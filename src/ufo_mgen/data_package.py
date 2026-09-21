"""Stage-III data packaging and tensor construction utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


CHEM_COLS = [
    'chem_n_elements',
    'chem_mean_Z',
    'chem_mean_X',
    'chem_max_frac',
    'chem_formula_atoms',
]
LATTICE_COLS = ['lattice_a', 'lattice_b', 'lattice_c', 'alpha', 'beta', 'gamma']



def _read_table(root: Path, candidates: List[str]) -> pd.DataFrame:
    """Read the first table that exists, accepting parquet or CSV.

    The published release ships parquet under ``UFO_MGen_*`` names; the raw
    training packages use the original CSV names. Both are accepted so the same
    loader drives training and the released data.
    """
    for name in candidates:
        path = root / name
        if path.exists():
            return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    raise FileNotFoundError(
        f"none of {candidates} found under {root}. Point --data-root at a "
        "Stage-III data package or at the released data/ directory."
    )


def _normalise_release_columns(*frames: pd.DataFrame) -> None:
    """Map released column names back onto the names the loader expects."""
    aliases = {
        'scaffold_id': 'scaffold',
        'wyckoff_sequence': 'source_raw_scaffold',
        'space_group': 'spacegroup',
        'support_count': 'n_structures',
        'K': 'n_orbits',
        'source_id': 'path',
    }
    for df in frames:
        for src, dst in aliases.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]

@dataclass
class WyckoffStage2Config:
    root: Path
    split: str | None = None
    scaffold: str | None = None
    normalize: bool = True
    max_rows: int = 0


class WyckoffStage2Dataset(Dataset):
    def __init__(self, cfg: WyckoffStage2Config):
        self.root = cfg.root
        self.split = cfg.split
        self.normalize = cfg.normalize

        self.struct_df = _read_table(self.root, [
            'UFO_MGen_unified_dataset.parquet', 'per_structure_vectors.parquet',
            'per_structure_vectors.csv'])
        self.orbit_df = _read_table(self.root, [
            'UFO_MGen_per_orbit_blocks.parquet', 'per_orbit_blocks.parquet',
            'per_orbit_blocks.csv'])
        self.schema_df = _read_table(self.root, [
            'UFO_MGen_variable_schema.csv', 'variable_schema.parquet',
            'variable_schema.csv'])
        self.scaffold_df = _read_table(self.root, [
            'wyckoff_scaffolds.csv', 'scaffold_summary.parquet',
            'scaffold_summary.csv'])
        _normalise_release_columns(self.struct_df, self.orbit_df,
                                   self.schema_df, self.scaffold_df)

        if self.split is not None:
            self.struct_df = self.struct_df[self.struct_df['split'] == self.split].copy()
        if cfg.scaffold is not None:
            self.struct_df = self.struct_df[self.struct_df['scaffold'] == cfg.scaffold].copy()
        if cfg.max_rows and cfg.max_rows > 0:
            self.struct_df = self.struct_df.head(int(cfg.max_rows)).copy()
        self.struct_df = self.struct_df.reset_index(drop=True)

        # Tensor widths are properties of the SCOPE a model was trained on.
        # A per-route Stage-III checkpoint was fitted on a single-scaffold
        # package, so its orbit/DOF widths are that scaffold's, not the whole
        # corpus's. Scoping these to the scaffold filter is what lets a released
        # route checkpoint load against the released union dataset.
        scope_scaffolds = self.scaffold_df
        scope_orbits = self.orbit_df
        if cfg.scaffold is not None:
            scope_scaffolds = self.scaffold_df[self.scaffold_df['scaffold'] == cfg.scaffold]
            scope_orbits = self.orbit_df[self.orbit_df['scaffold'] == cfg.scaffold]
            if scope_scaffolds.empty:
                raise KeyError(
                    f"scaffold {cfg.scaffold!r} not present in {self.root}. "
                    "Scaffold ids look like 'SG216|1a;1b;1c;1d;6f'."
                )

        self.scaffold_list = scope_scaffolds['scaffold'].tolist()
        self.scaffold_to_idx = {s: i for i, s in enumerate(self.scaffold_list)}
        self.max_orbits = int(scope_scaffolds['n_orbits'].max())
        # May legitimately be 0: for scaffolds whose every orbit sits on a fully
        # pinned special position there are no free coordinates at all, and the
        # model generates only the lattice. Do not clamp this to 1 -- the
        # released checkpoints for those six routes have zero-width coordinate
        # heads and would not load against a padded tensor.
        self.max_orbit_dof = int(
            scope_orbits.groupby(['scaffold', 'orbit_index'])['dof'].max().fillna(0).max()
        )
        self.max_feature_dim = int(scope_scaffolds['feature_dim_total'].max())

        self.fp_cols = sorted([c for c in self.struct_df.columns if c.startswith('fp_')], key=lambda x: int(x.split('_')[1]))
        self.chem_cols = [c for c in CHEM_COLS if c in self.struct_df.columns]

        train_df = _read_table(self.root, [
            'UFO_MGen_unified_dataset.parquet', 'per_structure_vectors.parquet',
            'per_structure_vectors.csv'])
        train_df = train_df[train_df['split'] == 'train'].copy()
        self.lattice_mean = train_df[LATTICE_COLS].mean().to_numpy(dtype=np.float32)
        self.lattice_std = train_df[LATTICE_COLS].std().replace(0, 1.0).to_numpy(dtype=np.float32)
        self.chem_mean = train_df[self.chem_cols].mean().to_numpy(dtype=np.float32) if self.chem_cols else np.zeros(0, dtype=np.float32)
        self.chem_std = train_df[self.chem_cols].std().replace(0, 1.0).to_numpy(dtype=np.float32) if self.chem_cols else np.zeros(0, dtype=np.float32)

        self.orbits_by_path: Dict[str, pd.DataFrame] = {
            path: g.sort_values('orbit_index').reset_index(drop=True)
            for path, g in self.orbit_df.groupby('path')
        }

    def __len__(self) -> int:
        return len(self.struct_df)

    def _encode_lattice(self, row: pd.Series) -> np.ndarray:
        x = row[LATTICE_COLS].to_numpy(dtype=np.float32)
        if self.normalize:
            x = (x - self.lattice_mean) / self.lattice_std
        return x

    def _encode_chem(self, row: pd.Series) -> np.ndarray:
        if not self.chem_cols:
            return np.zeros(0, dtype=np.float32)
        x = row[self.chem_cols].to_numpy(dtype=np.float32)
        if self.normalize:
            x = (x - self.chem_mean) / self.chem_std
        return x

    def _encode_flat_vector(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        vals = row[LATTICE_COLS + self.fp_cols].to_numpy(dtype=np.float32)
        mask = ~np.isnan(vals)
        vals = np.nan_to_num(vals, nan=0.0)
        if self.normalize:
            vals[:6] = (vals[:6] - self.lattice_mean) / self.lattice_std
        return vals, mask.astype(np.float32)

    def _encode_orbit_blocks(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        orbit_tensor = np.zeros((self.max_orbits, self.max_orbit_dof), dtype=np.float32)
        orbit_value_mask = np.zeros((self.max_orbits, self.max_orbit_dof), dtype=np.float32)
        orbit_active_mask = np.zeros(self.max_orbits, dtype=np.float32)
        orbit_dof = np.zeros(self.max_orbits, dtype=np.int64)
        per_path = self.orbits_by_path.get(path)
        if per_path is None:
            return orbit_tensor, orbit_value_mask, orbit_active_mask, orbit_dof
        for _, r in per_path.iterrows():
            i = int(r['orbit_index'])
            vals = json.loads(r['block_values_json'])
            dof = int(r['dof'])
            if dof > 0:
                arr = np.asarray(vals, dtype=np.float32)
                orbit_tensor[i, :dof] = arr
                orbit_value_mask[i, :dof] = 1.0
            orbit_active_mask[i] = 1.0
            orbit_dof[i] = dof
        return orbit_tensor, orbit_value_mask, orbit_active_mask, orbit_dof

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        row = self.struct_df.iloc[idx]
        path = str(row['path'])
        flat_vec, var_mask = self._encode_flat_vector(row)
        lattice = self._encode_lattice(row)
        chem = self._encode_chem(row)
        orbit_tensor, orbit_value_mask, orbit_active_mask, orbit_dof = self._encode_orbit_blocks(path)
        return {
            'path': path,
            'mp_id': str(row.get('mp_id', '')),
            'formula': str(row.get('formula', '')),
            'split': str(row['split']),
            'spacegroup': int(row['spacegroup']),
            'scaffold': str(row['scaffold']),
            'scaffold_idx': torch.tensor(self.scaffold_to_idx[str(row['scaffold'])], dtype=torch.long),
            'feature_dim': torch.tensor(int(row['feature_dim']), dtype=torch.long),
            'flat_vector': torch.tensor(flat_vec, dtype=torch.float32),
            'variable_mask': torch.tensor(var_mask, dtype=torch.float32),
            'lattice_block': torch.tensor(lattice, dtype=torch.float32),
            'chem_block': torch.tensor(chem, dtype=torch.float32),
            'orbit_tensor': torch.tensor(orbit_tensor, dtype=torch.float32),
            'orbit_value_mask': torch.tensor(orbit_value_mask, dtype=torch.float32),
            'orbit_active_mask': torch.tensor(orbit_active_mask, dtype=torch.float32),
            'orbit_dof': torch.tensor(orbit_dof, dtype=torch.long),
        }


def collate_wyckoff_stage2(batch: List[Dict[str, torch.Tensor | str | int]]) -> Dict[str, torch.Tensor | List[str]]:
    tensor_keys = [
        'scaffold_idx', 'feature_dim', 'flat_vector', 'variable_mask', 'lattice_block',
        'chem_block', 'orbit_tensor', 'orbit_value_mask', 'orbit_active_mask', 'orbit_dof'
    ]
    out: Dict[str, torch.Tensor | List[str]] = {}
    for k in tensor_keys:
        out[k] = torch.stack([item[k] for item in batch])
    for k in ['path', 'mp_id', 'formula', 'split', 'scaffold']:
        out[k] = [str(item[k]) for item in batch]
    out['spacegroup'] = torch.tensor([int(item['spacegroup']) for item in batch], dtype=torch.long)
    return out


def build_loader(
    root: Path,
    split: str | None,
    batch_size: int,
    shuffle: bool,
    scaffold: str | None = None,
    max_rows: int = 0,
    num_workers: int = 4,
    sample_weights: np.ndarray | None = None,
) -> Tuple[WyckoffStage2Dataset, DataLoader]:
    ds = WyckoffStage2Dataset(WyckoffStage2Config(root=root, split=split, scaffold=scaffold, normalize=True, max_rows=max_rows))
    use_persistent = num_workers > 0
    sampler = None
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if len(weights) != len(ds):
            raise ValueError(f'sample_weights length {len(weights)} != dataset length {len(ds)}')
        weights = np.clip(weights, a_min=1e-12, a_max=None)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(ds),
            replacement=True,
        )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=(shuffle and sampler is None), sampler=sampler, collate_fn=collate_wyckoff_stage2,
        num_workers=num_workers, pin_memory=True, persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return ds, loader
