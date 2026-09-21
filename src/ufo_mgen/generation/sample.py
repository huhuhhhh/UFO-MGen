"""Stage-III sampling and crystal decoding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data_package import (
    CHEM_COLS,
    LATTICE_COLS,
    WyckoffStage2Config,
    WyckoffStage2Dataset,
)
from ..models.stage3_swg import sample_trajectories
from ..wyckoff.decode import decode_structure
from ..wyckoff.lattice_projection import project_raw_samples_by_spacegroup

#: Manuscript ODE settings.
PAPER_INTEGRATION_STEPS = 48       # Stage-III training / default sampling
PAPER_ROUTE_OVERRIDE_STEPS = 64    # used by route overrides in the 50k run
PAPER_SAMPLE_TEMP = 0.5


@dataclass
class SamplingConfig:
    """Knobs for one sampling call."""

    n_samples: int = 16
    integration_steps: int = PAPER_INTEGRATION_STEPS
    sample_temp: float = PAPER_SAMPLE_TEMP
    project_lattice: bool = True
    split: str = "train"
    device: str = "cpu"
    seed: Optional[int] = None


def lattice_statistics(dataset: WyckoffStage2Dataset) -> tuple:
    """Train-split lattice mean/std used to invert the model's standardisation."""
    df = dataset.struct_df
    train = df[df["split"].astype(str).str.lower() == "train"]
    if train.empty:
        train = df
    mean = train[LATTICE_COLS].mean().to_numpy(dtype=np.float32)
    std = train[LATTICE_COLS].std().replace(0, 1.0).to_numpy(dtype=np.float32)
    return mean, std


def flatten_raw(lattice: np.ndarray, orbit: np.ndarray, value_mask: np.ndarray) -> np.ndarray:
    """Flatten ``(lattice, orbit)`` blocks into the raw decode vector layout."""
    n = lattice.shape[0]
    out: List[np.ndarray] = []
    for i in range(n):
        free = orbit[i][value_mask[i] > 0.5].reshape(-1)
        out.append(np.concatenate([lattice[i].reshape(-1), free]))
    width = max(v.shape[0] for v in out)
    padded = np.zeros((n, width), dtype=np.float32)
    for i, v in enumerate(out):
        padded[i, : v.shape[0]] = v
    return padded


@torch.no_grad()
def sample_raw(
    model,
    adapter,
    dataset: WyckoffStage2Dataset,
    spacegroup: int,
    cfg: SamplingConfig,
    *,
    chem_block: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Sample ``cfg.n_samples`` raw representation vectors."""
    from torch.utils.data import DataLoader

    if cfg.seed is not None:
        torch.manual_seed(int(cfg.seed))

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    base = next(iter(loader))

    batch: Dict[str, torch.Tensor] = {}
    for key, value in base.items():
        if isinstance(value, torch.Tensor):
            reps = [cfg.n_samples] + [1] * (value.dim() - 1)
            batch[key] = value.repeat(*reps).to(cfg.device)
    if chem_block is not None:
        batch["chem_block"] = chem_block.to(cfg.device)

    lattice, orbit = sample_trajectories(
        model, adapter, batch, temp=cfg.sample_temp, steps=int(cfg.integration_steps)
    )

    mean, std = lattice_statistics(dataset)
    lattice_raw = lattice.cpu().numpy() * std[None, :] + mean[None, :]
    orbit_raw = orbit.cpu().numpy()
    value_mask = batch["orbit_value_mask"].cpu().numpy()

    raw = flatten_raw(lattice_raw, orbit_raw, value_mask)
    if cfg.project_lattice:
        raw = project_raw_samples_by_spacegroup(raw, spacegroup)
    return raw


def decode_samples(
    raw: np.ndarray,
    spacegroup: int,
    scaffold: str,
    orbit_species: Optional[Sequence[str]] = None,
    *,
    skip_failures: bool = True,
) -> List:
    """Decode raw vectors into structures, optionally skipping failures."""
    out = []
    for vec in raw:
        try:
            out.append(decode_structure(vec, spacegroup, scaffold, orbit_species))
        except Exception:
            if not skip_failures:
                raise
    return out


def sample_structures(
    checkpoint,
    data_root,
    spacegroup: int,
    scaffold: str,
    cfg: Optional[SamplingConfig] = None,
    orbit_species: Optional[Sequence[str]] = None,
    *,
    context_mode: Optional[str] = None,
) -> List:
    """End-to-end convenience: load a Stage-III route, sample, decode."""
    from ..models.io import load_stage3_swg

    cfg = cfg or SamplingConfig()
    model, adapter, _ = load_stage3_swg(
        checkpoint, data_root, scaffold=scaffold, split=cfg.split, device=cfg.device,
        context_mode=context_mode,
    )
    dataset = WyckoffStage2Dataset(
        WyckoffStage2Config(root=Path(str(data_root)), split=cfg.split,
                            scaffold=scaffold, normalize=True)
    )
    raw = sample_raw(model, adapter, dataset, spacegroup, cfg)
    return decode_samples(raw, spacegroup, scaffold, orbit_species)
