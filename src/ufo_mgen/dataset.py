"""Utilities for loading the released processed UFO-MGen corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

#: Free-parameter columns are packed as ``fp_0 … fp_N``.
FP_PREFIX = "fp_"
LATTICE_COLS = ["lattice_a", "lattice_b", "lattice_c", "alpha", "beta", "gamma"]


def data_dir() -> Path:
    """The repository's ``data/`` directory."""
    return Path(__file__).resolve().parents[2] / "data"


def _read(name: str, root: Optional[Path] = None):
    import pandas as pd

    path = (root or data_dir()) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run from a full checkout")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_corpus(split: Optional[str] = None, root: Optional[Path] = None):
    """Load the processed corpus, optionally one split of it."""
    df = _read("UFO_MGen_unified_dataset.parquet", root)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def load_scaffolds(root: Optional[Path] = None):
    """The 119 trainable scaffolds with their complexity and support."""
    return _read("wyckoff_scaffolds.csv", root)


def load_variable_schema(root: Optional[Path] = None):
    """Per-scaffold meaning of each free-parameter slot."""
    return _read("UFO_MGen_variable_schema.csv", root)


def free_coords_of(row, schema=None) -> List[float]:
    """Pull one structure's free Wyckoff coordinates out of its packed row.

    The corpus stores free parameters in fixed-width ``fp_*`` columns padded to
    the widest scaffold, so the tail is meaningless for narrower ones.
    ``n_fp_present`` records how many slots are real.
    """
    n = int(row.get("n_fp_present", 0) or 0)
    return [float(row[f"{FP_PREFIX}{i}"]) for i in range(n)]


def structure_vector(row) -> List[float]:
    """Build the raw decode vector ``[a,b,c,alpha,beta,gamma, free coords…]``.

    Feed the result straight to :func:`ufo_mgen.wyckoff.decode_structure`.
    """
    return [float(row[c]) for c in LATTICE_COLS] + free_coords_of(row)


def corpus_summary(root: Optional[Path] = None) -> Dict[str, object]:
    """Headline counts, recomputed from the released files."""
    df = load_corpus(root=root)
    sc = load_scaffolds(root=root)
    return {
        "n_structures": int(len(df)),
        "n_scaffolds": int(df["scaffold_id"].nunique()),
        "n_spacegroups": int(df["spacegroup"].nunique()),
        "splits": {k: int(v) for k, v in df["split"].value_counts().items()},
        "scaffold_table_rows": int(len(sc)),
        "D_rep_range": [int(sc["D_rep"].min()), int(sc["D_rep"].max())],
        "rho_range": [float(sc["rho"].min()), float(sc["rho"].max())],
    }
