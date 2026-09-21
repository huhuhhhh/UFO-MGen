"""Seeding and run provenance.

The manuscript generation runs use ``seed=42`` for the pilot profile and
``seed=44`` for the production profile; the pooled 50k corpus sweeps seeds
44..58. Record the seed you used alongside any structures you publish.
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
from pathlib import Path
from typing import Any, Dict

#: Seeds used by the manuscript runs.
PAPER_SEEDS = {"pilot": 42, "production": 44, "pooled_50k": list(range(44, 59))}


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch."""
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def sha256_file(path, *, chunk_size: int = 1 << 20) -> str:
    """Stream a SHA256 of a file."""
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_report() -> Dict[str, Any]:
    """Versions that materially affect generated geometry."""
    report: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "scipy", "torch", "pymatgen", "spglib", "pyxtal", "ase"):
        try:
            mod = __import__(name)
            report[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            report[name] = None
    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        report["cuda_available"] = None
    return report
