"""COV-R and COV-P coverage metrics for generated crystal sets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def compute_structural_fingerprints(
    structures: List[Structure],
    verbose: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute CrystalNN structural fingerprints for a list of structures.

    Uses matminer SiteStatsFingerprint(CrystalNNFingerprint) with mean aggregation.

    Returns:
        fingerprints: np.ndarray of shape (N_valid, d_SF)
        valid_indices: list of indices that succeeded
    """
    try:
        from matminer.featurizers.site import CrystalNNFingerprint
        from matminer.featurizers.structure import SiteStatsFingerprint
    except ImportError:
        raise ImportError(
            "matminer is required for structural fingerprints.\n"
            "Install with: pip install matminer"
        )

    cnn_fp = CrystalNNFingerprint.from_preset("ops")
    ssf = SiteStatsFingerprint(cnn_fp, stats=["mean", "std_dev"])

    fingerprints = []
    valid_indices = []
    failed = []

    iterator = tqdm(enumerate(structures), total=len(structures), desc="Structural fingerprints") if verbose else enumerate(structures)

    for idx, struct in iterator:
        try:
            fp = ssf.featurize(struct)
            fingerprints.append(fp)
            valid_indices.append(idx)
        except Exception as e:
            failed.append((idx, str(e)))

    if failed:
        print(f"[WARNING] Structural fingerprint failed for {len(failed)} structures.")

    return np.array(fingerprints, dtype=np.float32), valid_indices


def compute_compositional_fingerprints(
    structures: List[Structure],
    verbose: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute normalized Magpie compositional fingerprints.

    Uses matminer ElementProperty with Magpie statistics.
    Normalizes each fingerprint vector to unit L2 norm.

    Returns:
        fingerprints: np.ndarray of shape (N_valid, d_CF) — L2-normalized
        valid_indices: list of indices that succeeded
    """
    try:
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition
    except ImportError:
        raise ImportError(
            "matminer is required for compositional fingerprints.\n"
            "Install with: pip install matminer"
        )

    ep = ElementProperty.from_preset("magpie")

    fingerprints = []
    valid_indices = []
    failed = []

    iterator = tqdm(enumerate(structures), total=len(structures), desc="Compositional fingerprints") if verbose else enumerate(structures)

    for idx, struct in iterator:
        try:
            comp = struct.composition
            fp = ep.featurize(comp)
            fp = np.array(fp, dtype=np.float32)
            # L2 normalize
            norm = np.linalg.norm(fp)
            if norm > 0:
                fp = fp / norm
            fingerprints.append(fp)
            valid_indices.append(idx)
        except Exception as e:
            failed.append((idx, str(e)))

    if failed:
        print(f"[WARNING] Compositional fingerprint failed for {len(failed)} structures.")

    return np.array(fingerprints, dtype=np.float32), valid_indices


# ---------------------------------------------------------------------------
# COV-R and COV-P computation
# ---------------------------------------------------------------------------


def compute_cov_r(
    test_fps: np.ndarray,
    gen_fps: np.ndarray,
    threshold: float,
    batch_size: int = 1000,
) -> float:
    """
    COV-R: fraction of test structures with min distance to any generated structure <= threshold.

    Args:
        test_fps: (N_test, d) fingerprint array for test set
        gen_fps:  (N_gen, d) fingerprint array for generated set
        threshold: distance threshold
        batch_size: process test structures in batches to avoid OOM

    Returns:
        cov_r: float in [0, 1]
    """
    n_test = len(test_fps)
    covered = 0

    for i in range(0, n_test, batch_size):
        test_batch = test_fps[i:i + batch_size]  # (B, d)
        # Compute pairwise L2 distances: (B, N_gen)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        diff = test_batch[:, np.newaxis, :] - gen_fps[np.newaxis, :, :]  # (B, N_gen, d)
        dists = np.linalg.norm(diff, axis=-1)  # (B, N_gen)
        min_dists = dists.min(axis=1)  # (B,)
        covered += (min_dists <= threshold).sum()

    return covered / n_test


def compute_cov_p(
    test_fps: np.ndarray,
    gen_fps: np.ndarray,
    threshold: float,
    batch_size: int = 1000,
) -> float:
    """
    COV-P: fraction of generated structures with min distance to any test structure <= threshold.
    """
    n_gen = len(gen_fps)
    covered = 0

    for i in range(0, n_gen, batch_size):
        gen_batch = gen_fps[i:i + batch_size]
        diff = gen_batch[:, np.newaxis, :] - test_fps[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        min_dists = dists.min(axis=1)
        covered += (min_dists <= threshold).sum()

    return covered / n_gen


def compute_coverage_metrics(
    test_fps: np.ndarray,
    gen_fps: np.ndarray,
    thresholds: List[float],
    fp_type: str = "structural",
) -> Dict[str, Any]:
    """Compute COV-R and COV-P at multiple thresholds."""
    results = {
        "fingerprint_type": fp_type,
        "n_test": len(test_fps),
        "n_gen": len(gen_fps),
        "thresholds": {},
    }

    for tau in thresholds:
        print(f"  Computing COV-R/P at threshold={tau} ({fp_type})...")
        cov_r = compute_cov_r(test_fps, gen_fps, threshold=tau)
        cov_p = compute_cov_p(test_fps, gen_fps, threshold=tau)
        results["thresholds"][str(tau)] = {
            "threshold": tau,
            "COV_R": cov_r,
            "COV_R_pct": 100.0 * cov_r,
            "COV_P": cov_p,
            "COV_P_pct": 100.0 * cov_p,
        }

    return results


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
