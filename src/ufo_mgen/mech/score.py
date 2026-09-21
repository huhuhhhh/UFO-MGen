"""Mechanical K/G/E scoring and sample-weight utilities for UFO-Mech."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

#: Relative weight of each log-modulus in the score.
SCORE_WEIGHTS: Dict[str, float] = {
    "log_K": 1.0,
    "log_G": 1.0,
    "log_E": 1.0,
}

#: Percentiles at which scores are binned.
SCORE_QUANTILES: Tuple[float, ...] = (0.50, 0.70, 0.90, 0.95, 0.99)

#: Sample weights are clamped to this range before sampling.
SAMPLE_WEIGHT_RANGE: Tuple[float, float] = (1.0, 5.0)

#: Poisson ratio outside this range is treated as unphysical.
VALID_POISSON_RANGE: Tuple[float, float] = (0.0, 0.45)


def _finite(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def stability_penalty(
    poisson_ratio: Optional[float] = None,
    elastic_stable: Optional[bool] = None,
) -> float:
    """Penalty for elastically unstable or unphysical entries."""
    penalty = 0.0
    if elastic_stable is False:
        penalty += 1.0
    nu = _finite(poisson_ratio)
    lo, hi = VALID_POISSON_RANGE
    if np.isfinite(nu) and not (lo <= nu <= hi):
        penalty += 0.5
    return float(penalty)


def compute_score_from_logs(
    log_k: Any,
    log_g: Any,
    log_e: Any,
    score_stats: Dict[str, Dict[str, float]],
    penalty: float = 0.0,
) -> float:
    """The extreme score. ``score_stats`` holds the corpus mean/std per log-modulus."""

    def z(col: str, value: Any) -> float:
        v = _finite(value)
        if not np.isfinite(v):
            return 0.0                      # a missing modulus is neutral, not extreme
        mean = score_stats[col]["mean"]
        std = max(score_stats[col]["std"], 1e-8)
        return (v - mean) / std

    return float(
        SCORE_WEIGHTS["log_K"] * z("log_K", log_k)
        + SCORE_WEIGHTS["log_G"] * z("log_G", log_g)
        + SCORE_WEIGHTS["log_E"] * z("log_E", log_e)
        - penalty
    )


def score_statistics(df, columns=("K", "G", "E")) -> Dict[str, Dict[str, float]]:
    """Corpus mean/std of each log-modulus, for z-scoring."""
    stats: Dict[str, Dict[str, float]] = {}
    for src in columns:
        v = np.log(np.clip(np.asarray(df[src], dtype=float), 1e-6, None))
        v = v[np.isfinite(v)]
        stats[f"log_{src}"] = {"mean": float(v.mean()), "std": float(v.std())}
    return stats


def add_mechanical_scores(df, score_stats: Optional[Dict[str, Dict[str, float]]] = None):
    """Attach ``penalty_unstable`` and ``mechanical_extreme_score`` to a label table."""
    df = df.copy()
    stats = score_stats or score_statistics(df)
    penalties, scores = [], []
    for row in df.itertuples(index=False):
        p = stability_penalty(
            getattr(row, "nu", None) if hasattr(row, "nu") else getattr(row, "poisson_ratio", None),
            getattr(row, "elastic_stability", None),
        )
        penalties.append(p)
        scores.append(
            compute_score_from_logs(
                np.log(max(_finite(getattr(row, "K", np.nan)), 1e-6)),
                np.log(max(_finite(getattr(row, "G", np.nan)), 1e-6)),
                np.log(max(_finite(getattr(row, "E", np.nan)), 1e-6)),
                stats,
                p,
            )
        )
    df["penalty_unstable"] = penalties
    df["mechanical_extreme_score"] = scores
    return df


def bin_and_weight(
    scores: Sequence[float],
    quantiles: Sequence[float] = SCORE_QUANTILES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin scores at the given percentiles and map bins to sample weights.

    Returns ``(bin_index, sample_weight)``. Bin 0 is the bottom half and keeps
    weight 1.0; the top bin reaches the cap.
    """
    s = np.asarray(scores, dtype=float)
    cuts = [float(np.nanquantile(s, q)) for q in quantiles]
    bins = np.digitize(s, cuts)
    lo, hi = SAMPLE_WEIGHT_RANGE
    weights = np.clip(1.0 + bins.astype(float), lo, hi)
    return bins, weights
