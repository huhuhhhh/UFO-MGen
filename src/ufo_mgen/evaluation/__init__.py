"""Evaluation: public benchmark metrics and the MSE stability funnel."""

from .coverage import compute_coverage_metrics, compute_cov_p, compute_cov_r
from .interpolation_extrapolation import (
    CoverageSplit,
    detect_spacegroup,
    spacegroup_coverage,
    split_by_training_coverage,
)
from .mse import (
    DEFAULT_THRESHOLDS,
    MLIP_BACKENDS,
    MLIP_MODEL_NAMES,
    MSERecord,
    MSEThresholds,
    consensus_pass,
    funnel_counts,
)
from .public_metrics import evaluate_generated, load_structures_from_dir
from .validity import (
    compute_compositional_validity,
    compute_structural_validity,
    is_structurally_valid,
)

__all__ = [
    "is_structurally_valid", "compute_structural_validity",
    "compute_compositional_validity",
    "compute_cov_r", "compute_cov_p", "compute_coverage_metrics",
    "split_by_training_coverage", "spacegroup_coverage", "detect_spacegroup",
    "CoverageSplit",
    "MSERecord", "MSEThresholds", "DEFAULT_THRESHOLDS",
    "consensus_pass", "funnel_counts", "MLIP_BACKENDS", "MLIP_MODEL_NAMES",
    "evaluate_generated", "load_structures_from_dir",
]
