"""UFO-Mech utilities for property-guided generation and MatterSim validation."""

from .element_mask import (
    ELEMENT_SETS,
    apply_element_mask,
    build_logit_mask,
    parse_allowed_z,
)
from .finetune import (
    FINETUNE_SETTINGS,
    build_weighted_sampler,
    finetune_com,
    overfitting_report,
)
from .predictor import (
    TARGET_COLUMNS,
    build_predictor,
    fit_predictor,
    predict_moduli,
)
from ..mlip.harness import MECHANICAL_BACKEND
from .score import (
    SAMPLE_WEIGHT_RANGE,
    SCORE_QUANTILES,
    SCORE_WEIGHTS,
    add_mechanical_scores,
    bin_and_weight,
    compute_score_from_logs,
    score_statistics,
    stability_penalty,
)

__all__ = [
    "MECHANICAL_BACKEND",
    "SCORE_WEIGHTS", "SCORE_QUANTILES", "SAMPLE_WEIGHT_RANGE",
    "compute_score_from_logs", "score_statistics", "add_mechanical_scores",
    "bin_and_weight", "stability_penalty",
    "build_predictor", "fit_predictor", "predict_moduli", "TARGET_COLUMNS",
    "finetune_com", "build_weighted_sampler", "overfitting_report",
    "FINETUNE_SETTINGS",
    "apply_element_mask", "build_logit_mask", "parse_allowed_z", "ELEMENT_SETS",
]
