"""Post-generation screening utilities and funnel orchestration."""

from .canonical import check_scaffold, detect_scaffold
from .chemistry import check_chemistry
from .funnel import (
    CandidateAudit,
    ScreeningFunnel,
    chemistry_stage,
    default_funnel,
    geometry_stage,
    mlff_stage,
    novelty_stage,
    quick_ef_prescreen_stage,
    relaxation_stage,
    scaffold_stage,
    summarise,
)
from .geometry import (
    check_geometry,
    min_pair_distance,
    min_pair_ratio,
    volume_per_atom_scale,
)
from .mlff_score import check_mlff, rank_candidates, risk_score, single_point
from .novelty import (
    DUPLICATE,
    ISOMER,
    NEAR_FAMILY,
    NOVEL,
    check_novelty,
    classify_novelty,
    deduplicate_batch,
)
from .relax import (
    check_post_relax,
    check_quick_ef_prescreen,
    formation_energy_per_atom,
    relax_structure,
    relaxation_rmsd,
)
from .result import StageResult
from .thresholds import (
    DEFAULT_THRESHOLDS,
    GEOMETRY_PRESETS,
    GeometryThresholds,
    MLFFThresholds,
    NoveltyThresholds,
    QuickEfPrescreen,
    RelaxThresholds,
    ScreeningThresholds,
)

__all__ = [
    "StageResult", "ScreeningFunnel", "CandidateAudit", "default_funnel", "summarise",
    "scaffold_stage", "geometry_stage", "chemistry_stage", "novelty_stage",
    "mlff_stage", "quick_ef_prescreen_stage", "relaxation_stage",
    "check_scaffold", "detect_scaffold",
    "check_geometry", "min_pair_distance", "min_pair_ratio", "volume_per_atom_scale",
    "check_chemistry",
    "check_novelty", "classify_novelty", "deduplicate_batch",
    "DUPLICATE", "ISOMER", "NEAR_FAMILY", "NOVEL",
    "check_mlff", "single_point", "risk_score", "rank_candidates",
    "relax_structure", "relaxation_rmsd", "check_post_relax",
    "check_quick_ef_prescreen", "formation_energy_per_atom",
    "DEFAULT_THRESHOLDS", "ScreeningThresholds", "GeometryThresholds",
    "NoveltyThresholds", "MLFFThresholds", "RelaxThresholds", "QuickEfPrescreen",
    "GEOMETRY_PRESETS",
]
