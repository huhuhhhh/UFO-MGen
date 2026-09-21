"""Threshold definitions for the post-generation screening protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

#: Complexity bands used by the geometry presets.
COMPLEXITY_BANDS = ("low", "mid", "high")


@dataclass(frozen=True)
class GeometryThresholds:
    """Geometric plausibility limits for a decoded cell."""

    #: Absolute floor on the shortest interatomic distance, in angstrom.
    min_valid_dist: float = 0.9
    #: Shortest distance as a fraction of the summed covalent radii.
    min_pair_ratio: float = 0.5
    #: Volume per atom, as a multiple of a reference estimate.
    min_vpa_scale: float = 0.5
    max_vpa_scale: float = 1.6

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


#: Complexity-aware geometry presets.
GEOMETRY_PRESETS: Dict[str, GeometryThresholds] = {
    "low": GeometryThresholds(min_valid_dist=0.85, min_pair_ratio=0.45,
                              min_vpa_scale=0.45, max_vpa_scale=1.8),
    "mid": GeometryThresholds(min_valid_dist=0.92, min_pair_ratio=0.52,
                              min_vpa_scale=0.45, max_vpa_scale=1.6),
    "high": GeometryThresholds(min_valid_dist=1.00, min_pair_ratio=0.55,
                               min_vpa_scale=0.50, max_vpa_scale=1.5),
}


@dataclass(frozen=True)
class NoveltyThresholds:
    """``StructureMatcher`` tolerances for novelty and deduplication.

    The batch tolerances are deliberately looser than the reference ones:
    within a single generated batch, near-identical samples are usually the
    same structure drawn twice, whereas a near-match against a reference
    database is a weaker claim and should not hard-reject.
    """

    ltol: float = 0.1
    stol: float = 0.2
    angle_tol: float = 5.0
    batch_ltol: float = 0.15
    batch_stol: float = 0.25
    batch_angle_tol: float = 5.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class MLFFThresholds:
    """Limits and risk flags for the cheap online MLFF pass."""

    #: Hard reject above this force; a decoded cell this far off is unusable.
    max_force_ev_per_a: float = 100.0
    #: Soft risk flags -- they mark a candidate, they do not reject it.
    risk_positive_energy_ev_per_atom: float = 0.1
    risk_force_ev_per_a: float = 50.0
    risk_stress_l2: float = 30.0
    #: Risk assigned when scoring fails outright.
    missing_score_risk: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class RelaxThresholds:
    """Online relaxation, and the filter applied to what comes out of it."""

    fmax: float = 0.15
    steps: int = 50
    #: Post-relax limits.
    max_relaxed_rmsd: float = 2.0
    min_relaxed_pair_distance: float = 0.7
    max_formation_energy_ev_per_atom: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class QuickEfPrescreen:
    """Formation-energy prescreen applied *before* relaxation.

    Disabled by default, and it should stay that way unless you know why you
    want it. Judging an unrelaxed decoded cell on formation energy is a harsh
    test: many structures that relax into perfectly sound crystals look
    unfavourable before relaxation, so the stage trades recall for compute.
    The production profile enabled it to fit a fixed relaxation budget.
    """

    enabled: bool = False
    max_formation_energy_ev_per_atom: float = 0.05

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScreeningThresholds:
    """The whole protocol, in one object."""

    geometry: GeometryThresholds = field(default_factory=GeometryThresholds)
    novelty: NoveltyThresholds = field(default_factory=NoveltyThresholds)
    mlff: MLFFThresholds = field(default_factory=MLFFThresholds)
    relax: RelaxThresholds = field(default_factory=RelaxThresholds)
    quick_ef_prescreen: QuickEfPrescreen = field(default_factory=QuickEfPrescreen)

    def geometry_for(self, complexity: Optional[str] = None) -> GeometryThresholds:
        """Geometry limits for a complexity band, or the defaults."""
        if complexity is None:
            return self.geometry
        band = str(complexity).lower()
        if band not in GEOMETRY_PRESETS:
            raise KeyError(f"unknown complexity band {complexity!r}; "
                           f"expected one of {COMPLEXITY_BANDS}")
        return GEOMETRY_PRESETS[band]

    def to_dict(self) -> Dict[str, object]:
        return {
            "geometry": self.geometry.to_dict(),
            "novelty": self.novelty.to_dict(),
            "mlff": self.mlff.to_dict(),
            "relax": self.relax.to_dict(),
            "quick_ef_prescreen": self.quick_ef_prescreen.to_dict(),
        }


#: The protocol as released.
DEFAULT_THRESHOLDS = ScreeningThresholds()
