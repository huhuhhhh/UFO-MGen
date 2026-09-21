"""Space-group interpolation and extrapolation coverage utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

#: Looser tolerance for detecting symmetry in unrelaxed generated cells.
COVERAGE_SYMPREC = 0.2
COVERAGE_ANGLE_TOL = 5.0


def detect_spacegroup(structure, *, symprec: float = COVERAGE_SYMPREC,
                      angle_tolerance: float = COVERAGE_ANGLE_TOL) -> Optional[int]:
    """Detected space-group number, or ``None`` if analysis fails."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
        return int(sga.get_space_group_number())
    except Exception:
        return None


@dataclass
class CoverageSplit:
    """Interpolation / extrapolation breakdown of a generated set."""

    n_generated: int
    n_valid: int
    training_spacegroups: Set[int]
    interpolation_sgs: Set[int]
    extrapolation_sgs: Set[int]
    n_interpolation_structures: int
    n_extrapolation_structures: int

    @property
    def n_extrapolation_sgs(self) -> int:
        return len(self.extrapolation_sgs)

    @property
    def extrapolation_structure_rate(self) -> float:
        return (self.n_extrapolation_structures / self.n_valid) if self.n_valid else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_generated": self.n_generated,
            "n_valid": self.n_valid,
            "n_training_spacegroups": len(self.training_spacegroups),
            "n_interpolation_spacegroups": len(self.interpolation_sgs),
            "n_extrapolation_spacegroups": self.n_extrapolation_sgs,
            "n_interpolation_structures": self.n_interpolation_structures,
            "n_extrapolation_structures": self.n_extrapolation_structures,
            "extrapolation_structure_rate_pct": 100.0 * self.extrapolation_structure_rate,
            "extrapolation_spacegroup_list": sorted(self.extrapolation_sgs),
        }


def split_by_training_coverage(
    structures: Sequence,
    training_spacegroups: Iterable[int],
    *,
    symprec: float = COVERAGE_SYMPREC,
    detected: Optional[Sequence[Optional[int]]] = None,
) -> CoverageSplit:
    """Classify a generated set into interpolation and extrapolation."""
    train = {int(s) for s in training_spacegroups}
    if detected is None:
        detected = [detect_spacegroup(s, symprec=symprec) for s in structures]

    interp_sgs: Set[int] = set()
    extrap_sgs: Set[int] = set()
    n_interp = n_extrap = 0
    for sg in detected:
        if sg is None:
            continue
        if sg in train:
            interp_sgs.add(sg)
            n_interp += 1
        else:
            extrap_sgs.add(sg)
            n_extrap += 1

    return CoverageSplit(
        n_generated=len(structures),
        n_valid=n_interp + n_extrap,
        training_spacegroups=train,
        interpolation_sgs=interp_sgs,
        extrapolation_sgs=extrap_sgs,
        n_interpolation_structures=n_interp,
        n_extrapolation_structures=n_extrap,
    )


def spacegroup_coverage(detected_spacegroups: Iterable[Optional[int]], total: int = 230) -> Dict[str, object]:
    """How much of the 230-group table a generated set reaches."""
    seen = {int(sg) for sg in detected_spacegroups if sg is not None}
    return {
        "n_spacegroups_covered": len(seen),
        "total_spacegroups": total,
        "coverage_pct": 100.0 * len(seen) / total,
        "covered": sorted(seen),
        "missing": [sg for sg in range(1, total + 1) if sg not in seen],
    }
