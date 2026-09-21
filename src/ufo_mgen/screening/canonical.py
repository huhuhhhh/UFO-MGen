"""Scaffold and space-group consistency checks for generated structures."""

from __future__ import annotations

from typing import Optional

from ..wyckoff import canonicalize_scaffold, encode_structure
from ..wyckoff.decode import strip_scaffold_prefix
from .result import StageResult

STAGE = "scaffold_correctness"


def detect_scaffold(structure, symprec: float = 0.01, angle_tolerance: float = 5.0):
    """Re-detect ``(spacegroup, scaffold, canonical_scaffold)`` from a structure."""
    enc = encode_structure(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    return enc.spacegroup, enc.scaffold, enc.canonical_scaffold


def check_scaffold(
    structure,
    target_spacegroup: Optional[int] = None,
    target_scaffold: Optional[str] = None,
    *,
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
    require_scaffold_match: bool = True,
) -> StageResult:
    """Compare a structure's detected topology against its generation target."""
    try:
        sg, scaffold, canonical = detect_scaffold(
            structure, symprec=symprec, angle_tolerance=angle_tolerance)
    except Exception as exc:
        return StageResult.fail(STAGE, "symmetry_analysis_failed",
                                error=f"{type(exc).__name__}: {exc}")

    diag = {
        "detected_spacegroup": sg,
        "detected_scaffold": scaffold,
        "detected_canonical_scaffold": canonical,
        "target_spacegroup": target_spacegroup,
        "target_scaffold": target_scaffold,
    }

    if target_spacegroup is not None and int(sg) != int(target_spacegroup):
        return StageResult.fail(STAGE, "spacegroup_mismatch", **diag)

    if target_scaffold is not None and require_scaffold_match:
        want = canonicalize_scaffold(
            int(target_spacegroup if target_spacegroup is not None else sg),
            strip_scaffold_prefix(target_scaffold),
        )
        diag["target_canonical_scaffold"] = want
        if canonical != want:
            return StageResult.fail(STAGE, "scaffold_mismatch", **diag)

    return StageResult.ok(STAGE, **diag)
