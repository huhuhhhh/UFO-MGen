"""Loose relaxation and post-relaxation filtering utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .result import StageResult
from .thresholds import DEFAULT_THRESHOLDS, RelaxThresholds

RELAX_STAGE = "relaxation"
POST_STAGE = "post_relax_filter"
PRESCREEN_STAGE = "quick_formation_energy_prescreen"


def relax_structure(
    structure,
    calculator,
    thresholds: Optional[RelaxThresholds] = None,
) -> Dict[str, Any]:
    """Relax with a loose convergence target. Returns the relaxed cell and energy."""
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor

    t = thresholds or DEFAULT_THRESHOLDS.relax
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator

    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=t.fmax, steps=t.steps)

    relaxed = AseAtomsAdaptor.get_structure(atoms)
    return {
        "structure": relaxed,
        "energy_per_atom_ev": float(atoms.get_potential_energy()) / max(len(atoms), 1),
        "converged": bool(opt.converged()),
        "steps": int(opt.get_number_of_steps()),
    }


def relaxation_rmsd(before, after, ltol=0.2, stol=0.3, angle_tol=5.0) -> Optional[float]:
    """RMSD between the input and relaxed cells, or ``None`` if they do not match."""
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                               primitive_cell=True, scale=True)
    try:
        result = matcher.get_rms_dist(before, after)
    except Exception:
        return None
    return float(result[0]) if result else None


def formation_energy_per_atom(
    structure,
    energy_per_atom_ev: float,
    elemental_reference_ev: Dict[str, float],
) -> Optional[float]:
    """Formation energy against per-element reference energies.

    ``elemental_reference_ev`` maps element symbol to energy per atom in the
    element's reference state, computed with the same potential.
    """
    comp = structure.composition
    total = comp.num_atoms
    if total <= 0:
        return None
    reference = 0.0
    for element, amount in comp.get_el_amt_dict().items():
        if element not in elemental_reference_ev:
            return None
        reference += float(elemental_reference_ev[element]) * float(amount)
    return float(energy_per_atom_ev - reference / total)


def check_post_relax(
    original,
    relaxed,
    energy_per_atom_ev: float,
    thresholds: Optional[RelaxThresholds] = None,
    elemental_reference_ev: Optional[Dict[str, float]] = None,
) -> StageResult:
    """Filter a relaxed candidate on movement, geometry and formation energy."""
    from .geometry import min_pair_distance

    t = thresholds or DEFAULT_THRESHOLDS.relax
    rmsd = relaxation_rmsd(original, relaxed)
    d_min = min_pair_distance(relaxed)
    diag: Dict[str, Any] = {"relaxed_rmsd": rmsd, "min_pair_distance": d_min,
                            "energy_per_atom_ev": energy_per_atom_ev}

    if rmsd is not None and rmsd > t.max_relaxed_rmsd:
        return StageResult.fail(POST_STAGE, "moved_too_far_during_relaxation", **diag)
    if d_min < t.min_relaxed_pair_distance:
        return StageResult.fail(POST_STAGE, "collapsed_during_relaxation", **diag)

    if elemental_reference_ev:
        ef = formation_energy_per_atom(relaxed, energy_per_atom_ev, elemental_reference_ev)
        diag["formation_energy_per_atom_ev"] = ef
        if ef is not None and ef > t.max_formation_energy_ev_per_atom:
            return StageResult.fail(POST_STAGE, "formation_energy_above_threshold", **diag)

    return StageResult.ok(POST_STAGE, **diag)


def check_quick_ef_prescreen(
    structure,
    energy_per_atom_ev: float,
    elemental_reference_ev: Dict[str, float],
    *,
    enabled: bool = False,
    max_formation_energy_ev_per_atom: Optional[float] = None,
) -> StageResult:
    """Formation-energy gate applied *before* relaxation. Off by default.

    Judging an unrelaxed decoded cell on formation energy is aggressive: many
    structures that relax into sound crystals look unfavourable beforehand, so
    this trades recall for compute. Enable it only when the relaxation budget
    is the binding constraint, and expect to lose true positives.
    """
    if not enabled:
        return StageResult.skip(PRESCREEN_STAGE, "disabled_by_default")

    limit = (max_formation_energy_ev_per_atom
             if max_formation_energy_ev_per_atom is not None
             else DEFAULT_THRESHOLDS.quick_ef_prescreen.max_formation_energy_ev_per_atom)
    ef = formation_energy_per_atom(structure, energy_per_atom_ev, elemental_reference_ev)
    diag = {"formation_energy_per_atom_ev": ef, "threshold": limit}
    if ef is None:
        return StageResult.skip(PRESCREEN_STAGE, "missing_elemental_reference")
    if ef > limit:
        return StageResult.fail(PRESCREEN_STAGE, "formation_energy_above_threshold", **diag)
    return StageResult.ok(PRESCREEN_STAGE, **diag)
