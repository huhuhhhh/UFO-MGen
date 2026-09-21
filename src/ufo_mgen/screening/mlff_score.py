"""Online MLFF scoring, risk flags, and candidate ranking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .result import StageResult
from .thresholds import DEFAULT_THRESHOLDS, MLFFThresholds

STAGE = "mlff_screening"


def load_scorer(backend: str = "chgnet", device: str = "auto"):
    """Load a single-point calculator. Imported lazily; see :mod:`ufo_mgen.mlip`."""
    from ..mlip.harness import load_backend

    return load_backend(backend, device=device)


def single_point(structure, calculator) -> Dict[str, Any]:
    """Energy per atom, maximum force and stress norm for one structure."""
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator

    energy = float(atoms.get_potential_energy()) / max(len(atoms), 1)
    forces = np.asarray(atoms.get_forces(), dtype=float)
    max_force = float(np.abs(forces).max()) if forces.size else float("nan")
    try:
        stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
        stress_l2 = float(np.linalg.norm(stress))
    except Exception:
        stress_l2 = float("nan")

    return {"energy_per_atom_ev": energy, "max_force_ev_per_a": max_force,
            "stress_l2": stress_l2}


def risk_score(scores: Dict[str, Any],
               thresholds: Optional[MLFFThresholds] = None) -> float:
    """Accumulate soft risk flags into a single advisory number."""
    t = thresholds or DEFAULT_THRESHOLDS.mlff
    if not scores:
        return t.missing_score_risk

    risk = 0.0
    e = scores.get("energy_per_atom_ev")
    f = scores.get("max_force_ev_per_a")
    s = scores.get("stress_l2")
    if e is None or not np.isfinite(e):
        risk += t.missing_score_risk
    elif e > t.risk_positive_energy_ev_per_atom:
        risk += 1.0
    if f is not None and np.isfinite(f) and f > t.risk_force_ev_per_a:
        risk += 1.0
    if s is not None and np.isfinite(s) and s > t.risk_stress_l2:
        risk += 1.0
    return float(risk)


def check_mlff(
    structure,
    calculator,
    thresholds: Optional[MLFFThresholds] = None,
) -> StageResult:
    """Score one structure and reject only if it is unusable."""
    t = thresholds or DEFAULT_THRESHOLDS.mlff
    try:
        scores = single_point(structure, calculator)
    except Exception as exc:
        return StageResult(stage=STAGE, passed=True,
                           reason=f"scoring_failed:{type(exc).__name__}",
                           diagnostics={}, soft_penalty=t.missing_score_risk)

    risk = risk_score(scores, t)
    diag = dict(scores, risk=risk)

    f = scores.get("max_force_ev_per_a")
    if f is not None and np.isfinite(f) and f > t.max_force_ev_per_a:
        return StageResult.fail(STAGE, "force_above_hard_limit", **diag)
    return StageResult(stage=STAGE, passed=True, reason="ok",
                       diagnostics=diag, soft_penalty=risk)


def rank_candidates(
    records: Sequence[Dict[str, Any]],
    *,
    keep: Optional[int] = None,
    energy_key: str = "energy_per_atom_ev",
    risk_key: str = "risk",
) -> List[int]:
    """Rank by (risk, energy) ascending and return the indices to keep.

    ``keep`` is a caller decision, not a property of the protocol: how many
    candidates survive depends on the relaxation budget available, not on
    anything intrinsic to the method.
    """
    def key(i: int):
        r = records[i]
        e = r.get(energy_key)
        risk = r.get(risk_key, 0.0)
        return (float(risk), float(e) if e is not None and np.isfinite(e) else np.inf)

    order = sorted(range(len(records)), key=key)
    return order if keep is None else order[: int(keep)]
