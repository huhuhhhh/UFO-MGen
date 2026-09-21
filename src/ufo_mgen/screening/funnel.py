"""Post-generation screening funnel and audit records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .result import StageResult


@dataclass
class CandidateAudit:
    """The full screening trace for one candidate."""

    candidate_id: str
    stages: List[StageResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.stages)

    @property
    def rejected_at(self) -> Optional[str]:
        for s in self.stages:
            if not s.passed:
                return s.stage
        return None

    @property
    def reason(self) -> str:
        for s in self.stages:
            if not s.passed:
                return s.reason
        return "ok"

    @property
    def total_penalty(self) -> float:
        return float(sum(s.soft_penalty for s in self.stages))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "passed": self.passed,
            "rejected_at": self.rejected_at,
            "reason": self.reason,
            "total_penalty": self.total_penalty,
            "stages": [s.to_dict() for s in self.stages],
        }


#: One stage: takes a candidate context, returns a StageResult.
Stage = Callable[[Dict[str, Any]], StageResult]


@dataclass
class ScreeningFunnel:
    """An ordered list of named stages, run until one rejects."""

    stages: List[Stage] = field(default_factory=list)
    stop_on_failure: bool = True

    def add(self, stage: Stage) -> "ScreeningFunnel":
        self.stages.append(stage)
        return self

    def run(self, context: Dict[str, Any], candidate_id: str = "candidate") -> CandidateAudit:
        """Run one candidate through the funnel."""
        audit = CandidateAudit(candidate_id=candidate_id)
        for stage in self.stages:
            try:
                result = stage(context)
            except Exception as exc:
                result = StageResult.fail(
                    getattr(stage, "stage_name", "unknown_stage"),
                    f"stage_error:{type(exc).__name__}", error=str(exc))
            audit.stages.append(result)
            if not result.passed and self.stop_on_failure:
                break
        return audit

    def run_batch(self, contexts: Sequence[Dict[str, Any]],
                  ids: Optional[Sequence[str]] = None) -> List[CandidateAudit]:
        labels = list(ids) if ids is not None else [f"candidate_{i}" for i in range(len(contexts))]
        return [self.run(c, labels[i]) for i, c in enumerate(contexts)]


def summarise(audits: Iterable[CandidateAudit]) -> Dict[str, Any]:
    """Per-stage pass/fail/skip tallies for a set of audits.

    This counts whatever you ran. It is a diagnostic for your own funnel, not a
    record of any published run.
    """
    audits = list(audits)
    stages: Dict[str, Dict[str, int]] = {}
    for a in audits:
        for s in a.stages:
            row = stages.setdefault(s.stage, {"passed": 0, "failed": 0, "skipped": 0})
            if s.skipped:
                row["skipped"] += 1
            elif s.passed:
                row["passed"] += 1
            else:
                row["failed"] += 1
    return {
        "n_candidates": len(audits),
        "n_passed": sum(1 for a in audits if a.passed),
        "by_stage": stages,
    }


# ----------------------------------------------------------------- builders

def _named(fn: Stage, name: str) -> Stage:
    fn.stage_name = name  # type: ignore[attr-defined]
    return fn


def scaffold_stage(**kwargs) -> Stage:
    """Scaffold consistency against the generation target."""
    from .canonical import STAGE, check_scaffold

    def run(ctx: Dict[str, Any]) -> StageResult:
        return check_scaffold(ctx["structure"],
                              ctx.get("target_spacegroup"),
                              ctx.get("target_scaffold"), **kwargs)
    return _named(run, STAGE)


def geometry_stage(**kwargs) -> Stage:
    """Geometry sanity, complexity-aware if the context supplies a band."""
    from .geometry import STAGE, check_geometry

    def run(ctx: Dict[str, Any]) -> StageResult:
        return check_geometry(ctx["structure"],
                              complexity=ctx.get("complexity"), **kwargs)
    return _named(run, STAGE)


def chemistry_stage(**kwargs) -> Stage:
    """Composition plausibility."""
    from .chemistry import STAGE, check_chemistry

    def run(ctx: Dict[str, Any]) -> StageResult:
        return check_chemistry(ctx["structure"], **kwargs)
    return _named(run, STAGE)


def novelty_stage(reference: Sequence, **kwargs) -> Stage:
    """Novelty against a reference set."""
    from .novelty import STAGE, check_novelty

    def run(ctx: Dict[str, Any]) -> StageResult:
        return check_novelty(ctx["structure"], reference, **kwargs)
    return _named(run, STAGE)


def mlff_stage(calculator, **kwargs) -> Stage:
    """Single-point MLFF screening."""
    from .mlff_score import STAGE, check_mlff

    def run(ctx: Dict[str, Any]) -> StageResult:
        return check_mlff(ctx["structure"], calculator, **kwargs)
    return _named(run, STAGE)


def quick_ef_prescreen_stage(elemental_reference_ev: Dict[str, float],
                             *, enabled: bool = False, **kwargs) -> Stage:
    """Pre-relaxation formation-energy gate. Disabled unless asked for."""
    from .relax import PRESCREEN_STAGE, check_quick_ef_prescreen

    def run(ctx: Dict[str, Any]) -> StageResult:
        return check_quick_ef_prescreen(
            ctx["structure"], ctx.get("energy_per_atom_ev", 0.0),
            elemental_reference_ev, enabled=enabled, **kwargs)
    return _named(run, PRESCREEN_STAGE)


def relaxation_stage(calculator, elemental_reference_ev=None, **kwargs) -> Stage:
    """Relax, then filter on movement, geometry and formation energy.

    Writes the relaxed cell back into the context under ``relaxed_structure``.
    """
    from .relax import POST_STAGE, check_post_relax, relax_structure

    def run(ctx: Dict[str, Any]) -> StageResult:
        original = ctx["structure"]
        out = relax_structure(original, calculator, **kwargs)
        ctx["relaxed_structure"] = out["structure"]
        ctx["relaxed_energy_per_atom_ev"] = out["energy_per_atom_ev"]
        return check_post_relax(original, out["structure"], out["energy_per_atom_ev"],
                                elemental_reference_ev=elemental_reference_ev)
    return _named(run, POST_STAGE)


def default_funnel(
    *,
    reference: Optional[Sequence] = None,
    calculator=None,
    elemental_reference_ev: Optional[Dict[str, float]] = None,
    enable_quick_ef_prescreen: bool = False,
) -> ScreeningFunnel:
    """The screening protocol, cheapest stages first.

    Stages needing a calculator are omitted when none is supplied, so the
    cheap structural half of the funnel runs without any potential installed.
    """
    funnel = ScreeningFunnel()
    funnel.add(scaffold_stage())
    funnel.add(geometry_stage())
    funnel.add(chemistry_stage())
    if reference:
        funnel.add(novelty_stage(reference))
    if calculator is not None:
        funnel.add(mlff_stage(calculator))
        if enable_quick_ef_prescreen and elemental_reference_ev:
            funnel.add(quick_ef_prescreen_stage(elemental_reference_ev, enabled=True))
        funnel.add(relaxation_stage(calculator, elemental_reference_ev))
    return funnel
