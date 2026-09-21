"""Chemical plausibility checks for generated compositions."""

from __future__ import annotations

from .result import StageResult

STAGE = "chemistry_plausibility"


def check_chemistry(structure, *, reject_on_fail: bool = False,
                    soft_penalty: float = 0.25) -> StageResult:
    """Screen a structure's composition for chemical plausibility."""
    try:
        from ..evaluation.validity import smact_validity_check
    except Exception as exc:  # pragma: no cover - import guard
        return StageResult.skip(STAGE, f"validity_unavailable:{type(exc).__name__}")

    try:
        valid = bool(smact_validity_check(structure.composition))
    except ImportError:
        # smact is an optional dependency; without it the stage abstains
        # rather than silently passing everything as if it had checked.
        return StageResult.skip(STAGE, "smact_not_installed")
    except Exception as exc:
        return StageResult.skip(STAGE, f"check_failed:{type(exc).__name__}")

    diag = {"composition": str(structure.composition.reduced_formula),
            "smact_valid": valid}
    if valid:
        return StageResult.ok(STAGE, **diag)
    if reject_on_fail:
        return StageResult.fail(STAGE, "implausible_composition", **diag)
    return StageResult(stage=STAGE, passed=True, reason="implausible_composition_flagged",
                       diagnostics=diag, soft_penalty=soft_penalty)
