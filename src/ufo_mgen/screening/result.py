"""Result records shared by screening stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StageResult:
    """Outcome and diagnostics for one screening stage."""

    stage: str
    passed: bool
    reason: str = "ok"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    #: Advisory penalty; a stage may flag a candidate without rejecting it.
    soft_penalty: float = 0.0
    skipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def ok(cls, stage: str, **diagnostics: Any) -> "StageResult":
        return cls(stage=stage, passed=True, reason="ok", diagnostics=diagnostics)

    @classmethod
    def fail(cls, stage: str, reason: str, **diagnostics: Any) -> "StageResult":
        return cls(stage=stage, passed=False, reason=reason, diagnostics=diagnostics)

    @classmethod
    def skip(cls, stage: str, reason: str = "not_configured") -> "StageResult":
        return cls(stage=stage, passed=True, reason=reason, skipped=True)
