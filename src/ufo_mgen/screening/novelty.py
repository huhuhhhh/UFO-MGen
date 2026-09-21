"""Novelty classification and within-batch structure deduplication."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from .result import StageResult
from .thresholds import DEFAULT_THRESHOLDS, NoveltyThresholds

STAGE = "novelty"

#: Outcome labels.
DUPLICATE = "duplicate"
ISOMER = "same_composition_distinct"
NEAR_FAMILY = "near_known_family"
NOVEL = "novel"


def _matcher(ltol: float, stol: float, angle_tol: float):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    return StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                            primitive_cell=True, scale=True)


def classify_novelty(
    structure,
    reference: Sequence,
    thresholds: Optional[NoveltyThresholds] = None,
) -> Tuple[str, Optional[int]]:
    """Classify a structure against a reference set.

    Returns ``(label, matched_index)``; the index is the reference entry that
    triggered the label, or ``None``.
    """
    t = thresholds or DEFAULT_THRESHOLDS.novelty
    if not reference:
        return NOVEL, None

    strict = _matcher(t.ltol, t.stol, t.angle_tol)
    loose = _matcher(t.batch_ltol, t.batch_stol, t.batch_angle_tol)
    comp = structure.composition.reduced_composition

    same_composition = None
    near = None
    for i, ref in enumerate(reference):
        try:
            if strict.fit(structure, ref):
                return DUPLICATE, i
            if ref.composition.reduced_composition == comp and same_composition is None:
                same_composition = i
            if near is None and loose.fit(structure, ref):
                near = i
        except Exception:
            continue

    if same_composition is not None:
        return ISOMER, same_composition
    if near is not None:
        return NEAR_FAMILY, near
    return NOVEL, None


def check_novelty(
    structure,
    reference: Sequence,
    thresholds: Optional[NoveltyThresholds] = None,
    *,
    near_family_penalty: float = 0.25,
) -> StageResult:
    """Screen one structure. Only an exact duplicate is rejected."""
    if not reference:
        return StageResult.skip(STAGE, "no_reference_set")

    label, idx = classify_novelty(structure, reference, thresholds)
    diag = {"label": label, "matched_index": idx,
            "n_reference": len(reference)}

    if label == DUPLICATE:
        return StageResult.fail(STAGE, DUPLICATE, **diag)
    if label == ISOMER:
        return StageResult(stage=STAGE, passed=True, reason=ISOMER, diagnostics=diag)
    if label == NEAR_FAMILY:
        return StageResult(stage=STAGE, passed=True, reason=NEAR_FAMILY,
                           diagnostics=diag, soft_penalty=near_family_penalty)
    return StageResult.ok(STAGE, **diag)


def deduplicate_batch(
    structures: Sequence,
    thresholds: Optional[NoveltyThresholds] = None,
) -> List[int]:
    """Indices of the structures to keep, dropping within-batch repeats.

    The first occurrence of each distinct structure is kept.
    """
    t = thresholds or DEFAULT_THRESHOLDS.novelty
    matcher = _matcher(t.batch_ltol, t.batch_stol, t.batch_angle_tol)

    keep: List[int] = []
    kept_structures: List = []
    for i, s in enumerate(structures):
        duplicate = False
        for k in kept_structures:
            try:
                if matcher.fit(s, k):
                    duplicate = True
                    break
            except Exception:
                continue
        if not duplicate:
            keep.append(i)
            kept_structures.append(s)
    return keep
