"""Canonicalization utilities for equivalent Wyckoff scaffold labels."""

from __future__ import annotations

import re
from typing import List

#: Centring letters that multiply site multiplicities, and their factors.
CENTERING_FACTORS = {"A": 2, "B": 2, "C": 2, "I": 2, "F": 4, "R": 3}

_TOKEN_RE = re.compile(r"(\d+)([A-Za-z])")
_LEADING_INT_RE = re.compile(r"(\d+)")


def centering_factor_for_sg(sg: int) -> int:
    """Return the centring multiplicity factor of a space group (1 if primitive)."""
    from pyxtal.symmetry import Group

    symbol = str(Group(int(sg)).symbol)
    return CENTERING_FACTORS.get(symbol[0].upper(), 1)


def canonicalize_scaffold(sg: int, scaffold: str, *, reduce_centering: bool = True) -> str:
    """Canonicalise a raw scaffold string for space group ``sg``.

    Orbit tokens are lower-cased, optionally divided by the centring factor, and
    sorted by ``(letter, multiplicity)``. The result is the canonical scaffold
    family identifier used throughout the release.
    """
    factor = centering_factor_for_sg(sg) if reduce_centering else 1
    tokens: List[str] = []
    for raw in str(scaffold).split(";"):
        raw = raw.strip()
        if not raw:
            continue
        m = _TOKEN_RE.fullmatch(raw)
        if m:
            mult = int(m.group(1))
            letter = m.group(2).lower()
            if reduce_centering and factor > 1 and mult % factor == 0:
                mult //= factor
            tokens.append(f"{mult}{letter}")
        else:
            tokens.append(raw.lower())

    def sort_key(tok: str):
        lead = _LEADING_INT_RE.match(tok)
        return (tok[-1], int(lead.group(1)) if lead else 0, tok)

    tokens.sort(key=sort_key)
    return ";".join(tokens)


def scaffold_family_equivalent(sg: int, scaffold_a: str, scaffold_b: str) -> bool:
    """Whether two scaffolds belong to the same canonical family.

    Beyond plain canonicalisation this folds in the SG-227 (Fd-3m) origin-choice
    equivalences, where ``2a`` and ``2b`` describe the same orbit under the two
    standard origin settings.
    """
    canon_a = canonicalize_scaffold(sg, scaffold_a, reduce_centering=True)
    canon_b = canonicalize_scaffold(sg, scaffold_b, reduce_centering=True)
    if canon_a == canon_b:
        return True
    if int(sg) == 227:
        classes = [
            {
                canonicalize_scaffold(227, "2a;4d;8e", reduce_centering=True),
                canonicalize_scaffold(227, "2b;4c;8e", reduce_centering=True),
            },
            {
                canonicalize_scaffold(227, "2a;4c;4d;12f", reduce_centering=True),
                canonicalize_scaffold(227, "2b;4c;4d;12f", reduce_centering=True),
            },
        ]
        for klass in classes:
            if canon_a in klass and canon_b in klass:
                return True
    return False


def canonicalize_wyckoff(sg: int, wyckoff_symbols, **kwargs) -> str:
    """Convenience wrapper: canonicalise a sequence of per-orbit symbols."""
    if isinstance(wyckoff_symbols, str):
        scaffold = wyckoff_symbols
    else:
        scaffold = ";".join(str(w) for w in wyckoff_symbols)
    return canonicalize_scaffold(sg, scaffold, **kwargs)
