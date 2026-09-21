"""Wyckoff special-position and free-DOF utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .scaffold import parse_wyckoff_letter, wyckoff_dof_map


@dataclass(frozen=True)
class SpecialPositionInfo:
    """Result of a special-position query for one orbit."""

    spacegroup: int
    letter: str
    dof: int
    general_dof: int
    is_special: bool
    multiplicity: int | None = None

    @property
    def n_pinned(self) -> int:
        """Number of coordinates fixed by site symmetry."""
        return int(self.general_dof - self.dof)


def general_position_dof(spacegroup: int) -> int:
    """Free DOF of the general position (the maximum over all letters)."""
    dof_map = wyckoff_dof_map(spacegroup)
    return int(max(dof_map.values())) if dof_map else 3


def detect_special_position(spacegroup: int, wyckoff_symbol: str) -> SpecialPositionInfo:
    """Classify one Wyckoff orbit as general or special."""
    letter = parse_wyckoff_letter(wyckoff_symbol)
    dof_map = wyckoff_dof_map(spacegroup)
    if letter not in dof_map:
        raise KeyError(f"Wyckoff letter {letter!r} not in SG {int(spacegroup)} table")
    dof = int(dof_map[letter])
    gdof = general_position_dof(spacegroup)
    mult = None
    token = str(wyckoff_symbol).strip()
    if token and token[0].isdigit():
        digits = "".join(ch for ch in token if ch.isdigit())
        mult = int(digits) if digits else None
    return SpecialPositionInfo(
        spacegroup=int(spacegroup),
        letter=letter,
        dof=dof,
        general_dof=gdof,
        is_special=dof < gdof,
        multiplicity=mult,
    )


def orbit_dof_vector(spacegroup: int, scaffold: str) -> Dict[str, int]:
    """Map each orbit token of a scaffold to its free DOF."""
    out: Dict[str, int] = {}
    for raw in str(scaffold).split(";"):
        raw = raw.strip()
        if raw:
            out[raw] = detect_special_position(spacegroup, raw).dof
    return out
