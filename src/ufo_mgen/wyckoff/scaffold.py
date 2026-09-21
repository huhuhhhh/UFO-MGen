"""Wyckoff scaffold parsing and representation-dimension utilities."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

#: Free lattice parameters per crystal system (``D_L``).
CRYSTAL_D_L: Dict[str, int] = {
    "triclinic": 6,
    "monoclinic": 4,
    "orthorhombic": 3,
    "tetragonal": 2,
    "trigonal": 2,
    "hexagonal": 2,
    "cubic": 1,
}

_ORBIT_RE = re.compile(r"(\d+)([A-Za-z])")
_GROUP_DOF_CACHE: Dict[int, Dict[str, int]] = {}


def wyckoff_dof_map(spacegroup: int) -> Dict[str, int]:
    """Return ``{wyckoff_letter: free_dof}`` for a space group, via pyxtal."""
    sg = int(spacegroup)
    cached = _GROUP_DOF_CACHE.get(sg)
    if cached is not None:
        return cached
    from pyxtal.symmetry import Group  # imported lazily: pyxtal is slow to load

    group = Group(sg)
    dof = {wp.letter.lower(): int(wp.get_dof()) for wp in group.Wyckoff_positions}
    _GROUP_DOF_CACHE[sg] = dof
    return dof


def parse_wyckoff_letter(symbol: str) -> str:
    """``"4a" -> "a"``, ``"2i" -> "i"``. Raises ``ValueError`` on a bad token."""
    for ch in reversed(str(symbol).strip()):
        if ch.isalpha():
            return ch.lower()
    raise ValueError(f"Invalid Wyckoff symbol: {symbol!r}")


def parse_scaffold(scaffold: str) -> List[Tuple[int, str]]:
    """Split ``"1a;2c;3g"`` into ``[(1, 'a'), (2, 'c'), (3, 'g')]``."""
    out: List[Tuple[int, str]] = []
    for raw in str(scaffold).split(";"):
        raw = raw.strip()
        if not raw:
            continue
        m = _ORBIT_RE.fullmatch(raw)
        if not m:
            raise ValueError(f"Invalid orbit token: {raw!r}")
        out.append((int(m.group(1)), m.group(2).lower()))
    return out


def build_scaffold(wyckoff_symbols: Iterable[str]) -> str:
    """Join per-orbit Wyckoff symbols into the ``;``-separated scaffold string."""
    return ";".join(str(w).strip() for w in wyckoff_symbols)


def compute_Drep(
    spacegroup: int,
    wyckoff_symbols: Sequence[str],
    crystal_system: str,
) -> int:
    """``D_rep = D_L + sum_k d_k``."""
    dof_map = wyckoff_dof_map(spacegroup)
    d_free_sum = 0
    for symbol in wyckoff_symbols:
        letter = parse_wyckoff_letter(symbol)
        if letter not in dof_map:
            raise KeyError(f"Wyckoff letter {letter!r} not in SG {int(spacegroup)} table")
        d_free_sum += dof_map[letter]
    d_l = CRYSTAL_D_L.get(str(crystal_system).lower(), 6)
    return int(d_l + d_free_sum)


def compute_rho(d_rep: int, n_atoms: int) -> float:
    """``rho = D_rep / (3N + 6)`` -- the representation compression ratio."""
    return float(int(d_rep) / (3 * int(n_atoms) + 6))

