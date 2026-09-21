"""Decode UFO-MGen Wyckoff representations into crystal structures."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .lattice_projection import project_lattice_by_spacegroup

#: Placeholder species used when decoding geometry without a chemistry model.
_PLACEHOLDER_SPECIES = [
    "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr",
]


def strip_scaffold_prefix(scaffold: str) -> str:
    """Drop a leading ``"SG<number>|"`` namespace from a scaffold key."""
    s = str(scaffold)
    return s.split("|", 1)[1] if "|" in s else s


def placeholder_species_for_orbit(orbit_idx: int) -> str:
    """Deterministic placeholder element for geometry-only decoding."""
    return _PLACEHOLDER_SPECIES[int(orbit_idx) % len(_PLACEHOLDER_SPECIES)]



def _expected_free_dof(sg: int, letters) -> int:
    """Total free coordinates a scaffold needs, for error messages."""
    from pyxtal.symmetry import Wyckoff_position

    total = 0
    for token in letters:
        letter = token[-1] if len(token) > 1 else token
        total += int(Wyckoff_position.from_group_and_letter(int(sg), letter).get_dof())
    return total

def decode_structure(
    vec: Sequence[float],
    sg: int,
    scaffold: str,
    orbit_species: Optional[Sequence[str]] = None,
    *,
    project_lattice: bool = False,
):
    """Decode a raw representation vector into a pymatgen ``Structure``.

    Parameters
    ----------
    vec
        ``[a, b, c, alpha, beta, gamma]`` followed by the concatenated free
        Wyckoff coordinates, orbit by orbit, in scaffold order.
    sg
        Target space group number. The decoded structure lies in this group by
        construction.
    scaffold
        ``;``-separated orbit tokens, optionally ``"SG<n>|"``-prefixed.
    orbit_species
        One species per orbit. If omitted, deterministic placeholder elements
        are used, which is appropriate for geometry-only evaluation.
    project_lattice
        Project the lattice onto its crystal-system manifold before decoding.
    """
    from pymatgen.core import Lattice, Structure
    from pyxtal.symmetry import Wyckoff_position

    vec = np.asarray(vec, dtype=float).reshape(-1)
    letters = strip_scaffold_prefix(scaffold).split(";")
    letters = [t.strip() for t in letters if t.strip()]

    lat6 = vec[:6]
    if project_lattice:
        lat6 = project_lattice_by_spacegroup(lat6, sg)
    a, b, c, alpha, beta, gamma = (float(v) for v in lat6)
    lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

    species: List[str] = []
    coords: List[np.ndarray] = []
    cursor = 6
    for orbit_idx, token in enumerate(letters):
        letter = token[-1] if len(token) > 1 else token
        wp = Wyckoff_position.from_group_and_letter(int(sg), letter)
        dof = int(wp.get_dof())
        if cursor + dof > vec.shape[0]:
            raise ValueError(
                f"representation vector is too short for scaffold {scaffold!r} "
                f"in SG {int(sg)}: orbit {orbit_idx} ({token}) needs {dof} free "
                f"coordinate(s) at offset {cursor}, but the vector holds "
                f"{vec.shape[0]} numbers. Expected 6 lattice parameters plus "
                f"{_expected_free_dof(sg, letters)} free coordinate(s)."
            )
        free = np.asarray(vec[cursor: cursor + dof], dtype=float)
        cursor += dof
        pos = np.asarray(
            wp.get_position_from_free_xyzs(np.mod(free, 1.0)), dtype=float
        ).reshape(-1)
        orbit = wp.apply_ops(pos)
        if orbit_species is not None and orbit_idx < len(orbit_species):
            name = str(orbit_species[orbit_idx])
        elif orbit_species is not None:
            raise ValueError(
                f"Missing orbit species for orbit {orbit_idx} in scaffold {scaffold!r}"
            )
        else:
            name = placeholder_species_for_orbit(orbit_idx)
        for site in orbit:
            coords.append(np.mod(np.asarray(site, dtype=float), 1.0))
            species.append(name)

    return Structure(lattice, species, coords, coords_are_cartesian=False)


def decode_to_cif(vec, sg, scaffold, orbit_species=None, path=None, **kwargs) -> str:
    """Decode and return CIF text, optionally writing it to ``path``."""
    structure = decode_structure(vec, sg, scaffold, orbit_species, **kwargs)
    cif = structure.to(fmt="cif")
    if path is not None:
        with open(path, "w") as fh:
            fh.write(cif)
    return cif
