"""Encode crystal structures into the UFO-MGen Wyckoff representation."""

from __future__ import annotations

import io
import warnings
from contextlib import redirect_stderr
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .canonicalize import canonicalize_scaffold
from .scaffold import (
    CRYSTAL_D_L,
    build_scaffold,
    compute_Drep,
    compute_rho,
)
from .special_positions import detect_special_position

#: Manuscript symmetry-analysis defaults.
DEFAULT_SYMPREC = 0.01
DEFAULT_ANGLE_TOL = 5.0


@dataclass
class WyckoffEncoding:
    """The UFO-MGen representation of one crystal."""

    spacegroup: int
    crystal_system: str
    n_atoms: int
    K: int
    wyckoff_symbols: List[str]
    scaffold: str
    canonical_scaffold: str
    orbit_species: List[str]
    d_L: int
    d_free_sum: int
    D_rep: int
    rho: float
    lattice: List[float]
    free_coords: List[float] = field(default_factory=list)
    orbit_dof: List[int] = field(default_factory=list)
    is_special: List[bool] = field(default_factory=list)
    formula: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def encode_structure(
    structure,
    *,
    symprec: float = DEFAULT_SYMPREC,
    angle_tolerance: float = DEFAULT_ANGLE_TOL,
) -> WyckoffEncoding:
    """Encode a pymatgen ``Structure`` into its Wyckoff representation."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stderr(io.StringIO()):
            sga0 = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
            refined = sga0.get_refined_structure()
            sga = SpacegroupAnalyzer(refined, symprec=symprec, angle_tolerance=angle_tolerance)
            symm = sga.get_symmetrized_structure()
            sg = int(sga.get_space_group_number())
            crystal_system = str(sga.get_crystal_system())

    wyckoff_symbols = [str(w) for w in symm.wyckoff_symbols]
    scaffold = build_scaffold(wyckoff_symbols)
    canonical = canonicalize_scaffold(sg, scaffold)

    orbit_species: List[str] = []
    for group in symm.equivalent_sites:
        orbit_species.append(str(group[0].specie))

    orbit_dof: List[int] = []
    is_special: List[bool] = []
    for symbol in wyckoff_symbols:
        info = detect_special_position(sg, symbol)
        orbit_dof.append(info.dof)
        is_special.append(info.is_special)

    n_atoms = int(len(refined))
    d_free_sum = int(sum(orbit_dof))
    d_l = int(CRYSTAL_D_L.get(crystal_system.lower(), 6))
    d_rep = compute_Drep(sg, wyckoff_symbols, crystal_system)
    rho = compute_rho(d_rep, n_atoms)

    lat = refined.lattice
    lattice = [
        float(lat.a), float(lat.b), float(lat.c),
        float(lat.alpha), float(lat.beta), float(lat.gamma),
    ]

    free_coords: List[float] = []
    for group, dof in zip(symm.equivalent_sites, orbit_dof):
        rep = np.asarray(group[0].frac_coords, dtype=float)
        free_coords.extend(float(v) for v in rep[:dof])

    return WyckoffEncoding(
        spacegroup=sg,
        crystal_system=crystal_system,
        n_atoms=n_atoms,
        K=int(len(wyckoff_symbols)),
        wyckoff_symbols=wyckoff_symbols,
        scaffold=scaffold,
        canonical_scaffold=canonical,
        orbit_species=orbit_species,
        d_L=d_l,
        d_free_sum=d_free_sum,
        D_rep=int(d_rep),
        rho=float(rho),
        lattice=lattice,
        free_coords=free_coords,
        orbit_dof=orbit_dof,
        is_special=is_special,
        formula=str(refined.composition.reduced_formula),
    )


def encode_cif(path, **kwargs) -> WyckoffEncoding:
    """Read a CIF from disk and encode it."""
    from pymatgen.core import Structure

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = Structure.from_file(str(path))
    return encode_structure(structure, **kwargs)


def encode_many(
    structures: Sequence,
    **kwargs,
) -> List[WyckoffEncoding]:
    """Encode a sequence of structures, skipping ones that fail analysis."""
    out: List[WyckoffEncoding] = []
    for s in structures:
        try:
            out.append(encode_structure(s, **kwargs))
        except Exception:
            continue
    return out
