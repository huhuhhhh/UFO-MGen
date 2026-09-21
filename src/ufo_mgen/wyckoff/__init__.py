"""Encoding, decoding, and scaffold utilities for the Wyckoff representation."""

from .canonicalize import (
    canonicalize_scaffold,
    canonicalize_wyckoff,
    centering_factor_for_sg,
    scaffold_family_equivalent,
)
from .decode import decode_structure, decode_to_cif, strip_scaffold_prefix
from .encode import (
    DEFAULT_ANGLE_TOL,
    DEFAULT_SYMPREC,
    WyckoffEncoding,
    encode_cif,
    encode_many,
    encode_structure,
)
from .lattice_projection import (
    crystal_system_from_spacegroup,
    project_lattice_by_spacegroup,
    project_raw_samples_by_spacegroup,
)
from .scaffold import (
    CRYSTAL_D_L,
    build_scaffold,
    compute_Drep,
    compute_rho,
    parse_scaffold,
    parse_wyckoff_letter,
    wyckoff_dof_map,
)
from .special_positions import (
    SpecialPositionInfo,
    detect_special_position,
    general_position_dof,
    orbit_dof_vector,
)

__all__ = [
    "WyckoffEncoding", "encode_structure", "encode_cif", "encode_many",
    "DEFAULT_SYMPREC", "DEFAULT_ANGLE_TOL",
    "build_scaffold", "parse_scaffold", "parse_wyckoff_letter",
    "compute_Drep", "compute_rho", "wyckoff_dof_map",
    "CRYSTAL_D_L",
    "detect_special_position", "general_position_dof", "orbit_dof_vector",
    "SpecialPositionInfo",
    "canonicalize_scaffold", "canonicalize_wyckoff",
    "scaffold_family_equivalent", "centering_factor_for_sg",
    "decode_structure", "decode_to_cif", "strip_scaffold_prefix",
    "project_lattice_by_spacegroup", "project_raw_samples_by_spacegroup",
    "crystal_system_from_spacegroup",
]
