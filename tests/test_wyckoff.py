"""Tests for the Wyckoff encoder.

    python tests/test_wyckoff.py      # or: pytest tests/test_wyckoff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _nacl():
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _rutile():
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup(
        "P4_2/mnm", Lattice.tetragonal(4.594, 2.959), ["Ti", "O"],
        [[0, 0, 0], [0.305, 0.305, 0]]
    )


def test_import_is_local():
    import ufo_mgen

    origin = Path(ufo_mgen.__file__).resolve()
    assert origin.is_relative_to(SRC), f"imported from {origin}, outside {SRC}"


def test_encode_nacl():
    """Rock salt: every atom sits on a fully pinned special position."""
    from ufo_mgen.wyckoff import encode_structure

    enc = encode_structure(_nacl())
    assert enc.spacegroup == 225
    assert enc.crystal_system == "cubic"
    assert enc.K == 2
    assert set(enc.orbit_species) == {"Na", "Cl"}
    assert all(enc.is_special)
    assert enc.orbit_dof == [0, 0]
    assert enc.d_L == 1              # cubic -> one free lattice parameter
    assert enc.d_free_sum == 0
    assert enc.D_rep == 1
    assert enc.free_coords == []
    assert abs(enc.rho - 1 / (3 * enc.n_atoms + 6)) < 1e-12


def test_encode_rutile_has_one_free_coordinate():
    """Rutile's oxygen sits on 4f, which carries a single free parameter."""
    from ufo_mgen.wyckoff import encode_structure

    enc = encode_structure(_rutile())
    assert enc.spacegroup == 136
    assert enc.crystal_system == "tetragonal"
    assert enc.K == 2
    assert enc.d_L == 2              # tetragonal -> a and c
    assert enc.d_free_sum == 1       # the oxygen x parameter
    assert enc.D_rep == 3
    assert len(enc.free_coords) == 1
    assert 0.0 <= enc.free_coords[0] < 1.0


def test_drep_is_lattice_plus_orbit_dof():
    from ufo_mgen.wyckoff import compute_Drep, encode_structure
    from ufo_mgen.wyckoff.scaffold import CRYSTAL_D_L

    for structure in (_nacl(), _rutile()):
        enc = encode_structure(structure)
        assert enc.D_rep == enc.d_L + enc.d_free_sum
        assert enc.d_L == CRYSTAL_D_L[enc.crystal_system]
        assert compute_Drep(enc.spacegroup, enc.wyckoff_symbols, enc.crystal_system) == enc.D_rep


def test_rho_is_compression_against_atom_list():
    from ufo_mgen.wyckoff import encode_structure

    for structure in (_nacl(), _rutile()):
        enc = encode_structure(structure)
        assert abs(enc.rho - enc.D_rep / (3 * enc.n_atoms + 6)) < 1e-12
        assert enc.rho < 1.0, "representation should be smaller than the atom list"


def test_free_coords_match_declared_dof():
    from ufo_mgen.wyckoff import encode_structure

    enc = encode_structure(_rutile())
    assert len(enc.free_coords) == sum(enc.orbit_dof) == enc.d_free_sum


def test_special_position_detection():
    from ufo_mgen.wyckoff import detect_special_position

    pinned = detect_special_position(225, "4a")
    assert pinned.is_special and pinned.dof == 0 and pinned.n_pinned == 3

    general = detect_special_position(1, "1a")
    assert not general.is_special and general.dof == 3


def test_canonicalization_collapses_centering_and_orbit_order():
    from ufo_mgen.wyckoff import canonicalize_scaffold, scaffold_family_equivalent

    # SG139 is I-centred: reported multiplicities carry a factor of 2
    assert canonicalize_scaffold(139, "8d;4a") == canonicalize_scaffold(139, "4a;8d")
    assert canonicalize_scaffold(139, "8d;4a") == "2a;4d"
    # SG227 origin-choice equivalence
    assert scaffold_family_equivalent(227, "2a;4d;8e", "2b;4c;8e")


def test_scaffold_parsing_roundtrip():
    from ufo_mgen.wyckoff import build_scaffold, parse_scaffold

    scaffold = "1a;2c;3g"
    assert parse_scaffold(scaffold) == [(1, "a"), (2, "c"), (3, "g")]
    assert build_scaffold(["1a", "2c", "3g"]) == scaffold


def test_wyckoff_dof_table_matches_pyxtal():
    from ufo_mgen.wyckoff import wyckoff_dof_map

    dof = wyckoff_dof_map(225)
    assert dof["a"] == 0 and dof["b"] == 0     # fully pinned
    assert max(dof.values()) == 3              # the general position


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
