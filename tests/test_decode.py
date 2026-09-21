"""Tests for decoding: Wyckoff representation -> crystal."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _structures():
    from pymatgen.core import Lattice, Structure

    return [
        ("NaCl", 225, Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]])),
        ("SrTiO3", 221, Structure.from_spacegroup(
            "Pm-3m", Lattice.cubic(3.905), ["Sr", "Ti", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]])),
        ("rutile", 136, Structure.from_spacegroup(
            "P4_2/mnm", Lattice.tetragonal(4.594, 2.959), ["Ti", "O"],
            [[0, 0, 0], [0.305, 0.305, 0]])),
    ]


def test_roundtrip_preserves_symmetry_composition_and_lattice():
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from ufo_mgen.wyckoff import decode_structure, encode_structure

    for label, sg, s in _structures():
        enc = encode_structure(s)
        back = decode_structure(
            list(enc.lattice) + list(enc.free_coords),
            enc.spacegroup, enc.scaffold, enc.orbit_species,
        )
        assert back.composition.reduced_formula == enc.formula, label
        assert len(back) == enc.n_atoms, label
        assert SpacegroupAnalyzer(back, symprec=0.01).get_space_group_number() == sg, label
        assert abs(back.lattice.a - enc.lattice[0]) < 1e-6, label


def test_decode_without_species_uses_placeholders():
    from ufo_mgen.wyckoff import decode_structure

    s = decode_structure([4.0, 4.0, 4.0, 90.0, 90.0, 90.0], 221, "1a;1b;3c")
    assert len(s) == 5
    assert len({str(site.specie) for site in s}) == 3


def test_decode_rejects_short_vector():
    """Too few free coordinates must raise a clear error, not IndexError."""
    from ufo_mgen.wyckoff import decode_structure

    try:
        decode_structure([4.6, 4.6, 3.0, 90.0, 90.0, 90.0], 136, "2a;4f", ["Ti", "O"])
    except ValueError as exc:
        assert "too short" in str(exc) and "free coordinate" in str(exc)
        return
    raise AssertionError("expected ValueError for a vector missing free coordinates")


def test_decode_rejects_short_species_list():
    from ufo_mgen.wyckoff import decode_structure

    try:
        decode_structure([4.0, 4.0, 4.0, 90.0, 90.0, 90.0], 221, "1a;1b;3c", ["Sr"])
    except ValueError:
        return
    raise AssertionError("expected ValueError for a species list shorter than the scaffold")


def test_free_coordinates_are_wrapped():
    """A coordinate given as 1.7 must wrap into the cell, not escape it."""
    from ufo_mgen.wyckoff import decode_structure

    s = decode_structure([4.6, 4.6, 3.0, 90.0, 90.0, 90.0, 1.7], 136, "2a;4f", ["Ti", "O"])
    assert all((0.0 <= c < 1.0 + 1e-9) for site in s for c in site.frac_coords)


def test_lattice_projection_enforces_crystal_system():
    from ufo_mgen.wyckoff import project_lattice_by_spacegroup

    raw = [5.01, 4.98, 5.03, 89.7, 90.2, 90.1]

    a, b, c, al, be, ga = project_lattice_by_spacegroup(raw, 225)   # cubic
    assert a == b == c and (al, be, ga) == (90.0, 90.0, 90.0)

    a, b, c, al, be, ga = project_lattice_by_spacegroup(raw, 123)   # tetragonal
    assert a == b and c != a and (al, be, ga) == (90.0, 90.0, 90.0)

    a, b, c, al, be, ga = project_lattice_by_spacegroup(raw, 191)   # hexagonal
    assert a == b and ga == 120.0

    out = project_lattice_by_spacegroup(raw, 1)                     # triclinic: untouched
    assert [round(float(v), 4) for v in out] == [round(v, 4) for v in raw]


def test_projection_batch_wraps_coordinates():
    import numpy as np

    from ufo_mgen.wyckoff import project_raw_samples_by_spacegroup

    raw = np.array([[5.0, 4.9, 5.1, 91.0, 89.0, 90.0, 1.4, -0.3]], dtype=np.float32)
    out = project_raw_samples_by_spacegroup(raw, 225)
    assert out[0, 0] == out[0, 1] == out[0, 2]
    assert all(0.0 <= v < 1.0 for v in out[0, 6:])


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except Exception as exc:
            failed += 1; print(f"  FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
