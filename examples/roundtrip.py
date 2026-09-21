"""Encode and decode a crystal through the Wyckoff representation."""

from __future__ import annotations

import sys

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ufo_mgen.wyckoff import decode_structure, encode_cif, encode_structure


def reference_structures():
    from pymatgen.core import Lattice, Structure

    return [
        ("NaCl", Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]])),
        ("SrTiO3", Structure.from_spacegroup(
            "Pm-3m", Lattice.cubic(3.905), ["Sr", "Ti", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]])),
        ("TiO2 rutile", Structure.from_spacegroup(
            "P4_2/mnm", Lattice.tetragonal(4.594, 2.959), ["Ti", "O"],
            [[0, 0, 0], [0.305, 0.305, 0]])),
    ]


def roundtrip(label, enc) -> None:
    vec = list(enc.lattice) + list(enc.free_coords)
    back = decode_structure(vec, enc.spacegroup, enc.scaffold, enc.orbit_species)
    sg_back = SpacegroupAnalyzer(back, symprec=0.01).get_space_group_number()

    print(f"\n=== {label} ===")
    print(f"  representation      {len(vec)} numbers "
          f"({enc.D_rep} free) vs {3 * enc.n_atoms + 6} for the atom list")
    print(f"  formula             {enc.formula}  ->  {back.composition.reduced_formula}")
    print(f"  sites               {enc.n_atoms}  ->  {len(back)}")
    print(f"  space group         {enc.spacegroup}  ->  {sg_back}"
          f"   {'OK' if sg_back == enc.spacegroup else 'CHANGED'}")
    print(f"  lattice a           {enc.lattice[0]:.4f}  ->  {back.lattice.a:.4f}")
    print(f"  scaffold            {enc.scaffold}")


def main() -> int:
    paths = sys.argv[1:]
    if paths:
        for p in paths:
            roundtrip(p, encode_cif(p))
    else:
        print("(no CIF given -- using built-in reference structures)")
        for label, s in reference_structures():
            roundtrip(label, encode_structure(s))

    print("\nLattice projection puts a generated cell back on its crystal-system")
    print("manifold, which is what keeps a sampled structure on its target group:")
    from ufo_mgen.wyckoff import project_lattice_by_spacegroup
    raw = [5.01, 4.98, 5.03, 89.7, 90.2, 90.1]
    print(f"  raw        {[round(v, 2) for v in raw]}")
    print(f"  SG 225     {[round(float(v), 2) for v in project_lattice_by_spacegroup(raw, 225)]}  (cubic)")
    print(f"  SG 123     {[round(float(v), 2) for v in project_lattice_by_spacegroup(raw, 123)]}  (tetragonal)")
    print(f"  SG 191     {[round(float(v), 2) for v in project_lattice_by_spacegroup(raw, 191)]}  (hexagonal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
