"""Crystal-to-Wyckoff encoding example."""

from __future__ import annotations

import sys

from ufo_mgen.wyckoff import encode_cif, encode_structure


def reference_structures():
    """Three crystals spanning very different representation sizes."""
    from pymatgen.core import Lattice, Structure

    return [
        ("NaCl (rock salt)", Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]])),
        ("SrTiO3 (perovskite)", Structure.from_spacegroup(
            "Pm-3m", Lattice.cubic(3.905), ["Sr", "Ti", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]])),
        ("TiO2 (rutile)", Structure.from_spacegroup(
            "P4_2/mnm", Lattice.tetragonal(4.594, 2.959), ["Ti", "O"],
            [[0, 0, 0], [0.305, 0.305, 0]])),
    ]


def report(label, enc) -> None:
    print(f"\n=== {label} ===")
    print(f"  formula             {enc.formula}")
    print(f"  space group         {enc.spacegroup}  ({enc.crystal_system})")
    print(f"  atoms in cell   N = {enc.n_atoms}")
    print(f"  orbits          K = {enc.K}")
    print(f"  Wyckoff scaffold    {enc.scaffold}")
    print(f"  canonical scaffold  {enc.canonical_scaffold}")
    print(f"  species per orbit   {enc.orbit_species}")
    print(f"  special position?   {enc.is_special}")
    print(f"  free DOF per orbit  {enc.orbit_dof}")
    print(f"  lattice DOF   D_L = {enc.d_L}")
    print(f"  sum of d_k        = {enc.d_free_sum}")
    print(f"  D_rep             = {enc.D_rep}")
    print(f"  rho               = {enc.rho:.4f}"
          f"   ({enc.D_rep} / (3*{enc.n_atoms}+6) = {enc.D_rep}/{3*enc.n_atoms+6})")
    print(f"  free coordinates    {[round(v, 4) for v in enc.free_coords]}")
    serialized_dim = 6 + len(enc.free_coords)
    print(f"  intrinsic DOF     = {enc.D_rep}")
    print(f"  serialized vector = {serialized_dim} values "
          f"(6 lattice values + {len(enc.free_coords)} free coordinates)")


def main() -> int:
    paths = sys.argv[1:]
    if paths:
        for p in paths:
            report(p, encode_cif(p))
    else:
        print("(no CIF given -- encoding built-in reference structures)")
        for label, structure in reference_structures():
            report(label, encode_structure(structure))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
