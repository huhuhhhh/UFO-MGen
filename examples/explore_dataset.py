"""Minimal example for inspecting the processed UFO-MGen corpus."""

from __future__ import annotations

from ufo_mgen.dataset import (
    corpus_summary,
    load_corpus,
    load_scaffolds,
    structure_vector,
)
from ufo_mgen.wyckoff import decode_structure


def main() -> int:
    s = corpus_summary()
    print("=== corpus ===")
    print(f"  structures     {s['n_structures']:,}")
    print(f"  scaffolds      {s['n_scaffolds']}")
    print(f"  space groups   {s['n_spacegroups']}")
    for k in ("train", "val", "test"):
        print(f"  {k:<14} {s['splits'].get(k, 0):,}")

    sc = load_scaffolds()
    print(f"\n=== scaffold complexity ({len(sc)} scaffolds) ===")
    print(f"  D_rep   {sc.D_rep.min()} … {sc.D_rep.max()}   (median {int(sc.D_rep.median())})")
    print(f"  rho     {sc.rho.min():.4f} … {sc.rho.max():.4f}")
    print(f"  K       {sc.K.min()} … {sc.K.max()} orbits")
    print("\n  by crystal system:")
    for system, n in sc.crystal_system.value_counts().items():
        sub = sc[sc.crystal_system == system]
        print(f"    {system:<14} {n:>3} scaffolds   median D_rep {int(sub.D_rep.median()):>3}")

    print("\n  simplest and most complex:")
    for _, r in sc.nsmallest(2, "D_rep").iterrows():
        print(f"    D_rep {r.D_rep:>3}  rho {r.rho:.4f}  {r.scaffold_id}")
    for _, r in sc.nlargest(2, "D_rep").iterrows():
        print(f"    D_rep {r.D_rep:>3}  rho {r.rho:.4f}  {r.scaffold_id[:60]}")

    print("\n=== geometry-only decode from the corpus ===")
    df = load_corpus(split="test")
    row = df.iloc[0]
    vec = structure_vector(row)
    sg = int(row["spacegroup"])
    scaffold = str(row["wyckoff_sequence"])
    structure = decode_structure(vec, sg, scaffold)
    print(f"  {row['formula']}  SG {sg}  scaffold {scaffold}")
    print(f"  stored vector: {len(vec)} values -> decoded {len(structure)} sites")
    print("  note: orbit chemistry is not supplied here, so deterministic placeholder species are used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
