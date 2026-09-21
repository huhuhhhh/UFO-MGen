"""Minimal post-generation screening example on toy structures."""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

from ufo_mgen.mlip import BACKENDS, BACKEND_MODELS, MECHANICAL_BACKEND
from ufo_mgen.screening import (
    DEFAULT_THRESHOLDS,
    ScreeningFunnel,
    chemistry_stage,
    deduplicate_batch,
    geometry_stage,
    novelty_stage,
    scaffold_stage,
    summarise,
)


def rocksalt(a: float, species):
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), list(species),
                                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def known_reference():
    """Stand-in for a reference database: one structure already on record."""
    return [rocksalt(6.60, ["K", "Br"])]


def cells():
    """A sound crystal, plus one failure of each kind the funnel is meant to catch."""
    perovskite = Structure.from_spacegroup("Pm-3m", Lattice.cubic(3.905),
                                           ["Sr", "Ti", "O"],
                                           [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]])
    collapsed = Structure(Lattice.cubic(3.0), ["Na", "Cl"], [[0, 0, 0], [0.02, 0, 0]])
    return [
        ("sound cell",      rocksalt(5.64, ["Na", "Cl"]), 225, "4a;4b"),
        ("collapsed cell",  collapsed,                    None, None),
        ("over-expanded",   rocksalt(40.0, ["Na", "Cl"]), 225, "4a;4b"),
        ("wrong target",    perovskite,                   225, "4a;4b"),
        ("already known",   rocksalt(6.60, ["K", "Br"]),  225, "4a;4b"),
    ]


def main() -> int:
    print("=== protocol ===")
    t = DEFAULT_THRESHOLDS
    print(f"  geometry   min distance {t.geometry.min_valid_dist} A, "
          f"pair ratio {t.geometry.min_pair_ratio}, "
          f"volume scale {t.geometry.min_vpa_scale}-{t.geometry.max_vpa_scale}")
    print(f"  novelty    ltol {t.novelty.ltol}, stol {t.novelty.stol}, "
          f"angle {t.novelty.angle_tol} deg")
    print(f"  relax      fmax {t.relax.fmax}, {t.relax.steps} steps")
    print(f"  quick Ef prescreen enabled: {t.quick_ef_prescreen.enabled}"
          f"   <- opt-in; it trades recall for compute")

    reference = known_reference()
    funnel = (ScreeningFunnel()
              .add(scaffold_stage())
              .add(geometry_stage())
              .add(chemistry_stage())
              .add(novelty_stage(reference)))

    print("\n=== running the structural funnel ===")
    audits = []
    for label, structure, sg, scaffold in cells():
        audit = funnel.run(
            {"structure": structure, "target_spacegroup": sg, "target_scaffold": scaffold},
            candidate_id=label,
        )
        audits.append(audit)
        verdict = "pass" if audit.passed else f"reject at {audit.rejected_at}"
        print(f"  {label:<16} {verdict:<34} {audit.reason}")

    print("\n=== per-stage tally for this toy run ===")
    s = summarise(audits)
    print(f"  {'stage':<26}{'passed':>8}{'failed':>8}{'skipped':>9}")
    for stage, row in s["by_stage"].items():
        print(f"  {stage:<26}{row['passed']:>8}{row['failed']:>8}{row['skipped']:>9}")
    print(f"\n  {s['n_passed']}/{s['n_candidates']} candidates survived")

    print("\n=== batch deduplication ===")
    batch = [cells()[0][1], cells()[3][1], cells()[0][1]]
    print(f"  three structures, the first and third identical -> "
          f"keep indices {deduplicate_batch(batch)}")

    print("\n=== physical validation backends ===")
    for b in BACKENDS:
        mark = "  <- final K/G/E for UFO-Mech" if b == MECHANICAL_BACKEND else ""
        print(f"  {b:<12} {BACKEND_MODELS[b]}{mark}")
    print("\n  Weights are not redistributed; install the upstream package and")
    print("  it fetches its own. Nothing above required one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
