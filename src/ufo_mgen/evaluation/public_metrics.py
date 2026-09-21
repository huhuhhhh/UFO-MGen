"""Public validity, coverage, and optional S.U.N. evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.io import write_json


def load_structures_from_dir(cif_dir, limit: Optional[int] = None) -> List:
    """Load CIFs from a directory, skipping unreadable files."""
    from pymatgen.core import Structure

    paths = sorted(Path(str(cif_dir)).glob("*.cif"))
    if limit:
        paths = paths[:limit]
    out = []
    for p in paths:
        try:
            out.append(Structure.from_file(str(p)))
        except Exception:
            continue
    return out


def evaluate_generated(
    structures: List,
    *,
    reference_structures: Optional[List] = None,
    run_sun: bool = False,
    sun_sample: int = 1024,
    coverage_thresholds=(0.4, 0.5, 1.0),
) -> Dict[str, object]:
    """Compute the public metric suite for a generated set."""
    from .coverage import compute_coverage_metrics
    from .validity import compute_compositional_validity, compute_structural_validity

    results: Dict[str, object] = {"n_generated": len(structures)}

    struct_valid = compute_structural_validity(structures)
    comp_valid = compute_compositional_validity(structures)
    results["validity"] = {
        "structural": struct_valid,
        "compositional": comp_valid,
    }

    if reference_structures:
        results["coverage"] = compute_coverage_metrics(
            structures, reference_structures, thresholds=list(coverage_thresholds)
        )

    if run_sun:
        if not reference_structures:
            raise ValueError("S.U.N. needs reference_structures for the novelty check")
        from .sun import compute_novelty, compute_uniqueness

        sample = structures[:sun_sample]
        results["sun"] = {
            "n_sampled": len(sample),
            "unique": compute_uniqueness(sample),
            "novel": compute_novelty(sample, reference_structures),
            "note": (
                "Stability requires MLIP relaxation and a convex hull; run "
                "ufo_mgen.evaluation.sun directly with a CHGNet/MACE backend."
            ),
        }

    return results


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UFO-MGen public metric suite")
    p.add_argument("--generated", required=True, help="directory of generated CIFs")
    p.add_argument("--reference", default=None, help="directory of reference CIFs")
    p.add_argument("--output", default="metrics.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run-sun", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    gen = load_structures_from_dir(args.generated, limit=args.limit)
    if not gen:
        raise SystemExit(f"no readable CIFs in {args.generated}")
    ref = load_structures_from_dir(args.reference) if args.reference else None
    results = evaluate_generated(gen, reference_structures=ref, run_sun=args.run_sun)
    write_json(results, args.output)
    print(f"wrote {args.output}")
    for k, v in results.items():
        print(f"  {k}: {v if not isinstance(v, dict) else '...'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
