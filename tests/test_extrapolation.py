"""Tests for the space-group extrapolation generation path.

These tests cover the protocol and seed construction only. Manuscript result
counts and conclusion tables are intentionally not duplicated in the codebase.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_seed_building_works_for_dense_groups():
    from ufo_mgen.generation.generate_extrapolation import pyxtal_seed_scaffolds

    seeds = pyxtal_seed_scaffolds(225, n_orbits=3, max_seeds=3)
    assert len(seeds) == 3
    assert all(len(s.split(";")) == 3 for s in seeds)
    assert len(set(seeds)) == len(seeds)


def test_seed_building_works_for_sparse_groups():
    """Sparse Wyckoff tables should still yield candidate scaffold seeds."""
    from ufo_mgen.generation.generate_extrapolation import pyxtal_seed_scaffolds

    for sg in (92, 1, 181):
        seeds = pyxtal_seed_scaffolds(sg, n_orbits=3, max_seeds=3)
        assert seeds, f"SG {sg} produced no seeds"
        assert all(len(s.split(";")) == 3 for s in seeds)
        assert len(set(seeds)) == len(seeds)


def test_held_out_spacegroups():
    from ufo_mgen.generation.generate_extrapolation import held_out_spacegroups

    held = held_out_spacegroups([1, 2, 225])
    assert len(held) == 227
    assert 1 not in held and 225 not in held
    assert 92 in held


def test_extrapolation_context_mode_is_identity_free():
    from ufo_mgen.generation.generate_extrapolation import EXTRAPOLATION_CONTEXT_MODE

    assert EXTRAPOLATION_CONTEXT_MODE == "topology_chem_only"


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
