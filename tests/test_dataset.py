"""Integrity checks on the released corpus."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_corpus_counts():
    from ufo_mgen.dataset import corpus_summary

    s = corpus_summary()
    assert s["n_structures"] == 38124
    assert s["n_scaffolds"] == 119
    assert s["n_spacegroups"] == 35
    assert s["splits"] == {"train": 30798, "val": 3676, "test": 3650}
    assert sum(s["splits"].values()) == s["n_structures"]


def test_scaffold_table_agrees_with_corpus():
    from ufo_mgen.dataset import load_corpus, load_scaffolds

    df, sc = load_corpus(), load_scaffolds()
    assert len(sc) == 119
    assert set(sc.scaffold_id) == set(df.scaffold_id.unique())
    assert int(sc.support_count.sum()) == len(df)
    assert int(sc.train_count.sum()) == 30798
    for _, r in sc.iterrows():
        assert r.support_count == r.train_count + r.val_count + r.test_count


def test_scaffold_drep_matches_the_paper_definition():
    """D_rep in the table must be D_L + sum(d_k), recomputable from scratch."""
    from ufo_mgen.wyckoff import compute_Drep
    from ufo_mgen.dataset import load_scaffolds

    sc = load_scaffolds()
    for _, r in sc.head(40).iterrows():
        expected = compute_Drep(
            int(r.space_group), str(r.wyckoff_sequence).split(";"), r.crystal_system
        )
        assert int(r.D_rep) == expected, r.scaffold_id


def test_rho_is_consistent():
    from ufo_mgen.dataset import load_scaffolds

    sc = load_scaffolds()
    for _, r in sc.iterrows():
        expected = r.D_rep / (3 * r.median_n_atoms + 6)
        assert abs(r.rho - expected) < 1e-4, r.scaffold_id
        assert 0.0 < r.rho <= 1.0


def test_complexity_spans_the_stratification_range():
    """The corpus must cover both trivial and 40-orbit scaffolds."""
    from ufo_mgen.dataset import load_scaffolds

    sc = load_scaffolds()
    assert sc.D_rep.min() == 1
    assert sc.D_rep.max() >= 100
    assert sc.K.min() == 2 and sc.K.max() == 40
    assert sc.crystal_system.nunique() == 7


def test_every_corpus_row_decodes():
    """The stored representation must be enough to rebuild the crystal."""
    from ufo_mgen.dataset import load_corpus, structure_vector
    from ufo_mgen.wyckoff import decode_structure

    df = load_corpus(split="test")
    for _, row in df.sample(25, random_state=0).iterrows():
        s = decode_structure(
            structure_vector(row), int(row.spacegroup), str(row.wyckoff_sequence)
        )
        assert len(s) > 0
        assert all(0.0 <= c < 1.0 + 1e-9 for site in s for c in site.frac_coords)


def test_free_coordinate_count_matches_scaffold_dof():
    from ufo_mgen.dataset import free_coords_of, load_corpus
    from ufo_mgen.wyckoff import detect_special_position

    df = load_corpus(split="test")
    for _, row in df.sample(15, random_state=1).iterrows():
        tokens = [t for t in str(row.wyckoff_sequence).split(";") if t]
        expected = sum(
            detect_special_position(int(row.spacegroup), t).dof for t in tokens
        )
        assert len(free_coords_of(row)) == expected


def test_no_local_paths_leaked_into_the_corpus():
    from ufo_mgen.dataset import load_corpus

    df = load_corpus().head(2000)
    for col in df.columns:
        if df[col].dtype == object:
            assert not df[col].astype(str).str.contains("/home/").any(), col


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
