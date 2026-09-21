"""Tests for UFO-Mech: scoring, weighting, element masking, fine-tune record."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _table():
    import pandas as pd

    return pd.DataFrame([
        (440, 520, 1100, 0.07, True),
        (300, 250, 600, 0.18, True),
        (280, 40, 115, 0.42, True),
        (160, 90, 225, 0.26, True),
        (70, 25, 68, 0.35, True),
        (350, 300, 720, 0.15, False),
    ], columns=["K", "G", "E", "nu", "elastic_stability"])


# ---------------------------------------------------------------- score

def test_k_g_e_are_equally_weighted():
    from ufo_mgen.mech import SCORE_WEIGHTS

    assert SCORE_WEIGHTS == {"log_K": 1.0, "log_G": 1.0, "log_E": 1.0}


def test_incompressible_but_soft_scores_below_hard():
    """The score should reward jointly large K, G and E."""
    from ufo_mgen.mech import add_mechanical_scores

    d = add_mechanical_scores(_table())
    hard = d.iloc[0].mechanical_extreme_score        # K 440, G 520
    dense_soft = d.iloc[2].mechanical_extreme_score  # K 280, G 40
    ordinary = d.iloc[3].mechanical_extreme_score    # K 160, G 90
    assert hard > ordinary > dense_soft


def test_instability_penalty_applies():
    from ufo_mgen.mech import add_mechanical_scores, stability_penalty

    d = add_mechanical_scores(_table())
    assert d.iloc[5].penalty_unstable >= 1.0        # elastic_stability False
    assert d.iloc[0].penalty_unstable == 0.0
    # nu = 0.42 sits inside the valid range, so it must NOT be penalised
    assert d.iloc[2].penalty_unstable == 0.0

    assert stability_penalty(0.25, True) == 0.0
    assert stability_penalty(0.45, True) == 0.0     # inclusive upper bound
    assert stability_penalty(0.46, True) == 0.5     # just outside
    assert stability_penalty(0.25, False) == 1.0
    assert stability_penalty(0.60, True) == 0.5
    assert stability_penalty(-0.1, False) == 1.5


def test_missing_modulus_is_neutral_not_extreme():
    """A NaN must not push an entry into the tail."""
    import numpy as np

    from ufo_mgen.mech import compute_score_from_logs

    stats = {c: {"mean": 0.0, "std": 1.0} for c in ("log_K", "log_G", "log_E")}
    full = compute_score_from_logs(0.0, 0.0, 0.0, stats)
    missing = compute_score_from_logs(np.nan, 0.0, 0.0, stats)
    assert full == missing == 0.0


def test_weights_stay_in_range_and_track_the_score():
    from ufo_mgen.mech import SAMPLE_WEIGHT_RANGE, add_mechanical_scores, bin_and_weight

    d = add_mechanical_scores(_table()).sort_values(
        "mechanical_extreme_score", ascending=False)
    bins, w = bin_and_weight(d.mechanical_extreme_score)
    lo, hi = SAMPLE_WEIGHT_RANGE
    assert w.min() >= lo and w.max() <= hi
    assert w[0] == w.max(), "the top-scoring entry should carry the largest weight"
    assert list(w) == sorted(w, reverse=True), "weights must be monotone in score"


def test_bulk_of_corpus_keeps_weight_one():
    """Steering, not narrowing: the lower half must stay at baseline weight."""
    import numpy as np

    from ufo_mgen.mech import bin_and_weight

    scores = np.linspace(-3, 3, 200)
    _, w = bin_and_weight(scores)
    assert (w[:100] == 1.0).all()
    assert w.max() > 1.0


# ---------------------------------------------------------------- predictor

def test_predictor_architecture():
    from ufo_mgen.mech import TARGET_COLUMNS
    from ufo_mgen.mech.predictor import PRODUCTION_ARCH

    assert TARGET_COLUMNS == ["log_K", "log_G", "log_E"]
    assert PRODUCTION_ARCH["n_estimators"] == 220
    assert PRODUCTION_ARCH["imputation"] == "median"


def test_predictor_fits_and_predicts_three_targets():
    import numpy as np
    import pandas as pd

    from ufo_mgen.mech import TARGET_COLUMNS, fit_predictor, predict_moduli

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(60, 8)), columns=[f"f{i}" for i in range(8)])
    y = pd.DataFrame(rng.normal(size=(60, 3)), columns=TARGET_COLUMNS)
    model = fit_predictor(x, y, random_state=0)
    out = predict_moduli(model, x)
    assert len(out) == 60
    for col in ("K", "G", "E"):
        assert col in out.columns
    assert (out.K > 0).all(), "moduli come back from logs, so must be positive"





# ---------------------------------------------------------------- fine-tune

def test_finetune_settings_are_gentle():
    from ufo_mgen.mech import FINETUNE_SETTINGS as s

    assert s["epochs"] == 5
    assert s["lr"] == 2e-5, "a larger step would overwrite the base chemistry"
    assert s["batch_size"] == 192
    assert s["weighted_sampler"] is True




def test_weighted_sampler_clamps_and_oversamples():
    import numpy as np

    from ufo_mgen.mech import build_weighted_sampler

    sampler = build_weighted_sampler([1.0, 1.0, 5.0, 99.0])
    drawn = np.asarray(list(sampler))
    assert len(drawn) == 4
    assert drawn.max() <= 3
    # the heavy entries should dominate a long draw
    many = np.asarray([i for _ in range(400) for i in build_weighted_sampler(
        [1.0, 1.0, 5.0, 5.0])])
    assert (many >= 2).mean() > 0.6


def test_overfitting_report_reads_a_curve():
    """The diagnostic must work on any curve, not a baked-in one."""
    from ufo_mgen.mech import overfitting_report

    plateaued = [
        {"epoch": 1, "train_nll": 1.00, "val_nll": 1.40, "val_chem": 300.0},
        {"epoch": 2, "train_nll": 0.99, "val_nll": 1.38, "val_chem": 300.0},
        {"epoch": 3, "train_nll": 0.99, "val_nll": 1.375, "val_chem": 300.0},
        {"epoch": 4, "train_nll": 0.98, "val_nll": 1.374, "val_chem": 300.0},
    ]
    r = overfitting_report(plateaued)
    assert r["best_epoch"] == 4
    assert r["val_nll_plateaued"] is True
    assert r["val_nll_diverged"] is False
    assert r["chemical_constraint_preserved"] is True
    assert abs(r["generalisation_gap"] - 0.394) < 1e-6


def test_overfitting_report_flags_divergence_and_chemistry_drift():
    from ufo_mgen.mech import overfitting_report

    bad = [
        {"epoch": 1, "train_nll": 1.00, "val_nll": 1.30, "val_chem": 300.0},
        {"epoch": 2, "train_nll": 0.80, "val_nll": 1.45, "val_chem": 310.0},
        {"epoch": 3, "train_nll": 0.60, "val_nll": 1.70, "val_chem": 340.0},
    ]
    r = overfitting_report(bad)
    assert r["val_nll_diverged"] is True
    assert r["val_nll_plateaued"] is False
    assert r["chemical_constraint_preserved"] is False, (
        "a drifting chemistry term means the fine-tune traded validity for bias"
    )


def test_overfitting_report_rejects_an_empty_curve():
    from ufo_mgen.mech import overfitting_report

    try:
        overfitting_report([])
    except ValueError:
        return
    raise AssertionError("expected ValueError for an empty curve")


# ---------------------------------------------------------------- mask

def test_element_sets_and_parsing():
    from ufo_mgen.mech import ELEMENT_SETS, parse_allowed_z

    assert ELEMENT_SETS["light_covalent"] == [5, 6, 7]
    assert parse_allowed_z("5,6,7") == {5, 6, 7}
    assert parse_allowed_z("light_covalent") == {5, 6, 7}
    assert parse_allowed_z([5, 6]) == {5, 6}


def test_logit_mask_blocks_everything_else():
    import torch

    from ufo_mgen.mech import build_logit_mask

    mask = build_logit_mask([5, 6, 7], num_elements=118)
    assert mask.shape == (118,)
    finite = torch.isfinite(mask)
    assert int(finite.sum()) == 3
    # logits are zero-indexed by atomic number
    for z in (5, 6, 7):
        assert mask[z - 1] == 0.0
    assert torch.isinf(mask[0])          # hydrogen blocked


def test_logit_mask_rejects_an_empty_set():
    from ufo_mgen.mech import build_logit_mask

    try:
        build_logit_mask([500, 900], num_elements=118)
    except ValueError:
        return
    raise AssertionError("an all-blocked mask should raise")


def test_apply_and_lift_the_mask():
    import torch

    from ufo_mgen.mech import apply_element_mask

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_elements = 118
            self.out_head = torch.nn.Linear(4, 118)

    m = Toy()
    original = apply_element_mask(m, "light_covalent")
    out = m.out_head(torch.zeros(2, 4))
    assert int(torch.isfinite(out[0]).sum()) == 3
    m.out_head = original
    assert torch.isfinite(m.out_head(torch.zeros(2, 4))).all(), "mask must lift cleanly"


def test_violation_detection():
    from ufo_mgen.mech.element_mask import violations

    assert violations(["B", "C", "N"], "light_covalent") == []
    assert violations(["B", "Fe"], "light_covalent") == ["Fe"]


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
