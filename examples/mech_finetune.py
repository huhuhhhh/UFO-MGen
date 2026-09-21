"""Minimal UFO-Mech scoring, masking, and fine-tuning example."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ufo_mgen.mech import (
    ELEMENT_SETS,
    FINETUNE_SETTINGS,
    SCORE_QUANTILES,
    SCORE_WEIGHTS,
    add_mechanical_scores,
    bin_and_weight,
    build_logit_mask,
    overfitting_report,
    parse_allowed_z,
)
from ufo_mgen.mech.predictor import PRODUCTION_ARCH, TARGET_COLUMNS


def toy_table() -> pd.DataFrame:
    """Phases spanning soft, incompressible-but-soft, and genuinely hard."""
    return pd.DataFrame([
        ("diamond-like",    440,  520, 1100, 0.07, True),
        ("hard carbide",    300,  250,  600, 0.18, True),
        ("dense but soft",  280,   40,  115, 0.42, True),
        ("ordinary oxide",  160,   90,  225, 0.26, True),
        ("soft metal",       70,   25,   68, 0.35, True),
        ("unstable phase",  350,  300,  720, 0.15, False),
    ], columns=["name", "K", "G", "E", "nu", "elastic_stability"])


def main() -> int:
    print("=== 1. the score ===")
    for k, v in SCORE_WEIGHTS.items():
        print(f"    {k:<8} weight {v}")
    print("  K, G and E enter the fine-tuning score with equal weight.")

    df = add_mechanical_scores(toy_table()).sort_values(
        "mechanical_extreme_score", ascending=False)
    bins, weights = bin_and_weight(df.mechanical_extreme_score)
    df["bin"], df["sample_weight"] = bins, weights

    print(f"\n  {'phase':<16}{'K':>6}{'G':>6}{'E':>7}{'penalty':>9}{'score':>8}{'weight':>8}")
    for _, r in df.iterrows():
        print(f"  {r['name']:<16}{r.K:>6.0f}{r.G:>6.0f}{r.E:>7.0f}"
              f"{r.penalty_unstable:>9.1f}{r.mechanical_extreme_score:>8.2f}"
              f"{r.sample_weight:>8.1f}")
    print("\n  'dense but soft' has the third-highest K yet scores near the bottom.")
    print("  'unstable phase' is competitive on moduli but carries a penalty.")

    print(f"\n  scores are binned at {[f'p{int(q*100)}' for q in SCORE_QUANTILES]}")
    scores = np.linspace(-3, 3, 200)
    _, w = bin_and_weight(scores)
    print(f"  over an even spread, the lower half keeps weight 1.0 and only")
    print(f"  {int((w > 1).sum())}/200 entries are up-weighted -- steering, not narrowing.")

    print("\n=== 2. element restriction ===")
    for name, zs in ELEMENT_SETS.items():
        print(f"    {name:<16} {len(zs):>2} elements  {zs[:8]}{' ...' if len(zs) > 8 else ''}")
    mask = build_logit_mask(parse_allowed_z("light_covalent"))
    allowed = [int(i) + 1 for i in np.where(np.isfinite(mask.numpy()))[0]]
    print(f"\n  a mask over 118 elements leaves {len(allowed)} usable: Z = {allowed}")
    print("  additive on the head logits, reversible, composes with a fine-tune")

    print("\n=== 3. the fine-tuning protocol ===")
    for k, v in FINETUNE_SETTINGS.items():
        print(f"    {k:<16} {v}")
    print("  A larger learning rate lets the model collapse onto the hard-phase")
    print("  corner and forget the chemistry the base checkpoint learned.")

    print("\n  diagnostics on a synthetic curve:")
    curve = [
        {"epoch": 1, "train_nll": 1.00, "val_nll": 1.40, "val_chem": 300.0},
        {"epoch": 2, "train_nll": 0.99, "val_nll": 1.38, "val_chem": 300.0},
        {"epoch": 3, "train_nll": 0.99, "val_nll": 1.375, "val_chem": 300.0},
        {"epoch": 4, "train_nll": 0.98, "val_nll": 1.374, "val_chem": 300.0},
    ]
    for k, v in overfitting_report(curve).items():
        print(f"    {k:<32} {v}")
    print("\n  A gap alone is not fatal. What matters is whether validation")
    print("  plateaued rather than diverging, and whether the chemical-constraint")
    print("  term held -- if it degrades, the fine-tune traded chemical validity")
    print("  for property bias.")

    print("\n=== the surrogate ranks, it does not measure ===")
    print(f"    targets     {TARGET_COLUMNS}")
    for k, v in PRODUCTION_ARCH.items():
        print(f"    {k:<16} {v}")
    print("\n  Final K/G/E values are recomputed with MatterSim,")
    print("  never taken from the ranking surrogate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
