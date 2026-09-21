"""Training and fine-tuning entry point for Stage II COM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..models.stage2_com import PRODUCTION_ARCH, build_com, load_com
from ..utils.reproducibility import set_seed


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UFO-MGen Stage II (COM)")
    p.add_argument("--dataset", required=True, help="scaffold/species training set (JSON)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--init", default=None, help="checkpoint to initialise from")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-cb", type=float, default=0.1, help="charge-balance weight")
    p.add_argument("--lambda-rad", type=float, default=0.01, help="ionic-radius weight")
    p.add_argument("--sample-weights", default=None,
                   help="optional per-sample weights (property-guided fine-tuning)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    if args.init:
        model = load_com(args.init, device=args.device)
    else:
        model = build_com(lambda_cb=args.lambda_cb, lambda_rad=args.lambda_rad).to(args.device)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(PRODUCTION_ARCH)
    cfg.update({"lambda_cb": args.lambda_cb, "lambda_rad": args.lambda_rad})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    raise SystemExit(
        "Stage-II training needs the scaffold/species training set, which is not "
        "part of this release (it is derived from the full Materials Project "
        "corpus). Build it with ufo_mgen.wyckoff.encode over your own structures, "
        "then drive ufo_mgen.models.com.ChemicalOccupancyModule with the "
        "objective documented in its docstring. The released Stage-II checkpoint "
        "can be loaded and used for generation without retraining."
    )


if __name__ == "__main__":
    raise SystemExit(main())
