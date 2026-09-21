"""Training entry point for Stage III SWG route models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data_package import LATTICE_COLS, WyckoffStage2Config, WyckoffStage2Dataset, build_loader
from ..models.stage3_swg import (
    StructuredWyckoffVectorField,
    WyckoffStage2InputAdapter,
    build_context,
)
from ..utils.reproducibility import set_seed
from .flow_matching import batch_to_device, fm_loss, sample_training_bridge


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UFO-MGen Stage III (SWG)")
    p.add_argument("--data-root", required=True)
    p.add_argument("--scaffold", default=None)
    p.add_argument("--spacegroup", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--adapter-dim", type=int, default=128)
    p.add_argument("--n-attn-heads", type=int, default=4)
    p.add_argument("--n-attn-layers", type=int, default=2)
    p.add_argument("--context-mode", default="discrete_chem_topology_only")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = args.device
    root = Path(args.data_root)

    train_ds, train_loader = build_loader(
        root, "train", args.batch_size, True, scaffold=args.scaffold,
    )
    val_ds, val_loader = build_loader(
        root, "val", args.batch_size, False, scaffold=args.scaffold,
    )
    ds: WyckoffStage2Dataset = train_ds

    base = ds[0]
    adapter = WyckoffStage2InputAdapter(
        lattice_dim=int(base["lattice_block"].shape[-1]),
        chem_dim=int(base["chem_block"].shape[-1]),
        max_orbits=ds.max_orbits,
        max_orbit_dof=ds.max_orbit_dof,
        n_scaffolds=len(ds.scaffold_list),
        hidden_dim=args.adapter_dim,
        context_mode=args.context_mode,
        max_feature_dim=ds.max_feature_dim,
    ).to(device)
    model = StructuredWyckoffVectorField(
        args.adapter_dim, ds.max_orbits, ds.max_orbit_dof,
        args.hidden_dim, args.n_attn_heads, args.n_attn_layers,
    ).to(device)

    params = list(model.parameters()) + list(adapter.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    history, best, best_state = [], {"epoch": -1, "val_loss": float("inf")}, None
    for epoch in range(1, args.epochs + 1):
        model.train(); adapter.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            bridge = sample_training_bridge(batch, device)
            context = build_context(adapter, batch)
            pred = model(context, bridge["lattice_xt"], bridge["orbit_xt"],
                         batch["orbit_dof"], batch["orbit_active_mask"], bridge["t"])
            loss = fm_loss(pred, bridge, batch["orbit_value_mask"]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            bs = len(batch["path"])
            tot += float(loss.item()) * bs
            n += bs

        model.eval(); adapter.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch_to_device(batch, device)
                bridge = sample_training_bridge(batch, device)
                context = build_context(adapter, batch)
                pred = model(context, bridge["lattice_xt"], bridge["orbit_xt"],
                             batch["orbit_dof"], batch["orbit_active_mask"], bridge["t"])
                vals.append(float(fm_loss(pred, bridge, batch["orbit_value_mask"]).mean().item()))

        row = {"epoch": epoch, "train_loss": tot / max(n, 1),
               "val_loss": float(np.mean(vals)) if vals else float("inf")}
        history.append(row)
        if row["val_loss"] < best["val_loss"]:
            best = dict(row)
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "adapter": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
            }
        print(json.dumps(row))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    summary = {
        "spacegroup": args.spacegroup, "scaffold": args.scaffold,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim, "adapter_dim": args.adapter_dim,
        "n_attn_heads": args.n_attn_heads, "n_attn_layers": args.n_attn_layers,
        "context_mode": args.context_mode, "seed": args.seed,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
    }
    torch.save({**best_state, "summary": summary}, out / "model.pt")
    pd.DataFrame(history).to_csv(out / "train_history.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out/'model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
