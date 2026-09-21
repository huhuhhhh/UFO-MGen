"""Weighted Stage-II COM fine-tuning for mechanical-property guidance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

#: Fine-tuning hyper-parameters. Small steps, few epochs, weighted resampling.
FINETUNE_SETTINGS: Dict[str, Any] = {
    "epochs": 5,
    "lr": 2e-5,
    "weight_decay": 1e-4,
    "batch_size": 192,
    "max_orbits": 32,
    "weighted_sampler": True,
}


def build_weighted_sampler(sample_weights: Sequence[float]):
    """A sampler that oversamples the extreme tail without dropping the bulk."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    from .score import SAMPLE_WEIGHT_RANGE

    w = np.clip(np.asarray(sample_weights, dtype=np.float64),
                1e-8, SAMPLE_WEIGHT_RANGE[1])
    return WeightedRandomSampler(
        torch.as_tensor(w, dtype=torch.double),
        num_samples=len(w),
        replacement=True,
    )


def finetune_com(
    init_checkpoint,
    train_loader,
    val_loader,
    run_epoch: Callable,
    out_dir,
    *,
    epochs: int = FINETUNE_SETTINGS["epochs"],
    lr: float = FINETUNE_SETTINGS["lr"],
    weight_decay: float = FINETUNE_SETTINGS["weight_decay"],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Fine-tune a Stage-II checkpoint on a weighted corpus.

    ``run_epoch(model, loader, optimizer, device, train) -> (nll, chem)`` is the
    Stage-II epoch step; pass the one used to train the base model so the
    objective is unchanged. The best epoch by validation NLL is written to
    ``out_dir/best_model.pt``.
    """
    import torch

    from ..models.stage2_com import load_com

    out = Path(str(out_dir))
    out.mkdir(parents=True, exist_ok=True)

    model = load_com(init_checkpoint, device=device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, float]] = []
    best: Dict[str, Any] = {"epoch": -1, "val_nll": float("inf")}

    for epoch in range(1, int(epochs) + 1):
        tr_nll, tr_chem = run_epoch(model, train_loader, optimizer, device, True)
        vl_nll, vl_chem = run_epoch(model, val_loader, None, device, False)
        row = {"epoch": epoch, "train_nll": float(tr_nll), "train_chem": float(tr_chem),
               "val_nll": float(vl_nll), "val_chem": float(vl_chem)}
        history.append(row)
        print(json.dumps(row), flush=True)

        if row["val_nll"] < best["val_nll"]:
            best = dict(row)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_nll": row["val_nll"],
                    "args": {
                        "mechanical_finetune": True,
                        "lr": lr,
                        "epochs": int(epochs),
                        "weighted_sampler": True,
                    },
                },
                out / "best_model.pt",
            )

    (out / "train_history.json").write_text(json.dumps(history, indent=2))
    (out / "summary.json").write_text(
        json.dumps({"best": best, "history": history}, indent=2))
    return {"best": best, "history": history}


def overfitting_report(curve: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    """Diagnose a fine-tuning curve: the gap, the plateau, the chemistry term.

    ``curve`` is the list of per-epoch dicts returned by :func:`finetune_com`.
    """
    c = list(curve)
    if not c:
        raise ValueError("empty curve")
    best = min(c, key=lambda r: r["val_nll"])
    chem = {round(float(r["val_chem"]), 4) for r in c if r.get("val_chem") is not None}
    # A plateau needs at least two points to be meaningful: a one-point tail
    # has zero spread and would otherwise always look flat.
    vals = [r["val_nll"] for r in c]
    tail = vals[2:] if len(vals[2:]) >= 2 else vals[-3:]
    plateaued = len(tail) >= 2 and (max(tail) - min(tail)) < 0.01
    return {
        "best_epoch": best["epoch"],
        "best_val_nll": best["val_nll"],
        "train_nll_at_best": best.get("train_nll"),
        "generalisation_gap": (
            round(best["val_nll"] - best["train_nll"], 4)
            if best.get("train_nll") is not None else None
        ),
        "val_nll_plateaued": plateaued,
        "val_nll_diverged": c[-1]["val_nll"] > c[0]["val_nll"],
        "chemical_constraint_preserved": len(chem) == 1,
    }
