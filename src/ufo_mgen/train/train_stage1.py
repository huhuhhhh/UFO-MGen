"""Training entry point for Stage I HTS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..models.stage1_hts import (
    PARENT_MAP,
    TARGET_ORDER,
    CoreHierStage1,
    LabelVocab,
    featurize_formula,
)
from ..utils.reproducibility import set_seed


class Stage1Dataset(Dataset):
    """Composition features plus the three discrete targets."""

    def __init__(self, df: pd.DataFrame, vocabs):
        self.x = np.stack([
            featurize_formula(r.get("formula"), r.get("n_atoms", 0), r.get("bucket", ""))
            for _, r in df.iterrows()
        ]).astype(np.float32)
        self.y = {t: np.array([vocabs[t].encode(v) for v in df[t]], dtype=np.int64)
                  for t in TARGET_ORDER}

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i):
        item = {"x": torch.from_numpy(self.x[i])}
        for t in TARGET_ORDER:
            item[t] = torch.tensor(self.y[t][i])
        return item


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UFO-MGen Stage I (HTS)")
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = args.device

    df = pd.read_csv(args.labels_csv, low_memory=False)
    for t in TARGET_ORDER:
        if t not in df.columns:
            raise SystemExit(f"labels CSV is missing target column {t!r}")
    split = df["split"].astype(str).str.lower() if "split" in df.columns else None
    train_df = df[split == "train"] if split is not None else df
    val_df = df[split == "val"] if split is not None else df.iloc[:0]

    vocabs = {t: LabelVocab.from_train_series(train_df[t]) for t in TARGET_ORDER}
    train_ds, val_ds = Stage1Dataset(train_df, vocabs), Stage1Dataset(val_df, vocabs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if len(val_ds) else None

    model = CoreHierStage1(
        input_dim=train_ds.x.shape[1], vocabs=vocabs,
        hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.forward_train(batch)
            loss = sum(loss_fn(out[t], batch[t]) for t in TARGET_ORDER)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * batch["x"].shape[0]
            n += batch["x"].shape[0]

        row = {"epoch": epoch, "train_loss": tot / max(n, 1)}
        if val_loader is not None:
            model.eval()
            correct = {t: 0 for t in TARGET_ORDER}
            seen = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logits = model.forward_greedy(batch["x"])
                    for t in TARGET_ORDER:
                        correct[t] += int((logits[t].argmax(-1) == batch[t]).sum().item())
                    seen += batch["x"].shape[0]
            for t in TARGET_ORDER:
                row[f"val_acc_{t}"] = correct[t] / max(seen, 1)
        history.append(row)
        print(json.dumps(row))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_dim": int(train_ds.x.shape[1]), "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim, "epochs": args.epochs, "seed": args.seed,
        "n_train": len(train_ds), "n_val": len(val_ds),
    }
    torch.save(
        {"model": model.state_dict(),
         "vocabs": {t: {"stoi": v.stoi, "itos": v.itos} for t, v in vocabs.items()},
         "summary": summary},
        out / "model.pt",
    )
    pd.DataFrame(history).to_csv(out / "train_history.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out/'model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
