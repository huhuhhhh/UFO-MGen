"""Checkpoint loading utilities for HTS, COM, and SWG."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from ..data_package import WyckoffStage2Config, WyckoffStage2Dataset
from .stage2_com import load_com  # noqa: F401  (re-exported)
from .stage3_swg import StructuredWyckoffVectorField, WyckoffStage2InputAdapter



def _shapes_from_state(model_state, adapter_state=None) -> Dict[str, int]:
    """Recover tensor widths from saved weights.

    ``orbit_value_proj.0.weight``  is ``[hidden, max_orbit_dof]``
    ``orbit_idx_emb.weight``       is ``[max_orbits + 1, hidden]``
    ``scaffold_emb.weight``        is ``[n_scaffolds, embed]``
    """
    out: Dict[str, int] = {}
    w = model_state.get("orbit_value_proj.0.weight")
    if w is not None:
        out["max_orbit_dof"] = int(w.shape[1])
    e = model_state.get("orbit_idx_emb.weight")
    if e is not None:
        out["max_orbits"] = int(e.shape[0]) - 1
    if adapter_state:
        s = adapter_state.get("scaffold_emb.weight")
        if s is not None:
            out["n_scaffolds"] = int(s.shape[0])
    return out

def load_stage3_swg(
    checkpoint_path,
    data_root,
    *,
    scaffold: Optional[str] = None,
    split: str = "train",
    device: str = "cpu",
    context_mode: Optional[str] = None,
) -> Tuple[StructuredWyckoffVectorField, WyckoffStage2InputAdapter, Dict[str, Any]]:
    """Load a Stage-III (SWG) route checkpoint together with its adapter.

    Parameters
    ----------
    checkpoint_path
        Path to a released ``model.pt``.
    data_root
        The Stage-III data package the route was trained on. Supplies
        ``max_orbits``/``max_orbit_dof``/``n_scaffolds``/``max_feature_dim``.
    scaffold
        Restrict the package to one scaffold. Defaults to the package's sole
        scaffold when it holds only one.
    context_mode
        Overrides the checkpoint's recorded mode. Production checkpoints use
        ``discrete_chem_topology_only``.

    Returns
    -------
    ``(model, adapter, summary)`` -- both modules in ``eval()`` mode.
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(
            f"{checkpoint_path}: not a Stage-III checkpoint "
            "(expected keys 'model' and 'adapter')"
        )
    summary = dict(ckpt.get("summary") or {})

    ds = WyckoffStage2Dataset(
        WyckoffStage2Config(
            root=Path(data_root),
            split=split,
            scaffold=scaffold,
            normalize=True,
        )
    )
    if len(ds) == 0:
        raise ValueError(f"{data_root}: no rows for split={split!r} scaffold={scaffold!r}")

    base = ds[0]
    lattice_dim = int(base["lattice_block"].shape[-1])
    chem_dim = int(base["chem_block"].shape[-1])

    mode = context_mode or summary.get("context_mode") or "discrete_chem_topology_only"
    adapter_dim = int(summary.get("adapter_dim", 128))
    hidden_dim = int(summary.get("hidden_dim", 192))
    n_heads = int(summary.get("n_attn_heads", 4))
    n_layers = int(summary.get("n_attn_layers", 2))

    # Tensor widths are a property of the package a route was trained on, and
    # every route saw a different one. Read them off the saved weights rather
    # than re-deriving them, so a checkpoint loads against any data package
    # that carries its scaffold.
    shapes = _shapes_from_state(ckpt["model"], ckpt.get("adapter"))
    max_orbits = shapes.get("max_orbits", ds.max_orbits)
    max_orbit_dof = shapes.get("max_orbit_dof", ds.max_orbit_dof)
    n_scaffolds = shapes.get("n_scaffolds", len(ds.scaffold_list))

    adapter = WyckoffStage2InputAdapter(
        lattice_dim=lattice_dim,
        chem_dim=chem_dim,
        max_orbits=max_orbits,
        max_orbit_dof=max_orbit_dof,
        n_scaffolds=n_scaffolds,
        hidden_dim=adapter_dim,
        context_mode=mode,
        max_feature_dim=ds.max_feature_dim,
    ).to(device)
    model = StructuredWyckoffVectorField(
        adapter_dim, max_orbits, max_orbit_dof, hidden_dim, n_heads, n_layers
    ).to(device)

    model.load_state_dict(ckpt["model"])
    adapter.load_state_dict(ckpt["adapter"])
    model.eval()
    adapter.eval()
    return model, adapter, summary


def load_stage1_hts(checkpoint_path, *, device: str = "cpu"):
    """Load the Stage-I (HTS) checkpoint and its label vocabularies."""
    from .stage1_hts import CoreHierStage1, LabelVocab

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    summary = dict(ckpt.get("summary") or {})
    raw_vocabs = ckpt.get("vocabs") or {}
    vocabs = {
        k: (v if isinstance(v, LabelVocab) else LabelVocab(stoi=v["stoi"], itos=v["itos"]))
        for k, v in raw_vocabs.items()
    }
    route_vocab = summary.get("route_vocab")
    if isinstance(route_vocab, dict):
        route_vocab = LabelVocab(stoi=route_vocab["stoi"], itos=route_vocab["itos"])

    model = CoreHierStage1(
        input_dim=int(summary.get("input_dim", 123)),
        vocabs=vocabs,
        hidden_dim=int(summary.get("hidden_dim", 256)),
        embed_dim=int(summary.get("embed_dim", 64)),
        route_vocab=route_vocab,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, vocabs, summary


def checkpoint_summary(checkpoint_path) -> Dict[str, Any]:
    """Inspect a checkpoint without building the model."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return {"type": type(ckpt).__name__}
    info: Dict[str, Any] = {"keys": sorted(ckpt.keys())}
    if "summary" in ckpt and isinstance(ckpt["summary"], dict):
        info["summary"] = ckpt["summary"]
    if "args" in ckpt:
        args = ckpt["args"]
        info["args"] = args if isinstance(args, dict) else vars(args)
    for bad in ("optimizer", "optimizer_state_dict", "scheduler", "scaler", "ema"):
        if bad in ckpt:
            info.setdefault("unexpected_training_state", []).append(bad)
    return info


__all__ = ["load_stage1_hts", "load_com", "load_stage3_swg", "checkpoint_summary"]
