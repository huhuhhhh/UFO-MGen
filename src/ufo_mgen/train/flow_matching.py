"""Flow-matching bridges and losses for Stage III SWG."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def wrap_delta(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Shortest signed displacement from ``mu`` to ``x`` on the unit torus."""
    return torch.remainder(x - mu + 0.5, 1.0) - 0.5


def sample_training_bridge(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    """Draw a flow-matching training pair at a random time ``t``."""
    bsz = batch["lattice_block"].shape[0]
    t = torch.rand(bsz, 1, device=device)

    x1_l = batch["lattice_block"]
    x0_l = torch.randn_like(x1_l)
    xt_l = (1.0 - t) * x0_l + t * x1_l
    target_l = x1_l - x0_l

    x1_o = batch["orbit_tensor"]
    x0_o = torch.rand_like(x1_o)
    delta_o = wrap_delta(x1_o, x0_o) * batch["orbit_value_mask"]
    xt_o = torch.remainder(x0_o + t.unsqueeze(-1) * delta_o, 1.0) * batch["orbit_value_mask"]

    return {
        "t": t,
        "lattice_xt": xt_l,
        "orbit_xt": xt_o,
        "target_lattice_v": target_l,
        "target_orbit_v": delta_o,
    }


def fm_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    orbit_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample flow-matching loss (lattice MSE + masked coordinate MSE)."""
    loss_l = ((pred["lattice_v"] - target["target_lattice_v"]) ** 2).mean(dim=-1)
    loss_o = ((pred["orbit_v"] - target["target_orbit_v"]) ** 2) * orbit_mask
    loss_o = loss_o.sum(dim=(1, 2)) / orbit_mask.sum(dim=(1, 2)).clamp(min=1.0)
    return loss_l + loss_o


def batch_to_device(batch: Dict, device) -> Dict:
    """Move tensor entries of a batch to ``device``, leaving metadata alone."""
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
