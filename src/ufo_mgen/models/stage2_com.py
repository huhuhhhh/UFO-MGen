"""Stage II: Chemical Occupancy Module (COM)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .com import (
    ChargeBalanceConstraint,
    ChemicalOccupancyModule,
    CompositionValidator,
    IonicRadiusConstraint,
)

#: Default architecture of the released Stage-II production checkpoint.
PRODUCTION_ARCH = {
    "num_elements": 118,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 4,
    "d_learn": 64,
    "d_sg": 32,
    "d_orbit": 128,
    "max_orbits": 32,
}


def build_com(**overrides: Any) -> ChemicalOccupancyModule:
    """Instantiate a COM with the production architecture."""
    cfg = dict(PRODUCTION_ARCH)
    cfg.update(overrides)
    return ChemicalOccupancyModule(**cfg)


def load_com(
    checkpoint_path,
    *,
    device: str = "cpu",
    strict: bool = True,
    **overrides: Any,
) -> ChemicalOccupancyModule:
    """Load a released Stage-II checkpoint.

    Accepts the production checkpoint layout ``{epoch, model_state, val_nll,
    args}``, as well as a bare ``state_dict``. Architecture hyper-parameters
    recorded in the checkpoint's ``args`` take precedence over
    :data:`PRODUCTION_ARCH`.
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    state: Dict[str, torch.Tensor]
    cfg = dict(PRODUCTION_ARCH)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        args = ckpt.get("args") or {}
        if not isinstance(args, dict):
            args = vars(args)
        for key in PRODUCTION_ARCH:
            if key in args and args[key] is not None:
                cfg[key] = args[key]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    cfg.update(overrides)
    model = ChemicalOccupancyModule(**cfg)
    model.load_state_dict(state, strict=strict)
    model.to(device).eval()
    return model


__all__ = [
    "ChemicalOccupancyModule",
    "build_com", "load_com", "PRODUCTION_ARCH",
    "ChargeBalanceConstraint", "IonicRadiusConstraint", "CompositionValidator",
]
