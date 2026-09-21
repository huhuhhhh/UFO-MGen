#!/usr/bin/env python3
"""Element and orbit-aware embeddings used by COM."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np

# ── Periodic table physical features ─────────────────────────────────────────
# Index: 1-indexed atomic number (Z=1 → H, Z=118 → Og)
# We pre-build these once and cache.

def _build_physical_feature_table() -> torch.Tensor:
    """
    Returns a (119, D_PHYS) tensor.  Row 0 is unused (padding).
    Rows 1..118 correspond to Z=1..118.

    Features (D_PHYS = 9):
        0: Z (normalized 0–1)
        1: period (1–7, normalized)
        2: group (1–18, normalized, 0 for f-block)
        3: Pauling electronegativity (0 if unknown)
        4: atomic radius (pm, normalized; 0 if unknown)
        5: first ionization energy (eV, normalized; 0 if unknown)
        6: mean common oxidation state (0 if none)
        7: max common oxidation state
        8: min common oxidation state
    """
    try:
        from pymatgen.core import Element
    except ImportError:
        raise ImportError("pymatgen is required for element embeddings.")

    D_PHYS = 9
    table = torch.zeros(119, D_PHYS)

    # Collect raw values for normalization
    raw: Dict[str, List[float]] = {k: [] for k in
                                    ["Z", "period", "group", "X", "r", "IE", "ox_mean", "ox_max", "ox_min"]}

    elements = []
    for Z in range(1, 119):
        try:
            el = Element.from_Z(Z)
        except Exception:
            elements.append(None)
            continue
        elements.append(el)

        raw["Z"].append(float(Z))
        raw["period"].append(float(el.row))
        raw["group"].append(float(el.group) if el.group else 0.0)

        try:
            raw["X"].append(float(el.X) if el.X else 0.0)
        except Exception:
            raw["X"].append(0.0)

        try:
            r = el.atomic_radius
            raw["r"].append(float(r) if r else 0.0)
        except Exception:
            raw["r"].append(0.0)

        try:
            ie = el.ionization_energy
            raw["IE"].append(float(ie) if ie else 0.0)
        except Exception:
            raw["IE"].append(0.0)

        try:
            ox = list(el.common_oxidation_states)
            if ox:
                raw["ox_mean"].append(float(np.mean(ox)))
                raw["ox_max"].append(float(max(ox)))
                raw["ox_min"].append(float(min(ox)))
            else:
                raw["ox_mean"].append(0.0)
                raw["ox_max"].append(0.0)
                raw["ox_min"].append(0.0)
        except Exception:
            raw["ox_mean"].append(0.0)
            raw["ox_max"].append(0.0)
            raw["ox_min"].append(0.0)

    # Normalize each feature to [0, 1] using known max
    def _norm(vals: List[float], vmax: Optional[float] = None) -> List[float]:
        if vmax is None:
            vmax = max(v for v in vals if v != 0.0) if any(v != 0.0 for v in vals) else 1.0
        return [v / vmax if vmax > 0 else 0.0 for v in vals]

    Z_norm    = _norm(raw["Z"],    118.0)
    per_norm  = _norm(raw["period"], 7.0)
    grp_norm  = _norm(raw["group"], 18.0)
    X_norm    = _norm(raw["X"])
    r_norm    = _norm(raw["r"])
    IE_norm   = _norm(raw["IE"])
    oxm_vals  = raw["ox_mean"]
    oxmx_vals = raw["ox_max"]
    oxmn_vals = raw["ox_min"]

    # Fill table (offset +1 because row 0 = padding)
    for i, el in enumerate(elements):
        if el is None:
            continue
        Z = i + 1
        table[Z, 0] = Z_norm[i]
        table[Z, 1] = per_norm[i]
        table[Z, 2] = grp_norm[i]
        table[Z, 3] = X_norm[i]
        table[Z, 4] = r_norm[i]
        table[Z, 5] = IE_norm[i]
        # Oxidation states: scale by dividing by 8 (common max abs ox state)
        table[Z, 6] = oxm_vals[i]  / 8.0
        table[Z, 7] = oxmx_vals[i] / 8.0
        table[Z, 8] = oxmn_vals[i] / 8.0

    return table   # (119, 9)


# Build once at import time (lazy, cached as module-level constant)
_PHYS_TABLE: Optional[torch.Tensor] = None


def get_physical_table() -> torch.Tensor:
    global _PHYS_TABLE
    if _PHYS_TABLE is None:
        _PHYS_TABLE = _build_physical_feature_table()
    return _PHYS_TABLE


D_PHYS = 9  # must match _build_physical_feature_table


# ── IntrinsicElementEmbedding ─────────────────────────────────────────────────

class IntrinsicElementEmbedding(nn.Module):
    """
    Level-1 element embedding: e_int(s) ∈ R^{d_learn + D_PHYS}.

    Args:
        num_elements: size of element vocabulary (default 118, 1-indexed)
        d_learn:      dimension of learnable part
        pretrained_init: 'physical' initializes learned part from physical
                          features via a linear projection; 'random' is Xavier.
    """

    def __init__(
        self,
        num_elements: int = 118,
        d_learn: int = 64,
        pretrained_init: str = "physical",
    ) -> None:
        super().__init__()
        self.num_elements = num_elements
        self.d_learn = d_learn
        # +1 for padding index 0
        self.embed = nn.Embedding(num_elements + 1, d_learn, padding_idx=0)

        # Register fixed physical features as a buffer (not a parameter)
        phys_table = get_physical_table()  # (119, D_PHYS)
        self.register_buffer("phys_table", phys_table)

        if pretrained_init == "physical":
            self._init_from_physical()
        else:
            nn.init.xavier_uniform_(self.embed.weight)

    def _init_from_physical(self) -> None:
        """Initialize learned embedding weights from a linear projection of
        physical features, so that physically similar elements start nearby."""
        with torch.no_grad():
            phys = self.phys_table[1:]  # (118, D_PHYS)
            # Project D_PHYS → d_learn via a random linear map (fixed seed)
            gen = torch.Generator()
            gen.manual_seed(42)
            proj = torch.randn(D_PHYS, self.d_learn, generator=gen)
            proj = proj / math.sqrt(D_PHYS)
            init = phys @ proj  # (118, d_learn)
            # Normalize rows
            init = init / (init.norm(dim=1, keepdim=True).clamp(min=1e-6))
            self.embed.weight.data[1:] = init   # row 0 (padding) stays zero

    @property
    def d_e(self) -> int:
        return self.d_learn + D_PHYS

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: LongTensor[..., K]  element indices, 1-indexed (0 = padding)
        Returns:
            e_int: FloatTensor[..., K, d_e]
        """
        learned = self.embed(s)                       # [..., K, d_learn]
        phys    = self.phys_table[s]                  # [..., K, D_PHYS]
        return torch.cat([learned, phys], dim=-1)     # [..., K, d_e]


# ── Wyckoff position encoding ─────────────────────────────────────────────────

# We encode a Wyckoff token (e.g. "4a") by its multiplicity and letter index.
# Letter 'a'=0, 'b'=1, ..., 'z'=25 (not all are used, but covers ITA range).

WYCKOFF_LETTERS = "abcdefghijklmnopqrstuvwxyz"
MAX_MULTIPLICITY = 192   # highest Wyckoff multiplicity in any SG


def encode_wyckoff_token(token: str) -> tuple[int, int]:
    """
    Parse a Wyckoff token like '4a' or '16e'.
    Returns (multiplicity, letter_idx).
    """
    import re
    m = re.match(r"(\d+)([a-z])", token.strip().lower())
    if m is None:
        return (1, 0)
    mult = int(m.group(1))
    letter = WYCKOFF_LETTERS.index(m.group(2)) if m.group(2) in WYCKOFF_LETTERS else 0
    return (mult, letter)


class OrbitAwareEmbedding(nn.Module):
    """
    Level-2 embedding: h_k = MLP(e_int(s_k) ‖ w(w_k) ‖ Emb(m_k) ‖ Emb(d_k)).

    Args:
        d_e:       intrinsic embedding dim (= d_learn + D_PHYS)
        d_wyckoff: embedding dim for Wyckoff letter
        d_h:       output orbit embedding dim
        max_mult:  maximum Wyckoff multiplicity (for positional embedding)
        max_dof:   maximum degrees of freedom per orbit (typically ≤ 3)
    """

    def __init__(
        self,
        d_e: int = 96,
        d_wyckoff: int = 16,
        d_h: int = 128,
        max_mult: int = MAX_MULTIPLICITY,
        max_dof: int = 4,
    ) -> None:
        super().__init__()
        self.letter_embed = nn.Embedding(len(WYCKOFF_LETTERS) + 1, d_wyckoff)
        # Multiplicity and DOF as scalar features (normalized)
        self.max_mult = max_mult
        self.max_dof  = max_dof

        d_in = d_e + d_wyckoff + 2   # +2 for scalar mult and dof
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, d_h),
        )

    def forward(
        self,
        e_int: torch.Tensor,          # [..., K, d_e]
        wyckoff_letters: torch.Tensor,  # [..., K] — letter indices (LongTensor)
        multiplicities: torch.Tensor,   # [..., K] — int multiplicities
        dofs: torch.Tensor,             # [..., K] — int DOFs
    ) -> torch.Tensor:
        """Returns h: [..., K, d_h]"""
        w_emb  = self.letter_embed(wyckoff_letters)           # [..., K, d_wyckoff]
        mult_n = multiplicities.float().unsqueeze(-1) / self.max_mult  # [..., K, 1]
        dof_n  = dofs.float().unsqueeze(-1) / self.max_dof             # [..., K, 1]
        x = torch.cat([e_int, w_emb, mult_n, dof_n], dim=-1)          # [..., K, d_in]
        return self.mlp(x)
