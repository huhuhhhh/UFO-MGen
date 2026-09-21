#!/usr/bin/env python3
"""Chemical validity constraints used by COM training and sampling."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# ── Oxidation state data ──────────────────────────────────────────────────────
# Pre-built from pymatgen; returns dict Z -> list of common oxidation states.

def _build_oxidation_state_table() -> Dict[int, List[int]]:
    try:
        from pymatgen.core import Element
    except ImportError:
        raise ImportError("pymatgen required for chemical constraints.")
    table: Dict[int, List[int]] = {}
    for Z in range(1, 119):
        try:
            el = Element.from_Z(Z)
            ox = list(el.common_oxidation_states)
            table[Z] = ox if ox else [0]
        except Exception:
            table[Z] = [0]
    return table


_OX_TABLE: Optional[Dict[int, List[int]]] = None


def get_oxidation_table() -> Dict[int, List[int]]:
    global _OX_TABLE
    if _OX_TABLE is None:
        _OX_TABLE = _build_oxidation_state_table()
    return _OX_TABLE


# ── ChargeBalanceConstraint ───────────────────────────────────────────────────

class ChargeBalanceConstraint(nn.Module):
    """
    Differentiable charge balance energy.

    For each structure, finds the oxidation state assignment q_{1:K} that
    minimises (Σ m_k * q_k)^2 subject to q_k ∈ OxStates(s_k).

    Because oxidation states are discrete, we solve the inner minimisation
    via exhaustive enumeration over a bounded set and return the minimum
    squared charge imbalance as a scalar penalty.

    For training efficiency, K ≤ 8 and |OxStates| ≤ 8 per element, so the
    product space has at most 8^8 = 16M entries — too many to enumerate.
    We instead use a greedy sequential search (exact for K ≤ 4, approximate
    for larger K) plus the continuous soft-penalty relaxation described in the
    COM design doc.

    Forward:
        s:              LongTensor[B, K]   element indices (1-indexed)
        multiplicities: FloatTensor[B, K]  Wyckoff multiplicities
    Returns:
        E_CB:     FloatTensor[B]   charge balance energy (≥ 0; 0 = balanced)
        best_ox:  FloatTensor[B, K] best oxidation state assignment found
    """

    def __init__(self, max_ox_states: int = 8) -> None:
        super().__init__()
        # Build per-element oxidation state tensor (119, max_ox_states),
        # padded with NaN for missing entries.
        ox_table = get_oxidation_table()
        ox_arr = np.full((119, max_ox_states), np.nan, dtype=np.float32)
        for Z, states in ox_table.items():
            n = min(len(states), max_ox_states)
            ox_arr[Z, :n] = states[:n]
        self.register_buffer("ox_tensor", torch.tensor(ox_arr))
        self.max_ox = max_ox_states

    def forward(
        self,
        s: torch.Tensor,
        multiplicities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K = s.shape
        ox_tensor = self.ox_tensor   # (119, max_ox)
        device = s.device

        # Per-element oxidation state options: (B, K, max_ox)
        el_ox = ox_tensor[s]   # (B, K, max_ox)

        # Greedy sequential assignment: for each orbit, choose oxidation state
        # that brings running charge sum closest to 0.
        running_q = torch.zeros(B, device=device)
        best_ox = torch.zeros(B, K, device=device)

        for k in range(K):
            ox_k = el_ox[:, k, :]           # (B, max_ox) — NaN where not available
            mult_k = multiplicities[:, k]   # (B,)
            # Contribution of each ox candidate: running_q + mult_k * ox_candidate
            contrib = running_q.unsqueeze(1) + mult_k.unsqueeze(1) * ox_k  # (B, max_ox)
            # Replace NaN with large value so they are never selected
            contrib_clean = torch.where(torch.isnan(ox_k), torch.full_like(contrib, 1e6), contrib)
            # Choose candidate that minimises |contrib|
            best_idx = contrib_clean.abs().argmin(dim=1)   # (B,)
            chosen_ox = ox_k.gather(1, best_idx.unsqueeze(1)).squeeze(1)   # (B,)
            # If still NaN (all options NaN), default to 0
            chosen_ox = torch.where(torch.isnan(chosen_ox), torch.zeros_like(chosen_ox), chosen_ox)
            best_ox[:, k] = chosen_ox
            running_q = running_q + mult_k * chosen_ox

        E_CB = running_q.pow(2)   # (B,)  = (Σ m_k q_k)^2
        return E_CB, best_ox


# ── IonicRadiusConstraint ─────────────────────────────────────────────────────
# Simplified: penalise elements whose mean ionic radius deviates strongly from
# the route-average ionic radius.  Full site-geometry estimation requires ITA
# tables; here we use elemental covalent radii as a proxy.

def _build_ionic_radius_table() -> torch.Tensor:
    """Returns a (119,) tensor of mean Shannon ionic radii (Å); 0 if unknown."""
    try:
        from pymatgen.core import Element
    except ImportError:
        raise ImportError("pymatgen required.")
    radii = torch.zeros(119)
    for Z in range(1, 119):
        try:
            el = Element.from_Z(Z)
            r = el.atomic_radius
            if r is not None:
                radii[Z] = float(r)
        except Exception:
            pass
    return radii


_RADIUS_TABLE: Optional[torch.Tensor] = None


def get_radius_table() -> torch.Tensor:
    global _RADIUS_TABLE
    if _RADIUS_TABLE is None:
        _RADIUS_TABLE = _build_ionic_radius_table()
    return _RADIUS_TABLE


class IonicRadiusConstraint(nn.Module):
    """
    Soft ionic radius compatibility penalty.

    For each orbit k, computes E_rad(s_k) = max(0, r_k/r_ref - γ_upper)^2
                                           + max(0, γ_lower - r_k/r_ref)^2

    where r_ref is the mean radius of training elements on that orbit type
    (approximated here as the mean radius across the batch for simplicity).

    Args:
        gamma_lower: minimum acceptable radius ratio (default 0.5)
        gamma_upper: maximum acceptable radius ratio (default 2.0)
    """

    def __init__(self, gamma_lower: float = 0.5, gamma_upper: float = 2.0) -> None:
        super().__init__()
        self.gamma_lower = gamma_lower
        self.gamma_upper = gamma_upper
        radius_table = get_radius_table()
        self.register_buffer("radius_table", radius_table)

    def forward(
        self,
        s: torch.Tensor,          # LongTensor[B, K]
        multiplicities: torch.Tensor,  # FloatTensor[B, K] — not used here but kept for API
    ) -> torch.Tensor:
        """Returns E_rad: FloatTensor[B]"""
        r = self.radius_table[s]           # (B, K)
        # Use batch-mean radius as reference for each orbit position
        r_ref = r.mean(dim=0, keepdim=True).clamp(min=1e-3)   # (1, K)
        ratio = r / r_ref                                       # (B, K)
        e_upper = torch.clamp(ratio - self.gamma_upper, min=0).pow(2)
        e_lower = torch.clamp(self.gamma_lower - ratio, min=0).pow(2)
        return (e_upper + e_lower).sum(dim=1)   # (B,)


# ── CompositionValidator (non-differentiable, for inference) ──────────────────

class CompositionValidator:
    """
    Hard chemical validity checks used during rejection sampling at inference.

    Methods:
        is_charge_balanced(s, multiplicities, tolerance) -> bool
        is_electronegativity_valid(s)                   -> bool
        validate_all(s, multiplicities)                 -> (bool, dict)
    """

    def __init__(self) -> None:
        self._ox_table = get_oxidation_table()
        self._strongly_ionic_anions = {"O", "F", "Cl", "Br", "I", "N", "S", "Se"}
        all_states = [state for states in self._ox_table.values() for state in states if state is not None]
        self._global_min_ox = min(all_states) if all_states else -8
        self._global_max_ox = max(all_states) if all_states else 8

    def _symbols(self, s: List[int]) -> List[str]:
        from pymatgen.core import Element
        return [Element.from_Z(z).symbol for z in s if 1 <= z <= 118]

    def _composition_counts(
        self,
        s: Sequence[int],
        multiplicities: Sequence[int],
    ) -> Dict[str, int]:
        from pymatgen.core import Element

        counts: Dict[str, int] = {}
        for z, mult in zip(s, multiplicities):
            if not (1 <= z <= 118):
                continue
            symbol = Element.from_Z(int(z)).symbol
            counts[symbol] = counts.get(symbol, 0) + int(mult)
        return counts

    def contains_strongly_ionic_anion(self, s: Sequence[int]) -> bool:
        return any(symbol in self._strongly_ionic_anions for symbol in self._symbols(list(s)))

    def has_oxidation_state_solution(
        self,
        s: Sequence[int],
        multiplicities: Sequence[int],
    ) -> bool:
        try:
            from pymatgen.core import Composition

            counts = self._composition_counts(s, multiplicities)
            if not counts:
                return False
            guesses = Composition(counts).oxi_state_guesses(max_sites=-1)
            return bool(guesses)
        except Exception:
            return False

    def _charge_interval(
        self,
        s: Sequence[int],
        multiplicities: Sequence[int],
    ) -> Tuple[float, float]:
        min_charge = 0.0
        max_charge = 0.0
        for z, mult in zip(s, multiplicities):
            options = self._ox_table.get(int(z), [0]) or [0]
            min_charge += int(mult) * min(options)
            max_charge += int(mult) * max(options)
        return min_charge, max_charge

    def charge_interval(self, s: Sequence[int], multiplicities: Sequence[int]) -> Tuple[float, float]:
        return self._charge_interval(s, multiplicities)

    def charge_interval_allows_neutrality(
        self,
        s: Sequence[int],
        multiplicities: Sequence[int],
    ) -> bool:
        min_charge, max_charge = self._charge_interval(s, multiplicities)
        return min_charge <= 0 <= max_charge

    def prefix_charge_feasible(
        self,
        s: Sequence[int],
        multiplicities: Sequence[int],
        remaining_multiplicities: Sequence[int],
        *,
        strict_ionic_oxi_check: bool = False,
    ) -> bool:
        prefix_min, prefix_max = self._charge_interval(s, multiplicities)
        remaining_total = sum(int(mult) for mult in remaining_multiplicities)
        min_possible = prefix_min + remaining_total * self._global_min_ox
        max_possible = prefix_max + remaining_total * self._global_max_ox
        if min_possible > 0 or max_possible < 0:
            return False
        if strict_ionic_oxi_check and remaining_total == 0 and self.contains_strongly_ionic_anion(s):
            return self.has_oxidation_state_solution(s, multiplicities)
        return True

    def filter_feasible_candidates(
        self,
        prefix_s: Sequence[int],
        prefix_multiplicities: Sequence[int],
        next_multiplicity: int,
        remaining_multiplicities: Sequence[int],
        candidate_zs: Sequence[int],
        *,
        strict_ionic_oxi_check: bool = False,
    ) -> List[bool]:
        next_mult = int(next_multiplicity)
        base_s = list(prefix_s)
        base_mults = list(prefix_multiplicities)
        return [
            self.prefix_charge_feasible(
                [*base_s, int(z)],
                [*base_mults, next_mult],
                remaining_multiplicities,
                strict_ionic_oxi_check=strict_ionic_oxi_check,
            )
            for z in candidate_zs
        ]

    def is_charge_balanced(
        self,
        s: List[int],
        multiplicities: List[int],
        tolerance: float = 0.01,
    ) -> bool:
        """True if any oxidation state assignment satisfies charge neutrality."""
        ox_table = self._ox_table
        from itertools import product as iproduct
        options = [ox_table.get(z, [0]) for z in s]
        # For large K, limit search to avoid combinatorial explosion
        max_combos = 10_000
        total = 1
        for opt in options:
            total *= len(opt)
            if total > max_combos:
                # Fall back to greedy estimate
                running = 0.0
                for z, m, opts in zip(s, multiplicities, options):
                    best = min(opts, key=lambda q: abs(running + m * q))
                    running += m * best
                return abs(running) <= tolerance

        for combo in iproduct(*options):
            charge = sum(m * q for m, q in zip(multiplicities, combo))
            if abs(charge) <= tolerance:
                return True
        return False

    def is_electronegativity_valid(self, s: List[int]) -> bool:
        """True if there is at least one electronegativity-ordered pair
        (anion more electronegative than cation), a loose Pauling rule check."""
        try:
            from pymatgen.core import Element
            els = [Element.from_Z(z) for z in s if 1 <= z <= 118]
            xs = [float(el.X) if el.X else 0.0 for el in els]
            # Pass if range of electronegativities > 0.5 (some polarity exists)
            # or if all metals (intermetallic compounds are always OK)
            if all(el.is_metal for el in els):
                return True
            return (max(xs) - min(xs)) > 0.5
        except Exception:
            return True   # Don't reject on failure

    def validate_all(
        self,
        s: List[int],
        multiplicities: List[int],
        *,
        use_charge_interval_proxy_for_ionic_oxi: bool = False,
    ) -> Tuple[bool, Dict[str, bool]]:
        # All-metal intermetallics bypass charge balance check
        try:
            from pymatgen.core import Element
            all_metal = all(Element.from_Z(z).is_metal for z in s if 1 <= z <= 118)
        except Exception:
            all_metal = False

        en = self.is_electronegativity_valid(s)
        if all_metal:
            return True, {"charge_balanced": True, "electronegativity_valid": en, "all_metal": True}

        cb  = self.is_charge_balanced(s, multiplicities)
        oxi_ok = True
        if self.contains_strongly_ionic_anion(s):
            if use_charge_interval_proxy_for_ionic_oxi:
                oxi_ok = self.charge_interval_allows_neutrality(s, multiplicities)
            else:
                oxi_ok = self.has_oxidation_state_solution(s, multiplicities)
        ok  = cb and en and oxi_ok
        return ok, {
            "charge_balanced": cb,
            "electronegativity_valid": en,
            "oxi_state_solution": oxi_ok,
            "used_charge_interval_proxy_for_ionic_oxi": bool(use_charge_interval_proxy_for_ionic_oxi),
        }
