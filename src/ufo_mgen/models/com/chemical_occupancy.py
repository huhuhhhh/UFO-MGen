#!/usr/bin/env python3
"""Autoregressive Chemical Occupancy Module (COM)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .element_embedding import (
    IntrinsicElementEmbedding,
    OrbitAwareEmbedding,
    encode_wyckoff_token,
    WYCKOFF_LETTERS,
)
from .chemical_constraints import (
    ChargeBalanceConstraint,
    IonicRadiusConstraint,
    CompositionValidator,
)


# ── Scaffold data containers ──────────────────────────────────────────────────

@dataclass
class TopologicalScaffold:
    """
    tau = (G, K, w_{1:K}) — topology only, no species.

    Attributes:
        sg:           LongTensor[B]     space group number (1–230)
        wyckoff_letters: LongTensor[B, K]  letter indices (0=a, 1=b, …)
        multiplicities:  LongTensor[B, K]  Wyckoff multiplicities
        dofs:            LongTensor[B, K]  degrees of freedom per orbit
        orbit_mask:      BoolTensor[B, K]  True = valid orbit, False = padding
    """
    sg: torch.Tensor
    wyckoff_letters: torch.Tensor
    multiplicities: torch.Tensor
    dofs: torch.Tensor
    orbit_mask: torch.Tensor

    @classmethod
    def from_scaffold_strings(
        cls,
        sgs: List[int],
        scaffolds: List[str],
        dofs_list: Optional[List[List[int]]] = None,
        device: str = "cpu",
    ) -> "TopologicalScaffold":
        """
        Construct from lists of space group ints and scaffold strings like "4a;4a;4a".
        dofs_list: optional per-orbit DOF counts.  Defaults to 0 for all orbits.
        """
        B = len(sgs)
        K_max = max(len(s.split(";")) for s in scaffolds)
        wyckoff_letters = torch.zeros(B, K_max, dtype=torch.long)
        multiplicities  = torch.zeros(B, K_max, dtype=torch.long)
        dofs_t          = torch.zeros(B, K_max, dtype=torch.long)
        orbit_mask      = torch.zeros(B, K_max, dtype=torch.bool)

        for b, (sg, sc) in enumerate(zip(sgs, scaffolds)):
            tokens = sc.split(";")
            for k, tok in enumerate(tokens):
                mult, letter_idx = encode_wyckoff_token(tok)
                wyckoff_letters[b, k] = letter_idx
                multiplicities[b, k]  = mult
                orbit_mask[b, k]      = True
                if dofs_list is not None:
                    dofs_t[b, k] = dofs_list[b][k] if k < len(dofs_list[b]) else 0

        return cls(
            sg=torch.tensor(sgs, dtype=torch.long, device=device),
            wyckoff_letters=wyckoff_letters.to(device),
            multiplicities=multiplicities.to(device),
            dofs=dofs_t.to(device),
            orbit_mask=orbit_mask.to(device),
        )


# ── Space group embedding ─────────────────────────────────────────────────────

class SpaceGroupEmbedding(nn.Module):
    """Simple learned embedding for space groups 1–230."""

    def __init__(self, d_sg: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(231, d_sg)   # 0 = padding

    def forward(self, sg: torch.Tensor) -> torch.Tensor:
        """sg: LongTensor[B] → FloatTensor[B, d_sg]"""
        return self.embed(sg)


# ── COM Transformer ───────────────────────────────────────────────────────────

class ChemicalOccupancyModule(nn.Module):
    """
    Autoregressive species predictor.

    Architecture overview:
      1. Encode scaffold tau: space group embedding + Wyckoff orbit embeddings
         as a "memory" sequence for cross-attention.
      2. Autoregressively predict s_1, ..., s_K:
         - Embed previously assigned s_{1:k-1} → h_{1:k-1}
         - Append running charge-balance scalar and step-position encoding
         - Feed through causal Transformer decoder
         - Linear head → logits over 118 elements
      3. At training time: compute NLL + chemical constraint losses.
      4. At inference time: sample with optional rejection sampling.
    """

    def __init__(
        self,
        num_elements: int = 118,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_learn: int = 64,
        d_sg: int = 32,
        d_orbit: int = 128,
        max_orbits: int = 32,
        lambda_cb: float = 0.1,
        lambda_rad: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_elements = num_elements
        self.max_orbits   = max_orbits
        self.lambda_cb    = lambda_cb
        self.lambda_rad   = lambda_rad

        # ── Embedding modules ────────────────────────────────────────────────
        self.elem_embed  = IntrinsicElementEmbedding(num_elements, d_learn)
        d_e              = self.elem_embed.d_e

        self.orbit_embed = OrbitAwareEmbedding(d_e=d_e, d_h=d_orbit)
        self.sg_embed    = SpaceGroupEmbedding(d_sg=d_sg)

        # Positional embedding for each orbit step
        self.pos_embed = nn.Embedding(max_orbits + 1, d_model)

        # Project orbit embeddings to d_model
        self.orbit_proj = nn.Linear(d_orbit, d_model)
        # Project SG context to d_model (used as a prefix token in memory)
        self.sg_proj    = nn.Linear(d_sg, d_model)

        # Running charge balance: scalar → d_model projection
        self.cb_proj = nn.Linear(1, d_model)

        # ── Transformer decoder ──────────────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # ── Output head ──────────────────────────────────────────────────────
        self.out_head = nn.Linear(d_model, num_elements)

        # ── Chemical constraints ─────────────────────────────────────────────
        self.cb_constraint  = ChargeBalanceConstraint()
        self.rad_constraint = IonicRadiusConstraint()
        self.validator      = CompositionValidator()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_memory(self, tau: TopologicalScaffold) -> torch.Tensor:
        """
        Encode topological scaffold tau as a memory sequence for cross-attention.

        Memory = [sg_token, orbit_token_0, orbit_token_1, ..., orbit_token_{K-1}]
        shape: (B, 1 + K_max, d_model)
        """
        B, K = tau.wyckoff_letters.shape

        # SG prefix token
        sg_emb  = self.sg_proj(self.sg_embed(tau.sg))         # (B, d_model)
        sg_tok  = sg_emb.unsqueeze(1)                         # (B, 1, d_model)

        # Orbit tokens: we embed using a "dummy" species (all zeros → padding)
        # because we only want the structural context here, not chemistry.
        dummy_s = torch.zeros(B, K, dtype=torch.long, device=tau.sg.device)
        e_int   = self.elem_embed(dummy_s)                    # (B, K, d_e)
        h_orbit = self.orbit_embed(
            e_int,
            tau.wyckoff_letters,
            tau.multiplicities.float(),
            tau.dofs.float(),
        )                                                      # (B, K, d_orbit)
        orbit_toks = self.orbit_proj(h_orbit)                 # (B, K, d_model)

        memory = torch.cat([sg_tok, orbit_toks], dim=1)       # (B, 1+K, d_model)
        return memory

    def _embed_assigned(
        self,
        s_assigned: torch.Tensor,          # (B, k) species assigned so far
        tau: TopologicalScaffold,
        running_q: torch.Tensor,           # (B,) running charge sum
    ) -> torch.Tensor:
        """
        Embed the k already-assigned species as a query sequence.
        Returns query: (B, k, d_model)
        """
        B, k = s_assigned.shape
        device = s_assigned.device

        e_int   = self.elem_embed(s_assigned)                     # (B, k, d_e)
        letters = tau.wyckoff_letters[:, :k]                      # (B, k)
        mults   = tau.multiplicities[:, :k].float()               # (B, k)
        dofs    = tau.dofs[:, :k].float()                         # (B, k)
        h       = self.orbit_embed(e_int, letters, mults, dofs)   # (B, k, d_orbit)
        q_proj  = self.orbit_proj(h)                              # (B, k, d_model)

        # Add positional embeddings (clamp to max supported position for K > max_orbits)
        _max_pos = self.pos_embed.num_embeddings - 1
        pos_ids = torch.arange(k, device=device).clamp(max=_max_pos).unsqueeze(0).expand(B, -1)
        q_proj  = q_proj + self.pos_embed(pos_ids)

        # Inject running charge balance as an additional bias on the last token
        cb_bias = self.cb_proj(running_q.unsqueeze(-1).unsqueeze(-1))  # (B, 1, d_model)
        if k > 0:
            q_proj[:, -1:, :] = q_proj[:, -1:, :] + cb_bias

        return q_proj   # (B, k, d_model)

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        tau: TopologicalScaffold,
        s_target: torch.Tensor,    # LongTensor[B, K] ground truth species (1-indexed)
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher-forced forward pass.

        Returns dict with keys:
            logits:    (B, K, num_elements)
            loss_nll:  scalar
            loss_chem: scalar
            loss_total: scalar
        """
        B, K = s_target.shape
        device = s_target.device

        memory  = self._build_memory(tau)                     # (B, 1+K, d_model)

        # Build full query sequence (teacher forcing): s_{0}, s_{1}, ..., s_{K-1}
        # We use a BOS token (element 0 = padding as start-of-sequence)
        bos     = torch.zeros(B, 1, dtype=torch.long, device=device)
        s_in    = torch.cat([bos, s_target[:, :-1]], dim=1)   # (B, K) — shifted right

        # Compute running charge balances for teacher-forced sequence
        # (approximate: use ground-truth species to track Q)
        mult_f  = tau.multiplicities.float()                  # (B, K)
        # For simplicity, compute cumulative charge for all positions at once
        ox_tbl  = self.cb_constraint.ox_tensor                # (119, max_ox)
        # Use the first valid oxidation state as a proxy for teacher-forcing CB
        first_ox = ox_tbl[s_target][:, :, 0].clamp(-8, 8)    # (B, K)
        first_ox = torch.where(torch.isnan(first_ox), torch.zeros_like(first_ox), first_ox)
        cum_q   = torch.cumsum(mult_f * first_ox, dim=1)      # (B, K)
        # Shift: running_q at step k is the cumulative charge *before* step k
        running_q_seq = torch.cat([
            torch.zeros(B, 1, device=device),
            cum_q[:, :-1]
        ], dim=1)   # (B, K)

        # Embed all input tokens jointly
        e_int   = self.elem_embed(s_in)                       # (B, K, d_e)
        letters = tau.wyckoff_letters                         # (B, K)
        h       = self.orbit_embed(e_int, letters, mult_f, tau.dofs.float())  # (B, K, d_orbit)
        q_seq   = self.orbit_proj(h)                          # (B, K, d_model)

        # Clamp to max supported position for K > max_orbits
        _max_pos = self.pos_embed.num_embeddings - 1
        pos_ids = torch.arange(K, device=device).clamp(max=_max_pos).unsqueeze(0).expand(B, -1)
        q_seq   = q_seq + self.pos_embed(pos_ids)

        # Inject charge balance bias
        cb_bias = self.cb_proj(running_q_seq.unsqueeze(-1))   # (B, K, d_model)
        q_seq   = q_seq + cb_bias

        # Causal mask for decoder
        causal_mask = nn.Transformer.generate_square_subsequent_mask(K, device=device)

        # Transformer decode
        dec_out = self.decoder(
            tgt=q_seq,
            memory=memory,
            tgt_mask=causal_mask,
        )   # (B, K, d_model)

        logits = self.out_head(dec_out)   # (B, K, num_elements)

        # ── NLL loss ─────────────────────────────────────────────────────────
        # s_target is 1-indexed; shift to 0-indexed for CE
        target_0idx = (s_target - 1).clamp(min=0)   # (B, K)
        mask = tau.orbit_mask                         # (B, K)

        loss_nll = F.cross_entropy(
            logits.reshape(B * K, self.num_elements),
            target_0idx.reshape(B * K),
            reduction="none",
        ).reshape(B, K)
        loss_nll = (loss_nll * mask.float()).sum() / mask.float().sum().clamp(min=1)

        # ── Chemical constraint losses ────────────────────────────────────────
        # Skip chemical constraint computation entirely when weights are zero
        # (avoids 0 * NaN = NaN pathology from unmasked padding positions)
        if self.lambda_cb > 0 or self.lambda_rad > 0:
            # Only compute on valid (non-padding) positions: use orbit_mask to zero out padding mults
            mask_f  = tau.orbit_mask.float()          # (B, K) 1=valid, 0=padding
            mult_masked = mult_f * mask_f             # zero out padding multiplicities
            E_CB, _ = self.cb_constraint(s_target, mult_masked)
            E_rad   = self.rad_constraint(s_target, mult_masked)
            loss_chem = self.lambda_cb * E_CB.mean() + self.lambda_rad * E_rad.mean()
        else:
            loss_chem = torch.zeros(1, device=s_target.device).squeeze()

        loss_total = loss_nll + loss_chem

        return {
            "logits":      logits,
            "loss_nll":    loss_nll,
            "loss_chem":   loss_chem,
            "loss_total":  loss_total,
        }

    # ── Inference / generation ────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        tau: TopologicalScaffold,
        temperature: float = 1.0,
        constraint_mode: str = "reject",
        max_attempts: int = 100,
        top_k: Optional[int] = 20,
        forward_charge_check: bool = False,
        strict_ionic_oxi_check: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autoregressively sample species assignments.

        Returns:
            s:        LongTensor[B, K]   sampled species (1-indexed; 0 = failed)
            log_prob: FloatTensor[B]     sum log-prob of the assignment
            is_valid: BoolTensor[B]      True if CompositionValidator passed
        """
        B, K = tau.wyckoff_letters.shape
        device = tau.sg.device

        def _single_sample() -> Tuple[torch.Tensor, torch.Tensor]:
            memory    = self._build_memory(tau)
            s_out     = torch.zeros(B, K, dtype=torch.long, device=device)
            lp_out    = torch.zeros(B, device=device)
            running_q = torch.zeros(B, device=device)

            for k in range(K):
                if k == 0:
                    # BOS
                    s_prev = torch.zeros(B, 1, dtype=torch.long, device=device)
                else:
                    s_prev = s_out[:, :k]   # (B, k)

                query = self._embed_assigned(s_prev, tau, running_q)  # (B, k or 1, d_model)

                # Causal mask
                qlen = query.shape[1]
                if qlen > 1:
                    causal_mask = nn.Transformer.generate_square_subsequent_mask(qlen, device=device)
                else:
                    causal_mask = None

                dec_out = self.decoder(
                    tgt=query,
                    memory=memory,
                    tgt_mask=causal_mask,
                )   # (B, qlen, d_model)

                logits_k = self.out_head(dec_out[:, -1, :])   # (B, num_elements)

                # Temperature + top-k masking
                if top_k is not None and top_k < self.num_elements:
                    topk_vals, _ = logits_k.topk(top_k, dim=-1)
                    threshold = topk_vals[:, -1].unsqueeze(-1)
                    logits_k = logits_k.masked_fill(logits_k < threshold, -1e9)

                if temperature != 1.0 and temperature > 0:
                    logits_k = logits_k / temperature

                hard_mask_before_charge_check = logits_k <= -1e8
                if forward_charge_check:
                    mask_value = -1e9
                    for b in range(B):
                        if not bool(tau.orbit_mask[b, k].item()):
                            continue
                        original_logits_b = logits_k[b].clone()
                        candidate_idx = torch.nonzero(logits_k[b] > mask_value, as_tuple=False).flatten()
                        if candidate_idx.numel() == 0:
                            continue
                        prefix_positions = [idx for idx in range(k) if bool(tau.orbit_mask[b, idx].item())]
                        prefix_s = [int(s_out[b, idx].item()) for idx in prefix_positions]
                        prefix_mults = [int(tau.multiplicities[b, idx].item()) for idx in prefix_positions]
                        remaining_mults = [
                            int(tau.multiplicities[b, idx].item())
                            for idx in range(k + 1, K)
                            if bool(tau.orbit_mask[b, idx].item())
                        ]
                        next_mult = int(tau.multiplicities[b, k].item())
                        candidate_zs = [int(idx.item()) + 1 for idx in candidate_idx]
                        feasible = self.validator.filter_feasible_candidates(
                            prefix_s,
                            prefix_mults,
                            next_mult,
                            remaining_mults,
                            candidate_zs,
                            strict_ionic_oxi_check=strict_ionic_oxi_check,
                        )
                        if not any(feasible) and candidate_idx.numel() < self.num_elements:
                            expanded_idx = torch.nonzero(
                                ~hard_mask_before_charge_check[b],
                                as_tuple=False,
                            ).flatten()
                            if expanded_idx.numel() == 0:
                                continue
                            expanded_zs = [int(idx.item()) + 1 for idx in expanded_idx]
                            expanded_feasible = self.validator.filter_feasible_candidates(
                                prefix_s,
                                prefix_mults,
                                next_mult,
                                remaining_mults,
                                expanded_zs,
                                strict_ionic_oxi_check=strict_ionic_oxi_check,
                            )
                            if any(expanded_feasible):
                                logits_k[b] = torch.full_like(logits_k[b], mask_value)
                                keep_idx = expanded_idx[
                                    torch.tensor(expanded_feasible, dtype=torch.bool, device=device)
                                ]
                                logits_k[b, keep_idx] = original_logits_b[keep_idx]
                                continue
                        if any(feasible):
                            reject_idx = candidate_idx[
                                ~torch.tensor(feasible, dtype=torch.bool, device=device)
                            ]
                            logits_k[b, reject_idx] = mask_value

                hard_mask = logits_k <= -1e8

                # Clamp to prevent overflow before softmax
                logits_k = logits_k - logits_k.amax(dim=-1, keepdim=True)
                logits_k = logits_k.clamp(min=-50.0)
                probs_k = F.softmax(logits_k, dim=-1)           # (B, num_elements)
                probs_k = torch.nan_to_num(probs_k, nan=0.0, posinf=0.0, neginf=0.0)
                probs_k = probs_k.masked_fill(hard_mask, 0.0).clamp(min=0.0)

                # Rarely, aggressive masking or CUDA numerical issues can leave a
                # row with NaN/zero probability mass. Fall back only over logits
                # that were not hard-masked, so environment element masks remain
                # strict.
                row_sums = probs_k.sum(dim=-1, keepdim=True)
                invalid_rows = row_sums.squeeze(1) <= 0
                if invalid_rows.any():
                    fallback_mask = (~hard_mask[invalid_rows]).float()
                    fallback_sums = fallback_mask.sum(dim=-1, keepdim=True)
                    fallback_invalid = fallback_sums.squeeze(1) <= 0
                    if fallback_invalid.any():
                        fallback_mask[fallback_invalid] = 1.0
                        fallback_sums = fallback_mask.sum(dim=-1, keepdim=True)
                    probs_k[invalid_rows] = fallback_mask / fallback_sums.clamp(min=1.0)
                    row_sums = probs_k.sum(dim=-1, keepdim=True)

                probs_k = probs_k / row_sums
                s_k_0   = torch.multinomial(probs_k, 1).squeeze(1)  # (B,) 0-indexed
                s_k     = s_k_0 + 1                                  # 1-indexed

                lp_out    = lp_out + torch.log(probs_k.gather(1, s_k_0.unsqueeze(1)).squeeze(1) + 1e-12)
                s_out[:, k] = s_k

                # Update running charge (greedy ox state for speed)
                ox_tbl = self.cb_constraint.ox_tensor
                ox_k   = ox_tbl[s_k][:, 0].clamp(-8, 8)
                ox_k   = torch.where(torch.isnan(ox_k), torch.zeros_like(ox_k), ox_k)
                running_q = running_q + tau.multiplicities[:, k].float() * ox_k

            return s_out, lp_out

        if constraint_mode in ("soft", "greedy"):
            s, lp = _single_sample()
            is_valid = torch.ones(B, dtype=torch.bool, device=device)
            return s, lp, is_valid

        # Rejection sampling
        best_s  = torch.zeros(B, K, dtype=torch.long, device=device)
        best_lp = torch.full((B,), -1e9, device=device)
        is_valid = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_attempts):
            if is_valid.all():
                break
            s_cand, lp_cand = _single_sample()

            for b in range(B):
                if is_valid[b]:
                    continue
                s_b    = s_cand[b].tolist()
                mults_b = tau.multiplicities[b].tolist()
                K_b    = int(tau.orbit_mask[b].sum().item())
                ok, _  = self.validator.validate_all(s_b[:K_b], mults_b[:K_b])
                if ok:
                    best_s[b]  = s_cand[b]
                    best_lp[b] = lp_cand[b]
                    is_valid[b] = True

        # For structures that never passed, use last sample anyway
        unresolved = ~is_valid
        if unresolved.any():
            s_last, lp_last = _single_sample()
            best_s[unresolved]  = s_last[unresolved]
            best_lp[unresolved] = lp_last[unresolved]

        return best_s, best_lp, is_valid
