"""Stage III: Structured Wyckoff-aware Generation (SWG)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

#: Architecture of every released Stage-III production checkpoint.
PRODUCTION_ARCH = {
    "hidden_dim": 192,
    "adapter_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "context_mode": "discrete_chem_topology_only",
}


def build_context(adapter: "WyckoffStage2InputAdapter", batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Run the input adapter and return the fused conditioning vector."""
    return adapter(batch)["fused_context"]


class WyckoffStage2InputAdapter(nn.Module):
    def __init__(self, lattice_dim: int, chem_dim: int, max_orbits: int, max_orbit_dof: int,
                 n_scaffolds: int, hidden_dim: int = 128, scaffold_embed_dim: int = 16,
                 context_mode: str = 'full', max_feature_dim: int = 64, max_spacegroup: int = 230,
                 include_topology_summary: bool = True, include_spacegroup_embed: bool = True):
        super().__init__()
        if context_mode not in {'full', 'discrete_chem_only', 'discrete_chem_topology_only', 'non_fiber_chem_only', 'topology_chem_only'}:
            raise ValueError(f'Unsupported context_mode: {context_mode}')
        self.context_mode = context_mode
        self.max_orbits = max_orbits
        self.max_orbit_dof = max_orbit_dof
        self.max_feature_dim = max(max_feature_dim, 1)
        self.include_topology_summary = include_topology_summary
        self.include_spacegroup_embed = include_spacegroup_embed
        self.scaffold_emb = nn.Embedding(n_scaffolds, scaffold_embed_dim)
        if include_spacegroup_embed:
            self.spacegroup_emb = nn.Embedding(max_spacegroup + 1, scaffold_embed_dim)
        else:
            self.spacegroup_emb = None
        self.lattice_proj = nn.Sequential(nn.Linear(lattice_dim, hidden_dim), nn.SiLU())
        self.chem_proj = nn.Sequential(nn.Linear(max(chem_dim, 1), hidden_dim), nn.SiLU())
        self.orbit_value_proj = nn.Sequential(nn.Linear(max_orbit_dof, hidden_dim), nn.SiLU())
        self.orbit_dof_emb = nn.Embedding(8, hidden_dim)
        self.orbit_idx_emb = nn.Embedding(max_orbits + 1, hidden_dim)
        if include_topology_summary:
            self.topology_proj = nn.Sequential(nn.Linear(8, hidden_dim), nn.SiLU())
        else:
            self.topology_proj = None
        fuse_in_dim = hidden_dim * 4 + scaffold_embed_dim
        if include_topology_summary:
            fuse_in_dim += hidden_dim
        if include_spacegroup_embed:
            fuse_in_dim += scaffold_embed_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.context_mode == 'full':
            lattice_in = batch['lattice_block']
        else:
            lattice_in = torch.zeros_like(batch['lattice_block'])
        lattice_h = self.lattice_proj(lattice_in)
        chem_in = batch['chem_block']
        if chem_in.shape[-1] == 0:
            chem_in = torch.zeros((chem_in.shape[0], 1), device=chem_in.device, dtype=chem_in.dtype)
        chem_h = self.chem_proj(chem_in)

        B, O, D = batch['orbit_tensor'].shape
        if self.context_mode == 'full':
            orbit_in = batch['orbit_tensor']
            orbit_active_mask = batch['orbit_active_mask']
        elif self.context_mode in {'discrete_chem_topology_only', 'topology_chem_only'}:
            orbit_in = torch.zeros_like(batch['orbit_tensor'])
            orbit_active_mask = batch['orbit_active_mask']
        else:
            orbit_in = torch.zeros_like(batch['orbit_tensor'])
            orbit_active_mask = torch.zeros_like(batch['orbit_active_mask'])
        orbit_val_h = self.orbit_value_proj(orbit_in.reshape(B * O, D)).reshape(B, O, -1)
        orbit_dof_h = self.orbit_dof_emb(batch['orbit_dof'].long().clamp(min=0, max=7))
        orbit_idx = torch.arange(O, device=batch['orbit_tensor'].device).unsqueeze(0).expand(B, O)
        orbit_idx_h = self.orbit_idx_emb((orbit_idx + 1).clamp(max=self.orbit_idx_emb.num_embeddings - 1))
        orbit_h = (orbit_val_h + orbit_dof_h + orbit_idx_h) * orbit_active_mask.unsqueeze(-1)
        orbit_h_mean = orbit_h.sum(dim=1) / orbit_active_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)

        if self.context_mode == 'full':
            lattice_mask = batch['variable_mask'][:, :6].mean(dim=1, keepdim=True)
            if batch['variable_mask'].shape[1] > 6:
                orbit_mask = batch['variable_mask'][:, 6:].mean(dim=1, keepdim=True)
            else:
                orbit_mask = torch.zeros((B, 1), device=lattice_h.device, dtype=lattice_h.dtype)
        else:
            lattice_mask = torch.zeros((B, 1), device=lattice_h.device, dtype=lattice_h.dtype)
            orbit_mask = batch['orbit_active_mask'].mean(dim=1, keepdim=True).to(lattice_h.dtype)
        var_mask_h = torch.cat([lattice_mask, orbit_mask], dim=1)
        var_mask_h = torch.cat([var_mask_h, torch.zeros((B, lattice_h.shape[-1] - 2), device=lattice_h.device)], dim=1)

        scaffold_h = self.scaffold_emb(batch['scaffold_idx'].long().clamp(min=0, max=self.scaffold_emb.num_embeddings - 1))
        if self.context_mode in {'non_fiber_chem_only', 'topology_chem_only'}:
            # topology_chem_only drops the per-scaffold ID lookup so the model
            # conditions only on generalizable Wyckoff topology + chemistry, enabling
            # generation for train-external space groups (the SG is imposed at decode).
            scaffold_h = torch.zeros_like(scaffold_h)
        if self.spacegroup_emb is not None:
            sg_h = self.spacegroup_emb(batch['spacegroup'].long().clamp(min=0, max=self.spacegroup_emb.num_embeddings - 1))
        else:
            sg_h = torch.zeros((B, scaffold_h.shape[-1]), device=lattice_h.device, dtype=lattice_h.dtype)

        # Topology summary without using target continuous values.
        active = batch['orbit_active_mask']
        k_orbits = active.sum(dim=1, keepdim=True) / float(max(self.max_orbits, 1))
        dof_hist = []
        active_count = active.sum(dim=1, keepdim=True).clamp(min=1.0)
        for dof in range(4):
            count = (((batch['orbit_dof'] == dof).to(active.dtype)) * active).sum(dim=1, keepdim=True) / active_count
            dof_hist.append(count)
        active_dim_frac = batch['orbit_value_mask'].sum(dim=(1, 2), keepdim=False).unsqueeze(-1) / float(max(self.max_orbits * self.max_orbit_dof, 1))
        feature_dim_norm = batch['feature_dim'].to(lattice_h.dtype).unsqueeze(-1) / float(self.max_feature_dim)
        mean_dof = ((batch['orbit_dof'].to(active.dtype) * active).sum(dim=1, keepdim=True) / active_count)
        topo_stats = torch.cat([k_orbits, *dof_hist, active_dim_frac, feature_dim_norm, mean_dof], dim=-1)
        if not self.include_topology_summary:
            topo_h = torch.zeros_like(lattice_h)
            sg_h = torch.zeros_like(sg_h)
        elif self.context_mode in {'discrete_chem_only', 'non_fiber_chem_only'}:
            topo_h = torch.zeros_like(lattice_h)
            sg_h = torch.zeros_like(sg_h)
        else:
            topo_h = self.topology_proj(topo_stats)
            if self.context_mode == 'topology_chem_only':
                # keep generalizable Wyckoff topology, drop the per-SG-number embedding
                sg_h = torch.zeros_like(sg_h)
        fuse_parts = [lattice_h, chem_h, orbit_h_mean, var_mask_h]
        if self.include_topology_summary:
            fuse_parts.append(topo_h)
        fuse_parts.append(scaffold_h)
        if self.include_spacegroup_embed:
            fuse_parts.append(sg_h)
        fused = self.fuse(torch.cat(fuse_parts, dim=-1))
        return {
            'fused_context': fused,
            'lattice_context': lattice_h,
            'chem_context': chem_h,
            'orbit_context': orbit_h,
            'orbit_pooled': orbit_h_mean,
            'topology_context': topo_h,
            'scaffold_context': scaffold_h,
            'spacegroup_context': sg_h,
            'context_mode': self.context_mode,
        }


class TimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class StructuredWyckoffVectorField(nn.Module):
    def __init__(self, adapter_dim: int, max_orbits: int, max_orbit_dof: int, hidden_dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.max_orbits = max_orbits
        self.max_orbit_dof = max_orbit_dof
        self.hidden_dim = hidden_dim

        self.time_emb = TimeEmbed(hidden_dim)
        self.lattice_state = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU())
        self.orbit_value_proj = nn.Sequential(nn.Linear(max_orbit_dof, hidden_dim), nn.SiLU())
        self.orbit_dof_emb = nn.Embedding(8, hidden_dim)
        self.orbit_idx_emb = nn.Embedding(max_orbits + 1, hidden_dim)
        self.t_to_orbit = nn.Linear(hidden_dim, hidden_dim)
        self.context_to_orbit = nn.Linear(adapter_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2, dropout=0.0,
            activation='gelu', batch_first=True
        )
        self.orbit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.lattice_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + adapter_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )
        self.orbit_head = nn.Sequential(
            nn.Linear(hidden_dim + adapter_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_orbit_dof),
        )

    def forward(self, context: torch.Tensor, lattice_xt: torch.Tensor, orbit_xt: torch.Tensor, orbit_dof: torch.Tensor, orbit_active_mask: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, n_orbits, _ = orbit_xt.shape
        t_h = self.time_emb(t)
        l_h = self.lattice_state(lattice_xt)

        orbit_val_h = self.orbit_value_proj(orbit_xt.reshape(bsz * n_orbits, -1)).reshape(bsz, n_orbits, self.hidden_dim)
        idx = torch.arange(n_orbits, device=orbit_xt.device).unsqueeze(0).expand(bsz, n_orbits)
        orbit_h = (
            orbit_val_h
            + self.orbit_dof_emb(orbit_dof.clamp(min=0, max=7))
            + self.orbit_idx_emb((idx + 1).clamp(max=self.orbit_idx_emb.num_embeddings - 1))
            + self.t_to_orbit(t_h).unsqueeze(1)
            + self.context_to_orbit(context).unsqueeze(1)
        )
        orbit_h = orbit_h * orbit_active_mask.unsqueeze(-1)
        pad_mask = orbit_active_mask < 0.5
        orbit_h = self.orbit_encoder(orbit_h, src_key_padding_mask=pad_mask)
        orbit_h = orbit_h * orbit_active_mask.unsqueeze(-1)
        orbit_pool = orbit_h.sum(dim=1) / orbit_active_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        lattice_v = self.lattice_head(torch.cat([context, t_h, l_h, orbit_pool], dim=-1))

        orbit_context = torch.cat([orbit_h, context.unsqueeze(1).expand(-1, n_orbits, -1)], dim=-1)
        orbit_v = self.orbit_head(orbit_context) * orbit_active_mask.unsqueeze(-1)
        return {'lattice_v': lattice_v, 'orbit_v': orbit_v}


def sample_trajectories(model, adapter, batch: Dict[str, torch.Tensor], temp: float, steps: int):
    context = build_context(adapter, batch)
    lattice = torch.randn_like(batch['lattice_block']) * temp
    orbit = torch.rand_like(batch['orbit_tensor'])
    dt = 1.0 / steps
    for s in range(steps):
        t = torch.full((lattice.shape[0], 1), (s + 0.5) / steps, device=lattice.device, dtype=lattice.dtype)
        pred = model(context, lattice, orbit, batch['orbit_dof'], batch['orbit_active_mask'], t)
        lattice = lattice + dt * pred['lattice_v']
        orbit = torch.remainder(orbit + dt * pred['orbit_v'], 1.0) * batch['orbit_value_mask']
    return lattice, orbit

