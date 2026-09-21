"""Minimal example for inspecting HTS, COM, and SWG interfaces."""

from __future__ import annotations

import torch

from ufo_mgen.generation import RouteManifest
from ufo_mgen.models import (
    PRODUCTION_ARCH,
    CoreHierStage1,
    StructuredWyckoffVectorField,
    WyckoffStage2InputAdapter,
    build_com,
    build_context,
    sample_trajectories,
)
from ufo_mgen.models.stage1_hts import TARGET_ORDER, LabelVocab
from ufo_mgen.train import fm_loss, sample_training_bridge


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> int:
    print("=== Stage I topology interface ===")
    manifest = RouteManifest.load()
    s = manifest.summary()
    print(f"  routes              {s['n_routes']}")
    print(f"  distinct scaffolds  {s['n_unique_scaffolds']}")
    print(f"  space groups        {s['n_spacegroups']}")
    print(f"  unique route IDs    {s['n_unique_route_ids']}")
    print("\n  The released production topology interface is materialized")
    print("  through the route manifest.")

    print("\n=== Stage I -- HTS ===")
    vocabs = {t: LabelVocab.from_train_series([f"{t}_{i}" for i in range(50)])
              for t in TARGET_ORDER}
    hts = CoreHierStage1(input_dim=123, vocabs=vocabs, hidden_dim=256, embed_dim=64)
    print(f"  label chain           {' -> '.join(TARGET_ORDER)}")
    print("  synthetic vocabularies are used below only to exercise the public interface")
    logits = hts.forward_greedy(torch.randn(4, 123))
    print(f"  greedy decode         {[f'{k}:{tuple(v.shape)}' for k, v in logits.items()]}")

    print("\n=== Stage II -- COM ===")
    com = build_com()
    from ufo_mgen.models.stage2_com import PRODUCTION_ARCH as COM_ARCH
    print(f"  ChemicalOccupancyModule  {n_params(com):>12,} params")
    print(f"  architecture          {COM_ARCH}")

    print("\n=== Stage III -- SWG ===")
    print(f"  architecture          {PRODUCTION_ARCH}")
    B, O, D = 4, 6, 3
    adapter = WyckoffStage2InputAdapter(
        lattice_dim=6, chem_dim=5, max_orbits=O, max_orbit_dof=D, n_scaffolds=1,
        hidden_dim=128, context_mode="discrete_chem_topology_only", max_feature_dim=24,
    )
    model = StructuredWyckoffVectorField(128, O, D, 192, 4, 2)
    print(f"  adapter               {n_params(adapter):>12,} params")
    print(f"  vector field          {n_params(model):>12,} params")

    batch = {
        "lattice_block": torch.randn(B, 6),
        "orbit_tensor": torch.rand(B, O, D),
        "orbit_active_mask": torch.ones(B, O),
        "orbit_value_mask": torch.ones(B, O, D),
        "orbit_dof": torch.full((B, O), D, dtype=torch.long),
        "chem_block": torch.randn(B, 5),
        "scaffold_idx": torch.zeros(B, dtype=torch.long),
        "spacegroup": torch.full((B,), 225, dtype=torch.long),
        "feature_dim": torch.full((B,), 24, dtype=torch.long),
        "variable_mask": torch.ones(B, 24),
    }

    bridge = sample_training_bridge(batch, "cpu")
    context = build_context(adapter, batch)
    pred = model(context, bridge["lattice_xt"], bridge["orbit_xt"],
                 batch["orbit_dof"], batch["orbit_active_mask"], bridge["t"])
    loss = fm_loss(pred, bridge, batch["orbit_value_mask"]).mean()
    print(f"\n  one training step     loss = {loss.item():.4f}")
    print(f"    coordinates move on the torus: xt in "
          f"[{bridge['orbit_xt'].min():.3f}, {bridge['orbit_xt'].max():.3f}]")

    with torch.no_grad():
        lattice, orbit = sample_trajectories(model, adapter, batch, temp=0.5, steps=48)
    print(f"\n  one sampling run      48 ODE steps")
    print(f"    lattice  {tuple(lattice.shape)}")
    print(f"    coords   {tuple(orbit.shape)}  in "
          f"[{orbit.min():.3f}, {orbit.max():.3f}]  (wrapped to the unit cell)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
