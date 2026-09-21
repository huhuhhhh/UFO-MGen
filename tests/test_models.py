"""Tests for the three model stages.

No weights required: every stage is built at its production configuration and
exercised on synthetic tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _batch(B=4, O=6, D=3, feat=24):
    import torch

    return {
        "lattice_block": torch.randn(B, 6),
        "orbit_tensor": torch.rand(B, O, D),
        "orbit_active_mask": torch.ones(B, O),
        "orbit_value_mask": torch.ones(B, O, D),
        "orbit_dof": torch.full((B, O), D, dtype=torch.long),
        "chem_block": torch.randn(B, 5),
        "scaffold_idx": torch.zeros(B, dtype=torch.long),
        "spacegroup": torch.full((B,), 225, dtype=torch.long),
        "feature_dim": torch.full((B,), feat, dtype=torch.long),
        "variable_mask": torch.ones(B, feat),
    }


def _swg(O=6, D=3, feat=24):
    from ufo_mgen.models import StructuredWyckoffVectorField, WyckoffStage2InputAdapter

    adapter = WyckoffStage2InputAdapter(
        lattice_dim=6, chem_dim=5, max_orbits=O, max_orbit_dof=D, n_scaffolds=1,
        hidden_dim=128, context_mode="discrete_chem_topology_only", max_feature_dim=feat,
    )
    model = StructuredWyckoffVectorField(128, O, D, 192, 4, 2)
    return model, adapter


# ---------------------------------------------------------------- Stage I

def test_stage1_label_vocab_is_a_working_dataclass():
    from ufo_mgen.models.stage1_hts import LabelVocab

    v = LabelVocab.from_train_series(["a", "b", "a", "c"])
    assert v.size() == 4                    # UNK + three labels
    assert v.encode("b") == v.stoi["b"]
    assert v.encode("not-seen") == 0        # falls back to UNK


def test_stage1_hierarchy_conditions_on_parents():
    from ufo_mgen.models.stage1_hts import PARENT_MAP, TARGET_ORDER

    assert TARGET_ORDER == ["spacegroup", "k_orbits", "wyckoff_config"]
    assert PARENT_MAP["spacegroup"] == []
    assert PARENT_MAP["k_orbits"] == ["spacegroup"]
    assert PARENT_MAP["wyckoff_config"] == ["spacegroup", "k_orbits"]


def test_stage1_greedy_decode_emits_all_levels():
    import torch

    from ufo_mgen.models import CoreHierStage1
    from ufo_mgen.models.stage1_hts import TARGET_ORDER, LabelVocab

    vocabs = {t: LabelVocab.from_train_series([f"{t}{i}" for i in range(7)])
              for t in TARGET_ORDER}
    model = CoreHierStage1(input_dim=123, vocabs=vocabs, hidden_dim=64, embed_dim=16)
    logits = model.forward_greedy(torch.randn(3, 123))
    for t in TARGET_ORDER:
        assert logits[t].shape == (3, vocabs[t].size())


def test_stage1_composition_featuriser():
    """Layout: element fractions, log(1+N), then a four-slot size one-hot."""
    from ufo_mgen.models import featurize_formula

    v = featurize_formula("TiO2", 6, "N0-20")
    assert v.ndim == 1
    assert abs(v[:-5].sum() - 1.0) < 1e-5, "element fractions should sum to 1"
    assert v[-5] > 0, "log(1+N) slot"
    assert v[-4:].sum() == 1.0, "exactly one size bucket should be hot"
    assert v[-4] == 1.0, "N0-20 is the first bucket"


def test_stage1_featuriser_tolerates_unknown_bucket():
    """An unrecognised bucket must zero the size slots, not index past the end."""
    from ufo_mgen.models import featurize_formula

    v = featurize_formula("TiO2", 6, "not-a-bucket")
    assert v[-4:].sum() == 0.0
    assert abs(v[:-5].sum() - 1.0) < 1e-5


# ---------------------------------------------------------------- Stage II

def test_stage2_builds_at_production_arch():
    from ufo_mgen.models import build_com
    from ufo_mgen.models.stage2_com import PRODUCTION_ARCH

    assert PRODUCTION_ARCH["d_model"] == 256
    assert PRODUCTION_ARCH["n_heads"] == 8
    assert PRODUCTION_ARCH["n_layers"] == 4
    com = build_com()
    assert sum(p.numel() for p in com.parameters()) > 1_000_000


def test_stage2_exposes_chemical_constraints():
    from ufo_mgen.models.stage2_com import (
        ChargeBalanceConstraint,
        CompositionValidator,
        IonicRadiusConstraint,
    )

    for cls in (ChargeBalanceConstraint, IonicRadiusConstraint, CompositionValidator):
        assert cls is not None


# ---------------------------------------------------------------- Stage III

def test_stage3_production_arch():
    from ufo_mgen.models import PRODUCTION_ARCH

    assert PRODUCTION_ARCH == {
        "hidden_dim": 192, "adapter_dim": 128, "n_heads": 4, "n_layers": 2,
        "context_mode": "discrete_chem_topology_only",
    }


def test_stage3_forward_shapes():
    import torch

    from ufo_mgen.models import build_context

    model, adapter = _swg()
    b = _batch()
    ctx = build_context(adapter, b)
    t = torch.rand(4, 1)
    out = model(ctx, b["lattice_block"], b["orbit_tensor"],
                b["orbit_dof"], b["orbit_active_mask"], t)
    assert out["lattice_v"].shape == (4, 6)
    assert out["orbit_v"].shape == (4, 6, 3)


def test_stage3_masks_padding_orbits():
    """Inactive orbits must receive no velocity."""
    import torch

    from ufo_mgen.models import build_context

    model, adapter = _swg()
    b = _batch()
    b["orbit_active_mask"][:, 3:] = 0.0
    ctx = build_context(adapter, b)
    out = model(ctx, b["lattice_block"], b["orbit_tensor"],
                b["orbit_dof"], b["orbit_active_mask"], torch.rand(4, 1))
    assert torch.allclose(out["orbit_v"][:, 3:], torch.zeros_like(out["orbit_v"][:, 3:]))


def test_flow_matching_bridge_is_wrapped():
    """The coordinate bridge must live on the torus, and use shortest paths."""
    import torch

    from ufo_mgen.train import sample_training_bridge
    from ufo_mgen.train.flow_matching import wrap_delta

    b = _batch()
    bridge = sample_training_bridge(b, "cpu")
    assert (bridge["orbit_xt"] >= 0).all() and (bridge["orbit_xt"] <= 1).all()
    assert (bridge["target_orbit_v"].abs() <= 0.5 + 1e-6).all(), "delta must be the short way round"

    # 0.05 and 0.95 are neighbours on the torus, 0.1 apart, not 0.9
    d = wrap_delta(torch.tensor([0.05]), torch.tensor([0.95]))
    assert abs(float(d) - 0.10) < 1e-6


def test_flow_matching_loss_is_masked():
    """Masked-out coordinate slots must not contribute to the loss."""
    import torch

    from ufo_mgen.models import build_context
    from ufo_mgen.train import fm_loss, sample_training_bridge

    model, adapter = _swg()
    b = _batch()
    bridge = sample_training_bridge(b, "cpu")
    ctx = build_context(adapter, b)
    pred = model(ctx, bridge["lattice_xt"], bridge["orbit_xt"],
                 b["orbit_dof"], b["orbit_active_mask"], bridge["t"])

    full = fm_loss(pred, bridge, b["orbit_value_mask"])
    half_mask = b["orbit_value_mask"].clone()
    half_mask[:, :, 2:] = 0.0
    partial = fm_loss(pred, bridge, half_mask)
    assert full.shape == partial.shape == (4,)
    assert not torch.allclose(full, partial)


def test_stage3_sampling_wraps_coordinates():
    import torch

    from ufo_mgen.models import sample_trajectories

    model, adapter = _swg()
    b = _batch()
    with torch.no_grad():
        lattice, orbit = sample_trajectories(model, adapter, b, temp=0.5, steps=8)
    assert lattice.shape == (4, 6)
    assert (orbit >= 0).all() and (orbit <= 1).all()


# ---------------------------------------------------------------- routes

def test_route_manifest_matches_the_corpus():
    from ufo_mgen.dataset import load_scaffolds
    from ufo_mgen.generation import RouteManifest

    m = RouteManifest.load()
    s = m.summary()
    assert s["n_routes"] == 124
    assert s["n_unique_scaffolds"] == 119
    assert s["n_spacegroups"] == 35
    assert s["total_n_train"] == 32550        # route-weighted, see README
    assert s["n_unique_route_ids"] == 124
    assert s["n_unique_route_keys"] == 119
    assert set(m.unique_scaffolds()) == set(load_scaffolds().scaffold_id)


def test_route_checkpoints_are_120_not_124():
    """Five scaffolds occur twice in the production plan, so rows outnumber models."""
    import pandas as pd

    inv = pd.read_csv(REPO / "data" / "route_inventory_124.csv")
    assert len(inv) == 124
    assert inv.checkpoint.nunique() == 120


def test_model_manifest_covers_every_checkpoint():
    import pandas as pd

    m = pd.read_csv(REPO / "manifests" / "UFO_MGen_models.csv")
    assert (m.model_id.str.startswith("stage3_")).sum() == 120
    assert {"stage1_hts", "stage2_com"} <= set(m.model_id)
    assert len(m) == 122
    assert m.sha256.str.len().eq(64).all()

    inv = pd.read_csv(REPO / "data" / "route_inventory_124.csv")
    assert set(inv.checkpoint) <= set(m.checkpoint_filename)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except Exception as exc:
            failed += 1; print(f"  FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
