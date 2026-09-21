"""Tests for post-generation screening and the MLIP harness interface.

Everything here runs on toy structures. No universal potential is loaded: the
harness is tested for dispatch and validation only, since actually running
CHGNet, MACE or MatterSim is not a unit test.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _nacl(a: float = 5.64):
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), ["Na", "Cl"],
                                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def _perovskite():
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup("Pm-3m", Lattice.cubic(3.905),
                                     ["Sr", "Ti", "O"],
                                     [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]])


def _overlapping():
    from pymatgen.core import Lattice, Structure

    return Structure(Lattice.cubic(3.0), ["Na", "Cl"], [[0, 0, 0], [0.02, 0.0, 0.0]])


# ---------------------------------------------------------------- thresholds

def test_threshold_constants():
    from ufo_mgen.screening import DEFAULT_THRESHOLDS as t

    assert t.geometry.min_valid_dist == 0.9
    assert t.geometry.min_pair_ratio == 0.5
    assert (t.geometry.min_vpa_scale, t.geometry.max_vpa_scale) == (0.5, 1.6)
    assert (t.novelty.ltol, t.novelty.stol, t.novelty.angle_tol) == (0.1, 0.2, 5.0)
    assert (t.novelty.batch_ltol, t.novelty.batch_stol) == (0.15, 0.25)
    assert t.mlff.max_force_ev_per_a == 100.0
    assert t.mlff.risk_positive_energy_ev_per_atom == 0.1
    assert t.mlff.risk_force_ev_per_a == 50.0
    assert t.mlff.risk_stress_l2 == 30.0
    assert (t.relax.fmax, t.relax.steps) == (0.15, 50)
    assert t.relax.max_relaxed_rmsd == 2.0
    assert t.relax.min_relaxed_pair_distance == 0.7
    assert t.relax.max_formation_energy_ev_per_atom == 0.0


def test_quick_ef_prescreen_is_disabled_by_default():
    """The aggressive stage must be opt-in."""
    from ufo_mgen.screening import DEFAULT_THRESHOLDS

    assert DEFAULT_THRESHOLDS.quick_ef_prescreen.enabled is False


def test_batch_tolerances_are_looser_than_reference():
    from ufo_mgen.screening import DEFAULT_THRESHOLDS as t

    assert t.novelty.batch_ltol > t.novelty.ltol
    assert t.novelty.batch_stol > t.novelty.stol


def test_complexity_presets():
    from ufo_mgen.screening import DEFAULT_THRESHOLDS, GEOMETRY_PRESETS

    assert set(GEOMETRY_PRESETS) == {"low", "mid", "high"}
    low = DEFAULT_THRESHOLDS.geometry_for("low")
    high = DEFAULT_THRESHOLDS.geometry_for("high")
    assert low.min_valid_dist < high.min_valid_dist, (
        "complex scaffolds should face a stricter distance floor"
    )
    assert DEFAULT_THRESHOLDS.geometry_for(None) is DEFAULT_THRESHOLDS.geometry
    try:
        DEFAULT_THRESHOLDS.geometry_for("enormous")
    except KeyError:
        return
    raise AssertionError("an unknown complexity band should raise")


# ---------------------------------------------------------------- geometry

def test_geometry_accepts_a_sound_cell():
    from ufo_mgen.screening import check_geometry

    r = check_geometry(_nacl())
    assert r.passed and r.reason == "ok"
    assert r.diagnostics["min_pair_distance"] > 2.0


def test_geometry_rejects_overlapping_atoms():
    from ufo_mgen.screening import check_geometry

    r = check_geometry(_overlapping())
    assert not r.passed
    assert r.reason == "atoms_too_close"
    assert r.diagnostics["min_pair_distance"] < 0.9


def test_geometry_rejects_an_absurd_cell_volume():
    from ufo_mgen.screening import check_geometry

    r = check_geometry(_nacl(a=40.0))
    assert not r.passed
    assert r.reason == "implausible_cell_volume"


def test_geometry_helpers():
    from ufo_mgen.screening import (
        min_pair_distance,
        min_pair_ratio,
        volume_per_atom_scale,
    )

    s = _nacl()
    assert min_pair_distance(s) > 0
    assert min_pair_ratio(s) > 0
    assert volume_per_atom_scale(s) > 0


# ---------------------------------------------------------------- scaffold

def test_scaffold_check_accepts_the_generation_target():
    from ufo_mgen.screening import check_scaffold

    r = check_scaffold(_nacl(), target_spacegroup=225, target_scaffold="4a;4b")
    assert r.passed, r.reason
    assert r.diagnostics["detected_spacegroup"] == 225


def test_scaffold_check_catches_a_spacegroup_mismatch():
    from ufo_mgen.screening import check_scaffold

    r = check_scaffold(_nacl(), target_spacegroup=221, target_scaffold="1a;1b")
    assert not r.passed
    assert r.reason == "spacegroup_mismatch"


def test_scaffold_check_catches_a_scaffold_mismatch():
    from ufo_mgen.screening import check_scaffold

    r = check_scaffold(_nacl(), target_spacegroup=225, target_scaffold="4a;4a")
    assert not r.passed
    assert r.reason == "scaffold_mismatch"


def test_scaffold_check_without_a_target_just_reports():
    from ufo_mgen.screening import check_scaffold

    r = check_scaffold(_nacl())
    assert r.passed
    assert r.diagnostics["detected_scaffold"]


# ---------------------------------------------------------------- novelty

def test_novelty_rejects_an_exact_duplicate():
    from ufo_mgen.screening import DUPLICATE, check_novelty

    r = check_novelty(_nacl(), [_nacl()])
    assert not r.passed
    assert r.reason == DUPLICATE


def test_novelty_passes_something_new():
    from ufo_mgen.screening import check_novelty

    r = check_novelty(_perovskite(), [_nacl()])
    assert r.passed


def test_novelty_abstains_without_a_reference():
    from ufo_mgen.screening import check_novelty

    r = check_novelty(_nacl(), [])
    assert r.passed and r.skipped


def test_same_composition_isomer_is_allowed_and_marked():
    """A new polymorph of a known composition must not be thrown away."""
    from pymatgen.core import Lattice, Structure

    from ufo_mgen.screening import ISOMER, classify_novelty

    rocksalt = _nacl()
    caesium_chloride_type = Structure(Lattice.cubic(3.5), ["Na", "Cl"],
                                      [[0, 0, 0], [0.5, 0.5, 0.5]])
    label, idx = classify_novelty(caesium_chloride_type, [rocksalt])
    assert label == ISOMER
    assert idx == 0


def test_batch_deduplication_keeps_first_occurrence():
    from ufo_mgen.screening import deduplicate_batch

    keep = deduplicate_batch([_nacl(), _nacl(), _perovskite()])
    assert keep == [0, 2]


def test_batch_deduplication_keeps_distinct_structures():
    from ufo_mgen.screening import deduplicate_batch

    keep = deduplicate_batch([_nacl(), _perovskite()])
    assert keep == [0, 1]


# ---------------------------------------------------------------- mlff

def test_risk_score_accumulates_flags():
    from ufo_mgen.screening import risk_score

    clean = {"energy_per_atom_ev": -5.0, "max_force_ev_per_a": 0.2, "stress_l2": 1.0}
    assert risk_score(clean) == 0.0

    bad = {"energy_per_atom_ev": 0.5, "max_force_ev_per_a": 80.0, "stress_l2": 40.0}
    assert risk_score(bad) == 3.0

    assert risk_score({}) == 0.5, "a missing score is a risk, not a pass"


def test_ranking_prefers_low_risk_then_low_energy():
    from ufo_mgen.screening import rank_candidates

    records = [
        {"energy_per_atom_ev": -9.0, "risk": 2.0},   # lowest energy, risky
        {"energy_per_atom_ev": -3.0, "risk": 0.0},
        {"energy_per_atom_ev": -5.0, "risk": 0.0},
    ]
    assert rank_candidates(records) == [2, 1, 0]
    assert rank_candidates(records, keep=1) == [2]


# ---------------------------------------------------------------- relax

def test_formation_energy_needs_every_reference():
    from ufo_mgen.screening import formation_energy_per_atom

    s = _nacl()
    assert formation_energy_per_atom(s, -4.0, {"Na": -1.0}) is None
    ef = formation_energy_per_atom(s, -4.0, {"Na": -1.0, "Cl": -1.0})
    assert ef is not None and ef < 0


def test_quick_ef_prescreen_skips_when_disabled():
    from ufo_mgen.screening import check_quick_ef_prescreen

    r = check_quick_ef_prescreen(_nacl(), -4.0, {"Na": -1.0, "Cl": -1.0})
    assert r.passed and r.skipped
    assert r.reason == "disabled_by_default"


def test_quick_ef_prescreen_applies_when_explicitly_enabled():
    from ufo_mgen.screening import check_quick_ef_prescreen

    refs = {"Na": -1.0, "Cl": -1.0}
    good = check_quick_ef_prescreen(_nacl(), -4.0, refs, enabled=True)
    assert good.passed and not good.skipped

    bad = check_quick_ef_prescreen(_nacl(), 5.0, refs, enabled=True)
    assert not bad.passed
    assert bad.reason == "formation_energy_above_threshold"


# ---------------------------------------------------------------- funnel

def test_funnel_audit_records_every_stage():
    from ufo_mgen.screening import ScreeningFunnel, geometry_stage, scaffold_stage

    funnel = ScreeningFunnel().add(scaffold_stage()).add(geometry_stage())
    audit = funnel.run({"structure": _nacl(), "target_spacegroup": 225,
                        "target_scaffold": "4a;4b"}, "toy-1")
    assert audit.candidate_id == "toy-1"
    assert audit.passed
    assert audit.rejected_at is None
    assert [s.stage for s in audit.stages] == ["scaffold_correctness", "geometry_sanity"]
    assert audit.to_dict()["passed"] is True


def test_funnel_stops_at_the_first_rejection():
    from ufo_mgen.screening import ScreeningFunnel, geometry_stage, scaffold_stage

    funnel = ScreeningFunnel().add(geometry_stage()).add(scaffold_stage())
    audit = funnel.run({"structure": _overlapping()}, "toy-2")
    assert not audit.passed
    assert audit.rejected_at == "geometry_sanity"
    assert audit.reason == "atoms_too_close"
    assert len(audit.stages) == 1, "later stages must not run after a rejection"


def test_funnel_accumulates_soft_penalties_without_rejecting():
    from ufo_mgen.screening import ScreeningFunnel
    from ufo_mgen.screening.result import StageResult

    def flagging(ctx):
        return StageResult(stage="flagger", passed=True, reason="flagged",
                           soft_penalty=0.25)

    funnel = ScreeningFunnel().add(flagging).add(flagging)
    audit = funnel.run({"structure": _nacl()})
    assert audit.passed
    assert abs(audit.total_penalty - 0.5) < 1e-9


def test_funnel_survives_a_raising_stage():
    from ufo_mgen.screening import ScreeningFunnel

    def broken(ctx):
        raise RuntimeError("boom")

    audit = ScreeningFunnel().add(broken).run({"structure": _nacl()})
    assert not audit.passed
    assert "stage_error" in audit.reason


def test_default_funnel_runs_without_a_calculator():
    """The structural half must work with no potential installed."""
    from ufo_mgen.screening import default_funnel

    funnel = default_funnel()
    stages = [getattr(s, "stage_name", "?") for s in funnel.stages]
    assert "mlff_screening" not in stages
    assert "scaffold_correctness" in stages and "geometry_sanity" in stages

    audit = funnel.run({"structure": _nacl(), "target_spacegroup": 225,
                        "target_scaffold": "4a;4b"})
    assert audit.passed


def test_default_funnel_omits_prescreen_unless_requested():
    from ufo_mgen.screening import default_funnel

    funnel = default_funnel(reference=[_perovskite()])
    stages = [getattr(s, "stage_name", "?") for s in funnel.stages]
    assert "quick_formation_energy_prescreen" not in stages
    assert "novelty" in stages


def test_summarise_counts_stages():
    from ufo_mgen.screening import ScreeningFunnel, geometry_stage, summarise

    funnel = ScreeningFunnel().add(geometry_stage())
    audits = funnel.run_batch([{"structure": _nacl()}, {"structure": _overlapping()}])
    s = summarise(audits)
    assert s["n_candidates"] == 2
    assert s["n_passed"] == 1
    assert s["by_stage"]["geometry_sanity"] == {"passed": 1, "failed": 1, "skipped": 0}


# ---------------------------------------------------------------- mlip

def test_mlip_backends_are_the_three_supported_ones():
    from ufo_mgen.mlip import BACKEND_MODELS, BACKENDS

    assert BACKENDS == ("chgnet", "mace", "mattersim")
    assert set(BACKEND_MODELS) == set(BACKENDS)


def test_mlip_backend_name_validation():
    from ufo_mgen.mlip import normalise_backend

    assert normalise_backend("CHGNet") == "chgnet"
    assert normalise_backend("mace-mpa") == "mace"
    assert normalise_backend("MatterSim") == "mattersim"
    for bad in ("sevennet", "nequip", ""):
        try:
            normalise_backend(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not be accepted as a backend")


def test_mattersim_is_the_mechanical_backend():
    from ufo_mgen.mech import MECHANICAL_BACKEND
    from ufo_mgen.mlip import MECHANICAL_BACKEND as HARNESS_BACKEND

    assert MECHANICAL_BACKEND == HARNESS_BACKEND == "mattersim"


def test_mlip_protocol_constants():
    from ufo_mgen.mlip import (
        AIMD_STEPS,
        AIMD_TEMPERATURE_K,
        PHONON_MIN_SUPERCELL,
        STRICT_FMAX,
        STRICT_STEPS,
    )

    assert STRICT_FMAX == 0.005 and STRICT_STEPS == 600
    assert PHONON_MIN_SUPERCELL == 2
    assert AIMD_TEMPERATURE_K == 300.0 and AIMD_STEPS == 2000


def test_mlip_imports_are_lazy():
    """Importing the harness must not pull in a multi-gigabyte potential."""
    import ufo_mgen.mlip.harness  # noqa: F401

    for heavy in ("chgnet", "mace", "mattersim"):
        assert heavy not in sys.modules, f"{heavy} was imported eagerly"


def test_mse_gates_use_the_same_three_backends():
    from ufo_mgen.evaluation import MLIP_BACKENDS
    from ufo_mgen.mlip import normalise_backend

    assert {normalise_backend(b) for b in MLIP_BACKENDS} == {"chgnet", "mace", "mattersim"}


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
