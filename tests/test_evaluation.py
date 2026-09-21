"""Tests for the benchmark evaluators."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
GEN = REPO / "generated"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _cell(a=4.0, species=("Na", "Cl")):
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), list(species), [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


# ---------------------------------------------------------------- validity

def test_structural_validity_rejects_overlapping_atoms():
    from pymatgen.core import Lattice, Structure

    from ufo_mgen.evaluation import is_structurally_valid

    assert is_structurally_valid(_cell())
    collapsed = Structure(Lattice.cubic(3.0), ["Na", "Cl"],
                          [[0, 0, 0], [0.02, 0.0, 0.0]])
    assert not is_structurally_valid(collapsed)


def test_structural_validity_threshold_is_configurable():
    from ufo_mgen.evaluation import is_structurally_valid

    s = _cell(a=4.0)
    assert is_structurally_valid(s, min_dist=0.5)
    assert not is_structurally_valid(s, min_dist=99.0)


# ---------------------------------------------------------------- MSE gates

def test_mse_thresholds_are_the_production_values():
    from ufo_mgen.evaluation import DEFAULT_THRESHOLDS as t

    assert t.ef_max_ev_per_atom == 0.0
    assert t.ef_physical_floor_ev_per_atom == -8.0
    assert t.phonon_band_min_thz == -0.3
    assert t.imaginary_dos_integral_max == 0.05
    assert t.aimd_temperature_k == 300.0
    assert t.aimd_energy_drift_max_ev_per_atom == 0.05


def test_gate_two_needs_both_conditions():
    """Band minimum alone is not enough; the DOS integral must also pass."""
    from ufo_mgen.evaluation import MSERecord

    base = dict(ef_per_atom_ev=-1.0, aimd_energy_drift_ev_per_atom=0.01,
                aimd_min_pair_distance_a=2.0, aimd_ran=True)

    both = MSERecord("a", "chgnet", phonon_band_min_thz=-0.1,
                     imaginary_dos_integral=0.01, **base)
    assert both.phonon_pass()

    bad_dos = MSERecord("b", "chgnet", phonon_band_min_thz=-0.1,
                        imaginary_dos_integral=0.40, **base)
    assert not bad_dos.phonon_pass()

    bad_band = MSERecord("c", "chgnet", phonon_band_min_thz=-2.0,
                         imaginary_dos_integral=0.01, **base)
    assert not bad_band.phonon_pass()


def test_thermodynamic_gate_has_a_physical_floor():
    """An absurdly negative formation energy signals a broken relaxation."""
    from ufo_mgen.evaluation import MSERecord

    assert MSERecord("a", "chgnet", ef_per_atom_ev=-1.0).thermodynamic_pass()
    assert not MSERecord("b", "chgnet", ef_per_atom_ev=-40.0).thermodynamic_pass()
    assert not MSERecord("c", "chgnet", ef_per_atom_ev=+0.5).thermodynamic_pass()


def test_thermal_gate_requires_aimd_to_have_run():
    from ufo_mgen.evaluation import MSERecord

    never_ran = MSERecord("a", "chgnet", ef_per_atom_ev=-1.0,
                          phonon_band_min_thz=-0.1, imaginary_dos_integral=0.01,
                          aimd_ran=False)
    assert never_ran.lattice_dynamical_pass()
    assert not never_ran.thermal_pass()
    assert not never_ran.mse_pass()


def test_gates_compose_in_order():
    from ufo_mgen.evaluation import MSERecord

    r = MSERecord("a", "chgnet", ef_per_atom_ev=-1.0, phonon_band_min_thz=-0.1,
                  imaginary_dos_integral=0.01,
                  aimd_energy_drift_ev_per_atom=0.30,
                  aimd_min_pair_distance_a=2.0, aimd_ran=True)
    assert r.thermodynamic_pass() and r.phonon_pass()
    assert r.lattice_dynamical_pass()
    assert not r.thermal_pass()
    assert not r.mse_pass(), "MSE must require all three gates"


def test_consensus_selects_when_at_least_two_potentials_pass():
    from ufo_mgen.evaluation import MLIP_BACKENDS, MSERecord, consensus_pass

    good = dict(ef_per_atom_ev=-1.0, phonon_band_min_thz=-0.1,
                imaginary_dos_integral=0.01, aimd_energy_drift_ev_per_atom=0.01,
                aimd_min_pair_distance_a=2.0, aimd_ran=True)
    passing = [MSERecord("x", b, **good) for b in MLIP_BACKENDS]
    dissent = passing[:2] + [MSERecord("x", "mattersim", ef_per_atom_ev=-1.0,
                                       phonon_band_min_thz=-3.0,
                                       imaginary_dos_integral=0.8)]
    assert consensus_pass(passing)
    assert consensus_pass(dissent)
    one_pass = [passing[0],
                MSERecord("x", "mace-mpa", ef_per_atom_ev=-1.0,
                          phonon_band_min_thz=-3.0, imaginary_dos_integral=0.8),
                MSERecord("x", "mattersim", ef_per_atom_ev=-1.0,
                          phonon_band_min_thz=-3.0, imaginary_dos_integral=0.8)]
    assert not consensus_pass(one_pass)


def test_funnel_counts_are_monotone():
    from ufo_mgen.evaluation import MLIP_BACKENDS, MSERecord, funnel_counts

    recs = [
        MSERecord("a", "chgnet", ef_per_atom_ev=-1.0, phonon_band_min_thz=-0.1,
                  imaginary_dos_integral=0.01, aimd_energy_drift_ev_per_atom=0.01,
                  aimd_min_pair_distance_a=2.0, aimd_ran=True),
        MSERecord("b", "chgnet", ef_per_atom_ev=-1.0, phonon_band_min_thz=-5.0,
                  imaginary_dos_integral=0.9),
        MSERecord("c", "chgnet", ef_per_atom_ev=+1.0),
    ]
    c = funnel_counts(recs)["chgnet"]
    assert c["attempted"] == 3
    assert c["attempted"] >= c["thermodynamic"] >= c["lattice_dynamical"] >= c["thermal"]
    assert c["thermodynamic"] == 2 and c["lattice_dynamical"] == 1 and c["thermal"] == 1


def test_mlip_names_are_documented_not_redistributed():
    from ufo_mgen.evaluation import MLIP_BACKENDS, MLIP_MODEL_NAMES

    assert set(MLIP_BACKENDS) == {"chgnet", "mace-mpa", "mattersim"}
    for b in MLIP_BACKENDS:
        assert b in MLIP_MODEL_NAMES and len(MLIP_MODEL_NAMES[b]) > 10


# ------------------------------------------- interpolation vs extrapolation

def test_split_by_training_coverage():
    from ufo_mgen.evaluation import split_by_training_coverage

    detected = [225, 225, 62, 14, None, 92]
    split = split_by_training_coverage(
        [None] * len(detected), training_spacegroups={225, 62}, detected=detected
    )
    assert split.n_interpolation_structures == 3     # 225, 225, 62
    assert split.n_extrapolation_structures == 2     # 14, 92
    assert split.extrapolation_sgs == {14, 92}
    assert split.n_valid == 5                        # the None is dropped
    assert abs(split.extrapolation_structure_rate - 2 / 5) < 1e-9


def test_spacegroup_coverage_reports_gaps():
    from ufo_mgen.evaluation import spacegroup_coverage

    cov = spacegroup_coverage([1, 2, 2, 225, None], total=230)
    assert cov["n_spacegroups_covered"] == 3
    assert cov["covered"] == [1, 2, 225]
    assert len(cov["missing"]) == 227


def test_coverage_detection_uses_a_looser_symprec():
    """Generated cells are unrelaxed, so detection is looser than encoding."""
    from ufo_mgen.evaluation.interpolation_extrapolation import COVERAGE_SYMPREC
    from ufo_mgen.wyckoff.encode import DEFAULT_SYMPREC

    assert COVERAGE_SYMPREC > DEFAULT_SYMPREC


def test_detect_spacegroup_on_a_real_cell():
    from ufo_mgen.evaluation import detect_spacegroup

    assert detect_spacegroup(_cell()) == 225


# ---------------------------------------------------------- MP20 model manifest

def test_mp20_model_manifest():
    import pandas as pd

    m = pd.read_csv(REPO / "manifests" / "MP20_models.csv")
    assert len(m) == 164
    assert (m.checkpoint_filename.str.startswith("union/")).sum() == 3
    assert (m.checkpoint_filename.str.startswith("per_sg/")).sum() == 161
    assert m.sha256.str.len().eq(64).all()
    assert "internal_source_path" not in m.columns


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
