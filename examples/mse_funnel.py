"""Minimal MSE gate example using synthetic records."""

from __future__ import annotations

from ufo_mgen.evaluation import (
    DEFAULT_THRESHOLDS,
    MLIP_BACKENDS,
    MLIP_MODEL_NAMES,
    MSERecord,
    consensus_pass,
    funnel_counts,
)


def main() -> int:
    t = DEFAULT_THRESHOLDS
    print("=== gates ===")
    print(f"  1 thermodynamic      {t.ef_physical_floor_ev_per_atom} < Ef/atom < "
          f"{t.ef_max_ev_per_atom} eV/atom")
    print(f"  2 lattice-dynamical  phonon band min >= {t.phonon_band_min_thz} THz")
    print(f"                       AND imaginary DOS integral < {t.imaginary_dos_integral_max}")
    print(f"  3 thermal            {t.aimd_temperature_k:.0f} K MD, |drift| < "
          f"{t.aimd_energy_drift_max_ev_per_atom} eV/atom, no overlap")
    print("\n  Gate 2 needs both conditions: a single numerically soft q-point should")
    print("  not disqualify a sound structure, while a genuinely unstable one fails")
    print("  the DOS integral.")

    print("\n=== potentials (weights are not redistributed) ===")
    for b in MLIP_BACKENDS:
        print(f"  {b:<12} {MLIP_MODEL_NAMES[b]}")

    print("\n=== worked examples ===")
    cases = {
        "sound crystal": dict(ef_per_atom_ev=-1.2, phonon_band_min_thz=-0.08,
                              imaginary_dos_integral=0.004,
                              aimd_energy_drift_ev_per_atom=0.01,
                              aimd_min_pair_distance_a=2.1, aimd_ran=True),
        "soft mode": dict(ef_per_atom_ev=-1.4, phonon_band_min_thz=-1.9,
                          imaginary_dos_integral=0.33, aimd_ran=False),
        "borderline phonon": dict(ef_per_atom_ev=-0.9, phonon_band_min_thz=-0.22,
                                  imaginary_dos_integral=0.01,
                                  aimd_energy_drift_ev_per_atom=0.02,
                                  aimd_min_pair_distance_a=1.9, aimd_ran=True),
        "melts at 300 K": dict(ef_per_atom_ev=-1.1, phonon_band_min_thz=-0.05,
                               imaginary_dos_integral=0.002,
                               aimd_energy_drift_ev_per_atom=0.31,
                               aimd_min_pair_distance_a=0.7, aimd_ran=True),
    }
    print(f"  {'case':<20}{'thermo':>8}{'phonon':>8}{'thermal':>9}{'MSE':>6}")
    for label, kw in cases.items():
        r = MSERecord(label, "chgnet", **kw)
        print(f"  {label:<20}{str(r.thermodynamic_pass()):>8}{str(r.phonon_pass()):>8}"
              f"{str(r.thermal_pass()):>9}{str(r.mse_pass()):>6}")

    print("\n=== selection across three potentials ===")
    soft = MSERecord("x", "mace-mpa", **cases["soft mode"])
    three_good = [MSERecord("x", b, **cases["sound crystal"]) for b in MLIP_BACKENDS]
    two_good = three_good[:2] + [soft]
    one_good = [three_good[0], soft,
                MSERecord("x", "mattersim", **cases["soft mode"])]
    print(f"  3 stable backends      selected = {consensus_pass(three_good)}")
    print(f"  2 stable backends      selected = {consensus_pass(two_good)}")
    print(f"  1 stable backend       selected = {consensus_pass(one_good)}")

    print("\n=== funnel counts ===")
    for backend, counts in funnel_counts(three_good + [soft]).items():
        print(f"  {backend:<12} {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
