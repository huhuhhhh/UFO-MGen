"""Multi-stage stability evaluation (MSE) gates and consensus logic."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence

#: Potentials used for the manuscript MSE funnel.
MLIP_BACKENDS = ("chgnet", "mace-mpa", "mattersim")

#: Upstream package + model identifiers. Weights are fetched by those packages.
MLIP_MODEL_NAMES = {
    "chgnet": "CHGNet (chgnet package, pretrained CHGNet-MPtrj)",
    "mace-mpa": "MACE-MPA-0 (mace-torch package, foundation model 'medium-mpa-0')",
    "mattersim": "MatterSim-v1 5M (mattersim package)",
}


@dataclass(frozen=True)
class MSEThresholds:
    """Gate thresholds. Defaults are the manuscript values."""

    ef_max_ev_per_atom: float = 0.0
    ef_physical_floor_ev_per_atom: float = -8.0
    phonon_band_min_thz: float = -0.3
    imaginary_dos_integral_max: float = 0.05
    aimd_temperature_k: float = 300.0
    aimd_energy_drift_max_ev_per_atom: float = 0.05
    aimd_min_pair_distance_a: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_THRESHOLDS = MSEThresholds()


@dataclass
class MSERecord:
    """Per-structure, per-potential MSE outcome."""

    structure_id: str
    backend: str
    ef_per_atom_ev: Optional[float] = None
    phonon_band_min_thz: Optional[float] = None
    imaginary_dos_integral: Optional[float] = None
    aimd_energy_drift_ev_per_atom: Optional[float] = None
    aimd_min_pair_distance_a: Optional[float] = None
    aimd_ran: bool = False

    def thermodynamic_pass(self, t: MSEThresholds = DEFAULT_THRESHOLDS) -> bool:
        if self.ef_per_atom_ev is None:
            return False
        return t.ef_physical_floor_ev_per_atom < self.ef_per_atom_ev < t.ef_max_ev_per_atom

    def phonon_pass(self, t: MSEThresholds = DEFAULT_THRESHOLDS) -> bool:
        if self.phonon_band_min_thz is None or self.imaginary_dos_integral is None:
            return False
        return (
            self.phonon_band_min_thz >= t.phonon_band_min_thz
            and self.imaginary_dos_integral < t.imaginary_dos_integral_max
        )

    def thermal_pass(self, t: MSEThresholds = DEFAULT_THRESHOLDS) -> bool:
        if not self.aimd_ran or self.aimd_energy_drift_ev_per_atom is None:
            return False
        if abs(self.aimd_energy_drift_ev_per_atom) >= t.aimd_energy_drift_max_ev_per_atom:
            return False
        if self.aimd_min_pair_distance_a is None:
            return True
        return self.aimd_min_pair_distance_a >= t.aimd_min_pair_distance_a

    def lattice_dynamical_pass(self, t: MSEThresholds = DEFAULT_THRESHOLDS) -> bool:
        """Gates 1 and 2 -- the manuscript's 'lattice-dynamically stable' count."""
        return self.thermodynamic_pass(t) and self.phonon_pass(t)

    def mse_pass(self, t: MSEThresholds = DEFAULT_THRESHOLDS) -> bool:
        """All three gates."""
        return self.lattice_dynamical_pass(t) and self.thermal_pass(t)


def consensus_pass(
    records: Sequence[MSERecord],
    *,
    thresholds: MSEThresholds = DEFAULT_THRESHOLDS,
    require: int = 2,
) -> bool:
    """Whether at least ``require`` potentials pass the full funnel.

    The released selection rule uses agreement from at least two MLIP backends.
    """
    return sum(1 for r in records if r.mse_pass(thresholds)) >= int(require)


def funnel_counts(
    records: Iterable[MSERecord],
    *,
    thresholds: MSEThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Dict[str, int]]:
    """Per-backend counts at each gate, in manuscript reporting order."""
    out: Dict[str, Dict[str, int]] = {}
    for r in records:
        b = out.setdefault(
            r.backend,
            {"attempted": 0, "thermodynamic": 0, "lattice_dynamical": 0, "thermal": 0},
        )
        b["attempted"] += 1
        if r.thermodynamic_pass(thresholds):
            b["thermodynamic"] += 1
        if r.lattice_dynamical_pass(thresholds):
            b["lattice_dynamical"] += 1
        if r.mse_pass(thresholds):
            b["thermal"] += 1
    return out


def records_from_dataframe(df, *, backend_col: str = "backend", id_col: str = "structure_id") -> List[MSERecord]:
    """Build :class:`MSERecord` objects from a metadata table."""
    alias = {
        "ef_per_atom_ev": ("ef_per_atom_ev", "Ef_per_atom_eV", "tight_Ef_per_atom_eV"),
        "phonon_band_min_thz": ("phonon_band_min_thz", "phonon_min_freq_THz"),
        "imaginary_dos_integral": ("imaginary_dos_integral",),
        "aimd_energy_drift_ev_per_atom": ("aimd_energy_drift_ev_per_atom", "aimd_energy_drift_eV"),
        "aimd_min_pair_distance_a": ("aimd_min_pair_distance_a", "aimd_final_min_pair_distance_A"),
    }

    def pick(row, names):
        for n in names:
            if n in row and row[n] == row[n]:  # not NaN
                return float(row[n])
        return None

    out: List[MSERecord] = []
    for _, row in df.iterrows():
        out.append(
            MSERecord(
                structure_id=str(row.get(id_col, row.get("cand_id", ""))),
                backend=str(row.get(backend_col, row.get("model", "unknown"))),
                ef_per_atom_ev=pick(row, alias["ef_per_atom_ev"]),
                phonon_band_min_thz=pick(row, alias["phonon_band_min_thz"]),
                imaginary_dos_integral=pick(row, alias["imaginary_dos_integral"]),
                aimd_energy_drift_ev_per_atom=pick(row, alias["aimd_energy_drift_ev_per_atom"]),
                aimd_min_pair_distance_a=pick(row, alias["aimd_min_pair_distance_a"]),
                aimd_ran=bool(row.get("aimd_ran", row.get("aimd300k_pass", False)) is not False),
            )
        )
    return out
