"""Unified MLIP calculations for CHGNet, MACE, and MatterSim."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

#: The three supported backends.
BACKENDS = ("chgnet", "mace", "mattersim")

#: Upstream package and model identity for each backend. Weights are fetched by
#: those packages; nothing here redistributes them.
BACKEND_MODELS: Dict[str, str] = {
    "chgnet": "CHGNet (chgnet package, pretrained CHGNet-MPtrj)",
    "mace": "MACE-MPA-0 (mace-torch package, foundation model 'medium-mpa-0')",
    "mattersim": "MatterSim-v1 5M (mattersim package)",
}

#: The backend used for final mechanical-property evaluation in UFO-Mech.
#: K, G and E reported for a generated crystal come from here.
MECHANICAL_BACKEND = "mattersim"

#: Strict-relaxation protocol.
STRICT_FMAX = 0.005
STRICT_STEPS = 600

#: Phonon protocol.
PHONON_MIN_SUPERCELL = 2
PHONON_DISPLACEMENT = 0.03

#: Molecular-dynamics protocol.
AIMD_TEMPERATURE_K = 300.0
AIMD_STEPS = 2000
AIMD_TIMESTEP_FS = 2.0


def normalise_backend(name: str) -> str:
    """Canonical backend name, or a clear error naming the alternatives."""
    key = str(name).strip().lower().replace("_", "-")
    aliases = {"mace-mpa": "mace", "mace-mp": "mace", "matter-sim": "mattersim",
               "chg-net": "chgnet"}
    key = aliases.get(key, key)
    if key not in BACKENDS:
        raise ValueError(
            f"unknown MLIP backend {name!r}; supported backends are "
            f"{', '.join(BACKENDS)}"
        )
    return key


def load_backend(name: str, device: str = "auto"):
    """Load a calculator. The heavy import happens here, not at module import."""
    backend = normalise_backend(name)
    dev = None if device == "auto" else device

    if backend == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator() if dev is None else CHGNetCalculator(use_device=dev)

    if backend == "mace":
        from mace.calculators import mace_mp

        return mace_mp(model="medium-mpa-0", default_dtype="float64",
                       device=dev or "cpu")

    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(device=dev or "cpu")


def _atoms(structure, calculator):
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator
    return atoms


def min_interatomic_distance(structure) -> float:
    """Shortest interatomic distance, used as an overlap guard after dynamics."""
    if len(structure) < 2:
        return float("inf")
    d = np.array(structure.distance_matrix, dtype=float, copy=True)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def relax_strict(structure, calculator, *, fmax: float = STRICT_FMAX,
                 steps: int = STRICT_STEPS, fix_symmetry: bool = True) -> Dict[str, Any]:
    """Tight relaxation with symmetry held fixed.

    Symmetry is constrained because a phonon calculation on a cell that has
    silently drifted to a lower space group is answering a different question
    than the one asked.
    """
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = _atoms(structure, calculator)
    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    opt.run(fmax=fmax, steps=steps)

    return {
        "structure": AseAtomsAdaptor.get_structure(atoms),
        "energy_per_atom_ev": float(atoms.get_potential_energy()) / max(len(atoms), 1),
        "converged": bool(opt.converged()),
        "steps": int(opt.get_number_of_steps()),
        "fmax": fmax,
    }


def formation_energy_proxy(structure, calculator,
                           elemental_reference_ev: Dict[str, float]) -> Dict[str, Any]:
    """Formation energy per atom against per-element reference energies.

    The references must come from the *same* backend: mixing potentials makes
    the difference meaningless.
    """
    atoms = _atoms(structure, calculator)
    e_per_atom = float(atoms.get_potential_energy()) / max(len(atoms), 1)

    comp = structure.composition
    total = comp.num_atoms
    missing = [el for el in comp.get_el_amt_dict() if el not in elemental_reference_ev]
    if missing or total <= 0:
        return {"energy_per_atom_ev": e_per_atom,
                "formation_energy_per_atom_ev": None,
                "missing_references": missing}

    reference = sum(float(elemental_reference_ev[el]) * float(n)
                    for el, n in comp.get_el_amt_dict().items())
    return {"energy_per_atom_ev": e_per_atom,
            "formation_energy_per_atom_ev": float(e_per_atom - reference / total),
            "missing_references": []}


def _supercell_matrix(structure, min_reps: int = PHONON_MIN_SUPERCELL,
                      max_atoms: Optional[int] = None) -> List[int]:
    reps = [min_reps, min_reps, min_reps]
    if max_atoms:
        while np.prod(reps) * len(structure) > max_atoms and max(reps) > 1:
            reps[int(np.argmax(reps))] -= 1
    return reps


def phonon_rigorous(structure, calculator, *, min_reps: int = PHONON_MIN_SUPERCELL,
                    displacement: float = PHONON_DISPLACEMENT,
                    symmetrize: bool = True) -> Dict[str, Any]:
    """Phonons on at least a 2x2x2 supercell with symmetrised force constants."""
    import phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    cell = PhonopyAtoms(symbols=[str(s.specie) for s in structure],
                        scaled_positions=structure.frac_coords,
                        cell=structure.lattice.matrix)
    reps = _supercell_matrix(structure, min_reps)
    ph = phonopy.Phonopy(cell, supercell_matrix=np.diag(reps))
    ph.generate_displacements(distance=displacement)

    from pymatgen.core import Structure as PmgStructure

    forces = []
    for sc in ph.supercells_with_displacements:
        s = PmgStructure(sc.cell, sc.symbols, sc.scaled_positions)
        atoms = _atoms(s, calculator)
        forces.append(np.asarray(atoms.get_forces(), dtype=float))
    ph.forces = np.asarray(forces)
    ph.produce_force_constants()
    if symmetrize:
        ph.symmetrize_force_constants()

    ph.run_mesh([8, 8, 8])
    mesh = ph.get_mesh_dict()
    freqs = np.asarray(mesh["frequencies"], dtype=float)
    return {
        "supercell": reps,
        "min_frequency_thz": float(freqs.min()),
        "imaginary_fraction": float((freqs < 0).mean()),
        "symmetrized": symmetrize,
    }


def phonon_proxy(structure, calculator, *, max_atoms: int = 120,
                 displacement: float = PHONON_DISPLACEMENT) -> Dict[str, Any]:
    """Cheaper phonon estimate with a supercell capped by atom count.

    Use it for triage only. A capped supercell can produce soft modes that are
    artefacts of the cell size, which is exactly what ``phonon_rigorous``
    exists to avoid.
    """
    reps = _supercell_matrix(structure, PHONON_MIN_SUPERCELL, max_atoms=max_atoms)
    result = phonon_rigorous(structure, calculator, min_reps=min(reps),
                             displacement=displacement, symmetrize=True)
    result["proxy"] = True
    result["supercell"] = reps
    return result


def aimd_proxy(structure, calculator, *, temperature_k: float = AIMD_TEMPERATURE_K,
               steps: int = AIMD_STEPS, timestep_fs: float = AIMD_TIMESTEP_FS,
               min_pair_distance: float = 0.5) -> Dict[str, Any]:
    """Finite-temperature stability check by molecular dynamics.

    Reports the energy drift and whether any atoms overlapped. A structure that
    melts or collapses at 300 K is not a crystal, whatever its phonons say.
    """
    from ase import units
    from ase.md.langevin import Langevin
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = _atoms(structure, calculator)
    n = max(len(atoms), 1)
    e_start = float(atoms.get_potential_energy()) / n

    dyn = Langevin(atoms, timestep_fs * units.fs, temperature_K=temperature_k,
                   friction=0.01, logfile=None)
    dyn.run(steps)

    e_end = float(atoms.get_potential_energy()) / n
    final = AseAtomsAdaptor.get_structure(atoms)
    d_min = min_interatomic_distance(final)
    return {
        "temperature_k": temperature_k,
        "steps": steps,
        "energy_drift_ev_per_atom": float(e_end - e_start),
        "min_pair_distance_a": d_min,
        "overlapped": bool(d_min < min_pair_distance),
        "final_structure": final,
    }


def elastic_properties(structure, calculator, *,
                       deltas: Sequence[float] = (-0.01, -0.005, 0.005, 0.01),
                       relax_first: bool = True) -> Dict[str, Any]:
    """Elastic tensor and the Voigt-Reuss-Hill moduli K, G and E.

    Six Voigt strain modes at several amplitudes, fitted for the stiffness
    tensor. **This is the backend behind every reported K/G/E value in
    UFO-Mech** -- see :data:`MECHANICAL_BACKEND`.
    """
    from pymatgen.analysis.elasticity import DeformedStructureSet, ElasticTensor, Strain, Stress
    from pymatgen.io.ase import AseAtomsAdaptor

    work = structure
    if relax_first:
        work = relax_strict(structure, calculator)["structure"]

    deformed = DeformedStructureSet(work, symmetry=False,
                                    norm_strains=list(deltas), shear_strains=list(deltas))
    strains, stresses = [], []
    for ds, deformation in zip(deformed, deformed.deformations):
        atoms = _atoms(ds, calculator)
        raw = np.asarray(atoms.get_stress(voigt=False), dtype=float)
        strains.append(Strain.from_deformation(deformation))
        stresses.append(Stress(raw * -1.0 / units_GPa()))

    tensor = ElasticTensor.from_independent_strains(strains, stresses, eq_stress=None)
    k = float(tensor.k_vrh)
    g = float(tensor.g_vrh)
    e = float(9 * k * g / (3 * k + g)) if (3 * k + g) else float("nan")
    nu = float((3 * k - 2 * g) / (2 * (3 * k + g))) if (3 * k + g) else float("nan")
    return {
        "K": k, "G": g, "E": e, "poisson_ratio": nu,
        "born_stable": bool(np.all(np.linalg.eigvals(tensor.voigt) > 0)),
        "backend_note": f"computed with the loaded calculator; "
                        f"UFO-Mech reports {MECHANICAL_BACKEND}",
    }


def units_GPa() -> float:
    """Conversion from eV/A^3 to GPa."""
    from ase import units

    return float(units.GPa)
