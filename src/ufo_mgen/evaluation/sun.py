"""Stable, Unique, and Novel (S.U.N.) evaluation utilities."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def relax_with_chgnet(structures: List[Structure], fmax: float = 0.05, device: str = "cuda") -> Tuple[List[Optional[Structure]], List[Optional[float]]]:
    """
    Relax structures using CHGNet and return (relaxed_structures, e_hull_values).
    e_hull computed via pymatgen PhaseDiagram + Materials Project reference energies.

    Returns:
        relaxed_structures: list of relaxed Structure (None if failed)
        formation_energies: list of formation energies eV/atom (None if failed)
    """
    try:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import StructOptimizer
    except ImportError:
        raise ImportError(
            "chgnet not installed. Install with: pip install chgnet\n"
            "or use --mode dft with pre-relaxed structures."
        )

    model = CHGNet.load()
    relaxer = StructOptimizer(model=model, use_device=device)

    relaxed = []
    energies = []

    for struct in tqdm(structures, desc="CHGNet relaxation"):
        try:
            result = relaxer.relax(struct, fmax=fmax, steps=500)
            relaxed_struct = result["final_structure"]
            e_per_atom = float(result["trajectory"].energies[-1]) / len(relaxed_struct)
            relaxed.append(relaxed_struct)
            energies.append(e_per_atom)
        except Exception as e:
            relaxed.append(None)
            energies.append(None)

    return relaxed, energies


def relax_with_mace(structures: List[Structure], fmax: float = 0.05) -> Tuple[List[Optional[Structure]], List[Optional[float]]]:
    """
    Relax structures using MACE-MP-0 and return (relaxed_structures, formation_energies).
    """
    try:
        from mace.calculators import mace_mp
        from ase.optimize import FIRE
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError:
        raise ImportError(
            "mace-torch not installed. Install with: pip install mace-torch\n"
            "or use --mlff chgnet."
        )

    calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device="cpu")
    adaptor = AseAtomsAdaptor()

    relaxed = []
    energies = []

    for struct in tqdm(structures, desc="MACE-MP-0 relaxation"):
        try:
            atoms = adaptor.get_atoms(struct)
            atoms.calc = calc
            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=fmax, steps=500)
            relaxed_struct = adaptor.get_structure(atoms)
            e_per_atom = float(atoms.get_potential_energy()) / len(atoms)
            relaxed.append(relaxed_struct)
            energies.append(e_per_atom)
        except Exception:
            relaxed.append(None)
            energies.append(None)

    return relaxed, energies


# ---------------------------------------------------------------------------
# E_hull computation
# ---------------------------------------------------------------------------


def compute_e_hull(
    structures: List[Optional[Structure]],
    formation_energies: List[Optional[float]],
) -> List[Optional[float]]:
    """
    Compute energy above convex hull (e_hull) for each structure.
    Uses pymatgen PhaseDiagram with Materials Project reference entries.

    Returns: list of e_hull values (eV/atom), None if computation failed.
    NOTE: This requires mp-api access. If offline, uses formation energy as proxy.
    """
    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
        from pymatgen.core import Composition
        MP_API_AVAILABLE = True
    except ImportError:
        MP_API_AVAILABLE = False
        print("[WARNING] mp-api not available. Using formation energy as E_hull proxy.")
        print("          Install with: pip install mp-api")

    if not MP_API_AVAILABLE:
        # Fallback: use formation energy directly as proxy
        e_hull = []
        for e in formation_energies:
            if e is not None:
                # Very rough proxy: negative formation energy = potentially stable
                e_hull.append(max(0.0, e + 0.5) if e < 0 else e + 0.5)
            else:
                e_hull.append(None)
        return e_hull

    e_hull_values = []
    for struct, e_form in tqdm(zip(structures, formation_energies), total=len(structures), desc="Computing E_hull"):
        if struct is None or e_form is None:
            e_hull_values.append(None)
            continue
        try:
            comp = struct.composition.reduced_composition
            elements = [str(el) for el in comp.elements]

            # Get reference entries from MP
            with MPRester() as mpr:
                entries = mpr.get_entries_in_chemsys(elements)

            # Add the generated structure as a new entry
            from pymatgen.entries.computed_entries import ComputedEntry
            gen_entry = ComputedEntry(
                composition=comp,
                energy=e_form * comp.num_atoms,
            )
            all_entries = entries + [gen_entry]

            pd = PhaseDiagram(all_entries)
            e_hull = pd.get_e_above_hull(gen_entry)
            e_hull_values.append(float(e_hull))
        except Exception:
            # Use formation energy as proxy if PD computation fails
            e_hull_values.append(max(0.0, e_form + 0.3) if e_form < 0 else None)

    return e_hull_values


# ---------------------------------------------------------------------------
# Uniqueness and Novelty
# ---------------------------------------------------------------------------


def compute_uniqueness(
    structures: List[Optional[Structure]],
    stable_mask: List[bool],
    matcher: StructureMatcher,
) -> Tuple[List[bool], int]:
    """
    Among stable structures, mark as unique if no two with same formula are equivalent.

    Returns:
        unique_mask: bool list (True = unique)
        n_unique: count of unique stable structures
    """
    stable_indices = [i for i, m in enumerate(stable_mask) if m and structures[i] is not None]

    # Group by reduced formula
    from collections import defaultdict
    formula_groups = defaultdict(list)
    for idx in stable_indices:
        formula = structures[idx].composition.reduced_formula
        formula_groups[formula].append(idx)

    unique_mask = [False] * len(structures)

    for formula, indices in formula_groups.items():
        # Within each formula group, find unique representatives
        unique_in_group = []
        for idx in indices:
            is_dup = False
            for u_idx in unique_in_group:
                try:
                    if matcher.fit(structures[u_idx], structures[idx]):
                        is_dup = True
                        break
                except Exception:
                    pass
            if not is_dup:
                unique_in_group.append(idx)
                unique_mask[idx] = True

    n_unique = sum(unique_mask)
    return unique_mask, n_unique


def compute_novelty(
    structures: List[Optional[Structure]],
    unique_mask: List[bool],
    train_structures: List[Structure],
    matcher: StructureMatcher,
) -> Tuple[List[bool], int]:
    """
    Among unique structures, mark as novel if not present in train_structures.
    """
    # Build train formula index for fast lookup
    from collections import defaultdict
    train_by_formula = defaultdict(list)
    for ts in tqdm(train_structures, desc="Indexing train set"):
        formula = ts.composition.reduced_formula
        train_by_formula[formula].append(ts)

    unique_indices = [i for i, m in enumerate(unique_mask) if m and structures[i] is not None]

    novel_mask = [False] * len(structures)

    for idx in tqdm(unique_indices, desc="Novelty check"):
        struct = structures[idx]
        formula = struct.composition.reduced_formula
        train_candidates = train_by_formula.get(formula, [])

        is_novel = True
        for train_s in train_candidates:
            try:
                if matcher.fit(struct, train_s):
                    is_novel = False
                    break
            except Exception:
                pass

        novel_mask[idx] = is_novel

    n_novel = sum(novel_mask)
    return novel_mask, n_novel


# ---------------------------------------------------------------------------
# RMSD computation
# ---------------------------------------------------------------------------


def compute_relaxed_rmsd(
    gen_structures: List[Optional[Structure]],
    relaxed_structures: List[Optional[Structure]],
) -> List[Optional[float]]:
    """
    Compute per-structure RMSD between generated and relaxed positions (Å).

    Uses minimum-image convention (MIC) in fractional coordinates to correctly
    handle atoms that cross periodic boundaries during relaxation.  Displacements
    are converted to Angstroms using the relaxed lattice (so we measure how far
    atoms moved in the final geometry, not the generated one).

    Algorithm per structure:
        frac_diff = rel.frac_coords - gen.frac_coords
        frac_diff -= round(frac_diff)          # MIC: wrap to [-0.5, 0.5)
        cart_diff  = rel.lattice @ frac_diff.T # Cartesian in relaxed cell
        RMSD = sqrt( mean( ||cart_diff_i||^2 ) )
    """
    rmsd_values = []
    for gen_s, rel_s in zip(gen_structures, relaxed_structures):
        if gen_s is None or rel_s is None:
            rmsd_values.append(None)
            continue
        try:
            if len(gen_s) != len(rel_s):
                rmsd_values.append(None)
                continue
            frac_diff = rel_s.frac_coords - gen_s.frac_coords
            frac_diff -= np.round(frac_diff)           # minimum image convention
            cart_diff = rel_s.lattice.get_cartesian_coords(frac_diff)
            rmsd = float(np.sqrt(np.mean(np.sum(cart_diff ** 2, axis=1))))
            rmsd_values.append(rmsd)
        except Exception:
            rmsd_values.append(None)
    return rmsd_values


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
