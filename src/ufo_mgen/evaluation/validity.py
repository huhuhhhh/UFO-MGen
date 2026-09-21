"""Structural and compositional validity checks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Composition, Structure

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import smact
    from smact.screening import smact_filter
    SMACT_AVAILABLE = True
except ImportError:
    smact = None
    smact_filter = None
    SMACT_AVAILABLE = False


def is_structurally_valid(structure: Structure, min_dist: float = 0.5) -> bool:
    """Return True if all pairwise atomic distances (MIC) > min_dist Angstrom."""
    try:
        # get_all_neighbors returns neighbors within cutoff radius
        # If any neighbor found within min_dist, structure is invalid
        neighbors = structure.get_all_neighbors(r=min_dist, include_index=True)
        for site_neighbors in neighbors:
            if len(site_neighbors) > 0:
                return False
        return True
    except Exception:
        return False


def compute_structural_validity(
    structures: List[Structure],
    min_dist: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute structural validity for a list of pymatgen Structure objects.

    Returns dict with:
        valid_count, total_count, validity_rate, invalid_indices
    """
    valid_count = 0
    invalid_indices = []

    iterator = tqdm(enumerate(structures), total=len(structures), desc="Structural validity") if verbose else enumerate(structures)

    for idx, struct in iterator:
        if is_structurally_valid(struct, min_dist=min_dist):
            valid_count += 1
        else:
            invalid_indices.append(idx)

    total = len(structures)
    return {
        "metric": "structural_validity",
        "min_dist_threshold_angstrom": min_dist,
        "valid_count": valid_count,
        "total_count": total,
        "validity_rate": valid_count / total if total > 0 else 0.0,
        "validity_pct": 100.0 * valid_count / total if total > 0 else 0.0,
        "invalid_indices": invalid_indices,
    }


# ---------------------------------------------------------------------------
# Compositional validity (SMACT)
# ---------------------------------------------------------------------------


def smact_validity_check(comp: "pymatgen.core.Composition") -> bool:
    """
    Run SMACT check on a pymatgen Composition.

    A composition passes if there exists at least one valid assignment of
    oxidation states satisfying:
      1. Electrical neutrality: sum(n_i * v_i) == 0
      2. Electronegativity ordering: more electronegative species are anions
    """
    if not SMACT_AVAILABLE:
        return True  # skip if smact not installed

    # Common relaxation for intermetallic / alloy compositions:
    # if all species are metals, treat the composition as valid instead of
    # forcing an ionic oxidation-state assignment that SMACT may reject.
    elems = list(comp.elements)
    if elems and all(getattr(el, "is_metal", False) for el in elems):
        return True

    elem_symbols = [str(el) for el in comp.elements]
    stoich = [int(comp[el]) for el in comp.elements]

    # Normalize stoichiometry by GCD
    from math import gcd
    from functools import reduce
    g = reduce(gcd, stoich)
    stoich = [s // g for s in stoich]

    try:
        smact_elems = [smact.Element(sym) for sym in elem_symbols]
    except Exception:
        # Element not in SMACT database
        return False

    # Build list of (element, stoichiometry) tuples
    # smact_filter checks all combinations of oxidation states
    # SMACT v4 requires stoichs as list[list[int]], one list per element
    stoichs_v4 = [[s] for s in stoich]
    result = smact_filter(
        els=smact_elems,
        stoichs=stoichs_v4,
        threshold=8,  # max oxidation state magnitude to consider
    )

    return len(result) > 0


def compute_compositional_validity(
    structures: List[Structure],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute SMACT-based compositional validity.

    Returns dict with:
        valid_count, total_count, validity_rate, smact_available
    """
    if not SMACT_AVAILABLE:
        return {
            "metric": "compositional_validity",
            "smact_available": False,
            "note": "smact not installed; skipped",
            "valid_count": None,
            "total_count": len(structures),
            "validity_rate": None,
        }

    valid_count = 0
    invalid_indices = []

    iterator = tqdm(enumerate(structures), total=len(structures), desc="Compositional validity (SMACT)") if verbose else enumerate(structures)

    for idx, struct in iterator:
        try:
            comp = struct.composition
            if smact_validity_check(comp):
                valid_count += 1
            else:
                invalid_indices.append(idx)
        except Exception:
            invalid_indices.append(idx)

    total = len(structures)
    return {
        "metric": "compositional_validity",
        "smact_available": True,
        "valid_count": valid_count,
        "total_count": total,
        "validity_rate": valid_count / total if total > 0 else 0.0,
        "validity_pct": 100.0 * valid_count / total if total > 0 else 0.0,
        "invalid_indices": invalid_indices,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
