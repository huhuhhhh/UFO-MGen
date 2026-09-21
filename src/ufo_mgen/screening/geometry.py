"""Geometry sanity checks for decoded crystal candidates."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .result import StageResult
from .thresholds import DEFAULT_THRESHOLDS, GeometryThresholds

STAGE = "geometry_sanity"


def min_pair_distance(structure) -> float:
    """Shortest interatomic distance under periodic boundary conditions."""
    if len(structure) < 2:
        return float("inf")
    d = structure.distance_matrix.copy()
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def min_pair_ratio(structure) -> float:
    """Shortest distance as a fraction of the two elements' summed radii.

    Returns ``inf`` when no radius is available for a species, so a missing
    radius never causes a rejection on its own.
    """
    if len(structure) < 2:
        return float("inf")

    radii = []
    for site in structure:
        try:
            r = site.specie.atomic_radius
        except Exception:
            r = None
        radii.append(float(r) if r else None)

    d = structure.distance_matrix
    worst = float("inf")
    n = len(structure)
    for i in range(n):
        if radii[i] is None:
            continue
        for j in range(i + 1, n):
            if radii[j] is None:
                continue
            expected = radii[i] + radii[j]
            if expected <= 0:
                continue
            worst = min(worst, float(d[i, j]) / expected)
    return worst


def volume_per_atom_scale(structure) -> float:
    """Observed volume per atom over a radius-based estimate.

    The estimate is the summed atomic sphere volume divided by a typical
    packing fraction, which is crude but sufficient to catch a cell that
    decoded an order of magnitude too large or too small.
    """
    if len(structure) == 0:
        return float("nan")

    packed = 0.0
    counted = 0
    for site in structure:
        try:
            r = site.specie.atomic_radius
        except Exception:
            r = None
        if r:
            packed += (4.0 / 3.0) * np.pi * float(r) ** 3
            counted += 1
    if counted == 0 or packed <= 0:
        return float("nan")

    # Scale the sphere volume to a plausible packed cell.
    expected_volume = packed / 0.64 * (len(structure) / counted)
    return float(structure.volume / expected_volume)


def check_geometry(
    structure,
    thresholds: Optional[GeometryThresholds] = None,
    complexity: Optional[str] = None,
) -> StageResult:
    """Run all three geometry checks and report the first hard failure."""
    t = thresholds or DEFAULT_THRESHOLDS.geometry_for(complexity)

    d_min = min_pair_distance(structure)
    ratio = min_pair_ratio(structure)
    vpa = volume_per_atom_scale(structure)
    diag = {"min_pair_distance": d_min, "min_pair_ratio": ratio,
            "vpa_scale": vpa, "n_sites": len(structure)}

    if len(structure) == 0:
        return StageResult.fail(STAGE, "empty_structure", **diag)
    if d_min < t.min_valid_dist:
        return StageResult.fail(STAGE, "atoms_too_close", **diag)
    if np.isfinite(ratio) and ratio < t.min_pair_ratio:
        return StageResult.fail(STAGE, "pair_distance_below_radii_ratio", **diag)
    if np.isfinite(vpa) and not (t.min_vpa_scale <= vpa <= t.max_vpa_scale):
        return StageResult.fail(STAGE, "implausible_cell_volume", **diag)
    return StageResult.ok(STAGE, **diag)
