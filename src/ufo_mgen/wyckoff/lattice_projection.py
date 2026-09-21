"""Project generated lattice parameters onto crystal-system constraints."""

from __future__ import annotations

import numpy as np


def crystal_system_from_spacegroup(spacegroup: int) -> str:
    from pymatgen.symmetry.groups import SpaceGroup

    return str(SpaceGroup.from_int_number(int(spacegroup)).crystal_system)


def project_lattice_by_spacegroup(lattice_vec, spacegroup: int) -> np.ndarray:
    """Project ``[a, b, c, alpha, beta, gamma]`` onto its crystal system."""
    x = np.asarray(lattice_vec, dtype=np.float32).copy()
    a, b, c, alpha, beta, gamma = (float(v) for v in x)
    cs = crystal_system_from_spacegroup(spacegroup)

    if cs == "cubic":
        m = float(np.mean([a, b, c]))
        return np.asarray([m, m, m, 90.0, 90.0, 90.0], dtype=np.float32)
    if cs == "tetragonal":
        m = float(np.mean([a, b]))
        return np.asarray([m, m, c, 90.0, 90.0, 90.0], dtype=np.float32)
    if cs == "orthorhombic":
        return np.asarray([a, b, c, 90.0, 90.0, 90.0], dtype=np.float32)
    if cs == "hexagonal":
        m = float(np.mean([a, b]))
        return np.asarray([m, m, c, 90.0, 90.0, 120.0], dtype=np.float32)
    if cs == "trigonal":
        m = float(np.mean([a, b, c]))
        ang = float(np.mean([alpha, beta, gamma]))
        return np.asarray([m, m, m, ang, ang, ang], dtype=np.float32)
    if cs == "monoclinic":
        return np.asarray([a, b, c, 90.0, beta, 90.0], dtype=np.float32)
    return x


def project_raw_samples_by_spacegroup(samples_raw, spacegroup: int) -> np.ndarray:
    """Project a batch of raw vectors: lattice onto the manifold, coords mod 1."""
    out = np.asarray(samples_raw, dtype=np.float32).copy()
    for i in range(out.shape[0]):
        out[i, :6] = project_lattice_by_spacegroup(out[i, :6], spacegroup)
        if out.shape[1] > 6:
            out[i, 6:] = np.mod(out[i, 6:], 1.0)
    return out
