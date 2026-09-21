"""Shared-model generation into target space groups outside training coverage."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from ..utils.config import load_config, repo_root
from ..utils.io import ensure_dir, write_cif
from ..utils.reproducibility import set_seed
from .sample import SamplingConfig, sample_structures

#: Context mode that drops scaffold and space-group identity.
EXTRAPOLATION_CONTEXT_MODE = "topology_chem_only"


def pyxtal_seed_scaffolds(
    spacegroup: int,
    n_orbits: int = 3,
    max_seeds: int = 8,
) -> List[str]:
    """Build Wyckoff-scaffold seeds for a space group from its public tables.

    For a group with no training structures there is nothing to copy a scaffold
    from, so valid orbit combinations are enumerated from pyxtal's Wyckoff
    tables instead. The model then supplies coordinates for whichever
    combination is handed to it.
    """
    from pyxtal.symmetry import Group

    group = Group(int(spacegroup))
    letters = [(int(wp.multiplicity), str(wp.letter).lower())
               for wp in group.Wyckoff_positions]
    letters.sort()
    if not letters:
        return []

    seeds: List[str] = []
    seen = set()

    def add(chunk):
        token = ";".join(f"{m}{l}" for m, l in chunk)
        if token not in seen:
            seen.add(token)
            seeds.append(token)

    # Windows of distinct positions, when the group has enough of them.
    for i in range(max(0, len(letters) - n_orbits + 1)):
        if len(seeds) >= max_seeds:
            return seeds
        add(letters[i: i + n_orbits])

    # Sparse groups have fewer distinct Wyckoff positions than the requested
    # orbit count. A position may legitimately host several independent orbits,
    # so pad with repeats of the lowest-multiplicity ones rather than giving up
    # -- otherwise groups like SG92 would be unreachable.
    i = 0
    while len(seeds) < max_seeds and letters:
        base = list(letters[: min(n_orbits, len(letters))])
        pad = letters[i % len(letters)]
        while len(base) < n_orbits:
            base.append(pad)
        add(sorted(base))
        i += 1
        if i > len(letters) * 2:
            break
    return seeds


def held_out_spacegroups(training_spacegroups: Iterable[int], total: int = 230) -> List[int]:
    """The space groups absent from a training set."""
    seen = {int(s) for s in training_spacegroups}
    return [sg for sg in range(1, total + 1) if sg not in seen]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UFO-MGen space-group extrapolation")
    p.add_argument("--config", default="configs/ufo_mgen_final.yaml")
    p.add_argument("--spacegroup", type=int, required=True,
                   help="target space group, typically one absent from training")
    p.add_argument("--num-samples", type=int, default=24, help="samples per seed scaffold")
    p.add_argument("--max-seeds", type=int, default=8)
    p.add_argument("--n-orbits", type=int, default=3)
    p.add_argument("--output", default="generated/extrapolation")
    p.add_argument("--weights-dir", default=None)
    p.add_argument("--union-checkpoint", default=None,
                   help="shared Stage-III checkpoint (default: from config)")
    p.add_argument("--union-package", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    set_seed(args.seed)

    extrap = cfg.section("extrapolation") if cfg.get("extrapolation") else None
    if extrap is None and cfg.get("models"):
        models = cfg.section("models")
        if models.get("extrapolation"):
            extrap = models.section("extrapolation")
    ckpt_rel = args.union_checkpoint or (extrap.get("checkpoint") if extrap else None)
    pkg_rel = args.union_package or (extrap.get("data_package") if extrap else None)
    if not ckpt_rel or not pkg_rel:
        raise SystemExit(
            "Need a shared Stage-III checkpoint and its data package: pass "
            "--union-checkpoint and --union-package, or add an 'extrapolation' "
            "section to the config."
        )

    weights_dir = Path(args.weights_dir) if args.weights_dir else cfg.path("weights_dir")
    if weights_dir is None:
        raise SystemExit(
            "No weights directory. Download the weight archive named in the "
            "config and pass --weights-dir."
        )
    ckpt = weights_dir / str(ckpt_rel)
    pkg = repo_root() / str(pkg_rel)
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint {ckpt}")

    sg = int(args.spacegroup)
    seeds = pyxtal_seed_scaffolds(sg, n_orbits=args.n_orbits, max_seeds=args.max_seeds)
    if not seeds:
        raise SystemExit(f"no Wyckoff seeds buildable for SG {sg}")

    sampling = SamplingConfig(
        n_samples=args.num_samples,
        integration_steps=int(extrap.get("integration_steps", 64) if extrap else 64),
        sample_temp=float(extrap.get("sample_temp", 0.5) if extrap else 0.5),
        project_lattice=True,
        device=args.device,
        seed=args.seed,
    )

    out = ensure_dir(Path(args.output) / f"sg{sg}")
    total = 0
    for scaffold in seeds:
        try:
            structures = sample_structures(
                ckpt, pkg, sg, scaffold, sampling,
                context_mode=EXTRAPOLATION_CONTEXT_MODE,
            )
        except Exception as exc:
            print(f"[fail] sg{sg} {scaffold}: {exc}")
            continue
        slug = scaffold.replace(";", "_")
        for i, s in enumerate(structures, start=1):
            write_cif(s, out / f"sg{sg}_{slug}_{i:04d}.cif")
            total += 1
        print(f"[ok]   sg{sg} {scaffold}: {len(structures)}")

    print(f"\nwrote {total} CIFs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
