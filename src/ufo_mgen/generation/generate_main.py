"""Route-manifest generation entry point for the main UFO-MGen pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from ..utils.config import load_config, repo_root
from ..utils.io import ensure_dir, write_cif
from ..utils.reproducibility import set_seed
from .routes import RouteManifest
from .sample import SamplingConfig, sample_structures


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UFO-MGen main generation (route manifest path)")
    p.add_argument("--config", default="configs/ufo_mgen_final.yaml")
    p.add_argument("--num-samples", type=int, default=None,
                   help="samples per route (default: generation.candidates_per_route from config)")
    p.add_argument("--output", default="generated/ufo_mgen")
    p.add_argument("--routes", default=None, help="route manifest CSV (default: from config)")
    p.add_argument("--route-id", default=None, help="generate one manifest row, e.g. R001")
    p.add_argument("--route-key", default=None, help="generate all rows carrying one route key")
    p.add_argument("--weights-dir", default=None, help="directory holding Stage-III checkpoints")
    p.add_argument("--integration-steps", type=int, default=None)
    p.add_argument("--sample-temp", type=float, default=None)
    p.add_argument("--seed", type=int, default=44, help="production seed (paper: 44)")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    set_seed(args.seed)

    manifest = RouteManifest.load(args.routes or cfg.get("routes"))
    routes = list(manifest)
    if args.route_id:
        one = manifest.get(args.route_id)
        routes = [one] if one else []
    if args.route_key:
        routes = manifest.by_route_key(args.route_key)
    if not routes:
        raise SystemExit("no routes selected")

    gen = cfg.section("generation") if cfg.get("generation") else None
    configured_n = gen.get("candidates_per_route") if gen else None
    n_samples = args.num_samples if args.num_samples is not None else configured_n
    if n_samples is None:
        raise SystemExit(
            "Set --num-samples or generation.candidates_per_route in the config."
        )

    sampling = SamplingConfig(
        n_samples=int(n_samples),
        integration_steps=int(
            args.integration_steps
            or (gen.get("integration_steps", 48) if gen else 48)
        ),
        sample_temp=float(
            args.sample_temp or (gen.get("sample_temp", 0.5) if gen else 0.5)
        ),
        device=args.device,
        seed=args.seed,
    )

    weights_dir = Path(args.weights_dir) if args.weights_dir else cfg.path("weights_dir")
    if weights_dir is None:
        raise SystemExit(
            "No Stage-III weights directory. Download the weight archive from the "
            "Zenodo record in README.md and pass --weights-dir."
        )

    out_root = ensure_dir(args.output)
    written = 0
    for route in routes:
        if not route.checkpoint or not route.data_package:
            print(f"[skip] {route.route_key}: manifest has no checkpoint/data package")
            continue
        ckpt = weights_dir / route.checkpoint
        pkg = repo_root() / route.data_package
        if not ckpt.exists():
            print(f"[skip] {route.route_key}: missing checkpoint {ckpt}")
            continue
        try:
            structures = sample_structures(
                ckpt, pkg, route.spacegroup, route.scaffold, sampling
            )
        except Exception as exc:
            print(f"[fail] {route.route_key}: {exc}")
            continue
        route_dir = ensure_dir(out_root / route.route_id / route.route_key)
        for i, s in enumerate(structures, start=1):
            write_cif(s, route_dir / f"{route.route_key}_{i:04d}.cif")
            written += 1
        print(f"[ok]   {route.route_key}: {len(structures)} structures")

    print(f"\nwrote {written} CIFs to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
