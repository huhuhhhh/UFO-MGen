"""Small IO helpers: CIF writing, JSON/JSONL, archive listing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def ensure_dir(path) -> Path:
    p = Path(str(path))
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_cif(structure, path) -> Path:
    """Write a pymatgen ``Structure`` to a CIF file."""
    p = Path(str(path))
    ensure_dir(p.parent)
    p.write_text(structure.to(fmt="cif"))
    return p


def write_cifs(structures: Iterable, out_dir, *, prefix: str = "ufo_mgen", start: int = 1) -> List[Path]:
    """Write many structures as ``<prefix>_<index>.cif``."""
    out = ensure_dir(out_dir)
    written: List[Path] = []
    for i, s in enumerate(structures, start=start):
        written.append(write_cif(s, out / f"{prefix}_{i:06d}.cif"))
    return written


def read_json(path) -> Any:
    with open(str(path)) as fh:
        return json.load(fh)


def write_json(obj: Any, path, *, indent: int = 2) -> Path:
    p = Path(str(path))
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=indent, default=str))
    return p


def read_jsonl(path) -> Iterator[Dict[str, Any]]:
    with open(str(path)) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[Dict[str, Any]], path) -> Path:
    p = Path(str(path))
    ensure_dir(p.parent)
    with open(p, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    return p


def iter_archive_cifs(archive_path) -> Iterator[tuple]:
    """Yield ``(name, text)`` for each CIF in a ``.tar.zst`` or ``.tar.*`` archive.

    Requires ``zstandard`` for ``.zst`` archives.
    """
    import tarfile

    path = Path(str(archive_path))
    if path.suffixes[-1:] == [".zst"]:
        import zstandard as zstd

        with open(path, "rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as tar:
                    for member in tar:
                        if member.isfile() and member.name.endswith(".cif"):
                            fh = tar.extractfile(member)
                            if fh is not None:
                                yield member.name, fh.read().decode()
    else:
        with tarfile.open(path, "r:*") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".cif"):
                    fh = tar.extractfile(member)
                    if fh is not None:
                        yield member.name, fh.read().decode()
