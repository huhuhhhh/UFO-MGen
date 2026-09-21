"""Configuration loading.

Configs are plain YAML. Paths inside a config are resolved relative to the
repository root (the directory containing ``configs/``), never against an
absolute location, so a fresh checkout works unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

_PATH_KEYS = (
    "data_root", "dataset", "checkpoint", "ckpt", "manifest", "output",
    "out_dir", "scaffolds", "routes", "weights_dir", "package",
)


def repo_root() -> Path:
    """Repository root, overridable with ``UFO_MGEN_ROOT``."""
    env = os.environ.get("UFO_MGEN_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def resolve_path(value, *, root: Optional[Path] = None) -> Path:
    """Resolve a config path against the repository root."""
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    return ((root or repo_root()) / p).resolve()


@dataclass
class Config:
    """A loaded config plus the root its relative paths resolve against."""

    data: Dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def path(self, key: str, default: Any = None) -> Optional[Path]:
        value = self.data.get(key, default)
        return None if value is None else resolve_path(value, root=self.root)

    def section(self, key: str) -> "Config":
        value = self.data.get(key) or {}
        if not isinstance(value, dict):
            raise TypeError(f"config section {key!r} is not a mapping")
        return Config(data=value, root=self.root)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)


def load_config(path, *, root: Optional[Path] = None) -> Config:
    """Load a YAML config."""
    import yaml

    p = Path(str(path))
    if not p.is_absolute():
        p = resolve_path(p, root=root)
    with open(p) as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data=data, root=root or repo_root())
