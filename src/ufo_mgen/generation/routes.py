"""Production route-manifest loading and lookup utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

from ..utils.config import repo_root

DEFAULT_ROUTE_CSV = "data/route_inventory_124.csv"


@dataclass
class Route:
    """One production generation route."""

    route_id: str
    route_key: str
    spacegroup: int
    scaffold: str
    route_train_count: Optional[int] = None
    data_package: Optional[str] = None
    checkpoint: Optional[str] = None

    @property
    def scaffold_id(self) -> str:
        return f"SG{self.spacegroup}|{self.scaffold}"


class RouteManifest:
    """The 124-route production manifest."""

    def __init__(self, routes: List[Route]):
        self.routes = list(routes)

    def __len__(self) -> int:
        return len(self.routes)

    def __iter__(self) -> Iterator[Route]:
        return iter(self.routes)

    @classmethod
    def load(cls, path=None) -> "RouteManifest":
        p = Path(str(path)) if path else (repo_root() / DEFAULT_ROUTE_CSV)
        df = pd.read_csv(p)
        routes: List[Route] = []
        for _, r in df.iterrows():
            routes.append(
                Route(
                    route_id=str(r.get("route_id")),
                    route_key=str(r.get("route_key")),
                    spacegroup=int(r.get("spacegroup")),
                    scaffold=str(r.get("scaffold")),
                    route_train_count=(
                        int(r["route_train_count"])
                        if pd.notna(r.get("route_train_count")) else None
                    ),
                    data_package=(str(r["data_package"]) if pd.notna(r.get("data_package")) else None),
                    checkpoint=(str(r["checkpoint"]) if pd.notna(r.get("checkpoint")) else None),
                )
            )
        return cls(routes)

    def by_spacegroup(self, spacegroup: int) -> List[Route]:
        return [r for r in self.routes if r.spacegroup == int(spacegroup)]

    def get(self, route_id: str) -> Optional[Route]:
        """Return one manifest row by its stable public route ID."""
        for r in self.routes:
            if r.route_id == route_id:
                return r
        return None

    def by_route_key(self, route_key: str) -> List[Route]:
        """Return all production rows carrying one route key."""
        return [r for r in self.routes if r.route_key == route_key]

    def unique_scaffolds(self) -> List[str]:
        return sorted({r.scaffold_id for r in self.routes})

    def summary(self) -> Dict[str, object]:
        return {
            "n_routes": len(self.routes),
            "n_unique_scaffolds": len(self.unique_scaffolds()),
            "n_spacegroups": len({r.spacegroup for r in self.routes}),
            "n_unique_route_ids": len({r.route_id for r in self.routes}),
            "n_unique_route_keys": len({r.route_key for r in self.routes}),
            "total_route_train_count": sum(
                r.route_train_count or 0 for r in self.routes
            ),
        }
