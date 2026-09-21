"""Stage I: Hierarchical Topology Selection (HTS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pymatgen.core import Composition, Element

#: Label chain, coarsest first.
TARGET_ORDER = ["spacegroup", "k_orbits", "wyckoff_config"]

#: Which earlier labels each level conditions on.
PARENT_MAP = {
    "spacegroup": [],
    "k_orbits": ["spacegroup"],
    "wyckoff_config": ["spacegroup", "k_orbits"],
}

#: Human-readable names used in the manuscript tables.
PAPER_LABELS = {
    "spacegroup": "Space group G",
    "k_orbits": "Orbit count K | G",
    "wyckoff_config": "Wyckoff config w1:K | G,K",
}

UNK = "__UNK__"
ROUTE_NONE = "__NONE__"

#: Size buckets used by the composition featuriser.
BUCKET_MAP = {"N0-20": 0, "N21-100": 1, "N101-200": 2, "N201-400": 3}


def featurize_formula(formula: str, n_atoms: int, bucket: str) -> np.ndarray:
    if pd.isna(formula) or str(formula).strip() == "":
        formula = "H"
    comp = Composition(str(formula))
    frac = comp.fractional_composition
    vec = np.zeros(len(Element) + 5, dtype=np.float32)
    for i, el in enumerate(Element):
        amt = frac.get_el_amt_dict().get(el.symbol)
        if amt is not None:
            vec[i] = float(amt)
    vec[len(Element)] = np.log1p(float(n_atoms))
    idx = BUCKET_MAP.get(str(bucket))
    if idx is not None:
        # An unrecognised bucket leaves every size slot at zero rather than
        # indexing past the end of the vector.
        vec[len(Element) + 1 + idx] = 1.0
    return vec


@dataclass
class LabelVocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def from_train_series(cls, series: Iterable[str]) -> "LabelVocab":
        vals = [str(x) for x in series]
        uniq = [UNK] + sorted(set(vals))
        return cls(stoi={v: i for i, v in enumerate(uniq)}, itos=uniq)

    def encode(self, value: object) -> int:
        return self.stoi.get(str(value), 0)

    def size(self) -> int:
        return len(self.itos)


class CoreHierStage1(nn.Module):
    def __init__(self, input_dim: int, vocabs: Dict[str, LabelVocab], hidden_dim: int, embed_dim: int, route_vocab: LabelVocab | None = None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.embeddings = nn.ModuleDict({
            target: nn.Embedding(vocabs[target].size(), embed_dim)
            for target in TARGET_ORDER
        })
        self.heads = nn.ModuleDict()
        for target in TARGET_ORDER:
            in_dim = hidden_dim + embed_dim * len(PARENT_MAP[target])
            self.heads[target] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, vocabs[target].size()),
            )
        self.route_vocab = route_vocab
        self.route_head = None
        if route_vocab is not None:
            self.route_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, route_vocab.size()),
            )

    def _cat(self, z: torch.Tensor, parent_targets: Sequence[str], parent_ids: Sequence[torch.Tensor]) -> torch.Tensor:
        if not parent_targets:
            return z
        embs = [self.embeddings[t](ids) for t, ids in zip(parent_targets, parent_ids)]
        return torch.cat([z, *embs], dim=-1)

    def forward_train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z = self.encoder(batch["x"])
        out = {}
        for target in TARGET_ORDER:
            parents = PARENT_MAP[target]
            parent_ids = [batch[p] for p in parents]
            out[target] = self.heads[target](self._cat(z, parents, parent_ids))
        if self.route_head is not None:
            out["route_role"] = self.route_head(z)
        return out

    def forward_oracle(self, batch: Dict[str, torch.Tensor], target: str) -> torch.Tensor:
        z = self.encoder(batch["x"])
        parents = PARENT_MAP[target]
        parent_ids = [batch[p] for p in parents]
        return self.heads[target](self._cat(z, parents, parent_ids))

    def forward_greedy(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        preds = {}
        logits = {}
        for target in TARGET_ORDER:
            parents = PARENT_MAP[target]
            parent_ids = [preds[p] for p in parents]
            logit = self.heads[target](self._cat(z, parents, parent_ids))
            logits[target] = logit
            preds[target] = torch.argmax(logit, dim=-1)
        if self.route_head is not None:
            logits["route_role"] = self.route_head(z)
        return logits

