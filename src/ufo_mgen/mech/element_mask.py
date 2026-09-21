"""Element-restricted sampling utilities for Stage II COM."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set

#: Element sets used for hard-phase generation, by atomic number.
ELEMENT_SETS = {
    #: Light covalent formers -- the classic superhard corner.
    "light_covalent": [5, 6, 7],
    #: Covalent formers plus the main-group hardeners.
    "hard_formers": [4, 5, 6, 7, 8, 12, 13, 14, 15],
    #: Hard formers plus refractory transition metals.
    "refractory": [4, 5, 6, 7, 12, 13, 14, 22, 23, 24, 26,
                   41, 42, 44, 73, 74, 75, 76],
}


def parse_allowed_z(spec: str | Iterable[int]) -> Set[int]:
    """Parse ``"5,6,7"``, an iterable of atomic numbers, or a named set."""
    if isinstance(spec, str):
        name = spec.strip()
        if name in ELEMENT_SETS:
            return set(ELEMENT_SETS[name])
        return {int(x) for x in name.split(",") if x.strip()}
    return {int(x) for x in spec}


def build_logit_mask(allowed_z: Iterable[int], num_elements: int = 118, device=None):
    """An additive logit mask: ``0`` for allowed elements, ``-inf`` elsewhere.

    Logits are zero-indexed by atomic number, so element ``Z`` is column
    ``Z - 1``. Atomic numbers outside ``[1, num_elements]`` are ignored.
    """
    import torch

    mask = torch.full((int(num_elements),), float("-inf"), device=device)
    for z in allowed_z:
        z = int(z)
        if 1 <= z <= num_elements:
            mask[z - 1] = 0.0
    if bool(torch.isinf(mask).all()):
        raise ValueError("allowed element set is empty after filtering")
    return mask


def apply_element_mask(model, allowed_z: str | Iterable[int], device=None):
    """Wrap a Stage-II model's output head so it can only emit allowed elements.

    Returns the original head, so the restriction can be lifted:

        original = apply_element_mask(com, "light_covalent")
        ...
        com.out_head = original
    """
    import torch

    allowed = parse_allowed_z(allowed_z)
    num_elements = int(getattr(model, "num_elements", 118))
    mask = build_logit_mask(allowed, num_elements, device=device)
    original = model.out_head

    class _MaskedHead(torch.nn.Module):
        def __init__(self, head, m):
            super().__init__()
            self.head = head
            self.register_buffer("mask", m, persistent=False)

        def forward(self, *args, **kwargs):
            return self.head(*args, **kwargs) + self.mask

    model.out_head = _MaskedHead(original, mask)
    return original


def violations(symbols: Sequence[str], allowed_z: str | Iterable[int]) -> List[str]:
    """Species in ``symbols`` that fall outside the allowed set."""
    from pymatgen.core import Element

    allowed = parse_allowed_z(allowed_z)
    bad = []
    for s in symbols:
        try:
            if Element(str(s)).Z not in allowed:
                bad.append(str(s))
        except Exception:
            bad.append(str(s))
    return bad
