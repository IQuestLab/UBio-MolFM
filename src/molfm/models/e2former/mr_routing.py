"""Canonical mixed-resolution light-atom routing.

Priority order:

    explicit ``batched_data["is_light"]``
        > ``atomic_numbers in lf_atoms`` OR ``hoh_o_mask``
        > ``atomic_numbers in lf_atoms``

The explicit mask wins because it can carry information that is NOT a function
of the atom's own species -- most importantly the HOH-O mask, where a water
oxygen's routing depends on its neighbours and therefore cannot be recovered
from its atomic number alone. Inference is single-GPU, so the mask is simply
carried on the input.

When ``verify=True`` (or ``MOLFM_MR_ROUTING_CHECK=1``) and both an explicit mask
and the ingredients for a recomputation are present, the two are asserted equal.
That turns a silent interface/model disagreement into a loud failure, at the
cost of one comparison -- so it is off by default on the hot path.
"""

from __future__ import annotations

import os

import torch


_TRUE_ENV = ("1", "true", "TRUE", "yes", "on")


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "") in _TRUE_ENV


def compute_mr_is_light(
    atomic_numbers: torch.Tensor,
    lf_atoms_t: torch.Tensor,
    hoh_o_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recompute the light mask from species alone (plus the optional HOH-O mask).

    ``atomic_numbers`` keeps whatever shape the caller passes; the result has
    that same shape.
    """
    is_light = torch.isin(atomic_numbers, lf_atoms_t)
    if hoh_o_mask is not None:
        is_light = is_light | hoh_o_mask.to(device=is_light.device).bool()
    return is_light


def resolve_mr_is_light(
    batched_data,
    lf_atoms_t: torch.Tensor,
    *,
    verify: bool | None = None,
    where: str = "",
) -> torch.Tensor:
    """Return the canonical per-atom light mask for ``batched_data``.

    ``batched_data`` may be a plain dict or a ``torch_geometric.data.Data``;
    both support ``.get``. Only the keys ``atomic_numbers``, ``is_light`` and
    ``hoh_o_mask`` are read.

    Args:
        batched_data: model input container.
        lf_atoms_t: 1-D long tensor of atomic numbers routed to the light stream.
        verify: assert the explicit mask equals a fresh recomputation. ``None``
            (default) reads ``MOLFM_MR_ROUTING_CHECK`` from the environment.
        where: short caller tag used in the assertion message.
    """
    if verify is None:
        verify = _env_enabled("MOLFM_MR_ROUTING_CHECK")

    explicit = batched_data.get("is_light", None)
    hoh_o_mask = batched_data.get("hoh_o_mask", None)
    atomic_numbers = batched_data["atomic_numbers"]

    if explicit is None:
        return compute_mr_is_light(atomic_numbers, lf_atoms_t, hoh_o_mask)

    explicit = explicit.bool()
    if verify and hoh_o_mask is not None:
        recomputed = compute_mr_is_light(atomic_numbers, lf_atoms_t, hoh_o_mask)
        if explicit.shape != recomputed.shape or not bool(
            torch.equal(explicit.reshape(-1), recomputed.reshape(-1))
        ):
            n_diff = int(
                (explicit.reshape(-1) != recomputed.reshape(-1)).sum().item()
            )
            raise AssertionError(
                f"[MR routing{(' @ ' + where) if where else ''}] the explicit "
                f"'is_light' mask disagrees with the atomic-number + hoh_o_mask "
                f"recomputation on {n_diff} atom(s). The interface and the model "
                f"are routing atoms differently, which silently changes the energy "
                f"surface. Fix the producer of 'is_light' or unset "
                f"MOLFM_MR_ROUTING_CHECK."
            )
    return explicit
