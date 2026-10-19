"""Function-preserving edits for :class:`ValmorphMLP`.

Widen follows the Net2Net split: the copied unit duplicates the incoming
weights, and the outgoing weights of the original and the copy are halved,
so the represented function is unchanged. Identity deepen inserts a linear
map ``I`` with zero bias and no activation, which is the identity function.
"""

from __future__ import annotations

import json

import torch
from torch import nn

from valmorph.core.ledger import EditProposal, Ledger, LineageEdge
from valmorph.models.mlp import ValmorphMLP


class EditError(ValueError):
    """This edit cannot be applied to the network in its current shape."""


def _linear(weight: torch.Tensor, bias: torch.Tensor) -> nn.Linear:
    layer = nn.Linear(weight.shape[1], weight.shape[0], device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    return layer


def _record(
    ledger: Ledger | None,
    *,
    step: int,
    edit_type: str,
    target: str,
    params: dict[str, object],
    edges: list[LineageEdge],
) -> None:
    if ledger is None:
        return
    proposal_id = f"{edit_type}-{step}-{target}"
    ledger.append_proposals(
        [
            EditProposal(
                proposal_id=proposal_id,
                step=step,
                controller="L0",
                edit_type=edit_type,
                target=target,
                params_json=json.dumps(params, sort_keys=True),
            )
        ]
    )
    if edges:
        ledger.append_lineage(edges)


def widen(
    net: ValmorphMLP,
    layer: int,
    unit: int,
    *,
    ledger: Ledger | None = None,
    step: int = 0,
) -> str:
    """Copy hidden unit ``unit`` in ``layer``. The function is unchanged.

    ``layer`` is an index into ``net.layers`` and must not be the output layer.
    The new unit id is returned. Incoming weights match the parent. Outgoing
    weights of the parent and the child are each half of the parent's old column.
    """

    if layer < 0 or layer >= len(net.layers) - 1:
        raise EditError("widen applies to a layer that has a following layer")
    source = net.linear(layer)
    dest = net.linear(layer + 1)
    width = int(source.out_features)
    if unit < 0 or unit >= width:
        raise EditError(f"unit {unit} is outside layer {layer} of width {width}")
    parent_id = net.unit_ids[layer][unit]
    child_id = net._alloc_unit()
    with torch.no_grad():
        incoming = source.weight[unit : unit + 1].clone()
        bias = source.bias[unit : unit + 1].clone()
        outgoing = dest.weight[:, unit : unit + 1].clone()
        new_source_weight = torch.cat([source.weight, incoming], dim=0)
        new_source_bias = torch.cat([source.bias, bias], dim=0)
        halved = dest.weight.clone()
        halved[:, unit] *= 0.5
        new_dest_weight = torch.cat([halved, outgoing * 0.5], dim=1)
        new_dest_bias = dest.bias.clone()
    net.layers[layer] = _linear(new_source_weight, new_source_bias)
    net.layers[layer + 1] = _linear(new_dest_weight, new_dest_bias)
    net.unit_ids[layer].append(child_id)
    _record(
        ledger,
        step=step,
        edit_type="widen",
        target=f"layer{layer}",
        params={"unit": unit, "parent": parent_id, "child": child_id},
        edges=[
            LineageEdge(step=step, parent_unit=parent_id, child_unit=child_id, relation="split")
        ],
    )
    return child_id


def identity_deepen(
    net: ValmorphMLP,
    after_layer: int,
    *,
    ledger: Ledger | None = None,
    step: int = 0,
) -> list[str]:
    """Insert an identity linear layer after ``after_layer``.

    The inserted map is ``y = x`` with no ReLU, so every finite input is unchanged.
    ``after_layer`` must be a layer that is not the output. New unit ids, one per
    output of ``after_layer``, are returned in order.
    """

    if after_layer < 0 or after_layer >= len(net.layers) - 1:
        raise EditError("identity deepen inserts between two existing layers")
    previous = net.linear(after_layer)
    width = int(previous.out_features)
    device = previous.weight.device
    dtype = previous.weight.dtype
    weight = torch.eye(width, device=device, dtype=dtype)
    bias = torch.zeros(width, device=device, dtype=dtype)
    child_ids = [net._alloc_unit() for _ in range(width)]
    parent_ids = list(net.unit_ids[after_layer])
    insert_at = after_layer + 1
    net.layers.insert(insert_at, _linear(weight, bias))
    net.use_relu.insert(insert_at, False)
    net.unit_ids.insert(insert_at, child_ids)
    _record(
        ledger,
        step=step,
        edit_type="identity_deepen",
        target=f"after{after_layer}",
        params={"width": width, "children": child_ids},
        edges=[
            LineageEdge(step=step, parent_unit=parent, child_unit=child, relation="identity")
            for parent, child in zip(parent_ids, child_ids, strict=True)
        ],
    )
    return child_ids


def prune(
    net: ValmorphMLP,
    layer: int,
    unit: int,
    *,
    ledger: Ledger | None = None,
    step: int = 0,
) -> str:
    """Remove hidden unit ``unit``. The represented function usually changes.

    The last remaining unit in a layer cannot be removed. The removed unit id
    is returned. This is not a function-preserving edit.
    """

    if layer < 0 or layer >= len(net.layers) - 1:
        raise EditError("prune applies to a layer that has a following layer")
    source = net.linear(layer)
    dest = net.linear(layer + 1)
    width = int(source.out_features)
    if width <= 1:
        raise EditError("refuse to prune the last unit in a layer")
    if unit < 0 or unit >= width:
        raise EditError(f"unit {unit} is outside layer {layer} of width {width}")
    removed = net.unit_ids[layer].pop(unit)
    keep = [index for index in range(width) if index != unit]
    with torch.no_grad():
        new_source_weight = source.weight[keep].clone()
        new_source_bias = source.bias[keep].clone()
        new_dest_weight = dest.weight[:, keep].clone()
        new_dest_bias = dest.bias.clone()
    net.layers[layer] = _linear(new_source_weight, new_source_bias)
    net.layers[layer + 1] = _linear(new_dest_weight, new_dest_bias)
    _record(
        ledger,
        step=step,
        edit_type="prune",
        target=f"layer{layer}",
        params={"unit": unit, "removed": removed},
        edges=[LineageEdge(step=step, parent_unit=removed, child_unit=removed, relation="prune")],
    )
    return removed
