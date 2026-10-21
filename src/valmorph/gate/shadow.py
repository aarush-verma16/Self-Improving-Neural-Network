"""Train a copy, score it on the probe, then accept or reject the edit.

``probe_x`` is ``[probe_batch, in_features]`` and ``probe_y`` is
``[probe_batch, out_features]``. The live network changes only when the
edited copy beats a matched copy that trained for the same number of steps
without the edit.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable

import torch

from valmorph.core.ledger import EditProposal, GateDecision, Ledger, LineageEdge
from valmorph.models.mlp import ValmorphMLP


def probe_mse(net: ValmorphMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean squared error of ``net`` on one batch. A scalar."""

    with torch.no_grad():
        return float(torch.mean((net(x) - y) ** 2))


def train_steps(
    net: ValmorphMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
) -> list[float]:
    """SGD for ``steps`` updates. Returns one loss scalar per step."""

    if steps < 0:
        raise ValueError("steps must be >= 0")
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.mean((net(x) - y) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def clone_net(net: ValmorphMLP) -> ValmorphMLP:
    """Deep copy, including unit ids."""

    return copy.deepcopy(net)


def adopt_shadow(live: ValmorphMLP, shadow: ValmorphMLP) -> None:
    """Make ``live`` the same network as ``shadow``, including new layers."""

    live.layers = copy.deepcopy(shadow.layers)
    live.use_relu = list(shadow.use_relu)
    live.unit_ids = copy.deepcopy(shadow.unit_ids)
    live._next_unit = shadow._next_unit


def _unit_ids(net: ValmorphMLP) -> set[str]:
    return {unit for layer in net.unit_ids for unit in layer}


def consider_edit(
    net: ValmorphMLP,
    edit: Callable[[ValmorphMLP], object],
    *,
    probe_x: torch.Tensor,
    probe_y: torch.Tensor,
    ledger: Ledger | None = None,
    step: int = 0,
    proposal_id: str,
    edit_type: str,
    target: str,
    params: dict[str, object] | None = None,
    train_x: torch.Tensor | None = None,
    train_y: torch.Tensor | None = None,
    shadow_steps: int = 0,
    lr: float = 1e-2,
    min_gain: float = 1e-4,
    score: Callable[[ValmorphMLP, torch.Tensor, torch.Tensor], float] | None = None,
) -> GateDecision:
    """Trial ``edit`` on a copy. Keep it only when probe loss falls enough.

    When ``shadow_steps`` is positive, both the edited copy and an unedited
    copy train for that many steps. ``validated_gain`` is probe loss after
    minus probe loss before, matching the ledger. A kept edit has a negative
    gain: the loss went down.
    """

    metric = probe_mse if score is None else score
    baseline = clone_net(net)
    if shadow_steps:
        if train_x is None or train_y is None:
            raise ValueError("shadow training requires train_x and train_y")
        trained_x = train_x
        trained_y = train_y
        train_steps(baseline, trained_x, trained_y, shadow_steps, lr)
    before = metric(baseline, probe_x, probe_y)
    before_ids = _unit_ids(net)
    shadow = clone_net(net)
    edit(shadow)
    if shadow_steps:
        if train_x is None or train_y is None:
            raise ValueError("shadow training requires train_x and train_y")
        train_steps(shadow, train_x, train_y, shadow_steps, lr)
    after = metric(shadow, probe_x, probe_y)
    gain = after - before
    accepted = gain < -min_gain
    reason = "probe loss fell" if accepted else "probe loss did not fall enough"
    if ledger is not None:
        ledger.append_proposals(
            [
                EditProposal(
                    proposal_id=proposal_id,
                    step=step,
                    controller="L0",
                    edit_type=edit_type,
                    target=target,
                    params_json=json.dumps(params or {}, sort_keys=True),
                )
            ]
        )
        ledger.append_decisions(
            [
                GateDecision(
                    proposal_id=proposal_id,
                    step=step,
                    accepted=accepted,
                    probe_before=before,
                    probe_after=after,
                    validated_gain=gain,
                    reason=reason,
                )
            ]
        )
    if accepted:
        added = _unit_ids(shadow) - before_ids
        removed = before_ids - _unit_ids(shadow)
        adopt_shadow(net, shadow)
        if ledger is not None and (added or removed):
            edges = [
                LineageEdge(step=step, parent_unit=unit, child_unit=unit, relation="split")
                for unit in sorted(added)
            ]
            edges.extend(
                LineageEdge(step=step, parent_unit=unit, child_unit=unit, relation="prune")
                for unit in sorted(removed)
            )
            ledger.append_lineage(edges)
    return GateDecision(
        proposal_id=proposal_id,
        step=step,
        accepted=accepted,
        probe_before=before,
        probe_after=after,
        validated_gain=gain,
        reason=reason,
    )
