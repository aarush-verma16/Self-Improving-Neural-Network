"""A first controller: propose pruning a unit that is silent on the current batch.

The gate still decides. A silent unit often does not lower probe loss when it
is removed, so the proposal can be refused. That outcome is recorded.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from valmorph.benchmarks.classify import batch_accuracy
from valmorph.core.ledger import Ledger, LineageEdge
from valmorph.gate import consider_edit, train_cross_entropy_steps
from valmorph.introspect import hidden_activations, most_silent_unit
from valmorph.models import ValmorphMLP, prune, widen


def train_with_silent_prune(
    net: ValmorphMLP,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    probe_images: torch.Tensor,
    probe_labels: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    ledger: Ledger,
    every: int = 10,
) -> list[float]:
    """Train and, every ``every`` steps, propose pruning a silent hidden unit.

    Returns probe accuracy after each step. ``images`` is ``[n, features]``.
    """

    accuracies: list[float] = []
    count = int(images.shape[0])
    for step in range(steps):
        start = (step * batch_size) % count
        index = torch.arange(start, start + batch_size) % count
        batch_x = images[index]
        batch_y = labels[index]
        if every > 0 and step % every == 0 and int(net.linear(0).out_features) > 1:
            activations = hidden_activations(net, batch_x)[0]
            unit = most_silent_unit(activations)
            if unit is not None:
                consider_edit(
                    net,
                    lambda shadow, chosen=unit: prune(shadow, 0, chosen),
                    probe_x=probe_images,
                    probe_y=probe_labels,
                    score=_cross_entropy,
                    ledger=ledger,
                    step=step,
                    proposal_id=f"prune-silent-{step}",
                    edit_type="prune",
                    target="layer0",
                    params={"unit": unit},
                )
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        optimizer.zero_grad()
        loss = F.cross_entropy(net(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            accuracies.append(batch_accuracy(net(probe_images), probe_labels))
    return accuracies


def train_with_widen_trials(
    net: ValmorphMLP,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    probe_images: torch.Tensor,
    probe_labels: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    ledger: Ledger,
    every: int = 20,
    shadow_steps: int = 8,
) -> list[float]:
    """Train and periodically propose a function-preserving widen.

    The gate trains an edited copy and an unedited copy for ``shadow_steps``
    on the full training set. The widen is kept only when the edited copy has
    lower probe loss. Returns probe accuracy after each minibatch step.
    """

    accuracies: list[float] = []
    count = int(images.shape[0])
    for step in range(steps):
        if every > 0 and step > 0 and step % every == 0:

            def propose_widen(shadow: ValmorphMLP, at: int = step) -> LineageEdge:
                parent = shadow.unit_ids[0][0]
                child = widen(shadow, 0, 0)
                return LineageEdge(
                    step=at,
                    parent_unit=parent,
                    child_unit=child,
                    relation="split",
                )

            consider_edit(
                net,
                propose_widen,
                probe_x=probe_images,
                probe_y=probe_labels,
                train_x=images,
                train_y=labels,
                shadow_steps=shadow_steps,
                lr=lr,
                score=_cross_entropy,
                train_fn=train_cross_entropy_steps,
                ledger=ledger,
                step=step,
                proposal_id=f"widen-{step}",
                edit_type="widen",
                target="layer0",
                params={"unit": 0, "shadow_steps": shadow_steps},
            )
        start = (step * batch_size) % count
        index = torch.arange(start, start + batch_size) % count
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        optimizer.zero_grad()
        loss = F.cross_entropy(net(images[index]), labels[index])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            accuracies.append(batch_accuracy(net(probe_images), probe_labels))
    return accuracies


def _cross_entropy(net: ValmorphMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(net(x), y))
