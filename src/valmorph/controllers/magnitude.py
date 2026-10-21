"""Propose pruning the smallest incoming-magnitude hidden unit.

The gate still decides. A kept prune is applied at the live weights. The
matched copy takes the shadow's extra steps and does not prune.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from valmorph.benchmarks.classify import batch_accuracy
from valmorph.core.ledger import Ledger, LineageEdge
from valmorph.gate import adopt_shadow, clone_net, consider_edit, train_cross_entropy_steps
from valmorph.introspect import smallest_magnitude_unit
from valmorph.models import ValmorphMLP, prune


def train_with_magnitude_prune(
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
    matched: ValmorphMLP,
    every: int = 20,
    shadow_steps: int = 8,
    min_gain: float = 1e-4,
) -> tuple[list[float], list[float]]:
    """Train and periodically propose pruning the smallest unit in layer 0.

    Magnitude is the L2 norm of the incoming weights and bias. Returns probe
    accuracy for the editor and for ``matched``.
    """

    accuracies: list[float] = []
    matched_accuracies: list[float] = []
    count = int(images.shape[0])

    def minibatch(model: ValmorphMLP, index: torch.Tensor) -> None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images[index]), labels[index])
        loss.backward()
        optimizer.step()

    for step in range(steps):
        if every > 0 and step > 0 and step % every == 0:
            unit = smallest_magnitude_unit(net, 0)
            if unit is not None:
                unit_id = net.unit_ids[0][unit]

                def propose(
                    shadow: ValmorphMLP,
                    at: int = step,
                    chosen: int = unit,
                ) -> LineageEdge:
                    removed = prune(shadow, 0, chosen)
                    return LineageEdge(
                        step=at,
                        parent_unit=removed,
                        child_unit=removed,
                        relation="prune",
                    )

                decision = consider_edit(
                    net,
                    propose,
                    probe_x=probe_images,
                    probe_y=probe_labels,
                    train_x=images,
                    train_y=labels,
                    shadow_steps=shadow_steps,
                    lr=lr,
                    min_gain=min_gain,
                    score=_cross_entropy,
                    train_fn=train_cross_entropy_steps,
                    ledger=ledger,
                    step=step,
                    proposal_id=f"magnitude-{step}",
                    edit_type="prune",
                    target="layer0",
                    params={"unit": unit, "unit_id": unit_id, "shadow_steps": shadow_steps},
                )
                if decision.accepted:
                    twin = clone_net(matched)
                    train_cross_entropy_steps(twin, images, labels, shadow_steps, lr)
                    adopt_shadow(matched, twin)
        start = (step * batch_size) % count
        index = torch.arange(start, start + batch_size) % count
        minibatch(net, index)
        minibatch(matched, index)
        with torch.no_grad():
            accuracies.append(batch_accuracy(net(probe_images), probe_labels))
            matched_accuracies.append(batch_accuracy(matched(probe_images), probe_labels))
    return accuracies, matched_accuracies


def _cross_entropy(net: ValmorphMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(net(x), y))
