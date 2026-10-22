"""Random-edit control. Same times and the same gate as the other controllers.

Only the choice of edit is random: a prune or a widen of one hidden unit in
the first layer. The gate still keeps an edit only when probe loss falls.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from valmorph.benchmarks.classify import batch_accuracy
from valmorph.core.ledger import Ledger, LineageEdge
from valmorph.gate import adopt_shadow, clone_net, consider_edit, train_cross_entropy_steps
from valmorph.models import ValmorphMLP, lr_change, prune, widen


def choose_edit(net: ValmorphMLP, generator: torch.Generator) -> tuple[str, int]:
    """Pick prune, widen, or a learning-rate halving, then a unit.

    A layer with one unit cannot be pruned. The unit draw is unused when the
    kind is ``lr_change``.
    """

    width = int(net.linear(0).out_features)
    kinds = ("prune", "widen", "lr_change") if width > 1 else ("widen", "lr_change")
    kind = kinds[int(torch.randint(0, len(kinds), (1,), generator=generator))]
    unit = int(torch.randint(0, width, (1,), generator=generator))
    return kind, unit


def train_with_random_edits(
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
    generator: torch.Generator,
    every: int = 20,
    shadow_steps: int = 8,
    min_gain: float = 1e-4,
) -> tuple[list[float], list[float]]:
    """Train and periodically propose a random prune or widen.

    The gate matches ``shadow_steps`` of training on the edited and unedited
    copies. The live network receives a kept edit at its current weights.
    ``matched`` sees the same minibatches and, when an edit is kept, the
    shadow's extra steps without the edit. Returns probe accuracy for the
    editor and for ``matched``.
    """

    accuracies: list[float] = []
    matched_accuracies: list[float] = []
    count = int(images.shape[0])

    def minibatch(model: ValmorphMLP, index: torch.Tensor) -> None:
        train_cross_entropy_steps(model, images[index], labels[index], 1, lr)

    for step in range(steps):
        if every > 0 and step > 0 and step % every == 0:
            kind, unit = choose_edit(net, generator)
            unit_id = net.unit_ids[0][unit]

            def propose(
                shadow: ValmorphMLP,
                at: int = step,
                chosen: int = unit,
                edit_kind: str = kind,
            ) -> LineageEdge | None:
                if edit_kind == "widen":
                    parent = shadow.unit_ids[0][chosen]
                    child = widen(shadow, 0, chosen)
                    return LineageEdge(
                        step=at,
                        parent_unit=parent,
                        child_unit=child,
                        relation="split",
                    )
                if edit_kind == "prune":
                    removed = prune(shadow, 0, chosen)
                    return LineageEdge(
                        step=at,
                        parent_unit=removed,
                        child_unit=removed,
                        relation="prune",
                    )
                lr_change(shadow, 0, 0.5)
                return None

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
                proposal_id=f"random-{kind}-{step}",
                edit_type=kind,
                target="layer0",
                controller="random",
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
