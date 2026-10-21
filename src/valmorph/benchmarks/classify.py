"""A few SGD steps of classification. Used by the task notebook."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from valmorph.models.mlp import ValmorphMLP


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of correct rows. ``logits`` is ``[batch, classes]``."""

    return float((logits.argmax(dim=1) == labels).to(dtype=torch.float32).mean())


def train_classifier(
    net: ValmorphMLP,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    probe_images: torch.Tensor,
    probe_labels: torch.Tensor,
) -> tuple[list[float], list[float]]:
    """Return training loss and probe accuracy, one pair per step.

    ``images`` is ``[n, features]``. The probe is scored in full each step.
    """

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    losses: list[float] = []
    accuracies: list[float] = []
    count = int(images.shape[0])
    for step in range(steps):
        start = (step * batch_size) % count
        index = torch.arange(start, start + batch_size) % count
        batch_x = images[index]
        batch_y = labels[index]
        optimizer.zero_grad()
        logits = net(batch_x)
        loss = F.cross_entropy(logits, batch_y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            accuracies.append(batch_accuracy(net(probe_images), probe_labels))
    return losses, accuracies
