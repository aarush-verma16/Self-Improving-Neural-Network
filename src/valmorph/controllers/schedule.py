"""Widen on a fixed schedule. The gate is not asked.

Each widen is function-preserving at the moment it is applied. Later
minibatch steps can use the new unit. This is the ungated growth control.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from valmorph.benchmarks.classify import batch_accuracy
from valmorph.models import ValmorphMLP, widen


def train_with_scheduled_widen(
    net: ValmorphMLP,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    probe_images: torch.Tensor,
    probe_labels: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    every: int = 10,
    until: int | None = None,
) -> list[float]:
    """Train and widen hidden unit 0 of layer 0 every ``every`` steps.

    ``until`` is the last step that may widen. Later steps only train.
    Returns probe accuracy after each minibatch step.
    """

    accuracies: list[float] = []
    count = int(images.shape[0])
    for step in range(steps):
        due = every > 0 and step > 0 and step % every == 0
        if due and (until is None or step <= until):
            widen(net, 0, 0)
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
