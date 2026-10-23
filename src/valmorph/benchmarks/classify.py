"""A few SGD steps of classification. Used by the task notebook."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from valmorph.gate import train_cross_entropy_steps
from valmorph.models import freeze
from valmorph.models.mlp import ValmorphMLP


def probe_cross_entropy(net: ValmorphMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean cross-entropy. ``y`` is integer class labels of shape ``[batch]``."""

    with torch.no_grad():
        return float(F.cross_entropy(net(x), y))


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


def train_on_tasks(
    net: ValmorphMLP,
    tasks: list[tuple[torch.Tensor, torch.Tensor]],
    probes: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    steps_per_task: int | list[int],
    batch_size: int,
    lr: float,
) -> list[list[float]]:
    """Train tasks in order and score every probe after every step.

    Each task and each probe is ``(images [n, features], labels [n])``.
    ``steps_per_task`` is one count for every task, or one count per task.
    The return value has one accuracy list per probe, and each list has
    length equal to the sum of those counts. One SGD optimizer covers the run.
    """

    if not tasks or not probes:
        raise ValueError("tasks and probes must be non-empty")
    if isinstance(steps_per_task, int):
        counts = [steps_per_task] * len(tasks)
    else:
        counts = list(steps_per_task)
    if len(counts) != len(tasks) or any(count < 1 for count in counts):
        raise ValueError("steps_per_task must be a positive count for each task")
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    series: list[list[float]] = [[] for _ in probes]
    for (images, labels), task_steps in zip(tasks, counts, strict=True):
        count = int(images.shape[0])
        for step in range(task_steps):
            start = (step * batch_size) % count
            index = torch.arange(start, start + batch_size) % count
            optimizer.zero_grad()
            loss = F.cross_entropy(net(images[index]), labels[index])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for slot, (probe_x, probe_y) in enumerate(probes):
                    series[slot].append(batch_accuracy(net(probe_x), probe_y))
    return series


def train_with_frozen_layer(
    net: ValmorphMLP,
    tasks: list[tuple[torch.Tensor, torch.Tensor]],
    probes: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    steps_per_task: int,
    batch_size: int,
    lr: float,
    freeze_at: int,
    layer: int = 0,
) -> list[list[float]]:
    """Train tasks in order. At global step ``freeze_at``, stop updating ``layer``.

    Steps before the freeze use each layer's scale, which starts at 1. The
    returned lists have one probe accuracy per step, length
    ``steps_per_task * len(tasks)``.
    """

    if not tasks or len(tasks) != len(probes):
        raise ValueError("tasks and probes must be the same non-empty length")
    if freeze_at < 0:
        raise ValueError("freeze_at must be non-negative")
    series: list[list[float]] = [[] for _ in probes]
    for task_index, (images, labels) in enumerate(tasks):
        count = int(images.shape[0])
        for local in range(steps_per_task):
            step = task_index * steps_per_task + local
            if step == freeze_at:
                freeze(net, layer)
            start = (local * batch_size) % count
            index = torch.arange(start, start + batch_size) % count
            train_cross_entropy_steps(net, images[index], labels[index], 1, lr)
            with torch.no_grad():
                for slot, (probe_x, probe_y) in enumerate(probes):
                    series[slot].append(batch_accuracy(net(probe_x), probe_y))
    return series
