"""L0 widen trials across a sequence of tasks.

The gate scores every task probe seen so far, concatenated, so each task
weighs the same when the probes have the same number of rows. Shadow
training uses only the task being trained. Both choices are ours.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from valmorph.benchmarks.classify import batch_accuracy
from valmorph.core.ledger import Ledger, LineageEdge
from valmorph.gate import consider_edit, train_cross_entropy_steps
from valmorph.models import ValmorphMLP, widen


def train_across_tasks(
    net: ValmorphMLP,
    tasks: list[tuple[torch.Tensor, torch.Tensor]],
    probes: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    steps_per_task: int,
    batch_size: int,
    lr: float,
    ledger: Ledger,
    every: int = 20,
    shadow_steps: int = 8,
    min_gain: float = 1e-4,
) -> list[list[float]]:
    """Train tasks in order and propose a widen of unit 0 on a fixed interval.

    Returns one probe-accuracy list per task. Each list has length
    ``steps_per_task * len(tasks)``. A kept widen is applied at the current
    weights. The shadow's extra steps stay off the live network.
    """

    if not tasks or len(tasks) != len(probes):
        raise ValueError("tasks and probes must be the same non-empty length")
    if steps_per_task < 1:
        raise ValueError("steps_per_task must be positive")
    series: list[list[float]] = [[] for _ in probes]
    for task_index, (images, labels) in enumerate(tasks):
        count = int(images.shape[0])
        probe_x = torch.cat([probes[i][0] for i in range(task_index + 1)])
        probe_y = torch.cat([probes[i][1] for i in range(task_index + 1)])
        for local in range(steps_per_task):
            step = task_index * steps_per_task + local
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
                    probe_x=probe_x,
                    probe_y=probe_y,
                    train_x=images,
                    train_y=labels,
                    shadow_steps=shadow_steps,
                    lr=lr,
                    min_gain=min_gain,
                    score=_cross_entropy,
                    train_fn=train_cross_entropy_steps,
                    ledger=ledger,
                    step=step,
                    proposal_id=f"widen-{step}",
                    edit_type="widen",
                    target="layer0",
                    params={
                        "unit": 0,
                        "shadow_steps": shadow_steps,
                        "task": task_index,
                        "probes": task_index + 1,
                    },
                )
            start = (local * batch_size) % count
            index = torch.arange(start, start + batch_size) % count
            optimizer = torch.optim.SGD(net.parameters(), lr=lr)
            optimizer.zero_grad()
            loss = F.cross_entropy(net(images[index]), labels[index])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for slot, (seen_x, seen_y) in enumerate(probes):
                    series[slot].append(batch_accuracy(net(seen_x), seen_y))
    return series


def _cross_entropy(net: ValmorphMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(net(x), y))
