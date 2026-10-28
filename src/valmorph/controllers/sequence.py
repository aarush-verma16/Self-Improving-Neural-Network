"""L0 widen trials across a sequence of tasks.

The gate scores every task probe seen so far, concatenated, so each task
weighs the same when the probes have the same number of rows. Shadow
training uses only the task being trained. Both choices are ours.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional as F

from valmorph.benchmarks.classify import batch_accuracy
from valmorph.core.ledger import GateDecision, Ledger, LineageEdge
from valmorph.gate import consider_edit, train_cross_entropy_steps
from valmorph.models import ValmorphMLP, lr_change, prune, widen


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
    choose: Callable[[ValmorphMLP, torch.Generator], tuple[str, int]] | None = None,
    controller: str = "L0",
    generator: torch.Generator | None = None,
    on_decision: Callable[[str, GateDecision], None] | None = None,
    on_task_end: Callable[[int, ValmorphMLP], None] | None = None,
) -> list[list[float]]:
    """Train tasks in order and propose an edit on a fixed interval.

    The default edit is a widen of unit 0, recorded as controller L0.
    ``choose`` returns ``prune`` or ``widen`` and a unit index. The unit draw
    uses ``generator``. Returns one probe-accuracy list per task. Each list
    has length ``steps_per_task * len(tasks)``. A kept edit is applied at the
    current weights. The shadow's extra steps stay off the live network.
    ``on_task_end`` is called with the task index after that task's pass.
    """

    if not tasks or len(tasks) != len(probes):
        raise ValueError("tasks and probes must be the same non-empty length")
    if steps_per_task < 1:
        raise ValueError("steps_per_task must be positive")
    if choose is not None and generator is None:
        raise ValueError("choose requires a generator")
    series: list[list[float]] = [[] for _ in probes]
    for task_index, (images, labels) in enumerate(tasks):
        count = int(images.shape[0])
        probe_x = torch.cat([probes[i][0] for i in range(task_index + 1)])
        probe_y = torch.cat([probes[i][1] for i in range(task_index + 1)])
        for local in range(steps_per_task):
            step = task_index * steps_per_task + local
            if every > 0 and step > 0 and step % every == 0:
                if choose is None:
                    kind, unit = "widen", 0
                    proposal_id = f"widen-{step}"
                    edit_controller = "L0"
                else:
                    assert generator is not None
                    kind, unit = choose(net, generator)
                    proposal_id = f"{controller.lower()}-{kind}-{step}"
                    edit_controller = controller
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
                    proposal_id=proposal_id,
                    edit_type=kind,
                    target="layer0",
                    controller=edit_controller,
                    params={
                        "unit": unit,
                        "unit_id": unit_id,
                        "shadow_steps": shadow_steps,
                        "task": task_index,
                        "probes": task_index + 1,
                        "lr_factor": 0.5 if kind == "lr_change" else None,
                    },
                )
                if on_decision is not None:
                    on_decision(kind, decision)
            start = (local * batch_size) % count
            index = torch.arange(start, start + batch_size) % count
            train_cross_entropy_steps(net, images[index], labels[index], 1, lr)
            with torch.no_grad():
                for slot, (seen_x, seen_y) in enumerate(probes):
                    series[slot].append(batch_accuracy(net(seen_x), seen_y))
        if on_task_end is not None:
            on_task_end(task_index, net)
    return series


def _cross_entropy(net: ValmorphMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(net(x), y))
