"""Sequential task training scores every probe at every step."""

from __future__ import annotations

import torch

from valmorph.benchmarks.classify import train_classifier, train_on_tasks
from valmorph.gate import clone_net
from valmorph.models import ValmorphMLP


def test_one_task_matches_the_classifier() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    images = torch.rand(8, 4)
    labels = torch.randint(0, 2, (8,))
    probe_x = torch.rand(5, 4)
    probe_y = torch.randint(0, 2, (5,))
    copy = clone_net(net)
    _loss, accuracy = train_classifier(
        net,
        images,
        labels,
        steps=4,
        batch_size=3,
        lr=0.1,
        probe_images=probe_x,
        probe_labels=probe_y,
    )
    series = train_on_tasks(
        copy,
        [(images, labels)],
        [(probe_x, probe_y)],
        steps_per_task=4,
        batch_size=3,
        lr=0.1,
    )
    assert series[0] == accuracy


def test_two_tasks_score_both_probes() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    tasks = []
    probes = []
    for _ in range(2):
        tasks.append((torch.rand(6, 4), torch.randint(0, 2, (6,))))
        probes.append((torch.rand(4, 4), torch.randint(0, 2, (4,))))
    series = train_on_tasks(net, tasks, probes, steps_per_task=3, batch_size=2, lr=0.1)
    assert len(series) == 2
    assert [len(values) for values in series] == [6, 6]
    assert all(0.0 <= value <= 1.0 for values in series for value in values)
