"""Sequential task training scores every probe at every step."""

from __future__ import annotations

import torch

from valmorph.benchmarks.classify import (
    train_classifier,
    train_on_tasks,
    train_rehearsal_end_scores,
    train_sequence_end_scores,
    train_with_frozen_layer,
    train_with_rehearsal,
)
from valmorph.gate import clone_net, train_cross_entropy_steps
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


def test_task_step_counts_can_differ() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    tasks = [(torch.rand(6, 4), torch.randint(0, 2, (6,))) for _ in range(2)]
    probes = [(torch.rand(4, 4), torch.randint(0, 2, (4,)))]
    series = train_on_tasks(net, tasks, probes, steps_per_task=[2, 5], batch_size=2, lr=0.1)
    assert len(series[0]) == 7


def test_freeze_at_a_step_holds_the_hidden_weights() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    images = torch.rand(6, 4)
    labels = torch.randint(0, 2, (6,))
    probes = [(torch.rand(4, 4), torch.randint(0, 2, (4,)))]
    train_with_frozen_layer(
        net,
        [(images, labels)],
        probes,
        steps_per_task=3,
        batch_size=2,
        lr=0.1,
        freeze_at=1,
    )
    held = net.linear(0).weight.detach().clone()
    train_cross_entropy_steps(net, images, labels, 1, 0.1)
    assert net.lr_scale[0] == 0.0
    assert torch.equal(net.linear(0).weight.detach(), held)


def test_freeze_can_hold_the_output_layer() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    images = torch.rand(6, 4)
    labels = torch.randint(0, 2, (6,))
    probes = [(torch.rand(4, 4), torch.randint(0, 2, (4,)))]
    train_with_frozen_layer(
        net,
        [(images, labels)],
        probes,
        steps_per_task=3,
        batch_size=2,
        lr=0.1,
        freeze_at=1,
        layer=1,
    )
    held = net.linear(1).weight.detach().clone()
    train_cross_entropy_steps(net, images, labels, 1, 0.1)
    assert net.lr_scale == [1.0, 0.0]
    assert torch.equal(net.linear(1).weight.detach(), held)


def test_rehearsal_without_stored_rows_matches_plain_training() -> None:
    torch.manual_seed(0)
    images = torch.rand(8, 4)
    labels = torch.randint(0, 2, (8,))
    probes = [(torch.rand(4, 4), torch.randint(0, 2, (4,)))]
    tasks = [(images, labels), (torch.rand(8, 4), torch.randint(0, 2, (8,)))]
    plain = ValmorphMLP([4, 3, 2])
    mixed = clone_net(plain)
    plain_series = train_on_tasks(plain, tasks, probes, steps_per_task=3, batch_size=4, lr=0.1)
    mixed_series = train_with_rehearsal(
        mixed,
        tasks,
        probes,
        steps_per_task=3,
        batch_size=4,
        lr=0.1,
        memory=0,
    )
    assert mixed_series == plain_series


def test_rehearsal_keeps_a_prefix_of_each_finished_task() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    tasks = [(torch.rand(8, 4), torch.randint(0, 2, (8,))) for _ in range(2)]
    probes = [(torch.rand(4, 4), torch.randint(0, 2, (4,))) for _ in range(2)]
    series = train_with_rehearsal(
        net,
        tasks,
        probes,
        steps_per_task=2,
        batch_size=4,
        lr=0.1,
        memory=3,
    )
    assert [len(values) for values in series] == [4, 4]
    assert all(0.0 <= value <= 1.0 for values in series for value in values)


def test_one_pass_rehearsal_without_stored_rows_matches_the_plain_pass() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    tasks = [(torch.rand(5, 4), torch.randint(0, 2, (5,))) for _ in range(2)]
    tests = [(torch.rand(3, 4), torch.randint(0, 2, (3,))) for _ in range(2)]
    plain = train_sequence_end_scores(clone_net(net), tasks, tests, batch_size=4, lr=0.1)
    mix = train_rehearsal_end_scores(clone_net(net), tasks, tests, batch_size=4, lr=0.1, memory=0)
    assert mix == plain


def test_one_pass_rehearsal_keeps_the_first_task_and_changes_the_second() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    tasks = [(torch.rand(5, 4), torch.randint(0, 2, (5,))) for _ in range(2)]
    tests = [(torch.rand(3, 4), torch.randint(0, 2, (3,))) for _ in range(2)]
    plain_net = clone_net(net)
    mix_net = clone_net(net)
    plain = train_sequence_end_scores(plain_net, tasks, tests, batch_size=4, lr=0.1)
    mix = train_rehearsal_end_scores(mix_net, tasks, tests, batch_size=4, lr=0.1, memory=None)
    assert mix[0] == plain[0]
    pairs = zip(plain_net.parameters(), mix_net.parameters(), strict=True)
    assert any(not torch.allclose(left, right) for left, right in pairs)


def test_sequence_end_scores_once_per_task() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    tasks = [(torch.rand(5, 4), torch.randint(0, 2, (5,))) for _ in range(2)]
    tests = [(torch.rand(3, 4), torch.randint(0, 2, (3,))) for _ in range(2)]
    rows = train_sequence_end_scores(net, tasks, tests, batch_size=4, lr=0.1)
    assert [len(row) for row in rows] == [2, 2]
    assert all(0.0 <= value <= 1.0 for row in rows for value in row)
