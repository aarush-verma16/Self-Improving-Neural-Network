"""Controllers keep a matched copy that receives extra steps and no edit."""

from __future__ import annotations

from pathlib import Path

import torch

from valmorph.controllers import (
    EditBandit,
    train_across_tasks,
    train_with_random_edits,
    train_with_scheduled_widen,
    train_with_widen_trials,
)
from valmorph.core.ledger import LEDGER_VERSION, Ledger, RunMeta
from valmorph.gate import clone_net
from valmorph.models import ValmorphMLP


def _ledger(tmp_path: Path, run_id: str = "widen") -> Ledger:
    return Ledger.create(
        tmp_path,
        RunMeta(
            ledger_version=LEDGER_VERSION,
            run_id=run_id,
            created_utc="2026-10-20T00:00:00Z",
            synthetic=False,
            experiment="widen",
            seed=0,
            config_hash="widen",
            config={"experiment": "widen"},
            git_commit="test",
            device_type="cpu",
            device_name="cpu",
            device_index=None,
            capability=None,
            torch_version="test",
            cuda_version=None,
            python_version="3.12",
            platform="test",
            libraries={},
        ),
    )


def test_matched_copy_keeps_its_width_when_a_widen_is_accepted(tmp_path: Path) -> None:
    torch.manual_seed(0)
    editor = ValmorphMLP([4, 3, 2]).cpu()
    matched = clone_net(editor)
    images = torch.randn(16, 4)
    labels = torch.randint(0, 2, (16,))
    editor_accuracy, matched_accuracy = train_with_widen_trials(
        editor,
        images,
        labels,
        probe_images=torch.randn(8, 4),
        probe_labels=torch.randint(0, 2, (8,)),
        steps=2,
        batch_size=4,
        lr=0.1,
        ledger=_ledger(tmp_path),
        matched=matched,
        every=1,
        shadow_steps=0,
        min_gain=-1.0,
    )
    assert len(editor_accuracy) == 2
    assert len(matched_accuracy) == 2
    assert int(editor.linear(0).out_features) == 4
    assert int(matched.linear(0).out_features) == 3
    edges = Ledger.reopen(tmp_path, "widen").read("lineage")
    assert edges.column("relation").to_pylist() == ["split"]
    assert edges.column("parent_unit").to_pylist() != edges.column("child_unit").to_pylist()


def test_random_controller_is_labeled_and_leaves_the_matched_width(tmp_path: Path) -> None:
    torch.manual_seed(0)
    editor = ValmorphMLP([4, 3, 2]).cpu()
    matched = clone_net(editor)
    images = torch.randn(16, 4)
    labels = torch.randint(0, 2, (16,))
    train_with_random_edits(
        editor,
        images,
        labels,
        probe_images=torch.randn(8, 4),
        probe_labels=torch.randint(0, 2, (8,)),
        steps=4,
        batch_size=4,
        lr=0.1,
        ledger=_ledger(tmp_path, "random"),
        matched=matched,
        generator=torch.Generator().manual_seed(0),
        every=1,
        shadow_steps=0,
        min_gain=-1.0e9,
    )
    opened = Ledger.reopen(tmp_path, "random")
    proposals = opened.read("edit_proposals")
    assert proposals.column("controller").to_pylist() == ["random", "random", "random"]
    assert set(proposals.column("edit_type").to_pylist()) <= {"prune", "widen"}
    assert opened.read("gate_decisions").column("accepted").to_pylist() == [True, True, True]
    assert int(matched.linear(0).out_features) == 3
    assert opened.read("lineage").num_rows == 3


def test_scheduled_widen_grows_on_the_interval() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2]).cpu()
    images = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,))
    accuracy = train_with_scheduled_widen(
        net,
        images,
        labels,
        probe_images=images,
        probe_labels=labels,
        steps=3,
        batch_size=4,
        lr=0.1,
        every=1,
    )
    assert len(accuracy) == 3
    assert int(net.linear(0).out_features) == 5
    held = ValmorphMLP([4, 3, 2]).cpu()
    train_with_scheduled_widen(
        held,
        images,
        labels,
        probe_images=images,
        probe_labels=labels,
        steps=3,
        batch_size=4,
        lr=0.1,
        every=1,
        until=1,
    )
    assert int(held.linear(0).out_features) == 4


def test_task_widen_trials_grow_only_when_the_gate_keeps_them(tmp_path: Path) -> None:
    torch.manual_seed(0)
    tasks = [(torch.rand(6, 4), torch.randint(0, 2, (6,))) for _ in range(2)]
    probes = [(torch.rand(4, 4), torch.randint(0, 2, (4,))) for _ in range(2)]
    kept = ValmorphMLP([4, 3, 2])
    series = train_across_tasks(
        kept,
        tasks,
        probes,
        steps_per_task=4,
        batch_size=2,
        lr=0.1,
        ledger=_ledger(tmp_path, "kept"),
        every=2,
        shadow_steps=0,
        min_gain=-1e9,
    )
    assert [len(values) for values in series] == [8, 8]
    assert int(kept.linear(0).out_features) == 6
    refused = ValmorphMLP([4, 3, 2])
    train_across_tasks(
        refused,
        tasks,
        probes,
        steps_per_task=4,
        batch_size=2,
        lr=0.1,
        ledger=_ledger(tmp_path, "refused"),
        every=2,
        shadow_steps=0,
        min_gain=1e9,
    )
    assert int(refused.linear(0).out_features) == 3


def test_bandit_tries_each_arm_then_follows_the_mean() -> None:
    bandit = EditBandit()
    generator = torch.Generator().manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    assert bandit.choose(net, generator)[0] == "prune"
    bandit.observe("prune", 0.4)
    assert bandit.choose(net, generator)[0] == "widen"
    bandit.observe("widen", -0.2)
    assert bandit.choose(net, generator)[0] == "widen"
    bandit.observe("widen", 1.2)
    assert bandit.choose(net, generator)[0] == "prune"


def test_bandit_explore_step_retries_the_less_used_arm() -> None:
    bandit = EditBandit()
    generator = torch.Generator().manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    bandit.choose(net, generator)
    bandit.observe("prune", 0.0)
    bandit.choose(net, generator)
    bandit.observe("widen", -1.0)
    assert bandit.choose(net, generator)[0] == "widen"
    bandit.observe("widen", -1.0)
    assert bandit.choose(net, generator)[0] == "widen"
    bandit.observe("widen", -1.0)
    assert bandit.choose(net, generator)[0] == "prune"


def test_l1_choice_is_labeled_on_the_ledger(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    bandit = EditBandit()
    generator = torch.Generator().manual_seed(0)
    tasks = [(torch.rand(6, 4), torch.randint(0, 2, (6,)))]
    probes = [(torch.rand(4, 4), torch.randint(0, 2, (4,)))]
    ledger = _ledger(tmp_path, "l1")
    series = train_across_tasks(
        net,
        tasks,
        probes,
        steps_per_task=4,
        batch_size=2,
        lr=0.1,
        ledger=ledger,
        every=2,
        shadow_steps=0,
        min_gain=-1e9,
        choose=bandit.choose,
        controller="L1",
        generator=generator,
        on_decision=lambda kind, decision: bandit.observe(kind, decision.validated_gain),
    )
    assert len(series[0]) == 4
    assert ledger.read("edit_proposals").column("controller").to_pylist() == ["L1"]
    assert bandit.count["prune"] == 1
