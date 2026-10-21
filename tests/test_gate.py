"""The gate keeps a helpful edit and refuses a harmful one."""

from __future__ import annotations

from pathlib import Path

import torch

from valmorph.core.ledger import LEDGER_VERSION, Ledger, LineageEdge, RunMeta
from valmorph.gate import clone_net, consider_edit, probe_mse, train_steps
from valmorph.models import ValmorphMLP, prune, widen


def _ledger(tmp_path: Path) -> Ledger:
    return Ledger.create(
        tmp_path,
        RunMeta(
            ledger_version=LEDGER_VERSION,
            run_id="gate",
            created_utc="2026-10-19T00:00:00Z",
            synthetic=False,
            experiment="gate",
            seed=0,
            config_hash="gate",
            config={"experiment": "gate"},
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


def _bad_net() -> tuple[ValmorphMLP, torch.Tensor, torch.Tensor]:
    net = ValmorphMLP([1, 2, 1]).cpu()
    with torch.no_grad():
        net.linear(0).weight.zero_()
        net.linear(0).bias.copy_(torch.tensor([0.0, 10.0]))
        net.linear(1).weight.copy_(torch.tensor([[0.0, 10.0]]))
        net.linear(1).bias.zero_()
    x = torch.tensor([[1.0], [0.0]])
    y = torch.zeros(2, 1)
    return net, x, y


def test_harmful_prune_is_rejected_and_live_net_stays(tmp_path: Path) -> None:
    net = ValmorphMLP([1, 2, 1]).cpu()
    with torch.no_grad():
        net.linear(0).weight.fill_(1.0)
        net.linear(0).bias.zero_()
        net.linear(1).weight.fill_(1.0)
        net.linear(1).bias.zero_()
    x = torch.tensor([[1.0]])
    y = torch.tensor([[2.0]])
    before = net(x).clone()
    decision = consider_edit(
        net,
        lambda shadow: prune(shadow, 0, 0),
        probe_x=x,
        probe_y=y,
        ledger=_ledger(tmp_path),
        step=1,
        proposal_id="prune-needed",
        edit_type="prune",
        target="layer0",
    )
    assert decision.accepted is False
    assert decision.validated_gain > 0
    assert torch.allclose(net(x), before)
    reopened = Ledger.reopen(tmp_path, "gate")
    assert reopened.read("gate_decisions").column("accepted").to_pylist() == [False]
    assert reopened.read("lineage").num_rows == 0


def test_helpful_prune_is_accepted_and_live_matches_shadow(tmp_path: Path) -> None:
    net, x, y = _bad_net()
    assert probe_mse(net, x, y) > 1.0
    decision = consider_edit(
        net,
        lambda shadow: prune(shadow, 0, 1),
        probe_x=x,
        probe_y=y,
        ledger=_ledger(tmp_path),
        step=0,
        proposal_id="prune-bad",
        edit_type="prune",
        target="layer0",
    )
    assert decision.accepted is True
    assert decision.validated_gain < 0
    assert probe_mse(net, x, y) < 1e-6
    assert int(net.linear(0).out_features) == 1
    decisions = Ledger.reopen(tmp_path, "gate").read("gate_decisions")
    assert decisions.column("accepted").to_pylist() == [True]
    assert Ledger.reopen(tmp_path, "gate").read("lineage").num_rows == 1


def test_accepted_widen_lineage_names_parent_and_child(tmp_path: Path) -> None:
    net, x, y = _bad_net()

    def edit(shadow: ValmorphMLP) -> LineageEdge:
        parent = shadow.unit_ids[0][0]
        child = widen(shadow, 0, 0)
        with torch.no_grad():
            shadow.linear(1).weight[:, 1] = 0.0
        return LineageEdge(step=0, parent_unit=parent, child_unit=child, relation="split")

    decision = consider_edit(
        net,
        edit,
        probe_x=x,
        probe_y=y,
        ledger=_ledger(tmp_path),
        step=0,
        proposal_id="widen-repair",
        edit_type="widen",
        target="layer0",
    )
    assert decision.accepted is True
    edges = Ledger.reopen(tmp_path, "gate").read("lineage")
    assert edges.column("parent_unit").to_pylist() == [net.unit_ids[0][0]]
    assert edges.column("child_unit").to_pylist() != edges.column("parent_unit").to_pylist()
    assert edges.column("relation").to_pylist() == ["split"]


def test_accept_keeps_the_edit_not_the_shadow_training(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([2, 2, 1]).cpu()
    original = clone_net(net)
    x = torch.randn(16, 2)
    y = torch.randn(16, 1)
    decision = consider_edit(
        net,
        lambda shadow: widen(shadow, 0, 0),
        probe_x=x[:4],
        probe_y=y[:4],
        train_x=x,
        train_y=y,
        shadow_steps=5,
        lr=0.5,
        min_gain=-1.0e9,
        ledger=_ledger(tmp_path),
        step=1,
        proposal_id="widen-shadow",
        edit_type="widen",
        target="layer0",
    )
    assert decision.accepted is True
    expected = clone_net(original)
    widen(expected, 0, 0)
    assert torch.allclose(net.linear(0).weight, expected.linear(0).weight)
    assert torch.allclose(net.linear(1).weight, expected.linear(1).weight)
    trained = clone_net(expected)
    train_steps(trained, x, y, 5, 0.5)
    assert not torch.allclose(net.linear(0).weight, trained.linear(0).weight)
