"""The first neural-network milestone: function-preserving edits on CPU."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from valmorph.core.ledger import LEDGER_VERSION, Ledger, RunMeta
from valmorph.gate import adopt_shadow, clone_net, train_cross_entropy_steps
from valmorph.models.edits import EditError, freeze, identity_deepen, lr_change, widen
from valmorph.models.mlp import SubstrateError, ValmorphMLP


def _net(widths: list[int], seed: int) -> ValmorphMLP:
    torch.manual_seed(seed)
    return ValmorphMLP(widths).cpu()


def test_hand_checked_widen_keeps_the_logit() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([1, 1, 1]).cpu()
    with torch.no_grad():
        net.linear(0).weight.fill_(2.0)
        net.linear(0).bias.fill_(0.5)
        net.linear(1).weight.fill_(3.0)
        net.linear(1).bias.fill_(-1.0)
    x = torch.tensor([[1.0]])
    before = net(x).clone()
    assert torch.allclose(before, torch.tensor([[6.5]]))
    widen(net, 0, 0)
    assert torch.allclose(net(x), before)


def test_identity_deepen_is_the_identity_map() -> None:
    net = _net([3, 4, 2], seed=1)
    x = torch.randn(6, 3)
    before = net(x).clone()
    identity_deepen(net, 0)
    assert torch.allclose(net(x), before)
    assert net.use_relu[1] is False


def test_widen_then_deepen_on_a_fresh_batch() -> None:
    net = _net([4, 5, 3], seed=2)
    first = torch.randn(8, 4)
    second = torch.randn(5, 4)
    before_first = net(first).clone()
    before_second = net(second).clone()
    widen(net, 0, 2)
    identity_deepen(net, 0)
    assert torch.allclose(net(first), before_first)
    assert torch.allclose(net(second), before_second)


def test_rejects_bad_shapes() -> None:
    with pytest.raises(SubstrateError):
        ValmorphMLP([4, 2])
    net = _net([3, 4, 2], seed=0)
    with pytest.raises(EditError):
        widen(net, 1, 0)
    with pytest.raises(EditError):
        widen(net, 0, 9)


def test_widen_records_a_split_in_the_ledger(tmp_path: Path) -> None:
    net = _net([3, 4, 2], seed=3)
    parent = net.unit_ids[0][1]
    meta = RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id="substrate",
        created_utc="2026-10-10T00:00:00Z",
        synthetic=False,
        experiment="substrate",
        seed=3,
        config_hash="substrate",
        config={"experiment": "substrate"},
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
    )
    ledger = Ledger.create(tmp_path, meta)
    child = widen(net, 0, 1, ledger=ledger, step=0)
    edges = ledger.read("lineage")
    assert edges.column("parent_unit").to_pylist() == [parent]
    assert edges.column("child_unit").to_pylist() == [child]
    assert edges.column("relation").to_pylist() == ["split"]
    assert ledger.read("edit_proposals").column("edit_type").to_pylist() == ["widen"]


@settings(max_examples=30, deadline=None)
@given(
    st.lists(st.integers(min_value=2, max_value=6), min_size=3, max_size=4),
    st.integers(min_value=0, max_value=10_000),
)
def test_edits_preserve_the_function(widths: list[int], seed: int) -> None:
    net = _net(widths, seed)
    batch = torch.randn(7, widths[0])
    held_out = torch.randn(4, widths[0])
    before_batch = net(batch).clone()
    before_held = net(held_out).clone()
    hidden = len(net.layers) - 1
    widen(net, seed % hidden, seed % int(net.linear(seed % hidden).out_features))
    identity_deepen(net, 0)
    assert torch.allclose(net(batch), before_batch, rtol=1e-5, atol=1e-5)
    assert torch.allclose(net(held_out), before_held, rtol=1e-5, atol=1e-5)


def test_lr_change_keeps_logits_and_halves_the_hidden_step() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    plain = clone_net(net)
    images = torch.rand(6, 4)
    labels = torch.randint(0, 2, (6,))
    before = net(images).detach().clone()
    assert lr_change(net, 0, 0.5) == 0.5
    assert torch.equal(net(images), before)
    hidden_before = net.linear(0).weight.detach().clone()
    output_before = net.linear(1).weight.detach().clone()
    train_cross_entropy_steps(net, images, labels, 1, 0.2)
    train_cross_entropy_steps(plain, images, labels, 1, 0.2)
    hidden_delta = net.linear(0).weight.detach() - hidden_before
    plain_hidden = plain.linear(0).weight.detach() - hidden_before
    assert torch.allclose(hidden_delta, 0.5 * plain_hidden)
    output_delta = net.linear(1).weight.detach() - output_before
    plain_output = plain.linear(1).weight.detach() - output_before
    assert torch.allclose(output_delta, plain_output)
    with pytest.raises(EditError):
        lr_change(net, 0, 0)


def test_adopted_shadow_keeps_the_learning_rate_scale() -> None:
    live = ValmorphMLP([4, 3, 2])
    shadow = clone_net(live)
    lr_change(shadow, 0, 0.5)
    adopt_shadow(live, shadow)
    assert live.lr_scale == [0.5, 1.0]


def test_identity_deepen_inserts_a_unit_learning_rate_scale() -> None:
    net = ValmorphMLP([4, 3, 2])
    identity_deepen(net, 0)
    assert net.lr_scale == [1.0, 1.0, 1.0]


def test_freeze_stops_one_layer() -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([4, 3, 2])
    images = torch.rand(6, 4)
    labels = torch.randint(0, 2, (6,))
    before = net(images).detach().clone()
    assert freeze(net, 0) == 0.0
    assert torch.equal(net(images), before)
    hidden = net.linear(0).weight.detach().clone()
    output = net.linear(1).weight.detach().clone()
    train_cross_entropy_steps(net, images, labels, 1, 0.2)
    assert torch.equal(net.linear(0).weight.detach(), hidden)
    assert not torch.equal(net.linear(1).weight.detach(), output)
