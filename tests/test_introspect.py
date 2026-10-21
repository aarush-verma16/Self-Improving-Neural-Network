"""Known-tensor checks for the diagnostics."""

from __future__ import annotations

from pathlib import Path

import torch

from valmorph.core.ledger import LEDGER_VERSION, Diagnostic, Ledger, RunMeta
from valmorph.introspect import (
    dead_unit_fraction,
    singular_entropy_rank,
)
from valmorph.models import ValmorphMLP


def test_dead_fraction_on_a_known_batch() -> None:
    activations = torch.tensor([[0.0, 2.0], [0.0, 3.0]])
    assert dead_unit_fraction(activations) == 0.5
    assert dead_unit_fraction(torch.ones(4, 3)) == 0.0
    assert dead_unit_fraction(torch.zeros(4, 5)) == 1.0


def test_singular_entropy_rank_on_identity_and_rank_one() -> None:
    eye = torch.eye(4)
    assert abs(singular_entropy_rank(eye) - 4.0) < 1e-5
    rank_one = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    assert abs(singular_entropy_rank(rank_one) - 1.0) < 1e-5
    assert singular_entropy_rank(torch.zeros(3, 3)) == 0.0


def test_gradient_norm_and_ledger_row(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = ValmorphMLP([2, 3, 1]).cpu()
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)
    from valmorph.introspect import hidden_activations, layer_gradient_norms

    activations = hidden_activations(net, x)
    assert activations[0].shape == (8, 3)
    norms = layer_gradient_norms(net, x, y)
    assert len(norms) == 2
    assert norms[0] > 0.0
    ledger = Ledger.create(
        tmp_path,
        RunMeta(
            ledger_version=LEDGER_VERSION,
            run_id="diag",
            created_utc="2026-10-20T00:00:00Z",
            synthetic=False,
            experiment="diagnostics",
            seed=0,
            config_hash="diag",
            config={},
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
    ledger.append_diagnostics(
        [
            Diagnostic(
                step=0,
                layer="hidden0",
                name="dead_unit_fraction",
                value=dead_unit_fraction(activations[0]),
            )
        ]
    )
    assert ledger.read("diagnostics").num_rows == 1
