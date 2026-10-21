"""The widen controller's matched copy receives extra steps and does not widen."""

from __future__ import annotations

from pathlib import Path

import torch

from valmorph.controllers import train_with_widen_trials
from valmorph.core.ledger import LEDGER_VERSION, Ledger, RunMeta
from valmorph.gate import clone_net
from valmorph.models import ValmorphMLP


def _ledger(tmp_path: Path) -> Ledger:
    return Ledger.create(
        tmp_path,
        RunMeta(
            ledger_version=LEDGER_VERSION,
            run_id="widen",
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
