"""Seeding and the memory cap."""

from __future__ import annotations

import os

import pytest
import torch

from valmorph.core.compute import ComputeCapError, apply_compute_cap
from valmorph.core.seed import seed_everything


def test_seed_repeats_cpu_tensor() -> None:
    seed_everything(7)
    first = torch.rand(6)
    seed_everything(7)
    second = torch.rand(6)
    assert torch.equal(first, second)


def test_cpu_cap_checks_policy_without_an_allocator() -> None:
    assert apply_compute_cap(torch.device("cpu"), 0.2) == 0.2


def test_fraction_above_ceiling_is_refused() -> None:
    with pytest.raises(ComputeCapError):
        apply_compute_cap(torch.device("cpu"), 0.5)


def test_override_allows_a_higher_fraction() -> None:
    assert apply_compute_cap(torch.device("cpu"), 0.5, allow_high_memory=True) == 0.5


def test_mps_cap_sets_the_watermark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    apply_compute_cap(torch.device("mps"), 0.2)
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.2"
    assert os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.2"
