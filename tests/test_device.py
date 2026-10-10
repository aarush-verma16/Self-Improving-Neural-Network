"""Device resolution never downgrades an explicit request."""

from __future__ import annotations

import pytest
import torch

from valmorph.core.device import DeviceError, resolve_device, verify_matmul


def test_cpu_matmul() -> None:
    result = verify_matmul(torch.device("cpu"))
    assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_explicit_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_auto_falls_through_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


def test_auto_uses_mps_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "mps"


def test_explicit_cuda_does_not_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceError):
        resolve_device("cuda")


def test_unknown_preference() -> None:
    with pytest.raises(DeviceError):
        resolve_device("tpu")
