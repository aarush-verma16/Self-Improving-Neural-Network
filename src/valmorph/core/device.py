"""Device selection. Never hard-codes cuda, mps, or cpu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch


class DeviceError(RuntimeError):
    """The requested device cannot be used on this machine."""


@dataclass(frozen=True)
class DeviceInfo:
    """Identity of the device a run actually used.

    ``capability`` is ``major.minor`` for CUDA and None otherwise.
    """

    type: str
    name: str
    index: int | None
    capability: str | None


def resolve_device(preference: str = "auto") -> torch.device:
    """Return a device for ``preference``.

    ``auto`` picks CUDA, then MPS, then CPU. An explicit preference never
    downgrades: if ``cuda`` or ``mps`` was asked for and is missing, this
    raises ``DeviceError``.
    """

    pref = preference.strip().lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise DeviceError("cuda was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise DeviceError("mps was requested but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise DeviceError(f"unknown device preference: {preference}")


def cuda_index(device: torch.device) -> int:
    """CUDA device index. ``torch.device('cuda')`` leaves index unset; use 0."""

    raw = cast("int | None", device.index)
    return 0 if raw is None else raw


def describe_device(device: torch.device) -> DeviceInfo:
    """Name, index, and compute capability for ``device``."""

    if device.type == "cuda":
        index = cuda_index(device)
        name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        return DeviceInfo(type="cuda", name=name, index=index, capability=f"{major}.{minor}")
    if device.type == "mps":
        return DeviceInfo(type="mps", name="Apple MPS", index=None, capability=None)
    if device.type == "cpu":
        return DeviceInfo(type="cpu", name="cpu", index=None, capability=None)
    raise DeviceError(f"unsupported device type: {device.type}")


def verify_matmul(device: torch.device) -> list[list[float]]:
    """Run a 2x2 identity matmul on ``device`` and return the CPU result.

    The expected value is ``[[1, 2], [3, 4]]``. A failure here means the
    selected backend did not execute a real matmul.
    """

    eye = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    result = value @ eye
    cpu = result.detach().cpu()
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    if not torch.allclose(cpu, expected):
        raise DeviceError(f"matmul on {device} returned {cpu.tolist()}")
    rows: list[list[float]] = cpu.tolist()
    return rows
