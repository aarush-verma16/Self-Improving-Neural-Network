"""Process memory cap. Default is 20% and 25% is the hard ceiling."""

from __future__ import annotations

import os

import torch

from valmorph.core.device import cuda_index

# Shared 16 GB GPU: 0.20 is about 3.3 GB. A 24 GB Mac: 0.20 is about 4.8 GB.
DEFAULT_MEMORY_FRACTION = 0.2
MAX_FRACTION_WITHOUT_OVERRIDE = 0.25


class ComputeCapError(RuntimeError):
    """The requested memory fraction is outside project policy."""


def apply_compute_cap(
    device: torch.device,
    fraction: float = DEFAULT_MEMORY_FRACTION,
    *,
    allow_high_memory: bool = False,
) -> float:
    """Apply ``fraction`` and return it.

    CUDA uses ``set_per_process_memory_fraction``. MPS sets
    ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` and
    ``PYTORCH_MPS_LOW_WATERMARK_RATIO`` to ``fraction`` before the first
    allocation. PyTorch reads both when the MPS allocator starts. The low
    ratio must be at most the high ratio; the default low ratio is 1.4, so
    setting only the high ratio to 0.20 makes the first MPS op fail. CPU
    has no allocator cap; the fraction is still checked so a config cannot
    silently ask for the whole machine. Call this before the first tensor
    on ``device``.
    """

    if fraction <= 0.0 or fraction > 1.0:
        raise ComputeCapError(f"memory fraction must be in (0, 1], got {fraction}")
    if fraction > MAX_FRACTION_WITHOUT_OVERRIDE and not allow_high_memory:
        raise ComputeCapError(
            f"memory fraction {fraction} exceeds {MAX_FRACTION_WITHOUT_OVERRIDE} "
            "without allow_high_memory"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ComputeCapError("cuda memory cap requested but CUDA is not available")
        torch.cuda.set_per_process_memory_fraction(fraction, cuda_index(device))
    elif device.type == "mps":
        # The low watermark is a soft limit and must be <= the high watermark.
        # Leaving it at PyTorch's default (1.4) raises "invalid low watermark ratio".
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(fraction)
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(fraction)
    return fraction
