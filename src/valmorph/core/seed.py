"""Seeding. Best-effort determinism, never bit-exact across devices."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, str]:
    """Seed Python, NumPy, and PyTorch.

    When ``deterministic`` is true, cuDNN benchmarking is off and
    deterministic algorithms are requested with ``warn_only=True``.
    CUDA atomics, MPS, and different GPU architectures can still disagree.
    Do not claim bit-exact results across machines.
    """

    if seed < 0:
        raise ValueError("seed must be >= 0")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    notes = [
        "Seeds cover random, numpy, and torch. They do not make CUDA, MPS, "
        "and CPU bit-exact with each other."
    ]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        notes.append("deterministic algorithms are warn_only; unsupported ops still run")
    return {"seed": str(seed), "nondeterminism": " ".join(notes)}
