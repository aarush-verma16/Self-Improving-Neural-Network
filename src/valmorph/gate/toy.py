"""A one-dimensional sine fit used by the gate notebook. Not a benchmark."""

from __future__ import annotations

import torch

from valmorph.models.mlp import ValmorphMLP


def sine_split(
    seed: int = 0,
    n_train: int = 128,
    n_probe: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return train and probe pairs. ``x`` is ``[n, 1]`` in ``[-2, 2]``, ``y`` is ``sin(x)``."""

    generator = torch.Generator().manual_seed(seed)
    train_x = torch.rand(n_train, 1, generator=generator) * 4 - 2
    probe_x = torch.rand(n_probe, 1, generator=generator) * 4 - 2
    return train_x, torch.sin(train_x), probe_x, torch.sin(probe_x)


def prediction_grid(net: ValmorphMLP, points: int = 200) -> tuple[list[float], list[float]]:
    """Evenly spaced inputs and the net's outputs, for a line plot."""

    grid = torch.linspace(-2, 2, points).unsqueeze(1)
    with torch.no_grad():
        predicted = net(grid).squeeze(1)
    return grid.squeeze(1).tolist(), predicted.tolist()
