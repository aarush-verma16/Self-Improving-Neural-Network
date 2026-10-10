"""Small feedforward network whose hidden units can be edited.

``x`` is ``[batch, in_features]``. Hidden layers are linear then ReLU.
The output layer is linear. Unit ids stay on the module so lineage can
name a unit after later edits.
"""

from __future__ import annotations

import torch
from torch import nn


class SubstrateError(ValueError):
    """The requested widths cannot form a network."""


class ValmorphMLP(nn.Module):
    """MLP with stable output-unit ids on every linear layer.

    ``widths`` has at least three entries: input, one hidden width, output.
    ``layers[i].weight`` has shape ``[widths[i + 1], widths[i]]`` at construction.
    Later edits change those widths. ``use_relu[i]`` is true when ReLU follows
    layer ``i``. An inserted identity layer has ``use_relu`` false.
    """

    def __init__(self, widths: list[int]) -> None:
        super().__init__()
        if len(widths) < 3 or any(width < 1 for width in widths):
            raise SubstrateError("widths need an input, a hidden layer, and an output")
        self.layers = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)
        )
        self.use_relu: list[bool] = [True] * (len(self.layers) - 1) + [False]
        self._next_unit = 0
        self.unit_ids: list[list[str]] = []
        for module in self.layers:
            if not isinstance(module, nn.Linear):
                raise SubstrateError("substrate layers are linear maps")
            count = int(module.out_features)
            self.unit_ids.append([self._alloc_unit() for _ in range(count)])

    def linear(self, index: int) -> nn.Linear:
        """Return layer ``index`` as a linear map."""

        module = self.layers[index]
        if not isinstance(module, nn.Linear):
            raise SubstrateError("substrate layers are linear maps")
        return module

    def _alloc_unit(self) -> str:
        unit_id = f"u{self._next_unit}"
        self._next_unit += 1
        return unit_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``[batch, out_features]``."""

        hidden = x
        for index in range(len(self.layers)):
            hidden = self.linear(index)(hidden)
            if self.use_relu[index]:
                hidden = torch.relu(hidden)
        return hidden

    def parameter_count(self) -> int:
        """Number of trainable parameters, including frozen ones."""

        return sum(parameter.numel() for parameter in self.parameters())
