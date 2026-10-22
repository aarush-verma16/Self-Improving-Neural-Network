"""L1 edit choice. Prune, widen, or halve the first layer's learning rate.

The rule is ADR-012. The unit inside a structural arm is a uniform draw.
The reward is the gate's validated gain, and a lower gain is better.
"""

from __future__ import annotations

import torch

from valmorph.models import ValmorphMLP

ARM_ORDER = ("prune", "widen", "lr_change")


class EditBandit:
    """Mean validated gain for prune, widen, and a learning-rate halving.

    Untried arms go first, in ``ARM_ORDER``. Every fourth choice after that
    takes the arm with fewer pulls. The rest take the lower mean. A tie keeps
    the earlier arm. A layer of width 1 cannot be pruned. ``lr_change`` halves
    the first layer's scale and ignores the drawn unit.
    """

    def __init__(self) -> None:
        self.total = {arm: 0.0 for arm in ARM_ORDER}
        self.count = {arm: 0 for arm in ARM_ORDER}
        self.pulls = 0

    def choose(self, net: ValmorphMLP, generator: torch.Generator) -> tuple[str, int]:
        """Return the edit kind and a unit index in the first hidden layer."""

        width = int(net.linear(0).out_features)
        arms = tuple(arm for arm in ARM_ORDER if arm != "prune" or width > 1)
        untried = [arm for arm in arms if self.count[arm] == 0]
        if untried:
            kind = untried[0]
        elif self.pulls % 4 == 0:
            kind = min(arms, key=lambda arm: (self.count[arm], ARM_ORDER.index(arm)))
        else:
            kind = min(
                arms,
                key=lambda arm: (self.total[arm] / self.count[arm], ARM_ORDER.index(arm)),
            )
        self.pulls += 1
        unit = int(torch.randint(0, width, (1,), generator=generator))
        return kind, unit

    def observe(self, kind: str, gain: float) -> None:
        """Record one gate gain for ``kind``."""

        if kind not in self.total:
            raise ValueError(f"unknown arm {kind}")
        self.total[kind] += gain
        self.count[kind] += 1

    def mean(self, kind: str) -> float | None:
        """Mean gain for ``kind``, or ``None`` when that arm is untried."""

        if self.count[kind] == 0:
            return None
        return self.total[kind] / self.count[kind]
