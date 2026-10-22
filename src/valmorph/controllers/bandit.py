"""L1 edit choice. Two arms, prune and widen. The rule is ADR-011.

The unit inside the chosen arm is a uniform draw. The reward is the gate's
validated gain, and a lower gain is better.
"""

from __future__ import annotations

import torch

from valmorph.models import ValmorphMLP


class EditBandit:
    """Mean validated gain for prune and for widen.

    Untried arms go first, prune before widen. Every fourth choice after that
    takes the arm with fewer pulls. The rest take the lower mean. A tie keeps
    prune. A layer of width 1 can only widen.
    """

    def __init__(self) -> None:
        self.total = {"prune": 0.0, "widen": 0.0}
        self.count = {"prune": 0, "widen": 0}
        self.pulls = 0

    def choose(self, net: ValmorphMLP, generator: torch.Generator) -> tuple[str, int]:
        """Return the edit kind and a unit index in the first hidden layer."""

        width = int(net.linear(0).out_features)
        arms = ("widen",) if width <= 1 else ("prune", "widen")
        untried = [arm for arm in arms if self.count[arm] == 0]
        if untried:
            kind = untried[0]
        elif self.pulls % 4 == 0:
            kind = min(arms, key=lambda arm: (self.count[arm], arm))
        else:
            kind = min(arms, key=lambda arm: (self.total[arm] / self.count[arm], arm))
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
