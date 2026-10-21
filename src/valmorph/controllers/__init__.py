"""The first edit controller: prune a silent unit, then let the gate decide."""

from valmorph.controllers.l0 import train_with_silent_prune, train_with_widen_trials

__all__ = ["train_with_silent_prune", "train_with_widen_trials"]
