"""Edit controllers. The gate decides whether a proposal is kept."""

from valmorph.controllers.l0 import train_with_silent_prune, train_with_widen_trials
from valmorph.controllers.random_controller import choose_edit, train_with_random_edits

__all__ = [
    "choose_edit",
    "train_with_random_edits",
    "train_with_silent_prune",
    "train_with_widen_trials",
]
