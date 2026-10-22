"""Edit controllers. The gate decides whether a proposal is kept."""

from valmorph.controllers.bandit import EditBandit
from valmorph.controllers.l0 import train_with_silent_prune, train_with_widen_trials
from valmorph.controllers.magnitude import train_with_magnitude_prune
from valmorph.controllers.random_controller import choose_edit, train_with_random_edits
from valmorph.controllers.schedule import train_with_scheduled_widen
from valmorph.controllers.sequence import train_across_tasks

__all__ = [
    "EditBandit",
    "choose_edit",
    "train_across_tasks",
    "train_with_magnitude_prune",
    "train_with_random_edits",
    "train_with_scheduled_widen",
    "train_with_silent_prune",
    "train_with_widen_trials",
]
