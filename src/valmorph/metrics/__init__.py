"""Continual-learning summaries of a task-by-task accuracy matrix."""

from valmorph.metrics.scores import (
    average_accuracy,
    backward_transfer,
    forgetting,
    learning_accuracy,
)

__all__ = ["average_accuracy", "backward_transfer", "forgetting", "learning_accuracy"]
