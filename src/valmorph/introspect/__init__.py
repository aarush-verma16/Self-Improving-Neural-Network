"""Hidden-unit measurements taken while a network trains."""

from valmorph.introspect.signals import (
    activation_covariance,
    dead_unit_fraction,
    hidden_activations,
    layer_gradient_norms,
    mean_off_diagonal,
    most_silent_unit,
    singular_entropy_rank,
)

__all__ = [
    "activation_covariance",
    "dead_unit_fraction",
    "hidden_activations",
    "layer_gradient_norms",
    "mean_off_diagonal",
    "most_silent_unit",
    "singular_entropy_rank",
]
