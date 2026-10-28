"""Hidden-unit measurements taken while a network trains."""

from valmorph.introspect.signals import (
    activation_covariance,
    dead_unit_fraction,
    hidden_activations,
    layer_gradient_norms,
    linear_cka,
    mean_off_diagonal,
    most_silent_unit,
    per_example_gradient_dispersion,
    representation_drift,
    singular_entropy_rank,
    smallest_magnitude_unit,
)

__all__ = [
    "activation_covariance",
    "dead_unit_fraction",
    "hidden_activations",
    "layer_gradient_norms",
    "linear_cka",
    "mean_off_diagonal",
    "most_silent_unit",
    "per_example_gradient_dispersion",
    "representation_drift",
    "singular_entropy_rank",
    "smallest_magnitude_unit",
]
