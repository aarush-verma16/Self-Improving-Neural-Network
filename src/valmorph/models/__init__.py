"""Substrate networks and edits."""

from valmorph.models.edits import identity_deepen, prune, widen
from valmorph.models.mlp import ValmorphMLP

__all__ = ["ValmorphMLP", "identity_deepen", "prune", "widen"]
