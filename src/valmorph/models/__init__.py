"""Substrate networks and function-preserving edits."""

from valmorph.models.edits import identity_deepen, widen
from valmorph.models.mlp import ValmorphMLP

__all__ = ["ValmorphMLP", "identity_deepen", "widen"]
