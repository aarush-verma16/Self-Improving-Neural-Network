"""Substrate networks and edits."""

from valmorph.models.edits import freeze, identity_deepen, lr_change, prune, widen
from valmorph.models.mlp import ValmorphMLP

__all__ = ["ValmorphMLP", "freeze", "identity_deepen", "lr_change", "prune", "widen"]
