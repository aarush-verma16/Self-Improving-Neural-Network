"""Scalars computed from a batch of hidden activations.

``activations`` is ``[batch, units]`` unless a docstring says otherwise.
``linear_cka`` is the linear case of Kornblith et al. 2019. The other
functions are our definitions.
"""

from __future__ import annotations

import math

import torch

from valmorph.models.mlp import ValmorphMLP


def smallest_magnitude_unit(
    net: ValmorphMLP,
    layer: int = 0,
    *,
    skip: set[str] | None = None,
) -> int | None:
    """Index of the hidden unit with the smallest incoming L2 norm.

    The vector is that unit's incoming weight row with its bias appended.
    Ties keep the lowest index. ``skip`` holds unit ids to pass over.
    Returns None when the layer has fewer than two units, or when every
    remaining unit is skipped. This is our definition, not a formula from
    an unopened paper.
    """

    if layer < 0 or layer >= len(net.layers) - 1:
        return None
    linear = net.linear(layer)
    if int(linear.out_features) < 2 or linear.bias is None:
        return None
    rows = torch.cat([linear.weight.detach(), linear.bias.detach().unsqueeze(1)], dim=1)
    blocked = skip or set()
    norms = torch.linalg.vector_norm(rows, dim=1)
    for index in torch.argsort(norms).tolist():
        chosen = int(index)
        if net.unit_ids[layer][chosen] not in blocked:
            return chosen
    return None


def dead_unit_fraction(activations: torch.Tensor, *, eps: float = 1e-6) -> float:
    """Fraction of units that stay silent on this batch.

    A unit is silent when every value in its column has absolute value ``<= eps``.
    """

    if activations.ndim != 2:
        raise ValueError("activations must have shape [batch, units]")
    units = int(activations.shape[1])
    if units == 0:
        return 0.0
    silent = activations.detach().abs().amax(dim=0) <= eps
    return float(silent.to(dtype=torch.float32).mean())


def most_silent_unit(activations: torch.Tensor, *, eps: float = 1e-6) -> int | None:
    """Index of a unit that is silent on this batch, or None.

    ``activations`` has shape ``[batch, units]``. If several units are silent,
    the one with the smallest peak activation is returned.
    """

    if activations.ndim != 2 or int(activations.shape[1]) == 0:
        return None
    peak = activations.detach().abs().amax(dim=0)
    if float(peak.min()) > eps:
        return None
    return int(torch.argmin(peak))


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear centered kernel alignment. Kornblith et al. 2019, equations 3 and 4.

    ``x`` and ``y`` have shape ``[examples, features]`` and the same number of
    examples. Columns are centered, which is the case where their equation 2
    holds. The kernel is the dot product. An orthogonal map of the features,
    or one isotropic scale, leaves the value unchanged. A constant matrix
    returns 0.
    """

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must have shape [examples, features]")
    if int(x.shape[0]) != int(y.shape[0]):
        raise ValueError("x and y must have the same number of examples")
    if int(x.shape[0]) < 2:
        raise ValueError("linear CKA needs at least two examples")
    x_centered = x.detach().to(dtype=torch.float64)
    y_centered = y.detach().to(dtype=torch.float64)
    x_centered = x_centered - x_centered.mean(dim=0, keepdim=True)
    y_centered = y_centered - y_centered.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(y_centered.transpose(0, 1) @ x_centered, ord="fro") ** 2
    left = torch.linalg.matrix_norm(x_centered.transpose(0, 1) @ x_centered, ord="fro")
    right = torch.linalg.matrix_norm(y_centered.transpose(0, 1) @ y_centered, ord="fro")
    denom = left * right
    if float(denom) == 0.0:
        return 0.0
    return float(cross / denom)


def singular_entropy_rank(activations: torch.Tensor) -> float:
    """Exponential entropy of the normalized singular values.

    Equal nonzero singular values that span ``k`` directions return ``k``.
    A zero matrix returns 0. This is not a published CKA number.
    """

    if activations.ndim != 2:
        raise ValueError("activations must have shape [batch, units]")
    if activations.numel() == 0:
        return 0.0
    values = torch.linalg.svdvals(activations.detach().to(dtype=torch.float64))
    kept = values[values > values.max() * 1e-8]
    if int(kept.numel()) == 0:
        return 0.0
    probabilities = kept / kept.sum()
    entropy = -torch.sum(probabilities * torch.log(probabilities))
    return float(torch.exp(entropy))


def activation_covariance(activations: torch.Tensor) -> torch.Tensor:
    """Column covariance, shape ``[units, units]``.

    One batch row is not enough to estimate a covariance, so this returns zeros.
    """

    if activations.ndim != 2:
        raise ValueError("activations must have shape [batch, units]")
    batch, units = int(activations.shape[0]), int(activations.shape[1])
    if batch < 2 or units == 0:
        return torch.zeros(units, units, dtype=torch.float32)
    centered = activations.detach().to(dtype=torch.float64)
    centered = centered - centered.mean(dim=0, keepdim=True)
    covariance = (centered.transpose(0, 1) @ centered) / (batch - 1)
    return covariance.to(dtype=torch.float32)


def hidden_activations(net: ValmorphMLP, x: torch.Tensor) -> list[torch.Tensor]:
    """Post-ReLU tensors. Item ``i`` matches the ``i``-th ReLU layer, shape ``[batch, units]``."""

    hidden = x
    recorded: list[torch.Tensor] = []
    for index in range(len(net.layers)):
        hidden = net.linear(index)(hidden)
        if net.use_relu[index]:
            hidden = torch.relu(hidden)
            recorded.append(hidden)
    return recorded


def layer_gradient_norms(net: ValmorphMLP, x: torch.Tensor, y: torch.Tensor) -> list[float]:
    """L2 norm of each linear layer's weight and bias gradient.

    One mean-squared-error backward. Gradients on ``net`` are cleared afterward.
    The returned list has one float per linear layer.
    """

    net.zero_grad(set_to_none=True)
    loss = torch.mean((net(x) - y) ** 2)
    loss.backward()
    norms: list[float] = []
    for index in range(len(net.layers)):
        layer = net.linear(index)
        weight_grad = layer.weight.grad
        bias_grad = layer.bias.grad
        if weight_grad is None or bias_grad is None:
            norms.append(0.0)
            continue
        total = float(weight_grad.detach().pow(2).sum() + bias_grad.detach().pow(2).sum())
        norms.append(math.sqrt(total))
    net.zero_grad(set_to_none=True)
    return norms


def mean_off_diagonal(covariance: torch.Tensor) -> float:
    """Mean absolute off-diagonal of a square covariance. A scalar."""

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    units = int(covariance.shape[0])
    if units < 2:
        return 0.0
    mask = ~torch.eye(units, dtype=torch.bool, device=covariance.device)
    return float(covariance.detach().abs()[mask].mean())
