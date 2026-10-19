"""Curves the gate notebook shows while a toy fit is running."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from valmorph.viz.theme import COLOR_CYCLE, apply_matplotlib_style


def plot_loss_curve(losses: list[float]) -> go.Figure:
    """Interactive training loss. One finite scalar per step."""

    figure = go.Figure(
        data=[
            go.Scatter(
                y=losses,
                mode="lines",
                line={"color": COLOR_CYCLE[1], "width": 2},
                name="train loss",
            )
        ]
    )
    figure.update_layout(
        template="plotly_white",
        title="training loss",
        xaxis_title="step",
        yaxis_title="mean squared error",
        font={"family": "Arial", "size": 13},
    )
    return figure


def plot_fit(
    x: list[float],
    truth: list[float],
    before: list[float],
    after: list[float],
) -> go.Figure:
    """Input versus target, the net before the gate, and the net after it."""

    figure = go.Figure(
        data=[
            go.Scatter(x=x, y=truth, mode="lines", name="target", line={"color": COLOR_CYCLE[5]}),
            go.Scatter(
                x=x, y=before, mode="lines", name="before gate", line={"color": COLOR_CYCLE[0]}
            ),
            go.Scatter(
                x=x, y=after, mode="lines", name="after gate", line={"color": COLOR_CYCLE[2]}
            ),
        ]
    )
    figure.update_layout(
        template="plotly_white",
        title="toy fit before and after the gate",
        xaxis_title="x",
        yaxis_title="y",
        font={"family": "Arial", "size": 13},
    )
    return figure


def save_loss_curve_png(losses: list[float], path: Path) -> None:
    """Static copy of the training-loss curve."""

    apply_matplotlib_style()
    figure, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(losses, color=COLOR_CYCLE[1])
    ax.set_title("training loss")
    ax.set_xlabel("step")
    ax.set_ylabel("mean squared error")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
