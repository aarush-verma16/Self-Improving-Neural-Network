"""Diagnostic traces and a covariance heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from valmorph.viz.theme import COLOR_CYCLE, apply_matplotlib_style


def plot_diagnostic_traces(
    series: dict[str, list[float]],
    *,
    title: str = "diagnostics during training",
    y_title: str = "value",
) -> go.Figure:
    """One line per diagnostic. Each list is one finite scalar per recorded step."""

    figure = go.Figure()
    for index, (name, values) in enumerate(series.items()):
        figure.add_trace(
            go.Scatter(
                y=values,
                mode="lines",
                name=name,
                line={"color": COLOR_CYCLE[index % len(COLOR_CYCLE)], "width": 2},
            )
        )
    figure.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="step",
        yaxis_title=y_title,
        font={"family": "Arial", "size": 13},
    )
    return figure


def plot_covariance_heatmap(matrix: list[list[float]]) -> go.Figure:
    """Hidden-unit covariance. ``matrix`` is square, ``[units, units]``."""

    figure = go.Figure(data=[go.Heatmap(z=matrix, colorscale="Viridis")])
    figure.update_layout(
        template="plotly_white",
        title="hidden activation covariance",
        font={"family": "Arial", "size": 13},
    )
    return figure


def save_diagnostic_traces_png(
    series: dict[str, list[float]],
    path: Path,
    *,
    title: str = "diagnostics during training",
    y_title: str = "value",
) -> None:
    """Static copy of the diagnostic traces."""

    apply_matplotlib_style()
    figure, ax = plt.subplots(figsize=(6.0, 3.2))
    for name, values in series.items():
        ax.plot(values, label=name)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(y_title)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
