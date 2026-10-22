"""Diagnostic traces and a covariance heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from valmorph.viz.theme import COLOR_CYCLE, apply_matplotlib_style

_PLOTLY_DASH = {"solid": "solid", "dash": "dash", "dot": "dot", "dashdot": "dashdot"}
_MPL_DASH = {"solid": "solid", "dash": "dashed", "dot": "dotted", "dashdot": "dashdot"}


def plot_diagnostic_traces(
    series: dict[str, list[float]],
    *,
    title: str = "diagnostics during training",
    y_title: str = "value",
    marks: list[int] | None = None,
    dashes: dict[str, str] | None = None,
    mark_label: str = "kept",
) -> go.Figure:
    """One line per diagnostic. Each list is one scalar per recorded step.

    ``marks`` are step numbers drawn as dotted vertical lines. ``mark_label``
    is their annotation. ``dashes`` maps a series name to ``solid``, ``dash``,
    ``dot``, or ``dashdot``.
    """

    figure = go.Figure()
    style = dashes or {}
    for index, (name, values) in enumerate(series.items()):
        figure.add_trace(
            go.Scatter(
                y=values,
                mode="lines",
                name=name,
                line={
                    "color": COLOR_CYCLE[index % len(COLOR_CYCLE)],
                    "width": 2,
                    "dash": _PLOTLY_DASH[style.get(name, "solid")],
                },
            )
        )
    for step in marks or []:
        figure.add_vline(
            x=step,
            line_dash="dot",
            line_color="#000000",
            annotation_text=mark_label,
            annotation_position="top",
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
    marks: list[int] | None = None,
    dashes: dict[str, str] | None = None,
    mark_label: str = "kept",
) -> None:
    """Static copy of the diagnostic traces. ``marks`` are dotted vertical lines."""

    apply_matplotlib_style()
    figure, ax = plt.subplots(figsize=(6.0, 3.2))
    style = dashes or {}
    for name, values in series.items():
        ax.plot(values, label=name, linestyle=_MPL_DASH[style.get(name, "solid")])
    labeled = False
    for step in marks or []:
        ax.axvline(
            step,
            color="#000000",
            linestyle=":",
            linewidth=1,
            label=None if labeled else mark_label,
        )
        labeled = True
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(y_title)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
