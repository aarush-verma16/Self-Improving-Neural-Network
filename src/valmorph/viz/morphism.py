"""Plot the output change after function-preserving edits. One scalar per edit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from valmorph.viz.theme import COLOR_CYCLE, apply_matplotlib_style


def plot_morphism_error(labels: list[str], errors: list[float]) -> go.Figure:
    """Interactive max-absolute output change. ``errors[i]`` is one finite float."""

    figure = go.Figure(
        data=[
            go.Scatter(
                x=list(range(len(errors))),
                y=errors,
                mode="markers+lines",
                marker={"size": 10, "color": COLOR_CYCLE[2]},
                text=labels,
                name="max abs change",
                hovertemplate="%{text}<br>%{y:.3e}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        template="plotly_white",
        title="output change after function-preserving edits",
        xaxis_title="edit",
        yaxis_title="max |Δ output|",
        font={"family": "Arial", "size": 13},
    )
    return figure


def save_morphism_error_png(labels: list[str], errors: list[float], path: Path) -> None:
    """Static twin of ``plot_morphism_error``."""

    apply_matplotlib_style()
    figure, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.axhline(0.0, color="#000000", linewidth=0.6)
    ax.plot(range(len(errors)), errors, marker="o", color=COLOR_CYCLE[2])
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_title("output change after function-preserving edits")
    ax.set_ylabel("max |Δ output|")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
