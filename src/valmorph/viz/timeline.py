"""Gate-timeline figures from a ledger's ``gate_decisions`` table."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pyarrow as pa

from valmorph.viz.theme import COLOR_CYCLE, apply_matplotlib_style


def _columns(table: pa.Table) -> tuple[list[int], list[bool], list[float], list[str]]:
    steps = [int(value) for value in table.column("step").to_pylist()]
    accepted = [bool(value) for value in table.column("accepted").to_pylist()]
    gains = [float(value) for value in table.column("validated_gain").to_pylist()]
    return steps, accepted, gains, [str(value) for value in table.column("proposal_id").to_pylist()]


def plot_gate_timeline(table: pa.Table, *, synthetic: bool) -> go.Figure:
    """Interactive validated-gain timeline. One point per gate decision.

    ``table`` is a ``gate_decisions`` table. The title says SYNTHETIC when
    ``synthetic`` is true.
    """

    steps, accepted, gains, proposals = _columns(table)
    colors = [COLOR_CYCLE[0] if flag else COLOR_CYCLE[5] for flag in accepted]
    figure = go.Figure(
        data=[
            go.Scatter(
                x=steps,
                y=gains,
                mode="markers+text",
                marker={"color": colors, "size": 12},
                text=[
                    f"{proposal}<br>{gain:.3g}"
                    for proposal, gain in zip(proposals, gains, strict=True)
                ],
                textposition="top center",
                hovertemplate="step %{x}<br>gain %{y:.4f}<br>%{text}<extra></extra>",
                name="validated gain",
            )
        ]
    )
    prefix = "SYNTHETIC — " if synthetic else ""
    figure.update_layout(
        template="plotly_white",
        title=f"{prefix}validated gain per proposal",
        xaxis_title="step",
        yaxis_title="probe gain",
        font={"family": "Arial", "size": 13},
        colorway=list(COLOR_CYCLE),
    )
    return figure


def save_gate_timeline_png(table: pa.Table, path: Path, *, synthetic: bool) -> None:
    """Static twin of ``plot_gate_timeline``."""

    apply_matplotlib_style()
    steps, accepted, gains, _proposals = _columns(table)
    figure, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.axhline(0.0, color="#000000", linewidth=0.6)
    ax.plot(steps, gains, color=COLOR_CYCLE[1], linewidth=1.0)
    accept_x = [step for step, flag in zip(steps, accepted, strict=True) if flag]
    accept_y = [gain for gain, flag in zip(gains, accepted, strict=True) if flag]
    reject_x = [step for step, flag in zip(steps, accepted, strict=True) if not flag]
    reject_y = [gain for gain, flag in zip(gains, accepted, strict=True) if not flag]
    ax.scatter(accept_x, accept_y, color=COLOR_CYCLE[0], label="accepted", zorder=3)
    ax.scatter(reject_x, reject_y, color=COLOR_CYCLE[5], label="rejected", zorder=3)
    prefix = "SYNTHETIC — " if synthetic else ""
    ax.set_title(f"{prefix}validated gain per proposal")
    ax.set_xlabel("step")
    ax.set_ylabel("probe gain")
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
