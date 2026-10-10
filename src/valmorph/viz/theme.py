"""Colorblind-safe theme shared by Plotly and matplotlib figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

# Okabe-Ito. Black is reserved for text; the cycle starts at orange.
OKABE_ITO: tuple[str, ...] = (
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
)
COLOR_CYCLE: tuple[str, ...] = OKABE_ITO[1:]


def apply_matplotlib_style() -> None:
    """Publication style. Call before drawing a static figure."""

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["axes.prop_cycle"] = cycler(color=list(COLOR_CYCLE))
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["figure.dpi"] = 150
