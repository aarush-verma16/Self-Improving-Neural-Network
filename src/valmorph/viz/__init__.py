"""Interactive and publication figures."""

from valmorph.viz.diagnostics import (
    plot_covariance_heatmap,
    plot_diagnostic_traces,
    save_diagnostic_traces_png,
)
from valmorph.viz.morphism import plot_morphism_error, save_morphism_error_png
from valmorph.viz.provenance import assert_citable, write_provenance
from valmorph.viz.timeline import plot_gate_timeline, save_gate_timeline_png
from valmorph.viz.training import plot_fit, plot_loss_curve, save_loss_curve_png

__all__ = [
    "assert_citable",
    "plot_covariance_heatmap",
    "plot_diagnostic_traces",
    "plot_fit",
    "plot_gate_timeline",
    "plot_loss_curve",
    "plot_morphism_error",
    "save_diagnostic_traces_png",
    "save_gate_timeline_png",
    "save_loss_curve_png",
    "save_morphism_error_png",
    "write_provenance",
]
