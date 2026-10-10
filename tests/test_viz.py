"""Figure provenance. Synthetic output is labeled and not citable."""

from __future__ import annotations

from pathlib import Path

import pytest

from valmorph.core.synthetic import write_synthetic_demo
from valmorph.viz.provenance import ProvenanceError, assert_citable, write_provenance
from valmorph.viz.timeline import plot_gate_timeline, save_gate_timeline_png


def test_synthetic_timeline_is_labeled_and_not_citable(tmp_path: Path) -> None:
    ledger = write_synthetic_demo(tmp_path)
    table = ledger.read("gate_decisions")
    figure = plot_gate_timeline(table, synthetic=True)
    layout = figure.to_dict()["layout"]
    title = layout["title"]
    text = title["text"] if isinstance(title, dict) else str(title)
    assert text.startswith("SYNTHETIC")
    png = tmp_path / "timeline.png"
    save_gate_timeline_png(table, png, synthetic=True)
    assert png.is_file()
    manifest = write_provenance(
        tmp_path / "timeline.provenance.json",
        figure="timeline",
        ledgers=[ledger],
        config_hash="synthetic-demo",
        seeds=[0],
    )
    assert manifest["synthetic"] is True
    with pytest.raises(ProvenanceError):
        assert_citable(manifest)
