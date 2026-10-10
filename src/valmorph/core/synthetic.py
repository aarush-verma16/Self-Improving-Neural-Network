"""Synthetic ledger used by the setup notebook. Not evidence."""

from __future__ import annotations

import sys
from pathlib import Path

from valmorph import __version__
from valmorph.core.ledger import (
    LEDGER_VERSION,
    Diagnostic,
    EditProposal,
    Finding,
    GateDecision,
    Ledger,
    LineageEdge,
    RunMeta,
    StepMetric,
    make_run_id,
)
from valmorph.core.runtime import git_commit, library_versions, package_version, platform_string


def write_synthetic_demo(root: Path, *, seed: int = 0) -> Ledger:
    """Write a tiny labeled demo ledger under ``root``.

    ``synthetic`` is true and ``device_type`` is ``synthetic``, so
    ``assert_same_machine`` will not pool it with a real run.
    """

    config_hash = "synthetic-demo"
    run_id = make_run_id("synthetic-demo", seed, config_hash)
    meta = RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc=run_id.rsplit("-", 1)[-1],
        synthetic=True,
        experiment="synthetic-demo",
        seed=seed,
        config_hash=config_hash,
        config={"experiment": "synthetic-demo", "seed": seed},
        git_commit=git_commit(),
        device_type="synthetic",
        device_name="synthetic",
        device_index=None,
        capability=None,
        torch_version=package_version("torch"),
        cuda_version=None,
        python_version=sys.version,
        platform=platform_string(),
        libraries={**library_versions(), "valmorph": __version__},
    )
    ledger = Ledger.create(root, meta)
    metrics: list[StepMetric] = []
    diagnostics: list[Diagnostic] = []
    proposals: list[EditProposal] = []
    decisions: list[GateDecision] = []
    lineage: list[LineageEdge] = []
    for step in range(12):
        acc = 0.4 + 0.03 * step
        metrics.append(StepMetric(step=step, split="probe", name="accuracy", value=acc))
        diagnostics.append(
            Diagnostic(
                step=step, layer="hidden", name="dead_neuron_fraction", value=0.2 - 0.01 * step
            )
        )
        accepted = step % 3 != 0
        gain = 0.01 * (step + 1) if accepted else -0.005
        proposals.append(
            EditProposal(
                proposal_id=f"p{step}",
                step=step,
                controller="random" if step % 2 else "L0",
                edit_type="widen" if accepted else "prune",
                target="hidden",
                params_json="{}",
            )
        )
        decisions.append(
            GateDecision(
                proposal_id=f"p{step}",
                step=step,
                accepted=accepted,
                probe_before=acc,
                probe_after=acc + gain,
                validated_gain=gain,
                reason="synthetic",
            )
        )
        if accepted:
            lineage.append(
                LineageEdge(
                    step=step,
                    parent_unit=f"u{step}",
                    child_unit=f"u{step}-a",
                    relation="split",
                )
            )
    ledger.append_step_metrics(metrics)
    ledger.append_diagnostics(diagnostics)
    ledger.append_proposals(proposals)
    ledger.append_decisions(decisions)
    ledger.append_lineage(lineage)
    ledger.append_findings(
        [
            Finding(
                finding_id="f0",
                kind="method",
                statement="This ledger is synthetic and is not evidence.",
                evidence_ref="notebooks/00_setup.py",
            )
        ]
    )
    return ledger
