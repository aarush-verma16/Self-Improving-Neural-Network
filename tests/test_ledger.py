"""Ledger round-trips, vocabulary, and the device pool."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from valmorph.core.ledger import (
    LEDGER_VERSION,
    EditProposal,
    Finding,
    GateDecision,
    Ledger,
    LedgerError,
    MixedDeviceError,
    RunMeta,
    StepMetric,
    assert_same_machine,
)
from valmorph.core.synthetic import write_synthetic_demo


def _meta(run_id: str, *, device_type: str = "cpu", synthetic: bool = False) -> RunMeta:
    return RunMeta(
        ledger_version=LEDGER_VERSION,
        run_id=run_id,
        created_utc="2026-10-10T00:00:00Z",
        synthetic=synthetic,
        experiment="unit",
        seed=0,
        config_hash="abc",
        config={"experiment": "unit"},
        git_commit="no-commit",
        device_type=device_type,
        device_name=device_type,
        device_index=None,
        capability=None,
        torch_version="test",
        cuda_version=None,
        python_version="3.12",
        platform="test-platform",
        libraries={"valmorph": "0.1.0"},
    )


def test_round_trip_and_reopen(tmp_path: Path) -> None:
    ledger = Ledger.create(tmp_path, _meta("run-a"))
    ledger.append_step_metrics([StepMetric(step=1, split="probe", name="accuracy", value=0.5)])
    reopened = Ledger.reopen(tmp_path, "run-a")
    reopened.append_step_metrics([StepMetric(step=2, split="test", name="accuracy", value=0.25)])
    table = reopened.read("step_metrics")
    assert table.num_rows == 2
    assert table.column("value").to_pylist() == [0.5, 0.25]
    assert reopened.read("findings").num_rows == 0


def test_rejects_bad_split_and_inconsistent_gain(tmp_path: Path) -> None:
    ledger = Ledger.create(tmp_path, _meta("run-b"))
    with pytest.raises(LedgerError):
        ledger.append_step_metrics(
            [StepMetric(step=0, split="holdout", name="accuracy", value=1.0)]
        )
    with pytest.raises(LedgerError):
        ledger.append_decisions(
            [
                GateDecision(
                    proposal_id="p",
                    step=0,
                    accepted=True,
                    probe_before=0.2,
                    probe_after=0.3,
                    validated_gain=0.5,
                    reason="mismatch",
                )
            ]
        )


def test_proposal_and_finding_vocab(tmp_path: Path) -> None:
    ledger = Ledger.create(tmp_path, _meta("run-c"))
    ledger.append_proposals(
        [
            EditProposal(
                proposal_id="p0",
                step=0,
                controller="random",
                edit_type="widen",
                target="h",
                params_json='{"n": 1}',
            )
        ]
    )
    with pytest.raises(LedgerError):
        ledger.append_proposals(
            [
                EditProposal(
                    proposal_id="p1",
                    step=1,
                    controller="random",
                    edit_type="magic",
                    target="h",
                    params_json="{}",
                )
            ]
        )
    with pytest.raises(LedgerError):
        ledger.append_findings(
            [Finding(finding_id="f", kind="claim", statement="  ", evidence_ref="")]
        )


def test_synthetic_cannot_pool_with_real(tmp_path: Path) -> None:
    demo = write_synthetic_demo(tmp_path)
    real = Ledger.create(tmp_path, _meta("real"))
    with pytest.raises(MixedDeviceError):
        assert_same_machine([demo, real])
    with pytest.raises(MixedDeviceError):
        assert_same_machine([demo])
    assert_same_machine([demo], allow_synthetic=True)
    assert demo.meta.synthetic is True
    assert demo.content_hash() == Ledger.reopen(tmp_path, demo.meta.run_id).content_hash()


def test_device_mismatch(tmp_path: Path) -> None:
    cpu = Ledger.create(tmp_path, _meta("cpu-run", device_type="cpu"))
    cuda = Ledger.create(tmp_path, _meta("cuda-run", device_type="cuda"))
    with pytest.raises(MixedDeviceError):
        assert_same_machine([cpu, cuda])


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=30),
            st.sampled_from(["train", "probe", "val", "test"]),
            st.sampled_from(["accuracy", "loss"]),
            st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=8,
    )
)
def test_metric_roundtrip_property(rows: list[tuple[int, str, str, float]]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ledger = Ledger.create(root, _meta(f"prop-{uuid.uuid4().hex[:8]}"))
        ledger.append_step_metrics(
            [
                StepMetric(step=step, split=split, name=name, value=value)
                for step, split, name, value in rows
            ]
        )
        loaded = ledger.read("step_metrics")
        assert loaded.num_rows == len(rows)
        assert loaded.column("name").to_pylist() == [name for _, _, name, _ in rows]
