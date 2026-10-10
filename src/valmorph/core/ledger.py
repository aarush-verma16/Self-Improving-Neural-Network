"""Typed Parquet event ledger.

A run is a directory ``{root}/{run_id}/`` with ``run.json`` and one Parquet
file per table. Every row is one scalar event. There is no batch dimension.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LEDGER_VERSION = "1"

SPLITS = frozenset({"train", "probe", "val", "test"})
CONTROLLERS = frozenset({"L0", "L1", "L2", "random"})
EDIT_TYPES = frozenset(
    {
        "widen",
        "identity_deepen",
        "prune",
        "prune_regrow",
        "lr_change",
        "regularization_change",
        "consolidate",
        "freeze",
    }
)
RELATIONS = frozenset({"split", "identity", "regrow", "prune", "freeze"})
FINDING_KINDS = frozenset({"observation", "negative", "claim", "method"})
TABLES = (
    "step_metrics",
    "diagnostics",
    "edit_proposals",
    "gate_decisions",
    "lineage",
    "findings",
)

SCHEMAS: dict[str, pa.Schema] = {
    "step_metrics": pa.schema(
        [
            ("run_id", pa.string()),
            ("step", pa.int64()),
            ("split", pa.string()),
            ("name", pa.string()),
            ("value", pa.float64()),
        ]
    ),
    "diagnostics": pa.schema(
        [
            ("run_id", pa.string()),
            ("step", pa.int64()),
            ("layer", pa.string()),
            ("name", pa.string()),
            ("value", pa.float64()),
        ]
    ),
    "edit_proposals": pa.schema(
        [
            ("run_id", pa.string()),
            ("proposal_id", pa.string()),
            ("step", pa.int64()),
            ("controller", pa.string()),
            ("edit_type", pa.string()),
            ("target", pa.string()),
            ("params_json", pa.string()),
            ("parent_proposal_id", pa.string()),
        ]
    ),
    "gate_decisions": pa.schema(
        [
            ("run_id", pa.string()),
            ("proposal_id", pa.string()),
            ("step", pa.int64()),
            ("accepted", pa.bool_()),
            ("probe_before", pa.float64()),
            ("probe_after", pa.float64()),
            ("validated_gain", pa.float64()),
            ("reason", pa.string()),
        ]
    ),
    "lineage": pa.schema(
        [
            ("run_id", pa.string()),
            ("step", pa.int64()),
            ("parent_unit", pa.string()),
            ("child_unit", pa.string()),
            ("relation", pa.string()),
        ]
    ),
    "findings": pa.schema(
        [
            ("run_id", pa.string()),
            ("finding_id", pa.string()),
            ("step", pa.int64()),
            ("kind", pa.string()),
            ("statement", pa.string()),
            ("evidence_ref", pa.string()),
        ]
    ),
}


class LedgerError(ValueError):
    """A row, schema, or run header is invalid."""


class MixedDeviceError(RuntimeError):
    """Two runs that must not be pooled were compared."""


@dataclass(frozen=True)
class RunMeta:
    """Header stored in ``run.json``. One machine, one config, one seed."""

    ledger_version: str
    run_id: str
    created_utc: str
    synthetic: bool
    experiment: str
    seed: int
    config_hash: str
    config: dict[str, object]
    git_commit: str
    device_type: str
    device_name: str
    device_index: int | None
    capability: str | None
    torch_version: str
    cuda_version: str | None
    python_version: str
    platform: str
    libraries: dict[str, str]


@dataclass(frozen=True)
class StepMetric:
    """One scalar metric. ``split`` is train, probe, val, or test."""

    step: int
    split: str
    name: str
    value: float


@dataclass(frozen=True)
class Diagnostic:
    """One scalar diagnostic. ``layer`` is empty when the value is global."""

    step: int
    layer: str
    name: str
    value: float


@dataclass(frozen=True)
class EditProposal:
    """One proposed edit, whether or not the gate later accepts it."""

    proposal_id: str
    step: int
    controller: str
    edit_type: str
    target: str
    params_json: str
    parent_proposal_id: str = ""


@dataclass(frozen=True)
class GateDecision:
    """Probe-set outcome for one proposal. Rejections are stored."""

    proposal_id: str
    step: int
    accepted: bool
    probe_before: float
    probe_after: float
    validated_gain: float
    reason: str


@dataclass(frozen=True)
class LineageEdge:
    """One parent-to-child unit link created by an accepted edit."""

    step: int
    parent_unit: str
    child_unit: str
    relation: str


@dataclass(frozen=True)
class Finding:
    """A recorded observation, negative result, claim, or method note.

    ``step`` is -1 when the finding is about the whole run.
    """

    finding_id: str
    kind: str
    statement: str
    evidence_ref: str
    step: int = -1


def _require_finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise LedgerError(f"{label} must be finite")
    return value


def _require_vocab(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed:
        raise LedgerError(f"{label} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _require_json(payload: str) -> str:
    try:
        json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LedgerError("params_json is not valid JSON") from exc
    return payload


class Ledger:
    """Writer and reader for one run directory."""

    def __init__(self, path: Path, meta: RunMeta) -> None:
        self.path = path
        self.meta = meta

    @classmethod
    def create(cls, root: Path, meta: RunMeta) -> Ledger:
        """Create a new run directory. Refuses to overwrite."""

        if meta.ledger_version != LEDGER_VERSION:
            raise LedgerError(f"unsupported ledger version {meta.ledger_version}")
        path = root / meta.run_id
        if path.exists():
            raise LedgerError(f"run already exists: {path}")
        path.mkdir(parents=True)
        ledger = cls(path, meta)
        ledger.path.joinpath("run.json").write_text(
            json.dumps(asdict(meta), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ledger

    @classmethod
    def reopen(cls, root: Path, run_id: str) -> Ledger:
        """Open an existing run for further appends."""

        path = root / run_id
        raw = json.loads(path.joinpath("run.json").read_text(encoding="utf-8"))
        meta = RunMeta(**raw)
        if meta.run_id != run_id:
            raise LedgerError("run.json run_id does not match the directory")
        if meta.ledger_version != LEDGER_VERSION:
            raise LedgerError(f"unsupported ledger version {meta.ledger_version}")
        return cls(path, meta)

    def append_step_metrics(self, rows: list[StepMetric]) -> None:
        """Append scalar metrics. ``value`` is one finite float per row."""

        payload = []
        for row in rows:
            _require_vocab(row.split, SPLITS, "split")
            payload.append(
                {
                    "run_id": self.meta.run_id,
                    "step": row.step,
                    "split": row.split,
                    "name": row.name,
                    "value": _require_finite(row.value, "metric"),
                }
            )
        self._append("step_metrics", payload)

    def append_diagnostics(self, rows: list[Diagnostic]) -> None:
        """Append scalar diagnostics. ``layer`` may be an empty string."""

        payload = []
        for row in rows:
            payload.append(
                {
                    "run_id": self.meta.run_id,
                    "step": row.step,
                    "layer": row.layer,
                    "name": row.name,
                    "value": _require_finite(row.value, "diagnostic"),
                }
            )
        self._append("diagnostics", payload)

    def append_proposals(self, rows: list[EditProposal]) -> None:
        """Append proposals. Rejected proposals are written here too."""

        payload = []
        for row in rows:
            payload.append(
                {
                    "run_id": self.meta.run_id,
                    "proposal_id": row.proposal_id,
                    "step": row.step,
                    "controller": _require_vocab(row.controller, CONTROLLERS, "controller"),
                    "edit_type": _require_vocab(row.edit_type, EDIT_TYPES, "edit_type"),
                    "target": row.target,
                    "params_json": _require_json(row.params_json),
                    "parent_proposal_id": row.parent_proposal_id,
                }
            )
        self._append("edit_proposals", payload)

    def append_decisions(self, rows: list[GateDecision]) -> None:
        """Append gate decisions, accepted and rejected."""

        payload = []
        for row in rows:
            before = _require_finite(row.probe_before, "probe_before")
            after = _require_finite(row.probe_after, "probe_after")
            gain = _require_finite(row.validated_gain, "validated_gain")
            if abs((after - before) - gain) > 1e-6:
                raise LedgerError("validated_gain must equal probe_after - probe_before")
            payload.append(
                {
                    "run_id": self.meta.run_id,
                    "proposal_id": row.proposal_id,
                    "step": row.step,
                    "accepted": row.accepted,
                    "probe_before": before,
                    "probe_after": after,
                    "validated_gain": gain,
                    "reason": row.reason,
                }
            )
        self._append("gate_decisions", payload)

    def append_lineage(self, rows: list[LineageEdge]) -> None:
        """Append unit-lineage edges."""

        payload = []
        for row in rows:
            payload.append(
                {
                    "run_id": self.meta.run_id,
                    "step": row.step,
                    "parent_unit": row.parent_unit,
                    "child_unit": row.child_unit,
                    "relation": _require_vocab(row.relation, RELATIONS, "relation"),
                }
            )
        self._append("lineage", payload)

    def append_findings(self, rows: list[Finding]) -> None:
        """Append findings. Claims still need a citable figure or table."""

        payload = []
        for row in rows:
            if not row.statement.strip():
                raise LedgerError("finding statement is empty")
            payload.append(
                {
                    "run_id": self.meta.run_id,
                    "finding_id": row.finding_id,
                    "step": row.step,
                    "kind": _require_vocab(row.kind, FINDING_KINDS, "kind"),
                    "statement": row.statement,
                    "evidence_ref": row.evidence_ref,
                }
            )
        self._append("findings", payload)

    def read(self, table: str) -> pa.Table:
        """Read one table. Missing files return an empty table with the schema."""

        if table not in SCHEMAS:
            raise LedgerError(f"unknown table {table}")
        path = self.path / f"{table}.parquet"
        if not path.exists():
            return SCHEMAS[table].empty_table()
        loaded = pq.read_table(path)
        if list(loaded.schema.names) != list(SCHEMAS[table].names):
            raise LedgerError(f"{table} schema does not match ledger version {LEDGER_VERSION}")
        return loaded

    def content_hash(self) -> str:
        """SHA-256 of the header plus each table's bytes, in table order."""

        digest = hashlib.sha256()
        digest.update(self.path.joinpath("run.json").read_bytes())
        for name in TABLES:
            digest.update(name.encode("utf-8"))
            path = self.path / f"{name}.parquet"
            digest.update(path.read_bytes() if path.exists() else b"-")
        return digest.hexdigest()

    def _append(self, table: str, rows: list[dict[str, object]]) -> None:
        if not rows:
            raise LedgerError("refusing to append an empty batch")
        schema = SCHEMAS[table]
        incoming = pa.Table.from_pylist(rows, schema=schema)
        path = self.path / f"{table}.parquet"
        if path.exists():
            previous = pq.read_table(path)
            if list(previous.schema.names) != list(schema.names):
                raise LedgerError(f"{table} schema mismatch on append")
            incoming = pa.concat_tables([previous, incoming])
        pq.write_table(incoming, path)


def assert_same_machine(ledgers: list[Ledger], *, allow_synthetic: bool = False) -> None:
    """Raise if runs differ in device, torch build, or platform.

    Synthetic runs are refused unless ``allow_synthetic`` is true, and even
    then they cannot be mixed with real runs.
    """

    if not ledgers:
        raise MixedDeviceError("no ledgers to compare")
    if not allow_synthetic and any(item.meta.synthetic for item in ledgers):
        raise MixedDeviceError("synthetic ledger cannot enter an evidence pool")
    synthetic_flags = {item.meta.synthetic for item in ledgers}
    if allow_synthetic and True in synthetic_flags and False in synthetic_flags:
        raise MixedDeviceError("synthetic and real ledgers cannot be pooled")
    keys = {
        (
            item.meta.device_type,
            item.meta.device_name,
            item.meta.torch_version,
            item.meta.platform,
        )
        for item in ledgers
    }
    if len(keys) != 1:
        raise MixedDeviceError(f"runs span multiple machines: {sorted(keys)}")


def make_run_id(experiment: str, seed: int, config_hash: str, now: datetime | None = None) -> str:
    """Filesystem-safe run id. The timestamp is UTC."""

    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in experiment).strip("-")
    return f"{safe}-s{seed}-{config_hash[:12]}-{stamp}"
