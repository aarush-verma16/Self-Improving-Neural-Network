"""Provenance manifests for figures. Synthetic figures are not citable."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from valmorph.core.ledger import LEDGER_VERSION, Ledger
from valmorph.core.runtime import git_commit


class ProvenanceError(RuntimeError):
    """A figure cannot be cited."""


def write_provenance(
    path: Path,
    *,
    figure: str,
    ledgers: list[Ledger],
    config_hash: str,
    seeds: list[int],
) -> dict[str, object]:
    """Write a JSON manifest next to an exported figure and return it."""

    synthetic = any(item.meta.synthetic for item in ledgers)
    manifest: dict[str, object] = {
        "figure": figure,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "config_hash": config_hash,
        "seeds": seeds,
        "ledger_version": LEDGER_VERSION,
        "run_ids": [item.meta.run_id for item in ledgers],
        "ledger_hashes": {item.meta.run_id: item.content_hash() for item in ledgers},
        "synthetic": synthetic,
        "device_types": sorted({item.meta.device_type for item in ledgers}),
        "device_names": sorted({item.meta.device_name for item in ledgers}),
        "torch_versions": sorted({item.meta.torch_version for item in ledgers}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def assert_citable(manifest: dict[str, object]) -> None:
    """Refuse a synthetic manifest as support for a paper claim."""

    if manifest.get("synthetic") is True:
        raise ProvenanceError("synthetic figure is not citable")
    if not manifest.get("ledger_hashes"):
        raise ProvenanceError("citable figure is missing ledger hashes")
