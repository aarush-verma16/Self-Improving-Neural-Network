"""Typed run configs. The hash is the identity of a configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf


class ConfigError(ValueError):
    """A config file does not match ``RunConfig``."""


@dataclass
class ComputeConfig:
    """Memory cap. Fractions above 0.25 require ``allow_high_memory``."""

    memory_fraction: float = 0.2
    allow_high_memory: bool = False


@dataclass
class RunConfig:
    """One experiment's resolved settings.

    ``device`` is ``auto``, ``cuda``, ``mps``, or ``cpu``.
    """

    experiment: str = "setup"
    seed: int = 0
    device: str = "auto"
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    ledger_dir: str = "results/ledgers"
    notes: str = ""


def config_payload(cfg: RunConfig) -> dict[str, object]:
    """JSON-ready mapping. Key order is not significant; ``config_hash`` sorts."""

    return cast(dict[str, object], asdict(cfg))


def config_hash(cfg: RunConfig) -> str:
    """First 16 hex chars of the canonical JSON SHA-256."""

    raw = json.dumps(config_payload(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_run_config(path: Path) -> RunConfig:
    """Load YAML onto the ``RunConfig`` schema. Unknown keys raise."""

    base = OmegaConf.structured(RunConfig)
    loaded = OmegaConf.load(path)
    merged = OmegaConf.merge(base, loaded)
    obj = OmegaConf.to_object(merged)
    if not isinstance(obj, RunConfig):
        raise ConfigError(f"{path} did not resolve to RunConfig")
    if obj.seed < 0:
        raise ConfigError("seed must be >= 0")
    return obj
