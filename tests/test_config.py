"""Config load and hash."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from omegaconf.errors import ConfigKeyError

from valmorph.core.config import RunConfig, config_hash, load_run_config


def test_setup_yaml_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_run_config(root / "configs" / "setup.yaml")
    assert cfg.experiment == "setup"
    assert cfg.compute.memory_fraction == 0.2
    assert config_hash(cfg) == config_hash(load_run_config(root / "configs" / "setup.yaml"))


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("experiment: x\nnot_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ConfigKeyError):
        load_run_config(path)


def test_negative_seed_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("seed: -1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        load_run_config(path)


@given(st.integers(min_value=0, max_value=10_000), st.text(min_size=1, max_size=20))
def test_hash_is_stable(seed: int, experiment: str) -> None:
    cfg = RunConfig(experiment=experiment, seed=seed)
    assert config_hash(cfg) == config_hash(cfg)
    other = RunConfig(experiment=experiment, seed=seed + 1)
    assert config_hash(cfg) != config_hash(other)
