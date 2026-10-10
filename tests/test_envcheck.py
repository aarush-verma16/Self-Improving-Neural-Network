"""CPU environment record. The CUDA check is the ``gpu`` marker."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from valmorph.core.envcheck import collect_environment, write_environment


def test_cpu_environment_matmul(tmp_path: Path) -> None:
    record = collect_environment(
        "cpu",
        memory_fraction=0.2,
        allow_high_memory=False,
        seed=0,
    )
    assert record["matmul_ok"] is True
    assert record["device_type"] == "cpu"
    assert record["matmul_result"] == [[1.0, 2.0], [3.0, 4.0]]
    path = tmp_path / "env.json"
    write_environment(record, path)
    assert path.is_file()


@pytest.mark.gpu
def test_cuda_environment_matmul() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    record = collect_environment(
        "cuda",
        memory_fraction=0.2,
        allow_high_memory=False,
        seed=0,
    )
    assert record["matmul_ok"] is True
    assert record["device_type"] == "cuda"
