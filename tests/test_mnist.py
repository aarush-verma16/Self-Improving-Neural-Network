"""IDX parsing and pixel permutation, without downloading MNIST."""

from __future__ import annotations

import gzip
from pathlib import Path

import torch

from valmorph.benchmarks.mnist import apply_permutation, pixel_permutation, read_idx


def _write_idx(path: Path, magic: int, array: torch.Tensor) -> None:
    payload = bytearray()
    payload += magic.to_bytes(4, "big")
    payload += array.shape[0].to_bytes(4, "big")
    if magic == 2051:
        payload += array.shape[1].to_bytes(4, "big")
        payload += array.shape[2].to_bytes(4, "big")
    payload += array.numpy().tobytes()
    with gzip.open(path, "wb") as handle:
        handle.write(payload)


def test_read_idx_roundtrip_and_permutation(tmp_path: Path) -> None:
    images = torch.arange(2 * 28 * 28, dtype=torch.uint8).reshape(2, 28, 28)
    labels = torch.tensor([3, 9], dtype=torch.uint8)
    _write_idx(tmp_path / "images.gz", 2051, images)
    _write_idx(tmp_path / "labels.gz", 2049, labels)
    assert torch.equal(read_idx(tmp_path / "images.gz"), images)
    assert torch.equal(read_idx(tmp_path / "labels.gz"), labels)
    flat = images.to(dtype=torch.float32).reshape(2, -1)
    permutation = pixel_permutation(0)
    shuffled = apply_permutation(flat, permutation)
    assert shuffled.shape == flat.shape
    assert not torch.equal(shuffled[0], flat[0])
    assert torch.equal(shuffled[0, torch.argsort(permutation)], flat[0])
