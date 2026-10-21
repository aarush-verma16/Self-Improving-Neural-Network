"""Permuted MNIST for a short smoke run.

Images are cached under ``data/mnist``, which is gitignored. Each image is a
float tensor of 784 pixels in ``[0, 1]``. A permutation is a vector of 784
indices applied to every image in a task.
"""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import torch

MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "train_x",
    "train-labels-idx1-ubyte.gz": "train_y",
    "t10k-images-idx3-ubyte.gz": "test_x",
    "t10k-labels-idx1-ubyte.gz": "test_y",
}
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"


def read_idx(path: Path) -> torch.Tensor:
    """Read one gzipped IDX file into a tensor.

    Images come back as ``uint8`` with shape ``[n, 28, 28]``. Labels come back
    as ``uint8`` with shape ``[n]``.
    """

    with gzip.open(path, "rb") as handle:
        payload = handle.read()
    magic = int.from_bytes(payload[0:4], "big")
    count = int.from_bytes(payload[4:8], "big")
    if magic == 2049:
        return torch.frombuffer(bytearray(payload[8 : 8 + count]), dtype=torch.uint8).clone()
    if magic == 2051:
        rows = int.from_bytes(payload[8:12], "big")
        cols = int.from_bytes(payload[12:16], "big")
        body = payload[16 : 16 + count * rows * cols]
        images = torch.frombuffer(bytearray(body), dtype=torch.uint8).clone()
        return images.reshape(count, rows, cols)
    raise ValueError(f"unrecognized IDX magic {magic} in {path}")


def download_mnist(root: Path) -> Path:
    """Download the four MNIST archives into ``root`` if they are not already there."""

    root.mkdir(parents=True, exist_ok=True)
    for name in MNIST_FILES:
        destination = root / name
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        url = f"{MNIST_MIRROR}/{name}"
        urllib.request.urlretrieve(url, destination)
    return root


def load_split(root: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pixels ``[n, 784]`` in ``[0, 1]`` and integer labels ``[n]``.

    ``split`` is ``train`` or ``test``.
    """

    if split == "train":
        images = read_idx(root / "train-images-idx3-ubyte.gz")
        labels = read_idx(root / "train-labels-idx1-ubyte.gz")
    elif split == "test":
        images = read_idx(root / "t10k-images-idx3-ubyte.gz")
        labels = read_idx(root / "t10k-labels-idx1-ubyte.gz")
    else:
        raise ValueError("split must be train or test")
    flat = images.to(dtype=torch.float32).reshape(images.shape[0], -1) / 255.0
    return flat, labels.to(dtype=torch.long)


def pixel_permutation(seed: int, pixels: int = 784) -> torch.Tensor:
    """A fixed permutation of pixel indices, shape ``[pixels]``."""

    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(pixels, generator=generator)


def apply_permutation(images: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Permute the last dimension of ``images``. Shape is unchanged."""

    return images[:, permutation]
