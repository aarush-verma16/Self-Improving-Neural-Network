"""A row of digit images, original or permuted."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_digit_row(images: torch.Tensor, path: Path, *, title: str) -> None:
    """Save ``images`` of shape ``[count, 784]`` as a horizontal strip."""

    count = int(images.shape[0])
    figure, axes = plt.subplots(1, count, figsize=(count, 1.4))
    if count == 1:
        axes = [axes]
    for axis, image in zip(axes, images, strict=True):
        axis.imshow(image.reshape(28, 28), cmap="gray_r")
        axis.axis("off")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
