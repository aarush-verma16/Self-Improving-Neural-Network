"""Image-task loaders. MNIST files stay in ``data/`` and are not committed."""

from valmorph.benchmarks.mnist import apply_permutation, pixel_permutation, read_idx

__all__ = ["apply_permutation", "pixel_permutation", "read_idx"]
