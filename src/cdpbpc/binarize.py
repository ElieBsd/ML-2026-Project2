"""Binarization of a captured CDP image into an estimated binary template."""

import numpy as np
from skimage.filters import threshold_otsu


def _blocks(y: np.ndarray, k: int) -> np.ndarray:
    """Reshape y: (kH, kW) into (H, W, k*k) blocks of k x k pixels."""
    if y.ndim != 2 or y.shape[0] % k or y.shape[1] % k:
        raise ValueError(f"expected a 2-D image with sides divisible by {k}, got {y.shape}")
    H, W = y.shape[0] // k, y.shape[1] // k
    return y.reshape(H, k, W, k).transpose(0, 2, 1, 3).reshape(H, W, k * k)


def binarize_vote(y: np.ndarray, k: int = 4, threshold: float | None = None) -> np.ndarray:
    """Global Otsu threshold, then majority vote in each k x k block.

    y: (kL, kL) float in [0, 1]. Returns (L, L) uint8, 1 = white.
    A block is white if more than half of its pixels are white. On an exact tie
    (only possible for even k*k, e.g. 8/16) the block mean is compared to the threshold.
    """
    y = np.asarray(y, dtype=np.float32)
    thr = threshold_otsu(y) if threshold is None else threshold
    b = _blocks(y, k)
    n_white = (b > thr).sum(axis=-1)
    half = k * k / 2
    out = n_white > half
    tie = n_white == half
    if tie.any():
        out = np.where(tie, b.mean(axis=-1) > thr, out)
    return out.astype(np.uint8)


def binarize_mean(y: np.ndarray, k: int = 4) -> np.ndarray:
    """Ablation: average each k x k block, then Otsu on the (L, L) image of means.

    y: (kL, kL) float in [0, 1]. Returns (L, L) uint8, 1 = white.
    """
    m = _blocks(np.asarray(y, dtype=np.float32), k).mean(axis=-1)
    return (m > threshold_otsu(m)).astype(np.uint8)


def binarize_otsu(y: np.ndarray) -> np.ndarray:
    """Plain Otsu (k = 1, e.g. Indigo mobile rcod_hist). Returns uint8, 1 = white."""
    y = np.asarray(y, dtype=np.float32)
    return (y > threshold_otsu(y)).astype(np.uint8)
