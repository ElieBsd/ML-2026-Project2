"""Authentication scores. All scores are oriented so that large = suspect."""

import numpy as np


def pcor(t: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Minus the Pearson correlation between the template upsampled k times and the grey image.

    t: (L, L) binary, 1 = white; y: (kL, kL) grey levels. The only score computed on grey
    levels rather than on the binarized image. NaN if t or y is constant.
    """
    t_up = np.repeat(np.repeat(np.asarray(t, dtype=np.float64), k, axis=0), k, axis=1)
    y = np.asarray(y, dtype=np.float64)
    if t_up.shape != y.shape:
        raise ValueError(f"template {np.shape(t)} upsampled x{k} does not match image {y.shape}")
    tc = t_up - t_up.mean()
    yc = y - y.mean()
    den = np.sqrt((tc * tc).sum() * (yc * yc).sum())
    if den == 0:
        return float("nan")
    return float(-(tc * yc).sum() / den)
