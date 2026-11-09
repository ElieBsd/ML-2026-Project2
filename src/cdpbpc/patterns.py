"""Vectorized 3x3 pattern index.

Bits are read row by row from top-left to bottom-right; top-left is the most
significant bit (2^8), the center is 2^4, bottom-right is 2^0. Same convention
as Tutt (2025), Fig. 4.6.
"""

import numpy as np

N_PATTERNS = 512


def pattern_index(t: np.ndarray, pad: bool = False) -> np.ndarray:
    """3x3 neighbourhood index of each symbol.

    t: (..., L, L) binary, 1 = white. Returns int16 of shape (..., L-2, L-2)
    (interior symbols only), or (..., L, L) if pad=True (border padded with white).
    """
    t = np.asarray(t).astype(np.int16)
    if pad:
        width = [(0, 0)] * (t.ndim - 2) + [(1, 1), (1, 1)]
        t = np.pad(t, width, constant_values=1)
    L = t.shape[-1]
    if L < 3 or t.shape[-2] != L:
        raise ValueError(f"expected square templates with side >= 3, got {t.shape}")
    idx = np.zeros(t.shape[:-2] + (L - 2, L - 2), dtype=np.int16)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            w = 8 - 3 * (a + 1) - (b + 1)
            idx += t[..., 1 + a : L - 1 + a, 1 + b : L - 1 + b] << w
    return idx


def interior(x: np.ndarray) -> np.ndarray:
    """Drop the outer ring of symbols: (..., L, L) -> (..., L-2, L-2)."""
    return x[..., 1:-1, 1:-1]
