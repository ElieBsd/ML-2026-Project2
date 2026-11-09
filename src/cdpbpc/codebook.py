"""Pattern codebook P_b(omega) of the binary pattern-based channel (BPC) model.

Tutt (2025), Sec. 4.3 and Alg. 1: the flip probability of a symbol depends only
on its 3x3 template neighbourhood omega.
"""

import numpy as np

from cdpbpc.patterns import N_PATTERNS, interior, pattern_index


def flips(t: np.ndarray, t_hat: np.ndarray) -> np.ndarray:
    """Flip map delta = 1{t_hat != t}. t, t_hat: (..., L, L). Returns uint8."""
    return (np.asarray(t) != np.asarray(t_hat)).astype(np.uint8)


def pattern_counts(t: np.ndarray, t_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Occurrences n and flips k of each pattern, summed over all given images.

    t, t_hat: (L, L) or (N, L, L), interior symbols only.
    Returns n, k: (512,) int64.
    """
    idx = pattern_index(t).ravel()
    d = interior(flips(t, t_hat)).ravel()
    n = np.bincount(idx, minlength=N_PATTERNS)
    k = np.bincount(idx, weights=d, minlength=N_PATTERNS).astype(np.int64)
    return n, k


def estimate(n: np.ndarray, k: np.ndarray, smoothing: str = "jeffreys") -> np.ndarray:
    """P_b from counts. 'jeffreys': (k + 0.5) / (n + 1); 'raw': k / n (NaN if n = 0)."""
    n = np.asarray(n, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    if smoothing == "jeffreys":
        return (k + 0.5) / (n + 1.0)
    if smoothing == "raw":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, k / n, np.nan)
    raise ValueError(f"unknown smoothing {smoothing!r}")


def fit_codebook(t: np.ndarray, t_hat: np.ndarray, smoothing: str = "jeffreys") -> np.ndarray:
    """Codebook P_b: (512,) from originals. t, t_hat: (N, L, L) or (L, L)."""
    n, k = pattern_counts(t, t_hat)
    return estimate(n, k, smoothing)


def sample_codebook(t: np.ndarray, t_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-image codebook P_b^y = D / L (NaN where L = 0), with raw counts D and L.

    t, t_hat: (N, L, L). Returns three (N, 512) arrays: P, D, L.
    """
    t = np.asarray(t)
    t_hat = np.asarray(t_hat)
    if t.ndim == 2:
        t, t_hat = t[None], t_hat[None]
    N = t.shape[0]
    idx = pattern_index(t).reshape(N, -1).astype(np.int64)
    d = interior(flips(t, t_hat)).reshape(N, -1)
    flat = idx + N_PATTERNS * np.arange(N)[:, None]
    Lw = np.bincount(flat.ravel(), minlength=N * N_PATTERNS).reshape(N, N_PATTERNS)
    Dw = np.bincount(flat.ravel(), weights=d.ravel(), minlength=N * N_PATTERNS)
    Dw = Dw.reshape(N, N_PATTERNS).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(Lw > 0, Dw / Lw, np.nan)
    return P, Dw, Lw


def predicted_flip_prob(t: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """p_ij = P_b(omega_ij) for interior symbols. t: (..., L, L) -> (..., L-2, L-2)."""
    return np.asarray(codebook)[pattern_index(t)]
