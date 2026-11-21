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


# Scores from pattern counts. D, L: (N, 512) flips and occurrences per image (counts.py);
# codebook: (512,) flip probabilities P_b of originals. Each returns (N,) float64.

P_CLIP = 1e-4  # p is kept in [1e-4, 1 - 1e-4] inside scores


def hamming(D: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Share of interior symbols read wrong. The trivial baseline."""
    return np.asarray(D).sum(axis=1) / np.asarray(L).sum(axis=1)


def centered_z(D: np.ndarray, L: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Flips minus their BPC expectation, over the BPC standard deviation.

    About N(0, 1) for originals if the BPC model holds.
    """
    p = np.clip(np.asarray(codebook, dtype=np.float64), P_CLIP, 1 - P_CLIP)
    D = np.asarray(D, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    return (D - L * p).sum(axis=1) / np.sqrt((L * p * (1 - p)).sum(axis=1))


def masked_hamming(
    D: np.ndarray, L: np.ndarray, codebook: np.ndarray, mu: float = 0.01
) -> np.ndarray:
    """Hamming distance on the reliable patterns only, those with P_b <= mu (thesis p. 104).

    NaN for an image with no reliable symbol.
    """
    mask = np.asarray(codebook) <= mu
    flips = np.asarray(D)[:, mask].sum(axis=1).astype(np.float64)
    symbols = np.asarray(L)[:, mask].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(symbols > 0, flips / symbols, np.nan)


# Reliability classes: class b if edge_b <= P_b < edge_{b+1}, last edge
# included. 8 classes for the profiles and the oracle; 4 for binned_mahalanobis.
CLASS_EDGES_8 = (0.0, 0.01, 0.05, 0.15, 0.30, 0.50, 0.70, 0.90, 1.0)
CLASS_EDGES_4 = (0.0, 0.05, 0.30, 0.70, 1.0)


def reliability_class(p: np.ndarray, edges=CLASS_EDGES_8) -> np.ndarray:
    """Reliability class of each flip probability p: ints in 0..len(edges)-2."""
    return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)


def binned_z(D: np.ndarray, L: np.ndarray, codebook: np.ndarray, edges=CLASS_EDGES_8) -> np.ndarray:
    """centered_z inside each reliability class.

    z_b = sum over the patterns of class b of (D - L p), over sqrt(sum of L p (1 - p)); 0
    when the image has no symbol of class b. Classes follow the codebook P_b, p is kept in
    [1e-4, 1 - 1e-4]. Returns (N, len(edges) - 1).
    """
    codebook = np.asarray(codebook, dtype=np.float64)
    onehot = reliability_class(codebook, edges)[:, None] == np.arange(len(edges) - 1)[None]
    p = np.clip(codebook, P_CLIP, 1 - P_CLIP)
    D = np.asarray(D, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    num = (D - L * p) @ onehot
    var = (L * p * (1 - p)) @ onehot
    return np.where(var > 0, num / np.sqrt(np.where(var > 0, var, 1.0)), 0.0)


LRT_CLIP = 1e-3  # p and q are kept in [1e-3, 1 - 1e-3] in likelihood ratios


def lrt(D: np.ndarray, L: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Log-likelihood ratio of fakes (flip probabilities q) against originals (p).

    sum over patterns of D log(q / p) + (L - D) log((1 - q) / (1 - p)); large = suspect.
    p: codebook of originals; q: flip probabilities of fakes, learned on fakes (lrt_plugin,
    oracle) or predicted without fakes (lrt_vote, cdpbpc.two_pass). Both (512,), kept in
    [1e-3, 1 - 1e-3].
    """
    p = np.clip(np.asarray(p, dtype=np.float64), LRT_CLIP, 1 - LRT_CLIP)
    q = np.clip(np.asarray(q, dtype=np.float64), LRT_CLIP, 1 - LRT_CLIP)
    D = np.asarray(D, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    return D @ np.log(q / p) + (L - D) @ np.log((1 - q) / (1 - p))
