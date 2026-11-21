"""Draws under the BPC model on given templates: null reference of the diagnostics and tests.

Under the BPC, each interior symbol of a template flips independently with the
probability P_b(omega) of its 3x3 pattern (Tutt 2025, Sec. 4.3); the outer ring is never
judged and is left as it is. Three levels of detail, all with the same law:

    simulate_reads        n binarized reads of each template (positions);
    simulate_flip_counts  K = number of flips per position over n reads, drawn directly
                          as Binomial(n, p), without storing the reads;
    simulate_counts       per-image pattern counts D ~ Binomial(L, p), for statistics
                          that only use counts.

The null reference redoes the whole computation, codebook included, on
such draws (200 of them in experiments/c1a_noise.py).
"""

import numpy as np

from cdpbpc.patterns import N_PATTERNS, interior, pattern_index


def _check(codebook: np.ndarray) -> np.ndarray:
    codebook = np.asarray(codebook, dtype=np.float64)
    if codebook.shape != (N_PATTERNS,):
        raise ValueError(f"expected a codebook of {N_PATTERNS} probabilities, got {codebook.shape}")
    if not ((codebook >= 0) & (codebook <= 1)).all():
        raise ValueError("codebook probabilities must be in [0, 1]")
    return codebook


def simulate_reads(
    t: np.ndarray, codebook: np.ndarray, n: int, rng: np.random.Generator, chunk: int = 64
) -> np.ndarray:
    """n independent reads of each template under the BPC.

    t: (N, side, side) binary templates, 1 = white; codebook: (512,) flip probabilities.
    Returns (n, N, side, side) uint8: each read is t with interior symbols flipped.
    """
    codebook = _check(codebook)
    t = np.asarray(t, dtype=np.uint8)
    if n < 1:
        raise ValueError("need at least one read")
    reads = np.repeat(t[None], n, axis=0)
    for start in range(0, len(t), chunk):
        p = codebook[pattern_index(t[start : start + chunk])].astype(np.float32)
        for r in range(n):
            flip = rng.random(p.shape, dtype=np.float32) < p
            interior(reads[r, start : start + chunk])[...] ^= flip.astype(np.uint8)
    return reads


def simulate_flip_counts(
    t: np.ndarray, codebook: np.ndarray, n: int, rng: np.random.Generator, chunk: int = 64
) -> np.ndarray:
    """Flip counts over n reads under the BPC, the law of diagnostics.flip_counts.

    t: (N, side, side); returns (N, side, side) uint8 with K ~ Binomial(n, P_b(omega))
    on interior symbols and 0 on the outer ring.
    """
    codebook = _check(codebook)
    t = np.asarray(t)
    if not 1 <= n <= 255:
        raise ValueError("need between 1 and 255 reads")
    K = np.zeros(t.shape, dtype=np.uint8)
    for start in range(0, len(t), chunk):
        p = codebook[pattern_index(t[start : start + chunk])]
        interior(K[start : start + chunk])[...] = rng.binomial(n, p)
    return K


def simulate_counts(L: np.ndarray, codebook: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pattern flip counts of one BPC read of each image: D ~ Binomial(L, P_b).

    L: (N, 512) pattern occurrences of each image's template (counts.image_counts).
    Returns D: (N, 512) int64, the law of image_counts(t, simulate_reads(t, codebook, 1)[0]).
    """
    codebook = _check(codebook)
    return rng.binomial(np.asarray(L, dtype=np.int64), codebook)
