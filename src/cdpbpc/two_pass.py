"""Flip profile of a fake predicted without any fake: the two passes of a copy.

A copy of a CDP goes through two passes. The attacker estimates the template from a
print (t_hat), then prints that estimate again. Under the BPC (Tutt 2025, Sec. 4.3), the
reprinted symbol at (i, j) differs from t_hat[i, j] with probability P_b(omega_hat_ij),
where omega_hat_ij is the 3x3 pattern of t_hat around (i, j). Measured against the true
template t, the fake is therefore wrong at (i, j) with probability

    q_ij = P_b(omega_hat_ij)        if t_hat[i, j] == t[i, j]
    q_ij = 1 - P_b(omega_hat_ij)    otherwise.

Pooled by the pattern of the true template, as for originals, this gives the predicted
codebook of fakes Q_pred(omega) = mean of q_ij over the positions of pattern omega in t.
P_b is the codebook of originals captured in the same condition as the fakes; no fake is
used. The same model without context is in Phan Ho et al. (2014), eqs. 9-11, and in
Baras and Cayre (2012).

t_hat is given by the caller (an estimated template of the archive, a binarized
original, a vote over originals...). Nothing here produces an image of a fake.

vote_estimate gives one such t_hat from the codebook alone: the estimate of an attacker
who takes a majority vote over many reads of the same print (the averaging attack).
vote_profile is the predicted codebook of the fakes of that attacker.
"""

import numpy as np

from cdpbpc.patterns import N_PATTERNS, interior, pattern_index


def _check_codebook(codebook: np.ndarray) -> np.ndarray:
    codebook = np.asarray(codebook, dtype=np.float64)
    if codebook.shape != (N_PATTERNS,):
        raise ValueError(f"expected a codebook of {N_PATTERNS} probabilities, got {codebook.shape}")
    if not ((codebook >= 0) & (codebook <= 1)).all():
        raise ValueError("codebook probabilities must be in [0, 1]")
    return codebook


def _check_pair(t: np.ndarray, t_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t)
    t_hat = np.asarray(t_hat)
    if t.shape != t_hat.shape:
        raise ValueError(f"t and t_hat differ in shape: {t.shape} and {t_hat.shape}")
    if not (np.isin(t, (0, 1)).all() and np.isin(t_hat, (0, 1)).all()):
        raise ValueError("t and t_hat must be 0/1")
    return t, t_hat


def predicted_flip_prob(t: np.ndarray, t_hat: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """q_ij: probability that the reprinted estimate t_hat is read wrong against t.

    t, t_hat: (..., L, L) binary, 1 = white; codebook: (512,) P_b of originals.
    Returns float64 (..., L-2, L-2), interior symbols only.
    """
    codebook = _check_codebook(codebook)
    t, t_hat = _check_pair(t, t_hat)
    p = codebook[pattern_index(t_hat)]
    same = interior(t == t_hat)
    return np.where(same, p, 1.0 - p)


def predicted_sums(
    t: np.ndarray, t_hat: np.ndarray, codebook: np.ndarray, chunk: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Sum of q and number of positions per pattern of the true template, per template.

    t, t_hat: (N, side, side) binary; t_hat[i] is the estimate of template t[i]. The
    expected flip count of a fake of template i on pattern omega is Sq[i, omega], out of
    L[i, omega] symbols, as D and L for a captured image (counts.image_counts).
    Returns Sq: (N, 512) float64 and L: (N, 512) float64.
    """
    codebook = _check_codebook(codebook)
    t, t_hat = _check_pair(t, t_hat)
    if t.ndim != 3:
        raise ValueError(f"expected (N, side, side) arrays, got {t.shape}")
    Sq = np.zeros((len(t), N_PATTERNS))
    L = np.zeros((len(t), N_PATTERNS))
    for start in range(0, len(t), chunk):
        sl = slice(start, start + chunk)
        idx = pattern_index(t[sl]).astype(np.int64)
        m = len(idx)
        flat = (idx + N_PATTERNS * np.arange(m)[:, None, None]).ravel()
        q = predicted_flip_prob(t[sl], t_hat[sl], codebook).ravel()
        size = m * N_PATTERNS
        Sq[sl] = np.bincount(flat, weights=q, minlength=size).reshape(m, N_PATTERNS)
        L[sl] = np.bincount(flat, minlength=size).reshape(m, N_PATTERNS)
    return Sq, L


def pooled_profile(
    Sq: np.ndarray, L: np.ndarray, weights: np.ndarray | None = None, fill: float = np.nan
) -> np.ndarray:
    """Predicted codebook of fakes Q_pred(omega) = sum of q / sum of L over templates.

    Sq, L: (N, 512) from predicted_sums; weights: (N,) template multiplicities (None = 1).
    Patterns with no position get `fill`. Returns (512,) float64.
    """
    Sq = np.asarray(Sq, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    if Sq.shape != L.shape or Sq.ndim != 2 or Sq.shape[1] != N_PATTERNS:
        raise ValueError(f"expected two (N, {N_PATTERNS}) arrays, got {Sq.shape} and {L.shape}")
    w = np.ones(len(Sq)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(Sq),) or (w < 0).any():
        raise ValueError(f"expected {len(Sq)} non-negative weights, got shape {w.shape}")
    num = w @ Sq
    den = w @ L
    out = np.full(N_PATTERNS, fill, dtype=np.float64)
    seen = den > 0
    out[seen] = num[seen] / den[seen]
    return out


def vote_estimate(t: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Template estimate of an attacker who votes over many reads of a print.

    Under the BPC each read of a print of t gets the symbol at (i, j) wrong with
    probability P_b(omega_ij), omega_ij being the 3x3 pattern of t, independently from
    read to read. A majority vote over m reads is wrong with probability
    P(Binomial(m, P_b) > m / 2), which tends to 1{P_b(omega_ij) > 0.5} as m grows. The
    estimate is therefore t with every interior symbol of P_b > 0.5 flipped; a tie
    (P_b = 0.5) keeps the symbol, and the outer ring, which has no pattern, is kept.

    t: (..., L, L) binary, 1 = white; codebook: (512,) P_b of originals.
    Returns an array like t.
    """
    codebook = _check_codebook(codebook)
    t, _ = _check_pair(t, t)
    if t.ndim < 2:
        raise ValueError(f"expected (..., L, L) templates, got {t.shape}")
    t_hat = t.copy()
    flip = codebook[pattern_index(t)] > 0.5
    interior(t_hat)[...] ^= flip.astype(t_hat.dtype)
    return t_hat


def vote_profile(
    t: np.ndarray, codebook: np.ndarray, weights: np.ndarray | None = None, chunk: int = 64
) -> np.ndarray:
    """Predicted codebook of the fakes of the voting attacker, Q_vote(omega).

    t: (N, side, side) templates whose prints the codebook was learned on; codebook:
    (512,) P_b of those originals; weights: (N,) multiplicities (None = 1). The
    estimate of each template (vote_estimate) goes through the two passes
    (predicted_sums) and is pooled by pattern of the true template (pooled_profile).
    A pattern absent from t gets Q = P_b: it then adds nothing to a likelihood ratio of
    Q against P_b. Returns (512,) float64.
    """
    codebook = _check_codebook(codebook)
    t = np.asarray(t)
    t_hat = vote_estimate(t, codebook)
    Sq, L = predicted_sums(t, t_hat, codebook, chunk)
    Q = pooled_profile(Sq, L, weights)
    absent = np.isnan(Q)
    Q[absent] = codebook[absent]
    return Q
