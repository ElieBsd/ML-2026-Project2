"""Thresholds set on originals only (the defender has no fakes)."""

import math
from typing import NamedTuple

import numpy as np
from scipy import stats


def conformal_rank(n: int, alpha: float) -> int:
    """k = ceil((1 - alpha)(n + 1)): the threshold is the k-th smallest of n scores."""
    if n < 1 or not 0 < alpha < 1:
        raise ValueError(f"need n >= 1 and 0 < alpha < 1, got n={n}, alpha={alpha}")
    # Rounded first so that float noise in (1 - alpha)(n + 1) cannot push k one step up.
    return math.ceil(round((1 - alpha) * (n + 1), 9))


def conformal_threshold(original_scores: np.ndarray, alpha: float) -> float:
    """Conformal threshold from n scores of calibration originals.

    Guarantees P_miss <= alpha on new originals exchangeable with the calibration ones.
    +inf (never reject) when n is too small for alpha, i.e. when k > n.
    """
    s = np.sort(np.asarray(original_scores, dtype=np.float64))
    k = conformal_rank(len(s), alpha)
    return float("inf") if k > len(s) else float(s[k - 1])


def p_miss_band(n: int, alpha: float, level: float = 0.90) -> tuple[float, float]:
    """Range of the realized P_miss of a conformal threshold, under exchangeability.

    The realized P_miss follows Beta(n + 1 - k, k); returns its central interval.
    """
    k = conformal_rank(n, alpha)
    if k > n:
        return 0.0, 0.0
    a = (1 - level) / 2
    dist = stats.beta(n + 1 - k, k)
    return float(dist.ppf(a)), float(dist.ppf(1 - a))


def cross_conformal_thresholds(
    scores: np.ndarray, in_b: np.ndarray, alpha: float
) -> tuple[float, float]:
    """Cross-conformal thresholds for two halves of templates.

    scores: scores of originals; in_b: True for originals of half b, False for half a.
    Returns (tau_a, tau_b): tau_a is the conformal threshold of the originals of half b
    and is applied to the images of half a, and the other way round. Every image is
    then judged by a threshold that never saw it. The level holds only if the two
    halves are exchangeable.
    """
    s = np.asarray(scores, dtype=np.float64)
    b = np.asarray(in_b)
    if b.shape != s.shape:
        raise ValueError(f"in_b has shape {b.shape}, scores have shape {s.shape}")
    if b.dtype != bool:
        if not np.isin(b, (0, 1)).all():
            raise ValueError("in_b must say for each original whether it is in half b")
        b = b.astype(bool)
    if b.all() or not b.any():
        raise ValueError("both halves need at least one original")
    return conformal_threshold(s[b], alpha), conformal_threshold(s[~b], alpha)


class Recalibration(NamedTuple):
    """One recalibration experiment: one row per random draw of K originals."""

    tau: np.ndarray  # (n_draws,) conformal threshold on the K drawn originals
    p_miss: np.ndarray  # (n_draws,) share of the other originals rejected by tau
    drawn: np.ndarray  # (n_draws, K) indices of the drawn originals


def recalibrate(
    target_scores: np.ndarray,
    K: int,
    alpha: float,
    rng: np.random.Generator,
    n_draws: int = 50,
) -> Recalibration:
    """Conformal threshold redone on K originals of a new condition.

    Each draw takes K originals of the target without replacement, sets the conformal
    threshold on them and measures the realized P_miss on the remaining originals of
    the target. Under exchangeability the realized P_miss follows the Beta band of
    p_miss_band(K, alpha). One score per template: with several images of a template,
    drawn and judged images would share a template.
    """
    s = np.asarray(target_scores, dtype=np.float64)
    n = len(s)
    if not 1 <= K < n:
        raise ValueError(f"need 1 <= K < n so that some originals remain, got K={K}, n={n}")
    drawn = np.stack([rng.choice(n, size=K, replace=False) for _ in range(n_draws)])
    tau = np.empty(n_draws)
    p_miss = np.empty(n_draws)
    for r, idx in enumerate(drawn):
        rest = np.ones(n, dtype=bool)
        rest[idx] = False
        tau[r] = conformal_threshold(s[idx], alpha)
        p_miss[r] = np.mean(s[rest] > tau[r])
    return Recalibration(tau, p_miss, drawn)


def mean_shift_threshold(
    reference_scores: np.ndarray, target_scores: np.ndarray, alpha: float
) -> float:
    """Comparison point only: the reference conformal threshold moved by the mean gap.

    tau = tau_ref - mean(reference) + mean(target). It ignores a change of spread, so it
    does not hold the level when the spread of the scores changes between conditions.
    """
    ref = np.asarray(reference_scores, dtype=np.float64)
    tgt = np.asarray(target_scores, dtype=np.float64)
    return conformal_threshold(ref, alpha) - float(ref.mean()) + float(tgt.mean())


def analytic_threshold(alpha: float) -> float:
    """Comparison point only: z_(1 - alpha), the threshold of centered_z if BPC held exactly."""
    if not 0 < alpha < 1:
        raise ValueError(f"need 0 < alpha < 1, got {alpha}")
    return float(stats.norm.ppf(1 - alpha))
