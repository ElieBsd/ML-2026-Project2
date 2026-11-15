"""Thresholds set on originals only (the defender has no fakes)."""

import math

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
