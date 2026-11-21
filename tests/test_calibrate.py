import numpy as np
import pytest

from cdpbpc.calibrate import conformal_rank, conformal_threshold, p_miss_band


def test_conformal_rank():
    assert conformal_rank(20, 0.05) == 20
    assert conformal_rank(100, 0.05) == 96
    assert conformal_rank(99, 0.01) == 99  # 1 % needs n >= 99
    assert conformal_rank(98, 0.01) == 99
    with pytest.raises(ValueError):
        conformal_rank(10, 0)


def test_threshold_is_the_kth_smallest_score():
    scores = np.arange(1, 101, dtype=float)[::-1]  # 100 scores, unsorted
    assert conformal_threshold(scores, 0.05) == 96.0
    assert conformal_threshold(np.arange(98.0), 0.01) == float("inf")


def test_realized_p_miss_follows_the_beta_band():
    rng = np.random.default_rng(0)
    n, alpha, reps = 100, 0.05, 1000
    realized = np.empty(reps)
    for r in range(reps):
        tau = conformal_threshold(rng.standard_normal(n), alpha)
        realized[r] = np.mean(rng.standard_normal(5000) > tau)
    lo, hi = p_miss_band(n, alpha, level=0.90)
    assert realized.mean() == pytest.approx(alpha, abs=0.005)  # E = (n + 1 - k)/(n + 1) <= alpha
    assert 0.86 < np.mean((realized >= lo) & (realized <= hi)) < 0.94


def test_band_values_of_methods_md():
    lo, hi = p_miss_band(20, 0.05)
    assert (round(lo, 3), round(hi, 3)) == (0.003, 0.139)
    lo, hi = p_miss_band(100, 0.05)
    assert (round(lo, 3), round(hi, 3)) == (0.020, 0.089)
    lo, hi = p_miss_band(500, 0.01)
    assert (round(lo, 3), round(hi, 3)) == (0.004, 0.018)
