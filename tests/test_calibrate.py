import numpy as np
import pytest

from cdpbpc.calibrate import (
    analytic_threshold,
    conformal_rank,
    conformal_threshold,
    cross_conformal_thresholds,
    mean_shift_threshold,
    p_miss_band,
    recalibrate,
)


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


# --- cross-conformal threshold -------------------------------------------------------


def test_cross_conformal_uses_the_other_half():
    rng = np.random.default_rng(1)
    scores = rng.standard_normal(400)
    in_b = np.arange(400) % 2 == 1
    tau_a, tau_b = cross_conformal_thresholds(scores, in_b, 0.05)
    assert tau_a == conformal_threshold(scores[in_b], 0.05)
    assert tau_b == conformal_threshold(scores[~in_b], 0.05)


def test_cross_conformal_hand_example():
    # half a = 1..20, half b = 101..120; k = ceil(0.95 * 21) = 20, the largest score.
    scores = np.r_[np.arange(1.0, 21.0), np.arange(101.0, 121.0)]
    in_b = np.r_[np.zeros(20, bool), np.ones(20, bool)]
    assert cross_conformal_thresholds(scores, in_b, 0.05) == (120.0, 20.0)


def test_cross_conformal_holds_the_level_on_exchangeable_halves():
    rng = np.random.default_rng(2)
    n, alpha, reps = 200, 0.05, 1000
    in_b = np.arange(n) % 2 == 1
    realized = np.empty(reps)
    for r in range(reps):
        s = rng.standard_normal(n)
        tau_a, tau_b = cross_conformal_thresholds(s, in_b, alpha)
        realized[r] = np.mean(s > np.where(in_b, tau_b, tau_a))
    assert realized.mean() == pytest.approx(alpha, abs=0.005)


def test_cross_conformal_misses_the_level_when_halves_differ():
    # The guarantee needs exchangeable halves: a shifted half b is rejected too often.
    rng = np.random.default_rng(3)
    s = np.r_[rng.standard_normal(5000), rng.standard_normal(5000) + 0.5]
    in_b = np.r_[np.zeros(5000, bool), np.ones(5000, bool)]
    tau_a, tau_b = cross_conformal_thresholds(s, in_b, 0.05)
    assert np.mean(s[in_b] > tau_b) > 0.10
    assert np.mean(s[~in_b] > tau_a) < 0.02


def test_cross_conformal_rejects_bad_input():
    s = np.arange(10.0)
    with pytest.raises(ValueError):
        cross_conformal_thresholds(s, np.zeros(10, bool), 0.05)  # half b empty
    with pytest.raises(ValueError):
        cross_conformal_thresholds(s, np.zeros(9, bool), 0.05)  # wrong length
    with pytest.raises(ValueError):
        cross_conformal_thresholds(s, np.arange(10) % 3, 0.05)  # not a two-way split


# --- recalibration on K originals of the target ---------------------------------------


def test_recalibration_draws_and_evaluates_on_the_rest():
    rng = np.random.default_rng(4)
    s = rng.standard_normal(300)
    rec = recalibrate(s, 50, 0.05, np.random.default_rng(0), n_draws=20)
    assert rec.tau.shape == rec.p_miss.shape == (20,)
    assert rec.drawn.shape == (20, 50)
    for r in range(20):
        idx = rec.drawn[r]
        assert len(np.unique(idx)) == 50
        rest = np.setdiff1d(np.arange(300), idx)
        assert rec.tau[r] == conformal_threshold(s[idx], 0.05)
        assert rec.p_miss[r] == np.mean(s[rest] > rec.tau[r])


def test_recalibration_is_deterministic_for_a_seed():
    s = np.random.default_rng(5).standard_normal(200)
    a = recalibrate(s, 20, 0.05, np.random.default_rng(7))
    b = recalibrate(s, 20, 0.05, np.random.default_rng(7))
    assert np.array_equal(a.drawn, b.drawn) and np.array_equal(a.p_miss, b.p_miss)
    assert a.p_miss.shape == (50,)  # 50 draws by default


def test_recalibration_follows_the_beta_band():
    s = np.random.default_rng(6).standard_normal(20000)
    K, alpha = 100, 0.05
    rec = recalibrate(s, K, alpha, np.random.default_rng(8), n_draws=1000)
    lo, hi = p_miss_band(K, alpha, level=0.90)
    assert rec.p_miss.mean() == pytest.approx(alpha, abs=0.005)
    assert 0.86 < np.mean((rec.p_miss >= lo) & (rec.p_miss <= hi)) < 0.94


def test_recalibration_rejects_bad_k():
    s = np.arange(50.0)
    for K in (0, 50, 60):  # needs 1 <= K < n so that originals remain to judge
        with pytest.raises(ValueError):
            recalibrate(s, K, 0.05, np.random.default_rng(0))


# --- comparison points -----------------------------------------------------------------


def test_mean_shift_moves_the_reference_threshold_by_the_mean_gap():
    ref = np.arange(1.0, 101.0)  # conformal threshold at 5 %: 96
    target = np.array([10.0, 20.0])  # mean 15, reference mean 50.5
    assert mean_shift_threshold(ref, target, 0.05) == pytest.approx(96.0 - 50.5 + 15.0)


def test_mean_shift_fails_when_the_spread_changes():
    # Same mean, twice the spread: the shifted threshold rejects far more than alpha.
    rng = np.random.default_rng(9)
    ref, target = rng.standard_normal(5000), 2 * rng.standard_normal(5000)
    tau = mean_shift_threshold(ref, target[:100], 0.05)
    assert np.mean(target[100:] > tau) > 0.15


def test_analytic_threshold_is_the_normal_quantile():
    assert analytic_threshold(0.05) == pytest.approx(1.6449, abs=1e-4)
    assert analytic_threshold(0.01) == pytest.approx(2.3263, abs=1e-4)
