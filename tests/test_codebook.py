import numpy as np

from cdpbpc.codebook import (
    estimate,
    fit_codebook,
    flips,
    pattern_counts,
    predicted_flip_prob,
    sample_codebook,
)
from cdpbpc.patterns import interior, pattern_index


def _simulate(P_b, n_img, L, rng):
    """Templates and BPC estimates: each interior symbol flips with P_b[omega]."""
    t = rng.integers(0, 2, (n_img, L, L), dtype=np.uint8)
    p = P_b[pattern_index(t)]
    d = (rng.random(p.shape) < p).astype(np.uint8)
    t_hat = t.copy()
    interior(t_hat)[...] ^= d
    return t, t_hat


def test_recovers_known_flip_probabilities():
    rng = np.random.default_rng(0)
    P_b = rng.uniform(0.01, 0.4, 512)
    t, t_hat = _simulate(P_b, 400, 52, rng)
    n, k = pattern_counts(t, t_hat)
    P_hat = estimate(n, k, "raw")
    sd = np.sqrt(P_b * (1 - P_b) / n)
    assert n.sum() == 400 * 50 * 50
    assert (np.abs(P_hat - P_b) < 3 * sd).mean() > 0.99
    np.testing.assert_allclose(fit_codebook(t, t_hat), (k + 0.5) / (n + 1))


def test_jeffreys_and_raw():
    n = np.array([0, 10, 4])
    k = np.array([0, 0, 4])
    np.testing.assert_allclose(estimate(n, k), [0.5, 0.5 / 11, 4.5 / 5])
    raw = estimate(n, k, "raw")
    assert np.isnan(raw[0]) and raw[1] == 0 and raw[2] == 1


def test_no_flips_when_identical():
    t = np.random.default_rng(1).integers(0, 2, (3, 10, 10), dtype=np.uint8)
    n, k = pattern_counts(t, t)
    assert k.sum() == 0 and n.sum() == 3 * 64
    assert flips(t, 1 - t).all()


def test_sample_codebook_sums_to_global_counts():
    rng = np.random.default_rng(2)
    t, t_hat = _simulate(np.full(512, 0.2), 6, 20, rng)
    P, D, Lw = sample_codebook(t, t_hat)
    n, k = pattern_counts(t, t_hat)
    assert P.shape == D.shape == Lw.shape == (6, 512)
    np.testing.assert_array_equal(Lw.sum(0), n)
    np.testing.assert_array_equal(D.sum(0), k)
    assert (Lw.sum(1) == 18 * 18).all()
    assert np.isnan(P[Lw == 0]).all()
    _, D1, _ = sample_codebook(t[3], t_hat[3])
    np.testing.assert_array_equal(D1[0], D[3])


def test_predicted_flip_prob_lookup():
    cb = np.arange(512) / 512
    t = np.ones((5, 5), np.uint8)
    assert (predicted_flip_prob(t, cb) == 511 / 512).all()
