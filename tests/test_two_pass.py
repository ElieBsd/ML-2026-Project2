import numpy as np
import pytest

from cdpbpc.patterns import N_PATTERNS, interior, pattern_index
from cdpbpc.simulate import simulate_reads
from cdpbpc.two_pass import pooled_profile, predicted_flip_prob, predicted_sums


def _templates(n, side, seed):
    return np.random.default_rng(seed).integers(0, 2, (n, side, side), dtype=np.uint8)


def _codebook(seed, low=0.0, high=1.0):
    return np.random.default_rng(seed).uniform(low, high, N_PATTERNS)


def test_exact_estimate_gives_the_codebook():
    t = _templates(20, 30, 0)
    cb = _codebook(1)
    q = predicted_flip_prob(t, t, cb)
    np.testing.assert_array_equal(q, cb[pattern_index(t)])
    Sq, L = predicted_sums(t, t, cb)
    Q = pooled_profile(Sq, L)
    seen = L.sum(axis=0) > 0
    np.testing.assert_allclose(Q[seen], cb[seen], rtol=1e-12)


def test_one_wrong_symbol_changes_only_its_3x3_block():
    t = _templates(1, 12, 2)
    cb = _codebook(3)  # distinct values almost surely: any pattern change changes q
    r, c = 5, 6  # at distance >= 2 from the border: its 8 neighbours are interior
    t_hat = t.copy()
    t_hat[0, r, c] ^= 1
    q0 = predicted_flip_prob(t, t, cb)[0]
    q1 = predicted_flip_prob(t, t_hat, cb)[0]
    new = pattern_index(t_hat)[0]
    # interior coordinates are shifted by one
    assert q1[r - 1, c - 1] == pytest.approx(1 - cb[new[r - 1, c - 1]])
    changed = np.argwhere(q1 != q0)
    block = {(r - 1 + a, c - 1 + b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
    assert {tuple(x) for x in changed} == block
    for i, j in block - {(r - 1, c - 1)}:
        assert q1[i, j] == cb[new[i, j]]  # right symbol, new pattern around it


def test_sums_match_a_direct_computation():
    rng = np.random.default_rng(4)
    t = _templates(3, 9, 5)
    t_hat = t ^ (rng.random(t.shape) < 0.2).astype(np.uint8)
    cb = _codebook(6)
    Sq, L = predicted_sums(t, t_hat, cb, chunk=2)
    Sq_direct = np.zeros((3, N_PATTERNS))
    L_direct = np.zeros((3, N_PATTERNS))
    for n in range(3):
        for i in range(1, 8):
            for j in range(1, 8):
                w = 0
                w_hat = 0
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        w = 2 * w + int(t[n, i + a, j + b])
                        w_hat = 2 * w_hat + int(t_hat[n, i + a, j + b])
                p = cb[w_hat]
                Sq_direct[n, w] += p if t_hat[n, i, j] == t[n, i, j] else 1 - p
                L_direct[n, w] += 1
    np.testing.assert_allclose(Sq, Sq_direct, rtol=1e-12)
    np.testing.assert_array_equal(L, L_direct)


def test_pooled_profile_weights_and_fill():
    rng = np.random.default_rng(7)
    t = _templates(4, 10, 8)
    t_hat = t ^ (rng.random(t.shape) < 0.1).astype(np.uint8)
    cb = _codebook(9)
    Sq, L = predicted_sums(t, t_hat, cb)
    w = np.array([2, 0, 1, 3])
    rows = np.repeat(np.arange(4), w)
    np.testing.assert_allclose(
        pooled_profile(Sq, L, weights=w), pooled_profile(Sq[rows], L[rows]), equal_nan=True
    )
    Q = pooled_profile(Sq[:1], L[:1], fill=-1.0)
    empty = L[0] == 0
    assert empty.any()
    assert (Q[empty] == -1.0).all() and np.isnan(pooled_profile(Sq[:1], L[:1])[empty]).all()


def test_shapes_and_bad_input():
    t = _templates(2, 8, 10)
    cb = _codebook(11)
    assert predicted_flip_prob(t, t, cb).shape == (2, 6, 6)
    assert predicted_flip_prob(t[0], t[0], cb).shape == (6, 6)
    Sq, L = predicted_sums(t, t, cb)
    assert Sq.shape == L.shape == (2, N_PATTERNS)
    assert L.sum() == 2 * 36
    assert pooled_profile(Sq, L).shape == (N_PATTERNS,)
    with pytest.raises(ValueError):
        predicted_flip_prob(t, t, cb[:-1])
    with pytest.raises(ValueError):
        predicted_flip_prob(t, t, np.full(N_PATTERNS, 1.5))
    with pytest.raises(ValueError):
        predicted_flip_prob(t, t, np.full(N_PATTERNS, np.nan))
    with pytest.raises(ValueError):
        predicted_sums(t, t[:1], cb)
    with pytest.raises(ValueError):
        predicted_sums(t * 2, t, cb)
    with pytest.raises(ValueError):
        predicted_sums(t[0], t[0], cb)


def test_reprinted_estimate_matches_the_prediction():
    # Fakes simulated as the BPC applied to the estimate t_hat: their disagreement with the
    # true template t, pooled by pattern of t, follows Q_pred.
    rng = np.random.default_rng(12)
    t = _templates(100, 40, 13)
    t_hat = t ^ (rng.random(t.shape) < 0.15).astype(np.uint8)
    cb = _codebook(14, 0.02, 0.4)
    Sq, L = predicted_sums(t, t_hat, cb)
    Q = pooled_profile(Sq, L)
    n = 20
    reads = simulate_reads(t_hat, cb, n, rng)
    idx = pattern_index(t).ravel().astype(np.int64)
    wrong = interior(reads != t[None]).reshape(n, -1).sum(axis=0)
    obs_k = np.bincount(idx, weights=wrong, minlength=N_PATTERNS)
    obs_n = n * np.bincount(idx, minlength=N_PATTERNS)
    ok = obs_n >= 2000
    assert ok.sum() > 400
    # reads of one position are independent given t_hat; positions too
    z = (obs_k[ok] / obs_n[ok] - Q[ok]) / np.sqrt(Q[ok] * (1 - Q[ok]) / obs_n[ok])
    assert np.abs(z).max() < 5
    assert abs(z.mean()) < 0.2
