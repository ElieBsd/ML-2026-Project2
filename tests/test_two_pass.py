import numpy as np
import pytest

from cdpbpc.patterns import N_PATTERNS, interior, pattern_index
from cdpbpc.simulate import simulate_reads
from cdpbpc.two_pass import (
    majority_estimate,
    pooled_profile,
    predicted_flip_prob,
    predicted_sums,
    vote_estimate,
    vote_profile,
)


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


# ------------------------------------------------------------------- averaging attacker


def test_vote_estimate_keeps_the_template_when_no_pattern_is_flipped_often():
    t = _templates(5, 12, 15)
    t_hat = vote_estimate(t, np.full(N_PATTERNS, 0.4))
    np.testing.assert_array_equal(t_hat, t)
    assert t_hat.dtype == t.dtype and t_hat is not t
    # exactly 0.5 is not a majority: the symbol is kept
    np.testing.assert_array_equal(vote_estimate(t, np.full(N_PATTERNS, 0.5)), t)


def test_vote_estimate_flips_the_whole_interior_and_keeps_the_border():
    t = _templates(3, 10, 16)
    t_hat = vote_estimate(t, np.full(N_PATTERNS, 0.6))
    np.testing.assert_array_equal(interior(t_hat), 1 - interior(t))
    border = np.ones(t.shape[1:], bool)
    border[1:-1, 1:-1] = False
    np.testing.assert_array_equal(t_hat[:, border], t[:, border])


def test_vote_estimate_by_hand():
    # 4x4 template: interior symbols (1,1), (1,2), (2,1), (2,2)
    t = np.array(
        [[1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0], [1, 1, 0, 0]],
        dtype=np.uint8,
    )
    idx = pattern_index(t)  # patterns of the TRUE template around each interior symbol
    cb = np.full(N_PATTERNS, 0.1)
    cb[idx[0, 1]] = 0.9  # only the pattern around (1, 2) is read wrong most of the time
    assert len({idx[0, 0], idx[0, 1], idx[1, 0], idx[1, 1]}) == 4
    expected = t.copy()
    expected[1, 2] = 0
    np.testing.assert_array_equal(vote_estimate(t, cb), expected)
    # the estimate follows the patterns of t, not those of its own flipped symbols
    np.testing.assert_array_equal(vote_estimate(t[None], cb)[0], expected)


def test_vote_estimate_rejects_bad_input():
    t = _templates(1, 8, 17)[0]
    cb = _codebook(18)
    with pytest.raises(ValueError):
        vote_estimate(t, cb[:-1])
    with pytest.raises(ValueError):
        vote_estimate(t, np.full(N_PATTERNS, -0.1))
    with pytest.raises(ValueError):
        vote_estimate(t * 2, cb)
    with pytest.raises(ValueError):
        vote_estimate(t[0], cb)


def test_vote_profile_fills_absent_patterns_with_the_codebook():
    t = _templates(3, 9, 19)
    cb = _codebook(20)
    Q = vote_profile(t, cb)
    Sq, L = predicted_sums(t, vote_estimate(t, cb), cb)
    seen = L.sum(axis=0) > 0
    assert (~seen).any()
    np.testing.assert_allclose(Q[seen], pooled_profile(Sq, L)[seen], rtol=1e-12)
    np.testing.assert_array_equal(Q[~seen], cb[~seen])
    w = np.array([1, 0, 2])
    rows = np.repeat(np.arange(3), w)
    np.testing.assert_allclose(vote_profile(t, cb, weights=w), vote_profile(t[rows], cb))


def test_vote_profile_of_an_attacker_who_flips_nothing_is_the_codebook():
    t = _templates(10, 20, 21)
    cb = _codebook(22, 0.0, 0.5)
    np.testing.assert_allclose(vote_profile(t, cb), cb, rtol=1e-12)


def test_inputs_must_be_binary_and_weights_non_negative():
    t = _templates(2, 8, 23)
    cb = _codebook(24)
    half = t.astype(float)
    half[0, 3, 3] = 0.5
    with pytest.raises(ValueError):
        predicted_flip_prob(half, t, cb)
    with pytest.raises(ValueError):
        vote_estimate(half, cb)
    Sq, L = predicted_sums(t, t, cb)
    with pytest.raises(ValueError):
        pooled_profile(Sq, L, weights=np.array([1.0, -1.0]))
    with pytest.raises(ValueError):
        pooled_profile(Sq, L, weights=np.array([1.0, 1.0, 1.0]))


# ----------------------------------------------------------- majority of several reads


def test_majority_estimate_by_hand_and_ties():
    reads = np.array(
        [
            [[1, 1], [0, 0]],
            [[1, 0], [0, 1]],
            [[1, 0], [1, 1]],
            [[0, 0], [1, 1]],
        ],
        dtype=np.uint8,
    )
    # white votes per position: 3, 1, 2, 3 out of 4; the tie (2 of 4) goes to black (0)
    np.testing.assert_array_equal(majority_estimate(reads), [[1, 0], [0, 1]])
    np.testing.assert_array_equal(majority_estimate(reads[:3]), [[1, 0], [0, 1]])
    np.testing.assert_array_equal(majority_estimate(reads[:1]), reads[0])
    out = majority_estimate(reads)
    assert out.dtype == np.uint8 and out.shape == (2, 2)


def test_majority_of_many_bpc_reads_flips_exactly_the_unreliable_patterns():
    t = _templates(3, 20, 7)
    cb = _codebook(8, 0.0, 0.3)
    cb[_codebook(9) > 0.8] = 0.85  # some patterns read wrong most of the time
    rng = np.random.default_rng(10)
    reads = simulate_reads(t, cb, 301, rng)  # (301, 3, 20, 20)
    est = np.stack([majority_estimate(reads[:, i]) for i in range(len(t))])
    np.testing.assert_array_equal(interior(est), interior(vote_estimate(t, cb)))


def test_majority_estimate_refuses_bad_inputs():
    with pytest.raises(ValueError):
        majority_estimate(np.zeros((4, 4), np.uint8))  # a single image, not a stack
    with pytest.raises(ValueError):
        majority_estimate(np.zeros((0, 4, 4), np.uint8))  # no read
    with pytest.raises(ValueError):
        majority_estimate(np.full((3, 4, 4), 2, np.uint8))  # not binary
