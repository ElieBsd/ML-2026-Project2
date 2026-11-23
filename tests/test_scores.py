import itertools

import numpy as np
import pytest

from cdpbpc.counts import image_counts
from cdpbpc.patterns import interior, pattern_index
from cdpbpc.scores import (
    CLASS_EDGES_4,
    CLASS_EDGES_8,
    MahalanobisFit,
    binned_mahalanobis,
    binned_z,
    centered_z,
    fit_binned_mahalanobis,
    hamming,
    lrt,
    masked_hamming,
    pcor,
    pll,
    reliability_class,
)
from cdpbpc.simulate import simulate_counts


@pytest.mark.parametrize("k", [1, 3])
def test_pcor_matches_numpy(k):
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, (30, 30))
    y = np.kron(t, np.ones((k, k))) + rng.normal(0, 0.8, (30 * k, 30 * k))
    expected = -np.corrcoef(np.kron(t, np.ones((k, k))).ravel(), y.ravel())[0, 1]
    assert pcor(t, y, k) == pytest.approx(expected, rel=1e-12)


def test_pcor_orientation():
    t = np.random.default_rng(1).integers(0, 2, (20, 20))
    assert pcor(t, t.astype(float)) == pytest.approx(-1.0)  # perfect original: least suspect
    assert pcor(t, 1.0 - t) == pytest.approx(1.0)


def test_pcor_constant_image_is_nan():
    t = np.random.default_rng(2).integers(0, 2, (10, 10))
    assert np.isnan(pcor(t, np.ones((10, 10))))


def test_pcor_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        pcor(np.zeros((10, 10)), np.zeros((20, 20)), k=3)


def test_count_scores_match_direct_computation():
    rng = np.random.default_rng(3)
    t = rng.integers(0, 2, (4, 20, 20), dtype=np.uint8)
    t_hat = t ^ (rng.random(t.shape) < 0.2).astype(np.uint8)
    codebook = rng.uniform(0, 0.4, 512)
    D, L = image_counts(t, t_hat)
    d = interior(t != t_hat).reshape(4, -1).astype(float)
    p = np.clip(codebook[pattern_index(t)].reshape(4, -1), 1e-4, 1 - 1e-4)
    np.testing.assert_allclose(hamming(D, L), d.mean(1))
    np.testing.assert_allclose(
        centered_z(D, L, codebook), (d - p).sum(1) / np.sqrt((p * (1 - p)).sum(1))
    )
    m = codebook[pattern_index(t)].reshape(4, -1) <= 0.1
    np.testing.assert_allclose(masked_hamming(D, L, codebook, 0.1), (d * m).sum(1) / m.sum(1))


def test_centered_z_is_standard_under_bpc():
    rng = np.random.default_rng(4)
    codebook = rng.uniform(0.01, 0.5, 512)
    t = rng.integers(0, 2, (2000, 20, 20), dtype=np.uint8)
    t_hat = t.copy()
    interior(t_hat)[...] ^= (rng.random((2000, 18, 18)) < codebook[pattern_index(t)]).astype(
        np.uint8
    )
    z = centered_z(*image_counts(t, t_hat), codebook)
    assert abs(z.mean()) < 0.1 and abs(z.var() - 1) < 0.1


def test_masked_hamming_without_reliable_symbols_is_nan():
    D = np.zeros((1, 512), int)
    L = np.ones((1, 512), int)
    assert np.isnan(masked_hamming(D, L, np.full(512, 0.5)))


# ----------------------------------------------------------------------------- classes, lrt


def test_reliability_classes_follow_methods_7_2():
    p = np.array([0.0, 0.009, 0.01, 0.0499, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0])
    np.testing.assert_array_equal(reliability_class(p), [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 7])
    assert CLASS_EDGES_8 == (0.0, 0.01, 0.05, 0.15, 0.30, 0.50, 0.70, 0.90, 1.0)
    assert CLASS_EDGES_4 == (0.0, 0.05, 0.30, 0.70, 1.0)
    q = np.array([0.0, 0.049, 0.05, 0.29, 0.3, 0.69, 0.7, 1.0])
    np.testing.assert_array_equal(reliability_class(q, CLASS_EDGES_4), [0, 0, 1, 1, 2, 2, 3, 3])


def _example(seed, n=6):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, (n, 20, 20), dtype=np.uint8)
    t_hat = t ^ (rng.random(t.shape) < 0.25).astype(np.uint8)
    codebook = rng.uniform(0, 1, 512)
    codebook[:5] = [0.0, 1.0, 0.003, 0.995, 0.5]  # values at and near the bounds
    D, L = image_counts(t, t_hat)
    d = interior(t != t_hat).reshape(n, -1).astype(float)
    p_raw = codebook[pattern_index(t)].reshape(n, -1)
    return rng, D, L, d, p_raw, codebook


@pytest.mark.parametrize("edges", [CLASS_EDGES_8, CLASS_EDGES_4])
def test_binned_z_matches_direct_computation(edges):
    _, D, L, d, p_raw, codebook = _example(5)
    z = binned_z(D, L, codebook, edges)
    assert z.shape == (len(D), len(edges) - 1)
    cls = reliability_class(p_raw, edges)
    p = np.clip(p_raw, 1e-4, 1 - 1e-4)
    for b in range(len(edges) - 1):
        m = cls == b
        num = ((d - p) * m).sum(1)
        den = np.sqrt((p * (1 - p) * m).sum(1))
        expected = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
        np.testing.assert_allclose(z[:, b], expected, atol=1e-12)


def test_binned_z_with_one_class_is_centered_z():
    _, D, L, _, _, codebook = _example(6)
    np.testing.assert_allclose(
        binned_z(D, L, codebook, (0.0, 1.0))[:, 0], centered_z(D, L, codebook)
    )


def test_empty_class_gives_zero_without_nan():
    _, D, L, _, _, _ = _example(7)
    z = binned_z(D, L, np.full(512, 0.2))  # every pattern in class 3
    assert np.isfinite(z).all()
    assert (z[:, [0, 1, 2, 4, 5, 6, 7]] == 0).all() and (z[:, 3] != 0).any()


def test_binned_z_is_standard_per_class_under_bpc():
    rng = np.random.default_rng(8)
    t = rng.integers(0, 2, (3000, 24, 24), dtype=np.uint8)
    codebook = np.concatenate(
        [rng.uniform(lo, hi, 64) for lo, hi in itertools.pairwise(CLASS_EDGES_8)]
    )
    rng.shuffle(codebook)
    _, L = image_counts(t, t)
    z = binned_z(simulate_counts(L, codebook, rng), L, codebook)
    np.testing.assert_allclose(z.mean(0), 0, atol=0.08)
    np.testing.assert_allclose(z.var(0), 1, atol=0.12)


def test_lrt_matches_the_formula_and_is_zero_when_q_equals_p():
    rng = np.random.default_rng(9)
    t = rng.integers(0, 2, (5, 20, 20), dtype=np.uint8)
    t_hat = t ^ (rng.random(t.shape) < 0.25).astype(np.uint8)
    p_book, q_book = rng.uniform(0, 1, 512), rng.uniform(0, 1, 512)
    p_book[:4], q_book[:4] = [0.0, 1.0, 0.0005, 0.5], [0.5, 0.0, 0.9995, 1.0]  # clipped
    D, L = image_counts(t, t_hat)
    idx = pattern_index(t).reshape(5, -1)
    d = interior(t != t_hat).reshape(5, -1).astype(float)
    p = np.clip(p_book[idx], 1e-3, 1 - 1e-3)
    q = np.clip(q_book[idx], 1e-3, 1 - 1e-3)
    direct = (d * np.log(q / p) + (1 - d) * np.log((1 - q) / (1 - p))).sum(1)
    s = lrt(D, L, p_book, q_book)
    assert s.shape == (5,) and np.isfinite(s).all()
    np.testing.assert_allclose(s, direct, rtol=1e-12)
    np.testing.assert_allclose(lrt(D, L, p_book, p_book), 0, atol=1e-9)


def test_lrt_grows_with_flips_where_fakes_flip_more():
    L = np.full((1, 512), 100)
    p = np.full(512, 0.1)
    q = p.copy()
    q[:256] = 0.4  # fakes flip more on half of the patterns
    D = np.full((1, 512), 10)
    more = D.copy()
    more[0, :256] += 5  # more flips where q > p: more suspect
    less = D.copy()
    less[0, 256:] += 5  # more flips where q = p: no change
    assert lrt(more, L, p, q)[0] > lrt(D, L, p, q)[0]
    assert lrt(less, L, p, q)[0] == pytest.approx(lrt(D, L, p, q)[0])


def test_lrt_is_larger_for_reads_drawn_like_fakes():
    rng = np.random.default_rng(10)
    t = rng.integers(0, 2, (500, 20, 20), dtype=np.uint8)
    p = rng.uniform(0.01, 0.4, 512)
    q = np.clip(p + 0.1, 0, 1)
    _, L = image_counts(t, t)
    originals = lrt(simulate_counts(L, p, rng), L, p, q)
    fakes = lrt(simulate_counts(L, q, rng), L, p, q)
    assert fakes.mean() > originals.mean() and (fakes > np.median(originals)).mean() > 0.9


# ----------------------------------------------------------------------------- pll


def test_pll_hand_example():
    codebook = np.full(512, 0.2)
    codebook[:2] = [0.1, 0.5]
    D = np.zeros((1, 512), int)
    L = np.zeros((1, 512), int)
    D[0, 0], L[0, 0], L[0, 1] = 1, 3, 2  # pattern 0: 1 flip in 3; pattern 1: 0 flip in 2
    expected = -(np.log(0.1) + 2 * np.log(0.9) + 2 * np.log(0.5))
    assert pll(D, L, codebook)[0] == pytest.approx(expected, rel=1e-12)


def test_pll_matches_thesis_formula_4_9_per_symbol():
    # PLL = sum over symbols of log pi, pi = 1 - P_b if read right, P_b if flipped.
    _, D, L, d, p_raw, codebook = _example(10)
    p = np.clip(p_raw, 1e-4, 1 - 1e-4)
    log_pi = np.where(d == 1, np.log(p), np.log(1 - p))
    np.testing.assert_allclose(pll(D, L, codebook), -log_pi.sum(1), rtol=1e-12)


def test_pll_is_large_when_reliable_patterns_flip():
    codebook = np.full(512, 0.3)
    codebook[0] = 0.001
    L = np.zeros((2, 512), int)
    L[:, 0] = L[:, 1] = 10
    D = np.zeros((2, 512), int)
    D[0, 1] = 1  # one flip on an unreliable pattern
    D[1, 0] = 1  # one flip on a reliable pattern
    s = pll(D, L, codebook)
    assert s[1] > s[0]
    assert np.isfinite(pll(D, L, np.zeros(512))).all()  # p = 0 is clipped, no inf


# ----------------------------------------------------------------------------- binned_mahalanobis


def _class_codebook(rng, edges=CLASS_EDGES_4):
    cb = np.concatenate([rng.uniform(lo, hi, 128) for lo, hi in itertools.pairwise(edges)])
    rng.shuffle(cb)
    return cb


def _bpc_images(rng, codebook, n):
    L = rng.multinomial(22 * 22, np.full(512, 1 / 512), size=n)
    return simulate_counts(L, codebook, rng), L


def test_binned_mahalanobis_matches_direct_computation():
    rng = np.random.default_rng(11)
    cb = _class_codebook(rng)
    DM, LM = _bpc_images(rng, cb, 150)
    D, L = _bpc_images(rng, cb, 20)
    fit = fit_binned_mahalanobis(DM, LM, cb)
    zM = binned_z(DM, LM, cb, CLASS_EDGES_4)
    m = zM.mean(0)
    S = np.cov(zM, rowvar=False) + 1e-6 * np.eye(4)
    z = binned_z(D, L, cb, CLASS_EDGES_4)
    expected = np.einsum("ni,ij,nj->n", z - m, np.linalg.inv(S), z - m)
    np.testing.assert_allclose(binned_mahalanobis(D, L, cb, fit), expected, rtol=1e-9)
    np.testing.assert_allclose(fit.mean, m)


def test_binned_mahalanobis_with_identity_fit_is_sum_of_squares():
    _, D, L, _, _, codebook = _example(12)
    fit = MahalanobisFit(np.zeros(4), np.eye(4))
    z = binned_z(D, L, codebook, CLASS_EDGES_4)
    np.testing.assert_allclose(binned_mahalanobis(D, L, codebook, fit), (z**2).sum(1))


def test_binned_mahalanobis_mean_is_four_under_bpc():
    # Exact m and S would give a mean of K = 4 (number of classes).
    rng = np.random.default_rng(13)
    cb = _class_codebook(rng)
    fit = fit_binned_mahalanobis(*_bpc_images(rng, cb, 20000), cb)
    s = binned_mahalanobis(*_bpc_images(rng, cb, 20000), cb, fit)
    assert s.mean() == pytest.approx(4.0, abs=0.1)
    assert (s >= 0).all()


def test_binned_mahalanobis_mean_with_150_originals_in_m():
    # m and S estimated on n = 150 originals: a new original scores on average
    # K (n + 1)(n - 1) / (n (n - K - 2)) = 4.17, not 4. Not a bug.
    rng = np.random.default_rng(14)
    cb = _class_codebook(rng)
    n, K = 150, 4
    means = []
    for _ in range(300):
        fit = fit_binned_mahalanobis(*_bpc_images(rng, cb, n), cb)
        means.append(binned_mahalanobis(*_bpc_images(rng, cb, 100), cb, fit).mean())
    expected = K * (n + 1) * (n - 1) / (n * (n - K - 2))
    assert np.mean(means) == pytest.approx(expected, abs=0.06)
    assert np.mean(means) > 4.05


def test_binned_mahalanobis_empty_class_gives_finite_scores():
    _, D, L, _, _, _ = _example(15, n=30)
    cb = np.full(512, 0.2)  # every pattern in class 1 of 4
    fit = fit_binned_mahalanobis(D[:20], L[:20], cb)
    s = binned_mahalanobis(D[20:], L[20:], cb, fit)
    assert np.isfinite(fit.inv_cov).all() and np.isfinite(s).all()


def test_binned_mahalanobis_fit_depends_only_on_block_m():
    rng = np.random.default_rng(16)
    cb = _class_codebook(rng)
    DM, LM = _bpc_images(rng, cb, 150)
    fit = fit_binned_mahalanobis(DM, LM, cb)
    D, L = _bpc_images(rng, cb, 10)
    one = binned_mahalanobis(D, L, cb, fit)
    both = binned_mahalanobis(np.r_[D, D + 3], np.r_[L, L + 5], cb, fit)
    np.testing.assert_allclose(both[:10], one)  # judging more images changes nothing


def test_binned_mahalanobis_fit_needs_two_originals():
    _, D, L, _, _, codebook = _example(17)
    with pytest.raises(ValueError):
        fit_binned_mahalanobis(D[:1], L[:1], codebook)
