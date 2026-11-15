import numpy as np
import pytest

from cdpbpc.counts import image_counts
from cdpbpc.patterns import interior, pattern_index
from cdpbpc.scores import centered_z, hamming, masked_hamming, pcor


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
