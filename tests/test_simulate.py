import numpy as np
import pytest

from cdpbpc.codebook import fit_codebook
from cdpbpc.counts import image_counts
from cdpbpc.patterns import interior, pattern_index
from cdpbpc.simulate import simulate_counts, simulate_flip_counts, simulate_reads


def _templates(seed, n=200, side=20):
    rng = np.random.default_rng(seed)
    return rng, rng.integers(0, 2, (n, side, side), dtype=np.uint8)


def test_reads_keep_the_border_and_follow_the_codebook():
    rng, t = _templates(0)
    codebook = rng.uniform(0.05, 0.6, 512)
    reads = simulate_reads(t, codebook, 3, rng)
    assert reads.shape == (3,) + t.shape and reads.dtype == np.uint8
    assert set(np.unique(reads)) <= {0, 1}
    border = np.ones(t.shape[1:], bool)
    border[1:-1, 1:-1] = False
    assert (reads[:, :, border] == t[:, border]).all()  # the outer ring never flips
    # Every pattern flips at its codebook rate, within 4 standard errors.
    n, k = np.zeros(512), np.zeros(512)
    for r in reads:
        D, L = image_counts(t, r)
        k += D.sum(0)
        n += L.sum(0)
    seen = n > 0
    se = np.sqrt(codebook * (1 - codebook) / np.maximum(n, 1))
    assert (np.abs(k / np.maximum(n, 1) - codebook)[seen] < 4 * se[seen] + 1e-9).all()


def test_reads_are_reproducible_and_independent():
    _, t = _templates(1, n=50)
    codebook = np.full(512, 0.3)
    a = simulate_reads(t, codebook, 2, np.random.default_rng(7))
    b = simulate_reads(t, codebook, 2, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    da, db = (interior(r != t).ravel().astype(float) for r in a)
    assert abs(np.corrcoef(da, db)[0, 1]) < 0.02  # two reads flip independently


def test_flip_counts_have_the_law_of_n_reads():
    rng, t = _templates(2, n=300)
    codebook = rng.uniform(0.05, 0.6, 512)
    n = 6
    K = simulate_flip_counts(t, codebook, n, rng)
    assert K.shape == t.shape and K.dtype == np.uint8
    assert (K[:, 0, :] == 0).all() and (K[:, :, -1] == 0).all()
    p = codebook[pattern_index(t)]
    k = interior(K).astype(float)
    # Binomial(n, p) per position: mean n p, variance n p (1 - p).
    assert k.mean() == pytest.approx(n * p.mean(), rel=0.01)
    assert ((k - n * p) ** 2).mean() == pytest.approx((n * p * (1 - p)).mean(), rel=0.02)
    assert k.max() <= n


def test_simulated_reads_recover_the_codebook():
    rng, t = _templates(3, n=400)
    codebook = rng.uniform(0.05, 0.6, 512)
    reads = simulate_reads(t, codebook, 1, rng)[0]
    fitted = fit_codebook(t, reads)
    n = np.bincount(pattern_index(t).ravel(), minlength=512)
    se = np.sqrt(codebook * (1 - codebook) / np.maximum(n, 1))
    assert (np.abs(fitted - codebook) < 4 * se + 2 / np.maximum(n, 1)).all()


def test_counts_are_binomial_given_the_template():
    rng, t = _templates(4, n=40)
    codebook = rng.uniform(0.05, 0.6, 512)
    _, L = image_counts(t, t)
    draws = np.stack([simulate_counts(L, codebook, rng) for _ in range(300)])
    assert draws.shape == (300,) + L.shape and (draws <= L).all() and (draws >= 0).all()
    mean, var = draws.mean(0), draws.var(0)
    expected_var = L * codebook * (1 - codebook)
    assert np.abs(mean - L * codebook).max() < 5 * np.sqrt(expected_var.max() / 300) + 1e-9
    big = L >= 5
    assert np.median(var[big] / expected_var[big]) == pytest.approx(1.0, abs=0.08)


def test_counts_agree_with_simulated_images():
    """Counts drawn directly and counts of simulated images have the same law."""
    rng, t = _templates(5, n=300)
    codebook = rng.uniform(0.05, 0.6, 512)
    _, L = image_counts(t, t)
    D_direct = simulate_counts(L, codebook, rng)
    D_image, _ = image_counts(t, simulate_reads(t, codebook, 1, rng)[0])
    h_direct = D_direct.sum(1) / L.sum(1)
    h_image = D_image.sum(1) / L.sum(1)
    assert h_direct.mean() == pytest.approx(h_image.mean(), abs=0.004)
    assert h_direct.std() == pytest.approx(h_image.std(), rel=0.2)


def test_rejects_bad_input():
    rng, t = _templates(6, n=4)
    with pytest.raises(ValueError):
        simulate_reads(t, np.full(100, 0.1), 1, rng)  # not a 512-entry codebook
    with pytest.raises(ValueError):
        simulate_flip_counts(t, np.full(512, 0.1), 0, rng)
    with pytest.raises(ValueError):
        simulate_reads(t, np.full(512, 1.5), 1, rng)
