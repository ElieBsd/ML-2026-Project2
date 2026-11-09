import numpy as np
import pytest

from cdpbpc.binarize import binarize_mean, binarize_otsu, binarize_vote


def _render(t, k, noise, rng):
    y = np.kron(t, np.ones((k, k))).astype(np.float32)
    y = 0.15 + 0.7 * y + noise * rng.standard_normal(y.shape)
    return np.clip(y, 0, 1).astype(np.float32)


@pytest.mark.parametrize("k", [3, 4])
def test_vote_recovers_clean_template(k):
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, (52, 52), dtype=np.uint8)
    y = _render(t, k, 0.05, rng)
    np.testing.assert_array_equal(binarize_vote(y, k), t)
    # Synthetic block means have an empty gap between classes; skimage's Otsu then picks
    # the lowest threshold of the flat optimum, so a few dark blocks at the edge can flip.
    assert (binarize_mean(y, k) != t).mean() < 0.005


def test_otsu_k1():
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, (20, 20), dtype=np.uint8)
    np.testing.assert_array_equal(binarize_otsu(_render(t, 1, 0.02, rng)), t)


def test_vote_threshold_9_of_16():
    a =np.zeros((4, 4), np.float32)
    a.ravel()[:9] = 1.0  # 9 white pixels -> white
    b = np.zeros((4, 4), np.float32)
    b.ravel()[:7] = 1.0  # 7 white pixels -> black
    y = np.hstack([a, b])
    assert binarize_vote(y, 4, threshold=0.5).tolist() == [[1, 0]]


def test_vote_tie_uses_block_mean():
    # 8 of 16 pixels above the threshold in both blocks; block means differ.
    a = np.r_[np.full(8, 0.9), np.full(8, 0.3)].reshape(4, 4)  # mean 0.6 > 0.5
    b = np.r_[np.full(8, 0.6), np.full(8, 0.0)].reshape(4, 4)  # mean 0.3 < 0.5
    y = np.hstack([a, b]).astype(np.float32)
    assert binarize_vote(y, 4, threshold=0.5).tolist() == [[1, 0]]


def test_vote_shape_and_dtype():
    y = np.random.default_rng(2).random((208, 208)).astype(np.float32)
    out = binarize_vote(y, 4)
    assert out.shape == (52, 52) and out.dtype == np.uint8


def test_vote_rejects_bad_shape():
    with pytest.raises(ValueError):
        binarize_vote(np.zeros((10, 10)), 4)
