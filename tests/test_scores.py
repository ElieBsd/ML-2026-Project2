import numpy as np
import pytest

from cdpbpc.scores import pcor


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
