import numpy as np
import pytest

from cdpbpc.patterns import interior, pattern_index


def test_all_white_is_511():
    assert pattern_index(np.ones((3, 3), np.uint8))[0, 0] == 511


def test_all_black_is_0():
    assert pattern_index(np.zeros((3, 3), np.uint8))[0, 0] == 0


def test_top_left_black_is_255():
    t = np.ones((3, 3), np.uint8)
    t[0, 0] = 0
    assert pattern_index(t)[0, 0] == 255


def test_center_bit_is_16():
    t = np.zeros((3, 3), np.uint8)
    t[1, 1] = 1
    assert pattern_index(t)[0, 0] == 16


def test_matches_naive_loop():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, (12, 12), dtype=np.uint8)
    idx = pattern_index(t)
    w = 2 ** np.arange(8, -1, -1)
    for i in range(1, 11):
        for j in range(1, 11):
            assert idx[i - 1, j - 1] == (t[i - 1 : i + 2, j - 1 : j + 2].ravel() * w).sum()


def test_shapes_and_batch():
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, (5, 52, 52), dtype=np.uint8)
    assert pattern_index(t).shape == (5, 50, 50)
    assert pattern_index(t, pad=True).shape == (5, 52, 52)
    np.testing.assert_array_equal(pattern_index(t)[2], pattern_index(t[2]))
    assert interior(t).shape == (5, 50, 50)


def test_pad_is_white():
    t = np.ones((4, 4), np.uint8)
    assert (pattern_index(t, pad=True) == 511).all()


def test_rejects_non_square():
    with pytest.raises(ValueError):
        pattern_index(np.ones((3, 4)))
