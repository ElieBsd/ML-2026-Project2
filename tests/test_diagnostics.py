import numpy as np
import pytest

from cdpbpc.diagnostics import (
    adjusted_eta2,
    flip_counts,
    icc_oneway,
    reproducibility,
    two_way_components,
)
from cdpbpc.patterns import interior, pattern_index

N, SIDE = 600, 24


def _setup(seed):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, (N, SIDE, SIDE), dtype=np.uint8)
    p_b = rng.uniform(0.15, 0.45, 512)
    return rng, t, p_b[pattern_index(t)]


def _reads(rng, t, pi, n):
    """n reads in which each interior symbol flips with probability pi."""
    out = []
    for _ in range(n):
        r = t.copy()
        interior(r)[...] ^= (rng.random(pi.shape) < pi).astype(np.uint8)
        out.append(r)
    return out


def _injected(rng, p, d=0.1):
    """Position flip probabilities p +/- d: a reproducible position effect of variance d^2."""
    return p + d * rng.choice([-1.0, 1.0], size=p.shape)


def _expected_rho(p, pi):
    return float(((pi - p) ** 2).mean() / (p * (1 - p)).mean())


def test_flip_counts():
    t = np.zeros((2, 4, 4), np.uint8)
    a, b = t.copy(), t.copy()
    a[0, 1, 1] = b[0, 1, 1] = b[1, 2, 2] = 1
    K = flip_counts(t, [a, b])
    assert K.dtype == np.uint8 and K[0, 1, 1] == 2 and K[1, 2, 2] == 1 and K.sum() == 3
    with pytest.raises(ValueError):
        flip_counts(t, [a[:1]])


def test_rho_is_zero_under_bpc():
    rng, t, p = _setup(0)
    K = flip_counts(t, _reads(rng, t, p, 6))
    r = reproducibility(t, K, 6)
    assert abs(r["rho"]) < 0.02
    assert r["D_obs"] == pytest.approx(r["D_bpc"], rel=0.03)


def test_rho_recovers_an_injected_position_effect():
    rng, t, p = _setup(1)
    pi = _injected(rng, p)
    r = reproducibility(t, flip_counts(t, _reads(rng, t, pi, 6)), 6)
    assert r["rho"] == pytest.approx(_expected_rho(p, pi), abs=0.02)


def test_rho_does_not_depend_on_the_number_of_reads():
    rng, t, p = _setup(2)
    pi = _injected(rng, p)
    reads = _reads(rng, t, pi, 8)
    r4 = reproducibility(t, flip_counts(t, reads[:4]), 4)["rho"]
    r8 = reproducibility(t, flip_counts(t, reads), 8)["rho"]
    assert r4 == pytest.approx(r8, abs=0.02)


def test_shared_part_between_two_conditions():
    rng, t, p = _setup(3)
    pi = _injected(rng, p)
    K = flip_counts(t, _reads(rng, t, pi, 6))
    same = reproducibility(t, K, 6, flip_counts(t, _reads(rng, t, pi, 3)), 3)
    assert same["shared_over_var_a"] == pytest.approx(1.0, abs=0.1)
    assert same["var_b"] == pytest.approx(same["var_a"], rel=0.15)
    other = _injected(rng, p)  # an unrelated position effect
    diff = reproducibility(t, K, 6, flip_counts(t, _reads(rng, t, other, 1)), 1)
    assert abs(diff["shared_over_var_a"]) < 0.1 and "var_b" not in diff


def test_icc_oneway():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(300), 20)
    noise = rng.standard_normal(groups.size)
    assert icc_oneway(noise, groups) == pytest.approx(0.0, abs=0.03)
    effect = rng.standard_normal(300)[groups]  # group variance 1, residual variance 1
    assert icc_oneway(effect + noise, groups) == pytest.approx(0.5, abs=0.06)


def test_adjusted_eta2():
    rng = np.random.default_rng(6)
    labels = np.arange(1440) % 144
    assert adjusted_eta2(rng.standard_normal(1440), labels) == pytest.approx(0.0, abs=0.05)
    means = rng.standard_normal(144)
    assert adjusted_eta2(means[labels] + 0.01 * rng.standard_normal(1440), labels) > 0.99


def test_two_way_components_recover_simulated_effects():
    rng = np.random.default_rng(7)
    n, r = 2000, 10
    H = (
        0.03 * rng.standard_normal((n, 1))  # template effect, 3 points
        + 0.01 * rng.standard_normal((1, r))  # read effect, 1 point
        + 0.02 * rng.standard_normal((n, r))  # residual, 2 points
    )
    c = two_way_components(H)
    assert c["sd_template_pts"] == pytest.approx(3.0, abs=0.15)
    assert c["sd_residual_pts"] == pytest.approx(2.0, abs=0.05)
    assert 0.3 < c["sd_read_pts"] < 2.0  # only 10 reads: a rough estimate
