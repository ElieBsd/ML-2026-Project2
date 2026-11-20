import numpy as np
import pytest

from cdpbpc.counts import image_counts
from cdpbpc.diagnostics import (
    RELIABILITY_EDGES,
    adjusted_eta2,
    cross_fitted_codebooks,
    crossed_components,
    flip_counts,
    icc_oneway,
    map_stats,
    pattern_sums,
    reliability_class,
    reproducibility,
    reproducibility_from_sums,
    residual_flip_rate,
    sigma_image,
    simulate_pattern_sums,
    template_maps,
    two_point,
    two_way_components,
)
from cdpbpc.patterns import interior, pattern_index
from cdpbpc.simulate import simulate_counts, simulate_flip_counts, simulate_reads

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


# ----------------------------------------------------------------------------- C1a suite


def test_reliability_classes_follow_methods_7_2():
    p = np.array([0.0, 0.009, 0.01, 0.0499, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0])
    np.testing.assert_array_equal(reliability_class(p), [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 7])
    assert len(RELIABILITY_EDGES) == 9


def test_pattern_sums_match_a_direct_count():
    rng, t, p = _setup(10)
    t, p = t[:5], p[:5]
    K = flip_counts(t, _reads(rng, t, p, 3))
    s = pattern_sums(t, K, 3)
    idx, k = pattern_index(t[2]).ravel(), interior(K[2]).ravel().astype(float)
    w = 37
    assert s["L"][2, w] == (idx == w).sum()
    assert s["S1"][2, w] == k[idx == w].sum()
    assert s["SD"][2, w] == pytest.approx((2 * k * (3 - k) / 6)[idx == w].sum())
    _, L = image_counts(t, t)
    np.testing.assert_array_equal(s["L"], L)


def test_default_folds_are_the_two_halves_by_rank():
    rng, t, p = _setup(11)
    K = flip_counts(t, _reads(rng, t, _injected(rng, p), 4))
    a = reproducibility(t, K, 4)
    b = reproducibility(t, K, 4, folds=np.arange(N) % 2)
    assert a["rho"] == b["rho"] and a["D_bpc"] == b["D_bpc"]


def test_weights_are_template_multiplicities():
    rng, t, p = _setup(12)
    K = flip_counts(t, _reads(rng, t, _injected(rng, p), 3))
    folds = np.arange(N) % 2
    doubled = reproducibility(np.r_[t, t], np.r_[K, K], 3, folds=np.r_[folds, folds])
    weighted = reproducibility_from_sums(
        pattern_sums(t, K, 3), 3, folds=folds, weights=np.full(N, 2.0)
    )
    assert weighted["rho"] == pytest.approx(doubled["rho"], abs=1e-12)
    assert weighted["D_obs"] == pytest.approx(doubled["D_obs"], abs=1e-12)


def test_rho_is_zero_under_simulated_bpc_and_by_class():
    rng, t, _ = _setup(13)
    codebook = rng.uniform(0.005, 0.95, 512)
    r = reproducibility(t, simulate_flip_counts(t, codebook, 5, rng), 5)
    assert abs(r["rho"]) < 0.02
    filled = np.array(r["share_by_class"]) > 0.02
    assert np.nanmax(np.abs(np.array(r["rho_by_class"])[filled])) < 0.08


def test_rho_by_class_finds_an_effect_in_one_class():
    rng = np.random.default_rng(14)
    t = rng.integers(0, 2, (N, SIDE, SIDE), dtype=np.uint8)
    codebook = rng.uniform(0.15, 0.7, 512)
    p = codebook[pattern_index(t)]
    target = (p >= 0.30) & (p < 0.50)  # class 4
    pi = np.where(target, p + 0.12 * rng.choice([-1.0, 1.0], size=p.shape), p)
    r = reproducibility(t, flip_counts(t, _reads(rng, t, pi, 6)), 6)
    by = np.array(r["rho_by_class"])
    assert by[4] == pytest.approx(0.0144 / (p[target] * (1 - p[target])).mean(), abs=0.03)
    assert abs(by[3]) < 0.03 and abs(by[5]) < 0.03


def test_external_codebook_for_the_second_condition():
    rng, t, p = _setup(15)
    pi = _injected(rng, p)
    K = flip_counts(t, _reads(rng, t, pi, 6))
    Kb = flip_counts(t, _reads(rng, t, pi, 1))
    idx = pattern_index(t).ravel()
    seen = np.maximum(np.bincount(idx, minlength=512), 1)
    true_codebook = np.bincount(idx, weights=p.ravel(), minlength=512) / seen
    r = reproducibility(t, K, 6, Kb, 1, codebook_b=true_codebook)
    assert r["shared_over_var_a"] == pytest.approx(1.0, abs=0.1)
    shifted = reproducibility(t, K, 6, Kb, 1, codebook_b=np.clip(true_codebook + 0.05, 0, 1))
    assert shifted["X_bpc"] != pytest.approx(r["X_bpc"], abs=1e-3)  # the external one is used


def test_simulated_pattern_sums_have_the_law_of_simulated_positions():
    rng = np.random.default_rng(16)
    t = rng.integers(0, 2, (200, SIDE, SIDE), dtype=np.uint8)
    codebook = rng.uniform(0.01, 0.9, 512)
    folds = np.arange(200) % 2
    _, L = image_counts(t, t)
    cells = np.stack([L[folds == g].sum(0) for g in (0, 1)]).astype(float)
    rho_pos, rho_cell, x_pos, x_cell = [], [], [], []
    for _ in range(60):
        K = simulate_flip_counts(t, codebook, 4, rng)
        rho_pos.append(reproducibility(t, K, 4)["rho"])
        Kb = simulate_flip_counts(t, codebook, 2, rng)
        x_pos.append(reproducibility(t, K, 4, Kb, 2)["shared"])
        s = simulate_pattern_sums(cells, codebook, 4, rng)
        assert s["L"].shape == (2, 512) and (s["S1"] <= 4 * s["L"]).all()
        rho_cell.append(reproducibility_from_sums(s, 4, folds=np.array([0, 1]))["rho"])
        sx = simulate_pattern_sums(cells, codebook, 4, rng, codebook, 2)
        x_cell.append(reproducibility_from_sums(sx, 4, 2, folds=np.array([0, 1]))["shared"])
    assert np.mean(rho_cell) == pytest.approx(np.mean(rho_pos), abs=0.004)
    assert np.std(rho_cell) == pytest.approx(np.std(rho_pos), rel=0.35)
    # Small counts: the Jeffreys prior gives the shared part a small bias under the BPC,
    # the same whichever way the reads are drawn (that is what the null reference measures).
    se = np.sqrt((np.var(x_cell) + np.var(x_pos)) / 60)
    assert abs(np.mean(x_cell) - np.mean(x_pos)) < 4 * se


def test_simulated_sums_expectations():
    rng = np.random.default_rng(17)
    cells = rng.integers(50, 200, (3, 512)).astype(float)
    codebook = rng.uniform(0.05, 0.8, 512)
    s = [simulate_pattern_sums(cells, codebook, 6, rng, codebook, 3) for _ in range(200)]
    mean = {k: np.mean([x[k] for x in s], axis=0) for k in s[0]}
    np.testing.assert_allclose(mean["S1"].sum(), 6 * (cells * codebook).sum(), rtol=0.01)
    pred = 2 * codebook * (1 - codebook)
    np.testing.assert_allclose(mean["SD"].sum(), (cells * pred).sum(), rtol=0.01)
    np.testing.assert_allclose(mean["SDb"].sum(), (cells * pred).sum(), rtol=0.01)
    np.testing.assert_allclose(mean["SX"].sum(), (cells * pred).sum(), rtol=0.01)


def test_cross_fitted_codebooks_never_use_the_judged_fold():
    rng, t, p = _setup(18)
    K = flip_counts(t, _reads(rng, t, p, 2))
    s = pattern_sums(t, K, 2)
    folds = np.arange(N) % 3
    cb = cross_fitted_codebooks(s, 2, folds)
    assert cb.shape == (3, 512)
    train = folds != 1
    expected = (s["S1"][train].sum(0) + 0.5) / (2 * s["L"][train].sum(0) + 1)
    np.testing.assert_allclose(cb[1], expected)


def _image_counts_with_effect(rng, L, codebook, sd):
    """One read per image; image i flips with probability P_b + e_i, e_i ~ N(0, sd)."""
    e = sd * rng.standard_normal((len(L), 1))
    return rng.binomial(L, np.clip(codebook[None] + e, 0, 1))


def _uneven_templates(rng, n, side):
    """Templates whose white density varies a lot, so that their expected flip rates differ."""
    density = rng.uniform(0.15, 0.85, (n, 1, 1))
    return (rng.random((n, side, side)) < density).astype(np.uint8)


def test_sigma_image_is_zero_under_bpc_even_when_templates_differ():
    rng = np.random.default_rng(19)
    t = _uneven_templates(rng, 3000, 14)
    codebook = rng.uniform(0.01, 0.95, 512)
    _, L = image_counts(t, t)
    vals = []
    for _ in range(20):
        r, b = residual_flip_rate(simulate_counts(L, codebook, rng), L)
        vals.append(sigma_image(r, b)["var_image"])
    s = sigma_image(r, b)
    assert abs(np.mean(vals)) < 0.06 * s["sd_binomial_pts"] ** 2 / 1e4
    assert s["sd_total_over_binomial"] == pytest.approx(1.0, abs=0.06)
    # The raw flip rate would have shown a large spurious image effect.
    h = simulate_counts(L, codebook, rng).sum(1) / L.sum(1)
    assert h.var() - b.mean() > 0.5 * b.mean()


def test_sigma_image_recovers_an_injected_image_effect():
    rng = np.random.default_rng(20)
    t = _uneven_templates(rng, 3000, 40)
    codebook = rng.uniform(0.05, 0.6, 512)
    _, L = image_counts(t, t)
    r, b = residual_flip_rate(_image_counts_with_effect(rng, L, codebook, 0.02), L)
    s = sigma_image(r, b)
    assert s["sd_image_pts"] == pytest.approx(2.0, abs=0.12)
    assert s["sd_image_over_binomial"] == pytest.approx(2.0 / s["sd_binomial_pts"], rel=0.1)


def test_residual_flip_rate_on_a_table_pools_the_reads():
    rng = np.random.default_rng(21)
    t = _uneven_templates(rng, 400, 20)
    codebook = rng.uniform(0.05, 0.6, 512)
    _, L = image_counts(t, t)
    D = np.stack([simulate_counts(L, codebook, rng) for _ in range(3)], axis=1)  # (N, 3, 512)
    shifted = D.copy()
    shifted[:, 2] = rng.binomial(L, np.clip(codebook + 0.05, 0, 1))
    r, b = residual_flip_rate(shifted, np.repeat(L[:, None], 3, 1))
    assert r.shape == (400, 3) and b.shape == (400, 3)
    assert r[:, 2].mean() - r[:, :2].mean() == pytest.approx(0.05, abs=0.005)  # read shift kept
    c = two_way_components(r)
    assert c["sd_read_pts"] > 2


def test_weighted_two_way_components_equal_duplicated_rows():
    rng = np.random.default_rng(22)
    H = rng.standard_normal((50, 4)) + rng.standard_normal((50, 1))
    w = rng.integers(0, 3, 50).astype(float)
    dup = np.repeat(H, w.astype(int), axis=0)
    a, b = two_way_components(H, weights=w), two_way_components(dup)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-9)
    labels = np.arange(50) % 5
    assert adjusted_eta2(H[:, 0], labels, weights=w) == pytest.approx(
        adjusted_eta2(dup[:, 0], np.repeat(labels, w.astype(int))), rel=1e-9
    )


def test_crossed_components_recover_simulated_effects():
    rng = np.random.default_rng(23)
    a, b, c = 400, 5, 10
    A = (
        0.03 * rng.standard_normal((a, 1, 1))  # template
        + 0.02 * rng.standard_normal((1, b, 1))  # sheet (only 5: rough)
        + 0.01 * rng.standard_normal((a, b, 1))  # template x sheet
        + 0.04 * rng.standard_normal((a, b, c))  # copy
    )
    comp = crossed_components(A)
    assert comp["sd_template_pts"] == pytest.approx(3.0, abs=0.2)
    assert comp["sd_template_x_sheet_pts"] == pytest.approx(1.0, abs=0.2)
    assert comp["sd_copy_pts"] == pytest.approx(4.0, abs=0.05)
    assert 0.5 < comp["sd_sheet_pts"] < 4.0
    w = rng.integers(0, 3, a).astype(float)
    dup = np.repeat(A, w.astype(int), axis=0)
    for k, v in crossed_components(A, weights=w).items():
        assert v == pytest.approx(crossed_components(dup)[k], rel=1e-9)


def _map_setup(seed, effect=None, n_templates=800, side=30, n=4):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, (n_templates, side, side), dtype=np.uint8)
    codebook = rng.uniform(0.1, 0.6, 512)
    p = codebook[pattern_index(t)]
    pi = p if effect is None else np.clip(p + effect[None], 0, 1)
    K = flip_counts(t, _reads(rng, t, pi, n))
    folds = np.arange(n_templates) % 2
    cb = cross_fitted_codebooks(pattern_sums(t, K, n), n, folds)
    return rng, t, K, folds, cb


def test_position_map_is_flat_under_bpc():
    _, t, K, folds, cb = _map_setup(24)
    M = template_maps(t, K, 4, cb, folds)
    assert M.shape == (800, 28, 28) and M.dtype == np.float32
    s = map_stats(M, folds)
    assert s["sd_reliable_inner_pts"] < 0.4
    assert abs(s["corr_halves_inner"]) < 0.15


def test_position_map_finds_an_injected_position_effect():
    rng = np.random.default_rng(25)
    effect = 0.03 * rng.choice([-1.0, 1.0], size=(28, 28))
    _, t, K, folds, cb = _map_setup(26, effect)
    s = map_stats(template_maps(t, K, 4, cb, folds), folds)
    assert s["sd_reliable_inner_pts"] == pytest.approx(3.0, abs=0.3)
    assert s["corr_halves_inner"] > 0.8
    w = rng.integers(0, 3, 800).astype(float)
    sw = map_stats(template_maps(t, K, 4, cb, folds), folds, weights=w[None])
    assert sw["sd_reliable_inner_pts"].shape == (1,)
    assert sw["sd_reliable_inner_pts"][0] == pytest.approx(3.0, abs=0.4)


def test_map_edge_rows_show_a_border_effect():
    effect = np.zeros((28, 28))
    effect[0] = 0.05
    _, t, K, folds, cb = _map_setup(27, effect)
    s = map_stats(template_maps(t, K, 4, cb, folds), folds)
    assert s["row_profile_pts"][0] == pytest.approx(5.0, abs=0.7)
    assert abs(s["row_profile_pts"][10]) < 0.7


LAGS = ((0, 1), (1, 0), (0, 20), (20, 0))


def test_two_point_is_zero_under_bpc():
    rng = np.random.default_rng(28)
    t = rng.integers(0, 2, (300, 40, 40), dtype=np.uint8)
    codebook = rng.uniform(0.05, 0.9, 512)
    reads = simulate_reads(t, codebook, 1, rng)[0]
    c = two_point(t, reads, np.stack([codebook, codebook]), np.arange(300) % 2, LAGS)
    assert c.shape == (300, 4)
    assert np.abs(c.mean(0)).max() < 0.01


def test_two_point_sees_local_and_image_correlation():
    rng = np.random.default_rng(29)
    t = rng.integers(0, 2, (300, 40, 40), dtype=np.uint8)
    codebook = np.full(512, 0.3)
    cbs = np.stack([codebook, codebook])
    folds = np.arange(300) % 2
    # Image effect: a whole image flips more or less, so even far pairs correlate.
    e = 0.1 * rng.choice([-1.0, 1.0], size=(300, 1, 1))
    img = t.copy()
    interior(img)[...] ^= (rng.random((300, 38, 38)) < 0.3 + e).astype(np.uint8)
    far = two_point(t, img, cbs, folds, LAGS).mean(0)
    assert far[2] == pytest.approx(0.01 / 0.21, abs=0.01)  # e^2 / (p (1 - p))
    # Local effect: horizontal neighbours flip together.
    loc = t.copy()
    f = rng.random((300, 38, 19)) < 0.3
    interior(loc)[...] ^= np.repeat(f, 2, axis=2).astype(np.uint8)
    near = two_point(t, loc, cbs, folds, LAGS).mean(0)
    assert near[0] > 0.3 and abs(near[2]) < 0.01


def test_a_subset_of_templates_keeps_its_fold_codebook():
    rng = np.random.default_rng(30)
    t = rng.integers(0, 2, (40, 30, 30), dtype=np.uint8)
    reads = simulate_reads(t, np.full(512, 0.3), 1, rng)[0]
    cbs = np.stack([np.full(512, 0.1), np.full(512, 0.5)])
    folds = np.arange(40) % 2
    full = two_point(t, reads, cbs, folds, LAGS)
    odd = two_point(t[1::2], reads[1::2], cbs, folds[1::2], LAGS)  # only fold 1
    np.testing.assert_allclose(odd, full[1::2])
    with pytest.raises(ValueError):
        two_point(t, reads, cbs, folds + 5, LAGS)
