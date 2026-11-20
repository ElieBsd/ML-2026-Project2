import numpy as np
import pytest

from cdpbpc.codebook import fit_codebook
from cdpbpc.context import (
    LEVELS,
    TOP_BITS,
    backoff_predict,
    binomial_rows,
    calibration,
    ceiling_log_loss,
    context_index,
    crop,
    fit_ring_logistic,
    fit_trees,
    log_loss,
    observed_disagreement,
    parent_index,
    per_template_disagreement,
    per_template_log_loss,
    per_template_predicted_disagreement,
    predicted_disagreement,
    remaining_rho,
    ring_features,
    select_masses,
    table_counts,
    trees_predict,
    window_bits,
)
from cdpbpc.patterns import interior, pattern_index
from cdpbpc.simulate import simulate_flip_counts


def _templates(seed, n=200, side=20):
    rng = np.random.default_rng(seed)
    return rng, rng.integers(0, 2, (n, side, side), dtype=np.uint8)


def _ring_dependent_flips(t, n, rng, margin=2):
    """K over n reads where a symbol flips with 0.05 if the symbol two rows above is
    white and 0.4 otherwise: a dependence the 3x3 table cannot see."""
    above = np.zeros(t.shape, bool)
    above[:, 2:, :] = t[:, :-2, :] == 1
    p = np.where(above, 0.05, 0.4)
    K = rng.binomial(n, p).astype(np.uint8)
    K[:, :margin], K[:, -margin:], K[:, :, :margin], K[:, :, -margin:] = 0, 0, 0, 0
    return K


def test_levels_follow_the_spec():
    assert [len(LEVELS[b]) for b in (9, 13, 21, 25)] == [9, 13, 21, 25]
    assert LEVELS[9] == [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    assert LEVELS[13][9:] == [(-2, 0), (0, -2), (0, 2), (2, 0)]
    assert LEVELS[25][21:] == [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    full = {(a, b) for a in range(-2, 3) for b in range(-2, 3)}
    assert set(LEVELS[25]) == full and len(set(LEVELS[25])) == 25


def test_nine_bit_index_is_the_3x3_pattern_index():
    _, t = _templates(0, n=5)
    np.testing.assert_array_equal(context_index(t, 9, margin=1), pattern_index(t))
    # Same positions at a larger margin: the interior of the interior.
    np.testing.assert_array_equal(context_index(t, 9, margin=2), interior(pattern_index(t)))


def test_index_of_the_parent_is_the_index_of_the_child_shifted():
    _, t = _templates(1, n=10)
    for child, parent in ((13, 9), (21, 13), (25, 21), (25, 9)):
        np.testing.assert_array_equal(
            parent_index(context_index(t, child, 2), child, parent), context_index(t, parent, 2)
        )


def test_index_bits_by_hand():
    t = np.ones((1, 5, 5), np.uint8)
    assert context_index(t, 25, 2).item() == 2**25 - 1
    assert context_index(np.zeros_like(t), 25, 2).item() == 0
    t[0, 0, 2] = 0  # offset (-2, 0): the first of ARMS, bit 25 - 1 - 9 = 15
    assert context_index(t, 25, 2).item() == 2**25 - 1 - 2**15
    with pytest.raises(ValueError):
        context_index(t, 25, 1)  # the 5x5 does not fit at distance 1 from the border


def test_window_bits_match_the_template():
    _, t = _templates(2, n=3, side=12)
    F = window_bits(t, radius=3, margin=3)
    assert F.shape == (3 * 6 * 6, 49) and F.dtype == np.uint8
    np.testing.assert_array_equal(F[:, 24], crop(t, 3).ravel())  # centre
    np.testing.assert_array_equal(F[:, 0], t[:, 0:6, 0:6].ravel())  # offset (-3, -3)
    # The 5x5 window holds the same bits as the 25-bit index, in another order.
    F5 = window_bits(t, radius=2, margin=3)
    idx = context_index(t, 25, 3).ravel()
    for k, (a, b) in enumerate(LEVELS[25]):
        col = (a + 2) * 5 + (b + 2)
        np.testing.assert_array_equal(F5[:, col], (idx >> (24 - k)) & 1)


def test_nine_bit_table_is_the_jeffreys_codebook():
    rng, t = _templates(3)
    codebook = rng.uniform(0.02, 0.6, 512)
    K = simulate_flip_counts(t, codebook, 1, rng)
    counts = table_counts(t, K, 1, margin=1, bits=9)
    t_hat = (t ^ K).astype(np.uint8)  # one read: the flips themselves
    cb = fit_codebook(t, t_hat)
    idx = context_index(t, 9, 1)
    q = backoff_predict(counts, {}, idx, bits=9, top=9)
    np.testing.assert_allclose(q, cb[idx])


def test_counts_of_a_parent_are_the_sums_of_its_children():
    rng, t = _templates(4, n=50)
    K = simulate_flip_counts(t, rng.uniform(0.05, 0.5, 512), 3, rng)
    c = table_counts(t, K, 3, margin=2)
    for child, parent in ((13, 9), (21, 13), (25, 21)):
        np.testing.assert_array_equal(c.k[child].reshape(2**parent, -1).sum(1), c.k[parent])
    assert c.n[9].sum() == 3 * 50 * 16 * 16  # n reads of every position at distance >= 2
    assert c.k[25].sum() == crop(K, 2).sum()


def test_infinite_backoff_mass_gives_the_3x3_table():
    rng, t = _templates(5)
    K = _ring_dependent_flips(t, 2, rng)
    counts = table_counts(t, K, 2, margin=2)
    idx = context_index(t, TOP_BITS, 3)
    p9 = backoff_predict(counts, {}, idx, bits=9)
    for b in (13, 21, 25):
        q = backoff_predict(counts, {13: np.inf, 21: np.inf, 25: np.inf}, idx, bits=b)
        np.testing.assert_allclose(q, p9)


def test_backoff_formula_by_hand():
    rng, t = _templates(6, n=30)
    K = _ring_dependent_flips(t, 1, rng)
    counts = table_counts(t, K, 1, margin=2)
    idx = context_index(t, 25, 2).ravel()[:50]
    m = {13: 5.0, 21: 20.0, 25: 50.0}
    p = (counts.k[9][idx >> 16] + 0.5) / (counts.n[9][idx >> 16] + 1)
    for b, par in ((13, 9), (21, 13), (25, 21)):
        c = idx >> (25 - b)
        p = (counts.k[b][c] + m[b] * p) / (counts.n[b][c] + m[b])
    np.testing.assert_allclose(backoff_predict(counts, m, idx, bits=25), p)


def test_5x5_table_beats_3x3_on_held_out_templates_when_the_ring_matters():
    rng, t = _templates(7, n=400, side=24)
    K = _ring_dependent_flips(t, 1, rng)
    tr, va, te = slice(0, 250), slice(250, 320), slice(320, 400)
    counts = table_counts(t[tr], K[tr], 1, margin=2)
    masses, _ = select_masses([(counts, t[va], K[va], 1)])
    assert set(masses) == {13, 21, 25}
    idx = context_index(t[te], TOP_BITS, 3)
    Kte = crop(K[te], 3)
    ll9 = log_loss(Kte, 1, backoff_predict(counts, masses, idx, bits=9))
    ll25 = log_loss(Kte, 1, backoff_predict(counts, masses, idx, bits=25))
    assert ll25 < 0.9 * ll9


def test_selected_masses_minimize_the_validation_loss():
    rng, t = _templates(8, n=200, side=16)
    K = _ring_dependent_flips(t, 1, rng)
    counts = table_counts(t[:150], K[:150], 1, margin=2)
    masses, losses = select_masses([(counts, t[150:], K[150:], 1)], grid=(2, 50, 500))
    for b in (13, 21, 25):
        assert masses[b] in (2, 50, 500)
        assert losses[b][masses[b]] == min(losses[b].values())


def test_log_loss_by_hand():
    K = np.array([1, 0])
    q = np.array([0.5, 0.1])
    expected = -((np.log(0.5) + np.log(0.5)) + 2 * np.log(0.9)) / 2 / 2
    assert log_loss(K, 2, q) == pytest.approx(expected)
    # Per template: the mean over its positions.
    Kt = np.array([[[1, 0]], [[0, 0]]])
    qt = np.array([[[0.5, 0.1]], [[0.1, 0.1]]])
    per = per_template_log_loss(Kt, 2, qt)
    assert per[0] == pytest.approx(expected)
    assert per.mean() == pytest.approx(log_loss(Kt, 2, qt))
    # q is kept in [1e-5, 1 - 1e-5]: a certain miss costs log(1e5), not infinity.
    assert log_loss(np.array([1]), 1, np.array([0.0])) == pytest.approx(np.log(1e5))


def test_disagreement_and_rho_are_zero_under_the_bpc():
    rng, t = _templates(9, n=300, side=24)
    codebook = rng.uniform(0.02, 0.6, 512)
    n = 10
    K = interior(simulate_flip_counts(t, codebook, n, rng))
    q = codebook[pattern_index(t)]
    assert observed_disagreement(K, n) == pytest.approx(predicted_disagreement(q), rel=0.01)
    assert abs(remaining_rho(K, n, q)) < 0.01
    with pytest.raises(ValueError):
        remaining_rho(K, 1, q)  # needs two reads at least


def test_per_template_disagreement_with_many_reads_in_uint8():
    # 50 reads: 2 K (n - K) reaches 1250, beyond uint8. K = 25 of 50 gives 2 * 25 * 25 / (50 * 49).
    K = np.full((2, 3, 3), 25, np.uint8)
    K[1] = 0
    per = per_template_disagreement(K, 50)
    np.testing.assert_allclose(per, [2 * 25 * 25 / (50 * 49), 0.0])
    assert per.mean() == pytest.approx(observed_disagreement(K, 50))
    q = np.stack([np.full((3, 3), 0.5), np.full((3, 3), 0.1)])
    np.testing.assert_allclose(per_template_predicted_disagreement(q), [0.5, 0.18])
    assert per_template_predicted_disagreement(q).mean() == pytest.approx(predicted_disagreement(q))


def test_ceiling_tends_to_the_model_when_a_grows():
    rng, t = _templates(10, n=100, side=16)
    K = _ring_dependent_flips(t, 5, rng)
    q = np.full(K.shape, 0.2)
    assert ceiling_log_loss(K, 5, q, 1e9) == pytest.approx(log_loss(K, 5, q), rel=1e-6)
    # One position read 3 times with 1 flip, q = 0.5, a = 1: the flipped read is predicted
    # (0 + 0.5) / 3 from the other two, each clean read (1 + 0.5) / 3.
    by_hand = -(np.log(0.5 / 3) + 2 * np.log(1 - 1.5 / 3)) / 3
    assert ceiling_log_loss(np.array([1]), 3, np.array([0.5]), 1.0) == pytest.approx(by_hand)


def test_calibration_bins():
    K = np.array([0, 1, 1, 2])
    q = np.array([0.005, 0.02, 0.6, 0.6])
    cal = calibration(K, 2, q, edges=(0.0, 0.01, 0.5, 1.0))
    assert [c["share"] for c in cal] == [0.25, 0.25, 0.5]
    assert cal[2]["mean_q"] == pytest.approx(0.6) and cal[2]["observed"] == pytest.approx(0.75)


def test_binomial_rows_carry_the_counts():
    F = np.array([[0, 1], [1, 1], [1, 0]], np.uint8)
    K = np.array([0, 2, 3])
    X, y, w = binomial_rows(F, K, 3)
    # A position gives a flip row (weight K) if K > 0 and a clean row (n - K) if K < n.
    assert len(X) == 4
    assert w[y == 1].sum() == K.sum() and w[y == 0].sum() == (3 - K).sum()
    np.testing.assert_array_equal(X[y == 1], F[[1, 2]])


def test_ring_features():
    rng, t = _templates(11, n=4, side=10)
    idx = context_index(t, 25, 3).ravel()
    p9 = rng.uniform(0.01, 0.9, idx.shape)
    F = ring_features(idx, p9)
    assert F.shape == (len(idx), 34)
    np.testing.assert_allclose(F[:, 0], np.log(p9 / (1 - p9)), rtol=1e-5)
    centre = crop(t, 3).ravel()
    np.testing.assert_array_equal(F[:, 33], centre)
    np.testing.assert_array_equal(F[:, 17:33], F[:, 1:17] * centre[:, None])
    ring = window_bits(t, 2, 3)[:, [(a + 2) * 5 + b + 2 for a, b in LEVELS[25][9:]]]
    np.testing.assert_array_equal(F[:, 1:17], ring)


def test_logistic_and_trees_learn_a_ring_dependence():
    rng, t = _templates(12, n=160, side=20)
    K = _ring_dependent_flips(t, 1, rng, margin=3)
    tr, te = slice(0, 120), slice(120, 160)
    counts = table_counts(t[tr], K[tr], 1, margin=2)
    idx_tr, idx_te = context_index(t[tr], 25, 3).ravel(), context_index(t[te], 25, 3).ravel()
    Ktr, Kte = crop(K[tr], 3).ravel(), crop(K[te], 3).ravel()
    p9_tr = backoff_predict(counts, {}, idx_tr, bits=9)
    p9_te = backoff_predict(counts, {}, idx_te, bits=9)
    ll9 = log_loss(Kte, 1, p9_te)

    lr = fit_ring_logistic(ring_features(idx_tr, p9_tr), Ktr, 1, C=1.0)
    q_lr = lr.predict_proba(ring_features(idx_te, p9_te))[:, 1]
    assert log_loss(Kte, 1, q_lr) < 0.9 * ll9

    trees = fit_trees(window_bits(t[tr], 2, 3), Ktr, 1, max_iter=30, min_samples_leaf=20, seed=0)
    q_gb = trees_predict(trees, window_bits(t[te], 2, 3), n_iter=30)
    assert log_loss(Kte, 1, q_gb) < 0.9 * ll9
    # Fewer iterations: the first stages of the same model.
    np.testing.assert_allclose(
        trees_predict(trees, window_bits(t[te], 2, 3), n_iter=5),
        list(trees.staged_predict_proba(window_bits(t[te], 2, 3)))[4][:, 1],
    )
