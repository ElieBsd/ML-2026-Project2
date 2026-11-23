import numpy as np
import pytest

from cdpbpc.baselines.supervised import (
    fake_codebook,
    fit_logistic,
    grouped_folds,
    logistic_score,
    lrt_plugin,
    select_C,
)
from cdpbpc.codebook import estimate
from cdpbpc.patterns import N_PATTERNS
from cdpbpc.scores import binned_z, lrt
from cdpbpc.simulate import simulate_counts

CS = (0.01, 0.1, 1.0, 10.0)


def _features(seed=0, n_templates=60, per_template=4, shift=1.5, n_fake_templates=None):
    """binned_z of originals (BPC) and of fakes that flip more on a few classes.

    Each template has `per_template` originals and as many fakes. Returns Z, y, groups.
    """
    rng = np.random.default_rng(seed)
    cb = rng.uniform(0.0, 1.0, N_PATTERNS)
    L = rng.integers(20, 60, (n_templates * per_template, N_PATTERNS))
    groups = np.repeat(np.arange(n_templates), per_template)
    D_o = simulate_counts(L, cb, rng)
    q = np.clip(cb + np.where(cb < 0.3, 0.03 * shift, 0.0), 0, 1)
    D_f = simulate_counts(L, q, rng)
    Z = np.concatenate([binned_z(D_o, L, cb), binned_z(D_f, L, cb)])
    y = np.r_[np.zeros(len(D_o), int), np.ones(len(D_f), int)]
    g = np.r_[groups, groups]
    if n_fake_templates is not None:  # unbalanced: fakes of the first templates only
        keep = (y == 0) | (g < n_fake_templates)
        Z, y, g = Z[keep], y[keep], g[keep]
    return Z, y, g


def test_logistic_separates_simulated_fakes():
    Z, y, g = _features(0)
    model, C, cv = fit_logistic(Z, y, g, CS, n_splits=5, seed=0)
    Zt, yt, _ = _features(1)  # new templates, same channel
    s = logistic_score(model, Zt)
    assert s.shape == (len(yt),)
    # large = suspect: fakes score higher
    from cdpbpc.metrics import auc

    assert auc(s[yt == 0], s[yt == 1]) > 0.95
    assert C in CS
    assert set(cv) == set(CS) and all(np.isfinite(v) for v in cv.values())
    assert C == min(CS, key=lambda c: (cv[c], c))


def test_folds_never_split_a_template():
    _, y, g = _features(2, n_templates=23, per_template=3)
    folds = grouped_folds(y, g, n_splits=4, seed=3)
    assert len(folds) == 4
    seen = np.zeros(len(y), int)
    for train, val in folds:
        assert not set(g[train]) & set(g[val])
        seen[val] += 1
        assert set(y[train]) == {0, 1}
    np.testing.assert_array_equal(seen, 1)  # every image validated exactly once


def test_balanced_class_weight_and_deterministic_for_a_seed():
    Z, y, g = _features(4, n_fake_templates=10)  # 6 times more originals than fakes
    model, C, cv = fit_logistic(Z, y, g, CS, n_splits=3, seed=7)
    assert model.named_steps["logisticregression"].class_weight == "balanced"
    model2, C2, cv2 = fit_logistic(Z, y, g, CS, n_splits=3, seed=7)
    assert C2 == C and cv2 == cv
    np.testing.assert_array_equal(logistic_score(model, Z), logistic_score(model2, Z))
    # balanced weights: about as many fakes as originals predicted fake on the training
    # set despite the imbalance (an unweighted fit would call almost everything original)
    pred = logistic_score(model, Z) > 0
    assert pred[y == 1].mean() > 0.6


def test_logistic_refuses_bad_inputs():
    Z, y, g = _features(5, n_templates=12)
    with pytest.raises(ValueError):
        fit_logistic(Z[:, :3][:, :, None], y, g, CS)  # not 2-d
    with pytest.raises(ValueError):
        fit_logistic(Z, y[:-1], g, CS)  # lengths differ
    with pytest.raises(ValueError):
        fit_logistic(Z, y + 1, g, CS)  # labels not 0/1
    with pytest.raises(ValueError):
        fit_logistic(Z, y, g, ())  # empty grid
    with pytest.raises(ValueError):
        fit_logistic(Z, y, g, (1.0, -1.0))  # C must be > 0
    with pytest.raises(ValueError):
        fit_logistic(Z, y, g, CS, n_splits=13)  # more folds than templates
    with pytest.raises(ValueError):
        fit_logistic(Z, np.zeros_like(y), g, CS)  # one class only
    Zn = Z.copy()
    Zn[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_logistic(Zn, y, g, CS)


def test_fake_codebook_is_the_pooled_jeffreys_rate_of_the_fakes():
    rng = np.random.default_rng(6)
    L = rng.integers(0, 30, (40, N_PATTERNS))
    D = rng.binomial(L, 0.2)
    Q = fake_codebook(D, L)
    np.testing.assert_allclose(Q, (D.sum(0) + 0.5) / (L.sum(0) + 1.0))
    np.testing.assert_allclose(Q, estimate(L.sum(0), D.sum(0)))
    L[:, 7] = 0
    D[:, 7] = 0
    assert fake_codebook(D, L)[7] == 0.5  # unseen pattern: Jeffreys prior mean
    with pytest.raises(ValueError):
        fake_codebook(D[:, :10], L[:, :10])
    with pytest.raises(ValueError):
        fake_codebook(L + 1, L)  # more flips than symbols


def test_lrt_plugin_is_zero_when_q_equals_p_and_matches_lrt():
    rng = np.random.default_rng(8)
    P = rng.uniform(0.01, 0.99, N_PATTERNS)
    L = rng.integers(0, 30, (5, N_PATTERNS))
    D = rng.binomial(L, 0.3)
    np.testing.assert_allclose(lrt_plugin(D, L, P, P), 0.0, atol=1e-9)
    Q = np.clip(P + 0.1, 0, 1)
    np.testing.assert_allclose(lrt_plugin(D, L, P, Q), lrt(D, L, P, Q))
    # learned on fakes that flip more: fakes score higher than originals
    D_o = simulate_counts(np.repeat(L[:1], 200, 0), P, rng)
    D_f = simulate_counts(np.repeat(L[:1], 200, 0), Q, rng)
    Qhat = fake_codebook(D_f[:100], np.repeat(L[:1], 100, 0))
    so = lrt_plugin(D_o[100:], np.repeat(L[:1], 100, 0), P, Qhat)
    sf = lrt_plugin(D_f[100:], np.repeat(L[:1], 100, 0), P, Qhat)
    assert sf.mean() > so.mean()
    with pytest.raises(ValueError):
        lrt_plugin(D, L, P[:10], Q)


def test_select_C_takes_the_smallest_loss_and_the_strongest_penalty_on_ties():
    assert select_C({0.1: 0.5, 1.0: 0.4, 10.0: 0.45}) == 1.0
    assert select_C({10.0: 0.4, 0.1: 0.4, 1.0: 0.4}) == 0.1
    with pytest.raises(ValueError):
        select_C({})


def test_returned_model_is_refit_on_all_the_training_data():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Z, y, g = _features(9, n_templates=30)
    model, C, _ = fit_logistic(Z, y, g, CS, n_splits=3, seed=0)
    ref = make_pipeline(
        StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    ).fit(Z, y)
    np.testing.assert_allclose(logistic_score(model, Z), ref.decision_function(Z), atol=1e-6)
