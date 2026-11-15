import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from cdpbpc.metrics import (
    auc,
    bootstrap_mean_ci,
    clopper_pearson,
    design_effect,
    mcnemar_exact,
    p_err_min,
    p_fa,
    p_miss,
    rate_per_group,
    wilcoxon_paired,
)


def test_rates_follow_the_thesis_convention():
    orig = np.array([0.1, 0.2, 0.3, 0.9])
    fake = np.array([0.25, 0.5, 0.8])
    assert p_miss(orig, 0.3) == 0.25  # only 0.9 is above: rejected
    assert p_fa(fake, 0.3) == pytest.approx(1 / 3)  # 0.25 <= tau: accepted
    assert p_fa(fake, 0.25) == pytest.approx(1 / 3)  # equality means accepted


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 10, 200).astype(float)
    fake = rng.integers(3, 13, 150).astype(float)
    labels = np.r_[np.zeros(200), np.ones(150)]
    assert auc(orig, fake) == pytest.approx(roc_auc_score(labels, np.r_[orig, fake]))


def test_p_err_min():
    assert p_err_min([0, 1, 2], [3, 4]) == 0.0  # separable
    assert p_err_min([1, 1], [1, 1]) == 0.5  # identical
    # Brute force over every threshold between and around the values.
    rng = np.random.default_rng(1)
    orig, fake = rng.normal(0, 1, 50), rng.normal(1, 1, 40)
    taus = np.r_[-np.inf, np.sort(np.r_[orig, fake])]
    brute = min((p_miss(orig, t) + p_fa(fake, t)) / 2 for t in taus)
    assert p_err_min(orig, fake) == pytest.approx(brute)


def test_clopper_pearson():
    lo, hi = clopper_pearson(0, 500)
    assert lo == 0 and hi == pytest.approx(0.0074, abs=5e-5)
    lo, hi = clopper_pearson(0, 1152)
    assert hi == pytest.approx(0.0032, abs=5e-5)  # methods.md §11
    lo, hi = clopper_pearson(10, 10)
    assert hi == 1 and 0.6 < lo < 0.8
    lo, hi = clopper_pearson(50, 100)
    assert lo < 0.5 < hi and hi - 0.5 == pytest.approx(0.5 - lo)
    with pytest.raises(ValueError):
        clopper_pearson(3, 2)


def test_mcnemar_exact():
    a = np.array([1, 1, 1, 1, 1, 1, 0, 1], bool)
    b = np.array([0, 0, 0, 0, 0, 0, 0, 1], bool)
    assert mcnemar_exact(a, b) == pytest.approx(2 * 0.5**6)  # 6 discordant pairs, all for a
    assert mcnemar_exact(a, a) == 1.0


def test_rate_per_group():
    labels, rates = rate_per_group([1, 0, 1, 1, 0, 0], [7, 7, 3, 3, 9, 9])
    assert labels.tolist() == [3, 7, 9] and rates.tolist() == [1.0, 0.5, 0.0]


def test_template_bootstrap_is_wider_under_intra_template_correlation():
    rng = np.random.default_rng(2)
    n_templates, per = 40, 50
    p_template = rng.beta(2, 2, n_templates)  # strong template effect
    flags = rng.random((n_templates, per)) < p_template[:, None]
    groups = np.repeat(np.arange(n_templates), per)
    _, per_template = rate_per_group(flags.ravel(), groups)
    lo_t, hi_t = bootstrap_mean_ci(per_template, np.random.default_rng(3))
    lo_i, hi_i = bootstrap_mean_ci(flags.ravel().astype(float), np.random.default_rng(3))
    assert hi_t - lo_t > 3 * (hi_i - lo_i)
    assert lo_t < flags.mean() < hi_t


def test_design_effect():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(200), 50)
    independent = rng.random(groups.size) < 0.3
    assert design_effect(independent, groups) == pytest.approx(1.0, abs=0.25)
    p_template = rng.beta(2, 2, 200)
    correlated = rng.random(groups.size) < p_template[groups]
    assert design_effect(correlated, groups) > 5
    assert np.isnan(design_effect(np.zeros(10), np.arange(10) // 5))


def test_wilcoxon_paired():
    a = np.arange(20) / 20
    assert wilcoxon_paired(a, a) == 1.0
    assert wilcoxon_paired(a + 0.1, a) < 0.001
