"""Error rates, confidence intervals and paired tests.

Thesis convention: scores are oriented so that large = suspect, an image is declared
fake when its score is above the threshold tau, P_miss = P(score > tau | original) and
P_fa = P(score <= tau | fake).
"""

import numpy as np
from scipy import stats


def p_miss(original_scores: np.ndarray, tau: float) -> float:
    """Share of originals rejected (score > tau)."""
    return float(np.mean(np.asarray(original_scores) > tau))


def p_fa(fake_scores: np.ndarray, tau: float) -> float:
    """Share of fakes accepted (score <= tau)."""
    return float(np.mean(np.asarray(fake_scores) <= tau))


def auc(original_scores: np.ndarray, fake_scores: np.ndarray) -> float:
    """P(fake score > original score), ties counted half (Mann-Whitney)."""
    o = np.asarray(original_scores, dtype=np.float64)
    f = np.asarray(fake_scores, dtype=np.float64)
    ranks = stats.rankdata(np.concatenate([o, f]))
    u = ranks[len(o) :].sum() - len(f) * (len(f) + 1) / 2
    return float(u / (len(o) * len(f)))


def p_err_min(original_scores: np.ndarray, fake_scores: np.ndarray) -> float:
    """Oracle error min over tau of (P_miss + P_fa) / 2 (thesis p. 31).

    The threshold is chosen on the very scores it is measured on: report it only to
    compare with the thesis, labelled "oracle".
    """
    o = np.sort(np.asarray(original_scores, dtype=np.float64))
    f = np.sort(np.asarray(fake_scores, dtype=np.float64))
    taus = np.concatenate([[-np.inf], np.unique(np.concatenate([o, f]))])
    miss = 1 - np.searchsorted(o, taus, side="right") / len(o)
    fa = np.searchsorted(f, taus, side="right") / len(f)
    return float(np.min((miss + fa) / 2))


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact confidence interval for a proportion of k successes out of n."""
    if not 0 <= k <= n or n == 0:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Two-sided exact McNemar p-value for two methods judged on the same images.

    correct_a, correct_b: (N,) bool, whether each method got each image right. Only
    images where they disagree count. Not valid on batch_cdp images (D-017).
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    if only_a + only_b == 0:
        return 1.0
    return float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue)


def rate_per_group(flags: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of a 0/1 outcome within each group (e.g. P_fa per template).

    Returns the sorted group labels and their rates.
    """
    labels, inverse = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(flags, dtype=np.float64))
    return labels, sums / np.bincount(inverse)


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean of independent units.

    For batch_cdp, the units are templates: pass one value per template (e.g. its
    P_fa), not one per image.
    """
    v = np.asarray(values, dtype=np.float64)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float(lo), float(hi)


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value on paired values (e.g. P_fa per template).

    1.0 when all pairs are equal.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(d).pvalue)


def design_effect(flags: np.ndarray, groups: np.ndarray) -> float:
    """Variance of the per-group rate over its value under independence.

    deff = Var_groups(rate_g) / (p (1 - p) / m), with p the overall rate and m the mean
    group size. About 1 for independent images; the 50 fakes of a batch_cdp template
    are worth about 50 / deff independent ones.
    """
    labels, rates = rate_per_group(flags, groups)
    p = float(np.mean(np.asarray(flags, dtype=np.float64)))
    if p in (0.0, 1.0):
        return float("nan")
    m = len(np.asarray(flags)) / len(labels)
    return float(np.var(rates, ddof=1) / (p * (1 - p) / m))
