"""Diagnostics of the BPC model from repeated reads of the same templates (originals only).

A read is one capture of one print. If a position flips with probability pi, two
independent reads of it disagree with probability 2 pi (1 - pi). The BPC model says
pi = P_b(omega) for every position with pattern omega. If pi in fact varies around P_b
from position to position, reads agree more often than the BPC predicts, and

    rho = 1 - D_obs / D_bpc

measures the share of the BPC noise that is reproducible from one read to the next
(0 under the BPC; methods.md §6.1). It does not depend on the number of reads.

The codebook that judges a template is always learned on other templates: templates
are split into two halves by rank (even, odd), each judged with the other's codebook.
"""

import numpy as np

from cdpbpc.patterns import N_PATTERNS, interior, pattern_index


def flip_counts(t: np.ndarray, reads: list[np.ndarray]) -> np.ndarray:
    """K = number of reads in which each symbol flipped.

    t: (N, side, side) templates; reads: list of (N, side, side) binarized images, each
    aligned with t (same templates, same order). Returns (N, side, side) uint8.
    """
    if not reads or len(reads) > 255:
        raise ValueError("need between 1 and 255 reads")
    K = np.zeros(np.shape(t), dtype=np.uint8)
    for r in reads:
        if np.shape(r) != np.shape(t):
            raise ValueError(f"read of shape {np.shape(r)} does not match templates {np.shape(t)}")
        K += np.asarray(r) != np.asarray(t)
    return K


def _half_sums(t, K, n, K_b, n_b, chunk):
    """Per-pattern sums over interior positions, for each half of the templates.

    Returns a dict of (2, 512) arrays: L positions, S1 flips, SD within-a pair
    disagreements; with K_b also S1b, SDb, and SX cross a-b disagreements.
    """
    names = ["L", "S1", "SD"] + (["S1b", "SDb", "SX"] if K_b is not None else [])
    out = {name: np.zeros((2, N_PATTERNS)) for name in names}
    for start in range(0, len(t), chunk):
        sl = slice(start, start + chunk)
        idx = pattern_index(t[sl]).astype(np.int64)
        half = (np.arange(start, start + len(idx)) % 2)[:, None, None]
        flat = (idx + N_PATTERNS * half).ravel()
        k = interior(K[sl]).astype(np.float64).ravel()
        terms = {"L": np.ones_like(k), "S1": k}
        if n > 1:
            terms["SD"] = 2 * k * (n - k) / (n * (n - 1))
        if K_b is not None:
            kb = interior(K_b[sl]).astype(np.float64).ravel()
            terms["S1b"] = kb
            terms["SX"] = (k * (n_b - kb) + kb * (n - k)) / (n * n_b)
            if n_b > 1:
                terms["SDb"] = 2 * kb * (n_b - kb) / (n_b * (n_b - 1))
        for name, w in terms.items():
            out[name] += np.bincount(flat, weights=w, minlength=2 * N_PATTERNS).reshape(2, -1)
    return out


def reproducibility(
    t: np.ndarray,
    K: np.ndarray,
    n: int,
    K_b: np.ndarray | None = None,
    n_b: int | None = None,
    chunk: int = 64,
) -> dict:
    """Disagreement statistics of methods.md §6.1, with a cross-fitted codebook.

    t: (N, side, side) templates; K: flip counts over n reads of condition a (flip_counts);
    K_b, n_b: the same for a second condition b on the same templates.
    Returns D_obs, D_bpc and rho for a (D_obs and rho need n > 1). With b: X_obs, X_bpc,
    shared = covariance of the position effects of a and b, var_a and var_b (their
    variances, when n, n_b > 1), and shared / var_a (1 if a and b share the same
    reproducible pattern).
    """
    s = _half_sums(np.asarray(t), K, n, K_b, n_b, chunk)
    tot = dict.fromkeys(("L", "pred", "obs", "xpred", "xobs", "predb", "obsb"), 0.0)
    for g in (0, 1):
        train, test = 1 - g, g  # judge half g with the codebook of the other half
        p = (s["S1"][train] + 0.5) / (n * s["L"][train] + 1.0)
        tot["L"] += s["L"][test].sum()
        tot["pred"] += (s["L"][test] * 2 * p * (1 - p)).sum()
        tot["obs"] += s["SD"][test].sum()
        if K_b is not None:
            pb = (s["S1b"][train] + 0.5) / (n_b * s["L"][train] + 1.0)
            tot["xpred"] += (s["L"][test] * (p + pb - 2 * p * pb)).sum()
            tot["xobs"] += s["SX"][test].sum()
            tot["predb"] += (s["L"][test] * 2 * pb * (1 - pb)).sum()
            tot["obsb"] += s["SDb"][test].sum()
    L = tot["L"]
    res = {"n_templates": len(t), "n_reads": n, "D_bpc": tot["pred"] / L}
    if n > 1:
        res.update(D_obs=tot["obs"] / L, rho=1 - tot["obs"] / tot["pred"])
        res["var_a"] = (tot["pred"] - tot["obs"]) / (2 * L)
    if K_b is not None:
        res.update(n_reads_b=n_b, X_obs=tot["xobs"] / L, X_bpc=tot["xpred"] / L)
        res["shared"] = (tot["xpred"] - tot["xobs"]) / (2 * L)
        if n_b > 1:
            res["var_b"] = (tot["predb"] - tot["obsb"]) / (2 * L)
        if n > 1:
            res["shared_over_var_a"] = res["shared"] / res["var_a"]
    return res


def icc_oneway(values: np.ndarray, groups: np.ndarray) -> float:
    """Intraclass correlation from a one-way ANOVA with equal-sized groups.

    Share of the variance of values due to the group (e.g. the template): about 0 when
    the group does not matter, at most 1.
    """
    labels, inverse = np.unique(groups, return_inverse=True)
    k, n = len(labels), len(values)
    m = n / k
    means = np.bincount(inverse, weights=values) / np.bincount(inverse)
    msb = m * ((means - values.mean()) ** 2).sum() / (k - 1)
    msw = ((values - means[inverse]) ** 2).sum() / (n - k)
    return float((msb - msw) / (msb + (m - 1) * msw))


def adjusted_eta2(values: np.ndarray, labels: np.ndarray) -> float:
    """Adjusted share of the variance of values explained by a grouping (methods.md §6.3).

    Corrected for the number of groups, so a grouping that explains nothing gives about 0.
    """
    u, inverse = np.unique(labels, return_inverse=True)
    centred = values - values.mean()
    means = np.bincount(inverse, weights=centred) / np.bincount(inverse)
    ssb = (np.bincount(inverse) * means**2).sum()
    sst = (centred**2).sum()
    n, k = len(values), len(u)
    return float(1 - (1 - ssb / sst) * (n - 1) / (n - k))


def two_way_components(H: np.ndarray) -> dict:
    """Variance components of a templates x reads table without replication (methods.md §6.3).

    H: (n_templates, n_reads), e.g. the flip rate of each template in each print run.
    Returns the standard deviations of the template, read and residual effects, in points
    (x 100), estimated by moments.
    """
    n, r = H.shape
    grand = H.mean()
    row = H.mean(axis=1, keepdims=True)
    col = H.mean(axis=0, keepdims=True)
    ms_row = r * ((row - grand) ** 2).sum() / (n - 1)
    ms_col = n * ((col - grand) ** 2).sum() / (r - 1)
    ms_res = ((H - row - col + grand) ** 2).sum() / ((n - 1) * (r - 1))
    return {
        "sd_template_pts": 100 * float(np.sqrt(max(ms_row - ms_res, 0) / r)),
        "sd_read_pts": 100 * float(np.sqrt(max(ms_col - ms_res, 0) / n)),
        "sd_residual_pts": 100 * float(np.sqrt(ms_res)),
    }
