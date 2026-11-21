"""Diagnostics of the BPC model on originals.

Locality. A read is one capture of one print. If a position flips with
probability pi, two independent reads of it disagree with probability 2 pi (1 - pi).
The BPC says pi = P_b(omega) for every position with pattern omega. If pi in fact varies
around P_b from position to position, reads agree more often than the BPC predicts, and

    rho = 1 - D_obs / D_bpc

is the share of the BPC noise that is reproducible from one read to the next (0 under
the BPC). It does not depend on the number of reads. Between two conditions a and b,
the covariance of their position effects over the variance in a is the shared part.

Independence. sigma_image is the spread of the flip rate between images beyond
what the BPC predicts for each image, in points of flip rate.

Stationarity and two-point correlation (Tutt 2025, p. 138) work on positions.

The codebook that judges a template is always learned on other templates: templates are
split into folds (two halves by rank by default, groups of sheets for batch_cdp), and
each fold is judged with the codebook of the others. Everything that can be is computed
from per-template, per-pattern sums, so that a bootstrap on templates only reweights
rows (weights = how many times each template is drawn).
"""

import numpy as np
from scipy.stats import binom

from cdpbpc.patterns import N_PATTERNS, interior, pattern_index
from cdpbpc.scores import CLASS_EDGES_8, reliability_class


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


# ----------------------------------------------------------------------------- locality


def pattern_sums(
    t: np.ndarray,
    K: np.ndarray,
    n: int,
    K_b: np.ndarray | None = None,
    n_b: int | None = None,
    chunk: int = 64,
) -> dict:
    """Per-template, per-pattern sums over interior positions.

    t: (N, side, side) templates; K: flip counts over n reads of condition a; K_b, n_b:
    the same for condition b on the same templates. Returns (N, 512) float64 arrays:
    L positions, S1 flips, SD within-a pair disagreements 2K(n-K)/(n(n-1)) (n > 1); with
    b also S1b, SDb (n_b > 1) and SX cross a-b disagreements.
    """
    t = np.asarray(t)
    names = ["L", "S1"] + (["SD"] if n > 1 else [])
    if K_b is not None:
        names += ["S1b", "SX"] + (["SDb"] if n_b > 1 else [])
    out = {name: np.zeros((len(t), N_PATTERNS)) for name in names}
    for start in range(0, len(t), chunk):
        sl = slice(start, start + chunk)
        idx = pattern_index(t[sl]).astype(np.int64)
        m = len(idx)
        flat = (idx + N_PATTERNS * np.arange(m)[:, None, None]).ravel()
        k = interior(K[sl]).astype(np.float64).ravel()
        terms = {"L": None, "S1": k}
        if n > 1:
            terms["SD"] = 2 * k * (n - k) / (n * (n - 1))
        if K_b is not None:
            kb = interior(K_b[sl]).astype(np.float64).ravel()
            terms["S1b"] = kb
            terms["SX"] = (k * (n_b - kb) + kb * (n - k)) / (n * n_b)
            if n_b > 1:
                terms["SDb"] = 2 * kb * (n_b - kb) / (n_b * (n_b - 1))
        for name, w in terms.items():
            out[name][sl] = np.bincount(flat, weights=w, minlength=m * N_PATTERNS).reshape(m, -1)
    return out


def _fold_totals(sums: dict, folds: np.ndarray, weights: np.ndarray) -> dict:
    """Weighted sums of each key over the templates of each fold: (n_folds, 512)."""
    labels = np.unique(folds)
    onehot = (folds[:, None] == labels[None, :]) * weights[:, None]
    return {k: onehot.T @ v for k, v in sums.items()}


def _prepare(sums, folds, weights):
    n_rows = len(sums["L"])
    folds = np.arange(n_rows) % 2 if folds is None else np.asarray(folds)
    weights = np.ones(n_rows) if weights is None else np.asarray(weights, dtype=np.float64)
    return folds, weights


def cross_fitted_codebooks(
    sums: dict, n: int, folds: np.ndarray | None = None, weights: np.ndarray | None = None
) -> np.ndarray:
    """Codebook that judges each fold, learned on the other folds (Jeffreys prior).

    sums: output of pattern_sums (S1 and L are used). Returns (n_folds, 512), one row per
    fold label in sorted order.
    """
    folds, weights = _prepare(sums, folds, weights)
    F = _fold_totals({"L": sums["L"], "S1": sums["S1"]}, folds, weights)
    train_S1 = F["S1"].sum(0) - F["S1"]
    train_L = F["L"].sum(0) - F["L"]
    return (train_S1 + 0.5) / (n * train_L + 1.0)


def reproducibility_from_sums(
    sums: dict,
    n: int,
    n_b: int | None = None,
    folds: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    codebook_b: np.ndarray | None = None,
    edges=CLASS_EDGES_8,
) -> dict:
    """Disagreement statistics (rho, shared part) from pattern_sums, cross-fitted by fold.

    folds: (N,) fold of each template (default: two halves by rank); weights: (N,)
    template multiplicities (bootstrap); codebook_b: a fixed codebook for condition b,
    used instead of the cross-fitted one (e.g. learned on other templates of b).
    Returns D_obs, D_bpc, rho, var_a and rho by reliability class (classes of the
    cross-fitted P_b of a); with b: X_obs, X_bpc, shared (covariance of the position
    effects), var_b (n_b > 1) and shared_over_var_a.
    """
    folds, weights = _prepare(sums, folds, weights)
    F = _fold_totals(sums, folds, weights)
    total = {k: v.sum(0) for k, v in F.items()}
    has_b = "S1b" in sums
    n_cls = len(edges) - 1
    acc = {k: np.zeros(n_cls) for k in ("L", "pred", "obs", "xpred", "xobs", "predb", "obsb")}
    for g in range(len(F["L"])):
        train = {k: total[k] - F[k][g] for k in ("L", "S1", "S1b") if k in F}
        test = {k: v[g] for k, v in F.items()}
        p = (train["S1"] + 0.5) / (n * train["L"] + 1.0)
        cls = reliability_class(p, edges)
        terms = {"L": test["L"], "pred": test["L"] * 2 * p * (1 - p)}
        if n > 1:
            terms["obs"] = test["SD"]
        if has_b:
            if codebook_b is not None:
                pb = np.asarray(codebook_b, dtype=np.float64)
            else:
                pb = (train["S1b"] + 0.5) / (n_b * train["L"] + 1.0)
            terms["xpred"] = test["L"] * (p + pb - 2 * p * pb)
            terms["xobs"] = test["SX"]
            terms["predb"] = test["L"] * 2 * pb * (1 - pb)
            if n_b > 1:
                terms["obsb"] = test["SDb"]
        for k, v in terms.items():
            acc[k] += np.bincount(cls, weights=v, minlength=n_cls)
    tot = {k: v.sum() for k, v in acc.items()}
    L = tot["L"]
    res = {"n_templates": len(folds), "n_reads": n, "D_bpc": tot["pred"] / L}
    res["share_by_class"] = (acc["L"] / L).tolist()
    if n > 1:
        res.update(D_obs=tot["obs"] / L, rho=1 - tot["obs"] / tot["pred"])
        res["var_a"] = (tot["pred"] - tot["obs"]) / (2 * L)
        with np.errstate(invalid="ignore", divide="ignore"):
            res["rho_by_class"] = (1 - acc["obs"] / acc["pred"]).tolist()
    if has_b:
        res.update(n_reads_b=n_b, X_obs=tot["xobs"] / L, X_bpc=tot["xpred"] / L)
        res["shared"] = (tot["xpred"] - tot["xobs"]) / (2 * L)
        if n_b > 1:
            res["var_b"] = (tot["predb"] - tot["obsb"]) / (2 * L)
        if n > 1:
            res["shared_over_var_a"] = res["shared"] / res["var_a"]
    return res


def reproducibility(
    t: np.ndarray,
    K: np.ndarray,
    n: int,
    K_b: np.ndarray | None = None,
    n_b: int | None = None,
    chunk: int = 64,
    folds: np.ndarray | None = None,
    codebook_b: np.ndarray | None = None,
) -> dict:
    """Disagreement statistics (rho, shared part), with a cross-fitted codebook.

    t: (N, side, side) templates; K: flip counts over n reads of condition a (flip_counts);
    K_b, n_b: the same for a second condition b on the same templates. See
    reproducibility_from_sums for folds, codebook_b and the returned keys.
    """
    sums = pattern_sums(np.asarray(t), K, n, K_b, n_b, chunk)
    return reproducibility_from_sums(sums, n, n_b, folds=folds, codebook_b=codebook_b)


def simulate_pattern_sums(
    cells: np.ndarray,
    codebook: np.ndarray,
    n: int,
    rng: np.random.Generator,
    codebook_b: np.ndarray | None = None,
    n_b: int | None = None,
) -> dict:
    """pattern_sums of BPC reads, drawn directly at the level of cells.

    cells: (M, 512) number of positions of each pattern in each cell (e.g. the positions of
    one fold). Under the BPC all positions of a pattern are exchangeable: only how many of
    them have 0, 1, ..., n flips matters, and that is multinomial. This gives the exact law
    of the sums of pattern_sums over the cell, without drawing each symbol. With
    codebook_b and n_b, condition b is drawn independently on the same positions.
    """
    cells = np.asarray(cells, dtype=np.int64)
    codebook = np.asarray(codebook, dtype=np.float64)
    k = np.arange(n + 1, dtype=np.float64)
    pmf = binom.pmf(k[None, :], n, codebook[:, None])  # (512, n + 1)
    values = {"S1": k}
    if n > 1:
        values["SD"] = 2 * k * (n - k) / (n * (n - 1))
    if codebook_b is not None:
        kb = np.arange(n_b + 1, dtype=np.float64)
        pmf_b = binom.pmf(kb[None, :], n_b, np.asarray(codebook_b, np.float64)[:, None])
        pmf = (pmf[:, :, None] * pmf_b[:, None, :]).reshape(N_PATTERNS, -1)
        ka, kb = np.repeat(k, n_b + 1), np.tile(kb, n + 1)
        values = {"S1": ka, "S1b": kb, "SX": (ka * (n_b - kb) + kb * (n - ka)) / (n * n_b)}
        if n > 1:
            values["SD"] = 2 * ka * (n - ka) / (n * (n - 1))
        if n_b > 1:
            values["SDb"] = 2 * kb * (n_b - kb) / (n_b * (n_b - 1))
    pmf /= pmf.sum(1, keepdims=True)
    draws = rng.multinomial(cells.ravel(), np.tile(pmf, (len(cells), 1)))
    draws = draws.reshape(len(cells), N_PATTERNS, -1).astype(np.float64)
    out = {"L": cells.astype(np.float64)}
    out.update({name: draws @ v for name, v in values.items()})
    return out


# ----------------------------------------------------------------------------- image effect


def residual_flip_rate(
    D: np.ndarray, L: np.ndarray, folds: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Flip rate of each image minus the rate the BPC predicts for its template.

    D, L: (N, 512) pattern counts of one read per template, or (N, R, 512) for R reads of
    the same N templates (a table). r = (sum D - sum L p) / sum L, with p the codebook of
    the other folds, pooled over all reads; also the binomial variance of the flip rate
    sum L p (1 - p) / (sum L)^2. Returns r, var_binomial: (N,) or (N, R).
    """
    D = np.asarray(D, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    folds = np.arange(len(D)) % 2 if folds is None else np.asarray(folds)
    per_template = {
        "S1": D.reshape(len(D), -1, N_PATTERNS).sum(1),
        "L": L.reshape(len(L), -1, N_PATTERNS).sum(1),
    }
    codebooks = cross_fitted_codebooks(per_template, 1, folds)
    p = codebooks[np.searchsorted(np.unique(folds), folds)]  # (N, 512)
    if D.ndim == 3:
        p = p[:, None, :]
    B = L.sum(-1)
    r = (D.sum(-1) - (L * p).sum(-1)) / B
    var_binomial = (L * p * (1 - p)).sum(-1) / B**2
    return r, var_binomial


def _weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    """Sample variance of x with integer multiplicities w (ddof = 1 on sum w)."""
    m = (w * x).sum() / w.sum()
    return float((w * (x - m) ** 2).sum() / (w.sum() - 1))


def sigma_image(r: np.ndarray, var_binomial: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Image effect sigma_image from residual flip rates (one read per template).

    r, var_binomial: (N,) from residual_flip_rate. var_image = Var(r) - mean binomial
    variance (may be < 0 by chance); sd_image = sqrt(max(var_image, 0)). Standard
    deviations in points (x 100) and their ratios to the binomial one.
    """
    w = np.ones(len(r)) if weights is None else np.asarray(weights, dtype=np.float64)
    var_total = _weighted_var(np.asarray(r), w)
    var_bin = float((w * var_binomial).sum() / w.sum())
    var_image = var_total - var_bin
    sd_bin = np.sqrt(var_bin)
    return {
        "var_image": var_image,
        "sd_image_pts": 100 * np.sqrt(max(var_image, 0.0)),
        "sd_binomial_pts": 100 * sd_bin,
        "sd_total_pts": 100 * np.sqrt(var_total),
        "sd_image_over_binomial": np.sqrt(max(var_image, 0.0)) / sd_bin,
        "sd_total_over_binomial": np.sqrt(var_total) / sd_bin,
    }


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


def adjusted_eta2(
    values: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Adjusted share of the variance of values explained by a grouping.

    Corrected for the number of groups, so a grouping that explains nothing gives about 0.
    weights: integer multiplicities of the values (bootstrap), as if rows were repeated.
    """
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=np.float64)
    keep = w > 0
    values, labels, w = np.asarray(values)[keep], np.asarray(labels)[keep], w[keep]
    u, inverse = np.unique(labels, return_inverse=True)
    centred = values - (w * values).sum() / w.sum()
    size = np.bincount(inverse, weights=w)
    means = np.bincount(inverse, weights=w * centred) / size
    ssb = (size * means**2).sum()
    sst = (w * centred**2).sum()
    n, k = w.sum(), len(u)
    return float(1 - (1 - ssb / sst) * (n - 1) / (n - k))


def two_way_components(H: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Variance components of a templates x reads table without replication.

    H: (n_templates, n_reads), e.g. the residual flip rate of each template in each print
    run. weights: (n_templates,) multiplicities (bootstrap). Returns the standard
    deviations of the template, read and residual effects, in points (x 100), by moments.
    """
    H = np.asarray(H, dtype=np.float64)
    w = np.ones(len(H)) if weights is None else np.asarray(weights, dtype=np.float64)
    n, r = w.sum(), H.shape[1]
    col = (w[:, None] * H).sum(0, keepdims=True) / n
    grand = col.mean()
    row = H.mean(axis=1, keepdims=True)
    ms_row = r * (w[:, None] * (row - grand) ** 2).sum() / (n - 1)
    ms_col = n * ((col - grand) ** 2).sum() / (r - 1)
    ms_res = (w[:, None] * (H - row - col + grand) ** 2).sum() / ((n - 1) * (r - 1))
    return {
        "sd_template_pts": 100 * float(np.sqrt(max(ms_row - ms_res, 0) / r)),
        "sd_read_pts": 100 * float(np.sqrt(max(ms_col - ms_res, 0) / n)),
        "sd_residual_pts": 100 * float(np.sqrt(ms_res)),
    }


def crossed_components(A: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Variance components of a templates x sheets table with copies (Scantrust batch_cdp).

    A: (n_templates, n_sheets, n_copies), e.g. residual flip rates of one group. Random
    effects by moments: template, sheet, template x sheet, copy. weights: (n_templates,)
    multiplicities. Returns standard deviations in points (x 100).
    """
    A = np.asarray(A, dtype=np.float64)
    w = np.ones(len(A)) if weights is None else np.asarray(weights, dtype=np.float64)
    a, (b, c) = w.sum(), A.shape[1:]
    cell = A.mean(2)
    tm = cell.mean(1)
    sm = (w[:, None] * cell).sum(0) / a
    grand = sm.mean()
    ms_t = b * c * (w * (tm - grand) ** 2).sum() / (a - 1)
    ms_s = a * c * ((sm - grand) ** 2).sum() / (b - 1)
    inter = cell - tm[:, None] - sm[None, :] + grand
    ms_ts = c * (w[:, None] * inter**2).sum() / ((a - 1) * (b - 1))
    ms_e = (w[:, None, None] * (A - cell[:, :, None]) ** 2).sum() / (a * b * (c - 1))

    def sd(x):
        return 100 * float(np.sqrt(max(x, 0.0)))

    return {
        "sd_template_pts": sd((ms_t - ms_ts) / (b * c)),
        "sd_sheet_pts": sd((ms_s - ms_ts) / (a * c)),
        "sd_template_x_sheet_pts": sd((ms_ts - ms_e) / c),
        "sd_copy_pts": sd(ms_e),
    }


# ----------------------------------------------------------------------------- positions


def _fold_rows(folds: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """Row of codebooks for each template. Folds must be labelled 0..n_folds-1, the order of
    cross_fitted_codebooks, so that a subset of templates keeps the right codebook."""
    rows = np.asarray(folds)
    if rows.min() < 0 or rows.max() >= len(codebooks) or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("folds must be integer labels 0..n_folds-1 indexing the codebooks")
    return rows


def template_maps(
    t: np.ndarray,
    K: np.ndarray,
    n: int,
    codebooks: np.ndarray,
    folds: np.ndarray,
    chunk: int = 64,
) -> np.ndarray:
    """Residual map of each template: K / n - p at each interior position.

    codebooks: (n_folds, 512) from cross_fitted_codebooks; folds: (N,) labels 0..n_folds-1.
    Returns (N, side-2, side-2) float32.
    """
    rows = _fold_rows(folds, codebooks)
    t = np.asarray(t)
    out = np.empty((len(t),) + interior(t[:1]).shape[1:], dtype=np.float32)
    for start in range(0, len(t), chunk):
        sl = slice(start, start + chunk)
        p = np.asarray(codebooks)[rows[sl]][:, None, :]  # (m, 1, 512)
        idx = pattern_index(t[sl]).astype(np.int64)
        pp = np.take_along_axis(p, idx.reshape(len(idx), 1, -1), axis=2).reshape(idx.shape)
        out[sl] = interior(K[sl]).astype(np.float32) / n - pp
    return out


def map_stats(
    maps: np.ndarray, halves: np.ndarray, weights: np.ndarray | None = None, ring: int = 2
) -> dict:
    """Stationarity statistics of the residual maps.

    maps: (N, s, s) from template_maps; halves: (N,) 0 or 1, the two halves of templates
    whose mean maps are compared. ring: outer rings of the map left out of the "inner"
    statistics. Without weights: correlation between the two half maps (all, inner),
    standard deviation of the mean map (inner), reliable standard deviation sqrt(cov)
    between halves (inner), row and column profiles of the mean map, in points.
    With weights (B, N) (bootstrap): the three inner scalars as (B,) arrays.
    """
    halves = np.asarray(halves)
    N = len(maps)
    flat = maps.reshape(N, -1)
    side = maps.shape[-1]
    inner = np.zeros((side, side), bool)
    inner[ring : side - ring, ring : side - ring] = True
    inner = inner.ravel()
    # Weights in the maps' dtype, so that a float32 stack of maps is never copied to float64.
    W = np.ones((1, N), flat.dtype) if weights is None else np.asarray(weights, flat.dtype)
    half_maps = []
    for h in (0, 1):
        Wh = W * (halves == h)[None]
        hm = (Wh @ flat) / Wh.sum(1, keepdims=True)  # (B, s*s)
        half_maps.append(hm.astype(np.float64))
    a, b = half_maps[0][:, inner], half_maps[1][:, inner]
    ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    cov = (ac * bc).sum(1) / (inner.sum() - 1)
    corr = (ac * bc).sum(1) / np.sqrt((ac**2).sum(1) * (bc**2).sum(1))
    mean_map = (half_maps[0] + half_maps[1]) / 2
    sd_map = 100 * mean_map[:, inner].std(1, ddof=1)
    sd_rel = 100 * np.sqrt(np.maximum(cov, 0))
    if weights is not None:
        return {
            "corr_halves_inner": corr,
            "sd_map_inner_pts": sd_map,
            "sd_reliable_inner_pts": sd_rel,
        }
    a_all, b_all = half_maps[0][0], half_maps[1][0]
    m = mean_map[0].reshape(side, side)
    return {
        "corr_halves_all": float(np.corrcoef(a_all, b_all)[0, 1]),
        "corr_halves_inner": float(corr[0]),
        "sd_map_inner_pts": float(sd_map[0]),
        "sd_reliable_inner_pts": float(sd_rel[0]),
        "row_profile_pts": (100 * m.mean(1)).tolist(),
        "col_profile_pts": (100 * m.mean(0)).tolist(),
    }


TWO_POINT_LAGS = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (0, 3), (3, 0), (0, 20), (20, 0))


def two_point(
    t: np.ndarray,
    t_hat: np.ndarray,
    codebooks: np.ndarray,
    folds: np.ndarray,
    lags=TWO_POINT_LAGS,
    chunk: int = 64,
) -> np.ndarray:
    """Two-point correlation of standardized residuals (Tutt 2025, p. 138), per template.

    r = (delta - p) / sqrt(p (1 - p)) on interior symbols, p from the codebook of the
    template's fold (p bounded in [1e-4, 1 - 1e-4]); for each lag (dy, dx), the mean of
    r_ij r_(i+dy)(j+dx) over the template's positions. t, t_hat: (N, side, side) for one
    read per template; folds: (N,) labels 0..n_folds-1 indexing codebooks. Returns
    (N, n_lags); the dataset value is the mean over templates.
    """
    rows = _fold_rows(folds, codebooks)
    cb = np.clip(np.asarray(codebooks, dtype=np.float64), 1e-4, 1 - 1e-4)
    t = np.asarray(t)
    out = np.empty((len(t), len(lags)))
    for start in range(0, len(t), chunk):
        sl = slice(start, start + chunk)
        idx = pattern_index(t[sl]).astype(np.int64)
        p = np.take_along_axis(cb[rows[sl]][:, None, :], idx.reshape(len(idx), 1, -1), axis=2)
        p = p.reshape(idx.shape).astype(np.float32)
        r = (interior(t[sl] != t_hat[sl]).astype(np.float32) - p) / np.sqrt(p * (1 - p))
        h = r.shape[-1]
        for j, (dy, dx) in enumerate(lags):
            out[sl, j] = (r[:, : h - dy, : h - dx] * r[:, dy:, dx:]).mean((1, 2))
    return out
