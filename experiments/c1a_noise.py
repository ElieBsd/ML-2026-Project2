"""C1a: what the 3x3 BPC model misses, measured on originals only (methods.md §6).

Locality (§6.1, §6.2): rho, the share of the noise predicted by the BPC that is in fact
reproducible from one read to the next, between prints (Scantrust batch_cdp, Indigo HPI55
and HPI76), between scans of one print and between phone captures; hence the split of
the noise into template, print and capture, and the part of the reproducible pattern
shared between sessions, printers and phones.
Independence (§6.3): sigma_image, the spread of the flip rate between images beyond the
binomial noise, and its components (position in the page, print, scan, capture; sheet
for Scantrust). Stationarity (§6.4): position maps. Two-point correlation (§6.5).

Each statistic comes with a 95 % percentile interval from a bootstrap on templates, and
a null reference: the same computation, codebook included, on 200 draws under the BPC
on the same templates (cdpbpc.simulate, diagnostics.simulate_pattern_sums). The
codebook that judges a template is always learned on other templates.

No fake is read: only files of originals (checked in their metadata) and the digital
templates.

Usage:
    python -m experiments.c1a_noise --config configs/c1a.yaml
    python -m experiments.c1a_noise --figure results/c1a_<hash>.json   # redraw only
Writes results/c1a_<config hash>.json and report/figures/fig1_left.pdf.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from cdpbpc.counts import image_counts, indigo_counts
from cdpbpc.diagnostics import (
    TWO_POINT_LAGS,
    adjusted_eta2,
    cross_fitted_codebooks,
    crossed_components,
    map_stats,
    pattern_sums,
    reproducibility_from_sums,
    residual_flip_rate,
    sigma_image,
    simulate_pattern_sums,
    template_maps,
    two_point,
    two_way_components,
)
from cdpbpc.io import (
    INDIGO_N_TEMPLATES,
    INDIGO_SIDE,
    load_compact,
    load_scantrust,
    read_compact_header,
    read_manifest,
)
from cdpbpc.simulate import simulate_counts, simulate_flip_counts, simulate_reads
from experiments.e0_sanity import _round, git_state

ROOT = Path(__file__).resolve().parents[1]
STAGE = "dev"  # only originals and templates are read; the stage cut does not touch them
INDIGO_PAGE = 144
BATCH_FIRST_ID = 3501
BATCH_TEMPLATES, BATCH_SHEETS, BATCH_COPIES = 50, 5, 10  # per template: 5 sheets x 10 copies
GROUP_SIZE = 10
MAP_SCALARS = ("corr_halves_all", "corr_halves_inner", "sd_map_inner_pts", "sd_reliable_inner_pts")
# The template bootstrap repeats templates, which adds noise to a resampled map: the
# correlation between halves and the raw spread of the map depend on that noise level, so
# they get no bootstrap interval (their null reference gives the noise floor). The
# reliable spread, a covariance between independent halves, does.
BOOT_SCALARS = ("sd_reliable_inner_pts",)
NO_INTERVAL = {"map_corr_halves_all", "map_corr_halves_inner", "map_sd_map_inner_pts"}
NO_NULL = {"shared_U_over_var_B"}  # under the BPC var_B is 0: the ratio has no null value
LAG_NAMES = [f"{dy},{dx}" for dy, dx in TWO_POINT_LAGS]


# ----------------------------------------------------------------------------- helpers


def jeffreys(S1: np.ndarray, L: np.ndarray, n: int = 1) -> np.ndarray:
    """Codebook (k + 0.5) / (n L + 1) from flips S1 over n reads of L positions."""
    return (S1 + 0.5) / (n * L + 1.0)


def interval(values) -> list:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])] if v.size else [np.nan, np.nan]


def null_summary(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    q = np.percentile(v, [2.5, 97.5])
    return {"mean": float(v.mean()), "q025": float(q[0]), "q975": float(q[1]), "draws": int(v.size)}


def column(dicts: list, key: str) -> np.ndarray:
    return np.array([d[key] for d in dicts], dtype=np.float64)


class TemplateBootstrap:
    """Bootstrap on templates: multiplicities of each template of a universe, per draw.

    Every statistic of a dataset reads its weights from the same draws, so that
    differences between statistics get joint intervals.
    """

    def __init__(self, universe: np.ndarray, n_boot: int, rng: np.random.Generator):
        self.universe = np.asarray(universe)
        u = len(self.universe)
        draws = rng.integers(0, u, (n_boot, u))
        self.W = np.stack([np.bincount(d, minlength=u) for d in draws]).astype(np.float32)

    def weights(self, ids: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.universe, ids)
        if (self.universe[pos] != ids).any():
            raise ValueError("some templates are not in the bootstrap universe")
        return self.W[:, pos]


def fold_cells(L: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Positions of each pattern in each fold: (n_folds, 512)."""
    return np.stack([L[folds == g].sum(0) for g in range(folds.max() + 1)])


def batched(W: np.ndarray, size: int = 100):
    for start in range(0, len(W), size):
        yield W[start : start + size]


# ----------------------------------------------------------------------------- statistics


def locality_within(sums, n, folds, W, rng, n_null):
    """rho (and by class) with bootstrap interval and BPC null, for one set of reads."""
    point = reproducibility_from_sums(sums, n, folds=folds)
    reps = [reproducibility_from_sums(sums, n, folds=folds, weights=w) for w in W]
    cells = fold_cells(sums["L"], folds)
    cb = jeffreys(sums["S1"].sum(0), sums["L"].sum(0), n)
    null_folds = np.arange(len(cells))
    nulls = [
        reproducibility_from_sums(simulate_pattern_sums(cells, cb, n, rng), n, folds=null_folds)
        for _ in range(n_null)
    ]
    by_class = np.array([r["rho_by_class"] for r in reps])
    by_class_null = np.array([r["rho_by_class"] for r in nulls])
    out = {k: point[k] for k in ("n_templates", "n_reads", "D_obs", "D_bpc", "rho")}
    out["rho_ci"] = interval(column(reps, "rho"))
    out["null_rho"] = null_summary(column(nulls, "rho"))
    out["share_by_class"] = point["share_by_class"]
    out["rho_by_class"] = point["rho_by_class"]
    out["rho_by_class_ci"] = [interval(by_class[:, c]) for c in range(by_class.shape[1])]
    out["null_rho_by_class"] = [
        interval(by_class_null[:, c]) for c in range(by_class_null.shape[1])
    ]
    return out, column(reps, "rho"), column(nulls, "rho")


def locality_cross(sums, n, n_b, folds, W, rng, n_null):
    """Shared part of the reproducible pattern between conditions a and b (§6.1)."""
    point = reproducibility_from_sums(sums, n, n_b, folds=folds)
    reps = [reproducibility_from_sums(sums, n, n_b, folds=folds, weights=w) for w in W]
    cells = fold_cells(sums["L"], folds)
    L = sums["L"].sum(0)
    cb_a, cb_b = jeffreys(sums["S1"].sum(0), L, n), jeffreys(sums["S1b"].sum(0), L, n_b)
    null_folds = np.arange(len(cells))
    nulls = [
        reproducibility_from_sums(
            simulate_pattern_sums(cells, cb_a, n, rng, cb_b, n_b), n, n_b, folds=null_folds
        )
        for _ in range(n_null)
    ]
    keys = ("n_templates", "n_reads", "n_reads_b", "X_obs", "X_bpc", "shared", "var_a")
    out = {k: point[k] for k in keys}
    out["rho_a"] = point["rho"]
    if "var_b" in point:
        out["var_b"] = point["var_b"]
    out["shared_over_var_a"] = point["shared_over_var_a"]
    out["shared_over_var_a_ci"] = interval(column(reps, "shared_over_var_a"))
    out["shared_ci"] = interval(column(reps, "shared"))
    out["null_shared"] = null_summary(column(nulls, "shared"))
    return out


def decomposition(rho_prints, rho_reads, reps_prints, reps_reads, null_prints, null_reads):
    """methods.md §6.2: template, print and capture parts of the BPC noise."""

    def parts(p, r):
        return {"template": p, "print": r - p, "capture": 1 - r}

    point = parts(rho_prints, rho_reads)
    reps = parts(reps_prints, reps_reads)
    nulls = parts(null_prints, null_reads)
    return {
        k: {"value": float(point[k]), "ci": interval(reps[k]), "null": null_summary(nulls[k])}
        for k in point
    }


def image_effect_file(D, L, W, rng, n_null):
    """sigma_image of one file (one read per template), cross-fitted by halves of templates.

    The bootstrap redraws templates with the residuals kept as they are (the codebook,
    learned on hundreds of templates, is not refit in each draw); the null redoes it all.
    """
    folds = np.arange(len(D)) % 2
    r, b = residual_flip_rate(D, L, folds)
    point = sigma_image(r, b)
    reps = [sigma_image(r, b, w) for w in W]
    cb = jeffreys(D.sum(0), L.sum(0))
    nulls = [
        sigma_image(*residual_flip_rate(simulate_counts(L, cb, rng), L, folds))
        for _ in range(n_null)
    ]
    out = dict(point)
    out["n_templates"] = len(D)
    for k in ("sd_image_pts", "sd_image_over_binomial", "sd_total_over_binomial"):
        out[f"{k}_ci"] = interval(column(reps, k))
        out[f"null_{k}"] = null_summary(column(nulls, k))
    return out


def table_components(r, ids, w=None):
    c = two_way_components(r, weights=w)
    pos = (ids - 1) % INDIGO_PAGE
    eta = adjusted_eta2(r.mean(1), pos, weights=w)
    c["eta2_position_in_page"] = eta
    c["sd_position_pts"] = c["sd_template_pts"] * float(np.sqrt(max(eta, 0.0)))
    return c


def image_effect_table(D3, L3, ids, W, rng, n_null):
    """Components of sigma_image on a templates x reads table (§6.3), Indigo."""
    folds = np.arange(len(D3)) % 2
    r, b = residual_flip_rate(D3, L3, folds)
    point = table_components(r, ids)
    point["sd_binomial_pts"] = 100 * float(np.sqrt(b.mean()))
    point["n_templates"], point["n_reads"] = r.shape
    reps = [table_components(r, ids, w) for w in W]
    cb = jeffreys(D3.sum((0, 1)), L3.sum((0, 1)))
    nulls = []
    for _ in range(n_null):
        D_sim = np.stack([simulate_counts(L3[:, j], cb, rng) for j in range(r.shape[1])], 1)
        nulls.append(table_components(residual_flip_rate(D_sim, L3, folds)[0], ids))
    for k in reps[0]:
        point[f"{k}_ci"] = interval(column(reps, k))
        point[f"null_{k}"] = null_summary(column(nulls, k))
    return point


def map_summary(stats: dict) -> dict:
    out = {k: stats[k] for k in MAP_SCALARS}
    for axis in ("row", "col"):
        prof = np.asarray(stats[f"{axis}_profile_pts"])
        out[f"{axis}_profile_pts"] = prof.tolist()
        out[f"{axis}_border_pts"] = np.r_[prof[:3], prof[-3:]].tolist()
        out[f"{axis}_max_abs_pts"] = float(np.abs(prof).max())
    return out


def map_with_bootstrap(M, halves, W, ring):
    point = map_summary(map_stats(M, halves, ring=ring))
    reps = {k: [] for k in BOOT_SCALARS}
    for Wb in batched(W):
        s = map_stats(M, halves, weights=Wb, ring=ring)
        for k in BOOT_SCALARS:
            reps[k].append(s[k])
    for k in BOOT_SCALARS:
        point[f"{k}_ci"] = interval(np.concatenate(reps[k]))
    return point


# ----------------------------------------------------------------------------- null tasks
# Run in worker processes: each gets its arrays and its own seed, so results do not
# depend on the number of workers.


def null_map_task(t, cb_full, n, folds, halves, ring, seed, n_draws):
    """Position-map statistics on n_draws BPC draws of K (codebooks refit each time)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        K = simulate_flip_counts(t, cb_full, n, rng)
        cbs = cross_fitted_codebooks(pattern_sums(t, K, n), n, folds)
        s = map_stats(template_maps(t, K, n, cbs, folds), halves, ring=ring)
        out.append({k: s[k] for k in MAP_SCALARS})
    return out


def null_two_point_task(t_sub, L_all, sub, folds, cb_full, seed, n_draws):
    """Two-point correlations on n_draws BPC reads of the same templates."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        reads = simulate_reads(t_sub, cb_full, 1, rng)[0]
        D = simulate_counts(L_all, cb_full, rng)
        D[sub] = image_counts(t_sub, reads)[0]
        cbs = cross_fitted_codebooks({"S1": D, "L": L_all}, 1, folds)
        out.append(two_point(t_sub, reads, cbs, folds[sub]).mean(0))
    return out


# ----------------------------------------------------------------------------- Indigo


class Indigo:
    """Files of originals of the compact pack, read by template_id."""

    def __init__(self, data_dir: Path, cfg: dict):
        self.pack = data_dir / "indigo_compact"
        self.sets = cfg["within"]
        tpl = load_compact(self.pack / "templates_original.npz", stage=STAGE)
        self.T = np.zeros((INDIGO_N_TEMPLATES + 1, INDIGO_SIDE, INDIGO_SIDE), np.uint8)
        self.T[tpl.ids] = tpl.bits

    def files(self, item) -> list[str]:
        if isinstance(item, str):
            return list(self.sets[item]) if item in self.sets else [item]
        return list(item)

    def path(self, name: str) -> Path:
        path = self.pack / f"{name}.npz"
        _, meta = read_compact_header(path)
        if meta["class"] != "original":
            raise ValueError(f"{name} is not a file of originals: C1a reads no fake")
        return path

    def common_ids(self, names) -> np.ndarray:
        sets = [set(read_compact_header(self.path(n))[0].tolist()) for n in names]
        return np.array(sorted(set.intersection(*sets)))

    def flip_counts(self, names, ids) -> np.ndarray:
        K = np.zeros((len(ids), INDIGO_SIDE, INDIGO_SIDE), np.uint8)
        t = self.T[ids]
        for name in names:
            f = load_compact(self.path(name), stage=STAGE)
            K += f.bits[np.searchsorted(f.ids, ids)] != t
        return K

    def counts(self, name):
        return indigo_counts(self.path(name), stage=STAGE)


def run_indigo(data_dir, cfg, n_null, n_boot, ring, rng, pool, seeds, log):
    ind = Indigo(data_dir, cfg)
    boot = TemplateBootstrap(np.arange(1, INDIGO_N_TEMPLATES + 1), n_boot, rng)
    res = {"within": {}, "cross": {}, "decomposition": {}, "files": {}, "tables": {}}
    rho_reps, rho_null = {}, {}

    log("Indigo: rho within sets of reads")
    for name, files in cfg["within"].items():
        ids = ind.common_ids(files)
        K = ind.flip_counts(files, ids)
        sums = pattern_sums(ind.T[ids], K, len(files))
        out, rho_reps[name], rho_null[name] = locality_within(
            sums, len(files), np.arange(len(ids)) % 2, boot.weights(ids), rng, n_null
        )
        res["within"][name] = out
        log(f"  {name}: {len(ids)} templates x {len(files)} reads, rho = {out['rho']:.3f}")
        del K

    log("Indigo: shared parts between conditions")
    for name, (a, b) in cfg["cross"].items():
        fa, fb = ind.files(a), ind.files(b)
        ids = ind.common_ids(fa + fb)
        sums = pattern_sums(
            ind.T[ids], ind.flip_counts(fa, ids), len(fa), ind.flip_counts(fb, ids), len(fb)
        )
        folds = np.arange(len(ids)) % 2
        res["cross"][name] = locality_cross(
            sums, len(fa), len(fb), folds, boot.weights(ids), rng, n_null
        )
        res["cross"][name]["a"], res["cross"][name]["b"] = a, b
        log(f"  {name}: shared / var_a = {res['cross'][name]['shared_over_var_a']:.3f}")

    for name, (prints, reads) in cfg["decomposition"].items():
        res["decomposition"][name] = decomposition(
            res["within"][prints]["rho"],
            res["within"][reads]["rho"],
            rho_reps[prints],
            rho_reps[reads],
            rho_null[prints],
            rho_null[reads],
        ) | {"prints": prints, "reads": reads}

    log("Indigo: sigma_image of each file and components of each table")
    names = sorted({f for files in cfg["within"].values() for f in files} | set(cfg["extra_files"]))
    counts = {}
    for name in names:
        c = ind.counts(name)
        counts[name] = c
        res["files"][name] = image_effect_file(c.D, c.L, boot.weights(c.ids), rng, n_null)
    for name, files in cfg["within"].items():
        ids = ind.common_ids(files)
        rows = [np.searchsorted(counts[f].ids, ids) for f in files]
        D3 = np.stack([counts[f].D[r] for f, r in zip(files, rows)], 1)
        L3 = np.stack([counts[f].L[r] for f, r in zip(files, rows)], 1)
        res["tables"][name] = image_effect_table(D3, L3, ids, boot.weights(ids), rng, n_null)
        log(f"  table {name}: template {res['tables'][name]['sd_template_pts']:.2f} pts")

    log("Indigo: position maps (null draws sent to the workers)")
    res["maps"], pending = {}, {}
    for name in cfg["maps"]:
        files = ind.files(name)
        ids = ind.common_ids(files)
        t, K, n = ind.T[ids], ind.flip_counts(files, ids), len(files)
        folds = np.arange(len(ids)) % 2
        sums = pattern_sums(t, K, n)
        M = template_maps(t, K, n, cross_fitted_codebooks(sums, n, folds), folds)
        res["maps"][name] = map_with_bootstrap(M, folds, boot.weights(ids), ring)
        res["maps"][name].update(n_templates=len(ids), n_reads=n)
        del M, K
        cb_full = jeffreys(sums["S1"].sum(0), sums["L"].sum(0), n)
        per = cfg["map_draws_per_task"]
        pending[name] = [
            pool.submit(null_map_task, t, cb_full, n, folds, folds, ring, next(seeds), k)
            for k in [per] * (n_null // per) + ([n_null % per] if n_null % per else [])
        ]
        log(f"  {name}: sd reliable {res['maps'][name]['sd_reliable_inner_pts']:.2f} pts")

    log("Indigo: two-point correlation")
    res["two_point"], pending_tp = {}, {}
    for name in cfg["two_point_files"]:
        f = load_compact(ind.path(name), stage=STAGE)
        c = counts.get(name) or ind.counts(name)
        if not np.array_equal(c.ids, f.ids):
            raise ValueError(f"counts and images of {name} do not match")
        folds = np.arange(len(f.ids)) % 2
        sub = np.arange(min(cfg["two_point_max_templates"], len(f.ids)))
        t_sub = ind.T[f.ids[sub]]
        cbs = cross_fitted_codebooks({"S1": c.D, "L": c.L}, 1, folds)
        per_t = two_point(t_sub, f.bits[sub], cbs, folds[sub])
        W = boot.weights(f.ids[sub])
        reps = (W @ per_t) / W.sum(1, keepdims=True)
        res["two_point"][name] = {
            "n_templates": len(sub),
            "lags": LAG_NAMES,
            "value": per_t.mean(0).tolist(),
            "ci": [interval(reps[:, j]) for j in range(len(LAG_NAMES))],
        }
        cb_full = jeffreys(c.D.sum(0), c.L.sum(0))
        pending_tp[name] = pool.submit(
            null_two_point_task, t_sub, c.L, sub, folds, cb_full, next(seeds), n_null
        )
    return res, pending, pending_tp


# ----------------------------------------------------------------------------- Scantrust


def scantrust_images(data_dir: Path, cfg: dict):
    """Originals only: unique templates of the pool and 3501..3550, and all of batch_cdp."""
    manifest = read_manifest(data_dir / "manifest.csv")
    orig = manifest[manifest["class"] == "original"]
    lo, hi = cfg["unique_pool"]
    last_batch = BATCH_FIRST_ID + BATCH_TEMPLATES - 1
    uniq = orig[(orig["dataset"] == "unique") & orig["template_id"].between(lo, last_batch)]
    batch = orig[orig["dataset"] == "batch"]
    u = load_scantrust(data_dir, uniq.sort_values("template_id"), stage=STAGE)
    b = load_scantrust(data_dir, batch.sort_values(["template_id", "print_id"]), stage=STAGE)
    for imgs in (u, b):
        if not (imgs.rows["class"] == "original").all():
            raise ValueError("C1a reads no fake")

    # batch layout: file number p = 100 sheet + 10 rank + copy + 1, sheets 5g..5g+4.
    p = b.rows["print_id"].to_numpy(int) - 1
    tid = b.rows["template_id"].to_numpy(int)
    group, rank, sheet = p // 500, (p // 10) % 10, p // 100
    if not (tid == BATCH_FIRST_ID + GROUP_SIZE * group + rank).all():
        raise ValueError("batch_cdp numbering does not match the manifest")
    if not (b.rows["sheet"].to_numpy(int) == sheet).all():
        raise ValueError("batch_cdp sheets do not match the manifest")
    shape = (BATCH_TEMPLATES, BATCH_SHEETS * BATCH_COPIES)
    if not (sheet.reshape(shape) % BATCH_SHEETS == np.arange(shape[1])[None] // 10).all():
        raise ValueError("batch_cdp reads are not ordered by sheet then copy")
    side = b.t.shape[-1]
    t50 = b.t[:: shape[1]]
    reads = b.bits.reshape(shape + (side, side))
    if not (b.t.reshape(reads.shape) == t50[:, None]).all():
        raise ValueError("batch_cdp images are not grouped by template")

    utid = u.rows["template_id"].to_numpy(int)
    pool = (utid >= lo) & (utid <= hi)
    ub = utid >= BATCH_FIRST_ID
    if not np.array_equal(utid[ub], np.arange(BATCH_FIRST_ID, last_batch + 1)):
        raise ValueError("missing unique originals of the batch templates")
    if not (u.t[ub] == t50).all():
        raise ValueError("unique templates 3501..3550 differ from the batch templates (D-014)")
    return {
        "t_pool": u.t[pool],
        "pool_reads": u.bits[pool],
        "t50": t50,
        "reads": reads,
        "u_read": u.bits[ub],
    }


def batch_prepare(t50, reads, u_read, ring):
    """Per-template arrays of batch_cdp from which every statistic is a weighted sum."""
    n_t, n_r = reads.shape[:2]
    folds = np.arange(n_t) // GROUP_SIZE  # the codebook comes from the other groups
    flips = reads != t50[:, None]
    K_all = flips.sum(1).astype(np.uint8)
    units = flips.reshape(n_t, BATCH_SHEETS, BATCH_COPIES, *t50.shape[1:]).sum(2)
    t_units = np.repeat(t50, BATCH_SHEETS, axis=0)
    sums_units = pattern_sums(t_units, units.reshape(-1, *t50.shape[1:]).astype(np.uint8), 10)
    sums_w = {k: v.reshape(n_t, BATCH_SHEETS, -1).sum(1) for k, v in sums_units.items()}
    sums_x = pattern_sums(t50, K_all, n_r, (u_read != t50).astype(np.uint8), 1)
    D, L = image_counts(np.repeat(t50, n_r, axis=0), reads.reshape(-1, *t50.shape[1:]))
    D3, L3 = D.reshape(n_t, n_r, -1), L.reshape(n_t, n_r, -1)
    r, b = residual_flip_rate(D3, L3, folds)
    cbs = cross_fitted_codebooks(sums_x, n_r, folds)
    M = template_maps(t50, K_all, n_r, cbs, folds)
    cbs1 = cross_fitted_codebooks({"S1": D3.sum(1), "L": L3.sum(1)}, 1, folds)
    tp = two_point(
        np.repeat(t50, n_r, axis=0), reads.reshape(-1, *t50.shape[1:]), cbs1, np.repeat(folds, n_r)
    ).reshape(n_t, n_r, -1)
    return {
        "folds": folds,
        "n": n_r,
        "sums_x": sums_x,
        "sums_w": sums_w,
        "r": r,
        "b": b,
        "M": M,
        "tp": tp.mean(1),
        "ring": ring,
    }


def batch_stats(prep, cbU, w=None):
    """Every batch_cdp statistic of C1a, with template weights w (None = the data)."""
    folds, n = prep["folds"], prep["n"]
    sx = prep["sums_x"]
    a = reproducibility_from_sums(sx, n, 1, folds=folds, weights=w, codebook_b=cbU)
    within = reproducibility_from_sums(prep["sums_w"], 10, folds=folds, weights=w)
    pairs_all = n * (n - 1) / 2
    pairs_within = BATCH_SHEETS * BATCH_COPIES * (BATCH_COPIES - 1) / 2
    d_between = (pairs_all * a["D_obs"] - pairs_within * within["D_obs"]) / (
        pairs_all - pairs_within
    )
    ww = np.ones(len(folds)) if w is None else np.asarray(w, np.float64)
    s = sigma_image(prep["r"].ravel(), prep["b"].ravel(), np.repeat(ww, prep["r"].shape[1]))
    comps = []
    for g in range(folds.max() + 1):
        rows = folds == g
        if ww[rows].sum() < 2:
            continue
        A = prep["r"][rows].reshape(rows.sum(), BATCH_SHEETS, BATCH_COPIES)
        comps.append(crossed_components(A, weights=ww[rows]))
    comp = {k: float(np.sqrt(np.mean([c[k] ** 2 for c in comps]))) for k in comps[0]}
    halves = np.arange(len(folds)) % 2
    m = map_stats(prep["M"], halves, weights=ww[None], ring=prep["ring"])
    tp = (ww @ prep["tp"]) / ww.sum()
    out = {
        "rho": a["rho"],
        "D_obs": a["D_obs"],
        "D_bpc": a["D_bpc"],
        "rho_by_class": a["rho_by_class"],
        "share_by_class": a["share_by_class"],
        "rho_within_sheet": within["rho"],
        "rho_between_sheets": 1 - d_between / a["D_bpc"],
        "shared_U_over_var_B": a["shared_over_var_a"],
        "shared_U": a["shared"],
        **{f"comp_{k}": v for k, v in comp.items()},
        **{f"map_{k}": float(v[0]) for k, v in m.items()},
        **{f"two_point_{name}": float(v) for name, v in zip(LAG_NAMES, tp)},
    }
    out.update({f"image_{k}": v for k, v in s.items()})
    return out


def pool_prepare(t, reads, ring):
    folds = np.arange(len(t)) % 2
    D, L = image_counts(t, reads)
    r, b = residual_flip_rate(D, L, folds)
    K = (reads != t).astype(np.uint8)
    cbs = cross_fitted_codebooks({"S1": D, "L": L}, 1, folds)
    return {
        "folds": folds,
        "r": r,
        "b": b,
        "M": template_maps(t, K, 1, cbs, folds),
        "tp": two_point(t, reads, cbs, folds),
        "ring": ring,
        "D": D,
        "L": L,
    }


def pool_stats(prep, w=None):
    ww = np.ones(len(prep["r"])) if w is None else np.asarray(w, np.float64)
    s = sigma_image(prep["r"], prep["b"], ww)
    m = map_stats(prep["M"], prep["folds"], weights=ww[None], ring=prep["ring"])
    tp = (ww @ prep["tp"]) / ww.sum()
    out = {f"image_{k}": v for k, v in s.items()}
    out.update({f"map_{k}": float(v[0]) for k, v in m.items()})
    out.update({f"two_point_{name}": float(v) for name, v in zip(LAG_NAMES, tp)})
    return out


def with_intervals(point: dict, reps: list, nulls: list) -> dict:
    out = {}
    for k, v in point.items():
        if isinstance(v, list):
            out[k] = v
            continue
        out[k] = {
            "value": v,
            "ci": None if k in NO_INTERVAL else interval(column(reps, k)),
            "null": None if k in NO_NULL else null_summary(column(nulls, k)),
        }
    return out


def run_scantrust(data_dir, cfg, n_null, n_boot, ring, rng, log):
    log("Scantrust: reading the originals")
    d = scantrust_images(data_dir, cfg)
    res = {}

    pool = pool_prepare(d["t_pool"], d["pool_reads"], ring)
    cbU = jeffreys(pool["D"].sum(0), pool["L"].sum(0))  # session U, other templates

    log("Scantrust batch_cdp: data, bootstrap, null")
    prep = batch_prepare(d["t50"], d["reads"], d["u_read"], ring)
    point = batch_stats(prep, cbU)
    boot = TemplateBootstrap(np.arange(BATCH_TEMPLATES), n_boot, rng)
    reps = [batch_stats(prep, cbU, w) for w in boot.W]
    sums_all = prep["sums_x"]
    cb_b = jeffreys(sums_all["S1"].sum(0), sums_all["L"].sum(0), prep["n"])
    nulls = []
    for _ in range(n_null):
        sim = simulate_reads(d["t50"], cb_b, prep["n"], rng).transpose(1, 0, 2, 3)
        u_sim = simulate_reads(d["t50"], cbU, 1, rng)[0]
        cbU_sim = jeffreys(simulate_counts(pool["L"], cbU, rng).sum(0), pool["L"].sum(0))
        nulls.append(batch_stats(batch_prepare(d["t50"], sim, u_sim, ring), cbU_sim))
    res["batch"] = with_intervals(point, reps, nulls)
    by_class = np.array([r["rho_by_class"] for r in reps])
    res["batch"]["rho_by_class_ci"] = [interval(by_class[:, c]) for c in range(by_class.shape[1])]
    res["batch"]["n_templates"], res["batch"]["n_reads"] = BATCH_TEMPLATES, prep["n"]

    log("Scantrust unique pool: data, bootstrap, null")
    point = pool_stats(pool)
    boot_u = TemplateBootstrap(np.arange(len(d["t_pool"])), n_boot, rng)
    reps = [pool_stats(pool, w) for w in boot_u.W]
    nulls = []
    for _ in range(n_null):
        sim = simulate_reads(d["t_pool"], cbU, 1, rng)[0]
        nulls.append(pool_stats(pool_prepare(d["t_pool"], sim, ring)))
    res["unique_pool"] = with_intervals(point, reps, nulls)
    res["unique_pool"]["n_templates"] = len(d["t_pool"])
    m = map_stats(pool["M"], pool["folds"], ring=ring)
    res["unique_pool"]["map_profiles"] = map_summary(m)
    m = map_stats(prep["M"], np.arange(BATCH_TEMPLATES) % 2, ring=ring)
    res["batch"]["map_profiles"] = map_summary(m)
    return res


# ----------------------------------------------------------------------------- predictions


def predictions(res: dict, cfg: dict) -> dict:
    """protocol.md §5.1, H1a, H1c, H1d, H1e, judged as written on 2026-11-10."""
    p = cfg["predictions"]
    ind, st = res["indigo"], res["scantrust"]
    rho = {
        "Scantrust batch_cdp": (st["batch"]["rho"]["value"], st["batch"]["rho"]["ci"]),
        "Indigo HPI55 (session 1, 10 prints)": (
            ind["within"]["prints_HPI55_s1"]["rho"],
            ind["within"]["prints_HPI55_s1"]["rho_ci"],
        ),
        "Indigo HPI76 (session 1, 3 prints)": (
            ind["within"]["prints_HPI76_s1"]["rho"],
            ind["within"]["prints_HPI76_s1"]["rho_ci"],
        ),
    }
    h1a = {
        k: {
            "rho": v,
            "ci": ci,
            "holds": bool(v > p["H1a"]["rho_min"] and ci[0] > p["H1a"]["ci_low_min"]),
        }
        for k, (v, ci) in rho.items()
    }
    dec = ind["decomposition"]["HPI55"]["print"]
    h1c = {
        "rho_scans_minus_rho_prints": dec["value"],
        "ci": dec["ci"],
        "holds": bool(dec["value"] > p["H1c"]["margin"]),
    }
    files = ind["files"]
    ratios = {k: v["sd_image_over_binomial"] for k, v in files.items()}
    scan = [v for k, v in ratios.items() if k.startswith("variability")]
    phone = [v for k, v in ratios.items() if k.startswith("mobile")]
    st_ratio = {
        "batch_cdp": st["batch"]["image_sd_image_over_binomial"]["value"],
        "unique (templates of the pool)": st["unique_pool"]["image_sd_image_over_binomial"][
            "value"
        ],
    }
    h1d = {
        "indigo_scanner_ratio_min_max": [min(scan), max(scan)],
        "indigo_phone_ratio_min_max": [min(phone), max(phone)],
        "indigo_holds": bool(min(scan + phone) > p["H1d"]["indigo_ratio"]),
        "scantrust_ratio": st_ratio,
        "scantrust_holds": bool(min(st_ratio.values()) > p["H1d"]["scantrust_ratio"]),
        "scantrust_total_over_binomial": {
            "batch_cdp": st["batch"]["image_sd_total_over_binomial"]["value"],
            "unique (templates of the pool)": st["unique_pool"]["image_sd_total_over_binomial"][
                "value"
            ],
        },
    }
    sd = {k: v["sd_reliable_inner_pts"] for k, v in ind["maps"].items()}
    sd["Scantrust unique"] = st["unique_pool"]["map_sd_reliable_inner_pts"]["value"]
    sd["Scantrust batch_cdp"] = st["batch"]["map_sd_reliable_inner_pts"]["value"]
    h1e = {
        "sd_reliable_inner_pts": sd,
        "holds": bool(max(sd.values()) < p["H1e"]["sd_reliable_max_pts"]),
    }
    return {"H1a": h1a, "H1c": h1c, "H1d": h1d, "H1e": h1e}


# ----------------------------------------------------------------------------- figure


def plot_fig1_left(result: dict, path: Path) -> None:
    """Fig. 1 left: the noise predicted by the BPC, split by where it comes from."""
    import matplotlib.pyplot as plt

    from cdpbpc.plotting import NOISE_COLORS, setup

    setup()
    ind, st = result["indigo"], result["scantrust"]["batch"]
    rows = []
    for label, key in (("Indigo HPI76*", "HPI76_scans_session0"), ("Indigo HPI55", "HPI55")):
        d = ind["decomposition"][key]
        rows.append(
            (
                label,
                [("template", d["template"]), ("print", d["print"]), ("capture", d["capture"])],
                [ind["within"][d["prints"]]["rho_ci"], ind["within"][d["reads"]]["rho_ci"]],
            )
        )
    r = st["rho"]
    rows.append(
        (
            "Scantrust",
            [
                ("template", {"value": r["value"]}),
                ("print_and_capture", {"value": 1 - r["value"]}),
            ],
            [r["ci"]],
        )
    )
    names = {
        "template": "reproducible from print to print",
        "print": "specific to one print",
        "capture": "capture (scan or photo)",
        "print_and_capture": "print and capture (not split)",
    }
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    seen = set()
    for y, (label, parts, cis) in enumerate(rows):
        left = 0.0
        for part, d in parts:
            width = 100 * d["value"]
            ax.barh(
                y,
                width,
                left=left,
                color=NOISE_COLORS[part],
                edgecolor="white",
                linewidth=0.5,
                label=None if part in seen else names[part],
            )
            seen.add(part)
            tone = "white" if part in ("template", "print") else "black"
            ax.text(
                left + width / 2,
                y,
                f"{width:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color=tone,
            )
            left += width
        for ci in cis:
            lo, hi = 100 * ci[0], 100 * ci[1]
            ax.errorbar(
                (lo + hi) / 2,
                y + 0.33,
                xerr=(hi - lo) / 2,
                fmt="none",
                ecolor="black",
                elinewidth=0.7,
                capsize=1.5,
            )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("share of the flip noise predicted by the BPC (%)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.4, -0.36),
        ncol=2,
        frameon=False,
        handlelength=1,
        columnspacing=0.8,
    )
    note = "* prints of session 1, scans of session 0. Bars: 95 % bootstrap intervals on templates."
    ax.text(-0.3, -0.82, note, transform=ax.transAxes, fontsize=6, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C1a: what the 3x3 BPC model misses")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "c1a.yaml")
    parser.add_argument("--figure", type=Path, help="only redraw Fig. 1 left from a result")
    args = parser.parse_args(argv)
    fig_path = ROOT / "report" / "figures" / "fig1_left.pdf"
    if args.figure:
        plot_fig1_left(json.loads(args.figure.read_text()), fig_path)
        print(f"written to {fig_path.relative_to(ROOT)}")
        return 0

    text = args.config.read_text()
    cfg = yaml.safe_load(text)
    data_dir = ROOT / cfg["data_dir"]
    start = time.time()

    def log(msg):
        print(f"[{time.time() - start:6.0f} s] {msg}", flush=True)

    rng = np.random.default_rng(cfg["seed"])
    seeds = (np.random.SeedSequence([cfg["seed"], k]) for k in range(1, 10**6))
    n_null, n_boot, ring = cfg["n_null"], cfg["n_boot"], cfg["map_ring"]
    with ProcessPoolExecutor(cfg["workers"], mp_context=mp.get_context("spawn")) as pool:
        indigo, pending, pending_tp = run_indigo(
            data_dir, cfg["indigo"], n_null, n_boot, ring, rng, pool, seeds, log
        )
        scantrust = run_scantrust(data_dir, cfg["scantrust"], n_null, n_boot, ring, rng, log)
        log("waiting for the null draws of the workers")
        for name, futures in pending.items():
            draws = [d for f in futures for d in f.result()]
            for k in MAP_SCALARS:
                indigo["maps"][name][f"null_{k}"] = null_summary(column(draws, k))
        for name, future in pending_tp.items():
            draws = np.array(future.result())
            indigo["two_point"][name]["null"] = [
                null_summary(draws[:, j]) for j in range(draws.shape[1])
            ]

    result = {
        "experiment": "c1a_noise",
        "data": "originals and templates only",
        "git": git_state(),
        "config": cfg,
        "indigo": indigo,
        "scantrust": scantrust,
    }
    result["predictions"] = predictions(result, cfg)
    result["seconds"] = round(time.time() - start)
    out = ROOT / "results" / f"c1a_{hashlib.sha1(text.encode()).hexdigest()[:8]}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(_round(result, 5), indent=1))
    plot_fig1_left(json.loads(out.read_text()), fig_path)
    print(json.dumps(_round(result["predictions"], 3), indent=1))
    print(f"written to {out.relative_to(ROOT)} and {fig_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
