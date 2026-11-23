"""Where a fake gives itself away, predicted without any fake: the two passes of a copy.

A fake is the attacker's estimate of the template, t_hat, printed and captured again.
Under the BPC (Tutt 2025, Sec. 4.3) its flip probability against the true template is
q = P_b(pattern of t_hat) where t_hat is right and 1 - P_b(pattern of t_hat) where it
is wrong (cdpbpc.two_pass; same model without context: Phan Ho et al. 2014, eqs. 9-11;
Baras and Cayre 2012). Pooled by the pattern of the true template, q gives the
predicted flip rate of fakes Q_pred. This script compares Q_pred with the flip rate of
real fakes Q_obs. No image of a fake is produced; P_b and the estimate come from
originals and from the estimated templates published with the data.

Conditions:
- Indigo mobile, each phone, capture 1: t_hat = the estimated templates of the mobile
  archive (EHPI55); P_b = codebook of the originals of block E of that phone, capture 1.
- Indigo scanner, fakes estimated from prints of printer nn and reprinted on printer mm:
  t_hat = estimated templates of the base archive for nn; P_b = codebook of the
  originals of block E of printer mm.
- Scantrust unique_cdp, single-scan attacks: t_hat is not published; it is approximated
  by our binarization of the original of the same template. P_b from block E.
- Scantrust batch_cdp, averaging attacks: t_hat approximated by the per-position
  majority vote of the 50 binarized originals of the template. P_b from block E.

Everything goes through per-pattern counts. For the judged templates: Q_pred from
cdpbpc.two_pass.predicted_sums (pattern of the true template), Q_obs from the counts
D, L of the real fakes, P_obs from the counts of the originals of the same templates in
the same condition, and the attacker's error e from the counts of t_hat against t.
Reliability classes (8) follow P_b of block E. Reported per condition and attack:
- per class: share of symbols, P_obs, P_b, e, Q_pred, Q_obs, and the gaps
  Q_pred - P_obs and Q_obs - P_obs with a bootstrap interval over templates;
- per pattern (patterns with more than min_occurrences symbols in the judged fakes):
  correlation between Q_pred - P and Q_obs - P (plain and weighted by the number of
  symbols), weighted slope, bootstrap intervals over templates; P = P_obs, and P = P_b
  as a secondary line. With P = P_obs both gaps contain the sampling noise of P_obs,
  which pushes the correlation up (about 1/sqrt(2) when fakes are just originals); with
  P = P_b the predicted gap does not share that noise. Resampling templates adds noise
  to every pattern, which pulls a correlation down: its percentile interval can lie
  below the point value (the median of the bootstrap correlations shows the shift);
- masked Hamming (P_b <= mu) of the fakes, predicted and observed, and of the originals;
- overall flip rate of the fakes, predicted and observed; error rate of t_hat.
The prediction set in the config is judged at the test stage only; at the development
stage it is reported as an indication.

--stage dev reads only the development fakes and estimates (Indigo templates 1-288,
batch group 0, unique templates 3551-4000). --stage test runs once after the methods
freeze and needs freeze_commit in the config.

Usage:
    python -m experiments.c3b_two_pass --config configs/c3b.yaml --stage dev
Writes results/c3b_<config hash>.json at the development stage and
results/c3b_<config hash>_test.json at the test stage, which refuses to overwrite it.
"""

import argparse
import hashlib
import json
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cdpbpc.counts import image_counts, indigo_counts, scantrust_counts
from cdpbpc.io import (
    INDIGO_DEV_MAX_ID,
    SCANTRUST_DEV_GROUP,
    SCANTRUST_DEV_UNIQUE,
    STAGES,
    load_compact,
    load_scantrust,
)
from cdpbpc.patterns import N_PATTERNS
from cdpbpc.scores import CLASS_EDGES_8, reliability_class
from cdpbpc.splits import INDIGO_TEST, batch_fold, indigo_blocks, unique_blocks
from cdpbpc.two_pass import majority_estimate, predicted_sums
from experiments.e0_sanity import _round, codebook_of, git_state

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEMPLATES_FILE = "templates_original.npz"
N_CLASSES = len(CLASS_EDGES_8) - 1


# ----------------------------------------------------------------------------- guards


def check_indigo_ids(ids, stage: str, what: str) -> None:
    """Refuse, at the development stage, any Indigo fake or estimate of a test template."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if stage == "dev" and (np.asarray(ids) > INDIGO_DEV_MAX_ID).any():
        raise RuntimeError(f"a test {what} was read at the development stage")


def _is(rows: pd.DataFrame, col: str, value) -> np.ndarray:
    """rows[col] == value as a plain boolean array (missing values give False)."""
    return (rows[col] == value).fillna(False).to_numpy(dtype=bool)


def _ints(rows: pd.DataFrame, col: str) -> np.ndarray:
    """A column of ids as int64, missing values as -1."""
    return rows[col].fillna(-1).to_numpy(dtype=np.int64)


def check_scantrust_fakes(rows: pd.DataFrame, stage: str) -> None:
    """Refuse, at the development stage, a Scantrust fake outside batch group 0 and V_U."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if stage == "dev":
        fake = _is(rows, "class", "fake")
        batch = _is(rows, "dataset", "batch")
        lo, hi = SCANTRUST_DEV_UNIQUE
        tid = _ints(rows, "template_id")
        ok_batch = _is(rows, "group", SCANTRUST_DEV_GROUP)
        ok_unique = (tid >= lo) & (tid <= hi)
        if (fake & np.where(batch, ~ok_batch, ~ok_unique)).any():
            raise RuntimeError("a test fake was read at the development stage")


# ------------------------------------------------------------------------- one condition


@dataclass
class Condition:
    """Judged templates of one condition and attack.

    P: (512,) codebook of block E. ids: (N,) template ids; t, t_hat: (N, side, side)
    true and estimated templates; nf: (N,) number of fakes of each template;
    Df, Lf: (N, 512) counts summed over the fakes of each template; Do, Lo: (N, 512)
    counts summed over the originals of each template in the same condition.
    """

    P: np.ndarray
    ids: np.ndarray
    t: np.ndarray
    t_hat: np.ndarray
    nf: np.ndarray
    Df: np.ndarray
    Lf: np.ndarray
    Do: np.ndarray
    Lo: np.ndarray


def per_template(tids, D, L, ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts of images summed per template, in the order of ids; and images per template."""
    tids = np.asarray(tids)
    ids = np.asarray(ids)
    pos = np.searchsorted(ids, tids)
    pos = np.clip(pos, 0, len(ids) - 1)
    if len(ids) == 0 or (ids[pos] != tids).any():
        raise ValueError("an image belongs to no listed template")
    n = np.bincount(pos, minlength=len(ids))
    Ds = np.zeros((len(ids), N_PATTERNS))
    Ls = np.zeros((len(ids), N_PATTERNS))
    np.add.at(Ds, pos, np.asarray(D, dtype=np.float64))
    np.add.at(Ls, pos, np.asarray(L, dtype=np.float64))
    return Ds, Ls, n


def bootstrap_weights(n: int, rng, n_boot: int) -> np.ndarray:
    """(n_boot, n) multiplicities of n templates drawn with replacement."""
    idx = rng.integers(0, n, (n_boot, n)) + n * np.arange(n_boot)[:, None]
    return np.bincount(idx.ravel(), minlength=n * n_boot).reshape(n_boot, n).astype(np.float64)


def _div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def weighted_corr_slope(x, y, w) -> tuple[np.ndarray, np.ndarray]:
    """Correlation and slope of y on x with weights w, along the last axis (w = 0 ignored)."""
    w = np.asarray(w, dtype=np.float64)
    x = np.where(w > 0, x, 0.0)
    y = np.where(w > 0, y, 0.0)
    sw = w.sum(axis=-1, keepdims=True)
    mx = _div((w * x).sum(-1, keepdims=True), sw)
    my = _div((w * y).sum(-1, keepdims=True), sw)
    dx, dy = np.where(w > 0, x - mx, 0.0), np.where(w > 0, y - my, 0.0)
    cxy = (w * dx * dy).sum(-1)
    vx, vy = (w * dx * dx).sum(-1), (w * dy * dy).sum(-1)
    return _div(cxy, np.sqrt(vx * vy)), _div(cxy, vx)


def _ci(values) -> list:
    v = np.asarray(values, dtype=np.float64)
    with warnings.catch_warnings():  # a class or pattern absent from every replicate: NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanquantile(v, [0.025, 0.975], axis=0)
    return [lo.tolist(), hi.tolist()] if np.ndim(lo) else [float(lo), float(hi)]


def _median(values) -> float:
    with warnings.catch_warnings():  # no pattern in any replicate: NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmedian(values))


def compare(c: Condition, mu: float, min_occurrences: int, rng, n_boot: int) -> dict:
    """Predicted and observed flip profile of the fakes of one condition."""
    n = len(c.ids)
    if n == 0:
        raise ValueError("no judged template")
    Sq, L = predicted_sums(c.t, c.t_hat, c.P)
    De, Le = image_counts(c.t, c.t_hat)
    w = np.asarray(c.nf, dtype=np.float64)[:, None]
    if not np.array_equal(Le, L) or not np.array_equal(np.asarray(c.Lf, np.float64), w * L):
        raise ValueError("counts of the fakes do not match their templates")
    if (np.asarray(c.Lo).sum(axis=1) == 0).any():
        raise ValueError("a judged template has no original")
    X = {
        "Lw": w * L,
        "Sq": w * Sq,
        "De": w * De,
        "Pb": w * L * c.P,
        "Df": np.asarray(c.Df, np.float64),
        "Lf": np.asarray(c.Lf, np.float64),
        "Do": np.asarray(c.Do, np.float64),
        "Lo": np.asarray(c.Lo, np.float64),
    }
    W = bootstrap_weights(n, rng, n_boot)
    cls = reliability_class(c.P)
    onehot = (cls[:, None] == np.arange(N_CLASSES)[None]).astype(np.float64)

    # ---------------------------------------------------------------- per class
    Xc = {k: v @ onehot for k, v in X.items()}
    tot = {k: v.sum(axis=0) for k, v in Xc.items()}
    boot = {k: W @ v for k, v in Xc.items()}

    def class_stats(T):
        p_obs = _div(T["Do"], T["Lo"])
        q_pred = _div(T["Sq"], T["Lw"])
        q_obs = _div(T["Df"], T["Lf"])
        return p_obs, q_pred, q_obs, q_pred - p_obs, q_obs - p_obs

    p_obs, q_pred, q_obs, g_pred, g_obs = class_stats(tot)
    _, _, _, bg_pred, bg_obs = class_stats(boot)
    ci_pred, ci_obs = _ci(bg_pred), _ci(bg_obs)
    share = tot["Lw"] / tot["Lw"].sum()
    by_class = []
    for k in range(N_CLASSES):
        by_class.append(
            {
                "class": [CLASS_EDGES_8[k], CLASS_EDGES_8[k + 1]],
                "share": float(share[k]),
                "P_obs": float(p_obs[k]),
                "P_b": float(_div(tot["Pb"], tot["Lw"])[k]),
                "e": float(_div(tot["De"], tot["Lw"])[k]),
                "Q_pred": float(q_pred[k]),
                "Q_obs": float(q_obs[k]),
                "gap_pred": float(g_pred[k]),
                "gap_pred_ci": [ci_pred[0][k], ci_pred[1][k]],
                "gap_obs": float(g_obs[k]),
                "gap_obs_ci": [ci_obs[0][k], ci_obs[1][k]],
            }
        )

    # -------------------------------------------------------------- per pattern
    totp = {k: v.sum(axis=0) for k, v in X.items()}
    keep = (totp["Lf"] > min_occurrences) & (totp["Lo"] > 0)
    bootp = {k: W @ v for k, v in X.items() if k in ("Lw", "Sq", "Df", "Lf", "Do", "Lo")}

    def pattern_stats(T, ref):
        q_pred, q_obs = _div(T["Sq"], T["Lw"]), _div(T["Df"], T["Lf"])
        p = _div(T["Do"], T["Lo"]) if ref == "P_obs" else np.broadcast_to(c.P, q_pred.shape)
        valid = keep & (T["Lw"] > 0) & (T["Lo"] > 0)
        x, y = q_pred - p, q_obs - p
        corr, _ = weighted_corr_slope(x, y, valid.astype(np.float64))
        wcorr, slope = weighted_corr_slope(x, y, np.where(valid, T["Lf"], 0.0))
        return corr, wcorr, slope

    per_pattern = {"min_occurrences": min_occurrences, "n_patterns": int(keep.sum())}
    for ref in ("P_obs", "P_b"):
        corr, wcorr, slope = pattern_stats(totp, ref)
        bc, bw, bs = pattern_stats(bootp, ref)
        per_pattern[ref] = {
            "corr": float(corr),
            "corr_ci": _ci(bc),
            "corr_bootstrap_median": _median(bc),
            "weighted_corr": float(wcorr),
            "weighted_corr_ci": _ci(bw),
            "weighted_slope": float(slope),
            "weighted_slope_ci": _ci(bs),
        }

    # -------------------------------------------------------- masked Hamming, rates
    m = (c.P <= mu).astype(np.float64)

    def masked(T):
        return (
            _div(T["Sq"] @ m, T["Lw"] @ m),
            _div(T["Df"] @ m, T["Lf"] @ m),
            _div(T["Do"] @ m, T["Lo"] @ m),
        )

    mh = masked(totp)
    mb = masked({k: W @ v for k, v in X.items()})
    out = {
        "n_templates": n,
        "n_fakes": int(np.sum(c.nf)),
        "n_originals_per_template": float(np.mean(np.asarray(c.Lo).sum(1) / L.sum(1))),
        "by_class": by_class,
        "per_pattern": per_pattern,
        "masked_hamming": {
            "mu": mu,
            "share_of_symbols": float((totp["Lw"] @ m) / totp["Lw"].sum()),
            "fakes_predicted": float(mh[0]),
            "fakes_predicted_ci": _ci(mb[0]),
            "fakes_observed": float(mh[1]),
            "fakes_observed_ci": _ci(mb[1]),
            "originals": float(mh[2]),
            "originals_ci": _ci(mb[2]),
        },
        "flip_rate": {
            "fakes_predicted": float(totp["Sq"].sum() / totp["Lw"].sum()),
            "fakes_observed": float(totp["Df"].sum() / totp["Lf"].sum()),
            "originals": float(totp["Do"].sum() / totp["Lo"].sum()),
        },
        "estimate_error_rate": float(De.sum() / L.sum()),
    }
    return out


# --------------------------------------------------------------------------- Indigo


def _rows_of(all_ids, ids) -> np.ndarray:
    """Positions of ids in all_ids (each present once)."""
    all_ids = np.asarray(all_ids)
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("a template appears twice in one file")
    order = np.argsort(all_ids)
    pos = order[np.searchsorted(all_ids[order], ids)]
    if (all_ids[pos] != ids).any():
        raise ValueError("a template is missing")
    return pos


def indigo_condition(
    pack: Path, setting: str, original: str, fake: str, estimate: str, cfg: dict, stage: str
) -> tuple[Condition, dict]:
    """Condition of an Indigo file of fakes, its estimates and the originals it is judged with."""
    est = load_compact(pack / f"{estimate}.npz", stage=stage)
    if est.meta["class"] != "estimated":
        raise ValueError(f"{estimate} holds no estimated templates")
    check_indigo_ids(est.ids, stage, "estimated template")
    o = indigo_counts(pack / f"{original}.npz", stage=stage)
    f = indigo_counts(pack / f"{fake}.npz", stage=stage)
    if o.meta["class"] != "original" or f.meta["class"] != "fake":
        raise ValueError(f"expected originals in {original} and fakes in {fake}")
    check_indigo_ids(f.ids, stage, "fake")
    blocks = indigo_blocks(setting, stage=stage, available=o.ids, seed=cfg["seed"])
    if stage == "dev":
        judged_range = blocks["V"]
    else:
        judged_range = np.concatenate([blocks["T_a"], blocks["T_b"]])
    ids = np.intersect1d(np.intersect1d(judged_range, f.ids), est.ids)
    check_indigo_ids(ids, stage, "fake")
    tmpl = load_compact(pack / TEMPLATES_FILE, stage=stage)
    E = np.isin(o.ids, blocks["E"])
    po, pf = _rows_of(o.ids, ids), _rows_of(f.ids, ids)
    cond = Condition(
        P=codebook_of(o.D[E], o.L[E]),
        ids=ids,
        t=tmpl.bits[_rows_of(tmpl.ids, ids)],
        t_hat=est.bits[_rows_of(est.ids, ids)],
        nf=np.ones(len(ids), int),
        Df=f.D[pf],
        Lf=f.L[pf],
        Do=o.D[po],
        Lo=o.L[po],
    )
    lo, hi = (1, INDIGO_DEV_MAX_ID) if stage == "dev" else INDIGO_TEST[setting]
    in_range = np.arange(lo, hi + 1)
    info = {
        "originals": original,
        "fakes": fake,
        "estimates": estimate,
        "E_originals": int(E.sum()),
        "E_templates_range": [int(blocks["E"].min()), int(blocks["E"].max())],
        "fakes_not_judged": sorted(
            set(f.ids[np.isin(f.ids, in_range)].tolist()) - set(ids.tolist())
        ),
    }
    return cond, info


def run_indigo(pack: Path, cfg: dict, stage: str, rng, log) -> dict:
    mob, scan = cfg["indigo"]["mobile"], cfg["indigo"]["scanner"]
    out = {"mobile": {}, "scanner": {}}
    r = mob["capture"]
    for device in mob["devices"]:
        cond, info = indigo_condition(
            pack,
            "mobile",
            f"mobile_original_{device}_run{r}",
            f"mobile_fake_{mob['estimate']}_{device}_run{r}",
            f"templates_estimated_mobile_{mob['estimate']}",
            cfg,
            stage,
        )
        res = compare(cond, cfg["mu"], cfg["min_occurrences"], rng, cfg["n_boot"])
        out["mobile"][device] = {"info": info, **res}
        log(_line(f"Indigo mobile {device}", res))
    for printer, estimate in scan["pairs"]:
        cond, info = indigo_condition(
            pack,
            "scanner",
            f"base_original_{printer}",
            f"base_fake_{printer}_{estimate}",
            f"templates_estimated_base_{estimate}",
            cfg,
            stage,
        )
        res = compare(cond, cfg["mu"], cfg["min_occurrences"], rng, cfg["n_boot"])
        out["scanner"][f"{printer}_{estimate}"] = {"info": info, **res}
        log(_line(f"Indigo scanner {printer} reprint of {estimate}", res))
    return out


# ------------------------------------------------------------------------- Scantrust


def _used(rows: pd.DataFrame) -> np.ndarray:
    return (rows["use"].astype(str).str.lower() == "true").to_numpy(dtype=bool)


def _originals_bits(data_dir: Path, rows: pd.DataFrame, stage: str):
    if (rows["class"] != "original").any():
        raise RuntimeError("only originals are read as images")
    return load_scantrust(data_dir, rows, stage=stage)


def _fake_conditions(base: dict, rows, D, L, fake_mask, attacks, judged) -> dict:
    """One Condition per attack: the templates with fakes of that attack."""
    out = {}
    for a in attacks:
        F = fake_mask & _is(rows, "attack", a) & np.isin(base["tids"], judged)
        Df, Lf, nf = per_template(base["tids"][F], D[F], L[F], base["ids"])
        k = nf > 0
        out[a] = Condition(
            P=base["P"],
            ids=base["ids"][k],
            t=base["t"][k],
            t_hat=base["t_hat"][k],
            nf=nf[k],
            Df=Df[k],
            Lf=Lf[k],
            Do=base["Do"][k],
            Lo=base["Lo"][k],
        )
    return out


def run_scantrust(data_dir: Path, cfg: dict, stage: str, rng, log) -> dict:
    c = scantrust_counts(data_dir, stage=stage)
    rows = c.rows.reset_index(drop=True)
    check_scantrust_fakes(rows, stage)
    st = cfg["scantrust"]
    tids = _ints(rows, "template_id")
    used = _used(rows)
    is_orig = _is(rows, "class", "original") & used
    is_fake = _is(rows, "class", "fake") & used
    out = {"unique": {}, "batch": {}}

    # unique_cdp: one original per template, t_hat = its binarization
    uni = _is(rows, "dataset", "unique")
    blocks = unique_blocks(cfg["seed"])
    judged = blocks["V"] if stage == "dev" else blocks["T"]
    E = uni & is_orig & np.isin(tids, blocks["E"])
    O = uni & is_orig & np.isin(tids, judged)
    img = _originals_bits(data_dir, rows[O], stage)
    ids, first = np.unique(tids[O], return_index=True)
    if len(ids) != O.sum():
        raise ValueError("unique_cdp: more than one original for a template")
    base = {
        "P": codebook_of(c.D[E], c.L[E]),
        "ids": ids,
        "tids": tids,
        "t": img.t[first],
        "t_hat": img.bits[first],
        "Do": c.D[O][first].astype(np.float64),
        "Lo": c.L[O][first].astype(np.float64),
    }
    conds = _fake_conditions(base, rows, c.D, c.L, uni & is_fake, st["unique_attacks"], judged)
    for a, cond in conds.items():
        check_scantrust_fakes(rows[uni & is_fake & np.isin(tids, cond.ids)], stage)
        res = compare(cond, cfg["mu"], cfg["min_occurrences"], rng, cfg["n_boot"])
        info = {"E_originals": int(E.sum()), "judged_range": [int(judged[0]), int(judged[-1])]}
        out["unique"][a] = {"info": info, **res}
        log(_line(f"Scantrust unique {a}", res))

    # batch_cdp: t_hat = per-position majority of the 50 binarized originals
    bat = _is(rows, "dataset", "batch")
    prints = _ints(rows, "print_id")
    groups = [SCANTRUST_DEV_GROUP] if stage == "dev" else [1, 2, 3, 4]
    folds = [batch_fold(g) for g in groups]
    e_ids = folds[0]["e_print_ids"]
    if any(not np.array_equal(f["e_print_ids"], e_ids) for f in folds):
        raise ValueError("the batch folds do not share block E")
    judged = np.concatenate([f["blocks"]["T"] for f in folds])
    E = bat & is_orig & np.isin(prints, e_ids)
    O = bat & is_orig & np.isin(tids, judged)
    if (E & O).any():
        raise RuntimeError("an original of block E belongs to a judged template")
    img = _originals_bits(data_dir, rows[O], stage)
    ids = np.unique(tids[O])
    Do, Lo, n_o = per_template(tids[O], c.D[O], c.L[O], ids)
    t = np.stack([img.t[tids[O] == i][0] for i in ids])
    t_hat = np.stack([majority_estimate(img.bits[tids[O] == i]) for i in ids])
    base = {
        "P": codebook_of(c.D[E], c.L[E]),
        "ids": ids,
        "tids": tids,
        "t": t,
        "t_hat": t_hat,
        "Do": Do,
        "Lo": Lo,
    }
    conds = _fake_conditions(base, rows, c.D, c.L, bat & is_fake, st["batch_attacks"], judged)
    for a, cond in conds.items():
        res = compare(cond, cfg["mu"], cfg["min_occurrences"], rng, cfg["n_boot"])
        info = {
            "E_originals": int(E.sum()),
            "groups_judged": groups,
            "originals_per_template": [int(n_o.min()), int(n_o.max())],
        }
        out["batch"][a] = {"info": info, **res}
        log(_line(f"Scantrust batch {a}", res))
    return out


# ------------------------------------------------------------------------ prediction


def judge_prediction(res: dict, pred: dict) -> dict:
    """The prediction set in the config, checked on the results.

    Phone: plain per-pattern correlation of Q - P above min_corr for every reference P
    listed (P_obs = originals of the same templates; P_b = codebook of block E, which does
    not share the sampling noise of P_obs). Averaging attacks: Q_pred - P_obs and
    Q_obs - P_obs both positive in every class with lower edge >= p_min.
    """
    ph = pred["phone"]
    refs = list(ph["references"])
    if not refs or not set(refs) <= {"P_obs", "P_b"}:
        raise ValueError(f"phone references must be among P_obs and P_b, got {refs}")
    pp = res["indigo"]["mobile"][ph["device"]]["per_pattern"]
    corr = {ref: pp[ref]["corr"] for ref in refs}
    out = {
        "phone": {
            "device": ph["device"],
            "corr": corr,
            "holds": bool(all(c > ph["min_corr"] for c in corr.values())),
        }
    }
    av = pred["averaging"]
    for a in av["attacks"]:
        rows = [r for r in res["scantrust"]["batch"][a]["by_class"] if r["class"][0] >= av["p_min"]]
        ok = all(r["gap_pred"] > 0 and r["gap_obs"] > 0 for r in rows)
        out[f"batch_{a}"] = {
            "gap_pred": [r["gap_pred"] for r in rows],
            "gap_obs": [r["gap_obs"] for r in rows],
            "holds": bool(ok),
        }
    return out


def _line(name: str, r: dict) -> str:
    pp = r["per_pattern"]["P_obs"]
    mh = r["masked_hamming"]
    return (
        f"{name}: {r['n_templates']} templates, corr of (Q - P) {pp['corr']:.3f}, "
        f"masked Hamming of fakes predicted {mh['fakes_predicted']:.4f}, "
        f"observed {mh['fakes_observed']:.4f}, originals {mh['originals']:.4f}"
    )


# ------------------------------------------------------------------------------ main


def run_all(cfg: dict, stage: str, rng, log) -> dict:
    data_dir = ROOT / cfg["data_dir"]
    out = {
        "indigo": run_indigo(data_dir / "indigo_compact", cfg, stage, rng, log),
        "scantrust": run_scantrust(data_dir, cfg, stage, rng, log),
    }
    out["prediction"] = {
        "status": "judged (test stage)" if stage == "test" else "indication only (development)",
        **judge_prediction(out, cfg["prediction"]),
    }
    return out


def result_path(text: str, stage: str) -> Path:
    """Result file of a config text: c3b_<hash>.json, or c3b_<hash>_test.json at the test stage.

    The test result has its own name so that a later development run with the same
    config can never overwrite it.
    """
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    return RESULTS / f"c3b_{h}{'_test' if stage == 'test' else ''}.json"


def check_freeze_commit(sha) -> None:
    """Refuse a freeze_commit that is not a commit of this repository."""
    found = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if found.returncode != 0:
        raise SystemExit(f"freeze_commit {sha!r} is not a commit of this repository")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Flip profile of fakes predicted without fakes")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "c3b.yaml")
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args(argv)
    text = args.config.read_text()
    cfg = yaml.safe_load(text)
    if args.stage == "test" and not cfg.get("freeze_commit"):
        raise SystemExit("--stage test needs freeze_commit in the config (methods freeze)")
    out = result_path(text, args.stage)
    if args.stage == "test" and out.exists():
        raise SystemExit(f"{out.name} exists: the test stage runs once")
    if args.stage == "test":
        check_freeze_commit(cfg["freeze_commit"])
    start = time.time()

    def log(msg):
        print(f"[{time.time() - start:6.0f} s] {msg}", flush=True)

    rng = np.random.default_rng(cfg["seed"])
    result = {"experiment": "c3b_two_pass", "stage": args.stage, "git": None, "config": cfg}
    result.update(run_all(cfg, args.stage, rng, log))
    result["seconds"] = round(time.time() - start)
    result["git"] = git_state()
    out.parent.mkdir(exist_ok=True)
    with out.open("x" if args.stage == "test" else "w") as f:
        f.write(json.dumps(_round(result, 5), indent=1))
    print(json.dumps(_round(result["prediction"], 4), indent=1))
    print(f"written to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
