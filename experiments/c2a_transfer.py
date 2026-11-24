"""Does a threshold set on originals still hold when the condition changes?

A detector is enrolled in a reference condition (one file of originals): codebook on
E = 50 originals, mean and covariance of binned_mahalanobis on M = 200, conformal
threshold on S = 400, templates drawn at random. It is then applied to the originals of
a target condition (another file), leaving out the templates used for enrolment, and
we measure the realized P_miss for a target P_miss alpha. Changes: none (same file,
other templates), other pages, scan, capture, print run, session, printer, phone.

Scantrust adds three rows, with its own fixed blocks:
- same session: unique_cdp, E = 50, M = 500, S = 1000 drawn among templates 1..3500,
  judged on the 1000 originals of T_orig (10 draws);
- other group of sheets: batch_cdp, S = the 500 originals of group r, judged on the
  500 originals of group t != r (20 pairs); E = the 50 first copies of a third group,
  M = the 500 originals of a fourth (the next groups after r, skipping t);
- other session: the unique_cdp detectors judged on the 2500 batch_cdp originals (10),
  and the batch_cdp detector of group r (E and M from groups r + 1 and r + 2) judged on
  the unique_cdp originals of templates 1..3500 (5).
A batch_cdp template has 50 originals: there, rates are per template and the threshold
is redone on whole templates (K = 50, 100 or 200 images = 1, 2 or 4 templates).

Then the threshold is redone on K originals of the target (codebook kept), 50 random
draws per K; the realized P_miss on the other originals of the target is compared with
the Beta band of a conformal threshold on K originals. Two comparison points only: the
reference threshold moved by the mean gap (mean shift) and, for centered_z, the
analytic BPC threshold z_(1 - alpha).

Originals only: every file must hold originals (checked in its metadata), no fake is
read. Every pair of Indigo files is written in the config.

Usage:
    python -m experiments.c2a_transfer --config configs/c2a.yaml
    python -m experiments.c2a_transfer --figure results/c2a_<hash>.json   # redraw only
Writes results/c2a_<config hash>.json and report/figures/fig2.pdf.
"""

import argparse
import hashlib
import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from cdpbpc.calibrate import (
    analytic_threshold,
    conformal_threshold,
    mean_shift_threshold,
    p_miss_band,
    recalibrate,
)
from cdpbpc.codebook import estimate
from cdpbpc.counts import indigo_counts, scantrust_counts
from cdpbpc.metrics import rate_per_group
from cdpbpc.scores import (
    MahalanobisFit,
    binned_mahalanobis,
    centered_z,
    fit_binned_mahalanobis,
    hamming,
    masked_hamming,
    pll,
)
from cdpbpc.splits import UNIQUE_POOL, batch_e_print_ids, batch_group_ids, unique_blocks
from experiments.e0_sanity import _round, git_state

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STAGE = "dev"  # only originals are read; the stage cut does not touch them
SCORES = ("hamming", "centered_z", "pll", "masked_hamming", "binned_mahalanobis")
INDIGO_PAGE = 144


# ----------------------------------------------------------------------------- enrolment


@dataclass
class Detector:
    """A detector enrolled on originals of one reference condition."""

    used: np.ndarray  # template_ids of E, M and S: never judged in a target
    codebook: np.ndarray  # (512,) from E
    fit: MahalanobisFit  # from M
    cal: dict  # score name -> (n_S,) scores of the originals of S
    mu: float


def all_scores(D, L, codebook, fit, mu) -> dict:
    """The five scores compared in this experiment, from pattern counts. Large = suspect."""
    return {
        "hamming": hamming(D, L),
        "centered_z": centered_z(D, L, codebook),
        "pll": pll(D, L, codebook),
        "masked_hamming": masked_hamming(D, L, codebook, mu),
        "binned_mahalanobis": binned_mahalanobis(D, L, codebook, fit),
    }


def enrol(ids, D, L, sizes: dict, rng: np.random.Generator, mu: float, pool=None) -> Detector:
    """Draw E, M, S among the originals of the reference (restricted to pool if given)."""
    ids = np.asarray(ids)
    cand = np.flatnonzero(np.isin(ids, pool)) if pool is not None else np.arange(len(ids))
    nE, nM, nS = sizes["E"], sizes["M"], sizes["S"]
    if nE + nM + nS > len(cand):
        raise ValueError(f"{len(cand)} originals cannot hold E + M + S = {nE + nM + nS}")
    pick = rng.permutation(cand)
    iE, iM, iS = pick[:nE], pick[nE : nE + nM], pick[nE + nM : nE + nM + nS]
    codebook = estimate(L[iE].sum(axis=0), D[iE].sum(axis=0))
    fit = fit_binned_mahalanobis(D[iM], L[iM], codebook)
    cal = all_scores(D[iS], L[iS], codebook, fit, mu)
    return Detector(ids[np.r_[iE, iM, iS]], codebook, fit, cal, mu)


def judge(det: Detector, ids, D, L, keep=None) -> dict:
    """Scores of the target originals whose template was not used to enrol det."""
    m = ~np.isin(ids, det.used)
    if keep is not None:
        m &= np.isin(ids, keep)
    return all_scores(D[m], L[m], det.codebook, det.fit, det.mu)


# ----------------------------------------------------------------------------- one pair


def pair_result(det: Detector, target: dict, cfg: dict, rng: np.random.Generator) -> dict:
    """Realized P_miss of the fixed threshold, then after recalibration on K originals."""
    alphas, Ks, n_draws = cfg["alphas"], cfg["recal_K"], cfg["recal_draws"]
    level = cfg["band_level"]
    n = len(target["hamming"])
    out = {"n_target": n, "fixed": {}, "analytic_centered_z": {}, "recal": {}}
    for a in alphas:
        out["analytic_centered_z"][a] = float(np.mean(target["centered_z"] > analytic_threshold(a)))
    for k in SCORES:
        s = target[k]
        out["fixed"][k] = {
            a: float(np.mean(s > conformal_threshold(det.cal[k], a))) for a in alphas
        }
        out["recal"][k] = {}
        for a in alphas:
            out["recal"][k][a] = {}
            for K in Ks:
                if K >= n:
                    continue
                rec = recalibrate(s, K, a, rng, n_draws)
                shift = np.empty(n_draws)
                for r, idx in enumerate(rec.drawn):
                    rest = np.ones(n, dtype=bool)
                    rest[idx] = False
                    tau = mean_shift_threshold(det.cal[k], s[idx], a)
                    shift[r] = np.mean(s[rest] > tau)
                lo, hi = p_miss_band(K, a, level)
                out["recal"][k][a][K] = {
                    "mean": float(rec.p_miss.mean()),
                    "q10": float(np.quantile(rec.p_miss, 0.1)),
                    "q90": float(np.quantile(rec.p_miss, 0.9)),
                    "share_in_band": float(np.mean((rec.p_miss >= lo) & (rec.p_miss <= hi))),
                    "p_miss": rec.p_miss.tolist(),
                    "mean_shift_mean": float(shift.mean()),
                }
    return out


# ----------------------------------------------------------------------------- pairs


def expand_pairs(change: dict) -> list[dict]:
    """The (reference, target) pairs of one change of the config.

    Each group lists refs and targets; every ordered pair (ref, target) with ref != target
    is a pair, or (file, file) for the groups of `same_file`. The group is the unit of
    the change (e.g. one couple of sessions).
    """
    pairs = []
    for g in change["groups"]:
        if g.get("same_file"):
            pairs += [{"group": g["name"], "ref": f, "target": f} for f in g["files"]]
            continue
        for r in g["refs"]:
            for t in g["targets"]:
                if r != t:
                    pairs.append({"group": g["name"], "ref": r, "target": t})
    return pairs


def check_original(name: str, meta: dict) -> None:
    """Refuse any file that does not hold originals: this experiment reads no fake."""
    if meta.get("class") != "original":
        raise SystemExit(f"{name} holds {meta.get('class')!r}, not originals: no fake is read here")


def seeded(seed: int, *keys: str) -> np.random.Generator:
    """A generator that depends on the seed and on names only, not on the run order."""
    return np.random.default_rng([seed, *(zlib.crc32(k.encode()) for k in keys)])


def page_ids(pages) -> np.ndarray:
    """template_ids of pages lo..hi (1-based, 144 templates per page)."""
    lo, hi = pages
    return np.arange((lo - 1) * INDIGO_PAGE + 1, hi * INDIGO_PAGE + 1)


# ----------------------------------------------------------------------------- Scantrust


def detector_from_blocks(ids, D, L, iE, iM, iS, mu: float) -> Detector:
    """A detector enrolled on fixed blocks (boolean masks over the images)."""
    codebook = estimate(L[iE].sum(axis=0), D[iE].sum(axis=0))
    fit = fit_binned_mahalanobis(D[iM], L[iM], codebook)
    cal = all_scores(D[iS], L[iS], codebook, fit, mu)
    return Detector(np.unique(ids[iE | iM | iS]), codebook, fit, cal, mu)


def pair_result_by_template(
    det: Detector, target: dict, tids, cfg: dict, rng: np.random.Generator
) -> dict:
    """As pair_result, for a target with several images per template (batch_cdp).

    Rates are the mean over templates of the rate per template. The threshold is redone
    on all the images of k templates drawn at random (K = k x images per template), and
    the realized P_miss is measured on the other templates.
    """
    tids = np.asarray(tids)
    groups, per = np.unique(tids, return_counts=True)
    if len(set(per)) != 1:
        raise ValueError("every template must have the same number of images")
    per = int(per[0])
    alphas, n_draws, level = cfg["alphas"], cfg["recal_draws"], cfg["band_level"]

    def rate(reject, m=None):
        m = np.ones(len(tids), dtype=bool) if m is None else m
        return float(rate_per_group(reject[m], tids[m])[1].mean())

    out = {"n_target": len(tids), "n_templates": len(groups), "fixed": {}, "recal": {}}
    out["analytic_centered_z"] = {
        a: rate(target["centered_z"] > analytic_threshold(a)) for a in alphas
    }
    Ks = [K for K in cfg["recal_K"] if K % per == 0 and K // per < len(groups)]
    draws = {K: [rng.choice(groups, K // per, replace=False) for _ in range(n_draws)] for K in Ks}
    out["recal_templates"] = {K: [d.tolist() for d in ds] for K, ds in draws.items()}
    for k in SCORES:
        s = target[k]
        out["fixed"][k] = {a: rate(s > conformal_threshold(det.cal[k], a)) for a in alphas}
        out["recal"][k] = {}
        for a in alphas:
            out["recal"][k][a] = {}
            for K, ds in draws.items():
                miss, shift = np.empty(n_draws), np.empty(n_draws)
                for r, drawn in enumerate(ds):
                    m = np.isin(tids, drawn)
                    miss[r] = rate(s > conformal_threshold(s[m], a), ~m)
                    shift[r] = rate(s > mean_shift_threshold(det.cal[k], s[m], a), ~m)
                lo, hi = p_miss_band(K, a, level)
                out["recal"][k][a][K] = {
                    "mean": float(miss.mean()),
                    "q10": float(np.quantile(miss, 0.1)),
                    "q90": float(np.quantile(miss, 0.9)),
                    "share_in_band": float(np.mean((miss >= lo) & (miss <= hi))),
                    "p_miss": miss.tolist(),
                    "templates_per_draw": K // per,
                    "mean_shift_mean": float(shift.mean()),
                }
    return out


def run_scantrust(cfg: dict, st, log) -> dict:
    """The three Scantrust rows. st: originals with ids, prints, dataset, D, L."""
    sc = cfg["scantrust"]
    is_u, is_b = st.dataset == "unique", st.dataset == "batch"
    mu = cfg["mu"]

    def of(mask):
        return {"ids": st.ids[mask], "D": st.D[mask], "L": st.L[mask]}

    def judged(det, mask):
        t = of(mask & ~np.isin(st.ids, det.used))
        return all_scores(t["D"], t["L"], det.codebook, det.fit, mu), t["ids"]

    def unique_det(seed):
        b = unique_blocks(seed)
        m = {k: is_u & np.isin(st.ids, b[k]) for k in ("E", "M", "S", "T_orig")}
        return detector_from_blocks(st.ids, st.D, st.L, m["E"], m["M"], m["S"], mu), m["T_orig"]

    def batch_det(r, skip=None):
        e, m = [g % 5 for g in range(r + 1, r + 6) if g % 5 not in (r, skip)][:2]
        iE = is_b & np.isin(st.prints, batch_e_print_ids(e))
        iM = is_b & np.isin(st.ids, batch_group_ids(m))
        iS = is_b & np.isin(st.ids, batch_group_ids(r))
        return detector_from_blocks(st.ids, st.D, st.L, iE, iM, iS, mu), (e, m)

    def rng_of(name, ref, target):
        return seeded(cfg["seed"], "recal", name, ref, target)

    changes = {}
    rows = []
    for seed in range(sc["unique_seeds"]):
        det, t_orig = unique_det(seed)
        target, _ = judged(det, t_orig)
        p = {"group": "unique", "ref": f"unique_seed{seed}", "target": "unique_T_orig"}
        rows.append(
            {**p, **pair_result(det, target, cfg, rng_of("scantrust_same", p["ref"], p["target"]))}
        )
    changes["scantrust_same"] = rows

    rows = []
    for r in sc["batch_groups"]:
        for t in sc["batch_groups"]:
            if t == r:
                continue
            det, (e, m) = batch_det(r, skip=t)
            target, tids = judged(det, is_b & np.isin(st.ids, batch_group_ids(t)))
            p = {"group": "batch", "ref": f"batch_group{r}", "target": f"batch_group{t}"}
            res = pair_result_by_template(
                det, target, tids, cfg, rng_of("scantrust_sheets", p["ref"], p["target"])
            )
            rows.append({**p, "E_group": e, "M_group": m, **res})
    changes["scantrust_sheets"] = rows

    rows = []
    for seed in range(sc["unique_seeds"]):
        det, _ = unique_det(seed)
        target, tids = judged(det, is_b)
        p = {"group": "unique_to_batch", "ref": f"unique_seed{seed}", "target": "batch_all"}
        res = pair_result_by_template(
            det, target, tids, cfg, rng_of("scantrust_session", p["ref"], p["target"])
        )
        rows.append({**p, **res})
    lo, hi = UNIQUE_POOL
    for r in sc["batch_groups"]:
        det, (e, m) = batch_det(r)
        target, _ = judged(det, is_u & (st.ids >= lo) & (st.ids <= hi))
        p = {"group": "batch_to_unique", "ref": f"batch_group{r}", "target": "unique_1_3500"}
        res = pair_result(det, target, cfg, rng_of("scantrust_session", p["ref"], p["target"]))
        rows.append({**p, "E_group": e, "M_group": m, **res})
    changes["scantrust_session"] = rows
    for name, rows in changes.items():
        log(f"{name}: {len(rows)} pairs")
    return {
        name: {"pairs": rows, "summary": summarize(rows, cfg)} for name, rows in changes.items()
    }


# ----------------------------------------------------------------------------- summaries


def summarize(pairs: list[dict], cfg: dict) -> dict:
    """Median and range over pairs; per group too (the unit of the change)."""
    out = {"n_pairs": len(pairs), "groups": {}}

    def stats(rows):
        res = {"fixed": {}, "recal_mean": {}}
        for k in SCORES:
            res["fixed"][k] = {}
            res["recal_mean"][k] = {}
            for a in cfg["alphas"]:
                v = [r["fixed"][k][a] for r in rows]
                res["fixed"][k][a] = {
                    "median": float(np.median(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                }
                res["recal_mean"][k][a] = {}
                for K in cfg["recal_K"]:
                    w = [r["recal"][k][a][K]["mean"] for r in rows if K in r["recal"][k][a]]
                    if w:
                        res["recal_mean"][k][a][K] = {
                            "median": float(np.median(w)),
                            "min": float(np.min(w)),
                            "max": float(np.max(w)),
                        }
        v = {a: [r["analytic_centered_z"][a] for r in rows] for a in cfg["alphas"]}
        res["analytic_centered_z"] = {
            a: {"median": float(np.median(x)), "min": float(np.min(x)), "max": float(np.max(x))}
            for a, x in v.items()
        }
        return res

    out.update(stats(pairs))
    for g in dict.fromkeys(p["group"] for p in pairs):
        rows = [p for p in pairs if p["group"] == g]
        out["groups"][g] = {"n_pairs": len(rows), **stats(rows)}
    return out


def predictions(result: dict, cfg: dict) -> dict:
    """The four predictions of the config, written down before any run.

    same_run: within one print run the fixed threshold holds (Beta band of S).
    between_sessions: across sessions hamming rejects either far too many or almost none.
    mask_vs_count: where hamming rejects too many, masked_hamming rejects at most half.
    recalibration: a threshold redone on K target originals holds, whatever the change.
    result: as written to the JSON file (alphas and K are string keys).
    """
    P = cfg["predictions"]
    ch = result["changes"]
    out = {}

    a = P["same_run"]["alpha"]
    lo, hi = p_miss_band(cfg["sizes"]["S"], a, cfg["band_level"])
    rows = [p for p in ch["none"]["pairs"] if p["group"] in P["same_run"]["groups"]]
    inside = [lo <= p["fixed"]["hamming"][str(a)] <= hi for p in rows]
    out["same_run"] = {
        "rule": f"same print run: fixed P_miss of hamming at {a} in the Beta band "
        f"[{lo:.4f}, {hi:.4f}] for at least {P['same_run']['min_share']:.0%} of the runs",
        "share": float(np.mean(inside)),
        "n_runs": len(rows),
        "holds": bool(np.mean(inside) >= P["same_run"]["min_share"]),
    }

    a = P["between_sessions"]["alpha"]
    groups = ch["session"]["summary"]["groups"]
    med = {g: v["fixed"]["hamming"][str(a)]["median"] for g, v in groups.items()}
    out["between_sessions"] = {
        "rule": f"each couple of sessions: median P_miss of hamming at {a} above "
        f"{P['between_sessions']['high']} or below {P['between_sessions']['low']}",
        "median_per_couple": med,
        "holds": bool(
            all(
                m > P["between_sessions"]["high"] or m < P["between_sessions"]["low"]
                for m in med.values()
            )
        ),
    }

    ratio = {
        g: groups[g]["fixed"]["masked_hamming"][str(a)]["median"] / m
        for g, m in med.items()
        if m > a
    }
    out["mask_vs_count"] = {
        "rule": "couples of sessions where hamming rejects more than the target: "
        f"masked_hamming rejects at most {P['mask_vs_count']['max_ratio']} times as many",
        "masked_over_hamming": ratio,
        "holds": bool(ratio) and all(r <= P["mask_vs_count"]["max_ratio"] for r in ratio.values()),
    }

    if "scantrust_session" in ch and "mask_vs_count_scantrust" in P:
        q = P["mask_vs_count_scantrust"]
        a = q["alpha"]
        g = ch["scantrust_session"]["summary"]["groups"]
        med = {d: v["fixed"]["hamming"][str(a)]["median"] for d, v in g.items()}
        ratio = {
            d: g[d]["fixed"]["masked_hamming"][str(a)]["median"] / m
            for d, m in med.items()
            if m > a
        }
        out["mask_vs_count_scantrust"] = {
            "rule": "Scantrust, unique_cdp <-> batch_cdp, directions where hamming rejects more "
            f"than the target: masked_hamming does not reject {q['max_ratio']} times as many or fewer",
            "masked_over_hamming": ratio,
            "holds": bool(ratio) and all(r > q["max_ratio"] for r in ratio.values()),
        }

    a, K = str(P["recalibration"]["alpha"]), str(P["recalibration"]["K"])
    lo_d, hi_d = P["recalibration"]["range"]
    share = {}
    for name, c in ch.items():
        v = np.concatenate(
            [
                p["recal"]["hamming"][a][K]["p_miss"]
                for p in c["pairs"]
                if K in p["recal"]["hamming"][a]
            ]
        )
        share[name] = float(np.mean((v >= lo_d) & (v <= hi_d)))
    out["recalibration"] = {
        "rule": f"recalibration on K = {K}: realized P_miss of hamming at {a} in "
        f"[{lo_d}, {hi_d}] for at least {P['recalibration']['min_share']:.0%} of the draws, every change",
        "share_per_change": share,
        "holds": bool(all(s >= P["recalibration"]["min_share"] for s in share.values())),
    }
    return out


# ----------------------------------------------------------------------------- figure

FIG_CHANGES = (
    "none",
    "pages",
    "scan",
    "capture",
    "print_run",
    "session",
    "printer",
    "phone",
    "scantrust_same",
    "scantrust_sheets",
    "scantrust_session",
)
FIG_LABELS = {
    "none": "none",
    "pages": "pages",
    "scan": "scan",
    "capture": "capture",
    "print_run": "print run",
    "session": "session",
    "printer": "printer",
    "phone": "phone",
    "scantrust_same": "none",
    "scantrust_sheets": "sheets",
    "scantrust_session": "session",
}


def draw_figure(result: dict, out: Path) -> None:
    """Fig. 2: realized P_miss at the figure alpha, one point per pair, two scores.

    The grey band is the Beta band of a conformal threshold on n exchangeable images
    (n = S on the left, K on the right). It is drawn over the Indigo columns only: on
    Scantrust, S has another size, and on batch_cdp the unit is the template, not the
    image, so a band counted in images would be far too narrow. On the right each point
    is the mean of the recalibration draws, not one draw.
    """
    import matplotlib.pyplot as plt

    from cdpbpc.plotting import C2_SCORE_COLORS, setup

    setup()
    cfg = result["config"]
    a, K = cfg["figure"]["alpha"], cfg["figure"]["K"]
    scores = cfg["figure"]["scores"]
    names = [c for c in FIG_CHANGES if c in result["changes"]]
    n_ind = sum(not c.startswith("scantrust") for c in names)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    rng = np.random.default_rng(0)  # jitter only
    panels = (
        ("threshold set in the reference condition", cfg["sizes"]["S"], "fixed"),
        (f"redone on K = {K} target originals (mean of draws)", K, "recal"),
    )
    for ax, (title, n_band, kind) in zip(axes, panels):
        lo, hi = p_miss_band(n_band, a, cfg["band_level"])
        ax.fill_between([-0.5, n_ind - 0.5], lo, hi, color="0.88", lw=0, zorder=0)
        ax.axhline(a, color="0.4", lw=0.6, ls="--", zorder=1)
        for j, k in enumerate(scores):
            off = (j - (len(scores) - 1) / 2) * 0.28
            for i, c in enumerate(names):
                pairs = result["changes"][c]["pairs"]
                if kind == "fixed":
                    v = np.array([p["fixed"][k][str(a)] for p in pairs])
                else:
                    v = np.array(
                        [
                            p["recal"][k][str(a)][str(K)]["mean"]
                            for p in pairs
                            if str(K) in p["recal"][k][str(a)]
                        ]
                    )
                x = i + off + rng.uniform(-0.08, 0.08, len(v))
                ax.scatter(
                    x,
                    v,
                    s=6,
                    color=C2_SCORE_COLORS[k],
                    alpha=0.7,
                    lw=0,
                    marker="o" if j == 0 else "s",
                    label=k.replace("_", " ") if i == 0 else None,
                    zorder=2,
                )
        ax.set_xticks(
            range(len(names)),
            [FIG_LABELS[c] for c in names],
            fontsize=6,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        if n_ind < len(names):  # Indigo on the left, Scantrust on the right
            ax.axvline(n_ind - 0.5, color="0.6", lw=0.5)
            for x, txt in (
                ((n_ind - 1) / 2, "Indigo"),
                ((n_ind + len(names) - 1) / 2, "Scantrust"),
            ):
                ax.text(
                    x,
                    0.97,
                    txt,
                    ha="center",
                    va="top",
                    fontsize=6,
                    color="0.35",
                    transform=ax.get_xaxis_transform(),
                )
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.set_title(title)
        ax.set_xlabel("what changes between reference and target")
        ax.set_yscale("function", functions=(np.sqrt, np.square))
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0], ["0", "1", "5", "10", "20", "50", "100"])
    axes[0].set_ylabel(f"realized P_miss (%), target {a:.0%}")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.92), frameon=False, markerscale=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- run


def run(cfg: dict, load, log, scantrust=None) -> dict:
    """All changes of the config. load(name) -> counts of one file of originals;
    scantrust: the Scantrust originals (ids, prints, dataset, D, L), for its rows."""
    cache, dets = {}, {}

    def counts(name):
        if name not in cache:
            c = load(name)
            check_original(name, c.meta)
            cache[name] = c
        return cache[name]

    def detector(name, pages=None):
        key = (name, tuple(pages) if pages else None)
        if key not in dets:
            c = counts(name)
            pool = page_ids(pages) if pages else None
            rng = seeded(cfg["seed"], "enrol", name, str(pages))
            dets[key] = enrol(c.ids, c.D, c.L, cfg["sizes"], rng, cfg["mu"], pool)
        return dets[key]

    changes = {}
    for name, change in cfg["changes"].items():
        pairs = expand_pairs(change)
        log(f"{name}: {len(pairs)} pairs")
        rows = []
        for p in pairs:
            det = detector(p["ref"], change.get("ref_pages"))
            t = counts(p["target"])
            keep = page_ids(change["target_pages"]) if change.get("target_pages") else None
            target = judge(det, t.ids, t.D, t.L, keep)
            rng = seeded(cfg["seed"], "recal", name, p["ref"], p["target"])
            rows.append({**p, **pair_result(det, target, cfg, rng)})
        changes[name] = {"pairs": rows}
        changes[name]["summary"] = summarize(rows, cfg)
    if cfg.get("scantrust"):
        changes.update(run_scantrust(cfg, scantrust, log))
    return {"changes": changes}


def scantrust_originals(data_dir: Path) -> SimpleNamespace:
    """The Scantrust originals (all of them are free to use)."""
    sc = scantrust_counts(data_dir, stage=STAGE)
    r = sc.rows.reset_index(drop=True)
    o = (r["class"] == "original").to_numpy()
    return SimpleNamespace(
        ids=r["template_id"].to_numpy(int)[o],
        prints=r["print_id"].to_numpy()[o],
        dataset=r["dataset"].to_numpy()[o],
        D=sc.D[o],
        L=sc.L[o],
    )


def result_path(text: str) -> Path:
    return RESULTS / f"c2a_{hashlib.sha1(text.encode()).hexdigest()[:8]}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transfer of thresholds set on originals")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "c2a.yaml")
    parser.add_argument("--figure", type=Path, help="redraw Fig. 2 from a result file")
    args = parser.parse_args(argv)
    fig_path = ROOT / "report" / "figures" / "fig2.pdf"
    if args.figure:
        draw_figure(json.loads(args.figure.read_text()), fig_path)
        print(f"written to {fig_path.relative_to(ROOT)}")
        return 0

    text = args.config.read_text()
    cfg = yaml.safe_load(text)
    start = time.time()

    def log(msg):
        print(f"[{time.time() - start:6.0f} s] {msg}", flush=True)

    pack = ROOT / cfg["data_dir"] / "indigo_compact"

    def load(name):
        return indigo_counts(pack / f"{name}.npz", stage=STAGE)

    result = {"experiment": "c2a_transfer", "git": None, "config": cfg}
    result.update(run(cfg, load, log, scantrust_originals(ROOT / cfg["data_dir"])))
    result = json.loads(json.dumps(_round(result, 5)))  # keys as strings, as in the file
    result["predictions"] = predictions(result, cfg)
    result["seconds"] = round(time.time() - start)
    result["git"] = git_state()
    out = result_path(text)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    draw_figure(result, fig_path)
    for name, c in result["changes"].items():
        s = c["summary"]["fixed"]
        log(
            f"{name:10s} {c['summary']['n_pairs']:3d} pairs  P_miss at 5 %: hamming "
            f"{s['hamming']['0.05']['median']:.3f} ({s['hamming']['0.05']['min']:.3f}-"
            f"{s['hamming']['0.05']['max']:.3f}), masked {s['masked_hamming']['0.05']['median']:.3f}"
        )
    print(json.dumps({k: v["holds"] for k, v in result["predictions"].items()}, indent=1))
    print(f"written to {out.relative_to(ROOT)} and {fig_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
