"""Does a threshold set on originals still hold when the condition changes?

A detector is enrolled in a reference condition (one file of originals): codebook on
E = 50 originals, mean and covariance of binned_mahalanobis on M = 200, conformal
threshold on S = 400, templates drawn at random. It is then applied to the originals of
a target condition (another file), leaving out the templates used for enrolment, and
we measure the realized P_miss for a target P_miss alpha. Changes: none (same file,
other templates), other pages, scan, capture, print run, session, printer, phone.

Then the threshold is redone on K originals of the target (codebook kept), 50 random
draws per K; the realized P_miss on the other originals of the target is compared with
the Beta band of a conformal threshold on K originals. Two comparison points only: the
reference threshold moved by the mean gap (mean shift) and, for centered_z, the
analytic BPC threshold z_(1 - alpha).

Originals only: every file must hold originals (checked in its metadata), no fake is
read. Every pair of files is written in the config.

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
from cdpbpc.counts import indigo_counts
from cdpbpc.scores import (
    MahalanobisFit,
    binned_mahalanobis,
    centered_z,
    fit_binned_mahalanobis,
    hamming,
    masked_hamming,
    pll,
)
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

FIG_CHANGES = ("none", "pages", "scan", "capture", "print_run", "session", "printer", "phone")
FIG_LABELS = {
    "none": "none",
    "pages": "pages",
    "scan": "scan",
    "capture": "capture",
    "print_run": "print\nrun",
    "session": "session",
    "printer": "printer",
    "phone": "phone",
}


def draw_figure(result: dict, out: Path) -> None:
    """Fig. 2: realized P_miss at the figure alpha, one point per pair, two scores."""
    import matplotlib.pyplot as plt

    from cdpbpc.plotting import C2_SCORE_COLORS, setup

    setup()
    cfg = result["config"]
    a, K = cfg["figure"]["alpha"], cfg["figure"]["K"]
    scores = cfg["figure"]["scores"]
    names = [c for c in FIG_CHANGES if c in result["changes"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    rng = np.random.default_rng(0)  # jitter only
    panels = (
        ("threshold set in the reference condition", cfg["sizes"]["S"], "fixed"),
        (f"threshold redone on K = {K} target originals", K, "recal"),
    )
    for ax, (title, n_band, kind) in zip(axes, panels):
        lo, hi = p_miss_band(n_band, a, cfg["band_level"])
        ax.axhspan(lo, hi, color="0.88", lw=0, zorder=0)
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
        ax.set_xticks(range(len(names)), [FIG_LABELS[c] for c in names])
        ax.set_title(title)
        ax.set_xlabel("what changes between reference and target")
        ax.set_yscale("function", functions=(np.sqrt, np.square))
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0], ["0", "1", "5", "10", "20", "50", "100"])
    axes[0].set_ylabel(f"realized P_miss (%), target {a:.0%}")
    axes[0].legend(loc="upper left", frameon=False, markerscale=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- run


def run(cfg: dict, load, log) -> dict:
    """All changes of the config. load(name) -> counts of one file of originals."""
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
    return {"changes": changes}


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
    result.update(run(cfg, load, log))
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
