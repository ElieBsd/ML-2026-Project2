"""A detector enrolled in one condition, judged on the originals and fakes of another.

Fakes exist in one session only (session 0 at the scanner, capture 1 of the phones in
these files), so the transfer with fakes covers two changes:
- session (scanner): enrolled on the originals of session 1 (print run 1, scan 1) of a
  printer, applied to session 0 of the same printer: its originals and the four kinds of
  fakes of the base scanner set;
- phone: enrolled on the originals of one phone (capture 1), applied to the originals
  and fakes of the other phone (capture 1).

The codebook (E) and the mean and covariance of binned_mahalanobis (M) always come from
the reference. Two thresholds:
- fixed: the conformal threshold of reference originals;
- recalibrated: the conformal threshold of K target originals, 50 random draws per K
  (the same draws for every score), codebook kept.
For each we report the realized P_miss on the judged target originals and the P_fa on
the judged fakes, with Clopper-Pearson intervals for the fixed threshold, and the AUC
(which tells whether the codebook still ranks fakes above originals).

Blocks follow the detection experiment, so the rows compare with its same-condition
rows. Templates are those with an original in both files.
- --stage dev: E, M, S drawn among reference templates >= 289; judged: target templates
  1..288; K target originals drawn among the templates of S.
- --stage test: E, M drawn among reference templates 1..288; judged: the two halves of
  the test range, the threshold of one half coming from the reference originals of the
  other half (cross-conformal), and the K target originals of one half drawn in the other.
A template of E or M is never judged and never used to recalibrate.

--stage dev reads only the development fakes. --stage test runs once after the methods
freeze and needs freeze_commit in the config.

Usage:
    python -m experiments.c2a_fakes --config configs/c2a_fakes.yaml --stage dev
Writes results/c2a_fakes_<config hash>.json at the development stage and
results/c2a_fakes_<config hash>_test.json at the test stage, which refuses to overwrite it.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from cdpbpc.calibrate import conformal_threshold, cross_conformal_thresholds, p_miss_band
from cdpbpc.codebook import estimate
from cdpbpc.counts import indigo_counts
from cdpbpc.io import STAGES
from cdpbpc.metrics import auc, clopper_pearson
from cdpbpc.scores import fit_binned_mahalanobis
from cdpbpc.splits import indigo_blocks
from experiments.c2a_transfer import SCORES, all_scores, seeded
from experiments.c3a_detection import check_freeze_commit
from experiments.e0_sanity import _round, git_state

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


# ----------------------------------------------------------------------------- data


class Images:
    """Pattern counts of one file. ids: (N,) template_id."""

    def __init__(self, ids, D, L):
        self.ids, self.D, self.L = np.asarray(ids), np.asarray(D), np.asarray(L)

    def of(self, template_ids) -> "Images":
        m = np.isin(self.ids, template_ids)
        return Images(self.ids[m], self.D[m], self.L[m])

    def __len__(self):
        return len(self.ids)


def load(pack: Path, name: str, stage: str, cls: str) -> Images:
    c = indigo_counts(pack / f"{name}.npz", stage=stage)
    if c.meta["class"] != cls:
        raise ValueError(f"{name}: expected {cls}, found {c.meta['class']}")
    return Images(c.ids, c.D, c.L)


class Detector:
    """Codebook from E, binned_mahalanobis fitted on M (reference originals)."""

    def __init__(self, E: Images, M: Images, mu: float):
        self.codebook = estimate(E.L.sum(axis=0), E.D.sum(axis=0))
        self.fit = fit_binned_mahalanobis(M.D, M.L, self.codebook)
        self.mu = mu
        self.n_E, self.n_M = len(E), len(M)

    def scores(self, c: Images) -> dict:
        return all_scores(c.D, c.L, self.codebook, self.fit, self.mu)


# ----------------------------------------------------------------------------- measures


def fixed_rates(o, f, tau_o, tau_f) -> dict:
    """Realized P_miss and P_fa, one image per template, Clopper-Pearson intervals."""
    miss, acc = o > tau_o, f <= tau_f
    return {
        "p_miss": float(miss.mean()),
        "p_miss_ci": list(clopper_pearson(int(miss.sum()), len(miss))),
        "p_fa": float(acc.mean()),
        "p_fa_ci": list(clopper_pearson(int(acc.sum()), len(acc))),
    }


def spread(v: np.ndarray) -> dict:
    return {
        "mean": float(v.mean()),
        "q10": float(np.quantile(v, 0.1)),
        "q90": float(np.quantile(v, 0.9)),
    }


# ----------------------------------------------------------------------------- one change


def transfer(setting, ref: Images, tgt: Images, fakes: dict, cfg: dict, alphas, stage, rng):
    """One reference -> target change: fixed and recalibrated thresholds, every fake kind.

    ref, tgt: originals of the reference and target files; fakes: {kind: Images} of the
    target condition. Returns the measures and the sizes of the blocks.
    """
    available = np.intersect1d(ref.ids, tgt.ids)
    b = indigo_blocks(setting, stage=stage, available=available, seed=cfg["seed"])
    det = Detector(ref.of(b["E"]), ref.of(b["M"]), cfg["mu"])
    if stage == "dev":
        halves = [b["V"]]  # one judged part, threshold and recalibration pool outside it
        cal = det.scores(ref.of(b["S"]))
        pools = [tgt.of(b["S"])]
        ref_tau = {k: {a: [conformal_threshold(cal[k], a)] for a in alphas} for k in SCORES}
    else:
        halves = [b["T_a"], b["T_b"]]
        both = ref.of(np.r_[b["T_a"], b["T_b"]])
        in_b = np.isin(both.ids, b["T_b"])
        cal = det.scores(both)
        pools = [tgt.of(b["T_b"]), tgt.of(b["T_a"])]  # the other half
        ref_tau = {
            k: {a: list(cross_conformal_thresholds(cal[k], in_b, a)) for a in alphas}
            for k in SCORES
        }
    judged_o = [det.scores(tgt.of(h)) for h in halves]
    judged_f = {kind: [det.scores(fk.of(h)) for h in halves] for kind, fk in fakes.items()}
    pool_s = [det.scores(p) for p in pools]
    used = np.r_[b["E"], b["M"]]
    assert not np.isin(np.concatenate(halves), used).any()
    assert not any(np.isin(p.ids, used).any() for p in pools)

    def cat(parts, k):
        return np.concatenate([p[k] for p in parts])

    def per_image(taus, parts):
        return np.concatenate([np.full(len(p["hamming"]), t) for t, p in zip(taus, parts)])

    # the same K draws for every score and alpha
    draws = {
        K: [
            [rng.choice(len(p), K, replace=False) for p in pools] for _ in range(cfg["recal_draws"])
        ]
        for K in cfg["recal_K"]
        if all(K < len(p) for p in pools)
    }
    out = {
        "n_E": det.n_E,
        "n_M": det.n_M,
        "n_reference_threshold": len(cal["hamming"]),
        "n_recal_pool": [len(p) for p in pools],
        "n_originals": int(sum(len(o["hamming"]) for o in judged_o)),
        "n_fakes": {kind: int(sum(len(x["hamming"]) for x in v)) for kind, v in judged_f.items()},
        "fakes": {},
    }
    for kind, jf in judged_f.items():
        res = {}
        for k in SCORES:
            o, f = cat(judged_o, k), cat(jf, k)
            r = {"auc": auc(o, f)}
            for a in alphas:
                ra = {
                    "fixed": fixed_rates(
                        o, f, per_image(ref_tau[k][a], judged_o), per_image(ref_tau[k][a], jf)
                    )
                }
                ra["recal"] = {}
                for K, ds in draws.items():
                    miss, fa = np.empty(len(ds)), np.empty(len(ds))
                    for i, idx in enumerate(ds):
                        taus = [conformal_threshold(ps[k][j], a) for ps, j in zip(pool_s, idx)]
                        miss[i] = np.mean(o > per_image(taus, judged_o))
                        fa[i] = np.mean(f <= per_image(taus, jf))
                    lo, hi = p_miss_band(K, a, cfg["band_level"])
                    ra["recal"][K] = {
                        "p_miss": spread(miss),
                        "p_fa": spread(fa),
                        "share_p_miss_in_band": float(np.mean((miss >= lo) & (miss <= hi))),
                    }
                r[f"alpha_{a}"] = ra
            res[k] = r
        out["fakes"][kind] = res
    return out


# ----------------------------------------------------------------------------- run


def run(data_dir: Path, cfg: dict, stage: str, log) -> dict:
    pack = data_dir / "indigo_compact"
    out = {"session": {}, "phone": {}}
    sc = cfg["session"]
    fakes = {
        f"f{p[3:]}/{e[4:]}": load(pack, f"base_fake_{p}_{e}", stage, "fake")
        for p, e in (f.split("_") for f in sc["fakes"])
    }
    for printer in sc["printers"]:
        ref = load(pack, sc["reference"].format(printer=printer), stage, "original")
        tgt = load(pack, f"base_original_{printer}", stage, "original")
        rng = seeded(cfg["seed"], "session", printer)
        res = transfer("scanner", ref, tgt, fakes, cfg, sc["alphas"], stage, rng)
        out["session"][printer] = res
        _log_row(log, f"session 1 -> 0, {printer}", res, sc["alphas"][-1], cfg["recal_K"])
    ph = cfg["phone"]
    for refd, tgtd in ph["pairs"]:
        run_ = ph["capture_run"]
        ref = load(pack, f"mobile_original_{refd}_run{run_}", stage, "original")
        tgt = load(pack, f"mobile_original_{tgtd}_run{run_}", stage, "original")
        fk = {"EHPI55": load(pack, f"mobile_fake_EHPI55_{tgtd}_run{run_}", stage, "fake")}
        rng = seeded(cfg["seed"], "phone", refd, tgtd)
        res = transfer("mobile", ref, tgt, fk, cfg, ph["alphas"], stage, rng)
        out["phone"][f"{refd}_to_{tgtd}"] = res
        _log_row(log, f"phone {refd} -> {tgtd}", res, ph["alphas"][-1], cfg["recal_K"])
    return out


def _log_row(log, title, res, a, Ks):
    K = max(
        k for k in Ks if k in next(iter(res["fakes"].values()))["hamming"][f"alpha_{a}"]["recal"]
    )
    for kind, m in res["fakes"].items():
        parts = []
        for k in ("hamming", "masked_hamming"):
            x = m[k][f"alpha_{a}"]
            parts.append(
                f"{k} fixed {x['fixed']['p_miss']:.3f}/{x['fixed']['p_fa']:.3f}, "
                f"K={K} {x['recal'][K]['p_miss']['mean']:.3f}/{x['recal'][K]['p_fa']['mean']:.3f}"
            )
        log(f"{title} {kind} (P_miss/P_fa at {a}): " + "; ".join(parts))


def result_path(text: str, stage: str) -> Path:
    """c2a_fakes_<hash>.json, or c2a_fakes_<hash>_test.json at the test stage."""
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    return RESULTS / f"c2a_fakes_{h}{'_test' if stage == 'test' else ''}.json"


def main(argv: list[str] | None = None, run_fn=run) -> int:
    parser = argparse.ArgumentParser(description="Transfer of a detector, with fakes")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "c2a_fakes.yaml")
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

    result = {"experiment": "c2a_fakes", "stage": args.stage, "git": git_state(), "config": cfg}
    result.update(run_fn(ROOT / cfg["data_dir"], cfg, args.stage, log))
    result["seconds"] = round(time.time() - start)
    out.parent.mkdir(exist_ok=True)
    with out.open("x" if args.stage == "test" else "w") as f:
        f.write(json.dumps(_round(result, 5), indent=1))
    print(f"written to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
