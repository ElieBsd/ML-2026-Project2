"""Which fake-free score detects which fakes, at the operating points.

Scores: hamming, centered_z, pcor, pll, masked_hamming (mu = 0.01) and binned_mahalanobis
(4 classes). The codebook comes from block E, the mean and covariance of
binned_mahalanobis from block M; thresholds are conformal, set on originals only.

Measures: P_fa at the target P_miss with the realized P_miss next to it; P_fa at equal
P_miss (threshold = quantile of the judged originals, labelled as such: it compares
scores without the calibration error); AUC; P_err^min (oracle, only to compare with
Table 5.3 of Tutt 2025). Flip rate of originals and of each kind of fake in each of the
8 reliability classes.

Data and blocks:
- Indigo phones (capture 1) and scanner (both printers, four kinds of fakes): blocks E,
  M, S, V at the development stage; at the test stage E and M from development
  templates and the cross-conformal threshold between the two halves of the test range.
  Clopper-Pearson intervals.
- Scantrust unique_cdp: 20 draws of E, M, S, T_orig among templates 1..3500; fakes of
  3551..4000 (development) or 4001..5000 (test). Mean and spread over draws,
  Clopper-Pearson for draw 0.
- Scantrust batch_cdp: one group of 10 templates out (group 0 at the development stage,
  groups 1-4 at the test stage). Rates per template, bootstrap on templates.

--stage dev reads only the development fakes. --stage test runs once after the methods
freeze and needs freeze_commit in the config.

Usage:
    python -m experiments.c3a_detection --config configs/c3a.yaml --stage dev
Writes results/c3a_<config hash>.json at the development stage and
results/c3a_<config hash>_test.json at the test stage, which refuses to overwrite it.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from cdpbpc.calibrate import conformal_threshold, cross_conformal_thresholds
from cdpbpc.codebook import estimate
from cdpbpc.counts import indigo_counts, scantrust_counts
from cdpbpc.io import STAGES, load_compact
from cdpbpc.metrics import (
    auc,
    bootstrap_mean_ci,
    clopper_pearson,
    design_effect,
    p_err_min,
    rate_per_group,
)
from cdpbpc.scores import (
    CLASS_EDGES_8,
    binned_mahalanobis,
    centered_z,
    fit_binned_mahalanobis,
    hamming,
    masked_hamming,
    pll,
    reliability_class,
)
from cdpbpc.splits import batch_fold, indigo_blocks, unique_blocks
from experiments.e0_sanity import _round, git_state

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCORES = ("hamming", "centered_z", "pcor", "pll", "masked_hamming", "binned_mahalanobis")


# ----------------------------------------------------------------------------- data


class Counts:
    """Pattern counts and pcor of a set of images. ids: (N,) template_id."""

    def __init__(self, ids, D, L, pcor, prints=None):
        self.ids, self.D, self.L = np.asarray(ids), np.asarray(D), np.asarray(L)
        self.pcor = np.asarray(pcor, dtype=np.float64)
        self.prints = None if prints is None else np.asarray(prints)

    def take(self, mask):
        p = None if self.prints is None else self.prints[mask]
        return Counts(self.ids[mask], self.D[mask], self.L[mask], self.pcor[mask], p)

    def of(self, template_ids):
        return self.take(np.isin(self.ids, template_ids))

    def __len__(self):
        return len(self.ids)


def concat(*parts: Counts) -> Counts:
    return Counts(
        np.concatenate([p.ids for p in parts]),
        np.concatenate([p.D for p in parts]),
        np.concatenate([p.L for p in parts]),
        np.concatenate([p.pcor for p in parts]),
    )


def check_pcor(c: Counts, what: str) -> None:
    """pcor of test fakes is withheld in the compact pack until the methods freeze."""
    if np.isnan(c.pcor).any():
        raise SystemExit(
            f"{what}: pcor is missing for {int(np.isnan(c.pcor).sum())} images; after the "
            "freeze, rebuild the fakes with: python scripts/make_compact.py --test-pcor "
            '--overwrite --only "*fake*"'
        )


# ----------------------------------------------------------------------------- detector


class Detector:
    """Codebook from E, binned_mahalanobis fitted on M; all scores from counts."""

    def __init__(self, E: Counts, M: Counts, mu: float):
        self.codebook = estimate(E.L.sum(axis=0), E.D.sum(axis=0))
        self.fit = fit_binned_mahalanobis(M.D, M.L, self.codebook)
        self.mu = mu
        self.n_E, self.n_M = len(E), len(M)

    def scores(self, c: Counts) -> dict:
        cb = self.codebook
        return {
            "hamming": hamming(c.D, c.L),
            "centered_z": centered_z(c.D, c.L, cb),
            "pcor": c.pcor,
            "pll": pll(c.D, c.L, cb),
            "masked_hamming": masked_hamming(c.D, c.L, cb, self.mu),
            "binned_mahalanobis": binned_mahalanobis(c.D, c.L, cb, self.fit),
        }

    def class_sums(self, c: Counts) -> np.ndarray:
        """(2, 8): flips and symbols of the images in each reliability class of the codebook."""
        onehot = reliability_class(self.codebook, CLASS_EDGES_8)[:, None] == np.arange(8)
        return np.stack([c.D.sum(axis=0) @ onehot, c.L.sum(axis=0) @ onehot]).astype(float)


def profile(sums: np.ndarray) -> list:
    """Flip rate per class from (2, 8) sums; None for an empty class."""
    return [float(d / n) if n > 0 else None for d, n in zip(*sums)]


# ----------------------------------------------------------------------------- measures


def rates_images(o, f, tau_o, tau_f) -> dict:
    """One image per template: rates with Clopper-Pearson intervals. tau: scalar or per image."""
    miss, acc = o > tau_o, f <= tau_f
    return {
        "p_miss": float(miss.mean()),
        "p_miss_ci": list(clopper_pearson(int(miss.sum()), len(miss))),
        "p_fa": float(acc.mean()),
        "p_fa_ci": list(clopper_pearson(int(acc.sum()), len(acc))),
        "n_originals": len(miss),
        "n_fakes": len(acc),
        "fake_correct": (~acc).astype(int).tolist(),  # for McNemar between scores
    }


def measures_images(o: dict, f: dict, tau: dict, alphas) -> dict:
    """Indigo and unique_cdp. o, f: {score: scores}; tau: {score: {alpha: (tau_o, tau_f)}}."""
    out = {}
    for k in SCORES:
        res = {"auc": auc(o[k], f[k]), "p_err_min_oracle": p_err_min(o[k], f[k])}
        for a in alphas:
            r = rates_images(o[k], f[k], *tau[k][a])
            eq = float(np.quantile(o[k], 1 - a))
            r["equal_p_miss_test_originals"] = rates_images(o[k], f[k], eq, eq)
            del r["equal_p_miss_test_originals"]["fake_correct"]
            res[f"alpha_{a}"] = r
        out[k] = res
    return out


def scalar_tau(cal: dict, alphas, n_o: int, n_f: int) -> dict:
    """Conformal threshold of the calibration originals, the same for every judged image."""
    out = {}
    for k in SCORES:
        out[k] = {}
        for a in alphas:
            t = conformal_threshold(cal[k], a)
            out[k][a] = (np.full(n_o, t), np.full(n_f, t))
    return out


# ----------------------------------------------------------------------------- Indigo


def load_indigo(pack: Path, name: str, stage: str, cls: str) -> Counts:
    path = pack / f"{name}.npz"
    c = indigo_counts(path, stage=stage)
    if c.meta["class"] != cls:
        raise ValueError(f"{name}: expected {cls}, found {c.meta['class']}")
    f = load_compact(path, stage=stage)
    if not np.array_equal(f.ids, c.ids):
        raise ValueError(f"{name}: counts and compact file disagree on templates")
    return Counts(c.ids, c.D, c.L, f.pcor)


def run_indigo_condition(setting, orig: Counts, fakes: dict, alphas, cfg, stage, log) -> dict:
    blocks = indigo_blocks(setting, stage=stage, available=orig.ids, seed=cfg["seed"])
    det = Detector(orig.of(blocks["E"]), orig.of(blocks["M"]), cfg["mu"])
    out = {"n_E": det.n_E, "n_M": det.n_M, "fakes": {}, "class_profile": {}}
    if stage == "dev":
        S, J = orig.of(blocks["S"]), orig.of(blocks["V"])
        cal = det.scores(S)
        out["n_S"] = len(S)
        judged = {name: fk.of(blocks["V"]) for name, fk in fakes.items()}
    else:
        J = concat(orig.of(blocks["T_a"]), orig.of(blocks["T_b"]))
        judged = {
            name: concat(fk.of(blocks["T_a"]), fk.of(blocks["T_b"])) for name, fk in fakes.items()
        }
        for name, fk in judged.items():
            check_pcor(fk, f"{setting} {name}")
    so = det.scores(J)
    out["class_profile"]["originals"] = profile(det.class_sums(J))
    for name, fk in judged.items():
        sf = det.scores(fk)
        if stage == "dev":
            tau = scalar_tau(cal, alphas, len(J), len(fk))
        else:
            in_b_o = np.isin(J.ids, blocks["T_b"])
            in_b_f = np.isin(fk.ids, blocks["T_b"])
            tau = {}
            for k in SCORES:
                tau[k] = {}
                for a in alphas:
                    ta, tb = cross_conformal_thresholds(so[k], in_b_o, a)
                    tau[k][a] = (np.where(in_b_o, tb, ta), np.where(in_b_f, tb, ta))
        out["fakes"][name] = measures_images(so, sf, tau, alphas)
        out["class_profile"][name] = profile(det.class_sums(fk))
        m = out["fakes"][name]
        a = alphas[-1]
        log(
            f"  {name}: P_fa at {a}: hamming {m['hamming'][f'alpha_{a}']['p_fa']:.3f}, "
            f"masked {m['masked_hamming'][f'alpha_{a}']['p_fa']:.3f}, "
            f"pcor {m['pcor'][f'alpha_{a}']['p_fa']:.3f}, pll {m['pll'][f'alpha_{a}']['p_fa']:.3f}"
        )
    return out


def run_indigo(data_dir: Path, cfg: dict, stage: str, log) -> dict:
    pack = data_dir / "indigo_compact"
    out = {}
    mob = cfg["indigo"]["mobile"]
    for dev in mob["devices"]:
        run = mob["capture_run"]
        log(f"Indigo mobile {dev}, capture {run}")
        orig = load_indigo(pack, f"mobile_original_{dev}_run{run}", stage, "original")
        fake = load_indigo(pack, f"mobile_fake_EHPI55_{dev}_run{run}", stage, "fake")
        out[f"mobile_{dev}"] = run_indigo_condition(
            "mobile", orig, {"EHPI55": fake}, mob["alphas"], cfg, stage, log
        )
    sc = cfg["indigo"]["scanner"]
    fakes = {
        f"f{p[3:]}/{e[4:]}": load_indigo(pack, f"base_fake_{p}_{e}", stage, "fake")
        for p, e in (f.split("_") for f in sc["fakes"])
    }
    for printer in sc["printers"]:
        log(f"Indigo scanner, originals {printer}")
        orig = load_indigo(pack, f"base_original_{printer}", stage, "original")
        out[f"scanner_{printer}"] = run_indigo_condition(
            "scanner", orig, fakes, sc["alphas"], cfg, stage, log
        )
    return out


# ----------------------------------------------------------------------------- Scantrust


def scantrust_sets(data_dir: Path, stage: str):
    """Counts of the Scantrust images allowed at this stage, and their manifest rows."""
    sc = scantrust_counts(data_dir, stage=stage)
    r = sc.rows.reset_index(drop=True)
    allc = Counts(r["template_id"].to_numpy(int), sc.D, sc.L, sc.pcor, r["print_id"].to_numpy())
    return allc, r


def run_unique(allc: Counts, r, cfg: dict, stage: str, log) -> dict:
    u = cfg["scantrust"]["unique"]
    is_u = (r["dataset"] == "unique").to_numpy()
    is_o = (r["class"] == "original").to_numpy()
    orig = allc.take(is_u & is_o)
    fakes = {a: allc.take(is_u & (r["attack"] == a).fillna(False).to_numpy()) for a in u["attacks"]}
    per_seed = {a: [] for a in u["attacks"]}
    profiles = {a: np.zeros((2, 8)) for a in ["originals", *u["attacks"]]}
    for seed in range(u["n_seeds"]):
        b = unique_blocks(seed)
        det = Detector(orig.of(b["E"]), orig.of(b["M"]), cfg["mu"])
        cal = det.scores(orig.of(b["S"]))
        T = orig.of(b["T_orig"])
        so = det.scores(T)
        profiles["originals"] += det.class_sums(T)
        block = b["V"] if stage == "dev" else b["T"]
        for a, fk in fakes.items():
            fj = fk.of(block)
            if stage == "test":
                check_pcor(fj, f"unique {a}")
            per_seed[a].append(
                measures_images(
                    so, det.scores(fj), scalar_tau(cal, u["alphas"], len(T), len(fj)), u["alphas"]
                )
            )
            profiles[a] += det.class_sums(fj)
    out = {
        "n_seeds": u["n_seeds"],
        "attacks": {},
        "class_profile": {k: profile(v) for k, v in profiles.items()},
    }
    for a, rows in per_seed.items():
        res = {}
        for k in SCORES:
            aucs = [x[k]["auc"] for x in rows]
            res[k] = {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1))}
            for al in u["alphas"]:
                key = f"alpha_{al}"
                fa = [x[k][key]["p_fa"] for x in rows]
                miss = [x[k][key]["p_miss"] for x in rows]
                res[k][key] = {
                    "p_fa_mean": float(np.mean(fa)),
                    "p_fa_sd": float(np.std(fa, ddof=1)),
                    "p_miss_mean": float(np.mean(miss)),
                    "p_miss_sd": float(np.std(miss, ddof=1)),
                    "seed_0": {kk: v for kk, v in rows[0][k][key].items() if kk != "fake_correct"},
                }
        out["attacks"][a] = res
        log(
            f"  unique {a}: P_fa at 0.05: hamming {res['hamming']['alpha_0.05']['p_fa_mean']:.3f}, "
            f"masked {res['masked_hamming']['alpha_0.05']['p_fa_mean']:.3f}"
        )
    return out


def run_batch(allc: Counts, r, cfg: dict, stage: str, rng, log) -> dict:
    bc = cfg["scantrust"]["batch"]
    is_b = (r["dataset"] == "batch").to_numpy()
    orig = allc.take(is_b & (r["class"] == "original").to_numpy())
    fakes = {
        a: allc.take(is_b & (r["attack"] == a).fillna(False).to_numpy()) for a in bc["attacks"]
    }
    groups = [0] if stage == "dev" else [1, 2, 3, 4]
    folds, info = {}, {}
    profiles = {a: np.zeros((2, 8)) for a in ["originals", *bc["attacks"]]}
    for g in groups:
        fold = batch_fold(g)
        E = orig.take(np.isin(orig.prints, fold["e_print_ids"]))
        det = Detector(E, orig.of(fold["blocks"]["M"]), cfg["mu"])
        info[g] = {
            "n_E": len(E),
            "n_M": det.n_M,
            "n_S": int(np.isin(orig.ids, fold["blocks"]["S"]).sum()),
            "E_print_ids": np.sort(E.prints).tolist(),
        }
        T = orig.of(fold["blocks"]["T"])
        folds[g] = {
            "S": det.scores(orig.of(fold["blocks"]["S"])),
            "O": det.scores(T),
            "ids_o": T.ids,
            "F": {},
        }
        profiles["originals"] += det.class_sums(T)
        for a, fk in fakes.items():
            ft = fk.of(fold["blocks"]["T"])
            if stage == "test":
                check_pcor(ft, f"batch {a}")
            folds[g]["F"][a] = (det.scores(ft), ft.ids)
            profiles[a] += det.class_sums(ft)
    out = {
        "groups": groups,
        "folds": info,
        "attacks": {},
        "class_profile": {k: profile(v) for k, v in profiles.items()},
    }
    for a in bc["attacks"]:
        res = {}
        for k in SCORES:
            o = np.concatenate([folds[g]["O"][k] for g in groups])
            f = np.concatenate([folds[g]["F"][a][0][k] for g in groups])
            ids_o = np.concatenate([folds[g]["ids_o"] for g in groups])
            ids_f = np.concatenate([folds[g]["F"][a][1] for g in groups])
            rk = {"auc_pooled": auc(o, f), "p_err_min_oracle_pooled": p_err_min(o, f)}
            for al in bc["alphas"]:
                rows = {}
                th = {
                    "conformal": {g: conformal_threshold(folds[g]["S"][k], al) for g in groups},
                    "equal_p_miss_test_originals": {
                        g: float(np.quantile(folds[g]["O"][k], 1 - al)) for g in groups
                    },
                }
                for name, t in th.items():
                    tau_o = np.concatenate([np.full(len(folds[g]["ids_o"]), t[g]) for g in groups])
                    tau_f = np.concatenate(
                        [np.full(len(folds[g]["F"][a][1]), t[g]) for g in groups]
                    )
                    _, miss = rate_per_group(o > tau_o, ids_o)
                    _, fa = rate_per_group(f <= tau_f, ids_f)
                    rows[name] = {
                        "p_miss": float(miss.mean()),
                        "p_fa": float(fa.mean()),
                        "p_fa_ci_template_bootstrap": list(
                            bootstrap_mean_ci(fa, rng, cfg["n_boot"])
                        ),
                        "p_fa_per_template": fa.tolist(),
                        "n_templates": len(fa),
                    }
                    if name == "conformal":
                        rows[name]["deff_fakes"] = design_effect(f <= tau_f, ids_f)
                rk[f"alpha_{al}"] = rows
            res[k] = rk
        out["attacks"][a] = res
        log(
            f"  batch {a}: P_fa at 0.05: hamming {res['hamming']['alpha_0.05']['conformal']['p_fa']:.3f}, "
            f"masked {res['masked_hamming']['alpha_0.05']['conformal']['p_fa']:.3f}"
        )
    return out


# ----------------------------------------------------------------------------- predictions


def predictions(result: dict, cfg: dict) -> dict:
    """The predictions written before any run. At the development stage they are a check
    of the code on templates already seen, not a result."""
    P = cfg["predictions"]
    out = {}
    p = P["replication"]
    m = result["indigo"][f"mobile_{p['device']}"]["fakes"]["EHPI55"]
    key = f"alpha_{p['alpha']}"
    fa_m, fa_h = m["masked_hamming"][key]["p_fa"], m["hamming"][key]["p_fa"]
    out["replication"] = {
        "rule": f"{p['device']}, target P_miss {p['alpha']}: P_fa of masked_hamming < "
        f"{p['max_masked']} and P_fa of hamming > {p['min_hamming']}",
        "p_fa_masked_hamming": fa_m,
        "p_fa_hamming": fa_h,
        "holds": bool(fa_m < p["max_masked"] and fa_h > p["min_hamming"]),
    }
    p = P["operating_point"]
    m = result["indigo"][f"mobile_{p['device']}"]["fakes"]["EHPI55"]
    key = f"alpha_{p['alpha']}"
    fa = {k: m[k][key]["p_fa"] for k in SCORES}
    out["operating_point"] = {
        "rule": f"{p['device']}, target P_miss {p['alpha']}: P_fa > {p['min_p_fa']} for every score",
        "p_fa": fa,
        "holds": bool(all(v > p["min_p_fa"] for v in fa.values())),
    }
    p = P["averaging_breaks_mask"]
    m = result["scantrust_batch"]["attacks"][p["attack"]]
    key = f"alpha_{p['alpha']}"
    fa_m = m["masked_hamming"][key]["conformal"]["p_fa"]
    fa_h = m["hamming"][key]["conformal"]["p_fa"]
    out["averaging_breaks_mask"] = {
        "rule": f"batch {p['attack']}, target P_miss {p['alpha']}: P_fa of masked_hamming >= hamming",
        "p_fa_masked_hamming": fa_m,
        "p_fa_hamming": fa_h,
        "holds": bool(fa_m >= fa_h),
    }
    return out


# ----------------------------------------------------------------------------- main


def result_path(text: str, stage: str) -> Path:
    """c3a_<hash>.json, or c3a_<hash>_test.json at the test stage (never overwritten)."""
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    return RESULTS / f"c3a_{h}{'_test' if stage == 'test' else ''}.json"


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


def run(data_dir: Path, cfg: dict, stage: str, log) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    result = {"indigo": run_indigo(data_dir, cfg, stage, log)}
    log("Scantrust")
    allc, r = scantrust_sets(data_dir, stage)
    log("Scantrust unique_cdp")
    result["scantrust_unique"] = run_unique(allc, r, cfg, stage, log)
    log("Scantrust batch_cdp")
    result["scantrust_batch"] = run_batch(allc, r, cfg, stage, rng, log)
    return result


def main(argv: list[str] | None = None, run_fn=run) -> int:
    parser = argparse.ArgumentParser(description="Fake-free scores at the operating points")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "c3a.yaml")
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

    result = {"experiment": "c3a_detection", "stage": args.stage, "git": git_state(), "config": cfg}
    result.update(run_fn(ROOT / cfg["data_dir"], cfg, args.stage, log))
    result["predictions"] = predictions(result, cfg)
    result["seconds"] = round(time.time() - start)
    out.parent.mkdir(exist_ok=True)
    with out.open("x" if args.stage == "test" else "w") as f:
        f.write(json.dumps(_round(result, 5), indent=1))
    print(json.dumps(_round({k: v["holds"] for k, v in result["predictions"].items()}), indent=1))
    print(f"written to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
