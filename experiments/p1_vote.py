"""A likelihood-ratio test against the averaging attack, built without any fake.

The Neyman-Pearson test of the thesis (Tutt 2025, eqs. 6.18-6.21, p. 125-127) compares
the flip probabilities of originals, P_b, with those of fakes, Q_b, which it learns on
fakes. Here Q_b is predicted instead, from the originals alone:

1. Simulated attacker (cdpbpc.two_pass.vote_estimate): a majority vote over many reads
   of a print gets an interior symbol wrong exactly where P_b of its pattern is > 0.5.
2. Q_vote: that estimate goes through the two passes (cdpbpc.two_pass.vote_profile),
   pooled over the templates of block E, with the codebook P_b of block E.
3. lrt_vote = sum over patterns of D log(Q/P) + (L - D) log((1 - Q)/(1 - P)),
   P and Q kept in [1e-3, 1 - 1e-3] (cdpbpc.scores.lrt).
4. Conformal threshold on the originals of block S.

Scores compared on Scantrust batch_cdp: hamming (trivial baseline), centered_z on the
patterns with P_b >= 0.5 only (the same idea without simulation), lrt_vote, and
masked_hamming (mu = 0.01) for reference. Measures, per template as the 50 fakes of a
template are not independent: P_fa at the target P_miss with the realized P_miss,
bootstrap interval over templates, P_fa at equal P_miss (threshold on the test
originals), AUC, and P_err^min (oracle, to compare with the thesis only).

Decision rule, set before the run (development, group 0, otsu+unsharp): the test is kept
if AUC(lrt_vote) >= AUC(hamming) + 0.05. Only then the "union" variant is reported:
reject if masked_hamming or lrt_vote is above its own threshold, each set at alpha / 2.

--stage dev reads only the fakes of batch group 0 (blocks: E = 50 originals of group 1,
S = groups 3 and 4, T = group 0). --stage test runs once after the methods freeze, on
groups 1 to 4 (leave one group out), and needs freeze_commit in the config.

Usage:
    python -m experiments.p1_vote --config configs/p1.yaml --stage dev
Writes results/p1_<config hash>.json at the development stage and
results/p1_<config hash>_test.json at the test stage, which refuses to overwrite it.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cdpbpc.calibrate import conformal_threshold
from cdpbpc.counts import scantrust_counts
from cdpbpc.io import SCANTRUST_DEV_GROUP, STAGES, decode_template
from cdpbpc.metrics import (
    auc,
    bootstrap_mean_ci,
    design_effect,
    p_err_min,
    rate_per_group,
    wilcoxon_paired,
)
from cdpbpc.patterns import interior
from cdpbpc.scores import centered_z, hamming, lrt, masked_hamming
from cdpbpc.splits import batch_fold
from cdpbpc.two_pass import vote_estimate, vote_profile
from experiments.e0_sanity import _round, codebook_of, git_state

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCORES = ("hamming", "centered_z_high", "lrt_vote", "masked_hamming")


# ------------------------------------------------------------------------------- data


def check_stage_fakes(rows: pd.DataFrame, stage: str) -> None:
    """Refuse, at the development stage, any batch fake outside the development group."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if stage == "dev":
        fake = ((rows["dataset"] == "batch") & (rows["class"] == "fake")).to_numpy()
        if (rows["group"].to_numpy()[fake] != SCANTRUST_DEV_GROUP).any():
            raise RuntimeError("a test fake was read at the development stage")


def batch_rows(rows: pd.DataFrame, attacks) -> np.ndarray:
    """Positions of the batch originals and of the fakes of the given attacks."""
    use = rows["use"].astype(str).str.lower() == "true"
    keep = (rows["dataset"] == "batch") & use
    keep &= (rows["class"] == "original") | rows["attack"].isin(list(attacks))
    return np.flatnonzero(keep.to_numpy())


def read_templates(data_dir: Path, template_ids) -> dict:
    """Templates of the given ids (batch template k is unique template 3500 + k)."""
    folder = Path(data_dir) / "unique_cdp" / "template"
    return {
        int(t): decode_template((folder / f"{int(t):05d}.png").read_bytes())
        for t in sorted(set(np.asarray(template_ids).tolist()))
    }


class Batch:
    """Counts of batch images. ids, prints: (N,); D, L: (N, 512); kind: (N,) str."""

    def __init__(self, ids, prints, kind, D, L):
        self.ids, self.prints, self.kind = np.asarray(ids), np.asarray(prints), np.asarray(kind)
        self.D, self.L = D, L

    def take(self, mask):
        return Batch(self.ids[mask], self.prints[mask], self.kind[mask], self.D[mask], self.L[mask])

    def of(self, template_ids):
        return self.take(np.isin(self.ids, template_ids))

    def __len__(self):
        return len(self.ids)


# ----------------------------------------------------------------------------- scores


class Detector:
    """Codebook of block E and the vote profile predicted from it, without any fake."""

    def __init__(self, E: Batch, templates: dict, mu: float):
        self.P = codebook_of(E.D, E.L)
        tids, mult = np.unique(E.ids, return_counts=True)
        t = np.stack([templates[int(i)] for i in tids])
        self.Q = vote_profile(t, self.P, weights=mult)
        self.high = self.P >= 0.5
        self.mu = mu
        # facts on the simulated attacker, over the templates of E
        t_hat = vote_estimate(t, self.P)
        L_E = E.L.sum(axis=0)
        self.info = {
            "E_images": len(E),
            "E_templates": len(tids),
            "patterns_flipped_by_the_vote": int((self.P > 0.5).sum()),
            "patterns_with_P_b_ge_0.5": int(self.high.sum()),
            "estimate_error_rate_E": float((interior(t_hat) != interior(t)).mean()),
            "share_of_E_symbols_with_P_b_ge_0.5": float(L_E[self.high].sum() / L_E.sum()),
            "patterns_absent_from_E": int((L_E == 0).sum()),
            "mean_Q_minus_P_over_E_symbols": float(L_E @ (self.Q - self.P) / L_E.sum()),
        }

    def scores(self, D: np.ndarray, L: np.ndarray) -> dict:
        h = self.high.astype(D.dtype)
        out = {
            "hamming": hamming(D, L),
            "centered_z_high": centered_z(D * h, L * h, self.P),
            "lrt_vote": lrt(D, L, self.P, self.Q),
            "masked_hamming": masked_hamming(D, L, self.P, self.mu),
        }
        for k, v in out.items():
            if not np.isfinite(v).all():
                raise ValueError(f"{k}: an image has no symbol to score")
        return out


# ---------------------------------------------------------------------------- measures


def per_template(o, f, tau_o, tau_f, ids_o, ids_f):
    """P_miss and P_fa per template at thresholds tau (scalars or per image)."""
    _, miss = rate_per_group(o > tau_o, ids_o)
    _, fa = rate_per_group(f <= tau_f, ids_f)
    return miss, fa


def summarize(folds: dict, alphas, rng, n_boot: int) -> dict:
    """Measures of every score for one attack.

    folds: {g: {"S": {score: (n,)}, "O": {score: (n,)}, "F": {score: (m,)},
    "ids_o": (n,), "ids_f": (m,)}}; thresholds of a fold come from its S.
    """
    gs = sorted(folds)

    def cat(key, k):
        return np.concatenate([folds[g][key][k] for g in gs])

    ids_o = np.concatenate([folds[g]["ids_o"] for g in gs])
    ids_f = np.concatenate([folds[g]["ids_f"] for g in gs])
    out, fa_conf = {}, {}
    for k in SCORES:
        o, f = cat("O", k), cat("F", k)
        res = {"auc": auc(o, f), "p_err_min_oracle": p_err_min(o, f)}
        for a in alphas:
            tau = {g: conformal_threshold(folds[g]["S"][k], a) for g in gs}
            eq = {g: float(np.quantile(folds[g]["O"][k], 1 - a)) for g in gs}
            rows = {}
            for name, th in (("conformal", tau), ("equal_p_miss_test_originals", eq)):
                tau_o = np.concatenate([np.full(len(folds[g]["ids_o"]), th[g]) for g in gs])
                tau_f = np.concatenate([np.full(len(folds[g]["ids_f"]), th[g]) for g in gs])
                miss, fa = per_template(o, f, tau_o, tau_f, ids_o, ids_f)
                rows[name] = {
                    "p_miss": float(miss.mean()),
                    "p_fa": float(fa.mean()),
                    "p_fa_ci_template_bootstrap": list(bootstrap_mean_ci(fa, rng, n_boot)),
                    "p_fa_per_template": fa.tolist(),
                    "n_templates": len(fa),
                }
                if name == "conformal":
                    rows[name]["tau"] = [tau[g] for g in gs]
                    rows[name]["deff_fakes"] = design_effect(f <= tau_f, ids_f)
                    fa_conf[(k, a)] = fa
            res[f"alpha_{a}"] = rows
        out[k] = res
    out["wilcoxon_lrt_vote_vs_hamming"] = {
        f"alpha_{a}": wilcoxon_paired(fa_conf[("lrt_vote", a)], fa_conf[("hamming", a)])
        for a in alphas
    }
    return out


def union(folds: dict, alphas, rng, n_boot: int) -> dict:
    """Reject if masked_hamming or lrt_vote is above its own conformal threshold at a / 2."""
    gs = sorted(folds)
    out = {}
    for a in alphas:
        miss_all, fa_all = [], []
        for g in gs:
            fd = folds[g]
            rej_o = np.zeros(len(fd["ids_o"]), bool)
            rej_f = np.zeros(len(fd["ids_f"]), bool)
            for k in ("masked_hamming", "lrt_vote"):
                tau = conformal_threshold(fd["S"][k], a / 2)
                rej_o |= fd["O"][k] > tau
                rej_f |= fd["F"][k] > tau
            miss_all.append(rate_per_group(rej_o, fd["ids_o"])[1])
            fa_all.append(rate_per_group(~rej_f, fd["ids_f"])[1])
        miss, fa = np.concatenate(miss_all), np.concatenate(fa_all)
        out[f"alpha_{a}"] = {
            "p_miss": float(miss.mean()),
            "p_fa": float(fa.mean()),
            "p_fa_ci_template_bootstrap": list(bootstrap_mean_ci(fa, rng, n_boot)),
            "p_fa_per_template": fa.tolist(),
            "n_templates": len(fa),
        }
    return out


# ------------------------------------------------------------------------------- main


def run(data_dir: Path, cfg: dict, stage: str, rng, log) -> dict:
    c = scantrust_counts(data_dir, stage=stage)
    pos = batch_rows(c.rows, cfg["attacks"])
    rows = c.rows.iloc[pos]
    check_stage_fakes(rows, stage)
    kind = np.where(rows["class"] == "original", "original", rows["attack"].astype(str))
    allb = Batch(
        rows["template_id"].to_numpy(int), rows["print_id"].to_numpy(int), kind, c.D[pos], c.L[pos]
    )
    templates = read_templates(data_dir, allb.ids)
    orig = allb.take(allb.kind == "original")
    groups = [SCANTRUST_DEV_GROUP] if stage == "dev" else [1, 2, 3, 4]
    folds = {a: {} for a in cfg["attacks"]}
    info = {}
    for g in groups:
        fold = batch_fold(g)
        E = orig.take(np.isin(orig.prints, fold["e_print_ids"]))
        det = Detector(E, templates, cfg["mu"])
        S, O = orig.of(fold["blocks"]["S"]), orig.of(fold["blocks"]["T"])
        s_S, s_O = det.scores(S.D, S.L), det.scores(O.D, O.L)
        info[g] = {**det.info, "n_S": len(S), "n_T_originals": len(O)}
        log(f"fold {g}: {det.info}")
        for a in cfg["attacks"]:
            F = allb.take(allb.kind == a).of(fold["blocks"]["T"])
            folds[a][g] = {
                "S": s_S,
                "O": s_O,
                "F": det.scores(F.D, F.L),
                "ids_o": O.ids,
                "ids_f": F.ids,
            }
            info[g][f"n_fakes_{a}"] = len(F)
    out = {"folds": info, "attacks": {}}
    for a in cfg["attacks"]:
        out["attacks"][a] = summarize(folds[a], cfg["alphas"], rng, cfg["n_boot"])
        r = out["attacks"][a]
        log(
            f"  {a}: AUC hamming {r['hamming']['auc']:.4f}, "
            f"centered_z_high {r['centered_z_high']['auc']:.4f}, "
            f"lrt_vote {r['lrt_vote']['auc']:.4f}, masked {r['masked_hamming']['auc']:.4f}"
        )
    d = cfg["decision"]
    if stage == "dev":
        r = out["attacks"][d["attack"]]
        gain = r["lrt_vote"]["auc"] - r["hamming"]["auc"]
        out["decision"] = {
            "rule": f"group 0, {d['attack']}: AUC(lrt_vote) >= AUC(hamming) + {d['min_gain']}",
            "auc_hamming": r["hamming"]["auc"],
            "auc_lrt_vote": r["lrt_vote"]["auc"],
            "gain": gain,
            "keep": bool(gain >= d["min_gain"]),
        }
        do_union = out["decision"]["keep"]
    else:
        do_union = bool(cfg.get("union"))
    if do_union:
        out["union"] = {
            a: union(folds[a], cfg["alphas"], rng, cfg["n_boot"]) for a in cfg["attacks"]
        }
    return out


def result_path(text: str, stage: str) -> Path:
    """Result file of a config text: p1_<hash>.json, or p1_<hash>_test.json at the test stage.

    The test result has its own name so that a later development run with the same
    config can never overwrite it.
    """
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    return RESULTS / f"p1_{h}{'_test' if stage == 'test' else ''}.json"


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
    parser = argparse.ArgumentParser(description="A fake-free test against the averaging attack")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p1.yaml")
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args(argv)
    text = args.config.read_text()
    cfg = yaml.safe_load(text)
    if args.stage == "test" and not cfg.get("freeze_commit"):
        raise SystemExit("--stage test needs freeze_commit in the config (methods freeze)")
    if args.stage == "test" and not isinstance(cfg.get("union"), bool):
        raise SystemExit("--stage test needs union: true or false in the config")
    out = result_path(text, args.stage)
    if args.stage == "test" and out.exists():
        raise SystemExit(f"{out.name} exists: the test stage runs once")
    if args.stage == "test":
        check_freeze_commit(cfg["freeze_commit"])
    start = time.time()

    def log(msg):
        print(f"[{time.time() - start:6.0f} s] {msg}", flush=True)

    rng = np.random.default_rng(cfg["seed"])
    result = {"experiment": "p1_vote", "stage": args.stage, "git": None, "config": cfg}
    result["scantrust_batch"] = run(ROOT / cfg["data_dir"], cfg, args.stage, rng, log)
    result["seconds"] = round(time.time() - start)
    result["git"] = git_state()
    out.parent.mkdir(exist_ok=True)
    with out.open("x" if args.stage == "test" else "w") as f:
        f.write(json.dumps(_round(result, 5), indent=1))
    if "decision" in result["scantrust_batch"]:
        print(json.dumps(_round(result["scantrust_batch"]["decision"], 4), indent=1))
    print(f"written to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
