"""Fusing three phone captures of the same print: keep the best one.

A phone capture can be blurred, badly lit or slightly off; an original then looks like a
fake. With three captures of the same print, the fused score is the minimum over the
captures of the masked Hamming distance (mu = 0.01, thesis p. 104): the most
favourable capture decides. It is compared with:
- one capture (capture 1 alone);
- pooled counts: D and L summed over the three captures, then the score, which is the
  multi-shot fusion of the thesis (Tutt 2025, Table 6.1, p. 135).
The trivial baseline hamming is fused the same three ways.

All rules use the same codebook: P_b of the originals of block E, capture 1. Thresholds
are conformal, set on originals fused exactly like the judged images. Only templates
with all three captures are judged (an original needs its three captures; a fake needs
its three captures and the three of its original); the others are counted and listed.

Measures, per phone, rule and level (1 % and 5 %): P_fa at the target P_miss with the
realized P_miss, Clopper-Pearson intervals (one image per template), AUC, P_err^min
(oracle, to compare with the thesis only), and the exact McNemar test of the minimum
against one capture on the same fakes.

Decision rule, set before the run (development only): on Samsung, the minimum of three
captures accepts fewer than 20 % of fakes at a target P_miss of 1 %.

--stage dev reads only the development fakes (Indigo templates 1-288); E and S are drawn
among the originals of templates >= 289. --stage test runs once after the methods freeze
(cross-conformal threshold on the two halves of the test range) and needs
freeze_commit in the config.

Usage:
    python -m experiments.p2_best_of_3 --config configs/p2.yaml --stage dev
Writes results/p2_<config hash>.json (the stage is written inside).
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from cdpbpc.calibrate import conformal_threshold
from cdpbpc.counts import indigo_counts
from cdpbpc.io import INDIGO_DEV_MAX_ID, STAGES
from cdpbpc.metrics import auc, clopper_pearson, mcnemar_exact, p_err_min
from cdpbpc.scores import hamming, masked_hamming
from cdpbpc.splits import indigo_blocks
from experiments.e0_sanity import _round, codebook_of, git_state

ROOT = Path(__file__).resolve().parents[1]
SCORES = ("masked_hamming", "hamming")
RULES = ("single", "pooled", "min3")


# ------------------------------------------------------------------------------- data


def check_dev_fakes(fake_ids: np.ndarray, stage: str) -> None:
    """Refuse, at the development stage, any fake of a test template."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if stage == "dev" and (np.asarray(fake_ids) > INDIGO_DEV_MAX_ID).any():
        raise RuntimeError("a test fake was read at the development stage")


def align(counts) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Templates present in every capture, with their counts in template order.

    counts: one object per capture with ids (N_k,), D and L (N_k, 512).
    Returns ids (N,) sorted, D and L (K, N, 512), and the sorted ids present in some
    captures but not all.
    """
    sets = [set(np.asarray(c.ids).tolist()) for c in counts]
    ids = np.array(sorted(set.intersection(*sets)), dtype=np.int64)
    missing = np.array(sorted(set.union(*sets) - set(ids.tolist())), dtype=np.int64)
    D, L = [], []
    for c in counts:
        order = np.argsort(c.ids)
        pos = order[np.searchsorted(np.asarray(c.ids)[order], ids)]
        D.append(np.asarray(c.D)[pos])
        L.append(np.asarray(c.L)[pos])
    return ids, np.stack(D), np.stack(L), missing


# ----------------------------------------------------------------------------- scores


def fused_scores(D: np.ndarray, L: np.ndarray, codebook: np.ndarray, mu: float) -> dict:
    """Each score under each fusion rule. D, L: (K, N, 512), capture 0 = the single one.

    Returns {score: {rule: (N,)}}: "single" = capture 0, "pooled" = counts summed over
    the captures, "min3" = minimum of the per-capture scores.
    """

    def score(name, d, n):
        if name == "hamming":
            return hamming(d, n)
        return masked_hamming(d, n, codebook, mu)

    out = {}
    for k in SCORES:
        per = np.stack([score(k, D[c], L[c]) for c in range(len(D))])
        out[k] = {
            "single": per[0],
            "pooled": score(k, D.sum(axis=0), L.sum(axis=0)),
            "min3": per.min(axis=0),
        }
        if not np.isfinite(per).all():
            raise ValueError(f"{k}: an image has no symbol to score")
    return out


# ---------------------------------------------------------------------------- measures


def evaluate(parts: list, alphas) -> dict:
    """Measures of every score and rule, pooled over parts.

    parts: [{"cal": fs, "orig": fs, "fake": fs}], fs = fused_scores output. The images
    of a part are judged with the conformal threshold of its "cal" originals: one part
    at the development stage (block S), two at the test stage (the other half).
    """
    out, correct = {}, {}
    for k in SCORES:
        out[k] = {}
        for r in RULES:
            o = np.concatenate([p["orig"][k][r] for p in parts])
            f = np.concatenate([p["fake"][k][r] for p in parts])
            res = {"auc": auc(o, f), "p_err_min_oracle": p_err_min(o, f)}
            for a in alphas:
                taus = [conformal_threshold(p["cal"][k][r], a) for p in parts]
                miss = np.concatenate([p["orig"][k][r] > t for p, t in zip(parts, taus)])
                acc = np.concatenate([p["fake"][k][r] <= t for p, t in zip(parts, taus)])
                res[f"alpha_{a}"] = {
                    "tau": taus,
                    "p_miss": float(miss.mean()),
                    "p_miss_ci": list(clopper_pearson(int(miss.sum()), len(miss))),
                    "p_fa": float(acc.mean()),
                    "p_fa_ci": list(clopper_pearson(int(acc.sum()), len(acc))),
                    "n_originals": len(miss),
                    "n_fakes": len(acc),
                }
                correct[(k, r, a)] = ~acc
            out[k][r] = res
    out["mcnemar_min3_vs_single"] = {}
    for a in alphas:
        m, s = correct[("masked_hamming", "min3", a)], correct[("masked_hamming", "single", a)]
        out["mcnemar_min3_vs_single"][f"alpha_{a}"] = {
            "p_value": mcnemar_exact(m, s),
            "fakes_only_min3_rejects": int((m & ~s).sum()),
            "fakes_only_single_rejects": int((~m & s).sum()),
        }
    return out


# ------------------------------------------------------------------------------- main


def run_phone(pack: Path, device: str, cfg: dict, stage: str, log) -> dict:
    runs = cfg["captures"]
    orig = [indigo_counts(pack / f"mobile_original_{device}_run{r}.npz", stage=stage) for r in runs]
    fake = [
        indigo_counts(pack / f"mobile_fake_EHPI55_{device}_run{r}.npz", stage=stage) for r in runs
    ]
    for c, cls in [(c, "original") for c in orig] + [(c, "fake") for c in fake]:
        if c.meta["class"] != cls:
            raise ValueError(f"expected {cls}, found {c.meta['class']}")
    for c in fake:
        check_dev_fakes(c.ids, stage)
    ids_o, Do, Lo, miss_o = align(orig)
    ids_f, Df, Lf, miss_f = align(fake)
    blocks = indigo_blocks("mobile", stage=stage, available=ids_o, seed=cfg["seed"])
    E = np.isin(ids_o, blocks["E"])
    cb = codebook_of(Do[0][E], Lo[0][E])

    def fs_orig(block):
        m = np.isin(ids_o, block)
        return fused_scores(Do[:, m], Lo[:, m], cb, cfg["mu"]), int(m.sum())

    def fs_fake(block):
        m = np.isin(ids_f, block)
        return fused_scores(Df[:, m], Lf[:, m], cb, cfg["mu"]), ids_f[m]

    judged = blocks["V"] if stage == "dev" else np.concatenate([blocks["T_a"], blocks["T_b"]])
    lo, hi = (1, INDIGO_DEV_MAX_ID) if stage == "dev" else (INDIGO_DEV_MAX_ID + 1, 1440)
    fake_any = set().union(*[set(c.ids.tolist()) for c in fake])
    fake_any = {i for i in fake_any if lo <= i <= hi}
    info = {
        "captures": runs,
        "E_originals": int(E.sum()),
        "patterns_in_mask": int((cb <= cfg["mu"]).sum()),
        "share_of_E_symbols_in_mask": float(
            Lo[0][E].sum(axis=0)[cb <= cfg["mu"]].sum() / Lo[0][E].sum()
        ),
        "originals_missing_a_capture": miss_o.tolist(),
        "fakes_missing_a_capture": [int(i) for i in miss_f if lo <= i <= hi],
    }
    if stage == "dev":
        cal, info["n_S"] = fs_orig(blocks["S"])
        o, info["n_V_originals"] = fs_orig(blocks["V"])
        f, f_ids = fs_fake(blocks["V"])
        parts = [{"cal": cal, "orig": o, "fake": f}]
    else:
        oa, _ = fs_orig(blocks["T_a"])
        ob, _ = fs_orig(blocks["T_b"])
        fa, ids_a = fs_fake(blocks["T_a"])
        fb, ids_b = fs_fake(blocks["T_b"])
        parts = [{"cal": ob, "orig": oa, "fake": fa}, {"cal": oa, "orig": ob, "fake": fb}]
        f_ids = np.concatenate([ids_a, ids_b])
    info["n_fakes_judged"] = len(f_ids)
    info["fakes_not_judged"] = sorted(fake_any - set(f_ids.tolist()))
    if not set(f_ids.tolist()) <= set(np.asarray(judged).tolist()):
        raise RuntimeError("a fake outside the judged blocks")
    res = evaluate(parts, cfg["alphas"])
    for a in cfg["alphas"]:
        line = ", ".join(
            f"{r} {res['masked_hamming'][r][f'alpha_{a}']['p_fa']:.3f} "
            f"(P_miss {res['masked_hamming'][r][f'alpha_{a}']['p_miss']:.3f})"
            for r in RULES
        )
        log(f"  {device}, masked Hamming, P_fa at target P_miss {a}: {line}")
    return {"info": info, "results": res}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fusing three phone captures of the same print")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p2.yaml")
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args(argv)
    text = args.config.read_text()
    cfg = yaml.safe_load(text)
    if args.stage == "test" and not cfg.get("freeze_commit"):
        raise SystemExit("--stage test needs freeze_commit in the config (methods freeze)")
    start = time.time()

    def log(msg):
        print(f"[{time.time() - start:6.0f} s] {msg}", flush=True)

    pack = ROOT / cfg["data_dir"] / "indigo_compact"
    result = {"experiment": "p2_best_of_3", "stage": args.stage, "git": None, "config": cfg}
    result["indigo_mobile"] = {}
    for device in cfg["devices"]:
        log(f"Indigo mobile {device}, captures {cfg['captures']}")
        result["indigo_mobile"][device] = run_phone(pack, device, cfg, args.stage, log)
    d = cfg["decision"]
    if args.stage == "dev":
        r = result["indigo_mobile"][d["device"]]["results"]["masked_hamming"]
        p = r["min3"][f"alpha_{d['alpha']}"]
        result["decision"] = {
            "rule": (
                f"development, {d['device']}: P_fa of the minimum of 3 captures < "
                f"{d['max_p_fa']} at target P_miss {d['alpha']}"
            ),
            "p_fa_single": r["single"][f"alpha_{d['alpha']}"]["p_fa"],
            "p_fa_pooled": r["pooled"][f"alpha_{d['alpha']}"]["p_fa"],
            "p_fa_min3": p["p_fa"],
            "p_miss_min3": p["p_miss"],
            "keep": bool(p["p_fa"] < d["max_p_fa"]),
        }
    result["seconds"] = round(time.time() - start)
    result["git"] = git_state()
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    out = ROOT / "results" / f"p2_{h}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(_round(result, 5), indent=1))
    if "decision" in result:
        print(json.dumps(_round(result["decision"], 4), indent=1))
    print(f"written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
