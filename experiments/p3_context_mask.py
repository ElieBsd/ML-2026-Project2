"""The 5x5 flip model in the mask of the masked Hamming distance.

masked_hamming counts flips only on the symbols that originals almost never get wrong
(flip probability q <= mu, mu = 0.01). Here q comes from the 5x5 back-off table
(cdpbpc.context) instead of the 3x3 table. Both tables are learned on the same
originals of the tested condition, whose templates are never judged (blocks E and M);
the back-off masses are chosen by log-loss on originals (two halves of E + M). Both
scores use the same positions, at distance >= 2 from the border. The thesis score
(3x3 codebook from E only, all interior symbols) is reported alongside.

Thresholds: conformal, set on originals only (block S, or the other half of the test
range for the cross-conformal threshold at the test stage). Measures: P_fa at the
target P_miss with the realized P_miss, AUC, and P_err^min (oracle, to compare with the
thesis only). Conditions: Indigo phones (capture 1), Indigo scanner (both printers,
four kinds of fakes), Scantrust batch_cdp (the two unsharp attacks).

Decision rule, set before the run (development only): on Samsung, the 5x5 mask cuts
the P_err^min of the 3x3 mask by at least 30 %.

--stage dev reads only the development fakes (Indigo templates 1-288, batch group 0).
--stage test runs once after the methods freeze and needs freeze_commit in the config.

Usage:
    python -m experiments.p3_context_mask --config configs/p3.yaml --stage dev
Writes results/p3_<config hash>.json (the stage is written inside).
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from cdpbpc import context as cx
from cdpbpc.calibrate import conformal_threshold
from cdpbpc.codebook import fit_codebook
from cdpbpc.counts import image_counts
from cdpbpc.io import (
    INDIGO_N_TEMPLATES,
    INDIGO_SIDE,
    STAGES,
    load_compact,
    load_scantrust,
    read_manifest,
)
from cdpbpc.metrics import auc, bootstrap_mean_ci, clopper_pearson, p_err_min, rate_per_group
from cdpbpc.scores import masked_hamming
from cdpbpc.splits import batch_fold, indigo_blocks
from experiments.e0_sanity import _round, git_state

ROOT = Path(__file__).resolve().parents[1]
SCORES = ("mask_5x5", "mask_3x3", "thesis_mask_3x3_E")
CHUNK = 64


class Images:
    """Binarized images with their templates. ids: (N,) template_id; t, bits: (N, L, L)."""

    def __init__(self, ids, t, bits, prints=None):
        self.ids, self.t, self.bits = np.asarray(ids), t, bits
        self.prints = None if prints is None else np.asarray(prints)

    def take(self, mask):
        p = None if self.prints is None else self.prints[mask]
        return Images(self.ids[mask], self.t[mask], self.bits[mask], p)

    def of(self, template_ids):
        return self.take(np.isin(self.ids, template_ids))

    def __len__(self):
        return len(self.ids)


class Masks:
    """The 3x3 and 5x5 tables learned on training originals, and the thesis codebook."""

    def __init__(self, train: Images, e_block: Images, cfg: dict):
        m = cfg["margin"]
        self.margin, self.mu = m, cfg["mu"]
        K = (train.t != train.bits).astype(np.uint8)
        # Back-off masses by log-loss on originals: two halves of the training templates.
        tids = np.unique(train.ids)
        half = np.isin(train.ids, tids[0::2])
        folds = []
        for fit, judge in ((half, ~half), (~half, half)):
            c = cx.table_counts(train.t[fit], K[fit], 1, margin=m)
            folds.append((c, train.t[judge], K[judge], 1))
        self.masses, losses = cx.select_masses(folds, grid=cfg["backoff_masses"], margin=m)
        del folds
        self.counts = cx.table_counts(train.t, K, 1, margin=m)
        self.codebook_E = fit_codebook(e_block.t, e_block.bits)
        self.info = {
            "train_images": len(train),
            "train_templates": len(tids),
            "E_images": len(e_block),
            "backoff_masses": self.masses,
            "backoff_loss": losses,
        }

    def scores(self, imgs: Images) -> dict:
        """Each score of SCORES for each image, and the share of its symbols in each mask."""
        n = len(imgs)
        out = {k: np.empty(n) for k in SCORES}
        share = {k: np.empty(n) for k in ("mask_5x5", "mask_3x3")}
        for s in range(0, n, CHUNK):
            t, b = imgs.t[s : s + CHUNK], imgs.bits[s : s + CHUNK]
            idx = cx.context_index(t, cx.TOP_BITS, self.margin)
            d = cx.crop(t != b, self.margin)
            for name, bits in (("mask_5x5", 25), ("mask_3x3", 9)):
                mask = cx.backoff_predict(self.counts, self.masses, idx, bits) <= self.mu
                sym = mask.sum((1, 2))
                if (sym == 0).any():
                    raise ValueError(f"{name}: an image has no reliable symbol")
                out[name][s : s + len(t)] = (d & mask).sum((1, 2)) / sym
                share[name][s : s + len(t)] = sym / mask[0].size
            D, L = image_counts(t, b)
            out["thesis_mask_3x3_E"][s : s + len(t)] = masked_hamming(
                D, L, self.codebook_E, self.mu
            )
        return {"scores": out, "share": share}


# ----------------------------------------------------------------------------- measures


def rates(o, f, tau, groups_o=None, groups_f=None) -> dict:
    """P_miss and P_fa at threshold tau; per template (then averaged) if groups given."""
    miss, acc = o > tau, f <= tau
    if groups_o is None:
        return {
            "tau": tau,
            "p_miss": float(miss.mean()),
            "p_fa": float(acc.mean()),
            "p_fa_ci": list(clopper_pearson(int(acc.sum()), len(acc))),
            "n_originals": len(o),
            "n_fakes": len(f),
        }
    _, miss_t = rate_per_group(miss, groups_o)
    _, fa_t = rate_per_group(acc, groups_f)
    return {
        "tau": tau,
        "p_miss": float(miss_t.mean()),
        "p_fa": float(fa_t.mean()),
        "p_miss_per_template": miss_t,
        "p_fa_per_template": fa_t,
        "n_templates": len(fa_t),
    }


def summary(cal: dict, orig: dict, fake: dict, alphas) -> dict:
    """Measures of each score. cal, orig, fake: {score: (N,)}; thresholds come from cal."""
    out = {}
    for k in SCORES:
        o, f = orig[k], fake[k]
        res = {"auc": auc(o, f), "p_err_min_oracle": p_err_min(o, f)}
        for a in alphas:
            res[f"alpha_{a}"] = rates(o, f, conformal_threshold(cal[k], a))
        out[k] = res
    out["p_err_min_cut_5x5_vs_3x3"] = 1 - out["mask_5x5"]["p_err_min_oracle"] / max(
        out["mask_3x3"]["p_err_min_oracle"], 1e-12
    )
    return out


def cross_summary(masks, halves, fakes_by_half, alphas) -> dict:
    """Test stage, Indigo: the threshold for each half comes from the originals of the other."""
    so = [masks.scores(h)["scores"] for h in halves]
    sf = [masks.scores(f)["scores"] for f in fakes_by_half]
    out = {}
    for k in SCORES:
        o_all = np.concatenate([so[0][k], so[1][k]])
        f_all = np.concatenate([sf[0][k], sf[1][k]])
        res = {"auc": auc(o_all, f_all), "p_err_min_oracle": p_err_min(o_all, f_all)}
        for a in alphas:
            miss, acc = [], []
            for j in (0, 1):
                tau = conformal_threshold(so[1 - j][k], a)
                miss.append(so[j][k] > tau)
                acc.append(sf[j][k] <= tau)
            miss, acc = np.concatenate(miss), np.concatenate(acc)
            res[f"alpha_{a}"] = {
                "p_miss": float(miss.mean()),
                "p_fa": float(acc.mean()),
                "p_fa_ci": list(clopper_pearson(int(acc.sum()), len(acc))),
                "n_originals": len(miss),
                "n_fakes": len(acc),
                "fake_correct": (~acc).astype(int).tolist(),  # for McNemar against the 3x3 mask
            }
        out[k] = res
    out["p_err_min_cut_5x5_vs_3x3"] = 1 - out["mask_5x5"]["p_err_min_oracle"] / max(
        out["mask_3x3"]["p_err_min_oracle"], 1e-12
    )
    return out


def mask_facts(masks: Masks, orig: Images) -> dict:
    """Share of judged symbols in each mask and flip rate of originals there."""
    out = {}
    for name, bits in (("mask_5x5", 25), ("mask_3x3", 9)):
        flips = sym = 0
        for s in range(0, len(orig), CHUNK):
            t, b = orig.t[s : s + CHUNK], orig.bits[s : s + CHUNK]
            idx = cx.context_index(t, cx.TOP_BITS, masks.margin)
            mask = cx.backoff_predict(masks.counts, masks.masses, idx, bits) <= masks.mu
            flips += int((cx.crop(t != b, masks.margin) & mask).sum())
            sym += int(mask.sum())
        h = orig.t.shape[-1] - 2 * masks.margin
        out[name] = {"share": sym / (len(orig) * h * h), "flip_rate_originals": flips / sym}
    return out


# ------------------------------------------------------------------------------- Indigo


def indigo_images(pack: Path, name: str, T: np.ndarray, stage: str, cls: str) -> Images:
    f = load_compact(pack / f"{name}.npz", stage=stage)
    if f.meta["class"] != cls:
        raise ValueError(f"{name}: expected {cls}, found {f.meta['class']}")
    return Images(f.ids, T[f.ids], f.bits)


def run_indigo_condition(setting, orig, fakes: dict, alphas, cfg, stage, log) -> dict:
    blocks = indigo_blocks(setting, stage=stage, available=orig.ids, seed=cfg["seed"])
    E, M = orig.of(blocks["E"]), orig.of(blocks["M"])
    train = Images(
        np.concatenate([E.ids, M.ids]),
        np.concatenate([E.t, M.t]),
        np.concatenate([E.bits, M.bits]),
    )
    masks = Masks(train, E, cfg)
    out = {"masks": masks.info, "fakes": {}}
    if stage == "dev":
        S, V = orig.of(blocks["S"]), orig.of(blocks["V"])
        s_cal = masks.scores(S)["scores"]
        sc = masks.scores(V)
        out["n_S"], out["n_V_originals"] = len(S), len(V)
        out["mask_facts_V_originals"] = mask_facts(masks, V)
        out["share_in_mask_V"] = {k: float(v.mean()) for k, v in sc["share"].items()}
        for fname, fk in fakes.items():
            fv = fk.of(blocks["V"])
            out["fakes"][fname] = summary(s_cal, sc["scores"], masks.scores(fv)["scores"], alphas)
            out["fakes"][fname]["n_fakes"] = len(fv)
    else:
        halves = [orig.of(blocks["T_a"]), orig.of(blocks["T_b"])]
        out["mask_facts_T_originals"] = mask_facts(masks, halves[0])
        for fname, fk in fakes.items():
            fh = [fk.of(blocks["T_a"]), fk.of(blocks["T_b"])]
            out["fakes"][fname] = cross_summary(masks, halves, fh, alphas)
    for fname, r in out["fakes"].items():
        log(
            f"  {fname}: P_err^min 3x3 {r['mask_3x3']['p_err_min_oracle']:.4f} -> "
            f"5x5 {r['mask_5x5']['p_err_min_oracle']:.4f} "
            f"(thesis {r['thesis_mask_3x3_E']['p_err_min_oracle']:.4f})"
        )
    return out


def run_indigo(data_dir: Path, cfg: dict, stage: str, log) -> dict:
    pack = data_dir / "indigo_compact"
    tpl = load_compact(pack / "templates_original.npz", stage=stage)
    T = np.zeros((INDIGO_N_TEMPLATES + 1, INDIGO_SIDE, INDIGO_SIDE), np.uint8)
    T[tpl.ids] = tpl.bits
    out = {}
    mob = cfg["indigo"]["mobile"]
    for dev in mob["devices"]:
        run = mob["capture_run"]
        log(f"Indigo mobile {dev}, capture {run}")
        orig = indigo_images(pack, f"mobile_original_{dev}_run{run}", T, stage, "original")
        fake = indigo_images(pack, f"mobile_fake_EHPI55_{dev}_run{run}", T, stage, "fake")
        out[f"mobile_{dev}"] = run_indigo_condition(
            "mobile", orig, {"EHPI55": fake}, mob["alphas"], cfg, stage, log
        )
    sc = cfg["indigo"]["scanner"]
    fakes = {
        f"f{p[3:]}/{e[4:]}": indigo_images(pack, f"base_fake_{p}_{e}", T, stage, "fake")
        for p, e in (f.split("_") for f in sc["fakes"])
    }
    for printer in sc["printers"]:
        log(f"Indigo scanner, originals {printer}")
        orig = indigo_images(pack, f"base_original_{printer}", T, stage, "original")
        out[f"scanner_{printer}"] = run_indigo_condition(
            "scanner", orig, fakes, sc["alphas"], cfg, stage, log
        )
    return out


# ---------------------------------------------------------------------------- Scantrust


def run_scantrust(data_dir: Path, cfg: dict, stage: str, rng, log) -> dict:
    sc = cfg["scantrust"]
    manifest = read_manifest(data_dir / "manifest.csv")
    batch = manifest[manifest["dataset"] == "batch"]
    rows = batch[(batch["class"] == "original") | batch["attack"].isin(sc["attacks"])]
    imgs = load_scantrust(data_dir, rows, stage=stage)  # drops the fakes not allowed at stage
    r = imgs.rows
    is_orig = (r["class"] == "original").to_numpy()
    all_imgs = Images(
        r["template_id"].to_numpy(int), imgs.t, imgs.bits, r["print_id"].to_numpy(int)
    )
    orig = all_imgs.take(is_orig)
    fakes = {a: all_imgs.take((r["attack"] == a).fillna(False).to_numpy()) for a in sc["attacks"]}
    groups = [0] if stage == "dev" else [1, 2, 3, 4]
    per_fold = {}
    for g in groups:
        fold = batch_fold(g)
        E = orig.take(np.isin(orig.prints, fold["e_print_ids"]))
        M = orig.of(fold["blocks"]["M"])
        train = Images(
            np.concatenate([E.ids, M.ids]),
            np.concatenate([E.t, M.t]),
            np.concatenate([E.bits, M.bits]),
        )
        masks = Masks(train, E, cfg)
        S, Tt = orig.of(fold["blocks"]["S"]), orig.of(fold["blocks"]["T"])
        s_cal = masks.scores(S)["scores"]
        so = masks.scores(Tt)["scores"]
        per_fold[g] = {"masks": masks.info, "S": s_cal, "T_orig": so, "T_ids": Tt.ids, "fakes": {}}
        for a, fk in fakes.items():
            ft = fk.of(fold["blocks"]["T"])
            per_fold[g]["fakes"][a] = (masks.scores(ft)["scores"], ft.ids)
        if g == groups[0]:
            per_fold[g]["mask_facts_T_originals"] = mask_facts(masks, Tt)
    out = {"folds": {g: {"masks": v["masks"]} for g, v in per_fold.items()}}
    out["mask_facts_T_originals"] = per_fold[groups[0]]["mask_facts_T_originals"]
    for a in sc["attacks"]:
        res = {}
        for k in SCORES:
            o_all = np.concatenate([per_fold[g]["T_orig"][k] for g in groups])
            f_all = np.concatenate([per_fold[g]["fakes"][a][0][k] for g in groups])
            rk = {
                "auc_pooled": auc(o_all, f_all),
                "p_err_min_oracle_pooled": p_err_min(o_all, f_all),
            }
            for al in sc["alphas"]:
                fa_t, miss_t = [], []
                for g in groups:
                    tau = conformal_threshold(per_fold[g]["S"][k], al)
                    rr = rates(
                        per_fold[g]["T_orig"][k],
                        per_fold[g]["fakes"][a][0][k],
                        tau,
                        per_fold[g]["T_ids"],
                        per_fold[g]["fakes"][a][1],
                    )
                    fa_t.append(rr["p_fa_per_template"])
                    miss_t.append(rr["p_miss_per_template"])
                fa_t, miss_t = np.concatenate(fa_t), np.concatenate(miss_t)
                rk[f"alpha_{al}"] = {
                    "p_miss": float(miss_t.mean()),
                    "p_fa": float(fa_t.mean()),
                    "p_fa_ci_template_bootstrap": list(bootstrap_mean_ci(fa_t, rng, cfg["n_boot"])),
                    "p_fa_per_template": fa_t.tolist(),
                    "n_templates": len(fa_t),
                }
            res[k] = rk
        res["p_err_min_cut_5x5_vs_3x3"] = 1 - res["mask_5x5"]["p_err_min_oracle_pooled"] / max(
            res["mask_3x3"]["p_err_min_oracle_pooled"], 1e-12
        )
        out[a] = res
        log(
            f"  batch {a}: P_err^min 3x3 {res['mask_3x3']['p_err_min_oracle_pooled']:.4f} -> "
            f"5x5 {res['mask_5x5']['p_err_min_oracle_pooled']:.4f}"
        )
    return out


# --------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The 5x5 flip model in the mask")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p3.yaml")
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args(argv)
    text = args.config.read_text()
    cfg = yaml.safe_load(text)
    if args.stage == "test" and not cfg.get("freeze_commit"):
        raise SystemExit("--stage test needs freeze_commit in the config (methods freeze)")
    data_dir = ROOT / cfg["data_dir"]
    start = time.time()

    def log(msg):
        print(f"[{time.time() - start:6.0f} s] {msg}", flush=True)

    rng = np.random.default_rng(cfg["seed"])
    result = {
        "experiment": "p3_context_mask",
        "stage": args.stage,
        "git": git_state(),
        "config": cfg,
    }
    result["indigo"] = run_indigo(data_dir, cfg, args.stage, log)
    log("Scantrust batch_cdp")
    result["scantrust_batch"] = run_scantrust(data_dir, cfg, args.stage, rng, log)
    d = cfg["decision"]
    if args.stage == "dev":
        r = result["indigo"][d["condition"]]["fakes"]["EHPI55"]
        result["decision"] = {
            "rule": "on development, Samsung: the 5x5 mask cuts P_err^min of the 3x3 mask by >= 30 %",
            "p_err_min_3x3": r["mask_3x3"]["p_err_min_oracle"],
            "p_err_min_5x5": r["mask_5x5"]["p_err_min_oracle"],
            "cut": r["p_err_min_cut_5x5_vs_3x3"],
            "continue_P3": r["p_err_min_cut_5x5_vs_3x3"] >= d["min_cut"],
        }
    result["seconds"] = round(time.time() - start)
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    out = ROOT / "results" / f"p3_{h}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(_round(result, 5), indent=1))
    if "decision" in result:
        print(json.dumps(_round(result["decision"], 4), indent=1))
    print(f"written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
