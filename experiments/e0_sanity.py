"""Sanity checks: facts about the data, redone with the project code.

Scantrust: identity of the templates of the two datasets (N1), the otsu / deep duplicate
of batch_cdp (N2), flip rates, shape of the codebook (N8), session shift (N3), sheet
effect (N4), design effect of the fakes (N7).
Indigo: inventory of the compact pack, flip rate per file (F7), page structure (F1), link
between the phone originals and print run 1 (F2), error rate of the attacker's estimated
templates, tail of the masked Hamming distance of phone originals (F5).

Only stage "dev" data are read: all originals and templates, and the fakes of batch
group 0, of unique templates 3551..4000 and of Indigo templates 1..288.

Usage:
    python -m experiments.e0_sanity --config configs/e0.yaml
Writes results/e0_<config hash>.json.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cdpbpc.calibrate import conformal_threshold
from cdpbpc.codebook import estimate
from cdpbpc.counts import indigo_counts, scantrust_counts
from cdpbpc.diagnostics import (
    adjusted_eta2,
    flip_counts,
    icc_oneway,
    reproducibility,
    two_way_components,
)
from cdpbpc.io import (
    INDIGO_DEV_MAX_ID,
    INDIGO_N_TEMPLATES,
    decode_image,
    decode_template,
    load_compact,
    load_scantrust,
    read_manifest,
    to_gray,
)
from cdpbpc.metrics import design_effect, p_miss, rate_per_group
from cdpbpc.scores import centered_z, hamming, masked_hamming

ROOT = Path(__file__).resolve().parents[1]
STAGE = "dev"
ATTACKS_USED = ("otsu", "otsu+unsharp", "deep+unsharp")  # batch deep is a duplicate of batch otsu
INDIGO_PAGE = 144


# ----------------------------------------------------------------------------- helpers


def codebook_of(D: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Jeffreys-smoothed codebook from the counts of a set of images."""
    return estimate(L.sum(axis=0), D.sum(axis=0))


def git_state() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout
        return commit + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _round(x, digits=4):
    """JSON-friendly copy with floats rounded."""
    if isinstance(x, dict):
        return {str(k): _round(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_round(v, digits) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), digits)
    return x


# ----------------------------------------------------------------------------- Scantrust


def scantrust_facts(data_dir: Path, cfg: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    c = scantrust_counts(data_dir, stage=STAGE)
    rows = c.rows.reset_index(drop=True)
    h = hamming(c.D, c.L)
    out = {}

    # N1: batch template k is unique template 3500 + k; no duplicate among unique templates.
    def template(path):
        return decode_template(path.read_bytes())

    unique_t = {i: template(data_dir / f"unique_cdp/template/{i:05d}.png") for i in range(1, 5001)}
    same = sum(
        np.array_equal(template(data_dir / f"batch_cdp/template/{k}.png"), unique_t[3500 + k])
        for k in range(1, 51)
    )
    out["N1"] = {
        "batch_templates_equal_to_unique_3500_plus_k": same,
        "distinct_unique_templates": len({t.tobytes() for t in unique_t.values()}),
        "white_share_of_templates": float(np.mean([t.mean() for t in unique_t.values()])),
    }

    # N2: copy/deep and copy/otsu of batch hold the same scans (checked on group 0).
    otsu = rows[(rows["dataset"] == "batch") & rows["attack"].eq("otsu").fillna(False)]
    otsu = otsu.head(cfg["correlation_pairs"])
    deep_paths = otsu["path"].str.replace("/copy/otsu/", "/copy/deep/", regex=False)
    corr, identical = [], 0
    for a, b in zip(otsu["path"], deep_paths):
        ba, bb = (data_dir / a).read_bytes(), (data_dir / b).read_bytes()
        identical += hashlib.md5(ba).digest() == hashlib.md5(bb).digest()
        ya, yb = to_gray(decode_image(ba)).ravel(), to_gray(decode_image(bb)).ravel()
        corr.append(np.corrcoef(ya, yb)[0, 1])
    deep_rows = rows[rows["path"].isin(set(deep_paths))].set_index("path").loc[deep_paths]
    bits_o = load_scantrust(data_dir, otsu, stage=STAGE).bits
    bits_d = load_scantrust(data_dir, deep_rows.reset_index(), stage=STAGE).bits
    out["N2"] = {
        "pairs": len(otsu),
        "identical_files": identical,
        "grey_correlation_mean": float(np.mean(corr)),
        "grey_correlation_min": float(np.min(corr)),
        "binarized_agreement": float((bits_o == bits_d).mean()),
    }

    # Flip rates per dataset, class and attack (fakes: development subsets only).
    rates = {}
    keys = rows[["dataset", "class", "attack"]].fillna("-").astype(str).agg("/".join, axis=1)
    for key in sorted(keys.unique()):
        v = h[(keys == key).to_numpy()]
        rates[key] = {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    out["flip_rates"] = rates

    unique_orig = (rows["dataset"] == "unique") & (rows["class"] == "original")
    pool = np.flatnonzero(unique_orig & (rows["template_id"] <= 3500))
    held_out = np.flatnonzero(unique_orig & (rows["template_id"] > 3500))
    batch_orig = np.flatnonzero((rows["dataset"] == "batch") & (rows["class"] == "original"))

    # N3 and N8: codebook from unique originals, threshold on other unique originals,
    # applied to the batch originals (another print session).
    miss = {a: {"batch": [], "unique_held_out": []} for a in cfg["alphas"]}
    for draw in range(cfg["n_draws"]):
        perm = rng.permutation(pool)
        E, S = perm[: cfg["codebook_size"]], perm[cfg["codebook_size"] :]
        cb = codebook_of(c.D[E], c.L[E])
        if draw == 0:
            reliable = cb <= 0.01
            out["N8"] = {
                "codebook_quantiles_25_50_75_95": np.quantile(cb, [0.25, 0.5, 0.75, 0.95]),
                "patterns_at_most_0.01": int(reliable.sum()),
                "symbol_share_at_most_0.01": float(c.L[E][:, reliable].sum() / c.L[E].sum()),
                "patterns_above_0.5": int((cb > 0.5).sum()),
            }
        z_s = centered_z(c.D[S], c.L[S], cb)
        for a in cfg["alphas"]:
            tau = conformal_threshold(z_s, a)
            miss[a]["batch"].append(p_miss(centered_z(c.D[batch_orig], c.L[batch_orig], cb), tau))
            miss[a]["unique_held_out"].append(
                p_miss(centered_z(c.D[held_out], c.L[held_out], cb), tau)
            )
    cb_u = codebook_of(c.D[unique_orig.to_numpy()], c.L[unique_orig.to_numpy()])
    cb_b = codebook_of(c.D[batch_orig], c.L[batch_orig])
    n_u = c.L[unique_orig.to_numpy()].sum(0)
    n_b = c.L[batch_orig].sum(0)
    se = np.sqrt(cb_u * (1 - cb_u) / n_u + cb_b * (1 - cb_b) / n_b)
    out["N3"] = {
        "flip_rate_unique_originals": float(h[unique_orig.to_numpy()].mean()),
        "flip_rate_batch_originals": float(h[batch_orig].mean()),
        "p_miss_by_alpha": {
            str(a): {
                k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                for k, v in m.items()
            }
            for a, m in miss.items()
        },
        "codebook_correlation_unique_batch": float(np.corrcoef(cb_u, cb_b)[0, 1]),
        "patterns_differing_by_3_se": float(np.mean(np.abs(cb_u - cb_b) > 3 * se)),
    }

    # N4: sheet effect on batch originals, codebook cross-fitted by template folds.
    b = rows.iloc[batch_orig]
    templates = np.sort(b["template_id"].unique())
    fold = dict(zip(templates, np.arange(len(templates)) % cfg["sheet_folds"]))
    folds = b["template_id"].map(fold).to_numpy()
    z = np.empty(len(b))
    for f in range(cfg["sheet_folds"]):
        train, test = batch_orig[folds != f], batch_orig[folds == f]
        z[folds == f] = centered_z(c.D[test], c.L[test], codebook_of(c.D[train], c.L[train]))
    sheet_means = pd.Series(z).groupby(b["sheet"].to_numpy()).mean()
    per_sheet = int(b.groupby("sheet").size().iloc[0])
    out["N4"] = {
        "sheets": len(sheet_means),
        "mean_z_per_sheet_min": float(sheet_means.min()),
        "mean_z_per_sheet_max": float(sheet_means.max()),
        "sd_of_sheet_means": float(sheet_means.std(ddof=1)),
        "sd_expected_if_independent": float(z.std(ddof=1) / np.sqrt(per_sheet)),
        "icc_of_z_by_template": icc_oneway(z, b["template_id"].to_numpy()),
    }

    # N7: fakes of one template are alike (batch group 0 only before the freeze).
    tau = float(np.quantile(h[batch_orig], cfg["fake_quantile"]))
    n7 = {"threshold_quantile_of_batch_originals": cfg["fake_quantile"], "threshold": tau}
    for attack in ATTACKS_USED:
        is_attack = rows["attack"].eq(attack).fillna(False).to_numpy(bool)
        sel = np.flatnonzero((rows["dataset"] == "batch").to_numpy() & is_attack)
        tid = rows["template_id"].to_numpy()[sel]
        accepted = h[sel] <= tau
        _, p_fa_per_template = rate_per_group(accepted, tid)
        n7[attack] = {
            "templates": len(np.unique(tid)),
            "icc_of_flip_rate": icc_oneway(h[sel], tid),
            "p_fa_per_template_min": float(p_fa_per_template.min()),
            "p_fa_per_template_max": float(p_fa_per_template.max()),
            "design_effect": design_effect(accepted, tid),
        }
    out["N7_group0"] = n7
    return out


# ----------------------------------------------------------------------------- Indigo


def indigo_facts(data_dir: Path, cfg: dict, seed: int) -> dict:
    pack = data_dir / "indigo_compact"
    manifest = read_manifest(data_dir / "manifest_indigo.csv")
    templates = load_compact(pack / "templates_original.npz", stage=STAGE)
    T = np.zeros((INDIGO_N_TEMPLATES + 1,) + templates.bits.shape[1:], dtype=np.uint8)
    T[templates.ids] = templates.bits
    out = {}

    # Inventory: images per file and missing template ids.
    inventory = {}
    for name, g in manifest.groupby("file", sort=True):
        small = g["subset"].iloc[0] == "base" or g["session"].iloc[0] == 0
        expected = 720 if small else INDIGO_N_TEMPLATES
        missing = sorted(set(range(1, expected + 1)) - set(g["template_id"].tolist()))
        inventory[name] = {"images": len(g), "missing": len(missing)}
    out["inventory"] = inventory

    # Flip rate per file (fakes: templates 1..288 only).
    counts = {name: indigo_counts(pack / f"{name}.npz", stage=STAGE) for name in inventory}
    out["flip_rate_per_file"] = {
        name: {"n": len(c.ids), "flip_rate": float(c.D.sum() / c.L.sum())}
        for name, c in counts.items()
    }
    runs = [f"variability_HPI55_session1_print{r}_scan1" for r in range(1, 11)]
    run_rates = [out["flip_rate_per_file"][r]["flip_rate"] for r in runs]
    out["F7"] = {
        "HPI55_session1_10_print_runs_min": min(run_rates),
        "HPI55_session1_10_print_runs_max": max(run_rates),
        "HPI55_session0": out["flip_rate_per_file"]["variability_HPI55_session0_print1_scan1"][
            "flip_rate"
        ],
        "HPI55_session2": out["flip_rate_per_file"]["variability_HPI55_session2_print1_scan1"][
            "flip_rate"
        ],
    }

    # F1: the template effect of the flip rate follows the position on the page of 144.
    common = sorted(set.intersection(*[set(counts[r].ids.tolist()) for r in runs]))
    ids = np.array(common)
    H = np.stack(
        [hamming(counts[r].D, counts[r].L)[np.searchsorted(counts[r].ids, ids)] for r in runs],
        axis=1,
    )
    effect = H.mean(axis=1)
    page, pos = (ids - 1) // INDIGO_PAGE, (ids - 1) % INDIGO_PAGE
    out["F1"] = {
        "templates": len(ids),
        "eta2_position_in_page": adjusted_eta2(effect, pos),
        "eta2_page": adjusted_eta2(effect, page),
        "eta2_row_in_page": adjusted_eta2(effect, pos // 12),
        "eta2_column_in_page": adjusted_eta2(effect, pos % 12),
        "page_means_pts": [
            100 * float(effect[page == p].mean() - effect.mean()) for p in np.unique(page)
        ],
        **two_way_components(H),
    }

    # F2: the phone originals share more reproducible pattern with print run 1.
    phone = [f"mobile_original_iPhone12Pro_run{r}" for r in range(1, 7)]
    scans = [f"variability_HPI55_session1_print{r}_scan1" for r in (1, 2, 3)]
    files = {name: load_compact(pack / f"{name}.npz", stage=STAGE) for name in phone + scans}
    common = sorted(set.intersection(*[set(f.ids.tolist()) for f in files.values()]))
    ids = np.array(common)

    def aligned(f):
        return f.bits[np.searchsorted(f.ids, ids)]

    t = T[ids]
    K_phone = flip_counts(t, [aligned(files[n]) for n in phone])
    f2 = {"templates": len(ids)}
    for name in scans:
        r = reproducibility(t, K_phone, len(phone), flip_counts(t, [aligned(files[name])]), 1)
        f2[f"shared_with_print{name.split('print')[1][0]}"] = r["shared"]
    f2["rho_iPhone_6_captures"] = r["rho"]
    out["F2"] = f2
    del files

    # Error rate of the attacker's estimated templates (templates 1..288).
    ber = {}
    for est in sorted(pack.glob("templates_estimated_*.npz")):
        e = load_compact(est, stage=STAGE)
        ber[est.stem.replace("templates_estimated_", "")] = {
            "templates": len(e.ids),
            "max_id": int(e.ids.max()),
            "ber": float((e.bits != T[e.ids]).mean()),
        }
    out["estimated_template_ber"] = ber

    # F5: tail of the masked Hamming distance of phone originals, capture 1. The codebook
    # comes from 50 originals of templates above 288; the first draw is also reported alone,
    # the others show how much the mask depends on which 50 originals are drawn.
    rng = np.random.default_rng(seed)
    f5 = {}
    for device in ("iPhone12Pro", "SamsungGN20U"):
        cs = [counts[f"mobile_original_{device}_run{r}"] for r in range(1, 7)]
        common = np.array(sorted(set.intersection(*[set(c.ids.tolist()) for c in cs])))
        first = cs[0]
        rows = np.searchsorted(first.ids, common)
        D, L = first.D[rows], first.L[rows]
        pool = np.flatnonzero(common > INDIGO_DEV_MAX_ID)
        draws = []
        for _ in range(cfg["codebook_draws"]):
            enrol = rng.permutation(pool)[: cfg["codebook_size"]]
            cb = codebook_of(D[enrol], L[enrol])
            mh = masked_hamming(D, L, cb, cfg["mask_mu"])
            draws.append((np.quantile(mh, cfg["quantiles"]), int((cb <= cfg["mask_mu"]).sum())))
        q = np.array([d[0] for d in draws])
        f5[device] = {
            "originals": len(common),
            "codebook_images": cfg["codebook_size"],
            "quantiles_first_draw": dict(zip(map(str, cfg["quantiles"]), q[0])),
            "quantiles_min_over_draws": dict(zip(map(str, cfg["quantiles"]), q.min(axis=0))),
            "quantiles_max_over_draws": dict(zip(map(str, cfg["quantiles"]), q.max(axis=0))),
            "masked_patterns_min_max": [min(d[1] for d in draws), max(d[1] for d in draws)],
        }
    out["F5"] = f5
    return out


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sanity checks: facts about the data")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "e0.yaml")
    args = parser.parse_args(argv)
    text = args.config.read_text()
    cfg = yaml.safe_load(text)
    data_dir = ROOT / cfg["data_dir"]

    result = {
        "experiment": "e0_sanity",
        "stage": STAGE,
        "git": git_state(),
        "config": cfg,
        "scantrust": scantrust_facts(data_dir, cfg["scantrust"], cfg["seed"]),
        "indigo": indigo_facts(data_dir, cfg["indigo"], cfg["seed"]),
    }
    out = ROOT / "results" / f"e0_{hashlib.sha1(text.encode()).hexdigest()[:8]}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(_round(result), indent=1))
    print(json.dumps(_round(result, 3), indent=1))
    print(f"written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
