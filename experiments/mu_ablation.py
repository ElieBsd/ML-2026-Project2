"""Sensitivity of masked_hamming to the reliability threshold mu, on development data only.

The masked Hamming distance counts errors only on symbols whose 3x3 pattern has a flip
probability <= mu in the codebook (Tutt 2025, p. 104). The thesis uses mu = 0.01, found by
looking at the error (p. 107, Fig. 5.8); we keep mu = 0.01 for every reported number and
only show here how much the results move for mu in {0.003, 0.01, 0.03}. Nothing is chosen
from this file.

Same data, blocks, codebooks and conformal thresholds as the detection experiment
(experiments/c3a_detection.py, imported unchanged, development stage): Indigo templates
1-288, Scantrust batch_cdp group 0, development fakes of unique_cdp. No test fake is read.
The row mu = 0.01 is the masked_hamming row of that experiment.

Usage:
    python -m experiments.mu_ablation --config configs/mu_ablation.yaml
Writes results/mu_ablation_<config hash>.json.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import yaml

from experiments import c3a_detection as c3a
from experiments.e0_sanity import _round, git_state

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCORE = "masked_hamming"
REFERENCE = "hamming"  # does not depend on mu: shown once, for scale
DROP = ("class_profile", "fake_correct", "p_fa_per_template", "p_miss_per_template")


def keep_score(tree, score: str):
    """The tree of a detection result with only the entries of one score.

    A dict that has a score name among its keys keeps that score only; class profiles and
    per-image or per-template lists are dropped. Other keys and values are kept.
    """
    if not isinstance(tree, dict):
        return tree
    if score in tree and any(k in tree for k in c3a.SCORES if k != score):
        return keep_score(tree[score], score)
    return {k: keep_score(v, score) for k, v in tree.items() if k not in DROP}


def main_rows(result: dict, cfg: dict) -> dict:
    """The detection result without the extra phone captures (their range is not ablated)."""
    mob = cfg["indigo"]["mobile"]
    extra = [f"mobile_{d}_capture{r}" for d in mob["devices"] for r in mob["range_capture_runs"]]
    indigo = {k: v for k, v in result["indigo"].items() if k not in extra}
    return {
        "indigo": indigo,
        "scantrust_unique": result["scantrust_unique"],
        "scantrust_batch": result["scantrust_batch"],
    }


def summary(rows: dict, mus, alpha: float) -> dict:
    """P_fa at the target P_miss alpha, one list (one value per mu) per condition."""
    key = f"alpha_{alpha}"
    out = {}
    first = rows[mus[0]]
    for name, cond in first["indigo"].items():
        for fake in cond["fakes"]:
            out[f"{name} {fake}"] = [
                rows[m]["indigo"][name]["fakes"][fake][key]["p_fa"] for m in mus
            ]
    for attack in first["scantrust_unique"]["attacks"]:
        out[f"unique {attack}"] = [
            rows[m]["scantrust_unique"]["attacks"][attack][key]["p_fa_mean"] for m in mus
        ]
    for attack in first["scantrust_batch"]["attacks"]:
        out[f"batch {attack}"] = [
            rows[m]["scantrust_batch"]["attacks"][attack][key]["conformal"]["p_fa"] for m in mus
        ]
    return out


def run(data_dir: Path, base: dict, mus, log) -> dict:
    rows, reference = {}, None
    for mu in mus:
        log(f"mu = {mu}")
        result = main_rows(c3a.run(data_dir, {**base, "mu": mu}, "dev", log), base)
        rows[mu] = keep_score(result, SCORE)
        if reference is None:
            reference = keep_score(result, REFERENCE)
    return {"rows": rows, "reference": reference}


def main(argv: list[str] | None = None, run_fn=run) -> int:
    parser = argparse.ArgumentParser(description="masked_hamming for several mu, development")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "mu_ablation.yaml")
    args = parser.parse_args(argv)
    text = args.config.read_text()
    cfg = yaml.safe_load(text)
    mus = [float(m) for m in cfg["mus"]]
    if cfg["reported_mu"] not in mus:
        raise SystemExit("the reported mu must be one of the ablated values")
    base_text = (ROOT / cfg["base_config"]).read_text()
    base = yaml.safe_load(base_text)
    start = time.time()

    def log(msg):
        print(f"[{time.time() - start:6.0f} s] {msg}", flush=True)

    out = run_fn(ROOT / base["data_dir"], base, mus, log)
    alpha = cfg["summary_alpha"]
    result = {
        "experiment": "mu_ablation",
        "stage": "dev",
        "data": "development fakes only (Indigo 1-288, batch_cdp group 0, unique_cdp V_U)",
        "git": git_state(),
        "config": cfg,
        "base_config": base,
        "score": SCORE,
        "mus": mus,
        "reported_mu": cfg["reported_mu"],
        "rows": {str(m): r for m, r in out["rows"].items()},
        "reference_" + REFERENCE: out["reference"],
        f"summary_p_fa_alpha_{alpha}": summary(out["rows"], mus, alpha),
        "seconds": round(time.time() - start),
    }
    h = hashlib.sha1((text + base_text).encode()).hexdigest()[:8]
    path = RESULTS / f"mu_ablation_{h}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(_round(result, 5), indent=1))
    print(f"P_fa at target P_miss {alpha}, mu = {mus}:")
    for name, values in result[f"summary_p_fa_alpha_{alpha}"].items():
        print(f"  {name:40s} " + "  ".join(f"{v:.3f}" for v in values))
    print(f"written to results/{path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
