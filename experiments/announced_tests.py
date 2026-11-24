"""Paired tests and predictions announced before the test stage, read from its result files.

Six comparisons were announced before any test fake was opened, all on the share of
fakes accepted (P_fa) at the target P_miss, with a conformal threshold:
1. masked_hamming against hamming, each phone (exact McNemar on the same fakes);
2. masked_hamming against hamming, batch_cdp, otsu+unsharp (paired Wilcoxon on the P_fa
   per template);
3. binned_mahalanobis against hamming, batch_cdp, both averaging attacks (Wilcoxon);
4. the vote likelihood-ratio test against hamming, batch_cdp, otsu+unsharp (Wilcoxon);
5. the minimum of three captures against one capture, Samsung, target P_miss 1 % (McNemar);
6. the 5x5 mask against the 3x3 mask on the same positions, Samsung (McNemar).
All tests are two-sided. On batch_cdp the unit is the template (the 50 fakes of a
template are not independent), on the phones it is the fake (one per template).

Three predictions on the add-on methods were written down with them; the config holds
their numbers. They are judged here, on the test-stage files only.

This script reads no image and no count: only the test-stage result files of
c3a_detection, p1_vote, p2_best_of_3 and p3_context_mask (exactly one each in results/).

Usage:
    python -m experiments.announced_tests --config configs/announced_tests.yaml
Writes results/announced_tests_<config hash>.json.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from cdpbpc.metrics import mcnemar_exact, wilcoxon_paired
from experiments.e0_sanity import git_state

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCES = {"detection": "c3a", "vote": "p1", "best_of_3": "p2", "context_mask": "p3"}


def load_test_result(results_dir: Path, prefix: str) -> dict:
    """The single test-stage result file of an experiment; refuse none, several or a dev file."""
    files = sorted(results_dir.glob(f"{prefix}_*_test.json"))
    if len(files) != 1:
        raise SystemExit(f"expected one {prefix}_*_test.json in {results_dir}, found {len(files)}")
    r = json.loads(files[0].read_text())
    if r.get("stage") != "test":
        raise SystemExit(f"{files[0].name} is not a test-stage result")
    return {**r, "_file": files[0].name}


def mcnemar(correct_a, correct_b) -> dict:
    """Exact two-sided McNemar test: a and b say, fake by fake, whether it was rejected."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    if a.shape != b.shape:
        raise ValueError("the two methods did not judge the same fakes")
    return {
        "test": "mcnemar_exact",
        "n_fakes": len(a),
        "only_first_rejects": int((a & ~b).sum()),
        "only_second_rejects": int((~a & b).sum()),
        "p_value": mcnemar_exact(a, b),
    }


def mcnemar_from_counts(only_a: int, only_b: int) -> float:
    if only_a + only_b == 0:
        return 1.0
    return float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue)


def wilcoxon(fa_a, fa_b) -> dict:
    """Two-sided paired Wilcoxon test on the P_fa per template of two scores."""
    a, b = np.asarray(fa_a, float), np.asarray(fa_b, float)
    if a.shape != b.shape:
        raise ValueError("the two scores did not judge the same templates")
    return {
        "test": "wilcoxon_paired",
        "n_templates": len(a),
        "mean_p_fa_first": float(a.mean()),
        "mean_p_fa_second": float(b.mean()),
        "p_value": wilcoxon_paired(a, b),
    }


def comparisons(res: dict, cfg: dict) -> dict:
    """The six announced comparisons. res: {source: test-stage result}."""
    key = f"alpha_{cfg['alpha']}"
    det, out = res["detection"], {}
    for dev in cfg["phones"]:
        m = det["indigo"][f"mobile_{dev}"]["fakes"]["EHPI55"]
        out[f"1_masked_hamming_vs_hamming_{dev}"] = {
            "p_fa_first": m["masked_hamming"][key]["p_fa"],
            "p_fa_second": m["hamming"][key]["p_fa"],
            **mcnemar(m["masked_hamming"][key]["fake_correct"], m["hamming"][key]["fake_correct"]),
        }
    batch = det["scantrust_batch"]["attacks"]

    def per_template(attack, score):
        return batch[attack][score][key]["conformal"]["p_fa_per_template"]

    a = cfg["averaging_attacks"][0]
    out[f"2_masked_hamming_vs_hamming_batch_{a}"] = wilcoxon(
        per_template(a, "masked_hamming"), per_template(a, "hamming")
    )
    for a in cfg["averaging_attacks"]:
        out[f"3_binned_mahalanobis_vs_hamming_batch_{a}"] = wilcoxon(
            per_template(a, "binned_mahalanobis"), per_template(a, "hamming")
        )
    a = cfg["averaging_attacks"][0]
    v = res["vote"]["scantrust_batch"]["attacks"][a]
    out[f"4_lrt_vote_vs_hamming_batch_{a}"] = wilcoxon(
        v["lrt_vote"][key]["conformal"]["p_fa_per_template"],
        v["hamming"][key]["conformal"]["p_fa_per_template"],
    )
    b = cfg["best_of_3"]
    r = res["best_of_3"]["indigo_mobile"][b["device"]]["results"]
    k3 = f"alpha_{b['alpha']}"
    mc = r["mcnemar_min3_vs_single"][k3]
    out[f"5_min3_vs_single_{b['device']}"] = {
        "test": "mcnemar_exact",
        "target_p_miss": b["alpha"],
        "p_fa_first": r["masked_hamming"]["min3"][k3]["p_fa"],
        "p_fa_second": r["masked_hamming"]["single"][k3]["p_fa"],
        "n_fakes": r["masked_hamming"]["min3"][k3]["n_fakes"],
        "only_first_rejects": mc["fakes_only_min3_rejects"],
        "only_second_rejects": mc["fakes_only_single_rejects"],
        "p_value": mcnemar_from_counts(
            mc["fakes_only_min3_rejects"], mc["fakes_only_single_rejects"]
        ),
    }
    dev = cfg["context_mask_phone"]
    m = res["context_mask"]["indigo"][f"mobile_{dev}"]["fakes"]["EHPI55"]
    out[f"6_mask_5x5_vs_mask_3x3_{dev}"] = {
        "p_fa_first": m["mask_5x5"][key]["p_fa"],
        "p_fa_second": m["mask_3x3"][key]["p_fa"],
        **mcnemar(m["mask_5x5"][key]["fake_correct"], m["mask_3x3"][key]["fake_correct"]),
    }
    return out


def predictions(res: dict, P: dict) -> dict:
    """The three predictions on the add-on methods, with the numbers of the config."""
    out = {}
    p = P["vote"]
    key = f"alpha_{p['alpha']}"
    v = res["vote"]["scantrust_batch"]["attacks"][p["attack"]]
    eq = {s: v[s][key]["equal_p_miss_test_originals"] for s in ("lrt_vote", "hamming")}
    w = wilcoxon(eq["lrt_vote"]["p_fa_per_template"], eq["hamming"]["p_fa_per_template"])
    half = eq["lrt_vote"]["p_fa"] < p["max_ratio"] * eq["hamming"]["p_fa"]
    out["vote"] = {
        "rule": f"batch {p['attack']}, P_fa at equal P_miss {p['alpha']} (threshold on the test "
        f"originals): lrt_vote below {p['max_ratio']} times hamming, and paired Wilcoxon on the "
        f"P_fa per template with p < {p['max_p_value']}",
        "p_fa_lrt_vote": eq["lrt_vote"]["p_fa"],
        "p_fa_hamming": eq["hamming"]["p_fa"],
        "wilcoxon": w,
        "holds": bool(half and w["p_value"] < p["max_p_value"]),
    }
    p = P["best_of_3"]
    r = res["best_of_3"]["indigo_mobile"][p["device"]]["results"]["masked_hamming"]["min3"]
    r = r[f"alpha_{p['alpha']}"]
    out["best_of_3"] = {
        "rule": f"{p['device']}, minimum of 3 captures, target P_miss {p['alpha']}: P_fa < "
        f"{p['max_p_fa']} with a realized P_miss <= {p['max_p_miss']}",
        "p_fa": r["p_fa"],
        "p_miss": r["p_miss"],
        "holds": bool(r["p_fa"] < p["max_p_fa"] and r["p_miss"] <= p["max_p_miss"]),
    }
    p = P["context_mask"]
    key = f"alpha_{p['alpha']}"
    cm = res["context_mask"]

    def phone(dev):
        m = cm["indigo"][f"mobile_{dev}"]["fakes"]["EHPI55"]
        return m["mask_5x5"][key]["p_fa"], m["mask_3x3"][key]["p_fa"]

    better, same = phone(p["lower_on"]), phone(p["not_higher_on"])
    batch = {
        a: (
            cm["scantrust_batch"][a]["mask_5x5"][key]["p_fa"],
            cm["scantrust_batch"][a]["mask_3x3"][key]["p_fa"],
        )
        for a in p["batch_attacks"]
    }
    out["context_mask"] = {
        "rule": f"P_fa at target P_miss {p['alpha']}, 5x5 mask against 3x3 mask on the same "
        f"positions: lower on {p['lower_on']}, not higher on {p['not_higher_on']}, and not "
        f"lower on batch_cdp ({', '.join(p['batch_attacks'])})",
        "p_fa_5x5_3x3": {p["lower_on"]: better, p["not_higher_on"]: same, **batch},
        "holds": bool(
            better[0] < better[1]
            and same[0] <= same[1]
            and all(x5 >= x3 for x5, x3 in batch.values())
        ),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Announced paired tests and predictions")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "announced_tests.yaml")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    text = args.config.read_text()
    cfg = yaml.safe_load(text)
    res = {name: load_test_result(args.results_dir, prefix) for name, prefix in SOURCES.items()}
    result = {
        "experiment": "announced_tests",
        "stage": "test",
        "git": git_state(),
        "config": cfg,
        "sources": {name: {"file": r["_file"], "git": r.get("git")} for name, r in res.items()},
        "comparisons": comparisons(res, cfg),
        "predictions": predictions(res, cfg["predictions"]),
    }
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    out = args.results_dir / f"announced_tests_{h}.json"
    out.write_text(json.dumps(result, indent=1))  # p-values are kept unrounded
    print(json.dumps({k: v["p_value"] for k, v in result["comparisons"].items()}, indent=1))
    print(json.dumps({k: v["holds"] for k, v in result["predictions"].items()}, indent=1))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
