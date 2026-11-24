"""Tests of the announced paired tests and predictions, on hand-made result files."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from experiments import announced_tests as at

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "announced_tests.yaml").read_text())


def _phone(correct_masked, correct_hamming):
    def row(c):
        return {"alpha_0.05": {"p_fa": 1 - float(np.mean(c)), "fake_correct": list(c)}}

    return {
        "fakes": {
            "EHPI55": {"masked_hamming": row(correct_masked), "hamming": row(correct_hamming)}
        }
    }


def _batch(fa):
    """fa: {score: per-template P_fa}."""
    return {
        s: {
            "alpha_0.05": {
                kind: {"p_fa": float(np.mean(v)), "p_fa_per_template": list(v)}
                for kind in ("conformal", "equal_p_miss_test_originals")
            }
        }
        for s, v in fa.items()
    }


def _results(vote_fa=(0.1, 0.0, 0.2, 0.1, 0.0, 0.1), min3=(0.03, 0.02), mask=None):
    ham = [0.6, 0.5, 0.7, 0.4, 0.5, 0.6]
    det = {
        "stage": "test",
        "indigo": {
            "mobile_iPhone12Pro": _phone([1, 1, 1, 1, 0, 1], [0, 0, 1, 0, 0, 1]),
            "mobile_SamsungGN20U": _phone([1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1]),
        },
        "scantrust_batch": {
            "attacks": {
                a: _batch(
                    {
                        "hamming": ham,
                        "masked_hamming": [0.7, 0.5, 0.8, 0.6, 0.5, 0.9],
                        "binned_mahalanobis": [0.3, 0.2, 0.4, 0.1, 0.2, 0.3],
                    }
                )
                for a in ("otsu+unsharp", "deep+unsharp")
            }
        },
    }
    vote = {
        "stage": "test",
        "scantrust_batch": {
            "attacks": {"otsu+unsharp": _batch({"hamming": ham, "lrt_vote": list(vote_fa)})}
        },
    }
    row = {"alpha_0.01": {"p_fa": min3[0], "p_miss": min3[1], "n_fakes": 100}}
    best = {
        "stage": "test",
        "indigo_mobile": {
            "SamsungGN20U": {
                "results": {
                    "masked_hamming": {
                        "min3": row,
                        "single": {"alpha_0.01": {"p_fa": 0.9, "p_miss": 0.01, "n_fakes": 100}},
                    },
                    "mcnemar_min3_vs_single": {
                        "alpha_0.01": {
                            "p_value": 0.0,
                            "fakes_only_min3_rejects": 87,
                            "fakes_only_single_rejects": 0,
                        }
                    },
                }
            }
        },
    }
    mask = mask or {
        "SamsungGN20U": (0.5, 0.7),
        "iPhone12Pro": (0.0, 0.0),
        "otsu+unsharp": (0.9, 0.6),
        "deep+unsharp": (0.6, 0.6),
    }

    def mk(p5, p3, n=10):
        c5 = [1] * round(n * (1 - p5)) + [0] * round(n * p5)
        c3 = [1] * round(n * (1 - p3)) + [0] * round(n * p3)
        return {
            "mask_5x5": {"alpha_0.05": {"p_fa": p5, "fake_correct": c5}},
            "mask_3x3": {"alpha_0.05": {"p_fa": p3, "fake_correct": c3}},
        }

    ctx = {
        "stage": "test",
        "indigo": {
            f"mobile_{d}": {"fakes": {"EHPI55": mk(*mask[d])}}
            for d in ("SamsungGN20U", "iPhone12Pro")
        },
        "scantrust_batch": {a: mk(*mask[a]) for a in ("otsu+unsharp", "deep+unsharp")},
    }
    return {"detection": det, "vote": vote, "best_of_3": best, "context_mask": ctx}


def test_comparisons_use_the_announced_tests_and_numbers():
    res = _results()
    c = at.comparisons(res, CFG)
    assert len(c) == 8  # six comparisons; the first on two phones, the third on two attacks
    i = c["1_masked_hamming_vs_hamming_iPhone12Pro"]
    assert (i["only_first_rejects"], i["only_second_rejects"]) == (3, 0)
    assert i["p_value"] == pytest.approx(stats.binomtest(3, 3, 0.5).pvalue)
    assert c["1_masked_hamming_vs_hamming_SamsungGN20U"]["p_value"] == 1.0  # same answers
    ham = res["detection"]["scantrust_batch"]["attacks"]["otsu+unsharp"]["hamming"]["alpha_0.05"][
        "conformal"
    ]["p_fa_per_template"]
    w = c["2_masked_hamming_vs_hamming_batch_otsu+unsharp"]
    assert w["n_templates"] == 6 and w["test"] == "wilcoxon_paired"
    assert w["p_value"] == pytest.approx(
        stats.wilcoxon(np.array([0.7, 0.5, 0.8, 0.6, 0.5, 0.9]) - np.array(ham)).pvalue
    )
    assert "3_binned_mahalanobis_vs_hamming_batch_deep+unsharp" in c
    assert c["5_min3_vs_single_SamsungGN20U"]["p_value"] == pytest.approx(2 * 0.5**87)
    assert c["6_mask_5x5_vs_mask_3x3_SamsungGN20U"]["only_first_rejects"] == 2


def test_predictions_hold_and_fail_as_written():
    p = at.predictions(_results(), CFG["predictions"])
    assert p["vote"]["holds"] and p["best_of_3"]["holds"] and p["context_mask"]["holds"]
    # vote: above half of hamming -> fails even if the test is significant
    assert not at.predictions(_results(vote_fa=(0.4, 0.3, 0.5, 0.3, 0.4, 0.5)), CFG["predictions"])[
        "vote"
    ]["holds"]
    # best of 3: realized P_miss above the bound -> fails
    assert not at.predictions(_results(min3=(0.03, 0.05)), CFG["predictions"])["best_of_3"]["holds"]
    # context mask: better on batch_cdp too -> fails (the prediction says it does not help there)
    m = {
        "SamsungGN20U": (0.5, 0.7),
        "iPhone12Pro": (0.0, 0.0),
        "otsu+unsharp": (0.5, 0.6),
        "deep+unsharp": (0.6, 0.6),
    }
    assert not at.predictions(_results(mask=m), CFG["predictions"])["context_mask"]["holds"]
    m["otsu+unsharp"], m["iPhone12Pro"] = (0.6, 0.6), (0.1, 0.0)  # worse on the other phone
    assert not at.predictions(_results(mask=m), CFG["predictions"])["context_mask"]["holds"]


def test_only_single_test_stage_files_are_read(tmp_path):
    with pytest.raises(SystemExit, match="found 0"):
        at.load_test_result(tmp_path, "c3a")
    (tmp_path / "c3a_aaaa_test.json").write_text(json.dumps({"stage": "dev"}))
    with pytest.raises(SystemExit, match="not a test-stage"):
        at.load_test_result(tmp_path, "c3a")
    (tmp_path / "c3a_bbbb_test.json").write_text(json.dumps({"stage": "test"}))
    with pytest.raises(SystemExit, match="found 2"):
        at.load_test_result(tmp_path, "c3a")
    (tmp_path / "c3a_dev.json").write_text("{}")  # a development file is never picked up
    (tmp_path / "c3a_aaaa_test.json").unlink()
    assert at.load_test_result(tmp_path, "c3a")["_file"] == "c3a_bbbb_test.json"


def test_main_writes_the_summary(tmp_path):
    for name, prefix in at.SOURCES.items():
        (tmp_path / f"{prefix}_x_test.json").write_text(json.dumps(_results()[name]))
    assert at.main(["--results-dir", str(tmp_path)]) == 0
    out = json.loads(next(tmp_path.glob("announced_tests_*.json")).read_text())
    assert set(out["predictions"]) == {"vote", "best_of_3", "context_mask"}
    assert out["sources"]["detection"]["file"] == "c3a_x_test.json"
