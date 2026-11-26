import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from experiments import c3a_detection as c3a
from experiments import mu_ablation as ab


def _measures(value):
    """One condition of the detection result: every score, with a per-image list."""
    row = {"p_fa": value, "p_miss": 0.05, "fake_correct": [1, 0]}
    return {k: {"auc": value, "alpha_0.05": dict(row)} for k in c3a.SCORES}


def _result(value):
    batch_row = {"p_fa": value, "p_miss": 0.05, "p_fa_per_template": [value]}
    return {
        "indigo": {
            "mobile_iPhone12Pro": {"n_E": 50, "fakes": {"EHPI55": _measures(value)}},
            "mobile_iPhone12Pro_capture2": {"n_E": 50, "fakes": {"EHPI55": _measures(value)}},
            "scanner_HPI55": {"n_E": 50, "fakes": {"f55/55": _measures(value)}},
        },
        "indigo_capture_range": {"iPhone12Pro": {}},
        "scantrust_unique": {
            "attacks": {"otsu": {k: {"alpha_0.05": {"p_fa_mean": value}} for k in c3a.SCORES}},
            "class_profile": {"originals": [0.1]},
        },
        "scantrust_batch": {
            "attacks": {
                "otsu+unsharp": {
                    k: {"alpha_0.05": {"conformal": dict(batch_row)}} for k in c3a.SCORES
                }
            }
        },
    }


BASE = {
    "data_dir": "dataset",
    "indigo": {"mobile": {"devices": ["iPhone12Pro"], "range_capture_runs": [2]}},
}


def test_keep_score_keeps_one_score_and_drops_lists():
    kept = ab.keep_score(_result(0.3), "masked_hamming")
    row = kept["indigo"]["mobile_iPhone12Pro"]["fakes"]["EHPI55"]
    assert row == {"auc": 0.3, "alpha_0.05": {"p_fa": 0.3, "p_miss": 0.05}}
    assert "class_profile" not in kept["scantrust_unique"]
    batch = kept["scantrust_batch"]["attacks"]["otsu+unsharp"]
    assert batch == {"alpha_0.05": {"conformal": {"p_fa": 0.3, "p_miss": 0.05}}}


def test_main_rows_drop_the_extra_captures():
    rows = ab.main_rows(_result(0.3), BASE)
    assert set(rows["indigo"]) == {"mobile_iPhone12Pro", "scanner_HPI55"}
    assert "indigo_capture_range" not in rows


def test_run_changes_only_mu_and_stays_on_development(monkeypatch):
    calls = []

    def fake_run(data_dir, cfg, stage, log):
        calls.append((cfg["mu"], stage, {k: v for k, v in cfg.items() if k != "mu"}))
        return _result(cfg["mu"])

    monkeypatch.setattr(c3a, "run", fake_run)
    out = ab.run(Path("nowhere"), {**BASE, "mu": 0.01}, [0.003, 0.01, 0.03], lambda m: None)
    assert [c[:2] for c in calls] == [(0.003, "dev"), (0.01, "dev"), (0.03, "dev")]
    assert all(c[2] == BASE for c in calls)
    assert ab.summary(out["rows"], [0.003, 0.01, 0.03], 0.05) == {
        "mobile_iPhone12Pro EHPI55": [0.003, 0.01, 0.03],
        "scanner_HPI55 f55/55": [0.003, 0.01, 0.03],
        "unique otsu": [0.003, 0.01, 0.03],
        "batch otsu+unsharp": [0.003, 0.01, 0.03],
    }


def _config(tmp_path, mus, reported=0.01):
    path = tmp_path / "mu.yaml"
    cfg = {"base_config": "configs/c3a.yaml", "mus": mus, "reported_mu": reported}
    path.write_text(yaml.safe_dump({**cfg, "summary_alpha": 0.05}))
    return path


def test_main_writes_the_rows_without_reading_data(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "RESULTS", tmp_path / "results")

    def no_data(data_dir, base, mus, log):
        rows = {m: ab.keep_score(ab.main_rows(_result(m), BASE), "masked_hamming") for m in mus}
        return {"rows": rows, "reference": ab.keep_score(_result(0.5), "hamming")}

    assert ab.main(["--config", str(_config(tmp_path, [0.003, 0.01]))], run_fn=no_data) == 0
    (path,) = (tmp_path / "results").glob("mu_ablation_*.json")
    result = json.loads(path.read_text())
    assert result["stage"] == "dev" and result["mus"] == [0.003, 0.01]
    assert result["summary_p_fa_alpha_0.05"]["batch otsu+unsharp"] == [0.003, 0.01]


def test_the_reported_mu_must_be_ablated(tmp_path):
    with pytest.raises(SystemExit, match="reported mu"):
        ab.main(["--config", str(_config(tmp_path, [0.003, 0.03]))], run_fn=None)
