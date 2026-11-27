import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from experiments import fig3

EDGES = (0.0, 0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0)


def _block(shift=0.0):
    rows = []
    for k, (lo, hi) in enumerate(itertools.pairwise(EDGES)):
        g = 0.01 * (k - 3) + shift
        rows.append(
            {
                "class": [lo, hi],
                "gap_pred": g,
                "gap_pred_ci": [g - 0.001, g + 0.002],
                "gap_obs": g + 0.003,
                "gap_obs_ci": [g + 0.001, g + 0.005],
            }
        )
    corr = {"corr": 0.7, "corr_ci": [0.68, 0.72]}
    return {"by_class": rows, "per_pattern": {"P_obs": corr, "P_b": corr}}


def _c3b(stage="test"):
    batch = {a: _block(0.02) for a in ("otsu", "otsu+unsharp", "deep+unsharp")}
    return {
        "experiment": "c3b_two_pass",
        "stage": stage,
        "git": "abc",
        "indigo": {"mobile": {"iPhone12Pro": _block()}},
        "scantrust": {"batch": batch},
    }


def test_class_labels_read_the_bounds():
    labels = fig3.class_labels(_block()["by_class"])
    assert labels == ["<.01", ".01-.05", ".05-.15", ".15-.3", ".3-.5", ".5-.7", ".7-.9", ">.9"]


def test_series_are_in_percentage_points_with_their_intervals():
    s = fig3.series(_block()["by_class"], "gap_obs")
    assert s["value"][0] == pytest.approx(100 * (-0.03 + 0.003))
    assert s["low"][0] == pytest.approx(100 * (-0.03 + 0.001))
    assert s["high"][0] == pytest.approx(100 * (-0.03 + 0.005))
    assert len(s["value"]) == 8


def test_first_unreliable_class_starts_at_one_half():
    assert fig3.first_unreliable(_block()["by_class"]) == 5


def test_the_two_averaging_attacks_share_one_prediction():
    c3b = _c3b()
    c3b["scantrust"]["batch"]["deep+unsharp"]["by_class"][2]["gap_pred"] = 0.5
    with pytest.raises(SystemExit, match="same prediction"):
        fig3.batch_prediction(c3b)


def test_main_draws_the_figure_from_the_test_result(tmp_path, monkeypatch):
    monkeypatch.setattr(fig3, "ROOT", tmp_path)
    path = tmp_path / "c3b.json"
    path.write_text(json.dumps(_c3b()))
    assert fig3.main(["--c3b", str(path)]) == 0
    assert (tmp_path / "report" / "figures" / "fig3.pdf").stat().st_size > 0


def test_a_development_result_is_refused(tmp_path):
    path = tmp_path / "c3b.json"
    path.write_text(json.dumps(_c3b(stage="dev")))
    with pytest.raises(SystemExit, match="not a test-stage result"):
        fig3.load(path)


def test_a_result_of_another_experiment_is_refused(tmp_path):
    path = tmp_path / "c3b.json"
    path.write_text(json.dumps({**_c3b(), "experiment": "c3a_detection"}))
    with pytest.raises(SystemExit, match="not a result of c3b_two_pass"):
        fig3.load(path)


def test_default_input_follows_the_frozen_config():
    path = fig3.result_path("c3b")
    assert path.parent == fig3.ROOT / "results"
    assert path.name.startswith("c3b_") and path.name.endswith("_test.json")
