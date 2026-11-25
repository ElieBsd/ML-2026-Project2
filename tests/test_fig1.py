import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from experiments import fig1


def _noise():
    part = {"value": 0.3}
    decomposition = {"template": {"value": 0.2}, "print": part, "capture": {"value": 0.5}}
    rho = {"rho_ci": [0.18, 0.22]}
    return {
        "experiment": "c1a_noise",
        "data": "originals and templates only",
        "git": "abc",
        "indigo": {
            "decomposition": {
                "HPI55": {**decomposition, "prints": "p55", "reads": "r55"},
                "HPI76_scans_session0": {**decomposition, "prints": "p76", "reads": "r76"},
            },
            "within": {name: rho for name in ("p55", "r55", "p76", "r76")},
        },
        "scantrust": {"batch": {"rho": {"value": 0.23, "ci": [0.21, 0.25]}}},
    }


def _models():
    model = {"logloss": 0.3, "logloss_ci": [0.29, 0.31]}
    tests = {name: model for name, _ in fig1.MODELS}
    return {
        "experiment": "c1b_context",
        "data": "originals and templates only",
        "git": "abc",
        "indigo": {"tests": {"test": tests}},
        "scantrust": {"tests": {"test_unique": tests, "test_batch": tests}},
    }


def test_noise_rows_add_up_to_the_whole_noise():
    rows = fig1.noise_rows(_noise())
    assert [r[0] for r in rows] == ["Indigo HPI76*", "Indigo HPI55", "Scantrust"]
    for _, parts, _ in rows:
        assert sum(v for _, v in parts) == pytest.approx(1.0)
    assert rows[-1][1] == [("template", 0.23), ("print_and_capture", pytest.approx(0.77))]


def test_main_draws_the_figure_from_the_two_results(tmp_path, monkeypatch):
    monkeypatch.setattr(fig1, "ROOT", tmp_path)
    noise, models = tmp_path / "noise.json", tmp_path / "models.json"
    noise.write_text(json.dumps(_noise()))
    models.write_text(json.dumps(_models()))
    assert fig1.main(["--noise", str(noise), "--models", str(models)]) == 0
    assert (tmp_path / "report" / "figures" / "fig1.pdf").stat().st_size > 0


def test_a_result_of_another_experiment_is_refused(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(_models()))
    with pytest.raises(SystemExit, match="not a result of c1a_noise"):
        fig1.load(path, "c1a_noise")


def test_default_inputs_follow_the_current_configs():
    for name in ("c1a", "c1b"):
        path = fig1.result_path(name)
        assert path.parent == fig1.ROOT / "results"
        assert path.name.startswith(f"{name}_") and len(path.stem) == len(name) + 9
