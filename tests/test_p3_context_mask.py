import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

import experiments.p3_context_mask as p3


@pytest.fixture
def no_data(monkeypatch):
    """A main() test must never reach real data."""

    def refuse(*args, **kwargs):
        raise AssertionError("a main() test reached the data")

    for name in ("run_indigo", "run_scantrust", "load_compact", "load_scantrust"):
        monkeypatch.setattr(p3, name, refuse)


def _config(tmp_path, freeze):
    path = tmp_path / "p3.yaml"
    path.write_text(yaml.safe_dump({"data_dir": "dataset", "seed": 0, "freeze_commit": freeze}))
    return path


def test_test_stage_needs_a_real_freeze_commit(tmp_path, no_data):
    with pytest.raises(SystemExit, match="freeze_commit"):
        p3.main(["--config", str(_config(tmp_path, None)), "--stage", "test"])
    with pytest.raises(SystemExit, match="not a commit"):
        p3.main(["--config", str(_config(tmp_path, "abc")), "--stage", "test"])


def test_the_test_result_is_never_overwritten(tmp_path, monkeypatch, no_data):
    monkeypatch.setattr(p3, "RESULTS", tmp_path / "results")
    path = _config(tmp_path, "abc")
    text = path.read_text()
    dev, test = p3.result_path(text, "dev"), p3.result_path(text, "test")
    assert dev != test and test.name.endswith("_test.json") and dev.name.startswith("p3_")
    test.parent.mkdir()
    test.write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        p3.main(["--config", str(path), "--stage", "test"])
    assert test.read_text() == "{}"
