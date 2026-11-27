import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from experiments import table1

A = table1.ALPHA


def _img(p_fa, p_miss=0.05):
    return {"p_fa": p_fa, "p_miss": p_miss, "p_fa_ci": [max(p_fa - 0.01, 0), p_fa + 0.01]}


def _tpl(p_fa, p_miss=0.06):
    return {"p_fa": p_fa, "p_miss": p_miss, "p_fa_ci_template_bootstrap": [p_fa - 0.1, p_fa + 0.1]}


SHORT = ("f55/55", "f55/76", "f76/55", "f76/76")
LONG = ("HPI55_EHPI55", "HPI55_EHPI76", "HPI76_EHPI55", "HPI76_EHPI76")


def _indigo(keys, value):
    """Detection-like indigo block: P_fa of a fake type = its index / 100 + value."""
    out = {}
    for ph in table1.PHONES:
        out[f"mobile_{ph}"] = {"fakes": {"EHPI55": {k: {A: _img(value)} for k in keys}}}
    for p in table1.PRINTERS:
        fakes = {fk: {k: {A: _img(value + i / 100)} for k in keys} for i, fk in enumerate(SHORT)}
        out[f"scanner_{p}"] = {"fakes": fakes}
    return out


def _results():
    scores = [k for k, _ in table1.SCORES]
    masks = [k for k, _ in table1.MASKS]
    c3a = {
        "experiment": "c3a_detection",
        "stage": "test",
        "git": "abc",
        "indigo": _indigo(scores, 0.2),
        "scantrust_batch": {
            "attacks": {
                a: {k: {A: {"conformal": _tpl(0.5)}} for k in scores} for a in table1.ATTACKS
            }
        },
    }
    p3 = {
        "experiment": "p3_context_mask",
        "stage": "test",
        "git": "abc",
        "indigo": _indigo(masks, 0.1),
        "scantrust_batch": {a: {k: {A: _tpl(0.7)} for k in masks} for a in table1.ATTACKS},
    }
    p1 = {
        "experiment": "p1_vote",
        "stage": "test",
        "git": "abc",
        "scantrust_batch": {
            "attacks": {a: {"lrt_vote": {A: {"conformal": _tpl(0.04)}}} for a in table1.ATTACKS}
        },
    }
    mh = {"single": {A: _img(0.6)}, "min3": {A: _img(0.003)}}
    p2 = {
        "experiment": "p2_best_of_3",
        "stage": "test",
        "git": "abc",
        "indigo_mobile": {ph: {"results": {"masked_hamming": mh}} for ph in table1.PHONES},
    }

    def logistic(judged, value):
        return {fk: {A: {"conformal": _img(value)}} for fk in judged}

    conditions = {}
    for ph, other in zip(table1.PHONES, table1.PHONES[::-1]):
        conditions[f"mobile {ph}"] = {
            "oracle": {
                f"trained on {ph}": {"logistic": logistic(["EHPI55"], 0.01)},
                f"trained on {other}": {"logistic": logistic(["EHPI55"], 0.3)},
            }
        }
    for p in table1.PRINTERS:
        # trained on the printer of the originals: 0.0 on its own fakes, 0.4 on the others;
        # trained on the other printer: 0.0 on its own fakes, 0.8 on those of the originals
        oracle = {}
        for row in LONG:
            own = row.startswith(p)
            oracle[row] = {
                "logistic": {
                    fk: {A: {"conformal": _img(0.0 if fk == row else (0.4 if own else 0.8))}}
                    for fk in LONG
                }
            }
        conditions[f"scanner {p}"] = {"oracle": oracle}
    by_c = {
        c: {a: {A: {"conformal": {"p_fa": v}}} for a in table1.ATTACKS}
        for c, v in (("0.1", 0.9), ("10.0", 0.7))
    }
    batch = {a: {"logistic": logistic(table1.ATTACKS, 0.05)} for a in table1.ATTACKS}
    batch["unique V_U (all attacks)"] = {
        "logistic": logistic(table1.ATTACKS, 0.7),
        "logistic_by_C": by_c,
        "models": [{"C": 10.0}, {"C": 10.0}],
    }
    conditions["batch"] = {"oracle": batch}
    c3c = {"experiment": "c3c_oracle", "stage": "test", "git": "abc", "conditions": conditions}
    return {"c3a": c3a, "c3c": c3c, "p1": p1, "p2": p2, "p3": p3}


def test_scanner_fakes_split_by_the_printer_of_the_fake():
    assert table1.scanner_fakes("HPI55", True, "short") == ["f55/55", "f55/76"]
    assert table1.scanner_fakes("HPI55", False, "short") == ["f76/55", "f76/76"]
    assert table1.scanner_fakes("HPI76", True, "long") == ["HPI76_EHPI55", "HPI76_EHPI76"]


def test_scanner_columns_are_ranges_without_interval():
    rows = dict(table1.detection_rows(_results()["c3a"]))
    same = rows["Hamming distance"]["scanner_same"]
    # originals HPI55: f55/* are fakes 0 and 1; originals HPI76: f76/* are fakes 2 and 3
    assert same.p_fa == pytest.approx((0.2, 0.23))
    assert same.ci is None


def test_oracle_scanner_cells_cross_the_printers():
    same, other = (cells for _, cells in table1.oracle_rows(_results()["c3c"]))
    assert same["scanner_same"].p_fa == (0.0, 0.0)
    # same-printer fakes judged by models trained on the other printer's fakes
    assert other["scanner_same"].p_fa == (0.8, 0.8)
    # other-printer fakes judged by models trained on the same printer's fakes
    assert other["scanner_other"].p_fa == (0.4, 0.4)
    assert other["iPhone"].p_fa == (0.3,)


def test_extension_rows_fill_only_their_columns():
    rows = dict(table1.extension_rows(*(_results()[k] for k in ("p1", "p2", "p3"))))
    assert set(rows["likelihood ratio, averaging attacker"]) == set(table1.ATTACKS)
    assert set(rows["masked Hamming, best of three captures"]) == {"iPhone", "Samsung"}
    assert rows["masked Hamming, best of three captures"]["Samsung"].p_fa == (0.003,)


def test_bold_marks_the_lowest_rate_without_fakes_never_the_oracle():
    blocks, by_c = table1.build(_results())
    best = table1.best_fake_free(blocks[:2])
    assert best["otsu+unsharp"] == pytest.approx(0.04)
    assert best["Samsung"] == pytest.approx(0.003)
    tex = table1.render(blocks, by_c, {"x.json": "abc"})
    assert r"\textbf{4.0}" in tex and r"\textbf{0.3}" in tex
    assert r"\textbf{1.0}" not in tex  # oracle iPhone, 0.01, lower than the best without fakes


def test_other_session_range_over_c():
    by_c = table1.other_session_by_c(_results()["c3c"])
    assert by_c["chosen"] == [10.0]
    assert by_c["range"]["otsu+unsharp"] == (0.7, 0.9)


def test_formatting():
    assert table1.pct(0.5804) == "58.0"
    assert table1.span((0.0, 0.00004)) == "0.0"
    assert table1.span((0.346, 1.0)) == "34.6--100.0"
    assert table1.shade(0.0) == 0 and table1.shade(1.0) == table1.SHADE_MAX
    assert table1.cell_tex(None, False) == "--"


def test_main_writes_the_table(tmp_path, monkeypatch):
    monkeypatch.setattr(table1, "ROOT", tmp_path)
    argv = []
    for name, res in _results().items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(res))
        argv += [f"--{name}", str(path)]
    assert table1.main(argv) == 0
    tex = (tmp_path / "report" / "table1.tex").read_text()
    assert tex.count(r"\\") >= 14 and r"\label{tab:detection}" in tex


def test_a_development_result_is_refused(tmp_path):
    path = tmp_path / "c3a.json"
    path.write_text(json.dumps({**_results()["c3a"], "stage": "dev"}))
    with pytest.raises(SystemExit, match="not a test-stage result"):
        table1.load(path, "c3a_detection")


def test_default_inputs_follow_the_frozen_configs():
    for name in table1.SOURCES:
        path = table1.result_path(name)
        assert path.name.startswith(f"{name}_") and path.name.endswith("_test.json")
