"""Tests of the detection experiment on simulated counts: no data is read."""

import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from cdpbpc.calibrate import conformal_threshold
from cdpbpc.scores import CLASS_EDGES_4, hamming
from cdpbpc.simulate import simulate_counts
from cdpbpc.splits import batch_e_print_ids, indigo_halves
from experiments import c3a_detection as c3a

ROOT = Path(__file__).resolve().parents[1]
CFG = {"seed": 0, "mu": 0.01, "n_boot": 200}


def _codebook(rng):
    cb = np.concatenate([rng.uniform(lo, hi, 128) for lo, hi in itertools.pairwise(CLASS_EDGES_4)])
    rng.shuffle(cb)
    return cb


def _counts(rng, codebook, ids, prints=None, pcor=None):
    ids = np.asarray(ids)
    L = rng.multinomial(22 * 22, np.full(512, 1 / 512), size=len(ids))
    D = simulate_counts(L, codebook, rng)
    pc = rng.normal(0, 1, len(ids)) if pcor is None else pcor
    return c3a.Counts(ids, D, L, pc, prints)


def _log(msg):
    pass


# ----------------------------------------------------------------------------- pieces


def test_detector_scores_from_counts():
    rng = np.random.default_rng(0)
    cb = _codebook(rng)
    E, M, J = (_counts(rng, cb, np.arange(n)) for n in (50, 150, 30))
    det = c3a.Detector(E, M, 0.01)
    np.testing.assert_allclose(det.codebook, (E.D.sum(0) + 0.5) / (E.L.sum(0) + 1))
    s = det.scores(J)
    assert set(s) == set(c3a.SCORES)
    np.testing.assert_array_equal(s["pcor"], J.pcor)  # pcor is read, not computed
    np.testing.assert_allclose(s["hamming"], hamming(J.D, J.L))


def test_profile_of_empty_class_is_none():
    assert c3a.profile(np.array([[1.0, 0.0], [4.0, 0.0]])) == [0.25, None]


def test_class_sums_add_up_to_all_flips():
    rng = np.random.default_rng(1)
    cb = _codebook(rng)
    det = c3a.Detector(_counts(rng, cb, range(50)), _counts(rng, cb, range(100)), 0.01)
    J = _counts(rng, cb, range(20))
    s = det.class_sums(J)
    assert s.shape == (2, 8)
    assert s[0].sum() == J.D.sum() and s[1].sum() == J.L.sum()


def test_measures_with_clopper_pearson_and_equal_p_miss():
    rng = np.random.default_rng(2)
    o = {k: rng.normal(0, 1, 400) for k in c3a.SCORES}
    f = {k: rng.normal(2, 1, 300) for k in c3a.SCORES}
    cal = {k: rng.normal(0, 1, 1000) for k in c3a.SCORES}
    m = c3a.measures_images(o, f, c3a.scalar_tau(cal, [0.05], 400, 300), [0.05])
    r = m["hamming"]["alpha_0.05"]
    tau = conformal_threshold(cal["hamming"], 0.05)
    assert r["p_fa"] == np.mean(f["hamming"] <= tau)
    assert r["p_miss"] == np.mean(o["hamming"] > tau)
    assert r["p_fa_ci"][0] <= r["p_fa"] <= r["p_fa_ci"][1]
    assert len(r["fake_correct"]) == 300
    eq = r["equal_p_miss_test_originals"]
    assert eq["p_miss"] == pytest.approx(0.05, abs=0.005)
    assert 0.5 < m["hamming"]["auc"] <= 1


# ----------------------------------------------------------------------------- Indigo


def _indigo(rng, setting):
    cb = _codebook(rng)
    top = 1440 if setting == "mobile" else 720
    orig = _counts(rng, cb, np.arange(1, top + 1))
    fake = _counts(rng, np.clip(cb * 2, 0, 0.99), np.arange(1, top + 1))
    return orig, fake


def test_indigo_dev_judges_only_development_templates():
    rng = np.random.default_rng(3)
    orig, fake = _indigo(rng, "scanner")
    out = c3a.run_indigo_condition("scanner", orig, {"f": fake}, [0.05], CFG, "dev", _log)
    r = out["fakes"]["f"]["hamming"]["alpha_0.05"]
    assert r["n_originals"] == r["n_fakes"] == 288
    assert out["n_E"] == 50 and out["n_M"] == 150 and out["n_S"] == 720 - 288 - 200
    assert r["p_fa"] < 0.05  # fakes flip twice as often
    assert set(out["class_profile"]) == {"originals", "f"}


def test_indigo_test_uses_the_threshold_of_the_other_half():
    rng = np.random.default_rng(4)
    orig, fake = _indigo(rng, "scanner")
    out = c3a.run_indigo_condition("scanner", orig, {"f": fake}, [0.05], CFG, "test", _log)
    r = out["fakes"]["f"]["hamming"]["alpha_0.05"]
    assert r["n_originals"] == r["n_fakes"] == 432
    a, b = indigo_halves("scanner")
    h = {name: hamming(orig.of(x).D, orig.of(x).L) for name, x in (("a", a), ("b", b))}
    hf = {name: hamming(fake.of(x).D, fake.of(x).L) for name, x in (("a", a), ("b", b))}
    tau_a = conformal_threshold(h["b"], 0.05)  # half a judged with the originals of b
    tau_b = conformal_threshold(h["a"], 0.05)
    expected = np.mean(np.r_[hf["a"] <= tau_a, hf["b"] <= tau_b])
    assert r["p_fa"] == pytest.approx(expected)
    exp_miss = np.mean(np.r_[h["a"] > tau_a, h["b"] > tau_b])
    assert r["p_miss"] == pytest.approx(exp_miss)


def test_indigo_test_refuses_missing_pcor():
    rng = np.random.default_rng(5)
    orig, fake = _indigo(rng, "scanner")
    fake.pcor[fake.ids >= 289] = np.nan  # as in the compact pack before the freeze
    with pytest.raises(SystemExit):
        c3a.run_indigo_condition("scanner", orig, {"f": fake}, [0.05], CFG, "test", _log)
    c3a.run_indigo_condition("scanner", orig, {"f": fake}, [0.05], CFG, "dev", _log)  # fine


# ----------------------------------------------------------------------------- Scantrust


def _batch_frame(rng, attacks=("otsu+unsharp",)):
    cb = _codebook(rng)
    rows = []
    for k in range(50):
        g, rank = divmod(k, 10)
        for s in range(5 * g, 5 * g + 5):
            for c in range(10):
                rows.append(("batch", 3501 + k, "original", None, 100 * s + 10 * rank + c + 1))
    n_o = len(rows)
    for a in attacks:
        for k in range(10):  # fakes of group 0 only, as at the development stage
            for c in range(50):
                rows.append(("batch", 3501 + k, "fake", a, -1))
    r = pd.DataFrame(rows, columns=["dataset", "template_id", "class", "attack", "print_id"])
    o = _counts(rng, cb, r["template_id"][:n_o])
    f = _counts(rng, np.clip(cb * 2, 0, 0.99), r["template_id"][n_o:])
    allc = c3a.concat(o, f)
    allc.prints = r["print_id"].to_numpy()
    return allc, r


def test_batch_rates_per_template_on_group_0():
    rng = np.random.default_rng(6)
    allc, r = _batch_frame(rng)
    cfg = {**CFG, "scantrust": {"batch": {"attacks": ["otsu+unsharp"], "alphas": [0.05]}}}
    out = c3a.run_batch(allc, r, cfg, "dev", rng, _log)
    res = out["attacks"]["otsu+unsharp"]["hamming"]["alpha_0.05"]["conformal"]
    assert res["n_templates"] == 10 and len(res["p_fa_per_template"]) == 10
    assert res["p_fa"] < 0.05
    lo, hi = res["p_fa_ci_template_bootstrap"]
    assert lo <= res["p_fa"] <= hi
    assert out["groups"] == [0]
    fold = out["folds"][0]
    assert fold["n_E"] == 50 and fold["n_M"] == 500 and fold["n_S"] == 1000
    # E: copy 0 of each template of group 1 on each of its sheets, never a judged image
    assert fold["E_print_ids"] == sorted(batch_e_print_ids(1).tolist())


def test_unique_uses_development_fakes_at_dev():
    rng = np.random.default_rng(7)
    cb = _codebook(rng)
    o_ids = np.arange(1, 5001)
    f_ids = np.arange(3551, 4001)
    r = pd.DataFrame(
        {
            "dataset": "unique",
            "template_id": np.r_[o_ids, f_ids],
            "class": ["original"] * len(o_ids) + ["fake"] * len(f_ids),
            "attack": [None] * len(o_ids) + ["otsu"] * len(f_ids),
            "print_id": -1,
        }
    )
    allc = c3a.concat(_counts(rng, cb, o_ids), _counts(rng, np.clip(cb * 2, 0, 0.99), f_ids))
    cfg = {**CFG, "scantrust": {"unique": {"n_seeds": 2, "attacks": ["otsu"], "alphas": [0.05]}}}
    out = c3a.run_unique(allc, r, cfg, "dev", _log)
    s0 = out["attacks"]["otsu"]["hamming"]["alpha_0.05"]["seed_0"]
    assert s0["n_fakes"] == 450 and s0["n_originals"] == 1000
    assert out["n_seeds"] == 2


# ----------------------------------------------------------------------------- predictions


def test_predictions_read_the_right_numbers():
    def mob(fa):
        return {
            "fakes": {
                "EHPI55": {
                    k: {"alpha_0.05": {"p_fa": fa[k]}, "alpha_0.01": {"p_fa": 0.9}}
                    for k in c3a.SCORES
                }
            }
        }

    fa_i = dict.fromkeys(c3a.SCORES, 0.8) | {"masked_hamming": 0.0}
    res = {
        "indigo": {
            "mobile_iPhone12Pro": mob(fa_i),
            "mobile_SamsungGN20U": mob(dict.fromkeys(c3a.SCORES, 0.9)),
        },
        "scantrust_batch": {
            "attacks": {
                "otsu+unsharp": {
                    k: {
                        "alpha_0.05": {
                            "conformal": {"p_fa": 0.56 if k == "masked_hamming" else 0.53}
                        }
                    }
                    for k in c3a.SCORES
                }
            }
        },
    }
    cfg = {
        "predictions": {
            "replication": {
                "device": "iPhone12Pro",
                "alpha": 0.05,
                "max_masked": 0.02,
                "min_hamming": 0.5,
            },
            "operating_point": {"device": "SamsungGN20U", "alpha": 0.01, "min_p_fa": 0.5},
            "averaging_breaks_mask": {"attack": "otsu+unsharp", "alpha": 0.05},
        }
    }
    p = c3a.predictions(res, cfg)
    assert p["replication"]["holds"] and p["operating_point"]["holds"]
    assert p["averaging_breaks_mask"]["holds"]
    res["indigo"]["mobile_iPhone12Pro"] = mob(fa_i | {"masked_hamming": 0.03})
    assert not c3a.predictions(res, cfg)["replication"]["holds"]
    res["indigo"]["mobile_iPhone12Pro"] = mob(fa_i | {"hamming": 0.4})
    assert not c3a.predictions(res, cfg)["replication"]["holds"]


# ----------------------------------------------------------------------------- main


def _no_data(*args, **kwargs):
    raise AssertionError("a test must never read data")


def _config(tmp_path, freeze):
    text = (
        (ROOT / "configs" / "c3a.yaml")
        .read_text()
        .replace("freeze_commit: null", f"freeze_commit: {freeze}")
    )
    p = tmp_path / "c3a.yaml"
    p.write_text(text)
    return p


def test_main_test_stage_needs_a_freeze_commit(tmp_path):
    with pytest.raises(SystemExit):
        c3a.main(["--config", str(_config(tmp_path, "null")), "--stage", "test"], run_fn=_no_data)


def test_main_test_stage_refuses_a_fake_commit(tmp_path):
    with pytest.raises(SystemExit):
        c3a.main(
            ["--config", str(_config(tmp_path, "deadbeef")), "--stage", "test"], run_fn=_no_data
        )


def test_main_test_stage_never_overwrites(tmp_path, monkeypatch):
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    cfg = _config(tmp_path, head)
    monkeypatch.setattr(c3a, "RESULTS", tmp_path)
    out = c3a.result_path(cfg.read_text(), "test")
    assert out.name.endswith("_test.json") and out.parent == tmp_path
    out.write_text("{}")
    with pytest.raises(SystemExit):
        c3a.main(["--config", str(cfg), "--stage", "test"], run_fn=_no_data)
    assert out.read_text() == "{}"


def test_dev_and_test_results_have_different_names():
    assert c3a.result_path("x", "dev") != c3a.result_path("x", "test")
