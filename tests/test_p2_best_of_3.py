import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from cdpbpc.calibrate import conformal_threshold
from cdpbpc.patterns import N_PATTERNS
from experiments.p2_best_of_3 import (
    RULES,
    SCORES,
    align,
    check_dev_fakes,
    evaluate,
    fused_scores,
)


def _counts(ids, seed):
    rng = np.random.default_rng(seed)
    n = len(ids)
    L = rng.integers(5, 20, (n, N_PATTERNS))
    D = rng.binomial(L, 0.1)
    return SimpleNamespace(ids=np.asarray(ids), D=D, L=L)


def test_development_stage_refuses_test_fakes():
    check_dev_fakes(np.array([1, 200, 288]), "dev")
    with pytest.raises(RuntimeError):
        check_dev_fakes(np.array([1, 289]), "dev")
    check_dev_fakes(np.array([1, 289]), "test")
    with pytest.raises(ValueError):
        check_dev_fakes(np.array([1]), "all")


def test_align_keeps_templates_with_every_capture_and_lists_the_others():
    a = _counts([3, 1, 2, 5], 0)
    b = _counts([1, 2, 3], 1)
    c = _counts([2, 3, 1, 4], 2)
    ids, D, L, missing = align([a, b, c])
    np.testing.assert_array_equal(ids, [1, 2, 3])
    np.testing.assert_array_equal(missing, [4, 5])
    assert D.shape == L.shape == (3, 3, N_PATTERNS)
    # each capture row follows the template order, whatever the order in its file
    np.testing.assert_array_equal(D[0], a.D[[1, 2, 0]])
    np.testing.assert_array_equal(L[2], c.L[[2, 0, 1]])
    np.testing.assert_array_equal(D[1], b.D)


def test_fused_scores_by_hand():
    # two images, three captures, only patterns 0 (reliable) and 1 (not) occur
    cb = np.full(N_PATTERNS, 0.5)
    cb[0] = 0.005
    D = np.zeros((3, 2, N_PATTERNS), int)
    L = np.zeros((3, 2, N_PATTERNS), int)
    L[:, :, 0] = 100
    L[:, :, 1] = 50
    D[:, 0, 0] = [4, 1, 2]  # image 0: reliable flips per capture
    D[:, 1, 0] = [0, 3, 3]
    D[:, :, 1] = [[10, 20], [0, 5], [30, 30]]
    s = fused_scores(D, L, cb, mu=0.01)
    assert set(s) == set(SCORES) and all(set(s[k]) == set(RULES) for k in SCORES)
    m = s["masked_hamming"]
    np.testing.assert_allclose(m["single"], [0.04, 0.0])
    np.testing.assert_allclose(m["pooled"], [7 / 300, 6 / 300])
    np.testing.assert_allclose(m["min3"], [0.01, 0.0])
    h = s["hamming"]
    np.testing.assert_allclose(h["single"], [14 / 150, 20 / 150])
    np.testing.assert_allclose(h["pooled"], [47 / 450, 61 / 450])
    np.testing.assert_allclose(h["min3"], [min(14, 1, 32) / 150, min(20, 8, 33) / 150])


def test_fused_scores_refuse_an_image_without_reliable_symbol():
    cb = np.full(N_PATTERNS, 0.5)
    D = np.zeros((3, 1, N_PATTERNS), int)
    L = np.ones((3, 1, N_PATTERNS), int)
    with pytest.raises(ValueError):
        fused_scores(D, L, cb, mu=0.01)


def _fs(rng, n, shift):
    return {k: {r: rng.normal(shift, 1, n) for r in RULES} for k in SCORES}


def test_thresholds_come_from_originals_fused_the_same_way():
    rng = np.random.default_rng(3)
    cal, orig, fake = _fs(rng, 400, 0), _fs(rng, 300, 0), _fs(rng, 200, 2)
    cal["masked_hamming"]["min3"] = cal["masked_hamming"]["single"] - 5  # a different law
    out = evaluate([{"cal": cal, "orig": orig, "fake": fake}], [0.01, 0.05])
    for r in RULES:
        for a in (0.01, 0.05):
            res = out["masked_hamming"][r][f"alpha_{a}"]
            tau = conformal_threshold(cal["masked_hamming"][r], a)
            assert res["tau"] == [tau]
            assert res["p_miss"] == pytest.approx(np.mean(orig["masked_hamming"][r] > tau))
            assert res["p_fa"] == pytest.approx(np.mean(fake["masked_hamming"][r] <= tau))
            lo, hi = res["p_fa_ci"]
            assert lo <= res["p_fa"] <= hi
    assert out["masked_hamming"]["min3"]["alpha_0.01"]["p_miss"] > 0.9  # threshold too low
    assert set(out["mcnemar_min3_vs_single"]) == {"alpha_0.01", "alpha_0.05"}


def test_cross_conformal_uses_the_other_half():
    rng = np.random.default_rng(4)
    a_o, b_o = _fs(rng, 300, 0), _fs(rng, 300, 0)
    a_f, b_f = _fs(rng, 100, 2), _fs(rng, 100, 2)
    out = evaluate(
        [{"cal": b_o, "orig": a_o, "fake": a_f}, {"cal": a_o, "orig": b_o, "fake": b_f}], [0.05]
    )
    res = out["hamming"]["pooled"]["alpha_0.05"]
    ta = conformal_threshold(b_o["hamming"]["pooled"], 0.05)
    tb = conformal_threshold(a_o["hamming"]["pooled"], 0.05)
    assert res["tau"] == [ta, tb]
    miss = np.concatenate([a_o["hamming"]["pooled"] > ta, b_o["hamming"]["pooled"] > tb])
    assert res["p_miss"] == pytest.approx(miss.mean())
    assert res["n_originals"] == 600 and res["n_fakes"] == 200


# ----------------------------------------------------------------- test-stage paths


def _phone_files(seed=0):
    """Synthetic counts of 3 captures; pattern 0 reliable, pattern 1 not.

    Originals of pages 3-6 (templates 289-864) are noisier than those of pages 7-10, so a
    cross-conformal threshold rejects many of them and a self-calibrated one would not.
    Original 300 misses capture 2; fake 301 misses capture 3.
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(1, 1441)
    files = {}
    for run in (1, 2, 3):
        for cls in ("original", "fake"):
            keep = ids
            if cls == "original" and run == 2:
                keep = ids[ids != 300]
            if cls == "fake":
                keep = ids[ids >= 289] if run != 3 else ids[(ids >= 289) & (ids != 301)]
            L = np.zeros((len(keep), N_PATTERNS), np.int64)
            L[:, 0], L[:, 1] = 1000, 200
            if cls == "fake":
                p0 = np.full(len(keep), 0.05)
            else:
                p0 = np.where((keep >= 289) & (keep <= 864), 0.02, 0.001)
            D = np.zeros_like(L)
            D[:, 0] = rng.binomial(1000, p0)
            D[:, 1] = rng.binomial(200, 0.3, len(keep))
            name = (
                f"mobile_original_X_run{run}.npz"
                if cls == "original"
                else f"mobile_fake_EHPI55_X_run{run}.npz"
            )
            files[name] = SimpleNamespace(ids=keep, D=D, L=L, meta={"class": cls})
    return files


def test_test_stage_judges_each_half_with_the_other_and_skips_incomplete_templates(
    monkeypatch,
):
    import experiments.p2_best_of_3 as p2

    files = _phone_files()
    monkeypatch.setattr(p2, "indigo_counts", lambda path, stage: files[Path(path).name])
    cfg = {"captures": [1, 2, 3], "seed": 0, "mu": 0.01, "alphas": [0.01, 0.05]}
    out = p2.run_phone(Path("pack"), "X", cfg, "test", lambda m: None)
    info, res = out["info"], out["results"]
    assert info["fakes_not_judged"] == [300, 301]
    assert info["n_fakes_judged"] == 1152 - 2
    assert 300 in info["originals_missing_a_capture"]
    r = res["masked_hamming"]["single"]["alpha_0.01"]
    assert r["n_originals"] == 1152 - 1 and r["n_fakes"] == 1150
    # half a (noisy) judged with the threshold of half b (clean): most of it is rejected
    assert len(r["tau"]) == 2 and r["tau"][0] < r["tau"][1]
    assert r["p_miss"] > 0.3


def test_main_refuses_the_test_stage_without_freeze_and_never_overwrites(tmp_path, monkeypatch):
    import yaml

    import experiments.p2_best_of_3 as p2

    def refuse(*args, **kwargs):
        raise AssertionError("a main() test reached the data")

    monkeypatch.setattr(p2, "indigo_counts", refuse)
    monkeypatch.setattr(p2, "run_phone", refuse)
    path = tmp_path / "p2.yaml"
    path.write_text(yaml.safe_dump({"data_dir": "dataset", "freeze_commit": None}))
    with pytest.raises(SystemExit, match="freeze_commit"):
        p2.main(["--config", str(path), "--stage", "test"])
    monkeypatch.setattr(p2, "RESULTS", tmp_path / "results")
    path.write_text(yaml.safe_dump({"data_dir": "dataset", "freeze_commit": "abc"}))
    test = p2.result_path(path.read_text(), "test")
    assert test != p2.result_path(path.read_text(), "dev")
    test.parent.mkdir()
    test.write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        p2.main(["--config", str(path), "--stage", "test"])
    assert test.read_text() == "{}"
    test.unlink()
    with pytest.raises(SystemExit, match="not a commit"):
        p2.main(["--config", str(path), "--stage", "test"])
