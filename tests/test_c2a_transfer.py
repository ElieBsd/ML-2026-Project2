"""Tests of the transfer experiment on simulated counts: no data is read."""

import collections
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from cdpbpc.calibrate import conformal_threshold
from cdpbpc.scores import CLASS_EDGES_4, hamming
from cdpbpc.simulate import simulate_counts
from experiments import c2a_transfer as c2a

ROOT = Path(__file__).resolve().parents[1]


def _codebook(rng):
    cb = np.concatenate([rng.uniform(lo, hi, 128) for lo, hi in itertools.pairwise(CLASS_EDGES_4)])
    rng.shuffle(cb)
    return cb


def _file(rng, codebook, ids, cls="original"):
    L = rng.multinomial(22 * 22, np.full(512, 1 / 512), size=len(ids))
    D = simulate_counts(L, codebook, rng)
    return SimpleNamespace(ids=np.asarray(ids), D=D, L=L, meta={"class": cls})


SIZES = {"E": 50, "M": 100, "S": 200}
CFG = {"alphas": [0.05], "recal_K": [50], "recal_draws": 50, "band_level": 0.9, "mu": 0.01}


def test_config_pairs_follow_the_transfer_matrix():
    cfg = yaml.safe_load((ROOT / "configs" / "c2a.yaml").read_text())
    n = {k: len(c2a.expand_pairs(v)) for k, v in cfg["changes"].items()}
    assert n == {
        "none": 25,  # 10 (HPI55) + 3 (HPI76) + 6 per phone
        "pages": 3,
        "scan": 12,
        "capture": 60,  # 30 per phone
        "print_run": 96,  # 90 (HPI55) + 6 (HPI76)
        "session": 50,
        "printer": 24,  # 12 each way
        "phone": 18,  # 9 each way
    }
    groups = collections.Counter(p["group"] for p in c2a.expand_pairs(cfg["changes"]["session"]))
    assert len(groups) == 8  # 4 couples of sessions per printer
    assert all(
        p["ref"] != p["target"]
        for k in ("scan", "capture", "print_run", "session", "printer", "phone")
        for p in c2a.expand_pairs(cfg["changes"][k])
    )
    names = {
        f
        for v in cfg["changes"].values()
        for p in c2a.expand_pairs(v)
        for f in (p["ref"], p["target"])
    }
    assert not any("fake" in f or "estimated" in f for f in names)


def test_enrol_draws_disjoint_blocks_in_the_pool():
    rng = np.random.default_rng(0)
    cb = _codebook(rng)
    f = _file(rng, cb, np.arange(1, 501))
    det = c2a.enrol(f.ids, f.D, f.L, SIZES, np.random.default_rng(1), 0.01, pool=np.arange(1, 401))
    assert len(det.used) == len(np.unique(det.used)) == 350
    assert det.used.max() <= 400
    assert det.cal["hamming"].shape == (200,)
    with pytest.raises(ValueError):
        c2a.enrol(f.ids, f.D, f.L, SIZES, rng, 0.01, pool=np.arange(1, 300))


def test_judge_leaves_out_enrolment_templates():
    rng = np.random.default_rng(2)
    cb = _codebook(rng)
    f = _file(rng, cb, np.arange(1, 501))
    det = c2a.enrol(f.ids, f.D, f.L, SIZES, rng, 0.01)
    s = c2a.judge(det, f.ids, f.D, f.L)
    keep = ~np.isin(f.ids, det.used)
    assert len(s["hamming"]) == 150
    np.testing.assert_allclose(s["hamming"], hamming(f.D[keep], f.L[keep]))
    s2 = c2a.judge(det, f.ids, f.D, f.L, keep=np.arange(1, 251))
    assert len(s2["hamming"]) == np.sum(keep & (f.ids <= 250))


def test_fixed_threshold_holds_in_the_same_condition():
    rng = np.random.default_rng(3)
    cb = _codebook(rng)
    miss = []
    for r in range(30):
        f = _file(rng, cb, np.arange(1, 1001))
        det = c2a.enrol(f.ids, f.D, f.L, SIZES, rng, 0.01)
        res = c2a.pair_result(det, c2a.judge(det, f.ids, f.D, f.L), CFG, rng)
        miss.append(res["fixed"]["hamming"][0.05])
        assert set(res["fixed"]) == set(c2a.SCORES)
    assert np.mean(miss) == pytest.approx(0.05, abs=0.015)


def test_shifted_target_breaks_the_threshold_and_recalibration_repairs_it():
    rng = np.random.default_rng(4)
    cb = _codebook(rng)
    ref = _file(rng, cb, np.arange(1, 1001))
    tgt = _file(rng, np.clip(cb * 1.3, 0, 0.99), np.arange(1, 1001))  # more flips everywhere
    det = c2a.enrol(ref.ids, ref.D, ref.L, SIZES, rng, 0.01)
    target = c2a.judge(det, tgt.ids, tgt.D, tgt.L)
    res = c2a.pair_result(det, target, CFG, rng)
    assert res["fixed"]["hamming"][0.05] > 0.5
    rec = res["recal"]["hamming"][0.05][50]
    assert rec["mean"] == pytest.approx(0.05, abs=0.02)
    assert len(rec["p_miss"]) == 50
    # fixed threshold: the one of S, applied to the target originals
    tau = conformal_threshold(det.cal["hamming"], 0.05)
    assert res["fixed"]["hamming"][0.05] == np.mean(target["hamming"] > tau)


def test_recalibration_skips_k_larger_than_the_target():
    rng = np.random.default_rng(5)
    cb = _codebook(rng)
    f = _file(rng, cb, np.arange(1, 401))
    det = c2a.enrol(f.ids, f.D, f.L, SIZES, rng, 0.01)  # 50 originals left
    res = c2a.pair_result(det, c2a.judge(det, f.ids, f.D, f.L), CFG, rng)
    assert res["recal"]["hamming"][0.05] == {}


def test_a_file_of_fakes_is_refused():
    with pytest.raises(SystemExit):
        c2a.check_original("mobile_fake_x", {"class": "fake"})
    c2a.check_original("mobile_original_x", {"class": "original"})


def test_seeded_depends_on_names_only():
    a = c2a.seeded(0, "enrol", "f1").random(3)
    assert np.array_equal(a, c2a.seeded(0, "enrol", "f1").random(3))
    assert not np.array_equal(a, c2a.seeded(0, "enrol", "f2").random(3))
    assert not np.array_equal(a, c2a.seeded(1, "enrol", "f1").random(3))


def test_page_ids():
    np.testing.assert_array_equal(c2a.page_ids((1, 1)), np.arange(1, 145))
    assert c2a.page_ids((6, 10))[0] == 721 and c2a.page_ids((6, 10))[-1] == 1440


def _small_config():
    groups = lambda refs, targets: [{"name": "g", "refs": refs, "targets": targets}]
    return {
        "seed": 0,
        "sizes": {"E": 20, "M": 40, "S": 70},
        "mu": 0.01,
        "alphas": [0.05],
        "recal_K": [20],
        "recal_draws": 10,
        "band_level": 0.9,
        "changes": {
            "none": {
                "groups": [{"name": "HPI55_session1", "same_file": True, "files": ["a", "b"]}]
            },
            "pages": {
                "ref_pages": [1, 1],
                "target_pages": [2, 2],
                "groups": [{"name": "p", "same_file": True, "files": ["a"]}],
            },
            "session": {"groups": groups(["a", "b"], ["c"])},
            "phone": {"groups": groups(["a"], ["b", "c"])},
        },
        "figure": {"alpha": 0.05, "K": 20, "scores": ["hamming", "masked_hamming"]},
        "predictions": {
            "same_run": {"alpha": 0.05, "min_share": 0.8, "groups": ["HPI55_session1"]},
            "between_sessions": {"alpha": 0.05, "high": 0.2, "low": 0.01},
            "mask_vs_count": {"max_ratio": 0.5},
            "recalibration": {"alpha": 0.05, "K": 20, "range": [0.02, 0.09], "min_share": 0.9},
        },
    }


def test_run_end_to_end_on_simulated_files(tmp_path):
    rng = np.random.default_rng(6)
    cb = _codebook(rng)
    files = {
        "a": _file(rng, cb, np.arange(1, 289)),
        "b": _file(rng, cb, np.arange(1, 289)),
        "c": _file(rng, np.clip(cb * 1.5, 0, 0.99), np.arange(1, 289)),  # another session
    }
    cfg = _small_config()
    res = c2a.run(cfg, files.__getitem__, lambda m: None)
    assert res["changes"]["pages"]["pairs"][0]["n_target"] == 144  # page 2 only
    assert res["changes"]["session"]["summary"]["n_pairs"] == 2
    assert res["changes"]["session"]["summary"]["fixed"]["hamming"][0.05]["median"] > 0.5
    import json

    from experiments.e0_sanity import _round

    written = json.loads(json.dumps(_round({"config": cfg, **res}, 5)))
    pred = c2a.predictions(written, cfg)
    assert set(pred) == {"same_run", "between_sessions", "mask_vs_count", "recalibration"}
    assert pred["between_sessions"]["holds"]  # the shifted session rejects far more than 20 %
    c2a.draw_figure(written, tmp_path / "fig2.pdf")
    assert (tmp_path / "fig2.pdf").stat().st_size > 0


def test_run_refuses_a_file_of_fakes():
    rng = np.random.default_rng(7)
    cb = _codebook(rng)
    files = {
        "a": _file(rng, cb, np.arange(1, 289)),
        "b": _file(rng, cb, np.arange(1, 289), cls="fake"),
        "c": _file(rng, cb, np.arange(1, 289)),
    }
    with pytest.raises(SystemExit):
        c2a.run(_small_config(), files.__getitem__, lambda m: None)


# ----------------------------------------------------------------------------- Scantrust


def _scantrust(rng, shift_batch=1.0):
    """Simulated Scantrust originals: unique 1..5000, batch 3501..3550 x 50 copies."""
    cb = _codebook(rng)
    s, r, c = np.meshgrid(np.arange(25), np.arange(10), np.arange(10), indexing="ij")
    s, r, c = s.ravel(), r.ravel(), c.ravel()
    b_ids = 3501 + 10 * (s // 5) + r
    b_prints = 100 * s + 10 * r + c + 1
    u = _file(rng, cb, np.arange(1, 5001))
    b = _file(rng, np.clip(cb * shift_batch, 0, 0.99), b_ids)
    return SimpleNamespace(
        ids=np.r_[u.ids, b.ids],
        prints=np.r_[np.full(5000, np.nan), b_prints],
        dataset=np.array(["unique"] * 5000 + ["batch"] * 2500),
        D=np.r_[u.D, b.D],
        L=np.r_[u.L, b.L],
    )


def test_rates_by_template_and_recalibration_on_whole_templates():
    rng = np.random.default_rng(8)
    cb = _codebook(rng)
    tids = np.repeat(np.arange(10), 50)
    f = _file(rng, cb, np.arange(600))
    det = c2a.enrol(f.ids, f.D, f.L, SIZES, rng, 0.01)
    t = _file(rng, np.clip(cb * 1.3, 0, 0.99), tids)
    t.D[tids == 3] = 0  # one template never rejected
    target = c2a.all_scores(t.D, t.L, det.codebook, det.fit, 0.01)
    cfg = {**CFG, "recal_K": [20, 50, 100, 500], "recal_draws": 30}
    res = c2a.pair_result_by_template(det, target, tids, cfg, rng)
    tau = conformal_threshold(det.cal["hamming"], 0.05)
    per = [np.mean(target["hamming"][tids == g] > tau) for g in range(10)]
    assert res["fixed"]["hamming"][0.05] == pytest.approx(np.mean(per))
    assert per[3] == 0 and res["n_templates"] == 10
    rec = res["recal"]["hamming"][0.05]
    assert set(rec) == {50, 100}  # 20 is not whole templates, 500 leaves none to judge
    assert rec[100]["templates_per_draw"] == 2 and len(rec[100]["p_miss"]) == 30
    with pytest.raises(ValueError):
        c2a.pair_result_by_template(det, target, tids[:-1].tolist() + [99], cfg, rng)


def test_recalibrated_p_miss_is_measured_on_the_other_templates():
    # every image of template g scores g: a threshold on templates {g1, g2} is max(g1, g2)
    tids = np.repeat(np.arange(10), 50)
    target = {k: tids.astype(float) for k in c2a.SCORES}
    det = SimpleNamespace(cal={k: np.arange(100.0) for k in c2a.SCORES})
    cfg = {**CFG, "recal_K": [100], "recal_draws": 20}
    res = c2a.pair_result_by_template(det, target, tids, cfg, np.random.default_rng(0))
    drawn = res["recal_templates"][100]
    expect = [np.mean([g > max(d) for g in range(10) if g not in d]) for d in drawn]
    np.testing.assert_allclose(res["recal"]["hamming"][0.05][100]["p_miss"], expect)


def test_scantrust_rows_keep_blocks_apart(monkeypatch):
    rng = np.random.default_rng(9)
    st = _scantrust(rng, shift_batch=1.3)
    seen = []
    real = c2a.detector_from_blocks

    def spy(ids, D, L, iE, iM, iS, mu):
        det = real(ids, D, L, iE, iM, iS, mu)
        seen.append((iE, iM, iS))
        return det

    monkeypatch.setattr(c2a, "detector_from_blocks", spy)
    cfg = {
        **CFG,
        "seed": 0,
        "recal_K": [50, 100],
        "recal_draws": 5,
        "scantrust": {"unique_seeds": 2, "batch_groups": [0, 1, 2, 3, 4]},
    }
    res = c2a.run_scantrust(cfg, st, lambda m: None)
    assert {k: len(v["pairs"]) for k, v in res.items()} == {
        "scantrust_same": 2,
        "scantrust_sheets": 20,
        "scantrust_session": 7,
    }
    for iE, iM, iS in seen:
        e, m, s = (set(st.ids[x]) for x in (iE, iM, iS))
        assert not (e & m or e & s or m & s)
        if st.dataset[iE][0] == "batch":
            assert iE.sum() == 50 and len(e) == 10  # one copy per template and sheet
            assert np.all((st.prints[iE] - 1) % 10 == 0)
    for p in res["scantrust_sheets"]["pairs"]:
        r, t = int(p["ref"][-1]), int(p["target"][-1])
        assert len({r, t, p["E_group"], p["M_group"]}) == 4
        assert p["n_target"] == 500 and p["n_templates"] == 10
    sess = res["scantrust_session"]["pairs"]
    assert [p["n_target"] for p in sess] == [2500, 2500, 3500, 3500, 3500, 3500, 3500]
    same = res["scantrust_same"]["pairs"][0]
    assert same["n_target"] == 1000
    # the batch sheets flip more: unique detectors reject most batch originals
    assert sess[0]["fixed"]["hamming"][0.05] > 0.5
    assert same["fixed"]["hamming"][0.05] < 0.15


def test_run_adds_scantrust_rows_when_configured():
    rng = np.random.default_rng(10)
    cb = _codebook(rng)
    files = {k: _file(rng, cb, np.arange(1, 289)) for k in "abc"}
    cfg = _small_config()
    cfg["scantrust"] = {"unique_seeds": 1, "batch_groups": [0, 1, 2, 3]}
    cfg["recal_K"] = [20, 100]  # 100 = two whole batch templates
    cfg["predictions"]["recalibration"]["K"] = 100
    cfg["predictions"]["mask_vs_count_scantrust"] = {"alpha": 0.05, "max_ratio": 0.5}
    res = c2a.run(cfg, files.__getitem__, lambda m: None, _scantrust(rng, 1.3))
    assert {"scantrust_same", "scantrust_sheets", "scantrust_session"} <= set(res["changes"])
    assert len(res["changes"]["scantrust_sheets"]["pairs"]) == 12
    import json

    from experiments.e0_sanity import _round

    written = json.loads(json.dumps(_round({"config": cfg, **res}, 5)))
    pred = c2a.predictions(written, cfg)
    assert "mask_vs_count_scantrust" in pred
    assert "scantrust_sheets" in pred["recalibration"]["share_per_change"]
