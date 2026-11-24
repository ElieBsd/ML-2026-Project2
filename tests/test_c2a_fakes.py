"""Tests of the transfer with fakes on simulated counts: no data is read."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from cdpbpc.calibrate import conformal_threshold
from cdpbpc.codebook import estimate
from cdpbpc.scores import CLASS_EDGES_4, hamming, masked_hamming
from cdpbpc.simulate import simulate_counts
from cdpbpc.splits import indigo_blocks, indigo_halves
from experiments import c2a_fakes as c2f

ROOT = Path(__file__).resolve().parents[1]
CFG = {"seed": 0, "mu": 0.01, "recal_K": [20, 50], "recal_draws": 20, "band_level": 0.9}


def _codebook(rng):
    cb = np.concatenate([rng.uniform(lo, hi, 128) for lo, hi in itertools.pairwise(CLASS_EDGES_4)])
    rng.shuffle(cb)
    return cb


def _images(rng, codebook, ids):
    ids = np.asarray(ids)
    L = rng.multinomial(22 * 22, np.full(512, 1 / 512), size=len(ids))
    return c2f.Images(ids, simulate_counts(L, codebook, rng), L)


def _setting(rng, n, shift=1.0, missing=()):
    """Reference, target (codebook scaled by shift) and fakes (more flips) of one setting.

    missing: templates without an original in the target file (the reference has them).
    """
    cb = _codebook(rng)
    ref = _images(rng, cb, np.arange(1, n + 1))
    tgt = _images(rng, np.clip(cb * shift, 0, 0.99), np.setdiff1d(np.arange(1, n + 1), missing))
    fk = _images(rng, np.clip(cb * shift + 0.08, 0, 0.99), np.arange(1, n + 1))
    return ref, tgt, {"f": fk}


def test_config_names_the_two_changes():
    cfg = yaml.safe_load((ROOT / "configs" / "c2a_fakes.yaml").read_text())
    assert cfg["session"]["printers"] == ["HPI55", "HPI76"]
    assert "session1" in cfg["session"]["reference"]
    assert cfg["phone"]["pairs"] == [
        ["iPhone12Pro", "SamsungGN20U"],
        ["SamsungGN20U", "iPhone12Pro"],
    ]
    assert cfg["freeze_commit"] is None


def test_dev_threshold_is_the_one_of_reference_originals_of_s():
    rng = np.random.default_rng(0)
    ref, tgt, fakes = _setting(rng, 720, shift=1.3)
    res = c2f.transfer("scanner", ref, tgt, fakes, CFG, [0.05], "dev", rng)
    b = indigo_blocks("scanner", stage="dev", available=ref.ids, seed=0)
    S = ref.of(b["S"])
    tau = conformal_threshold(hamming(S.D, S.L), 0.05)
    o, f = tgt.of(b["V"]), fakes["f"].of(b["V"])
    fixed = res["fakes"]["f"]["hamming"]["alpha_0.05"]["fixed"]
    assert fixed["p_miss"] == np.mean(hamming(o.D, o.L) > tau)
    assert fixed["p_fa"] == np.mean(hamming(f.D, f.L) <= tau)
    assert res["n_originals"] == res["n_fakes"]["f"] == 288
    assert res["n_reference_threshold"] == len(b["S"]) and res["n_recal_pool"] == [len(b["S"])]
    assert fixed["p_miss"] > 0.5  # the shifted target rejects far more than 5 %
    # codebook from E of the reference, even for the target
    E = ref.of(b["E"])
    cb = estimate(E.L.sum(axis=0), E.D.sum(axis=0))
    tau_m = conformal_threshold(masked_hamming(S.D, S.L, cb, 0.01), 0.05)
    m = res["fakes"]["f"]["masked_hamming"]["alpha_0.05"]["fixed"]
    assert m["p_miss"] == np.mean(masked_hamming(o.D, o.L, cb, 0.01) > tau_m)


def test_recalibration_on_target_originals_repairs_the_p_miss():
    # the pool and the judged originals are finite: average over a few settings
    cfg = {**CFG, "recal_draws": 50}
    fixed, rec, fa = [], [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        ref, tgt, fakes = _setting(rng, 720, shift=1.3)
        r = c2f.transfer("scanner", ref, tgt, fakes, cfg, [0.05], "dev", rng)
        x = r["fakes"]["f"]["hamming"]["alpha_0.05"]
        fixed.append(x["fixed"]["p_miss"])
        rec.append(x["recal"][50]["p_miss"]["mean"])
        fa.append(x["recal"][50]["p_fa"])
    assert min(fixed) > 0.5
    assert np.mean(rec) == pytest.approx(2 / 51, abs=0.015)  # E[P_miss], K = 50, k = 49
    assert all(0 <= v["q10"] <= v["mean"] <= v["q90"] <= 1 for v in fa)


def test_test_stage_threshold_comes_from_the_other_half_of_the_reference():
    rng = np.random.default_rng(2)
    ref, tgt, fakes = _setting(rng, 720, missing=(300, 301))
    res = c2f.transfer("scanner", ref, tgt, fakes, CFG, [0.05], "test", rng)
    a, bh = indigo_halves("scanner")
    keep = np.setdiff1d(np.arange(1, 721), (300, 301))
    a, bh = a[np.isin(a, keep)], bh[np.isin(bh, keep)]

    def h(c):
        return hamming(c.D, c.L)

    tau_a = conformal_threshold(h(ref.of(bh)), 0.05)  # for the half a: originals of b
    tau_b = conformal_threshold(h(ref.of(a)), 0.05)
    o = np.r_[h(tgt.of(a)) > tau_a, h(tgt.of(bh)) > tau_b]
    f = np.r_[h(fakes["f"].of(a)) <= tau_a, h(fakes["f"].of(bh)) <= tau_b]
    fixed = res["fakes"]["f"]["hamming"]["alpha_0.05"]["fixed"]
    assert fixed["p_miss"] == pytest.approx(o.mean())
    assert fixed["p_fa"] == pytest.approx(f.mean())
    # fakes of templates without an original (300, 301) are judged nowhere
    assert res["n_fakes"]["f"] == res["n_originals"] == len(a) + len(bh) == 430
    assert res["n_recal_pool"] == [len(bh), len(a)]


def test_test_stage_recalibrates_each_half_on_the_other_half():
    # the target half b rejects more: a threshold drawn in the other half shows it
    rng = np.random.default_rng(7)
    cb = _codebook(rng)
    a, bh = indigo_halves("scanner")
    ref = _images(rng, cb, np.arange(1, 721))
    ta = _images(rng, cb, np.r_[np.arange(1, 289), a])
    tb = _images(rng, np.clip(cb * 1.3, 0, 0.99), bh)
    tgt = c2f.Images(np.r_[ta.ids, tb.ids], np.r_[ta.D, tb.D], np.r_[ta.L, tb.L])
    fakes = {"f": _images(rng, cb, np.arange(1, 721))}
    res = c2f.transfer("scanner", ref, tgt, fakes, CFG, [0.05], "test", rng)
    rec = res["fakes"]["f"]["hamming"]["alpha_0.05"]["recal"][50]["p_miss"]["mean"]
    assert rec > 0.3  # half b judged with the threshold of half a; within each half: ~4 %


def test_no_template_of_e_or_m_is_judged_or_recalibrated(monkeypatch):
    rng = np.random.default_rng(3)
    ref, tgt, fakes = _setting(rng, 1440)
    seen = []
    real = c2f.Detector.scores

    def spy(self, c):
        seen.append(c.ids)
        return real(self, c)

    monkeypatch.setattr(c2f.Detector, "scores", spy)
    for stage in ("dev", "test"):
        seen.clear()
        c2f.transfer("mobile", ref, tgt, fakes, CFG, [0.05], stage, rng)
        b = indigo_blocks("mobile", stage=stage, available=ref.ids, seed=0)
        used = np.r_[b["E"], b["M"]]
        assert seen and not any(np.isin(ids, used).any() for ids in seen)
        if stage == "dev":
            assert all(ids.max() <= 288 or ids.min() >= 289 for ids in seen)


def test_k_larger_than_the_pool_is_skipped():
    rng = np.random.default_rng(4)
    ref, tgt, fakes = _setting(rng, 720)
    cfg = {**CFG, "recal_K": [20, 300]}
    res = c2f.transfer("scanner", ref, tgt, fakes, cfg, [0.05], "test", rng)
    assert set(res["fakes"]["f"]["hamming"]["alpha_0.05"]["recal"]) == {20}


def test_run_reads_files_at_the_given_stage(monkeypatch):
    rng = np.random.default_rng(5)
    scan_ref, scan_tgt, scan_f = _setting(rng, 720)
    mob_ref, mob_tgt, mob_f = _setting(rng, 1440)
    calls = []

    def fake_load(pack, name, stage, cls):
        calls.append((name, stage, cls))
        if name.startswith("base_fake"):
            return scan_f["f"]
        if name.startswith("variability"):
            return scan_ref
        if name.startswith("base_original"):
            return scan_tgt
        if name.startswith("mobile_fake"):
            return mob_f["f"]
        return mob_ref if "iPhone" in name else mob_tgt

    monkeypatch.setattr(c2f, "load", fake_load)
    cfg = {**CFG, **yaml.safe_load((ROOT / "configs" / "c2a_fakes.yaml").read_text())}
    cfg["recal_draws"] = 5
    res = c2f.run(Path("nowhere"), cfg, "dev", lambda m: None)
    assert {s for _, s, _ in calls} == {"dev"}
    assert set(res["session"]) == {"HPI55", "HPI76"}
    assert set(res["session"]["HPI55"]["fakes"]) == {"f55/55", "f55/76", "f76/55", "f76/76"}
    assert set(res["phone"]) == {"iPhone12Pro_to_SamsungGN20U", "SamsungGN20U_to_iPhone12Pro"}
    assert ("mobile_fake_EHPI55_SamsungGN20U_run1", "dev", "fake") in calls
    assert ("variability_HPI76_session1_print1_scan1", "dev", "original") in calls


def _fail(*args):
    raise AssertionError("no data may be read")


def test_main_refuses_the_test_stage_without_a_freeze_commit(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("data_dir: x\nfreeze_commit: null\n")
    with pytest.raises(SystemExit, match="freeze_commit"):
        c2f.main(["--config", str(cfg), "--stage", "test"], run_fn=_fail)


def test_main_refuses_an_unknown_freeze_commit(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("data_dir: x\nfreeze_commit: 0123456789abcdef0123456789abcdef01234567\n")
    with pytest.raises(SystemExit, match="not a commit"):
        c2f.main(["--config", str(cfg), "--stage", "test"], run_fn=_fail)


def test_main_never_overwrites_a_test_result(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    text = "data_dir: x\nfreeze_commit: HEAD\n"
    cfg.write_text(text)
    monkeypatch.setattr(c2f, "RESULTS", tmp_path)
    c2f.result_path(text, "test").write_text("{}")
    assert c2f.result_path(text, "test").name.endswith("_test.json")
    assert not c2f.result_path(text, "dev").name.endswith("_test.json")
    with pytest.raises(SystemExit, match="runs once"):
        c2f.main(["--config", str(cfg), "--stage", "test"], run_fn=_fail)
