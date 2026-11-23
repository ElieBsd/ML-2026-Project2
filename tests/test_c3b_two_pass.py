import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from cdpbpc.counts import image_counts
from cdpbpc.io import SCANTRUST_COLUMNS
from cdpbpc.patterns import N_PATTERNS, interior, pattern_index
from cdpbpc.scores import reliability_class
from cdpbpc.simulate import simulate_reads
from cdpbpc.two_pass import predicted_flip_prob
from experiments.c3b_two_pass import (
    Condition,
    bootstrap_weights,
    check_indigo_ids,
    check_scantrust_fakes,
    compare,
    judge_prediction,
    per_template,
    weighted_corr_slope,
)

SIDE = 20
# column types of the real manifest (nullable integers and strings, as cdpbpc.io reads it)
DTYPES = {k: v for k, v in SCANTRUST_COLUMNS.items() if k not in ("session", "sheet")}


def _codebook(seed):
    rng = np.random.default_rng(seed)
    cb = rng.uniform(0.0, 0.35, N_PATTERNS)
    cb[rng.choice(N_PATTERNS, 60, replace=False)] = rng.uniform(0.5, 0.95, 60)
    cb[rng.choice(N_PATTERNS, 80, replace=False)] = 0.002
    return cb


def _two_pass_world(seed=0, n=60, n_fakes=1, n_orig=1, error=0.1, side=SIDE):
    """Templates, estimates wrong at a rate `error`, and reads under the two-pass model."""
    rng = np.random.default_rng(seed)
    cb = _codebook(seed + 100)
    t = rng.integers(0, 2, (n, side, side), dtype=np.uint8)
    t_hat = t ^ (rng.random(t.shape) < error).astype(np.uint8)
    o = simulate_reads(t, cb, n_orig, rng)  # (n_orig, n, s, s)
    f = simulate_reads(t_hat, cb, n_fakes, rng)
    Do, Lo = image_counts(np.tile(t, (n_orig, 1, 1)), o.reshape(-1, side, side))
    Df, Lf = image_counts(np.tile(t, (n_fakes, 1, 1)), f.reshape(-1, side, side))
    ids = np.arange(1, n + 1)
    Do, Lo, _ = per_template(np.tile(ids, n_orig), Do, Lo, ids)
    Df, Lf, nf = per_template(np.tile(ids, n_fakes), Df, Lf, ids)
    return Condition(cb, ids, t, t_hat, nf, Df, Lf, Do, Lo), f.reshape(-1, side, side)


def test_by_class_and_masked_values_equal_the_direct_symbol_by_symbol_computation():
    c, fakes = _two_pass_world(1, n=30, n_fakes=2)
    r = compare(c, 0.01, 50, np.random.default_rng(0), 50)
    t2 = np.tile(c.t, (2, 1, 1))
    q = predicted_flip_prob(t2, np.tile(c.t_hat, (2, 1, 1)), c.P)
    flips = interior(t2 != fakes)
    p = c.P[pattern_index(t2)]
    cls = reliability_class(p)
    wrong = interior(t2 != np.tile(c.t_hat, (2, 1, 1)))
    for k, row in enumerate(r["by_class"]):
        m = cls == k
        if not m.any():
            continue
        assert row["share"] == pytest.approx(m.mean())
        assert row["Q_pred"] == pytest.approx(q[m].mean())
        assert row["Q_obs"] == pytest.approx(flips[m].mean())
        assert row["P_b"] == pytest.approx(p[m].mean())
        assert row["e"] == pytest.approx(wrong[m].mean())
        assert row["gap_obs"] == pytest.approx(row["Q_obs"] - row["P_obs"])
    # originals: the same templates, read once each
    rng = np.random.default_rng(11)
    o_reads = simulate_reads(c.t, c.P, 1, rng)[0]
    Do, Lo = image_counts(c.t, o_reads)
    c2 = Condition(c.P, c.ids, c.t, c.t_hat, c.nf, c.Df, c.Lf, Do, Lo)
    r2 = compare(c2, 0.01, 50, np.random.default_rng(0), 50)
    o_flips = interior(c.t != o_reads)
    p1 = c.P[pattern_index(c.t)]
    for k, row in enumerate(r2["by_class"]):
        m = reliability_class(p1) == k
        if m.any():
            assert row["P_obs"] == pytest.approx(o_flips[m].mean())
    # per pattern: plain correlation of the two gaps, computed directly
    idx = pattern_index(t2).ravel()
    n_f = np.bincount(idx, minlength=N_PATTERNS)
    qp = np.bincount(idx, q.ravel(), N_PATTERNS) / np.maximum(n_f, 1)
    qo = np.bincount(idx, flips.ravel(), N_PATTERNS) / np.maximum(n_f, 1)
    i1 = pattern_index(c.t).ravel()
    po = np.bincount(i1, o_flips.ravel(), N_PATTERNS) / np.maximum(
        np.bincount(i1, minlength=512), 1
    )
    keep = n_f > 50
    ref = np.corrcoef((qp - po)[keep], (qo - po)[keep])[0, 1]
    assert r2["per_pattern"]["P_obs"]["corr"] == pytest.approx(ref)
    wref = np.cov((qp - po)[keep], (qo - po)[keep], aweights=n_f[keep])
    assert r2["per_pattern"]["P_obs"]["weighted_corr"] == pytest.approx(
        wref[0, 1] / np.sqrt(wref[0, 0] * wref[1, 1])
    )
    assert r2["per_pattern"]["P_obs"]["weighted_corr"] != pytest.approx(ref, abs=1e-6)
    mk = p <= 0.01
    assert r["masked_hamming"]["fakes_predicted"] == pytest.approx(q[mk].mean())
    assert r["masked_hamming"]["fakes_observed"] == pytest.approx(flips[mk].mean())
    assert r["flip_rate"]["fakes_predicted"] == pytest.approx(q.mean())
    assert r["estimate_error_rate"] == pytest.approx(wrong[: len(c.t)].mean())
    assert r["n_templates"] == 30 and r["n_fakes"] == 60


def test_fakes_of_the_two_pass_model_follow_the_prediction():
    c, _ = _two_pass_world(2, n=150, n_orig=3, side=40)
    r = compare(c, 0.01, 200, np.random.default_rng(1), 200)
    pp = r["per_pattern"]
    assert pp["n_patterns"] > 50
    assert pp["P_obs"]["corr"] > 0.8 and pp["P_b"]["corr"] > 0.8
    assert 0.7 < pp["P_b"]["weighted_slope"] < 1.3
    # resampled templates add noise per pattern: the bootstrap correlations sit a little
    # below the point value, never far from it
    lo, hi = pp["P_obs"]["corr_ci"]
    assert lo <= pp["P_obs"]["corr_bootstrap_median"] <= hi <= pp["P_obs"]["corr"] + 0.01
    assert pp["P_obs"]["corr"] - lo < 0.05
    for row in r["by_class"]:
        if row["share"] > 0.05:
            lo, hi = row["gap_obs_ci"]
            assert lo <= row["gap_obs"] <= hi
            assert abs(row["Q_obs"] - row["Q_pred"]) < 0.03
    mh = r["masked_hamming"]
    assert abs(mh["fakes_observed"] - mh["fakes_predicted"]) < 0.01
    assert mh["fakes_predicted"] > mh["originals"]


def test_fakes_like_originals_give_no_correlation_against_the_codebook():
    # the attacker's estimate is wrong at 10 %, but the "fakes" are prints of the true
    # template: the predicted gap is positive, the observed one is only noise
    rng = np.random.default_rng(3)
    c, _ = _two_pass_world(3, n=150, side=40)
    reads = simulate_reads(c.t, c.P, 1, rng)[0]
    Df, Lf = image_counts(c.t, reads)
    null = Condition(c.P, c.ids, c.t, c.t_hat, c.nf, Df, Lf, c.Do, c.Lo)
    r = compare(null, 0.01, 200, np.random.default_rng(2), 100)
    assert abs(r["per_pattern"]["P_b"]["corr"]) < 0.2


def test_shared_noise_of_p_obs_gives_about_one_over_sqrt_2():
    # t_hat = t and fakes = originals: Q_pred = P_b exactly; against P_obs the two gaps
    # are -noise(P_obs) and noise(Q_obs) - noise(P_obs)
    c, _ = _two_pass_world(4, n=300, error=0.0, side=40)
    r = compare(c, 0.01, 200, np.random.default_rng(2), 50)
    assert r["estimate_error_rate"] == 0
    assert r["per_pattern"]["P_obs"]["corr"] == pytest.approx(1 / np.sqrt(2), abs=0.08)


def test_fake_multiplicity_counts_like_repeated_templates():
    c, _ = _two_pass_world(4, n=20, n_fakes=3)
    r = compare(c, 0.01, 50, np.random.default_rng(0), 20)
    rep = Condition(
        c.P,
        np.arange(60),
        np.tile(c.t, (3, 1, 1)),
        np.tile(c.t_hat, (3, 1, 1)),
        np.ones(60, int),
        np.tile(c.Df / 3, (3, 1)),
        np.tile(c.Lf / 3, (3, 1)),
        np.tile(c.Do, (3, 1)),
        np.tile(c.Lo, (3, 1)),
    )
    r2 = compare(rep, 0.01, 50, np.random.default_rng(0), 20)
    for a, b in zip(r["by_class"], r2["by_class"]):
        for k in ("share", "P_obs", "P_b", "e", "Q_pred", "Q_obs"):
            assert a[k] == pytest.approx(b[k], nan_ok=True)
    assert r["per_pattern"]["P_obs"]["corr"] == pytest.approx(r2["per_pattern"]["P_obs"]["corr"])


def test_compare_refuses_misaligned_counts():
    c, _ = _two_pass_world(5, n=10)
    bad = Condition(c.P, c.ids, c.t, c.t_hat, c.nf, c.Df, c.Lf[::-1], c.Do, c.Lo)
    with pytest.raises(ValueError, match="do not match"):
        compare(bad, 0.01, 50, np.random.default_rng(0), 10)
    bad = Condition(c.P, c.ids, c.t, c.t_hat, c.nf + 1, c.Df, c.Lf, c.Do, c.Lo)
    with pytest.raises(ValueError, match="do not match"):
        compare(bad, 0.01, 50, np.random.default_rng(0), 10)
    with pytest.raises(ValueError, match="no original"):
        compare(
            Condition(c.P, c.ids, c.t, c.t_hat, c.nf, c.Df, c.Lf, c.Do, 0 * c.Lo),
            0.01,
            50,
            np.random.default_rng(0),
            10,
        )


def test_per_pattern_statistics_keep_only_frequent_patterns():
    # occurrences = symbols read in the fakes (a template with 3 fakes counts 3 times)
    c, _ = _two_pass_world(6, n=40, n_fakes=3)
    r = compare(c, 0.01, 60, np.random.default_rng(0), 10)
    n = int((c.Lf.sum(0) > 60).sum())
    assert r["per_pattern"]["n_patterns"] == n > 0
    assert int((c.Lf.sum(0) / 3 > 60).sum()) == 0


def test_weighted_corr_slope_by_hand():
    x = np.array([0.0, 1.0, 2.0, 5.0])
    y = np.array([1.0, 3.0, 5.0, -100.0])
    corr, slope = weighted_corr_slope(x, y, np.array([1.0, 1.0, 1.0, 0.0]))
    assert corr == pytest.approx(1.0) and slope == pytest.approx(2.0)
    w = np.array([1.0, 2.0, 3.0, 4.0])
    corr, slope = weighted_corr_slope(x, -x, w)
    assert corr == pytest.approx(-1.0) and slope == pytest.approx(-1.0)
    xs, ys = np.array([0.0, 1.0, 3.0]), np.array([0.0, 2.0, 1.0])
    ref = np.corrcoef(xs, ys)[0, 1]
    assert weighted_corr_slope(xs, ys, np.ones(3))[0] == pytest.approx(ref)


def test_bootstrap_weights_draw_n_templates_per_replicate():
    W = bootstrap_weights(7, np.random.default_rng(0), 500)
    assert W.shape == (500, 7)
    np.testing.assert_array_equal(W.sum(axis=1), 7)
    assert abs(W.mean() - 1) < 0.05


def test_per_template_sums_images_of_each_template():
    D = np.arange(4 * N_PATTERNS).reshape(4, N_PATTERNS)
    Ds, Ls, n = per_template([7, 3, 7, 3], D, D + 1, np.array([3, 7, 9]))
    np.testing.assert_array_equal(n, [2, 2, 0])
    np.testing.assert_array_equal(Ds[0], D[1] + D[3])
    np.testing.assert_array_equal(Ls[1], D[0] + D[2] + 2)
    assert (Ds[2] == 0).all()
    with pytest.raises(ValueError):
        per_template([5], D[:1], D[:1], np.array([3, 7]))


def test_judge_prediction():
    def classes(gp, go):
        edges = [0, 0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1]
        return [
            {"class": [edges[k], edges[k + 1]], "gap_pred": gp[k], "gap_obs": go[k]}
            for k in range(8)
        ]

    pos = classes([-1] * 5 + [0.1] * 3, [-1] * 5 + [0.05] * 3)
    neg = classes([0.1] * 8, [0.1] * 7 + [-0.01])
    res = {
        "indigo": {"mobile": {"P": {"per_pattern": {"P_obs": {"corr": 0.6}}}}},
        "scantrust": {"batch": {"a": {"by_class": pos}, "b": {"by_class": neg}}},
    }
    pred = {
        "phone": {"device": "P", "min_corr": 0.5},
        "averaging": {"attacks": ["a", "b"], "p_min": 0.5},
    }
    out = judge_prediction(res, pred)
    assert out["phone"]["holds"] and out["batch_a"]["holds"] and not out["batch_b"]["holds"]
    assert len(out["batch_a"]["gap_pred"]) == 3
    res["indigo"]["mobile"]["P"]["per_pattern"]["P_obs"]["corr"] = 0.5
    assert not judge_prediction(res, pred)["phone"]["holds"]


# ---------------------------------------------------------------------------- guards


def test_development_guards():
    check_indigo_ids(np.array([1, 288]), "dev", "fake")
    with pytest.raises(RuntimeError):
        check_indigo_ids(np.array([1, 289]), "dev", "fake")
    check_indigo_ids(np.array([289]), "test", "fake")
    with pytest.raises(ValueError):
        check_indigo_ids(np.array([1]), "all", "fake")
    rows = pd.DataFrame(
        {
            "dataset": ["batch", "batch", "unique", "unique", "unique"],
            "class": ["fake", "original", "fake", "original", "fake"],
            "group": [0.0, 3.0, np.nan, np.nan, np.nan],
            "template_id": [3501, 3531, 3551, 4500, 4000],
        }
    ).astype({k: DTYPES[k] for k in ("dataset", "class", "group", "template_id")})
    check_scantrust_fakes(rows, "dev")
    for i, col, val in [(0, "group", 1.0), (2, "template_id", 4001), (4, "template_id", 100)]:
        bad = rows.copy()
        bad.loc[i, col] = val
        with pytest.raises(RuntimeError):
            check_scantrust_fakes(bad, "dev")
        check_scantrust_fakes(bad, "test")


# ------------------------------------------------- whole runs on synthetic data only


class IndigoPack:
    """Synthetic compact pack: templates 1..1440, mobile and scanner files.

    Honest loaders cut fakes and estimates at 288 in dev, as cdpbpc.io does; a broken
    pack ignores the stage, to test the script's own guard. Every call is recorded.
    """

    def __init__(self, seed=0, broken=False, side=SIDE):
        rng = np.random.default_rng(seed)
        self.cb = _codebook(seed)
        self.broken, self.calls = broken, []
        ids = np.arange(1, 1441)
        self.t = rng.integers(0, 2, (1440, side, side), dtype=np.uint8)
        self.files = {"templates_original": ("template", ids, self.t)}
        for e in ("EHPI55", "EHPI76"):
            est = self.t ^ (rng.random(self.t.shape) < 0.06).astype(np.uint8)
            self.files[f"templates_estimated_mobile_{e}"] = ("estimated", ids, est)
            self.files[f"templates_estimated_base_{e}"] = ("estimated", ids[:720], est[:720])
        for dev in ("iPhone12Pro", "SamsungGN20U"):
            keep = ids != 1000  # one missing original
            self._capture(f"mobile_original_{dev}_run1", "original", ids[keep], self.t[keep], rng)
            est = self.files["templates_estimated_mobile_EHPI55"][2]
            self._capture(f"mobile_fake_EHPI55_{dev}_run1", "fake", ids, est, rng)
        for p in ("HPI55", "HPI76"):
            self._capture(f"base_original_{p}", "original", ids[:720], self.t[:720], rng)
            for e in ("EHPI55", "EHPI76"):
                est = self.files[f"templates_estimated_base_{e}"][2]
                self._capture(f"base_fake_{p}_{e}", "fake", ids[:720], est, rng)

    def _capture(self, name, cls, ids, printed, rng):
        reads = simulate_reads(printed, self.cb, 1, rng)[0]
        D, L = image_counts(self.t[ids - 1], reads)
        self.files[name] = (cls, ids, D, L)

    def _cut(self, cls, ids, stage):
        if stage == "dev" and cls in ("fake", "estimated") and not self.broken:
            return ids <= 288
        return np.ones(len(ids), bool)

    def load_compact(self, path, *, stage):
        name = Path(path).stem
        self.calls.append(("load_compact", name, stage))
        cls, ids, bits = self.files[name]
        k = self._cut(cls, ids, stage)
        return SimpleNamespace(ids=ids[k], bits=bits[k], meta={"class": cls})

    def indigo_counts(self, path, *, stage):
        name = Path(path).stem
        self.calls.append(("indigo_counts", name, stage))
        cls, ids, D, L = self.files[name]
        k = self._cut(cls, ids, stage)
        return SimpleNamespace(ids=ids[k], D=D[k], L=L[k], meta={"class": cls})


class ScantrustWorld:
    """Synthetic Scantrust images: unique E, V_U, T_U and all of batch_cdp."""

    def __init__(self, seed=0, unique_attacks=("otsu",), batch_attacks=("otsu+unsharp",)):
        from cdpbpc.splits import unique_blocks

        rng = np.random.default_rng(seed)
        self.cb = _codebook(seed + 1)
        self.side, self.calls = 12, []
        self.tpl = {}
        rows, tt, reads = [], [], []

        def add(dataset, cls, attack, group, tid, print_id, printed):
            rows.append((dataset, cls, attack, True, group, tid, print_id, f"img{len(rows)}"))
            tt.append(self.template(tid))
            reads.append(simulate_reads(printed[None], self.cb, 1, rng)[0, 0])

        b = unique_blocks(0)
        for tid in np.concatenate([b["E"], b["V"], b["T"]]):
            add("unique", "original", None, np.nan, tid, None, self.template(tid))
            if tid > 3550:
                for a in unique_attacks:  # single-scan attack: reprint of a read original
                    add("unique", "fake", a, np.nan, tid, None, reads[-1])
        for k in range(50):
            tid, g, r = 3501 + k, k // 10, k % 10
            for s in range(5 * g, 5 * g + 5):
                for copy in range(2):
                    add(
                        "batch",
                        "original",
                        None,
                        g,
                        tid,
                        100 * s + 10 * r + copy + 1,
                        self.template(tid),
                    )
            for a in batch_attacks:
                for j in range(3):
                    add("batch", "fake", a, g, tid, j + 1, self.template(tid))
        cols = ["dataset", "class", "attack", "use", "group", "template_id", "print_id", "path"]
        self.rows = pd.DataFrame(rows, columns=cols).astype(DTYPES)
        self.bits = np.stack(reads)
        self.t_all = np.stack(tt)
        self.D, self.L = image_counts(self.t_all, self.bits)

    def template(self, tid):
        if tid not in self.tpl:
            r = np.random.default_rng(int(tid))
            self.tpl[tid] = r.integers(0, 2, (self.side, self.side), dtype=np.uint8)
        return self.tpl[tid]

    def scantrust_counts(self, data_dir, *, stage):
        self.calls.append(("scantrust_counts", stage))
        from cdpbpc.io import scantrust_stage_rows

        keep = scantrust_stage_rows(self.rows, stage=stage).index.to_numpy()
        return SimpleNamespace(rows=self.rows.loc[keep], D=self.D[keep], L=self.L[keep])

    def load_scantrust(self, data_dir, rows, *, stage):
        self.calls.append(("load_scantrust", tuple(sorted(set(rows["class"])))))
        pos = np.array([int(p[3:]) for p in rows["path"]])  # by path, as the real loader
        return SimpleNamespace(rows=rows, t=self.t_all[pos], bits=self.bits[pos])


def _cfg(tmp_path, **extra):
    cfg = {
        "data_dir": str(tmp_path / "no_data_here"),
        "seed": 0,
        "n_boot": 20,
        "mu": 0.01,
        "min_occurrences": 20,
        "indigo": {
            "mobile": {
                "devices": ["iPhone12Pro", "SamsungGN20U"],
                "capture": 1,
                "estimate": "EHPI55",
            },
            "scanner": {"pairs": [["HPI55", "EHPI55"], ["HPI76", "EHPI76"]]},
        },
        "scantrust": {"unique_attacks": ["otsu"], "batch_attacks": ["otsu+unsharp"]},
        "freeze_commit": None,
        "prediction": {
            "phone": {"device": "iPhone12Pro", "min_corr": 0.5},
            "averaging": {"attacks": ["otsu+unsharp"], "p_min": 0.5},
        },
        **extra,
    }
    return cfg


def _patch(monkeypatch, module, pack=None, world=None):
    def refuse(*args, **kwargs):
        raise AssertionError("a test reached the real data")

    for name in ("load_compact", "indigo_counts", "scantrust_counts", "load_scantrust"):
        monkeypatch.setattr(module, name, refuse)
    if pack is not None:
        monkeypatch.setattr(module, "load_compact", pack.load_compact)
        monkeypatch.setattr(module, "indigo_counts", pack.indigo_counts)
    if world is not None:
        monkeypatch.setattr(module, "scantrust_counts", world.scantrust_counts)
        monkeypatch.setattr(module, "load_scantrust", world.load_scantrust)


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_indigo_run_uses_the_blocks_of_the_stage(tmp_path, monkeypatch, stage):
    import experiments.c3b_two_pass as c3b

    pack = IndigoPack(0)
    _patch(monkeypatch, c3b, pack=pack)
    cfg = _cfg(tmp_path)
    out = c3b.run_indigo(tmp_path / "nowhere", cfg, stage, np.random.default_rng(0), print)
    assert {s for _, _, s in pack.calls} == {stage}
    iph = out["mobile"]["iPhone12Pro"]
    lo, hi = iph["info"]["E_templates_range"]
    assert iph["info"]["E_originals"] == 50
    if stage == "dev":
        assert lo >= 289 and iph["n_templates"] == 288
        assert out["scanner"]["HPI55_EHPI55"]["n_templates"] == 288
    else:
        assert hi <= 288 and iph["n_templates"] == 1152 - 1  # template 1000 has no original
        assert iph["info"]["fakes_not_judged"] == [1000]
        assert out["scanner"]["HPI76_EHPI76"]["n_templates"] == 432
    # synthetic fakes follow the two-pass model: the prediction holds
    assert iph["per_pattern"]["P_obs"]["corr"] > 0.5


def test_indigo_development_run_refuses_a_loader_that_leaks_test_fakes(tmp_path, monkeypatch):
    import experiments.c3b_two_pass as c3b

    _patch(monkeypatch, c3b, pack=IndigoPack(1, broken=True))
    with pytest.raises(RuntimeError, match="development stage"):
        c3b.run_indigo(tmp_path / "nowhere", _cfg(tmp_path), "dev", np.random.default_rng(0), print)


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_scantrust_run_uses_the_blocks_of_the_stage(tmp_path, monkeypatch, stage):
    import experiments.c3b_two_pass as c3b
    from cdpbpc.splits import unique_blocks

    world = ScantrustWorld(2)
    _patch(monkeypatch, c3b, world=world)
    out = c3b.run_scantrust(
        tmp_path / "nowhere", _cfg(tmp_path), stage, np.random.default_rng(0), print
    )
    assert ("scantrust_counts", stage) in world.calls
    assert all(c == ("load_scantrust", ("original",)) for c in world.calls[1:])
    u = out["unique"]["otsu"]
    b = out["batch"]["otsu+unsharp"]
    assert u["info"]["E_originals"] == len(unique_blocks(0)["E"])
    assert b["info"]["E_originals"] == 50 and b["n_fakes"] == b["n_templates"] * 3
    assert b["info"]["originals_per_template"] == [10, 10]
    if stage == "dev":
        assert u["n_templates"] == 450 and u["info"]["judged_range"] == [3551, 4000]
        assert b["n_templates"] == 10 and b["info"]["groups_judged"] == [0]
    else:
        assert u["n_templates"] == 1000 and u["info"]["judged_range"] == [4001, 5000]
        assert b["n_templates"] == 40 and b["info"]["groups_judged"] == [1, 2, 3, 4]
    # batch: the estimate is the per-position majority of the originals of each template
    from cdpbpc.two_pass import majority_estimate

    judged = np.arange(3501, 3511) if stage == "dev" else np.arange(3511, 3551)
    rows = world.rows
    errs = []
    for tid in judged:
        sel = ((rows["dataset"] == "batch") & (rows["class"] == "original")).to_numpy()
        sel = sel & (rows["template_id"].to_numpy() == tid)
        est = majority_estimate(world.bits[sel])
        errs.append(interior(est != world.template(tid)).mean())
    assert b["estimate_error_rate"] == pytest.approx(np.mean(errs))
    # unique: the estimate is the binarized original, so e equals the flip rate of originals
    assert u["estimate_error_rate"] == pytest.approx(u["flip_rate"]["originals"])
    assert u["per_pattern"]["P_obs"]["corr"] > 0.5


def test_scantrust_development_run_refuses_test_fakes(tmp_path, monkeypatch):
    import experiments.c3b_two_pass as c3b

    world = ScantrustWorld(3)
    leaky = world.scantrust_counts

    def leak(data_dir, *, stage):
        return leaky(data_dir, stage="test")  # a loader that ignores the stage

    _patch(monkeypatch, c3b, world=world)
    monkeypatch.setattr(c3b, "scantrust_counts", leak)
    with pytest.raises(RuntimeError, match="development stage"):
        c3b.run_scantrust(
            tmp_path / "nowhere", _cfg(tmp_path), "dev", np.random.default_rng(0), print
        )


# ------------------------------------------------------------------------------ main


def _config(tmp_path, **extra):
    path = tmp_path / "c3b.yaml"
    path.write_text(yaml.safe_dump(_cfg(tmp_path, **extra)))
    return path


def _no_data(monkeypatch, module):
    def refuse(*args, **kwargs):
        raise AssertionError("a main() test reached the data")

    for name in ("load_compact", "indigo_counts", "scantrust_counts", "load_scantrust"):
        monkeypatch.setattr(module, name, refuse)
    monkeypatch.setattr(module, "run_all", refuse)


def test_main_refuses_the_test_stage_without_a_freeze_commit(tmp_path, monkeypatch):
    import experiments.c3b_two_pass as c3b

    _no_data(monkeypatch, c3b)
    monkeypatch.setattr(c3b, "RESULTS", tmp_path / "results")
    with pytest.raises(SystemExit, match="needs freeze_commit"):
        c3b.main(["--config", str(_config(tmp_path)), "--stage", "test"])
    with pytest.raises(SystemExit, match="not a commit"):
        c3b.main(["--config", str(_config(tmp_path, freeze_commit="abc")), "--stage", "test"])


def test_the_test_result_is_never_overwritten(tmp_path, monkeypatch):
    import experiments.c3b_two_pass as c3b

    _no_data(monkeypatch, c3b)
    monkeypatch.setattr(c3b, "RESULTS", tmp_path / "results")
    path = _config(tmp_path, freeze_commit="abc")
    text = path.read_text()
    dev, test = c3b.result_path(text, "dev"), c3b.result_path(text, "test")
    assert dev != test and test.name.endswith("_test.json") and dev.name.startswith("c3b_")
    test.parent.mkdir()
    test.write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        c3b.main(["--config", str(path), "--stage", "test"])
    assert test.read_text() == "{}"


def test_main_runs_end_to_end_on_synthetic_data(tmp_path, monkeypatch):
    import experiments.c3b_two_pass as c3b

    _patch(monkeypatch, c3b, pack=IndigoPack(4), world=ScantrustWorld(5))
    monkeypatch.setattr(c3b, "RESULTS", tmp_path / "results")
    path = _config(tmp_path)
    assert c3b.main(["--config", str(path), "--stage", "dev"]) == 0
    out = c3b.result_path(path.read_text(), "dev")
    import json

    res = json.loads(out.read_text())
    assert res["stage"] == "dev" and res["prediction"]["status"].startswith("indication")
    assert set(res["indigo"]["mobile"]) == {"iPhone12Pro", "SamsungGN20U"}
