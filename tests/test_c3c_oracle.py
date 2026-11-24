import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from cdpbpc.baselines.supervised import fit_logistic, logistic_score
from cdpbpc.calibrate import conformal_threshold
from cdpbpc.io import SCANTRUST_COLUMNS
from cdpbpc.metrics import clopper_pearson
from cdpbpc.patterns import N_PATTERNS
from cdpbpc.scores import binned_z
from experiments.c3c_oracle import (
    Cell,
    Condition,
    Images,
    check_part,
    judge_prediction,
    measure,
    run_cell,
)

# column types of the real manifest (nullable integers and strings, as cdpbpc.io reads it)
DTYPES = {k: v for k, v in SCANTRUST_COLUMNS.items() if k not in ("session", "sheet")}
SYMBOLS = 600  # interior symbols of a synthetic image


def _codebook(seed):
    r = np.random.default_rng(seed)
    p = r.beta(0.4, 1.2, N_PATTERNS)
    p[:80] = r.uniform(0.0, 0.01, 80)  # reliable patterns, as in real codebooks
    return p


def _fake_q(P, kind):
    """Flip probabilities of a synthetic fake type."""
    if kind == "low":  # more errors on the reliable patterns
        return np.clip(P + 0.04 * (P < 0.15), 0, 1)
    if kind == "high":  # more errors where P_b >= 0.5, as averaging does
        return np.clip(P + 0.15 * (P >= 0.5), 0, 1)
    if kind == "mid":
        return np.clip(P + 0.06 * ((P >= 0.15) & (P < 0.5)), 0, 1)
    raise ValueError(kind)


def _images(n, q, rng):
    L = rng.multinomial(SYMBOLS, np.full(N_PATTERNS, 1 / N_PATTERNS), size=n)
    return L, rng.binomial(L, q)


def _condition(name, ids, P, fake_kinds, rng, parts, unit="image", n_fakes=1, alphas=(0.05,)):
    L, D = _images(len(ids), P, rng)
    fakes = {}
    for t, kind in fake_kinds.items():
        fids = np.repeat(ids, n_fakes)
        Lf, Df = _images(len(fids), _fake_q(P, kind), rng)
        fakes[t] = Images(fids, Df, Lf)
    return Condition(
        name=name,
        unit=unit,
        P=P,
        originals=Images(ids, D, L),
        fakes=fakes,
        parts=parts,
        alphas=list(alphas),
        no_train=np.zeros(len(ids), bool),
    )


CFG = {"seed": 0, "n_boot": 50, "mu": 0.01, "Cs": [0.01, 1.0, 100.0], "n_splits": 3}


def _two_halves(ids_a, ids_b, thr):
    return [
        {"judged": ids_a, "threshold": thr, "train": ids_b},
        {"judged": ids_b, "threshold": thr, "train": ids_a},
    ]


# ---------------------------------------------------------------------------- measures


def test_measure_one_image_per_template_uses_the_threshold_originals_and_clopper_pearson():
    s = np.arange(100, dtype=float)  # conformal threshold at 5 %: 96th smallest = 95
    o = np.array([10.0, 96.0, 50.0, 99.0])  # two above 95
    f = np.array([94.0, 95.0, 96.0, 200.0, 300.0])  # two accepted (<= 95)
    part = {"o": o, "s": s, "f": f, "ids_o": np.arange(4), "ids_f": np.arange(5)}
    r = measure([part], "image", [0.05], np.random.default_rng(0), 10)
    c = r["alpha_0.05"]["conformal"]
    assert c["tau"] == [95.0] == [conformal_threshold(s, 0.05)]
    assert c["p_fa"] == pytest.approx(2 / 5) and c["p_miss"] == pytest.approx(2 / 4)
    assert c["p_fa_ci"] == pytest.approx(list(clopper_pearson(2, 5)))
    assert c["p_miss_ci"] == pytest.approx(list(clopper_pearson(2, 4)))
    eq = r["alpha_0.05"]["equal_p_miss_test_originals"]
    assert eq["tau"] == [pytest.approx(np.quantile(o, 0.95))]  # from the judged originals
    assert r["auc"] == pytest.approx((2 + 2 + 2.5 + 4 + 4) / 20)  # 96 ties one original


def test_measure_batch_rates_are_per_template():
    # template 1: 1 fake of 1 accepted; template 2: 0 of 3; image mean 1/4, template mean 1/2
    part = {
        "o": np.zeros(4),
        "s": np.zeros(50),
        "f": np.array([-1.0, 5.0, 5.0, 5.0]),
        "ids_o": np.array([1, 1, 2, 2]),
        "ids_f": np.array([1, 2, 2, 2]),
    }
    r = measure([part], "template", [0.05], np.random.default_rng(0), 200)
    c = r["alpha_0.05"]["conformal"]
    assert c["p_fa"] == pytest.approx(0.5) and c["p_fa_per_template"] == [1.0, 0.0]
    assert c["n_templates"] == 2 and "deff_fakes" in c
    assert "p_fa_ci" in c and c["p_fa_ci"][0] <= 0.5 <= c["p_fa_ci"][1]
    assert "p_fa_ci_clopper_pearson" not in c  # never on batch images


def test_measure_uses_the_threshold_of_each_part():
    part_a = {
        "o": np.array([1.0]),
        "s": np.zeros(30),
        "f": np.array([0.5]),
        "ids_o": [1],
        "ids_f": [1],
    }
    part_b = {
        "o": np.array([1.0]),
        "s": np.full(30, 2.0),
        "f": np.array([0.5]),
        "ids_o": [2],
        "ids_f": [2],
    }
    c = measure([part_a, part_b], "image", [0.05], np.random.default_rng(0), 10)["alpha_0.05"][
        "conformal"
    ]
    assert c["tau"] == [0.0, 2.0] and c["p_miss"] == 0.5 and c["p_fa"] == 0.5


def test_check_part_refuses_shared_templates():
    check_part(np.array([1, 2]), np.array([3, 4]), np.array([5, 6]))
    with pytest.raises(ValueError, match="trained on and judged"):
        check_part(np.array([1, 2]), np.array([2, 4]), np.array([5]))
    with pytest.raises(ValueError, match="threshold"):
        check_part(np.array([1, 2]), np.array([3]), np.array([1, 5]))
    with pytest.raises(ValueError, match="threshold and judged"):
        check_part(np.array([1]), np.array([3]), np.array([3, 5]))


# ------------------------------------------------------------------------------- cells


def test_logistic_trained_on_simulated_fakes_separates_new_ones():
    rng = np.random.default_rng(1)
    ids = np.arange(1, 401)
    parts = _two_halves(np.arange(1, 101), np.arange(101, 201), np.arange(201, 401))
    cond = _condition("A", ids, _codebook(1), {"low": "low", "high": "high"}, rng, parts)
    memo = {}
    res = run_cell(
        Cell(cond, cond, "low", ["low"], {"low": "same", "high": "same"}),
        CFG,
        memo,
        np.random.default_rng(0),
    )
    seen = res["logistic"]["low"]
    assert seen["auc"] > 0.99 and seen["alpha_0.05"]["conformal"]["p_fa"] < 0.05
    # one model per part, each trained on the other half only
    assert len(res["models"]) == 2 and len(memo) == 2
    assert res["models"][0]["train_templates"] == [101, 200]
    assert res["models"][0]["n_originals"] == 100 and res["models"][0]["n_fakes"] == 100
    assert res["lrt_plugin"]["low"]["auc"] > 0.99


def test_all_types_row_trains_on_every_type():
    rng = np.random.default_rng(2)
    ids = np.arange(1, 301)
    parts = _two_halves(np.arange(1, 101), np.arange(101, 201), np.arange(201, 301))
    cond = _condition("A", ids, _codebook(2), {"x": "low", "y": "mid", "z": "high"}, rng, parts)
    rel = {t: "same" for t in cond.fakes}
    res = run_cell(Cell(cond, cond, "all", ["x", "y", "z"], rel), CFG, {}, np.random.default_rng(0))
    assert res["models"][0]["n_fakes"] == 300 and res["models"][0]["n_originals"] == 100
    assert set(res["logistic"]) == {"x", "y", "z"} == set(res["relation"])


def test_features_use_the_codebook_of_their_own_condition():
    """Training images with the codebook of the training condition, judged images with that
    of the judged condition; the threshold from originals of the judged condition."""
    rng = np.random.default_rng(3)
    ids = np.arange(1, 301)
    parts = [
        {
            "judged": np.arange(1, 101),
            "threshold": np.arange(201, 301),
            "train": np.arange(101, 201),
        }
    ]
    PA, PB = _codebook(3), np.clip(_codebook(3) * 1.5 + 0.01, 0, 0.99)
    A = _condition("A", ids, PA, {"f": "low"}, rng, parts)
    B = _condition("B", ids, PB, {"f": "low"}, rng, parts)
    memo = {}
    res = run_cell(
        Cell(B, A, "from A", ["f"], {"f": "other_phone"}), CFG, memo, np.random.default_rng(0)
    )
    # the model by hand: A's images of templates 101..200 with A's codebook
    tr = np.isin(A.originals.ids, np.arange(101, 201))
    trf = np.isin(A.fakes["f"].ids, np.arange(101, 201))
    Z = np.r_[
        binned_z(A.originals.D[tr], A.originals.L[tr], PA),
        binned_z(A.fakes["f"].D[trf], A.fakes["f"].L[trf], PA),
    ]
    y = np.r_[np.zeros(tr.sum()), np.ones(trf.sum())]
    g = np.r_[A.originals.ids[tr], A.fakes["f"].ids[trf]]
    model, C, _ = fit_logistic(Z, y, g, CFG["Cs"], CFG["n_splits"], CFG["seed"])
    (fit,) = memo.values()
    assert fit["C"] == C
    np.testing.assert_allclose(fit["model"][-1].coef_, model[-1].coef_)
    # judged 1..100 of B with B's codebook, threshold on B's originals 201..300
    o = np.isin(B.originals.ids, np.arange(1, 101))
    s = np.isin(B.originals.ids, np.arange(201, 301))
    f = np.isin(B.fakes["f"].ids, np.arange(1, 101))

    def sc(m):
        return logistic_score(model, binned_z(B.originals.D[m], B.originals.L[m], PB))

    tau = conformal_threshold(sc(s), 0.05)
    fs = logistic_score(model, binned_z(B.fakes["f"].D[f], B.fakes["f"].L[f], PB))
    c = res["logistic"]["f"]["alpha_0.05"]["conformal"]
    assert c["tau"][0] == pytest.approx(tau)
    assert c["p_fa"] == pytest.approx(np.mean(fs <= tau))
    assert c["p_miss"] == pytest.approx(np.mean(sc(o) > tau))


def test_training_never_uses_images_of_block_e():
    rng = np.random.default_rng(4)
    ids = np.arange(1, 301)
    parts = [
        {
            "judged": np.arange(101, 201),
            "threshold": np.arange(201, 301),
            "train": np.arange(1, 101),
        }
    ]
    cond = _condition("A", ids, _codebook(4), {"f": "low"}, rng, parts)
    cond.no_train[:20] = True  # originals of templates 1..20 are block E
    res = run_cell(Cell(cond, cond, "f", ["f"], {"f": "same"}), CFG, {}, np.random.default_rng(0))
    assert res["models"][0]["n_originals"] == 80 and res["models"][0]["n_fakes"] == 100


def test_run_cell_refuses_overlapping_blocks():
    rng = np.random.default_rng(5)
    ids = np.arange(1, 301)
    bad = [
        {"judged": np.arange(1, 101), "threshold": np.arange(201, 301), "train": np.arange(90, 201)}
    ]
    cond = _condition("A", ids, _codebook(5), {"f": "low"}, rng, bad)
    with pytest.raises(ValueError, match="trained on and judged"):
        run_cell(Cell(cond, cond, "f", ["f"], {"f": "same"}), CFG, {}, np.random.default_rng(0))


# -------------------------------------------------------------------------- prediction


def _res(rows):
    """Results of one judged condition: rows = {row: (relation, logistic p_fa)}; fake-free
    p_fa: hamming 0.4, masked_hamming 0.1 on the test type t."""

    def m(p):
        return {"alpha_0.05": {"conformal": {"p_fa": p}}}

    return {
        "fake_free": {"hamming": {"t": m(0.4)}, "masked_hamming": {"t": m(0.1)}},
        "oracle": {
            r: {"relation": {"t": rel}, "logistic": {"t": m(p)}} for r, (rel, p) in rows.items()
        },
    }


PRED = {
    "alpha": 0.05,
    "fake_free": ["hamming", "masked_hamming"],
    "other_cases": ["other_session", "other_printer", "other_phone"],
}


def test_judge_prediction_by_hand():
    good = {
        "cond1": _res({"a": ("same", 0.1), "b": ("same", 0.0), "c": ("other_phone", 0.3)}),
        "cond2": _res({"a": ("other_printer", 0.05), "b": ("other_printer", 0.2)}),
        "cond3": _res({"a": ("other_session", 0.5)}),
    }
    out = judge_prediction(good, PRED)
    assert out["same_condition"]["holds"] and out["other_condition"]["holds"]
    assert out["other_condition"]["cases"]["other_printer"]["worse_cells"] == 1
    assert out["holds"]
    # one same-condition cell worse than masked_hamming: the first part falls
    bad = dict(good, cond1=_res({"a": ("same", 0.11), "c": ("other_phone", 0.3)}))
    out = judge_prediction(bad, PRED)
    assert not out["same_condition"]["holds"] and out["other_condition"]["holds"]
    assert not out["holds"]
    # one case where the logistic is never worse: the second part falls
    bad = dict(good, cond3=_res({"a": ("other_session", 0.1)}))
    out = judge_prediction(bad, PRED)
    assert out["same_condition"]["holds"] and not out["other_condition"]["holds"]
    # a case without any cell cannot hold
    bad = {k: v for k, v in good.items() if k != "cond3"}
    assert not judge_prediction(bad, PRED)["other_condition"]["holds"]


# ------------------------------------------------- whole runs on synthetic data only


class IndigoPack:
    """Synthetic pack of counts: templates 1..1440, mobile and scanner files.

    Honest loaders cut fakes at 288 in dev, as cdpbpc.io does; a broken pack ignores
    the stage, to test the script's own guard. Every call is recorded.
    """

    def __init__(self, seed=0, broken=False):
        rng = np.random.default_rng(seed)
        self.broken, self.calls, self.files = broken, [], {}
        ids = np.arange(1, 1441)
        for j, dev in enumerate(("iPhone12Pro", "SamsungGN20U")):
            P = _codebook(seed + j)
            keep = ids != 1000  # one missing original
            self._add(f"mobile_original_{dev}_run1", "original", ids[keep], P, rng)
            self._add(f"mobile_fake_EHPI55_{dev}_run1", "fake", ids, _fake_q(P, "low"), rng)
        kinds = {"EHPI55": "low", "EHPI76": "mid"}
        for j, p in enumerate(("HPI55", "HPI76")):
            P = _codebook(seed + 10 + j)
            self._add(f"base_original_{p}", "original", ids[:720], P, rng)
            for e, kind in kinds.items():
                q = _fake_q(P, kind if p == "HPI55" else "high")
                self._add(f"base_fake_{p}_{e}", "fake", ids[:720], q, rng)

    def _add(self, name, cls, ids, q, rng):
        L, D = _images(len(ids), q, rng)
        self.files[name] = (cls, ids, D, L)

    def indigo_counts(self, path, *, stage):
        name = Path(path).stem
        self.calls.append(("indigo_counts", name, stage))
        cls, ids, D, L = self.files[name]
        k = np.ones(len(ids), bool)
        if stage == "dev" and cls == "fake" and not self.broken:
            k = ids <= 288
        return SimpleNamespace(ids=ids[k], D=D[k], L=L[k], meta={"class": cls})


class ScantrustWorld:
    """Synthetic Scantrust counts: unique E, V_U, T_U and all of batch_cdp."""

    def __init__(
        self, seed=0, unique_attacks=("otsu", "deep"), batch_attacks=("otsu", "otsu+unsharp")
    ):
        from cdpbpc.splits import unique_blocks

        rng = np.random.default_rng(seed)
        Pu, Pb = _codebook(seed + 20), _codebook(seed + 21)
        self.calls = []
        rows, Ds, Ls = [], [], []

        def add(dataset, cls, attack, group, tid, print_id, q):
            L, D = _images(1, q, rng)
            rows.append((dataset, cls, attack, True, group, tid, print_id, f"img{len(rows)}"))
            Ds.append(D[0])
            Ls.append(L[0])

        b = unique_blocks(0)
        for tid in np.concatenate([b["E"], b["V"], b["T"]]):
            add("unique", "original", None, np.nan, tid, None, Pu)
            if tid > 3550:
                for a, kind in zip(unique_attacks, ("low", "mid")):
                    add("unique", "fake", a, np.nan, tid, None, _fake_q(Pu, kind))
        for k in range(50):
            tid, g, r = 3501 + k, k // 10, k % 10
            for s in range(5 * g, 5 * g + 5):
                for copy in range(2):
                    add("batch", "original", None, g, tid, 100 * s + 10 * r + copy + 1, Pb)
            for a, kind in zip(batch_attacks, ("low", "high")):
                for j in range(3):
                    add("batch", "fake", a, g, tid, j + 1, _fake_q(Pb, kind))
        cols = ["dataset", "class", "attack", "use", "group", "template_id", "print_id", "path"]
        self.rows = pd.DataFrame(rows, columns=cols).astype(DTYPES)
        self.D, self.L = np.stack(Ds), np.stack(Ls)

    def scantrust_counts(self, data_dir, *, stage):
        from cdpbpc.io import scantrust_stage_rows

        self.calls.append(("scantrust_counts", stage))
        keep = scantrust_stage_rows(self.rows, stage=stage).index.to_numpy()
        return SimpleNamespace(rows=self.rows.loc[keep], D=self.D[keep], L=self.L[keep])


def _cfg(tmp_path, **extra):
    return {
        "data_dir": str(tmp_path / "no_data_here"),
        "seed": 0,
        "n_boot": 20,
        "mu": 0.01,
        "Cs": [0.01, 1.0],
        "n_splits": 3,
        "indigo": {
            "mobile": {
                "devices": ["iPhone12Pro", "SamsungGN20U"],
                "capture": 1,
                "estimate": "EHPI55",
                "alphas": [0.01, 0.05],
            },
            "scanner": {
                "printers": ["HPI55", "HPI76"],
                "fakes": [["HPI55", "EHPI55"], ["HPI55", "EHPI76"], ["HPI76", "EHPI55"]],
                "alphas": [0.05],
            },
        },
        "scantrust": {
            "batch_attacks": ["otsu", "otsu+unsharp"],
            "unique_attacks": ["otsu", "deep"],
            "alphas": [0.05, 0.1],
        },
        "freeze_commit": None,
        "prediction": PRED,
        **extra,
    }


def _patch(monkeypatch, module, pack=None, world=None):
    def refuse(*args, **kwargs):
        raise AssertionError("a test reached the real data")

    for name in ("indigo_counts", "scantrust_counts"):
        monkeypatch.setattr(module, name, refuse)
    if pack is not None:
        monkeypatch.setattr(module, "indigo_counts", pack.indigo_counts)
    if world is not None:
        monkeypatch.setattr(module, "scantrust_counts", world.scantrust_counts)


def _ranges(res):
    return [
        (p["train_templates"], p["judged_templates"], p["threshold_templates"])
        for p in res["parts"]
    ]


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_indigo_run_uses_the_blocks_of_the_stage(tmp_path, monkeypatch, stage):
    import experiments.c3c_oracle as c3c

    pack = IndigoPack(0)
    _patch(monkeypatch, c3c, pack=pack)
    conds, cells = c3c.indigo_conditions(tmp_path / "nowhere", _cfg(tmp_path), stage)
    assert {s for _, _, s in pack.calls} == {stage}
    res = c3c.run_cells(conds, cells, _cfg(tmp_path), np.random.default_rng(0), print)
    iph = res["mobile iPhone12Pro"]
    scan = res["scanner HPI55"]
    # rows: own phone and the other phone; scanner: each fake type and all of them
    assert set(iph["oracle"]) == {"trained on iPhone12Pro", "trained on SamsungGN20U"}
    assert iph["oracle"]["trained on SamsungGN20U"]["relation"] == {"EHPI55": "other_phone"}
    types = ["HPI55_EHPI55", "HPI55_EHPI76", "HPI76_EHPI55"]
    assert set(scan["oracle"]) == set(types) | {"all"}
    assert scan["oracle"]["HPI55_EHPI55"]["relation"] == {
        "HPI55_EHPI55": "same",
        "HPI55_EHPI76": "same",
        "HPI76_EHPI55": "other_printer",
    }
    assert set(scan["oracle"]["all"]["relation"].values()) == {"same"}
    assert set(scan["fake_free"]) == {"hamming", "masked_hamming"}
    E = iph["info"]["E_templates"]
    ph = iph["oracle"]["trained on iPhone12Pro"]
    fakes = ph["logistic"]["EHPI55"]
    if stage == "dev":
        assert E[0] >= 289
        assert [r[:2] for r in _ranges(ph)] == [([145, 288], [1, 144]), ([1, 144], [145, 288])]
        assert all(r[2][0] >= 289 for r in _ranges(ph))  # threshold: block S
        assert fakes["n_fakes"] == 288 and fakes["n_originals"] == 288
        # scanner halves: odd and even ranks of 1..288
        sp = scan["oracle"]["all"]["parts"][0]
        assert sp["judged_templates"] == [1, 287] and sp["n_judged"] == 144
    else:
        assert E[1] <= 288
        tr = ph["models"][0]
        assert 1 <= tr["train_templates"][0] and tr["train_templates"][1] <= 288
        assert tr["n_templates"] == 288 - 50  # development templates without block E
        assert _ranges(ph)[0][1:] == ([289, 864], [865, 1440])
        assert fakes["n_fakes"] == 1152 - 1 and fakes["n_originals"] == 1152 - 1
        assert iph["info"]["fakes_not_judged"] == [1000]
    assert ph["logistic"]["EHPI55"]["auc"] > 0.95


def test_indigo_development_run_refuses_a_loader_that_leaks_test_fakes(tmp_path, monkeypatch):
    import experiments.c3c_oracle as c3c

    _patch(monkeypatch, c3c, pack=IndigoPack(1, broken=True))
    with pytest.raises(RuntimeError, match="development stage"):
        c3c.indigo_conditions(tmp_path / "nowhere", _cfg(tmp_path), "dev")


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_scantrust_run_uses_the_blocks_of_the_stage(tmp_path, monkeypatch, stage):
    import experiments.c3c_oracle as c3c

    world = ScantrustWorld(2)
    _patch(monkeypatch, c3c, world=world)
    cfg = _cfg(tmp_path)
    conds, cells = c3c.scantrust_conditions(tmp_path / "nowhere", cfg, stage)
    assert world.calls == [("scantrust_counts", stage)]
    res = c3c.run_cells(conds, cells, cfg, np.random.default_rng(0), print)["batch"]
    assert set(res["oracle"]) == {"otsu", "otsu+unsharp", "all", "unique V_U (all attacks)"}
    vu = res["oracle"]["unique V_U (all attacks)"]
    assert set(vu["relation"].values()) == {"other_session"}
    assert vu["models"][0]["train_templates"] == [3551, 4000]
    assert vu["models"][0]["n_fakes"] == 2 * 450 and vu["models"][0]["n_templates"] == 450
    m = res["oracle"]["otsu"]
    c = m["logistic"]["otsu"]["alpha_0.05"]["conformal"]
    assert "p_fa_per_template" in c and "deff_fakes" in c
    if stage == "dev":
        assert _ranges(m) == [
            ([3506, 3510], [3501, 3505], [3531, 3550]),
            ([3501, 3505], [3506, 3510], [3531, 3550]),
        ]
        assert m["models"][0]["n_originals"] == 50 and m["models"][0]["n_fakes"] == 15
        assert c["n_templates"] == 10
        assert res["info"]["E_print_groups"] == [1]
    else:
        # one model trained on group 0 without the 50 images of block E, four folds
        assert len({tuple(p["train_templates"]) for p in m["parts"]}) == 1
        assert m["models"][0]["train_templates"] == [3501, 3510]
        assert m["models"][0]["n_originals"] == 100 - 50 and m["models"][0]["n_fakes"] == 30
        assert [p["judged_templates"] for p in m["parts"]] == [
            [3511, 3520],
            [3521, 3530],
            [3531, 3540],
            [3541, 3550],
        ]
        assert c["n_templates"] == 40
        assert res["info"]["E_print_groups"] == [0]


def test_scantrust_development_run_refuses_test_fakes(tmp_path, monkeypatch):
    import experiments.c3c_oracle as c3c

    world = ScantrustWorld(3)
    leaky = world.scantrust_counts

    def leak(data_dir, *, stage):
        return leaky(data_dir, stage="test")  # a loader that ignores the stage

    _patch(monkeypatch, c3c, world=world)
    monkeypatch.setattr(c3c, "scantrust_counts", leak)
    with pytest.raises(RuntimeError, match="development stage"):
        c3c.scantrust_conditions(tmp_path / "nowhere", _cfg(tmp_path), "dev")


# ------------------------------------------------------------------------------ main


def _config(tmp_path, **extra):
    path = tmp_path / "c3c.yaml"
    path.write_text(yaml.safe_dump(_cfg(tmp_path, **extra)))
    return path


def _no_data(monkeypatch, module):
    def refuse(*args, **kwargs):
        raise AssertionError("a main() test reached the data")

    for name in ("indigo_counts", "scantrust_counts", "run_all"):
        monkeypatch.setattr(module, name, refuse)


def test_main_refuses_the_test_stage_without_a_freeze_commit(tmp_path, monkeypatch):
    import experiments.c3c_oracle as c3c

    _no_data(monkeypatch, c3c)
    monkeypatch.setattr(c3c, "RESULTS", tmp_path / "results")
    with pytest.raises(SystemExit, match="needs freeze_commit"):
        c3c.main(["--config", str(_config(tmp_path)), "--stage", "test"])
    with pytest.raises(SystemExit, match="not a commit"):
        c3c.main(["--config", str(_config(tmp_path, freeze_commit="abc")), "--stage", "test"])
    assert not (tmp_path / "results").exists()


def test_the_test_result_is_never_overwritten(tmp_path, monkeypatch):
    import experiments.c3c_oracle as c3c

    _no_data(monkeypatch, c3c)
    monkeypatch.setattr(c3c, "RESULTS", tmp_path / "results")
    path = _config(tmp_path, freeze_commit="abc")
    text = path.read_text()
    dev, test = c3c.result_path(text, "dev"), c3c.result_path(text, "test")
    assert dev != test and test.name.endswith("_test.json") and dev.name.startswith("c3c_")
    test.parent.mkdir()
    test.write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        c3c.main(["--config", str(path), "--stage", "test"])
    assert test.read_text() == "{}"


def test_main_runs_end_to_end_on_synthetic_data(tmp_path, monkeypatch):
    import experiments.c3c_oracle as c3c

    _patch(monkeypatch, c3c, pack=IndigoPack(4), world=ScantrustWorld(5))
    monkeypatch.setattr(c3c, "RESULTS", tmp_path / "results")
    path = _config(tmp_path)
    assert c3c.main(["--config", str(path), "--stage", "dev"]) == 0
    res = json.loads(c3c.result_path(path.read_text(), "dev").read_text())
    assert res["stage"] == "dev" and res["prediction"]["status"].startswith("indication")
    assert set(res["conditions"]) == {
        "mobile iPhone12Pro",
        "mobile SamsungGN20U",
        "scanner HPI55",
        "scanner HPI76",
        "batch",
    }
    assert "same_condition" in res["prediction"] and "other_condition" in res["prediction"]
