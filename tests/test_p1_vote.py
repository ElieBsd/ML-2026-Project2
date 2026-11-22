import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from cdpbpc.calibrate import conformal_threshold
from cdpbpc.counts import image_counts
from cdpbpc.metrics import auc
from cdpbpc.patterns import N_PATTERNS
from cdpbpc.simulate import simulate_reads
from cdpbpc.two_pass import vote_estimate
from experiments.p1_vote import (
    SCORES,
    Batch,
    Detector,
    batch_rows,
    check_stage_fakes,
    summarize,
    union,
)


def _rows():
    return pd.DataFrame(
        {
            "dataset": ["batch"] * 5 + ["unique"],
            "class": ["original", "fake", "fake", "fake", "original", "fake"],
            "attack": [None, "otsu", "deep", "otsu+unsharp", None, "otsu"],
            "use": [True, True, False, True, True, True],
            "group": [3.0, 0.0, 0.0, 0.0, 1.0, np.nan],
            "template_id": [3531, 3501, 3501, 3502, 3511, 3600],
        }
    )


def test_development_stage_refuses_test_fakes():
    rows = _rows()
    check_stage_fakes(rows, "dev")  # originals of any group, fakes of group 0
    bad = rows.copy()
    bad.loc[3, "group"] = 2.0
    with pytest.raises(RuntimeError):
        check_stage_fakes(bad, "dev")
    check_stage_fakes(bad, "test")
    with pytest.raises(ValueError):
        check_stage_fakes(rows, "all")


def test_batch_rows_keeps_used_batch_images_of_the_attacks():
    rows = _rows()
    np.testing.assert_array_equal(batch_rows(rows, ["otsu", "otsu+unsharp"]), [0, 1, 3, 4])
    np.testing.assert_array_equal(batch_rows(rows, ["otsu"]), [0, 1, 4])


def _simulated(seed=0, side=24):
    """Originals and fakes of the voting attacker, both read under a known BPC."""
    rng = np.random.default_rng(seed)
    cb = rng.uniform(0.0, 0.3, N_PATTERNS)
    cb[rng.choice(N_PATTERNS, 40, replace=False)] = rng.uniform(0.6, 0.9, 40)
    t = rng.integers(0, 2, (12, side, side), dtype=np.uint8)
    ids = np.arange(len(t))
    tpl = {int(i): t[i] for i in ids}

    def batch(templates, reads, kind):
        n = len(reads)
        tt = np.repeat(templates[None], n, 0).reshape(-1, side, side)
        D, L = image_counts(tt, reads.reshape(-1, side, side))
        return Batch(np.tile(ids, n), np.zeros(n * len(ids), int), [kind] * len(D), D, L)

    E = batch(t, simulate_reads(t, cb, 20, rng), "original")
    O = batch(t, simulate_reads(t, cb, 5, rng), "original")
    S = batch(t, simulate_reads(t, cb, 10, rng), "original")
    F = batch(t, simulate_reads(vote_estimate(t, cb), cb, 5, rng), "fake")
    return tpl, E, S, O, F


def test_lrt_vote_finds_the_fakes_of_the_voting_attacker():
    tpl, E, _, O, F = _simulated()
    det = Detector(E, tpl, mu=0.01)
    assert det.info["patterns_flipped_by_the_vote"] > 0
    so, sf = det.scores(O.D, O.L), det.scores(F.D, F.L)
    assert set(so) == {"hamming", "centered_z_high", "lrt_vote", "masked_hamming"}
    assert auc(so["lrt_vote"], sf["lrt_vote"]) > 0.99
    # under the BPC, centered_z on the patterns P_b >= 0.5 is about N(0, 1) for originals
    assert abs(so["centered_z_high"].mean()) < 0.5


def test_measures_are_per_template():
    tpl, E, S, O, F = _simulated(1)
    det = Detector(E, tpl, mu=0.01)
    fold = {
        "S": det.scores(S.D, S.L),
        "O": det.scores(O.D, O.L),
        "F": det.scores(F.D, F.L),
        "ids_o": O.ids,
        "ids_f": F.ids,
    }
    rng = np.random.default_rng(0)
    out = summarize({0: fold}, [0.05, 0.1], rng, 200)
    r = out["lrt_vote"]["alpha_0.05"]["conformal"]
    assert r["n_templates"] == 12 and len(r["p_fa_per_template"]) == 12
    lo, hi = r["p_fa_ci_template_bootstrap"]
    assert lo <= r["p_fa"] <= hi
    eq = out["hamming"]["alpha_0.1"]["equal_p_miss_test_originals"]
    assert eq["p_miss"] <= 0.1 + 1e-12
    u = union({0: fold}, [0.05], rng, 200)["alpha_0.05"]
    assert u["n_templates"] == 12 and 0 <= u["p_miss"] <= 1


# ----------------------------------------------------------------- test-stage paths


def test_fold_thresholds_follow_their_own_fold():
    # fold 1: 3 templates, its threshold keeps every original; fold 2: 1 template, its
    # threshold rejects every original. Swapped thresholds would give 0.75, not 0.25.
    rng = np.random.default_rng(5)

    def fold(n_templates, cal_shift):
        ids = np.repeat(np.arange(n_templates), 10)
        o = {k: rng.normal(0, 1, len(ids)) for k in SCORES}
        cal = {k: rng.normal(cal_shift, 0.01, 200) for k in o}
        return {"S": cal, "O": o, "F": o, "ids_o": ids, "ids_f": ids}

    folds = {1: fold(3, 50.0), 2: fold(1, -50.0)}
    folds[2]["ids_o"] = folds[2]["ids_f"] = np.full(10, 99)
    out = summarize(folds, [0.05], np.random.default_rng(0), 50)
    assert out["hamming"]["alpha_0.05"]["conformal"]["p_miss"] == pytest.approx(0.25)


def test_union_sets_each_threshold_at_half_the_level():
    rng = np.random.default_rng(6)
    S = {k: rng.normal(0, 1, 1000) for k in ("masked_hamming", "lrt_vote")}
    O = {k: rng.normal(0, 1, 400) for k in S}
    F = {k: rng.normal(1, 1, 400) for k in S}
    ids = np.repeat(np.arange(40), 10)
    fold = {"S": S, "O": O, "F": F, "ids_o": ids, "ids_f": ids}
    got = union({1: fold}, [0.1], np.random.default_rng(0), 50)["alpha_0.1"]
    rej_o = np.zeros(400, bool)
    rej_f = np.zeros(400, bool)
    for k, s in S.items():
        tau = conformal_threshold(s, 0.05)
        rej_o |= O[k] > tau
        rej_f |= F[k] > tau
    assert got["p_miss"] == pytest.approx(rej_o.mean())
    assert got["p_fa"] == pytest.approx((~rej_f).mean())


def _batch_counts(seed=0, side=40, copies=2, n_fakes=3):
    """Synthetic counts of all batch images (groups 0-4, three attacks) and their templates."""
    rng = np.random.default_rng(seed)
    cb = np.full(N_PATTERNS, 0.001)
    cb[rng.choice(N_PATTERNS, 200, replace=False)] = rng.uniform(0.05, 0.3, 200)
    cb[rng.choice(N_PATTERNS, 30, replace=False)] = rng.uniform(0.6, 0.9, 30)
    attacks = ["otsu", "otsu+unsharp", "deep+unsharp"]
    rows, tt, reads, tpl = [], [], [], {}
    for k in range(50):
        tid, g, r = 3501 + k, k // 10, k % 10
        t = rng.integers(0, 2, (1, side, side), dtype=np.uint8)
        tpl[tid] = t[0]
        orig = simulate_reads(t, cb, 5 * copies, rng)[:, 0]
        for j, (s, c) in enumerate((s, c) for s in range(5 * g, 5 * g + 5) for c in range(copies)):
            rows.append(("original", None, g, tid, 100 * s + 10 * r + c + 1))
            tt.append(t[0])
            reads.append(orig[j])
        for a in attacks:
            for j, read in enumerate(simulate_reads(vote_estimate(t, cb), cb, n_fakes, rng)[:, 0]):
                rows.append(("fake", a, g, tid, j + 1))
                tt.append(t[0])
                reads.append(read)
    D, L = image_counts(np.stack(tt), np.stack(reads))
    df = pd.DataFrame(rows, columns=["class", "attack", "group", "template_id", "print_id"])
    df["dataset"], df["use"] = "batch", True
    df["group"] = df["group"].astype(float)
    return SimpleNamespace(rows=df, D=D, L=L), tpl, attacks


@pytest.mark.parametrize("with_union", [True, False])
def test_test_stage_runs_four_folds_with_group_0_as_E(monkeypatch, with_union):
    import experiments.p1_vote as p1

    counts, tpl, attacks = _batch_counts()
    monkeypatch.setattr(p1, "scantrust_counts", lambda data_dir, stage: counts)
    monkeypatch.setattr(p1, "read_templates", lambda data_dir, ids: tpl)
    seen_E = []

    class Spy(p1.Detector):
        def __init__(self, E, templates, mu):
            seen_E.append(set(E.ids.tolist()))
            assert len(E) == 50 and set(np.unique(E.prints) % 10) == {1}  # copy 0 only
            super().__init__(E, templates, mu)

    monkeypatch.setattr(p1, "Detector", Spy)
    cfg = {
        "attacks": attacks,
        "alphas": [0.05, 0.1],
        "mu": 0.01,
        "n_boot": 50,
        "decision": {"attack": "otsu+unsharp", "min_gain": 0.05},
        "union": with_union,
    }
    out = p1.run(Path("unused"), cfg, "test", np.random.default_rng(0), lambda m: None)
    assert sorted(out["folds"]) == [1, 2, 3, 4]
    assert seen_E == [set(range(3501, 3511))] * 4
    for info in out["folds"].values():
        assert info["E_templates"] == 10
        assert info["n_S"] == 20 * 10 and info["n_T_originals"] == 10 * 10
        assert all(info[f"n_fakes_{a}"] == 10 * 3 for a in attacks)
    r = out["attacks"]["otsu+unsharp"]["lrt_vote"]["alpha_0.05"]["conformal"]
    assert r["n_templates"] == 40
    assert out["attacks"]["otsu+unsharp"]["lrt_vote"]["auc"] > 0.95
    assert "decision" not in out
    assert ("union" in out) == with_union


def _config(tmp_path, **extra):
    cfg = {"data_dir": "dataset", "seed": 0, "freeze_commit": None, "union": None, **extra}
    path = tmp_path / "p1.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _no_data(monkeypatch, module):
    def refuse(*args, **kwargs):
        raise AssertionError("a main() test reached the data")

    monkeypatch.setattr(module, "scantrust_counts", refuse)
    monkeypatch.setattr(module, "run", refuse)


def test_main_refuses_the_test_stage_without_freeze_or_union_choice(tmp_path, monkeypatch):
    import experiments.p1_vote as p1

    _no_data(monkeypatch, p1)
    with pytest.raises(SystemExit, match="not a commit"):
        p1.main(
            ["--config", str(_config(tmp_path, freeze_commit="abc", union=True)), "--stage", "test"]
        )
    with pytest.raises(SystemExit, match="freeze_commit"):
        p1.main(["--config", str(_config(tmp_path)), "--stage", "test"])
    with pytest.raises(SystemExit, match="union"):
        p1.main(["--config", str(_config(tmp_path, freeze_commit="abc")), "--stage", "test"])


def test_the_test_result_is_never_overwritten(tmp_path, monkeypatch):
    import experiments.p1_vote as p1

    _no_data(monkeypatch, p1)
    monkeypatch.setattr(p1, "RESULTS", tmp_path / "results")
    path = _config(tmp_path, freeze_commit="abc", union=True)
    text = path.read_text()
    dev, test = p1.result_path(text, "dev"), p1.result_path(text, "test")
    assert dev != test and test.name.endswith("_test.json")
    test.parent.mkdir()
    test.write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        p1.main(["--config", str(path), "--stage", "test"])
    assert test.read_text() == "{}"
