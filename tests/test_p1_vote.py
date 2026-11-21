import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the experiments package

from cdpbpc.counts import image_counts
from cdpbpc.metrics import auc
from cdpbpc.patterns import N_PATTERNS
from cdpbpc.simulate import simulate_reads
from cdpbpc.two_pass import vote_estimate
from experiments.p1_vote import (
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
