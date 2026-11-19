from pathlib import Path

import numpy as np
import pytest

from cdpbpc.splits import (
    batch_e_print_ids,
    batch_fold,
    batch_group,
    check_disjoint,
    indigo_blocks,
    indigo_halves,
    indigo_page,
    indigo_stage,
    unique_blocks,
)

MANIFEST = Path(__file__).resolve().parents[1] / "dataset" / "manifest.csv"

# Values of the protocol, written here and not imported
# from cdpbpc.splits: a wrong constant in the module must make a test fail.
UNIQUE_SIZES = {"E": 50, "M": 500, "S": 1000, "T_orig": 1000}
UNIQUE_SEEDS = range(20)
BATCH_TEST_GROUPS = (1, 2, 3, 4)
BATCH_M_GROUP = {1: 2, 2: 3, 3: 4, 4: 1}  # M = the group after g, cycling in 1..4


def test_check_disjoint():
    check_disjoint({"E": np.array([1, 2]), "S": np.array([3])})
    with pytest.raises(ValueError):
        check_disjoint({"E": np.array([1, 2]), "S": np.array([2])})
    with pytest.raises(ValueError):
        check_disjoint({"E": np.array([1, 1])})


@pytest.mark.parametrize("seed", UNIQUE_SEEDS)
def test_unique_blocks(seed):
    b = unique_blocks(seed)
    check_disjoint(b)
    for name, size in UNIQUE_SIZES.items():
        assert len(b[name]) == size
        assert b[name].min() >= 1 and b[name].max() <= 3500
    every = np.concatenate(list(b.values()))
    assert not np.any((every >= 3501) & (every <= 3550))  # batch templates (D-014)
    np.testing.assert_array_equal(b["V"], np.arange(3551, 4001))
    np.testing.assert_array_equal(b["T"], np.arange(4001, 5001))


def test_unique_blocks_deterministic():
    a, b, c = unique_blocks(0), unique_blocks(0), unique_blocks(1)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert not np.array_equal(a["E"], c["E"])


def test_batch_group():
    np.testing.assert_array_equal(batch_group([3501, 3510, 3511, 3550]), [0, 0, 1, 4])
    with pytest.raises(ValueError):
        batch_group([3500])
    with pytest.raises(ValueError):
        batch_group([3551])


@pytest.mark.parametrize("g", [0, 1, 2, 3, 4])
def test_batch_fold(g):
    f = batch_fold(g)
    groups = f["groups"]
    assert groups["T"] == [g]
    assert sorted(x for gs in groups.values() for x in gs) == [
        0,
        1,
        2,
        3,
        4,
    ]  # each group in one block
    assert not set(groups["S"]) & set(groups["T"])
    assert len(f["blocks"]["M"]) == 10 and len(f["blocks"]["S"]) == 20
    if g in BATCH_TEST_GROUPS:
        assert groups["E"] == [0]
        assert groups["M"] == [BATCH_M_GROUP[g]]
    else:
        assert groups == {"E": [1], "M": [2], "S": [3, 4], "T": [0]}


@pytest.mark.parametrize("g", [0, 1, 2, 3, 4])
def test_batch_fold_tests_the_10_templates_of_its_group(g):
    first = 3500 + 10 * g + 1  # batch template k is template_id 3500 + k
    np.testing.assert_array_equal(batch_fold(g)["blocks"]["T"], np.arange(first, first + 10))


def test_group_0_never_tested():
    for g in BATCH_TEST_GROUPS:
        f = batch_fold(g)
        assert 0 not in f["groups"]["T"] + f["groups"]["M"] + f["groups"]["S"]
    with pytest.raises(ValueError):
        batch_fold(5)


@pytest.mark.parametrize("group", [0, 1, 4])
def test_batch_e_one_image_per_template_and_sheet(group):
    p = batch_e_print_ids(group)
    n = p - 1
    template = 10 * (n // 500) + (n % 100) // 10 + 1  # checked on the 2500 files
    sheet = n // 100
    assert len(p) == 50 and len(np.unique(p)) == 50
    assert set(template) == set(range(10 * group + 1, 10 * group + 11))
    assert len({(t, s) for t, s in zip(template, sheet)}) == 50
    assert set(np.unique(sheet)) == set(range(5 * group, 5 * group + 5))


@pytest.mark.parametrize("group", [0, 1, 2, 3, 4])
def test_batch_e_takes_the_first_copy(group):
    n = batch_e_print_ids(group) - 1
    assert np.all(n % 10 == 0)  # copy = n % 10; E uses the first one


@pytest.mark.skipif(not MANIFEST.exists(), reason="needs dataset/manifest.csv")
def test_batch_e_matches_manifest():
    from cdpbpc.io import read_manifest

    m = read_manifest(MANIFEST)
    orig = m[(m["dataset"] == "batch") & (m["class"] == "original")].set_index("print_id")
    for g in range(5):
        rows = orig.loc[batch_e_print_ids(g)]
        assert set(rows["group"]) == {g}
        assert rows.groupby("template_id")["sheet"].nunique().eq(5).all()


def test_indigo_page_and_stage():
    np.testing.assert_array_equal(indigo_page([1, 144, 145, 1440]), [0, 0, 1, 9])
    np.testing.assert_array_equal(indigo_stage([1, 288, 289]), ["dev", "dev", "test"])


def test_indigo_halves():
    a, b = indigo_halves("mobile")
    np.testing.assert_array_equal(a, np.arange(289, 865))
    np.testing.assert_array_equal(b, np.arange(865, 1441))
    assert set(indigo_page(a)) == {2, 3, 4, 5}  # pages 3 to 6, counted from 1
    a, b = indigo_halves("scanner")
    assert len(a) == len(b) == 216
    assert a[0] == 289 and b[0] == 290 and b[-1] == 720
    for a, b in (indigo_halves("mobile"), indigo_halves("scanner")):
        assert not set(a) & set(b)
    with pytest.raises(ValueError):
        indigo_halves("tablet")


def test_indigo_scanner_halves_cover_the_test_range():
    a, b = indigo_halves("scanner")
    np.testing.assert_array_equal(np.sort(np.concatenate([a, b])), np.arange(289, 721))


@pytest.mark.parametrize("setting", ["mobile", "scanner"])
def test_indigo_test_blocks(setting):
    b = indigo_blocks(setting, stage="test")
    check_disjoint(b)
    assert len(b["E"]) == 50 and len(b["M"]) == 150
    assert b["E"].max() <= 288 and b["M"].max() <= 288
    test = np.concatenate([b["T_a"], b["T_b"]])
    assert test.min() >= 289  # no development template in T
    assert len(test) == (1152 if setting == "mobile" else 432)


@pytest.mark.parametrize("setting", ["mobile", "scanner"])
def test_indigo_dev_blocks(setting):
    b = indigo_blocks(setting, stage="dev")
    check_disjoint(b)
    assert min(b["E"].min(), b["M"].min(), b["S"].min()) >= 289  # originals only
    np.testing.assert_array_equal(b["V"], np.arange(1, 289))
    assert len(b["E"]) + len(b["M"]) + len(b["S"]) == (1152 if setting == "mobile" else 432)


def test_indigo_blocks_follow_available_templates():
    missing = np.arange(400, 666)  # up to 266 templates miss in a file
    available = np.setdiff1d(np.arange(1, 1441), np.r_[missing, 5, 6])
    for stage in ("dev", "test"):
        b = indigo_blocks("mobile", stage=stage, available=available)
        every = np.concatenate(list(b.values()))
        assert not np.isin(every, np.r_[missing, 5, 6]).any()
    with pytest.raises(ValueError):
        indigo_blocks("scanner", stage="test", available=np.arange(1, 100))


@pytest.mark.parametrize("setting, last", [("mobile", 1440), ("scanner", 720)])
def test_indigo_blocks_lose_no_available_template(setting, last):
    available = np.setdiff1d(np.arange(1, last + 1), [5, 6, 300, 718])
    dev_ids = available[available <= 288]
    test_ids = available[available >= 289]

    b = indigo_blocks(setting, stage="dev", available=available)
    np.testing.assert_array_equal(np.sort(np.concatenate([b["E"], b["M"], b["S"]])), test_ids)
    np.testing.assert_array_equal(b["V"], dev_ids)

    b = indigo_blocks(setting, stage="test", available=available)
    np.testing.assert_array_equal(np.sort(np.concatenate([b["T_a"], b["T_b"]])), test_ids)
    assert np.isin(b["E"], dev_ids).all() and np.isin(b["M"], dev_ids).all()


def test_indigo_blocks_deterministic_and_checked():
    a, b = (
        indigo_blocks("mobile", stage="test", seed=0),
        indigo_blocks("mobile", stage="test", seed=0),
    )
    np.testing.assert_array_equal(a["E"], b["E"])
    assert not np.array_equal(a["E"], indigo_blocks("mobile", stage="test", seed=1)["E"])
    with pytest.raises(ValueError):
        indigo_blocks("mobile", stage="all")


# "No Indigo fake >= 289 read with stage='dev'" is enforced when the data
# is read and tested in tests/test_io.py (test_dev_stage_drops_test_templates).
