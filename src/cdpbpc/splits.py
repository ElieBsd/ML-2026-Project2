"""Blocks of templates.

Every image of a template stays in the block of its template, so blocks are sets of
template_ids. Block names:

    E   codebook
    M   mean and covariance of the two-sided test (binned_mahalanobis only)
    S   threshold (conformal, originals only)
    V   development: the only fakes looked at before the methods freeze
    T   test

Which fakes may be read at all is decided when the data is read (io.load_compact and
io.scantrust_stage_rows with stage="dev"); the blocks here only say which templates play
which role. The development boundaries (Indigo templates 1..288, unique 3551..4000, batch
group 0) are imported from cdpbpc.io, so blocks and reading cuts cannot drift apart.
check_disjoint guards every split.
"""

import numpy as np

from cdpbpc.io import (
    INDIGO_DEV_MAX_ID,
    SCANTRUST_DEV_GROUP,
    SCANTRUST_DEV_UNIQUE,
    STAGES,
)

# Scantrust unique_cdp (session U, one image per template)
UNIQUE_POOL = (1, 3500)  # originals only: E, M, S, T_orig are drawn here
UNIQUE_BATCH = (3501, 3550)  # templates of batch_cdp (D-014): in no unique block
UNIQUE_V = SCANTRUST_DEV_UNIQUE  # V_U (3551..4000): originals and 4 attacks, development
UNIQUE_T = (4001, 5000)  # T_U: originals and 4 attacks, test
UNIQUE_SIZES = {"E": 50, "M": 500, "S": 1000, "T_orig": 1000}
UNIQUE_N_SEEDS = 20

# Scantrust batch_cdp (session B, 50 prints per template)
BATCH_FIRST_ID = 3501  # batch template k is template_id 3500 + k
BATCH_N_GROUPS = 5  # 10 templates per group
BATCH_GROUP_SIZE = 10
BATCH_DEV_GROUP = SCANTRUST_DEV_GROUP  # development group (0), never tested
BATCH_TEST_GROUPS = (1, 2, 3, 4)
BATCH_E_COPY = 0  # copy on the sheet used for E: one image per template and per sheet

# Indigo
INDIGO_PAGE = 144
INDIGO_DEV = (1, INDIGO_DEV_MAX_ID)  # pages 1 and 2
INDIGO_TEST = {"mobile": (289, 1440), "scanner": (289, 720)}
INDIGO_E_SIZE = 50
INDIGO_M_SIZE = 150


def _range(lo_hi: tuple[int, int]) -> np.ndarray:
    lo, hi = lo_hi
    return np.arange(lo, hi + 1)


def _check_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")


def check_disjoint(blocks: dict[str, np.ndarray]) -> None:
    """Raise ValueError if a template_id is in two blocks (or twice in one block)."""
    seen: dict[int, str] = {}
    for name, ids in blocks.items():
        ids = np.asarray(ids)
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"block {name} lists a template twice")
        for tid in ids.tolist():
            if tid in seen:
                raise ValueError(f"template {tid} is in blocks {seen[tid]} and {name}")
            seen[tid] = name


# ---------------------------------------------------------------- Scantrust unique_cdp


def unique_blocks(seed: int) -> dict[str, np.ndarray]:
    """Blocks of unique_cdp for one draw (seeds 0..19).

    E, M, S, T_orig: disjoint random draws among templates 1..3500 (originals only).
    V: templates 3551..4000 (V_U). T: templates 4001..5000 (T_U). Ids are sorted.
    """
    perm = np.random.default_rng(seed).permutation(_range(UNIQUE_POOL))
    blocks, start = {}, 0
    for name, size in UNIQUE_SIZES.items():
        blocks[name] = np.sort(perm[start : start + size])
        start += size
    blocks["V"] = _range(UNIQUE_V)
    blocks["T"] = _range(UNIQUE_T)
    check_disjoint(blocks)
    return blocks


# ----------------------------------------------------------------- Scantrust batch_cdp


def batch_group(template_id: np.ndarray) -> np.ndarray:
    """Group 0..4 of batch templates (template_id 3501..3550)."""
    tid = np.asarray(template_id)
    if np.any((tid < BATCH_FIRST_ID) | (tid >= BATCH_FIRST_ID + BATCH_N_GROUPS * BATCH_GROUP_SIZE)):
        raise ValueError("not a batch_cdp template_id (3501..3550)")
    return (tid - BATCH_FIRST_ID) // BATCH_GROUP_SIZE


def batch_group_ids(group: int) -> np.ndarray:
    """The 10 template_ids of one batch group."""
    first = BATCH_FIRST_ID + BATCH_GROUP_SIZE * group
    return np.arange(first, first + BATCH_GROUP_SIZE)


def batch_e_print_ids(group: int) -> np.ndarray:
    """print_ids of the 50 originals that form E: one per template and per sheet.

    A group spans 5 sheets of 100 files; on sheet s the template of rank r has files
    100 s + 10 r + c + 1, c = 0..9. We take copy c = BATCH_E_COPY.
    """
    sheets = np.arange(5 * group, 5 * group + 5)
    ranks = np.arange(BATCH_GROUP_SIZE)
    return (100 * sheets[:, None] + 10 * ranks[None, :] + BATCH_E_COPY + 1).ravel()


def batch_fold(group: int) -> dict:
    """Groups and E images of the batch fold that evaluates `group`.

    Test fold g in 1..4: E from group 0, M = group after g (cycling in 1..4), S = the two
    other groups, T = g. Development (g = 0): E from group 1, M = group 2, S = groups 3, 4,
    T = group 0. Deterministic, no seed.

    Returns {"groups": {block: [groups]}, "blocks": {block: template_ids},
    "e_print_ids": print_ids of E}. E holds only 50 images of its group, so the group of
    E is not listed in "blocks" with the others but is disjoint from them.
    """
    if group == BATCH_DEV_GROUP:
        groups = {"E": [1], "M": [2], "S": [3, 4], "T": [0]}
    elif group in BATCH_TEST_GROUPS:
        m = group % 4 + 1
        groups = {
            "E": [0],
            "M": [m],
            "S": [g for g in BATCH_TEST_GROUPS if g not in (group, m)],
            "T": [group],
        }
    else:
        raise ValueError(f"group must be in 0..4, got {group}")
    blocks = {b: np.concatenate([batch_group_ids(g) for g in gs]) for b, gs in groups.items()}
    check_disjoint(blocks)
    return {"groups": groups, "blocks": blocks, "e_print_ids": batch_e_print_ids(groups["E"][0])}


# ---------------------------------------------------------------------------- Indigo


def indigo_page(template_id: np.ndarray) -> np.ndarray:
    """Page 0..9 of an Indigo template (144 per page)."""
    return (np.asarray(template_id) - 1) // INDIGO_PAGE


def indigo_stage(template_id: np.ndarray) -> np.ndarray:
    """'dev' for templates 1..288, 'test' otherwise."""
    return np.where(np.asarray(template_id) <= INDIGO_DEV[1], "dev", "test")


def indigo_halves(setting: str) -> tuple[np.ndarray, np.ndarray]:
    """The two halves of the test range for the cross-conformal threshold.

    mobile: pages 3-6 (templates 289..864) and pages 7-10 (865..1440).
    scanner: even and odd rank in 289..720 (289, 291, ... and 290, 292, ...).
    """
    if setting not in INDIGO_TEST:
        raise ValueError(f"setting must be one of {tuple(INDIGO_TEST)}, got {setting!r}")
    ids = _range(INDIGO_TEST[setting])
    if setting == "mobile":
        return ids[: len(ids) // 2], ids[len(ids) // 2 :]
    return ids[0::2], ids[1::2]


def indigo_blocks(
    setting: str, *, stage: str, available: np.ndarray | None = None, seed: int = 0
) -> dict[str, np.ndarray]:
    """Blocks for one condition (a phone and a capture, or a printer at the scanner).

    available: template_ids that have an original in this condition (some files miss
    templates); every block is restricted to it, T_a and T_b included. None means all
    templates of the setting. A fake whose template has no original in the condition is
    therefore in no block and is never judged (13 test fakes in all of Indigo, listed in
    protocol.md §1.3).

    stage="test": E = 50 and M = 150 originals drawn (seed) among development templates
    1..288; the test range is split in two halves T_a and T_b for the cross-conformal
    threshold (the threshold for one half comes from the originals of the other).

    stage="dev": E, M, S are drawn among originals of templates >= 289 (originals are
    free), S taking all the rest; V = templates 1..288 (originals and fakes).
    """
    _check_stage(stage)
    if setting not in INDIGO_TEST:
        raise ValueError(f"setting must be one of {tuple(INDIGO_TEST)}, got {setting!r}")
    test_ids = _range(INDIGO_TEST[setting])
    dev_ids = _range(INDIGO_DEV)
    if available is not None:
        available = np.asarray(available)
        test_ids = test_ids[np.isin(test_ids, available)]
        dev_ids = dev_ids[np.isin(dev_ids, available)]
    rng = np.random.default_rng(seed)
    if stage == "test":
        pool = rng.permutation(dev_ids)
        a, b = indigo_halves(setting)
        blocks = {
            "E": np.sort(pool[:INDIGO_E_SIZE]),
            "M": np.sort(pool[INDIGO_E_SIZE : INDIGO_E_SIZE + INDIGO_M_SIZE]),
            "T_a": a[np.isin(a, test_ids)],
            "T_b": b[np.isin(b, test_ids)],
        }
        need = INDIGO_E_SIZE + INDIGO_M_SIZE
    else:
        pool = rng.permutation(test_ids)
        blocks = {
            "E": np.sort(pool[:INDIGO_E_SIZE]),
            "M": np.sort(pool[INDIGO_E_SIZE : INDIGO_E_SIZE + INDIGO_M_SIZE]),
            "S": np.sort(pool[INDIGO_E_SIZE + INDIGO_M_SIZE :]),
            "V": dev_ids,
        }
        need = INDIGO_E_SIZE + INDIGO_M_SIZE + 1
    if len(pool) < need:
        raise ValueError(f"only {len(pool)} templates available to draw E and M")
    check_disjoint(blocks)
    return blocks
