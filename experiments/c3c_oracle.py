"""Supervised detectors trained on real fakes. ORACLE references, never fake-free methods.

A defender usually has no fakes. This script measures what having some would buy, and
what is left of it when the fakes met later do not look like the training fakes. Two
oracles (cdpbpc.baselines.supervised):
- logistic: scaling then L2 logistic regression on the 8 per-class deviations binned_z,
  classes balanced, C chosen by cross-validation grouped by template inside the
  training data;
- lrt_plugin: the likelihood-ratio test of the thesis (Tutt 2025, eqs. 6.18-6.21,
  p. 125-127) with the flip probabilities of fakes Q_b learned on the training fakes.
Next to them, on exactly the same images and with the same kind of threshold, two
fake-free scores: hamming and masked_hamming (P_b <= mu, thesis p. 104).

What a defender learns where:
- the codebook P_b behind the features, and the threshold, come from originals of the
  condition being judged (originals are always available there);
- only the logistic weights and Q_b come from the training fakes, possibly from another
  condition. Training images use the codebook of their own condition.
- A template is never both trained on and judged, and the originals of the threshold
  are never in the training data (checked for every part).

Cells (training fakes -> judged fakes):
- Indigo mobile, capture 1: each phone judged with a model trained on the same phone
  and with one trained on the other phone.
- Indigo scanner, for the originals of each printer: one row per fake type (printer
  that reprinted the fake, prints the estimate was made from) and one row with all
  types, judged on every fake type. Fakes reprinted on the other printer than the
  training fakes form the "other printer" cells.
- Scantrust batch_cdp: one row per attack and one with all attacks, judged on every
  attack; and one row trained on unique_cdp templates 3551..4000 (all attacks), the
  other session.

Blocks.
- Development (--stage dev; fakes of Indigo templates 1..288, batch group 0 and unique
  3551..4000 only): Indigo, codebook E and threshold block S among the originals of
  templates >= 289; the templates 1..288 are cut in two halves (mobile: page 1 and page
  2; scanner: odd and even ranks), a model trained on one half judges the other, and
  the other way round. batch: E = 50 originals of group 1, S = groups 3 and 4; group 0
  is cut into templates 3501..3505 and 3506..3510, crossed the same way. The unique
  row is trained on 3551..4000 and judges all of group 0.
- Test (--stage test, once, after the methods freeze; needs freeze_commit): models are
  trained on the development templates (Indigo 1..288 without the templates of block E;
  batch group 0 without the 50 images of block E; unique 3551..4000) and judge the test
  templates with the thresholds of the fake-free scores: Indigo, cross-conformal on the
  two halves of the test range (the threshold of a half comes from the originals of the
  other); batch, groups 1..4 left out one at a time, threshold from the two groups of
  block S of the fold, E = 50 originals of group 0.

Measures: P_fa at the target P_miss (conformal threshold) with the realized P_miss; P_fa
at equal P_miss (threshold = quantile of the judged originals, labelled); AUC. Indigo:
one image per template, Clopper-Pearson intervals. batch_cdp: rates per template, mean,
bootstrap interval over templates, design effect. The prediction set in the config is
judged at the test stage only; at the development stage it is an indication.

Usage:
    python -m experiments.c3c_oracle --config configs/c3c.yaml --stage dev
Writes results/c3c_<config hash>.json at the development stage and
results/c3c_<config hash>_test.json at the test stage, which refuses to overwrite it.
"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from cdpbpc.baselines.supervised import fake_codebook, fit_logistic, logistic_score, lrt_plugin
from cdpbpc.calibrate import conformal_threshold
from cdpbpc.counts import indigo_counts, scantrust_counts
from cdpbpc.io import INDIGO_DEV_MAX_ID, STAGES
from cdpbpc.metrics import (
    auc,
    bootstrap_mean_ci,
    clopper_pearson,
    design_effect,
    rate_per_group,
)
from cdpbpc.scores import binned_z, hamming, masked_hamming
from cdpbpc.splits import batch_fold, batch_group_ids, indigo_blocks, unique_blocks
from experiments.c3b_two_pass import (
    _ints,
    _is,
    _used,
    check_freeze_commit,
    check_indigo_ids,
    check_scantrust_fakes,
)
from experiments.e0_sanity import _round, codebook_of, git_state

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FAKE_FREE = ("hamming", "masked_hamming")


# ------------------------------------------------------------------------ structures


@dataclass
class Images:
    """Counts of images. ids: (N,) template of each image; D, L: (N, 512)."""

    ids: np.ndarray
    D: np.ndarray
    L: np.ndarray

    def take(self, mask) -> "Images":
        return Images(self.ids[mask], self.D[mask], self.L[mask])

    def of(self, template_ids) -> "Images":
        return self.take(np.isin(self.ids, template_ids))

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class Condition:
    """Originals and fakes captured in one condition, with its codebook and blocks.

    unit: "image" (one image per template: Clopper-Pearson) or "template" (rates per
    template). P: (512,) codebook of block E. fakes: {fake type: Images}.
    parts: [{"judged", "threshold", "train"}] template ids: the templates judged in a
    part, the originals giving its threshold, and the templates a model of this
    condition is trained on when it judges the same part. no_train: (n originals,) bool,
    the images of block E, never trained on. fixed_train: template ids a model of this
    condition is always trained on (a condition that is never judged).
    """

    name: str
    unit: str
    P: np.ndarray
    originals: Images
    fakes: dict
    parts: list
    alphas: list
    no_train: np.ndarray
    info: dict | None = None
    fixed_train: np.ndarray | None = None

    def train_ids(self, k: int) -> np.ndarray:
        return self.fixed_train if self.fixed_train is not None else self.parts[k]["train"]


@dataclass
class Cell:
    """A row of the matrix: a model trained on fakes `types` of `train`, judging `judged`.

    relation: {judged fake type: "same", "other_session", "other_printer" or "other_phone"}.
    """

    judged: Condition
    train: Condition
    row: str
    types: list
    relation: dict


def check_part(train_ids, judged_ids, threshold_ids) -> None:
    """Refuse a part where a template is in two of training, judging and threshold."""
    for a, b, what in (
        (train_ids, judged_ids, "trained on and judged"),
        (train_ids, threshold_ids, "trained on and used for the threshold"),
        (threshold_ids, judged_ids, "used for the threshold and judged"),
    ):
        if np.intersect1d(a, b).size:
            raise ValueError(f"a template is both {what}")


def _span(ids) -> list:
    ids = np.asarray(ids)
    return [int(ids.min()), int(ids.max())] if len(ids) else []


# -------------------------------------------------------------------------- measures


def _rates(miss, acc, ids_o, ids_f, unit, rng, n_boot) -> dict:
    if unit == "image":
        k, n = int(acc.sum()), len(acc)
        km, nm = int(miss.sum()), len(miss)
        return {
            "p_fa": k / n,
            "p_fa_ci": list(clopper_pearson(k, n)),
            "n_fakes_accepted": k,
            "n_fakes": n,
            "p_miss": km / nm,
            "p_miss_ci": list(clopper_pearson(km, nm)),
            "n_originals_rejected": km,
            "n_originals": nm,
        }
    _, fa = rate_per_group(acc, ids_f)
    _, ms = rate_per_group(miss, ids_o)
    return {
        "p_fa": float(fa.mean()),
        "p_fa_ci": list(bootstrap_mean_ci(fa, rng, n_boot)),
        "p_fa_per_template": fa.tolist(),
        "deff_fakes": design_effect(acc, ids_f),
        "n_templates": len(fa),
        "p_miss": float(ms.mean()),
        "p_miss_ci": list(bootstrap_mean_ci(ms, rng, n_boot)),
    }


def measure(parts: list, unit: str, alphas, rng, n_boot: int) -> dict:
    """Measures of one score on one fake type.

    parts: [{"o", "s", "f", "ids_o", "ids_f"}]: scores of the judged originals, of the
    threshold originals and of the judged fakes of each part, with their templates.
    Each part has its own thresholds: conformal on "s", and, labelled, the quantile
    1 - alpha of its judged originals "o" (P_fa at equal P_miss).
    """
    if unit not in ("image", "template"):
        raise ValueError(f"unit must be image or template, got {unit!r}")
    o = np.concatenate([p["o"] for p in parts])
    f = np.concatenate([p["f"] for p in parts])
    if len(o) == 0 or len(f) == 0:
        raise ValueError("no judged original or fake")
    ids_o = np.concatenate([np.asarray(p["ids_o"]) for p in parts])
    ids_f = np.concatenate([np.asarray(p["ids_f"]) for p in parts])
    out = {"auc": auc(o, f), "n_originals": len(o), "n_fakes": len(f)}
    for a in alphas:
        rows = {}
        for name, taus in (
            ("conformal", [conformal_threshold(p["s"], a) for p in parts]),
            ("equal_p_miss_test_originals", [float(np.quantile(p["o"], 1 - a)) for p in parts]),
        ):
            miss = np.concatenate([p["o"] > t for p, t in zip(parts, taus)])
            acc = np.concatenate([p["f"] <= t for p, t in zip(parts, taus)])
            rows[name] = {
                "tau": [float(t) for t in taus],
                **_rates(miss, acc, ids_o, ids_f, unit, rng, n_boot),
            }
        out[f"alpha_{a}"] = rows
    return out


def evaluate(cond: Condition, score_fn, rng, n_boot: int) -> dict:
    """{score: {fake type: measures}} of the judged parts of a condition.

    score_fn(k, images) -> {score name: (n,)}, the scores of images in part k.
    """
    scored = []
    for k, part in enumerate(cond.parts):
        O, S = cond.originals.of(part["judged"]), cond.originals.of(part["threshold"])
        F = {t: f.of(part["judged"]) for t, f in cond.fakes.items()}
        scored.append(
            {
                "O": score_fn(k, O),
                "S": score_fn(k, S),
                "F": {t: score_fn(k, x) for t, x in F.items()},
                "ids_o": O.ids,
                "ids_f": {t: x.ids for t, x in F.items()},
            }
        )
    out = {}
    for name in scored[0]["O"]:
        out[name] = {}
        for t in cond.fakes:
            parts = [
                {
                    "o": p["O"][name],
                    "s": p["S"][name],
                    "f": p["F"][t][name],
                    "ids_o": p["ids_o"],
                    "ids_f": p["ids_f"][t],
                }
                for p in scored
            ]
            out[name][t] = measure(parts, cond.unit, cond.alphas, rng, n_boot)
    return out


# ---------------------------------------------------------------------------- models


def fit_oracle(cond: Condition, types, train_ids, cfg: dict) -> dict:
    """ORACLE. Logistic and Q_b of lrt_plugin, trained on originals and fakes of `types`.

    Images of templates train_ids, without the originals of block E; features binned_z
    with the codebook of `cond`, the condition the training images come from.
    """
    O = cond.originals.take(np.isin(cond.originals.ids, train_ids) & ~cond.no_train)
    F = [cond.fakes[t].of(train_ids) for t in types]
    D = np.concatenate([O.D] + [f.D for f in F])
    L = np.concatenate([O.L] + [f.L for f in F])
    groups = np.concatenate([O.ids] + [f.ids for f in F])
    y = np.r_[np.zeros(len(O)), np.ones(len(groups) - len(O))]
    model, C, cv = fit_logistic(
        binned_z(D, L, cond.P), y, groups, cfg["Cs"], cfg["n_splits"], cfg["seed"]
    )
    fk = y == 1
    return {
        "model": model,
        "q": fake_codebook(D[fk], L[fk]),
        "C": C,
        "cv_log_loss": cv,
        "train_templates": _span(train_ids),
        "n_templates": len(np.unique(groups)),
        "n_originals": len(O),
        "n_fakes": int(fk.sum()),
    }


def oracle_scores(fit: dict, cond: Condition, imgs: Images) -> dict:
    """ORACLE scores of images of the judged condition, features with its own codebook."""
    return {
        "logistic": logistic_score(fit["model"], binned_z(imgs.D, imgs.L, cond.P)),
        "lrt_plugin": lrt_plugin(imgs.D, imgs.L, cond.P, fit["q"]),
    }


def fake_free_scores(cond: Condition, imgs: Images, mu: float) -> dict:
    out = {
        "hamming": hamming(imgs.D, imgs.L),
        "masked_hamming": masked_hamming(imgs.D, imgs.L, cond.P, mu),
    }
    for k, v in out.items():
        if not np.isfinite(v).all():
            raise ValueError(f"{k}: an image has no symbol to score")
    return out


def run_cell(cell: Cell, cfg: dict, memo: dict, rng) -> dict:
    """One row of the matrix; models are shared between cells through memo."""
    J, T = cell.judged, cell.train
    fits, parts = [], []
    for k, part in enumerate(J.parts):
        tr = np.sort(np.asarray(T.train_ids(k)))
        check_part(tr, part["judged"], part["threshold"])
        key = (T.name, tuple(cell.types), tuple(tr.tolist()))
        if key not in memo:
            memo[key] = fit_oracle(T, cell.types, tr, cfg)
        fits.append(memo[key])
        parts.append(
            {
                "train_templates": _span(tr),
                "judged_templates": _span(part["judged"]),
                "n_judged": len(part["judged"]),
                "threshold_templates": _span(part["threshold"]),
                "n_threshold": len(part["threshold"]),
            }
        )
    res = evaluate(J, lambda k, imgs: oracle_scores(fits[k], J, imgs), rng, cfg["n_boot"])
    models = [{k: v for k, v in f.items() if k not in ("model", "q")} for f in fits]
    return {
        "trained_on": T.name,
        "training_types": list(cell.types),
        "relation": dict(cell.relation),
        "parts": parts,
        "models": models,
        **res,
    }


def c_sensitivity(cell: Cell, cfg: dict, rng) -> dict:
    """Sensitivity of a row to the penalty: the row refit with each C of the grid in turn.

    Nothing is chosen here and no prediction reads it: it only shows how much the P_fa
    of the logistic moves with a setting that cannot be tuned for a new condition
    without its fakes. Returns {C: {fake type: measures}}.
    """
    J, T = cell.judged, cell.train
    out = {}
    for C in cfg["Cs"]:
        fits = [
            fit_oracle(T, cell.types, np.sort(np.asarray(T.train_ids(k))), {**cfg, "Cs": [C]})
            for k in range(len(J.parts))
        ]
        res = evaluate(
            J,
            lambda k, imgs, fits=fits: {"logistic": oracle_scores(fits[k], J, imgs)["logistic"]},
            rng,
            cfg["n_boot"],
        )
        out[str(float(C))] = res["logistic"]
    return out


def run_cells(conds: dict, cells: list, cfg: dict, rng, log) -> dict:
    """Fake-free scores and every oracle row, for each judged condition."""
    memo = {}
    out = {}
    for cell in cells:
        J = cell.judged
        if J.name not in out:
            ff = evaluate(
                J, lambda k, imgs, J=J: fake_free_scores(J, imgs, cfg["mu"]), rng, cfg["n_boot"]
            )
            out[J.name] = {"unit": J.unit, "info": J.info, "fake_free": ff, "oracle": {}}
        r = run_cell(cell, cfg, memo, rng)
        if set(cell.relation.values()) & set(cfg.get("c_sensitivity", [])):
            # its own generator: the other numbers do not depend on this extra row
            r["logistic_by_C"] = c_sensitivity(cell, cfg, np.random.default_rng(cfg["seed"] + 1))
        out[J.name]["oracle"][cell.row] = r
        log(_line(J.name, cell.row, r, out[J.name]["fake_free"]))
    return out


def _line(name, row, r, ff) -> str:
    a = "alpha_0.05"
    cells = []
    for t, m in r["logistic"].items():
        best = min(ff[s][t][a]["conformal"]["p_fa"] for s in FAKE_FREE)
        c = m[a]["conformal"]
        cells.append(f"{t} {c['p_fa']:.3f} (P_miss {c['p_miss']:.3f}; fake-free {best:.3f})")
    return f"{name} | row {row}: logistic P_fa at 5 %: " + ", ".join(cells)


# --------------------------------------------------------------------------- Indigo


def indigo_dev_halves(setting: str) -> tuple[np.ndarray, np.ndarray]:
    """Two halves of the development templates 1..288, built like the test halves:
    mobile, page 1 and page 2; scanner, odd and even ranks."""
    ids = np.arange(1, INDIGO_DEV_MAX_ID + 1)
    if setting == "mobile":
        return ids[: len(ids) // 2], ids[len(ids) // 2 :]
    if setting == "scanner":
        return ids[0::2], ids[1::2]
    raise ValueError(f"setting must be mobile or scanner, got {setting!r}")


def indigo_condition(
    pack: Path, name: str, setting: str, original: str, fakes: dict, alphas, cfg, stage
) -> Condition:
    o = indigo_counts(pack / f"{original}.npz", stage=stage)
    if o.meta["class"] != "original":
        raise ValueError(f"{original} holds no originals")
    F = {}
    for t, file in fakes.items():
        f = indigo_counts(pack / f"{file}.npz", stage=stage)
        if f.meta["class"] != "fake":
            raise ValueError(f"{file} holds no fakes")
        check_indigo_ids(f.ids, stage, "fake")
        F[t] = Images(np.asarray(f.ids), f.D, f.L)
    O = Images(np.asarray(o.ids), o.D, o.L)
    b = indigo_blocks(setting, stage=stage, available=O.ids, seed=cfg["seed"])
    dev_ids = np.arange(1, INDIGO_DEV_MAX_ID + 1)
    dev_ids = dev_ids[np.isin(dev_ids, O.ids)]
    if stage == "dev":
        h = [x[np.isin(x, b["V"])] for x in indigo_dev_halves(setting)]
        parts = [{"judged": h[k], "threshold": b["S"], "train": h[1 - k]} for k in (0, 1)]
        judged = b["V"]
    else:
        train = np.setdiff1d(dev_ids, b["E"])
        parts = [
            {"judged": b["T_a"], "threshold": b["T_b"], "train": train},
            {"judged": b["T_b"], "threshold": b["T_a"], "train": train},
        ]
        judged = np.concatenate([b["T_a"], b["T_b"]])
    for p in parts:
        check_indigo_ids(p["judged"], stage, "fake")
    lo, hi = (1, INDIGO_DEV_MAX_ID) if stage == "dev" else (int(judged.min()), int(judged.max()))
    in_range = set().union(*(f.ids[(f.ids >= lo) & (f.ids <= hi)].tolist() for f in F.values()))
    info = {
        "originals": original,
        "fakes": dict(fakes),
        "E_templates": _span(b["E"]),
        "n_E": len(b["E"]),
        "fakes_not_judged": sorted(in_range - set(judged.tolist())),
    }
    return Condition(
        name,
        "image",
        codebook_of(O.D[np.isin(O.ids, b["E"])], O.L[np.isin(O.ids, b["E"])]),
        O,
        F,
        parts,
        list(alphas),
        np.isin(O.ids, b["E"]),
        info,
    )


def _reprint_printer(fake_type: str) -> str:
    return fake_type.split("_")[0]


def indigo_conditions(pack: Path, cfg: dict, stage: str) -> tuple[dict, list]:
    mob, scan = cfg["indigo"]["mobile"], cfg["indigo"]["scanner"]
    r, est = mob["capture"], mob["estimate"]
    conds, cells = {}, []
    phones = {}
    for d in mob["devices"]:
        phones[d] = indigo_condition(
            pack,
            f"mobile {d}",
            "mobile",
            f"mobile_original_{d}_run{r}",
            {est: f"mobile_fake_{est}_{d}_run{r}"},
            mob["alphas"],
            cfg,
            stage,
        )
    for dj, J in phones.items():
        for dt, T in phones.items():
            rel = "same" if dj == dt else "other_phone"
            cells.append(Cell(J, T, f"trained on {dt}", [est], {est: rel}))
    types = {f"{p}_{e}": f"base_fake_{p}_{e}" for p, e in scan["fakes"]}
    for printer in scan["printers"]:
        J = indigo_condition(
            pack,
            f"scanner {printer}",
            "scanner",
            f"base_original_{printer}",
            types,
            scan["alphas"],
            cfg,
            stage,
        )
        conds[J.name] = J
        for t in types:
            rel = {
                u: "same" if _reprint_printer(u) == _reprint_printer(t) else "other_printer"
                for u in types
            }
            cells.append(Cell(J, J, t, [t], rel))
        cells.append(Cell(J, J, "all", list(types), {u: "same" for u in types}))
    conds.update({p.name: p for p in phones.values()})
    return conds, cells


# ------------------------------------------------------------------------- Scantrust


def scantrust_conditions(data_dir: Path, cfg: dict, stage: str) -> tuple[dict, list]:
    c = scantrust_counts(data_dir, stage=stage)
    rows = c.rows.reset_index(drop=True)
    check_scantrust_fakes(rows, stage)
    st = cfg["scantrust"]
    tids = _ints(rows, "template_id")
    used = _used(rows)
    is_orig = _is(rows, "class", "original") & used
    is_fake = _is(rows, "class", "fake") & used

    def images(mask):
        return Images(tids[mask], c.D[mask], c.L[mask])

    # batch_cdp, the judged condition
    bat = _is(rows, "dataset", "batch")
    prints = _ints(rows, "print_id")
    groups = [0] if stage == "dev" else [1, 2, 3, 4]
    folds = {g: batch_fold(g) for g in groups}
    e_ids = folds[groups[0]]["e_print_ids"]
    if any(not np.array_equal(f["e_print_ids"], e_ids) for f in folds.values()):
        raise ValueError("the batch folds do not share block E")
    e_group = folds[groups[0]]["groups"]["E"]
    O = bat & is_orig
    E = O & np.isin(prints, e_ids)
    if stage == "dev":
        f0 = folds[0]["blocks"]
        h = (f0["T"][:5], f0["T"][5:])
        parts = [{"judged": h[k], "threshold": f0["S"], "train": h[1 - k]} for k in (0, 1)]
    else:
        train = batch_group_ids(0)
        parts = [
            {
                "judged": folds[g]["blocks"]["T"],
                "threshold": folds[g]["blocks"]["S"],
                "train": train,
            }
            for g in groups
        ]
    fakes = {a: images(bat & is_fake & _is(rows, "attack", a)) for a in st["batch_attacks"]}
    batch = Condition(
        "batch",
        "template",
        codebook_of(c.D[E], c.L[E]),
        images(O),
        fakes,
        parts,
        list(st["alphas"]),
        E[O],
        {"E_originals": int(E.sum()), "E_print_groups": e_group, "groups_judged": groups},
    )
    # unique_cdp templates 3551..4000 (V_U), the other session: training only
    uni = _is(rows, "dataset", "unique")
    ub = unique_blocks(cfg["seed"])
    Eu = uni & is_orig & np.isin(tids, ub["E"])
    Ou = uni & is_orig & np.isin(tids, ub["V"])
    fu = {
        a: images(uni & is_fake & _is(rows, "attack", a) & np.isin(tids, ub["V"]))
        for a in st["unique_attacks"]
    }
    vu = Condition(
        "unique V_U",
        "image",
        codebook_of(c.D[Eu], c.L[Eu]),
        images(Ou),
        fu,
        [],
        [],
        np.zeros(int(Ou.sum()), bool),
        {"E_originals": int(Eu.sum())},
        fixed_train=ub["V"],
    )
    cells = [Cell(batch, batch, a, [a], {t: "same" for t in fakes}) for a in fakes]
    cells.append(Cell(batch, batch, "all", list(fakes), {t: "same" for t in fakes}))
    cells.append(
        Cell(batch, vu, "unique V_U (all attacks)", list(fu), {t: "other_session" for t in fakes})
    )
    return {"batch": batch, "unique V_U": vu}, cells


# ------------------------------------------------------------------------ prediction


def judge_prediction(conditions: dict, pred: dict) -> dict:
    """The prediction set in the config, checked on the results.

    Same condition: in every cell where training and judged fakes come from the same
    phone, reprinting printer or session, P_fa of the logistic at the target P_miss is
    at most the smallest P_fa of the fake-free scores. Other condition: in each case
    listed, at least one cell where the logistic accepts more fakes than the best
    fake-free score.
    """
    key = f"alpha_{pred['alpha']}"
    same, other = [], {case: [] for case in pred["other_cases"]}
    for name, r in conditions.items():
        for row, o in r["oracle"].items():
            for t, m in o["logistic"].items():
                lg = m[key]["conformal"]["p_fa"]
                best = min(
                    r["fake_free"][s][t][key]["conformal"]["p_fa"] for s in pred["fake_free"]
                )
                cell = {
                    "judged": name,
                    "row": row,
                    "fakes": t,
                    "logistic": lg,
                    "best_fake_free": best,
                }
                rel = o["relation"][t]
                if rel == "same":
                    same.append({**cell, "holds": bool(lg <= best)})
                elif rel in other:
                    other[rel].append({**cell, "worse": bool(lg > best)})
    cases = {
        case: {"cells": len(v), "worse_cells": sum(c["worse"] for c in v), "detail": v}
        for case, v in other.items()
    }
    s_ok = bool(same) and all(c["holds"] for c in same)
    o_ok = all(c["worse_cells"] > 0 for c in cases.values())
    return {
        "same_condition": {
            "holds": s_ok,
            "cells": len(same),
            "failing": [c for c in same if not c["holds"]],
        },
        "other_condition": {"holds": bool(o_ok), "cases": cases},
        "holds": bool(s_ok and o_ok),
    }


# ------------------------------------------------------------------------------ main


def run_all(cfg: dict, stage: str, rng, log) -> dict:
    data_dir = ROOT / cfg["data_dir"]
    conds, cells = indigo_conditions(data_dir / "indigo_compact", cfg, stage)
    sconds, scells = scantrust_conditions(data_dir, cfg, stage)
    res = run_cells(conds | sconds, cells + scells, cfg, rng, log)
    return {
        "conditions": res,
        "prediction": {
            "status": "judged (test stage)" if stage == "test" else "indication only (development)",
            **judge_prediction(res, cfg["prediction"]),
        },
    }


def result_path(text: str, stage: str) -> Path:
    """Result file of a config text: c3c_<hash>.json, or c3c_<hash>_test.json at the test stage.

    The test result has its own name so that a later development run with the same
    config can never overwrite it.
    """
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    return RESULTS / f"c3c_{h}{'_test' if stage == 'test' else ''}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Supervised detectors trained on fakes (oracle)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "c3c.yaml")
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args(argv)
    text = args.config.read_text()
    cfg = yaml.safe_load(text)
    if args.stage == "test" and not cfg.get("freeze_commit"):
        raise SystemExit("--stage test needs freeze_commit in the config (methods freeze)")
    out = result_path(text, args.stage)
    if args.stage == "test" and out.exists():
        raise SystemExit(f"{out.name} exists: the test stage runs once")
    if args.stage == "test":
        check_freeze_commit(cfg["freeze_commit"])
    start = time.time()

    def log(msg):
        print(f"[{time.time() - start:6.0f} s] {msg}", flush=True)

    rng = np.random.default_rng(cfg["seed"])
    result = {"experiment": "c3c_oracle", "stage": args.stage, "git": None, "config": cfg}
    result.update(run_all(cfg, args.stage, rng, log))
    result["seconds"] = round(time.time() - start)
    result["git"] = git_state()
    out.parent.mkdir(exist_ok=True)
    with out.open("x" if args.stage == "test" else "w") as f:
        f.write(json.dumps(_round(result, 5), indent=1))
    p = result["prediction"]
    summary = {
        "status": p["status"],
        "same_condition": {k: p["same_condition"][k] for k in ("holds", "cells")},
        "failing_same_condition_cells": len(p["same_condition"]["failing"]),
        "other_condition": {
            c: {k: v[k] for k in ("cells", "worse_cells")}
            for c, v in p["other_condition"]["cases"].items()
        },
        "holds": p["holds"],
    }
    print(json.dumps(_round(summary, 4), indent=1))
    print(f"written to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
