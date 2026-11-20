"""C1b: does a context larger than 3x3 predict the flips of originals better?

Models of the flip probability (cdpbpc.context) are learned on originals of some
templates and judged by their log-loss on reads of other templates: constant rate, 3x3
table (the BPC codebook), back-off tables with 13, 21 and 25 bits, ring logistic,
boosted trees on the 5x5 and 7x7 windows; and the per-position ceiling, which predicts
each read from the other prints of the same template. Every hyperparameter is chosen on
validation templates. Measures on the test templates (methods.md §5.3): log-loss per
symbol read, predicted disagreement between two reads, remaining reproducible share
rho_left, calibration; 95 % intervals from a bootstrap on templates.

Indigo: scanner HPI55, session 1, 10 print runs. Train on templates 433-1440, validate
on 289-432, test on 1-288; transfer lines on HPI76 and on the other HPI55 sessions.
Scantrust: train on unique originals 1-3000, validate on 3001-3500, test on unique
4001-5000 (same session) and on the 50 batch templates x 50 prints (other session).

Decision (plan): the 25-bit table cuts the validation log-loss of the 3x3 table by at
least 5 % on both datasets.

Originals only: no file or image of fakes is opened.

Usage:
    python -m experiments.c1b_context --config configs/c1b.yaml
    python -m experiments.c1b_context --figure results/c1b_<hash>.json   # redraw only
Writes results/c1b_<config hash>.json and report/figures/fig1_right.pdf.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from cdpbpc import context as cx
from cdpbpc.io import (
    INDIGO_N_TEMPLATES,
    INDIGO_SIDE,
    load_compact,
    load_scantrust,
    read_compact_header,
    read_manifest,
)
from cdpbpc.plotting import TEST_SET_COLORS, setup
from experiments.e0_sanity import _round, git_state

ROOT = Path(__file__).resolve().parents[1]
STAGE = "dev"  # only originals and templates are read; the stage cut does not touch them
TABLES = ("table_9bits", "table_13bits", "table_21bits", "table_25bits")
LEARNED = ("logistic_ring", "trees_5x5", "trees_7x7")
MODELS = ("constant",) + TABLES + LEARNED
CHUNK = 16  # templates per chunk when building features of the learned models


def _between(ids, lo_hi):
    lo, hi = lo_hi
    return (ids >= lo) & (ids <= hi)


def _interval(v) -> list:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])] if v.size else [None, None]


class Data:
    """Templates t (N, L, L), flip counts K (N, L, L) over n reads, template ids."""

    def __init__(self, ids, t, K, n):
        self.ids, self.t, self.K, self.n = np.asarray(ids), t, K, int(n)

    def subset(self, mask):
        return Data(self.ids[mask], self.t[mask], self.K[mask], self.n)

    def draw(self, size, rng):
        if size >= len(self.ids):
            return self
        keep = np.zeros(len(self.ids), bool)
        keep[rng.choice(len(self.ids), size, replace=False)] = True
        return self.subset(keep)


# ------------------------------------------------------------------------------- models


class Models:
    """Every model of methods.md §5.2, fitted on train, tuned on validation."""

    def __init__(self, train: Data, val: Data, cfg: dict, ds_cfg: dict, rng, log):
        m = cfg["margin"]
        self.margin = m
        self.counts = cx.table_counts(train.t, train.K, train.n, margin=2)
        self.constant = float(self.counts.k[9].sum() / self.counts.n[9].sum())
        self.masses, mass_losses = cx.select_masses(
            [(self.counts, val.t, val.K, val.n)], grid=cfg["backoff_masses"], margin=m
        )
        log(f"tables counted; back-off masses {self.masses}")
        self.selection = {"backoff_masses": self.masses, "backoff_val_loss": mass_losses}

        val_l = val.draw(ds_cfg["learned_val_templates"], rng)
        Kv = cx.crop(val_l.K, m).ravel()

        tr_lr = train.draw(ds_cfg["logistic_templates"], rng)
        F_tr = self.ring_features(tr_lr.t)
        F_va = self.ring_features(val_l.t)
        lr_losses = {}
        best = None
        for C in cfg["logistic_C"]:
            model = cx.fit_ring_logistic(F_tr, cx.crop(tr_lr.K, m).ravel(), tr_lr.n, C=C)
            lr_losses[C] = cx.log_loss(Kv, val_l.n, model.predict_proba(F_va)[:, 1])
            if best is None or lr_losses[C] < lr_losses[best[0]]:
                best = (C, model)
        self.logistic = best[1]
        self.selection["logistic_C"] = best[0]
        self.selection["logistic_val_loss"] = lr_losses
        del F_tr, F_va
        log(f"ring logistic fitted on {len(tr_lr.ids)} templates, C = {best[0]}")

        tr_gb = train.draw(ds_cfg["trees_templates"], rng)
        self.trees = {}
        for name, radius in (("trees_5x5", 2), ("trees_7x7", 3)):
            F = cx.window_bits(tr_gb.t, radius, m)
            model = cx.fit_trees(F, cx.crop(tr_gb.K, m).ravel(), tr_gb.n, seed=0, **cfg["trees"])
            del F
            losses = cx.trees_stage_losses(model, cx.window_bits(val_l.t, radius, m), Kv, val_l.n)
            n_iter = int(np.argmin(losses)) + 1
            self.trees[name] = (model, radius, n_iter)
            self.selection[f"{name}_n_iter"] = n_iter
            self.selection[f"{name}_val_loss"] = float(losses[n_iter - 1])
            log(f"{name} fitted on {len(tr_gb.ids)} templates, {n_iter} trees kept")

        # Validation losses of every model on the same draw, to pick the ceiling's q.
        self.val_loss = {
            k: cx.log_loss(Kv, val_l.n, self.predict(k, val_l.t).ravel()) for k in MODELS
        }
        self.best = min((k for k in MODELS if k != "constant"), key=self.val_loss.get)
        self.selection["val_loss_learned_draw"] = self.val_loss
        self.selection["best_model"] = self.best
        # Decision of the plan: tables on the whole validation range.
        idx = cx.context_index(val.t, cx.TOP_BITS, m)
        Kva = cx.crop(val.K, m)
        self.val_tables = {
            k: cx.log_loss(Kva, val.n, cx.backoff_predict(self.counts, self.masses, idx, b))
            for k, b in zip(TABLES, (9, 13, 21, 25))
        }
        self.ceiling_a = None
        if val_l.n > 1:
            q = self.predict(self.best, val_l.t)
            Kc = cx.crop(val_l.K, m)
            ll = {a: cx.ceiling_log_loss(Kc, val_l.n, q, a) for a in cfg["ceiling_a"]}
            self.ceiling_a = min(ll, key=ll.get)
            self.selection["ceiling_a"] = self.ceiling_a
            self.selection["ceiling_val_loss"] = ll

    def ring_features(self, t):
        out = []
        for s in range(0, len(t), CHUNK):
            idx = cx.context_index(t[s : s + CHUNK], cx.TOP_BITS, self.margin).ravel()
            p9 = cx.backoff_predict(self.counts, self.masses, idx, bits=9)
            out.append(cx.ring_features(idx, p9))
        return np.concatenate(out)

    def predict(self, name: str, t: np.ndarray) -> np.ndarray:
        """q of model `name` at the judged positions of templates t: (N, H, H) float32."""
        h = t.shape[-1] - 2 * self.margin
        out = np.empty((len(t), h, h), np.float32)
        for s in range(0, len(t), CHUNK):
            tc = t[s : s + CHUNK]
            if name == "constant":
                q = np.full((len(tc), h, h), self.constant)
            elif name in TABLES:
                bits = int(name.split("_")[1].removesuffix("bits"))
                idx = cx.context_index(tc, cx.TOP_BITS, self.margin)
                q = cx.backoff_predict(self.counts, self.masses, idx, bits)
            elif name == "logistic_ring":
                q = self.logistic.predict_proba(self.ring_features(tc))[:, 1]
            else:
                model, radius, n_iter = self.trees[name]
                q = cx.trees_predict(model, cx.window_bits(tc, radius, self.margin), n_iter)
            out[s : s + len(tc)] = np.reshape(q, (len(tc), h, h))
        return out


# ----------------------------------------------------------------------------- measures


def evaluate(
    models: Models, data: Data, n_boot: int, rng, ceiling: str | None, ceiling_grid
) -> dict:
    """Measures of every model on one test set, with bootstrap intervals on templates.

    ceiling: None, "validation" (a chosen on validation) or "halves" (a chosen in
    ceiling_grid on one half of the test templates by rank and applied to the other).
    """
    m = models.margin
    K = cx.crop(data.K, m)
    n = data.n
    N = len(data.ids)
    W = np.stack([np.bincount(d, minlength=N) for d in rng.integers(0, N, (n_boot, N))])
    W = W.astype(np.float64)
    Wsum = W.sum(1)
    per_ll, per_dm = {}, {}
    out = {"n_templates": N, "n_reads": n, "positions_per_template": int(K[0].size)}
    dobs = None
    if n > 1:
        dobs = cx.per_template_disagreement(K, n)
        out["D_obs"] = float(dobs.mean())
    keep_q = None
    for name in MODELS:
        q = models.predict(name, data.t)
        per_ll[name] = cx.per_template_log_loss(K, n, q)
        per_dm[name] = cx.per_template_predicted_disagreement(q)
        res = {
            "logloss": float(per_ll[name].mean()),
            "logloss_ci": _interval(W @ per_ll[name] / Wsum),
            "D_model": float(per_dm[name].mean()),
        }
        if dobs is not None:
            res["rho_left"] = float(1 - dobs.mean() / per_dm[name].mean())
            res["rho_left_ci"] = _interval(1 - (W @ dobs) / (W @ per_dm[name]))
        if name in ("table_9bits", "table_25bits", "trees_7x7"):
            res["calibration"] = cx.calibration(K, n, q)
            reliable = q <= 0.01
            res["share_q_le_0.01"] = float(reliable.mean())
            hits = int(reliable.sum())
            res["flip_rate_where_q_le_0.01"] = (
                float(K[reliable].sum() / (hits * n)) if hits else None
            )
        out[name] = res
        if name == models.best:
            keep_q = q
    base = per_ll["table_9bits"]
    for name in MODELS:
        out[name]["gain_vs_3x3"] = float(1 - per_ll[name].mean() / base.mean())
        out[name]["gain_vs_3x3_ci"] = _interval(1 - (W @ per_ll[name]) / (W @ base))
        if dobs is not None:
            r9 = 1 - dobs.mean() / per_dm["table_9bits"].mean()
            out[name]["rho_cut_vs_3x3"] = float(1 - out[name]["rho_left"] / r9)
    if ceiling and n > 1:

        def per_template(a, rows):
            return np.array([cx.ceiling_log_loss(K[i], n, keep_q[i], a) for i in rows])

        if ceiling == "validation":
            a = models.ceiling_a
            per_c = per_template(a, range(N))
            info = {"a": a, "chosen_on": "validation templates"}
        else:
            halves = [np.arange(N)[0::2], np.arange(N)[1::2]]
            per_c = np.empty(N)
            chosen = []
            for fit, apply in (halves, halves[::-1]):
                losses = {a: per_template(a, fit).mean() for a in ceiling_grid}
                a = min(losses, key=losses.get)
                per_c[apply] = per_template(a, apply)
                chosen.append(a)
            info = {"a": chosen, "chosen_on": "the other half of the test templates"}
        out["ceiling"] = {
            "q_from": models.best,
            **info,
            "logloss": float(per_c.mean()),
            "logloss_ci": _interval(W @ per_c / Wsum),
            "gain_vs_3x3": float(1 - per_c.mean() / base.mean()),
        }
    return out


# --------------------------------------------------------------------------------- data


def indigo_data(data_dir: Path, cfg: dict, log) -> dict:
    pack = data_dir / "indigo_compact"
    tpl = load_compact(pack / "templates_original.npz", stage=STAGE)
    T = np.zeros((INDIGO_N_TEMPLATES + 1, INDIGO_SIDE, INDIGO_SIDE), np.uint8)
    T[tpl.ids] = tpl.bits

    def reads(names, ids=None):
        for name in names:
            if read_compact_header(pack / f"{name}.npz")[1]["class"] != "original":
                raise ValueError(f"{name} is not a file of originals: C1b reads no fake")
        common = set.intersection(
            *[set(read_compact_header(pack / f"{nm}.npz")[0].tolist()) for nm in names]
        )
        if ids is not None:
            common &= set(np.asarray(ids).tolist())
        ids = np.array(sorted(common))
        K = np.zeros((len(ids), INDIGO_SIDE, INDIGO_SIDE), np.uint8)
        for name in names:
            f = load_compact(pack / f"{name}.npz", stage=STAGE)
            K += f.bits[np.searchsorted(f.ids, ids)] != T[ids]
        return Data(ids, T[ids], K, len(names))

    allp = reads(cfg["prints"])
    log(f"Indigo: {len(allp.ids)} templates in all {allp.n} prints")
    sets = {k: allp.subset(_between(allp.ids, cfg[k])) for k in ("train", "validation", "test")}
    sets["transfer"] = {
        name: reads(files, sets["test"].ids) for name, files in cfg["transfer"].items()
    }
    return sets


def scantrust_data(data_dir: Path, cfg: dict, log) -> dict:
    manifest = read_manifest(data_dir / "manifest.csv")
    orig = manifest[manifest["class"] == "original"]
    uniq = orig[orig["dataset"] == "unique"].sort_values("template_id")
    tid = uniq["template_id"].to_numpy(int)
    wanted = _between(tid, cfg["train"]) | _between(tid, cfg["validation"])
    wanted |= _between(tid, cfg["test_unique"])
    u = load_scantrust(data_dir, uniq[wanted], stage=STAGE)
    b = load_scantrust(
        data_dir, orig[orig["dataset"] == "batch"].sort_values("template_id"), stage=STAGE
    )
    for imgs in (u, b):
        if not (imgs.rows["class"] == "original").all():
            raise ValueError("C1b reads no fake")
    uid = u.rows["template_id"].to_numpy(int)
    one = Data(uid, u.t, (u.bits != u.t).astype(np.uint8), 1)
    bid = b.rows["template_id"].to_numpy(int)
    ids = np.unique(bid)
    sizes = {int((bid == i).sum()) for i in ids}
    if len(ids) != 50 or sizes != {50}:
        raise ValueError(f"expected 50 batch templates x 50 prints, got {len(ids)} and {sizes}")
    t50 = np.stack([b.t[bid == i][0] for i in ids])
    K50 = np.stack([(b.bits[bid == i] != t50[j]).sum(0) for j, i in enumerate(ids)])
    log(f"Scantrust: {len(uid)} unique originals, batch {len(ids)} x 50")
    return {
        "train": one.subset(_between(uid, cfg["train"])),
        "validation": one.subset(_between(uid, cfg["validation"])),
        "test_unique": one.subset(_between(uid, cfg["test_unique"])),
        "test_batch": Data(ids, t50, K50.astype(np.uint8), 50),
    }


# ------------------------------------------------------------------------------- figure

FIG_MODELS = (
    ("constant", "constant rate"),
    ("table_9bits", "3x3 table (BPC)"),
    ("table_25bits", "5x5 table, back-off"),
    ("trees_7x7", "7x7 boosted trees"),
    ("ceiling", "per-position ceiling"),
)
FIG_SETS = (
    ("indigo", "test", "Indigo HPI55", "o"),
    ("scantrust", "test_unique", "Scantrust, same session", "s"),
    ("scantrust", "test_batch", "Scantrust, other session", "^"),
)


def plot_fig1_right(result: dict, path: Path) -> None:
    """Held-out log-loss of each model, one marker per test set, 95 % intervals."""
    import matplotlib.pyplot as plt

    setup()
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    y = np.arange(len(FIG_MODELS))[::-1]
    offsets = (-0.2, 0.0, 0.2)
    for (ds, key, label, marker), dy, color in zip(FIG_SETS, offsets, TEST_SET_COLORS.values()):
        res = result[ds]["tests"][key]
        for yi, (model, _) in zip(y, FIG_MODELS):
            if model not in res:
                continue
            v, (lo, hi) = res[model]["logloss"], res[model]["logloss_ci"]
            ax.errorbar(
                v,
                yi + dy,
                xerr=[[v - lo], [hi - v]],
                fmt=marker,
                color=color,
                ms=4,
                lw=1,
                capsize=0,
                label=label if model == "table_9bits" else None,
            )
    ax.set_yticks(y, [name for _, name in FIG_MODELS])
    ax.set_xlabel("held-out log-loss per symbol read (nats)")
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------------- main


def run(name, sets, tests, cfg, ds_cfg, rng, log) -> dict:
    t0 = time.time()
    models = Models(sets["train"], sets["validation"], cfg, ds_cfg, rng, log)
    out = {
        "train_templates": len(sets["train"].ids),
        "validation_templates": len(sets["validation"].ids),
        "selection": models.selection,
        "validation_tables": models.val_tables,
        "validation_gain_25bits": 1
        - models.val_tables["table_25bits"] / models.val_tables["table_9bits"],
        "tests": {},
    }
    for key, (data, ceiling) in tests.items():
        out["tests"][key] = evaluate(models, data, cfg["n_boot"], rng, ceiling, cfg["ceiling_a"])
        r = out["tests"][key]
        log(
            f"{name} / {key}: log-loss 3x3 {r['table_9bits']['logloss']:.4f}, "
            f"5x5 {r['table_25bits']['logloss']:.4f}, trees 7x7 {r['trees_7x7']['logloss']:.4f}"
        )
    out["seconds"] = round(time.time() - t0)
    return out


def decisions(res: dict, cfg: dict) -> dict:
    gains = {ds: res[ds]["validation_gain_25bits"] for ds in ("indigo", "scantrust")}
    dec = {
        "rule": "25-bit table cuts the validation log-loss of the 3x3 table by >= 5 % on both datasets",
        "validation_gain": gains,
        "keep_5x5_in_fig1": all(g >= cfg["decision"]["min_gain"] for g in gains.values()),
    }
    h = cfg["prediction_H1b"]
    tests = {"indigo": "test", "scantrust": "test_batch"}
    h1b = {}
    for ds, key in tests.items():
        r = res[ds]["tests"][key]["table_25bits"]
        h1b[ds] = {
            "gain": r["gain_vs_3x3"],
            "gain_ci": r["gain_vs_3x3_ci"],
            "rho_cut": r["rho_cut_vs_3x3"],
        }
    h1b["unique_gain"] = res["scantrust"]["tests"]["test_unique"]["table_25bits"]["gain_vs_3x3"]
    h1b["holds"] = (
        all(
            h1b[ds]["gain"] >= h["min_gain"] and h1b[ds]["rho_cut"] >= h["min_rho_cut"]
            for ds in tests
        )
        and h1b["unique_gain"] >= h["min_gain"]
    )
    dec["H1b"] = h1b
    return dec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C1b: flip models with a larger context")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "c1b.yaml")
    parser.add_argument("--figure", type=Path, help="only redraw Fig. 1 right from a result")
    args = parser.parse_args(argv)
    fig_path = ROOT / "report" / "figures" / "fig1_right.pdf"
    if args.figure:
        plot_fig1_right(json.loads(args.figure.read_text()), fig_path)
        print(f"written to {fig_path.relative_to(ROOT)}")
        return 0

    text = args.config.read_text()
    cfg = yaml.safe_load(text)
    data_dir = ROOT / cfg["data_dir"]
    start = time.time()

    def log(msg):
        print(f"[{time.time() - start:6.0f} s] {msg}", flush=True)

    rng = np.random.default_rng(cfg["seed"])
    result = {
        "experiment": "c1b_context",
        "data": "originals and templates only",
        "git": git_state(),
        "config": cfg,
    }
    s = indigo_data(data_dir, cfg["indigo"], log)
    tests = {"test": (s["test"], "validation")}
    tests.update({k: (v, None) for k, v in s["transfer"].items()})
    result["indigo"] = run("Indigo", s, tests, cfg, cfg["indigo"], rng, log)
    del s
    s = scantrust_data(data_dir, cfg["scantrust"], log)
    tests = {"test_unique": (s["test_unique"], None), "test_batch": (s["test_batch"], "halves")}
    result["scantrust"] = run("Scantrust", s, tests, cfg, cfg["scantrust"], rng, log)
    result["decisions"] = decisions(result, cfg)
    result["seconds"] = round(time.time() - start)
    out = ROOT / "results" / f"c1b_{hashlib.sha1(text.encode()).hexdigest()[:8]}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(_round(result, 5), indent=1))
    plot_fig1_right(json.loads(out.read_text()), fig_path)
    print(json.dumps(_round(result["decisions"], 4), indent=1))
    print(f"written to {out.relative_to(ROOT)} and {fig_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
