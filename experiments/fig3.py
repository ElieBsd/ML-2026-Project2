"""Fig. 3 of the report: where a fake gives itself away, predicted without any fake.

For each reliability class of the patterns (flip probability P_b of the BPC codebook),
the gap between the flip rate of fakes and that of originals: predicted from the two
passes of a copy (line and band) and observed on real fakes (points with 95 % bootstrap
intervals on templates).
(a) Indigo, iPhone: the attacker estimates the template from one print; the fake errs
    where the estimate errs, on reliable patterns.
(b) Scantrust batch: the attacker averages several prints, so the estimate is right on
    reliable patterns and the fake errs on unreliable ones.

No data is read: only the test-stage result of the two-pass experiment (c3b_two_pass),
which is a file of per-class counts. By default it is the result of the frozen config.

Usage:
    python -m experiments.fig3
    python -m experiments.fig3 --c3b results/c3b_<hash>_test.json
Writes report/figures/fig3.pdf.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cdpbpc.plotting import setup

ROOT = Path(__file__).resolve().parents[1]
PHONE = "iPhone12Pro"
AVERAGING = (("otsu+unsharp", "#eb6834", "s"), ("deep+unsharp", "#1baf7a", "^"))
PHONE_COLOR = "#2a78d6"
PREDICTION_COLOR = "#333333"
UNRELIABLE_SHADE = "#efefef"


def result_path(name: str) -> Path:
    """Test-stage result file written by an experiment for its frozen config."""
    text = (ROOT / "configs" / f"{name}.yaml").read_text()
    return ROOT / "results" / f"{name}_{hashlib.sha1(text.encode()).hexdigest()[:8]}_test.json"


def load(path: Path) -> dict:
    result = json.loads(path.read_text())
    if result.get("experiment") != "c3b_two_pass":
        raise SystemExit(f"{path} is not a result of c3b_two_pass")
    if result.get("stage") != "test":
        raise SystemExit(f"{path} is not a test-stage result")
    return result


def _bound(x: float) -> str:
    return f"{x:g}".lstrip("0") if 0 < x < 1 else f"{x:g}"


def class_labels(by_class: list) -> list[str]:
    """Short tick labels of the reliability classes, from their bounds."""
    labels = []
    for k, row in enumerate(by_class):
        lo, hi = row["class"]
        if k == 0:
            labels.append(f"<{_bound(hi)}")
        elif k == len(by_class) - 1:
            labels.append(f">{_bound(lo)}")
        else:
            labels.append(f"{_bound(lo)}-{_bound(hi)}")
    return labels


def series(by_class: list, key: str) -> dict:
    """One gap per class (gap_pred or gap_obs) and its interval, in percentage points."""
    return {
        "value": np.array([100 * r[key] for r in by_class]),
        "low": np.array([100 * r[f"{key}_ci"][0] for r in by_class]),
        "high": np.array([100 * r[f"{key}_ci"][1] for r in by_class]),
    }


def first_unreliable(by_class: list) -> int:
    """Index of the first class whose patterns flip at least half the time."""
    return next(k for k, r in enumerate(by_class) if r["class"][0] >= 0.5)


def batch_prediction(c3b: dict) -> list:
    """The prediction does not depend on the attack: check it, then return it once."""
    batch = c3b["scantrust"]["batch"]
    rows = [batch[a]["by_class"] for a, _, _ in AVERAGING]
    for other in rows[1:]:
        if not np.allclose([r["gap_pred"] for r in rows[0]], [r["gap_pred"] for r in other]):
            raise SystemExit("the averaging attacks do not have the same prediction")
    return rows[0]


def _shade_unreliable(ax, start: int, n: int, label: bool) -> None:
    ax.axvspan(start - 0.5, n - 0.5, color=UNRELIABLE_SHADE, zorder=0, lw=0)
    ax.axhline(0, color="#9a9a9a", lw=0.6, zorder=1)
    if label:
        kw = {"transform": ax.get_xaxis_transform(), "fontsize": 6.5, "color": "#555555"}
        ax.text(start / 2 - 0.5, 0.97, "reliable patterns", ha="center", va="top", **kw)
        ax.text((start + n - 1) / 2, 0.97, "unreliable patterns", ha="center", va="top", **kw)


def _prediction(ax, by_class: list, label: str) -> None:
    x = np.arange(len(by_class))
    p = series(by_class, "gap_pred")
    ax.fill_between(x, p["low"], p["high"], color=PREDICTION_COLOR, alpha=0.18, lw=0, zorder=2)
    ax.plot(x, p["value"], color=PREDICTION_COLOR, lw=1.2, zorder=3, label=label)


def _observed(ax, by_class: list, dx: float, color: str, marker: str, label: str) -> None:
    x = np.arange(len(by_class)) + dx
    o = series(by_class, "gap_obs")
    ax.errorbar(
        x,
        o["value"],
        yerr=[o["value"] - o["low"], o["high"] - o["value"]],
        fmt=marker,
        color=color,
        ms=3.5,
        mec="white",
        mew=0.4,
        elinewidth=0.8,
        capsize=1.2,
        zorder=4,
        label=label,
    )


def _legend(fig, top, bottom) -> None:
    """One legend under both panels; the prediction is drawn as its line over its band."""
    from matplotlib.patches import Patch

    lines, names = [], []
    for ax in (top, bottom):
        for handle, name in zip(*ax.get_legend_handles_labels()):
            if name in names:
                continue
            if name == "predicted without fakes":
                band = Patch(color=PREDICTION_COLOR, alpha=0.18, lw=0)
                handle = (band, handle)
            lines.append(handle)
            names.append(name)
    fig.legend(
        lines,
        names,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.035),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )


def plot_fig3(c3b: dict, path: Path) -> None:
    """Two panels stacked, at the width of one column; same axes so gaps compare."""
    import matplotlib.pyplot as plt

    setup()
    phone = c3b["indigo"]["mobile"][PHONE]
    batch = c3b["scantrust"]["batch"]
    pred_batch = batch_prediction(c3b)
    labels = class_labels(phone["by_class"])
    n, start = len(labels), first_unreliable(phone["by_class"])

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True, sharey=True)
    _shade_unreliable(top, start, n, label=True)
    _prediction(top, phone["by_class"], "predicted without fakes")
    _observed(top, phone["by_class"], 0.0, PHONE_COLOR, "o", "observed, iPhone fakes")
    corr = phone["per_pattern"]["P_obs"]
    lo, hi = corr["corr_ci"]
    top.text(
        0.02,
        0.05,
        f"per-pattern correlation\n{corr['corr']:.2f} [{lo:.2f}, {hi:.2f}]",
        transform=top.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#555555",
    )
    top.set_title("(a) Indigo, iPhone: template estimated from one print", loc="left")

    _shade_unreliable(bottom, start, n, label=False)
    _prediction(bottom, pred_batch, "predicted without fakes")
    for (attack, color, marker), dx in zip(AVERAGING, (-0.13, 0.13)):
        _observed(bottom, batch[attack]["by_class"], dx, color, marker, f"observed, {attack}")
    bottom.set_title("(b) Scantrust batch: template averaged over prints", loc="left")

    bottom.set_xticks(range(n), labels, rotation=30, ha="right", rotation_mode="anchor")
    bottom.set_xlim(-0.5, n - 0.5)
    bottom.set_xlabel(r"flip probability $P_b(\omega)$ of the pattern on originals")
    from matplotlib.ticker import MultipleLocator

    bottom.yaxis.set_major_locator(MultipleLocator(2))
    for ax in (top, bottom):
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="y", lw=0.4, alpha=0.35)
    fig.supylabel("flip rate, fakes minus originals (points)", fontsize=8, x=0.02)
    fig.tight_layout(h_pad=0.8)
    _legend(fig, top, bottom)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fig. 3 from the two-pass test result")
    parser.add_argument("--c3b", type=Path, help="c3b_two_pass test-stage result")
    args = parser.parse_args(argv)
    path = args.c3b or result_path("c3b")
    c3b = load(path)
    print(f"{path.name}: git {c3b['git']}")
    out = ROOT / "report" / "figures" / "fig3.pdf"
    plot_fig3(c3b, out)
    print(f"written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
