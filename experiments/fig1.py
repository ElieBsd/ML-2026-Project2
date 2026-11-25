"""Fig. 1 of the report: what the 3x3 BPC model misses, in one full-width figure.

Left: the flip noise predicted by the BPC, split by where it comes from (template, print,
capture), with 95 % bootstrap intervals on templates. Right: held-out log-loss per symbol
read of the flip models (constant rate, 3x3 table, 5x5 table with back-off, 7x7 boosted
trees, per-position ceiling), one marker per test set.

No data is read: only the result files of the noise experiment (c1a_noise) and of the
flip-model experiment (c1b_context), which use originals and templates only. By default
the result files are the ones of the current configs (same name as their scripts write).

Usage:
    python -m experiments.fig1
    python -m experiments.fig1 --noise results/c1a_<hash>.json --models results/c1b_<hash>.json
Writes report/figures/fig1.pdf.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cdpbpc.plotting import NOISE_COLORS, TEST_SET_COLORS, setup

ROOT = Path(__file__).resolve().parents[1]
NOISE_ROWS = (("Indigo HPI76*", "HPI76_scans_session0"), ("Indigo HPI55", "HPI55"))
NOISE_NAMES = {
    "template": "reproducible from print to print",
    "print": "specific to one print",
    "capture": "capture (scan or photo)",
    "print_and_capture": "print and capture (not split)",
}
MODELS = (
    ("constant", "constant rate"),
    ("table_9bits", "3x3 table (BPC)"),
    ("table_25bits", "5x5 table, back-off"),
    ("trees_7x7", "7x7 boosted trees"),
    ("ceiling", "per-position ceiling"),
)
TEST_SETS = (
    ("indigo", "test", "Indigo HPI55", "o"),
    ("scantrust", "test_unique", "Scantrust, same session", "s"),
    ("scantrust", "test_batch", "Scantrust, other session", "^"),
)


def result_path(name: str) -> Path:
    """Result file written by an experiment for its current config."""
    text = (ROOT / "configs" / f"{name}.yaml").read_text()
    return ROOT / "results" / f"{name}_{hashlib.sha1(text.encode()).hexdigest()[:8]}.json"


def load(path: Path, experiment: str) -> dict:
    result = json.loads(path.read_text())
    if result.get("experiment") != experiment:
        raise SystemExit(f"{path} is not a result of {experiment}")
    if not result["data"].startswith("originals"):
        raise SystemExit(f"{path} does not say it reads originals only")
    return result


def noise_rows(noise: dict) -> list:
    """One row per dataset: its parts of the noise and the intervals of its rho values."""
    ind, rho = noise["indigo"], noise["scantrust"]["batch"]["rho"]
    rows = []
    for label, key in NOISE_ROWS:
        d = ind["decomposition"][key]
        parts = [(part, d[part]["value"]) for part in ("template", "print", "capture")]
        cis = [ind["within"][d["prints"]]["rho_ci"], ind["within"][d["reads"]]["rho_ci"]]
        rows.append((label, parts, cis))
    parts = [("template", rho["value"]), ("print_and_capture", 1 - rho["value"])]
    rows.append(("Scantrust", parts, [rho["ci"]]))
    return rows


def draw_noise(ax, noise: dict) -> None:
    seen = set()
    rows = noise_rows(noise)
    for y, (label, parts, cis) in enumerate(rows):
        left = 0.0
        for part, value in parts:
            width = 100 * value
            ax.barh(
                y,
                width,
                left=left,
                color=NOISE_COLORS[part],
                edgecolor="white",
                linewidth=0.5,
                label=None if part in seen else NOISE_NAMES[part],
            )
            seen.add(part)
            tone = "white" if part in ("template", "print") else "black"
            ax.text(left + width / 2, y, f"{width:.0f}", ha="center", va="center", color=tone)
            left += width
        for lo, hi in cis:
            ax.errorbar(
                50 * (lo + hi),
                y + 0.33,
                xerr=50 * (hi - lo),
                fmt="none",
                ecolor="black",
                elinewidth=0.7,
                capsize=1.5,
            )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("share of the flip noise predicted by the BPC (%)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.45, -0.38),
        ncol=2,
        frameon=False,
        handlelength=1,
        columnspacing=0.8,
    )


def draw_models(ax, models: dict) -> None:
    y = np.arange(len(MODELS))[::-1]
    offsets = (-0.2, 0.0, 0.2)
    for (ds, key, label, marker), dy, color in zip(TEST_SETS, offsets, TEST_SET_COLORS.values()):
        res = models[ds]["tests"][key]
        for yi, (model, _) in zip(y, MODELS):
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
    ax.set_yticks(y, [name for _, name in MODELS])
    ax.set_xlabel("held-out log-loss per symbol read (nats)")
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.35, -0.38),
        ncol=2,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def plot_fig1(noise: dict, models: dict, path: Path) -> None:
    """Both panels side by side, at the width of a two-column page."""
    import matplotlib.pyplot as plt

    setup()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.4), width_ratios=(1, 1))
    draw_noise(left, noise)
    draw_models(right, models)
    left.set_title("(a) where the predicted noise comes from", loc="left")
    right.set_title("(b) flip models", loc="left")
    fig.tight_layout(w_pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fig. 1 from the two result files")
    parser.add_argument("--noise", type=Path, help="c1a_noise result")
    parser.add_argument("--models", type=Path, help="c1b_context result")
    args = parser.parse_args(argv)
    noise_path = args.noise or result_path("c1a")
    models_path = args.models or result_path("c1b")
    noise = load(noise_path, "c1a_noise")
    models = load(models_path, "c1b_context")
    for path, result in ((noise_path, noise), (models_path, models)):
        print(f"{path.name}: git {result['git']}")
    out = ROOT / "report" / "figures" / "fig1.pdf"
    plot_fig1(noise, models, out)
    print(f"written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
