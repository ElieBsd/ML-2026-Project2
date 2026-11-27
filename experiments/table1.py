"""Table 1 of the report: which test against which attacker, at the operating point.

P_fa at a target P_miss of 5 %, with the threshold set on originals only (conformal
threshold) and the test blocks judged once. Each cell gives P_fa in %, then on a second
line its 95 % interval and the realized P_miss in %. Columns: Indigo phones (iPhone,
Samsung, capture 1), Indigo scanner with fakes reprinted on the printer of the originals
or on the other one (range over the two printers of the originals and the two
estimates), Scantrust batch_cdp averaging attacks (rates per template, bootstrap over
templates).

Rows:
- scores without fakes: Hamming distance, Pearson correlation, posterior
  log-likelihood, masked Hamming distance of the thesis (mu = 0.01), binned Mahalanobis;
- extensions without fakes: masks learned with a 5x5 or 3x3 context and the plain
  Hamming distance on the same positions; likelihood ratio against the averaging
  attacker (vote); minimum of the masked Hamming distance over three phone captures,
  next to one capture of the same experiment;
- oracle (trained with fakes): logistic regression trained on fakes of the same
  condition, and on fakes of another condition (other phone, the other printer's fakes,
  another Scantrust session).

No data is read: only the test-stage results of the detection, oracle, vote, three-capture
and context-mask experiments. By default they are the results of the frozen configs.

Usage:
    python -m experiments.table1
Writes report/table1.tex (a full-width table, input by report/main.tex).
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALPHA = "alpha_0.05"
PHONES = ("iPhone12Pro", "SamsungGN20U")
PRINTERS = ("HPI55", "HPI76")
ATTACKS = ("otsu+unsharp", "deep+unsharp")
COLUMNS = ("iPhone", "Samsung", "scanner_same", "scanner_other", *ATTACKS)
SOURCES = {
    "c3a": "c3a_detection",
    "c3c": "c3c_oracle",
    "p1": "p1_vote",
    "p2": "p2_best_of_3",
    "p3": "p3_context_mask",
}
SCORES = (
    ("hamming", "Hamming distance"),
    ("pcor", "Pearson correlation"),
    ("pll", "posterior log-likelihood"),
    ("masked_hamming", r"masked Hamming ($\mu=0.01$, thesis)"),
    ("binned_mahalanobis", "binned Mahalanobis"),
)
MASKS = (
    ("mask_5x5", r"mask, $5\times5$ context$^\dagger$"),
    ("mask_3x3", r"mask, $3\times3$ context$^\dagger$"),
    ("hamming_same_positions", r"Hamming, same positions$^\dagger$"),
)
SHADE_MAX = 55  # percent of the shading colour at P_fa = 1


@dataclass(frozen=True)
class Cell:
    """P_fa and P_miss of one cell, each a value or a (low, high) range, and the CI."""

    p_fa: tuple
    p_miss: tuple
    ci: tuple | None = None

    @property
    def worst(self) -> float:
        return max(self.p_fa)


def point(res: dict, ci_key: str = "p_fa_ci") -> Cell:
    return Cell((res["p_fa"],), (res["p_miss"],), tuple(res[ci_key]))


def spread(cells: list) -> Cell:
    """Range over several cells (no interval: they are different conditions)."""
    fa = [v for c in cells for v in c.p_fa]
    miss = [v for c in cells for v in c.p_miss]
    return Cell((min(fa), max(fa)), (min(miss), max(miss)))


def result_path(name: str) -> Path:
    """Test-stage result file written by an experiment for its frozen config."""
    text = (ROOT / "configs" / f"{name}.yaml").read_text()
    return ROOT / "results" / f"{name}_{hashlib.sha1(text.encode()).hexdigest()[:8]}_test.json"


def load(path: Path, experiment: str) -> dict:
    result = json.loads(path.read_text())
    if result.get("experiment") != experiment:
        raise SystemExit(f"{path} is not a result of {experiment}")
    if result.get("stage") != "test":
        raise SystemExit(f"{path} is not a test-stage result")
    return result


def scanner_fakes(printer: str, same: bool, style: str) -> list[str]:
    """Keys of the fakes reprinted on the printer of the originals (same) or the other.

    style "short" is f<printer>/<estimate> (detection and mask results), "long" is
    HPI<printer>_EHPI<estimate> (oracle result).
    """
    out = []
    for fp in PRINTERS:
        if (fp == printer) != same:
            continue
        for est in PRINTERS:
            out.append(f"f{fp[3:]}/{est[3:]}" if style == "short" else f"{fp}_E{est}")
    return out


def image_row(cond: dict, key, short=True) -> dict:
    """Phone and scanner cells of a score in a detection-like result (indigo block)."""
    phones = {
        "iPhone": point(cond[f"mobile_{PHONES[0]}"]["fakes"]["EHPI55"][key][ALPHA]),
        "Samsung": point(cond[f"mobile_{PHONES[1]}"]["fakes"]["EHPI55"][key][ALPHA]),
    }
    for col, same in (("scanner_same", True), ("scanner_other", False)):
        phones[col] = spread(
            [
                point(cond[f"scanner_{p}"]["fakes"][fk][key][ALPHA])
                for p in PRINTERS
                for fk in scanner_fakes(p, same, "short")
            ]
        )
    return phones


def detection_rows(c3a: dict) -> list:
    rows = []
    for key, label in SCORES:
        cells = image_row(c3a["indigo"], key)
        for attack in ATTACKS:
            res = c3a["scantrust_batch"]["attacks"][attack][key][ALPHA]["conformal"]
            cells[attack] = point(res, "p_fa_ci_template_bootstrap")
        rows.append((label, cells))
    return rows


def extension_rows(p1: dict, p2: dict, p3: dict) -> list:
    rows = []
    for key, label in MASKS:
        cells = image_row(p3["indigo"], key)
        for attack in ATTACKS:
            res = p3["scantrust_batch"][attack][key][ALPHA]
            cells[attack] = point(res, "p_fa_ci_template_bootstrap")
        rows.append((label, cells))
    vote = {
        a: point(
            p1["scantrust_batch"]["attacks"][a]["lrt_vote"][ALPHA]["conformal"],
            "p_fa_ci_template_bootstrap",
        )
        for a in ATTACKS
    }
    rows.append(("likelihood ratio, averaging attacker", vote))
    for key, label in (
        ("single", r"masked Hamming, one capture$^\ddagger$"),
        ("min3", "masked Hamming, best of three captures"),
    ):
        res = p2["indigo_mobile"]
        cells = {
            col: point(res[ph]["results"]["masked_hamming"][key][ALPHA])
            for col, ph in zip(("iPhone", "Samsung"), PHONES)
        }
        rows.append((label, cells))
    return rows


def _logistic(c3c: dict, cond: str, row: str, fakes: str) -> Cell:
    res = c3c["conditions"][cond]["oracle"][row]["logistic"][fakes][ALPHA]["conformal"]
    return point(res)


def oracle_rows(c3c: dict) -> list:
    same, other = {}, {}
    for col, ph, other_ph in (
        ("iPhone", PHONES[0], PHONES[1]),
        ("Samsung", PHONES[1], PHONES[0]),
    ):
        same[col] = _logistic(c3c, f"mobile {ph}", f"trained on {ph}", "EHPI55")
        other[col] = _logistic(c3c, f"mobile {ph}", f"trained on {other_ph}", "EHPI55")
    for col, judged_same in (("scanner_same", True), ("scanner_other", False)):
        diag, cross = [], []
        for p in PRINTERS:
            judged = scanner_fakes(p, judged_same, "long")
            trained_other = scanner_fakes(p, not judged_same, "long")
            diag += [_logistic(c3c, f"scanner {p}", fk, fk) for fk in judged]
            cross += [
                _logistic(c3c, f"scanner {p}", row, fk) for row in trained_other for fk in judged
            ]
        same[col], other[col] = spread(diag), spread(cross)
    for attack in ATTACKS:
        same[attack] = _logistic(c3c, "batch", attack, attack)
        other[attack] = _logistic(c3c, "batch", "unique V_U (all attacks)", attack)
    return [
        ("logistic, same condition", same),
        (r"logistic, other condition$^\S$", other),
    ]


def other_session_by_c(c3c: dict) -> dict:
    """Range of P_fa over the grid of C, for the other-session oracle cells."""
    grid = c3c["conditions"]["batch"]["oracle"]["unique V_U (all attacks)"]["logistic_by_C"]
    out = {}
    for attack in ATTACKS:
        values = [grid[c][attack][ALPHA]["conformal"]["p_fa"] for c in grid]
        out[attack] = (min(values), max(values))
    chosen = {
        m["C"] for m in c3c["conditions"]["batch"]["oracle"]["unique V_U (all attacks)"]["models"]
    }
    return {"range": out, "chosen": sorted(chosen)}


def best_fake_free(blocks: list) -> dict:
    """Per column, the lowest P_fa (worst end of a range) among rows without fakes."""
    best = {}
    for _, rows in blocks:
        for _, cells in rows:
            for col, cell in cells.items():
                best[col] = min(best.get(col, 1.0), cell.worst)
    return best


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def span(values: tuple) -> str:
    lo, hi = min(values), max(values)
    return pct(lo) if pct(lo) == pct(hi) else f"{pct(lo)}--{pct(hi)}"


def shade(p_fa: float) -> int:
    """Strength of the cell shading, from 0 (no fake accepted) to SHADE_MAX."""
    return round(SHADE_MAX * min(max(p_fa, 0.0), 1.0))


def cell_tex(cell: Cell | None, bold: bool) -> str:
    if cell is None:
        return "--"
    top = span(cell.p_fa)
    if bold:
        top = rf"\textbf{{{top}}}"
    ci = f"[{pct(cell.ci[0])}, {pct(cell.ci[1])}]" if cell.ci else "range"
    return rf"\tcell{{{shade(cell.worst)}}}{{{top}}}{{{ci}}}{{{span(cell.p_miss)}}}"


def render(blocks: list, by_c: dict, sources: dict) -> str:
    """The whole table* environment, with its caption and its notes."""
    best = best_fake_free(blocks[:2])
    lines = [
        "% Written by experiments/table1.py from the test-stage results:",
        *(f"%   {name}: git {git}" for name, git in sources.items()),
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \definecolor{pfa}{HTML}{D6604D}",
        (
            r"  \providecommand{\tcell}[4]{\begin{tabular}[t]{@{}c@{}}"
            r"{\setlength{\fboxsep}{1.2pt}\colorbox{pfa!#1}{\makebox[4.6em]{\strut #2}}}\\[-2.5pt]"
            r"{\fontsize{5.5}{6}\selectfont #3\enspace\textit{#4}}\end{tabular}}"
        ),
        (
            r"  \caption{False-acceptance rate $\Pfa$ (\%) at a target $\Pmiss$ of 5\,\%:"
            r" threshold set on originals only, test templates judged once}"
        ),
        r"  \label{tab:detection}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3.5pt}",
        r"  \renewcommand{\arraystretch}{1.05}",
        r"  \begin{tabular}{@{}l cc cc cc@{}}",
        r"    \toprule",
        (
            r"    & \multicolumn{2}{c}{Indigo, phone} & \multicolumn{2}{c}{Indigo, scanner: fakes on}"
            r" & \multicolumn{2}{c}{Scantrust batch (averaging)} \\"
        ),
        r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"    & iPhone & Samsung & same printer & other printer & otsu+unsharp & deep+unsharp \\",
    ]
    for b, (title, rows) in enumerate(blocks):
        lines.append(r"    \midrule")
        lines.append(rf"    \multicolumn{{7}}{{@{{}}l}}{{\textit{{{title}}}}} \\[1pt]")
        for label, cells in rows:
            tex = []
            for col in COLUMNS:
                cell = cells.get(col)
                bold = b < 2 and cell is not None and abs(cell.worst - best[col]) < 5e-4
                tex.append(cell_tex(cell, bold))
            lines.append(f"    {label} & " + " & ".join(tex) + r" \\")
    notes = (
        r"Second line: 95\,\% CI (Clopper-Pearson on Indigo, bootstrap over templates on"
        r" Scantrust), then the realized $\Pmiss$ (\%) in italics. Scanner columns: range over"
        r" the two printers of the originals and the two template estimates. Shading grows with"
        r" $\Pfa$; bold: lowest $\Pfa$ without fakes in the column."
        r" $^\dagger$Positions where the context masks can be learned; on Scantrust they leave"
        r" out the frame, so these rows do not compare with the masked Hamming row."
        r" $^\ddagger$Codebook block of the three-capture experiment, not that of the rows above."
        r" $^\S$Other phone, fakes of the other printer, another Scantrust session; $C="
        + ", ".join(f"{c:g}" for c in by_c["chosen"])
        + r"$ chosen by cross-validation, and over the grid $C\in\{10^{-3},\dots,10^{2}\}$: "
        + ", ".join(f"{span(by_c['range'][a])} ({a})" for a in ATTACKS)
        + "."
    )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \par\vspace{3pt}",
        rf"  \parbox{{\textwidth}}{{\scriptsize {notes}}}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def build(results: dict) -> tuple[list, dict]:
    blocks = [
        ("Scores without fakes", detection_rows(results["c3a"])),
        (
            "Extensions without fakes",
            extension_rows(results["p1"], results["p2"], results["p3"]),
        ),
        ("Oracle: trained with fakes", oracle_rows(results["c3c"])),
    ]
    return blocks, other_session_by_c(results["c3c"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Table 1 from the test-stage results")
    for name, experiment in SOURCES.items():
        parser.add_argument(f"--{name}", type=Path, help=f"{experiment} test-stage result")
    args = parser.parse_args(argv)
    results, sources = {}, {}
    for name, experiment in SOURCES.items():
        path = getattr(args, name) or result_path(name)
        results[name] = load(path, experiment)
        sources[path.name] = results[name]["git"]
        print(f"{path.name}: git {results[name]['git']}")
    blocks, by_c = build(results)
    out = ROOT / "report" / "table1.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(blocks, by_c, sources))
    print(f"written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
