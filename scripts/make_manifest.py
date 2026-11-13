"""Build the image manifests: one row per image, with everything known about it.

- dataset/manifest.csv: the Scantrust images in dataset/unique_cdp/ and dataset/batch_cdp/
  (originals and fakes; templates are not listed).
- dataset/manifest_indigo.csv: the Indigo captures of the compact pack in
  dataset/indigo_compact/ (run scripts/make_compact.py first).

Experiments select images by filtering a manifest instead of parsing file names. Column
meanings are in cdpbpc.io (SCANTRUST_COLUMNS, INDIGO_COLUMNS); read the files with
cdpbpc.io.read_manifest. Empty cells mean "not applicable" or "not known".

Usage:
    python scripts/make_manifest.py             # every dataset found under dataset/
    python scripts/make_manifest.py --data-dir path/to/dataset
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from cdpbpc.io import (
    INDIGO_COLUMNS,
    INDIGO_DEV_MAX_ID,
    INDIGO_N_TEMPLATES,
    SCANTRUST_COLUMNS,
    read_compact_header,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "dataset"

ATTACKS = ("otsu", "otsu+unsharp", "deep", "deep+unsharp")
BATCH_DUPLICATE_ATTACK = "deep"  # batch_cdp/copy/deep holds the scans of copy/otsu
BATCH_ID_OFFSET = 3500  # batch template k is unique template 3500 + k
# Expected number of PNG files per folder of the Kaggle download (version 4).
SCANTRUST_EXPECTED = {
    ("unique", "template"): 5000,
    ("unique", "original"): 5000,
    ("unique", "fake"): 1500,  # per attack, templates 3501..5000
    ("batch", "template"): 50,
    ("batch", "original"): 2500,
    ("batch", "fake"): 2500,  # per attack
}
INDIGO_PAGE = 144  # templates per printed page


def _pngs(folder: Path) -> list[Path]:
    """PNG files of a folder (other files, such as .DS_Store, are ignored)."""
    return sorted(p for p in folder.iterdir() if p.suffix == ".png")


def _check_count(found: int, key: tuple[str, str], where: Path) -> None:
    expected = SCANTRUST_EXPECTED[key]
    if found != expected:
        raise ValueError(f"{where}: {found} PNG files, expected {expected}")


def _batch_template(print_id: int) -> int:
    """Template k (1..50) printed as batch file print_id (1..2500).

    Files come in blocks of 10 consecutive numbers per template, 10 templates per block
    of 100 (one sheet), and 5 sheets per group of 10 templates.
    """
    n = print_id - 1
    return 10 * (n // 500) + (n % 100) // 10 + 1


def scantrust_rows(data_dir: Path) -> pd.DataFrame:
    """Manifest of the Scantrust download under data_dir (unique_cdp/ and batch_cdp/)."""
    rows = []
    unique = data_dir / "unique_cdp"
    templates = {int(p.stem) for p in _pngs(unique / "template")}
    _check_count(len(templates), ("unique", "template"), unique / "template")

    def unique_row(path: Path, cls: str, attack: str | None) -> dict:
        tid = int(path.stem)
        if tid not in templates:
            raise ValueError(f"{path}: no template {tid}")
        return {
            "dataset": "unique",
            "session": "U",
            "template_id": tid,
            "class": cls,
            "attack": attack,
            "use": True,
            "path": path.relative_to(data_dir).as_posix(),
        }

    files = _pngs(unique / "original")
    _check_count(len(files), ("unique", "original"), unique / "original")
    rows += [unique_row(p, "original", None) for p in files]
    for attack in ATTACKS:
        folder = unique / "copy" / attack
        files = _pngs(folder)
        _check_count(len(files), ("unique", "fake"), folder)
        # Fakes exist for the last 1500 templates only.
        if [int(p.stem) for p in files] != list(range(BATCH_ID_OFFSET + 1, len(templates) + 1)):
            raise ValueError(f"{folder}: expected the templates {BATCH_ID_OFFSET + 1} and above")
        rows += [unique_row(p, "fake", attack) for p in files]

    batch = data_dir / "batch_cdp"
    n_templates = len(_pngs(batch / "template"))
    _check_count(n_templates, ("batch", "template"), batch / "template")

    def batch_rows(root: Path, cls: str, attack: str | None) -> list[dict]:
        out = []
        for k in range(1, n_templates + 1):
            for path in _pngs(root / str(k)):
                print_id = int(path.stem)
                if _batch_template(print_id) != k:
                    raise ValueError(f"{path}: file number does not match template {k}")
                out.append(
                    {
                        "dataset": "batch",
                        "session": "B",
                        "template_id": BATCH_ID_OFFSET + k,
                        "class": cls,
                        "attack": attack,
                        "use": attack != BATCH_DUPLICATE_ATTACK,
                        "print_id": print_id,
                        "group": (print_id - 1) // 500,
                        "sheet": (print_id - 1) // 100,
                        "path": path.relative_to(data_dir).as_posix(),
                    }
                )
        return out

    originals = batch_rows(batch / "original", "original", None)
    _check_count(len(originals), ("batch", "original"), batch / "original")
    print_ids = [r["print_id"] for r in originals]
    if len(set(print_ids)) != len(print_ids):
        raise ValueError(f"{batch / 'original'}: a file number appears twice")
    rows += originals
    for attack in ATTACKS:
        fakes = batch_rows(batch / "copy" / attack, "fake", attack)
        _check_count(len(fakes), ("batch", "fake"), batch / "copy" / attack)
        # Each fake carries the file name of an original of the same template.
        if sorted(r["print_id"] for r in fakes) != sorted(print_ids):
            raise ValueError(f"{batch / 'copy' / attack}: file names differ from original/")
        rows += fakes

    return pd.DataFrame(rows, columns=list(SCANTRUST_COLUMNS)).astype(SCANTRUST_COLUMNS)


def _known(value, missing):
    return None if value == missing else value


def indigo_rows(pack_dir: Path) -> pd.DataFrame:
    """Manifest of the captures of the compact Indigo pack in pack_dir."""
    rows = []
    for path in sorted(pack_dir.glob("*.npz")):
        ids, meta = read_compact_header(path)
        if meta["class"] not in ("original", "fake"):
            continue  # digital and estimated templates are not captures
        session = _known(meta["session"], -1)
        print_run = _known(meta["print_run"], -1)
        scan_run = _known(meta["scan_run"], -1)
        if meta["subset"] == "base" and meta["class"] == "original":
            # The archive names of the scanner dataset give no session. Its originals are,
            # pixel for pixel, scan 1 of print run 1 of session 0 in the variability
            # dataset (checked on both printers). The session of its fakes is not known.
            session, print_run, scan_run = 0, 1, 1
        for row, tid in enumerate(ids.tolist()):
            if not 1 <= tid <= INDIGO_N_TEMPLATES:
                raise ValueError(f"{path.name}: template_id {tid} out of range")
            rows.append(
                {
                    "subset": meta["subset"],
                    "template_id": tid,
                    "page": (tid - 1) // INDIGO_PAGE,
                    "page_pos": (tid - 1) % INDIGO_PAGE,
                    "class": meta["class"],
                    "fake_type": _known(meta["fake_type"], ""),
                    "printer": _known(meta["printer"], ""),
                    "session": session,
                    "print_run": print_run,
                    "scan_run": scan_run,
                    "device": _known(meta["device"], ""),
                    "capture_run": _known(meta["capture_run"], -1),
                    "stage": "dev" if tid <= INDIGO_DEV_MAX_ID else "test",
                    "file": path.stem,
                    "row": row,
                }
            )
    return pd.DataFrame(rows, columns=list(INDIGO_COLUMNS)).astype(INDIGO_COLUMNS)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write under a temporary name, then rename: no half-written manifest."""
    tmp = path.with_name(path.name + ".part")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="dataset folder")
    args = parser.parse_args(argv)
    data_dir = args.data_dir

    built = 0
    if (data_dir / "unique_cdp").is_dir() and (data_dir / "batch_cdp").is_dir():
        df = scantrust_rows(data_dir)
        write_csv(df, data_dir / "manifest.csv")
        n_fakes = (df["class"] == "fake").sum()
        print(f"manifest.csv: {len(df)} images ({n_fakes} fakes, {(~df['use']).sum()} unused)")
        built += 1
    else:
        print("Scantrust not found: run scripts/download_data.py scantrust")
    pack_dir = data_dir / "indigo_compact"
    if any(pack_dir.glob("*.npz")):
        df = indigo_rows(pack_dir)
        write_csv(df, data_dir / "manifest_indigo.csv")
        files = df["file"].nunique()
        print(f"manifest_indigo.csv: {len(df)} images from {files} files")
        built += 1
    else:
        print("compact Indigo pack not found: run scripts/make_compact.py")
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
