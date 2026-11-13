"""Build the compact Indigo pack from the SIP zips.

The experiments only need binarized 228x228 images, which are a tiny fraction of the
40 GB of zips. This script reads the zips in dataset/sip_yareta/ without unpacking them
and writes one .npz file per data file into dataset/indigo_compact/ (about 0.5 GB in
all). The file format is described in cdpbpc.io; read it with cdpbpc.io.load_compact.

What is stored for each image:
- the binarized image. Scanner images (684x684, 16 bit): Otsu threshold per image, then a
  3x3 majority vote (white if more than 4.5 of 9 pixels are white). Phone images: Otsu on
  the grey version of rcod_hist (k = 1). Same rules as Tutt (2025), p. 87;
- pcor, the only score that needs grey levels. For fakes it is only computed on the
  development templates (1 to 288) and left as NaN on the test templates: no score is
  computed on test fakes before the methods freeze;
- the mean grey level and the Otsu threshold.
Templates (digital, and estimated by the attacker) are stored as bits only.

Usage:
    python scripts/make_compact.py                  # everything found in dataset/sip_yareta/
    python scripts/make_compact.py --only "mobile_*"
    python scripts/make_compact.py --workers 4 --overwrite

After the methods freeze, fill in pcor for the test fakes:
    python scripts/make_compact.py --test-pcor --overwrite --only "*fake*"

Files already built are skipped, so an interrupted run can simply be started again.
The pack falls under the dataset license (CC BY-NC-ND 4.0): share it inside the team
only and never commit it.
"""

import argparse
import fnmatch
import os
import re
import sys
import time
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from skimage.filters import threshold_otsu

from cdpbpc.binarize import binarize_otsu, binarize_vote
from cdpbpc.io import (
    INDIGO_DEV_MAX_ID,
    INDIGO_N_TEMPLATES,
    INDIGO_SIDE,
    decode_image,
    decode_template,
    load_compact,
    save_compact,
    to_gray,
)
from cdpbpc.scores import pcor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIP_DIR = ROOT / "dataset" / "sip_yareta"
DEFAULT_OUT_DIR = ROOT / "dataset" / "indigo_compact"
TEMPLATES_NAME = "templates_original"

# Errors raised by a damaged zip member or image file.
READ_ERRORS = (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error)

_MOBILE = re.compile(
    r"^2023_Indigo_1x1_mobile/(?P<cls>orig|fake)_phone/"
    r"(?P<printer>HPI\d+)_printdpi[\d.]+_printrun(?P<print_run>\d+)_session(?P<session>\d+)"
    r"_[A-Za-z]+(?:_(?P<fake_type>EHPI\d+))?/"
    r"(?P<device>[A-Za-z0-9]+)_run(?P<capture_run>\d+)_[^/]*\.zip$"
)
_BASE = re.compile(
    r"^2021_Indigo_1x1_base/print_(?P<cls>original|fake)/"
    r"(?P<printer>HPI\d+)(?:_(?P<fake_type>EHPI\d+))?_des3_[\d.]+dpi_2400dpi\.zip$"
)
_VARIABILITY = re.compile(
    r"^2022_Indigo_1x1_variability/"
    r"(?P<printer>HPI\d+)_printdpi[\d.]+_printrun(?P<print_run>\d+)_session(?P<session>\d+)"
    r"_[A-Za-z]+\.zip$"
)
_SCAN_DIR = re.compile(r"/scanrun(?P<scan_run>\d+)_scandpi2400/rcod/$")
_MOBILE_TEMPLATES = re.compile(
    r"^2023_Indigo_1x1_mobile/fake_templates/(?P<fake_type>EHPI\d+)\.zip$"
)
_BASE_TEMPLATES = re.compile(
    r"^2021_Indigo_1x1_base/template_estimated/(?P<printer>HPI\d+)/dens50\.zip$"
)
_ORIGINAL_TEMPLATES = "2023_Indigo_1x1_mobile/orig_templates/orig_templates.zip"


@dataclass(frozen=True)
class Source:
    """One data file of the archive: a folder of images inside one zip."""

    name: str  # output file name, without .npz
    zip_path: str  # relative to the SIP directory
    member_dir: str  # folder inside the zip, ending with "/"
    k: int  # pixels per symbol side; 0 for templates
    meta: dict = field(hash=False)


def _meta(subset, cls, zip_path, member_dir, k, **fields) -> dict:
    """Metadata of a source. Fields not given by the archive path are -1 or ""."""
    meta = {
        "subset": subset,
        "class": cls,
        "fake_type": "",
        "printer": "",
        "session": -1,
        "print_run": -1,
        "scan_run": -1,
        "device": "",
        "capture_run": -1,
        "source": f"{zip_path}::{member_dir}",
        "binarization": {0: "none", 1: "otsu", 3: "otsu + 3x3 vote"}[k],
        "k": k,
    }
    for key, value in fields.items():
        if value is None:
            continue
        meta[key] = int(value) if isinstance(meta[key], int) else value
    return meta


def describe(zip_path: str, member_dirs: list[str]) -> list[Source]:
    """Sources inside one zip of the archive, from its path and its folders.

    zip_path is relative to the SIP directory. Zips that are not part of the pack
    (e.g. estimated templates at other densities) give an empty list.
    """
    if zip_path == _ORIGINAL_TEMPLATES:
        d = "orig_templates/"
        return [
            Source(TEMPLATES_NAME, zip_path, d, 0, _meta("templates", "template", zip_path, d, 0))
        ]

    if m := _MOBILE_TEMPLATES.match(zip_path):
        fake = m["fake_type"]
        d = f"{fake}/"
        meta = _meta("mobile", "estimated", zip_path, d, 0, fake_type=fake)
        return [Source(f"templates_estimated_mobile_{fake}", zip_path, d, 0, meta)]

    if m := _BASE_TEMPLATES.match(zip_path):
        # Templates estimated from prints of HPInn are the ones behind the EHPInn fakes.
        fake = "E" + m["printer"]
        d = "dens50/"
        meta = _meta("base", "estimated", zip_path, d, 0, fake_type=fake)
        return [Source(f"templates_estimated_base_{fake}", zip_path, d, 0, meta)]

    if m := _MOBILE.match(zip_path):
        g = m.groupdict()
        cls = "original" if g["cls"] == "orig" else "fake"
        d = Path(zip_path).stem + "/rcod_hist/"
        meta = _meta("mobile", cls, zip_path, d, 1, **{k: v for k, v in g.items() if k != "cls"})
        tag = "original" if cls == "original" else f"fake_{g['fake_type']}"
        name = f"mobile_{tag}_{g['device']}_run{g['capture_run']}"
        return [Source(name, zip_path, d, 1, meta)]

    if m := _BASE.match(zip_path):
        g = m.groupdict()
        d = Path(zip_path).stem + "/dens50/"
        meta = _meta(
            "base", g["cls"], zip_path, d, 3, printer=g["printer"], fake_type=g["fake_type"]
        )
        name = f"base_original_{g['printer']}"
        if g["cls"] == "fake":
            name = f"base_fake_{g['printer']}_{g['fake_type']}"
        return [Source(name, zip_path, d, 3, meta)]

    if m := _VARIABILITY.match(zip_path):
        g = m.groupdict()
        sources = []
        for d in sorted(member_dirs):
            s = _SCAN_DIR.search(d)
            if s is None:
                continue
            meta = _meta("variability", "original", zip_path, d, 3, **g, scan_run=s["scan_run"])
            name = (
                f"variability_{g['printer']}_session{g['session']}"
                f"_print{g['print_run']}_scan{s['scan_run']}"
            )
            sources.append(Source(name, zip_path, d, 3, meta))
        return sources

    return []


def _member_dirs(names: list[str]) -> list[str]:
    return sorted({n[: n.rfind("/") + 1] for n in names if not n.endswith("/")})


def _members(z: zipfile.ZipFile, member_dir: str) -> list[tuple[int, str]]:
    """(template_id, member name) of the images directly inside member_dir, by id."""
    out = []
    for name in z.namelist():
        if not name.startswith(member_dir):
            continue
        rest = name[len(member_dir) :]
        stem, ext = os.path.splitext(rest)
        if "/" in rest or ext.lower() not in (".tif", ".tiff", ".png") or not stem.isdigit():
            continue
        tid = int(stem)
        if not 1 <= tid <= INDIGO_N_TEMPLATES:
            raise ValueError(f"template_id {tid} out of range in {member_dir}{rest}")
        out.append((tid, name))
    out.sort()
    if len({tid for tid, _ in out}) != len(out):
        raise ValueError(f"duplicate template_id in {member_dir}")
    return out


def find_sources(sip_dir: Path) -> list[Source]:
    """All sources of the pack found under sip_dir, in a stable order."""
    sources = []
    for path in sorted(sip_dir.rglob("*.zip")):
        rel = path.relative_to(sip_dir).as_posix()
        with zipfile.ZipFile(path) as z:
            sources += describe(rel, _member_dirs(z.namelist()))
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError("two sources map to the same output file")
    return sources


def compact_templates(source: Source, sip_dir: Path, out_dir: Path) -> tuple[int, int]:
    """Write a template file. Returns (images written, images skipped)."""
    ids, bits, skipped = [], [], []
    with zipfile.ZipFile(sip_dir / source.zip_path) as z:
        for tid, name in _members(z, source.member_dir):
            try:
                t = decode_template(z.read(name))
            except READ_ERRORS as e:
                print(f"  {source.name}: skipped template {tid} ({type(e).__name__}: {e})")
                skipped.append(tid)
                continue
            if t.shape != (INDIGO_SIDE, INDIGO_SIDE):
                print(f"  {source.name}: skipped template {tid} (shape {t.shape})")
                skipped.append(tid)
                continue
            ids.append(tid)
            bits.append(t)
    save_compact(
        out_dir / f"{source.name}.npz",
        ids=np.array(ids),
        bits=np.array(bits, dtype=np.uint8).reshape(-1, INDIGO_SIDE, INDIGO_SIDE),
        meta=source.meta,
        skipped=np.array(skipped),
    )
    return len(ids), len(skipped)


_templates: np.ndarray | None = None  # (1441, 228, 228) uint8, indexed by template_id


def _load_templates(path: Path) -> None:
    """Worker initializer: keep the digital templates in memory, indexed by template_id."""
    global _templates
    f = load_compact(path, stage="dev")  # digital templates are never restricted
    _templates = np.zeros((INDIGO_N_TEMPLATES + 1, INDIGO_SIDE, INDIGO_SIDE), dtype=np.uint8)
    _templates[f.ids] = f.bits
    present = np.zeros(INDIGO_N_TEMPLATES + 1, dtype=bool)
    present[f.ids] = True
    if not present[1:].all():
        missing = np.flatnonzero(~present[1:]) + 1
        raise ValueError(f"digital templates missing from {path}: {missing.tolist()}")


def compact_captures(
    source: Source, sip_dir: Path, out_dir: Path, test_pcor: bool = False
) -> tuple[int, int]:
    """Write a capture file (scans or phone photos). Returns (images written, images skipped).

    pcor of fakes on test templates is NaN unless test_pcor is True.
    """
    withhold = source.meta["class"] == "fake" and not test_pcor
    k = source.k
    shape = (k * INDIGO_SIDE, k * INDIGO_SIDE)
    ids, bits, pcors, means, thresholds, skipped = [], [], [], [], [], []
    with zipfile.ZipFile(sip_dir / source.zip_path) as z:
        for tid, name in _members(z, source.member_dir):
            try:
                y = to_gray(decode_image(z.read(name)))
            except READ_ERRORS as e:
                print(f"  {source.name}: skipped image {tid} ({type(e).__name__}: {e})")
                skipped.append(tid)
                continue
            if y.shape != shape:
                print(f"  {source.name}: skipped image {tid} (shape {y.shape}, expected {shape})")
                skipped.append(tid)
                continue
            thr = float(threshold_otsu(y))
            b = binarize_otsu(y, threshold=thr) if k == 1 else binarize_vote(y, k, threshold=thr)
            ids.append(tid)
            bits.append(b)
            if withhold and tid > INDIGO_DEV_MAX_ID:
                pcors.append(np.nan)
            else:
                pcors.append(pcor(_templates[tid], y, k))
            means.append(float(y.mean()))
            thresholds.append(thr)
    save_compact(
        out_dir / f"{source.name}.npz",
        ids=np.array(ids),
        bits=np.array(bits, dtype=np.uint8).reshape(-1, INDIGO_SIDE, INDIGO_SIDE),
        meta={**source.meta, "pcor_withheld_on_test": withhold},
        skipped=np.array(skipped),
        pcor=np.array(pcors),
        mean=np.array(means),
        otsu=np.array(thresholds),
    )
    return len(ids), len(skipped)


def _report(done: int, total: int, name: str, n: int, n_skipped: int, start: float) -> None:
    print(f"[{done}/{total}] {name}: {n} images, {n_skipped} skipped ({time.time() - start:.0f} s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sip-dir", type=Path, default=DEFAULT_SIP_DIR, help="SIP zips")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument("--only", default="*", help="glob on output file names, e.g. 'mobile_*'")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--overwrite", action="store_true", help="rebuild existing files")
    parser.add_argument(
        "--test-pcor",
        action="store_true",
        help="also compute pcor for the fakes of test templates (after the methods freeze only)",
    )
    args = parser.parse_args(argv)

    if not args.sip_dir.is_dir():
        sys.exit(f"{args.sip_dir} not found: run scripts/download_data.py first")
    sources = find_sources(args.sip_dir)
    if not any(s.name == TEMPLATES_NAME for s in sources):
        sys.exit(f"{_ORIGINAL_TEMPLATES} not found: the digital templates are needed for pcor")
    args.out.mkdir(parents=True, exist_ok=True)

    # The digital templates come first: every capture file needs them for pcor.
    templates_file = args.out / f"{TEMPLATES_NAME}.npz"
    selected = [s for s in sources if fnmatch.fnmatch(s.name, args.only)]
    todo = [s for s in selected if args.overwrite or not (args.out / f"{s.name}.npz").exists()]
    n_built = len(selected) - len(todo)
    if not templates_file.exists() and all(s.name != TEMPLATES_NAME for s in todo):
        todo.insert(0, next(s for s in sources if s.name == TEMPLATES_NAME))
    print(f"{len(selected)} files selected, {n_built} already built, {len(todo)} to build")

    start = time.time()
    done = 0
    for s in [s for s in todo if s.k == 0]:
        n, n_skipped = compact_templates(s, args.sip_dir, args.out)
        done += 1
        _report(done, len(todo), s.name, n, n_skipped, start)

    captures = [s for s in todo if s.k > 0]
    if args.workers == 1:
        _load_templates(templates_file)
        for s in captures:
            n, n_skipped = compact_captures(s, args.sip_dir, args.out, args.test_pcor)
            done += 1
            _report(done, len(todo), s.name, n, n_skipped, start)
    elif captures:
        with ProcessPoolExecutor(
            args.workers, initializer=_load_templates, initargs=(templates_file,)
        ) as pool:
            jobs = {
                pool.submit(compact_captures, s, args.sip_dir, args.out, args.test_pcor): s
                for s in captures
            }
            for job in as_completed(jobs):
                n, n_skipped = job.result()
                done += 1
                _report(done, len(todo), jobs[job].name, n, n_skipped, start)

    total = sum(p.stat().st_size for p in args.out.glob("*.npz"))
    print(f"done in {time.time() - start:.0f} s; {args.out} holds {total / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
