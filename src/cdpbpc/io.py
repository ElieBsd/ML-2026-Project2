"""Reading data: SIP Indigo images, the compact Indigo pack, and the image manifests.

The compact pack (built by scripts/make_compact.py) holds one .npz file per data file of
the archive: one print run and scan run, one phone and capture run, originals or fakes,
or one set of templates. Arrays in each file:

    ids         (N,) int16          template_id, strictly increasing
    bits        (N, 6498) uint8     np.packbits of the (228, 228) binary image, 1 = white
    pcor        (N,) float32        pcor score of the grey image (captures only); NaN for
                                    fakes on test templates until the methods freeze
    mean        (N,) float32        mean grey level in [0, 1] (captures only)
    otsu        (N,) float32        Otsu threshold in [0, 1] (captures only)
    skipped     (M,) int16          template_ids whose image could not be used
    meta        () str              JSON: where the data comes from (see make_compact.py)

The pack also holds the fakes of the test templates. load_compact(..., stage="dev")
drops them, as required before the methods freeze.

The manifests (built by scripts/make_manifest.py) list every image with what is known
about it, one row per image: dataset/manifest.csv for Scantrust and
dataset/manifest_indigo.csv for Indigo. Read them with read_manifest.
"""

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

INDIGO_SIDE = 228
INDIGO_N_TEMPLATES = 1440
INDIGO_DEV_MAX_ID = 288  # development templates: 1..288 (pages 1 and 2)

STAGES = ("dev", "test")
CAPTURE_FIELDS = ("pcor", "mean", "otsu")
# Classes of data that the development stage must not see beyond INDIGO_DEV_MAX_ID.
RESTRICTED_CLASSES = ("fake", "estimated")

# Manifest columns and their types. Empty cells mean "not applicable" or "not known".
SCANTRUST_COLUMNS = {
    "dataset": "string",  # unique or batch
    "session": "string",  # U or B
    "template_id": "Int64",  # global id 1..5000; batch template k is 3500 + k
    "class": "string",  # original or fake
    "attack": "string",  # otsu, otsu+unsharp, deep, deep+unsharp; empty for originals
    "use": "bool",  # False for batch + deep, a duplicate of batch + otsu
    "print_id": "Int64",  # batch only: file number 1..2500
    "group": "Int64",  # batch only: (print_id - 1) // 500, 10 templates per group
    "sheet": "Int64",  # batch only: (print_id - 1) // 100, 25 sheets
    "path": "string",  # relative to dataset/
}
INDIGO_COLUMNS = {
    "subset": "string",  # mobile, base or variability
    "template_id": "Int64",  # 1..1440
    "page": "Int64",  # (template_id - 1) // 144
    "page_pos": "Int64",  # (template_id - 1) % 144
    "class": "string",  # original or fake
    "fake_type": "string",  # EHPI55 or EHPI76 for fakes
    "printer": "string",  # HPI55 or HPI76
    "session": "Int64",  # print session 0, 1 or 2
    "print_run": "Int64",
    "scan_run": "Int64",  # scanner only
    "device": "string",  # phone only: iPhone12Pro or SamsungGN20U
    "capture_run": "Int64",  # phone only: 1..6
    "stage": "string",  # dev if template_id <= 288, test otherwise
    "file": "string",  # compact pack file, without .npz
    "row": "Int64",  # position of the image in that file
}


def decode_image(data: bytes) -> np.ndarray:
    """Decode an image file (TIFF, PNG) into a numpy array, as stored."""
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im)


def to_gray(a: np.ndarray) -> np.ndarray:
    """Grey level in [0, 1]. a: (H, W) uint8 or uint16, or (H, W, 3|4) uint8 RGB(A).

    RGB is converted with 0.299 R + 0.587 G + 0.114 B. Returns (H, W) float32.
    """
    if a.dtype == np.uint8:
        scale = 255.0
    elif a.dtype == np.uint16:
        scale = 65535.0
    else:
        raise ValueError(f"unsupported image dtype {a.dtype}")
    if a.ndim == 3 and a.shape[2] in (3, 4):
        g = a[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    elif a.ndim == 2:
        g = a.astype(np.float64)
    else:
        raise ValueError(f"unsupported image shape {a.shape}")
    return (g / scale).astype(np.float32)


def decode_template(data: bytes) -> np.ndarray:
    """Decode a template image with values {0, 255}. Returns (H, W) uint8, 1 = white."""
    a = decode_image(data)
    if a.ndim != 2 or a.dtype != np.uint8:
        raise ValueError(f"expected a 2-D uint8 template, got {a.dtype} {a.shape}")
    if not np.isin(a, (0, 255)).all():
        raise ValueError("template values are not in {0, 255}")
    return (a == 255).astype(np.uint8)


def pack_bits(b: np.ndarray) -> np.ndarray:
    """b: (N, L, L) binary. Returns (N, L*L/8) uint8."""
    b = np.asarray(b)
    return np.packbits(b.reshape(len(b), -1).astype(bool), axis=1)


def unpack_bits(p: np.ndarray, side: int = INDIGO_SIDE) -> np.ndarray:
    """Inverse of pack_bits. p: (N, side*side/8) uint8. Returns (N, side, side) uint8."""
    return np.unpackbits(p, axis=1, count=side * side).reshape(len(p), side, side)


def save_compact(
    path: Path,
    *,
    ids: np.ndarray,
    bits: np.ndarray,
    meta: dict,
    skipped: np.ndarray | None = None,
    **per_image: np.ndarray,
) -> None:
    """Write one compact file. bits: (N, 228, 228) binary; per_image arrays: (N,).

    The file is written under a temporary name and renamed, so an interrupted run
    never leaves a truncated file behind.
    """
    ids = np.asarray(ids, dtype=np.int16)
    bits = np.asarray(bits)
    if bits.shape != (len(ids), INDIGO_SIDE, INDIGO_SIDE):
        raise ValueError(
            f"bits must be ({len(ids)}, {INDIGO_SIDE}, {INDIGO_SIDE}), got {bits.shape}"
        )
    if len(ids) > 1 and not (np.diff(ids) > 0).all():
        raise ValueError("ids must be strictly increasing")
    unknown = set(per_image) - set(CAPTURE_FIELDS)
    if unknown:
        raise ValueError(f"unknown per-image fields {sorted(unknown)}")
    arrays = {
        "ids": ids,
        "bits": pack_bits(bits),
        "skipped": np.asarray([] if skipped is None else skipped, dtype=np.int16),
        "meta": np.array(json.dumps(meta, sort_keys=True)),
    }
    for name, values in per_image.items():
        values = np.asarray(values, dtype=np.float32)
        if values.shape != ids.shape:
            raise ValueError(f"{name} must have shape {ids.shape}, got {values.shape}")
        arrays[name] = values
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def read_compact_header(path: Path) -> tuple[np.ndarray, dict]:
    """ids and metadata of a compact file, without reading any image."""
    with np.load(path) as z:
        return z["ids"], json.loads(str(z["meta"]))


def read_manifest(path: Path) -> pd.DataFrame:
    """Read manifest.csv or manifest_indigo.csv with the right column types."""
    columns = pd.read_csv(path, nrows=0).columns
    for known in (SCANTRUST_COLUMNS, INDIGO_COLUMNS):
        if list(columns) == list(known):
            return pd.read_csv(path, dtype=known, keep_default_na=False, na_values=[""])
    raise ValueError(f"{path} is not a manifest: unexpected columns {list(columns)}")


@dataclass(frozen=True)
class CompactFile:
    """Content of one compact file. bits: (N, 228, 228) uint8; per-image arrays: (N,)."""

    ids: np.ndarray
    bits: np.ndarray
    meta: dict
    skipped: np.ndarray
    pcor: np.ndarray | None = None
    mean: np.ndarray | None = None
    otsu: np.ndarray | None = None


def load_compact(path: Path, *, stage: str) -> CompactFile:
    """Read one compact file.

    stage="dev" keeps only templates 1..288 in files of fakes and estimated templates
    (originals and digital templates are always complete). stage="test" keeps everything
    and is only for runs after the methods freeze.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        ids = z["ids"]
        keep = np.ones(len(ids), dtype=bool)
        if stage == "dev" and meta["class"] in RESTRICTED_CLASSES:
            keep = ids <= INDIGO_DEV_MAX_ID
        fields = {name: z[name][keep] for name in CAPTURE_FIELDS if name in z.files}
        skipped = z["skipped"]
        if stage == "dev" and meta["class"] in RESTRICTED_CLASSES:
            skipped = skipped[skipped <= INDIGO_DEV_MAX_ID]
        return CompactFile(
            ids=ids[keep],
            bits=unpack_bits(z["bits"][keep]),
            meta=meta,
            skipped=skipped,
            **fields,
        )
