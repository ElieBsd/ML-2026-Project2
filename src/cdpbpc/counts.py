"""Per-image pattern counts, the input of every codebook-based score.

For each image: L[omega] = number of interior symbols of its template with 3x3
pattern omega, and D[omega] = number of those symbols that were read wrong (flips).
Counts are always taken against the true template of the image, for fakes too.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cdpbpc.io import RESTRICTED_CLASSES, load_compact
from cdpbpc.patterns import N_PATTERNS, interior, pattern_index

TEMPLATES_FILE = "templates_original.npz"


def image_counts(
    t: np.ndarray, t_hat: np.ndarray, chunk: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Flips D and occurrences L of each 3x3 pattern, per image.

    t, t_hat: (N, side, side) binary (1 = white); t[i] is the template of image i and
    t_hat[i] its binarized capture. Only interior symbols count. Images are processed
    chunk at a time to bound memory. Returns D, L: (N, 512) int32.
    """
    t = np.asarray(t)
    t_hat = np.asarray(t_hat)
    if t.ndim != 3 or t.shape != t_hat.shape:
        raise ValueError(f"expected two (N, side, side) arrays, got {t.shape} and {t_hat.shape}")
    if t.size and (t.max() > 1 or t_hat.max() > 1):
        raise ValueError("templates and binarized images must be 0/1")
    n_img = len(t)
    D = np.empty((n_img, N_PATTERNS), dtype=np.int32)
    L = np.empty((n_img, N_PATTERNS), dtype=np.int32)
    for start in range(0, n_img, chunk):
        tc = t[start : start + chunk]
        n = len(tc)
        # Offset each image's indices so that one bincount gives n rows of 512 counts.
        idx = pattern_index(tc).reshape(n, -1).astype(np.int64)
        idx += N_PATTERNS * np.arange(n, dtype=np.int64)[:, None]
        flipped = interior(tc != t_hat[start : start + n]).reshape(n, -1)
        size = n * N_PATTERNS
        L[start : start + n] = np.bincount(idx.ravel(), minlength=size).reshape(n, N_PATTERNS)
        D[start : start + n] = np.bincount(idx[flipped], minlength=size).reshape(n, N_PATTERNS)
    return D, L


@dataclass(frozen=True)
class Counts:
    """Pattern counts of one data file. ids: (N,); D, L: (N, 512) int32."""

    ids: np.ndarray
    D: np.ndarray
    L: np.ndarray
    meta: dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cache_path(path: Path, cache_dir: Path, meta: dict, stage: str) -> Path:
    # Files with fakes differ between stages; other files are the same at both stages.
    suffix = f".{stage}" if meta["class"] in RESTRICTED_CLASSES else ""
    return cache_dir / f"{path.stem}{suffix}.npz"


def _read_cache(cache: Path, key: dict) -> Counts | None:
    if not cache.exists():
        return None
    with np.load(cache) as z:
        if json.loads(str(z["key"])) != key:
            return None
        return Counts(
            ids=z["ids"],
            D=z["D"].astype(np.int32),
            L=z["L"].astype(np.int32),
            meta=json.loads(str(z["meta"])),
        )


def _write_cache(cache: Path, key: dict, counts: Counts) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Per-image counts never exceed 228 * 228 interior symbols, so uint16 is lossless here.
    dtype = np.uint16 if counts.L.max(initial=0) <= np.iinfo(np.uint16).max else np.int32
    tmp = cache.with_name(cache.name + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            ids=counts.ids,
            D=counts.D.astype(dtype),
            L=counts.L.astype(dtype),
            meta=np.array(json.dumps(counts.meta, sort_keys=True)),
            key=np.array(json.dumps(key, sort_keys=True)),
        )
    os.replace(tmp, cache)


def indigo_counts(path: Path, *, stage: str, cache_dir: Path | None = None) -> Counts:
    """Pattern counts of one capture file of the compact Indigo pack, cached.

    path: a capture file of dataset/indigo_compact/ (not a template file); the digital
    templates are read from templates_original.npz next to it. stage is passed to
    load_compact: with stage="dev", the fakes of test templates are neither read nor
    counted. The cache (default dataset/cache/indigo_counts/) is rebuilt whenever the
    compact file or the templates change.
    """
    path = Path(path)
    templates_path = path.parent / TEMPLATES_FILE
    cache_dir = path.parent.parent / "cache" / "indigo_counts" if cache_dir is None else cache_dir
    key = {"stage": stage, "source": _sha256(path), "templates": _sha256(templates_path)}

    f = load_compact(path, stage=stage)
    if f.meta["class"] not in ("original", "fake"):
        raise ValueError(f"{path.name} holds templates, not captures")
    cache = _cache_path(path, Path(cache_dir), f.meta, stage)
    cached = _read_cache(cache, key)
    if cached is not None:
        return cached

    templates = load_compact(templates_path, stage=stage)
    position = np.full(int(templates.ids.max(initial=0)) + 1, -1)
    position[templates.ids] = np.arange(len(templates.ids))
    if f.ids.size and (f.ids.max() >= len(position) or (position[f.ids] < 0).any()):
        raise ValueError(f"some templates of {path.name} are missing from {TEMPLATES_FILE}")
    D, L = image_counts(templates.bits[position[f.ids]], f.bits)
    counts = Counts(ids=f.ids, D=D, L=L, meta=f.meta)
    _write_cache(cache, key, counts)
    return counts
