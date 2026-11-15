import io as stdio

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cdpbpc import counts as counts_module
from cdpbpc.counts import image_counts, scantrust_counts
from cdpbpc.io import SCANTRUST_COLUMNS, load_scantrust, scantrust_stage_rows

# (dataset, template_id, class, attack, group): one image each.
IMAGES = [
    ("unique", 3520, "original", None, None),
    ("unique", 3520, "fake", "otsu", None),  # 3501-3550: not in V_U, hidden in dev
    ("unique", 3551, "fake", "otsu", None),  # V_U: visible in dev
    ("unique", 4001, "fake", "deep", None),  # T_U: hidden in dev
    ("batch", 3501, "original", None, 0),
    ("batch", 3501, "fake", "otsu+unsharp", 0),  # group 0: visible in dev
    ("batch", 3511, "fake", "otsu+unsharp", 1),  # group 1: hidden in dev
    ("batch", 3511, "original", None, 1),
]


def _png(a: np.ndarray) -> bytes:
    buf = stdio.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(0)
    templates = {}
    for tid in sorted({img[1] for img in IMAGES}):
        templates[tid] = rng.integers(0, 2, (52, 52), dtype=np.uint8)
        path = tmp_path / "unique_cdp" / "template" / f"{tid:05d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_png(255 * templates[tid]))
    rows = []
    for n, (dataset, tid, cls, attack, group) in enumerate(IMAGES):
        y = 0.25 + 0.5 * np.kron(templates[tid], np.ones((4, 4))) + rng.normal(0, 0.1, (208, 208))
        rel = f"{dataset}_cdp/images/{n:05d}.png"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(_png((255 * np.clip(y, 0, 1)).astype(np.uint8)))
        rows.append({
            "dataset": dataset, "session": "U" if dataset == "unique" else "B",
            "template_id": tid, "class": cls, "attack": attack, "use": True,
            "print_id": n + 1 if dataset == "batch" else None, "group": group,
            "sheet": group, "path": rel,
        })  # fmt: skip
    manifest = pd.DataFrame(rows, columns=list(SCANTRUST_COLUMNS)).astype(SCANTRUST_COLUMNS)
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    return tmp_path, manifest, templates


def test_dev_stage_keeps_originals_and_dev_fakes_only(data):
    _, manifest, _ = data
    dev = scantrust_stage_rows(manifest, stage="dev")
    assert list(dev.index) == [0, 2, 4, 5, 7]
    assert len(scantrust_stage_rows(manifest, stage="test")) == len(IMAGES)
    with pytest.raises(ValueError):
        scantrust_stage_rows(manifest, stage="all")


def test_load_scantrust(data):
    data_dir, manifest, templates = data
    images = load_scantrust(data_dir, manifest, stage="test")
    assert images.bits.shape == images.t.shape == (len(IMAGES), 52, 52)
    for i, tid in enumerate(manifest["template_id"]):
        np.testing.assert_array_equal(images.t[i], templates[tid])
        np.testing.assert_array_equal(images.bits[i], templates[tid])  # clean scans
    assert (images.pcor < -0.8).all()  # large = suspect: clean originals are very negative
    assert ((images.otsu > 0.25) & (images.otsu < 0.75)).all()
    assert len(load_scantrust(data_dir, manifest, stage="dev").rows) == 5


def test_scantrust_counts_and_cache(data, monkeypatch):
    data_dir, manifest, _ = data
    c = scantrust_counts(data_dir, stage="dev")
    images = load_scantrust(data_dir, manifest, stage="dev")
    D, L = image_counts(images.t, images.bits)
    np.testing.assert_array_equal(c.D, D)
    np.testing.assert_array_equal(c.L, L)
    assert c.rows["path"].tolist() == images.rows["path"].tolist()
    with np.load(data_dir / "cache" / "scantrust_counts.dev.npz") as z:
        assert z["path"].tolist() == images.rows["path"].tolist()  # no hidden fake cached

    def fail(*args, **kwargs):
        raise AssertionError("images were read again")

    monkeypatch.setattr(counts_module, "load_scantrust", fail)
    again = scantrust_counts(data_dir, stage="dev")
    np.testing.assert_array_equal(again.D, c.D)
    np.testing.assert_array_equal(again.pcor, c.pcor)


def test_scantrust_cache_follows_the_manifest(data):
    data_dir, manifest, _ = data
    scantrust_counts(data_dir, stage="dev")
    manifest.iloc[:3].to_csv(data_dir / "manifest.csv", index=False)
    assert len(scantrust_counts(data_dir, stage="dev").rows) == 2
