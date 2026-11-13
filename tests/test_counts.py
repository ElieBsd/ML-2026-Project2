import numpy as np
import pytest

from cdpbpc import counts as counts_module
from cdpbpc.codebook import pattern_counts
from cdpbpc.counts import image_counts, indigo_counts
from cdpbpc.io import INDIGO_DEV_MAX_ID, save_compact

SIDE = 228


def _pair(rng, n, side, flip_rate=0.2):
    t = rng.integers(0, 2, (n, side, side), dtype=np.uint8)
    t_hat = t ^ (rng.random(t.shape) < flip_rate).astype(np.uint8)
    return t, t_hat


def test_sums_match_symbols_and_flips():
    t, t_hat = _pair(np.random.default_rng(0), 5, 30)
    D, L = image_counts(t, t_hat)
    assert D.shape == L.shape == (5, 512) and D.dtype == L.dtype == np.int32
    assert (L.sum(1) == 28 * 28).all()
    np.testing.assert_array_equal(D.sum(1), (t != t_hat)[:, 1:-1, 1:-1].sum((1, 2)))
    assert (D <= L).all()


def test_matches_direct_count_on_one_image():
    rng = np.random.default_rng(1)
    t, t_hat = _pair(rng, 1, 12)
    w = 2 ** np.arange(8, -1, -1)
    D_ref = np.zeros(512, int)
    L_ref = np.zeros(512, int)
    for i in range(1, 11):
        for j in range(1, 11):
            omega = (t[0, i - 1 : i + 2, j - 1 : j + 2].ravel() * w).sum()
            L_ref[omega] += 1
            D_ref[omega] += t[0, i, j] != t_hat[0, i, j]
    D, L = image_counts(t, t_hat)
    np.testing.assert_array_equal(D[0], D_ref)
    np.testing.assert_array_equal(L[0], L_ref)


@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_chunking_does_not_change_counts(chunk):
    t, t_hat = _pair(np.random.default_rng(2), 7, 20)
    D, L = image_counts(t, t_hat, chunk=7)
    Dc, Lc = image_counts(t, t_hat, chunk=chunk)
    np.testing.assert_array_equal(D, Dc)
    np.testing.assert_array_equal(L, Lc)


def test_sums_over_images_give_codebook_counts():
    t, t_hat = _pair(np.random.default_rng(3), 6, 20)
    D, L = image_counts(t, t_hat)
    n, k = pattern_counts(t, t_hat)
    np.testing.assert_array_equal(L.sum(0), n)
    np.testing.assert_array_equal(D.sum(0), k)


def test_all_white_template_fully_flipped():
    t = np.ones((1, 6, 6), np.uint8)
    D, L = image_counts(t, 1 - t)
    assert L[0, 511] == D[0, 511] == 16 and L.sum() == D.sum() == 16


def test_no_images():
    D, L = image_counts(np.zeros((0, 8, 8), np.uint8), np.zeros((0, 8, 8), np.uint8))
    assert D.shape == L.shape == (0, 512)


def test_rejects_bad_inputs():
    t = np.zeros((2, 8, 8), np.uint8)
    with pytest.raises(ValueError):
        image_counts(t, 255 * np.ones_like(t))  # 0/255 instead of 0/1
    with pytest.raises(ValueError):
        image_counts(t, t[:1])
    with pytest.raises(ValueError):
        image_counts(t[0], t[0])


# ----------------------------------------------------------------- compact Indigo files

IDS = np.array([1, 2, INDIGO_DEV_MAX_ID + 1, INDIGO_DEV_MAX_ID + 2])


@pytest.fixture
def pack(tmp_path):
    """dataset/indigo_compact with 4 templates, an original file and a fake file."""
    rng = np.random.default_rng(4)
    pack_dir = tmp_path / "dataset" / "indigo_compact"
    pack_dir.mkdir(parents=True)
    t = rng.integers(0, 2, (len(IDS), SIDE, SIDE), dtype=np.uint8)
    save_compact(pack_dir / "templates_original.npz", ids=IDS, bits=t, meta={"class": "template"})
    captures = {}
    for cls in ("original", "fake"):
        b = t ^ (rng.random(t.shape) < 0.2).astype(np.uint8)
        zeros = np.zeros(len(IDS))
        path = pack_dir / f"mobile_{cls}.npz"
        save_compact(path, ids=IDS, bits=b, meta={"class": cls}, pcor=zeros, mean=zeros, otsu=zeros)
        captures[cls] = (path, b)
    return pack_dir, t, captures


def test_indigo_counts_match_image_counts(pack):
    pack_dir, t, captures = pack
    path, b = captures["original"]
    c = indigo_counts(path, stage="dev")
    D, L = image_counts(t, b)
    np.testing.assert_array_equal(c.ids, IDS)  # originals are complete at both stages
    np.testing.assert_array_equal(c.D, D)
    np.testing.assert_array_equal(c.L, L)
    assert c.meta == {"class": "original"}
    assert (pack_dir.parent / "cache" / "indigo_counts" / "mobile_original.npz").exists()


def test_dev_stage_never_counts_test_fakes(pack):
    pack_dir, t, captures = pack
    path, b = captures["fake"]
    cache_dir = pack_dir.parent / "cache" / "indigo_counts"
    dev = indigo_counts(path, stage="dev")
    np.testing.assert_array_equal(dev.ids, [1, 2])
    np.testing.assert_array_equal(dev.D, image_counts(t[:2], b[:2])[0])
    with np.load(cache_dir / "mobile_fake.dev.npz") as z:
        assert (z["ids"] <= INDIGO_DEV_MAX_ID).all()
    assert not (cache_dir / "mobile_fake.test.npz").exists()
    test = indigo_counts(path, stage="test")
    np.testing.assert_array_equal(test.ids, IDS)
    np.testing.assert_array_equal(test.D, image_counts(t, b)[0])


def test_cache_is_reused(pack, monkeypatch):
    _, _, captures = pack
    path, _ = captures["original"]
    first = indigo_counts(path, stage="dev")

    def fail(*args, **kwargs):
        raise AssertionError("counts were recomputed")

    monkeypatch.setattr(counts_module, "image_counts", fail)
    second = indigo_counts(path, stage="dev")
    np.testing.assert_array_equal(second.D, first.D)
    np.testing.assert_array_equal(second.L, first.L)
    assert second.D.dtype == np.int32


def test_cache_is_rebuilt_when_the_pack_changes(pack):
    _, t, captures = pack
    path, _ = captures["original"]
    indigo_counts(path, stage="dev")
    zeros = np.zeros(len(IDS))
    save_compact(
        path, ids=IDS, bits=t, meta={"class": "original"}, pcor=zeros, mean=zeros, otsu=zeros
    )
    assert indigo_counts(path, stage="dev").D.sum() == 0  # rebuilt with perfect captures


def test_rejects_template_files_and_missing_templates(pack):
    pack_dir, t, _ = pack
    with pytest.raises(ValueError):
        indigo_counts(pack_dir / "templates_original.npz", stage="dev")
    zeros = np.zeros(1)
    path = pack_dir / "mobile_orphan.npz"
    save_compact(
        path, ids=[5], bits=t[:1], meta={"class": "original"}, pcor=zeros, mean=zeros, otsu=zeros
    )
    with pytest.raises(ValueError):
        indigo_counts(path, stage="dev")
