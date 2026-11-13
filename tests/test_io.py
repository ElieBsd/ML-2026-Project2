import io as stdio

import numpy as np
import pytest
from PIL import Image

from cdpbpc.io import (
    INDIGO_DEV_MAX_ID,
    decode_image,
    decode_template,
    load_compact,
    pack_bits,
    save_compact,
    to_gray,
    unpack_bits,
)


def _encode(a: np.ndarray, fmt: str = "TIFF") -> bytes:
    buf = stdio.BytesIO()
    Image.fromarray(a).save(buf, format=fmt)
    return buf.getvalue()


def _meta(cls: str) -> dict:
    return {"class": cls, "subset": "mobile"}


def test_pack_unpack_roundtrip():
    b = np.random.default_rng(0).integers(0, 2, (5, 228, 228), dtype=np.uint8)
    p = pack_bits(b)
    assert p.shape == (5, 228 * 228 // 8) and p.dtype == np.uint8
    np.testing.assert_array_equal(unpack_bits(p), b)


def test_to_gray_scales_and_weights():
    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[0, 0] = (255, 255, 255)
    rgb[0, 1] = (255, 0, 0)
    g = to_gray(rgb)
    assert g.dtype == np.float32
    np.testing.assert_allclose(g[0, :2], [1.0, 0.299], rtol=1e-6)
    np.testing.assert_allclose(to_gray(np.array([[0, 65535]], np.uint16)), [[0.0, 1.0]])
    with pytest.raises(ValueError):
        to_gray(np.zeros((2, 2), np.float32))


def test_decode_16bit_tiff():
    a = np.array([[0, 1000], [40000, 65535]], np.uint16)
    out = decode_image(_encode(a))
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, a)


@pytest.mark.parametrize("fmt", ["TIFF", "PNG"])
def test_decode_template(fmt):
    t = np.random.default_rng(1).integers(0, 2, (228, 228), dtype=np.uint8)
    np.testing.assert_array_equal(decode_template(_encode(255 * t, fmt)), t)


def test_decode_template_rejects_grey_values():
    with pytest.raises(ValueError):
        decode_template(_encode(np.full((4, 4), 128, np.uint8)))


def _write(tmp_path, cls, ids, skipped=()):
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, (len(ids), 228, 228), dtype=np.uint8)
    pc = rng.random(len(ids)).astype(np.float32)
    path = tmp_path / f"{cls}.npz"
    save_compact(
        path,
        ids=np.array(ids),
        bits=bits,
        meta=_meta(cls),
        skipped=np.array(skipped),
        pcor=pc,
        mean=pc,
        otsu=pc,
    )
    return path, bits, pc


def test_save_load_roundtrip(tmp_path):
    ids = [1, 7, 300, 1440]
    path, bits, pc = _write(tmp_path, "original", ids, skipped=[5, 900])
    f = load_compact(path, stage="dev")
    np.testing.assert_array_equal(f.ids, ids)
    np.testing.assert_array_equal(f.bits, bits)
    np.testing.assert_array_equal(f.pcor, pc)
    np.testing.assert_array_equal(f.skipped, [5, 900])
    assert f.meta == _meta("original")
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("cls", ["fake", "estimated"])
def test_dev_stage_drops_test_templates(tmp_path, cls):
    ids = [1, INDIGO_DEV_MAX_ID, INDIGO_DEV_MAX_ID + 1, 1440]
    path, bits, pc = _write(tmp_path, cls, ids, skipped=[10, 500])
    dev = load_compact(path, stage="dev")
    np.testing.assert_array_equal(dev.ids, [1, INDIGO_DEV_MAX_ID])
    np.testing.assert_array_equal(dev.bits, bits[:2])
    np.testing.assert_array_equal(dev.pcor, pc[:2])
    np.testing.assert_array_equal(dev.skipped, [10])
    assert len(load_compact(path, stage="test").ids) == 4


def test_dev_stage_keeps_all_originals(tmp_path):
    path, _, _ = _write(tmp_path, "original", [1, 289, 1440])
    assert len(load_compact(path, stage="dev").ids) == 3


def test_stage_is_checked(tmp_path):
    path, _, _ = _write(tmp_path, "fake", [1])
    with pytest.raises(ValueError):
        load_compact(path, stage="all")


def test_save_checks_inputs(tmp_path):
    bits = np.zeros((2, 228, 228), np.uint8)
    with pytest.raises(ValueError):
        save_compact(tmp_path / "a.npz", ids=[2, 1], bits=bits, meta=_meta("original"))
    with pytest.raises(ValueError):
        save_compact(tmp_path / "a.npz", ids=[1, 2], bits=bits[:1], meta=_meta("original"))
    with pytest.raises(ValueError):
        save_compact(
            tmp_path / "a.npz", ids=[1, 2], bits=bits, meta=_meta("original"), pcor=np.zeros(3)
        )
