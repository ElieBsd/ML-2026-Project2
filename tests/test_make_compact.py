import importlib.util
import io as stdio
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cdpbpc.binarize import binarize_otsu, binarize_vote
from cdpbpc.io import load_compact, to_gray
from cdpbpc.scores import pcor

_spec = importlib.util.spec_from_file_location(
    "make_compact", Path(__file__).resolve().parents[1] / "scripts" / "make_compact.py"
)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

N = 4  # templates in the synthetic archive
DEV_MAX = 2  # development templates in the synthetic archive
MOBILE_DIR = "2023_Indigo_1x1_mobile/orig_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG"
MOBILE_FAKE_DIR = (
    "2023_Indigo_1x1_mobile/fake_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55"
)
PHONE = "iPhone12Pro_run1_ss100_focal12_apperture1"
VAR_ZIP = "2022_Indigo_1x1_variability/HPI76_printdpi812.8_printrun2_session1_InvercoteG.zip"


def _tiff(a: np.ndarray) -> bytes:
    buf = stdio.BytesIO()
    Image.fromarray(a).save(buf, format="TIFF")
    return buf.getvalue()


def _zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A tiny copy of the SIP archive layout: 4 templates, phone photos and scans."""
    monkeypatch.setattr(mc, "INDIGO_N_TEMPLATES", N)
    monkeypatch.setattr(mc, "INDIGO_DEV_MAX_ID", DEV_MAX)
    rng = np.random.default_rng(0)
    sip = tmp_path / "sip"
    t = rng.integers(0, 2, (N + 1, 228, 228), dtype=np.uint8)  # t[tid], t[0] unused
    _zip(
        sip / mc._ORIGINAL_TEMPLATES,
        {f"orig_templates/{i:06d}.tiff": _tiff(255 * t[i]) for i in range(1, N + 1)},
    )
    t_hat = t.copy()
    t_hat[:, 100, :50] ^= 1
    _zip(
        sip / "2023_Indigo_1x1_mobile/fake_templates/EHPI55.zip",
        {f"EHPI55/{i:06d}.tiff": _tiff(255 * t_hat[i]) for i in range(1, N + 1)},
    )
    # Phone photos: ids 1-3 readable, id 4 damaged; the rcod/ folder must be ignored.
    photos = {}
    for i in range(1, 4):
        g = np.clip(0.2 + 0.6 * t[i] + rng.normal(0, 0.15, (228, 228)), 0, 1)
        photos[i] = np.repeat((255 * g).astype(np.uint8)[..., None], 3, axis=2)
    members = {f"{PHONE}/rcod_hist/{i:06d}.tiff": _tiff(a) for i, a in photos.items()}
    members[f"{PHONE}/rcod_hist/000004.tiff"] = b"not an image"
    members[f"{PHONE}/rcod/000001.tiff"] = _tiff(photos[1])
    _zip(sip / MOBILE_DIR / f"{PHONE}.zip", members)
    _zip(
        sip / MOBILE_FAKE_DIR / f"{PHONE}.zip",
        {f"{PHONE}/rcod_hist/{i:06d}.tiff": _tiff(photos[i]) for i in (2, 3)},
    )
    # Scans: two scan runs of one print run, 16 bit, 3x3 pixels per symbol.
    scans, members = {}, {}
    for run, ids in ((1, (1, 2)), (2, (2, 3))):
        for i in ids:
            y = 10000 + 40000 * np.kron(t[i], np.ones((3, 3))) + rng.normal(0, 9000, (684, 684))
            scans[run, i] = np.clip(y, 0, 65535).astype(np.uint16)
            folder = "HPI76_printdpi812.8_printrun2_session1_InvercoteG"
            members[f"{folder}/scanrun{run}_scandpi2400/rcod/{i:06d}.tiff"] = _tiff(scans[run, i])
    _zip(sip / VAR_ZIP, members)
    # A zip that is not part of the pack.
    _zip(sip / "2021_Indigo_1x1_base/template_estimated/HPI55/dens30.zip", {"dens30/0001.png": b""})
    return sip, tmp_path / "out", t, t_hat, photos, scans


def test_describe_paths():
    (s,) = mc.describe(f"{MOBILE_FAKE_DIR}/SamsungGN20U_run6_ss100_focal12_apperture1.zip", [])
    assert s.name == "mobile_fake_EHPI55_SamsungGN20U_run6"
    assert s.member_dir == "SamsungGN20U_run6_ss100_focal12_apperture1/rcod_hist/"
    assert (s.k, s.meta["class"], s.meta["fake_type"], s.meta["device"]) == (
        1,
        "fake",
        "EHPI55",
        "SamsungGN20U",
    )
    assert (s.meta["printer"], s.meta["session"], s.meta["print_run"], s.meta["capture_run"]) == (
        "HPI55",
        1,
        1,
        6,
    )

    (s,) = mc.describe("2021_Indigo_1x1_base/print_fake/HPI76_EHPI55_des3_812.8dpi_2400dpi.zip", [])
    assert s.name == "base_fake_HPI76_EHPI55"
    assert s.member_dir == "HPI76_EHPI55_des3_812.8dpi_2400dpi/dens50/"
    assert (s.meta["printer"], s.meta["fake_type"], s.meta["session"], s.k) == (
        "HPI76",
        "EHPI55",
        -1,
        3,
    )

    (s,) = mc.describe("2021_Indigo_1x1_base/print_original/HPI55_des3_812.8dpi_2400dpi.zip", [])
    assert s.name == "base_original_HPI55" and s.meta["class"] == "original"

    (s,) = mc.describe("2021_Indigo_1x1_base/template_estimated/HPI76/dens50.zip", [])
    assert s.name == "templates_estimated_base_EHPI76" and s.meta["class"] == "estimated"

    dirs = [f"F/scanrun{r}_scandpi2400/rcod/" for r in (2, 1)] + ["F/other/"]
    sources = mc.describe(VAR_ZIP, dirs)
    assert [s.name for s in sources] == [
        "variability_HPI76_session1_print2_scan1",
        "variability_HPI76_session1_print2_scan2",
    ]
    assert sources[1].meta["scan_run"] == 2 and sources[1].meta["print_run"] == 2

    assert mc.describe("2021_Indigo_1x1_base/template_estimated/HPI55/dens30.zip", []) == []


def test_end_to_end(archive):
    sip, out, t, t_hat, photos, scans = archive
    assert mc.main(["--sip-dir", str(sip), "--out", str(out), "--workers", "1"]) == 0
    assert sorted(p.stem for p in out.glob("*.npz")) == [
        "mobile_fake_EHPI55_iPhone12Pro_run1",
        "mobile_original_iPhone12Pro_run1",
        "templates_estimated_mobile_EHPI55",
        "templates_original",
        "variability_HPI76_session1_print2_scan1",
        "variability_HPI76_session1_print2_scan2",
    ]

    f = load_compact(out / "templates_original.npz", stage="dev")
    np.testing.assert_array_equal(f.ids, [1, 2, 3, 4])
    np.testing.assert_array_equal(f.bits, t[1:])
    assert f.pcor is None

    f = load_compact(out / "templates_estimated_mobile_EHPI55.npz", stage="test")
    np.testing.assert_array_equal(f.bits, t_hat[1:])

    f = load_compact(out / "mobile_original_iPhone12Pro_run1.npz", stage="dev")
    np.testing.assert_array_equal(f.ids, [1, 2, 3])
    np.testing.assert_array_equal(f.skipped, [4])
    for j, i in enumerate(f.ids):
        y = to_gray(photos[i])
        np.testing.assert_array_equal(f.bits[j], binarize_otsu(y))
        assert f.pcor[j] == pytest.approx(pcor(t[i], y, 1), abs=1e-6)
        assert f.mean[j] == pytest.approx(y.mean(), abs=1e-6)
    assert (f.pcor < -0.5).all()
    assert f.meta["source"] == f"{MOBILE_DIR}/{PHONE}.zip::{PHONE}/rcod_hist/"

    f = load_compact(out / "variability_HPI76_session1_print2_scan2.npz", stage="dev")
    np.testing.assert_array_equal(f.ids, [2, 3])
    for j, i in enumerate(f.ids):
        y = to_gray(scans[2, i])
        np.testing.assert_array_equal(f.bits[j], binarize_vote(y, 3))
        assert f.pcor[j] == pytest.approx(pcor(t[i], y, 3), abs=1e-6)
    assert (f.meta["printer"], f.meta["session"], f.meta["print_run"], f.meta["scan_run"]) == (
        "HPI76",
        1,
        2,
        2,
    )


def test_rerun_skips_built_files(archive):
    sip, out, *_ = archive
    args = ["--sip-dir", str(sip), "--out", str(out), "--workers", "1"]
    mc.main(args)
    mtimes = {p.name: p.stat().st_mtime_ns for p in out.glob("*.npz")}
    mc.main(args)
    assert {p.name: p.stat().st_mtime_ns for p in out.glob("*.npz")} == mtimes


def test_only_still_builds_templates(archive):
    sip, out, *_ = archive
    mc.main(["--sip-dir", str(sip), "--out", str(out), "--workers", "1", "--only", "variab*"])
    assert sorted(p.stem for p in out.glob("*.npz")) == [
        "templates_original",
        "variability_HPI76_session1_print2_scan1",
        "variability_HPI76_session1_print2_scan2",
    ]


def test_pcor_withheld_on_test_fakes(archive):
    sip, out, t, _, photos, _ = archive
    base = ["--sip-dir", str(sip), "--out", str(out), "--workers", "1"]
    mc.main(base)
    fake = load_compact(out / "mobile_fake_EHPI55_iPhone12Pro_run1.npz", stage="test")
    np.testing.assert_array_equal(fake.ids, [2, 3])
    assert fake.pcor[0] == pytest.approx(pcor(t[2], to_gray(photos[2]), 1), abs=1e-6)
    assert np.isnan(fake.pcor[1])  # test template of a fake: withheld
    assert fake.meta["pcor_withheld_on_test"] is True
    original = load_compact(out / "mobile_original_iPhone12Pro_run1.npz", stage="test")
    assert np.isfinite(original.pcor).all()  # originals are free
    assert original.meta["pcor_withheld_on_test"] is False

    mc.main(base + ["--test-pcor", "--overwrite", "--only", "*fake*"])
    fake = load_compact(out / "mobile_fake_EHPI55_iPhone12Pro_run1.npz", stage="test")
    assert fake.pcor[1] == pytest.approx(pcor(t[3], to_gray(photos[3]), 1), abs=1e-6)
    assert fake.meta["pcor_withheld_on_test"] is False
