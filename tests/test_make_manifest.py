import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cdpbpc.io import INDIGO_COLUMNS, SCANTRUST_COLUMNS, read_manifest, save_compact

_spec = importlib.util.spec_from_file_location(
    "make_manifest", Path(__file__).resolve().parents[1] / "scripts" / "make_manifest.py"
)
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)

# Tiny Scantrust layout: 4 unique templates (fakes for 3 and 4), 2 batch templates
# (batch k = unique 2 + k). Batch file numbers follow the real numbering.
BATCH_FILES = {1: [1, 2, 101], 2: [11, 12]}
EXPECTED = {
    ("unique", "template"): 4,
    ("unique", "original"): 4,
    ("unique", "fake"): 2,
    ("batch", "template"): 2,
    ("batch", "original"): 5,
    ("batch", "fake"): 5,
}


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.fixture
def scantrust(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "SCANTRUST_EXPECTED", EXPECTED)
    monkeypatch.setattr(mm, "BATCH_ID_OFFSET", 2)
    u = tmp_path / "unique_cdp"
    for i in range(1, 5):
        _touch(u / "template" / f"{i:05d}.png")
        _touch(u / "original" / f"{i:05d}.png")
    for attack in mm.ATTACKS:
        for i in (3, 4):
            _touch(u / "copy" / attack / f"{i:05d}.png")
    _touch(u / ".DS_Store")
    b = tmp_path / "batch_cdp"
    for k, numbers in BATCH_FILES.items():
        _touch(b / "template" / f"{k}.png")
        for root in [b / "original"] + [b / "copy" / a for a in mm.ATTACKS]:
            for n in numbers:
                _touch(root / str(k) / f"{n:05d}.png")
    return tmp_path


def test_batch_numbering_on_the_real_layout():
    k = [mm._batch_template(n) for n in range(1, 2501)]
    assert Counter(k) == {t: 50 for t in range(1, 51)}
    # Each template sits on 5 sheets of 100 files, 10 copies per sheet.
    sheets = Counter((t, (n - 1) // 100) for n, t in zip(range(1, 2501), k))
    assert len(sheets) == 250 and set(sheets.values()) == {10}
    assert mm._batch_template(1) == 1 and mm._batch_template(11) == 2
    assert mm._batch_template(501) == 11 and mm._batch_template(2500) == 50


def test_scantrust_rows(scantrust):
    df = mm.scantrust_rows(scantrust)
    assert len(df) == 4 + 4 * 2 + 5 + 4 * 5
    assert list(df.columns) == list(SCANTRUST_COLUMNS)
    assert df["use"].dtype == bool and str(df["template_id"].dtype) == "Int64"
    assert str(df["attack"].dtype) == "string"
    unique = df[df["dataset"] == "unique"]
    assert set(unique["session"]) == {"U"} and unique["print_id"].isna().all()
    assert sorted(unique.loc[unique["class"] == "fake", "template_id"].unique()) == [3, 4]
    batch = df[df["dataset"] == "batch"]
    assert set(batch["template_id"]) == {3, 4} and set(batch["session"]) == {"B"}
    assert (~df["use"] == ((df["dataset"] == "batch") & (df["attack"] == "deep"))).all()
    row = batch[(batch["class"] == "original") & (batch["print_id"] == 101)].iloc[0]
    assert (row["template_id"], row["group"], row["sheet"]) == (3, 0, 1)
    assert row["path"] == "batch_cdp/original/1/00101.png"
    assert (df["attack"].isna() == (df["class"] == "original")).all()


def test_scantrust_rejects_misplaced_batch_file(scantrust):
    _touch(scantrust / "batch_cdp" / "original" / "1" / "00011.png")  # file 11 is template 2
    with pytest.raises(ValueError):
        mm.scantrust_rows(scantrust)


def test_scantrust_rejects_fakes_without_original(scantrust):
    folder = scantrust / "batch_cdp" / "copy" / "otsu" / "1"
    (folder / "00002.png").unlink()
    _touch(folder / "00003.png")
    with pytest.raises(ValueError):
        mm.scantrust_rows(scantrust)


def test_scantrust_rejects_missing_files(scantrust):
    (scantrust / "unique_cdp" / "original" / "00002.png").unlink()
    with pytest.raises(ValueError):
        mm.scantrust_rows(scantrust)


def _capture(pack, name, ids, **meta):
    full = {
        "subset": "mobile",
        "class": "original",
        "fake_type": "",
        "printer": "HPI55",
        "session": -1,
        "print_run": -1,
        "scan_run": -1,
        "device": "",
        "capture_run": -1,
    }
    full.update(meta)
    zeros = np.zeros(len(ids))
    bits = np.zeros((len(ids), 228, 228), np.uint8)
    save_compact(
        pack / f"{name}.npz", ids=ids, bits=bits, meta=full, pcor=zeros, mean=zeros, otsu=zeros
    )


@pytest.fixture
def pack(tmp_path):
    pack = tmp_path / "indigo_compact"
    pack.mkdir()
    ids = [1, 288, 289, 1440]
    _capture(pack, "base_original_HPI76", ids[:3], subset="base", printer="HPI76")
    _capture(
        pack,
        "base_fake_HPI55_EHPI76",
        ids[:3],
        subset="base",
        **{"class": "fake"},
        fake_type="EHPI76",
    )
    _capture(
        pack,
        "mobile_original_iPhone12Pro_run2",
        ids,
        session=1,
        print_run=1,
        device="iPhone12Pro",
        capture_run=2,
    )
    _capture(
        pack,
        "variability_HPI55_session2_print1_scan1",
        ids,
        subset="variability",
        session=2,
        print_run=1,
        scan_run=1,
    )
    t = np.zeros((4, 228, 228), np.uint8)
    save_compact(pack / "templates_original.npz", ids=ids, bits=t, meta={"class": "template"})
    return pack


def test_indigo_rows(pack):
    df = mm.indigo_rows(pack)
    assert list(df.columns) == list(INDIGO_COLUMNS)
    assert len(df) == 3 + 3 + 4 + 4  # templates are not listed
    assert (
        (df["page"] == (df["template_id"] - 1) // 144)
        & (df["page_pos"] == (df["template_id"] - 1) % 144)
    ).all()
    assert (df["stage"].eq("dev") == (df["template_id"] <= 288)).all()

    base = df[df["file"] == "base_original_HPI76"]
    assert (base[["session", "print_run", "scan_run"]] == [0, 1, 1]).all().all()
    fake = df[df["file"] == "base_fake_HPI55_EHPI76"]
    assert fake[["session", "print_run", "scan_run"]].isna().all().all()
    assert set(fake["fake_type"]) == {"EHPI76"}
    phone = df[df["subset"] == "mobile"]
    assert phone["scan_run"].isna().all() and set(phone["capture_run"]) == {2}
    assert phone["row"].tolist() == [0, 1, 2, 3]
    var = df[df["subset"] == "variability"]
    assert var["device"].isna().all() and set(var["session"]) == {2}


def test_main_writes_readable_manifests(scantrust, pack):
    pack.rename(scantrust / "indigo_compact")
    assert mm.main(["--data-dir", str(scantrust)]) == 0
    s = read_manifest(scantrust / "manifest.csv")
    i = read_manifest(scantrust / "manifest_indigo.csv")
    pd.testing.assert_frame_equal(s, mm.scantrust_rows(scantrust))
    pd.testing.assert_frame_equal(i, mm.indigo_rows(scantrust / "indigo_compact"))
    assert s["use"].dtype == bool and str(i["session"].dtype) == "Int64"


def test_main_without_data(tmp_path):
    assert mm.main(["--data-dir", str(tmp_path)]) == 1


def test_read_manifest_rejects_other_csv(tmp_path):
    path = tmp_path / "other.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_manifest(path)
