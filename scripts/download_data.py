"""Download the datasets used in this project into dataset/.

Two public sources, no account and no token needed:

- Scantrust CDP dataset (Kaggle, 0.5 GB): one zip, downloaded and unpacked into
  dataset/unique_cdp/ and dataset/batch_cdp/.
- SIP Indigo datasets (University of Geneva, Yareta): one public 123 GB zip. The server
  accepts HTTP range requests, so we read the zip index remotely and fetch only the
  members we need into dataset/sip_yareta/. They stay zipped.

Downloads resume when interrupted. SIP files are checked against their CRC, the
Scantrust zip is checked when it is unpacked.

Usage:
    python scripts/download_data.py --list
    python scripts/download_data.py scantrust mobile
    python scripts/download_data.py all

Standard library only. Both datasets are CC BY-NC-ND 4.0: do not redistribute them.
If a source changes and this script fails, see the manual steps in the README.
"""

import argparse
import json
import re
import struct
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "dataset"
CHUNK = 1 << 20

# Scantrust: https://www.kaggle.com/datasets/scantrust/copy-detection-pattern-dataset
KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/scantrust/copy-detection-pattern-dataset"
)
SCANTRUST_FILES = 28550
SCANTRUST_GB = 0.5

# SIP: https://yareta.unige.ch/archives/2eab9a72-dd60-4a5f-ba95-0144cc2983aa
# doi:10.26037/yareta:hof52wzycncgvkkhm4nd7ta3ma
SIP_URL = (
    "https://access.yareta.unige.ch/access/metadata/2eab9a72-dd60-4a5f-ba95-0144cc2983aa/download"
)
SIP_PREFIX = "researchdata/"
_VAR = r"2022_Indigo_1x1_variability/"
# Regular expressions on the member path inside the SIP archive.
SIP_GROUPS = {
    # Thesis dataset: 1440 templates, iPhone and Samsung, 6 captures, originals and fakes.
    "mobile": [r"^README\.txt$", r"^2023_Indigo_1x1_mobile/"],
    # Scanner dataset at 2400 dpi: 720 templates, two printers, four kinds of fakes.
    "scanner": [
        r"^README\.txt$",
        r"^2023_Indigo_1x1_mobile/orig_templates/",
        r"^2021_Indigo_1x1_base/print_fake/",
        r"^2021_Indigo_1x1_base/print_original/.*_2400dpi\.zip$",
        r"^2021_Indigo_1x1_base/template_estimated/",
        r"^2021_Indigo_1x1_base/template_original/2400dpi/",
    ],
    # Repeated prints of the same templates on InvercoteG paper: ten print runs on HPI55,
    # three on HPI76, and the other print sessions of both printers.
    "variability": [
        r"^README\.txt$",
        r"^2023_Indigo_1x1_mobile/orig_templates/",
        _VAR + r"HPI55_printdpi812\.8_printrun\d+_session1_InvercoteG\.zip$",
        _VAR + r"HPI76_printdpi812\.8_printrun\d+_session1_InvercoteG\.zip$",
        _VAR + r"HPI(55|76)_printdpi812\.8_printrun1_session[02]_InvercoteG\.zip$",
    ],
}
GROUPS = ["scantrust", *SIP_GROUPS]


def open_range(url: str, lo: int, hi: int | str = ""):
    """GET url with a byte range. Redirects are followed and keep the range."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
    return urllib.request.urlopen(req, timeout=120)


def download_range(url: str, start: int, length: int, path: Path, label: str) -> None:
    """Append bytes [start, start + length) of url to path until it is complete."""
    while True:
        done = path.stat().st_size if path.exists() else 0
        if done >= length:
            return
        try:
            with open_range(url, start + done, start + length - 1) as r, open(path, "ab") as f:
                while block := r.read(CHUNK):
                    f.write(block)
        except OSError as err:
            print(f"retry {label} ({type(err).__name__})")
            time.sleep(5)


# ---------------------------------------------------------------- Scantrust


def scantrust_present(data_dir: Path) -> bool:
    folders = [data_dir / "unique_cdp", data_dir / "batch_cdp"]
    if not all(f.is_dir() for f in folders):
        return False
    return sum(1 for f in folders for p in f.rglob("*.png")) == SCANTRUST_FILES


def fetch_scantrust(data_dir: Path) -> None:
    """Download the Kaggle zip and unpack unique_cdp/ and batch_cdp/ into data_dir."""
    if scantrust_present(data_dir):
        print("have  scantrust (unique_cdp, batch_cdp)")
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    with open_range(KAGGLE_URL, 0, 0) as r:
        size = int(r.headers["Content-Range"].split("/")[1])
    archive = data_dir / "scantrust.zip.part"
    download_range(KAGGLE_URL, 0, size, archive, "scantrust")

    root = data_dir.resolve()
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member {bad} in {archive}, delete it and retry")
        for info in z.infolist():
            # Members are stored as dataset/unique_cdp/... and dataset/batch_cdp/...
            rel = info.filename.removeprefix("dataset/")
            target = (root / rel).resolve()
            if info.is_dir() or not target.is_relative_to(root):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
    if not scantrust_present(data_dir):
        raise RuntimeError(f"expected {SCANTRUST_FILES} png files after unpacking {archive}")
    archive.unlink()
    print(f"ok    scantrust  {SCANTRUST_FILES} files")


# ---------------------------------------------------------------------- SIP


class RemoteFile:
    """Read-only file object over HTTP range requests, enough for zipfile to read the index."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.blocks: dict[int, bytes] = {}
        with open_range(url, 0, 0) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = (0, self.pos, self.size)[whence] + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        first, last = self.pos // CHUNK, (self.pos + n - 1) // CHUNK
        for b in range(first, last + 1):
            if b not in self.blocks:
                with open_range(self.url, b * CHUNK, min((b + 1) * CHUNK, self.size) - 1) as r:
                    self.blocks[b] = r.read()
        data = b"".join(self.blocks[b] for b in range(first, last + 1))
        start = self.pos - first * CHUNK
        self.pos += n
        return data[start : start + n]


def sip_index(dest: Path) -> list[dict]:
    """Members of the remote zip: name, offset, sizes, crc. Cached in dest/_index.json."""
    cache = dest / "_index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    with zipfile.ZipFile(RemoteFile(SIP_URL)) as z:
        index = [
            {
                "name": i.filename.removeprefix(SIP_PREFIX),
                "offset": i.header_offset,
                "csize": i.compress_size,
                "usize": i.file_size,
                "crc": i.CRC,
                "deflate": i.compress_type == zipfile.ZIP_DEFLATED,
            }
            for i in z.infolist()
            if not i.is_dir()
        ]
    dest.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(index))
    return index


def sip_select(index: list[dict], groups: list[str]) -> list[dict]:
    patterns = [p for g in groups for p in SIP_GROUPS[g]]
    chosen = [m for m in index if any(re.search(p, m["name"]) for p in patterns)]
    return sorted(chosen, key=lambda m: m["usize"])


def fetch_sip(member: dict, dest: Path) -> None:
    """Download one member of the SIP archive, inflate it and check size and CRC."""
    out = dest / member["name"]
    if out.exists() and out.stat().st_size == member["usize"]:
        print(f"have  {member['name']}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with open_range(SIP_URL, member["offset"], member["offset"] + 29) as r:
        header = r.read()
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"unexpected zip header for {member['name']}")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = member["offset"] + 30 + name_len + extra_len

    raw = out.with_name(out.name + ".part")
    download_range(SIP_URL, start, member["csize"], raw, member["name"])

    inflater = zlib.decompressobj(-15) if member["deflate"] else None
    crc, size = 0, 0
    tmp = out.with_name(out.name + ".tmp")
    with open(raw, "rb") as f, open(tmp, "wb") as g:
        while block := f.read(4 * CHUNK):
            data = inflater.decompress(block) if inflater else block
            crc, size = zlib.crc32(data, crc), size + len(data)
            g.write(data)
        if inflater:
            data = inflater.flush()
            crc, size = zlib.crc32(data, crc), size + len(data)
            g.write(data)
    if size != member["usize"] or crc != member["crc"]:
        tmp.unlink()
        raise RuntimeError(f"size or CRC mismatch for {member['name']}, delete {raw} and retry")
    tmp.replace(out)
    raw.unlink()
    print(f"ok    {member['name']}  {size / 1e6:.1f} MB")


# --------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("groups", nargs="*", help=f"any of: {', '.join(GROUPS)}, all")
    parser.add_argument(
        "--dir", type=Path, default=DEFAULT_DIR, help="data folder (default: dataset/)"
    )
    parser.add_argument(
        "--list", action="store_true", help="show what each group would download and exit"
    )
    args = parser.parse_args()

    groups = GROUPS if "all" in args.groups or not args.groups else args.groups
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        parser.error(f"unknown group(s): {', '.join(unknown)}")
    sip_groups = [g for g in groups if g in SIP_GROUPS]
    sip_dest = args.dir / "sip_yareta"

    if args.list:
        print(f"scantrust: 1 zip, {SCANTRUST_GB} GB, unpacked to {SCANTRUST_FILES} files")
        index = sip_index(sip_dest)
        for g in SIP_GROUPS:
            members = sip_select(index, [g])
            print(f"{g}: {len(members)} files, {sum(m['usize'] for m in members) / 1e9:.1f} GB")
        return

    if "scantrust" in groups:
        fetch_scantrust(args.dir)
    if sip_groups:
        members = sip_select(sip_index(sip_dest), sip_groups)
        total = sum(m["usize"] for m in members) / 1e9
        print(f"sip: {len(members)} files, {total:.1f} GB -> {sip_dest}")
        for member in members:
            fetch_sip(member, sip_dest)
    print("done")
    if sip_groups:
        sip_dir = "" if args.dir == DEFAULT_DIR else f" --sip-dir {sip_dest}"
        print(
            "\nNext step for the Indigo data: build the compact pack used by the experiments"
            f" (about 0.5 GB, a few minutes):\n    python scripts/make_compact.py{sip_dir}"
        )


if __name__ == "__main__":
    main()
