"""Download the parts of the SIP copy detection pattern archive that this project uses.

The archive (University of Geneva, Yareta) is one public 123 GB zip. The server accepts
HTTP range requests, so we read the zip index remotely and fetch only the members we
need. Each member is checked against its CRC. Interrupted downloads resume.

Source: https://yareta.unige.ch/archives/2eab9a72-dd60-4a5f-ba95-0144cc2983aa
DOI: 10.26037/yareta:hof52wzycncgvkkhm4nd7ta3ma (license CC BY-NC-ND 4.0)

Usage:
    python scripts/download_sip.py --list
    python scripts/download_sip.py mobile scanner
    python scripts/download_sip.py all

Standard library only. If the range requests stop working, download the archive by hand
from the page above and unzip the same folders into dataset/sip_yareta/.
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

URL = "https://access.yareta.unige.ch/access/metadata/2eab9a72-dd60-4a5f-ba95-0144cc2983aa/download"
PREFIX = "researchdata/"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "dataset" / "sip_yareta"
CHUNK = 1 << 20

# Regular expressions on the member path inside the archive.
_VAR = r"2022_Indigo_1x1_variability/"
GROUPS = {
    # Thesis dataset: 1440 templates, iPhone and Samsung, 6 captures, originals and fakes.
    "mobile": [r"^README\.txt$", r"^2023_Indigo_1x1_mobile/"],
    # Scanner dataset at 2400 dpi: 720 templates, two printers, four kinds of fakes.
    "scanner": [
        r"^README\.txt$",
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


class RemoteFile:
    """Minimal read-only file object over HTTP range requests, enough for zipfile to read the index."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.blocks: dict[int, bytes] = {}
        with self._open(0, 0) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def _open(self, lo: int, hi: int):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
        return urllib.request.urlopen(req, timeout=120)

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
                with self._open(b * CHUNK, min((b + 1) * CHUNK, self.size) - 1) as r:
                    self.blocks[b] = r.read()
        data = b"".join(self.blocks[b] for b in range(first, last + 1))
        start = self.pos - first * CHUNK
        self.pos += n
        return data[start : start + n]


def read_index(dest: Path) -> list[dict]:
    """Members of the remote zip: name, offset, sizes, crc. Cached in dest/_index.json."""
    cache = dest / "_index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    remote = RemoteFile(URL)
    with zipfile.ZipFile(remote) as z:
        index = [
            {
                "name": i.filename.removeprefix(PREFIX),
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


def select(index: list[dict], groups: list[str]) -> list[dict]:
    patterns = [p for g in groups for p in GROUPS[g]]
    chosen = [m for m in index if any(re.search(p, m["name"]) for p in patterns)]
    return sorted(chosen, key=lambda m: m["usize"])


def _range(lo: int, hi: int):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={lo}-{hi}"})
    return urllib.request.urlopen(req, timeout=120)


def fetch(member: dict, dest: Path) -> None:
    """Download one member, inflate it and check size and CRC."""
    out = dest / member["name"]
    if out.exists() and out.stat().st_size == member["usize"]:
        print(f"have  {member['name']}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with _range(member["offset"], member["offset"] + 29) as r:
        header = r.read()
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"unexpected zip header for {member['name']}")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = member["offset"] + 30 + name_len + extra_len

    raw = out.with_name(out.name + ".part")
    while True:
        done = raw.stat().st_size if raw.exists() else 0
        if done >= member["csize"]:
            break
        try:
            with _range(start + done, start + member["csize"] - 1) as r, open(raw, "ab") as f:
                while block := r.read(CHUNK):
                    f.write(block)
        except OSError as err:
            print(f"retry {member['name']} ({type(err).__name__})")
            time.sleep(5)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("groups", nargs="*", help=f"any of: {', '.join(GROUPS)}, all")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="target folder (default: dataset/sip_yareta)",
    )
    parser.add_argument(
        "--list", action="store_true", help="show what each group would download and exit"
    )
    args = parser.parse_args()

    groups = list(GROUPS) if "all" in args.groups or not args.groups else args.groups
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        parser.error(f"unknown group(s): {', '.join(unknown)}")

    index = read_index(args.dest)
    if args.list:
        for g in GROUPS:
            members = select(index, [g])
            print(f"{g}: {len(members)} files, {sum(m['usize'] for m in members) / 1e9:.1f} GB")
        return
    members = select(index, groups)
    print(f"{len(members)} files, {sum(m['usize'] for m in members) / 1e9:.1f} GB -> {args.dest}")
    for member in members:
        fetch(member, args.dest)
    print("done")


if __name__ == "__main__":
    main()
