"""Downloads the FCC's bulk amateur radio license database (the Universal
Licensing System "complete" amateur extract) and builds a compact local
index of active US callsigns, so the TUI can flag a decoded callsign that
looks like a US callsign but isn't a real active license - a likely decode
error worth double-checking before logging, since a wrong callsign is a
much worse mistake than a wrong class/section.

Source and format verified directly against the real data (not assumed),
given the CI-V lesson earlier in this project:
  - Download URL: https://data.fcc.gov/download/pub/uls/complete/l_amat.zip
    (confirmed via FCC's own public-access-files page; ~175MB, refreshed
    weekly by the FCC).
  - HD.dat (one of several pipe-delimited files in the zip) carries the
    per-license header: callsign, license_status, plus other fields not
    needed here. Field positions (after the leading "HD" record-type
    marker) were confirmed by downloading the real file and inspecting
    actual rows: index 4 = callsign, index 5 = license_status ("A" =
    active - confirmed by also seeing "C" for cancelled and "E" for
    expired in real sample rows).
  - US callsign prefix pattern - confirmed by checking EVERY one of the
    827,263 real active licenses against it (zero mismatches): K, N, or W
    followed by any second letter (or none), OR "A" followed specifically
    by A-L (not the full A-Z range - e.g. AM-AZ are allocated to other
    countries) - then a digit and 1-4 letters.

This only covers US-licensed stations. A callsign not found here is NOT
necessarily wrong - it may be Canadian (VE/VA/VO/VY), other DX, or simply
outside the US prefix pattern, none of which this can verify. Only
US-pattern callsigns are checked at all (is_us_pattern), so foreign
callsigns are never flagged as "not found" in the first place.

Usage:
  python inference/fcc_uls.py --download   # fetch the latest zip + rebuild the index
  python inference/fcc_uls.py --rebuild    # rebuild the index from an already-downloaded zip
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA_ROOT

ULS_URL = "https://data.fcc.gov/download/pub/uls/complete/l_amat.zip"
ULS_DIR = DATA_ROOT / "fcc_uls"
ZIP_PATH = ULS_DIR / "l_amat.zip"
INDEX_PATH = ULS_DIR / "active_callsigns.txt"

# K/N/W + any optional second letter, or A + [A-L] specifically (the US's
# actual ITU-allocated slice of the "A" prefix block) - see module
# docstring for how this was verified against the real dataset.
US_CALLSIGN_RE = re.compile(r"^(?:[KNW][A-Z]?|A[A-L])[0-9][A-Z]{1,4}$")


def is_us_pattern(callsign: str) -> bool:
    return bool(US_CALLSIGN_RE.match(callsign.upper()))


def download_uls(url: str = ULS_URL, zip_path: Path = ZIP_PATH) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    return zip_path


def build_active_callsigns_index(zip_path: Path = ZIP_PATH, index_path: Path = INDEX_PATH) -> int:
    """Streams HD.dat directly out of the zip (no need to extract the much
    larger archive contents to disk) and writes every active license's
    callsign to index_path, one per line. Returns the count written."""
    active = set()
    with zipfile.ZipFile(zip_path) as zf, zf.open("HD.dat") as f:
        for raw_line in f:
            fields = raw_line.decode("latin-1").rstrip("\n").split("|")
            if len(fields) < 6:
                continue
            callsign, status = fields[4], fields[5]
            if callsign and status == "A":
                active.add(callsign.upper())

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        for callsign in sorted(active):
            f.write(callsign + "\n")
    return len(active)


def load_active_callsigns(index_path: Path = INDEX_PATH) -> set[str]:
    """Returns the set of active US callsigns from the prebuilt index, or
    an empty set if it hasn't been built yet (verification is then simply
    unavailable, not an error - callers should treat that as "can't check"
    rather than crash)."""
    if not index_path.exists():
        return set()
    return set(index_path.read_text(encoding="utf-8").split())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true", help="download the latest ULS zip, then rebuild the index")
    group.add_argument("--rebuild", action="store_true", help="rebuild the index from an already-downloaded zip")
    args = parser.parse_args()

    if args.download:
        print(f"Downloading {ULS_URL} ...")
        download_uls()
        print(f"Saved to {ZIP_PATH}")

    if not ZIP_PATH.exists():
        sys.exit(f"{ZIP_PATH} not found - run with --download first")

    print("Building active-callsigns index from HD.dat ...")
    count = build_active_callsigns_index()
    print(f"Wrote {count} active US callsigns to {INDEX_PATH}")


if __name__ == "__main__":
    main()
