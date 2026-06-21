"""Download ARRL W1AW code practice MP3s + matching text transcripts.

These are real CW audio recordings (clean 750 Hz tone) with ground-truth text,
pulled from ARRL's per-speed archive pages, e.g.:
  https://www.arrl.org/20-wpm-code-archive

Each archive page lists pairs of files like:
  .../260106_20WPM.mp3   (audio)
  .../260106_20.txt      (transcript)

Usage:
  python fetch_arrl.py --speeds 5,10,13,15,18,20,25,30,35,40 --limit-per-speed 10
"""
import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

HEADERS = {
    "User-Agent": "morse-ai-dataprep/0.1 (personal research project; contact combat.medic@gmail.com)"
}

MP3_LINK_RE = re.compile(r'href="([^"]*?(\d{6})_(\d+)WPM\.mp3)"', re.IGNORECASE)

DEFAULT_SPEEDS = ["5", "10", "13", "15", "18", "20", "25", "30", "35", "40"]


def archive_url_for_speed(speed: str) -> str:
    slug = speed.replace(".", "-")
    return f"https://www.arrl.org/{slug}-wpm-code-archive"


def find_clip_pairs(archive_html: str, page_url: str) -> list[tuple[str, str]]:
    pairs = []
    for href, _date, _speed in MP3_LINK_RE.findall(archive_html):
        mp3_url = urljoin(page_url, href)
        txt_url = re.sub(r"_(\d+)WPM\.mp3$", r"_\1.txt", mp3_url, flags=re.IGNORECASE)
        pairs.append((mp3_url, txt_url))
    return pairs


def download(url: str, dest: Path, delay: float) -> bool:
    if dest.exists():
        return False
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        print(f"  skip (HTTP {resp.status_code}): {url}")
        return False
    dest.write_bytes(resp.content)
    time.sleep(delay)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds", default=",".join(DEFAULT_SPEEDS),
                         help="comma-separated WPM speeds, e.g. 5,10,20,40")
    parser.add_argument("--limit-per-speed", type=int, default=10,
                         help="max clip pairs to download per speed (0 = no limit)")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "raw" / "arrl"))
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between downloads")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    speeds = [s.strip() for s in args.speeds.split(",") if s.strip()]

    total_downloaded = 0
    for speed in speeds:
        archive_url = archive_url_for_speed(speed)
        print(f"[{speed} WPM] fetching archive page: {archive_url}")
        try:
            resp = requests.get(archive_url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  could not fetch archive page for {speed} WPM: {exc}")
            continue

        pairs = find_clip_pairs(resp.text, resp.url)
        if not pairs:
            print(f"  no clip links found for {speed} WPM, skipping")
            continue

        if args.limit_per_speed:
            pairs = pairs[: args.limit_per_speed]

        speed_dir = out_root / f"{speed}wpm"
        speed_dir.mkdir(parents=True, exist_ok=True)

        for mp3_url, txt_url in pairs:
            mp3_dest = speed_dir / Path(mp3_url).name
            txt_dest = speed_dir / Path(txt_url).name
            got_mp3 = download(mp3_url, mp3_dest, args.delay)
            got_txt = download(txt_url, txt_dest, args.delay)
            if got_mp3 or got_txt:
                total_downloaded += 1
                print(f"  saved {mp3_dest.name}")

    print(f"Done. {total_downloaded} clip pairs newly downloaded into {out_root}")


if __name__ == "__main__":
    main()
