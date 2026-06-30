"""Import manually-downloaded ON6ZQ CW practice audio + transcripts.

on6zq.be's robots.txt explicitly disallows AI-agent user agents (ClaudeBot,
GPTBot, Google-Extended, etc.) site-wide, so this is NOT an automated
fetcher - you download the files yourself, and this script just reshapes
them into the same raw/<source>/<speed>wpm/ layout that fetch_arrl.py
produces, so build_manifest.py can pick them up unmodified.

What to download (9 pages, 5 speeds each = 45 mp3s + 9 transcripts):
  https://on6zq.be/w/index.php/CW/100MostCommonEnglishWords
  https://on6zq.be/w/index.php/CW/500MostCommonEnglishWords
  https://on6zq.be/w/index.php/CW/If
  https://on6zq.be/w/index.php/CW/EnglishSayings
  https://on6zq.be/w/index.php/CW/NationalCapitals
  https://on6zq.be/w/index.php/CW/EnglishQuotations01
  https://on6zq.be/w/index.php/CW/EnglishQuotations02
  https://on6zq.be/w/index.php/CW/EnglishQuotations03
  https://on6zq.be/w/index.php/CW/EnglishQuotations04
(Wordsworth has no embedded transcript on the page - skip it.)

For each page:
  1. Download all 5 "NN Words Per Minute" mp3 links, keeping the
     browser-given filename as-is (e.g. "If12.mp3", "EnglishQuotations0128.mp3").
  2. Copy the plain-text passage shown on the page into a "body.txt" file.
  3. Put each page's mp3s + its body.txt in their own subfolder (any name) -
     default staging root: <DATA_ROOT>/raw/manual/. The script finds mp3s
     recursively and looks for a "body.txt" next to each one.

Then run:
  python import_on6zq.py
  python build_manifest.py --raw-dir <DATA_ROOT>/raw   # scans arrl/ and on6zq/ together

Anything under the staging dir that isn't a recognized "<page><speed>.mp3"
name (e.g. unrelated audio, files with no body.txt next to them) is skipped
and reported, not guessed at.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA_ROOT

PAGES = [
    "100MostCommonEnglishWords", "500MostCommonEnglishWords", "If",
    "EnglishSayings", "NationalCapitals",
    "EnglishQuotations01", "EnglishQuotations02", "EnglishQuotations03", "EnglishQuotations04",
]


def parse_mp3_name(stem: str):
    """'EnglishQuotations0112' -> ('EnglishQuotations01', 12), or None if unrecognized."""
    for page in PAGES:
        if stem.startswith(page) and stem[len(page):].isdigit():
            return page, int(stem[len(page):])
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging-dir", default=str(DATA_ROOT / "raw" / "manual"))
    parser.add_argument("--out-dir", default=str(DATA_ROOT / "raw" / "on6zq"))
    args = parser.parse_args()

    staging_dir = Path(args.staging_dir)
    out_dir = Path(args.out_dir)

    if not staging_dir.exists():
        print(f"Staging dir not found: {staging_dir}")
        print("Create it and drop the manually-downloaded mp3s + body.txt transcripts there first.")
        return

    imported, skipped = 0, 0
    for mp3_path in sorted(staging_dir.rglob("*.mp3")):
        parsed = parse_mp3_name(mp3_path.stem)
        if parsed is None:
            print(f"  skip (unrecognized filename): {mp3_path.relative_to(staging_dir)}")
            skipped += 1
            continue
        page, speed = parsed

        txt_path = mp3_path.parent / "body.txt"
        if not txt_path.exists():
            txt_path = staging_dir / f"{page}.txt"
        if not txt_path.exists():
            print(f"  skip (no body.txt next to it): {mp3_path.relative_to(staging_dir)}")
            skipped += 1
            continue
        text = re.sub(r"\s+", " ", txt_path.read_text(encoding="utf-8", errors="replace")).strip()

        speed_dir = out_dir / f"{speed}wpm"
        speed_dir.mkdir(parents=True, exist_ok=True)

        dest_mp3 = speed_dir / f"{page}_{speed}WPM.mp3"
        dest_txt = speed_dir / f"{page}_{speed}.txt"
        shutil.copy2(mp3_path, dest_mp3)
        dest_txt.write_text(text, encoding="utf-8")
        imported += 1

    print(f"Imported {imported} clip pairs into {out_dir} ({skipped} skipped)")


if __name__ == "__main__":
    main()
