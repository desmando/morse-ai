"""Combine clean + augmented (noisy) manifests, from both real and synthetic
audio, into one training manifest.

augment_hf_channel.py's output only contains noisy/faded/QRM'd variants, not
the clean originals - training on that alone means the model never sees an
easy example to anchor on, which risks CTC blank-collapse (the model settling
on "predict blank everywhere" rather than learning real structure). Mixing
clean clips back in gives the model both signals at once without the
distribution-shift risk of a clean-then-noisy curriculum. Synthetic clips
(exactly aligned, no proportional-slicing error) are mixed in alongside real
ARRL/on6zq clips for extra volume and cleaner labels.

Usage:
  python combine_manifests.py
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA_ROOT

FIELDS = ["clip_path", "label", "wpm", "source"]

DEFAULT_MANIFESTS = [
    str(DATA_ROOT / "manifests" / "clips_manifest.csv"),
    str(DATA_ROOT / "manifests" / "augmented_manifest.csv"),
    str(DATA_ROOT / "manifests" / "synthetic_manifest.csv"),
    str(DATA_ROOT / "manifests" / "augmented_synthetic_manifest.csv"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", nargs="+", default=DEFAULT_MANIFESTS)
    parser.add_argument("--out", default=str(DATA_ROOT / "manifests" / "combined_manifest.csv"))
    args = parser.parse_args()

    rows = []
    for path in args.manifests:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend({k: row[k] for k in FIELDS} for row in reader)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} combined clips ({' + '.join(args.manifests)}) to {args.out}")


if __name__ == "__main__":
    main()
