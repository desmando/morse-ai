"""Builds a series of manifests at increasing noisy/augmented-clip ratios,
for a gradual curriculum transition from the clean-only basis phase to the
full combined_manifest.csv. A single abrupt switch from 0% to ~75% noisy
data risks destabilizing training the same way the original silent
divergence did (see CLOUD_TRAINING.md) - this ramps it in smaller steps
instead, e.g. 5% -> 10% -> 20% -> 35% -> 50%, each resumed from the
previous step's best checkpoint.

All of synthetic_manifest.csv's clean clips are kept in every ramp step;
only the augmented (noisy) clip count varies, so "ratio" means the
fraction of the OUTPUT manifest that's noisy, not a fraction of the noisy
data itself: ratio = noisy / (clean + noisy).

The 75% ratio step is just combined_manifest.csv itself (all clean +
all available augmented clips) - no need to generate a separate file for
that endpoint.

Usage:
  python dataprep/build_noise_ramp_manifests.py --ratios 0.05,0.10,0.20,0.35,0.50
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA_ROOT


def load_rows(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-manifest", default=str(DATA_ROOT / "manifests" / "synthetic_manifest.csv"))
    parser.add_argument("--noisy-manifest",
                         default=str(DATA_ROOT / "manifests" / "augmented_synthetic_manifest.csv"))
    parser.add_argument("--out-dir", default=str(DATA_ROOT / "manifests" / "noise_ramp"))
    parser.add_argument("--ratios", default="0.05,0.10,0.20,0.35,0.50",
                         help="target fraction of each output manifest that's noisy/augmented")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clean_rows = load_rows(args.clean_manifest)
    noisy_rows = load_rows(args.noisy_manifest)
    print(f"clean rows: {len(clean_rows)}, noisy rows available: {len(noisy_rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(noisy_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The two manifests don't share an identical schema (e.g. augmented_synthetic
    # has an extra snr_db column) - union the fieldnames so DictWriter doesn't
    # choke on either side's rows; clean rows just get a blank snr_db.
    fieldnames = list(clean_rows[0].keys())
    for extra in noisy_rows[0].keys():
        if extra not in fieldnames:
            fieldnames.append(extra)

    n_clean = len(clean_rows)
    for ratio in (float(r) for r in args.ratios.split(",")):
        if not 0 < ratio < 1:
            print(f"skipping ratio {ratio} - must be between 0 and 1 (exclusive)")
            continue
        n_noisy_needed = int(round(ratio * n_clean / (1 - ratio)))
        if n_noisy_needed > len(noisy_rows):
            print(f"WARNING: ratio {ratio:.0%} needs {n_noisy_needed} noisy rows but only "
                  f"{len(noisy_rows)} are available - capping (actual ratio will be lower)")
            n_noisy_needed = len(noisy_rows)

        rows = clean_rows + noisy_rows[:n_noisy_needed]
        rng.shuffle(rows)
        actual_ratio = n_noisy_needed / len(rows)

        out_path = out_dir / f"ramp_{int(round(ratio * 100)):02d}pct.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{out_path.name}: {len(rows)} rows ({n_noisy_needed} noisy) - "
              f"target {ratio:.0%}, actual {actual_ratio:.1%}")

    print(f"\nWrote ramp manifests to {out_dir}")
    print("Next step beyond the last ratio here is combined_manifest.csv itself "
          "(all clean + all available augmented, ~75% noisy) - no separate file needed for that.")


if __name__ == "__main__":
    main()
