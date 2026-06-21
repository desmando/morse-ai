"""Segment ARRL audio+transcript pairs into ~4s clips with aligned text labels.

We don't have per-character timestamps, so alignment is approximate: each
transcript's average chars/second (derived from total clip duration and
transcript length) is used to slice the text proportionally to the audio
windows. This is the same approximation used by prior CW-decoder work (e.g.
AG1LE's CNN-LSTM-CTC decoder) and is good enough for CTC training, which
doesn't require exact frame-level alignment - it marginalizes over alignments
during training. Expect some character bleed across clip boundaries.

Usage:
  python build_manifest.py --clip-seconds 4.0
"""
import argparse
import csv
import re
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent


def clean_transcript(raw: str) -> str:
    # strip control chars (e.g. trailing 0x1A DOS EOF markers some of these old
    # files end with) before collapsing whitespace
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", raw)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slice_clips(audio, sr: int, text: str, clip_seconds: float):
    n_samples = len(audio)
    duration = n_samples / sr
    if duration <= 0 or not text:
        return
    chars_per_second = len(text) / duration
    clip_samples = int(clip_seconds * sr)

    start_sample = 0
    while start_sample < n_samples:
        end_sample = min(start_sample + clip_samples, n_samples)
        t0, t1 = start_sample / sr, end_sample / sr
        c0 = round(t0 * chars_per_second)
        c1 = round(t1 * chars_per_second)
        label = text[c0:c1].strip()
        if label:
            yield audio[start_sample:end_sample], label
        start_sample = end_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw" / "arrl"))
    parser.add_argument("--clips-dir", default=str(REPO_ROOT / "data" / "processed" / "clips"))
    parser.add_argument("--manifest-out", default=str(REPO_ROOT / "data" / "manifests" / "clips_manifest.csv"))
    parser.add_argument("--vocab-out", default=str(REPO_ROOT / "data" / "manifests" / "vocab.txt"))
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    clips_dir = Path(args.clips_dir)
    manifest_path = Path(args.manifest_out)
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    vocab = set()

    for mp3_path in sorted(raw_dir.rglob("*.mp3")):
        speed_dir = mp3_path.parent.name
        txt_path = re.sub(r"_(\d+)WPM\.mp3$", r"_\1.txt", str(mp3_path), flags=re.IGNORECASE)
        txt_path = Path(txt_path)
        if not txt_path.exists():
            print(f"  no transcript for {mp3_path.name}, skipping")
            continue

        audio, sr = sf.read(mp3_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        text = clean_transcript(txt_path.read_text(errors="replace"))

        out_dir = clips_dir / speed_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, (clip_audio, label) in enumerate(slice_clips(audio, sr, text, args.clip_seconds)):
            clip_name = f"{mp3_path.stem}_{i:03d}.wav"
            clip_path = out_dir / clip_name
            sf.write(clip_path, clip_audio, sr)
            rows.append({
                "clip_path": str(clip_path.relative_to(REPO_ROOT)),
                "label": label,
                "wpm": speed_dir,
                "source": mp3_path.name,
            })
            vocab.update(label)

        print(f"{mp3_path.name}: {len(text)} chars -> clips in {out_dir}")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_path", "label", "wpm", "source"])
        writer.writeheader()
        writer.writerows(rows)

    with open(args.vocab_out, "w", encoding="utf-8") as f:
        for ch in sorted(vocab):
            f.write(ch + "\n")

    print(f"Wrote {len(rows)} clips to manifest: {manifest_path}")
    print(f"Vocab ({len(vocab)} chars) written to: {args.vocab_out}")


if __name__ == "__main__":
    main()
