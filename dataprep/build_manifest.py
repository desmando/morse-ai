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
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA_ROOT


# Older W1AW transcripts wrap the actual practice text in a spoken announcer
# preamble/postamble ("NOW 10 WPM TEXT IS FROM JANUARY 2017 QST PAGE 56 ...
# END OF 10 WPM TEXT QST DE W1AW"). That boilerplate is real keyed content (it
# gets sent too), but it's near-identical across hundreds of files and sits at
# a fixed position - at low WPM a 4s clip only spans a few characters, so the
# *first* clip of nearly every old file ends up labeled "NOW " and nothing
# else, drowning out real content in training. Strip it before slicing so the
# chars/sec estimate and clip boundaries are computed from the body text only.
#
# This isn't one fixed template - confirmed variants in the actual data:
#   "NOW 10 WPM TEXT IS FROM JANUARY 2017 QST PAGE 56 <body>"            (no dividers)
#   "= NOW 15 WPM = TEXT IS FROM OCTOBER 2022 QST PAGE 36 = <body>"      (= dividers)
#   "= NOW 18 WPM transition file follows = <body>"                     (no citation)
#   "NOW 40 WPM <body>"                                                 (no citation, no divider)
# and footers:
#   "<body> END OF 25 WPM TEXT QST DE W1AW"
#   "<body> = END OF 5 WPM TEXT = QST DE W1AW <"
#   "<body> = END OF 18 WPM transition file <"
_WPM_NUM = r"[\d/ ]+?"  # almost always integer, but at least one file says "7 1/2"
HEADER_RE = re.compile(
    rf"^=?\s*NOW {_WPM_NUM} WPM\s*(?:=\s*)?"
    r"(?:(?:TEXT IS FROM .*?QST(?: PAGE \d+)?|transition file follows)\s*(?:=\s*)?)?"
)
FOOTER_RE = re.compile(
    rf"\s*(?:=\s*)?END OF {_WPM_NUM} WPM\s+"
    r"(?:TEXT(?:\s*=?\s*QST DE W1AW)?|transition file)"
    r"\s*[<>]?\.?\s*$"
)


def clean_transcript(raw: str) -> str:
    # strip control chars (e.g. trailing 0x1A DOS EOF markers some of these old
    # files end with), plus the C1 range (0x80-0x9F) some older ARRL transcripts
    # use as decorative bullets around "NOW xx WPM"/"QST DE W1AW" announcements,
    # and U+FFFD (produced by decoding those same stray bytes as UTF-8) - none of
    # this is actually keyed, so it shouldn't end up in the label vocab.
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f�]", "", raw)
    text = re.sub(r"\s+", " ", text).strip()
    text = HEADER_RE.sub("", text, count=1)
    text = FOOTER_RE.sub("", text, count=1)
    return text.strip()


def slice_clips(audio, sr: int, text: str, clip_seconds: float, min_label_chars: int = 5):
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
        # short labels are where proportional-alignment rounding error hurts
        # most (relative to label length) and where CTC has the cheapest
        # incentive to learn "predict blank" instead of real structure.
        if len(label) >= min_label_chars:
            yield audio[start_sample:end_sample], label
        start_sample = end_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DATA_ROOT / "raw" / "arrl"))
    parser.add_argument("--clips-dir", default=str(DATA_ROOT / "processed" / "clips"))
    parser.add_argument("--manifest-out", default=str(DATA_ROOT / "manifests" / "clips_manifest.csv"))
    parser.add_argument("--vocab-out", default=str(DATA_ROOT / "manifests" / "vocab.txt"))
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--min-label-chars", type=int, default=5,
                         help="drop clips whose sliced label is shorter than this")
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
        txt_path_str, n_subs = re.subn(r"_(\d+)WPM\.mp3$", r"_\1.txt", str(mp3_path), flags=re.IGNORECASE)
        if n_subs == 0:
            # filename doesn't match the "..._NNWPM.mp3" convention at all (e.g.
            # leftover staging files) - re.sub would silently return the mp3 path
            # unchanged, which then "exists" as itself and gets read as binary
            # audio mis-decoded as text. Skip instead of risking that.
            print(f"  unrecognized filename pattern, skipping: {mp3_path.name}")
            continue
        txt_path = Path(txt_path_str)
        if not txt_path.exists():
            print(f"  no transcript for {mp3_path.name}, skipping")
            continue

        audio, sr = sf.read(mp3_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        text = clean_transcript(txt_path.read_text(encoding="utf-8", errors="replace"))

        out_dir = clips_dir / speed_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, (clip_audio, label) in enumerate(
                slice_clips(audio, sr, text, args.clip_seconds, args.min_label_chars)):
            clip_name = f"{mp3_path.stem}_{i:03d}.wav"
            clip_path = out_dir / clip_name
            sf.write(clip_path, clip_audio, sr)
            rows.append({
                "clip_path": clip_path.relative_to(DATA_ROOT).as_posix(),
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
