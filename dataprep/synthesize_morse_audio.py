"""Synthesize Morse audio directly from text (e.g. the contest-style QSO corpus
from lm/generate_qso_corpus.py), so the acoustic model gets far more training
volume than the handful of real ARRL recordings alone can provide.

Unlike build_manifest.py (which has to *approximate* text/audio alignment
for the real ARRL recordings, since there are no per-character timestamps),
this script controls the synthesis directly, so clip labels are exactly
aligned to the audio - no bleed across clip boundaries.

Output uses the same manifest schema as build_manifest.py, so it can be fed
straight into augment_hf_channel.py for noise/fading/QRM impairments:

  python synthesize_morse_audio.py --text-file <path to qso_corpus.txt>
  python augment_hf_channel.py --manifest <synth manifest> --out-dir ...

Usage:
  python synthesize_morse_audio.py --wpm-range 18,32 --tone-hz-range 400,900
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA_ROOT

SAMPLE_RATE = 8000

MORSE_CODE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.", "=": "-...-",
    "-": "-....-", "'": ".----.", '"': ".-..-.", "(": "-.--.", ")": "-.--.-",
}


def char_elements(ch: str, dot_s: float, dash_s: float, gap_s: float):
    code = MORSE_CODE.get(ch)
    if code is None:
        return []
    elements = []
    for i, sym in enumerate(code):
        if i > 0:
            elements.append((gap_s, False))
        elements.append((dash_s if sym == "-" else dot_s, True))
    return elements


def _jitter(duration: float, rng, amount: float) -> float:
    """Real keying isn't metronome-exact - nudge each element's duration by a
    small random factor (clamped so nothing collapses to ~0 or doubles)."""
    if amount <= 0:
        return duration
    factor = max(0.5, min(1.5, rng.gauss(1.0, amount)))
    return duration * factor


def synthesize_line(text: str, wpm: float, tone_hz: float, sr: int = SAMPLE_RATE,
                     farnsworth_wpm: float = None, ramp_s: float = 0.004,
                     timing_jitter: float = 0.0, rng=None):
    """Returns (audio, char_spans) where char_spans is [(char, start_s, end_s), ...]
    for every transmitted (non-space, mapped) character - exact, not approximate,
    even with timing_jitter applied (each element's randomized duration is what
    actually gets rendered, so spans always match the audio precisely)."""
    rng = rng or random.Random()
    char_wpm = farnsworth_wpm if farnsworth_wpm else wpm
    dot_s = 1.2 / char_wpm
    dash_s = 3 * dot_s
    intra_gap_s = dot_s

    dot_s_target = 1.2 / wpm
    inter_char_gap_s = 3 * dot_s_target
    inter_word_gap_s = 7 * dot_s_target

    elements: list[tuple[float, bool]] = []
    char_spans: list[tuple[str, float, float]] = []
    t = 0.0

    words = text.split(" ")
    for wi, word in enumerate(words):
        chars_in_word = [c for c in word if c in MORSE_CODE]
        for ci, ch in enumerate(chars_in_word):
            start = t
            for d, is_tone in char_elements(ch, dot_s, dash_s, intra_gap_s):
                d = _jitter(d, rng, timing_jitter)
                elements.append((d, is_tone))
                t += d
            char_spans.append((ch, start, t))
            if ci < len(chars_in_word) - 1:
                d = _jitter(inter_char_gap_s, rng, timing_jitter)
                elements.append((d, False))
                t += d
        if wi < len(words) - 1 and chars_in_word:
            start = t
            d = _jitter(inter_word_gap_s, rng, timing_jitter)
            elements.append((d, False))
            t += d
            # label the word gap itself as a " " character span - otherwise
            # the model never sees a space as a training target at all (no
            # other code path produces one) and can never learn to predict
            # one, making every decode one run-on word with no boundaries
            char_spans.append((" ", start, t))

    total_samples = int(t * sr) + 1
    audio = np.zeros(total_samples, dtype=np.float64)
    ramp_n = max(1, int(ramp_s * sr))
    pos = 0
    for duration, is_tone in elements:
        n = int(duration * sr)
        if is_tone and n > 0:
            tt = np.arange(n) / sr
            tone = np.sin(2 * np.pi * tone_hz * tt)
            env = np.ones(n)
            r = min(ramp_n, n // 2)
            if r > 0:
                ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
                env[:r] *= ramp
                env[-r:] *= ramp[::-1]
            audio[pos:pos + n] = tone * env
        pos += n

    return audio[:pos], char_spans


def slice_into_clips(audio, char_spans, sr: int, clip_seconds: float):
    """Cuts each clip exactly at a character's end boundary - never mid
    dot/dash/gap - so the label always matches the audio exactly, with no
    bleed in either direction. Previously this cut audio at a fixed clock
    tick while assigning labels by character *start* time only: a character
    straddling the cut got counted in full in the label whose audio actually
    only contained its first fragment, and the next clip got that same
    character's leftover audio with no label credit at all. Every clip
    boundary in every phase of training had a chance to hit this. Clips now
    vary in length around clip_seconds (snapped to the nearest character
    end) instead of landing on a fixed tick - CTC handles variable length
    fine, exact alignment matters far more than uniform duration."""
    n_samples = len(audio)
    n_spans = len(char_spans)
    start_sample = 0
    span_idx = 0
    while span_idx < n_spans:
        label_chars = []
        end_sample = start_sample
        while span_idx < n_spans:
            ch, _s, e = char_spans[span_idx]
            candidate_end = min(int(round(e * sr)), n_samples)
            # always take at least one character, even if it alone exceeds
            # clip_seconds - a window can't be shorter than one character
            if label_chars and (candidate_end - start_sample) > clip_seconds * sr:
                break
            label_chars.append(ch)
            end_sample = candidate_end
            span_idx += 1
        if label_chars:
            yield audio[start_sample:end_sample], "".join(label_chars)
        start_sample = end_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", default=str(DATA_ROOT / "text_corpus" / "qso_corpus.txt"))
    parser.add_argument("--out-dir", default=str(DATA_ROOT / "synthetic" / "clips"))
    parser.add_argument("--manifest-out", default=str(DATA_ROOT / "manifests" / "synthetic_manifest.csv"))
    parser.add_argument("--wpm-range", default="18,32")
    parser.add_argument("--tone-hz-range", default="400,900")
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--max-lines", type=int, default=0, help="0 = all lines in the text file")
    parser.add_argument("--timing-jitter", type=float, default=0.0,
                         help="relative std-dev of random per-element timing variation (e.g. 0.08 = ~8%%), "
                              "0 = perfectly metronomic. Real human keying isn't perfectly even.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    wpm_lo, wpm_hi = (float(x) for x in args.wpm_range.split(","))
    hz_lo, hz_hi = (float(x) for x in args.tone_hz_range.split(","))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)

    lines = [l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.max_lines:
        lines = lines[: args.max_lines]

    rows = []
    for i, line in enumerate(lines):
        wpm = rng.uniform(wpm_lo, wpm_hi)
        tone_hz = rng.uniform(hz_lo, hz_hi)
        amp = rng.uniform(0.6, 1.0)

        audio, char_spans = synthesize_line(line, wpm, tone_hz, timing_jitter=args.timing_jitter, rng=rng)
        audio = audio * amp

        for j, (clip_audio, label) in enumerate(slice_into_clips(audio, char_spans, SAMPLE_RATE, args.clip_seconds)):
            clip_name = f"synth_{i:05d}_{j:03d}.wav"
            clip_path = out_dir / clip_name
            sf.write(clip_path, clip_audio, SAMPLE_RATE)
            rows.append({
                "clip_path": clip_path.relative_to(DATA_ROOT).as_posix(),
                "label": label,
                "wpm": f"{wpm:.0f}",
                # one "source" per synthesized line, not one constant value for
                # everything - split_rows_by_source holds out whole sources, so
                # a single shared source value makes synthetic-only training
                # put 100% of the data in either train or val, never both.
                "source": f"synth_line_{i:05d}",
            })

        if i % 200 == 0:
            print(f"  {i}/{len(lines)} lines synthesized, {len(rows)} clips so far")

    with open(args.manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_path", "label", "wpm", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic clips to {out_dir}")
    print(f"Manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
