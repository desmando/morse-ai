"""Relabel the real ARRL recordings with exact, model-derived character timing.

build_manifest.py had no per-character timestamps, so it sliced transcripts by
a uniform chars-per-second guess - labels routinely land mid-word, and every
clip boundary risks crediting a character to the wrong clip. That label noise
is baked into all real-audio clips, which is a real obstacle once real data is
mixed into training. This script replaces the guess with CTC forced alignment:

1. Run the trained model over each whole recording (overlap-stitched windows,
   same core-region scheme as decode_stream, but stitching log-prob FRAMES
   instead of decoded text - alignment needs one continuous lattice).
2. Viterbi forced alignment (model.decoder.ctc_forced_align) of the KNOWN
   transcript against that lattice -> exact per-character start/end times.
3. Re-slice into ~clip-seconds clips cut exactly at character end boundaries
   (never mid dot/dash), with labels that match the audio precisely.

Each clip also records a greedy_cer column - the model's free (greedy) decode
of that clip scored against the aligned label. It's a per-clip confidence
measure: near 0 means the model independently reads the same text, near 1
means either the audio is unusually hard or the transcript is wrong there.
Filter with --max-greedy-cer when building a training manifest; keep the
default (1.0 = keep everything) for a first pass and look at the distribution.

Chicken-and-egg note: alignment quality depends on the model being at least
roughly right about where characters are. Run this only once a checkpoint has
meaningful real-audio transfer (e.g. after noise-augmented training brings
real-ARRL CER well below ~50%), then fine-tune on the relabeled clips, and
optionally re-run with the improved model for even cleaner labels.

Usage:
  python dataprep/realign_arrl_labels.py --checkpoint <path> --device cuda
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataprep.build_manifest import clean_transcript
from model.decoder import cer, ctc_forced_align, ctc_greedy_decode, load_checkpoint_model
from model.features import SAMPLE_RATE, detect_tone_freq, extract_features, resample_to_model_rate
from model.vocab import Vocab
from paths import DATA_ROOT


def stitched_log_probs(audio: np.ndarray, model, device: str,
                        window_seconds: float = 8.0, stride_seconds: float = 4.0):
    """Whole-recording CTC lattice from overlapping windows.

    Same core-region tiling as model.decoder.decode_stream, but keeps each
    window's core log-prob FRAMES rather than its decoded text - forced
    alignment needs one continuous (T, vocab) lattice over the whole
    recording. Returns (log_probs (T, V) float32 numpy, frame_times (T,)
    seconds - absolute start time of each frame in the recording).

    The tone is detected ONCE for the whole recording (ARRL practice audio
    holds a constant sidetone) so a noisy window can't shift the feature band
    mid-file."""
    sr = SAMPLE_RATE
    n_samples = len(audio)
    window_samples = int(window_seconds * sr)
    stride_samples = int(stride_seconds * sr)
    guard_seconds = (window_seconds - stride_seconds) / 2
    total_seconds = n_samples / sr
    frame_s = model.frame_seconds(sr)
    tone = detect_tone_freq(audio[: min(n_samples, 60 * sr)], sr, hop_samples=model.hop_samples)

    lp_pieces, time_pieces = [], []
    window_start = 0
    while window_start < n_samples:
        window_end = min(window_start + window_samples, n_samples)
        is_first = window_start == 0
        is_last = window_end >= n_samples
        abs_start = window_start / sr
        core_start = 0.0 if is_first else abs_start + guard_seconds
        core_end = total_seconds if is_last else abs_start + guard_seconds + stride_seconds

        features = extract_features(audio[window_start:window_end], sr,
                                     hop_samples=model.hop_samples, tone_freq=tone)
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).to(device)
            lp = model(x)[0].cpu().numpy()  # (T_w, V)

        times = abs_start + np.arange(lp.shape[0]) * frame_s
        mask = (times >= core_start) & (times < core_end)
        lp_pieces.append(lp[mask])
        time_pieces.append(times[mask])

        if is_last:
            break
        window_start += stride_samples

    return np.concatenate(lp_pieces), np.concatenate(time_pieces)


def slice_aligned(kept_text: str, spans, frame_times: np.ndarray, frame_s: float,
                   clip_seconds: float):
    """Groups aligned characters into clips of ~clip_seconds, cutting only at
    character end boundaries. Yields (start_s, end_s, label, char_index_range).
    Characters the alignment gave zero frames inherit the running end time
    (they stay in the label - they're real transcript content, just squeezed)."""
    n = len(kept_text)
    i = 0
    clip_start = 0.0
    while i < n:
        label_chars = []
        end_s = clip_start
        first_i = i
        while i < n:
            f0, f1 = spans[i]
            char_end = end_s if f1 is None else float(frame_times[f1]) + frame_s
            if label_chars and (char_end - clip_start) > clip_seconds:
                break
            label_chars.append(kept_text[i])
            end_s = max(end_s, char_end)
            i += 1
        if label_chars:
            yield clip_start, end_s, "".join(label_chars).strip(), (first_i, i)
        clip_start = end_s


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                         help="checkpoint to align with - use the best real-audio-transfer model available")
    parser.add_argument("--vocab", default=str(DATA_ROOT / "manifests" / "vocab.txt"))
    parser.add_argument("--raw-dir", default=str(DATA_ROOT / "raw" / "arrl"))
    parser.add_argument("--clips-dir", default=str(DATA_ROOT / "realigned" / "clips"))
    parser.add_argument("--manifest-out", default=str(DATA_ROOT / "manifests" / "realigned_manifest.csv"))
    parser.add_argument("--clip-seconds", type=float, default=8.0,
                         help="target clip length (cut at the nearest character end boundary)")
    parser.add_argument("--min-label-chars", type=int, default=3)
    parser.add_argument("--max-greedy-cer", type=float, default=1.0,
                         help="drop clips whose greedy decode disagrees with the aligned label by more "
                              "than this CER (1.0 = keep everything, just record the column)")
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--stride-seconds", type=float, default=4.0)
    parser.add_argument("--max-files", type=int, default=0, help="0 = all")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    vocab = Vocab.from_file(args.vocab)
    model, _ckpt = load_checkpoint_model(args.checkpoint, vocab, device)
    print(f"model config: {model.config}")
    frame_s = model.frame_seconds(SAMPLE_RATE)

    raw_dir = Path(args.raw_dir)
    clips_dir = Path(args.clips_dir)
    manifest_path = Path(args.manifest_out)
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    mp3s = sorted(raw_dir.rglob("*.mp3"))
    if args.max_files:
        mp3s = mp3s[: args.max_files]

    rows = []
    n_dropped = 0
    for file_idx, mp3_path in enumerate(mp3s):
        txt_path_str, n_subs = re.subn(r"_(\d+)WPM\.mp3$", r"_\1.txt", str(mp3_path),
                                        flags=re.IGNORECASE)
        if n_subs == 0 or not Path(txt_path_str).exists():
            print(f"  no transcript for {mp3_path.name}, skipping")
            continue

        text = clean_transcript(Path(txt_path_str).read_text(encoding="utf-8", errors="replace")).upper()
        kept = "".join(c for c in text if c in vocab.char_to_idx)
        if len(kept) < args.min_label_chars:
            continue
        target_indices = [vocab.char_to_idx[c] for c in kept]

        audio, sr = sf.read(mp3_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = resample_to_model_rate(audio, sr)

        log_probs, frame_times = stitched_log_probs(audio, model, device,
                                                     args.window_seconds, args.stride_seconds)
        if log_probs.shape[0] < len(target_indices):
            print(f"  {mp3_path.name}: fewer lattice frames ({log_probs.shape[0]}) than target chars "
                  f"({len(target_indices)}) - transcript can't fit this audio, skipping")
            continue

        spans = ctc_forced_align(log_probs, target_indices)

        speed_dir = mp3_path.parent.name
        out_dir = clips_dir / speed_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        n_kept_clips = 0
        for j, (start_s, end_s, label, (ci0, ci1)) in enumerate(
                slice_aligned(kept, spans, frame_times, frame_s, args.clip_seconds)):
            if len(label) < args.min_label_chars:
                continue
            start_sample = max(0, int(start_s * SAMPLE_RATE))
            end_sample = min(len(audio), int(end_s * SAMPLE_RATE))
            if end_sample <= start_sample:
                continue

            # per-clip confidence: does the model's free decode of this
            # stretch of the lattice agree with the aligned label?
            f_lo, f_hi = np.searchsorted(frame_times, [start_s, end_s])
            pred = ctc_greedy_decode(torch.from_numpy(log_probs[f_lo:f_hi]).unsqueeze(0), vocab)[0]
            greedy_cer = cer(pred, label)
            if greedy_cer > args.max_greedy_cer:
                n_dropped += 1
                continue

            clip_name = f"{mp3_path.stem}_ra{j:03d}.wav"
            clip_path = out_dir / clip_name
            sf.write(clip_path, audio[start_sample:end_sample], SAMPLE_RATE)
            rows.append({
                "clip_path": clip_path.relative_to(DATA_ROOT).as_posix(),
                "label": label,
                "wpm": speed_dir,
                "source": mp3_path.name,
                "greedy_cer": f"{greedy_cer:.3f}",
            })
            n_kept_clips += 1

        print(f"[{file_idx + 1}/{len(mp3s)}] {mp3_path.name}: {len(kept)} chars -> "
              f"{n_kept_clips} clips")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_path", "label", "wpm", "source", "greedy_cer"])
        writer.writeheader()
        writer.writerows(rows)

    cers = [float(r["greedy_cer"]) for r in rows]
    print(f"\nWrote {len(rows)} realigned clips ({n_dropped} dropped by --max-greedy-cer) "
          f"to {clips_dir}")
    if cers:
        print(f"greedy_cer distribution: median {np.median(cers):.3f}, "
              f"90th pct {np.percentile(cers, 90):.3f} - re-run with --max-greedy-cer to filter")
    print(f"Manifest: {manifest_path}")
    print("Mix into training with dataprep/combine_manifests.py, or fine-tune on it directly "
          "(train.py --augment applies fresh impairments to these real clips too).")


if __name__ == "__main__":
    main()
