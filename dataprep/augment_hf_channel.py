"""Synthesize noisy/faded/QRM'd HF-channel variants of the clean ARRL clips.

The ARRL clips are clean studio-quality 750 Hz tone - nothing like what actually
comes out of an HF receiver (band-limited noise/QRN, ionospheric fading/QSB,
frequency drift, other CW signals bleeding through as QRM). This script adds
those impairments synthetically so the acoustic model sees something closer to
real off-air audio during training.

Ground-truth label text is unaffected by any of this - only the audio changes.

Usage:
  python augment_hf_channel.py --variants-per-clip 3 --snr-db-range -3,20
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, hilbert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA_ROOT


def bandlimited_noise(n_samples: int, sr: int, rng: np.random.Generator,
                       low_hz: float = 300.0, high_hz: float = 2500.0) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, n_samples)
    nyq = sr / 2
    b, a = butter(4, [low_hz / nyq, min(high_hz / nyq, 0.99)], btype="band")
    return filtfilt(b, a, noise)


def add_noise(audio: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = bandlimited_noise(len(audio), sr, rng)
    sig_power = np.mean(audio ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise *= np.sqrt(target_noise_power / noise_power)
    return audio + noise


def apply_fading(audio: np.ndarray, sr: int, rng: np.random.Generator,
                  fade_rate_hz: float = None, depth: float = 0.7) -> np.ndarray:
    if fade_rate_hz is None:
        fade_rate_hz = rng.uniform(0.1, 2.0)
    nyq = sr / 2
    b, a = butter(2, fade_rate_hz / nyq, btype="low")
    raw = rng.normal(0.0, 1.0, len(audio))
    envelope = np.abs(filtfilt(b, a, raw))
    envelope = envelope / (envelope.max() + 1e-12)
    gain = (1 - depth) + depth * envelope
    return audio * gain


def apply_freq_drift(audio: np.ndarray, sr: int, rng: np.random.Generator,
                      base_shift_hz: float = None, drift_amp_hz: float = None,
                      drift_rate_hz: float = None) -> np.ndarray:
    if base_shift_hz is None:
        base_shift_hz = rng.uniform(-30.0, 30.0)
    if drift_amp_hz is None:
        drift_amp_hz = rng.uniform(0.0, 8.0)
    if drift_rate_hz is None:
        drift_rate_hz = rng.uniform(0.02, 0.2)
    t = np.arange(len(audio)) / sr
    phase0 = rng.uniform(0, 2 * np.pi)
    shift_curve = base_shift_hz + drift_amp_hz * np.sin(2 * np.pi * drift_rate_hz * t + phase0)
    analytic = hilbert(audio)
    phase = 2 * np.pi * np.cumsum(shift_curve) / sr
    shifted = analytic * np.exp(1j * phase)
    return shifted.real


def add_qrm(audio: np.ndarray, sr: int, rng: np.random.Generator,
            amplitude: float = None, freq_hz: float = None,
            element_seconds: float = None) -> np.ndarray:
    if amplitude is None:
        amplitude = rng.uniform(0.1, 0.5)
    if freq_hz is None:
        freq_hz = rng.uniform(400.0, 1200.0)
    if element_seconds is None:
        element_seconds = rng.uniform(0.05, 0.15)

    n_samples = len(audio)
    t = np.arange(n_samples) / sr
    tone = np.sin(2 * np.pi * freq_hz * t)

    element_samples = max(1, int(element_seconds * sr))
    n_elements = n_samples // element_samples + 1
    keying = rng.random(n_elements) > 0.5
    keying = np.repeat(keying, element_samples)[:n_samples].astype(float)
    # smooth the on/off transitions a touch so it isn't pure clicks
    b, a = butter(2, 0.3)
    keying = filtfilt(b, a, keying)

    return audio + amplitude * tone * keying


def augment_clip(audio: np.ndarray, sr: int, rng: np.random.Generator, snr_range) -> np.ndarray:
    out = audio.copy()
    if rng.random() < 0.8:
        out = apply_freq_drift(out, sr, rng)
    if rng.random() < 0.7:
        out = apply_fading(out, sr, rng)
    if rng.random() < 0.5:
        out = add_qrm(out, sr, rng)
    snr_db = rng.uniform(*snr_range)
    out = add_noise(out, sr, snr_db, rng)
    peak = np.max(np.abs(out)) + 1e-12
    if peak > 1.0:
        out = out / peak
    return out, snr_db


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DATA_ROOT / "manifests" / "clips_manifest.csv"))
    parser.add_argument("--out-dir", default=str(DATA_ROOT / "augmented"))
    parser.add_argument("--manifest-out", default=str(DATA_ROOT / "manifests" / "augmented_manifest.csv"))
    parser.add_argument("--variants-per-clip", type=int, default=3)
    parser.add_argument("--snr-db-range", default="-3,20")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    snr_range = tuple(float(x) for x in args.snr_db_range.split(","))
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        clean_rows = list(reader)

    for row in clean_rows:
        clip_path = DATA_ROOT / row["clip_path"]
        audio, sr = sf.read(clip_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        speed_dir = out_dir / row["wpm"]
        speed_dir.mkdir(parents=True, exist_ok=True)

        for v in range(args.variants_per_clip):
            aug_audio, snr_db = augment_clip(audio, sr, rng, snr_range)
            aug_name = f"{clip_path.stem}_aug{v:02d}.wav"
            aug_path = speed_dir / aug_name
            sf.write(aug_path, aug_audio, sr)
            rows.append({
                "clip_path": str(aug_path.relative_to(DATA_ROOT)),
                "label": row["label"],
                "wpm": row["wpm"],
                "source": row["source"],
                "snr_db": f"{snr_db:.1f}",
            })

    with open(args.manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_path", "label", "wpm", "source", "snr_db"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} augmented clips ({args.variants_per_clip}x{len(clean_rows)}) to {out_dir}")
    print(f"Manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
