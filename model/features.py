"""Audio -> spectrogram-band features for the CW decoder.

Auto-detects the CW tone frequency per clip (handles different practice
sessions/radios using different sidetone pitches, and our synthetic frequency
drift augmentation), then extracts a fixed-width frequency band around it so
the model sees a consistent input shape regardless of tone pitch.
"""
import numpy as np
from scipy.signal import spectrogram

NFFT = 256
# Hop = NFFT - NOVERLAP = 112 samples (~14ms @ 8kHz). Frame count scales
# directly with clip duration regardless of how much real content is in it,
# while label length is capped by line/word boundaries - so longer clips
# alone make the frames-per-character ratio *worse*, not better (567 frames
# for ~7.5 chars at 4s; ~1139 frames for ~11 chars at 8s with the old hop of
# 56). Doubling the hop halves T at any clip length without losing audio
# data, directly improving that ratio. Still ~2 frames/dot at the fastest
# WPM we support (40 WPM dot = 30ms), enough resolution to resolve timing.
NOVERLAP = 144
HOP_SAMPLES = NFFT - NOVERLAP  # frame stride in samples, regardless of sr
N_FREQ_BINS = 32
TONE_FMIN_HZ = 300.0
TONE_FMAX_HZ = 1500.0


def detect_tone_freq(audio: np.ndarray, sr: int) -> float:
    freqs, _times, Sxx = spectrogram(audio, sr, nperseg=NFFT, noverlap=NOVERLAP, mode="magnitude")
    band_mask = (freqs >= TONE_FMIN_HZ) & (freqs <= TONE_FMAX_HZ)
    if not band_mask.any():
        return float(freqs[len(freqs) // 2])
    avg_power = Sxx[band_mask].mean(axis=1)
    return float(freqs[band_mask][np.argmax(avg_power)])


def extract_features(audio: np.ndarray, sr: int, n_freq_bins: int = N_FREQ_BINS) -> np.ndarray:
    """Returns a (T, n_freq_bins) log-magnitude feature matrix, normalized per-clip."""
    freqs, _times, Sxx = spectrogram(audio, sr, nperseg=NFFT, noverlap=NOVERLAP, mode="magnitude")
    tone_freq = detect_tone_freq(audio, sr)
    peak_bin = int(np.argmin(np.abs(freqs - tone_freq)))

    half = n_freq_bins // 2
    lo = max(0, peak_bin - half)
    hi = lo + n_freq_bins
    if hi > len(freqs):
        hi = len(freqs)
        lo = max(0, hi - n_freq_bins)
    band = Sxx[lo:hi, :]

    if band.shape[0] < n_freq_bins:
        band = np.pad(band, ((0, n_freq_bins - band.shape[0]), (0, 0)))

    log_band = np.log1p(band)
    log_band = (log_band - log_band.mean()) / (log_band.std() + 1e-6)
    return log_band.T.astype(np.float32)  # (T, n_freq_bins)
