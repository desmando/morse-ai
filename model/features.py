"""Audio -> spectrogram-band features for the CW decoder.

Auto-detects the CW tone frequency per clip (handles different practice
sessions/radios using different sidetone pitches, and our synthetic frequency
drift augmentation), then extracts a fixed-width frequency band around it so
the model sees a consistent input shape regardless of tone pitch.

The feature contract (sample rate, hop) is part of the trained model, not a
global constant: checkpoints record their hop_samples (see CWDecoder.config),
and everything downstream must use the checkpoint's value, not whatever this
module's default happens to be. Legacy checkpoints (no recorded config) used
LEGACY_HOP_SAMPLES with no CNN time-downsampling.
"""
import numpy as np
from scipy.signal import resample_poly, spectrogram

SAMPLE_RATE = 8000  # the model input contract - resample everything to this

NFFT = 256
# Legacy recipe: hop = 112 samples (~14ms @ 8kHz), no CNN time-downsampling.
# v4 recipe: hop = 56 samples (~7ms) into the CNN, which then downsamples
# time by 2x (CWDecoder time_stride=2) - the LSTM still sees ~14ms frames
# (the right rate for ~2 frames per dot at 40 WPM), but the CNN gets double
# the temporal resolution to find keying edges in before summarizing.
LEGACY_HOP_SAMPLES = 112
DEFAULT_HOP_SAMPLES = 56
N_FREQ_BINS = 32
TONE_FMIN_HZ = 300.0
TONE_FMAX_HZ = 1500.0


def resample_to_model_rate(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample to SAMPLE_RATE. The spectrogram parameters (NFFT, hop) are in
    samples, so feeding audio at any other rate silently changes the model's
    time/frequency resolution - always go through this first."""
    if sr == SAMPLE_RATE:
        return audio
    from math import gcd
    g = gcd(int(sr), SAMPLE_RATE)
    return resample_poly(audio, SAMPLE_RATE // g, int(sr) // g)


def _compute_spectrogram(audio: np.ndarray, sr: int, hop_samples: int):
    noverlap = NFFT - hop_samples
    return spectrogram(audio, sr, nperseg=NFFT, noverlap=noverlap, mode="magnitude")


def _detect_tone_from_spectrogram(freqs: np.ndarray, Sxx: np.ndarray) -> float:
    """Pick the bin that looks most like a keyed CW signal, not just the
    strongest one. Score = mean power * envelope std: a keyed tone is both
    strong AND switches on/off (high per-bin envelope variance), whereas a
    steady drifting carrier is strong but flat, and broadband noise varies
    but is weak everywhere. Plain strongest-mean-power selection could lock
    onto a QRM tone or carrier louder than the (possibly QSB-faded) desired
    signal - which, at training time, silently centered the feature band on
    the interferer while the label still described the real signal."""
    band_mask = (freqs >= TONE_FMIN_HZ) & (freqs <= TONE_FMAX_HZ)
    if not band_mask.any():
        return float(freqs[len(freqs) // 2])
    band = Sxx[band_mask]
    score = band.mean(axis=1) * band.std(axis=1)
    if not np.isfinite(score).any() or score.max() <= 0:
        score = band.mean(axis=1)  # degenerate clip (e.g. silence) - fall back
    return float(freqs[band_mask][np.argmax(score)])


def detect_tone_freq(audio: np.ndarray, sr: int, hop_samples: int = DEFAULT_HOP_SAMPLES) -> float:
    freqs, _times, Sxx = _compute_spectrogram(audio, sr, hop_samples)
    return _detect_tone_from_spectrogram(freqs, Sxx)


class ToneTracker:
    """Persistent CW tone estimate across analysis windows - the ONE shared
    policy for live (StreamDecoder) and offline (decode_stream) decoding, so
    batch ARRL evaluation exercises the same front-end behavior as the radio.

    The station being worked doesn't move: in-range detections update an EMA
    (tracks slow drift), a single outlier - e.g. a QRM burst dominating one
    window - is rejected, and a persistent change (retuned to a new station)
    takes over after `outlier_windows` consecutive windows agree on it."""

    def __init__(self, jump_hz: float = 60.0, ema_alpha: float = 0.3, outlier_windows: int = 3):
        self.jump_hz = jump_hz
        self.ema_alpha = ema_alpha
        self.outlier_windows = outlier_windows
        self.value: float | None = None
        self._outliers = 0

    def update(self, detected: float) -> float:
        if self.value is None:
            self.value = detected
        elif abs(detected - self.value) <= self.jump_hz:
            self.value += self.ema_alpha * (detected - self.value)
            self._outliers = 0
        else:
            self._outliers += 1
            if self._outliers >= self.outlier_windows:
                self.value = detected  # a real retune, not a blip
                self._outliers = 0
        return self.value


def cw_activity_db(audio: np.ndarray, sr: int, hop_samples: int = DEFAULT_HOP_SAMPLES,
                    tone_freq: float | None = None) -> float:
    """How CW-like this audio is, in dB: the p85/p15 envelope ratio of the
    (detected or given) tone bin. Keyed CW switches that bin on and off, so
    the ratio is large (typically well over 6 dB even at poor broadband SNR -
    the tone is narrowband, so per-bin SNR is ~19 dB better than the 2.5 kHz
    channel figure). Silence, broadband static, and steady carriers score low.
    Used as a squelch gate so dead air and tuning noise between transmissions
    don't get 'decoded' into confident-looking garbage characters."""
    freqs, _times, Sxx = _compute_spectrogram(audio, sr, hop_samples)
    if tone_freq is None:
        tone_freq = _detect_tone_from_spectrogram(freqs, Sxx)
    peak_bin = int(np.argmin(np.abs(freqs - tone_freq)))
    env = Sxx[peak_bin]
    if env.size < 8:
        return 0.0
    hi, lo = np.percentile(env, [85, 15])
    return float(10.0 * np.log10((hi + 1e-12) / (lo + 1e-12)))


def extract_features(audio: np.ndarray, sr: int, n_freq_bins: int = N_FREQ_BINS,
                      hop_samples: int = DEFAULT_HOP_SAMPLES,
                      tone_freq: float | None = None) -> np.ndarray:
    """Returns a (T, n_freq_bins) log-magnitude feature matrix, normalized per-clip.

    tone_freq: pass the known tone frequency (e.g. from a synthesis manifest)
    to skip detection entirely - eliminates detector mistakes as a source of
    training-label noise. None = auto-detect from the audio."""
    freqs, _times, Sxx = _compute_spectrogram(audio, sr, hop_samples)
    if tone_freq is None:
        tone_freq = _detect_tone_from_spectrogram(freqs, Sxx)
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
