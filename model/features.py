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


def estimate_signal_quality(audio: np.ndarray, sr: int, hop_samples: int = DEFAULT_HOP_SAMPLES,
                             tone_freq: float | None = None) -> tuple[float, float]:
    """Raw (activity_db, snr_db) for one window - the measurements behind
    bucket_readability()/bucket_strength() and SignalTracker, for generating
    a live signal report. One spectrogram pass (deliberately not sharing it
    with cw_activity_db's separate call at the squelch gate - the redundant
    FFT costs microseconds against the seconds-per-window decode budget, and
    keeping each caller self-contained is simpler than threading a shared
    Sxx through decode_window_core's existing squelch path).

    activity_db: same on/off keying contrast as cw_activity_db - drives R.

    snr_db: tone-bin near-peak power (90th percentile - keying is on/off, so
    the mean underestimates the "on" level) vs. the median power in nearby
    non-tone bins (a simple noise-floor estimate, skipping +-50Hz around the
    tone itself). This is an audio-domain proxy, NOT a calibrated S-meter
    reading - the software has no access to the receiver's RF gain/AGC
    state, only the digitized audio after it. It's closer to what an
    operator estimates by ear when not looking at the meter. Calibrated
    against dataprep/augment_hf_channel.py's controlled additive-noise sweep
    (-6..+30 dB target SNR maps to ~9..27 dB measured here, monotonically),
    not against real receiver behavior - treat bucket_strength()'s S-units
    as a reasoned first pass to refine with real operating experience, not
    a finished calibration.
    """
    freqs, _times, Sxx = _compute_spectrogram(audio, sr, hop_samples)
    if tone_freq is None:
        tone_freq = _detect_tone_from_spectrogram(freqs, Sxx)
    peak_bin = int(np.argmin(np.abs(freqs - tone_freq)))
    env = Sxx[peak_bin]
    if env.size < 8:
        return 0.0, -99.0
    hi, lo = np.percentile(env, [85, 15])
    activity_db = float(10.0 * np.log10((hi + 1e-12) / (lo + 1e-12)))

    bin_hz = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    guard = max(1, int(round(50.0 / bin_hz)))
    noise_bins = [i for i in range(len(freqs)) if guard < abs(i - peak_bin) < guard * 6]
    noise_power = float(np.median(Sxx[noise_bins])) if noise_bins else float(np.median(Sxx))
    signal_power = float(np.percentile(env, 90))
    snr_db = float(10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))
    return activity_db, snr_db


def bucket_readability(activity_db: float) -> int:
    """R (1-5, standard RST convention) from on/off keying contrast.
    Thresholds calibrated against the controlled SNR sweep in
    estimate_signal_quality's docstring, not real-operator ground truth."""
    if activity_db >= 20.0:
        return 5
    if activity_db >= 15.0:
        return 4
    if activity_db >= 11.0:
        return 3
    if activity_db >= 7.0:
        return 2
    return 1


def bucket_strength(snr_db: float) -> int:
    """S (1-9, standard RST convention) from tone-bin SNR. ~3 dB/S-unit over
    the calibrated -6..+22 dB range - see estimate_signal_quality's
    docstring for the important caveat: this is an audio-SNR proxy, not a
    calibrated receiver S-meter reading."""
    return int(np.clip(round((snr_db + 6.0) / 3.0), 1, 9))


class TransmissionSegmenter:
    """Detects transmission ("over") boundaries in a live audio stream by
    watching for a keying-silence gap substantially longer than the
    CURRENT estimated inter-word gap - the natural pause between complete
    transmissions (a repeated CQ, or the other station's reply), as
    distinct from the shorter gaps that occur WITHIN one (inter-character
    ~3 dots, inter-word ~7 dots). This is what lets a live display show
    "CQ TEST DE W7JET W7JET K" / "CQ TEST DE W7JET W7JET K" as separate
    lines rather than one endless scrolling run-on.

    Two independent concerns, deliberately not conflated:

    1. Activity/silence detection (update()) - reuses cw_activity_db's
       already-validated p85/p15 envelope-ratio squelch logic per analysis
       chunk (proven against known SNR levels and real noise/silence/
       carrier cases elsewhere in this module) rather than a frame-by-frame
       state machine. An earlier from-scratch attempt at the latter - a
       floor-tracking threshold with per-frame debounce - looked reasonable
       but broke on real pure noise: hop_samples << NFFT means adjacent
       "frames" share ~78% of their underlying samples, so they're highly
       correlated, and a single random elevated reading legitimately
       persists across several "consecutive" frames purely from that
       overlap, defeating debounce logic that assumes rough independence.
       Working at the whole-chunk level with an already-proven metric
       sidesteps that failure mode entirely rather than patching around it.

    2. WPM/dot-length estimation (set_wpm) - the inter-word gap is WPM-
       dependent (7 dot-lengths, and a dot at 15 WPM is 2x longer than at
       30 WPM), so a fixed silence threshold can't work across speeds. This
       does NOT try to infer WPM from raw envelope pulse timing (fragile -
       real elements are 30-250ms, comparable to or shorter than achievable
       envelope time resolution). Instead the caller (StreamDecoder, which
       already knows how many characters the neural decoder produced over
       how many seconds of audio) supplies a WPM estimate directly - a far
       more reliable source of truth than reconstructing it from noise
       statistics. A sane default is used until the first real estimate
       arrives.

    Boundary threshold: max(min_gap_s, gap_multiple * 7 * dot_s_estimate).
    The floor (min_gap_s) guards very fast/erratic sending where 7x a tiny
    dot length could otherwise be under a second; the multiple guards very
    slow sending where a fixed floor alone would fire mid-transmission on
    an ordinary inter-word gap.
    """

    def __init__(self, sr: int = SAMPLE_RATE, hop_samples: int = DEFAULT_HOP_SAMPLES,
                 gap_multiple: float = 2.5, min_gap_s: float = 1.2,
                 analysis_chunk_s: float = 0.3, activity_threshold_db: float = 9.0,
                 default_wpm: float = 20.0):
        # activity_threshold_db intentionally higher than decode_window_core's
        # squelch_db default (6.0): measured against real recorded room
        # noise, a plain 6dB threshold reads "active" on a majority of
        # pieces (median 6.4dB in one real sample) - not a rare fluke, a
        # genuinely borderline noise floor. Here, a too-low threshold just
        # delays boundary detection by resetting the silence clock on
        # spurious blips; in the live squelch gate, being too tolerant
        # means decoding noise into hallucinated text - a worse mistake -
        # so the two thresholds are allowed to differ.
        self.sr = sr
        self.hop_samples = hop_samples
        self.gap_multiple = gap_multiple
        self.min_gap_s = min_gap_s
        self.analysis_chunk_samples = max(1, int(analysis_chunk_s * sr))
        self.activity_threshold_db = activity_threshold_db
        self._dot_s_estimate = 1.2 / default_wpm
        self._buf = np.zeros(0, dtype=np.float32)
        self._silent = True
        self._silence_accum_s = 0.0
        self._boundary_fired_this_run = False

    @property
    def silence_s(self) -> float:
        return self._silence_accum_s

    @property
    def dot_s_estimate(self) -> float:
        return self._dot_s_estimate

    @property
    def boundary_threshold_s(self) -> float:
        return max(self.min_gap_s, self.gap_multiple * 7.0 * self._dot_s_estimate)

    def set_wpm(self, wpm: float):
        """Caller-supplied WPM estimate (e.g. from decoded chars / elapsed
        seconds) - see class docstring for why this isn't inferred from
        raw audio here. Ignores non-positive/absurd values defensively
        rather than letting a transient bad estimate corrupt the threshold."""
        if 1.0 <= wpm <= 100.0:
            self._dot_s_estimate = 1.2 / wpm

    def update(self, chunk: np.ndarray, tone_freq: float | None) -> bool:
        """Feed newly-arrived audio. Returns True exactly once, the moment
        accumulated silence first crosses the current boundary threshold -
        the signal to flush whatever's been decoded so far as one complete
        line. Won't fire again for the same silence stretch; resets once
        keying resumes and then stops again. tone_freq=None (no estimate
        yet, e.g. before the first decode window completes) defers
        analysis rather than guessing."""
        if tone_freq is None:
            return False
        self._buf = np.concatenate([self._buf, chunk])
        fired = False
        while len(self._buf) >= self.analysis_chunk_samples:
            piece = self._buf[: self.analysis_chunk_samples]
            self._buf = self._buf[self.analysis_chunk_samples:]
            fired = self._process_piece(piece, tone_freq) or fired
        return fired

    def _process_piece(self, piece: np.ndarray, tone_freq: float) -> bool:
        activity_db = cw_activity_db(piece, self.sr, hop_samples=self.hop_samples,
                                      tone_freq=tone_freq)
        piece_s = len(piece) / self.sr
        active = activity_db >= self.activity_threshold_db

        if active:
            self._silent = False
            self._silence_accum_s = 0.0
            self._boundary_fired_this_run = False  # keying resumed - allow the next silence to fire again
            return False

        self._silent = True
        self._silence_accum_s += piece_s
        if not self._boundary_fired_this_run and self._silence_accum_s >= self.boundary_threshold_s:
            self._boundary_fired_this_run = True
            return True
        return False


class SignalTracker:
    """Smoothed CW signal-quality estimate across windows, for reporting a
    live RST. Plain EMA - unlike ToneTracker's outlier rejection (a real
    station's frequency shouldn't jump around, so a sudden reading is
    probably an artifact), real QSB/interference SHOULD move a signal report
    responsively; there's no "outlier" to reject here."""

    def __init__(self, ema_alpha: float = 0.4):
        self.ema_alpha = ema_alpha
        self.activity_db: float | None = None
        self.snr_db: float | None = None

    def update(self, activity_db: float, snr_db: float) -> tuple[int, int]:
        if self.activity_db is None:
            self.activity_db, self.snr_db = activity_db, snr_db
        else:
            self.activity_db += self.ema_alpha * (activity_db - self.activity_db)
            self.snr_db += self.ema_alpha * (snr_db - self.snr_db)
        return bucket_readability(self.activity_db), bucket_strength(self.snr_db)

    @property
    def rst(self) -> str | None:
        """3-digit RST ('R', 'S', then a fixed '9' for Tone - no chirp/
        click/hum analysis exists to assess tone purity), or None before the
        first measurement."""
        if self.activity_db is None:
            return None
        r, s = bucket_readability(self.activity_db), bucket_strength(self.snr_db)
        return f"{r}{s}9"


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
