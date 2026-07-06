"""Real-time CW (Morse) decoder: capture audio from an input device (e.g. a
USB sound card fed by a radio's audio-out), run it through the trained
acoustic model, and print decoded text as it comes in.

Uses the same overlap-trim-stitch scheme as model.decoder.decode_stream
(see there for the full rationale): independently decoding non-overlapping
fixed windows corrupts whatever character straddles each window edge, since
the model has no context on the other side of the cut - real Morse doesn't
arrive in clean chunk-sized pieces. Overlapping windows are decoded instead,
keeping only each window's "core" (non-edge) region and stitching those
together - StreamDecoder below is the live, feed-incrementally version of
decode_stream, since a live mic feed doesn't have all its audio upfront.

The model was trained exclusively on audio at MODEL_SAMPLE_RATE (8kHz) -
captured audio is resampled to match before feature extraction, regardless
of the input device's native rate.

Usage:
  python inference/realtime_decode.py --list-devices
  python inference/realtime_decode.py --checkpoint <path to decoder_epochNNN.pt> --device "USB Audio"
"""
import argparse
import queue
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.decoder import decode_window_core, load_checkpoint_model
from model.features import (SAMPLE_RATE, SignalTracker, ToneTracker, detect_tone_freq,
                             estimate_signal_quality, resample_to_model_rate)
from model.vocab import Vocab
from paths import DATA_ROOT

MODEL_SAMPLE_RATE = SAMPLE_RATE


def load_model(checkpoint_path: str, vocab_path: str, device: str):
    """Rebuilds the model from the checkpoint's recorded model_config and
    vocab_chars (legacy checkpoints reconstruct as the legacy architecture
    automatically; vocab_path is only a fallback for checkpoints without
    recorded vocab_chars)."""
    model, vocab, _ckpt = load_checkpoint_model(checkpoint_path, device, vocab_path=vocab_path)
    return model, vocab


def make_fcc_rescorer(bonus: float = 2.0):
    """Final-beam rescorer for CTC beam search: boosts candidate decodes whose
    US-pattern callsign tokens are actual active FCC licenses. A character
    n-gram LM is weakest exactly where accuracy matters most - callsigns are
    near-random strings the LM actively penalizes - so this puts the FCC
    index (already used for post-hoc verification flags in the TUI) into the
    decoding loop itself, tipping close beams toward real licensed calls.
    Returns None (with a warning) if the FCC index hasn't been built."""
    from inference.fcc_uls import is_us_pattern, load_active_callsigns
    active = load_active_callsigns()
    if not active:
        print("warning: FCC index not found - run `python inference/fcc_uls.py --download` first; "
              "continuing without callsign rescoring", file=sys.stderr)
        return None

    def rescore(text: str) -> float:
        return sum(bonus for tok in text.split() if is_us_pattern(tok) and tok in active)

    return rescore


def parse_device(device: str | None):
    """sounddevice treats a numeric index passed as a string as a name
    substring match, not an index - convert if it looks numeric."""
    if device is not None and device.lstrip("-").isdigit():
        return int(device)
    return device


class StreamDecoder:
    """Live, feed-incrementally counterpart to model.decoder.decode_stream -
    same overlapping-window/core-trim scheme, but audio arrives piecemeal
    from a live mic feed instead of being available all at once. Call
    .feed(chunk) as audio arrives (returns newly decoded text, possibly
    empty if not enough has accumulated yet for another window), and
    .flush() once the stream actually ends to decode whatever's left in the
    buffer (its core extends to the true end, since there's no next window
    to cover the tail).

    Pass lm=CharNgramLM.load(...) to enable CTC beam search rescored by the
    ham-domain character LM instead of greedy decoding, and/or
    final_rescore=make_fcc_rescorer() to tip close beams toward decodes whose
    callsigns are real FCC licenses.

    The CW tone frequency is tracked ACROSS windows via the shared
    features.ToneTracker (same policy as offline decode_stream): EMA
    smoothing, one-off outlier rejection, retune acceptance after a few
    consistent windows.

    squelch_db gates each window on CW-likeness before decoding (see
    features.cw_activity_db) so dead air, static, and tuning noise between
    transmissions don't become hallucinated characters. Live radio audio is
    mostly not-CW, hence the nonzero default here; offline evaluation of
    known-CW recordings defaults it off.

    .signal_report exposes a live-measured RST string (e.g. "579") from the
    audio itself - see features.SignalTracker/estimate_signal_quality for
    what it measures and its calibration caveats. None until the first
    window is processed."""

    def __init__(self, model, vocab: Vocab, torch_device: str, sr: int,
                 window_seconds: float = 8.0, stride_seconds: float = 4.0,
                 lm=None, lm_weight: float = 0.3, beam_width: int = 20, top_k: int = 15,
                 final_rescore=None, squelch_db: float = 6.0,
                 length_bonus: float = 0.0, repeat_penalty: float = 0.0):
        self.model = model
        self.vocab = vocab
        self.torch_device = torch_device
        self.sr = sr
        self.window_samples = int(window_seconds * sr)
        self.stride_samples = int(stride_seconds * sr)
        self.guard_seconds = (window_seconds - stride_seconds) / 2
        self.stride_seconds = stride_seconds
        self.lm = lm
        self.lm_weight = lm_weight
        self.beam_width = beam_width
        self.top_k = top_k
        self.final_rescore = final_rescore
        self.squelch_db = squelch_db
        self.length_bonus = length_bonus
        self.repeat_penalty = repeat_penalty
        self.buf = np.zeros(0, dtype=np.float32)
        self.stream_pos_samples = 0
        self.is_first = True
        self._tracker = ToneTracker()
        self._signal = SignalTracker()

    @property
    def tone_freq(self):
        return self._tracker.value

    @property
    def signal_report(self):
        return self._signal.rst

    def _decode(self, window_audio, window_abs_start, core_start, core_end, tone) -> str:
        return decode_window_core(window_audio, window_abs_start, core_start, core_end,
                                   self.model, self.vocab, self.torch_device, self.sr,
                                   lm=self.lm, lm_weight=self.lm_weight,
                                   beam_width=self.beam_width, top_k=self.top_k,
                                   tone_freq=tone, final_rescore=self.final_rescore,
                                   squelch_db=self.squelch_db,
                                   length_bonus=self.length_bonus,
                                   repeat_penalty=self.repeat_penalty)

    def feed(self, chunk: np.ndarray) -> str:
        self.buf = np.concatenate([self.buf, chunk])
        pieces = []
        while len(self.buf) >= self.window_samples:
            window_audio = self.buf[: self.window_samples]
            window_abs_start = self.stream_pos_samples / self.sr
            core_start = 0.0 if self.is_first else window_abs_start + self.guard_seconds
            core_end = window_abs_start + self.guard_seconds + self.stride_seconds
            tone = self._tracker.update(detect_tone_freq(window_audio, self.sr,
                                                          hop_samples=self.model.hop_samples))
            activity_db, snr_db = estimate_signal_quality(window_audio, self.sr,
                                                            hop_samples=self.model.hop_samples,
                                                            tone_freq=tone)
            self._signal.update(activity_db, snr_db)
            pieces.append(self._decode(window_audio, window_abs_start, core_start, core_end, tone))
            self.is_first = False
            self.buf = self.buf[self.stride_samples:]
            self.stream_pos_samples += self.stride_samples
        return "".join(pieces)

    def flush(self) -> str:
        if len(self.buf) == 0:
            return ""
        window_abs_start = self.stream_pos_samples / self.sr
        core_start = 0.0 if self.is_first else window_abs_start + self.guard_seconds
        core_end = window_abs_start + len(self.buf) / self.sr
        # use the held tone estimate - the leftover buffer may be too short
        # (or too noise-dominated) for a reliable fresh detection
        if len(self.buf) >= 8:
            activity_db, snr_db = estimate_signal_quality(self.buf, self.sr,
                                                            hop_samples=self.model.hop_samples,
                                                            tone_freq=self._tracker.value)
            self._signal.update(activity_db, snr_db)
        text = self._decode(self.buf, window_abs_start, core_start, core_end, self._tracker.value)
        self.buf = np.zeros(0, dtype=np.float32)
        return text


def iter_decoded_stream(device, decoder: StreamDecoder):
    """Captures audio from `device` and yields newly decoded text as it
    becomes available - runs until the caller stops iterating (e.g. via
    `break`) or the input stream raises. Caller should call decoder.flush()
    afterward to get any text left in the buffer."""
    device_info = sd.query_devices(device, "input")
    native_sr = int(device_info["default_samplerate"])

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio status: {status}]", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    with sd.InputStream(device=device, channels=1, samplerate=native_sr,
                         dtype="float32", callback=callback):
        while True:
            chunk = resample_to_model_rate(audio_q.get(), native_sr)
            text = decoder.feed(chunk)
            if text:
                yield text


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="path to a decoder_epochNNN.pt checkpoint")
    parser.add_argument("--vocab", default=str(DATA_ROOT / "manifests" / "vocab.txt"))
    parser.add_argument("--device", default=None, help="input device name or index (see --list-devices)")
    parser.add_argument("--window-seconds", type=float, default=8.0,
                         help="decode window length - match the clip length the model was trained on")
    parser.add_argument("--stride-seconds", type=float, default=4.0,
                         help="how far the window advances each step - window_seconds/2 gives clean "
                              "non-overlapping core regions; smaller values add latency cost for no benefit, "
                              "larger values widen the unstitched edge gap")
    parser.add_argument("--torch-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--lm", default=None, metavar="PATH",
                         help="path to ham_char_lm.json to enable beam search + LM decoding")
    parser.add_argument("--lm-weight", type=float, default=0.3)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--fcc-rescore", action="store_true",
                         help="boost beam-search candidates whose callsigns are active FCC licenses "
                              "(requires the FCC index - see inference/fcc_uls.py - and --lm)")
    parser.add_argument("--squelch-db", type=float, default=6.0,
                         help="skip decoding windows whose tone-bin keying activity is below this "
                              "(dB, p85/p15 envelope ratio) - suppresses hallucinated characters from "
                              "dead air/static between transmissions. 0 = decode everything")
    parser.add_argument("--length-bonus", type=float, default=0.0,
                         help="beam-search score bonus per output character (sweep before trusting)")
    parser.add_argument("--repeat-penalty", type=float, default=0.0,
                         help="beam-search penalty per repeated-run character - targets the "
                              "runaway-repetition failure mode (sweep before trusting)")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required (unless using --list-devices)")

    model, vocab = load_model(args.checkpoint, args.vocab, args.torch_device)
    lm = None
    if args.lm:
        from lm.ngram_lm import CharNgramLM
        lm = CharNgramLM.load(args.lm)
    final_rescore = make_fcc_rescorer() if args.fcc_rescore and lm is not None else None
    if args.fcc_rescore and lm is None:
        print("warning: --fcc-rescore only applies with --lm (beam search); ignoring", file=sys.stderr)
    device = parse_device(args.device)
    device_info = sd.query_devices(device, "input")
    decoder = StreamDecoder(model, vocab, args.torch_device, MODEL_SAMPLE_RATE,
                             window_seconds=args.window_seconds, stride_seconds=args.stride_seconds,
                             lm=lm, lm_weight=args.lm_weight, beam_width=args.beam_width,
                             final_rescore=final_rescore, squelch_db=args.squelch_db,
                             length_bonus=args.length_bonus, repeat_penalty=args.repeat_penalty)

    print(f"Listening on {device_info['name']!r} at {int(device_info['default_samplerate'])} Hz, "
          f"{args.window_seconds}s windows / {args.stride_seconds}s stride (Ctrl+C to stop) ...")

    try:
        for text in iter_decoded_stream(device, decoder):
            if text:
                print(text, end="", flush=True)
    except KeyboardInterrupt:
        print(decoder.flush(), end="", flush=True)
        print("\nstopped.")


if __name__ == "__main__":
    main()
