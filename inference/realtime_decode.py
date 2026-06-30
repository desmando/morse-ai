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
from math import gcd
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.decoder import CWDecoder, decode_window_core
from model.vocab import Vocab
from paths import DATA_ROOT

MODEL_SAMPLE_RATE = 8000


def load_model(checkpoint_path: str, vocab_path: str, device: str):
    vocab = Vocab.from_file(vocab_path)
    model = CWDecoder(vocab_size=len(vocab))
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, vocab


def resample_to_model_rate(chunk: np.ndarray, native_sr: int) -> np.ndarray:
    if native_sr == MODEL_SAMPLE_RATE:
        return chunk
    g = gcd(native_sr, MODEL_SAMPLE_RATE)
    return resample_poly(chunk, MODEL_SAMPLE_RATE // g, native_sr // g)


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
    to cover the tail)."""

    def __init__(self, model, vocab: Vocab, torch_device: str, sr: int,
                 window_seconds: float = 8.0, stride_seconds: float = 4.0):
        self.model = model
        self.vocab = vocab
        self.torch_device = torch_device
        self.sr = sr
        self.window_samples = int(window_seconds * sr)
        self.stride_samples = int(stride_seconds * sr)
        self.guard_seconds = (window_seconds - stride_seconds) / 2
        self.stride_seconds = stride_seconds
        self.buf = np.zeros(0, dtype=np.float32)
        self.stream_pos_samples = 0
        self.is_first = True

    def feed(self, chunk: np.ndarray) -> str:
        self.buf = np.concatenate([self.buf, chunk])
        pieces = []
        while len(self.buf) >= self.window_samples:
            window_audio = self.buf[: self.window_samples]
            window_abs_start = self.stream_pos_samples / self.sr
            core_start = 0.0 if self.is_first else window_abs_start + self.guard_seconds
            core_end = window_abs_start + self.guard_seconds + self.stride_seconds
            pieces.append(decode_window_core(window_audio, window_abs_start, core_start, core_end,
                                              self.model, self.vocab, self.torch_device, self.sr))
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
        text = decode_window_core(self.buf, window_abs_start, core_start, core_end,
                                   self.model, self.vocab, self.torch_device, self.sr)
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
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required (unless using --list-devices)")

    model, vocab = load_model(args.checkpoint, args.vocab, args.torch_device)
    device = parse_device(args.device)
    device_info = sd.query_devices(device, "input")
    decoder = StreamDecoder(model, vocab, args.torch_device, MODEL_SAMPLE_RATE,
                             window_seconds=args.window_seconds, stride_seconds=args.stride_seconds)

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
