"""Sanity-check an ONNX export against the original PyTorch checkpoint:
confirms decoded text matches and reports streaming throughput, against real
continuous audio.

(Originally also benchmarked QNN HTP/NPU inference - dropped after measuring
it was 2-2.5x *slower* than plain ONNX Runtime CPUExecutionProvider on this
model size, with a lot more complexity (quantization, fixed shapes, ~1-2min
graph compile on first load) for no benefit. CPU EP alone gives 150-200x
real-time throughput, far more than the 1x actually needed.)

Usage:
  python benchmark_export.py --checkpoint <path> --onnx <path> --audio <real .mp3/.wav>
"""
import argparse
import sys
import time
from pathlib import Path

import onnxruntime as ort
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.decoder import decode_stream, load_checkpoint_model


class ONNXModel:
    """Wraps an ONNX Runtime session with the attributes decode_window_core
    needs (hop_samples, frame_seconds()) - decode_stream doesn't care
    whether the model underneath is PyTorch or ONNX, but it does read these
    off whatever object it's given, matching CWDecoder's own interface.
    Values come from the PyTorch checkpoint's recorded model_config (the
    ONNX graph itself doesn't carry them) - must be the same checkpoint the
    ONNX file was exported from, or these silently describe the wrong model."""

    def __init__(self, onnx_path: str, hop_samples: int, time_stride: int):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.hop_samples = hop_samples
        self.time_stride = time_stride

    def frame_seconds(self, sr: int) -> float:
        return self.hop_samples * self.time_stride / sr

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.session.run(None, {self.input_name: x.numpy()})[0]
        return torch.from_numpy(out)


def run_benchmark(name: str, model, audio, sr, vocab, window_seconds=8.0, stride_seconds=4.0):
    t0 = time.time()
    text = decode_stream(audio, sr, model, vocab, "cpu",
                          window_seconds=window_seconds, stride_seconds=stride_seconds)
    wall = time.time() - t0
    audio_seconds = len(audio) / sr
    rtf = audio_seconds / wall if wall > 0 else float("inf")
    print(f"\n=== {name} ===")
    print(f"  audio: {audio_seconds:.1f}s, wall time: {wall:.2f}s, real-time factor: {rtf:.1f}x")
    print(f"  decoded ({len(text)} chars): {text[:200]!r}{'...' if len(text) > 200 else ''}")
    return text, rtf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--stride-seconds", type=float, default=4.0)
    args = parser.parse_args()

    audio, sr = sf.read(args.audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    print(f"benchmarking against {args.audio} ({len(audio)/sr:.1f}s of real audio)")

    pt_model, vocab, _ckpt = load_checkpoint_model(args.checkpoint, "cpu")
    pt_text, _ = run_benchmark("PyTorch / CPU", pt_model, audio, sr, vocab,
                                args.window_seconds, args.stride_seconds)

    onnx_model = ONNXModel(args.onnx, pt_model.hop_samples, pt_model.time_stride)
    onnx_text, _ = run_benchmark("ONNX / CPU EP", onnx_model, audio, sr, vocab,
                                  args.window_seconds, args.stride_seconds)

    print("\nMATCH" if pt_text == onnx_text else "\nMISMATCH - investigate before trusting the export")


if __name__ == "__main__":
    main()
