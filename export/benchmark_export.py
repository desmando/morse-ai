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
from model.decoder import CWDecoder, decode_stream
from model.vocab import Vocab


class PyTorchModel:
    def __init__(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.vocab = Vocab(ckpt["vocab_chars"])
        self.model = CWDecoder(vocab_size=len(self.vocab))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)


class ONNXModel:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

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

    pt = PyTorchModel(args.checkpoint)
    pt_text, _ = run_benchmark("PyTorch / CPU", pt.model, audio, sr, pt.vocab,
                                args.window_seconds, args.stride_seconds)

    onnx_model = ONNXModel(args.onnx)
    onnx_text, _ = run_benchmark("ONNX / CPU EP", onnx_model, audio, sr, pt.vocab,
                                  args.window_seconds, args.stride_seconds)

    print("\nMATCH" if pt_text == onnx_text else "\nMISMATCH - investigate before trusting the export")


if __name__ == "__main__":
    main()
