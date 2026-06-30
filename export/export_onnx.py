"""Export a trained checkpoint to ONNX for CPU inference via onnxruntime.

(We tried NPU/QNN HTP inference and dropped it - on this model size, plain
ONNX Runtime CPUExecutionProvider beat the NPU by 2-2.5x, and CPU already
gives 150-200x real-time throughput, far more than the 1x actually needed
for live decoding. Not worth the quantization/fixed-shape complexity NPU
inference requires. See benchmark_export.py.)

Usage:
  python export_onnx.py --checkpoint <path to decoder_epochNNN.pt>
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.decoder import CWDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None, help="default: <checkpoint stem>.onnx next to the checkpoint")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab_chars = ckpt["vocab_chars"]
    vocab_size = len(vocab_chars) + 1  # + blank

    model = CWDecoder(vocab_size=vocab_size)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_path = Path(args.out) if args.out else ckpt_path.with_suffix(".onnx")
    dummy = torch.zeros(1, 200, 32, dtype=torch.float32)  # (batch, T, n_freq_bins) - T is dynamic

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["features"], output_names=["log_probs"],
        dynamic_axes={"features": {0: "batch", 1: "time"}, "log_probs": {0: "batch", 1: "time"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Exported to {out_path} (vocab_size={vocab_size}, epoch={ckpt.get('epoch')})")


if __name__ == "__main__":
    main()
