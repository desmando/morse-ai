# ONNX export + quantization (TODO)

Once `model/train.py` produces a real checkpoint (trained on the RTX 3080 box):

1. `torch.onnx.export()` the trained `CWDecoder` to `.onnx`.
2. Quantize on an **x64** machine (Qualcomm's own QNN docs recommend this -
   quantize on x64, then run the quantized model on ARM64 against the NPU).
   The RTX 3080 box is x64, so it can double as the quantization machine.
3. Copy the quantized `.onnx` file back to this laptop for `inference/`.

Needs: `onnxruntime` (export/quantization tooling) installed on the x64
machine doing the quantizing - separate from `onnxruntime-qnn` which goes on
this ARM64 laptop for actual NPU inference (see `inference/README.md`).
