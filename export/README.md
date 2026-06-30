# ONNX export

`export_onnx.py` exports a trained checkpoint to a dynamic-shape ONNX model
for CPU inference:

```
python export_onnx.py --checkpoint <path to decoder_epochNNN.pt>
```

`benchmark_export.py` sanity-checks the export (decoded text must match the
original PyTorch checkpoint exactly) and reports streaming throughput against
real audio.

**NPU/QNN HTP inference was tried and dropped.** Measured on the laptop's
Snapdragon NPU against a real 7.8-minute ARRL recording: plain ONNX Runtime
`CPUExecutionProvider` (202.9x real-time, fp32) beat QNN HTP/NPU (79.6x
real-time, required int8 quantization + fixed input shapes + a ~1-2 min
graph compile on first load). The model here (32-channel CNN + 2-layer
256->128 BiLSTM) is small enough that NPU dispatch overhead outweighs its
compute advantage, and LSTMs' sequential dependencies don't parallelize onto
a tensor accelerator the way conv-heavy models do. CPU alone gives 150-200x
real-time throughput, far more than the 1x actually needed for live
decoding, so the extra complexity wasn't worth it. Plain `onnxruntime` (no
`-qnn` package) is all `inference/` needs.
