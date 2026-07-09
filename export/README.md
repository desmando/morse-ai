# ONNX export

Needs two packages not in `requirements.txt` (export/benchmark-only, not
needed for `inference/` or training): `pip install onnx onnxruntime`.

`export_onnx.py` exports a trained checkpoint to a dynamic-shape ONNX model
for CPU inference. It rebuilds the model from the checkpoint's own recorded
`model_config` (via `load_checkpoint_model`) rather than assuming default
architecture params, so it stays correct across checkpoints trained with
different configs (feature hop, CNN time-stride, layer sizes, ...) -
constructing `CWDecoder` with only `vocab_size` would silently use whatever
this script's defaults happen to be, which only matches a given checkpoint's
real architecture by coincidence:

```
python export_onnx.py --checkpoint <path to decoder_epochNNN.pt>
```

`benchmark_export.py` sanity-checks the export (decoded text must match the
original PyTorch checkpoint exactly) and reports streaming throughput against
real audio. Its `ONNXModel` wrapper carries `hop_samples`/`frame_seconds()`
read from the PyTorch checkpoint's config - `decode_stream` needs these off
whatever model object it's given (ONNX or PyTorch), and the ONNX graph alone
doesn't carry them.

**NPU/QNN HTP inference was tried and dropped.** Measured on a laptop's
Snapdragon NPU against a real 7.8-minute ARRL recording: plain ONNX Runtime
`CPUExecutionProvider` (202.9x real-time, fp32) beat QNN HTP/NPU (79.6x
real-time, required int8 quantization + fixed input shapes + a ~1-2 min
graph compile on first load). The model here (32-channel CNN + 2-layer
256->128 BiLSTM) is small enough that NPU dispatch overhead outweighs its
compute advantage, and LSTMs' sequential dependencies don't parallelize onto
a tensor accelerator the way conv-heavy models do. Re-verified on a
different machine (RTX 3060 desktop, CPU-only inference) against a real
13-minute ARRL recording: ONNX CPU EP gave 125x real-time vs. 22x for plain
PyTorch CPU - a different absolute number (different hardware, and the v4
architecture's packed-sequence LSTM path adds some PyTorch-side overhead
this export bypasses), but the same conclusion: CPU alone is far more than
the 1x actually needed for live decoding, so chasing NPU/GPU acceleration
for this specific model isn't worth the complexity. Plain `onnxruntime` (no
`-qnn` package) is all `inference/` needs. (A Jetson-class real GPU is a
different story if a dedicated portable device is wanted for other reasons
- see the project discussion history - since it runs the exact same
PyTorch/CUDA code as training with no NPU-style quantization headaches, not
because raw speed is the problem here.)
