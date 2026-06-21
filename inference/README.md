# Real-time inference app (TODO)

Live pipeline once a quantized ONNX model exists (`export/`):

1. Capture audio from the HF radio (e.g. `sounddevice`, reading from a USB
   sound card / line-in fed by the radio's audio-out).
2. Run it through `model/features.py`'s feature extraction (same front end
   used in training - tone auto-detect + spectrogram band).
3. Run the quantized model via `onnxruntime` + `onnxruntime-qnn`
   (`onnxruntime-qnn>=2.0.0`, prebuilt Windows ARM64 wheel) targeting the
   Hexagon NPU (HTP backend).
4. Pipe raw decoded characters through `lm/` for contextual correction.
5. Stream the corrected text to the screen.

Needs both `onnxruntime` and the `onnxruntime-qnn` plugin package installed
on this laptop - not needed until there's an actual model to run.
