# morse-ai

ML pipeline to decode CW (Morse) audio off an HF radio into readable text, with
contextual correction, deployed for real-time inference on a Windows ARM64 NPU laptop.

## Machines involved

| Machine | Role |
|---|---|
| This laptop (Windows ARM64, Snapdragon NPU) | Data prep, code authoring, final inference deployment |
| Windows x64 desktop, RTX 3080 / 64GB RAM / 2TB SSD | Model training (CUDA), and later ONNX quantization (Qualcomm recommends quantizing on x64 before deploying the quantized model to ARM64) |
| Other CPU-only / high-RAM/storage machines | Bulk data prep & augmentation (CPU-bound, storage-heavy) before shipping the finished dataset to the 3080 box |

There is no shared git remote yet — move folders between machines manually (USB /
network share / OneDrive). Everything here is plain CLI Python, so it runs the same way
on any machine: `python script.py --args`.

## Pipeline stages

1. **`dataprep/`** — fetch real labeled audio (ARRL W1AW code practice files: clean
   750Hz tone MP3s + transcripts, 5-40 WPM), segment into clips aligned to text, and
   synthesize noisy/faded/QRM'd variants to approximate real off-air HF reception
   (the clean ARRL audio alone is not representative of what the radio will actually
   produce).
2. **`model/`** — CNN-LSTM-CTC acoustic decoder (spectrogram in, character sequence
   out). Architecture follows the approach validated by AG1LE's real-time Morse
   decoder (1.5% CER / 97.2% word accuracy training on these same ARRL files).
   Train on the RTX 3080 box: `python model/train.py --data <manifest> --epochs N`.
3. **`lm/`** — contextual correction stage. Ham CW text is heavily structured
   (callsigns, Q-codes, prosigns like CQ/DE/K/KN/73) — a small correction model
   cleans up raw decoder output using that structure. *(stub — built after a trained
   acoustic model exists)*
4. **`export/`** — PyTorch -> ONNX export, then quantization (run on an x64 machine).
   *(stub)*
5. **`inference/`** — real-time app: capture radio audio -> front end -> ONNX model via
   `onnxruntime` + `onnxruntime-qnn` (Hexagon NPU) -> correction stage -> text.
   *(stub)*

## Note on data sources

- `souryadey/morse-dataset` (GitHub) is **synthetic symbolic** data (bit/grayscale
  arrays, not audio) — not used as training audio. Kept only as a possible later
  sanity-check for symbol/timing logic, not wired into the real pipeline.
- ARRL W1AW practice files are real audio+transcript and are the actual training data.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
