# morse-ai

ML pipeline to decode CW (Morse) audio off an HF radio into readable text, with
contextual correction, deployed for real-time inference on a Windows ARM64 laptop (CPU -
the laptop's NPU was tried and dropped, see `export/README.md`).

## Machines involved

| Machine | Role |
|---|---|
| This laptop (Windows ARM64) | Data prep, code authoring, final inference deployment (CPU) |
| Windows x64 desktop, RTX 3080 / 64GB RAM / 2TB SSD | Model training (CUDA) |
| Other CPU-only / high-RAM/storage machines | Bulk data prep & augmentation (CPU-bound, storage-heavy) before shipping the finished dataset to the 3080 box |

There is no shared git remote yet — this repo itself lives under OneDrive so the
**code** syncs between machines automatically. Everything here is plain CLI Python, so
it runs the same way on any machine: `python script.py --args`.

## Data lives outside this repo

All generated/downloaded data (raw audio, processed clips, augmented audio, synthetic
corpora, model checkpoints) is written to `C:\morse-ai-data` (see `paths.py`,
override with the `MORSE_AI_DATA` env var) — **not** inside this OneDrive-synced
folder. That data is regenerable junk and often large (gigabytes of WAV files); syncing
it between machines would be wasted bandwidth/storage. Each machine just re-runs the
`dataprep/`/`model/`/`lm/` scripts to produce its own local copy under `C:\morse-ai-data`.

## Pipeline stages

1. **`dataprep/`** — fetch real labeled audio (ARRL W1AW code practice files: clean
   750Hz tone MP3s + transcripts, 5-40 WPM), segment into clips aligned to text, and
   synthesize noisy/faded/QRM'd variants to approximate real off-air HF reception
   (the clean ARRL audio alone is not representative of what the radio will actually
   produce). `synthesize_morse_audio.py` additionally generates keyed-tone audio
   directly from text (e.g. the `lm/` QSO corpus) for far more training volume than
   the real ARRL recordings alone, with exact (not approximate) label alignment -
   feed its manifest into `augment_hf_channel.py` the same way as the real clips.
2. **`model/`** — CNN-LSTM-CTC acoustic decoder (spectrogram in, character sequence
   out). Architecture follows the approach validated by AG1LE's real-time Morse
   decoder (1.5% CER / 97.2% word accuracy training on these same ARRL files).
   Train on the RTX 3080 box: `python model/train.py --epochs N`.
3. **`lm/`** — contextual correction stage. Ham CW text is heavily structured
   (callsigns, Q-codes, prosigns like CQ/DE/K/KN/73) — a small correction model
   cleans up raw decoder output using that structure.
   `generate_qso_corpus.py` builds a synthetic training corpus weighted toward
   contest-style traffic (Sweepstakes, Field Day, state QSO parties, NAQP, DX
   contests, POTA) rather than casual ragchew, since that's what this will mostly
   see in the field. The actual correction model is still a stub — built once
   there's a trained acoustic model to evaluate it against.
4. **`export/`** — PyTorch -> ONNX export for CPU inference via plain `onnxruntime`.
   NPU/QNN HTP inference was tried and measured 2-2.5x *slower* than CPU EP on this
   model size, plus needing quantization/fixed-shapes/a slow first-load compile - not
   worth it when CPU alone gives 150-200x real-time throughput. See its README.
5. **`inference/`** — real-time app: capture radio audio -> front end -> PyTorch/ONNX
   model (CPU) -> correction stage -> text, plus a full TUI operating console with CAT
   radio control, ADIF logging, and contest-assist tooling. See its README for details.

## Note on data sources

- `souryadey/morse-dataset` (GitHub) is **synthetic symbolic** data (bit/grayscale
  arrays, not audio) — not used as training audio. Kept only as a possible later
  sanity-check for symbol/timing logic, not wired into the real pipeline.
- ARRL W1AW practice files are real audio+transcript and are the actual training data.
- ON6ZQ's CW practice page (on6zq.be) has 45 more clips (9 passages x 5 speeds:
  12/16/20/24/28 WPM) but its `robots.txt` explicitly disallows AI-agent user
  agents site-wide, so these are **not** auto-fetched. See
  `dataprep/import_on6zq.py` for the manual-download + reshape process.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
