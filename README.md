# morse-ai

Real-time CW decoder and Field Day operating console. Listens to your radio's
audio output, decodes Morse code as it arrives, and suggests the correct contest
response — pre-filled, editable, and transmitted by pressing Enter.

## Hardware you need

- An HF radio capable of CW
- Audio from the radio into the computer — either the radio's built-in USB audio
  (the IC-7300 has this; most modern rigs do) or a USB sound card connected to
  the audio out
- A serial/USB connection to the radio for CW keying — the software keys CW by
  toggling the RTS or DTR control line, not a CI-V command. Which line your radio
  uses depends on the model and cable; check your manual for the USB SEND or
  keying assignment and use `--key-line rts` or `--key-line dtr` accordingly

## Installation

**Python 3.10 or later required.**

```bash
git clone <repo-url>
cd morse-ai
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Install PyTorch separately — pick the right build for your machine:

```bash
# CPU only (fine for inference — no GPU needed to run the TUI)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA (if you have an NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Get the model checkpoint

The trained model weights are not in this repo. Download the latest `.pt`
checkpoint file from the [Releases page](../../releases) and put it anywhere
convenient — you'll pass the path via `--checkpoint`.

The `vocab.txt` file from `manifests/` must also be accessible. By default the
software looks for it at `$MORSE_AI_DATA/manifests/vocab.txt`. Set that
environment variable to point at your data directory:

```bash
# Windows
set MORSE_AI_DATA=C:\morse-ai-data
# Mac/Linux
export MORSE_AI_DATA=/path/to/morse-ai-data
```

Then put `vocab.txt` at `<MORSE_AI_DATA>/manifests/vocab.txt`.

## Optional: FCC callsign verification

Downloads the FCC active amateur license database and builds a local index so
decoded US callsigns are flagged if they don't match any real license. Takes a
few minutes and about 300 MB. The TUI runs fine without it — verification is
just skipped.

```bash
python inference/fcc_uls.py --download
```

Run again periodically (the FCC refreshes the data weekly) or use `--rebuild`
to reindex a previously downloaded zip without re-downloading.

## Running the TUI

Provide your own callsign, Field Day class, and ARRL/RAC section:

```bash
python inference/tui.py \
    --checkpoint /path/to/decoder_epoch093.pt \
    --my-call W1ABC \
    --my-class 3A \
    --my-section ENY
```

On startup it prompts you to select:
1. The audio input device (the one receiving your radio's audio)
2. A serial port for CAT keying — or skip it if you only want to see decoded
   text without transmitting

### Before you key the transmitter — use `--cat-dry-run` first

This runs everything except actually opening the serial port. Key-down/up
transitions are logged to the transcript with timestamps so you can check the
timing looks right before going on-air:

```bash
python inference/tui.py \
    --checkpoint /path/to/decoder_epoch093.pt \
    --serial-port COM5 \
    --cat-dry-run \
    --my-call W1ABC --my-class 3A --my-section ENY
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to the `.pt` model file |
| `--my-call` | *(required)* | Your callsign |
| `--my-class` | *(required)* | Your Field Day class (e.g. `3A`, `1D`) |
| `--my-section` | *(required)* | Your ARRL/RAC section (e.g. `ENY`, `WCF`) |
| `--device` | *(prompted)* | Audio input device name or index — skips the startup prompt |
| `--serial-port` | *(prompted)* | Serial port for CW keying (e.g. `COM5`, `/dev/ttyUSB0`) |
| `--key-line` | `rts` | Which serial line keys CW: `rts` or `dtr` |
| `--civ-address` | `0x94` | Icom CI-V address for frequency polling (IC-7300 default) |
| `--cat-dry-run` | off | Simulate keying without opening the serial port |
| `--tx-wpm` | `20` | Starting transmit speed in WPM |
| `--tx-jitter` | `0.08` | Per-element timing randomness so it doesn't sound machine-generated |
| `--adif-log` | `<DATA>/logs/field_day.adi` | Where completed QSO records are written |
| `--udp-log-host` | `255.255.255.255` | UDP broadcast address for real-time logging integration |
| `--udp-log-port` | `2333` | UDP port |
| `--no-udp-log` | off | Disable UDP broadcast |
| `--window-seconds` | `8.0` | Decode window length — match this to what the checkpoint was trained on |
| `--stride-seconds` | `4.0` | Overlap stride — `window/2` gives clean stitching at boundaries |

## What the TUI does

- **Top pane**: the running decoded transcript, scrollable
- **Bottom pane**: the suggested response, auto-filled after each decoded
  chunk — edit it freely, then press **Enter** to transmit it as CW

**Transmitting**: Enter keys the response at `--tx-wpm` with `--tx-jitter`
applied so the timing isn't machine-perfect. Press **Escape** to abort a
send in progress — stops within one dot/dash, forces the key up, and does
not log the QSO.

**Speed adjustment**: if QRQ or QRS appears in decoded text, the transmit
speed steps up or down by `--tx-wpm-step` (default 5 WPM), clamped to
`[--tx-wpm-min, --tx-wpm-max]`. The current speed shows in the header.

**Duplicate contacts**: before filling the response field, the software
checks the ADIF log for the other station's callsign on the current band.
If they're already logged, it suggests a `QSO B4` reply instead of a
normal exchange response. If they're calling CQ and already logged, the
field shows an informational note that can't be accidentally transmitted
(any response starting with `[` is blocked from sending). You can type
over it to answer them anyway.

**QSO logging**: after transmitting a sign-off (`TU 73`), the software
polls the radio's current frequency over CI-V (Icom radios only — ignored
gracefully on other brands), logs the QSO to `--adif-log` in ADIF format,
and broadcasts it over UDP so other logging software on the network picks
it up in real time. The ADIF file can be imported into N1MM, N3FJP,
Log4OM, or any other contest logger after the event.

**Callsign verification**: if a decoded callsign matches the US callsign
prefix pattern but isn't in the FCC active-license index, the transcript
shows a "verify before logging" flag. Non-US callsigns (Canadian, other DX)
are never checked.

## CI-V frequency polling — Icom radios only

After each QSO sign-off, the software reads the radio's current operating
frequency over CI-V to record the band in the ADIF log. This only works
with Icom radios using CI-V. On any other brand, the poll fails silently
and the QSO is still logged without a frequency/band field. The default
CI-V address (`--civ-address 0x94`) matches the IC-7300; other Icom models
may need a different address — check the radio's CI-V settings menu.

## Testing without hardware — fake-decode mode

No radio, no audio device, no checkpoint needed. Opens the TUI with an
extra input field so you can type pretend decoded text and see what the
response logic generates:

```bash
python inference/tui.py --fake-decode \
    --my-call W1ABC --my-class 3A --my-section ENY
```

Tab/Shift+Tab cycle between the inject field and the response field. Try
inputs like:
- `CQ TEST DE K5XYZ K5XYZ K` — should suggest answering their CQ
- `K5XYZ DE W1ABC 3A ENY 3A ENY K` — exchange phase
- `R 3A ENY TU 73` — sign-off, should trigger QSO logging if a serial port
  is configured (or `--cat-dry-run`)
- `QRS` — should step the transmit speed down
- The same callsign twice — second time should show a duplicate warning

## UDP logging integration

Every completed QSO is broadcast as a raw ADIF record over UDP immediately
after it's written to the log file. Many ham logging programs (N1MM+, N3FJP,
Log4OM) listen on port 2333 to pick up contacts from other software on the
same network in real time. To target a specific machine instead of
broadcasting to the whole subnet:

```bash
--udp-log-host 192.168.1.50 --udp-log-port 2333
```

Disable entirely with `--no-udp-log`.
