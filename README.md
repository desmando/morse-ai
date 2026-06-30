# morse-ai

Real-time CW decoder and operating console. Listens to your radio's audio
output, decodes Morse code as it arrives, and suggests the correct response
pre-filled in an editable field — press Enter to transmit it as CW.

Works for **ARRL Field Day** (contest exchange), **normal HF contacts**
(RST, name, QTH), and **rag chewing** (decoder and logging active, response
field pre-fills the other station's callsign so you can type from there).

## Hardware you need

- An HF radio capable of CW
- Audio from the radio into the computer — either the radio's built-in USB
  audio (the IC-7300 has this; most modern rigs do) or a USB sound card
  connected to the audio out
- A serial/USB connection for CW keying — the software toggles the RTS or DTR
  control line in Morse timing. Which line your radio uses depends on the model
  and cable; check your manual for the USB SEND or keying assignment and use
  `--key-line rts` or `--key-line dtr` accordingly

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

The `vocab.txt` file must also be accessible. By default the software looks
for it at `$MORSE_AI_DATA/manifests/vocab.txt`. Set that environment variable
to point at your data directory:

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
to reindex without re-downloading.

---

## Operating modes

### Field Day

```bash
python inference/tui.py \
    --checkpoint /path/to/checkpoint.pt \
    --my-call W1ABC --my-class 3A --my-section ENY
```

Stages through a Field Day contest exchange: CQ → callsign → class/section →
`TU 73` sign-off. ADIF records include `CONTEST_ID`, `CLASS`, and `ARRL_SECT`.
Duplicate-contact detection checks the ADIF log and suggests `QSO B4` if the
callsign is already logged on the current band.

### Normal HF contact

```bash
python inference/tui.py \
    --checkpoint /path/to/checkpoint.pt \
    --mode contact \
    --my-call W1ABC --my-name Mike --my-qth "Chicago IL"
```

Stages through a standard contact: CQ → RST + name + QTH → acknowledge theirs
and send yours → 73/SK sign-off. The responder picks up the other station's
name and QTH from their decoded text and acknowledges them by name if it caught
it. ADIF records use `RST_SENT`, `RST_RCVD`, `NAME`, and `QTH`.

Use `--my-rst` to set the signal report you send (default `599`).

### Rag chew

```bash
python inference/tui.py \
    --checkpoint /path/to/checkpoint.pt \
    --mode ragchew \
    --my-call W1ABC
```

Decoder and logging work normally. The response field just pre-fills
`K5XYZ DE W1ABC ` so you're not hunting for the callsign in the transcript —
you type the rest yourself.

---

## Before you key the transmitter — use `--cat-dry-run` first

Simulates keying without opening the serial port. Key-down/up transitions are
logged to the transcript with timestamps so you can verify the timing before
going on-air:

```bash
python inference/tui.py \
    --checkpoint /path/to/checkpoint.pt \
    --serial-port COM5 --cat-dry-run \
    --mode contact --my-call W1ABC --my-name Mike --my-qth "Chicago IL"
```

---

## What the TUI does

- **Top pane**: the running decoded transcript, scrollable
- **Bottom pane**: the suggested response, auto-filled after each decoded
  chunk — edit it freely, then press **Enter** to transmit it as CW

**Transmitting**: Enter keys the response at `--tx-wpm` with `--tx-jitter`
timing randomness so it doesn't sound machine-generated. Press **Escape** to
abort mid-send — stops within one dot/dash, forces the key up, does not log
the QSO.

**Speed adjustment**: QRQ or QRS in decoded text steps the transmit speed up
or down by `--tx-wpm-step` (default 5 WPM), clamped to
`[--tx-wpm-min, --tx-wpm-max]`. Current speed shows in the header.

**QSO logging**: triggered when you transmit a sign-off (`TU 73` in Field Day
mode; `73` or `SK` in contact/ragchew mode). The software polls the radio's
operating frequency over CI-V (Icom only — ignored gracefully on other
brands), appends an ADIF record to `--adif-log`, and broadcasts it over UDP
so other logging software on the network picks it up in real time.

**Conversation log**: every decoded chunk received and every transmission you
send is appended to `--conversation-log` as timestamped JSONL — both sides of
every exchange in one file. Useful for reviewing contacts afterward or as
training data for improving the auto-responder.

**Callsign verification**: decoded US-pattern callsigns not found in the FCC
active-license index get a "verify before logging" flag in the transcript.
Non-US callsigns are never checked.

**Duplicate contacts** (Field Day and contact modes): before filling the
response field, checks the ADIF log for the other station's callsign on the
current band. Suggests `QSO B4` if already logged, or shows a non-transmittable
informational note if they're calling CQ.

---

## Key flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to the `.pt` model file |
| `--my-call` | *(required)* | Your callsign |
| `--mode` | `field-day` | `field-day`, `contact`, or `ragchew` |
| `--my-class` | *(required for field-day)* | Field Day class, e.g. `3A` |
| `--my-section` | *(required for field-day)* | ARRL/RAC section, e.g. `ENY` |
| `--my-name` | | Your name — used in contact/ragchew responses |
| `--my-qth` | | Your QTH — used in contact responses |
| `--my-rst` | `599` | RST to send the other station in contact mode |
| `--device` | *(prompted)* | Audio input device name or index |
| `--serial-port` | *(prompted)* | Serial port for CW keying, e.g. `COM5` |
| `--key-line` | `rts` | Which serial line keys CW: `rts` or `dtr` |
| `--civ-address` | `0x94` | Icom CI-V address for frequency polling (IC-7300 default) |
| `--cat-dry-run` | off | Log key-down/up transitions without opening the serial port |
| `--tx-wpm` | `20` | Starting transmit speed in WPM |
| `--tx-wpm-step` | `5` | WPM change per QRQ/QRS |
| `--tx-wpm-min` | `5` | Floor for QRS speed reduction |
| `--tx-wpm-max` | `40` | Ceiling for QRQ speed increase |
| `--tx-jitter` | `0.08` | Per-element timing randomness |
| `--adif-log` | `<DATA>/logs/qso_log.adi` | ADIF log file |
| `--conversation-log` | `<DATA>/logs/conversation_log.jsonl` | Timestamped sent/received log |
| `--udp-log-host` | `255.255.255.255` | UDP broadcast address |
| `--udp-log-port` | `2333` | UDP port |
| `--no-udp-log` | off | Disable UDP broadcast |
| `--window-seconds` | `8.0` | Decode window length |
| `--stride-seconds` | `4.0` | Overlap stride — `window/2` gives clean boundary stitching |

---

## CI-V frequency polling — Icom radios only

After each QSO sign-off, the software reads the operating frequency over CI-V
to record the band in the ADIF log. This only works with Icom radios. On any
other brand it fails silently and the QSO is logged without a
frequency/band field. The default CI-V address (`0x94`) matches the IC-7300;
other models may need a different value — check the radio's CI-V settings menu.

## UDP logging integration

Every completed QSO is broadcast as a raw ADIF record over UDP immediately
after it's written to the log file. Many ham logging programs (N1MM+, N3FJP,
Log4OM) listen on port 2333 to pick up contacts in real time. To target a
specific machine instead of broadcasting to the whole subnet:

```bash
--udp-log-host 192.168.1.50 --udp-log-port 2333
```

Disable entirely with `--no-udp-log`.

## Testing without hardware — fake-decode mode

No radio, no audio device, no checkpoint needed. Opens the TUI with an extra
input field to type pretend decoded text and see what the response logic
generates. Works with any `--mode`.

```bash
# Field Day
python inference/tui.py --fake-decode \
    --my-call W1ABC --my-class 3A --my-section ENY

# Normal contact
python inference/tui.py --fake-decode --mode contact \
    --my-call W1ABC --my-name Mike --my-qth "Chicago IL"

# Rag chew
python inference/tui.py --fake-decode --mode ragchew --my-call W1ABC
```

Tab/Shift+Tab cycle between the inject field and the response field.
