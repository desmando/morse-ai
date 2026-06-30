# inference/

Real-time CW decode and operating console. See the [main README](../README.md)
for full installation and usage instructions.

---

## tui.py — Field Day operating console

The main application. Full-screen TUI with a live decoded transcript pane and
an auto-filled editable response field. Press Enter to transmit the response as
CW; press Escape to abort mid-send.

See the main README for the full flag reference and feature walkthrough.

```bash
python inference/tui.py --checkpoint /path/to/checkpoint.pt \
    --my-call W1ABC --my-class 3A --my-section ENY
```

**Fake-decode mode** — no hardware or checkpoint required, for testing the
response logic:

```bash
python inference/tui.py --fake-decode \
    --my-call W1ABC --my-class 3A --my-section ENY
```

---

## realtime_decode.py — standalone decoder

Decodes audio from an input device and prints text continuously, without the
full operating console. Useful for testing that the model and audio path are
working before launching the TUI.

```bash
# list available input devices
python inference/realtime_decode.py --list-devices

# decode from a specific device
python inference/realtime_decode.py \
    --checkpoint /path/to/checkpoint.pt \
    --device "USB Audio"
```

Audio is decoded using overlapping sliding windows (`--window-seconds`,
`--stride-seconds`) — the model sees audio on both sides of every character,
so nothing gets corrupted at window boundaries the way a non-overlapping design
would. Each window's core region is committed and the segments are stitched
together as audio arrives.

The model was trained at 8 kHz — captured audio is resampled to match
automatically regardless of the input device's native rate.

---

## cat_keyer.py — CW keying via serial control line

Handles CW transmission and (on Icom radios) operating frequency polling.
Used by `tui.py`; can also be imported directly.

**Keying**: toggles the RTS or DTR hardware control line in Morse timing.
This is near-instantaneous at the OS level — a CI-V command takes ~8ms to
transmit at 9600 baud, which is too slow to accurately time a 20 WPM dot
(60ms). The radio must be configured to treat the chosen line as its CW SEND
input — check your radio's USB/Connectors menu.

**Frequency polling**: reads the operating frequency over CI-V after each QSO
sign-off so the band is recorded in the ADIF log. This only works with Icom
radios. On other brands the poll fails silently; the QSO is still logged
without a frequency/band field. The default CI-V address (`0x94`) matches the
IC-7300; other Icom models may need a different address via `--civ-address`.

```python
from inference.cat_keyer import CatKeyer

keyer = CatKeyer(port="COM5", key_line="rts", dry_run=False)
keyer.send_text("CQ TEST DE W1ABC K", wpm=20, timing_jitter=0.08)
freq_hz = keyer.read_frequency()   # Icom only
keyer.close()
```

Pass `dry_run=True` to log key-down/up transitions with timestamps instead of
opening the serial port — useful for verifying timing before keying live RF.

---

## fcc_uls.py — callsign verification

Downloads the FCC bulk amateur license database and builds a local index of
active US callsigns. The TUI checks decoded callsigns against this and flags
any US-pattern callsign that isn't in the database. Non-US callsigns are never
checked.

```bash
# download the latest database and build the index (~300 MB, a few minutes)
python inference/fcc_uls.py --download

# rebuild the index from an already-downloaded zip (faster, no re-download)
python inference/fcc_uls.py --rebuild
```

The FCC refreshes the data weekly. Re-run `--download` periodically to keep
the index current. The TUI runs without it — verification is simply skipped
and a warning is printed at startup.
