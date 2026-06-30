# Real-time inference app

`realtime_decode.py` is a working v1: captures audio from an input device
(e.g. a USB sound card fed by a radio's audio-out), decodes it through a
trained PyTorch checkpoint directly, and prints text as it comes in.

```
python inference/realtime_decode.py --list-devices
python inference/realtime_decode.py --checkpoint <path to decoder_epochNNN.pt> --device "USB Audio"
```

`tui.py` wraps the same decode loop in a text-based operating console
(`prompt_toolkit`, chosen over `curses` for reliable Windows terminal
support): a top pane shows the running decoded transcript, and a bottom
pane is an editable field auto-filled with the suggested reply
(`lm/field_day_responder.py`) every time a new chunk decodes.

```
python inference/tui.py --checkpoint <path> --my-call W1ABC --my-class 3A --my-section ENY
```

On startup it prompts (via `radiolist_dialog`/`button_dialog`) for the audio
input device and, separately, a serial port for CAT control - pass
`--device`/`--serial-port` to skip either prompt. If a serial port is
selected, it also asks whether to use the default 9600 8N1 settings or
change baud rate/data bits/stop bits/parity (one extra keypress for the
common case of keeping defaults). The header shows the current setup, e.g.
"CAT: COM5 @ 9600 8N1", and "- TRANSMITTING" while a send is in progress.

Press Enter in the response field to transmit it as CW over the CAT link
(`inference/cat_keyer.py`) by toggling the RTS or DTR hardware control line
(`--key-line`, default `rts`) in Morse timing - not a CI-V command, since a
CI-V frame takes too long to transmit (~8ms at 9600 baud) to time CW
elements accurately (a dot at 20 WPM is only 60ms). This requires the
IC-7300's own Connectors menu to be set to treat that line as the CW SEND
line for its USB port - check the manual for the exact menu path, since
toggling the line from software does nothing until the radio's side is
configured to act on it. `--tx-wpm` sets the speed, `--tx-jitter` adds
per-element timing randomness so it doesn't sound machine-generated, and
`--cat-dry-run` logs key-down/up transitions with timestamps instead of
opening the serial port so you can sanity-check timing before keying a real
radio. If no serial port is configured, Enter just clears the field.

After transmitting a sign-off ("TU 73" - end of a QSO per
`lm/field_day_responder.py`'s own stage model), it polls the radio's
current operating frequency over CI-V (`CatKeyer.read_frequency()`) and
logs it to the transcript, e.g. `[QSO complete @ 14.0740 MHz]`. Unlike
keying, this does use the actual CI-V data protocol (reading a value back
needs a real command/response exchange). Researched rather than recalled
from memory, given the earlier RTS/DTR-vs-PTT mistake:
- Command `03h` = read operating frequency, confirmed against Icom's
  official [IC-7300 CI-V command table](https://www.manualslib.com/manual/1106166/Icom-Ic-7300.html?page=156).
- 5-byte little-endian BCD frequency encoding and the default CI-V address
  `0x94` confirmed against [Hamlib's IC-7300 driver](https://github.com/Hamlib/Hamlib/blob/master/rigs/icom/ic7300.c)
  (open source, widely deployed) - decoding was checked against a
  documented worked example and round-tripped across several HF/VHF
  frequencies.
- The IC-7300 has a configurable "CI-V USB Echo Back" setting that some
  software needs ON, which echoes the request frame back before the real
  reply - `read_frequency()` discards that echo if present, tested against
  a faked serial connection in both the echo-on and echo-off cases.
- `--civ-address` overrides the address if the radio's been reconfigured.

That same sign-off also appends one ADIF QSO record (`lm/adif_log.py`) to
`--adif-log` (default `<DATA_ROOT>/logs/field_day.adi`), so the log can be
imported into real contest logging software afterward. The other station's
callsign/exchange are picked up from whatever was most recently heard
(`extract_callsigns`/`extract_exchange`, the same parsers
`field_day_responder.py` already uses), tracked in `current_qso` and reset
once logged - if no callsign was ever captured for the QSO, it logs "not
logged" to the transcript instead of writing a record with no CALL field
(which ADIF requires). Also researched against the spec rather than
assumed:
- `<TAG:LENGTH>VALUE` field syntax, `<EOH>`/`<EOR>` markers - confirmed
  against the [ADIF 3.1.7 specification](https://www.adif.org/317/ADIF_317.htm).
- `CLASS`/`ARRL_SECT`/`SRX_STRING` describe the *contacted* station;
  `MY_ARRL_SECT`/`STX_STRING` describe *your own* station (there's no
  `MY_CLASS` field) - confirmed against the same spec. The free-text
  `STX_STRING`/`SRX_STRING` pair carries the full "class section" exchange
  regardless, matching what real Field Day logging software does.
- `CONTEST_ID` enumeration value for ARRL Field Day is `ARRL-FIELD-DAY`,
  not the `ARRL-FD` abbreviation that would've been a reasonable guess -
  checked against the spec's Contest ID enumeration table directly.

**Aborting a transmission**: press Escape while a send is in progress to
cancel it. `CatKeyer.send_text()` checks an abort flag between elements (so
it stops within one dot/dash/gap, never mid-element) and still forces the
key up either way. An aborted send is not treated as a completed QSO - no
frequency poll, no ADIF record, `current_qso` is left untouched so a
corrected message can complete the QSO normally afterward.

**Duplicate-contact handling**: before populating the response field,
checks whether the other station's callsign is already logged on whatever
band the radio's currently tuned to (`lm/adif_log.py`'s `load_worked_calls`
parses `--adif-log` back into `{band: {callsign, ...}}` at startup, so this
survives restarting the TUI mid-event, not just duplicates within one
session). Two cases:
- They're sending us their exchange (responding to us) and they're a
  duplicate -> the suggested reply tells them we already have them (`QSO
  B4 <band>`) instead of completing the exchange - and since that reply has
  no "TU 73", it does *not* trigger another frequency poll/log entry.
- They're calling CQ and they're a duplicate -> no real response is
  suggested at all. The field shows an informational message instead
  (`[K5XYZ already logged on 20m - not calling]`), and a response starting
  with "[" can never be transmitted - Enter on it is a no-op. Typing a real
  message into the field over it sends normally, so this doesn't remove the
  operator's ability to call them anyway if they disagree with the check.

**QRQ/QRS**: if either appears in decoded text, our own sending speed
(`current_wpm`, shown live in the header) is bumped up/down by
`--tx-wpm-step` (default 5, clamped to `[--tx-wpm-min, --tx-wpm-max]`,
default 5-40) and used for every subsequent send - `--tx-wpm` is just the
starting value now. This needed no model changes at all: Q/R/S already
decode as plain text today (same as any other letter - QRZ/QSL etc. already
worked), so this is a text match on the existing decoder output, exactly
like the "TU 73" sign-off check. Actually *measuring* the other station's
real sending speed (rather than just reacting to an explicit request)
would need frame-level timestamps that `ctc_greedy_decode()` currently
discards when it collapses predictions to text - a real but separate
enhancement, not needed for this.

**Callsign verification**: a wrong callsign is worse than a wrong
class/section - it's potentially a different station, and it goes straight
into the log. `inference/fcc_uls.py` downloads the FCC's bulk amateur
license database (`l_amat.zip`, ~175MB, refreshed weekly by the FCC) and
builds a local index of active US callsigns:

```
python inference/fcc_uls.py --download   # fetch the latest zip + build the index
python inference/fcc_uls.py --rebuild    # rebuild from an already-downloaded zip
```

If the other station's callsign matches the US callsign prefix pattern
(`K`/`N`/`W` + any second letter, or `A` + `A`-`L` specifically - not the
full `A`-`Z` range, since e.g. `AM`-`AZ` are allocated to other countries)
but isn't in the index, the transcript gets a "verify before logging" flag
- informational only, nothing is blocked. Non-US-pattern callsigns
(Canadian VE/VA/VO/VY, other DX) are never checked at all, since the FCC
database wouldn't have them regardless of correctness - this matters
because Field Day/RAC sections mean Canadian contacts are expected, and
flagging every one of them as "not found" would be a constant false alarm.
Verified directly against the real data rather than assumed, given the
CI-V lesson: downloaded the actual ~175MB database, confirmed the
pipe-delimited field positions by inspecting real rows (`HD.dat` index 4 =
callsign, index 5 = license status), and checked the US prefix pattern
against *every one* of the 827,263 real active licenses with zero
mismatches (an earlier draft of the pattern was wrong - it only allowed
`K`/`N`/`W` followed by `A`-`L` like the `A` block, when real data showed
they each allow any second letter A-Z).

The transcript/response update logic, device/serial-port enumeration, the
full Enter-to-transmit chain, the sign-off detection (verified to *not*
fire for an ordinary exchange and to fire for a "TU 73" message), the BCD
frequency decode/echo-handling, the full QSO-to-ADIF-record flow (simulated
CQ -> exchange -> sign-off, checking the written record and that
`current_qso` resets afterward), ADIF round-trip parsing (writing several
records including the same callsign on two different bands, then
confirming `load_worked_calls` reconstructs the right per-band sets), the
duplicate-CQ block (confirmed it does *not* transmit on Enter), the
duplicate-exchange B4 reply (confirmed it *does* transmit but does not
re-log), a real mid-transmission abort (slowed the WPM down, fired
Escape partway through, confirmed the key released and "[transmission
aborted]" was logged), QRQ/QRS speed adjustment (repeated QRS clamping
at the floor, repeated QRQ clamping at the ceiling, and confirming
`send_text()` is actually called with the adjusted speed, not the original
default), and callsign verification (a real active callsign correctly not
flagged, a made-up US-pattern callsign correctly flagged, a Canadian
callsign correctly never checked) were all verified directly. RTS/DTR
toggling and
opening a real CatKeyer with a CI-V address were both confirmed against a
real COM port on this machine. What's NOT verified from here: the IC-7300's
actual CI-V response over real hardware (only the documented format), and
the IC-7300's own menu-configured response to RTS/DTR keying. The actual
interactive full-screen rendering also still needs a check in a real
terminal, since the sandboxed environment used to build this has no
attached console to render to - and a `--cat-dry-run` smoke test is worth
doing before ever keying live RF with it.

## How it works

1. `sounddevice` captures audio continuously at the input device's native
   sample rate.
2. Each non-overlapping `--chunk-seconds` window (default 8.0, matching the
   clip length the current model was trained on) is resampled to 8kHz -
   every checkpoint so far was trained exclusively on 8kHz audio
   (`dataprep/synthesize_morse_audio.py`'s `SAMPLE_RATE`), so this keeps
   inference input distribution identical to training regardless of what
   rate the capture device actually runs at.
3. `model/features.py`'s feature extraction (tone auto-detect + spectrogram
   band - same front end used in training) runs on the resampled chunk.
4. The acoustic model decodes it via CTC greedy decoding
   (`model/decoder.py::ctc_greedy_decode`), and any non-empty result prints.

Verified against a known synthesized phrase (not seen in training) - decoded
output matched the input text exactly.

## Known v1 limitations

- **Non-overlapping fixed windows, not a sliding window.** This means up to
  one window's worth of latency before any text appears, and the same
  character-bleed-across-boundaries approximation that
  `dataprep/build_manifest.py` already accepts for the same structural
  reason (a character keyed right at a window boundary may get split or
  dropped). A shorter `--chunk-seconds` trades training-distribution
  accuracy for lower latency - not yet tuned.
- **No contextual correction yet** - raw decoder output goes straight to
  the screen. Once `lm/`'s correction model exists, pipe through it before
  display.
- **CPU/CUDA, not the NPU - deliberately.** NPU/QNN HTP inference was tried
  and measured slower than plain ONNX Runtime CPU EP on this model size
  (2-2.5x slower, plus quantization + fixed-shape + ~1-2min first-load compile
  complexity) - see `export/README.md`. CPU alone gives 150-200x real-time
  throughput, far more than the 1x actually needed, so this runs the PyTorch
  checkpoint directly (or `export/export_onnx.py`'s ONNX export, which is
  faster still) on CPU and stops there.
