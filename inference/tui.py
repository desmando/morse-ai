"""Text-based live operating console for ARRL Field Day: top pane shows the
running decoded transcript, bottom pane is an editable response field that's
automatically pre-filled with the suggested reply
(lm/field_day_responder.py) every time a new (non-empty) chunk is decoded.

On startup, prompts for the audio input device and (optionally) a serial
port for CAT control - unless given via --device/--serial-port, in which
case that prompt is skipped.

The response field is fully editable - type over it freely, then press
Enter to transmit it as CW over the CAT link (see inference/cat_keyer.py -
the RTS/DTR line is toggled in Morse timing at --tx-wpm, with --tx-jitter
added so it doesn't sound machine-perfect). If no serial port was selected,
Enter just clears the field with nothing transmitted. The field still gets
overwritten by the next auto-suggestion as soon as new audio decodes, same
as before - that's unrelated to sending, not a side effect of it.

After transmitting a sign-off ("TU 73" - the end of a QSO per
lm/field_day_responder.py's own stage model), this polls the radio's
current operating frequency over CI-V, logs it to the transcript, and
appends an ADIF QSO record (lm/adif_log.py) to --adif-log - the other
station's callsign/exchange are picked up from whatever was most recently
heard, tracked in `current_qso` and reset after each logged QSO.

Press Escape to abort a transmission in progress (stops at the next
dot/dash/gap boundary and forces the key up - the app keeps running, this
only cancels the current send).

Duplicate-contact handling: before suggesting a response, checks whether
the other station's callsign is already logged on the band the radio's
currently tuned to (--adif-log, plus anything logged earlier this session).
If they're sending us their exchange (responding to us) and they're a
duplicate, the suggested reply tells them we already have them instead of
completing the exchange. If they're calling CQ and they're a duplicate, no
real response is suggested at all - the field shows an informational
"already logged" message instead, and that specific message can never be
transmitted (Enter on a response starting with "[" is a no-op), though
typing a real message into the field over it sends normally.

QRQ/QRS: if either appears in decoded text, our own sending speed (shown in
the header) is bumped up/down by --tx-wpm-step (clamped to
[--tx-wpm-min, --tx-wpm-max]) and used for all subsequent sends - this is a
plain text match on the decoder's existing output, not a new model
capability (Q/R/S already decode fine today, same as any other letter).
This doesn't try to measure the other station's actual speed - only reacts
to an explicit request - since that would need frame-level timestamps the
current decode path discards, not just a text match.

Callsign verification: a decoded callsign is a much worse kind of error
than a wrong class/section - it's potentially a different station
entirely. If the other station's callsign looks like a US callsign
(inference/fcc_uls.py's US_CALLSIGN_RE) but isn't in the locally-built FCC
active-license index (run that file with --download/--rebuild first), the
transcript gets a "verify before logging" flag - informational only, it
doesn't block anything. Non-US-pattern callsigns (Canadian, other DX)
are never checked at all, since the FCC database wouldn't have them
regardless of whether they're correct.

--fake-decode: skips audio/model entirely (no --checkpoint needed) and adds
a third "Inject decoded text" field above the response field - type pretend
decoded text and press Enter to feed it through the exact same on_text()
pipeline (QRQ/QRS, callsign verification, duplicate detection, response
generation) real decode would use, to test response behavior without a
radio, audio device, or trained checkpoint. Tab/Shift+Tab cycle focus
between fields. CAT control still works normally in this mode if a serial
port is selected (or pass --cat-dry-run to see simulated key-down/up
logging without real hardware) - only the decode side is faked.

Usage:
  # Field Day (default)
  python inference/tui.py --checkpoint <path> --my-call W1ABC --my-class 3A --my-section ENY

  # Normal HF contact
  python inference/tui.py --checkpoint <path> --mode contact \\
      --my-call W1ABC --my-name Mike --my-qth "Chicago IL"

  # Rag chew (auto-suggest just pre-fills callsign header, you type the rest)
  python inference/tui.py --checkpoint <path> --mode ragchew --my-call W1ABC

  # Fake-decode (no hardware or checkpoint needed, any mode)
  python inference/tui.py --fake-decode --mode contact \\
      --my-call W1ABC --my-name Mike --my-qth "Chicago IL"
"""
import argparse
import json
import sys
import threading
from pathlib import Path

import sounddevice as sd
import serial.tools.list_ports
from prompt_toolkit import Application
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.shortcuts import button_dialog, radiolist_dialog
from prompt_toolkit.widgets import Frame, Label, TextArea

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.cat_keyer import CatKeyer
from inference.fcc_uls import is_us_pattern, load_active_callsigns
from inference.realtime_decode import MODEL_SAMPLE_RATE, StreamDecoder, iter_decoded_stream, load_model, parse_device
from lm.adif_log import append_qso, broadcast_qso_udp, freq_to_band, load_worked_calls
from lm.contact_responder import (extract_name, extract_qth, extract_rst, generate_contact_response,
                                   generate_ragchew_response, is_signoff)
from lm.field_day_responder import extract_callsigns, extract_exchange, generate_response
from paths import DATA_ROOT

# 9600 8N1 - a conservative, widely-supported default for CAT serial links.
DEFAULT_SERIAL_SETTINGS = {"baudrate": 9600, "bytesize": 8, "stopbits": 1, "parity": "N"}
BAUD_RATE_CHOICES = [300, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200]
DATA_BITS_CHOICES = [5, 6, 7, 8]
STOP_BITS_CHOICES = [1, 1.5, 2]
PARITY_CHOICES = [("N", "None"), ("E", "Even"), ("O", "Odd"), ("M", "Mark"), ("S", "Space")]


def prompt_audio_device():
    """Returns (device_index, device_name) for an input-capable device, or
    exits the program if there are none or the user cancels - there's no
    reasonable way to proceed without one."""
    devices = sd.query_devices()
    choices = [(i, f"{i}: {d['name']}") for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not choices:
        sys.exit("No audio input devices found.")
    index = radiolist_dialog(
        title="Select audio input device",
        text="Receiving the radio's audio (e.g. a USB sound card fed by its audio-out):",
        values=choices,
    ).run()
    if index is None:
        sys.exit("No audio device selected.")
    return index, devices[index]["name"]


def prompt_serial_port():
    """Returns the chosen port device string (e.g. "COM5"), or None if the
    user explicitly skips CAT control."""
    ports = list(serial.tools.list_ports.comports())
    choices = [(p.device, f"{p.device} - {p.description}") for p in ports]
    choices.append((None, "(skip - no CAT control)"))
    return radiolist_dialog(
        title="Select CAT control serial port",
        text="For sending responses as CW via PTT keying (radio must already be in CW mode):",
        values=choices,
        default=None,
    ).run()


def prompt_serial_settings() -> dict:
    """Returns a dict of baudrate/bytesize/stopbits/parity. Asks once whether
    to use the 9600 8N1 default before drilling into individual fields, so
    the common case is one extra keypress, not four."""
    change = button_dialog(
        title="Serial settings",
        text="Use default serial settings (9600 baud, 8 data bits, 1 stop bit, no parity)?",
        buttons=[("Use defaults", False), ("Change", True)],
    ).run()
    if not change:
        return dict(DEFAULT_SERIAL_SETTINGS)

    def pick(title, values, default):
        result = radiolist_dialog(title=title, text=f"Select {title.lower()}:",
                                   values=values, default=default).run()
        return default if result is None else result

    return {
        "baudrate": pick("Baud rate", [(b, str(b)) for b in BAUD_RATE_CHOICES],
                          DEFAULT_SERIAL_SETTINGS["baudrate"]),
        "bytesize": pick("Data bits", [(b, str(b)) for b in DATA_BITS_CHOICES],
                          DEFAULT_SERIAL_SETTINGS["bytesize"]),
        "stopbits": pick("Stop bits", [(b, str(b)) for b in STOP_BITS_CHOICES],
                          DEFAULT_SERIAL_SETTINGS["stopbits"]),
        "parity": pick("Parity", PARITY_CHOICES, DEFAULT_SERIAL_SETTINGS["parity"]),
    }


def format_serial_settings(settings: dict) -> str:
    """e.g. {"baudrate": 9600, "bytesize": 8, "stopbits": 1, "parity": "N"} -> "9600 8N1"."""
    stopbits = settings["stopbits"]
    stopbits_str = str(int(stopbits)) if stopbits == int(stopbits) else str(stopbits)
    return f"{settings['baudrate']} {settings['bytesize']}{settings['parity']}{stopbits_str}"


def build_app(my_call: str, device_name: str, serial_port: str | None, serial_settings: dict | None,
              accept_handler=None, on_abort=None, inject_handler=None):
    transcript_area = TextArea(text="", read_only=True, scrollbar=True, wrap_lines=True)
    response_area = TextArea(text="", multiline=False, accept_handler=accept_handler)

    if serial_port:
        cat_status = f"{serial_port} @ {format_serial_settings(serial_settings)}"
    else:
        cat_status = "none"
    header = Label(text=f" Audio: {device_name!r} - CAT: {cat_status} - DE {my_call.upper()} - "
                        f"Enter to send, Esc to abort, Ctrl+C/Ctrl+Q to quit ")
    panes = [header, Frame(transcript_area, title="Decoded")]

    inject_area = None
    if inject_handler is not None:
        inject_area = TextArea(text="", multiline=False, accept_handler=inject_handler)
        panes.append(Frame(inject_area, title="Inject pretend decoded text (Enter to feed in, Tab to switch fields)"))

    panes.append(Frame(response_area, title="Suggested response (Enter to transmit, auto-filled, edit freely)"))
    root = HSplit(panes)

    kb = KeyBindings()

    @kb.add("c-c")
    @kb.add("c-q")
    def _(event):
        event.app.exit()

    @kb.add("escape")
    def _(event):
        if on_abort:
            on_abort()

    if inject_area is not None:
        kb.add("tab")(focus_next)
        kb.add("s-tab")(focus_previous)

    focused = inject_area if inject_area is not None else response_area
    app = Application(layout=Layout(root, focused_element=focused), key_bindings=kb, full_screen=True)
    return app, transcript_area, response_area, header, inject_area


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="required unless --fake-decode is set")
    parser.add_argument("--vocab", default=str(DATA_ROOT / "manifests" / "vocab.txt"))
    parser.add_argument("--fake-decode", action="store_true",
                         help="skip audio/model entirely - adds a text field to manually inject pretend "
                              "decoded text instead, for testing response behavior without hardware")
    parser.add_argument("--device", default=None,
                         help="input device name or index - skips the startup audio prompt if given")
    parser.add_argument("--serial-port", default=None,
                         help="CAT control serial port (e.g. COM5) - skips the startup serial prompt if given")
    parser.add_argument("--baud-rate", type=int, default=DEFAULT_SERIAL_SETTINGS["baudrate"])
    parser.add_argument("--data-bits", type=int, default=DEFAULT_SERIAL_SETTINGS["bytesize"])
    parser.add_argument("--stop-bits", type=float, default=DEFAULT_SERIAL_SETTINGS["stopbits"])
    parser.add_argument("--parity", default=DEFAULT_SERIAL_SETTINGS["parity"], choices=["N", "E", "O", "M", "S"])
    parser.add_argument("--tx-wpm", type=float, default=20.0,
                         help="starting CW transmit speed - adjusted at runtime by QRQ/QRS, see below")
    parser.add_argument("--tx-wpm-step", type=float, default=5.0,
                         help="WPM to adjust by per QRQ (faster) or QRS (slower) request heard")
    parser.add_argument("--tx-wpm-min", type=float, default=5.0)
    parser.add_argument("--tx-wpm-max", type=float, default=40.0)
    parser.add_argument("--tx-jitter", type=float, default=0.08,
                         help="relative per-element timing jitter on transmit (e.g. 0.08 = ~8%%) - "
                              "0 sounds machine-perfect, which is the thing to avoid")
    parser.add_argument("--key-line", default="rts", choices=["rts", "dtr"],
                         help="which serial control line keys CW - must match the radio's own SEND/keying "
                              "menu assignment for its USB port (check the manual)")
    parser.add_argument("--civ-address", default="0x94",
                         help="Icom CI-V address for frequency polling, e.g. 0x94 (IC-7300 default)")
    parser.add_argument("--cat-dry-run", action="store_true",
                         help="log key-down/up transitions and fake CI-V replies instead of opening the "
                              "serial port - verify tx timing safely first")
    parser.add_argument("--mode", default="field-day", choices=["field-day", "contact", "ragchew"],
                         help="operating mode: field-day (contest exchange), contact (RST/name/QTH), "
                              "ragchew (decoder + logging active, auto-suggest just pre-fills callsign)")
    parser.add_argument("--my-class", default=None, help="Field Day class, e.g. 3A (required for field-day mode)")
    parser.add_argument("--my-section", default=None, help="ARRL/RAC section (required for field-day mode)")
    parser.add_argument("--my-name", default="", help="Your name, used in contact/ragchew responses")
    parser.add_argument("--my-qth", default="", help="Your QTH, used in contact responses")
    parser.add_argument("--my-rst", default="599", help="Default RST to send in contact mode")
    parser.add_argument("--adif-log", default=str(DATA_ROOT / "logs" / "qso_log.adi"),
                         help="ADIF log file to append a record to after each completed QSO")
    parser.add_argument("--conversation-log", default=str(DATA_ROOT / "logs" / "conversation_log.jsonl"),
                         help="JSONL file logging every sent and received text chunk with timestamps - "
                              "useful for reviewing contacts and building training data for future AI responses")
    parser.add_argument("--no-udp-log", action="store_true",
                         help="disable UDP broadcast of each logged QSO (enabled by default)")
    parser.add_argument("--udp-log-host", default="255.255.255.255",
                         help="UDP broadcast/unicast address for logged QSOs - many ham logging programs "
                              "(N1MM, N3FJP, Log4OM, etc.) listen for this to stay in sync in real time")
    parser.add_argument("--udp-log-port", type=int, default=2333)
    parser.add_argument("--window-seconds", type=float, default=8.0,
                         help="decode window length - match the clip length the model was trained on")
    parser.add_argument("--stride-seconds", type=float, default=4.0,
                         help="how far the window advances each step - window_seconds/2 gives clean "
                              "non-overlapping core regions")
    parser.add_argument("--lm", default=None, metavar="PATH",
                         help="path to ham_char_lm.json to enable CTC beam search + LM decoding "
                              "instead of greedy; improves callsign and exchange accuracy")
    parser.add_argument("--lm-weight", type=float, default=0.3,
                         help="LM score weight (0 = pure acoustic greedy-equivalent, higher = more LM influence)")
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--torch-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--my-call", required=True)
    args = parser.parse_args()

    if args.mode == "field-day" and (not args.my_class or not args.my_section):
        parser.error("--my-class and --my-section are required for --mode field-day")

    if args.fake_decode:
        model, vocab, device, device_name, lm = None, None, None, "FAKE (manual injection)", None
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --fake-decode is set")
        model, vocab = load_model(args.checkpoint, args.vocab, args.torch_device)
        lm = None
        if args.lm:
            from lm.ngram_lm import CharNgramLM
            lm = CharNgramLM.load(args.lm)
        if args.device is not None:
            device = parse_device(args.device)
            device_name = sd.query_devices(device, "input")["name"]
        else:
            device, device_name = prompt_audio_device()

    if args.serial_port is not None:
        serial_port = args.serial_port
        serial_settings = {"baudrate": args.baud_rate, "bytesize": args.data_bits,
                            "stopbits": args.stop_bits, "parity": args.parity}
    else:
        serial_port = prompt_serial_port()
        serial_settings = prompt_serial_settings() if serial_port else None

    keyer = None
    if serial_port:
        try:
            keyer = CatKeyer(port=serial_port, key_line=args.key_line, civ_address=int(args.civ_address, 0),
                              dry_run=args.cat_dry_run, **serial_settings)
        except Exception as exc:
            sys.exit(f"Failed to open {serial_port} for CAT control: {exc}")

    def log_conversation(direction: str, text: str):
        """Appends one sent/received entry to the conversation log as JSONL.
        direction is 'received' or 'sent'. Both sides are logged so the full
        exchange can be replayed or used as training data later."""
        from datetime import datetime, timezone
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "direction": direction, "text": text}
        log_path = Path(args.conversation_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    worked_calls = load_worked_calls(args.adif_log)  # {band: {callsign, ...}} - survives a restart
    active_callsigns = load_active_callsigns()  # {} if inference/fcc_uls.py hasn't been run yet
    if not active_callsigns:
        print("warning: no FCC active-license index found - run inference/fcc_uls.py --download first "
              "to enable callsign verification (continuing without it)", file=sys.stderr)
    tx_lock = threading.Lock()
    current_qso = {"call": None, "exchange": None, "rst_rcvd": None, "their_name": None, "their_qth": None}
    current_tx_abort = {"event": None}  # the in-progress transmission's abort Event, if any
    current_wpm = {"value": args.tx_wpm}  # adjusted at runtime by QRQ/QRS

    def append_transcript_line(line: str):
        new_transcript = transcript_area.text + line + "\n"
        transcript_area.buffer.set_document(
            Document(new_transcript, cursor_position=len(new_transcript)), bypass_readonly=True)
        app.invalidate()

    def set_tx_status(active: bool):
        cat = f"{serial_port} @ {format_serial_settings(serial_settings)}" if serial_port else "none"
        suffix = " - TRANSMITTING (Esc to abort)" if active else ""
        header.text = (f" Audio: {device_name!r} - CAT: {cat}{suffix} - {current_wpm['value']:.0f} WPM - "
                        f"DE {args.my_call.upper()} - Enter to send, Esc to abort, Ctrl+C/Ctrl+Q to quit ")
        app.invalidate()

    def on_abort():
        abort_event = current_tx_abort["event"]
        if abort_event is not None:
            abort_event.set()
            append_transcript_line("[transmission aborted]")

    def on_response_accept(buf) -> bool:
        text = buf.text.strip()
        if text.startswith("["):
            return False  # informational message (e.g. duplicate-contact notice) - never transmit
        if not keyer or not text:
            return False
        if not tx_lock.acquire(blocking=False):
            return False  # already transmitting - drop this Enter rather than overlap on the same link

        def tx_worker():
            abort_event = threading.Event()
            current_tx_abort["event"] = abort_event
            try:
                set_tx_status(True)
                keyer.send_text(text, wpm=current_wpm["value"], timing_jitter=args.tx_jitter,
                                 abort_event=abort_event)
                if abort_event.is_set():
                    return  # cut short - don't treat as a completed QSO
                log_conversation("sent", text)

                sent_upper = text.upper()
                qso_complete = (
                    (args.mode == "field-day" and "TU 73" in sent_upper) or
                    (args.mode in ("contact", "ragchew") and is_signoff(sent_upper))
                )
                if qso_complete:
                    try:
                        freq_hz = keyer.read_frequency()
                        band = freq_to_band(freq_hz / 1e6)
                        append_transcript_line(f"[QSO complete @ {freq_hz / 1e6:.4f} MHz]")
                        if current_qso["call"]:
                            if args.mode == "field-day":
                                record = append_qso(args.adif_log, my_call=args.my_call,
                                                     their_call=current_qso["call"], freq_hz=freq_hz,
                                                     my_class=args.my_class, my_section=args.my_section,
                                                     their_exchange=current_qso["exchange"])
                            else:
                                record = append_qso(args.adif_log, my_call=args.my_call,
                                                     their_call=current_qso["call"], freq_hz=freq_hz,
                                                     rst_sent=args.my_rst,
                                                     rst_rcvd=current_qso["rst_rcvd"],
                                                     their_name=current_qso["their_name"],
                                                     their_qth=current_qso["their_qth"])
                            if band:
                                worked_calls.setdefault(band, set()).add(current_qso["call"])
                            append_transcript_line(f"[logged to {args.adif_log}]")
                            if not args.no_udp_log:
                                try:
                                    broadcast_qso_udp(record, host=args.udp_log_host, port=args.udp_log_port)
                                    append_transcript_line(
                                        f"[broadcast via UDP to {args.udp_log_host}:{args.udp_log_port}]")
                                except Exception as exc:
                                    append_transcript_line(f"[UDP broadcast failed: {exc}]")
                        else:
                            append_transcript_line("[no callsign captured for this QSO - not logged]")
                    except Exception as exc:
                        append_transcript_line(f"[frequency poll / log failed: {exc}]")
                    finally:
                        current_qso.update({"call": None, "exchange": None,
                                            "rst_rcvd": None, "their_name": None, "their_qth": None})
            finally:
                current_tx_abort["event"] = None
                set_tx_status(False)
                tx_lock.release()

        threading.Thread(target=tx_worker, daemon=True).start()
        return False  # clear the field once the send is kicked off

    def on_inject_accept(buf) -> bool:
        text = buf.text.strip()
        if text:
            on_text(text)
        return False  # clear the field after feeding it in

    app, transcript_area, response_area, header, inject_area = build_app(
        args.my_call, device_name, serial_port, serial_settings,
        accept_handler=on_response_accept, on_abort=on_abort,
        inject_handler=on_inject_accept if args.fake_decode else None)

    def on_text(text: str):
        append_transcript_line(text)
        log_conversation("received", text)
        text_upper = text.upper()

        if "QRQ" in text_upper:
            current_wpm["value"] = min(args.tx_wpm_max, current_wpm["value"] + args.tx_wpm_step)
            append_transcript_line(f"[QRQ heard - sending speed up to {current_wpm['value']:.0f} WPM]")
            set_tx_status(False)
        elif "QRS" in text_upper:
            current_wpm["value"] = max(args.tx_wpm_min, current_wpm["value"] - args.tx_wpm_step)
            append_transcript_line(f"[QRS heard - sending speed down to {current_wpm['value']:.0f} WPM]")
            set_tx_status(False)

        calls = extract_callsigns(text)
        their_call = next((c for c in calls if c != args.my_call.upper()), None)

        if (their_call and active_callsigns and is_us_pattern(their_call)
                and their_call not in active_callsigns):
            append_transcript_line(f"[{their_call} looks like a US callsign but isn't in the FCC "
                                    f"active-license list - verify before logging]")

        band = None
        if keyer and their_call:
            try:
                band = freq_to_band(keyer.read_frequency() / 1e6)
            except Exception:
                band = None
        already_worked = bool(band and their_call in worked_calls.get(band, set()))

        if args.mode == "field-day":
            exchange = extract_exchange(text, exclude=(args.my_class.upper(), args.my_section.upper()))
            if already_worked and exchange:
                response = f"{their_call} DE {args.my_call.upper()} QSO B4 {band.upper()} K"
            elif already_worked and "CQ" in text_upper:
                response = f"[{their_call} already logged on {band} - not calling]"
            else:
                if their_call:
                    current_qso["call"] = their_call
                if exchange:
                    current_qso["exchange"] = exchange
                response = generate_response(text, args.my_call, args.my_class, args.my_section)

        elif args.mode == "contact":
            if their_call:
                current_qso["call"] = their_call
            rst = extract_rst(text)
            if rst:
                current_qso["rst_rcvd"] = rst
            name = extract_name(text)
            if name:
                current_qso["their_name"] = name
            qth = extract_qth(text)
            if qth:
                current_qso["their_qth"] = qth
            if already_worked and "CQ" in text_upper:
                response = f"[{their_call} already logged on {band} - not calling]"
            else:
                response = generate_contact_response(text, args.my_call, args.my_name,
                                                      args.my_qth, args.my_rst)

        else:  # ragchew
            if their_call:
                current_qso["call"] = their_call
            response = generate_ragchew_response(text, args.my_call)

        response_area.buffer.set_document(Document(response, cursor_position=len(response)))
        app.invalidate()

    def worker():
        decoder = StreamDecoder(model, vocab, args.torch_device, MODEL_SAMPLE_RATE,
                                 window_seconds=args.window_seconds, stride_seconds=args.stride_seconds,
                                 lm=lm, lm_weight=args.lm_weight, beam_width=args.beam_width)
        try:
            for text in iter_decoded_stream(device, decoder):
                if text:
                    on_text(text)
        except Exception as exc:  # surface audio/decoding errors instead of silently dying
            append_transcript_line(f"[worker error: {exc}]")

    if not args.fake_decode:
        threading.Thread(target=worker, daemon=True).start()
    else:
        append_transcript_line("[fake-decode mode - type pretend decoded text in the Inject field below]")
    try:
        app.run()
    finally:
        if keyer:
            keyer.close()


if __name__ == "__main__":
    main()
