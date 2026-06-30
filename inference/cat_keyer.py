"""Keys Morse code over a CAT serial link by toggling the RTS (or DTR)
hardware control line in timed bursts shaped like dots and dashes. All
dot/dash/gap timing (plus jitter) is computed and played out here in
software - the radio supplies none of it - which is what makes it possible
to add the human-like timing variation a mechanically perfect keyer
wouldn't have.

RTS/DTR line toggling (not a CI-V protocol command) is the standard
technique for computer CW keying: it's a raw hardware control-line state
change with no framing/parsing latency, unlike a CI-V command frame, which
takes ~8ms just to transmit at 9600 baud before the radio even processes it
- far too slow and jittery relative to a 60ms dot at 20 WPM. It's also rig-
agnostic, working on any radio with a serial keying interface, not just
Icom CI-V-capable ones.

PREREQUISITE: the IC-7300 (like most Icom USB-CAT rigs) needs its own menu
configured to treat RTS or DTR as the CW "SEND"/key line for its USB port -
check the radio's Connectors menu (exact path/wording not verified here -
check your manual) and assign whichever line matches --key-line below.
Until that's set on the radio itself, toggling the line from software does
nothing - this is a prerequisite this code can't satisfy remotely.

Use --cat-dry-run (logs intended key-down/up transitions with timestamps
instead of opening the serial port) to verify timing/jitter completely
safely before keying a real radio.

read_frequency(), unlike keying, DOES use the CI-V data protocol - reading
a value back requires an actual command/response exchange, there's no
hardware-line shortcut for that. Verified via research rather than memory
(given the RTS/DTR-vs-PTT mistake earlier in this project):
  - Command 03h = "read operating frequency", confirmed against Icom's own
    IC-7300 CI-V command table (manualslib.com/manual/1106166/Icom-Ic-7300.html, page 156).
  - Response frequency encoding (5-byte BCD, little-endian byte order, two
    decimal digits per byte) confirmed against Hamlib's IC-7300 driver -
    open source, widely deployed, civ_731_mode=0 for this rig means the
    modern 5-byte (not legacy 4-byte) frequency format applies
    (github.com/Hamlib/Hamlib/blob/master/rigs/icom/ic7300.c and icom.c).
  - Default CI-V address 0x94, also confirmed via the same Hamlib driver.
  - The IC-7300's USB CI-V port has a configurable "Echo Back" menu setting
    that some software needs ON, which echoes a request frame back before
    the real reply - read_frequency() defensively discards an echo of its
    own request if one shows up first, so it works either way.
"""
import random
import sys
import time
from pathlib import Path

import serial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataprep.synthesize_morse_audio import MORSE_CODE, _jitter, char_elements

CONTROLLER_ADDRESS = 0xE0
DEFAULT_CIV_ADDRESS = 0x94  # IC-7300 factory default - override if yours differs
READ_FREQ_CMD = 0x03
FREQ_BCD_BYTES = 5  # IC-7300 doesn't use legacy civ_731_mode -> 5-byte BCD, not 4


def _bcd_le_to_freq(data: bytes) -> int:
    """Decodes Icom's little-endian BCD frequency encoding (2 decimal digits
    per byte, least significant byte first) to a frequency in Hz."""
    freq = 0
    for i, byte in enumerate(data):
        freq += (byte & 0x0F) * 10 ** (2 * i)
        freq += (byte >> 4) * 10 ** (2 * i + 1)
    return freq


def text_to_key_elements(text: str, wpm: float, timing_jitter: float = 0.0, rng=None):
    """Returns [(duration_s, is_key_down), ...] for `text` at `wpm` - the same
    dot/dash/gap timing model as synthesize_morse_audio.py's synthesize_line,
    minus Farnsworth spacing and audio rendering (not needed for live keying)."""
    rng = rng or random.Random()
    dot_s = 1.2 / wpm
    dash_s = 3 * dot_s
    intra_gap_s = dot_s
    inter_char_gap_s = 3 * dot_s
    inter_word_gap_s = 7 * dot_s

    elements: list[tuple[float, bool]] = []
    words = text.upper().split(" ")
    for wi, word in enumerate(words):
        chars_in_word = [c for c in word if c in MORSE_CODE]
        for ci, ch in enumerate(chars_in_word):
            for d, is_tone in char_elements(ch, dot_s, dash_s, intra_gap_s):
                elements.append((_jitter(d, rng, timing_jitter), is_tone))
            if ci < len(chars_in_word) - 1:
                elements.append((_jitter(inter_char_gap_s, rng, timing_jitter), False))
        if wi < len(words) - 1 and chars_in_word:
            elements.append((_jitter(inter_word_gap_s, rng, timing_jitter), False))

    return elements


class CatKeyer:
    def __init__(self, port: str, baudrate: int, bytesize: int, stopbits: float, parity: str,
                 key_line: str = "rts", civ_address: int = DEFAULT_CIV_ADDRESS, dry_run: bool = False):
        if key_line not in ("rts", "dtr"):
            raise ValueError(f"key_line must be 'rts' or 'dtr', got {key_line!r}")
        self.key_line = key_line
        self.civ_address = civ_address
        self.dry_run = dry_run
        # rtscts/dsrdtr=False: those enable automatic hardware flow control,
        # which would fight with manually driving the same lines as a key.
        self._conn = None if dry_run else serial.Serial(
            port=port, baudrate=baudrate, bytesize=bytesize, stopbits=stopbits, parity=parity,
            rtscts=False, dsrdtr=False, timeout=1)

    def _set_key_line(self, on: bool):
        if self.dry_run:
            print(f"[dry-run] {self.key_line.upper()} {'DOWN' if on else 'UP  '} @ {time.monotonic():.4f}")
            return
        if self.key_line == "rts":
            self._conn.rts = on
        else:
            self._conn.dtr = on

    def key_down(self):
        self._set_key_line(True)

    def key_up(self):
        self._set_key_line(False)

    def send_text(self, text: str, wpm: float, timing_jitter: float = 0.08, rng=None, abort_event=None):
        """Blocks for the duration of the transmission - call from a worker
        thread, never the UI thread. If abort_event is set partway through,
        stops at the next element boundary (at most one dash + one gap,
        well under a second even at slow WPM) rather than mid-element."""
        elements = text_to_key_elements(text, wpm, timing_jitter, rng)
        try:
            for duration, is_tone in elements:
                if abort_event is not None and abort_event.is_set():
                    break
                if is_tone:
                    self.key_down()
                    time.sleep(duration)
                    self.key_up()
                else:
                    time.sleep(duration)
        finally:
            self.key_up()  # never leave the key stuck down, even if interrupted

    def _civ_frame(self, command: bytes) -> bytes:
        return bytes([0xFE, 0xFE, self.civ_address, CONTROLLER_ADDRESS]) + command + bytes([0xFD])

    def _read_civ_frame(self, timeout: float) -> bytes:
        """Reads one FE FE ... FD - delimited CI-V frame, blocking up to
        `timeout` seconds total. Continuously re-syncs to the most recent
        FE FE marker seen so a stray leading byte (or two frames arriving
        back-to-back) can't desync the parser."""
        deadline = time.monotonic() + timeout
        buf = bytearray()
        while time.monotonic() < deadline:
            byte = self._conn.read(1)
            if not byte:
                continue
            buf += byte
            idx = buf.rfind(b"\xfe\xfe")
            if idx > 0:
                buf = buf[idx:]
            if len(buf) >= 6 and buf[:2] == b"\xfe\xfe" and buf[-1] == 0xFD:
                return bytes(buf)
        raise TimeoutError(f"no complete CI-V frame received within {timeout}s - "
                            f"check the radio is powered on, the port/baud rate are correct, "
                            f"and the CI-V address (0x{self.civ_address:02x}) matches the radio")

    def read_frequency(self, timeout: float = 1.0) -> int:
        """Returns the radio's current operating (VFO) frequency in Hz."""
        if self.dry_run:
            print("[dry-run] read_frequency -> 14074000 (placeholder, no real radio queried)")
            return 14074000

        request = self._civ_frame(bytes([READ_FREQ_CMD]))
        self._conn.reset_input_buffer()
        self._conn.write(request)

        frame = self._read_civ_frame(timeout)
        if frame == request:
            # CI-V USB Echo Back (if enabled on the radio) echoes our own
            # request before the real reply - skip it and read again
            frame = self._read_civ_frame(timeout)

        if len(frame) < 5 + FREQ_BCD_BYTES + 1 or frame[4] != READ_FREQ_CMD:
            raise ValueError(f"unexpected CI-V response to frequency query: {frame.hex(' ')}")

        return _bcd_le_to_freq(frame[5:5 + FREQ_BCD_BYTES])

    def close(self):
        if self._conn is not None:
            self.key_up()
            self._conn.close()
