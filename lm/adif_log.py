"""Appends ADIF (Amateur Data Interchange Format) QSO records to a log file
- the standard format ham logging software (N1MM, N3FJP, Log4OM, etc.)
imports, so this log can be merged into a "real" contest log afterward.

Verified against the ADIF 3.1.7 specification (adif.org/317/ADIF_317.htm)
rather than assumed:
  - File/record syntax: case-insensitive <TAG:LENGTH>VALUE data specifiers,
    header terminated by <EOH>, each QSO record terminated by <EOR>.
  - CLASS, ARRL_SECT, SRX_STRING are about the CONTACTED station; only
    STATION_CALLSIGN and MY_ARRL_SECT/STX_STRING are about one's own
    station - there's no MY_CLASS field, which is why the free-text
    STX_STRING/SRX_STRING exchange pair (what real Field Day logging
    software actually uses) carries the full "class section" text
    regardless, rather than relying solely on the structured fields.
  - CONTEST_ID enumeration value for ARRL Field Day is "ARRL-FIELD-DAY"
    (not the "ARRL-FD" abbreviation that would otherwise be a reasonable
    guess) - confirmed against the spec's Contest ID enumeration table.
"""
import re
import socket
from datetime import datetime, timezone
from pathlib import Path

ADIF_VERSION = "3.1.7"

_FIELD_RE = re.compile(r"<([A-Za-z_0-9]+)(?::(\d+)(?::[A-Za-z])?)?>")

# (lower_mhz, upper_mhz, band_name) - standard amateur HF/VHF band edges.
BAND_EDGES = [
    (1.8, 2.0, "160m"), (3.5, 4.0, "80m"), (5.3, 5.4, "60m"),
    (7.0, 7.3, "40m"), (10.1, 10.15, "30m"), (14.0, 14.35, "20m"),
    (18.068, 18.168, "17m"), (21.0, 21.45, "15m"), (24.89, 24.99, "12m"),
    (28.0, 29.7, "10m"), (50.0, 54.0, "6m"), (144.0, 148.0, "2m"),
]


def freq_to_band(freq_mhz: float) -> str:
    for lo, hi, name in BAND_EDGES:
        if lo <= freq_mhz <= hi:
            return name
    return ""


def _field(tag: str, value: str) -> str:
    return f"<{tag}:{len(value)}>{value}"


def format_qso_record(my_call: str, their_call: str, freq_hz: float, mode: str = "CW", when=None, *,
                       my_class: str | None = None, my_section: str | None = None,
                       their_exchange: tuple[str, str] | None = None,
                       rst_sent: str | None = None, rst_rcvd: str | None = None,
                       their_name: str | None = None, their_qth: str | None = None) -> str:
    """Returns one ADIF QSO record (fields + <EOR>, newline-terminated).

    For Field Day: pass my_class, my_section, and optionally their_exchange
    (class, section) - writes CONTEST_ID, MY_ARRL_SECT, STX/SRX_STRING.
    For general contacts: pass rst_sent/rst_rcvd/their_name/their_qth as
    available - writes RST_SENT, RST_RCVD, NAME, QTH instead."""
    when = when or datetime.now(timezone.utc)
    freq_mhz = freq_hz / 1e6

    fields = [
        _field("CALL", their_call.upper()),
        _field("QSO_DATE", when.strftime("%Y%m%d")),
        _field("TIME_ON", when.strftime("%H%M%S")),
        _field("FREQ", f"{freq_mhz:.6f}"),
        _field("MODE", mode.upper()),
        _field("STATION_CALLSIGN", my_call.upper()),
    ]
    band = freq_to_band(freq_mhz)
    if band:
        fields.append(_field("BAND", band))

    if my_section:
        my_exchange_str = f"{my_class} {my_section}" if my_class else my_section
        fields.append(_field("MY_ARRL_SECT", my_section.upper()))
        fields.append(_field("STX_STRING", my_exchange_str.upper()))
        fields.append(_field("CONTEST_ID", "ARRL-FIELD-DAY"))
    if their_exchange:
        their_class, their_section = their_exchange
        fields.append(_field("CLASS", their_class.upper()))
        fields.append(_field("ARRL_SECT", their_section.upper()))
        fields.append(_field("SRX_STRING", f"{their_class} {their_section}".upper()))

    if rst_sent:
        fields.append(_field("RST_SENT", rst_sent))
    if rst_rcvd:
        fields.append(_field("RST_RCVD", rst_rcvd))
    if their_name:
        fields.append(_field("NAME", their_name.upper()))
    if their_qth:
        fields.append(_field("QTH", their_qth.upper()))

    return " ".join(fields) + " <EOR>\n"


def _adif_header() -> str:
    return (
        "morse-ai Field Day log\n"
        f"<ADIF_VER:{len(ADIF_VERSION)}>{ADIF_VERSION}\n"
        "<PROGRAMID:8>morse-ai\n"
        "<EOH>\n"
    )


def append_qso(log_path, **kwargs) -> str:
    """Builds one QSO record via format_qso_record(**kwargs) and appends it
    to log_path, writing the ADIF header first if the file doesn't exist
    yet. Returns the record that was written."""
    record = format_qso_record(**kwargs)
    path = Path(log_path)
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if is_new:
            f.write(_adif_header())
        f.write(record)
    return record


def broadcast_qso_udp(record: str, host: str = "255.255.255.255", port: int = 2333) -> None:
    """Broadcasts one ADIF QSO record over UDP - the common integration
    mechanism several ham logging programs (N1MM, N3FJP, Log4OM, etc.) use
    to pick up contacts logged by another program on the same network in
    real time. Sends just the record (as returned by append_qso/
    format_qso_record), not the file header - listeners expect one ADIF
    record per datagram. Default host is the limited broadcast address so
    any listener on the local subnet receives it without needing to be
    configured with this machine's specific IP; pass a unicast address
    (e.g. "127.0.0.1") to target one specific listener instead."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        if host in ("255.255.255.255",) or host.endswith(".255"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(record.encode("utf-8"), (host, port))


def parse_records(text: str) -> list[dict]:
    """Parses ADIF <TAG:LENGTH>VALUE data back into a list of per-QSO field
    dicts, skipping the header (anything before <EOH>). <EOH>/<EOR> are
    bare markers with no length/value, handled separately from fielded tags."""
    records = []
    current: dict = {}
    in_header = True
    pos = 0
    while pos < len(text):
        m = _FIELD_RE.search(text, pos)
        if not m:
            break
        tag = m.group(1).upper()
        length = m.group(2)
        if tag == "EOH":
            in_header = False
            pos = m.end()
            continue
        if tag == "EOR":
            if not in_header:
                records.append(current)
            current = {}
            pos = m.end()
            continue
        if length is None:
            pos = m.end()
            continue
        length = int(length)
        value_start = m.end()
        value = text[value_start:value_start + length]
        pos = value_start + length
        if not in_header:
            current[tag] = value
    return records


def load_worked_calls(log_path) -> dict[str, set[str]]:
    """Returns {band: {callsign, ...}} from an existing ADIF log, so
    duplicate-contact checking survives restarting the TUI mid-event.
    Returns {} if the log doesn't exist yet."""
    path = Path(log_path)
    if not path.exists():
        return {}
    worked: dict[str, set[str]] = {}
    for rec in parse_records(path.read_text(encoding="utf-8")):
        call, band = rec.get("CALL"), rec.get("BAND")
        if call and band:
            worked.setdefault(band, set()).add(call.upper())
    return worked
