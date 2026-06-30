"""Generate suggested CW responses for normal HF contacts and rag chewing.

Unlike Field Day (fixed class/section exchange, strictly structured), a
general contact follows a looser pattern:
  1. They call CQ  ->  answer with their call + ours
  2. They answer our CQ or call us directly  ->  RST + name + QTH
  3. They send RST/name/QTH info  ->  acknowledge theirs, send ours
  4. Sign-off (73/SK) heard or sent  ->  73 and sign off

For rag chewing, auto-suggest just pre-fills the other station's callsign
and our DE line so the operator isn't looking it up - the rest is up to them.

Usage:
  python lm/contact_responder.py --my-call W1ABC --my-name Mike --my-qth "Chicago IL" \\
      --heard "CQ CQ DE K5XYZ K5XYZ K"
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lm.field_day_responder import extract_callsigns

RST_RE = re.compile(r'\b([1-5][1-9][1-9])\b')
NAME_RE = re.compile(r'\bNAME\s+([A-Z]{2,12})\b')
QTH_RE = re.compile(r'\bQTH\s+([A-Z]{2,20}(?:\s+[A-Z]{2,20})?)\b')
SIGNOFF_RE = re.compile(r'\b(73|SK|CUL|CUAGN)\b')


def extract_rst(text: str) -> str | None:
    m = RST_RE.search(text.upper())
    return m.group(1) if m else None


def extract_name(text: str) -> str | None:
    m = NAME_RE.search(text.upper())
    return m.group(1).strip() if m else None


def extract_qth(text: str) -> str | None:
    m = QTH_RE.search(text.upper())
    return m.group(1).strip() if m else None


def is_signoff(text: str) -> bool:
    return bool(SIGNOFF_RE.search(text.upper()))


def generate_contact_response(heard: str, my_call: str, my_name: str = "", my_qth: str = "",
                               my_rst: str = "599") -> str:
    heard_upper = heard.upper()
    my_call = my_call.upper()

    calls = extract_callsigns(heard_upper)
    their_call = next((c for c in calls if c != my_call), None)
    prefix = f"{their_call} DE {my_call}" if their_call else f"DE {my_call}"

    # Sign-off detected in what we heard
    if is_signoff(heard_upper):
        return f"{prefix} 73 TU SK"

    # They've sent signal info — acknowledge and send ours
    their_rst = extract_rst(heard_upper)
    their_name = extract_name(heard_upper)
    their_qth = extract_qth(heard_upper)

    if their_rst or their_name or their_qth:
        parts = ["R TU"]
        if their_name:
            parts.append(their_name)
        parts.append(f"UR RST {my_rst} {my_rst}")
        if my_name:
            parts.append(f"NAME {my_name.upper()}")
        if my_qth:
            parts.append(f"QTH {my_qth.upper()}")
        parts.append("HW? K")
        return f"{prefix} " + " ".join(parts)

    # They called CQ
    if their_call and "CQ" in heard_upper:
        return f"{their_call} DE {my_call} {my_call} K"

    # They called us directly (our call appears in their text)
    if their_call and my_call in heard_upper:
        info = f"GM UR RST {my_rst} {my_rst}"
        if my_name:
            info += f" NAME {my_name.upper()}"
        if my_qth:
            info += f" QTH {my_qth.upper()}"
        return f"{prefix} {info} HW? K"

    # Just a callsign but context unclear
    if their_call:
        return f"{prefix} K"

    return f"DE {my_call} AGN? K"


def generate_ragchew_response(heard: str, my_call: str) -> str:
    """Pre-fills the other station's callsign and our DE line so the operator
    isn't hunting for it in the transcript — they type the rest themselves."""
    calls = extract_callsigns(heard.upper())
    their_call = next((c for c in calls if c != my_call.upper()), None)
    if their_call:
        return f"{their_call} DE {my_call.upper()} "
    return f"DE {my_call.upper()} "


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--my-call", required=True)
    parser.add_argument("--my-name", default="")
    parser.add_argument("--my-qth", default="")
    parser.add_argument("--my-rst", default="599", help="RST to send the other station")
    parser.add_argument("--heard", required=True)
    args = parser.parse_args()
    print(generate_contact_response(args.heard, args.my_call, args.my_name, args.my_qth, args.my_rst))


if __name__ == "__main__":
    main()
