"""Generate the next CW transmission for an ARRL Field Day QSO, given the
decoded text of what the other station just sent.

Field Day exchanges follow a fixed structure - the same one
generate_qso_corpus.py generates synthetic training data from:
  1. They call CQ, or call us directly -> we send our callsign.
  2. They send their callsign (answering our CQ, or calling us) -> we send
     our exchange (class + section).
  3. They send their exchange (class + section, often after acknowledging
     ours with "R") -> we acknowledge theirs and sign off.
  4. We can't confidently parse a callsign or exchange out of it at all
     (noise, fading, a decode error) -> we ask them to repeat ("AGN") rather
     than staying silent - real CW conditions are noisy enough that this is
     normal, not exceptional.

This is a stateless, per-message heuristic, not a trained model - it
classifies the CURRENT message by what it contains (a callsign-only
exchange vs. a class+section exchange) and reacts accordingly. It doesn't
track conversation history, so it won't handle out-of-order or repeated
transmissions specially, but that matches how mechanical contest exchanges
normally proceed anyway.

Usage:
  python field_day_responder.py --my-call W1ABC --my-class 3A --my-section ENY \\
      --heard "CQ FD DE K5XYZ K5XYZ K5XYZ K"
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lm.generate_qso_corpus import ARRL_SECTIONS

CALLSIGN_RE = re.compile(r"\b[A-Z]{1,2}[0-9][A-Z]{1,4}\b")
# Class and section must be adjacent (just like they're actually sent, e.g.
# "3A ENY") - matched as class-code-followed-by-any-2-4-letter-token rather
# than requiring exact membership in ARRL_SECTIONS, since that list is a
# partial set built for synthetic-data variety, not full real-world section
# coverage (e.g. it's missing "WCF" - West Central Florida). Requiring
# adjacency to an actual class code is what avoids ambiguity with prosigns
# that collide with real section abbreviations (e.g. "DE" is also the
# Delaware section) - a whitelist isn't needed for that, and not using one
# means any real section is recognized, not just the ones in our list.
EXCHANGE_RE = re.compile(r"\b(\d{1,2}[A-F])\s+([A-Z]{2,4})\b")


def extract_callsigns(text: str) -> list[str]:
    return CALLSIGN_RE.findall(text.upper())


def extract_exchange(text: str, exclude: tuple[str, str] | None = None):
    """Returns the (class, section) pair found in text, preferring the last
    one that doesn't match `exclude` (our own exchange echoed back as part
    of their acknowledgment) - or None if no new exchange is present."""
    text = text.upper()
    candidates = EXCHANGE_RE.findall(text)
    if not candidates:
        return None
    if exclude:
        filtered = [c for c in candidates if c != exclude]
        return filtered[-1] if filtered else None
    return candidates[-1]


def generate_response(heard: str, my_call: str, my_class: str, my_section: str) -> str:
    heard_upper = heard.upper()
    my_call, my_class, my_section = my_call.upper(), my_class.upper(), my_section.upper()

    exchange = extract_exchange(heard_upper, exclude=(my_class, my_section))
    if exchange is not None:
        their_class, their_section = exchange
        return f"R {their_class} {their_section} TU 73"

    calls = extract_callsigns(heard_upper)
    their_call = next((c for c in calls if c != my_call), None)
    if their_call:
        if "CQ" in heard_upper:
            return f"{their_call} DE {my_call} {my_call} K"
        return f"{their_call} DE {my_call} {my_class} {my_section} {my_class} {my_section} K"

    # no callsign, no exchange - couldn't confidently parse anything, so ask
    # for a repeat instead of staying silent (their_call is always None here -
    # the block above already returned if it had found one).
    return f"DE {my_call} AGN K"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--my-call", required=True)
    parser.add_argument("--my-class", required=True, help="e.g. 3A, 1D, 5F")
    parser.add_argument("--my-section", required=True, help="ARRL/RAC section abbreviation, e.g. ENY, WCF")
    parser.add_argument("--heard", required=True, help="decoded text received from the other station")
    args = parser.parse_args()

    if args.my_section.upper() not in ARRL_SECTIONS:
        print(f"warning: {args.my_section!r} isn't in the known ARRL section list "
              f"({len(ARRL_SECTIONS)} known) - continuing anyway", file=sys.stderr)

    print(generate_response(args.heard, args.my_call, args.my_class, args.my_section))


if __name__ == "__main__":
    main()
