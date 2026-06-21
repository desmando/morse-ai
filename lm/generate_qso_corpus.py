"""Generate a synthetic corpus of ham-radio CW traffic, weighted toward
contest-style exchanges rather than ragchew.

Real CW contest traffic is extremely terse and formulaic: CQ calls, callsign
exchanges, RST/serial/section/zone codes, punctuated by a small set of
prosigns (DE, K, KN, AR, SK, BK, TU, R). That's a very different statistical
distribution than casual conversation (weather, health, life stories) - and
it's most of what this system will actually need to decode in the field, so
the synthetic text corpus is built to match it.

Output is plain text, one "band" of simulated activity per call to
generate_band(): a stream of CQs and QSOs concatenated the way they'd
actually key up on the air.

Usage:
  python generate_qso_corpus.py --num-qsos 2000
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA_ROOT

US_PREFIXES = ["K", "W", "N", "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AI",
               "KA", "KB", "KC", "KD", "KE", "KF", "KG", "WA", "WB", "WC", "WD"]
DX_PREFIXES = ["G", "M", "VE", "VA", "JA", "JH", "DL", "DK", "EA", "EI", "F",
               "I", "PA", "ON", "SM", "OZ", "LA", "9A", "S5", "PY", "LU", "ZL", "VK"]

NAMES = ["MIKE", "JOHN", "BOB", "STEVE", "DAVE", "TOM", "JIM", "BILL", "PAUL",
         "RICK", "GARY", "DAN", "ED", "FRANK", "JOE", "KEN", "LARRY", "PETE",
         "RON", "SAM", "TONY", "WALT", "ART", "CARL", "DOUG", "GREG", "PHIL"]

US_STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI",
             "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI",
             "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
             "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
             "VT", "VA", "WA", "WV", "WI", "WY"]

ARRL_SECTIONS = ["ENY", "WNY", "NLI", "NNJ", "SNJ", "EMA", "WMA", "CT", "RI",
                 "VT", "NH", "ME", "EPA", "WPA", "MDC", "DE", "AZ", "EWA",
                 "WWA", "OR", "SF", "EB", "LAX", "SDG", "SV", "AL", "GA", "SC",
                 "NC", "VA", "WV", "OH", "MI", "WI", "MN", "ND", "SD", "IA"]

SS_PRECEDENCE = ["Q", "A", "B", "M"]
FD_CLASSES = ["A", "B", "C", "D", "E", "F"]
ANTENNAS = ["VERTICAL", "DIPOLE", "YAGI", "LOOP", "INV V", "END FED", "MAGLOOP", "HEXBEAM"]
POWERS = ["5W", "QRP", "100W", "500W", "1KW"]
PARK_FLAVOR = ["IN THE PARK", "AT THE LAKE", "ON A HILLTOP", "PORTABLE OP", "AT THE SUMMIT"]
PROSIGNS_SIGNOFF = ["TU 73 GL", "TNX QSO 73", "GL IN THE TEST", "73 ES GL"]


def random_callsign(rng: random.Random) -> str:
    prefix = rng.choice(US_PREFIXES if rng.random() < 0.8 else DX_PREFIXES)
    digit = rng.randint(0, 9)
    suffix_len = rng.choice([1, 2, 2, 3])
    suffix = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(suffix_len))
    return f"{prefix}{digit}{suffix}"


def random_rst(rng: random.Random) -> str:
    # most contest reports are a perfect 599/5NN; sample occasional weaker ones
    return rng.choices(["599", "5NN", "579", "559", "449", "339"],
                        weights=[55, 20, 10, 8, 4, 3])[0]


def exchange_contest_serial(rng: random.Random) -> str:
    return f"{random_rst(rng)} {rng.randint(1, 999):03d}"


def exchange_sweepstakes(rng: random.Random) -> str:
    nr = rng.randint(1, 1500)
    prec = rng.choice(SS_PRECEDENCE)
    check = rng.randint(0, 99)  # last 2 digits of year first licensed
    section = rng.choice(ARRL_SECTIONS)
    return f"{nr} {prec} {check:02d} {section}"


def exchange_field_day(rng: random.Random) -> str:
    n_tx = rng.randint(1, 20)
    cls = rng.choice(FD_CLASSES)
    section = rng.choice(ARRL_SECTIONS)
    return f"{n_tx}{cls} {section}"


def exchange_state_qp(rng: random.Random) -> str:
    return f"{random_rst(rng)} {rng.choice(US_STATES)}"


def exchange_naqp(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    state = rng.choice(US_STATES)
    return f"{name} {state}"


def exchange_dx_zone(rng: random.Random) -> str:
    return f"{random_rst(rng)} {rng.randint(1, 40):02d}"


def station_chatter(rng: random.Random) -> str:
    """Brief, factual station/setup remarks - not personal narrative."""
    parts = []
    if rng.random() < 0.5:
        parts.append(f"RIG {rng.choice(['FT', 'IC', 'K', 'TS'])}{rng.randint(100,9999)}")
    if rng.random() < 0.6:
        parts.append(f"ANT {rng.choice(ANTENNAS)}")
    if rng.random() < 0.5:
        parts.append(f"PWR {rng.choice(POWERS)}")
    if rng.random() < 0.3:
        parts.append(rng.choice(PARK_FLAVOR))
    return " ".join(parts) if parts else "FB SIGNAL"


def generate_qso(rng: random.Random, scenario: str) -> str:
    a = random_callsign(rng)
    b = random_callsign(rng)

    if scenario == "sweepstakes":
        exch_a, exch_b = exchange_sweepstakes(rng), exchange_sweepstakes(rng)
    elif scenario == "field_day":
        exch_a, exch_b = exchange_field_day(rng), exchange_field_day(rng)
    elif scenario == "state_qp":
        exch_a, exch_b = exchange_state_qp(rng), exchange_state_qp(rng)
    elif scenario == "naqp":
        exch_a, exch_b = exchange_naqp(rng), exchange_naqp(rng)
    elif scenario == "dx_contest":
        exch_a, exch_b = exchange_dx_zone(rng), exchange_dx_zone(rng)
    elif scenario == "pota":
        exch_a, exch_b = f"{random_rst(rng)} {random_rst(rng)}", f"{random_rst(rng)} {random_rst(rng)}"
    else:  # generic contest serial
        exch_a, exch_b = exchange_contest_serial(rng), exchange_contest_serial(rng)

    lines = [
        f"{b} DE {a} {a} K",
        f"{a} DE {b} {exch_b} {exch_b} K",
        f"R {exch_a} TU" + (f" {station_chatter(rng)}" if rng.random() < 0.25 else "") + f" {rng.choice(PROSIGNS_SIGNOFF)}",
    ]
    return " = ".join(lines) if rng.random() < 0.3 else " ".join(lines)


SCENARIO_WEIGHTS = {
    "sweepstakes": 18,
    "field_day": 15,
    "state_qp": 15,
    "naqp": 12,
    "dx_contest": 18,
    "pota": 12,
    "generic_contest": 10,
}


def generate_band(rng: random.Random, n_qsos: int) -> str:
    scenarios = list(SCENARIO_WEIGHTS.keys())
    weights = list(SCENARIO_WEIGHTS.values())
    lines = []
    for _ in range(n_qsos):
        scenario = rng.choices(scenarios, weights=weights)[0]
        call = random_callsign(rng)
        cq_word = "TEST" if scenario != "pota" else "POTA"
        lines.append(f"CQ {cq_word} DE {call} {call} {call} K")
        lines.append(generate_qso(rng, scenario))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qsos", type=int, default=2000)
    parser.add_argument("--out", default=str(DATA_ROOT / "text_corpus" / "qso_corpus.txt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    text = generate_band(rng, args.num_qsos)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {args.num_qsos} QSOs ({len(text)} chars) to {out_path}")
    print("\nSample:\n" + "\n".join(text.splitlines()[:8]))


if __name__ == "__main__":
    main()
