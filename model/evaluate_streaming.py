"""Evaluate a checkpoint on whole, continuous real-ARRL recordings instead of
small pre-chopped clips.

evaluate.py scores against clips_manifest.csv, whose labels were sliced from
the transcript by a uniform chars-per-second approximation (build_manifest.py)
- that routinely lands mid-word, so a clip's "label" can be a meaningless
fragment of the real transcript even when the model decoded its audio
correctly. This script sidesteps that entirely: it runs model.decode_stream
over each full, unchopped recording and compares the *whole* decoded
transcript to the *whole* real transcript by edit distance. That also
matches how the radio actually delivers audio (continuously, not in 4-second
pieces), so it's a more honest measure of real-world readiness than the
clip-based eval ever was. decode_stream tracks the CW tone across windows
with the same ToneTracker policy the live path uses, so this measures the
actual deployed front-end behavior.

The reference text is normalize_transcript(), NOT build_manifest.py's
clean_transcript() - clean_transcript() strips the announcer header/footer
("NOW XX WPM = TEXT IS FROM...") because that's the right training target,
but the header really is spoken in the audio, so scoring against a
reference that's had it removed would count a correct decode of it as pure
error - this measurably distorted results before being fixed (one file's
CER dropped from 13.0% to 2.2% once compared against the un-stripped
text), hitting short recordings hardest since the fixed-size header is a
bigger fraction of a short reference.

Decoding defaults to beam search + the ham character LM when the LM file
exists (--lm auto; --lm none for greedy).

Besides CER/WER, callsign-level metrics are reported: CER alone underweights
the operationally important error - K7RDG -> K7RDC is a tiny CER miss but a
completely wrong station. Every callsign-shaped token in the reference is
checked for an exact match in the prediction (else its closest edit
distance), and predicted callsign-shaped tokens are checked against the FCC
active-license index when it's built.

Usage:
  python model/evaluate_streaming.py --checkpoint <path> --max-files-per-speed 2
"""
import argparse
import re
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataprep.build_manifest import normalize_transcript
from model.decoder import cer, decode_stream, edit_distance, load_checkpoint_model, wer
from model.evaluate import resolve_lm
from paths import DATA_ROOT

# callsign-shaped tokens, generic enough for US + common DX patterns
CALLSIGN_RE = re.compile(r"^[A-Z0-9]{1,3}[0-9][A-Z]{1,4}$")


def find_pairs(raw_dir: Path, max_files_per_speed: int):
    pairs = []
    for speed_dir in sorted(raw_dir.iterdir()):
        if not speed_dir.is_dir():
            continue
        mp3s = sorted(speed_dir.glob("*.mp3"))
        if max_files_per_speed:
            mp3s = mp3s[:max_files_per_speed]
        for mp3_path in mp3s:
            txt_path = Path(re.sub(r"_(\d+)WPM\.mp3$", r"_\1.txt", str(mp3_path),
                                    flags=re.IGNORECASE))
            if txt_path.exists():
                pairs.append((mp3_path, txt_path))
    return pairs


def callsign_tokens(text: str) -> list[str]:
    return [t for t in text.split() if CALLSIGN_RE.match(t) and any(c.isdigit() for c in t)
            and any(c.isalpha() for c in t)]


def callsign_metrics(pred: str, ref: str, active: set | None):
    """Per-recording callsign accounting: (n_ref_calls, n_exact, sum_edit,
    n_pred_calls, n_pred_valid_fcc)."""
    ref_calls = callsign_tokens(ref)
    pred_tokens = pred.split()
    pred_calls = callsign_tokens(pred)
    n_exact, sum_edit = 0, 0
    for call in ref_calls:
        if call in pred_tokens:
            n_exact += 1
        elif pred_tokens:
            sum_edit += min(edit_distance(call, t) for t in pred_tokens)
        else:
            sum_edit += len(call)
    n_valid = sum(1 for c in pred_calls if c in active) if active else 0
    return len(ref_calls), n_exact, sum_edit, len(pred_calls), n_valid


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-dir", default=str(DATA_ROOT / "raw" / "arrl"))
    parser.add_argument("--vocab", default=str(DATA_ROOT / "manifests" / "vocab.txt"),
                         help="fallback only - the checkpoint's own recorded vocab is used when present")
    parser.add_argument("--max-files-per-speed", type=int, default=2,
                         help="0 = all files - real recordings are several minutes each, "
                              "so a full run over everything is slow; sample for quick checks")
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--stride-seconds", type=float, default=4.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lm", default="auto", metavar="PATH|auto|none",
                         help="'auto' (default): beam search + the default ham LM if it exists; "
                              "'none': greedy decode; or an explicit ham_char_lm.json path")
    parser.add_argument("--lm-weight", type=float, default=0.1,
                         help="LM score weight relative to acoustic score - 0.1 measured best via "
                              "evaluate.py --sweep on real checkpoints, 0.3 already measurably hurts")
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--length-bonus", type=float, default=0.0)
    parser.add_argument("--repeat-penalty", type=float, default=0.0)
    parser.add_argument("--fcc-rescore", action="store_true",
                         help="boost beam candidates whose callsigns are active FCC licenses "
                              "(needs the FCC index and beam search)")
    parser.add_argument("--squelch-db", type=float, default=0.0,
                         help="CW-activity gate per window (dB); 0 = off, the right default for "
                              "known-CW recordings - use ~6 to mimic the live path's squelch")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model, vocab, _ckpt = load_checkpoint_model(args.checkpoint, device, vocab_path=args.vocab)
    print(f"model config: {model.config}  vocab size {len(vocab)} (incl. blank)")

    lm, lm_desc = resolve_lm(args.lm)
    print(f"decoding: {lm_desc} (weight={args.lm_weight}, beam={args.beam_width})" if lm
          else f"decoding: {lm_desc}")

    from inference.fcc_uls import load_active_callsigns
    active = load_active_callsigns()
    final_rescore = None
    if args.fcc_rescore and lm is not None and active:
        from inference.realtime_decode import make_fcc_rescorer
        final_rescore = make_fcc_rescorer()
    if not active:
        print("note: FCC index not built - valid-FCC callsign rate unavailable "
              "(python inference/fcc_uls.py --download)")

    pairs = find_pairs(Path(args.raw_dir), args.max_files_per_speed)
    print(f"evaluating {len(pairs)} whole recordings\n")

    total_cer, total_wer, n = 0.0, 0.0, 0
    cs_ref = cs_exact = cs_edit = cs_pred = cs_valid = 0
    for mp3_path, txt_path in pairs:
        audio, sr = sf.read(mp3_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        ref = normalize_transcript(txt_path.read_text(encoding="utf-8", errors="replace")).upper()
        if not ref:
            continue

        pred = decode_stream(audio, sr, model, vocab, device,
                              window_seconds=args.window_seconds, stride_seconds=args.stride_seconds,
                              lm=lm, lm_weight=args.lm_weight, beam_width=args.beam_width,
                              final_rescore=final_rescore, squelch_db=args.squelch_db,
                              length_bonus=args.length_bonus,
                              repeat_penalty=args.repeat_penalty).upper()
        c, w = cer(pred, ref), wer(pred, ref)
        total_cer += c
        total_wer += w
        n += 1
        nr, ne, se, np_, nv = callsign_metrics(pred, ref, active if active else None)
        cs_ref += nr; cs_exact += ne; cs_edit += se; cs_pred += np_; cs_valid += nv
        print(f"{mp3_path.name}: CER {c:.4f}  WER {w:.4f}  callsigns {ne}/{nr} exact"
              f"  ({len(audio)/sr:.0f}s audio, {len(ref)} ref chars)")

    print(f"\nfiles evaluated: {n}")
    print(f"avg CER: {total_cer / max(n, 1):.4f}")
    print(f"avg WER: {total_wer / max(n, 1):.4f}")
    if cs_ref:
        print(f"callsign exact match: {cs_exact}/{cs_ref} ({cs_exact / cs_ref:.1%})")
        n_missed = cs_ref - cs_exact
        if n_missed:
            print(f"callsign mean edit distance (missed ones): {cs_edit / n_missed:.2f}")
    if active and cs_pred:
        print(f"predicted callsigns valid in FCC index: {cs_valid}/{cs_pred} "
              f"({cs_valid / cs_pred:.1%})")


if __name__ == "__main__":
    main()
