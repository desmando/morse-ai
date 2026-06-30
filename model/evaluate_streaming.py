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
clip-based eval ever was.

Usage:
  python model/evaluate_streaming.py --checkpoint <path> --max-files-per-speed 2
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataprep.build_manifest import clean_transcript
from model.decoder import CWDecoder, cer, decode_stream, wer
from model.vocab import Vocab
from paths import DATA_ROOT


def find_pairs(raw_dir: Path, max_files_per_speed: int):
    pairs = []
    for speed_dir in sorted(raw_dir.iterdir()):
        if not speed_dir.is_dir():
            continue
        mp3s = sorted(speed_dir.glob("*.mp3"))
        if max_files_per_speed:
            mp3s = mp3s[:max_files_per_speed]
        for mp3_path in mp3s:
            txt_path = Path(__import__("re").sub(r"_(\d+)WPM\.mp3$", r"_\1.txt", str(mp3_path),
                                                   flags=__import__("re").IGNORECASE))
            if txt_path.exists():
                pairs.append((mp3_path, txt_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-dir", default=str(DATA_ROOT / "raw" / "arrl"))
    parser.add_argument("--vocab", default=str(DATA_ROOT / "manifests" / "vocab.txt"))
    parser.add_argument("--max-files-per-speed", type=int, default=2,
                         help="0 = all files - real recordings are several minutes each, "
                              "so a full run over everything is slow; sample for quick checks")
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--stride-seconds", type=float, default=4.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lm", default=None, metavar="PATH",
                         help="path to ham_char_lm.json - enables CTC beam search instead of greedy decode")
    parser.add_argument("--lm-weight", type=float, default=0.3,
                         help="LM score weight relative to acoustic score (0 = acoustic only)")
    parser.add_argument("--beam-width", type=int, default=20)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    vocab = Vocab.from_file(args.vocab)
    model = CWDecoder(vocab_size=len(vocab)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    lm = None
    if args.lm:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from lm.ngram_lm import CharNgramLM
        lm = CharNgramLM.load(args.lm)
        print(f"LM: {args.lm} (weight={args.lm_weight}, beam={args.beam_width})")

    pairs = find_pairs(Path(args.raw_dir), args.max_files_per_speed)
    print(f"evaluating {len(pairs)} whole recordings\n")

    total_cer, total_wer, n = 0.0, 0.0, 0
    for mp3_path, txt_path in pairs:
        audio, sr = sf.read(mp3_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        ref = clean_transcript(txt_path.read_text(encoding="utf-8", errors="replace")).upper()
        if not ref:
            continue

        pred = decode_stream(audio, sr, model, vocab, device,
                              window_seconds=args.window_seconds, stride_seconds=args.stride_seconds,
                              lm=lm, lm_weight=args.lm_weight, beam_width=args.beam_width).upper()
        c, w = cer(pred, ref), wer(pred, ref)
        total_cer += c
        total_wer += w
        n += 1
        print(f"{mp3_path.name}: CER {c:.4f}  WER {w:.4f}  ({len(audio)/sr:.0f}s audio, {len(ref)} ref chars)")

    print(f"\nfiles evaluated: {n}")
    print(f"avg CER: {total_cer / max(n, 1):.4f}")
    print(f"avg WER: {total_wer / max(n, 1):.4f}")


if __name__ == "__main__":
    main()
