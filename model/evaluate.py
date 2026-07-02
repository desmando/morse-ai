"""Evaluate a trained checkpoint's decode quality (CER/WER) on a held-out split.

Held-out clips are selected the same way train.py does (by source recording,
not by row, with the same --val-fraction/--seed), so augmented variants of one
clip never appear in both a training run and this held-out set - as long as
the checkpoint was trained with the same --val-fraction/--seed. If you're
evaluating a checkpoint trained on the *whole* manifest (--val-fraction 0),
this measures in-sample fit, not generalization.

The model architecture, feature parameters, and vocab are reconstructed from
the checkpoint itself - no flags needed to match them.

Decoding defaults to CTC beam search with the ham character LM when the LM
file exists (--lm auto); greedy CTC leaves accuracy on the table for ham
traffic, so greedy is now the explicit opt-out (--lm none), not the default.

--augment applies the same FIXED per-clip impairments train.py --augment uses
for its val split, so numbers here are comparable to the training log's
val_loss/decode_cer.

--sweep grid-searches lm_weight x beam_width over the cached model outputs
(one forward pass total - only the decode is repeated), printing a CER/WER
table. Use it to pick decoding parameters instead of trusting defaults:

  python model/evaluate.py --checkpoint <best.pt> --augment --sweep \
      --sweep-lm-weights 0,0.1,0.2,0.3,0.5,0.8 --sweep-beam-widths 10,20,50

Usage:
  python evaluate.py --checkpoint <path to decoder_epochNNN.pt> --max-clips 500
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.decoder import (MorseClipDataset, cer, collate_batch, ctc_beam_decode, ctc_greedy_decode,
                            load_checkpoint_model, load_manifest_rows, split_rows_by_source, wer)
from paths import DATA_ROOT


def resolve_lm(lm_arg: str):
    """'auto' = use the default LM if built, 'none' = greedy, else a path."""
    from lm.ngram_lm import DEFAULT_LM_PATH, CharNgramLM
    if lm_arg == "none":
        return None, "greedy (LM disabled)"
    if lm_arg == "auto":
        if Path(DEFAULT_LM_PATH).exists():
            return CharNgramLM.load(DEFAULT_LM_PATH), f"beam+LM ({DEFAULT_LM_PATH})"
        return None, f"greedy (no LM at {DEFAULT_LM_PATH} - build with: python lm/ngram_lm.py --build)"
    return CharNgramLM.load(lm_arg), f"beam+LM ({lm_arg})"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=str(DATA_ROOT / "manifests" / "augmented_manifest.csv"))
    parser.add_argument("--vocab", default=str(DATA_ROOT / "manifests" / "vocab.txt"),
                         help="fallback only - the checkpoint's own recorded vocab is used when present")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=500, help="cap held-out clips evaluated, 0 = all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--augment", action="store_true",
                         help="apply the fixed (per-clip-seeded, full-strength) impairments train.py "
                              "--augment uses for validation - evaluate noisy-conditions performance "
                              "on a clean manifest")
    parser.add_argument("--snr-db-range", default="-3,20",
                         help="SNR range for --augment (match the training run's)")
    parser.add_argument("--lm", default="auto", metavar="PATH|auto|none",
                         help="'auto' (default): beam search + the default ham LM if it exists; "
                              "'none': greedy decode; or an explicit ham_char_lm.json path")
    parser.add_argument("--lm-weight", type=float, default=0.3)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--length-bonus", type=float, default=0.0,
                         help="beam score bonus per output character (sweep before trusting)")
    parser.add_argument("--repeat-penalty", type=float, default=0.0,
                         help="beam score penalty per repeated-run character (sweep before trusting)")
    parser.add_argument("--sweep", action="store_true",
                         help="grid-search lm_weight x beam_width over cached model outputs and print "
                              "a CER/WER table (requires an LM)")
    parser.add_argument("--sweep-lm-weights", default="0,0.1,0.2,0.3,0.5,0.8")
    parser.add_argument("--sweep-beam-widths", default="10,20,50")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model, vocab, _ckpt = load_checkpoint_model(args.checkpoint, device, vocab_path=args.vocab)
    print(f"model config: {model.config}  vocab size {len(vocab)} (incl. blank)")

    lm, lm_desc = resolve_lm(args.lm)
    print(f"decoding: {lm_desc}")
    if args.sweep and lm is None:
        parser.error("--sweep needs an LM (build one or pass --lm <path>)")

    rows = load_manifest_rows(args.manifest)
    _, val_rows = split_rows_by_source(rows, args.val_fraction, args.seed)
    if args.max_clips:
        val_rows = val_rows[: args.max_clips]
    print(f"held-out clips: {len(val_rows)}" + ("  (fixed augmentation ON)" if args.augment else ""))

    snr_range = tuple(float(x) for x in args.snr_db_range.split(","))
    dataset = MorseClipDataset(val_rows, vocab, hop_samples=model.hop_samples,
                                augment="fixed" if args.augment else None,
                                snr_db_range=snr_range, aug_seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # One forward pass, cached per-clip lattices (trimmed to valid frames) -
    # decoding variants (greedy/beam/sweep points) all reuse them.
    cached = []
    with torch.no_grad():
        for features, targets, input_lengths, target_lengths in loader:
            log_probs = model(features.to(device), input_lengths).cpu()
            ctc_lengths = model.downsampled_lengths(input_lengths)
            offset = 0
            for i in range(log_probs.shape[0]):
                tlen = int(target_lengths[i])
                ref = vocab.decode(targets[offset: offset + tlen].tolist())
                offset += tlen
                cached.append((log_probs[i, : int(ctc_lengths[i])].clone(), ref))

    def run_decode(lm_, lm_weight, beam_width, length_bonus, repeat_penalty):
        total_cer, total_wer = 0.0, 0.0
        examples = []
        for lp, ref in cached:
            if lm_ is None:
                pred = ctc_greedy_decode(lp.unsqueeze(0), vocab)[0]
            else:
                pred = ctc_beam_decode(lp, vocab, lm=lm_, lm_weight=lm_weight,
                                        beam_width=beam_width, length_bonus=length_bonus,
                                        repeat_penalty=repeat_penalty)
            total_cer += cer(pred, ref)
            total_wer += wer(pred, ref)
            if len(examples) < 5:
                examples.append((ref, pred))
        n = max(len(cached), 1)
        return total_cer / n, total_wer / n, examples

    if args.sweep:
        weights = [float(x) for x in args.sweep_lm_weights.split(",")]
        widths = [int(x) for x in args.sweep_beam_widths.split(",")]
        print(f"\nsweep over {len(cached)} clips ({len(weights)}x{len(widths)} points):")
        print(f"{'lm_weight':>10} {'beam':>6} {'CER':>8} {'WER':>8}")
        results = []
        for w in weights:
            for bw in widths:
                c, wv, _ = run_decode(lm if w > 0 else None, w, bw,
                                       args.length_bonus, args.repeat_penalty)
                results.append((c, wv, w, bw))
                print(f"{w:>10.2f} {bw:>6d} {c:>8.4f} {wv:>8.4f}", flush=True)
        best = min(results)
        print(f"\nbest: lm_weight={best[2]} beam_width={best[3]} -> CER {best[0]:.4f}, WER {best[1]:.4f}")
        return

    avg_cer, avg_wer, examples = run_decode(lm, args.lm_weight, args.beam_width,
                                             args.length_bonus, args.repeat_penalty)
    print(f"clips evaluated: {len(cached)}")
    print(f"avg CER: {avg_cer:.4f}")
    print(f"avg WER: {avg_wer:.4f}")
    print("\nsample predictions (ref -> pred):")
    for ref, pred in examples:
        print(f"  {ref!r} -> {pred!r}")


if __name__ == "__main__":
    main()
