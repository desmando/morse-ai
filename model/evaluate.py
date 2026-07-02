"""Evaluate a trained checkpoint's decode quality (CER/WER) on a held-out split.

Held-out clips are selected the same way train.py does (by source recording,
not by row, with the same --val-fraction/--seed), so augmented variants of one
clip never appear in both a training run and this held-out set - as long as
the checkpoint was trained with the same --val-fraction/--seed. If you're
evaluating a checkpoint trained on the *whole* manifest (--val-fraction 0),
this measures in-sample fit, not generalization.

The model architecture and feature parameters are reconstructed from the
checkpoint's recorded model_config (legacy checkpoints get the legacy recipe
automatically) - no flags needed to match them.

--augment applies the same FIXED per-clip impairments train.py --augment uses
for its val split, so numbers here are comparable to the training log's
val_loss/decode_cer. --lm enables CTC beam search with the ham character LM
instead of greedy decoding, to measure what the LM actually buys end-to-end.

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
from model.vocab import Vocab
from paths import DATA_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=str(DATA_ROOT / "manifests" / "augmented_manifest.csv"))
    parser.add_argument("--vocab", default=str(DATA_ROOT / "manifests" / "vocab.txt"))
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
    parser.add_argument("--lm", default=None, metavar="PATH",
                         help="path to ham_char_lm.json - CTC beam search + LM instead of greedy")
    parser.add_argument("--lm-weight", type=float, default=0.3)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model, vocab, _ckpt = load_checkpoint_model(args.checkpoint, device, vocab_path=args.vocab)
    print(f"model config: {model.config}  vocab size {len(vocab)} (incl. blank)")

    lm = None
    if args.lm:
        from lm.ngram_lm import CharNgramLM
        lm = CharNgramLM.load(args.lm)
        print(f"LM: {args.lm} (weight={args.lm_weight}, beam={args.beam_width})")

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

    total_cer, total_wer, n = 0.0, 0.0, 0
    examples = []
    with torch.no_grad():
        for features, targets, input_lengths, target_lengths in loader:
            features = features.to(device)
            log_probs = model(features, input_lengths)
            ctc_lengths = model.downsampled_lengths(input_lengths)
            log_probs_cpu = log_probs.cpu()

            if lm is not None:
                preds = [ctc_beam_decode(log_probs_cpu[i, : int(ctc_lengths[i])], vocab, lm=lm,
                                          lm_weight=args.lm_weight, beam_width=args.beam_width)
                         for i in range(log_probs_cpu.shape[0])]
            else:
                preds = ctc_greedy_decode(log_probs_cpu, vocab, lengths=ctc_lengths)

            offset = 0
            for i, pred in enumerate(preds):
                tlen = int(target_lengths[i])
                ref = vocab.decode(targets[offset: offset + tlen].tolist())
                offset += tlen
                total_cer += cer(pred, ref)
                total_wer += wer(pred, ref)
                n += 1
                if len(examples) < 5:
                    examples.append((ref, pred))

    print(f"clips evaluated: {n}")
    print(f"avg CER: {total_cer / max(n, 1):.4f}")
    print(f"avg WER: {total_wer / max(n, 1):.4f}")
    print("\nsample predictions (ref -> pred):")
    for ref, pred in examples:
        print(f"  {ref!r} -> {pred!r}")


if __name__ == "__main__":
    main()
