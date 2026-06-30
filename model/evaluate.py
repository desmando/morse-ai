"""Evaluate a trained checkpoint's decode quality (CER/WER) on a held-out split.

Held-out clips are selected the same way train.py does (by source recording,
not by row, with the same --val-fraction/--seed), so augmented variants of one
clip never appear in both a training run and this held-out set - as long as
the checkpoint was trained with the same --val-fraction/--seed. If you're
evaluating a checkpoint trained on the *whole* manifest (--val-fraction 0),
this measures in-sample fit, not generalization.

Usage:
  python evaluate.py --checkpoint <path to decoder_epochNNN.pt> --max-clips 500
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.decoder import (CWDecoder, MorseClipDataset, cer, collate_batch, ctc_greedy_decode,
                            load_manifest_rows, split_rows_by_source, wer)
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    vocab = Vocab.from_file(args.vocab)
    rows = load_manifest_rows(args.manifest)
    _, val_rows = split_rows_by_source(rows, args.val_fraction, args.seed)
    if args.max_clips:
        val_rows = val_rows[: args.max_clips]
    print(f"held-out clips: {len(val_rows)}")

    dataset = MorseClipDataset(val_rows, vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = CWDecoder(vocab_size=len(vocab)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total_cer, total_wer, n = 0.0, 0.0, 0
    examples = []
    with torch.no_grad():
        for features, targets, input_lengths, target_lengths in loader:
            features = features.to(device)
            log_probs = model(features)
            preds = ctc_greedy_decode(log_probs.cpu(), vocab)

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
