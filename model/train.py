"""Train the CW acoustic decoder. Plain CLI script - run from a terminal on
whatever GPU machine you've got (e.g. the RTX 3080 box), or paste into a
notebook cell if you end up using one instead.

  python train.py --manifest ../data/manifests/augmented_manifest.csv --epochs 30

  --dry-run: runs a couple of CPU batches to smoke-test the pipeline (shapes,
  loss computation) without writing checkpoints. Use this to verify the script
  before handing it off to the training machine.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.decoder import CWDecoder, MorseClipDataset, collate_batch, ctc_greedy_decode
from model.vocab import Vocab

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "manifests" / "augmented_manifest.csv"))
    parser.add_argument("--vocab", default=str(REPO_ROOT / "data" / "manifests" / "vocab.txt"))
    parser.add_argument("--checkpoint-dir", default=str(REPO_ROOT / "model" / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true",
                         help="run a couple of batches on CPU to smoke-test, then exit without saving")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dry_run:
        device = "cpu"
    print(f"device: {device}")

    vocab = Vocab.from_file(args.vocab)
    dataset = MorseClipDataset(args.manifest, vocab)
    print(f"dataset: {len(dataset)} clips, vocab size {len(vocab)} (incl. blank)")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=not args.dry_run,
        collate_fn=collate_batch, num_workers=0,
    )

    model = CWDecoder(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    if args.dry_run:
        model.train()
        for step, (features, targets, input_lengths, target_lengths) in enumerate(loader):
            features, targets = features.to(device), targets.to(device)
            log_probs = model(features)  # (batch, T, vocab)
            log_probs_ctc = log_probs.permute(1, 0, 2)  # CTCLoss wants (T, batch, vocab)

            loss = ctc_loss(log_probs_ctc, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = ctc_greedy_decode(log_probs.detach(), vocab)
            print(f"[dry-run step {step}] loss={loss.item():.3f} sample_pred={preds[0]!r}")
            if step >= 2:
                break
        print("dry-run OK: forward/backward/step all completed without error.")
        return

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss, n_batches = 0.0, 0

        for features, targets, input_lengths, target_lengths in loader:
            features, targets = features.to(device), targets.to(device)
            log_probs = model(features)
            log_probs_ctc = log_probs.permute(1, 0, 2)

            loss = ctc_loss(log_probs_ctc, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - epoch_start
        print(f"epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.3f}  ({elapsed:.1f}s)")

        ckpt_path = checkpoint_dir / f"decoder_epoch{epoch:03d}.pt"
        torch.save({"model_state": model.state_dict(), "vocab_chars": vocab.chars}, ckpt_path)

    print(f"Training complete. Checkpoints in {checkpoint_dir}")


if __name__ == "__main__":
    main()
