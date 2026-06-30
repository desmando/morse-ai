"""Train the CW acoustic decoder. Plain CLI script - run from a terminal on
whatever GPU machine you've got (e.g. the RTX 3080 box), or paste into a
notebook cell if you end up using one instead.

  python train.py --epochs 30

  --dry-run: runs a couple of CPU batches to smoke-test the pipeline (shapes,
  loss computation) without writing checkpoints. Use this to verify the script
  before handing it off to the training machine.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.decoder import (CWDecoder, MorseClipDataset, cer, collate_batch, ctc_greedy_decode,
                            load_manifest_rows, split_rows_by_source)
from model.vocab import Vocab
from paths import DATA_ROOT


def write_run_status(checkpoint_dir: Path, status: str, epoch: int, reason: str):
    """Writes a small structured status file an orchestrator can check
    reliably, instead of grep-parsing log text - status is one of
    "converged" (LR-floor-stop, the designed successful outcome),
    "diverged" (decode-check-threshold exceeded - needs human review, never
    auto-advance from this), "stopped_manually" (stop-file), or
    "epochs_reached" (hit --epochs, rare given the usual --epochs 100000)."""
    with open(checkpoint_dir / "RUN_STATUS.json", "w", encoding="utf-8") as f:
        json.dump({"status": status, "epoch": epoch, "reason": reason}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DATA_ROOT / "manifests" / "augmented_manifest.csv"))
    parser.add_argument("--vocab", default=str(DATA_ROOT / "manifests" / "vocab.txt"))
    parser.add_argument("--checkpoint-dir", default=str(DATA_ROOT / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                         help="Adam weight decay - a light regularizer against weights drifting into "
                              "the kind of unbounded, self-reinforcing configuration that produces "
                              "runaway repetition output over very long runs. 0 to disable")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                         help="linear LR warmup over this many batches, 0 to disable")
    parser.add_argument("--grad-clip", type=float, default=5.0,
                         help="max gradient norm (LSTMs are prone to exploding gradients), 0 to disable")
    parser.add_argument("--lr-decay-factor", type=float, default=0.5,
                         help="multiply LR by this when val_loss plateaus after warmup, 0 to disable")
    parser.add_argument("--lr-decay-patience", type=int, default=5,
                         help="epochs with no val_loss improvement before decaying LR")
    parser.add_argument("--lr-min", type=float, default=1e-5,
                         help="once LR decays to this floor and still plateaus, STOP training rather "
                              "than continue decaying - a real incident saw LR decay tenfold to ~3e-7 "
                              "over 60+ epochs of negligible updates while decode quality kept getting "
                              "worse anyway (likely BatchNorm running stats still drifting on mixed "
                              "clean/noisy batches even with near-zero gradient steps). 0 to disable")
    parser.add_argument("--decode-check-clips", type=int, default=200,
                         help="greedy-decode this many held-out clips each epoch and check actual CER, "
                              "not just loss - CTC loss can stay smooth while greedy-decode quality "
                              "silently breaks (a real incident: 60+ epochs of runaway repetition output "
                              "went undetected this way since loss never flagged it). 0 to disable")
    parser.add_argument("--decode-check-threshold", type=float, default=0.5,
                         help="stop training if the decode-check CER exceeds this (e.g. 0.5 = 50%%) - "
                             "checkpoint is still saved first, so the bad epoch is there to diagnose")
    parser.add_argument("--val-fraction", type=float, default=0.02,
                         help="fraction of source recordings (not rows) held out for validation, 0 to disable")
    parser.add_argument("--seed", type=int, default=0, help="train/val split seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume-from", default=None, help="checkpoint path to resume training from")
    parser.add_argument("--reset-optimizer", action="store_true",
                         help="with --resume-from, load only model weights - start optimizer/LR "
                              "schedulers/plateau-tracking fresh and restart epoch numbering at 1. "
                              "Use this when starting a meaningfully different training phase (e.g. "
                              "introducing noisy data after a clean-only basis phase): the old phase's "
                              "Adam momentum and near-zero converged LR aren't appropriate for adapting "
                              "to new data, and its val_loss 'best' isn't a fair baseline once the "
                              "validation set itself gets harder. Pair with a fresh --checkpoint-dir.")
    parser.add_argument("--stop-file", default=str(DATA_ROOT / "STOP_TRAINING"),
                         help="if this file exists, finish the current epoch, save its checkpoint, "
                              "delete the file, and exit cleanly - checked once per epoch, not mid-epoch, "
                              "so a run in progress is never interrupted partway through")
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
    rows = load_manifest_rows(args.manifest)
    train_rows, val_rows = split_rows_by_source(rows, 0 if args.dry_run else args.val_fraction, args.seed)
    dataset = MorseClipDataset(train_rows, vocab)
    print(f"dataset: {len(dataset)} train clips, {len(val_rows)} held-out val clips, "
          f"vocab size {len(vocab)} (incl. blank)")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=not args.dry_run,
        collate_fn=collate_batch, num_workers=0 if args.dry_run else 8,
    )
    val_loader = None
    if val_rows:
        val_loader = DataLoader(MorseClipDataset(val_rows, vocab), batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_batch)

    model = CWDecoder(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    def lr_lambda(step):
        return min(1.0, (step + 1) / args.warmup_steps) if args.warmup_steps > 0 else 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Plateau-based decay only matters post-warmup, and only if there's a
    # val set to measure plateaus against. LambdaLR.step() resets lr back
    # to base_lr * lr_lambda(step) on every call, which would silently
    # undo any reduction this makes - so once warmup finishes, scheduler
    # (LambdaLR) simply stops being stepped at all (see the training loop
    # below), leaving plateau_scheduler as the sole thing adjusting lr.
    plateau_scheduler = None
    if val_loader is not None and args.lr_decay_factor > 0:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_decay_factor, patience=args.lr_decay_patience,
            min_lr=args.lr_min)

    start_epoch = 1
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if args.reset_optimizer:
            print(f"resumed from {args.resume_from} (model weights only - optimizer, LR schedulers, "
                  f"and plateau tracking all reset fresh; epoch numbering restarts at 1)")
        else:
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            if plateau_scheduler is not None and ckpt.get("plateau_scheduler_state"):
                plateau_scheduler.load_state_dict(ckpt["plateau_scheduler_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"resumed from {args.resume_from}, continuing at epoch {start_epoch}")

    if args.dry_run:
        model.train()
        for step, (features, targets, input_lengths, target_lengths) in enumerate(loader):
            features, targets = features.to(device), targets.to(device)
            log_probs = model(features)  # (batch, T, vocab)
            log_probs_ctc = log_probs.permute(1, 0, 2)  # CTCLoss wants (T, batch, vocab)

            loss = ctc_loss(log_probs_ctc, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            preds = ctc_greedy_decode(log_probs.detach(), vocab)
            print(f"[dry-run step {step}] loss={loss.item():.3f} sample_pred={preds[0]!r}")
            if step >= 2:
                break
        print("dry-run OK: forward/backward/step all completed without error.")
        return

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Tracked independently of plateau_scheduler.num_bad_epochs, which resets
    # to 0 on every triggered reduction attempt - including a no-op one
    # clamped at min_lr. That means num_bad_epochs alone can never be
    # observed exceeding patience while stuck at the floor, so "stuck at the
    # floor with no improvement" needs its own counter.
    floor_best_val_loss = None
    floor_bad_epochs = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss, n_batches, n_skipped = 0.0, 0, 0

        for features, targets, input_lengths, target_lengths in loader:
            features, targets = features.to(device), targets.to(device)
            log_probs = model(features)
            log_probs_ctc = log_probs.permute(1, 0, 2)

            loss = ctc_loss(log_probs_ctc, targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                # A NaN/Inf loss corrupts Adam's running moment estimates for
                # many subsequent steps if allowed through backward() - skip
                # the step entirely rather than train on it. zero_infinity on
                # CTCLoss already zeroes plain-Inf losses, but NaN can still
                # slip through, hence this backstop.
                n_skipped += 1
                print(f"  WARNING: non-finite loss ({loss.item()}) at epoch {epoch}, "
                      f"batch {n_batches + n_skipped} - skipping")
                continue

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler.last_epoch < args.warmup_steps:
                scheduler.step()  # warmup only - see plateau_scheduler comment above

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - epoch_start

        val_msg = ""
        decode_cer = None
        if val_loader is not None:
            model.eval()
            val_loss, val_batches = 0.0, 0
            decode_cer_total, decode_n = 0.0, 0
            decode_examples = []
            with torch.no_grad():
                for features, targets, input_lengths, target_lengths in val_loader:
                    features, targets = features.to(device), targets.to(device)
                    log_probs = model(features)
                    log_probs_ctc = log_probs.permute(1, 0, 2)
                    loss = ctc_loss(log_probs_ctc, targets, input_lengths, target_lengths)
                    val_loss += loss.item()
                    val_batches += 1

                    # Spot-check actual decode quality on a small subset every
                    # epoch - loss alone missed a real incident where greedy-
                    # decode output silently degenerated into runaway
                    # repetition for 60+ epochs while loss stayed smooth.
                    if args.decode_check_clips > 0 and decode_n < args.decode_check_clips:
                        preds = ctc_greedy_decode(log_probs.cpu(), vocab)
                        offset = 0
                        for i, pred in enumerate(preds):
                            tlen = int(target_lengths[i])
                            ref = vocab.decode(targets[offset: offset + tlen].tolist())
                            offset += tlen
                            decode_cer_total += cer(pred, ref)
                            decode_n += 1
                            if len(decode_examples) < 3:
                                decode_examples.append((ref, pred))
            avg_val_loss = val_loss / max(val_batches, 1)
            val_msg = f"  val_loss={avg_val_loss:.3f}"
            if plateau_scheduler is not None:
                plateau_scheduler.step(avg_val_loss)
            if decode_n > 0:
                decode_cer = decode_cer_total / decode_n
                val_msg += f"  decode_cer={decode_cer:.3f}"

        skip_msg = f"  skipped={n_skipped}" if n_skipped else ""
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.3f}{val_msg}  "
              f"lr={cur_lr:.2e}{skip_msg}  ({elapsed:.1f}s)")

        ckpt_path = checkpoint_dir / f"decoder_epoch{epoch:03d}.pt"
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "plateau_scheduler_state": plateau_scheduler.state_dict() if plateau_scheduler else None,
                    "epoch": epoch, "vocab_chars": vocab.chars}, ckpt_path)

        if decode_cer is not None and args.decode_check_threshold > 0 and decode_cer > args.decode_check_threshold:
            print(f"  WARNING: decode CER {decode_cer:.3f} exceeds --decode-check-threshold "
                  f"{args.decode_check_threshold} at epoch {epoch} - decode quality has broken even "
                  f"though loss may look fine. Examples: {decode_examples}")
            print(f"Stopping after epoch {epoch} (checkpoint saved) - resume from an earlier checkpoint, "
                  f"not this one.")
            write_run_status(checkpoint_dir, "diverged", epoch, f"decode_cer {decode_cer:.3f} exceeded "
                              f"--decode-check-threshold {args.decode_check_threshold}")
            break

        if plateau_scheduler is not None and args.lr_min > 0 and cur_lr <= args.lr_min:
            if floor_best_val_loss is None or avg_val_loss < floor_best_val_loss:
                floor_best_val_loss = avg_val_loss
                floor_bad_epochs = 0
            else:
                floor_bad_epochs += 1
            if floor_bad_epochs > args.lr_decay_patience:
                print(f"LR has been at the floor ({args.lr_min:.1e}) for more than "
                      f"--lr-decay-patience ({args.lr_decay_patience}) epochs with no val_loss "
                      f"improvement - stopping after epoch {epoch} (checkpoint saved) rather than "
                      f"continuing to train at a negligible LR, where loss stops being a useful "
                      f"signal but other state (e.g. BatchNorm running stats) can still silently drift.")
                write_run_status(checkpoint_dir, "converged", epoch,
                                  f"LR at floor {args.lr_min:.1e} for more than {args.lr_decay_patience} "
                                  f"epochs with no val_loss improvement")
                break

        stop_file = Path(args.stop_file)
        if stop_file.exists():
            stop_file.unlink()
            print(f"{stop_file} found - stopping cleanly after epoch {epoch} "
                  f"(checkpoint saved, stop file removed)")
            write_run_status(checkpoint_dir, "stopped_manually", epoch, "stop-file found")
            break
    else:
        write_run_status(checkpoint_dir, "epochs_reached", args.epochs, f"reached --epochs {args.epochs}")

    print(f"Training complete. Checkpoints in {checkpoint_dir}")


if __name__ == "__main__":
    main()
