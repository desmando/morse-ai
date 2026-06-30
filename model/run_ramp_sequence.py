"""Drives the noise-ramp curriculum sequence automatically: launches each
phase's train.py run, waits for it to exit, checks the structured
RUN_STATUS.json it writes (see train.py's write_run_status), and only
advances to the next phase if that status is "converged" - any other
status (diverged, stopped_manually, a crash, or a missing status file)
halts the sequence for human review rather than blindly continuing. A
diverged run is never a reason to advance; that needs a person to look at
the checkpoint(s) and decide what happened.

Also maintains a small persistent state file (ramp_sequence_state.json)
recording which phase/checkpoint is currently being attempted and, on a
halt, WHY - distinguishing "crashed_no_status" (an infrastructure failure
with no model-quality judgment involved - a real incident saw a checkpoint
directory vanish mid-run for no code-level reason) from "diverged"/
"stopped_manually" (a deliberate halt that needs human review, never
safe to retry blindly). This is what a cron-based relaunch guard
(ensure_ramp_sequence_running.sh) checks before deciding whether to
auto-relaunch - only the infra-crash case should ever auto-retry.

Cross-platform by construction: all paths go through pathlib/DATA_ROOT (no
hardcoded separators or absolute Linux paths), and train.py is launched via
subprocess with an argument list rather than a shell string, so this runs
identically on the Linux cloud box or for local Windows testing/dry-runs.

Each completed phase picks its best checkpoint by re-evaluating the last
few (not just trusting "latest" - a real incident showed the latest
checkpoint in a converged run can be a noise blip rather than the actual
best one), then resumes the next phase from that checkpoint with
--reset-optimizer, into its own fresh --checkpoint-dir, and archives the
completed phase's checkpoint dir.

Usage:
  python model/run_ramp_sequence.py --start-phase 10pct \\
      --start-checkpoint /root/morse-ai-data/checkpoints_ramp_05pct/decoder_epoch062.pt
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA_ROOT

NOISE_RAMP_DIR = DATA_ROOT / "manifests" / "noise_ramp"
STATE_PATH = DATA_ROOT / "ramp_sequence_state.json"

# Ordered curriculum: (phase label, manifest path). "combined" is the final
# step (~75% noisy, all available augmented clips) - combined_manifest.csv
# already exists, no separate ramp file needed for it.
PHASES = [
    ("05pct", NOISE_RAMP_DIR / "ramp_05pct.csv"),
    ("10pct", NOISE_RAMP_DIR / "ramp_10pct.csv"),
    ("20pct", NOISE_RAMP_DIR / "ramp_20pct.csv"),
    ("35pct", NOISE_RAMP_DIR / "ramp_35pct.csv"),
    ("50pct", NOISE_RAMP_DIR / "ramp_50pct.csv"),
    # 65pct was added after combined_manifest.csv (~75% noisy) twice diverged
    # resuming directly from 50pct - see CLOUD_TRAINING.md.
    ("65pct", NOISE_RAMP_DIR / "ramp_65pct.csv"),
    ("combined", DATA_ROOT / "manifests" / "combined_manifest.csv"),
]

TRAIN_ARGS = [
    "--epochs", "100000", "--batch-size", "64", "--lr", "3e-4", "--warmup-steps", "1000",
    "--grad-clip", "5.0", "--weight-decay", "1e-5", "--lr-decay-factor", "0.5",
    "--lr-decay-patience", "5", "--lr-min", "1e-5", "--decode-check-clips", "200",
    "--decode-check-threshold", "0.5",
]

_CER_RE = re.compile(r"avg CER:\s*([\d.]+)")


class PhaseCrashed(RuntimeError):
    """No RUN_STATUS.json at all - the phase never reached a normal exit
    point (process killed, infra failure, etc.), not a model-quality
    judgment call. Safe for a relaunch guard to auto-retry."""


class PhaseNotConverged(RuntimeError):
    """RUN_STATUS.json exists but status != "converged" (diverged,
    stopped_manually, epochs_reached) - a deliberate or quality-related
    halt. Never safe to auto-retry without a person looking at it first."""


def write_state(phase: str, resume_checkpoint, halted_reason: str | None = None, complete: bool = False):
    """Persists what the sequence is currently doing/where it stopped, so a
    separate process (the cron relaunch guard) can decide whether it's safe
    to auto-relaunch without re-deriving this from log text."""
    STATE_PATH.write_text(json.dumps({
        "phase": phase,
        "resume_checkpoint": str(resume_checkpoint),
        "halted_reason": halted_reason,
        "complete": complete,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")


def pick_best_checkpoint(checkpoint_dir: Path, manifest: Path, device: str = "auto",
                          last_n: int = 5) -> tuple[Path, float]:
    """Re-evaluates the last `last_n` checkpoints in checkpoint_dir against
    manifest and returns (path, cer) for whichever scores lowest - "latest"
    isn't always best even in a converged run (a single noise-blip epoch
    can land last)."""
    ckpts = sorted(checkpoint_dir.glob("decoder_epoch*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")
    candidates = ckpts[-last_n:]

    best_path, best_cer = None, float("inf")
    for ckpt in candidates:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "evaluate.py"),
             "--manifest", str(manifest), "--checkpoint", str(ckpt),
             "--max-clips", "2000", "--device", device],
            capture_output=True, text=True, check=True,
        )
        m = _CER_RE.search(result.stdout)
        if not m:
            print(f"  warning: couldn't parse CER from evaluate.py output for {ckpt.name}, skipping")
            continue
        cer = float(m.group(1))
        print(f"  {ckpt.name}: CER {cer:.4f}")
        if cer < best_cer:
            best_path, best_cer = ckpt, cer

    if best_path is None:
        raise RuntimeError(f"failed to evaluate any candidate checkpoint in {checkpoint_dir}")
    return best_path, best_cer


def run_phase(label: str, manifest: Path, resume_from: Path) -> Path:
    """Launches train.py for one phase, blocks until it exits, and returns
    its checkpoint directory. Raises if the phase doesn't end in a status
    that's safe to advance from."""
    checkpoint_dir = DATA_ROOT / f"checkpoints_ramp_{label}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = DATA_ROOT / f"train_ramp_{label}.log"

    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "train.py"),
        "--manifest", str(manifest),
        "--checkpoint-dir", str(checkpoint_dir),
        "--resume-from", str(resume_from),
        "--reset-optimizer",
    ] + TRAIN_ARGS

    print(f"\n=== phase {label}: launching, resuming from {resume_from} ===")
    print(" ".join(cmd))
    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)

    status_path = checkpoint_dir / "RUN_STATUS.json"
    if not status_path.exists():
        raise PhaseCrashed(f"phase {label} exited with no RUN_STATUS.json in {checkpoint_dir} - "
                            f"likely crashed before reaching a normal exit point (infra-level, not a "
                            f"model-quality issue). Check {log_path}.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    print(f"phase {label} finished: {status}")
    if status["status"] != "converged":
        raise PhaseNotConverged(f"phase {label} did not converge cleanly (status={status['status']!r}, "
                                 f"reason={status['reason']!r}) - stopping the sequence for review rather "
                                 f"than advancing. Check {log_path} and {checkpoint_dir}.")
    return checkpoint_dir


def archive_checkpoint_dir(checkpoint_dir: Path, label: str):
    archived = checkpoint_dir.parent / f"checkpoints_archived_{label}_{date.today():%Y%m%d}"
    shutil.move(str(checkpoint_dir), str(archived))
    print(f"archived {checkpoint_dir.name} -> {archived.name}")
    return archived


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-phase", required=True, choices=[p[0] for p in PHASES],
                         help="first phase to run - typically the one after whatever already converged")
    parser.add_argument("--start-checkpoint", required=True, type=Path,
                         help="checkpoint to resume --start-phase from (the previous phase's best)")
    args = parser.parse_args()

    start_idx = next(i for i, (label, _) in enumerate(PHASES) if label == args.start_phase)
    resume_from = args.start_checkpoint

    for label, manifest in PHASES[start_idx:]:
        write_state(label, resume_from)  # record what we're attempting BEFORE attempting it, so a
        try:                             # crash mid-attempt still leaves a correct resume point behind
            checkpoint_dir = run_phase(label, manifest, resume_from)
        except PhaseCrashed as exc:
            write_state(label, resume_from, halted_reason="crashed_no_status")
            raise
        except PhaseNotConverged as exc:
            write_state(label, resume_from, halted_reason="not_converged")
            raise
        best_ckpt, best_cer = pick_best_checkpoint(checkpoint_dir, manifest)
        print(f"phase {label} best checkpoint: {best_ckpt.name} (CER {best_cer:.4f})")
        archived_dir = archive_checkpoint_dir(checkpoint_dir, label)
        resume_from = archived_dir / best_ckpt.name

    write_state(PHASES[-1][0], resume_from, complete=True)
    print("\nRamp sequence complete - all phases converged.")


if __name__ == "__main__":
    main()
