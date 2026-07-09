# Training on a rented Ubuntu GPU

> **v4 recipe (current)** - supersedes the noise-ramp phase curriculum
> described in the "Context" section below. One run via
> `start_v4_training.sh`:
>
> - **On-the-fly augmentation** (`train.py --augment`): HF impairments
>   (noise, QSB, QRM, QRN, drift) are applied per clip in the dataloader -
>   fresh noise every epoch, annealed smoothly from clean to full strength
>   over `--aug-anneal-epochs`. No pre-rendered augmented manifests, no ramp
>   files, no phase resumes - and no abrupt distribution jumps, which is what
>   collapsed training twice at the 50pct->combined transition.
> - **Model config travels in the checkpoint** (`model_config`): the v4
>   architecture (feature hop 56 + CNN time-stride 2, LSTM dropout 0.2,
>   packed padded sequences) is reconstructed automatically by every
>   eval/inference script via `load_checkpoint_model()`. Legacy checkpoints
>   still load as the legacy architecture.
> - **Mixed precision** (AMP) is on by default on CUDA (`--no-amp` to disable).
> - **Synthesis realism** (regenerate clips before training): per-sender
>   keying style, Farnsworth spacing (with a realistic character-speed
>   floor - real slow-speed practice keys characters at a normal ~13-25 WPM
>   pace with stretched gaps, not everything uniformly slowed), **5-40 WPM**
>   (was 10-40 - real ARRL audio at 5 WPM was a near-total decode failure
>   until this widened, since the model had simply never heard anything
>   that slow), prosigns (`<AR>`/`<SK>`/`<KN>`/`<BT>`/`<AS>`), randomized
>   rise time, and a `tone_hz` manifest column so training never mis-detects
>   the tone under heavy augmentation.
> - **Vocab**: extend vocab.txt with `<` and `>` before training (49 chars
>   + blank; see start_v4_training.sh prerequisites) or prosign labels get
>   silently stripped from training targets. Safe for old checkpoints: every
>   checkpoint records its own vocab_chars and all loaders use those, so the
>   file only matters for NEW training runs. Rebuild the character LM
>   (`python lm/ngram_lm.py --build`) after regenerating the corpus.
> - **Decode params**: `--lm-weight 0.1` (not the old 0.3 default) measured
>   best via `evaluate.py --sweep` on real checkpoints - re-sweep on
>   anything you train yourself, this isn't guaranteed to transfer.
> - **Real-data path**: once a checkpoint transfers to real ARRL audio at
>   all, run `dataprep/realign_arrl_labels.py --checkpoint <best>` to
>   force-align the real recordings' transcripts into exactly-labeled clips
>   (fixes build_manifest.py's proportional-slicing label corruption), then
>   fine-tune on synthetic + realigned-real combined.
> - **`evaluate_streaming.py` used to score against a header-stripped
>   reference - fixed, but know the history if you see old numbers cited
>   anywhere (including earlier in this file, or the v4.1.0 release
>   notes).** `build_manifest.py`'s `clean_transcript()` strips the "NOW XX
>   WPM = TEXT IS FROM..." announcer header/footer from the reference text
>   (correct for training targets), but that header is genuinely spoken in
>   the audio and a working model decodes it - the old code compared
>   against the *stripped* reference, so real correct decoding of real
>   content got counted as pure error. Hit short files hardest (the
>   fixed-size header is a bigger fraction of a short reference) - one file
>   went from a reported 13.0% CER down to a true 2.2% once fixed. Now uses
>   `normalize_transcript()` (control-char/whitespace cleanup only, no
>   header/footer stripping) instead - see the "v4: on-the-fly
>   augmentation..." section below for the corrected per-speed numbers this
>   produced across the full real-ARRL eval set.

This project's acoustic model (`model/train.py`) is GPU-compute-bound, not
VRAM-bound — the LSTM's sequential nature keeps the GPU at ~100% utilization
even though it barely touches VRAM. So when scaling up to a long (1000+
epoch) run, prioritize raw/clock compute over VRAM capacity: a single RTX
4090 is likely the best cost/performance fit, not an A100/H100.

## 1. Pick an instance

Any rental provider's "PyTorch" or "CUDA" pre-built Ubuntu image saves you
from installing NVIDIA drivers/CUDA manually.

## 2. First login - verify the GPU

```bash
nvidia-smi          # confirm the GPU and driver are visible
python3 --version   # most images ship 3.10/3.11/3.12
```

## 3. Get the code onto the instance

No git remote exists for this repo yet, so push it up directly with rsync
(from Windows, via WSL/Git Bash's OpenSSH, or any scp-capable tool):

```bash
rsync -avz --exclude '.venv' --exclude '__pycache__' \
  "/path/to/morse-ai/" user@<instance-ip>:~/morse-ai/
```

## 4. Get the data onto the instance

Only ship what training actually reads - check `C:\morse-ai-data\manifests\`
for whichever manifest you're training on, and ship only the subfolders its
clip_path entries actually point into. As of the current fully-synthetic
recipe, `combined_manifest.csv` only references `synthetic/` and
`augmented_synthetic/` - the real ARRL/on6zq folders (`processed/`,
`augmented/`) aren't used by it at all. Skip `raw/` entirely either way -
manifests never reference it, it's only needed to regenerate clips, which
you won't be doing on the cloud box. If unsure which folders a given
manifest needs, check the `clip_path` column's prefixes directly rather than
guessing.

```bash
rsync -avz --progress C:\morse-ai-data\<subfolder>  user@<instance-ip>:/data/morse-ai-data/
```

(Run from WSL or Git Bash on Windows so `rsync` is available; substitute
`scp -r` if rsync isn't installed locally.) This will likely be the slowest
step - check your upload bandwidth before committing.

## 5. Point the code at the data

`paths.py` reads `MORSE_AI_DATA` - manifest CSVs store paths relative to
`DATA_ROOT`, so the absolute root can differ from the Windows box as long as
the subfolder layout matches:

```bash
export MORSE_AI_DATA=/data/morse-ai-data
echo 'export MORSE_AI_DATA=/data/morse-ai-data' >> ~/.bashrc
```

## 6. Set up the Python environment

```bash
cd ~/morse-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match the instance's CUDA version
```

## 7. Sanity-check before committing GPU-hours

```bash
python model/train.py --manifest /data/morse-ai-data/manifests/<manifest>.csv --dry-run
```

Confirms shapes/CTC loss/checkpointing work before paying for real epochs.

## 8. Launch training resiliently (survives an SSH disconnect)

For a run-until-you-say-stop session (the actual current plan - not a fixed
epoch count), pass a very large `--epochs` and rely on the stop-file check
instead:

```bash
tmux new -s train
python model/train.py --manifest /data/morse-ai-data/manifests/<manifest>.csv \
  --epochs 100000 --batch-size 64 --lr 3e-4 --warmup-steps 1000 --grad-clip 5.0 \
  --lr-decay-factor 0.5 --lr-decay-patience 5
# Ctrl+b then d to detach; `tmux attach -t train` to reattach later
```

`--warmup-steps`/`--grad-clip` are standard CTC stabilizers added during
local debugging - keep them. `--lr-decay-factor`/`--lr-decay-patience` halve
the LR whenever val_loss hasn't improved in 5 epochs (added after a real
incident: a run held a flat `lr=3e-4` for 40+ epochs while val_loss slowly
climbed - classic overfitting - and then hit a sharp NaN-driven divergence a
few epochs later that an unattended run wouldn't otherwise catch). Set
`--lr-decay-factor 0` to disable. Non-finite (NaN/Inf) losses are now also
skipped automatically rather than allowed through backward() - watch for a
`skipped=N` count appended to the epoch line; any nonzero count means
something upstream (a corrupt clip, a bad batch) is worth investigating even
though training itself won't crash or get poisoned by it.

`--resume-from <checkpoint>.pt` lets you extend a run later without losing
progress (model + optimizer + both scheduler states are all checkpointed
each epoch).

### Stopping an unlimited run cleanly

`train.py` checks once per epoch (right after that epoch's checkpoint is
saved) for a stop-file at `--stop-file` (defaults to
`$MORSE_AI_DATA/STOP_TRAINING`). When found, it deletes the file and exits -
it never gets killed mid-epoch. To stop the cloud run from your machine:

```bash
ssh user@<instance-ip> "touch /data/morse-ai-data/STOP_TRAINING"
```

Then wait for the current epoch to finish (check `tmux attach -t train`)
before pulling checkpoints or shutting the instance down - killing the VM
before it exits on its own risks a half-written checkpoint file.

## 9. Monitor

```bash
watch -n 2 nvidia-smi      # confirm GPU utilization
tmux attach -t train       # see live epoch/loss output
```

## 10. Check decode quality, not just loss

Loss alone is misleading for CTC (this is exactly how the original
blank-collapse bug went unnoticed for several runs - loss kept slowly
decreasing while greedy-decoded predictions were garbage). Use the real
metric instead:

```bash
python model/evaluate.py --manifest /data/morse-ai-data/manifests/<manifest>.csv \
  --checkpoint /data/morse-ai-data/checkpoints/decoder_epochNNN.pt --max-clips 2000
```

## 11. Retrieve checkpoints back to Windows

```bash
rsync -avz user@<instance-ip>:~/morse-ai-data/checkpoints/ "C:\morse-ai-data\checkpoints_cloud\"
```

## 12. Shut the instance down

Once you've pulled the checkpoints - billing is per-hour/minute on most
providers, so this is the easy way to accidentally overpay.

## Context as of this writing

A local run on clean-only fully-synthetic data (RTX 3060, denser 8s clips +
halved spectrogram hop) reached **1.86% CER** after 20 epochs -
see `checkpoints_run7_SUCCESS_clean_synthetic/`. That recipe was then
extended to mix in noise/fading/QRM augmentation (1.29M clips: clean + 3
noisy variants each) on the cloud box, unlimited-epochs via the stop-file
mechanism above.

**That combined-data run silently diverged.** Starting around epoch041, the
model began intermittently outputting runaway-repetition garbage on greedy
decode (e.g. `'JDKA6JDK' -> 'JDKA6JDKKAAAAAAAENAEAEAEAEAEAEAEA'`), fully
committing to the broken state by ~epoch050, and stayed there through
epoch101 (~60 epochs / 9+ hours of GPU time) before anyone noticed - because
CTC loss stayed smooth the entire time. Only greedy-decode quality broke,
which loss-based monitoring (including the LR-decay-on-plateau added
earlier) has no visibility into. Those checkpoints are archived at
`checkpoints_archived_combined_run_20260625/`, not deleted, in case they're
useful for comparison later.

Three things were added to `train.py` as a result, all active by default:
- **NaN/Inf-safe loss skipping** - a non-finite loss is skipped (no
  backward/step) rather than allowed to poison Adam's running moment
  estimates, logged as `skipped=N` in the epoch line.
- **Per-epoch decode-quality check** (`--decode-check-clips`/
  `--decode-check-threshold`) - greedy-decodes a small held-out subset every
  epoch and computes real CER, shown as `decode_cer=` in the log. This is
  what would have caught the divergence within one epoch instead of 60.
  Crosses the threshold -> training stops on its own (checkpoint already
  saved, so the bad epoch is there to diagnose, but you resume from an
  earlier one).
- **LR floor that stops training instead of decaying forever**
  (`--lr-min`), plus a small **weight decay** (`--weight-decay`). The
  diverged run had decayed LR tenfold (down to ~3e-7) while still getting
  worse - at a near-zero LR, weight updates are negligible, but BatchNorm's
  running statistics keep updating on every forward pass regardless of
  optimizer LR, and can keep drifting on a stream mixing clean and heavily
  noise-augmented batches. Stopping once LR bottoms out and genuinely
  plateaus removes that long, unproductive, drift-prone tail.

**Strategy: clean-only basis first, then a slow noise ramp - not a single
abrupt switch to `combined_manifest.csv`.** Even a curriculum-style
transition from 0% to ~75% noisy data in one jump risked being its own
destabilizing shock, similar in spirit to the original divergence. Instead:
`dataprep/build_noise_ramp_manifests.py` generates a series of manifests at
increasing noisy-clip ratios (all of `synthetic_manifest.csv`'s clean clips
plus a sampled fraction of `augmented_synthetic_manifest.csv`) -
`/root/morse-ai-data/manifests/noise_ramp/ramp_{05,10,20,35,50}pct.csv`
already generated. The endpoint beyond 50% is `combined_manifest.csv`
itself (~75% noisy, all available augmented clips) - no separate file
needed for that step.

**Phase 1 (clean-only, `synthetic_manifest.csv`)**: launched via
`start_clean_training.sh`, converged successfully via the LR-floor-stop at
**epoch086** (val_loss plateaued at 0.043, lr bottomed at 1e-5 - the
designed, successful outcome, not a failure). Best checkpoint was
**epoch084** (1.74% CER on a full 2000-clip eval - epoch086 itself was a
noise blip at 8.29%, not representative; always verify with a fuller
`evaluate.py` run before picking a checkpoint to resume from, not just
"latest"). Real-world sanity check against held-out real ARRL/on6zq audio
(see below) showed **97% CER** - i.e. essentially no transfer yet, which is
expected for a model with zero noise exposure. Now that phase 2 is running,
phase 1's checkpoints are archived (moved, not deleted) at
`/root/morse-ai-data/checkpoints_archived_phase1_clean_20260625/` - epoch084
lives there now, not at the original `checkpoints/decoder_epoch084.pt` path.

**Each ramp phase resumes from the previous phase's best checkpoint with
`--reset-optimizer`** (new flag) - loads model weights only, leaves
optimizer/LR-schedule/plateau-tracking fresh, and restarts epoch numbering
at 1 in a fresh `--checkpoint-dir`. This matters because the old phase's
Adam momentum and converged near-zero LR aren't suited to adapting to new
data, and its val_loss "best" isn't a fair baseline once validation itself
gets harder - carrying either forward would cause premature/wrong LR decay
decisions in the new phase. Verified with a local test (different `--lr`
on resume, confirmed it wasn't inherited from the checkpoint) before relying
on it. Phase 2 (5% ramp) launched the same way as phase 1 - `screen -dmS
train_ramp05`, log at `train_ramp_05pct.log`, checkpoints in
`checkpoints_ramp_05pct/`. Converged at epoch070; best was epoch063 (1.97%
CER, not the latest - same "verify, don't trust latest" lesson as phase 1).
Archived at `checkpoints_archived_05pct_20260625/`.

**From phase 3 (10% ramp) onward, `model/run_ramp_sequence.py` drives the
rest of the sequence automatically** - launched once via `screen -dmS
ramp_sequence start_ramp_sequence.sh` and left to run unattended through
10pct -> 20pct -> 35pct -> 50pct -> combined. It launches each phase's
`train.py`, blocks until it exits, reads the structured `RUN_STATUS.json`
that phase writes (`write_run_status()` in `train.py` - one of `"converged"`
/ `"diverged"` / `"stopped_manually"` / `"epochs_reached"`), and only
advances if status is `"converged"` - anything else halts the whole
sequence for human review rather than risk advancing past a bad phase.
On a successful phase it re-evaluates the last 5 checkpoints (not just
"latest") to pick the best, archives the completed phase's checkpoint dir,
and resumes the next phase from that checkpoint with `--reset-optimizer`.

This runs as a single long-lived process *on the cloud box*, independent of
any particular monitoring session - once launched it needs no further
human/Claude intervention to advance through the remaining phases, only to
react if it halts on something non-converged. Verified end-to-end (a
simulated 2-phase chain: subprocess launch, status-check, evaluate, archive,
correct resume args) on both Windows (local) and Linux (this cloud box, real
`cuda` device) before being trusted with the real sequence.

**Real ARRL audio is the standing generalization check, not a one-time
test.** A held-out sample of 3000 real ARRL/on6zq clips lives at
`/root/morse-ai-data/arrl_eval_sample/` (built by sampling
`clips_manifest.csv` locally, tarring just the sampled audio rather than
the full ~20GB/309k-file `processed/` folder, and extracting on the cloud
box). Re-run this against each phase's best checkpoint
(`--device cpu` so it doesn't compete with the GPU training job - the cloud
box's CPU is otherwise idle during training) to track whether real-world
transfer is actually improving as noise gets introduced. This number
matters more than synthetic `decode_cer` for judging whether the ramp
strategy is working.

**The 50pct -> combined (~75% noisy) jump turned out to be too large and
needed its own intermediate step.** Phase 50pct converged cleanly (best
checkpoint epoch066), but `combined_manifest.csv` diverged twice resuming
directly from it - first at epoch012 (decode_cer 0.855), then again on a
fresh `--reset-optimizer` retry from that run's own epoch009 at epoch018
(decode_cer 0.992) - both showing the same runaway-repetition collapse
pattern as the original incident (e.g. `'JDKA6JDK' -> 'JDKA6JDKKKTTLT3K'`),
and both runs showed noticeably more turbulent decode_cer in their early
epochs than any earlier phase transition. Two independent collapses on the
same jump, not a one-off blip, was the signal to add a step rather than
keep retrying blindly. `ramp_65pct.csv` was generated the same way as the
other ramp files (`python dataprep/build_noise_ramp_manifests.py --ratios
0.65`, no code changes needed - the script already took an arbitrary
ratio list) and inserted between 50pct and combined in both
`run_ramp_sequence.py`'s `PHASES` list and the actual cloud run, resuming
from phase 50pct's epoch066 checkpoint again (not from either diverged
combined attempt, to keep the curriculum's intent of incremental exposure
rather than carrying forward a run that had already partially adapted to
the full 75% distribution). If combined still diverges after this, the
next move is another intermediate step (e.g. 65pct -> 70pct) rather than
abandoning the ramp approach.

## v4: on-the-fly augmentation, then a 5-40 WPM widening (superseded the above)

The noise-ramp curriculum above was superseded entirely by on-the-fly
per-clip augmentation (see the callout at the top) - a single training run
with a smooth anneal, no phase resumes. A full v4 run on local hardware
(RTX 3060) converged cleanly; real-ARRL evaluation across all trained
speeds then found **near-total decode failure specifically at 5 WPM**
(CER ~1.0) while 10-40 WPM scored well - traced to `--wpm-range` never
including anything that slow (was `10,40`). Confirmed independently via
`dataprep/realign_arrl_labels.py`'s forced-alignment self-consistency
check, which also came back at ~1.0 CER for 5 WPM real audio - the model
had no real signal there at all, not just weaker performance.

**Fix and retrain**: widened `--wpm-range` to `5,40` and fixed a latent
Farnsworth realism bug the wider range would have made worse (character
speed floor was `wpm+3`, giving an unrealistically slow ~8 WPM character
speed at `wpm=5`; real slow-speed practice keys characters at a normal
~13-25 WPM pace with the gaps stretched - fixed to `max(wpm+3, 13)`).
Regenerated synthetic clips (243,874, up from 199,552 - slower speech
produces more clips per line) and **resumed from the prior v4 run's best
checkpoint** with `--reset-optimizer` rather than retraining from scratch -
transfer learning converged dramatically faster than the original run
(epoch 1 already near where the from-scratch run took ~50 epochs to
reach), since most of what the model knows doesn't need relearning, only
the WPM floor needed extending.

**Two infra-level crashes during this retrain, both resolved by the same
fix.** First a `CUDNN_STATUS_EXECUTION_FAILED` (had happened once before,
in the original v4 run, resumed without incident at the time), then on
retry a `CUDNN_STATUS_INTERNAL_ERROR` at the identical call site (LSTM
backward pass), then on a third attempt a `CUDA error: out of memory` at a
suspicious location (`pad_packed_sequence`'s `.cpu()` index move, which
cannot plausibly exhaust GPU memory on its own - PyTorch's own
asynchronous-error warning in that traceback makes it likely this was the
same underlying instability resurfacing through a different code path, not
three unrelated bugs). GPU health checks (`nvidia-smi`, Windows driver
event log) showed nothing - not thermal, not a driver reset. **Resuming
with `--no-amp --batch-size 32`** (half the normal batch size, mixed
precision disabled) fixed it - 28+ clean epochs before the next
interruption was an unrelated full machine reboot, cleanly recovered via
the normal checkpoint-resume path. If this recurs on a from-scratch v4 run
under normal settings, try this combination before assuming something
structural is wrong.

**Result, and an important evaluation-methodology finding.** The retrained
model (best checkpoint by re-evaluating the last several, not just latest -
epoch 95 of this phase) showed **zero regression at any of the original
10-40 WPM tiers** - confirms the wider range didn't thin out training
density at the speeds that already worked (checked directly: the 5-9 WPM
band alone was 68,808 of 243,874 clips, healthy, not diluted).

While investigating the 18-35 WPM numbers looking unexpectedly worse than
40 WPM despite being *easier* in principle (more time per character), found
that most of the apparent gap was `evaluate_streaming.py` itself scoring
against a header-stripped reference (see the callout box above) - **now
fixed** (`normalize_transcript()`, commit after this one). Re-running the
full 50-file/all-speed-tier evaluation with the fix gave the honest
picture, and the initial ~97-98%/~8% figures quoted in the first version of
this section (and in the v4.1.0 release notes) were themselves still partly
distorted by the same bug - corrected numbers, all speeds, all 5 files/tier:

| Speed | Accuracy | | Speed | Accuracy |
|---|---|---|---|---|
| 5 WPM | **40.4%** | | 25 WPM | 97.1% |
| 10 WPM | 94.8% | | 30 WPM | 97.4% |
| 13 WPM | 96.5% | | 35 WPM | 97.7% |
| 15 WPM | 96.0% | | 40 WPM | 98.0% |
| 18 WPM | 96.1% | | | |
| 20 WPM | 96.9% | | | |

**Every tier from 18-40 WPM is at or above 96%** - the model itself never
needed further work here, only the measurement did. **5 WPM is a real,
substantial improvement over the ~0% before this phase's retrain (not the
~8% first estimated) but still not usable** - still likely a harder
problem than the WPM-range gap alone; not pursued further given it's rare
traffic (beginner practice speed) rather than typical live QSO content -
revisit if that changes. Overall avg CER across all 50 files: 0.0891 (was
0.2057 under the buggy measurement) - the model didn't change, only how
honestly it was being scored.
