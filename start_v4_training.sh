#!/bin/bash
# v4 recipe: single run, on-the-fly augmentation with a smooth clean->noisy
# anneal. Replaces the whole noise-ramp phase sequence (no pre-rendered
# augmented manifests, no ramp files, no per-phase --reset-optimizer resumes).
#
# Prerequisites on the box:
#   1. Regenerate the corpus and synthetic clips with the v4 realism defaults
#      (prosigns, per-sender style, Farnsworth, 10-40 WPM, tone_hz column):
#        python lm/generate_qso_corpus.py --num-qsos 20000
#        python dataprep/synthesize_morse_audio.py --clip-seconds 8.0
#      (build_manifest.py's vocab.txt is unchanged - 47 chars + blank.)
#   2. No augmented_synthetic manifest needed - impairments happen in the
#      dataloader.
export MORSE_AI_DATA=/root/morse-ai-data
cd /root/morse-ai
source .venv/bin/activate
python model/train.py \
  --manifest /root/morse-ai-data/manifests/synthetic_manifest.csv \
  --augment \
  --snr-db-range -3,20 \
  --aug-anneal-epochs 20 \
  --epochs 100000 \
  --batch-size 64 \
  --lr 3e-4 \
  --warmup-steps 1000 \
  --grad-clip 5.0 \
  --weight-decay 1e-5 \
  --lr-decay-factor 0.5 \
  --lr-decay-patience 5 \
  --lr-min 1e-5 \
  --decode-check-clips 200 \
  --decode-check-threshold 0.5 \
  2>&1 | tee /root/morse-ai-data/train_v4.log
