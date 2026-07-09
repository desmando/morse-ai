#!/bin/bash
# DEPRECATED - superseded by start_v4_training.sh (on-the-fly augmentation
# replaces the ramp-phase curriculum, which twice collapsed at the
# 50pct->combined jump). Kept for reference only.
export MORSE_AI_DATA=/root/morse-ai-data
cd /root/morse-ai
source .venv/bin/activate
python model/run_ramp_sequence.py \
  --start-phase 10pct \
  --start-checkpoint /root/morse-ai-data/checkpoints_archived_05pct_20260625/decoder_epoch063.pt \
  > /root/morse-ai-data/run_ramp_sequence.log 2>&1
