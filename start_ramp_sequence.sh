#!/bin/bash
export MORSE_AI_DATA=/root/morse-ai-data
cd /root/morse-ai
source .venv/bin/activate
python model/run_ramp_sequence.py \
  --start-phase 10pct \
  --start-checkpoint /root/morse-ai-data/checkpoints_archived_05pct_20260625/decoder_epoch063.pt \
  > /root/morse-ai-data/run_ramp_sequence.log 2>&1
