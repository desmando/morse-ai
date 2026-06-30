#!/bin/bash
export MORSE_AI_DATA=/root/morse-ai-data
cd /root/morse-ai
source .venv/bin/activate
python model/train.py \
  --manifest /root/morse-ai-data/manifests/noise_ramp/ramp_05pct.csv \
  --checkpoint-dir /root/morse-ai-data/checkpoints_ramp_05pct \
  --resume-from /root/morse-ai-data/checkpoints/decoder_epoch084.pt \
  --reset-optimizer \
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
  2>&1 | tee /root/morse-ai-data/train_ramp_05pct.log
