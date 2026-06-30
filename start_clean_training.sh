#!/bin/bash
export MORSE_AI_DATA=/root/morse-ai-data
cd /root/morse-ai
source .venv/bin/activate
python model/train.py \
  --manifest /root/morse-ai-data/manifests/synthetic_manifest.csv \
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
  2>&1 | tee /root/morse-ai-data/train_clean_run.log
