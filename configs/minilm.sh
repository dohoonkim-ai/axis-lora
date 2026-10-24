#!/bin/bash
# all-MiniLM-L6-v2, paper setting: 4 GPUs x batch 64 (effective 256), 100 epochs.
# Usage: bash configs/minilm.sh [SEED]        (paper: seeds 42, 1, 7, 13)
set -euo pipefail
SEED=${1:-42}
OUT=runs/minilm/seed${SEED}

accelerate launch --num_processes 4 --mixed_precision bf16 scripts/train.py \
  --base-model sentence-transformers/all-MiniLM-L6-v2 --pooling mean --max-len 512 \
  --batch-size 64 --grad-accum 1 --epochs 100 --lr 2e-4 --lr-gate 1e-3 \
  --cf-per-axis 2 --kd-per-batch 8 --log-every 50 \
  --seed "$SEED" --out-dir "$OUT"

python scripts/evaluate.py --ckpt "$OUT/final" --out-dir "$OUT/eval" \
  --batch-encode 64 --batch-encode-m 32 --max-len-q 256 --max-len-m 512
