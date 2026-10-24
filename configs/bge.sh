#!/bin/bash
# bge-base-en-v1.5, paper setting: 4 GPUs x batch 16 (effective 64), 5 epochs.
# Usage: bash configs/bge.sh [SEED]           (paper: seeds 42, 1, 7, 13)
set -euo pipefail
SEED=${1:-42}
OUT=runs/bge/seed${SEED}
PREFIX='Represent this sentence for searching relevant passages: '

accelerate launch --num_processes 4 --mixed_precision bf16 scripts/train.py \
  --base-model BAAI/bge-base-en-v1.5 --pooling mean --max-len 512 \
  --batch-size 16 --grad-accum 1 --epochs 5 --lr 2e-4 --lr-gate 5e-4 \
  --cf-per-axis 2 --kd-per-batch 8 --instruction-prefix "$PREFIX" \
  --seed "$SEED" --out-dir "$OUT"

python scripts/evaluate.py --ckpt "$OUT/final" --out-dir "$OUT/eval" --instruction-prefix "$PREFIX" \
  --batch-encode 64 --batch-encode-m 32 --max-len-q 256 --max-len-m 512
