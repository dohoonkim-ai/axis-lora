#!/bin/bash
# Qwen3-Embedding-0.6B, paper setting: 4 GPUs x batch 2 x grad-accum 2 (effective 16), 5 epochs.
# Usage: bash configs/qwen3.sh [SEED]         (paper: seeds 42, 1, 7, 13)
set -euo pipefail
SEED=${1:-42}
OUT=runs/qwen3/seed${SEED}
PREFIX=$'Instruct: Given a personal-memory question, retrieve relevant memory entries.\nQuery: '
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --num_processes 4 --mixed_precision bf16 scripts/train.py \
  --base-model Qwen/Qwen3-Embedding-0.6B --pooling last --max-len 1536 \
  --batch-size 2 --grad-accum 2 --epochs 5 --lr 2e-4 --lr-gate 5e-4 \
  --cf-per-axis 1 --kd-per-batch 2 --instruction-prefix "$PREFIX" \
  --seed "$SEED" --out-dir "$OUT"

python scripts/evaluate.py --ckpt "$OUT/final" --out-dir "$OUT/eval" --instruction-prefix "$PREFIX" \
  --batch-encode 16 --batch-encode-m 4 --max-len-q 2048 --max-len-m 4096
