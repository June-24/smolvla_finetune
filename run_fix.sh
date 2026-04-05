#!/bin/bash
set -e
PYTHON=/home/junaid/miniforge3/envs/smolvla/bin/python
PROJ="/mnt/c/Users/Moham/OneDrive/Desktop/Claude Projects/smolvla_finetune"
cd "$PROJ"

echo "=== Fine-tuning SmolVLA on SO-100 ==="
$PYTHON train_so100.py \
    --data data/so100_pickplace \
    --output checkpoints/so100_run \
    --steps 5000 \
    --batch 4 \
    --lr 1e-4 \
    --bf16
