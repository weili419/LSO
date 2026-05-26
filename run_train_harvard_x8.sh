#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAROOT="${DATAROOT:-${PROJECT_DIR}/data/Harvard_x4}"
cd "${PROJECT_DIR}"
python scripts/train_harvard.py \
  --dataset Harvard_x8 \
  --dataroot "${DATAROOT}" \
  --model lso_x8 \
  --batchSize 16 \
  --testBatchSize 1 \
  --n_epochs 1000 \
  --lr 3e-4 \
  --num_token 8 \
  --num_basis 24 \
  --width 64 \
  --patch_size 4 \
  --n_resblocks 4 \
  --guide_dim 128
