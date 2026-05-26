#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAROOT="${DATAROOT:-${PROJECT_DIR}/data/Harvard_x4}"
GPU="${GPU:-0}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/checkpoints/harvard/lso_x4_Harvard_x4_202601221430/model_best.pth.tar}"
cd "${PROJECT_DIR}"
python scripts/eval_harvard.py \
  --dataset Harvard_x4 \
  --dataroot "${DATAROOT}" \
  --model lso_x4 \
  --model_path "${MODEL_PATH}" \
  --num_token 8 \
  --num_basis 24 \
  --width 64 \
  --patch_size 4 \
  --n_resblocks 4 \
  --guide_dim 128 \
  --gpu "${GPU}"
