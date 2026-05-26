#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAROOT="${DATAROOT:-${PROJECT_DIR}/data/cave_x4}"
GPU="${GPU:-0}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/checkpoints/cave/lso_x8_cave_x8_202601210442/model_best.pth.tar}"
cd "${PROJECT_DIR}"
python scripts/eval_cave.py \
  --dataset cave_x8 \
  --dataroot "${DATAROOT}" \
  --model lso_x8 \
  --model_path "${MODEL_PATH}" \
  --num_token 8 \
  --num_basis 24 \
  --width 64 \
  --patch_size 4 \
  --n_resblocks 4 \
  --guide_dim 128 \
  --gpu "${GPU}"
