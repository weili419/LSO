#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAROOT="${DATAROOT:-${PROJECT_DIR}/data/cave_x4}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/checkpoints/cave/lso_x4_cave_x4_202601210644/model_best.pth.tar}"
cd "${PROJECT_DIR}"
python scripts/eval_cave.py \
  --dataset cave_x4 \
  --dataroot "${DATAROOT}" \
  --model lso_x4 \
  --model_path "${MODEL_PATH}" \
  --num_token 8 \
  --num_basis 24 \
  --width 64 \
  --patch_size 4 \
  --n_resblocks 4 \
  --guide_dim 128
