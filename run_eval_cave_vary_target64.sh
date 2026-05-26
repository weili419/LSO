#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAROOT="${DATAROOT:-${PROJECT_DIR}/data/cave_x4}"
GPU="${GPU:-0}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/checkpoints/cave/lso_x8_cave_x8_202601210442/model_best.pth.tar}"
DATASETS="${DATASETS:-cave_x4 cave_x8 cave_x16 cave_x32}"

cd "${PROJECT_DIR}"
for DATASET in ${DATASETS}; do
  echo "======================================================================"
  echo "Evaluating ${DATASET} resized to 64x64"
  echo "======================================================================"
  python scripts/eval_cave_vary.py \
    --dataset "${DATASET}" \
    --dataroot "${DATAROOT}" \
    --targetsize 64,64 \
    --model lso_x8 \
    --model_path "${MODEL_PATH}" \
    --num_token 8 \
    --num_basis 24 \
    --width 64 \
    --patch_size 4 \
    --n_resblocks 4 \
    --guide_dim 128 \
    --gpu "${GPU}"
done
