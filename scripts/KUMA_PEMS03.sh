#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-demo}"

COMMON_ARGS=(
  --root_path ./dataset/PEMS
  --data_path PEMS03.npz
  --enc_in 358
  --activation gelu
  --gpu 0
)

if [[ "${MODE}" == "demo" ]]; then
  python -u run.py "${COMMON_ARGS[@]}" \
    --mode train \
    --random_seed 2025 \
    --d_model 32 \
    --d_ff 32 \
    --e_layers 1 \
    --batch_size 4 \
    --num_workers 0 \
    --train_epochs 1 \
    --max_train_batches 2 \
    --max_eval_batches 2
elif [[ "${MODE}" == "full" ]]; then
  python -u run.py "${COMMON_ARGS[@]}" \
    --mode train \
    --random_seed 2025 \
    --d_model 256 \
    --d_ff 256 \
    --e_layers 1 \
    --batch_size 32 \
    --num_workers 10 \
    --train_epochs 20 \
    --learning_rate 0.002
else
  echo "Usage: bash scripts/KUMA_PEMS03.sh [demo|full]" >&2
  exit 2
fi
