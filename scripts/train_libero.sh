#!/bin/bash
# Train MetaWAM on LIBERO.
#
# Effective batch size = (per-GPU batch) * NUM_GPUS * GRAD_ACCUM; the paper uses 4 GPUs with 4 accumulation steps.
#
#   export BASE_DATASETS_DIR=/path/to/datasets   # directory containing LIBERO-Cosmos-Policy/
#   bash scripts/train_libero.sh
set -euo pipefail

: "${BASE_DATASETS_DIR:?Set BASE_DATASETS_DIR to the directory containing LIBERO-Cosmos-Policy/}"
NUM_GPUS="${NUM_GPUS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MASTER_PORT="${MASTER_PORT:-12341}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-./output}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --extra cu128 --group libero --python 3.10 \
  torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="metawam_libero" \
  trainer.grad_accum_iter="${GRAD_ACCUM}" \
  ckpt_type=dcp \
  "$@"
