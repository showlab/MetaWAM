#!/bin/bash
# Train MetaWAM on a real-world Franka dataset in LeRobot format.
#
# One policy is trained jointly on every task in the dataset. REALWORLD_DATASET_DIR must contain
# `success_only/` (teleoperated demos + t5_embeddings.pkl) and `all_episodes/` (demos + policy rollouts,
# used for world-model / value learning).
#
#   export REALWORLD_DATASET_DIR=/path/to/franka_lerobot_dataset
#   bash scripts/train_realworld.sh
set -euo pipefail

: "${REALWORLD_DATASET_DIR:?Set REALWORLD_DATASET_DIR to the LeRobot-format Franka dataset}"
NUM_GPUS="${NUM_GPUS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MASTER_PORT="${MASTER_PORT:-12341}"
export REALWORLD_DATASET_DIR
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-./output}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --extra cu128 --group robocasa --python 3.10 \
  torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="metawam_realworld" \
  trainer.grad_accum_iter="${GRAD_ACCUM}" \
  ckpt_type=dcp \
  "$@"
