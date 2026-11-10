#!/bin/bash
# Evaluate MetaWAM on RoboCasa (24 tasks x 50 trials per task, one run per seed). Run from the repository root.
#
#   bash scripts/test_robocasa.sh                                                  # released checkpoint, all tasks, seeds 195-197
#   TASK_NAME=TurnOffMicrowave SEEDS=195 bash scripts/test_robocasa.sh             # one task, one seed
#   CKPT_PATH=checkpoints/metawam_robocasa_8400.pt bash scripts/test_robocasa.sh   # your own converted checkpoint
#
# Extra arguments are forwarded to run_robocasa_eval.py (e.g. `--use_wandb True`).
set -euo pipefail

# Local checkpoint folder or Hugging Face repo id (<org>/<repo>)
CKPT_DIR="${CKPT_DIR:-checkpoints/MetaWAM-RoboCasa-Predict2-2B}"
if [[ -z "${CKPT_PATH:-}" ]]; then
  if [[ -d "${CKPT_DIR}" ]]; then
    CKPT_PATH="${CKPT_DIR}/MetaWAM-RoboCasa-Predict2-2B.pt"
  else
    CKPT_PATH="${CKPT_DIR}"  # HF repo id: the .pt file in the repo is downloaded
  fi
fi
DATASET_STATS_PATH="${DATASET_STATS_PATH:-${CKPT_DIR}/robocasa_dataset_statistics.json}"
T5_EMBEDDINGS_PATH="${T5_EMBEDDINGS_PATH:-${CKPT_DIR}/robocasa_t5_embeddings.pkl}"
CONFIG="${CONFIG:-metawam_robocasa__inference}"

TASK_NAME="${TASK_NAME:-all}"                  # all | pnp | <TaskName>
SEEDS="${SEEDS:-195 196 197}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
GPU_ID="${GPU_ID:-0}"
NUM_PARALLEL_ENVS="${NUM_PARALLEL_ENVS:-10}"
INFERENCE_BATCH_TIMEOUT="${INFERENCE_BATCH_TIMEOUT:-0.05}"
RUN_NAME="${RUN_NAME:-metawam_robocasa}"
# run_robocasa_eval.py resolves relative log dirs against its own folder, so pass an absolute path
LOG_DIR="${LOG_DIR:-logs/robocasa}"
mkdir -p "${LOG_DIR}"
LOG_DIR="$(cd "${LOG_DIR}" && pwd)"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

for seed in ${SEEDS}; do
  echo "[test_robocasa] ckpt=${CKPT_PATH} task=${TASK_NAME} seed=${seed} gpu=${GPU_ID} parallel_envs=${NUM_PARALLEL_ENVS}"
  uv run --extra cu128 --group robocasa --python 3.10 \
    python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
      --config "${CONFIG}" \
      --ckpt_path "${CKPT_PATH}" \
      --config_file cosmos_policy/config/config.py \
      --use_wrist_image True \
      --num_wrist_images 1 \
      --use_proprio True \
      --normalize_proprio True \
      --unnormalize_actions True \
      --dataset_stats_path "${DATASET_STATS_PATH}" \
      --t5_text_embeddings_path "${T5_EMBEDDINGS_PATH}" \
      --trained_with_image_aug True \
      --chunk_size 32 \
      --num_open_loop_steps 16 \
      --task_name "${TASK_NAME}" \
      --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
      --run_id_note "${RUN_NAME}--5stepAct--seed${seed}--deterministic" \
      --local_log_dir "${LOG_DIR}" \
      --seed "${seed}" \
      --randomize_seed False \
      --deterministic True \
      --use_variance_scale False \
      --use_jpeg_compression True \
      --flip_images True \
      --num_denoising_steps_action 5 \
      --num_denoising_steps_future_state 1 \
      --num_denoising_steps_value 1 \
      --data_collection False \
      --rollout_prefix "${RUN_NAME}_seed${seed}" \
      --num_parallel_envs "${NUM_PARALLEL_ENVS}" \
      --inference_batch_timeout "${INFERENCE_BATCH_TIMEOUT}" \
      "$@"
done
