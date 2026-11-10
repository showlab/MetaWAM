#!/bin/bash
# Evaluate MetaWAM on the four LIBERO suites (10 tasks x 50 trials each), one run per suite and seed.
# Run from the repository root.
#
#   bash scripts/test_libero.sh                                                  # released checkpoint, 4 suites, seeds 195-197
#   SUITES=libero_10 SEEDS=195 bash scripts/test_libero.sh                       # one suite, one seed
#   CKPT_PATH=checkpoints/metawam_libero_10200.pt bash scripts/test_libero.sh    # your own converted checkpoint
#
# Extra arguments are forwarded to run_libero_eval.py (e.g. `--use_wandb True`).
set -euo pipefail

# Local checkpoint folder or Hugging Face repo id (<org>/<repo>)
CKPT_DIR="${CKPT_DIR:-checkpoints/MetaWAM-LIBERO-Predict2-2B}"
if [[ -z "${CKPT_PATH:-}" ]]; then
  if [[ -d "${CKPT_DIR}" ]]; then
    CKPT_PATH="${CKPT_DIR}/MetaWAM-LIBERO-Predict2-2B.pt"
  else
    CKPT_PATH="${CKPT_DIR}"  # HF repo id: the .pt file in the repo is downloaded
  fi
fi
DATASET_STATS_PATH="${DATASET_STATS_PATH:-${CKPT_DIR}/libero_dataset_statistics.json}"
T5_EMBEDDINGS_PATH="${T5_EMBEDDINGS_PATH:-${CKPT_DIR}/libero_t5_embeddings.pkl}"
CONFIG="${CONFIG:-metawam_libero__inference}"

SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
SEEDS="${SEEDS:-195 196 197}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
GPU_ID="${GPU_ID:-0}"
RUN_NAME="${RUN_NAME:-metawam_libero}"
LOG_DIR="${LOG_DIR:-logs/libero}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

for suite in ${SUITES}; do
  for seed in ${SEEDS}; do
    echo "[test_libero] ckpt=${CKPT_PATH} suite=${suite} seed=${seed} gpu=${GPU_ID}"
    uv run --extra cu128 --group libero --python 3.10 \
      python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
        --config "${CONFIG}" \
        --ckpt_path "${CKPT_PATH}" \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path "${DATASET_STATS_PATH}" \
        --t5_text_embeddings_path "${T5_EMBEDDINGS_PATH}" \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --task_suite_name "${suite}" \
        --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
        --local_log_dir "${LOG_DIR}" \
        --rollout_video_dir "${LOG_DIR}/rollouts/${RUN_NAME}--${suite}--seed${seed}" \
        --run_id_note "${RUN_NAME}--5stepAct--seed${seed}--deterministic" \
        --seed "${seed}" \
        --randomize_seed False \
        --deterministic True \
        --use_variance_scale False \
        --ar_future_prediction False \
        --ar_value_prediction False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --data_collection False \
        "$@"
  done
done
