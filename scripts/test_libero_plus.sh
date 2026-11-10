#!/bin/bash
# Evaluate MetaWAM on LIBERO-Plus, then summarize the success rate per perturbation category.
# Run from the repository root.
#
# Each suite is split into NUM_SHARDS shards that run in parallel, assigned round-robin to the GPUs in GPUS
# (several shards may share a GPU). Every LIBERO-Plus task is evaluated once. The shard logs are then merged by
# summarize_libero_plus.py.
#
#   export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus          # LIBERO-Plus checkout (provides the `libero` package)
#   export LIBERO_PLUS_T5_DIR=/path/to/liberoplus_t5      # --out_dir of save_libero_plus_t5_text_embeddings
#   GPUS="0 1 2 3" NUM_SHARDS=8 bash scripts/test_libero_plus.sh
#
# To resume an interrupted run, rerun with the same RUN_TAG and RESUME_DIR=$LOG_DIR/task_ids
# (written by the summary step): only the unfinished task ids of each suite are evaluated.
#
# Extra arguments are forwarded to run_libero_eval.py.
set -euo pipefail

: "${LIBERO_PLUS_ROOT:?Set LIBERO_PLUS_ROOT to the LIBERO-Plus checkout}"
: "${LIBERO_PLUS_T5_DIR:?Set LIBERO_PLUS_T5_DIR to the directory with libero_plus_t5_embeddings__<suite>.pkl}"

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
# The original 40 LIBERO instructions (used by the non-raw prompt modes), loaded as a supplemental T5 cache
EXTRA_T5_EMBEDDINGS_PATH="${EXTRA_T5_EMBEDDINGS_PATH:-${CKPT_DIR}/libero_t5_embeddings.pkl}"
CONFIG="${CONFIG:-metawam_libero__inference}"

SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
SEED="${SEED:-195}"
GPUS="${GPUS:-0}"
NUM_SHARDS="${NUM_SHARDS:-8}"
LOG_DIR="${LOG_DIR:-logs/libero_plus}"
RUN_TAG="${RUN_TAG:-metawam_liberoplus--seed${SEED}}"
RESUME_DIR="${RESUME_DIR:-}"

# Language-conditioning protocol: clean_nonlanguage (only the language perturbations change the instruction; the
# other categories keep the original LIBERO instruction) or raw (task.language as is for every task).
export LIBERO_PLUS_PROMPT_MODE="${LIBERO_PLUS_PROMPT_MODE:-clean_nonlanguage}"
if [[ "${LIBERO_PLUS_PROMPT_MODE}" != "raw" ]]; then
  # Must be a local copy of the original LIBERO T5 cache (its 40 keys are the original instructions)
  export LIBERO_PLUS_CANONICAL_PROMPTS_PATH="${LIBERO_PLUS_CANONICAL_PROMPTS_PATH:-${CKPT_DIR}/libero_t5_embeddings.pkl}"
fi
# The LIBERO-Plus checkout shadows the `libero` package of the uv `libero` group; keep its config separate.
export PYTHONPATH="${LIBERO_PLUS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero_plus}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

UV_RUN=(uv run --extra cu128 --group libero --python 3.10)
read -r -a gpu_list <<< "${GPUS}"
mkdir -p "${LOG_DIR}/stdout"

for suite in ${SUITES}; do
  task_ids_args=()
  if [[ -n "${RESUME_DIR}" ]]; then
    task_ids_args=(--task_ids "@${RESUME_DIR}/resume_${suite}.json")
  fi
  pids=()
  for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    gpu="${gpu_list[$((shard % ${#gpu_list[@]}))]}"
    stdout_log="${LOG_DIR}/stdout/${RUN_TAG}--${suite}--sh${shard}of${NUM_SHARDS}.log"
    echo "[test_libero_plus] suite=${suite} shard=${shard}/${NUM_SHARDS} gpu=${gpu} -> ${stdout_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${UV_RUN[@]}" \
      python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
        --config "${CONFIG}" \
        --ckpt_path "${CKPT_PATH}" \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path "${DATASET_STATS_PATH}" \
        --t5_text_embeddings_path "${LIBERO_PLUS_T5_DIR}/libero_plus_t5_embeddings__${suite}.pkl" \
        --extra_t5_text_embeddings_path "${EXTRA_T5_EMBEDDINGS_PATH}" \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --task_suite_name "${suite}" \
        --num_trials_per_task 1 \
        --task_num_shards "${NUM_SHARDS}" \
        --task_shard_index "${shard}" \
        "${task_ids_args[@]}" \
        --local_log_dir "${LOG_DIR}" \
        --rollout_video_dir "${LOG_DIR}/rollouts/${RUN_TAG}--${suite}" \
        --run_id_note "${RUN_TAG}--sh${shard}of${NUM_SHARDS}--deterministic" \
        --seed "${SEED}" \
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
        "$@" > "${stdout_log}" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || echo "[test_libero_plus] a ${suite} shard exited with an error; see ${LOG_DIR}/stdout/"
  done
done

# Per-category success table (+ JSON), and the unfinished task ids of each suite for resuming
"${UV_RUN[@]}" python -m cosmos_policy.experiments.robot.libero.summarize_libero_plus \
  --log_dir "${LOG_DIR}" \
  --libero_plus_root "${LIBERO_PLUS_ROOT}" \
  --run_tag "${RUN_TAG}" \
  --json_out "${LOG_DIR}/${RUN_TAG}.json" \
  --resume_out "${LOG_DIR}/task_ids"
