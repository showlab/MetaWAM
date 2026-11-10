#!/usr/bin/env bash
# Run the Franka deploy server (model / server side).
# Supports local uv execution and falls back to .venv/python if needed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${FRANKA_CONFIG:-metawam_realworld__inference}"
# Checkpoint (.pt), dataset statistics JSON and T5 embeddings cache of the real-world training run.
CKPT_PATH="${FRANKA_CKPT_PATH:?Set FRANKA_CKPT_PATH to the model checkpoint (.pt)}"
DATASET_STATS_PATH="${FRANKA_DATASET_STATS_PATH:?Set FRANKA_DATASET_STATS_PATH to the training dataset statistics JSON}"
T5_TEXT_EMBEDDINGS_PATH="${FRANKA_T5_EMBEDDINGS_PATH:?Set FRANKA_T5_EMBEDDINGS_PATH to the T5 text embeddings .pkl}"
HOST="${FRANKA_SERVER_HOST:-127.0.0.1}"
PORT="${FRANKA_SERVER_PORT:-8777}"
DEBUG_PREDICTIONS="${FRANKA_DEBUG_PREDICTIONS:-True}"
NORMALIZE_PROPRIO="${FRANKA_NORMALIZE_PROPRIO:-True}"
UNNORMALIZE_ACTIONS="${FRANKA_UNNORMALIZE_ACTIONS:-True}"
# Best-of-N planning: sample N action candidates per request and execute the one
# with the highest predicted value. Set to 1 to disable (default).
NUM_QUERIES_BEST_OF_N="${FRANKA_NUM_QUERIES_BEST_OF_N:-1}"
# Autoregressive planning (V(s') / Q(s,a)) and multi-level search. Defaults match
# the simplified deployment (value produced in parallel with the action chunk).
AR_FUTURE_PREDICTION="${FRANKA_AR_FUTURE_PREDICTION:-False}"
AR_VALUE_PREDICTION="${FRANKA_AR_VALUE_PREDICTION:-False}"
AR_QVALUE_PREDICTION="${FRANKA_AR_QVALUE_PREDICTION:-False}"
SEARCH_DEPTH="${FRANKA_SEARCH_DEPTH:-1}"
NUM_DENOISING_STEPS_FUTURE_STATE="${FRANKA_NUM_DENOISING_STEPS_FUTURE_STATE:-1}"
NUM_DENOISING_STEPS_VALUE="${FRANKA_NUM_DENOISING_STEPS_VALUE:-1}"
# Optional separate planning model (leave empty to plan with the policy model).
PLANNING_MODEL_CONFIG_NAME="${FRANKA_PLANNING_MODEL_CONFIG_NAME:-}"
PLANNING_MODEL_CKPT_PATH="${FRANKA_PLANNING_MODEL_CKPT_PATH:-}"
UV_EXTRA="${FRANKA_UV_EXTRA:-cu128}"
# The `aloha` uv dependency group provides the server dependencies (fastapi, uvicorn, json_numpy, draccus);
# see requirements_franka.txt when installing into an existing environment instead.
UV_GROUP="${FRANKA_UV_GROUP:-aloha}"
UV_PYTHON="${FRANKA_UV_PYTHON:-3.10}"

launcher=()
if [[ "${FRANKA_USE_UV:-1}" != "0" ]] && command -v uv >/dev/null 2>&1; then
    launcher=(uv run)
    launcher+=(--frozen )
    if [[ -n "${UV_EXTRA}" ]]; then
        launcher+=(--extra "${UV_EXTRA}")
    fi
    if [[ -n "${UV_GROUP}" ]]; then
        launcher+=(--group "${UV_GROUP}")
    fi
    if [[ -n "${UV_PYTHON}" ]]; then
        launcher+=(--python "${UV_PYTHON}")
    fi
    launcher+=(python)
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    launcher=("${REPO_ROOT}/.venv/bin/python")
else
    launcher=(python)
fi

echo "[Franka deploy] repo root: ${REPO_ROOT}"
echo "[Franka deploy] launcher: ${launcher[*]}"

exec "${launcher[@]}" -m cosmos_policy.experiments.robot.franka.deploy_franka \
    --config "${CONFIG}" \
    --ckpt_path "${CKPT_PATH}" \
    --dataset_stats_path "${DATASET_STATS_PATH}" \
    --t5_text_embeddings_path "${T5_TEXT_EMBEDDINGS_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --debug_predictions "${DEBUG_PREDICTIONS}" \
    --normalize_proprio "${NORMALIZE_PROPRIO}" \
    --unnormalize_actions "${UNNORMALIZE_ACTIONS}" \
    --num_queries_best_of_n "${NUM_QUERIES_BEST_OF_N}" \
    --ar_future_prediction "${AR_FUTURE_PREDICTION}" \
    --ar_value_prediction "${AR_VALUE_PREDICTION}" \
    --ar_qvalue_prediction "${AR_QVALUE_PREDICTION}" \
    --search_depth "${SEARCH_DEPTH}" \
    --num_denoising_steps_future_state "${NUM_DENOISING_STEPS_FUTURE_STATE}" \
    --num_denoising_steps_value "${NUM_DENOISING_STEPS_VALUE}" \
    --planning_model_config_name "${PLANNING_MODEL_CONFIG_NAME}" \
    --planning_model_ckpt_path "${PLANNING_MODEL_CKPT_PATH}" \
    "$@"
