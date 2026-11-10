#!/usr/bin/env bash
# Run the Franka evaluation client (robot / client side).
# Supports local uv execution and falls back to .venv/python if needed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# Client payload camera slots (the deploy server remaps them to the RealWorldLeRobotDataset views):
#   external -> primary_image, left -> left_wrist_image (secondary view), wrist -> right_wrist_image (wrist view).
# Hardware settings (set these for your setup):
#   FRANKA_NUC_IP           IP of the Franka NUC running the Polymetis ZeroRPC server
#   FRANKA_EXTERNAL_CAMERA  RealSense serial of the external camera (behind the robot, facing the workspace)
#   FRANKA_LEFT_CAMERA      RealSense serial of the second third-person camera (in front of the robot)
#   FRANKA_WRIST_CAMERA     RealSense serial of the wrist camera (facing downwards)
# List connected RealSense serials with: python -m cosmos_policy.experiments.robot.franka.camera_utils

NUC_IP="${FRANKA_NUC_IP:-127.0.0.1}"
NUC_PORT="${FRANKA_NUC_PORT:-4242}"
EXTERNAL_CAMERA="${FRANKA_EXTERNAL_CAMERA:-}"
WRIST_CAMERA="${FRANKA_WRIST_CAMERA:-}"
LEFT_CAMERA="${FRANKA_LEFT_CAMERA:-}"
CONTROL_MODE="${FRANKA_CONTROL_MODE:-eef}"
SERVER_IP="${FRANKA_SERVER_IP:-127.0.0.1}"
SERVER_PORT="${FRANKA_SERVER_PORT:-8777}"
MAX_TIMESTEPS="${FRANKA_MAX_TIMESTEPS:-600}"
SHOW_CAMERAS="${FRANKA_SHOW_CAMERAS:-False}"
SAVE_VIDEO="${FRANKA_SAVE_VIDEO:-False}"
DEBUG_PREDICTIONS="${FRANKA_DEBUG_PREDICTIONS:-True}"
DEBUG_LOG_DIR="${FRANKA_DEBUG_LOG_DIR:-franka_debug_predictions}"
DEBUG_IMAGE_LAYOUT="${FRANKA_DEBUG_IMAGE_LAYOUT:-separate}"
DEBUG_APPLY_MODEL_IMAGE_AUG="${FRANKA_DEBUG_APPLY_MODEL_IMAGE_AUG:-True}"
POLICY_ACTION_MODE="${FRANKA_POLICY_ACTION_MODE:-delta_eef_7d}"
POLICY_GRIPPER_PROPRIO_MODE="${FRANKA_POLICY_GRIPPER_PROPRIO_MODE:-measured_width_signed}"
POLICY_GRIPPER_POSITIVE_ACTION="${FRANKA_POLICY_GRIPPER_POSITIVE_ACTION:-open}"
START_GRIPPER_WIDTH="${FRANKA_START_GRIPPER_WIDTH:-}"
GRIPPER_RELEASE_CONFIRM_STEPS="${FRANKA_GRIPPER_RELEASE_CONFIRM_STEPS:-}"
GRIPPER_CHUNK_LOOKAHEAD_CLOSE="${FRANKA_GRIPPER_CHUNK_LOOKAHEAD_CLOSE:-True}"
GRIPPER_CHUNK_LOOKAHEAD_RELEASE="${FRANKA_GRIPPER_CHUNK_LOOKAHEAD_RELEASE:-True}"
GRIPPER_CHUNK_LOOKAHEAD_CONFIRM_STEPS="${FRANKA_GRIPPER_CHUNK_LOOKAHEAD_CONFIRM_STEPS:-3}"
GRIPPER_CHUNK_LOOKAHEAD_MIN_OFFSET="${FRANKA_GRIPPER_CHUNK_LOOKAHEAD_MIN_OFFSET:-6}"
OPEN_LOOP_HORIZON="${FRANKA_OPEN_LOOP_HORIZON:-16}"
DELTA_ACTION_BASE="${FRANKA_DELTA_ACTION_BASE:-target}"
LOCK_RPY_ACTIONS="${FRANKA_LOCK_RPY_ACTIONS:-False}"
LOCK_RPY_ACTION_AXES="${FRANKA_LOCK_RPY_ACTION_AXES:-}"
MAX_TARGET_POSITION_LEAD="${FRANKA_MAX_TARGET_POSITION_LEAD:-0.10}"
MAX_TARGET_VERTICAL_LEAD="${FRANKA_MAX_TARGET_VERTICAL_LEAD:-0.05}"
CLOSED_GRIPPER_MIN_TARGET_Z="${FRANKA_CLOSED_GRIPPER_MIN_TARGET_Z:-}"
BUILD_TASK_MIN_TARGET_Z="${FRANKA_BUILD_TASK_MIN_TARGET_Z:-0.139}"
BUILD_TASK_CLOSED_GRIPPER_MIN_TARGET_Z="${FRANKA_BUILD_TASK_CLOSED_GRIPPER_MIN_TARGET_Z:-0.178}"
TASK_NAME_FOR_PRESET="${FRANKA_TASK_NAME:-sweep}"
TASK_PRESET="${FRANKA_TASK_PRESET:-${TASK_NAME_FOR_PRESET##*/}}"
case "${TASK_PRESET,,}" in
    cup|d_cup)
        DEFAULT_PROMPT_VALUE="Place the cup on the green coaster"
        DEFAULT_START_JOINT_LOW="0.013538199, -0.607440637, -0.024424146, -2.810219556, -0.004018538, 2.206003447, 0.778393686"
        DEFAULT_START_JOINT_HIGH="0.027622807, -0.489711746, 0.006686491, -2.725664685, 0.019999191, 2.314832191, 0.824614995"
        DEFAULT_MIN_TARGET_Z="0.14"
        ;;
    build|build_block|d_build)
        DEFAULT_PROMPT_VALUE="Place the orange block on top of the gray block"
        DEFAULT_START_JOINT_LOW="0.006574900, -0.598931382, -0.025648478, -2.842746106, 0.003983285, 2.234394525, 0.782927812"
        DEFAULT_START_JOINT_HIGH="0.028517040, -0.479439665, 0.008107997, -2.794380222, 0.024404978, 2.361698553, 0.830549637"
        DEFAULT_MIN_TARGET_Z="0.14"
        ;;
    sweep|d_sweep)
        DEFAULT_PROMPT_VALUE="Sweep the green block into the dustpan"
        DEFAULT_START_JOINT_LOW="0.012812506, -0.603866181, -0.017325138, -2.793199597, -0.000767974, 2.164334780, 0.779078141"
        DEFAULT_START_JOINT_HIGH="0.039841094, -0.498352482, 0.015791032, -2.718271034, 0.030567631, 2.278225735, 0.834738195"
        DEFAULT_MIN_TARGET_Z="0.155"
        ;;
    *)
        DEFAULT_PROMPT_VALUE=""
        DEFAULT_START_JOINT_LOW=""
        DEFAULT_START_JOINT_HIGH=""
        DEFAULT_MIN_TARGET_Z="0.14"
        ;;
esac
DEFAULT_PROMPT="${FRANKA_DEFAULT_PROMPT-${DEFAULT_PROMPT_VALUE}}"
START_JOINT_LOW="${FRANKA_START_JOINT_LOW-${DEFAULT_START_JOINT_LOW}}"
START_JOINT_HIGH="${FRANKA_START_JOINT_HIGH-${DEFAULT_START_JOINT_HIGH}}"
START_JOINT_MOVE_TIME="${FRANKA_START_JOINT_MOVE_TIME:-3.0}"
START_JOINT_SETTLE_TIME="${FRANKA_START_JOINT_SETTLE_TIME:-0.5}"
MIN_TARGET_Z="${FRANKA_MIN_TARGET_Z:-${DEFAULT_MIN_TARGET_Z}}"
# Robot-side dependencies are listed in requirements_franka.txt.
CLIENT_UV_EXTRA="${FRANKA_CLIENT_UV_EXTRA:-}"
CLIENT_UV_GROUP="${FRANKA_CLIENT_UV_GROUP:-aloha}"
CLIENT_UV_PYTHON="${FRANKA_CLIENT_UV_PYTHON:-3.10}"

launcher=()
if [[ -n "${CONDA_PREFIX:-}" || -n "${VIRTUAL_ENV:-}" ]]; then
    launcher=(python)
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    launcher=("${REPO_ROOT}/.venv/bin/python")
elif [[ "${FRANKA_CLIENT_USE_UV:-1}" != "0" ]] && command -v uv >/dev/null 2>&1; then
    launcher=(uv run)
    if [[ -n "${CLIENT_UV_EXTRA}" ]]; then
        launcher+=(--extra "${CLIENT_UV_EXTRA}")
    fi
    if [[ -n "${CLIENT_UV_GROUP}" ]]; then
        launcher+=(--group "${CLIENT_UV_GROUP}")
    fi
    if [[ -n "${CLIENT_UV_PYTHON}" ]]; then
        launcher+=(--python "${CLIENT_UV_PYTHON}")
    fi
    launcher+=(python)
else
    launcher=(python)
fi

echo "[Franka eval] repo root: ${REPO_ROOT}"
echo "[Franka eval] launcher: ${launcher[*]}"
echo "[Franka eval] debug images: ${DEBUG_PREDICTIONS} -> ${REPO_ROOT}/${DEBUG_LOG_DIR}"
echo "[Franka eval] debug image layout: ${DEBUG_IMAGE_LAYOUT}"
echo "[Franka eval] debug current image model aug: ${DEBUG_APPLY_MODEL_IMAGE_AUG}"
echo "[Franka eval] lock RPY actions: ${LOCK_RPY_ACTIONS}"
echo "[Franka eval] lock RPY action axes: ${LOCK_RPY_ACTION_AXES:-none}"
echo "[Franka eval] task preset: ${TASK_PRESET:-none}"
echo "[Franka eval] default task: ${DEFAULT_PROMPT:-none}"
if [[ -n "${START_JOINT_LOW}" || -n "${START_JOINT_HIGH}" ]]; then
    echo "[Franka eval] start joint low:  ${START_JOINT_LOW}"
    echo "[Franka eval] start joint high: ${START_JOINT_HIGH}"
    echo "[Franka eval] start move:       ${START_JOINT_MOVE_TIME}s (+ settle ${START_JOINT_SETTLE_TIME}s)"
else
    echo "[Franka eval] start joint: disabled"
fi

if [[ -z "${START_JOINT_LOW}" && -n "${START_JOINT_HIGH}" ]] || [[ -n "${START_JOINT_LOW}" && -z "${START_JOINT_HIGH}" ]]; then
    echo "[Franka eval] FRANKA_START_JOINT_LOW and FRANKA_START_JOINT_HIGH must be provided together." >&2
    exit 1
fi

show_cameras_flag=()
case "${SHOW_CAMERAS,,}" in
    true|1|yes|y|on)
        show_cameras_flag=(--show-cameras)
        ;;
    false|0|no|n|off)
        show_cameras_flag=(--no-show-cameras)
        ;;
    *)
        echo "[Franka eval] Invalid FRANKA_SHOW_CAMERAS value: ${SHOW_CAMERAS}" >&2
        exit 1
        ;;
esac

save_video_flag=()
case "${SAVE_VIDEO,,}" in
    true|1|yes|y|on)
        save_video_flag=(--save-video)
        ;;
    false|0|no|n|off)
        save_video_flag=(--no-save-video)
        ;;
    *)
        echo "[Franka eval] Invalid FRANKA_SAVE_VIDEO value: ${SAVE_VIDEO}" >&2
        exit 1
        ;;
esac

debug_predictions_flag=()
case "${DEBUG_PREDICTIONS,,}" in
    true|1|yes|y|on)
        debug_predictions_flag=(--debug-predictions)
        ;;
    false|0|no|n|off)
        debug_predictions_flag=(--no-debug-predictions)
        ;;
    *)
        echo "[Franka eval] Invalid FRANKA_DEBUG_PREDICTIONS value: ${DEBUG_PREDICTIONS}" >&2
        exit 1
        ;;
esac

case "${DEBUG_IMAGE_LAYOUT,,}" in
    stitched|separate|both)
        ;;
    *)
        echo "[Franka eval] Invalid FRANKA_DEBUG_IMAGE_LAYOUT value: ${DEBUG_IMAGE_LAYOUT}" >&2
        echo "[Franka eval] Expected one of: stitched, separate, both" >&2
        exit 1
        ;;
esac

debug_apply_model_image_aug_flag=()
case "${DEBUG_APPLY_MODEL_IMAGE_AUG,,}" in
    true|1|yes|y|on)
        debug_apply_model_image_aug_flag=(--debug-apply-model-image-aug)
        ;;
    false|0|no|n|off)
        debug_apply_model_image_aug_flag=(--no-debug-apply-model-image-aug)
        ;;
    *)
        echo "[Franka eval] Invalid FRANKA_DEBUG_APPLY_MODEL_IMAGE_AUG value: ${DEBUG_APPLY_MODEL_IMAGE_AUG}" >&2
        exit 1
        ;;
esac

lock_rpy_actions_flag=()
case "${LOCK_RPY_ACTIONS,,}" in
    true|1|yes|y|on)
        lock_rpy_actions_flag=(--lock-rpy-actions)
        ;;
    false|0|no|n|off)
        lock_rpy_actions_flag=(--no-lock-rpy-actions)
        ;;
    *)
        echo "[Franka eval] Invalid FRANKA_LOCK_RPY_ACTIONS value: ${LOCK_RPY_ACTIONS}" >&2
        exit 1
        ;;
esac

gripper_release_confirm_flag=()
if [[ -n "${GRIPPER_RELEASE_CONFIRM_STEPS}" ]]; then
    gripper_release_confirm_flag=(--gripper-release-confirm-steps "${GRIPPER_RELEASE_CONFIRM_STEPS}")
fi

start_gripper_width_flag=()
if [[ -n "${START_GRIPPER_WIDTH}" ]]; then
    start_gripper_width_flag=(--start-gripper-width "${START_GRIPPER_WIDTH}")
fi

gripper_chunk_lookahead_close_flag=()
case "${GRIPPER_CHUNK_LOOKAHEAD_CLOSE,,}" in
    true|1|yes|y|on)
        gripper_chunk_lookahead_close_flag=(--gripper-chunk-lookahead-close)
        ;;
    false|0|no|n|off)
        gripper_chunk_lookahead_close_flag=(--no-gripper-chunk-lookahead-close)
        ;;
    *)
        echo "[Franka eval] Invalid FRANKA_GRIPPER_CHUNK_LOOKAHEAD_CLOSE value: ${GRIPPER_CHUNK_LOOKAHEAD_CLOSE}" >&2
        exit 1
        ;;
esac

gripper_chunk_lookahead_flag=()
case "${GRIPPER_CHUNK_LOOKAHEAD_RELEASE,,}" in
    true|1|yes|y|on)
        gripper_chunk_lookahead_flag=(--gripper-chunk-lookahead-release)
        ;;
    false|0|no|n|off)
        gripper_chunk_lookahead_flag=(--no-gripper-chunk-lookahead-release)
        ;;
    *)
        echo "[Franka eval] Invalid FRANKA_GRIPPER_CHUNK_LOOKAHEAD_RELEASE value: ${GRIPPER_CHUNK_LOOKAHEAD_RELEASE}" >&2
        exit 1
        ;;
esac

gripper_chunk_lookahead_min_offset_flag=()
if [[ -n "${GRIPPER_CHUNK_LOOKAHEAD_MIN_OFFSET}" ]]; then
    gripper_chunk_lookahead_min_offset_flag=(
        --gripper-chunk-lookahead-min-offset "${GRIPPER_CHUNK_LOOKAHEAD_MIN_OFFSET}"
    )
fi

target_z_floor_flags=()
if [[ -n "${MIN_TARGET_Z}" ]]; then
    target_z_floor_flags+=(--min-target-z "${MIN_TARGET_Z}")
fi
if [[ -n "${CLOSED_GRIPPER_MIN_TARGET_Z}" ]]; then
    target_z_floor_flags+=(--closed-gripper-min-target-z "${CLOSED_GRIPPER_MIN_TARGET_Z}")
fi
if [[ -n "${BUILD_TASK_MIN_TARGET_Z}" ]]; then
    target_z_floor_flags+=(--build-task-min-target-z "${BUILD_TASK_MIN_TARGET_Z}")
fi
if [[ -n "${BUILD_TASK_CLOSED_GRIPPER_MIN_TARGET_Z}" ]]; then
    target_z_floor_flags+=(--build-task-closed-gripper-min-target-z "${BUILD_TASK_CLOSED_GRIPPER_MIN_TARGET_Z}")
fi

default_prompt_flag=()
if [[ -n "${DEFAULT_PROMPT}" ]]; then
    default_prompt_flag=(--default-prompt "${DEFAULT_PROMPT}")
fi

camera_flags=()
if [[ -n "${EXTERNAL_CAMERA}" ]]; then
    camera_flags+=(--external-camera "${EXTERNAL_CAMERA}")
fi
if [[ -n "${WRIST_CAMERA}" ]]; then
    camera_flags+=(--wrist-camera "${WRIST_CAMERA}")
fi
if [[ -n "${LEFT_CAMERA}" ]]; then
    camera_flags+=(--left-camera "${LEFT_CAMERA}")
fi

start_joint_flags=()
if [[ -n "${START_JOINT_LOW}" ]]; then
    start_joint_flags=(
        --start-joint-low "${START_JOINT_LOW}"
        --start-joint-high "${START_JOINT_HIGH}"
        --start-joint-move-time "${START_JOINT_MOVE_TIME}"
        --start-joint-settle-time "${START_JOINT_SETTLE_TIME}"
    )
fi

exec "${launcher[@]}" -m cosmos_policy.experiments.robot.franka.run_franka_eval \
    --nuc-ip "${NUC_IP}" \
    --nuc-port "${NUC_PORT}" \
    "${camera_flags[@]}" \
    --control-mode "${CONTROL_MODE}" \
    --server-ip "${SERVER_IP}" \
    --server-port "${SERVER_PORT}" \
    --max-timesteps "${MAX_TIMESTEPS}" \
    --debug-log-dir "${DEBUG_LOG_DIR}" \
    --debug-image-layout "${DEBUG_IMAGE_LAYOUT,,}" \
    --policy-action-mode "${POLICY_ACTION_MODE}" \
    --policy-gripper-proprio-mode "${POLICY_GRIPPER_PROPRIO_MODE}" \
    --policy-gripper-positive-action "${POLICY_GRIPPER_POSITIVE_ACTION}" \
    "${start_gripper_width_flag[@]}" \
    "${gripper_release_confirm_flag[@]}" \
    "${gripper_chunk_lookahead_close_flag[@]}" \
    "${gripper_chunk_lookahead_flag[@]}" \
    --gripper-chunk-lookahead-confirm-steps "${GRIPPER_CHUNK_LOOKAHEAD_CONFIRM_STEPS}" \
    "${gripper_chunk_lookahead_min_offset_flag[@]}" \
    --open-loop-horizon "${OPEN_LOOP_HORIZON}" \
    --delta-action-base "${DELTA_ACTION_BASE}" \
    "${lock_rpy_actions_flag[@]}" \
    --lock-rpy-action-axes "${LOCK_RPY_ACTION_AXES}" \
    --max-target-position-lead "${MAX_TARGET_POSITION_LEAD}" \
    --max-target-vertical-lead "${MAX_TARGET_VERTICAL_LEAD}" \
    "${target_z_floor_flags[@]}" \
    "${show_cameras_flag[@]}" \
    "${save_video_flag[@]}" \
    "${debug_predictions_flag[@]}" \
    "${debug_apply_model_image_aug_flag[@]}" \
    "${default_prompt_flag[@]}" \
    "${start_joint_flags[@]}" \
    "$@"
