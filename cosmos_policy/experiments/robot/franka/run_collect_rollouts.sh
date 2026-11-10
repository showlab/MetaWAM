#!/usr/bin/env bash
# Collect on-policy Franka rollouts into a LeRobot parquet dataset.
#
# Mirrors run_franka_eval.sh for the robot/camera/policy plumbing, but invokes
# collect_rollouts.py instead — every rollout is buffered and saved (or
# discarded) per the operator's keyboard label.
#
# During a rollout, single keypresses control termination + labelling without
# Enter:
#   s = stop and save as SUCCESS
#   f = stop and save as FAILURE
#   d = stop and DISCARD
#   q = stop and ask interactively for the label
# If max_timesteps is reached first, the operator is prompted normally.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# --- Collector-specific knobs --------------------------------------------------
OUTPUT_ROOT="${FRANKA_OUTPUT_ROOT:-lerobot_rollouts}"
TASK_NAME="${FRANKA_TASK_NAME:-sweep}"
TASK_PRESET="${FRANKA_TASK_PRESET:-${TASK_NAME##*/}}"
NUM_ROLLOUTS="${FRANKA_NUM_ROLLOUTS:--1}"
# 15 Hz matches the teleop training control rate. The actual achievable rate
# still depends on camera + server inference latency.
CONTROL_FREQUENCY="${FRANKA_CONTROL_FREQUENCY:-15}"
# IMPORTANT for action-conditioned real-world training:
#   default to TARGET_HZ=0 so collect_rollouts.py writes one row per real
#   executed control step (no interpolation / no online resampling). After
#   collection, run retime_existing_rollouts.py offline to rewrite timestamps
#   to frame_index/15 while keeping the real step sequence.
TARGET_HZ="${FRANKA_TARGET_HZ:-0}"
# Only relevant when TARGET_HZ>0. With TARGET_HZ=0, raw frames already go
# directly to data/.
SAVE_RAW_TOO="${FRANKA_SAVE_RAW_TOO:-False}"
SAVE_WRIST_VIS="${FRANKA_SAVE_WRIST_VIS:-True}"
WRIST_VIS_FPS="${FRANKA_WRIST_VIS_FPS:-15}"

# Training-set start pose ranges, format: [x, y, z, roll, pitch, yaw].
# These are documented here for reference; collection uses the conservative
# joint ranges below to avoid adding online IK into the rollout path.
# cup_eef_low         = [0.345863178, -0.004296688, 0.352546722, 3.123013616, -0.007932360, -0.821529061]
# cup_eef_high        = [0.370646805,  0.008361207, 0.383113861, 3.149672453,  0.009553927, -0.785255134]
# build_block_eef_low = [0.338094312, -0.007675394, 0.333402783, 3.117949581, -0.003517302, -0.836524022]
# build_block_eef_high= [0.356436336,  0.009864779, 0.364659375, 3.146338431,  0.024911501, -0.799621832]
# sweep_eef_low       = [0.346927208, -0.003977372, 0.359197432, 3.118314314, -0.008967329, -0.823492074]
# sweep_eef_high      = [0.368510604,  0.016276091, 0.389380610, 3.150667302,  0.019456204, -0.781987369]
case "${TASK_PRESET,,}" in
    cup|d_cup)
        DEFAULT_PROMPT_VALUE="Place the cup on the green coaster"
        DEFAULT_START_JOINT_LOW="0.013538199, -0.607440637, -0.024424146, -2.810219556, -0.004018538, 2.206003447, 0.778393686"
        DEFAULT_START_JOINT_HIGH="0.027622807, -0.489711746, 0.006686491, -2.725664685, 0.019999191, 2.314832191, 0.824614995"
        DEFAULT_POST_ROLLOUT_HOME_BASE="0.020580503, -0.548576192, -0.008868828, -2.767942121, 0.007990327, 2.260417819, 0.801504341"
        DEFAULT_MIN_TARGET_Z="0.14"
        ;;
    build|build_block|d_build)
        DEFAULT_PROMPT_VALUE="Place the orange block on top of the gray block"
        DEFAULT_START_JOINT_LOW="0.006574900, -0.598931382, -0.025648478, -2.842746106, 0.003983285, 2.234394525, 0.782927812"
        DEFAULT_START_JOINT_HIGH="0.028517040, -0.479439665, 0.008107997, -2.794380222, 0.024404978, 2.361698553, 0.830549637"
        DEFAULT_POST_ROLLOUT_HOME_BASE="0.017545970, -0.539185524, -0.008770241, -2.818563164, 0.014194132, 2.298046539, 0.806738725"
        DEFAULT_MIN_TARGET_Z="0.14"
        ;;
    sweep|d_sweep)
        DEFAULT_PROMPT_VALUE="Sweep the green block into the dustpan"
        # Conservative joint-space sweep start range corresponding to the
        # reference EEF box above; preferred over online IK during collection.
        DEFAULT_START_JOINT_LOW="0.012812506, -0.603866181, -0.017325138, -2.793199597, -0.000767974, 2.164334780, 0.779078141"
        DEFAULT_START_JOINT_HIGH="0.039841094, -0.498352482, 0.015791032, -2.718271034, 0.030567631, 2.278225735, 0.834738195"
        DEFAULT_POST_ROLLOUT_HOME_BASE="0.027195957, -0.556973323, 0.000403217, -2.758499111, 0.015229879, 2.227127945, 0.813260110"
        DEFAULT_MIN_TARGET_Z="0.155"
        ;;
    *)
        DEFAULT_PROMPT_VALUE=""
        DEFAULT_START_JOINT_LOW=""
        DEFAULT_START_JOINT_HIGH=""
        DEFAULT_POST_ROLLOUT_HOME_BASE=""
        DEFAULT_MIN_TARGET_Z="0.14"
        ;;
esac
DEFAULT_PROMPT="${FRANKA_DEFAULT_PROMPT-${DEFAULT_PROMPT_VALUE}}"
START_JOINT_LOW="${FRANKA_START_JOINT_LOW-${DEFAULT_START_JOINT_LOW}}"
START_JOINT_HIGH="${FRANKA_START_JOINT_HIGH-${DEFAULT_START_JOINT_HIGH}}"
START_JOINT_MOVE_TIME="${FRANKA_START_JOINT_MOVE_TIME:-3.0}"
START_JOINT_SETTLE_TIME="${FRANKA_START_JOINT_SETTLE_TIME:-0.5}"

POST_ROLLOUT_HOME_BASE="${FRANKA_POST_ROLLOUT_HOME_BASE-${DEFAULT_POST_ROLLOUT_HOME_BASE}}"
POST_ROLLOUT_HOME_JITTER="${FRANKA_POST_ROLLOUT_HOME_JITTER:-0.03}"
POST_ROLLOUT_HOME_TIME="${FRANKA_POST_ROLLOUT_HOME_TIME:-3.0}"

# --- Plumbing copied from run_franka_eval.sh -----------------------------------
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
POLICY_ACTION_MODE="${FRANKA_POLICY_ACTION_MODE:-delta_eef_7d}"
POLICY_GRIPPER_PROPRIO_MODE="${FRANKA_POLICY_GRIPPER_PROPRIO_MODE:-measured_width_signed}"
POLICY_GRIPPER_POSITIVE_ACTION="${FRANKA_POLICY_GRIPPER_POSITIVE_ACTION:-open}"
GRIPPER_RELEASE_CONFIRM_STEPS="${FRANKA_GRIPPER_RELEASE_CONFIRM_STEPS:-2}"
GRIPPER_CHUNK_LOOKAHEAD_RELEASE="${FRANKA_GRIPPER_CHUNK_LOOKAHEAD_RELEASE:-True}"
GRIPPER_CHUNK_LOOKAHEAD_CONFIRM_STEPS="${FRANKA_GRIPPER_CHUNK_LOOKAHEAD_CONFIRM_STEPS:-2}"
GRIPPER_CHUNK_LOOKAHEAD_MIN_OFFSET="${FRANKA_GRIPPER_CHUNK_LOOKAHEAD_MIN_OFFSET:-2}"
# The real-world model action chunk is 16 steps, so open_loop_horizon must not
# exceed 16 unless the server/model is changed to output longer action chunks.
# Default to 10 for rollout collection: it is much more closed-loop than 16 and
# empirically improves success rate. The unexecuted second
# half of each predicted chunk is simply discarded; the parquet still stores
# the actual executed action sequence, and retime_existing_rollouts.py later
# rewrites timestamps without interpolation.
OPEN_LOOP_HORIZON="${FRANKA_OPEN_LOOP_HORIZON:-10}"
DELTA_ACTION_BASE="${FRANKA_DELTA_ACTION_BASE:-target}"
LOCK_RPY_ACTIONS="${FRANKA_LOCK_RPY_ACTIONS:-False}"
LOCK_RPY_ACTION_AXES="${FRANKA_LOCK_RPY_ACTION_AXES:-}"
MAX_TARGET_POSITION_LEAD="${FRANKA_MAX_TARGET_POSITION_LEAD:-0.15}"
MAX_TARGET_VERTICAL_LEAD="${FRANKA_MAX_TARGET_VERTICAL_LEAD:-0.05}"
MIN_TARGET_Z="${FRANKA_MIN_TARGET_Z:-${DEFAULT_MIN_TARGET_Z}}"
CLOSED_GRIPPER_MIN_TARGET_Z="${FRANKA_CLOSED_GRIPPER_MIN_TARGET_Z:-0.14}"
BUILD_TASK_MIN_TARGET_Z="${FRANKA_BUILD_TASK_MIN_TARGET_Z:-0.14}"
BUILD_TASK_CLOSED_GRIPPER_MIN_TARGET_Z="${FRANKA_BUILD_TASK_CLOSED_GRIPPER_MIN_TARGET_Z:-0.18}"
CLOSE_GRIPPER_ON_START="${FRANKA_CLOSE_GRIPPER_ON_START:-False}"
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

echo "[Franka collect] repo root:    ${REPO_ROOT}"
echo "[Franka collect] launcher:     ${launcher[*]}"
echo "[Franka collect] output root:  ${OUTPUT_ROOT}"
echo "[Franka collect] task name:    ${TASK_NAME}"
echo "[Franka collect] task preset:  ${TASK_PRESET}"
echo "[Franka collect] default task: ${DEFAULT_PROMPT}"
echo "[Franka collect] control freq: ${CONTROL_FREQUENCY} Hz"
echo "[Franka collect] target hz:    ${TARGET_HZ} (0 = no resample)"
echo "[Franka collect] save raw too: ${SAVE_RAW_TOO}"
echo "[Franka collect] open loop:    ${OPEN_LOOP_HORIZON} (must be <= model action chunk length, currently 16)"
echo "[Franka collect] gripper release confirm: ${GRIPPER_RELEASE_CONFIRM_STEPS} step(s)"
echo "[Franka collect] gripper lookahead: release=${GRIPPER_CHUNK_LOOKAHEAD_RELEASE}, confirm=${GRIPPER_CHUNK_LOOKAHEAD_CONFIRM_STEPS}, min_offset=${GRIPPER_CHUNK_LOOKAHEAD_MIN_OFFSET}"
echo "[Franka collect] num rollouts: ${NUM_ROLLOUTS} (-1 = until quit)"
echo "[Franka collect] wrist vis:    ${SAVE_WRIST_VIS} (${WRIST_VIS_FPS} FPS)"
if [[ -n "${START_JOINT_LOW}" || -n "${START_JOINT_HIGH}" ]]; then
    echo "[Franka collect] start joint low:  ${START_JOINT_LOW}"
    echo "[Franka collect] start joint high: ${START_JOINT_HIGH}"
    echo "[Franka collect] start move:       ${START_JOINT_MOVE_TIME}s (+ settle ${START_JOINT_SETTLE_TIME}s)"
else
    echo "[Franka collect] start joint: disabled"
fi
if [[ -n "${POST_ROLLOUT_HOME_BASE}" ]]; then
    echo "[Franka collect] post-home:    double-check after continue prompt (+/-${POST_ROLLOUT_HOME_JITTER})"
else
    echo "[Franka collect] post-home:    disabled"
fi

if [[ -z "${START_JOINT_LOW}" && -n "${START_JOINT_HIGH}" ]] || [[ -n "${START_JOINT_LOW}" && -z "${START_JOINT_HIGH}" ]]; then
    echo "[Franka collect] FRANKA_START_JOINT_LOW and FRANKA_START_JOINT_HIGH must be provided together." >&2
    exit 1
fi

if ! [[ "${OPEN_LOOP_HORIZON}" =~ ^[0-9]+$ ]]; then
    echo "[Franka collect] Invalid FRANKA_OPEN_LOOP_HORIZON value: ${OPEN_LOOP_HORIZON}" >&2
    exit 1
fi
if (( OPEN_LOOP_HORIZON > 16 )); then
    echo "[Franka collect] FRANKA_OPEN_LOOP_HORIZON=${OPEN_LOOP_HORIZON} exceeds current model action chunk length 16." >&2
    echo "[Franka collect] Refusing to run; use <=16 or change the server/model to output longer chunks." >&2
    exit 1
fi

show_cameras_flag=()
case "${SHOW_CAMERAS,,}" in
    true|1|yes|y|on)  show_cameras_flag=(--show-cameras) ;;
    false|0|no|n|off) show_cameras_flag=(--no-show-cameras) ;;
    *)
        echo "[Franka collect] Invalid FRANKA_SHOW_CAMERAS value: ${SHOW_CAMERAS}" >&2
        exit 1
        ;;
esac

save_wrist_vis_flag=()
case "${SAVE_WRIST_VIS,,}" in
    true|1|yes|y|on)  save_wrist_vis_flag=(--save-wrist-vis) ;;
    false|0|no|n|off) save_wrist_vis_flag=(--no-save-wrist-vis) ;;
    *)
        echo "[Franka collect] Invalid FRANKA_SAVE_WRIST_VIS value: ${SAVE_WRIST_VIS}" >&2
        exit 1
        ;;
esac

save_raw_too_flag=()
case "${SAVE_RAW_TOO,,}" in
    true|1|yes|y|on)  save_raw_too_flag=(--save-raw-too) ;;
    false|0|no|n|off) save_raw_too_flag=(--no-save-raw-too) ;;
    *)
        echo "[Franka collect] Invalid FRANKA_SAVE_RAW_TOO value: ${SAVE_RAW_TOO}" >&2
        exit 1
        ;;
esac

lock_rpy_actions_flag=()
case "${LOCK_RPY_ACTIONS,,}" in
    true|1|yes|y|on)  lock_rpy_actions_flag=(--lock-rpy-actions) ;;
    false|0|no|n|off) lock_rpy_actions_flag=(--no-lock-rpy-actions) ;;
    *)
        echo "[Franka collect] Invalid FRANKA_LOCK_RPY_ACTIONS value: ${LOCK_RPY_ACTIONS}" >&2
        exit 1
        ;;
esac

gripper_chunk_lookahead_flag=()
case "${GRIPPER_CHUNK_LOOKAHEAD_RELEASE,,}" in
    true|1|yes|y|on)  gripper_chunk_lookahead_flag=(--gripper-chunk-lookahead-release) ;;
    false|0|no|n|off) gripper_chunk_lookahead_flag=(--no-gripper-chunk-lookahead-release) ;;
    *)
        echo "[Franka collect] Invalid FRANKA_GRIPPER_CHUNK_LOOKAHEAD_RELEASE value: ${GRIPPER_CHUNK_LOOKAHEAD_RELEASE}" >&2
        exit 1
        ;;
esac

close_gripper_flag=()
case "${CLOSE_GRIPPER_ON_START,,}" in
    true|1|yes|y|on)  close_gripper_flag=(--close-gripper-on-start) ;;
    false|0|no|n|off) close_gripper_flag=(--no-close-gripper-on-start) ;;
    *)
        echo "[Franka collect] Invalid FRANKA_CLOSE_GRIPPER_ON_START value: ${CLOSE_GRIPPER_ON_START}" >&2
        exit 1
        ;;
esac

gripper_release_confirm_flag=()
if [[ -n "${GRIPPER_RELEASE_CONFIRM_STEPS}" ]]; then
    gripper_release_confirm_flag=(--gripper-release-confirm-steps "${GRIPPER_RELEASE_CONFIRM_STEPS}")
fi

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

post_rollout_home_flags=()
if [[ -n "${POST_ROLLOUT_HOME_BASE}" ]]; then
    post_rollout_home_flags=(
        --post-rollout-home-base "${POST_ROLLOUT_HOME_BASE}"
        --post-rollout-home-jitter "${POST_ROLLOUT_HOME_JITTER}"
        --post-rollout-home-time "${POST_ROLLOUT_HOME_TIME}"
    )
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

exec "${launcher[@]}" -m cosmos_policy.experiments.robot.franka.collect_rollouts \
    --output-root "${OUTPUT_ROOT}" \
    --task-name "${TASK_NAME}" \
    "${save_wrist_vis_flag[@]}" \
    --wrist-vis-fps "${WRIST_VIS_FPS}" \
    --target-hz "${TARGET_HZ}" \
    "${save_raw_too_flag[@]}" \
    --num-rollouts "${NUM_ROLLOUTS}" \
    --control-frequency "${CONTROL_FREQUENCY}" \
    --nuc-ip "${NUC_IP}" \
    --nuc-port "${NUC_PORT}" \
    "${camera_flags[@]}" \
    --control-mode "${CONTROL_MODE}" \
    --server-ip "${SERVER_IP}" \
    --server-port "${SERVER_PORT}" \
    --max-timesteps "${MAX_TIMESTEPS}" \
    --policy-action-mode "${POLICY_ACTION_MODE}" \
    --policy-gripper-proprio-mode "${POLICY_GRIPPER_PROPRIO_MODE}" \
    --policy-gripper-positive-action "${POLICY_GRIPPER_POSITIVE_ACTION}" \
    "${gripper_release_confirm_flag[@]}" \
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
    "${close_gripper_flag[@]}" \
    "${default_prompt_flag[@]}" \
    "${start_joint_flags[@]}" \
    "${post_rollout_home_flags[@]}" \
    "$@"
