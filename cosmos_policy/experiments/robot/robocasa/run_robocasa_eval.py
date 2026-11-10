# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
run_robocasa_eval.py

Evaluates a trained policy in a RoboCasa simulation benchmark task suite.

Parallel inference:
    (Only applicable when doing best-of-N search with N GPUs)
    To enable parallel inference across 8 GPUs, use:
        --use_parallel_inference True
        --num_queries_best_of_n 8
        --available_gpus "0,1,2,3,4,5,6,7"

Usage example (MetaWAM checkpoint; see ROBOCASA.md and scripts/test_robocasa.sh):
    #   Replace `task_name` with a RoboCasa task (e.g. TurnOffMicrowave), `pnp` or `all`
    #   Replace `seed` with one of {195, 196, 197}
    #   Replace `run_id_note` / `rollout_prefix` with unique identifiers for the run
    #   `ckpt_path` may also be a Hugging Face repo id, and the stats / T5 paths `<org>/<repo>/<file>`
    uv run --extra cu128 --group robocasa --python 3.10 \
      python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
        --config metawam_robocasa__inference \
        --ckpt_path checkpoints/MetaWAM-RoboCasa-Predict2-2B/MetaWAM-RoboCasa-Predict2-2B.pt \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --num_wrist_images 1 \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path checkpoints/MetaWAM-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
        --t5_text_embeddings_path checkpoints/MetaWAM-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 32 \
        --num_open_loop_steps 16 \
        --task_name TurnOffMicrowave \
        --num_trials_per_task 50 \
        --run_id_note metawam_robocasa--5stepAct--seed195--deterministic \
        --local_log_dir logs \
        --seed 195 \
        --randomize_seed False \
        --deterministic True \
        --use_variance_scale False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --data_collection False \
        --rollout_prefix metawam_robocasa_seed195 \
        --num_parallel_envs 10 \
        --inference_batch_timeout 0.05

"""

import ast
import copy
import multiprocessing as mp
import os
import re
import sys
import traceback
import pickle
import queue as _stdlib_queue
import secrets
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import h5py
import numpy as np
import robosuite
import torch
import wandb
# Importing RoboCasa's dataset registry registers RoboCasa environments with robosuite.
from robocasa.utils.dataset_registry import MULTI_STAGE_TASK_DATASETS, SINGLE_STAGE_TASK_DATASETS  # noqa: F401

from cosmos_policy.experiments.robot.cosmos_utils import (
    ACTION_DIM,
    WorkerPoolManager,
    extract_action_chunk,
    get_action,
    get_future_state_prediction,
    get_model,
    get_planning_model,
    get_qvalue_prediction,
    get_value_prediction,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    query_model_parallel,
    unnormalize_actions,
)
from cosmos_policy.experiments.robot.robocasa.robocasa_utils import (
    save_rollout_video,
    save_rollout_video_with_future_image_predictions,
)
from cosmos_policy.experiments.robot.robot_utils import DATE_TIME, log_message, setup_logging
from cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere

# Path to fixed controller configs
CONTROLLER_CONFIGS_PATH: str = "cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl"

# Define max steps for each RoboCasa task (based on horizons in dataset_registry)
TASK_MAX_STEPS = {
    # Pick and place tasks
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    # Door tasks
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    # Drawer tasks
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    # Stove tasks
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    # Sink tasks
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    # Coffee tasks
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    # Microwave tasks
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}


def _resolve_local_log_dir(local_log_dir: str) -> str:
    """Return an absolute path for eval logs and rollout videos.

    Resolution order:
    1. ``COSMOS_POLICY_LOG_DIR`` or ``ROBOCASA_LOCAL_LOG_DIR`` (if set): use as the full log directory.
    2. If ``local_log_dir`` is absolute: use it.
    3. If ``COSMOS_POLICY_ROBOCASA_DIR`` is set: ``join(ROBOCASA_DIR, local_log_dir)`` (relative paths).
    4. Else: ``join(dirname(run_robocasa_eval.py), local_log_dir)`` (not cwd).

    In Apptainer/Docker, ``/workspace`` only matches your repo if the job was started with the repo as
    ``--pwd`` / ``cwd``. If not, set ``COSMOS_POLICY_LOG_DIR`` to your host absolute path.
    """
    env_direct = os.environ.get("COSMOS_POLICY_LOG_DIR") or os.environ.get("ROBOCASA_LOCAL_LOG_DIR")
    if env_direct:
        return str(Path(env_direct).expanduser().resolve())
    p = Path(local_log_dir).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    anchor = os.environ.get("COSMOS_POLICY_ROBOCASA_DIR")
    if anchor:
        return str((Path(anchor).expanduser().resolve() / p).resolve())
    return str((Path(__file__).resolve().parent / p).resolve())


def _rollout_videos_parent(local_log_dir: str, rollout_prefix: str) -> str:
    """Parent of per-task dirs ``<task>--<timestamp>/`` (rollout MP4s live inside those)."""
    return os.path.join(local_log_dir, rollout_prefix.strip())


_OUTPUT_POINTER_NAME = "ROBOCASA_EVAL_OUTPUT_POINTER.txt"


def _write_output_pointer_files(local_log_dir: str, rollout_prefix: str = "rollout_data") -> tuple[str, Optional[str]]:
    """Write plain-text pointers so you always know where logs/videos went.

    1) Next to ``run_robocasa_eval.py`` (same folder as this file).
    2) Current working directory (often your repo root when you submit Slurm).
    """
    rollout_base = _rollout_videos_parent(local_log_dir, rollout_prefix)
    ts = os.environ.get("COSMOS_ROBOCASA_ROLLOUT_TS", "")
    body = "\n".join(
        [
            "RoboCasa eval — where outputs go for this run",
            "",
            f"LOG_DIR={local_log_dir}",
            f"ROLLOUT_VIDEOS={rollout_base}",
            f"SESSION_TS={ts}",
            "",
            f"Each task: {rollout_base}/<TaskName>--{ts}/",
            "",
            "Override directory: export COSMOS_POLICY_LOG_DIR=/absolute/path/to/logs",
            "",
        ]
    )
    script_dir = Path(__file__).resolve().parent
    path_a = script_dir / _OUTPUT_POINTER_NAME
    path_a.write_text(body + "\n", encoding="utf-8")
    path_b_str: Optional[str] = None
    try:
        path_b = Path.cwd() / _OUTPUT_POINTER_NAME
        path_b.write_text(body + "\n", encoding="utf-8")
        path_b_str = str(path_b.resolve())
    except OSError:
        pass
    return str(path_a.resolve()), path_b_str


def _write_where_outputs_are_file(local_log_dir: str, rollout_prefix: str = "rollout_data") -> str:
    """Drop a small text file so jobs can find outputs on shared filesystem."""
    path = os.path.join(local_log_dir, "WHERE_EVAL_OUTPUTS_ARE.txt")
    rp = rollout_prefix.strip()
    lines = [
        "RoboCasa eval writes text logs and rollout MP4s under this directory.",
        "",
        f"Absolute path (this machine / container): {local_log_dir}",
        f"Rollout MP4s: {os.path.join(local_log_dir, rp, '<task>--<timestamp>')}",
        "",
        "If you use Apptainer/Docker: this path is INSIDE the container.",
        "It matches your host only where you bind-mount (e.g. --bind /path/to/cosmos-policy:/workspace).",
        "Your Slurm job cwd must be the repo root if you use $(pwd):/workspace.",
        "",
        "Override output location without editing CLI:",
        "  export COSMOS_POLICY_LOG_DIR=/absolute/path/to/logs",
        "  or export COSMOS_POLICY_ROBOCASA_DIR=/absolute/path/to/cosmos_policy/experiments/robot/robocasa",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def _rollout_session_timestamp() -> str:
    """Timestamp string for ``<rollout_prefix>/<task>--<ts>/``; shared with spawn workers via env."""
    return os.environ.get("COSMOS_ROBOCASA_ROLLOUT_TS") or DATE_TIME


def _append_to_eval_log(eval_log_txt_path: Optional[str], line: str) -> None:
    """Append one line to the main ENV_EVAL log from worker processes (locked for parallel workers)."""
    if not eval_log_txt_path:
        return
    try:
        import fcntl

        with open(eval_log_txt_path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line.rstrip() + "\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _resolve_resume_log_path(resume_log_path: Optional[str], local_log_dir: str) -> Optional[str]:
    """Resolve an existing ENV_EVAL log path used for resume/appending."""
    if not resume_log_path:
        return None
    p = Path(resume_log_path).expanduser()
    if not p.is_absolute():
        p = Path(local_log_dir) / p
    return str(p.resolve())


def _resume_rollout_timestamp(resume_log_path: Optional[str]) -> Optional[str]:
    """Extract the run timestamp from an ENV_EVAL/DATA_COLLECTION log filename."""
    if not resume_log_path:
        return None
    match = re.search(r"(?:ENV_EVAL|DATA_COLLECTION)-.*?-(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})", Path(resume_log_path).name)
    if not match:
        return None
    return match.group(1)


def _task_list_from_cfg(cfg: "PolicyEvalConfig") -> list[str]:
    all_tasks = {**TASK_MAX_STEPS}
    if cfg.task_name == "all":
        return list(all_tasks.keys())
    if cfg.task_name == "pnp":
        return [t for t in TASK_MAX_STEPS.keys() if t.startswith("PnP")]
    return [cfg.task_name]


def _parse_task_from_rollout_path(path_text: str) -> Optional[str]:
    """Extract task name from .../<TaskName>--<timestamp>/<file>.mp4 if possible."""
    try:
        parent_name = Path(path_text.strip()).parent.name
    except OSError:
        return None
    if "--" not in parent_name:
        return None
    task_name = parent_name.split("--", 1)[0]
    return task_name or None


def _parse_resume_log(log_path: Optional[str]) -> dict[str, dict]:
    """Parse an existing eval log into per-task completed episodes and summary stats.

    The returned dict is intentionally simple:
        task -> {
            "episodes": {episode_idx: (success, length_or_none)},
            "summary_successes": int | None,
            "summary_episodes": int | None,
            "summary_avg_length": float | None,
        }

    Existing older parallel logs may only have per-task summaries for early tasks and
    per-episode rollout paths for later tasks, so both formats are supported.
    """
    if not log_path or not os.path.exists(log_path):
        return {}

    state: dict[str, dict] = {}
    current_task: Optional[str] = None
    summary_task: Optional[str] = None

    def task_state(task_name: str) -> dict:
        return state.setdefault(
            task_name,
            {
                "episodes": {},
                "summary_successes": None,
                "summary_episodes": None,
                "summary_avg_length": None,
            },
        )

    task_start_re = re.compile(r"(?:Starting evaluation for task:|Evaluating task:)\s+(?P<task>[A-Za-z0-9_]+)")
    serial_episode_re = re.compile(
        r"^\s*Episode\s+(?P<ep>\d+):\s+(?P<status>SUCCESS|FAILURE)\s+\(length:\s+(?P<length>\d+)\)"
    )
    worker_done_re = re.compile(
        r"\[Worker\s+\d+\]\s+(?P<task>\S+)\s+ep(?P<ep>\d+):\s+"
        r"(?P<status>SUCCESS|FAILURE)\s+\(length=(?P<length>\d+)\)"
    )
    worker_rollout_re = re.compile(
        r"\[Worker\s+\d+\]\s+(?P<task>\S+)\s+ep(?P<ep>\d+)\s+saved rollout:.*?--success=(?P<success>True|False)"
    )
    saved_rollout_re = re.compile(
        r"Saved rollout MP4 at path\s+(?P<path>.*?--episode=(?P<ep>\d+)--success=(?P<success>True|False)--task=.*)"
    )
    task_results_re = re.compile(r"^Task\s+(?P<task>[A-Za-z0-9_]+)\s+results:")
    avg_length_re = re.compile(r"^\s*Average episode length:\s+(?P<avg>[0-9.]+)")
    successes_re = re.compile(r"^\s*Successes:\s+(?P<successes>\d+)\s*/\s*(?P<episodes>\d+)")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if match := task_start_re.search(line):
                current_task = match.group("task")

            if match := worker_done_re.search(line):
                task = match.group("task")
                task_state(task)["episodes"][int(match.group("ep"))] = (
                    match.group("status") == "SUCCESS",
                    int(match.group("length")),
                )
                continue

            if match := serial_episode_re.search(line):
                if current_task is not None:
                    task_state(current_task)["episodes"][int(match.group("ep"))] = (
                        match.group("status") == "SUCCESS",
                        int(match.group("length")),
                    )
                continue

            if match := worker_rollout_re.search(line):
                task = match.group("task")
                episode_idx = int(match.group("ep"))
                episodes = task_state(task)["episodes"]
                episodes.setdefault(episode_idx, (match.group("success") == "True", None))
                continue

            if match := saved_rollout_re.search(line):
                task = _parse_task_from_rollout_path(match.group("path")) or current_task
                if task is not None:
                    episode_idx = int(match.group("ep"))
                    episodes = task_state(task)["episodes"]
                    episodes.setdefault(episode_idx, (match.group("success") == "True", None))
                continue

            if match := task_results_re.search(line):
                summary_task = match.group("task")
                task_state(summary_task)
                continue

            if summary_task and (match := avg_length_re.search(line)):
                task_state(summary_task)["summary_avg_length"] = float(match.group("avg"))
                continue

            if summary_task and (match := successes_re.search(line)):
                summary = task_state(summary_task)
                summary["summary_successes"] = int(match.group("successes"))
                summary["summary_episodes"] = int(match.group("episodes"))
                continue

    return state


def _parse_resume_rollout_videos(local_log_dir: str, rollout_prefix: str) -> dict[str, dict]:
    """Build resume state from rollout video files, treating videos as the source of truth.

    Scans ``<local_log_dir>/<rollout_prefix>/<TaskName>--<timestamp>/`` for files named
    ``...--episode=<idx>--success=<True|False>--task=...``. Both MP4 and GIF fallback files
    count. If duplicate videos exist for the same task/episode, the newest file wins.
    """
    rollout_base = Path(_rollout_videos_parent(local_log_dir, rollout_prefix)).expanduser()
    if not rollout_base.exists():
        return {}

    video_re = re.compile(r"--episode=(?P<ep>\d+)--success=(?P<success>True|False)--task=")
    latest_by_episode: dict[str, dict[int, tuple[float, bool]]] = {}

    for task_dir in rollout_base.glob("*--*"):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name.split("--", 1)[0]
        if task_name not in TASK_MAX_STEPS:
            continue
        for video_path in task_dir.iterdir():
            if video_path.suffix.lower() not in {".mp4", ".gif"}:
                continue
            match = video_re.search(video_path.name)
            if not match:
                continue
            episode_idx = int(match.group("ep"))
            success = match.group("success") == "True"
            try:
                mtime = video_path.stat().st_mtime
            except OSError:
                continue
            task_episodes = latest_by_episode.setdefault(task_name, {})
            previous = task_episodes.get(episode_idx)
            if previous is None or mtime >= previous[0]:
                task_episodes[episode_idx] = (mtime, success)

    state: dict[str, dict] = {}
    for task_name, episodes in latest_by_episode.items():
        state[task_name] = {
            "episodes": {episode_idx: (success, None) for episode_idx, (_, success) in episodes.items()},
            "summary_successes": None,
            "summary_episodes": None,
            "summary_avg_length": None,
        }
    return state


def _resume_episode_results(resume_state: dict[str, dict], task_name: str) -> dict[int, tuple[bool, Optional[int]]]:
    return dict(resume_state.get(task_name, {}).get("episodes", {}))


def _requested_episode_indices(
    num_trials: int,
    episode_start_idx: int = 0,
    episode_stride: int = 1,
) -> list[int]:
    return [episode_start_idx + local_idx * episode_stride for local_idx in range(num_trials)]


def _is_task_complete_from_summary(resume_state: dict[str, dict], task_name: str, num_trials: int) -> bool:
    task_state = resume_state.get(task_name, {})
    summary_episodes = task_state.get("summary_episodes")
    return summary_episodes is not None and summary_episodes >= num_trials


def _is_task_complete_from_episodes(
    resume_state: dict[str, dict],
    task_name: str,
    num_trials: int,
    episode_start_idx: int = 0,
    episode_stride: int = 1,
) -> bool:
    completed = _resume_episode_results(resume_state, task_name)
    return all(
        episode_idx in completed
        for episode_idx in _requested_episode_indices(num_trials, episode_start_idx, episode_stride)
    )


def _pending_episode_indices(
    resume_state: dict[str, dict],
    task_name: str,
    num_trials: int,
    episode_start_idx: int = 0,
    episode_stride: int = 1,
) -> list[int]:
    completed = _resume_episode_results(resume_state, task_name)
    return [
        episode_idx
        for episode_idx in _requested_episode_indices(num_trials, episode_start_idx, episode_stride)
        if episode_idx not in completed
    ]


def _stats_from_resume_state(
    resume_state: dict[str, dict],
    task_name: str,
    num_trials: int,
    episode_start_idx: int = 0,
    episode_stride: int = 1,
) -> tuple[float, float, list[bool]]:
    task_state = resume_state.get(task_name, {})
    if not task_state.get("episodes") and _is_task_complete_from_summary(resume_state, task_name, num_trials):
        success_count = int(task_state.get("summary_successes") or 0)
        episode_count = int(task_state.get("summary_episodes") or 0)
        successes = [True] * success_count + [False] * max(0, episode_count - success_count)
        success_rate = (success_count / episode_count) if episode_count > 0 else 0.0
        avg_length = float(task_state.get("summary_avg_length") or 0.0)
        return success_rate, avg_length, successes

    episode_results = _resume_episode_results(resume_state, task_name)
    ordered = [
        episode_results[i]
        for i in _requested_episode_indices(num_trials, episode_start_idx, episode_stride)
        if i in episode_results
    ]
    successes = [success for success, _ in ordered]
    lengths = [length for _, length in ordered if length is not None]
    success_rate = float(np.mean(successes)) if successes else 0.0
    avg_length = float(np.mean(lengths)) if lengths else 0.0
    return success_rate, avg_length, successes


@dataclass
class PolicyEvalConfig:
    # fmt: off
    suite: str = "robocasa"                                              # Evaluation suite name

    #################################################################################################################
    # Cosmos Policy-specific parameters
    #################################################################################################################
    model_family: str = "cosmos"                                         # Model family
    config: str = "metawam_robocasa__inference"                          # Inference config name
    ckpt_path: str = ""                                                  # Pretrained checkpoint path
    planning_model_config_name: str = ""                                 # Planning model config name
    planning_model_ckpt_path: str = ""                                   # Planning model checkpoint path
    config_file: str = "cosmos_policy/config/config.py"  # Cosmos default config file path

    use_third_person_image: bool = True                                  # Whether to include third-person ("primary") image in input
    num_third_person_images: int = 2                                     # Number of third-person images to include in input (RoboCasa: 1 for left, 1 for right)
    use_wrist_image: bool = True                                         # Whether to include wrist image in input
    num_wrist_images: int = 1                                            # Number of wrist images to include in input (RoboCasa: 1 wrist image)
    use_proprio: bool = True                                             # Whether to include proprio state in input
    flip_images: bool = True                                             # Whether to flip images vertically across x-axis (RoboCasa: True because environment returns images upside down)
    use_variance_scale: bool = False                                     # Whether to scale variance used to sample sigma max for denoising for increased diversity in generations
    use_jpeg_compression: bool = True                                    # Whether to use JPEG compression on images before querying policy
    ar_future_prediction: bool = False                                   # Whether to predict future state autoregressively
    ar_value_prediction: bool = False                                    # Whether to predict future state value autoregressively
    ar_qvalue_prediction: bool = False                                   # Whether to predict Q-value autoregressively
    num_denoising_steps_action: int = 5                                  # Number of denoising steps to take for action prediction
    num_denoising_steps_future_state: int = 1                            # Number of denoising steps to take for future state prediction (only applicable if ar_future_prediction is True; otherwise equal to num_denoising_steps_action)
    num_denoising_steps_value: int = 1                                   # Number of denoising steps to take for value prediction (only applicable if ar_value_prediction is True; otherwise equal to num_denoising_steps_action)
    unnormalize_actions: bool = True                                     # Unnormalize actions if trained with normalized actions
    normalize_proprio: bool = True                                       # Normalize proprio input if trained with normalized proprio
    dataset_stats_path: str = ""                                         # Path to dataset statistics file for action unnormalization and proprio normalization
    t5_text_embeddings_path: str = ""                                    # Path to precomputed T5 text embeddings dictionary (key: instruction, val: embedding)
    trained_with_image_aug: bool = True                                  # Whether the model was trained with image augmentations (needed for test-time image transformations)
    chunk_size: int = 32                                                 # Number of actions to predict in chunk
    num_open_loop_steps: int = 16                                        # Number of actions in predicted chunk to execute open-loop before requerying policy

    deterministic: bool = True                                           # Whether to run in deterministic mode
    deterministic_reset: bool = False                                    # Whether to run in deterministic reset mode (sets global random seed right before env reset)
    deterministic_reset_seed: int = None                                 # (Only applicable if deterministic_reset==True) The seed to set before deterministic reset; if not provided, defaults to the base seed

    #################################################################################################################
    # Planning model and best-of-N search parameters
    #################################################################################################################
    use_ensemble_future_state_predictions: bool = False                  # Whether to use ensemble of future state predictions
    num_future_state_predictions_in_ensemble: int = 3                    # Number of future state predictions in ensemble
    future_state_ensemble_aggregation_scheme: str = "average"            # How to aggregate future state predictions in an ensemble of future state predictions (options: "average", "last")
    use_ensemble_value_predictions: bool = False                         # Whether to use ensemble of value predictions
    num_value_predictions_in_ensemble: int = 5                           # Number of value predictions in ensemble
    value_ensemble_aggregation_scheme: str = "average"                   # How to aggregate values in an ensemble of value predictions (options: "average", "lcb", "success_vote", "majority_mean")
    search_depth: int = 1                                                # Number of levels to search through in the best-of-N search tree
    search_depth_value_aggregation_scheme: str = "use_last_value"        # How to aggregate value predictions across search depth (options: use_last_value, average)
    mask_current_state_action_for_value_prediction: bool = False         # Whether to use input masking to mask out certain inputs (current state and action) during value prediction
    mask_future_state_for_qvalue_prediction: bool = False                # Whether to use input masking to mask out certain inputs (future state) during Q(s, a) value prediction

    num_queries_best_of_n: int = 1                                       # Number of queries to make to the model (this is the N in best-of-N search)
    use_parallel_inference: bool = False                                 # Whether to use parallel inference across multiple GPUs
    available_gpus: str = "0,1,2,3,4,5,6,7"                              # Comma-separated list of GPU IDs available for use for parallel inference (defaults to all 8 GPUs on a node)
    parallel_timeout: int = 15                                           # Timeout in seconds for each parallel query

    #################################################################################################################
    # RoboCasa-specific parameters
    #################################################################################################################
    task_name: str = "PnPCounterToCab"                                   # Task name (must be in SINGLE_STAGE_TASK_DATASETS or MULTI_STAGE_TASK_DATASETS)
    num_trials_per_task: int = 50                                        # Number of rollouts per task
    episode_start_idx: int = 0                                           # Global episode index to start from (for sharded trial execution)
    episode_stride: int = 1                                              # Episode index stride between local trials (for interleaved sharding)
    env_img_res: int = 224                                               # Resolution for rendering environment images
    robots: str = "PandaMobile"                                          # Robot type for RoboCasa (PandaMobile is alias for PandaOmron)
    controllers: str = "OSC_POSE"                                        # Controller type (OSC_POSE = Operational Space Control with 6-DOF end-effector pose)
    obj_instance_split: str = "B"                                        # Object instance split - "B" = held-out test objects
    layout_and_style_ids: str = "((1,1),(2,2),(4,4),(6,9),(7,10))"       # Layout and style IDs - 5 test scenes
    randomize_cameras: bool = False                                      # Whether to randomize camera positions

    #################################################################################################################
    # Utils
    #################################################################################################################
    local_log_dir: str = "logs"                                          # Eval logs; see _resolve_local_log_dir (env COSMOS_POLICY_LOG_DIR overrides)
    rollout_prefix: str = "rollout_data"                                 # Subdir under local_log_dir for rollout MP4s: local_log_dir/rollout_prefix/<task>--<ts>/
    run_id_note: Optional[str] = None                                    # Extra note to add to end of run ID for logging
    resume_log_path: Optional[str] = None                                # Existing ENV_EVAL .txt to append to; completed episodes/tasks are skipped

    use_wandb: bool = False                                              # Whether to also log results in Weights & Biases
    wandb_entity: str = "YOUR_ENTITY"                                    # Name of WandB entity
    wandb_project: str = "YOUR_PROJECT"                                  # Name of WandB project

    seed: int = 195                                                      # Random seed (for reproducibility)
    randomize_seed: bool = False                                         # Whether to randomize the seed for sampling

    #################################################################################################################
    # Data collection parameters
    #################################################################################################################
    data_collection: bool = False                                        # If True, save policy rollouts for later offline use
    jpeg_compress: bool = True                                           # If True, apply JPEG compression to images before saving

    #################################################################################################################
    # Parallel environment parameters (for improved GPU utilization)
    #################################################################################################################
    num_parallel_envs: int = 1                                           # Number of environments to run in parallel on the same GPU.
    #                                                                    # When > 1, worker processes run CPU simulation while the main
    #                                                                    # process batches GPU inference requests, forming a CPU/GPU pipeline.
    #                                                                    # Recommended: 2-4 for H100 with num_open_loop_steps=16.
    inference_batch_timeout: float = 0.05                                # Seconds to wait collecting additional inference requests from
    #                                                                    # parallel workers before processing (enables opportunistic batching).

    # fmt: on


def validate_config(cfg: PolicyEvalConfig) -> None:
    """Validate that the evaluation configuration is valid."""
    # Check that the task name is valid
    all_tasks = {**TASK_MAX_STEPS}
    if cfg.task_name != "all" and cfg.task_name != "pnp" and cfg.task_name not in all_tasks:
        raise ValueError(
            f"Task name '{cfg.task_name}' not found in RoboCasa suite. Available tasks: {list(all_tasks.keys())}"
        )

    if not str(cfg.rollout_prefix).strip():
        raise ValueError("rollout_prefix must be a non-empty string (default: rollout_data).")

    # Check that num_third_person_images is 2 (1 for left, 1 for right)
    assert cfg.num_third_person_images == 2, (
        f"Expecting `num_third_person_images` to be 2 (1 for left agentview, 1 for right agentview), "
        f"but got `num_third_person_images={cfg.num_third_person_images}`"
    )

    # Check that the dataset stats path is provided if action unnormalization or proprio normalization is enabled
    if (cfg.unnormalize_actions or cfg.normalize_proprio) and cfg.dataset_stats_path == "":
        raise ValueError(
            "Must provide `dataset_stats_path` when `unnormalize_actions=True` or `normalize_proprio=True`"
        )

    # Check parallel inference configuration
    if cfg.use_parallel_inference and cfg.num_queries_best_of_n <= 1:
        raise ValueError("Parallel inference is enabled but `num_queries_best_of_n <= 1`!")

    if cfg.num_trials_per_task <= 0:
        raise ValueError("`num_trials_per_task` must be > 0")

    if cfg.episode_start_idx < 0:
        raise ValueError("`episode_start_idx` must be >= 0")

    if cfg.episode_stride <= 0:
        raise ValueError("`episode_stride` must be > 0")

    # Check parallel environment configuration
    if cfg.num_parallel_envs > 1:
        if cfg.use_parallel_inference:
            raise ValueError(
                "Cannot combine `num_parallel_envs > 1` with `use_parallel_inference=True`. "
                "Use one or the other."
            )
        if cfg.num_queries_best_of_n > 1:
            raise ValueError(
                "`num_parallel_envs > 1` does not support best-of-N search (`num_queries_best_of_n > 1`). "
                "Set `num_queries_best_of_n=1` when using parallel environments."
            )
        if cfg.data_collection:
            raise ValueError("`data_collection=True` is not supported with `num_parallel_envs > 1`.")
        if cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction:
            raise ValueError(
                "Autoregressive future/value prediction is not supported with `num_parallel_envs > 1`."
            )


def prepare_observation(obs, flip_images: bool = False):
    """Prepare observations from environment for policy input.

    Returns:
        dict: Observation dictionary with keys:
            - "primary_image": Left third-person image (primary camera)
            - "secondary_image": Right third-person image
            - "wrist_image": Eye-in-hand wrist camera image
            - "proprio": Proprioceptive state (eef pose + gripper state)
    """
    # Extract images based on available cameras
    primary_img = None
    secondary_img = None
    wrist_img = None
    # RoboCasa has multiple camera views: left third-person, right third-person, wrist camera
    # We call the left third-person image the primary image and the right third-person image the secondary image
    if "robot0_agentview_left_image" in obs:
        img = obs["robot0_agentview_left_image"]
        if flip_images:
            img = np.flipud(img)
        primary_img = img
    if "robot0_agentview_right_image" in obs:
        img = obs["robot0_agentview_right_image"]
        if flip_images:
            img = np.flipud(img)
        secondary_img = img
    if "robot0_eye_in_hand_image" in obs:
        img = obs["robot0_eye_in_hand_image"]
        if flip_images:
            img = np.flipud(img)
        wrist_img = img
    # Extract proprioceptive state
    proprio = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
    # Prepare observations dict
    observation = {
        "primary_image": primary_img,
        "secondary_image": secondary_img,
        "wrist_image": wrist_img,
        "proprio": proprio,
    }
    return observation


def create_robocasa_env(cfg: PolicyEvalConfig, seed=None, episode_idx=None):
    """Create a RoboCasa environment.

    Args:
        cfg: Configuration object
        seed: Random seed for environment
        episode_idx: Episode index for deterministic scene selection (if None, uses all scenes)
    """
    # Parse layout_and_style_ids
    if cfg.layout_and_style_ids:
        all_layout_style_ids = ast.literal_eval(cfg.layout_and_style_ids)
        # Deterministically select one scene based on episode index
        # Episodes 0-9 use scene 0, episodes 10-19 use scene 1, etc.
        if episode_idx is not None:
            scene_index = (episode_idx // 10) % len(all_layout_style_ids)
            layout_and_style_ids = (all_layout_style_ids[scene_index],)
        else:
            # If no episode index provided, use all scenes (random selection by env)
            layout_and_style_ids = all_layout_style_ids
    else:
        layout_and_style_ids = None

    with open(CONTROLLER_CONFIGS_PATH, "rb") as pickle_file:
        controller_configs = pickle.load(pickle_file)
    # Create environment
    # We use the same args used in the official RoboCasa evals (robocasa/utils/eval_utils.py)
    env_kwargs = dict(
        env_name=cfg.task_name,
        robots=cfg.robots,
        controller_configs=controller_configs,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=cfg.env_img_res,
        camera_heights=cfg.env_img_res,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=cfg.obj_instance_split,
        generative_textures=None,
        randomize_cameras=cfg.randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    env = robosuite.make(**env_kwargs)
    return env, env_kwargs


def run_episode(
    cfg: PolicyEvalConfig,
    env,
    task_description: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    episode_idx: int,
    log_file=None,
):
    """Run a single evaluation episode."""
    # Wait for objects to stabilize
    NUM_STEPS_WAIT = 10
    for _ in range(NUM_STEPS_WAIT):
        dummy_action = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy_action)
    # Get max steps for this task
    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
    # Important variables
    success = False
    episode_length = 0
    action_queue = deque()
    # Containers for episode replay images for saving videos
    replay_primary_images = []  # Left third-person camera
    replay_secondary_images = []  # Right third-person camera
    replay_wrist_images = []  # Wrist camera
    # Containers for episode data collection
    if cfg.data_collection:
        collected_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "success": [],
        }
        primary_images_list = []
        secondary_images_list = []
        wrist_images_list = []
        proprio_list = []
        actions_list = []
    # Best-of-N search variables
    future_image_predictions_list = []
    # Main episode loop
    for t in range(max_steps):
        observation = prepare_observation(obs, cfg.flip_images)
        # Store replay images for video saving
        replay_primary_images.append(observation["primary_image"])
        replay_secondary_images.append(observation["secondary_image"])
        replay_wrist_images.append(observation["wrist_image"])
        # Collect data if enabled
        if cfg.data_collection:
            primary_images_list.append(observation["primary_image"])
            secondary_images_list.append(observation["secondary_image"])
            wrist_images_list.append(observation["wrist_image"])
            proprio_list.append(observation["proprio"])
        # Query policy for new action chunk
        if len(action_queue) == 0:
            # Use parallel inference if enabled
            if cfg.use_parallel_inference and worker_pool and worker_pool.initialized:
                # Query model in parallel
                start_time = time.time()
                query_results = query_model_parallel(
                    cfg,
                    observation,
                    task_description,
                    worker_pool,
                    timeout=cfg.parallel_timeout,
                )
                total_query_time = time.time() - start_time

                log_message(
                    f"Parallel queries completed: {len(query_results)} results in {total_query_time:.3f}s", log_file
                )
                model_inference_times = [
                    result["model_inference_time"] for result in query_results if "model_inference_time" in result
                ]
                if model_inference_times:
                    num_actions = len(query_results[0]["actions"])
                    avg_model_inference_time = sum(model_inference_times) / len(model_inference_times)
                    avg_model_inference_speed = num_actions / avg_model_inference_time if avg_model_inference_time > 0 else 0.0
                    log_message(
                        f"Parallel pure model inference time: max={max(model_inference_times):.3f}s, "
                        f"avg={avg_model_inference_time:.3f}s ({num_actions} actions/query, "
                        f"{avg_model_inference_speed:.1f} actions/s/query)",
                        log_file,
                    )

            else:
                # Serial execution
                query_results = []
                for query_idx in range(cfg.num_queries_best_of_n):
                    actions_by_depth = []  # Action chunks across all depths of the search
                    future_image_predictions_by_depth = []  # Future image predictions across all depths of the search
                    value_predictions_by_depth = []  # Value predictions across all depths of the search
                    return_dict = {}
                    # Query model to get action
                    start_time = time.time()
                    action_return_dict = get_action(
                        cfg,
                        model,
                        dataset_stats,
                        observation,
                        task_description,
                        seed=cfg.seed + query_idx,
                        randomize_seed=cfg.randomize_seed,
                        num_denoising_steps_action=cfg.num_denoising_steps_action,
                        generate_future_state_and_value_in_parallel=not (
                            cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
                        ),
                    )
                    query_time = time.time() - start_time
                    log_message(
                        f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Action query time = {query_time:.3f} sec",
                        log_file,
                    )
                    model_inference_time = action_return_dict.get("model_inference_time")
                    if model_inference_time is not None:
                        num_actions = len(action_return_dict["actions"])
                        model_inference_speed = num_actions / model_inference_time if model_inference_time > 0 else 0.0
                        log_message(
                            f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Pure model inference time = "
                            f"{model_inference_time:.3f} sec ({num_actions} actions, "
                            f"{model_inference_speed:.1f} actions/s)",
                            log_file,
                        )
                    return_dict["actions"] = action_return_dict["actions"]
                    actions_by_depth.append(return_dict["actions"])

                    if cfg.ar_future_prediction:
                        # Autoregressively query model to get future state prediction
                        start_time = time.time()
                        future_state_return_dict = get_future_state_prediction(
                            cfg,
                            model=planning_model if planning_model is not None else model,
                            data_batch=action_return_dict["data_batch"],
                            generated_latent_with_action=action_return_dict["generated_latent"],
                            orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
                            future_proprio_latent_idx=action_return_dict["latent_indices"]["future_proprio_latent_idx"],
                            future_wrist_image_latent_idx=action_return_dict["latent_indices"][
                                "future_wrist_image_latent_idx"
                            ],
                            future_wrist_image2_latent_idx=action_return_dict["latent_indices"][
                                "future_wrist_image2_latent_idx"
                            ],
                            future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
                            future_image2_latent_idx=action_return_dict["latent_indices"]["future_image2_latent_idx"],
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                            use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                            num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                            future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Future state prediction query time = {query_time:.3f} sec",
                            log_file,
                        )
                        return_dict["future_image_predictions"] = future_state_return_dict["future_image_predictions"]
                        future_image_predictions_by_depth.append(return_dict["future_image_predictions"])

                    else:
                        return_dict["future_image_predictions"] = action_return_dict["future_image_predictions"]

                    if cfg.ar_value_prediction:
                        # Autoregressively query model to get value prediction
                        start_time = time.time()
                        value_return_dict = get_value_prediction(
                            cfg,
                            model=planning_model if planning_model is not None else model,
                            data_batch=action_return_dict["data_batch"],
                            future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_value=cfg.num_denoising_steps_value,
                            use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                            num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Value prediction query time = {query_time:.3f} sec",
                            log_file,
                        )
                        return_dict["value_prediction"] = value_return_dict["value_prediction"]
                        value_predictions_by_depth.append(return_dict["value_prediction"])
                        log_message(
                            f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Value prediction: {return_dict['value_prediction']:.4f}",
                            log_file,
                        )
                    elif cfg.ar_qvalue_prediction:
                        # Autoregressively query model to get Q-value prediction
                        start_time = time.time()
                        value_return_dict = get_qvalue_prediction(
                            cfg,
                            model=planning_model if planning_model is not None else model,
                            data_batch=action_return_dict["data_batch"],
                            action_sample=action_return_dict["generated_latent"],
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_value=cfg.num_denoising_steps_value,
                            use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                            num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Value prediction query time = {query_time:.3f} sec",
                            log_file,
                        )
                        return_dict["value_prediction"] = value_return_dict["value_prediction"]
                        value_predictions_by_depth.append(return_dict["value_prediction"])
                        log_message(
                            f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Value prediction: {return_dict['value_prediction']:.4f}",
                            log_file,
                        )
                    else:
                        return_dict["value_prediction"] = action_return_dict["value_prediction"]
                        value_predictions_by_depth.append(return_dict["value_prediction"])

                    if cfg.search_depth > 1:
                        assert not cfg.ar_qvalue_prediction, "Search depth > 1 not supported for Q-value prediction!"
                        for depth in range(2, cfg.search_depth + 1):
                            for future_state_latent in future_state_return_dict["future_state_samples_list"]:
                                next_generated_latent_with_future_state = future_state_latent.clone()
                                # Rearrange latent frames such that predicted future state replaces current state in the sequence.
                                # Use explicit modality index mapping (robust to different state_t layouts).
                                rearranged_next_latent_with_future_state = next_generated_latent_with_future_state.clone()
                                latent_indices = action_return_dict["latent_indices"]
                                index_pairs = [
                                    ("current_wrist_image_latent_idx", "future_wrist_image_latent_idx"),
                                    ("current_wrist_image2_latent_idx", "future_wrist_image2_latent_idx"),
                                    ("current_image_latent_idx", "future_image_latent_idx"),
                                    ("current_image2_latent_idx", "future_image2_latent_idx"),
                                ]
                                # Upstream layout has dedicated proprio frames; include that mapping too.
                                # (MetaWAM keeps proprio in the input/meta frames instead.)
                                if not getattr(model.config, "use_single_meta_frame", False):
                                    index_pairs.append(("current_proprio_latent_idx", "future_proprio_latent_idx"))
                                for curr_key, fut_key in index_pairs:
                                    curr_idx = int(action_return_dict["data_batch"][curr_key][0].item())
                                    fut_idx = int(latent_indices[fut_key])
                                    if curr_idx != -1 and fut_idx != -1:
                                        rearranged_next_latent_with_future_state[:, :, curr_idx, :, :] = (
                                            next_generated_latent_with_future_state[:, :, fut_idx, :, :]
                                        )
                                ################################
                                # Predict next action
                                ################################
                                data_batch = action_return_dict["data_batch"]
                                data_batch["num_conditional_frames"] = (
                                    model.config.min_num_conditional_frames
                                )  # Reset to the original value
                                data_batch["mask_current_state_action_for_value_prediction"] = (
                                    False  # Don't use input masking for action prediction
                                )
                                if cfg.randomize_seed:
                                    seed = secrets.randbits(32) % 256
                                else:
                                    seed = cfg.seed + query_idx
                                batch_size = 1
                                next_generated_latent_with_action, next_orig_clean_latent_frames = (
                                    model.generate_samples_from_batch(
                                        data_batch,
                                        n_sample=batch_size,
                                        num_steps=cfg.num_denoising_steps_action,
                                        seed=seed,
                                        is_negative_prompt=False,
                                        use_variance_scale=cfg.use_variance_scale,
                                        skip_vae_encoding=True,
                                        previous_generated_latent=rearranged_next_latent_with_future_state,  # Use future state sample since parts of value sample might be masked out
                                        return_orig_clean_latent_frames=True,
                                    )
                                )  # (B, C'=16, T', H'=28, W'=28)
                                # Extract the action chunk prediction from the generated samples
                                action_latent_idx = action_return_dict["latent_indices"]["action_latent_idx"]
                                action_indices = torch.full(
                                    (batch_size,),
                                    action_latent_idx,
                                    dtype=torch.int64,
                                    device=next_generated_latent_with_action.device,
                                )
                                next_actions = (
                                    extract_action_chunk(
                                        model,
                                        next_generated_latent_with_action,
                                        (cfg.chunk_size, ACTION_DIM),
                                        action_indices=action_indices,
                                    )
                                    .to(torch.float32)
                                    .cpu()
                                    .numpy()
                                )
                                # Unnormalize actions
                                if cfg.unnormalize_actions:
                                    next_actions = unnormalize_actions(next_actions, dataset_stats)
                                # Squeeze and convert to list
                                next_actions = next_actions[0]
                                next_actions = [next_actions[i] for i in range(len(next_actions))]
                                actions_by_depth.append(next_actions)
                                ################################
                                # Predict next future state
                                ################################
                                future_state_return_dict = get_future_state_prediction(
                                    cfg,
                                    model=planning_model if planning_model is not None else model,
                                    data_batch=action_return_dict["data_batch"],
                                    generated_latent_with_action=next_generated_latent_with_action,
                                    orig_clean_latent_frames=next_orig_clean_latent_frames,
                                    future_proprio_latent_idx=action_return_dict["latent_indices"][
                                        "future_proprio_latent_idx"
                                    ],
                                    future_wrist_image_latent_idx=action_return_dict["latent_indices"][
                                        "future_wrist_image_latent_idx"
                                    ],
                                    future_wrist_image2_latent_idx=action_return_dict["latent_indices"][
                                        "future_wrist_image2_latent_idx"
                                    ],
                                    future_image_latent_idx=action_return_dict["latent_indices"][
                                        "future_image_latent_idx"
                                    ],
                                    future_image2_latent_idx=action_return_dict["latent_indices"][
                                        "future_image2_latent_idx"
                                    ],
                                    seed=cfg.seed + query_idx,
                                    randomize_seed=cfg.randomize_seed,
                                    num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                                    use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                                    num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                                    future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                                )
                                # Track per-depth prediction
                                future_image_predictions_by_depth.append(
                                    future_state_return_dict["future_image_predictions"]
                                )
                                ################################
                                # Predict next value
                                ################################
                                value_return_dict = get_value_prediction(
                                    cfg,
                                    model=planning_model if planning_model is not None else model,
                                    data_batch=action_return_dict["data_batch"],
                                    future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                                    seed=cfg.seed + query_idx,
                                    randomize_seed=cfg.randomize_seed,
                                    num_denoising_steps_value=cfg.num_denoising_steps_value,
                                    use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                                    num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                                )
                                return_dict["value_prediction"] = value_return_dict["value_prediction"]
                                value_predictions_by_depth.append(return_dict["value_prediction"])
                                log_message(
                                    f"Query {query_idx + 1}/{cfg.num_queries_best_of_n}: Value prediction: {return_dict['value_prediction']:.4f}",
                                    log_file,
                                )
                    # Add results to the return dict
                    return_dict["future_image_predictions_by_depth"] = future_image_predictions_by_depth
                    return_dict["value_predictions_by_depth"] = value_predictions_by_depth
                    return_dict["actions_by_depth"] = actions_by_depth
                    query_results.append(return_dict)

            # Replace value of each query with aggregate value over all value predictions
            # This is only applicable when search depth > 1 because otherwise the aggregation is handled in get_value_prediction()
            # For search depth == 1, the return dict contains the aggregated value prediction already
            if cfg.search_depth > 1:
                for query_idx, return_dict in enumerate(query_results):
                    if cfg.search_depth_value_aggregation_scheme == "average":
                        return_dict["value_prediction"] = np.mean(return_dict["value_predictions_by_depth"]).item()
                    elif cfg.search_depth_value_aggregation_scheme == "use_last_value":
                        return_dict["value_prediction"] = return_dict["value_predictions_by_depth"][-1]
                    else:
                        raise ValueError(
                            f"Invalid search depth value aggregation scheme: {cfg.search_depth_value_aggregation_scheme}"
                        )
            # Print all value predictions
            for query_idx, return_dict in enumerate(query_results):
                predicted_value = return_dict["value_prediction"]
                log_message(
                    f"Query {query_idx + 1}/{cfg.num_queries_best_of_n} (seed {cfg.seed + query_idx}): Predicted value = {predicted_value:.4f}",
                    log_file,
                )
            # Only keep the first num_open_loop_steps timesteps of the action chunk
            for query_idx, return_dict in enumerate(query_results):
                return_dict["actions"] = return_dict["actions"][: cfg.num_open_loop_steps]
            # Get dict: seed number -> (action chunk, future state, value)
            seed_to_return_dict = {
                cfg.seed + query_idx: (
                    return_dict["actions"],
                    return_dict["future_image_predictions"],
                    return_dict["value_prediction"],
                )
                for query_idx, return_dict in enumerate(query_results)
            }
            # Get seed with highest value
            best_seed, best_return_dict = max(seed_to_return_dict.items(), key=lambda x: x[1][2])
            best_actions = best_return_dict[0]
            best_future_predictions = best_return_dict[1]
            best_value_predictions = best_return_dict[2]
            # Use the best actions, future predictions, and value predictions found
            action_queue.extend(best_actions)
            future_image_predictions_list.append(best_future_predictions)
            log_message(f"t={t}: Selected seed {best_seed} with value = {best_value_predictions:.4f}", log_file)

        # Get next action from chunk
        action = action_queue.popleft()
        # RoboCasa: Policy was trained on 7-dim manipulation actions, but env expects 12-dim (7 + 5 mobile base)
        # Append [0, 0, 0, 0, -1] for mobile base since we're not using it
        if action.shape[-1] == 7 and env.action_dim == 12:
            mobile_base_action = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
            action = np.concatenate([action, mobile_base_action])
        # Execute action
        print(f"t: {t}, action: {action}")
        obs, reward, done, info = env.step(action)
        episode_length += 1
        # Collect action data if enabled
        if cfg.data_collection:
            actions_list.append(action)
        # Check for success
        if env._check_success():
            success = True
            log_message(f"  Success detected at timestep {t}!", log_file)
            break

    # Log episode result
    log_message(
        f"  Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'} (length: {episode_length})",
        log_file,
    )
    # Prepare collected data if enabled
    if cfg.data_collection:
        collected_data = dict(
            primary_images=np.stack(primary_images_list, axis=0),  # (T, H, W, C) - left camera
            secondary_images=np.stack(secondary_images_list, axis=0),  # (T, H, W, C) - right camera
            wrist_images=np.stack(wrist_images_list, axis=0),  # (T, H, W, C)
            proprio=np.stack(proprio_list, axis=0),  # (T, D)
            actions=np.stack(actions_list, axis=0),  # (T, action_dim)
            success=success,
        )
        # Add future image predictions
        if len(future_image_predictions_list) > 0:
            # Primary camera predictions (left third-person)
            if (
                "future_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_image"] is not None
            ):
                future_primary_images = [x["future_image"] for x in future_image_predictions_list]
                collected_data["future_primary_images"] = np.stack(future_primary_images, axis=0)
            # Secondary camera predictions (right third-person)
            if (
                "future_image2" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_image2"] is not None
            ):
                future_secondary_images = [x["future_image2"] for x in future_image_predictions_list]
                collected_data["future_secondary_images"] = np.stack(future_secondary_images, axis=0)
            # Wrist camera predictions
            if (
                "future_wrist_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_wrist_image"] is not None
            ):
                future_wrist_images = [x["future_wrist_image"] for x in future_image_predictions_list]
                collected_data["future_wrist_images"] = np.stack(future_wrist_images, axis=0)
    else:
        collected_data = None
    return (
        success,
        episode_length,
        replay_primary_images,
        replay_secondary_images,
        replay_wrist_images,
        future_image_predictions_list,
        collected_data,
    )


def _iter_worker_work_items(work_items):
    """Yield worker tasks from either a static list or a shared multiprocessing queue."""
    if hasattr(work_items, "get"):
        while True:
            item = work_items.get()
            if item is None:
                break
            yield item
    else:
        yield from work_items


def _env_worker(
    worker_id: int,
    cfg: PolicyEvalConfig,
    work_items: list,
    obs_queue: mp.Queue,
    action_queue: mp.Queue,
    result_queue: mp.Queue,
    base_log_dir: str,
    eval_log_txt_path: Optional[str] = None,
) -> None:
    """Worker process: runs environment simulation and requests GPU inference from the main process.

    This function runs in a separate subprocess. It handles all CPU-bound simulation work
    (environment stepping, observation preparation) and delegates GPU inference to the main
    process via inter-process queues. Each worker can handle episodes from multiple tasks,
    enabling cross-task GPU pipeline parallelism.

    Args:
        worker_id: Unique worker identifier.
        cfg: Evaluation config (model-related fields are unused in the worker).
        work_items: Static list or shared queue of (task_name, episode_idx) tuples this worker should run.
        obs_queue: Shared queue for sending observations to the main process.
        action_queue: Per-worker queue for receiving action chunks from the main process.
        result_queue: Queue for returning per-episode results to the main process.
        base_log_dir: Root log directory; rollout videos are saved under
            ``base_log_dir / cfg.rollout_prefix / <task>--<timestamp> /``.

    Communication protocol:
        obs_queue (worker → main):
            - Inference request: (worker_id, observation_dict, task_description)
            - Done signal:       (worker_id, None, None)
        action_queue (main → worker):
            - Action chunk: list of numpy arrays of shape (action_dim,)
        result_queue (worker → main):
            - Per-episode result: (task_name, episode_idx, success, episode_length)
              (sent once per episode, immediately after it completes)
    """
    cfg = copy.copy(cfg)

    for task_name, episode_idx in _iter_worker_work_items(work_items):
        cfg.task_name = task_name

        # Deterministic seeding consistent with the serial path
        if cfg.deterministic or cfg.deterministic_reset:
            seed = cfg.seed * episode_idx * 256
        else:
            seed = None
        env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
        if cfg.deterministic_reset:
            reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
            set_seed_everywhere(reset_seed)
        env.reset()
        # Episode instruction is available only after reset because RoboCasa samples a concrete scene/instruction instance here.
        task_description = env.get_ep_meta()["lang"]
        print(f"[Worker {worker_id}] task_family={task_name} ep{episode_idx}: instruction={task_description}", flush=True)

        # Wait for objects to stabilize
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))

        max_steps = TASK_MAX_STEPS.get(task_name, 500)
        success = False
        episode_length = 0
        action_buf = deque()
        replay_primary, replay_secondary, replay_wrist = [], [], []
        future_image_predictions_list = []

        for t in range(max_steps):
            observation = prepare_observation(obs, cfg.flip_images)
            replay_primary.append(observation["primary_image"])
            replay_secondary.append(observation["secondary_image"])
            replay_wrist.append(observation["wrist_image"])

            if not action_buf:
                # Send observation to main process and block until actions + future preds arrive
                obs_queue.put((worker_id, observation, task_description))
                action_chunk, future_preds = action_queue.get()
                action_buf.extend(action_chunk[: cfg.num_open_loop_steps])
                future_image_predictions_list.append(future_preds)

            action = action_buf.popleft()
            # RoboCasa: policy predicts 7-DOF arm actions; env expects 12-DOF (7 arm + 5 mobile base)
            if action.shape[-1] == 7 and env.action_dim == 12:
                action = np.concatenate([action, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
            obs, _, _, _ = env.step(action)
            episode_length += 1

            if env._check_success():
                success = True
                print(f"[Worker {worker_id}] {task_name} ep{episode_idx}: SUCCESS at t={t}", flush=True)
                break

        env.close()
        status = "SUCCESS" if success else "FAILURE"
        print(
            f"[Worker {worker_id}] {task_name} ep{episode_idx}: {status} (length={episode_length})",
            flush=True,
        )
        _append_to_eval_log(
            eval_log_txt_path,
            f"[Worker {worker_id}] {task_name} ep{episode_idx}: {status} (length={episode_length})",
        )

        # Save rollout video directly from the worker process
        rollout_data_dir = os.path.join(
            base_log_dir, cfg.rollout_prefix.strip(), f"{task_name}--{_rollout_session_timestamp()}"
        )
        try:
            os.makedirs(rollout_data_dir, exist_ok=True)
            out_main = save_rollout_video(
                replay_primary,
                replay_secondary,
                replay_wrist,
                episode_idx,
                success=success,
                task_description=task_description,
                rollout_data_dir=rollout_data_dir,
                log_file=None,
            )
            _append_to_eval_log(
                eval_log_txt_path,
                f"[Worker {worker_id}] {task_name} ep{episode_idx} saved rollout: {out_main}",
            )
            # Save rollout video with future image predictions if the model generated them
            if len(future_image_predictions_list) > 0:
                future_primary = [x["future_image"] for x in future_image_predictions_list if "future_image" in x]
                future_secondary = [x["future_image2"] for x in future_image_predictions_list if "future_image2" in x]
                future_wrist = [x["future_wrist_image"] for x in future_image_predictions_list if "future_wrist_image" in x]
                if future_primary and future_secondary and future_wrist:
                    out_fut = save_rollout_video_with_future_image_predictions(
                        replay_primary,
                        replay_secondary,
                        replay_wrist,
                        episode_idx,
                        success=success,
                        task_description=task_description,
                        rollout_data_dir=rollout_data_dir,
                        chunk_size=cfg.chunk_size,
                        num_open_loop_steps=cfg.num_open_loop_steps,
                        future_primary_image_predictions=future_primary,
                        future_secondary_image_predictions=future_secondary,
                        future_wrist_image_predictions=future_wrist,
                        show_diff=False,
                        log_file=None,
                        show_timestep=True,
                    )
                    _append_to_eval_log(
                        eval_log_txt_path,
                        f"[Worker {worker_id}] {task_name} ep{episode_idx} saved future-pred video: {out_fut}",
                    )
        except Exception:
            err_txt = traceback.format_exc()
            print(
                f"[Worker {worker_id}] FAILED to save rollout video under {rollout_data_dir}:\n" + err_txt,
                flush=True,
            )
            _append_to_eval_log(
                eval_log_txt_path,
                f"[Worker {worker_id}] FAILED rollout save: {err_txt[:800]}",
            )

        # Report result immediately so the main process can track per-task progress
        result_queue.put((task_name, episode_idx, success, episode_length))

    # Signal to main process that this worker is done
    obs_queue.put((worker_id, None, None))


def _run_parallel_inference_loop(
    cfg: PolicyEvalConfig,
    model,
    dataset_stats,
    obs_queue: mp.Queue,
    action_queues: list,
    num_workers: int,
    log_file=None,
):
    """Main-process inference loop shared by both single-task and all-tasks parallel modes.

    Blocks until all workers signal completion. For each inference request received from
    a worker, optionally waits a short window (`inference_batch_timeout`) for additional
    requests to arrive so they can be dispatched back-to-back (opportunistic batching).

    Returns:
        result_queue containing per-episode (task_name, episode_idx, success, length) tuples
        was already populated by workers; this function only drives the inference side.
    """
    active_count = num_workers
    total_inferences = 0
    multi_batch_count = 0

    while active_count > 0:
        # Block until at least one inference request or sentinel arrives
        try:
            worker_id, obs, task_desc = obs_queue.get(timeout=2.0)
        except _stdlib_queue.Empty:
            continue

        if obs is None:
            active_count -= 1
            log_message(f"Worker {worker_id} finished. {active_count} worker(s) still active.", log_file)
            continue

        # Opportunistic batching: collect more requests within a short window
        pending = [(worker_id, obs, task_desc)]
        deadline = time.time() + cfg.inference_batch_timeout
        while len(pending) < active_count:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                w_id, o, t_desc = obs_queue.get(timeout=remaining)
                if o is None:
                    active_count -= 1
                else:
                    pending.append((w_id, o, t_desc))
            except _stdlib_queue.Empty:
                break

        batch_size = len(pending)
        if batch_size > 1:
            multi_batch_count += 1

        t_start = time.time()
        model_inference_times = []
        for w_id, obs_dict, task_description in pending:
            action_return_dict = get_action(
                cfg,
                model,
                dataset_stats,
                obs_dict,
                task_description,
                seed=cfg.seed,
                randomize_seed=cfg.randomize_seed,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
            if "model_inference_time" in action_return_dict:
                model_inference_times.append(action_return_dict["model_inference_time"])
            # Pass back both actions and future image predictions.
            # future_image_predictions is a small dict of (H,W,C) uint8 arrays; queue overhead is acceptable.
            future_preds = action_return_dict.get("future_image_predictions", {})
            action_queues[w_id].put((action_return_dict["actions"], future_preds))
            total_inferences += 1

        elapsed = time.time() - t_start
        log_message(
            f"[Inference] batch={batch_size} workers={[r[0] for r in pending]} "
            f"time={elapsed:.3f}s ({elapsed / batch_size:.3f}s/env)",
            log_file,
        )
        if model_inference_times:
            total_model_inference_time = sum(model_inference_times)
            num_actions = len(action_return_dict["actions"])
            avg_model_inference_time = total_model_inference_time / len(model_inference_times)
            avg_model_inference_speed = num_actions / avg_model_inference_time if avg_model_inference_time > 0 else 0.0
            log_message(
                f"[Inference] pure_model_time={total_model_inference_time:.3f}s "
                f"({avg_model_inference_time:.3f}s/env, {num_actions} actions/query, "
                f"{avg_model_inference_speed:.1f} actions/s/query)",
                log_file,
            )

    log_message(
        f"\n[Parallel eval] Total GPU inferences: {total_inferences}, "
        f"multi-request batches: {multi_batch_count}/{total_inferences}",
        log_file,
    )


def run_task_parallel_envs(
    cfg: PolicyEvalConfig,
    task_name: str,
    model,
    dataset_stats,
    log_file=None,
    eval_log_txt_path: Optional[str] = None,
    resume_state: Optional[dict[str, dict]] = None,
):
    """Run a single task with multiple parallel environments for improved GPU utilization.

    N worker subprocesses each handle a subset of episodes for `task_name`. All workers
    share one GPU (main process), creating a CPU/GPU pipeline: while the GPU processes
    one worker's observation, other workers continue executing open-loop actions on the CPU.

    Args:
        cfg: Evaluation config. `num_parallel_envs` controls how many workers to start.
        task_name: The RoboCasa task to evaluate.
        model: Loaded Cosmos policy model (main process only).
        dataset_stats: Dataset statistics for action unnormalization.
        log_file: Optional log file handle.

    Returns:
        (success_rate, avg_episode_length, successes_list)
    """
    num_envs = cfg.num_parallel_envs
    log_message(f"\nEvaluating task: {task_name} with {num_envs} parallel environments", log_file)
    if eval_log_txt_path:
        log_message(
            f"When each episode finishes, video paths are appended to this same log file: {eval_log_txt_path}",
            log_file,
        )

    resume_state = resume_state or {}
    completed_results = _resume_episode_results(resume_state, task_name)
    pending_episode_indices = _pending_episode_indices(
        resume_state,
        task_name,
        cfg.num_trials_per_task,
        cfg.episode_start_idx,
        cfg.episode_stride,
    )
    if len(pending_episode_indices) < cfg.num_trials_per_task:
        log_message(
            f"Resume: {task_name} has {cfg.num_trials_per_task - len(pending_episode_indices)}/"
            f"{cfg.num_trials_per_task} completed episodes; running {len(pending_episode_indices)} pending episodes",
            log_file,
        )
    if not pending_episode_indices:
        success_rate, avg_length, valid_successes = _stats_from_resume_state(
            resume_state,
            task_name,
            cfg.num_trials_per_task,
            cfg.episode_start_idx,
            cfg.episode_stride,
        )
        log_message(f"Resume: skipping completed task {task_name}", log_file)
        log_message(f"\nTask {task_name} results:", log_file)
        log_message(f"  Success rate: {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
        log_message(f"  Average episode length: {avg_length:.1f}", log_file)
        log_message(f"  Successes: {sum(valid_successes)}/{len(valid_successes)}", log_file)
        return success_rate, avg_length, valid_successes

    all_work_items = [(task_name, ep_idx) for ep_idx in pending_episode_indices]
    num_workers = min(num_envs, len(all_work_items))
    work_queue: mp.Queue = mp.Queue()
    for item in all_work_items:
        work_queue.put(item)
    for _ in range(num_workers):
        work_queue.put(None)

    obs_queue: mp.Queue = mp.Queue()
    action_queues: list[mp.Queue] = [mp.Queue() for _ in range(num_workers)]
    result_queue: mp.Queue = mp.Queue()

    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=_env_worker,
            args=(
                i,
                cfg,
                work_queue,
                obs_queue,
                action_queues[i],
                result_queue,
                cfg.local_log_dir,
                eval_log_txt_path,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    log_message(f"Started {len(processes)} worker processes for {len(all_work_items)} queued episodes", log_file)

    _run_parallel_inference_loop(cfg, model, dataset_stats, obs_queue, action_queues, len(processes), log_file)

    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    # Collect results
    task_results: dict[int, tuple[bool, Optional[int]]] = completed_results  # episode_idx → (success, length)
    while True:
        try:
            t_name, ep_idx, success, length = result_queue.get_nowait()
            task_results[ep_idx] = (success, length)
        except _stdlib_queue.Empty:
            break

    ordered = [
        task_results[i]
        for i in _requested_episode_indices(cfg.num_trials_per_task, cfg.episode_start_idx, cfg.episode_stride)
        if i in task_results
    ]
    valid_successes = [r[0] for r in ordered]
    valid_lengths = [r[1] for r in ordered if r[1] is not None]

    success_rate = float(np.mean(valid_successes)) if valid_successes else 0.0
    avg_length = float(np.mean(valid_lengths)) if valid_lengths else 0.0

    log_message(f"\nTask {task_name} results:", log_file)
    log_message(f"  Success rate: {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
    log_message(f"  Average episode length: {avg_length:.1f}", log_file)
    log_message(f"  Successes: {sum(valid_successes)}/{len(valid_successes)}", log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_name}": success_rate,
                f"avg_episode_length/{task_name}": avg_length,
                f"num_successes/{task_name}": sum(valid_successes),
                f"num_episodes/{task_name}": len(valid_successes),
            }
        )

    return success_rate, avg_length, valid_successes



def run_task(
    cfg: PolicyEvalConfig,
    task_name: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    log_file=None,
    resume_state: Optional[dict[str, dict]] = None,
):
    """Run evaluation for a single task."""
    log_message(f"\nEvaluating task: {task_name}", log_file)
    resume_state = resume_state or {}
    completed_results = _resume_episode_results(resume_state, task_name)
    pending_episode_indices = _pending_episode_indices(
        resume_state,
        task_name,
        cfg.num_trials_per_task,
        cfg.episode_start_idx,
        cfg.episode_stride,
    )
    if len(pending_episode_indices) < cfg.num_trials_per_task:
        log_message(
            f"Resume: {task_name} has {cfg.num_trials_per_task - len(pending_episode_indices)}/"
            f"{cfg.num_trials_per_task} completed episodes; running {len(pending_episode_indices)} pending episodes",
            log_file,
        )
    if not pending_episode_indices:
        success_rate, avg_length, successes = _stats_from_resume_state(
            resume_state,
            task_name,
            cfg.num_trials_per_task,
            cfg.episode_start_idx,
            cfg.episode_stride,
        )
        log_message(f"Resume: skipping completed task {task_name}", log_file)
        log_message(f"Task {task_name} results:", log_file)
        log_message(f"  Success rate: {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
        log_message(f"  Average episode length: {avg_length:.1f}", log_file)
        log_message(f"  Successes: {sum(successes)}/{len(successes)}", log_file)
        return success_rate, avg_length, successes

    episode_results: dict[int, tuple[bool, Optional[int]]] = completed_results
    total_episodes = len(completed_results)
    total_successes = sum(success for success, _ in completed_results.values())
    for local_episode_idx, episode_idx in enumerate(pending_episode_indices):
        log_message(
            f"Starting episode local={local_episode_idx + 1}/{len(pending_episode_indices)}, global_idx={episode_idx}...",
            log_file,
        )
        log_message(f"Task family: {cfg.task_name}", log_file)
        # Create environment with scene selection based on episode index
        # Episodes 0-9 use scene 0, 10-19 use scene 1, etc.
        if cfg.deterministic or cfg.deterministic_reset:
            # Deterministic seeding for reproducibility
            seed = cfg.seed * episode_idx * 256
        else:
            seed = None
        env, env_kwargs = create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
        # Reset environment
        # NOTE: Every reset samples a concrete scene/instruction instance for this task family, so reset only once.
        if cfg.deterministic_reset:
            reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
            set_seed_everywhere(reset_seed)
        env.reset()
        # Get the concrete language instruction for this episode instance.
        # IMPORTANT: Read it after reset because RoboCasa finalizes the scene/instruction at reset time.
        task_description = env.get_ep_meta()["lang"]
        log_message(f"\nInstruction: {task_description}", log_file)
        # Run episode
        (
            success,
            length,
            replay_primary_images,
            replay_secondary_images,
            replay_wrist_images,
            future_image_predictions_list,
            collected_data,
        ) = run_episode(
            cfg,
            env,
            task_description,
            model,
            planning_model,
            dataset_stats,
            worker_pool,
            episode_idx,
            log_file,
        )
        episode_results[episode_idx] = (success, length)
        # Update counters
        total_episodes += 1
        if success:
            total_successes += 1
        # Save rollout video
        rollout_data_dir = os.path.join(
            cfg.local_log_dir, cfg.rollout_prefix.strip(), f"{task_name}--{_rollout_session_timestamp()}"
        )
        try:
            os.makedirs(rollout_data_dir, exist_ok=True)
            save_rollout_video(
                replay_primary_images,
                replay_secondary_images,
                replay_wrist_images,
                episode_idx,
                success=success,
                task_description=task_description,
                rollout_data_dir=rollout_data_dir,
                log_file=log_file,
            )
            # Save rollout video with future image predictions
            if len(future_image_predictions_list) > 0:
                # Extract future predictions from the list
                future_primary_image_predictions = None
                future_secondary_image_predictions = None
                future_wrist_image_predictions = None
                if (
                    "future_image" in future_image_predictions_list[0]
                    and future_image_predictions_list[0]["future_image"] is not None
                ):
                    future_primary_image_predictions = [x["future_image"] for x in future_image_predictions_list]
                if (
                    "future_image2" in future_image_predictions_list[0]
                    and future_image_predictions_list[0]["future_image2"] is not None
                ):
                    future_secondary_image_predictions = [x["future_image2"] for x in future_image_predictions_list]
                if (
                    "future_wrist_image" in future_image_predictions_list[0]
                    and future_image_predictions_list[0]["future_wrist_image"] is not None
                ):
                    future_wrist_image_predictions = [x["future_wrist_image"] for x in future_image_predictions_list]
                # Save video with predictions if all three camera predictions are available
                if (
                    future_primary_image_predictions is not None
                    and future_secondary_image_predictions is not None
                    and future_wrist_image_predictions is not None
                ):
                    save_rollout_video_with_future_image_predictions(
                        replay_primary_images,
                        replay_secondary_images,
                        replay_wrist_images,
                        episode_idx,
                        success=success,
                        task_description=task_description,
                        rollout_data_dir=rollout_data_dir,
                        chunk_size=cfg.chunk_size,
                        num_open_loop_steps=cfg.num_open_loop_steps,
                        future_primary_image_predictions=future_primary_image_predictions,
                        future_secondary_image_predictions=future_secondary_image_predictions,
                        future_wrist_image_predictions=future_wrist_image_predictions,
                        show_diff=False,
                        log_file=log_file,
                        show_timestep=True,
                    )
                else:
                    log_message(
                        f"Skipping video with future predictions - not all camera predictions available "
                        f"(primary: {future_primary_image_predictions is not None}, "
                        f"secondary: {future_secondary_image_predictions is not None}, "
                        f"wrist: {future_wrist_image_predictions is not None})",
                        log_file,
                    )
        except Exception:
            log_message(f"FAILED to save rollout video under {rollout_data_dir}:\n{traceback.format_exc()}", log_file)
        # Save collected data if data_collection is enabled
        if cfg.data_collection and collected_data is not None:
            # Skip episodes that are less than 5 timesteps long (because sometimes success is detected immediately upon starting since the envs are buggy)
            if len(collected_data["actions"]) < 5:
                log_message(
                    f"Skipping saving this episode: less than 5 timesteps long (only {len(collected_data['actions'])} timesteps).",
                    log_file,
                )
                continue
            # Save episodic HDF5 data
            processed_task_description = (
                task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:35]
            )
            ep_filename = f"{_rollout_session_timestamp()}--episode_data--task={processed_task_description}--ep={episode_idx}--success={success}.hdf5"
            ep_filepath = os.path.join(rollout_data_dir, ep_filename)
            with h5py.File(ep_filepath, "w") as f:
                for k, v in collected_data.items():
                    if isinstance(v, np.ndarray):
                        is_image = v.ndim == 4 and v.shape[-1] == 3 and v.dtype == np.uint8
                        if is_image and cfg.jpeg_compress:
                            jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in v]
                            if len(jpeg_list) == 1:
                                # Skip saving the array if it only has one element (causes error during create_dataset())
                                continue
                            dt = h5py.vlen_dtype(np.dtype("uint8"))
                            f.create_dataset(k + "_jpeg", data=jpeg_list, dtype=dt)
                        else:
                            f.create_dataset(k, data=v)
                    else:
                        f.attrs[k] = v
                f.attrs["task_description"] = task_description
            log_message(f"Saved episode data to: {ep_filepath}", log_file)
        # Close environment after each episode
        env.close()
        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
    # Calculate statistics
    ordered = [episode_results[i] for i in sorted(episode_results)]
    successes = [success for success, _ in ordered]
    episode_lengths = [length for _, length in ordered if length is not None]
    success_rate = float(np.mean(successes)) if successes else 0.0
    avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    log_message(f"Task {task_name} results:", log_file)
    log_message(f"  Success rate: {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
    log_message(f"  Average episode length: {avg_length:.1f}", log_file)
    log_message(
        f"  Successes: {sum(successes)}/{len(successes)}",
        log_file,
    )
    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_name}": success_rate,
                f"avg_episode_length/{task_name}": avg_length,
                f"num_successes/{task_name}": sum(successes),
                f"num_episodes/{task_name}": len(successes),
            }
        )
    return success_rate, avg_length, successes


@draccus.wrap()
def eval_robocasa(cfg: PolicyEvalConfig) -> float:
    """Main function to evaluate a trained policy on RoboCasa tasks."""
    # Set DETERMINISTIC environment variable if on deterministic mode
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    # Set random seed
    set_seed_everywhere(cfg.seed)
    # Set multiprocessing start method for any subprocess-based parallelism.
    # "spawn" ensures worker processes start fresh (without inheriting GPU model state).
    if cfg.use_parallel_inference or cfg.num_parallel_envs > 1:
        mp.set_start_method("spawn", force=True)
    # Validate evaluation configuration
    validate_config(cfg)
    # Resolve log dir in the parent so spawn workers use the same path (relative to this file, not cwd).
    cfg.local_log_dir = _resolve_local_log_dir(cfg.local_log_dir)
    cfg.resume_log_path = _resolve_resume_log_path(cfg.resume_log_path, cfg.local_log_dir)
    if cfg.resume_log_path and not os.path.exists(cfg.resume_log_path):
        raise FileNotFoundError(f"resume_log_path does not exist: {cfg.resume_log_path}")
    resume_state = _parse_resume_rollout_videos(cfg.local_log_dir, cfg.rollout_prefix) if cfg.resume_log_path else {}
    # One timestamp for this eval run so main process and spawn workers use the same rollout subdir.
    # On resume, reuse the original ENV_EVAL timestamp so new MP4s land next to the old run's videos.
    os.environ["COSMOS_ROBOCASA_ROLLOUT_TS"] = _resume_rollout_timestamp(cfg.resume_log_path) or time.strftime(
        "%Y_%m_%d-%H_%M_%S"
    )
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    _readme = _write_where_outputs_are_file(cfg.local_log_dir, cfg.rollout_prefix)
    _ptr_script, _ptr_cwd = _write_output_pointer_files(cfg.local_log_dir, cfg.rollout_prefix)
    _ptr_extra = f"\nPointer files (open in editor): {_ptr_script}"
    if _ptr_cwd and _ptr_cwd != _ptr_script:
        _ptr_extra += f"\n                                  {_ptr_cwd}"
    _banner = (
        "\n"
        "================================================================================\n"
        "COSMOS RoboCasa eval — ALL text logs and rollout MP4/GIF go under:\n"
        f"  {cfg.local_log_dir}\n"
        f"  rollout videos: {_rollout_videos_parent(cfg.local_log_dir, cfg.rollout_prefix)}/<task>--<SESSION_TS>/\n"
        f"Also: {_readme}\n"
        f"{_ptr_extra}\n"
        "If wrong machine path: export COSMOS_POLICY_LOG_DIR=/your/absolute/logs\n"
        "================================================================================\n"
    )
    print(_banner, flush=True)
    print(_banner, file=sys.stderr, flush=True)
    print(
        f"[eval_robocasa] run_robocasa_eval.py (__file__): {Path(__file__).resolve()}",
        flush=True,
    )
    # Initialize T5 text embeddings cache
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    # Load Cosmos Policy dataset stats
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    # If using parallel inference, initialize worker pool
    worker_pool = None
    if cfg.use_parallel_inference:
        available_gpus = [int(gpu.strip()) for gpu in cfg.available_gpus.split(",")]
        available_gpus = available_gpus[: cfg.num_queries_best_of_n]  # Only need N parallel workers
        worker_pool = WorkerPoolManager(cfg, dataset_stats, available_gpus)
        worker_pool.start_workers()
        # Set model to None here because each worker will load its own copy
        model = None
        planning_model = None
    # If using serial inference, initialize model and Cosmos config
    else:
        model, cosmos_config = get_model(cfg)
        assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
            f"Mismatch found between train and test chunk sizes! Train: {cosmos_config.dataloader_train.dataset.chunk_size}, Test: {cfg.chunk_size}"
        )
        worker_pool = None
        # Initialize planning model if specified
        if cfg.planning_model_ckpt_path != "":
            planning_model, _ = get_planning_model(cfg)
        else:
            planning_model = None
    # Setup logging. For resume, append to the existing ENV_EVAL log instead of creating a new one.
    if cfg.resume_log_path:
        local_log_filepath = cfg.resume_log_path
        run_id = Path(local_log_filepath).stem
        log_file = open(local_log_filepath, "a", encoding="utf-8")
        if cfg.use_wandb:
            wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                name=run_id,
                tags=[cfg.task_name, "resume"],
                resume="allow",
            )
        log_message("\n" + "=" * 80, log_file)
        log_message("RESUME EVALUATION", log_file)
        log_message("=" * 80, log_file)
        log_message(f"Appending to existing log: {local_log_filepath}", log_file)
        log_message(
            f"Resume source of truth: rollout videos under {_rollout_videos_parent(cfg.local_log_dir, cfg.rollout_prefix)}",
            log_file,
        )
    else:
        log_file, local_log_filepath, run_id = setup_logging(
            cfg=cfg,
            task_identifier=cfg.task_name,
            log_dir=cfg.local_log_dir,
            run_id_note=cfg.run_id_note,
            use_wandb=cfg.use_wandb,
            wandb_entity=cfg.wandb_entity,
            wandb_project=cfg.wandb_project,
        )
    log_message(f"Eval config: {cfg}", log_file)
    rollout_data_base = _rollout_videos_parent(cfg.local_log_dir, cfg.rollout_prefix)
    os.makedirs(rollout_data_base, exist_ok=True)
    _rollout_msg = f"Rollout MP4 base directory (created at startup): {rollout_data_base}"
    print(f"[eval_robocasa] {_rollout_msg}", flush=True)
    log_message(_rollout_msg, log_file)
    # Log parallel inference configuration if enabled
    if cfg.use_parallel_inference and worker_pool:
        available_gpus = [int(gpu.strip()) for gpu in cfg.available_gpus.split(",")]
        log_message(f"Parallel inference enabled on GPUs: {available_gpus}", log_file)
        log_message(f"Parallel timeout: {cfg.parallel_timeout}s", log_file)
    # Run evaluation
    if cfg.task_name == "all":
        task_list = _task_list_from_cfg(cfg)
        log_message("\nStarting evaluation for all tasks", log_file)
        log_message(f"Total tasks: {len(task_list)}", log_file)
    elif cfg.task_name == "pnp":
        task_list = _task_list_from_cfg(cfg)
        log_message("\nStarting evaluation for all pick-and-place (pnp) tasks", log_file)
        log_message(f"Total pnp tasks: {len(task_list)}", log_file)
    else:
        task_list = _task_list_from_cfg(cfg)
        log_message(f"\nStarting evaluation for task: {cfg.task_name}", log_file)
        log_message(f"Number of trials: {cfg.num_trials_per_task}", log_file)
    if cfg.resume_log_path:
        completed_counts = {
            task_name: cfg.num_trials_per_task
            - len(
                _pending_episode_indices(
                    resume_state,
                    task_name,
                    cfg.num_trials_per_task,
                    cfg.episode_start_idx,
                    cfg.episode_stride,
                )
            )
            for task_name in task_list
        }
        log_message(f"Resume completed episode counts: {completed_counts}", log_file)
        missing_by_task = {
            task_name: _pending_episode_indices(
                resume_state,
                task_name,
                cfg.num_trials_per_task,
                cfg.episode_start_idx,
                cfg.episode_stride,
            )
            for task_name in task_list
            if _pending_episode_indices(
                resume_state,
                task_name,
                cfg.num_trials_per_task,
                cfg.episode_start_idx,
                cfg.episode_stride,
            )
        }
        log_message(f"Resume missing episode indices: {missing_by_task}", log_file)
    total_episodes = 0
    total_successes = 0
    total_length = 0.0
    per_task_results = []
    original_task_name = cfg.task_name

    cfg.task_name = original_task_name

    if cfg.num_parallel_envs > 1 and len(task_list) > 0:
        log_message(f"Using parallel environment mode: {cfg.num_parallel_envs} envs on same GPU across all tasks", log_file)

        # Pre-create all rollout directories so workers can write to them
        for task_name in task_list:
            run_rollout_dir = os.path.join(
                cfg.local_log_dir, cfg.rollout_prefix.strip(), f"{task_name}--{_rollout_session_timestamp()}"
            )
            os.makedirs(run_rollout_dir, exist_ok=True)

        all_work_items = []
        for task_name in task_list:
            for ep_idx in _pending_episode_indices(
                resume_state,
                task_name,
                cfg.num_trials_per_task,
                cfg.episode_start_idx,
                cfg.episode_stride,
            ):
                all_work_items.append((task_name, ep_idx))

        if not all_work_items:
            log_message("Resume: no pending episodes; all requested tasks are already complete", log_file)
            for task_name in task_list:
                success_rate, avg_length, valid_successes = _stats_from_resume_state(
                    resume_state,
                    task_name,
                    cfg.num_trials_per_task,
                    cfg.episode_start_idx,
                    cfg.episode_stride,
                )
                log_message(f"\nTask {task_name} results:", log_file)
                log_message(f"  Success rate: {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
                log_message(f"  Average episode length: {avg_length:.1f}", log_file)
                log_message(f"  Successes: {sum(valid_successes)}/{len(valid_successes)}", log_file)
                per_task_results.append((task_name, success_rate, avg_length, valid_successes))
                total_episodes += len(valid_successes)
                total_successes += sum(valid_successes)
                total_length += avg_length * len(valid_successes)
        else:
            log_message(f"Resume/all-tasks parallel mode queued {len(all_work_items)} pending episodes", log_file)

            num_envs = cfg.num_parallel_envs
            num_workers = min(num_envs, len(all_work_items))
            work_queue: mp.Queue = mp.Queue()
            for item in all_work_items:
                work_queue.put(item)
            for _ in range(num_workers):
                work_queue.put(None)

            obs_queue: mp.Queue = mp.Queue()
            action_queues: list[mp.Queue] = [mp.Queue() for _ in range(num_workers)]
            result_queue: mp.Queue = mp.Queue()

            processes = []
            for i in range(num_workers):
                p = mp.Process(
                    target=_env_worker,
                    args=(
                        i,
                        cfg,
                        work_queue,
                        obs_queue,
                        action_queues[i],
                        result_queue,
                        cfg.local_log_dir,
                        local_log_filepath,
                    ),
                    daemon=True,
                )
                p.start()
                processes.append(p)

            _run_parallel_inference_loop(cfg, model, dataset_stats, obs_queue, action_queues, len(processes), log_file)

            for p in processes:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()

            # Collect results
            task_results: dict[str, dict[int, tuple[bool, Optional[int]]]] = {
                t: _resume_episode_results(resume_state, t) for t in task_list
            }
            while True:
                try:
                    t_name, ep_idx, success, length = result_queue.get_nowait()
                    if t_name in task_results:
                        task_results[t_name][ep_idx] = (success, length)
                except _stdlib_queue.Empty:
                    break

            for task_name in task_list:
                if _is_task_complete_from_episodes(
                    resume_state,
                    task_name,
                    cfg.num_trials_per_task,
                    cfg.episode_start_idx,
                    cfg.episode_stride,
                ):
                    success_rate, avg_length, valid_successes = _stats_from_resume_state(
                        resume_state,
                        task_name,
                        cfg.num_trials_per_task,
                        cfg.episode_start_idx,
                        cfg.episode_stride,
                    )
                else:
                    ordered = [
                        task_results[task_name][i]
                        for i in _requested_episode_indices(
                            cfg.num_trials_per_task, cfg.episode_start_idx, cfg.episode_stride
                        )
                        if i in task_results[task_name]
                    ]
                    valid_successes = [r[0] for r in ordered]
                    valid_lengths = [r[1] for r in ordered if r[1] is not None]

                    success_rate = float(np.mean(valid_successes)) if valid_successes else 0.0
                    avg_length = float(np.mean(valid_lengths)) if valid_lengths else 0.0

                log_message(f"\nTask {task_name} results:", log_file)
                log_message(f"  Success rate: {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
                log_message(f"  Average episode length: {avg_length:.1f}", log_file)
                log_message(f"  Successes: {sum(valid_successes)}/{len(valid_successes)}", log_file)

                if cfg.use_wandb:
                    wandb.log(
                        {
                            f"success_rate/{task_name}": success_rate,
                            f"avg_episode_length/{task_name}": avg_length,
                            f"num_successes/{task_name}": sum(valid_successes),
                            f"num_episodes/{task_name}": len(valid_successes),
                        }
                    )
                per_task_results.append((task_name, success_rate, avg_length, valid_successes))
                total_episodes += len(valid_successes)
                total_successes += sum(valid_successes)
                total_length += avg_length * len(valid_successes)

    else:
        for task_name in task_list:
            cfg.task_name = task_name
            if original_task_name == "all":
                log_message(f"\nStarting evaluation for task: {task_name}", log_file)
                log_message(f"Number of trials: {cfg.num_trials_per_task}", log_file)
            # Create per-task rollout dir now so it exists before workers finish long episodes (MP4 still appears after each ep ends).
            run_rollout_dir = os.path.join(
                cfg.local_log_dir, cfg.rollout_prefix.strip(), f"{task_name}--{_rollout_session_timestamp()}"
            )
            os.makedirs(run_rollout_dir, exist_ok=True)
            _ep_msg = (
                f"Per-task rollout directory (MP4s written here when each episode completes): {run_rollout_dir}"
            )
            print(f"[eval_robocasa] {_ep_msg}", flush=True)
            log_message(_ep_msg, log_file)

            success_rate, avg_length, successes = run_task(
                cfg,
                task_name,
                model,
                planning_model,
                dataset_stats,
                worker_pool,
                log_file,
                resume_state,
            )
            per_task_results.append((task_name, success_rate, avg_length, successes))
            total_episodes += len(successes)
            total_successes += sum(successes)
            total_length += avg_length * len(successes)

    cfg.task_name = original_task_name
    overall_success_rate = (total_successes / total_episodes) if total_episodes > 0 else 0.0
    overall_avg_length = (total_length / total_episodes) if total_episodes > 0 else 0.0
    # Log final results
    log_message("\n" + "=" * 80, log_file)
    log_message("FINAL RESULTS", log_file)
    log_message("=" * 80, log_file)
    if cfg.task_name == "all":
        log_message("Task: all", log_file)
        for task_name, success_rate, avg_length, successes in per_task_results:
            log_message(f"  {task_name} | success: {success_rate:.4f} | avg_len: {avg_length:.1f}", log_file)
        log_message("-" * 80, log_file)
    else:
        log_message(f"Task: {cfg.task_name}", log_file)
    log_message(f"Success rate: {overall_success_rate:.4f} ({int(overall_success_rate * 100)}%)", log_file)
    log_message(f"Average episode length: {overall_avg_length:.1f}", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "final_success_rate": overall_success_rate,
                "final_avg_episode_length": overall_avg_length,
                "total_episodes": total_episodes,
                "total_successes": total_successes,
            }
        )
        wandb.save(local_log_filepath)
        wandb.finish()
    # Cleanup
    if worker_pool:
        worker_pool.shutdown()
    log_message(f"\nResults saved to: {local_log_filepath}", log_file)
    return overall_success_rate


if __name__ == "__main__":
    eval_robocasa()
