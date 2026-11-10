# SPDX-FileCopyrightText: Copyright (c) 2026 MetaWAM Authors.
# SPDX-FileCopyrightText: Portions adapted from Cosmos Policy, Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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
Real-world LeRobot-format dataset loader.

This dataset mirrors the RoboCasa-style interface while reading episodes stored
as LeRobot parquet files.
"""

import io
import json
import os
import pickle
import random
import hashlib
import re
from collections import OrderedDict
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.dataset_common import (
    build_rollout_step_index_mapping,
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    determine_sample_type,
    get_action_chunk_with_padding,
)
from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.utils.utils import duplicate_array

np.set_printoptions(precision=3, linewidth=np.inf)


LEROBOT_STATS_FORMAT_VERSION = 2.0
LEROBOT_ROLL_CANONICALIZATION_CODE = 1.0  # roll_fixed = wrap_to_pi(roll - pi)
LEROBOT_ACTION_SYMMETRIC_FIRST6_CODE = 1.0  # action[:6] uses symmetric min/max around zero


class RealWorldLeRobotDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 32,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        use_wrist_images: bool = True,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 0.5,
        success_rollout_sampling_prob: float = 0.5,
        # If >0, defines the per-step dt threshold above which a gap is
        # considered an inference "stall" (chunk-boundary). The filter then
        # rejects rollout windows that contain more than
        # `rollout_max_stalls_per_window` stalls. Set <=0 to disable the
        # filter entirely.
        # Typical values:
        #   K=8 (open_loop=8):  rollout_max_stalls_per_window=2 keeps everything
        #   K=16 (open_loop=16): =1 keeps everything
        rollout_max_raw_dt_ms: float = 0.0,
        rollout_max_stalls_per_window: int = 2,
        rollout_require_full_action_window: bool = True,
        # Drop the last N frames of each rollout episode before sampling
        # windows. Rationale: rollouts end with a manual keypress (s/f/d), so
        # the very last frames captured may show the operator reaching for the
        # keyboard or a half-completed final step. Default 0 = no truncation.
        rollout_tail_truncate_frames: int = 0,
        p_world_model: float = 0.5,
        treat_success_rollouts_as_demos: bool = False,
        return_value_function_returns: bool = True,
        gamma: float = 0.99,
        lazy_load_demos: bool = True,
        skip_computing_dataset_statistics: bool = False,
        # If True (MetaWAM), current proprio, action chunk, future proprio and value share a single meta
        # latent frame. If False, uses the upstream Cosmos Policy layout with one latent frame per modality.
        use_single_meta_frame: bool = False,
        episode_cache_size: int = 8,
        allowed_commands: list[str] | tuple[str, ...] | None = None,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.rollout_data_dir = rollout_data_dir
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.success_rollout_sampling_prob = success_rollout_sampling_prob
        self.rollout_max_raw_dt_ms = float(rollout_max_raw_dt_ms)
        self.rollout_max_stalls_per_window = int(rollout_max_stalls_per_window)
        self.rollout_require_full_action_window = bool(rollout_require_full_action_window)
        self.rollout_tail_truncate_frames = int(rollout_tail_truncate_frames)
        self.p_world_model = p_world_model
        self.treat_success_rollouts_as_demos = treat_success_rollouts_as_demos
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.lazy_load_demos = lazy_load_demos
        self.use_single_meta_frame = use_single_meta_frame
        self.episode_cache_size = max(int(episode_cache_size), 0)
        self.allowed_commands = None
        self._allowed_command_set = None
        if allowed_commands is not None:
            normalized_allowed_commands = sorted(
                {self._normalize_command_text(command) for command in allowed_commands if str(command).strip()}
            )
            if len(normalized_allowed_commands) == 0:
                raise ValueError("allowed_commands must contain at least one non-empty command when provided.")
            self.allowed_commands = tuple(normalized_allowed_commands)
            self._allowed_command_set = set(normalized_allowed_commands)

        if not self.use_wrist_images and not self.use_third_person_images:
            raise ValueError("Must use at least one of wrist images or third-person images.")

        self.action_dim = 7
        self.raw_proprio_dim = 8
        self.proprio_dim = 9
        self.data = {}
        self.demo_episode_metadata: dict[int, dict[str, Any]] = {}
        self.rollout_episode_metadata: dict[int, dict[str, Any]] = {}
        self.num_episodes = 0
        self.num_steps = 0
        self.rollout_num_episodes = 0
        self.rollout_num_steps = 0
        self.unique_commands: set[str] = set()
        self._episode_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

        self.t5_text_embeddings = None
        self._embedding_normalized_key_to_exact: dict[str, str] = {}
        self._rollout_raw_dt_ms_cache: dict[str, np.ndarray] = {}
        if self.t5_text_embeddings_path:
            with open(self.t5_text_embeddings_path, "rb") as file:
                self.t5_text_embeddings = pickle.load(file)
            self._embedding_normalized_key_to_exact = {
                self._normalize_command_text(key): key for key in self.t5_text_embeddings.keys()
            }

        demo_records = []
        if self.demonstration_sampling_prob > 0:
            demo_records = self._enumerate_episode_records(self.data_dir, default_success=True)
            for record in tqdm(demo_records, desc="Loading RealWorld-LeRobot demo metadata"):
                returns = (
                    compute_monte_carlo_returns(record["num_steps"], terminal_reward=1.0, gamma=self.gamma)
                    if self.return_value_function_returns
                    else None
                )
                self.demo_episode_metadata[self.num_episodes] = {
                    "file_path": record["file_path"],
                    "command": record["command"],
                    "num_steps": record["num_steps"],
                    "success": True,
                    "returns": returns.copy() if returns is not None else None,
                }
                for key in (
                    "source_type",
                    "open_loop_horizon",
                    "resampled_target_hz",
                    "raw_length",
                    "source_path",
                    "source_episode_index",
                ):
                    if key in record and record[key] is not None:
                        self.demo_episode_metadata[self.num_episodes][key] = record[key]
                self.unique_commands.add(record["command"])
                self.num_episodes += 1
                self.num_steps += record["num_steps"]

        self._build_step_index_mapping()

        if not skip_computing_dataset_statistics:
            self.dataset_stats = self._load_or_compute_dataset_statistics(demo_records)
            if self.normalize_actions or self.normalize_proprio:
                self.dataset_stats_post_norm = self._load_or_compute_post_normalization_statistics(demo_records)
            else:
                self.dataset_stats_post_norm = self.dataset_stats
        else:
            self.dataset_stats = None
            self.dataset_stats_post_norm = None

        rollout_records = []
        if self.rollout_data_dir:
            if not os.path.exists(self.rollout_data_dir):
                raise FileNotFoundError(f"Rollout data directory does not exist: {self.rollout_data_dir}")
            rollout_records = self._enumerate_episode_records(self.rollout_data_dir, default_success=True)
            for record in tqdm(rollout_records, desc="Loading RealWorld-LeRobot rollout metadata"):
                success = bool(record["success"])
                rollout_metadata = {
                    "file_path": record["file_path"],
                    "command": record["command"],
                    "num_steps": record["num_steps"],
                    "success": success,
                }
                for key in (
                    "source_type",
                    "open_loop_horizon",
                    "resampled_target_hz",
                    "raw_length",
                    "source_path",
                    "source_episode_index",
                ):
                    if key in record and record[key] is not None:
                        rollout_metadata[key] = record[key]
                if self.return_value_function_returns:
                    terminal_reward = 1.0 if success else 0.0
                    rollout_metadata["returns"] = compute_monte_carlo_returns(
                        record["num_steps"], terminal_reward=terminal_reward, gamma=self.gamma
                    )
                self.rollout_episode_metadata[self.rollout_num_episodes] = rollout_metadata
                self.unique_commands.add(record["command"])
                self.rollout_num_episodes += 1
                self.rollout_num_steps += record["num_steps"]

        if self.treat_success_rollouts_as_demos:
            for rollout_episode_idx, episode_metadata in self.rollout_episode_metadata.items():
                if not episode_metadata.get("success", False):
                    continue
                self.demo_episode_metadata[self.num_episodes] = {
                    "file_path": episode_metadata["file_path"],
                    "command": episode_metadata["command"],
                    "num_steps": episode_metadata["num_steps"],
                    "success": True,
                    "returns": (
                        episode_metadata["returns"].copy()
                        if episode_metadata.get("returns") is not None
                        else None
                    ),
                }
                for key in (
                    "source_type",
                    "open_loop_horizon",
                    "resampled_target_hz",
                    "raw_length",
                    "source_path",
                    "source_episode_index",
                ):
                    if key in episode_metadata and episode_metadata[key] is not None:
                        self.demo_episode_metadata[self.num_episodes][key] = episode_metadata[key]
                self.num_episodes += 1
                self.num_steps += episode_metadata["num_steps"]
            self._build_step_index_mapping()

        self._build_rollout_step_index_mapping()
        self._calculate_epoch_structure()
        print(f"Number of unique commands found: {len(self.unique_commands)}")

    @staticmethod
    def _normalize_command_text(command: str) -> str:
        return " ".join(str(command).strip().split()).rstrip(".").lower()

    @staticmethod
    def _is_split_root(path: str) -> bool:
        return os.path.isfile(os.path.join(path, "meta", "episodes.jsonl")) and os.path.isdir(os.path.join(path, "data"))

    def _discover_split_roots(self, base_dir: str) -> list[str]:
        if self._is_split_root(base_dir):
            return [base_dir]

        split_roots = []
        for child_name in sorted(os.listdir(base_dir)):
            child_path = os.path.join(base_dir, child_name)
            if os.path.isdir(child_path) and self._is_split_root(child_path):
                split_roots.append(child_path)

        if len(split_roots) == 0:
            raise FileNotFoundError(
                f"Could not find a LeRobot split root under '{base_dir}'. Expected meta/episodes.jsonl and data/."
            )
        return split_roots

    @staticmethod
    def _infer_success_from_path(path: str, default_success: bool) -> bool:
        norm_parts = os.path.normpath(path).split(os.sep)
        if any(part in {"failure", "fail", "failed", "failures"} for part in norm_parts):
            return False
        if any(part in {"success_only", "success", "successful"} for part in norm_parts):
            return True
        return default_success

    @staticmethod
    def _extract_episode_index_from_filename(filename: str) -> int | None:
        stem, ext = os.path.splitext(filename)
        if ext.lower() != ".parquet" or not stem.startswith("episode_"):
            return None
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return None

    def _build_episode_path_map(self, split_root: str) -> dict[int, str]:
        episode_path_map = {}
        data_root = os.path.join(split_root, "data")
        for root, _, files in os.walk(data_root):
            for file_name in files:
                episode_index = self._extract_episode_index_from_filename(file_name)
                if episode_index is None:
                    continue
                episode_path_map[episode_index] = os.path.join(root, file_name)
        return episode_path_map

    @staticmethod
    def _extract_command_from_episode_payload(payload: dict[str, Any]) -> str:
        tasks = payload.get("tasks")
        if isinstance(tasks, list) and len(tasks) > 0 and isinstance(tasks[0], str):
            return tasks[0].strip()
        task = payload.get("task")
        if isinstance(task, str):
            return task.strip()
        return ""

    @staticmethod
    def _read_jsonl(path: str) -> list[dict[str, Any]]:
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _infer_source_type_from_payload(cls, payload: dict[str, Any], path_success: bool) -> str:
        explicit_source_type = str(payload.get("source_type", "")).strip().lower()
        if explicit_source_type:
            return explicit_source_type
        if any(key in payload for key in ("raw_length", "resampled_target_hz", "collected_at", "wrist_vis_path")):
            return "rollout"
        if "success" in payload and not path_success:
            return "rollout"
        return "human"

    @classmethod
    def _infer_open_loop_horizon_from_payload(cls, payload: dict[str, Any]) -> int | None:
        explicit_horizon = cls._coerce_optional_int(payload.get("open_loop_horizon"))
        if explicit_horizon is not None and explicit_horizon > 0:
            return explicit_horizon

        for key in ("merged_from", "source_path"):
            value = payload.get(key)
            if not isinstance(value, str):
                continue
            match = re.search(r"(?:^|[_/])loop(\d+)(?:$|[_/])", value)
            if match:
                return int(match.group(1))
        return None

    def _enumerate_episode_records(self, base_dir: str, default_success: bool) -> list[dict[str, Any]]:
        records = []
        for split_root in self._discover_split_roots(base_dir):
            # Path-inferred success is the fallback for episodes without an
            # explicit `success` field in episodes.jsonl (e.g. legacy teleop
            # demos under success_only/).
            path_success = self._infer_success_from_path(split_root, default_success)
            episode_path_map = self._build_episode_path_map(split_root)
            episodes_path = os.path.join(split_root, "meta", "episodes.jsonl")

            for payload in self._read_jsonl(episodes_path):
                episode_index = int(payload["episode_index"])
                if episode_index not in episode_path_map:
                    raise FileNotFoundError(
                        f"Episode {episode_index} listed in {episodes_path} but parquet file was not found under {split_root}."
                    )
                command = self._extract_command_from_episode_payload(payload)
                if not command:
                    command = f"task_{payload.get('task_index', 0)}"
                if (
                    self._allowed_command_set is not None
                    and self._normalize_command_text(command) not in self._allowed_command_set
                ):
                    continue
                # Rollout collectors (collect_rollouts.py, retime_existing_rollouts.py)
                # write a per-episode `success` flag; trust it over the path
                # heuristic so success and failure rollouts can live in the same
                # split directory.
                episode_success = bool(payload["success"]) if "success" in payload else path_success
                source_type = self._infer_source_type_from_payload(payload, path_success=path_success)
                records.append(
                    {
                        "episode_index": episode_index,
                        "file_path": episode_path_map[episode_index],
                        "command": command,
                        "num_steps": int(payload["length"]),
                        "success": episode_success,
                        "source_type": source_type,
                        "open_loop_horizon": self._infer_open_loop_horizon_from_payload(payload),
                        "resampled_target_hz": payload.get("resampled_target_hz"),
                        "raw_length": (
                            int(payload["raw_length"]) if payload.get("raw_length") is not None else None
                        ),
                        "source_path": payload.get("source_path"),
                        "source_episode_index": self._coerce_optional_int(payload.get("source_episode_index")),
                    }
                )

        records.sort(key=lambda x: (x["file_path"], x["episode_index"]))
        return records

    def _get_dataset_statistics_path(self, post_norm: bool = False) -> str:
        file_stem = "dataset_statistics_post_norm" if post_norm else "dataset_statistics"
        if self.allowed_commands is None:
            return os.path.join(self.data_dir, f"{file_stem}.json")

        digest = hashlib.sha1("||".join(self.allowed_commands).encode("utf-8")).hexdigest()[:10]
        return os.path.join(self.data_dir, f"{file_stem}__cmdfilter_{digest}.json")

    @staticmethod
    def _decode_image_struct(image_struct: dict[str, Any]) -> np.ndarray:
        image_bytes = image_struct.get("bytes")
        image_path = image_struct.get("path")
        if image_bytes is not None:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return np.array(image).astype(np.uint8)
        if image_path:
            image = Image.open(image_path).convert("RGB")
            return np.array(image).astype(np.uint8)
        raise ValueError("Image struct does not contain bytes or path.")

    @staticmethod
    def _load_json_stats(path: str) -> dict[str, np.ndarray]:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {key: np.asarray(value, dtype=np.float32) for key, value in payload.items()}

    @staticmethod
    def _wrap_to_pi(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        return ((array + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)

    @classmethod
    def _canonicalize_state_roll(cls, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).copy()
        if state.shape[-1] < 4:
            raise ValueError(f"Expected state with at least 4 dims, got shape {state.shape}")
        state[..., 3] = cls._wrap_to_pi(state[..., 3] - np.pi)
        return state.astype(np.float32)

    @staticmethod
    def _clamp_action_gripper(actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32).copy()
        if actions.shape[-1] >= 7:
            actions[..., 6] = np.clip(actions[..., 6], 0.0, 1.0)
        return actions.astype(np.float32)

    def _build_action_normalization_bounds(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        actions = np.asarray(actions, dtype=np.float32)
        action_min = np.min(actions, axis=0).astype(np.float32)
        action_max = np.max(actions, axis=0).astype(np.float32)
        if action_min.shape[0] < 6:
            raise ValueError(f"Expected at least 6 action dims for symmetric normalization, got {action_min.shape}")
        symmetric_scale = np.maximum(np.abs(action_min[:6]), np.abs(action_max[:6])).astype(np.float32)
        action_min[:6] = -symmetric_scale
        action_max[:6] = symmetric_scale
        return action_min, action_max

    @staticmethod
    def _get_scalar_stat(stats: dict[str, np.ndarray], key: str, default: float = 0.0) -> float:
        if key not in stats:
            return float(default)
        value = np.asarray(stats[key], dtype=np.float32).reshape(-1)
        if value.size == 0:
            return float(default)
        return float(value[0])

    def _stats_match_expected_format(self, stats: dict[str, np.ndarray]) -> bool:
        if stats.get("actions_min", np.array([])).shape != (self.action_dim,):
            return False
        if stats.get("state_min", np.array([])).shape != (self.raw_proprio_dim,):
            return False
        version = self._get_scalar_stat(stats, "stats_format_version", default=0.0)
        roll_code = self._get_scalar_stat(stats, "state_roll_canonicalization_code", default=0.0)
        action_code = self._get_scalar_stat(stats, "action_symmetric_first6_code", default=0.0)
        return (
            version >= LEROBOT_STATS_FORMAT_VERSION
            and roll_code == LEROBOT_ROLL_CANONICALIZATION_CODE
            and action_code == LEROBOT_ACTION_SYMMETRIC_FIRST6_CODE
        )

    @staticmethod
    def _save_json_stats(path: str, stats: dict[str, np.ndarray]) -> None:
        payload = {key: value.tolist() for key, value in stats.items()}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)
            handle.write("\n")

    @staticmethod
    def _summarize_array(array: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "min": np.min(array, axis=0).astype(np.float32),
            "max": np.max(array, axis=0).astype(np.float32),
            "mean": np.mean(array, axis=0).astype(np.float32),
            "std": np.std(array, axis=0).astype(np.float32),
            "median": np.median(array, axis=0).astype(np.float32),
        }

    def _build_stats_dict(
        self,
        actions: np.ndarray,
        state: np.ndarray,
        use_action_normalization_bounds: bool = True,
    ) -> dict[str, np.ndarray]:
        action_summary = self._summarize_array(actions)
        state_summary = self._summarize_array(state)
        if use_action_normalization_bounds:
            actions_min, actions_max = self._build_action_normalization_bounds(actions)
        else:
            actions_min, actions_max = action_summary["min"], action_summary["max"]
        return {
            "actions_min": actions_min,
            "actions_max": actions_max,
            "actions_mean": action_summary["mean"],
            "actions_std": action_summary["std"],
            "actions_median": action_summary["median"],
            "state_min": state_summary["min"],
            "state_max": state_summary["max"],
            "state_mean": state_summary["mean"],
            "state_std": state_summary["std"],
            "state_median": state_summary["median"],
            "stats_format_version": np.asarray(LEROBOT_STATS_FORMAT_VERSION, dtype=np.float32),
            "state_roll_canonicalization_code": np.asarray(LEROBOT_ROLL_CANONICALIZATION_CODE, dtype=np.float32),
            "action_symmetric_first6_code": np.asarray(LEROBOT_ACTION_SYMMETRIC_FIRST6_CODE, dtype=np.float32),
        }

    def _compute_dataset_statistics(self, records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        if len(records) == 0:
            raise ValueError("Cannot compute dataset statistics without any demonstration episodes.")

        all_actions = []
        all_state = []
        for record in tqdm(records, desc="Computing RealWorld-LeRobot statistics"):
            table = pq.read_table(record["file_path"], columns=["state", "actions"])
            all_state.append(self._canonicalize_state_roll(np.asarray(table["state"].to_pylist(), dtype=np.float32)))
            all_actions.append(self._clamp_action_gripper(np.asarray(table["actions"].to_pylist(), dtype=np.float32)))

        return self._build_stats_dict(np.concatenate(all_actions, axis=0), np.concatenate(all_state, axis=0))

    def _load_or_compute_dataset_statistics(self, records: list[dict[str, Any]]) -> dict[str, np.ndarray] | None:
        if len(records) == 0:
            return None

        path = self._get_dataset_statistics_path(post_norm=False)
        if os.path.exists(path):
            stats = self._load_json_stats(path)
            if self._stats_match_expected_format(stats):
                print(f"Loaded dataset statistics from: {path}")
                return stats
            print(f"Ignoring incompatible statistics file at: {path}")

        stats = self._compute_dataset_statistics(records)
        self._save_json_stats(path, stats)
        print(f"Dataset statistics saved to: {path}")
        return stats

    def _normalize_array(self, array: np.ndarray, prefix: str) -> np.ndarray:
        if self.dataset_stats is None:
            return array.astype(np.float32)
        min_val = self.dataset_stats[f"{prefix}_min"]
        max_val = self.dataset_stats[f"{prefix}_max"]
        denom = np.where(max_val == min_val, 1.0, max_val - min_val)
        return (2.0 * ((array - min_val) / denom) - 1.0).astype(np.float32)

    def _expand_proprio_to_model_dim(self, proprio: np.ndarray) -> np.ndarray:
        proprio = np.asarray(proprio, dtype=np.float32)
        if proprio.shape[-1] >= self.proprio_dim:
            return proprio[..., : self.proprio_dim].astype(np.float32)

        # LeRobot real-world state is 8D, while the model expects 9D proprio.
        # Use zero padding to mark the missing channel neutrally instead of duplicating gripper/state values.
        pad_shape = list(proprio.shape)
        pad_shape[-1] = self.proprio_dim - proprio.shape[-1]
        padding = np.zeros(pad_shape, dtype=np.float32)
        return np.concatenate([proprio, padding], axis=-1).astype(np.float32)

    def _load_or_compute_post_normalization_statistics(self, records: list[dict[str, Any]]) -> dict[str, np.ndarray] | None:
        if len(records) == 0:
            return None

        path = self._get_dataset_statistics_path(post_norm=True)
        if os.path.exists(path):
            stats = self._load_json_stats(path)
            if self._stats_match_expected_format(stats):
                print(f"Loaded post-normalization dataset statistics from: {path}")
                return stats
            print(f"Ignoring incompatible post-normalization statistics file at: {path}")

        all_actions = []
        all_state = []
        for record in tqdm(records, desc="Computing RealWorld-LeRobot post-norm statistics"):
            table = pq.read_table(record["file_path"], columns=["state", "actions"])
            state = self._canonicalize_state_roll(np.asarray(table["state"].to_pylist(), dtype=np.float32))
            actions = self._clamp_action_gripper(np.asarray(table["actions"].to_pylist(), dtype=np.float32))
            all_state.append(self._normalize_array(state, "state"))
            all_actions.append(self._normalize_array(actions, "actions"))

        stats = self._build_stats_dict(
            np.concatenate(all_actions, axis=0),
            np.concatenate(all_state, axis=0),
            use_action_normalization_bounds=False,
        )
        self._save_json_stats(path, stats)
        print(f"Post-normalization dataset statistics saved to: {path}")
        return stats

    def _build_step_index_mapping(self) -> None:
        self._step_to_episode_map = {}
        self._total_steps = 0
        for episode_idx, episode_metadata in self.demo_episode_metadata.items():
            num_steps = int(episode_metadata["num_steps"])
            for step_idx in range(num_steps):
                self._step_to_episode_map[self._total_steps] = (episode_idx, step_idx)
                self._total_steps += 1

    def _build_rollout_step_index_mapping(self) -> None:
        result = build_rollout_step_index_mapping({}, self.rollout_episode_metadata)
        success_map = result["_rollout_success_step_to_episode_map"]
        failure_map = result["_rollout_failure_step_to_episode_map"]
        success_total = result["_rollout_success_total_steps"]
        failure_total = result["_rollout_failure_total_steps"]

        if self.rollout_max_raw_dt_ms > 0 or self.rollout_tail_truncate_frames > 0:
            success_map, success_total, filtered_success, success_breakdown = self._filter_rollout_map_by_raw_dt(
                success_map, success_total
            )
            failure_map, failure_total, filtered_failure, failure_breakdown = self._filter_rollout_map_by_raw_dt(
                failure_map, failure_total
            )
            print(
                "Rollout action-consistency filter: "
                f"stall_threshold_ms={self.rollout_max_raw_dt_ms:.1f}, "
                f"max_stalls_per_window={self.rollout_max_stalls_per_window}, "
                f"tail_truncate_frames={self.rollout_tail_truncate_frames}, "
                f"require_full_action_window={self.rollout_require_full_action_window}, "
                f"kept_success={success_total}, dropped_success={filtered_success}, "
                f"kept_failure={failure_total}, dropped_failure={filtered_failure}"
            )
            breakdown_labels = sorted(
                set(success_breakdown) | set(failure_breakdown),
                key=lambda label: (label not in {"loop4", "loop8", "loop16"}, label),
            )
            for label in breakdown_labels:
                success_entry = success_breakdown.get(label, {"kept": 0, "dropped_reasons": {}})
                failure_entry = failure_breakdown.get(label, {"kept": 0, "dropped_reasons": {}})
                reason_keys = sorted(
                    set(success_entry["dropped_reasons"].keys()) | set(failure_entry["dropped_reasons"].keys())
                )
                parts = [
                    f"{label} kept_success={success_entry['kept']}",
                    f"kept_failure={failure_entry['kept']}",
                ]
                for reason in reason_keys:
                    parts.append(
                        f"dropped_{reason}_success={success_entry['dropped_reasons'].get(reason, 0)}"
                    )
                    parts.append(
                        f"dropped_{reason}_failure={failure_entry['dropped_reasons'].get(reason, 0)}"
                    )
                print("  " + ", ".join(parts))

        self._rollout_success_step_to_episode_map = success_map
        self._rollout_failure_step_to_episode_map = failure_map
        self._rollout_success_total_steps = success_total
        self._rollout_failure_total_steps = failure_total
        self._rollout_total_steps = success_total + failure_total

    def _load_rollout_raw_dt_ms(self, episode_metadata: dict[str, Any]) -> np.ndarray:
        """Return outgoing raw dt (ms) for each row of a rollout parquet.

        New rollout files include a `raw_dt_ms` column. Older files do not; for
        those we fall back to timestamp diffs. The last row repeats the previous
        gap so callers can index a length-T vector uniformly.
        """
        file_path = episode_metadata["file_path"]
        cached = self._rollout_raw_dt_ms_cache.get(file_path)
        if cached is not None:
            return cached

        try:
            table = pq.read_table(file_path, columns=["raw_dt_ms"])
            raw_dt_ms = np.asarray(table["raw_dt_ms"].to_pylist(), dtype=np.float32)
        except Exception:
            table = pq.read_table(file_path, columns=["timestamp"])
            ts = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
            if ts.shape[0] >= 2:
                diffs = (np.diff(ts) * 1000.0).astype(np.float32)
                raw_dt_ms = np.concatenate([diffs, diffs[-1:]]).astype(np.float32)
            else:
                raw_dt_ms = np.zeros((ts.shape[0],), dtype=np.float32)

        self._rollout_raw_dt_ms_cache[file_path] = raw_dt_ms
        return raw_dt_ms

    @staticmethod
    def _rollout_horizon_label(episode_metadata: dict[str, Any]) -> str:
        horizon = episode_metadata.get("open_loop_horizon")
        if isinstance(horizon, (int, float)) and int(horizon) > 0:
            return f"loop{int(horizon)}"
        source_type = str(episode_metadata.get("source_type", "")).strip().lower()
        if source_type and source_type != "rollout":
            return source_type
        return "loop_unknown"

    def _evaluate_rollout_step_action_window(
        self, episode_idx: int, relative_step_idx: int
    ) -> tuple[bool, str, str]:
        metadata = self.rollout_episode_metadata[episode_idx]
        num_steps = int(metadata["num_steps"])
        source_type = str(metadata.get("source_type", "")).strip().lower() or "rollout"
        open_loop_horizon = self._coerce_optional_int(metadata.get("open_loop_horizon"))
        horizon_label = self._rollout_horizon_label(metadata)
        is_rollout_episode = source_type == "rollout"

        # Human episodes copied into all_episodes/success should keep their full
        # teleop tail. The manual keyboard reaction tail truncation only applies
        # to on-policy rollout files.
        effective_num_steps = (
            max(num_steps - self.rollout_tail_truncate_frames, 0) if is_rollout_episode else num_steps
        )

        # A full action-conditioned training sample uses actions[t:t+chunk_size]
        # and future frame t+chunk_size. If this is disabled, end-of-episode
        # padding remains allowed (legacy behavior), but the raw-dt check still
        # covers the available part of the action window.
        if self.rollout_require_full_action_window and relative_step_idx + self.chunk_size >= effective_num_steps:
            return False, "tail_or_short_window", horizon_label

        # Human episodes routed through all_episodes/success do not carry
        # rollout stalls, so keep them once the full-window check passes.
        if (not is_rollout_episode) or self.rollout_max_raw_dt_ms <= 0:
            return True, "kept", horizon_label

        raw_dt_ms = self._load_rollout_raw_dt_ms(metadata)
        if raw_dt_ms.shape[0] == 0:
            return False, "missing_raw_dt", horizon_label

        end = min(relative_step_idx + self.chunk_size, raw_dt_ms.shape[0])
        if end <= relative_step_idx:
            return False, "empty_window", horizon_label

        stall_offsets = np.flatnonzero(raw_dt_ms[relative_step_idx:end] > self.rollout_max_raw_dt_ms)
        if stall_offsets.size == 0:
            return True, "kept", horizon_label

        if open_loop_horizon is None or open_loop_horizon <= 0:
            n_stalls = int(stall_offsets.size)
            if n_stalls > self.rollout_max_stalls_per_window:
                return False, "excess_stalls", horizon_label
            return True, "kept", horizon_label

        absolute_frame_indices = relative_step_idx + stall_offsets
        if np.any((absolute_frame_indices % open_loop_horizon) != 0):
            return False, "non_boundary_stall", horizon_label

        natural_boundary_limit = max(self.chunk_size // open_loop_horizon, 1)
        allowed_stalls = min(self.rollout_max_stalls_per_window, natural_boundary_limit)
        if int(stall_offsets.size) > allowed_stalls:
            return False, "excess_boundary_stalls", horizon_label
        return True, "kept", horizon_label

    def _rollout_step_has_clean_action_window(self, episode_idx: int, relative_step_idx: int) -> bool:
        keep, _, _ = self._evaluate_rollout_step_action_window(episode_idx, relative_step_idx)
        return keep

    def _filter_rollout_map_by_raw_dt(
        self,
        step_map: dict[int, tuple[int, int]],
        total_steps: int,
    ) -> tuple[dict[int, tuple[int, int]], int, int, dict[str, dict[str, Any]]]:
        filtered_map = {}
        kept = 0
        dropped = 0
        breakdown: dict[str, dict[str, Any]] = {}
        for i in range(total_steps):
            episode_idx, relative_step_idx = step_map[i]
            keep, reason, horizon_label = self._evaluate_rollout_step_action_window(episode_idx, relative_step_idx)
            breakdown_entry = breakdown.setdefault(horizon_label, {"kept": 0, "dropped_reasons": {}})
            if keep:
                filtered_map[kept] = (episode_idx, relative_step_idx)
                kept += 1
                breakdown_entry["kept"] += 1
            else:
                dropped += 1
                breakdown_entry["dropped_reasons"][reason] = breakdown_entry["dropped_reasons"].get(reason, 0) + 1
        return filtered_map, kept, dropped, breakdown

    def _calculate_epoch_structure(self) -> None:
        result = calculate_epoch_structure(
            num_steps=self.num_steps,
            rollout_success_total_steps=self._rollout_success_total_steps,
            rollout_failure_total_steps=self._rollout_failure_total_steps,
            demonstration_sampling_prob=self.demonstration_sampling_prob,
            success_rollout_sampling_prob=self.success_rollout_sampling_prob,
        )
        self.adjusted_demo_count = result["adjusted_demo_count"]
        self.adjusted_success_rollout_count = result["adjusted_success_rollout_count"]
        self.adjusted_failure_rollout_count = result["adjusted_failure_rollout_count"]
        self.epoch_length = result["epoch_length"]

    def _cache_get(self, file_path: str) -> dict[str, Any] | None:
        cached = self._episode_cache.get(file_path)
        if cached is None:
            return None
        self._episode_cache.move_to_end(file_path)
        return cached

    def _cache_put(self, file_path: str, payload: dict[str, Any]) -> None:
        if self.episode_cache_size <= 0:
            return
        self._episode_cache[file_path] = payload
        self._episode_cache.move_to_end(file_path)
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)

    def _load_parquet_episode(self, file_path: str) -> dict[str, Any]:
        cached = self._cache_get(file_path)
        if cached is not None:
            return cached

        table = pq.read_table(
            file_path,
            columns=["image", "wrist_image", "left_image", "state", "actions"],
            use_threads=False,
            memory_map=False,
            pre_buffer=False,
        )
        payload = {
            "primary_image_column": table["image"],
            "wrist_image_column": table["wrist_image"],
            "secondary_image_column": table["left_image"],
            "proprio": self._canonicalize_state_roll(np.asarray(table["state"].to_pylist(), dtype=np.float32)),
            "actions": self._clamp_action_gripper(np.asarray(table["actions"].to_pylist(), dtype=np.float32)),
            "num_steps": int(table.num_rows),
        }
        self._cache_put(file_path, payload)
        return payload

    def _load_episode_data(self, episode_metadata: dict[str, Any], frame_indices: set[int] | None = None) -> dict[str, Any]:
        payload = self._load_parquet_episode(episode_metadata["file_path"])
        if frame_indices is None:
            raise ValueError("RealWorldLeRobotDataset expects explicit frame_indices for lazy loading.")

        sorted_indices = sorted(frame_indices)
        primary_images = {}
        wrist_images = {}
        secondary_images = {}
        for idx in sorted_indices:
            primary_images[idx] = self._decode_image_struct(payload["primary_image_column"][idx].as_py())
            wrist_images[idx] = self._decode_image_struct(payload["wrist_image_column"][idx].as_py())
            secondary_images[idx] = self._decode_image_struct(payload["secondary_image_column"][idx].as_py())

        proprio = payload["proprio"].copy()
        actions = payload["actions"].copy()
        if self.normalize_proprio:
            proprio = self._normalize_array(proprio, "state")
        proprio = self._expand_proprio_to_model_dim(proprio)
        if self.normalize_actions:
            actions = self._normalize_array(actions, "actions")

        return {
            "primary_images": primary_images,
            "wrist_images": wrist_images,
            "secondary_images": secondary_images,
            "proprio": proprio,
            "actions": actions,
            "command": episode_metadata["command"],
            "num_steps": int(episode_metadata["num_steps"]),
            "success": bool(episode_metadata.get("success", True)),
            "returns": episode_metadata.get("returns"),
        }

    def _resolve_embedding_key(self, command: str) -> str:
        if self.t5_text_embeddings is None:
            raise ValueError("t5_text_embeddings_path must be provided for RealWorldLeRobotDataset.")
        if command in self.t5_text_embeddings:
            return command
        normalized = self._normalize_command_text(command)
        if normalized in self._embedding_normalized_key_to_exact:
            return self._embedding_normalized_key_to_exact[normalized]
        raise KeyError(f"Command '{command}' not found in T5 embeddings.")

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_type = determine_sample_type(idx, self.adjusted_demo_count, self.adjusted_success_rollout_count)
        rollout_data_mask = 1 if sample_type != "demo" else 0
        rollout_data_success_mask = 1 if sample_type == "success_rollout" else 0

        if sample_type == "demo":
            global_step_idx = idx % self.num_steps
            episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
            episode_metadata = self.demo_episode_metadata[episode_idx]
            global_rollout_idx = -1
        elif sample_type == "success_rollout":
            success_idx = idx - self.adjusted_demo_count
            global_rollout_idx = success_idx % self._rollout_success_total_steps
            episode_idx, relative_step_idx = self._rollout_success_step_to_episode_map[global_rollout_idx]
            episode_metadata = self.rollout_episode_metadata[episode_idx]
        else:
            failure_idx = idx - self.adjusted_demo_count - self.adjusted_success_rollout_count
            global_rollout_idx = failure_idx % self._rollout_failure_total_steps
            episode_idx, relative_step_idx = self._rollout_failure_step_to_episode_map[global_rollout_idx]
            episode_metadata = self.rollout_episode_metadata[episode_idx]

        future_frame_idx = min(relative_step_idx + self.chunk_size, int(episode_metadata["num_steps"]) - 1)
        episode_data = self._load_episode_data(episode_metadata, frame_indices={relative_step_idx, future_frame_idx})

        is_world_model_sample = False
        is_value_function_sample = False
        if sample_type != "demo":
            if self.return_value_function_returns:
                if random.random() < self.p_world_model:
                    is_world_model_sample = True
                else:
                    is_value_function_sample = True
            else:
                is_world_model_sample = True

        current_primary = episode_data["primary_images"][relative_step_idx]
        current_secondary = episode_data["secondary_images"][relative_step_idx]
        current_wrist = episode_data["wrist_images"][relative_step_idx]
        future_primary = episode_data["primary_images"][future_frame_idx]
        future_secondary = episode_data["secondary_images"][future_frame_idx]
        future_wrist = episode_data["wrist_images"][future_frame_idx]
        current_proprio = episode_data["proprio"][relative_step_idx]
        future_proprio = episode_data["proprio"][future_frame_idx]

        image_list = []
        current_sequence_idx = 0
        current_proprio_latent_idx = -1
        current_wrist_image_latent_idx = -1
        current_image_latent_idx = -1
        current_image2_latent_idx = -1
        action_latent_idx = -1
        future_proprio_latent_idx = -1
        future_wrist_image_latent_idx = -1
        future_image_latent_idx = -1
        future_image2_latent_idx = -1
        value_latent_idx = -1

        first_input_image = np.expand_dims(np.zeros_like(current_primary), axis=0)
        image_list.append(first_input_image)
        current_sequence_idx += 1

        # Upstream mode: dedicated proprio latent frame. MetaWAM mode: proprio is written into the meta frame.
        if self.use_proprio and not self.use_single_meta_frame:
            blank_image = np.zeros_like(current_primary)
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            current_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        if self.use_wrist_images:
            wrist_image = duplicate_array(current_wrist, total_num_copies=self.num_duplicates_per_image)
            image_list.append(wrist_image)
            current_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        if self.use_third_person_images:
            current_left_image = duplicate_array(current_primary, total_num_copies=self.num_duplicates_per_image)
            image_list.append(current_left_image)
            current_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

            current_right_image = duplicate_array(current_secondary, total_num_copies=self.num_duplicates_per_image)
            image_list.append(current_right_image)
            current_image2_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        blank_image = np.zeros_like(current_primary)
        blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
        image_list.append(blank_image)
        action_latent_idx = current_sequence_idx
        current_sequence_idx += 1

        if self.use_proprio:
            if self.use_single_meta_frame:
                future_proprio_latent_idx = action_latent_idx
            else:
                blank_image = np.zeros_like(current_primary)
                blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
                image_list.append(blank_image)
                future_proprio_latent_idx = current_sequence_idx
                current_sequence_idx += 1

        if self.use_wrist_images:
            future_wrist_image = duplicate_array(future_wrist, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_wrist_image)
            future_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        if self.use_third_person_images:
            future_left_image = duplicate_array(future_primary, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_left_image)
            future_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

            future_right_image = duplicate_array(future_secondary, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_right_image)
            future_image2_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        if self.return_value_function_returns:
            if self.use_single_meta_frame:
                value_latent_idx = action_latent_idx
            else:
                value_image = np.zeros_like(current_primary)
                value_image = duplicate_array(value_image, total_num_copies=self.num_duplicates_per_image)
                image_list.append(value_image)
                value_latent_idx = current_sequence_idx
                current_sequence_idx += 1

        images = np.concatenate(image_list, axis=0)
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        if self.return_value_function_returns:
            value_function_return = episode_metadata["returns"][future_frame_idx]
        else:
            value_function_return = float("-100")

        embedding_key = self._resolve_embedding_key(episode_data["command"])
        sample_dict = {
            "video": images,
            "command": episode_data["command"],
            "actions": action_chunk,
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[embedding_key]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": current_proprio if self.use_proprio else np.zeros(self.proprio_dim, dtype=np.float32),
            "future_proprio": future_proprio if self.use_proprio else np.zeros(self.proprio_dim, dtype=np.float32),
            "__key__": idx,
            "rollout_data_mask": rollout_data_mask,
            "rollout_data_success_mask": rollout_data_success_mask,
            "world_model_sample_mask": 1 if is_world_model_sample else 0,
            "value_function_sample_mask": 1 if is_value_function_sample else 0,
            "global_rollout_idx": global_rollout_idx,
            "meta_latent_idx": action_latent_idx,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": value_latent_idx if self.return_value_function_returns else -1,
            "current_proprio_latent_idx": current_proprio_latent_idx if self.use_proprio else -1,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx if self.use_wrist_images else -1,
            "current_image_latent_idx": current_image_latent_idx if self.use_third_person_images else -1,
            "current_image2_latent_idx": current_image2_latent_idx if self.use_third_person_images else -1,
            "future_proprio_latent_idx": future_proprio_latent_idx if self.use_proprio else -1,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx if self.use_wrist_images else -1,
            "future_image_latent_idx": future_image_latent_idx if self.use_third_person_images else -1,
            "future_image2_latent_idx": future_image2_latent_idx if self.use_third_person_images else -1,
            "value_function_return": value_function_return,
        }
        return sample_dict
