"""
deploy_franka.py

Starts a Cosmos-Policy inference server for Franka robot control.
The server receives observations (images + proprio) via HTTP and returns actions.

This runs on the GPU machine (policy-server side).

Setup:
    Install the policy-server dependencies listed in requirements_franka.txt
    (or use the `aloha` uv dependency group, see run_deploy_franka.sh).

Usage:
    python -m cosmos_policy.experiments.robot.franka.deploy_franka \
        --config <CONFIG_NAME> \
        --ckpt_path <CHECKPOINT_PATH> \
        --dataset_stats_path <STATS_PATH> \
        --t5_text_embeddings_path <EMBEDDINGS_PATH>
"""

import json
import logging
import os
import secrets
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict

import draccus
import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from cosmos_policy.experiments.robot.cosmos_utils import (
    extract_action_chunk,
    get_action,
    get_action_shape,
    get_future_state_prediction,
    get_model,
    get_planning_model,
    get_qvalue_prediction,
    get_value_prediction,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    unnormalize_actions,
)
from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size
from cosmos_policy.utils.utils import set_seed_everywhere

json_numpy.patch()


@dataclass
class FrankaDeployConfig:
    # LeRobot Franka checkpoints use the RoboCasa-style policy input layout:
    # one wrist image plus two third-person images.
    suite: str = "robocasa"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8777

    # Model-specific parameters
    model_family: str = "cosmos"
    config: str = "metawam_realworld__inference"        # Inference config name
    ckpt_path: str = ""                                 # Model checkpoint path
    config_file: str = "cosmos_policy/config/config.py"

    # Data normalization
    dataset_stats_path: str = ""                        # Dataset statistics JSON of the training run
    t5_text_embeddings_path: str = ""                   # T5 text embeddings .pkl
    trained_with_image_aug: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True

    # Inference parameters
    chunk_size: int = 16
    num_denoising_steps_action: int = 5
    debug_predictions: bool = False                    # Return future proprio/images/value for real-robot debugging

    # Image settings
    use_third_person_image: bool = True
    num_third_person_images: int = 2
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = False
    use_jpeg_compression: bool = False
    use_variance_scale: bool = False

    # No planning model for simplified Franka deployment
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""

    # Reproducibility
    deterministic: bool = True
    seed: int = 195
    randomize_seed: bool = False

    # Best-of-N (disabled for simplified deployment)
    num_queries_best_of_n: int = 1
    use_parallel_inference: bool = False
    available_gpus: str = "0"
    parallel_timeout: int = 15

    # Ensemble settings (disabled)
    use_ensemble_future_state_predictions: bool = False
    num_future_state_predictions_in_ensemble: int = 3
    future_state_ensemble_aggregation_scheme: str = "average"
    use_ensemble_value_predictions: bool = False
    num_value_predictions_in_ensemble: int = 5
    value_ensemble_aggregation_scheme: str = "average"
    search_depth: int = 1
    search_depth_value_aggregation_scheme: str = "use_last_value"
    mask_current_state_action_for_value_prediction: bool = False
    mask_future_state_for_qvalue_prediction: bool = False


class FrankaPolicyServer:
    def __init__(self, cfg: FrankaDeployConfig):
        self.cfg = cfg

        # Set random seed
        set_seed_everywhere(cfg.seed)

        # Initialize T5 text embeddings cache
        print(f"[FrankaServer] Loading T5 text embeddings from: {cfg.t5_text_embeddings_path}")
        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

        # Load dataset stats
        print(f"[FrankaServer] Loading dataset stats from: {cfg.dataset_stats_path}")
        self.dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
        print(
            "[FrankaServer] Normalization flags: "
            f"normalize_proprio={cfg.normalize_proprio}, "
            f"unnormalize_actions={cfg.unnormalize_actions}"
        )
        print(
            "[FrankaServer] Input layout: "
            f"suite={cfg.suite}, "
            f"num_wrist_images={cfg.num_wrist_images}, "
            f"num_third_person_images={cfg.num_third_person_images}"
        )
        print(
            "[FrankaServer] Dataset stats: "
            f"actions_min={np.asarray(self.dataset_stats['actions_min'])}, "
            f"actions_max={np.asarray(self.dataset_stats['actions_max'])}, "
            f"state_keys={'state_min' in self.dataset_stats}, "
            f"proprio_keys={'proprio_min' in self.dataset_stats}"
        )

        # Initialize model
        print(f"[FrankaServer] Loading model from: {cfg.ckpt_path}")
        self.model, self.cosmos_config = get_model(cfg)
        print(
            "[FrankaServer] Model latent layout: "
            f"use_single_meta_frame={getattr(self.model.config, 'use_single_meta_frame', False)}"
        )

        # Optional separate planning model used for world-model (future state) and
        # value-function predictions. When omitted, the multi-task policy model
        # itself is used for planning (our checkpoints can switch inputs freely).
        if cfg.planning_model_ckpt_path:
            print(f"[FrankaServer] Loading planning model from: {cfg.planning_model_ckpt_path}")
            self.planning_model, self.planning_model_config = get_planning_model(cfg)
        else:
            self.planning_model, self.planning_model_config = None, None
            print("[FrankaServer] No separate planning model; using the policy model for planning.")

        # Validate the requested planning / best-of-N configuration up front.
        self._validate_planning_config()

        # Validate train/deploy alignment
        train_dataset = self.cosmos_config.dataloader_train.dataset
        train_chunk_size = train_dataset.chunk_size
        assert cfg.chunk_size == train_chunk_size, (
            f"Mismatch between train and test chunk sizes! "
            f"Train: {train_chunk_size}, Test: {cfg.chunk_size}"
        )
        self.action_shape = get_action_shape(cfg, self.dataset_stats)
        self._validate_training_alignment(train_dataset)

        # Get expected image dimensions
        self.resize_size = get_image_resize_size(self.cfg.model_family)
        self.cfg.resize_size = self.resize_size

        print(f"[FrankaServer] Model loaded successfully. Image resize: {self.resize_size}")
        print(
            f"[FrankaServer] Action shape inferred from config/stats: "
            f"(chunk_size={self.action_shape[0]}, action_dim={self.action_shape[1]})"
        )
        print(f"[FrankaServer] Ready to serve on {cfg.host}:{cfg.port}")

    def _require_alignment(self, condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"[FrankaServer] Training/deploy mismatch: {message}")

    def _validate_planning_config(self) -> None:
        """Validate the requested planning / best-of-N configuration.

        Mirrors the planning stack of the simulation evals: best-of-N search over
        action candidates, optional autoregressive future-state + V(s') value (or
        model-free Q(s, a)) prediction, and multi-level search. Parallel-worker
        inference is not wired up in this serial server.
        """
        cfg = self.cfg
        if cfg.use_parallel_inference:
            raise NotImplementedError(
                "[FrankaServer] use_parallel_inference is not supported by the Franka server "
                "(serial inference only). Set use_parallel_inference=False."
            )
        if cfg.ar_value_prediction and cfg.ar_qvalue_prediction:
            raise ValueError(
                "[FrankaServer] ar_value_prediction (V(s')) and ar_qvalue_prediction (Q(s, a)) "
                "are mutually exclusive."
            )
        if cfg.ar_value_prediction and not cfg.ar_future_prediction:
            raise ValueError(
                "[FrankaServer] ar_value_prediction (V(s')) requires ar_future_prediction=True "
                "so a future state is available to score."
            )
        if int(cfg.search_depth) > 1:
            if not cfg.ar_future_prediction:
                raise ValueError("[FrankaServer] search_depth > 1 requires ar_future_prediction=True.")
            if cfg.ar_qvalue_prediction:
                raise ValueError("[FrankaServer] search_depth > 1 is not supported with Q(s, a) prediction.")
        print(
            "[FrankaServer] Planning config: "
            f"num_queries_best_of_n={cfg.num_queries_best_of_n}, "
            f"ar_future_prediction={cfg.ar_future_prediction}, "
            f"ar_value_prediction={cfg.ar_value_prediction}, "
            f"ar_qvalue_prediction={cfg.ar_qvalue_prediction}, "
            f"search_depth={cfg.search_depth}"
        )

    def _run_query(
        self,
        observation: Dict[str, Any],
        task_description: Any,
        query_idx: int,
        generate_value_in_parallel: bool,
    ) -> Dict[str, Any]:
        """Run one best-of-N query: predict an action chunk, then (optionally) an
        autoregressive future-state + value (V(s')) / model-free Q(s, a) plan and
        a multi-level search. Returns a dict with the executed action chunk and
        its predicted value. Mirrors the robocasa eval flow.
        """
        cfg = self.cfg
        planning_model = self.planning_model if self.planning_model is not None else self.model
        query_seed = cfg.seed + query_idx

        return_dict: Dict[str, Any] = {}
        actions_by_depth = []
        future_image_predictions_by_depth = []
        value_predictions_by_depth = []

        # --- Predict the action chunk (and, in model-free parallel mode, the
        # future state + value jointly with the action). ---
        action_return_dict = get_action(
            cfg,
            self.model,
            self.dataset_stats,
            observation,
            task_description,
            seed=query_seed,
            randomize_seed=cfg.randomize_seed,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=generate_value_in_parallel,
        )
        return_dict["action_return_dict"] = action_return_dict
        return_dict["actions"] = action_return_dict["actions"]
        actions_by_depth.append(return_dict["actions"])

        latent_indices = action_return_dict["latent_indices"]

        # --- Autoregressive future-state prediction (depth 1). ---
        future_state_return_dict = None
        if cfg.ar_future_prediction:
            future_state_return_dict = get_future_state_prediction(
                cfg,
                model=planning_model,
                data_batch=action_return_dict["data_batch"],
                generated_latent_with_action=action_return_dict["generated_latent"],
                orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
                future_proprio_latent_idx=latent_indices["future_proprio_latent_idx"],
                future_wrist_image_latent_idx=latent_indices["future_wrist_image_latent_idx"],
                future_wrist_image2_latent_idx=latent_indices["future_wrist_image2_latent_idx"],
                future_image_latent_idx=latent_indices["future_image_latent_idx"],
                future_image2_latent_idx=latent_indices["future_image2_latent_idx"],
                seed=query_seed,
                randomize_seed=cfg.randomize_seed,
                num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
            )
            return_dict["future_image_predictions"] = future_state_return_dict["future_image_predictions"]
        elif not cfg.ar_qvalue_prediction:
            return_dict["future_image_predictions"] = action_return_dict.get("future_image_predictions")
        future_image_predictions_by_depth.append(return_dict.get("future_image_predictions"))

        # --- Value prediction (depth 1): V(s') from the future state, model-free
        # Q(s, a) from the action, or the value produced in parallel. ---
        if cfg.ar_value_prediction:
            value_return_dict = get_value_prediction(
                cfg,
                model=planning_model,
                data_batch=action_return_dict["data_batch"],
                future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                seed=query_seed,
                randomize_seed=cfg.randomize_seed,
                num_denoising_steps_value=cfg.num_denoising_steps_value,
                use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
            )
            return_dict["value_prediction"] = value_return_dict["value_prediction"]
        elif cfg.ar_qvalue_prediction:
            value_return_dict = get_qvalue_prediction(
                cfg,
                model=planning_model,
                data_batch=action_return_dict["data_batch"],
                action_sample=action_return_dict["generated_latent"],
                seed=query_seed,
                randomize_seed=cfg.randomize_seed,
                num_denoising_steps_value=cfg.num_denoising_steps_value,
                use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
            )
            return_dict["value_prediction"] = value_return_dict["value_prediction"]
        else:
            return_dict["value_prediction"] = action_return_dict.get("value_prediction")
        value_predictions_by_depth.append(return_dict["value_prediction"])

        # --- Multi-level search (depth >= 2): treat the imagined future state as
        # the next current state, then re-predict action -> future state -> value. ---
        if cfg.search_depth > 1:
            self._run_search_depth(
                planning_model=planning_model,
                action_return_dict=action_return_dict,
                future_state_return_dict=future_state_return_dict,
                query_idx=query_idx,
                actions_by_depth=actions_by_depth,
                future_image_predictions_by_depth=future_image_predictions_by_depth,
                value_predictions_by_depth=value_predictions_by_depth,
                return_dict=return_dict,
            )

        return_dict["actions_by_depth"] = actions_by_depth
        return_dict["future_image_predictions_by_depth"] = future_image_predictions_by_depth
        return_dict["value_predictions_by_depth"] = value_predictions_by_depth
        return return_dict

    def _run_search_depth(
        self,
        planning_model: torch.nn.Module,
        action_return_dict: Dict[str, Any],
        future_state_return_dict: Dict[str, Any],
        query_idx: int,
        actions_by_depth: list,
        future_image_predictions_by_depth: list,
        value_predictions_by_depth: list,
        return_dict: Dict[str, Any],
    ) -> None:
        """Expand the best-of-N search tree beyond depth 1.

        Mirrors the robocasa eval search loop, whose current->future latent frame
        remapping is derived from the latent indices of the request (so it works
        for both the MetaWAM and the upstream latent layouts).
        """
        cfg = self.cfg
        model = self.model
        action_dim = self.action_shape[1]
        data_batch = action_return_dict["data_batch"]
        latent_indices = action_return_dict["latent_indices"]
        action_latent_idx = latent_indices["action_latent_idx"]

        # current-state -> future-state latent frame mapping used to advance the
        # imagined rollout by one step. Images always have dedicated future frames;
        # proprio only in the upstream layout (MetaWAM keeps it in the input/meta frames).
        index_pairs = [
            ("current_wrist_image_latent_idx", "future_wrist_image_latent_idx"),
            ("current_wrist_image2_latent_idx", "future_wrist_image2_latent_idx"),
            ("current_image_latent_idx", "future_image_latent_idx"),
            ("current_image2_latent_idx", "future_image2_latent_idx"),
        ]
        if not getattr(model.config, "use_single_meta_frame", False):
            index_pairs.append(("current_proprio_latent_idx", "future_proprio_latent_idx"))

        for _depth in range(2, cfg.search_depth + 1):
            for future_state_latent in future_state_return_dict["future_state_samples_list"]:
                next_latent_with_future_state = future_state_latent.clone()
                # Rearrange so the predicted future state replaces the current state.
                rearranged = next_latent_with_future_state.clone()
                for curr_key, fut_key in index_pairs:
                    curr_idx = int(data_batch[curr_key][0].item())
                    fut_idx = int(latent_indices[fut_key])
                    if curr_idx != -1 and fut_idx != -1:
                        rearranged[:, :, curr_idx, :, :] = next_latent_with_future_state[:, :, fut_idx, :, :]

                # Predict the next action from the advanced state.
                data_batch["num_conditional_frames"] = model.config.min_num_conditional_frames
                data_batch["mask_current_state_action_for_value_prediction"] = False
                seed = secrets.randbits(32) % 256 if cfg.randomize_seed else cfg.seed + query_idx
                next_generated_latent_with_action, next_orig_clean_latent_frames = (
                    model.generate_samples_from_batch(
                        data_batch,
                        n_sample=1,
                        num_steps=cfg.num_denoising_steps_action,
                        seed=seed,
                        is_negative_prompt=False,
                        use_variance_scale=cfg.use_variance_scale,
                        skip_vae_encoding=True,
                        previous_generated_latent=rearranged,
                        return_orig_clean_latent_frames=True,
                    )
                )
                action_indices = torch.full(
                    (1,),
                    action_latent_idx,
                    dtype=torch.int64,
                    device=next_generated_latent_with_action.device,
                )
                next_actions = (
                    extract_action_chunk(
                        model,
                        next_generated_latent_with_action,
                        (cfg.chunk_size, action_dim),
                        action_indices=action_indices,
                    )
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                if cfg.unnormalize_actions:
                    next_actions = unnormalize_actions(next_actions, self.dataset_stats)
                next_actions = next_actions[0]
                next_actions = [next_actions[i] for i in range(len(next_actions))]
                actions_by_depth.append(next_actions)

                # Predict the next future state from the new action.
                future_state_return_dict = get_future_state_prediction(
                    cfg,
                    model=planning_model,
                    data_batch=data_batch,
                    generated_latent_with_action=next_generated_latent_with_action,
                    orig_clean_latent_frames=next_orig_clean_latent_frames,
                    future_proprio_latent_idx=latent_indices["future_proprio_latent_idx"],
                    future_wrist_image_latent_idx=latent_indices["future_wrist_image_latent_idx"],
                    future_wrist_image2_latent_idx=latent_indices["future_wrist_image2_latent_idx"],
                    future_image_latent_idx=latent_indices["future_image_latent_idx"],
                    future_image2_latent_idx=latent_indices["future_image2_latent_idx"],
                    seed=cfg.seed + query_idx,
                    randomize_seed=cfg.randomize_seed,
                    num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                    use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                    num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                    future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                )
                future_image_predictions_by_depth.append(future_state_return_dict["future_image_predictions"])

                # Predict the next value V(s').
                value_return_dict = get_value_prediction(
                    cfg,
                    model=planning_model,
                    data_batch=data_batch,
                    future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                    seed=cfg.seed + query_idx,
                    randomize_seed=cfg.randomize_seed,
                    num_denoising_steps_value=cfg.num_denoising_steps_value,
                    use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                    num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                )
                return_dict["value_prediction"] = value_return_dict["value_prediction"]
                value_predictions_by_depth.append(return_dict["value_prediction"])

    def _validate_training_alignment(self, train_dataset: Any) -> None:
        """Fail early when the loaded checkpoint config and deploy input layout diverge."""
        cfg = self.cfg
        model_cfg = self.model.config
        dataset_name = type(train_dataset).__name__

        action_dim_stats = int(np.asarray(self.dataset_stats["actions_min"]).shape[0])
        proprio_key = "state" if "state_min" in self.dataset_stats else "proprio"
        proprio_min = np.asarray(self.dataset_stats[f"{proprio_key}_min"])
        proprio_dim_stats = int(proprio_min.shape[0])

        train_action_dim = getattr(train_dataset, "action_dim", None)
        train_raw_proprio_dim = getattr(train_dataset, "raw_proprio_dim", None)
        train_proprio_dim = getattr(train_dataset, "proprio_dim", None)
        train_use_wrist_images = getattr(train_dataset, "use_wrist_images", None)
        train_use_third_person_images = getattr(train_dataset, "use_third_person_images", None)
        train_normalize_actions = getattr(train_dataset, "normalize_actions", None)
        train_normalize_proprio = getattr(train_dataset, "normalize_proprio", None)
        train_use_single_meta_frame = getattr(train_dataset, "use_single_meta_frame", None)

        print("[FrankaServer] Loaded training dataset/config summary:")
        print(f"  dataset={dataset_name}")
        print(
            "  train dataset: "
            f"chunk_size={getattr(train_dataset, 'chunk_size', None)}, "
            f"action_dim={train_action_dim}, "
            f"raw_proprio_dim={train_raw_proprio_dim}, "
            f"model_proprio_dim={train_proprio_dim}, "
            f"use_wrist_images={train_use_wrist_images}, "
            f"use_third_person_images={train_use_third_person_images}, "
            f"normalize_actions={train_normalize_actions}, "
            f"normalize_proprio={train_normalize_proprio}"
        )
        print(
            "  model latent config: "
            f"state_t={getattr(model_cfg, 'state_t', None)}, "
            f"min_cond={getattr(model_cfg, 'min_num_conditional_frames', None)}, "
            f"max_cond={getattr(model_cfg, 'max_num_conditional_frames', None)}, "
            f"use_single_meta_frame={getattr(model_cfg, 'use_single_meta_frame', None)}"
        )
        print(
            "  deploy input: "
            f"suite={cfg.suite}, "
            f"num_wrist_images={cfg.num_wrist_images}, "
            f"num_third_person_images={cfg.num_third_person_images}, "
            f"action_shape={self.action_shape}, "
            f"stats_action_dim={action_dim_stats}, "
            f"stats_{proprio_key}_dim={proprio_dim_stats}"
        )

        self._require_alignment(
            action_dim_stats == self.action_shape[1],
            f"stats action dim ({action_dim_stats}) != inferred action dim ({self.action_shape[1]})",
        )
        if train_action_dim is not None:
            self._require_alignment(
                int(train_action_dim) == action_dim_stats,
                f"train action_dim ({train_action_dim}) != stats action dim ({action_dim_stats})",
            )
        if train_raw_proprio_dim is not None:
            self._require_alignment(
                int(train_raw_proprio_dim) == proprio_dim_stats,
                f"train raw_proprio_dim ({train_raw_proprio_dim}) != stats {proprio_key}_dim ({proprio_dim_stats})",
            )
        if train_normalize_actions is not None:
            self._require_alignment(
                bool(train_normalize_actions) == bool(cfg.unnormalize_actions),
                f"train normalize_actions={train_normalize_actions} but deploy unnormalize_actions={cfg.unnormalize_actions}",
            )
        if train_normalize_proprio is not None:
            self._require_alignment(
                bool(train_normalize_proprio) == bool(cfg.normalize_proprio),
                f"train normalize_proprio={train_normalize_proprio} but deploy normalize_proprio={cfg.normalize_proprio}",
            )
        if train_use_wrist_images is not None:
            self._require_alignment(
                bool(train_use_wrist_images) == bool(cfg.use_wrist_image),
                f"train use_wrist_images={train_use_wrist_images} but deploy use_wrist_image={cfg.use_wrist_image}",
            )
        if train_use_third_person_images is not None:
            self._require_alignment(
                bool(train_use_third_person_images) == bool(cfg.use_third_person_image),
                "train/deploy third-person image usage differs",
            )

        if dataset_name == "RealWorldLeRobotDataset":
            stats_format_version = (
                float(np.asarray(self.dataset_stats.get("stats_format_version", 0.0)).reshape(-1)[0])
                if "stats_format_version" in self.dataset_stats
                else 0.0
            )
            roll_canonicalization_code = (
                float(np.asarray(self.dataset_stats.get("state_roll_canonicalization_code", 0.0)).reshape(-1)[0])
                if "state_roll_canonicalization_code" in self.dataset_stats
                else 0.0
            )
            action_symmetric_code = (
                float(np.asarray(self.dataset_stats.get("action_symmetric_first6_code", 0.0)).reshape(-1)[0])
                if "action_symmetric_first6_code" in self.dataset_stats
                else 0.0
            )
            self._require_alignment(cfg.suite == "robocasa", "RealWorldLeRobotDataset must use suite='robocasa'")
            self._require_alignment(
                cfg.num_wrist_images == 1 and cfg.num_third_person_images == 2,
                "RealWorldLeRobotDataset expects 1 wrist image and 2 third-person images",
            )
            self._require_alignment(action_dim_stats == 7, "RealWorldLeRobotDataset deploy expects 7D delta EEF actions")
            self._require_alignment(proprio_key == "state", "RealWorldLeRobotDataset stats should use state_min/state_max")
            self._require_alignment(proprio_dim_stats == 8, "RealWorldLeRobotDataset stats should be raw 8D EEF state")
            self._require_alignment(
                stats_format_version >= 2.0,
                "RealWorldLeRobotDataset stats must use stats format version >= 2 "
                "(roll canonicalization + symmetric action normalization)",
            )
            self._require_alignment(
                roll_canonicalization_code == 1.0,
                "RealWorldLeRobotDataset deploy expects roll_fixed = wrap_to_pi(roll - pi)",
            )
            self._require_alignment(
                action_symmetric_code == 1.0,
                "RealWorldLeRobotDataset deploy expects symmetric action normalization for action[:6]",
            )
            self._require_alignment(
                getattr(model_cfg, "state_t", None) == 8
                and getattr(model_cfg, "min_num_conditional_frames", None) == 4,
                "real-world MetaWAM model should use state_t=8 and min_num_conditional_frames=4",
            )
            self._require_alignment(
                bool(getattr(model_cfg, "use_single_meta_frame", False)),
                "real-world MetaWAM model should use the single meta-frame action/future/value layout",
            )
            if train_use_single_meta_frame is not None:
                self._require_alignment(bool(train_use_single_meta_frame), "training dataset should use a single meta frame")

    def get_server_action(self, payload: Dict[str, Any]) -> JSONResponse:
        try:
            if double_encode := "encoded" in payload:
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            observation = payload
            task_description = observation["task_description"]
            request_debug_predictions = bool(observation.pop("debug_predictions", self.cfg.debug_predictions))

            # Convert lists to numpy arrays
            observation["primary_image"] = np.array(observation["primary_image"], dtype=np.uint8)
            observation["left_wrist_image"] = np.array(observation["left_wrist_image"], dtype=np.uint8)
            observation["right_wrist_image"] = np.array(observation["right_wrist_image"], dtype=np.uint8)
            observation["proprio"] = np.array(observation["proprio"], dtype=np.float32)

            if self.cfg.suite == "robocasa":
                # RealWorldLeRobotDataset trains with [wrist, primary, secondary].
                # The Franka client payload uses `left_wrist_image` / `right_wrist_image`
                # keys for the two non-primary cameras, so remap them before calling the
                # shared Cosmos policy inference path.
                observation["wrist_image"] = observation["right_wrist_image"]
                observation["secondary_image"] = observation["left_wrist_image"]

            print("[FrankaServer] Received observation:")
            print(f"  primary_image: {observation['primary_image'].shape}")
            print(f"  left_wrist_image: {observation['left_wrist_image'].shape}")
            print(f"  right_wrist_image: {observation['right_wrist_image'].shape}")
            print(f"  proprio: {observation['proprio'].shape}")
            print(f"  task_description: {task_description}")

            # Run inference. Best-of-N: sample several action candidates (one per
            # seed); for each, optionally roll out an autoregressive future-state +
            # value (V(s')) / model-free Q(s, a) plan and a multi-level search.
            # Keep the candidate whose (aggregated) predicted value is highest.
            num_queries = max(1, int(self.cfg.num_queries_best_of_n))
            # A value is needed to rank candidates (best-of-N) or to fill the debug
            # payload. When not using an autoregressive value head, the value is
            # produced in parallel with the action chunk.
            need_value = bool(request_debug_predictions) or num_queries > 1
            parallel_value = need_value and not (
                self.cfg.ar_value_prediction or self.cfg.ar_qvalue_prediction
            )

            start_time = time.time()
            query_results = []
            for query_idx in range(num_queries):
                query_results.append(
                    self._run_query(
                        observation,
                        task_description,
                        query_idx,
                        generate_value_in_parallel=parallel_value,
                    )
                )

            # When search_depth > 1, collapse the per-depth values into one score.
            if self.cfg.search_depth > 1:
                for return_dict in query_results:
                    if self.cfg.search_depth_value_aggregation_scheme == "average":
                        return_dict["value_prediction"] = float(np.mean(return_dict["value_predictions_by_depth"]))
                    elif self.cfg.search_depth_value_aggregation_scheme == "use_last_value":
                        return_dict["value_prediction"] = return_dict["value_predictions_by_depth"][-1]
                    else:
                        raise ValueError(
                            "Invalid search depth value aggregation scheme: "
                            f"{self.cfg.search_depth_value_aggregation_scheme}"
                        )

            for query_idx, return_dict in enumerate(query_results):
                value = return_dict.get("value_prediction")
                value_str = f"{value:.4f}" if value is not None else "n/a"
                print(
                    f"[FrankaServer] Query {query_idx + 1}/{num_queries} "
                    f"(seed {self.cfg.seed + query_idx}): value = {value_str}"
                )

            # Select the highest-value candidate. The base seed (query 0) is tried
            # first, so ties keep the base-seed action.
            if num_queries > 1:
                if any(rd.get("value_prediction") is None for rd in query_results):
                    raise ValueError(
                        "Best-of-N selection requires value predictions, but none were produced. "
                        "Enable ar_value_prediction/ar_qvalue_prediction or use a checkpoint with a value head."
                    )
                best_query_idx = max(range(num_queries), key=lambda i: query_results[i]["value_prediction"])
                print(
                    f"[FrankaServer] Selected seed {self.cfg.seed + best_query_idx} with value = "
                    f"{query_results[best_query_idx]['value_prediction']:.4f} (best-of-{num_queries})"
                )
            else:
                best_query_idx = 0
            best_return_dict = query_results[best_query_idx]
            action_return_dict = best_return_dict["action_return_dict"]
            best_value = best_return_dict.get("value_prediction")
            inference_time_ms = (time.time() - start_time) * 1000

            actions = best_return_dict["actions"]
            normalized_proprio = action_return_dict.get("proprio")
            print(f"[FrankaServer] Inference complete in {inference_time_ms:.1f}ms")
            print(f"  Actions shape: {np.asarray(actions).shape}")
            if normalized_proprio is not None:
                normalized_proprio = np.asarray(normalized_proprio, dtype=np.float32).reshape(-1)
                out_of_range = np.where((normalized_proprio < -1.2) | (normalized_proprio > 1.2))[0]
                if out_of_range.size > 0:
                    print(
                        "  [WARNING] Normalized proprio outside expected range: "
                        f"dims={out_of_range.tolist()}, "
                        f"values={normalized_proprio[out_of_range]}"
                    )
            if request_debug_predictions:
                print(f"  Value prediction: {best_value}")
                future_proprio = action_return_dict.get("future_proprio_prediction")
                if future_proprio is not None:
                    print(f"  Future proprio prediction: {np.asarray(future_proprio)}")

            response = {
                "actions": actions,
                "inference_time_ms": inference_time_ms,
            }
            if num_queries > 1 or best_value is not None:
                response["value_prediction"] = best_value
            if num_queries > 1:
                response["num_queries_best_of_n"] = num_queries
            if request_debug_predictions:
                response.update(
                    {
                        "current_proprio_normalized": normalized_proprio,
                        "future_image_predictions": best_return_dict.get("future_image_predictions"),
                        "future_proprio": action_return_dict.get("future_proprio_prediction"),
                        "future_proprio_normalized": action_return_dict.get("future_proprio_prediction_normalized"),
                        "value_prediction": best_value,
                    }
                )

            if double_encode:
                return JSONResponse(json_numpy.dumps(response))
            else:
                return JSONResponse(response)

        except Exception:
            logging.error(traceback.format_exc())
            logging.warning(
                "Request error. Expected format:\n"
                "{'primary_image': [...], 'left_wrist_image': [...], "
                "'right_wrist_image': [...], 'proprio': [...], 'task_description': str}\n"
            )
            return JSONResponse(
                status_code=500,
                content={"error": "Inference failed", "details": traceback.format_exc()},
            )

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()

        @self.app.get("/health")
        def health():
            return {"status": "ok", "model_loaded": True}

        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


@draccus.wrap()
def deploy(cfg: FrankaDeployConfig) -> None:
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
        set_seed_everywhere(cfg.seed)

    server = FrankaPolicyServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
