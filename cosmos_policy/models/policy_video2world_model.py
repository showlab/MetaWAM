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
Cosmos Policy Video2World Model - extends CosmosPolicyDiffusionModel with video conditioning.

IMPORTANT: This class inherits from CosmosPolicyDiffusionModel (not Video2WorldModel)
to ensure it gets all the policy-specific functionality (training_step, compute_loss, etc.)
"""

import math
import os
from typing import Any, Callable, Dict, Optional, Tuple

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_policy._src.imaginaire.utils.denoise_prediction import DenoisePrediction
from cosmos_policy._src.imaginaire.utils.high_sigma_strategy import HighSigmaStrategy
from cosmos_policy._src.predict2.conditioner import DataType
from cosmos_policy._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY, ConditioningStrategy
from cosmos_policy.conditioner import Text2WorldCondition
from cosmos_policy.config.conditioner.video2world_conditioner import Video2WorldCondition
from cosmos_policy.models.policy_text2world_model import (
    CosmosPolicyDiffusionModel,
    CosmosPolicyModelConfig,
    _FREQUENCY_V2_ACTION_SLICE,
    _FREQUENCY_V2_CURRENT_PROPRIO_SLICE,
    _FREQUENCY_V2_FUTURE_PROPRIO_SLICE,
    _FREQUENCY_V2_TASK_SLICE,
    _FREQUENCY_V2_VALUE_SLICE,
    replace_latent_with_action_chunk,
    replace_latent_with_meta_regions,
    replace_latent_with_proprio,
)

LOG_200 = math.log(200)
LOG_100000 = math.log(100000)


@attrs.define(slots=False)
class CosmosPolicyVideo2WorldConfig(CosmosPolicyModelConfig):
    """
    Extended config for Cosmos Policy Video2World model.
    Inherits from CosmosPolicyModelConfig and adds some video-specific parameters in the same way that
    Video2WorldConfig adds video-specific parameters to Text2WorldModelConfig.
    """

    min_num_conditional_frames: int = 1  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    sigma_conditional: float = 0.0001  # Noise level used for conditional frames
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)  # What strategy to use for conditioning
    denoise_replace_gt_frames: bool = True  # Whether to denoise the ground truth frames
    high_sigma_strategy: str = str(HighSigmaStrategy.UNIFORM80_2000)  # What strategy to use for high sigma
    high_sigma_ratio: float = 0.05  # Ratio of high sigma frames
    low_sigma_ratio: float = 0.05  # Ratio of low sigma frames
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.conditioning_strategy in [
            str(ConditioningStrategy.FRAME_REPLACE),
        ]
        assert self.high_sigma_strategy in [
            str(HighSigmaStrategy.NONE),
            str(HighSigmaStrategy.UNIFORM80_2000),
            str(HighSigmaStrategy.LOGUNIFORM200_100000),
            str(HighSigmaStrategy.BALANCED_TWO_HEADS_V1),
            str(HighSigmaStrategy.SHIFT24),
            str(HighSigmaStrategy.HARDCODED_20steps),
        ]


class CosmosPolicyVideo2WorldModel(CosmosPolicyDiffusionModel):
    """
    Cosmos Policy Video2World Model - extends CosmosPolicyDiffusionModel with video conditioning.

    Inheritance: CosmosPolicyDiffusionModel → Text2WorldModel

    Adds Video2World functionality to the policy base:
    - Video frame conditioning with gt_frames
    - High sigma sampling strategies
    - FlowUniPC scheduler support
    - Policy-specific mask manipulation for world model/value function
    """

    def __init__(self, config: CosmosPolicyVideo2WorldConfig):
        super().__init__(config)
        self.config: CosmosPolicyVideo2WorldConfig = config
        self._validate_meta_frame_runtime()

    def _validate_meta_frame_runtime(self) -> None:
        if not self.config.use_single_meta_frame:
            return
        assert self._get_slot_condition_mask_channels() == _FREQUENCY_V2_VALUE_SLICE.stop, (
            "use_single_meta_frame requires "
            f"net.slot_condition_mask_channels={_FREQUENCY_V2_VALUE_SLICE.stop}"
        )

    def _build_task_masks(
        self, data_batch: dict[str, torch.Tensor], batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "rollout_data_mask" in data_batch and "world_model_sample_mask" in data_batch and "value_function_sample_mask" in data_batch:
            rollout_data_mask = data_batch["rollout_data_mask"].to(device=device).bool()
            world_model_sample_mask = data_batch["world_model_sample_mask"].to(device=device).bool()
            value_function_sample_mask = data_batch["value_function_sample_mask"].to(device=device).bool()
            demo_mask_B = ~rollout_data_mask
            world_mask_B = rollout_data_mask & world_model_sample_mask
            value_mask_B = rollout_data_mask & value_function_sample_mask
            return demo_mask_B, world_mask_B, value_mask_B

        # Inference fallback: default to policy task unless explicit flags are provided.
        demo_mask_B = torch.ones(batch_size, device=device, dtype=torch.bool)
        world_mask_B = torch.zeros(batch_size, device=device, dtype=torch.bool)
        value_mask_B = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if data_batch.get("mask_current_state_action_for_value_prediction", False) or data_batch.get(
            "mask_future_state_for_qvalue_prediction", False
        ):
            demo_mask_B[:] = False
            value_mask_B[:] = True
        return demo_mask_B, world_mask_B, value_mask_B

    def _build_task_type_region_vector(
        self,
        demo_mask_B: torch.Tensor,
        world_mask_B: torch.Tensor,
        value_mask_B: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = demo_mask_B.shape[0]
        task_type_region_vector = torch.zeros((batch_size, 3), device=device, dtype=dtype)
        task_type_region_vector[demo_mask_B, 0] = 1.0
        task_type_region_vector[world_mask_B, 1] = 1.0
        task_type_region_vector[value_mask_B, 2] = 1.0
        return task_type_region_vector

    def _apply_input_meta_frame_conditioning(
        self,
        condition: Video2WorldCondition,
        data_batch: dict[str, torch.Tensor],
    ) -> None:
        """Write the MetaWAM input meta frame (task type + current proprio + visible inputs) into latent slot 0."""
        B = condition.gt_frames.shape[0]
        _, _, _, H_latent, W_latent = condition.gt_frames.shape
        device = condition.gt_frames.device
        dtype = condition.gt_frames.dtype
        batch_indices = torch.arange(B, device=device)
        demo_mask_B, world_mask_B, value_mask_B = self._build_task_masks(data_batch, B, device)
        task_type_region_vector = self._build_task_type_region_vector(
            demo_mask_B,
            world_mask_B,
            value_mask_B,
            dtype=dtype,
            device=device,
        )
        assert "proprio" in data_batch and data_batch["proprio"] is not None, (
            "use_single_meta_frame requires proprio in data_batch"
        )
        current_proprio_region_vector = data_batch["proprio"].to(device=device, dtype=dtype).reshape(B, -1)
        mask_current_state_action_for_value_prediction = bool(
            data_batch.get("mask_current_state_action_for_value_prediction", False)
        )
        mask_future_state_for_qvalue_prediction = bool(data_batch.get("mask_future_state_for_qvalue_prediction", False))
        world_or_value_mask_B = world_mask_B | value_mask_B
        if mask_current_state_action_for_value_prediction:
            current_visible_B = (~value_mask_B) | world_mask_B
            action_visible_B = world_mask_B
        else:
            current_visible_B = torch.ones(B, device=device, dtype=torch.bool)
            action_visible_B = world_or_value_mask_B
        future_visible_B = torch.zeros(B, device=device, dtype=torch.bool)
        include_future_proprio_region = "future_proprio" in data_batch and data_batch["future_proprio"] is not None
        if include_future_proprio_region and not mask_future_state_for_qvalue_prediction:
            future_visible_B = value_mask_B

        current_proprio_region_vector = current_proprio_region_vector * current_visible_B.to(dtype).unsqueeze(1)
        action_region_vector = None
        if "actions" in data_batch and data_batch["actions"] is not None:
            action_region_vector = data_batch["actions"].to(device=device, dtype=dtype).reshape(B, -1)
            action_region_vector = action_region_vector * action_visible_B.to(dtype).unsqueeze(1)
        future_proprio_region_vector = None
        if include_future_proprio_region:
            future_proprio_region_vector = data_batch["future_proprio"].to(device=device, dtype=dtype).reshape(B, -1)
            future_proprio_region_vector = future_proprio_region_vector * future_visible_B.to(dtype).unsqueeze(1)

        mask_channels = self._get_working_condition_mask_channels()
        assert mask_channels >= _FREQUENCY_V2_VALUE_SLICE.stop, (
            "frequency input meta conditioning requires slot-aware working mask width >= "
            f"{_FREQUENCY_V2_VALUE_SLICE.stop}, got {mask_channels}"
        )
        input_meta_mask_B_C_H_W = self._build_meta_region_condition_mask(
            demo_mask_B=demo_mask_B,
            world_mask_B=world_mask_B,
            value_mask_B=value_mask_B,
            mask_channels=mask_channels,
            h_latent=H_latent,
            w_latent=W_latent,
            dtype=condition.condition_video_input_mask_B_C_T_H_W.dtype,
            device=condition.condition_video_input_mask_B_C_T_H_W.device,
            include_future_proprio_region=include_future_proprio_region,
            mask_current_state_action_for_value_prediction=mask_current_state_action_for_value_prediction,
            mask_future_state_for_qvalue_prediction=mask_future_state_for_qvalue_prediction,
        )
        # Slot 0 (input meta frame) is always conditioning input regardless of random conditional-frame sampling.
        condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, 0, :, :] = input_meta_mask_B_C_H_W

        input_meta_indices = torch.zeros((B,), dtype=torch.long, device=device)
        condition.gt_frames = replace_latent_with_meta_regions(
            condition.gt_frames,
            meta_indices=input_meta_indices,
            current_proprio_region_vector=current_proprio_region_vector,
            action_region_vector=action_region_vector,
            future_proprio_region_vector=future_proprio_region_vector,
            value_region_vector=None,
            task_type_region_vector=task_type_region_vector,
        )

    def _get_slot_condition_mask_channels(self) -> int:
        return int(getattr(self.net, "slot_condition_mask_channels", 0))

    def _get_working_condition_mask_channels(self) -> int:
        # With a slot-mask branch (MetaWAM), the working condition mask is slot-aware (one channel per latent
        # channel); otherwise it is the single coarse mask channel of the base model.
        slot_condition_mask_channels = self._get_slot_condition_mask_channels()
        return slot_condition_mask_channels if slot_condition_mask_channels > 0 else 1

    def _finalize_condition_masks_for_net(
        self, condition: Optional[Video2WorldCondition], *, context: str
    ) -> Optional[Video2WorldCondition]:
        if condition is None:
            return None
        working_mask = condition.condition_video_input_mask_B_C_T_H_W
        assert working_mask is not None, f"{context}: condition_video_input_mask_B_C_T_H_W should not be None"
        explicit_replace_mask = getattr(condition, "condition_video_input_replace_mask_B_C_T_H_W", None)

        expected_slot_condition_mask_channels = self._get_slot_condition_mask_channels()

        if expected_slot_condition_mask_channels > 0:
            assert working_mask.shape[1] == expected_slot_condition_mask_channels, (
                f"{context}: slot-aware working mask mismatch: "
                f"got {working_mask.shape[1]}, expected {expected_slot_condition_mask_channels}"
            )
            # - the slot-aware mask drives hard GT replacement (unless an explicit replace mask is given) and the
            #   net's separate slot-mask embedder
            # - a coarse single-channel mask (any slot conditioned) is concatenated into the base x_embedder
            if explicit_replace_mask is not None:
                assert explicit_replace_mask.shape[1] in (1, expected_slot_condition_mask_channels), (
                    f"{context}: explicit replacement mask channel mismatch: "
                    f"got {explicit_replace_mask.shape[1]}, expected 1 or {expected_slot_condition_mask_channels}"
                )
                condition.condition_video_input_replace_mask_B_C_T_H_W = explicit_replace_mask
            else:
                condition.condition_video_input_replace_mask_B_C_T_H_W = working_mask
            condition.slot_condition_video_input_mask_B_C_T_H_W = working_mask
            condition.condition_video_input_mask_B_C_T_H_W = (working_mask.amax(dim=1, keepdim=True) > 0).to(
                working_mask.dtype
            )
            return condition

        assert working_mask.shape[1] == 1, f"{context}: condition mask must have 1 channel, got {working_mask.shape[1]}"
        if explicit_replace_mask is not None:
            condition.condition_video_input_replace_mask_B_C_T_H_W = explicit_replace_mask
        else:
            condition.condition_video_input_replace_mask_B_C_T_H_W = condition.condition_video_input_mask_B_C_T_H_W
        condition.slot_condition_video_input_mask_B_C_T_H_W = None
        return condition

    def _build_meta_region_condition_mask(
        self,
        demo_mask_B: torch.Tensor,
        world_mask_B: torch.Tensor,
        value_mask_B: torch.Tensor,
        mask_channels: int,
        h_latent: int,
        w_latent: int,
        dtype: torch.dtype,
        device: torch.device,
        include_future_proprio_region: bool,
        mask_current_state_action_for_value_prediction: bool,
        mask_future_state_for_qvalue_prediction: bool,
    ) -> torch.Tensor:
        """Per-channel condition mask (1 = conditioning input, 0 = prediction target) for a meta frame."""
        batch_size = demo_mask_B.shape[0]
        mask_B_C_H_W = torch.zeros((batch_size, mask_channels, h_latent, w_latent), dtype=dtype, device=device)
        assert mask_channels >= _FREQUENCY_V2_VALUE_SLICE.stop, (
            "frequency meta frame requires condition mask channels >= "
            f"{_FREQUENCY_V2_VALUE_SLICE.stop}, got {mask_channels}"
        )
        world_or_value_mask_B = world_mask_B | value_mask_B
        # Q-value mode: future hidden, action visible.
        if mask_future_state_for_qvalue_prediction:
            future_cond_mask_B = torch.zeros_like(value_mask_B)
        else:
            future_cond_mask_B = value_mask_B
        # V(s') mode: action/current hidden for value samples.
        if mask_current_state_action_for_value_prediction:
            action_cond_mask_B = world_mask_B
            current_cond_mask_B = (~value_mask_B) | world_mask_B
        else:
            action_cond_mask_B = world_or_value_mask_B
            current_cond_mask_B = torch.ones_like(demo_mask_B)

        # task slot is always visible.
        mask_B_C_H_W[:, _FREQUENCY_V2_TASK_SLICE, :, :] = 1.0
        # current proprio slot is normally visible, except V(s') value samples.
        mask_B_C_H_W[:, _FREQUENCY_V2_CURRENT_PROPRIO_SLICE, :, :] = current_cond_mask_B.to(dtype).reshape(
            -1, 1, 1, 1
        )
        # action slot: world + (optionally) value.
        mask_B_C_H_W[:, _FREQUENCY_V2_ACTION_SLICE, :, :] = action_cond_mask_B.to(dtype).reshape(-1, 1, 1, 1)
        # future slot: value-only (unless Q-value mode).
        if include_future_proprio_region:
            mask_B_C_H_W[:, _FREQUENCY_V2_FUTURE_PROPRIO_SLICE, :, :] = future_cond_mask_B.to(dtype).reshape(
                -1, 1, 1, 1
            )
        # value slot: always target-only, keep 0.
        return mask_B_C_H_W

    def _apply_meta_frame_conditioning(
        self,
        condition: Video2WorldCondition,
        data_batch: dict[str, torch.Tensor],
        action_latent_idx: torch.Tensor,
    ) -> None:
        """Write the MetaWAM output meta frame (at the action index) and its per-channel condition mask."""
        B = condition.gt_frames.shape[0]
        _, _, _, H_latent, W_latent = condition.gt_frames.shape
        device = condition.gt_frames.device
        dtype = condition.gt_frames.dtype
        demo_mask_B, world_mask_B, value_mask_B = self._build_task_masks(data_batch, B, device)
        mask_channels = self._get_working_condition_mask_channels()
        include_future_proprio_region = "future_proprio" in data_batch and torch.all(
            data_batch["future_proprio_latent_idx"].to(device=device) != -1
        )
        mask_current_state_action_for_value_prediction = bool(
            data_batch.get("mask_current_state_action_for_value_prediction", False)
        )
        mask_future_state_for_qvalue_prediction = bool(data_batch.get("mask_future_state_for_qvalue_prediction", False))

        meta_mask_B_C_H_W = self._build_meta_region_condition_mask(
            demo_mask_B=demo_mask_B,
            world_mask_B=world_mask_B,
            value_mask_B=value_mask_B,
            mask_channels=mask_channels,
            h_latent=H_latent,
            w_latent=W_latent,
            dtype=condition.condition_video_input_mask_B_C_T_H_W.dtype,
            device=condition.condition_video_input_mask_B_C_T_H_W.device,
            include_future_proprio_region=include_future_proprio_region,
            mask_current_state_action_for_value_prediction=mask_current_state_action_for_value_prediction,
            mask_future_state_for_qvalue_prediction=mask_future_state_for_qvalue_prediction,
        )
        batch_indices = torch.arange(B, device=device)
        condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, action_latent_idx, :, :] = meta_mask_B_C_H_W

        action_region_vector = None
        if "actions" in data_batch and data_batch["actions"] is not None:
            action_region_vector = data_batch["actions"].to(device=device, dtype=dtype).reshape(B, -1)
        future_proprio_region_vector = (
            data_batch["future_proprio"].to(device=device, dtype=dtype).reshape(B, -1)
            if include_future_proprio_region
            else None
        )
        value_region_vector = None
        if "value_function_return" in data_batch and data_batch["value_function_return"] is not None:
            value_region_vector = data_batch["value_function_return"].to(device=device, dtype=dtype).reshape(B, 1)
        task_type_region_vector = self._build_task_type_region_vector(
            demo_mask_B,
            world_mask_B,
            value_mask_B,
            dtype=dtype,
            device=device,
        )
        current_proprio_region_vector = None
        if "proprio" in data_batch:
            current_proprio_region_vector = data_batch["proprio"].to(device=device, dtype=dtype).reshape(B, -1)

        condition.gt_frames = replace_latent_with_meta_regions(
            condition.gt_frames,
            meta_indices=action_latent_idx,
            current_proprio_region_vector=current_proprio_region_vector,
            action_region_vector=action_region_vector,
            future_proprio_region_vector=future_proprio_region_vector,
            value_region_vector=value_region_vector,
            task_type_region_vector=task_type_region_vector,
        )

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldCondition]:
        """
        Extended get_data_and_condition with policy-specific conditioning logic.

        Adds:
        - Condition mask manipulation for world model and value function samples
        - Latent injection for actions and proprio in gt_frames
        - Input masking for different prediction modes (V(s'), Q(s,a))
        """
        # generate random number of conditional frames for training
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        # Set video conditioning (from Video2WorldModel base functionality)
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        # Expand to the working mask width (slot-aware for MetaWAM); _finalize_condition_masks_for_net() derives the
        # masks the net consumes.
        expected_condition_mask_channels = self._get_working_condition_mask_channels()
        current_mask_channels = condition.condition_video_input_mask_B_C_T_H_W.shape[1]
        if current_mask_channels == 1 and expected_condition_mask_channels > 1:
            condition.condition_video_input_mask_B_C_T_H_W = condition.condition_video_input_mask_B_C_T_H_W.repeat(
                1, expected_condition_mask_channels, 1, 1, 1
            )
        else:
            assert current_mask_channels == expected_condition_mask_channels, (
                "condition mask channels mismatch: "
                f"got {current_mask_channels}, expected {expected_condition_mask_channels}"
            )

        action_latent_idx = data_batch["action_latent_idx"]
        value_latent_idx = action_latent_idx if self.config.use_single_meta_frame else data_batch["value_latent_idx"]
        original_future_proprio_latent_idx = data_batch["future_proprio_latent_idx"]
        future_proprio_latent_idx = (
            action_latent_idx if self.config.use_single_meta_frame else original_future_proprio_latent_idx
        )

        # NOTE (user):
        # If training future state prediction or value function on rollout data, adjust the condition_video_input_mask_B_C_T_H_W so that the
        # proper elements are treated as additional conditioning input (beyond the usual conditioning inputs)
        # - World model (future state prediction): action chunk is additionally treated as conditioning input
        # - Value function (expected return prediction): action chunk and future state are additionally treated as conditioning input
        if "rollout_data_mask" in data_batch:
            # For world model, set the video input mask to 1 for the action frame (i.e. make it a clean/denoised conditioning frame)
            # For value function, set mask to 1 for everything except the value function return frame (should be 0)
            world_model_sample_mask = data_batch["world_model_sample_mask"]
            value_function_sample_mask = data_batch["value_function_sample_mask"]
            # Expand masks from (B,) to (B, 1, 1, H', W')
            H_latent, W_latent = condition.condition_video_input_mask_B_C_T_H_W.shape[-2:]
            world_model_sample_mask = (
                world_model_sample_mask.unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .expand(-1, 1, 1, H_latent, W_latent)
            ).to(condition.condition_video_input_mask_B_C_T_H_W.dtype)
            value_function_sample_mask = (
                value_function_sample_mask.unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .expand(-1, 1, 1, H_latent, W_latent)
            ).to(condition.condition_video_input_mask_B_C_T_H_W.dtype)
            # World model: Set the action frame to 1
            batch_indices = torch.arange(world_model_sample_mask.shape[0], device=world_model_sample_mask.device)
            if not self.config.use_single_meta_frame:
                condition.condition_video_input_mask_B_C_T_H_W[
                    batch_indices, :, action_latent_idx, :, :
                ] = world_model_sample_mask[:, :, 0, :, :]
            # Value function: Set everything except the value function return frame to 1
            # First, set all frames to 1 for value function samples
            T = condition.condition_video_input_mask_B_C_T_H_W.shape[2]
            value_mask_all_frames = value_function_sample_mask.expand(-1, -1, T, -1, -1)  # (B, 1, T, H', W')
            condition.condition_video_input_mask_B_C_T_H_W = torch.where(
                value_mask_all_frames.bool(),
                torch.ones_like(condition.condition_video_input_mask_B_C_T_H_W),
                condition.condition_video_input_mask_B_C_T_H_W,
            )
            # Then set the value function return frame to 0
            condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, value_latent_idx, :, :] = (
                torch.where(
                    value_function_sample_mask[:, :, 0, :, :].bool(),
                    torch.zeros_like(
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, value_latent_idx, :, :
                        ]
                    ),
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_indices, :, value_latent_idx, :, :
                    ],
                )
            )
            # If we are predicting V(s') instead of V(s, a, s'), mask out the current state and action so that only the
            # future state is used for future state value prediction
            if self.config.mask_current_state_action_for_value_prediction:
                # Mask out current proprio frame
                if torch.all(data_batch["current_proprio_latent_idx"] != -1):  # -1 indicates proprio is not used
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_indices, :, data_batch["current_proprio_latent_idx"], :, :
                    ] = torch.where(
                        value_function_sample_mask[:, :, 0, :, :].bool(),
                        torch.zeros_like(
                            condition.condition_video_input_mask_B_C_T_H_W[
                                batch_indices, :, data_batch["current_proprio_latent_idx"], :, :
                            ]
                        ),
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, data_batch["current_proprio_latent_idx"], :, :
                        ],
                    )
                # Mask out current wrist image frame
                if torch.all(data_batch["current_wrist_image_latent_idx"] != -1):  # -1 indicates no wrist image is used
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_indices, :, data_batch["current_wrist_image_latent_idx"], :, :
                    ] = torch.where(
                        value_function_sample_mask[:, :, 0, :, :].bool(),
                        torch.zeros_like(
                            condition.condition_video_input_mask_B_C_T_H_W[
                                batch_indices, :, data_batch["current_wrist_image_latent_idx"], :, :
                            ]
                        ),
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, data_batch["current_wrist_image_latent_idx"], :, :
                        ],
                    )
                # Mask out current wrist image #2 frame
                if "current_wrist_image2_latent_idx" in data_batch and torch.all(
                    data_batch["current_wrist_image2_latent_idx"] != -1
                ):  # -1 indicates no wrist image is used
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_indices, :, data_batch["current_wrist_image2_latent_idx"], :, :
                    ] = torch.where(
                        value_function_sample_mask[:, :, 0, :, :].bool(),
                        torch.zeros_like(
                            condition.condition_video_input_mask_B_C_T_H_W[
                                batch_indices, :, data_batch["current_wrist_image2_latent_idx"], :, :
                            ]
                        ),
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, data_batch["current_wrist_image2_latent_idx"], :, :
                        ],
                    )
                # Mask out current image frame (primary image)
                if torch.all(data_batch["current_image_latent_idx"] != -1):  # -1 indicates primary image is not used
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_indices, :, data_batch["current_image_latent_idx"], :, :
                    ] = torch.where(
                        value_function_sample_mask[:, :, 0, :, :].bool(),
                        torch.zeros_like(
                            condition.condition_video_input_mask_B_C_T_H_W[
                                batch_indices, :, data_batch["current_image_latent_idx"], :, :
                            ]
                        ),
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, data_batch["current_image_latent_idx"], :, :
                        ],
                    )
                # Mask out current image #2 frame (secondary image)
                if "current_image2_latent_idx" in data_batch and torch.all(
                    data_batch["current_image2_latent_idx"] != -1
                ):  # -1 indicates secondary image is not used
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_indices, :, data_batch["current_image2_latent_idx"], :, :
                    ] = torch.where(
                        value_function_sample_mask[:, :, 0, :, :].bool(),
                        torch.zeros_like(
                            condition.condition_video_input_mask_B_C_T_H_W[
                                batch_indices, :, data_batch["current_image2_latent_idx"], :, :
                            ]
                        ),
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, data_batch["current_image2_latent_idx"], :, :
                        ],
                    )
                # Mask out action frame
                condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, action_latent_idx, :, :] = torch.where(
                    value_function_sample_mask[:, :, 0, :, :].bool(),
                    torch.zeros_like(
                        condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, action_latent_idx, :, :]
                    ),
                    condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, action_latent_idx, :, :],
                )
            # If we are predicting Q(s, a) instead of V(s, a, s'), mask out the future state so that only the
            # current state and action are used for Q(s, a) prediction
            if self.config.mask_future_state_for_qvalue_prediction:
                # Mask out future proprio frame
                # In single-meta mode, future proprio shares the meta/action slot. For Q(s,a) masking,
                # we should not mask the shared slot because action must remain visible.
                if (not self.config.use_single_meta_frame) and torch.all(
                    original_future_proprio_latent_idx != -1
                ):  # -1 indicates proprio is not used
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_indices, :, original_future_proprio_latent_idx, :, :
                    ] = torch.where(
                        value_function_sample_mask[:, :, 0, :, :].bool(),
                        torch.zeros_like(
                            condition.condition_video_input_mask_B_C_T_H_W[
                                batch_indices, :, original_future_proprio_latent_idx, :, :
                            ]
                        ),
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, original_future_proprio_latent_idx, :, :
                        ],
                    )
                # Mask out future wrist image frame
                if torch.all(data_batch["future_wrist_image_latent_idx"] != -1):  # -1 indicates no wrist image is used
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_indices, :, data_batch["future_wrist_image_latent_idx"], :, :
                    ] = torch.where(
                        value_function_sample_mask[:, :, 0, :, :].bool(),
                        torch.zeros_like(
                            condition.condition_video_input_mask_B_C_T_H_W[
                                batch_indices, :, data_batch["future_wrist_image_latent_idx"], :, :
                            ]
                        ),
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, data_batch["future_wrist_image_latent_idx"], :, :
                        ],
                    )
                # Mask out future wrist image #2 frame
                if "future_wrist_image2_latent_idx" in data_batch and torch.all(
                    data_batch["future_wrist_image2_latent_idx"] != -1
                ):  # -1 indicates no wrist image is used
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_indices, :, data_batch["future_wrist_image2_latent_idx"], :, :
                    ] = torch.where(
                        value_function_sample_mask[:, :, 0, :, :].bool(),
                        torch.zeros_like(
                            condition.condition_video_input_mask_B_C_T_H_W[
                                batch_indices, :, data_batch["future_wrist_image2_latent_idx"], :, :
                            ]
                        ),
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, data_batch["future_wrist_image2_latent_idx"], :, :
                        ],
                    )
                # Mask out future image frame (primary image)
                if torch.all(data_batch["future_image_latent_idx"] != -1):  # -1 indicates primary image is not used
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_indices, :, data_batch["future_image_latent_idx"], :, :
                    ] = torch.where(
                        value_function_sample_mask[:, :, 0, :, :].bool(),
                        torch.zeros_like(
                            condition.condition_video_input_mask_B_C_T_H_W[
                                batch_indices, :, data_batch["future_image_latent_idx"], :, :
                            ]
                        ),
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, data_batch["future_image_latent_idx"], :, :
                        ],
                    )
                # Mask out future image #2 frame (secondary image)
                if "future_image2_latent_idx" in data_batch and torch.all(
                    data_batch["future_image2_latent_idx"] != -1
                ):  # -1 indicates secondary image is not used
                    condition.condition_video_input_mask_B_C_T_H_W[
                        batch_indices, :, data_batch["future_image2_latent_idx"], :, :
                    ] = torch.where(
                        value_function_sample_mask[:, :, 0, :, :].bool(),
                        torch.zeros_like(
                            condition.condition_video_input_mask_B_C_T_H_W[
                                batch_indices, :, data_batch["future_image2_latent_idx"], :, :
                            ]
                        ),
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, data_batch["future_image2_latent_idx"], :, :
                        ],
                    )
            # Additionally, add the action chunk to the gt_frames so that the actions are added in later based on the mask
            # No need to do this for the other frames; actions are special because they are manually injected
            condition.orig_gt_frames = condition.gt_frames.clone()  # Keep a backup of the original gt_frames
            if not self.config.use_single_meta_frame:
                condition.gt_frames = replace_latent_with_action_chunk(
                    condition.gt_frames, data_batch["actions"], action_indices=action_latent_idx
                )

        # Manually add in the current and future proprio to the condition.gt_frames as well
        # (MetaWAM carries them in the meta frames instead)
        if (not self.config.use_single_meta_frame) and "proprio" in data_batch and torch.all(
            data_batch["current_proprio_latent_idx"] != -1
        ):  # -1 indicates proprio is not used
            condition.gt_frames = replace_latent_with_proprio(
                condition.gt_frames,
                data_batch["proprio"],
                proprio_indices=data_batch["current_proprio_latent_idx"],
            )
        if (
            (not self.config.use_single_meta_frame)
            and "future_proprio" in data_batch
            and torch.all(future_proprio_latent_idx != -1)
        ):  # -1 indicates proprio is not used
            condition.gt_frames = replace_latent_with_proprio(
                condition.gt_frames,
                data_batch["future_proprio"],
                proprio_indices=future_proprio_latent_idx,
            )

        # Manually add in value to the condition.gt_frames as well
        # (This is actually not needed for training because the value is not used as conditioning, but it may be useful
        # for visualizations when decoding the ground-truth latents to images)
        if (not self.config.use_single_meta_frame) and torch.all(value_latent_idx != -1) and "value_function_return" in data_batch:
            batch_indices = torch.arange(condition.gt_frames.shape[0], device=condition.gt_frames.device)
            _, C_latent, _, H_latent, W_latent = condition.gt_frames.shape
            condition.gt_frames[batch_indices, :, value_latent_idx, :, :] = (
                data_batch["value_function_return"]
                .reshape(-1, 1, 1, 1)
                .expand(-1, C_latent, H_latent, W_latent)
                .to(condition.gt_frames.dtype)
            )
        if self.config.use_single_meta_frame:
            self._apply_input_meta_frame_conditioning(condition, data_batch)
            self._apply_meta_frame_conditioning(condition, data_batch, action_latent_idx)
            # Slot 0 (input meta frame) is always hard-replaced with its ground truth.
            replace_mask = condition.condition_video_input_mask_B_C_T_H_W.clone()
            replace_mask[:, :, 0, :, :] = 1.0
            condition.condition_video_input_replace_mask_B_C_T_H_W = replace_mask

        condition = self._finalize_condition_masks_for_net(condition, context="training")
        return raw_state, latent_state, condition

    def denoise(
        self, xt_B_C_T_H_W: torch.Tensor, sigma: torch.Tensor, condition: Text2WorldCondition
    ) -> DenoisePrediction:
        """
        Performs denoising with optional debugging visualization support.

        Extended from base to add debugging code for visualizing latent frames.
        """
        if sigma.ndim == 1:
            sigma_B_T = rearrange(sigma, "b -> b 1")
        elif sigma.ndim == 2:
            sigma_B_T = sigma
        else:
            raise ValueError(f"sigma shape {sigma.shape} is not supported")

        sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
        # get precondition for the network
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(sigma=sigma_B_1_T_1_1)

        net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in_B_1_T_1_1

        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(net_state_in_B_C_T_H_W) / self.config.sigma_data
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            # Hard GT replacement uses the explicit replacement mask when provided (MetaWAM: slot-aware).
            raw_condition_video_mask = getattr(
                condition,
                "condition_video_input_replace_mask_B_C_T_H_W",
                None,
            )
            if raw_condition_video_mask is None:
                raw_condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W
            if raw_condition_video_mask.shape[1] == 1:
                condition_video_mask = raw_condition_video_mask.repeat(1, C, 1, 1, 1)
            else:
                assert raw_condition_video_mask.shape[1] == C, (
                    "Condition mask channel count must be 1 or equal latent channels: "
                    f"got {raw_condition_video_mask.shape[1]} vs C={C}"
                )
                condition_video_mask = raw_condition_video_mask
            condition_video_mask = condition_video_mask.type_as(net_state_in_B_C_T_H_W)

            # Replace the first few frames of the video with the conditional frames
            # Update the c_noise as the conditional frames are clean and have very low noise

            # Make the first few frames of x_t be the ground truth frames
            net_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + net_state_in_B_C_T_H_W * (
                1 - condition_video_mask
            )
            # Adjust c_noise for the conditional frames
            sigma_cond_B_1_T_1_1 = torch.ones_like(sigma_B_1_T_1_1) * self.config.sigma_conditional
            _, _, _, c_noise_cond_B_1_T_1_1 = self.scaling(sigma=sigma_cond_B_1_T_1_1)
            condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
            c_noise_B_1_T_1_1 = c_noise_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + c_noise_B_1_T_1_1 * (
                1 - condition_video_mask_B_1_T_1_1
            )

        # forward pass through the network
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
                **{
                    **self.tensor_kwargs,
                    "dtype": torch.float32 if self.config.use_wan_fp32_strategy else self.tensor_kwargs["dtype"],
                },
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        ).float()

        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        if condition.is_video and self.config.denoise_replace_gt_frames:
            # Set the first few frames to the ground truth frames. This will ensure that the loss is not computed for the first few frames.
            x0_pred_B_C_T_H_W = condition.gt_frames.type_as(
                x0_pred_B_C_T_H_W
            ) * condition_video_mask + x0_pred_B_C_T_H_W * (1 - condition_video_mask)

        # get noise prediction based on sde
        eps_pred_B_C_T_H_W = (xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / sigma_B_1_T_1_1

        # ####################################################################################
        # NOTE (user): Below is useful debugging code for visualizing the latent frames
        #                 Set DEBUGGING to True to enable
        # ####################################################################################
        DEBUGGING = False
        if DEBUGGING:
            SAMPLE_INDEX = 0  # Which sample to visualize from the batch
            import torch.distributed as dist

            if (not dist.is_initialized()) or (
                dist.is_initialized() and dist.get_rank() == 0
            ):  # Run on master process only
                from PIL import Image

                os.makedirs("temp", exist_ok=True)
                # Visualize the ground-truth latent frames (contaminated version that has visual artifacts due to latent injections)
                ground_truth_frames = condition.gt_frames
                decoded_ground_truth_frames = self.tokenizer.decode(ground_truth_frames)
                unnormalized_decoded_ground_truth_frames = (
                    ((decoded_ground_truth_frames + 1.0) * 127.5)
                    .clamp(0, 255)
                    .permute(0, 2, 3, 4, 1)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )  # (B, C=3, T=num_images, H=224, W=224)
                for idx in range(unnormalized_decoded_ground_truth_frames.shape[1]):
                    save_path = os.path.join("temp", f"contaminated_ground_truth_frames--{idx}.png")
                    Image.fromarray(unnormalized_decoded_ground_truth_frames[SAMPLE_INDEX, idx]).save(save_path)
                    print(f"Saved contaminated ground-truth frame at path: {save_path}")
                # Visualize the ground-truth latent frames (clean pre-injection version with no injected latents)
                ground_truth_frames = condition.orig_gt_frames
                decoded_ground_truth_frames = self.tokenizer.decode(ground_truth_frames)
                unnormalized_decoded_ground_truth_frames = (
                    ((decoded_ground_truth_frames + 1.0) * 127.5)
                    .clamp(0, 255)
                    .permute(0, 2, 3, 4, 1)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )  # (B, C=3, T=num_images, H=224, W=224)
                for idx in range(unnormalized_decoded_ground_truth_frames.shape[1]):
                    save_path = os.path.join("temp", f"clean_ground_truth_frames--{idx}.png")
                    Image.fromarray(unnormalized_decoded_ground_truth_frames[SAMPLE_INDEX, idx]).save(save_path)
                    print(f"Saved clean ground-truth frame at path: {save_path}")

                # # Visualize the non-image modalities by themselves (no images) - so that the decodings are cleaner
                # # This involves stitching together the non-image modalities and removing the images
                # # NOTE: HARDCODED for RoboCasa and ALOHA sequences (11 latent frames) - does not work for LIBERO
                # B, C, T, H, W = condition.gt_frames.shape
                # nonimage_sequence = (
                #     torch.zeros((B, C, 1 + 4 * 2, H, W)).to(condition.gt_frames.device).to(condition.gt_frames.dtype)
                # )  # 1 blank + 4 non-image modalities (proprio, action chunk, future proprio, value) with a blank frame in between each of them
                # nonimage_sequence[:, :, 1, :, :] = condition.gt_frames[:, :, 1, :, :]  # Current proprio
                # nonimage_sequence[:, :, 3, :, :] = condition.gt_frames[:, :, 5, :, :]  # Action chunk
                # nonimage_sequence[:, :, 5, :, :] = condition.gt_frames[:, :, 6, :, :]  # Future proprio
                # nonimage_sequence[:, :, 7, :, :] = condition.gt_frames[:, :, 10, :, :]  # Value
                # decoded_nonimage_sequence = self.tokenizer.decode(nonimage_sequence)
                # unnormalized_decoded_nonimage_sequence = (
                #     ((decoded_nonimage_sequence + 1.0) * 127.5)
                #     .clamp(0, 255)
                #     .permute(0, 2, 3, 4, 1)
                #     .to(torch.uint8)
                #     .cpu()
                #     .numpy()
                # )  # (B, C=3, T=num_images, H=224, W=224)
                # for idx in range(unnormalized_decoded_nonimage_sequence.shape[1]):
                #     save_path = os.path.join("temp", f"decoded_nonimage_frames--{idx}.png")
                #     Image.fromarray(unnormalized_decoded_nonimage_sequence[SAMPLE_INDEX, idx]).save(save_path)
                #     print(f"Saved decoded nonimage frame at path: {save_path}")

                # Visualize the noised latent frames
                noisy_input = net_state_in_B_C_T_H_W  # (B, C'=16, T'=num_video_latent_frames, H'=28, W'=28)
                decoded_noisy_input = self.tokenizer.decode(noisy_input)  # (B, C=3, T=num_images, H=224, W=224)
                unnormalized_decoded_noisy_input = (
                    ((decoded_noisy_input + 1.0) * 127.5)
                    .clamp(0, 255)
                    .permute(0, 2, 3, 4, 1)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )  # (B, C=3, T=num_images, H=224, W=224)
                for idx in range(unnormalized_decoded_noisy_input.shape[1]):
                    save_path = os.path.join("temp", f"noised_latent_frames--{idx}.png")
                    Image.fromarray(unnormalized_decoded_noisy_input[SAMPLE_INDEX, idx]).save(save_path)
                    print(f"Saved noised latent frame at path: {save_path}")
                # Visualize the denoised latent frames
                denoised_output = x0_pred_B_C_T_H_W  # (B, C'=16, T'=num_video_latent_frames, H'=28, W'=28)
                decoded_denoised_output = self.tokenizer.decode(denoised_output)  # (B, C=3, T=num_images, H=224, W=224)
                unnormalized_decoded_denoised_output = (
                    ((decoded_denoised_output + 1.0) * 127.5)
                    .clamp(0, 255)
                    .permute(0, 2, 3, 4, 1)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )  # (B, C=3, T=num_images, H=224, W=224)
                for idx in range(unnormalized_decoded_denoised_output.shape[1]):
                    save_path = os.path.join("temp", f"denoised_latent_frames--{idx}.png")
                    Image.fromarray(unnormalized_decoded_denoised_output[SAMPLE_INDEX, idx]).save(save_path)
                    print(f"Saved denoised latent frame at path: {save_path}")

        return DenoisePrediction(x0_pred_B_C_T_H_W, eps_pred_B_C_T_H_W, None)

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        skip_vae_encoding: bool = False,
        previous_generated_latent: torch.Tensor = None,
        return_orig_clean_latent_frames: bool = False,
    ) -> Callable:
        """
        NOTE (user): Modified to remove the negative prompt and "uncondition" since we are doing always-conditional generation instead of CFG.

        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        Extended to support:
        - Handling uncondition=None case (always-conditional generation)
        - Skip VAE encoding for autoregressive generation
        - Return original clean latent frames
        - Latent injection for proprio and actions in conditioning

        Args:
        - data_batch (Dict): A batch of data used for conditioning.
        - guidance (float, optional): Guidance scale. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true
        - skip_vae_encoding (bool): Skip VAE encoding if True
        - previous_generated_latent (torch.Tensor): Previous generated sample
        - return_orig_clean_latent_frames (bool): Whether to return the original condition

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that returns x0 prediction
        """

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        elif guidance == 0:
            # Inference fast path: when guidance is exactly 0, classifier-free guidance reduces
            # to the conditional branch, so we can skip constructing / running uncondition.
            cond_dropout_rates = {emb_name: 0.0 for emb_name in self.conditioner.embedders.keys()}
            condition = self.conditioner(data_batch, override_dropout_rate=cond_dropout_rates)
            uncondition = None
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        if uncondition is not None:
            uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        if skip_vae_encoding:
            assert previous_generated_latent is not None, (
                "previous_generated_latent must be provided if skip_vae_encoding is True!"
            )
            x0 = previous_generated_latent.clone()
        else:
            _, x0, _ = self.get_data_and_condition(data_batch)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        if uncondition is not None:
            uncondition = uncondition.set_video_condition(
                gt_frames=x0,
                random_min_num_conditional_frames=self.config.min_num_conditional_frames,
                random_max_num_conditional_frames=self.config.max_num_conditional_frames,
                num_conditional_frames=num_conditional_frames,
            )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        if uncondition is not None:
            uncondition = uncondition.edit_for_inference(
                is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
            )
        expected_condition_mask_channels = self._get_working_condition_mask_channels()

        def _align_condition_mask_channels(cond: Optional[Video2WorldCondition]) -> Optional[Video2WorldCondition]:
            if cond is None:
                return None
            current_channels = cond.condition_video_input_mask_B_C_T_H_W.shape[1]
            if current_channels == 1 and expected_condition_mask_channels > 1:
                cond.condition_video_input_mask_B_C_T_H_W = cond.condition_video_input_mask_B_C_T_H_W.repeat(
                    1, expected_condition_mask_channels, 1, 1, 1
                )
            else:
                assert current_channels == expected_condition_mask_channels, (
                    "condition mask channels mismatch in inference: "
                    f"got {current_channels}, expected {expected_condition_mask_channels}"
                )
            return cond

        condition = _align_condition_mask_channels(condition)
        uncondition = _align_condition_mask_channels(uncondition)

        # NOTE (user):
        # The original gt_frames latent is useful for decoding latents to images without distortions
        # caused by the latent injections below, but only when callers explicitly request it.
        condition.orig_gt_frames = condition.gt_frames.clone() if return_orig_clean_latent_frames else None

        B = condition.condition_video_input_mask_B_C_T_H_W.shape[0]
        action_latent_idx = data_batch["action_latent_idx"]
        original_future_proprio_latent_idx = data_batch["future_proprio_latent_idx"]

        # NOTE (user):
        # If generating samples with current proprio fed as condition via latent injection, adjust the condition_video_input_mask_B_C_T_H_W so that the
        # current proprio latent frame is treated as conditioning input (MetaWAM carries it in the input meta frame)
        if (not self.config.use_single_meta_frame) and "proprio" in data_batch and torch.all(
            data_batch["current_proprio_latent_idx"] != -1
        ):  # -1 indicates proprio is not used
            proprio = data_batch["proprio"]
            current_proprio_latent_idx = data_batch["current_proprio_latent_idx"]
            batch_indices = torch.arange(B, device=proprio.device)
            condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, current_proprio_latent_idx, :, :] = 1
            # Additionally, add the proprio to the gt_frames so that the proprio is added in later based on the mask
            condition.gt_frames = replace_latent_with_proprio(
                condition.gt_frames, proprio, proprio_indices=current_proprio_latent_idx
            )

        if (
            "mask_current_state_action_for_value_prediction" in data_batch
            and data_batch["mask_current_state_action_for_value_prediction"]
        ):
            batch_indices = torch.arange(B, device=condition.condition_video_input_mask_B_C_T_H_W.device)
            # Mask out current proprio frame
            if torch.all(data_batch["current_proprio_latent_idx"] != -1):  # -1 indicates proprio is not used
                condition.condition_video_input_mask_B_C_T_H_W[
                    batch_indices, :, data_batch["current_proprio_latent_idx"], :, :
                ] = 0
            # Mask out current wrist image frame
            if torch.all(data_batch["current_wrist_image_latent_idx"] != -1):  # -1 indicates wrist image is not used
                condition.condition_video_input_mask_B_C_T_H_W[
                    batch_indices, :, data_batch["current_wrist_image_latent_idx"], :, :
                ] = 0
            # Mask out current wrist image #2 frame
            if "current_wrist_image2_latent_idx" in data_batch and torch.all(
                data_batch["current_wrist_image2_latent_idx"] != -1
            ):  # -1 indicates wrist image #2 is not used
                condition.condition_video_input_mask_B_C_T_H_W[
                    batch_indices, :, data_batch["current_wrist_image2_latent_idx"], :, :
                ] = 0
            # Mask out current image frame (primary image)
            if torch.all(data_batch["current_image_latent_idx"] != -1):  # -1 indicates primary image is not used
                condition.condition_video_input_mask_B_C_T_H_W[
                    batch_indices, :, data_batch["current_image_latent_idx"], :, :
                ] = 0
            # Mask out current image #2 frame (secondary image)
            if "current_image2_latent_idx" in data_batch and torch.all(
                data_batch["current_image2_latent_idx"] != -1
            ):  # -1 indicates secondary image is not used
                condition.condition_video_input_mask_B_C_T_H_W[
                    batch_indices, :, data_batch["current_image2_latent_idx"], :, :
                ] = 0
            # Mask out action frame
            condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, action_latent_idx, :, :] = 0

        if (
            "mask_future_state_for_qvalue_prediction" in data_batch
            and data_batch["mask_future_state_for_qvalue_prediction"]
        ):
            batch_indices = torch.arange(B, device=condition.condition_video_input_mask_B_C_T_H_W.device)
            # Mask out future proprio frame
            # In single-meta mode, future proprio shares the meta/action slot. For Q(s,a) masking,
            # we should not mask the shared slot because action must remain visible.
            if (not self.config.use_single_meta_frame) and torch.all(
                original_future_proprio_latent_idx != -1
            ):  # -1 indicates proprio is not used
                condition.condition_video_input_mask_B_C_T_H_W[
                    batch_indices, :, original_future_proprio_latent_idx, :, :
                ] = 0
            # Mask out future wrist image frame
            if torch.all(data_batch["future_wrist_image_latent_idx"] != -1):  # -1 indicates wrist image is not used
                condition.condition_video_input_mask_B_C_T_H_W[
                    batch_indices, :, data_batch["future_wrist_image_latent_idx"], :, :
                ] = 0
            # Mask out future wrist image #2 frame
            if "future_wrist_image2_latent_idx" in data_batch and torch.all(
                data_batch["future_wrist_image2_latent_idx"] != -1
            ):  # -1 indicates wrist image #2 is not used
                condition.condition_video_input_mask_B_C_T_H_W[
                    batch_indices, :, data_batch["future_wrist_image2_latent_idx"], :, :
                ] = 0
            # Mask out future image frame (primary image)
            if torch.all(data_batch["future_image_latent_idx"] != -1):  # -1 indicates primary image is not used
                condition.condition_video_input_mask_B_C_T_H_W[
                    batch_indices, :, data_batch["future_image_latent_idx"], :, :
                ] = 0
            # Mask out future image #2 frame (secondary image)
            if "future_image2_latent_idx" in data_batch and torch.all(
                data_batch["future_image2_latent_idx"] != -1
            ):  # -1 indicates secondary image is not used
                condition.condition_video_input_mask_B_C_T_H_W[
                    batch_indices, :, data_batch["future_image2_latent_idx"], :, :
                ] = 0

        if self.config.use_single_meta_frame:
            self._apply_input_meta_frame_conditioning(condition, data_batch)
            if uncondition is not None:
                self._apply_input_meta_frame_conditioning(uncondition, data_batch)
            self._apply_meta_frame_conditioning(condition, data_batch, action_latent_idx)
            if uncondition is not None:
                self._apply_meta_frame_conditioning(uncondition, data_batch, action_latent_idx)
            replace_mask = condition.condition_video_input_mask_B_C_T_H_W.clone()
            replace_mask[:, :, 0, :, :] = 1.0
            condition.condition_video_input_replace_mask_B_C_T_H_W = replace_mask
            if uncondition is not None:
                uncondition_replace_mask = uncondition.condition_video_input_mask_B_C_T_H_W.clone()
                uncondition_replace_mask[:, :, 0, :, :] = 1.0
                uncondition.condition_video_input_replace_mask_B_C_T_H_W = uncondition_replace_mask

        condition = self._finalize_condition_masks_for_net(condition, context="inference-condition")
        uncondition = self._finalize_condition_masks_for_net(uncondition, context="inference-uncondition")
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        if uncondition is not None:
            _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            if self.config.use_flowunipc_scheduler:
                cond_velocity = self.denoise_with_velocity(noise_x, sigma, condition)
                uncond_velocity = self.denoise_with_velocity(noise_x, sigma, uncondition)
                velocity = uncond_velocity + guidance * (cond_velocity - uncond_velocity)
                return velocity
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            if uncondition is not None:
                uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
                raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
            else:
                raw_x0 = cond_x0
            if "guided_image" in data_batch:
                # replacement trick that enables inpainting with base model
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
            return raw_x0

        if return_orig_clean_latent_frames:
            return x0_fn, condition.orig_gt_frames
        else:
            return x0_fn

    def draw_training_sigma_and_epsilon(self, x0_size: int, condition: Any) -> torch.Tensor:
        """
        Extended sigma sampling with high sigma strategy support.

        Inherited from Video2WorldModel to support various high sigma strategies.
        """
        sigma_B_1, epsilon = super().draw_training_sigma_and_epsilon(x0_size, condition)
        is_video_batch = condition.data_type == DataType.VIDEO
        # if is_video_batch, with 5% ratio, we regenerate sigma_B_1 with uniformally from 80 to 2000
        # with remaining 95% ratio, we keep the original sigma_B_1
        if is_video_batch:
            if self.config.high_sigma_strategy == str(HighSigmaStrategy.UNIFORM80_2000):
                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
                new_sigma = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * 1920 + 80
                sigma_B_1 = torch.where(mask, new_sigma, sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.LOGUNIFORM200_100000):
                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
                log_new_sigma = (
                    torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * (LOG_100000 - LOG_200)
                    + LOG_200
                )
                sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.SHIFT24):
                # sample t from uniform distribution between 0 and 1, with same shape as sigma_B_1
                _t = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).double()
                _t = 24 * _t / (24 * _t + 1 - _t)
                sigma_B_1 = (_t / (1.0 - _t)).float()

                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
                new_sigma = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * 1920 + 80
                sigma_B_1 = torch.where(mask, new_sigma, sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.BALANCED_TWO_HEADS_V1):
                # replace high sigma parts
                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
                log_new_sigma = (
                    torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * (LOG_100000 - LOG_200)
                    + LOG_200
                )
                sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
                # replace low sigma parts
                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.low_sigma_ratio
                low_sigma_B_1 = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * 2.0 + 0.00001
                sigma_B_1 = torch.where(mask, low_sigma_B_1, sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.HARDCODED_20steps):
                if not hasattr(self, "hardcoded_20steps_sigma"):
                    from cosmos_policy._src.imaginaire.modules.res_sampler import get_rev_ts

                    hardcoded_20steps_sigma = get_rev_ts(
                        t_min=self.sde.sigma_min, t_max=self.sde.sigma_max, num_steps=20, ts_order=7.0
                    )
                    # add extra 100000 to the beginning
                    self.hardcoded_20steps_sigma = torch.cat(
                        [torch.tensor([100000.0], device=hardcoded_20steps_sigma.device), hardcoded_20steps_sigma],
                        dim=0,
                    )
                sigma_B_1 = self.hardcoded_20steps_sigma[
                    torch.randint(0, len(self.hardcoded_20steps_sigma), sigma_B_1.shape)
                ].type_as(sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.NONE):
                pass
            else:
                raise ValueError(f"High sigma strategy {self.config.high_sigma_strategy} is not supported")
        return sigma_B_1, epsilon

    def denoise_with_velocity(
        self, noise_x_in_t_space: torch.Tensor, t_B_T: torch.Tensor, condition: Text2WorldCondition
    ) -> torch.Tensor:
        """
        This function is used when self.config.use_flowunipc_scheduler is set.
        """
        if t_B_T.ndim == 1:
            t_B_T = rearrange(t_B_T, "b -> b 1")
        elif t_B_T.ndim == 2:
            t_B_T = t_B_T
        else:
            raise ValueError(f"t_B_T shape {t_B_T.shape} is not supported")
        # our model expects input of sigma and x_sigma, so convert t -> sigma, x_t to x_sigma
        sigma_B_T = t_B_T / (1.0 - t_B_T)
        x_B_C_T_H_W_in_sigma_space = noise_x_in_t_space * (1.0 + rearrange(sigma_B_T, "b t -> b 1 t 1 1"))
        denoise_output_B_C_T_H_W = self.denoise(x_B_C_T_H_W_in_sigma_space, sigma_B_T, condition)
        x0_pred_B_C_T_H_W = denoise_output_B_C_T_H_W.x0
        eps_pred_B_C_T_H_W = denoise_output_B_C_T_H_W.eps
        return eps_pred_B_C_T_H_W - x0_pred_B_C_T_H_W
