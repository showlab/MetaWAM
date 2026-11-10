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
Cosmos Policy Diffusion Model - extends Text2WorldModel with policy-specific functionality.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import attrs
import torch
import torch.nn.functional as F
from einops import rearrange

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_policy._src.imaginaire.modules.res_sampler import COMMON_SOLVER_OPTIONS
from cosmos_policy._src.imaginaire.utils import misc
from cosmos_policy._src.imaginaire.utils.context_parallel import broadcast_split_tensor, cat_outputs_cp
from cosmos_policy._src.predict2.models.text2world_model import (
    DiffusionModel as BaseDiffusionModel,
)
from cosmos_policy._src.predict2.models.text2world_model import (
    Text2WorldModelConfig as BaseText2WorldModelConfig,
)
from cosmos_policy.conditioner import Text2WorldCondition
from cosmos_policy.modules.cosmos_sampler import CosmosPolicySampler
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE


# MetaWAM frequency meta frame: channel-partitioned slots (16 latent channels expected).
# Slot values are row-0 (k_H=0) orthonormal 2D-DCT coefficients:
# - ch0: task one-hot
# - ch1-3: current proprio (9D split into 3x3)
# - ch4-11: action chunk (packed over row-0 8xW coeff slots)
# - ch12-14: future proprio (9D packed over 3xW coeff slots)
# - ch15: value (scalar)
_FREQUENCY_V2_TASK_SLICE = slice(0, 1)
_FREQUENCY_V2_CURRENT_PROPRIO_SLICE = slice(1, 4)
_FREQUENCY_V2_ACTION_SLICE = slice(4, 12)
_FREQUENCY_V2_FUTURE_PROPRIO_SLICE = slice(12, 15)
_FREQUENCY_V2_VALUE_SLICE = slice(15, 16)
_FREQUENCY_V2_LATENT_CHANNELS_MIN = 16
_FREQUENCY_V2_FUTURE_PROPRIO_ELEMENTS = 9
_FREQUENCY_V2_CURRENT_PROPRIO_ELEMENTS = 9
_DCT_ORTHO_BASIS_CACHE: dict[tuple[int, str, int], torch.Tensor] = {}


def _get_orthonormal_dct_basis(length: int, device: torch.device) -> torch.Tensor:
    """
    Build the orthonormal DCT-II basis matrix A (N x N):
    - dct(x) = x @ A
    - idct(X) = X @ A^T
    """
    key = (int(length), device.type, -1 if device.index is None else int(device.index))
    if key in _DCT_ORTHO_BASIS_CACHE:
        return _DCT_ORTHO_BASIS_CACHE[key]
    n = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)
    k = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N)
    basis = torch.cos((math.pi / float(length)) * (n + 0.5) * k)  # (N, N)
    basis = basis * math.sqrt(2.0 / float(length))
    basis[:, 0] = basis[:, 0] / math.sqrt(2.0)
    _DCT_ORTHO_BASIS_CACHE[key] = basis
    return basis


def _dct_ortho_last_dim(x: torch.Tensor) -> torch.Tensor:
    basis = _get_orthonormal_dct_basis(x.shape[-1], x.device)
    return torch.matmul(x.float(), basis)


def _idct_ortho_last_dim(x: torch.Tensor) -> torch.Tensor:
    basis = _get_orthonormal_dct_basis(x.shape[-1], x.device)
    return torch.matmul(x.float(), basis.transpose(0, 1))


def _idct2_row0_coeff_per_channel_ortho(coeff_row0_B_C_W: torch.Tensor, height: int) -> torch.Tensor:
    """
    Specialized orthonormal 2D IDCT for tensors whose only non-zero coefficients lie on row k_H=0.

    For row-0-only spectra, the H-axis IDCT reduces exactly to multiplying by the DC basis value
    1/sqrt(H) and broadcasting over the spatial H dimension.
    """
    spatial_row0_B_C_W = _idct_ortho_last_dim(coeff_row0_B_C_W)
    batch_size, latent_channels, latent_w = spatial_row0_B_C_W.shape
    scale = 1.0 / math.sqrt(float(height))
    return (spatial_row0_B_C_W * scale).unsqueeze(-2).expand(batch_size, latent_channels, height, latent_w)


def _dct2_per_channel_ortho(x_B_C_H_W: torch.Tensor) -> torch.Tensor:
    out = _dct_ortho_last_dim(x_B_C_H_W)
    out = _dct_ortho_last_dim(out.transpose(-1, -2)).transpose(-1, -2)
    return out


def _dct2_row0_coeff_per_channel_ortho(x_B_C_H_W: torch.Tensor) -> torch.Tensor:
    """
    Compute only the k_H=0 row of the orthonormal 2D DCT.

    For the orthonormal DCT basis, the H-axis DC basis vector is constant 1/sqrt(H),
    so row-0 coefficients can be obtained exactly by averaging over H (with the
    corresponding sqrt(H) normalization) and then applying a 1D DCT over W.
    """
    height = x_B_C_H_W.shape[-2]
    row0_signal_B_C_W = x_B_C_H_W.float().mean(dim=-2) * math.sqrt(float(height))
    return _dct_ortho_last_dim(row0_signal_B_C_W)


def _get_frequency_action_slot_capacity(latent_h: int, latent_w: int, action_slice: slice) -> int:
    del latent_h
    return (action_slice.stop - action_slice.start) * latent_w


def _pack_frequency_v2_action_slots(
    action_flat: torch.Tensor,
    latent_h: int,
    latent_w: int,
    action_slice: slice = _FREQUENCY_V2_ACTION_SLICE,
) -> torch.Tensor:
    """Pack a flattened action vector into the row-0 action slots."""
    batch_size = action_flat.shape[0]
    action_channels = action_slice.stop - action_slice.start
    action_slot_capacity = _get_frequency_action_slot_capacity(latent_h, latent_w, action_slice)
    assert action_flat.shape[1] <= action_slot_capacity, (
        "frequency action slot overflow: "
        f"{action_flat.shape[1]} > {action_slot_capacity}"
    )
    action_padded = torch.zeros((batch_size, action_slot_capacity), device=action_flat.device, dtype=action_flat.dtype)
    action_padded[:, : action_flat.shape[1]] = action_flat
    return action_padded.reshape(batch_size, action_channels, latent_w)


def _build_frequency_v2_action_slot_mask(
    batch_size: int,
    coeff_device: torch.device,
    coeff_dtype: torch.dtype,
    latent_h: int,
    latent_w: int,
    num_action_elements: int,
    action_slice: slice = _FREQUENCY_V2_ACTION_SLICE,
) -> torch.Tensor:
    """Binary mask for active row-0 action slots."""
    action_channels = action_slice.stop - action_slice.start
    action_mask = torch.zeros(
        (
            batch_size,
            action_channels,
            latent_h,
            latent_w,
        ),
        device=coeff_device,
        dtype=coeff_dtype,
    )
    if num_action_elements <= 0:
        return action_mask
    action_capacity = action_channels * latent_w
    active_elements = min(num_action_elements, action_capacity)
    action_mask[:, :, 0, :].reshape(batch_size, -1)[:, :active_elements] = 1.0
    return action_mask


def _pack_frequency_v2_vector_slots(
    vector_flat: torch.Tensor,
    slot_slice: slice,
    latent_w: int,
    slot_name: str,
) -> torch.Tensor:
    batch_size = vector_flat.shape[0]
    slot_channels = slot_slice.stop - slot_slice.start
    slot_capacity = slot_channels * latent_w
    assert vector_flat.shape[1] <= slot_capacity, (
        f"frequency {slot_name} slot overflow: {vector_flat.shape[1]} > {slot_capacity}"
    )
    vector_padded = torch.zeros((batch_size, slot_capacity), device=vector_flat.device, dtype=vector_flat.dtype)
    vector_padded[:, : vector_flat.shape[1]] = vector_flat
    return vector_padded.reshape(batch_size, slot_channels, latent_w)


def _build_frequency_v2_vector_slot_mask(
    batch_size: int,
    coeff_device: torch.device,
    coeff_dtype: torch.dtype,
    latent_h: int,
    latent_w: int,
    slot_slice: slice,
    num_elements: int,
) -> torch.Tensor:
    slot_channels = slot_slice.stop - slot_slice.start
    slot_mask = torch.zeros(
        (batch_size, slot_channels, latent_h, latent_w),
        device=coeff_device,
        dtype=coeff_dtype,
    )
    if num_elements <= 0:
        return slot_mask
    slot_capacity = slot_channels * latent_w
    active_elements = min(num_elements, slot_capacity)
    slot_mask[:, :, 0, :].reshape(batch_size, -1)[:, :active_elements] = 1.0
    return slot_mask


def _encode_frequency_meta_frame_v2(
    meta_latent: torch.Tensor,
    current_proprio_region_vector: Optional[torch.Tensor],
    action_region_vector: Optional[torch.Tensor],
    future_proprio_region_vector: Optional[torch.Tensor],
    value_region_vector: Optional[torch.Tensor],
    task_type_region_vector: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Encode a MetaWAM meta frame with channel-partitioned slots.

    Slot values are written in DCT space and projected back to latent space by orthonormal IDCT.
    """
    batch_size, latent_channels, latent_h, latent_w = meta_latent.shape
    assert latent_channels >= _FREQUENCY_V2_LATENT_CHANNELS_MIN, (
        f"frequency meta layout requires >={_FREQUENCY_V2_LATENT_CHANNELS_MIN} latent channels, got C={latent_channels}"
    )
    coeff_row0_B_C_W = torch.zeros(
        (batch_size, latent_channels, latent_w),
        device=meta_latent.device,
        dtype=torch.float32,
    )

    if task_type_region_vector is not None:
        task_type = task_type_region_vector.reshape(batch_size, -1).to(device=meta_latent.device, dtype=torch.float32)
        t_keep = min(3, latent_w, task_type.shape[1])
        coeff_row0_B_C_W[:, _FREQUENCY_V2_TASK_SLICE, :t_keep] = task_type[:, None, :t_keep]

    if current_proprio_region_vector is not None:
        current = current_proprio_region_vector.reshape(batch_size, -1).to(device=meta_latent.device, dtype=torch.float32)
        assert current.shape[1] >= _FREQUENCY_V2_CURRENT_PROPRIO_ELEMENTS, (
            f"frequency current proprio expects >={_FREQUENCY_V2_CURRENT_PROPRIO_ELEMENTS} elements, "
            f"got {current.shape[1]}"
        )
        # 3 coefficients in each of the 3 current-proprio channels (ch1-3).
        p_keep = min(3, latent_w)
        coeff_row0_B_C_W[:, 1, :p_keep] = current[:, :p_keep]
        coeff_row0_B_C_W[:, 2, :p_keep] = current[:, 3 : 3 + p_keep]
        coeff_row0_B_C_W[:, 3, :p_keep] = current[:, 6 : 6 + p_keep]

    if action_region_vector is not None:
        action_flat = action_region_vector.reshape(batch_size, -1).to(device=meta_latent.device, dtype=torch.float32)
        action_packed = _pack_frequency_v2_action_slots(action_flat, latent_h=latent_h, latent_w=latent_w)
        coeff_row0_B_C_W[:, _FREQUENCY_V2_ACTION_SLICE, :] = action_packed

    if future_proprio_region_vector is not None:
        future = future_proprio_region_vector.reshape(batch_size, -1).to(device=meta_latent.device, dtype=torch.float32)
        assert future.shape[1] >= _FREQUENCY_V2_FUTURE_PROPRIO_ELEMENTS, (
            f"frequency future proprio expects >={_FREQUENCY_V2_FUTURE_PROPRIO_ELEMENTS} elements, "
            f"got {future.shape[1]}"
        )
        future_packed = _pack_frequency_v2_vector_slots(
            future[:, :_FREQUENCY_V2_FUTURE_PROPRIO_ELEMENTS],
            _FREQUENCY_V2_FUTURE_PROPRIO_SLICE,
            latent_w,
            "future proprio",
        )
        coeff_row0_B_C_W[:, _FREQUENCY_V2_FUTURE_PROPRIO_SLICE, :] = future_packed

    if value_region_vector is not None:
        value_flat = value_region_vector.reshape(batch_size, -1).to(device=meta_latent.device, dtype=torch.float32)
        assert value_flat.shape[1] >= 1, "frequency value expects at least one element"
        coeff_row0_B_C_W[:, _FREQUENCY_V2_VALUE_SLICE.start, 0] = value_flat[:, 0]

    return _idct2_row0_coeff_per_channel_ortho(coeff_row0_B_C_W, height=latent_h).to(
        dtype=meta_latent.dtype
    ).contiguous()


def replace_latent_with_action_chunk(
    x0: torch.Tensor, action_chunk: torch.Tensor, action_indices: torch.Tensor
) -> torch.Tensor:
    """
    Replaces the image latent (at the specified action index) in clean input image latents x0 with the action chunk.

    Example:
    Let's say x0 has shape (B=32, C'=16, T', H'=28, W'=28) and action_chunk has shape (B=32, chunk_size=8, action_dim=7).
    Then, this function will overwrite the (C'=16, H'=28, W'=28) volume at x0[:,:,action_indices,:,:] with the action chunk,
    repeating it as many times as needed to fill the entire volume.

    Args:
        x0 (torch.Tensor): Clean image latents.
        action_chunk (torch.Tensor): Ground-truth action chunk.
        action_indices (torch.Tensor): Batch indices of the image latents to replace.

    Returns:
        torch.Tensor: Modified image latents.
    """
    # Get latent to be replaced
    batch_indices = torch.arange(x0.shape[0], device=x0.device)
    action_image_latent = x0[batch_indices, :, action_indices, :, :]

    # Create a new tensor with the same shape as action_image_latent, filled with zeros
    result = torch.zeros_like(action_image_latent)

    # Get shapes
    batch_size, latent_channels, latent_h, latent_w = action_image_latent.shape

    # Flatten action_chunk (preserving batch dimension)
    flat_action = action_chunk.reshape(batch_size, -1)
    num_action_elements = flat_action.shape[1]

    # Calculate total elements in the target tensor (per batch)
    latent_elements = latent_channels * latent_h * latent_w

    # Check that there is enough room in the target tensor for all the action elements
    assert num_action_elements <= latent_elements, (
        f"Not enough room in the latent tensor for the full action chunk: {num_action_elements} action elements > {latent_elements} latent elements!"
    )

    # Calculate how many times we need to repeat the action tensor
    # The expression below is a concise way of doing ceiling division to get the correct number of repeats
    num_repeats = (latent_elements + num_action_elements - 1) // num_action_elements

    # Repeat the action tensor along dimension 1
    repeated_action = flat_action.repeat(1, num_repeats)

    # Take only what we need to fill the result tensor
    repeated_action = repeated_action[:, :latent_elements]

    # Reshape the target tensor to put all channel and spatial dimensions together
    flat_result = result.reshape(batch_size, -1)

    # Place the action chunk values into the beginning of the flattened result
    flat_result[:, :] = repeated_action

    # Reshape back to original shape
    result = flat_result.reshape(batch_size, latent_channels, latent_h, latent_w)

    # Get final latents tensor
    new_x0 = x0
    new_x0[batch_indices, :, action_indices, :, :] = result

    return new_x0


def replace_latent_with_proprio(x0: torch.Tensor, proprio: torch.Tensor, proprio_indices: torch.Tensor) -> torch.Tensor:
    """
    Replaces the image latent (at the specified proprio index) in clean input image latents x0 with the proprio.

    Example:
    Let's say x0 has shape (B=32, C'=16, T', H'=28, W'=28) and proprio has shape (B=32, proprio_dim=9).
    Then, this function will overwrite the (C'=16, H'=28, W'=28) volume at x0[:,:,proprio_indices,:,:] with the proprio,
    repeating it as many times as needed to fill the entire volume.

    Args:
        x0 (torch.Tensor): Clean image latents.
        proprio (torch.Tensor): Ground-truth proprio.
        proprio_indices (torch.Tensor): Batch indices of the image latents to replace.

    Returns:
        torch.Tensor: Modified image latents.
    """
    # Get latent to be replaced
    batch_indices = torch.arange(x0.shape[0], device=x0.device)
    proprio_image_latent = x0[batch_indices, :, proprio_indices, :, :]

    # Create a new tensor with the same shape as proprio_image_latent, filled with zeros
    result = torch.zeros_like(proprio_image_latent)

    # Get shapes
    batch_size, latent_channels, latent_h, latent_w = proprio_image_latent.shape

    # Get number of proprio elements
    num_proprio_elements = proprio.shape[1]

    # Calculate total elements in the target tensor (per batch)
    latent_elements = latent_channels * latent_h * latent_w

    # Check that there is enough room in the target tensor for all the proprio elements
    assert num_proprio_elements <= latent_elements, (
        f"Not enough room in the latent tensor for the full proprio: {num_proprio_elements} proprio elements > {latent_elements} latent elements!"
    )

    # Calculate how many times we need to repeat the proprio tensor
    # The expression below is a concise way of doing ceiling division to get the correct number of repeats
    num_repeats = (latent_elements + num_proprio_elements - 1) // num_proprio_elements

    # Repeat the proprio tensor along dimension 1
    repeated_proprio = proprio.repeat(1, num_repeats)

    # Take only what we need to fill the result tensor
    repeated_proprio = repeated_proprio[:, :latent_elements]

    # Reshape the target tensor to put all channel and spatial dimensions together
    flat_result = result.reshape(batch_size, -1)

    # Place the proprio values into the beginning of the flattened result
    flat_result[:, :] = repeated_proprio

    # Reshape latent back to original shape
    result = flat_result.reshape(batch_size, latent_channels, latent_h, latent_w)

    # Get final latents tensor
    new_x0 = x0
    new_x0[batch_indices, :, proprio_indices, :, :] = result

    return new_x0


def replace_latent_with_meta_regions(
    x0: torch.Tensor,
    meta_indices: torch.Tensor,
    current_proprio_region_vector: Optional[torch.Tensor] = None,
    action_region_vector: Optional[torch.Tensor] = None,
    future_proprio_region_vector: Optional[torch.Tensor] = None,
    value_region_vector: Optional[torch.Tensor] = None,
    task_type_region_vector: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Replace one latent frame (at the specified meta index) in x0 with a MetaWAM frequency meta frame:
    channel-partitioned spectral slots projected back to the spatial latent by orthonormal IDCT.
    """
    batch_indices = torch.arange(x0.shape[0], device=x0.device)
    meta_latent = x0[batch_indices, :, meta_indices, :, :]
    result = _encode_frequency_meta_frame_v2(
        meta_latent=meta_latent,
        current_proprio_region_vector=current_proprio_region_vector,
        action_region_vector=action_region_vector,
        future_proprio_region_vector=future_proprio_region_vector,
        value_region_vector=value_region_vector,
        task_type_region_vector=task_type_region_vector,
    )
    new_x0 = x0
    new_x0[batch_indices, :, meta_indices, :, :] = result
    return new_x0


@attrs.define(slots=False)
class CosmosPolicyModelConfig(BaseText2WorldModelConfig):
    """
    Extended config for Cosmos Policy diffusion model.
    Uses Cosmos Policy's HybridEDMSDE instead of the original EDMSDE.
    Also adds policy-specific parameters for loss masking and action prediction.
    """

    sde: LazyDict = L(HybridEDMSDE)(
        # Note: Most of these values get overridden later in the experiment configs
        p_mean=0.0,
        p_std=1.0,
        sigma_max=80,
        sigma_min=0.0002,
        hybrid_sigma_distribution=True,
        uniform_lower=1.0,
        uniform_upper=85.0,
    )

    # Whether to use loss masking to separate action, future state, and value prediction
    # - Policy prediction: only take loss on action predictions
    # - World model prediction: only take loss on future state predictions
    # - Value function prediction: only take loss on value predictions
    mask_loss_for_action_future_state_prediction: bool = False
    # Whether to use loss masking on value prediction during policy predictions (so we only take loss on action + future state predictions)
    mask_value_prediction_loss_for_policy_prediction: bool = False
    # Whether to mask out some inputs (current state and action) during future state value prediction
    mask_current_state_action_for_value_prediction: bool = False
    # Whether to mask out some inputs (future state) during Q(s,a) prediction
    mask_future_state_for_qvalue_prediction: bool = False
    # MetaWAM: collapse action / future proprio / value prediction onto a single frequency meta latent frame,
    # and reuse latent slot 0 (the tokenizer-required blank frame) as an input meta frame carrying task type +
    # current proprio. When False, the model behaves like the original Cosmos Policy (dedicated latent frames).
    use_single_meta_frame: bool = False
    # Per-frame latent loss weights for image modalities.
    # These multipliers directly scale the corresponding latent-frame loss contribution.
    future_image_loss_weight: float = 1.0
    future_image2_loss_weight: float = 1.0
    future_wrist_image_loss_weight: float = 1.0
    future_wrist_image2_loss_weight: float = 1.0
    # MetaWAM auxiliary spectral losses on the meta frame (only used when use_single_meta_frame=True):
    # - active-slot Huber consistency loss in DCT domain (per-slot weights below)
    # - off-slot sparsity penalty in DCT domain
    frequency_active_slot_huber_delta: float = 1.0
    frequency_action_slot_loss_weight: float = 1.0
    frequency_future_proprio_slot_loss_weight: float = 1.0
    frequency_value_slot_loss_weight: float = 1.0
    frequency_offslot_sparsity_weight: float = 0.0

    # Action loss multiplier (if greater than 1, upweights loss on predicting actions relative to other losses)
    # (Must be an integer - or will be cast to an integer later!)
    action_loss_multiplier: int = 1

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert not (
            self.mask_loss_for_action_future_state_prediction and self.mask_value_prediction_loss_for_policy_prediction
        ), (
            "Cannot enable both mask_loss_for_action_future_state_prediction and mask_value_prediction_loss_for_policy_prediction!"
        )
        assert self.future_image_loss_weight > 0, (
            "future_image_loss_weight must be > 0, "
            f"got {self.future_image_loss_weight}"
        )
        assert self.future_image2_loss_weight > 0, (
            "future_image2_loss_weight must be > 0, "
            f"got {self.future_image2_loss_weight}"
        )
        assert self.future_wrist_image_loss_weight > 0, (
            "future_wrist_image_loss_weight must be > 0, "
            f"got {self.future_wrist_image_loss_weight}"
        )
        assert self.future_wrist_image2_loss_weight > 0, (
            "future_wrist_image2_loss_weight must be > 0, "
            f"got {self.future_wrist_image2_loss_weight}"
        )
        assert self.frequency_active_slot_huber_delta > 0, (
            "frequency_active_slot_huber_delta must be > 0, "
            f"got {self.frequency_active_slot_huber_delta}"
        )
        assert self.frequency_action_slot_loss_weight >= 0, (
            "frequency_action_slot_loss_weight must be >= 0, "
            f"got {self.frequency_action_slot_loss_weight}"
        )
        assert self.frequency_future_proprio_slot_loss_weight >= 0, (
            "frequency_future_proprio_slot_loss_weight must be >= 0, "
            f"got {self.frequency_future_proprio_slot_loss_weight}"
        )
        assert self.frequency_value_slot_loss_weight >= 0, (
            "frequency_value_slot_loss_weight must be >= 0, "
            f"got {self.frequency_value_slot_loss_weight}"
        )
        assert self.frequency_offslot_sparsity_weight >= 0, (
            "frequency_offslot_sparsity_weight must be >= 0, "
            f"got {self.frequency_offslot_sparsity_weight}"
        )


class CosmosPolicyDiffusionModel(BaseDiffusionModel):
    """
    Cosmos Policy Diffusion Model - extends Text2WorldModel with policy-specific functionality.

    Adds support for:
    - Action chunk prediction and injection
    - Proprioception (proprio) prediction and injection
    - Value function prediction
    - Loss masking for different prediction types (action, future state, value)
    - Multi-component loss tracking
    """

    def __init__(self, config: CosmosPolicyModelConfig):
        super().__init__(config)
        self.config: CosmosPolicyModelConfig = config

        # Cosmos Policy SDE and Sampler
        self.sde = lazy_instantiate(config.sde)
        self.sampler = CosmosPolicySampler()

    def _remap_meta_indices(
        self,
        action_indices: torch.Tensor,
        future_proprio_indices: torch.Tensor,
        value_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.use_single_meta_frame:
            meta_indices = action_indices
            return meta_indices, meta_indices, meta_indices
        return action_indices, future_proprio_indices, value_indices

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the Cosmos Policy diffusion model.

        Extended from base to pass policy-specific data (actions, proprio, values, masks).

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.
        """
        self._update_train_stats(data_batch)

        # Obtain text embeddings online
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(x0_B_C_T_H_W.size(), condition)

        # Broadcast and split the input data and condition for model parallelism
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )
        output_batch, kendall_loss, _, _ = self.compute_loss_with_epsilon_and_sigma(
            x0_B_C_T_H_W,
            condition,
            epsilon_B_C_T_H_W,
            sigma_B_T,
            action_chunk=data_batch["actions"],
            action_indices=data_batch["action_latent_idx"],
            proprio=data_batch["proprio"],
            current_proprio_indices=data_batch["current_proprio_latent_idx"],
            future_proprio=data_batch["future_proprio"],
            future_proprio_indices=data_batch["future_proprio_latent_idx"],
            future_wrist_image_indices=data_batch["future_wrist_image_latent_idx"],
            future_wrist_image2_indices=(
                data_batch["future_wrist_image2_latent_idx"] if "future_wrist_image2_latent_idx" in data_batch else None
            ),
            future_image_indices=data_batch["future_image_latent_idx"],
            future_image2_indices=(
                data_batch["future_image2_latent_idx"] if "future_image2_latent_idx" in data_batch else None
            ),
            rollout_data_mask=data_batch["rollout_data_mask"],
            world_model_sample_mask=data_batch["world_model_sample_mask"],
            value_function_sample_mask=data_batch["value_function_sample_mask"],
            value_function_return=data_batch["value_function_return"],
            value_indices=data_batch["value_latent_idx"],
        )

        frequency_active_slot_huber_loss = output_batch.get("frequency_active_slot_huber_loss")
        frequency_offslot_sparsity_loss = output_batch.get("frequency_offslot_sparsity_loss")
        if frequency_active_slot_huber_loss is None:
            frequency_active_slot_huber_loss = torch.tensor(0.0, device=kendall_loss.device, dtype=kendall_loss.dtype)
        if frequency_offslot_sparsity_loss is None:
            frequency_offslot_sparsity_loss = torch.tensor(0.0, device=kendall_loss.device, dtype=kendall_loss.dtype)
        frequency_aux_loss = frequency_active_slot_huber_loss.to(kendall_loss.dtype) + frequency_offslot_sparsity_loss.to(
            kendall_loss.dtype
        )

        if self.loss_reduce == "mean":
            kendall_loss = (kendall_loss.mean() + frequency_aux_loss) * self.loss_scale
        elif self.loss_reduce == "sum":
            kendall_loss = (kendall_loss.sum(dim=1).mean() + frequency_aux_loss) * self.loss_scale
        else:
            raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")

        output_batch["frequency_aux_loss"] = frequency_aux_loss.detach()
        return output_batch, kendall_loss

    def compute_loss_with_epsilon_and_sigma(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: Text2WorldCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
        action_chunk: torch.Tensor,
        action_indices: torch.Tensor,
        proprio: torch.Tensor,
        current_proprio_indices: torch.Tensor,
        future_proprio: torch.Tensor,
        future_proprio_indices: torch.Tensor,
        future_wrist_image_indices: torch.Tensor,
        future_wrist_image2_indices: Optional[torch.Tensor],
        future_image_indices: torch.Tensor,
        future_image2_indices: Optional[torch.Tensor],
        rollout_data_mask: torch.Tensor,
        world_model_sample_mask: torch.Tensor,
        value_function_sample_mask: torch.Tensor,
        value_function_return: torch.Tensor,
        value_indices: torch.Tensor,
    ):
        """
        NOTE (user): Modified to add action chunk prediction and future image prediction + action chunk loss logging.

        Compute loss given epsilon and sigma with policy-specific functionality.

        This method extends the base implementation to support:
        1. Latent injection for actions, proprioception, and values
        2. Loss masking for different prediction types (action, future state, value)
        3. Detailed per-component loss tracking

        Args:
            x0_B_C_T_H_W: image/video latent
            condition: text condition
            epsilon_B_C_T_H_W: noise
            sigma_B_T: noise level
            action_chunk: ground truth action chunk
            action_indices: indices for action latent frames
            proprio: current proprioception
            current_proprio_indices: indices for current proprio latent frames
            future_proprio: future proprioception
            future_proprio_indices: indices for future proprio latent frames
            future_wrist_image_indices: indices for future wrist image latent frames
            future_wrist_image2_indices: indices for future wrist image #2 latent frames
            future_image_indices: indices for future primary image latent frames
            future_image2_indices: indices for future secondary image latent frames
            rollout_data_mask: mask for rollout vs demo data
            world_model_sample_mask: mask for world model samples
            value_function_sample_mask: mask for value function samples
            value_function_return: ground truth value function returns
            value_indices: indices for value latent frames

        Returns:
            tuple: A tuple containing four elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor 1: kendall loss,
                - Tensor 2: MSE loss,
                - Tensor 3: EDM loss
        """
        original_future_proprio_indices = future_proprio_indices
        action_indices, future_proprio_indices, value_indices = self._remap_meta_indices(
            action_indices, future_proprio_indices, value_indices
        )
        # NOTE (user): Action chunk, proprio, value injection
        # x0_B_C_T_H_W: (B, C', T', H', W')
        # x0_B_C_T_H_W is the VAE-encoded latent representing several input images (which may be in a different order than below):
        # - conditional frames (e.g., proprio, wrist camera image, primary image, optional history)
        # - future image(s) and proprio state to predict
        # - action chunk to predict (blank image)
        # - value function return (rewards-to-go) to predict (blank image)
        condition.orig_x0_B_C_T_H_W = x0_B_C_T_H_W.clone()  # Keep a backup of the original gt_frames
        batch_indices = torch.arange(x0_B_C_T_H_W.shape[0], device=x0_B_C_T_H_W.device)
        C_latent, H_latent, W_latent = x0_B_C_T_H_W.shape[1], x0_B_C_T_H_W.shape[3], x0_B_C_T_H_W.shape[4]
        if self.config.use_single_meta_frame:
            demo_sample_mask_B = rollout_data_mask == 0
            world_sample_mask_B = (rollout_data_mask == 1) & (world_model_sample_mask == 1)
            value_sample_mask_B = (rollout_data_mask == 1) & (value_function_sample_mask == 1)
            world_or_value_mask_B = world_sample_mask_B | value_sample_mask_B
            # One-hot task type: [policy/demo, world-model, value-model]
            task_type_region_vector = torch.zeros((x0_B_C_T_H_W.shape[0], 3), device=x0_B_C_T_H_W.device).to(
                x0_B_C_T_H_W.dtype
            )
            task_type_region_vector[demo_sample_mask_B, 0] = 1.0
            task_type_region_vector[world_sample_mask_B, 1] = 1.0
            task_type_region_vector[value_sample_mask_B, 2] = 1.0

            # Input meta frame (latent slot 0): task type + current proprio, plus the inputs that are visible as
            # conditioning for the given task (action for world-model / value samples, future proprio for value
            # samples).
            input_current_proprio_region_vector = proprio.reshape(proprio.shape[0], -1).to(x0_B_C_T_H_W.dtype)
            if self.config.mask_current_state_action_for_value_prediction:
                current_visible_B = (~value_sample_mask_B) | world_sample_mask_B
                action_visible_B = world_sample_mask_B
            else:
                current_visible_B = torch.ones_like(demo_sample_mask_B)
                action_visible_B = world_or_value_mask_B
            input_current_proprio_region_vector = (
                input_current_proprio_region_vector * current_visible_B.to(x0_B_C_T_H_W.dtype).unsqueeze(1)
            )
            input_action_region_vector = action_chunk.reshape(action_chunk.shape[0], -1).to(x0_B_C_T_H_W.dtype)
            input_action_region_vector = (
                input_action_region_vector * action_visible_B.to(x0_B_C_T_H_W.dtype).unsqueeze(1)
            )
            input_future_proprio_region_vector = None
            has_future_proprio = torch.all(original_future_proprio_indices != -1)  # -1 indicates future proprio is not used
            if has_future_proprio:
                input_future_proprio_region_vector = future_proprio.reshape(future_proprio.shape[0], -1).to(
                    x0_B_C_T_H_W.dtype
                )
                if self.config.mask_future_state_for_qvalue_prediction:
                    future_visible_B = torch.zeros_like(value_sample_mask_B)
                else:
                    future_visible_B = value_sample_mask_B
                input_future_proprio_region_vector = (
                    input_future_proprio_region_vector * future_visible_B.to(x0_B_C_T_H_W.dtype).unsqueeze(1)
                )
            input_meta_indices = torch.zeros((x0_B_C_T_H_W.shape[0],), dtype=torch.long, device=x0_B_C_T_H_W.device)
            x0_B_C_T_H_W = replace_latent_with_meta_regions(
                x0_B_C_T_H_W,
                meta_indices=input_meta_indices,
                current_proprio_region_vector=input_current_proprio_region_vector,
                action_region_vector=input_action_region_vector,
                future_proprio_region_vector=input_future_proprio_region_vector,
                value_region_vector=None,
                task_type_region_vector=task_type_region_vector,
            )

            # Output meta frame (at the action index): action chunk, future proprio and value targets, plus the
            # task type and current proprio.
            action_region_vector = action_chunk.reshape(action_chunk.shape[0], -1).to(x0_B_C_T_H_W.dtype)
            future_proprio_region_vector = None
            if has_future_proprio:
                future_proprio_region_vector = future_proprio.reshape(future_proprio.shape[0], -1).to(
                    x0_B_C_T_H_W.dtype
                )
            value_region_vector = value_function_return.reshape(-1, 1).to(x0_B_C_T_H_W.dtype)
            current_proprio_region_vector = proprio.reshape(proprio.shape[0], -1).to(x0_B_C_T_H_W.dtype)
            x0_B_C_T_H_W = replace_latent_with_meta_regions(
                x0_B_C_T_H_W,
                meta_indices=action_indices,
                current_proprio_region_vector=current_proprio_region_vector,
                action_region_vector=action_region_vector,
                future_proprio_region_vector=future_proprio_region_vector,
                value_region_vector=value_region_vector,
                task_type_region_vector=task_type_region_vector,
            )
        else:
            # Proprio
            if torch.all(current_proprio_indices != -1):  # -1 indicates proprio is not used
                x0_B_C_T_H_W = replace_latent_with_proprio(
                    x0_B_C_T_H_W,
                    proprio,
                    proprio_indices=current_proprio_indices,
                )
            # Action
            x0_B_C_T_H_W = replace_latent_with_action_chunk(x0_B_C_T_H_W, action_chunk, action_indices=action_indices)
            # Future proprio
            if torch.all(future_proprio_indices != -1):  # -1 indicates future proprio is not used
                x0_B_C_T_H_W = replace_latent_with_proprio(
                    x0_B_C_T_H_W,
                    future_proprio,
                    proprio_indices=future_proprio_indices,
                )
            # Value
            x0_B_C_T_H_W[batch_indices, :, value_indices, :, :] = (
                value_function_return.reshape(-1, 1, 1, 1)
                .expand(-1, C_latent, H_latent, W_latent)
                .to(x0_B_C_T_H_W.dtype)
            )

        # Get the mean and stand deviation of the marginal probability distribution.
        mean_B_C_T_H_W, std_B_T = self.sde.marginal_prob(x0_B_C_T_H_W, sigma_B_T)
        # Generate noisy observations
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(std_B_T, "b t -> b 1 t 1 1")
        # make prediction
        model_pred = self.denoise(xt_B_C_T_H_W, sigma_B_T, condition)
        # loss weights for different noise levels
        weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)

        # Construct mask to support masking out loss (or scaling it by some multiplier) for different types of predictions
        B, T = x0_B_C_T_H_W.shape[0], x0_B_C_T_H_W.shape[2]
        final_mask_B_T = torch.ones(
            (B, T), dtype=x0_B_C_T_H_W.dtype, device=sigma_B_T.device
        )  # All 1s mask initially

        def _scale_mask_at_indices(frame_indices_B: Optional[torch.Tensor], multiplier: float) -> None:
            if frame_indices_B is None:
                return
            valid_B = frame_indices_B != -1
            if not torch.any(valid_B):
                return
            valid_batch_indices = torch.nonzero(valid_B, as_tuple=False).squeeze(-1).to(torch.long).to(sigma_B_T.device)
            valid_frame_indices = frame_indices_B[valid_batch_indices].to(torch.long).to(sigma_B_T.device)
            final_mask_B_T[valid_batch_indices, valid_frame_indices] = (
                final_mask_B_T[valid_batch_indices, valid_frame_indices] * float(multiplier)
            )

        meta_channel_loss_scale_B_C = None
        if self.config.use_single_meta_frame:
            # Channel-level supervision mask for the meta frame.
            demo_sample_mask_B = (rollout_data_mask == 0).to(device=x0_B_C_T_H_W.device).bool()
            world_sample_mask_B = (
                ((rollout_data_mask == 1) & (world_model_sample_mask == 1)).to(device=x0_B_C_T_H_W.device).bool()
            )
            value_sample_mask_B = (
                ((rollout_data_mask == 1) & (value_function_sample_mask == 1)).to(device=x0_B_C_T_H_W.device).bool()
            )
            demo_or_world_mask_B = demo_sample_mask_B | world_sample_mask_B
            any_sample_mask_B = demo_sample_mask_B | world_sample_mask_B | value_sample_mask_B

            per_sample_meta_channel_mask_B_C = torch.zeros(
                (B, C_latent), dtype=x0_B_C_T_H_W.dtype, device=x0_B_C_T_H_W.device
            )
            # task/current slots: pure condition, never supervised.
            per_sample_meta_channel_mask_B_C[:, _FREQUENCY_V2_ACTION_SLICE] = demo_sample_mask_B.to(
                x0_B_C_T_H_W.dtype
            ).reshape(-1, 1)
            if torch.all(original_future_proprio_indices != -1):
                per_sample_meta_channel_mask_B_C[:, _FREQUENCY_V2_FUTURE_PROPRIO_SLICE] = demo_or_world_mask_B.to(
                    x0_B_C_T_H_W.dtype
                ).reshape(-1, 1)
            per_sample_meta_channel_mask_B_C[:, _FREQUENCY_V2_VALUE_SLICE] = any_sample_mask_B.to(
                x0_B_C_T_H_W.dtype
            ).reshape(-1, 1)

            meta_channel_loss_scale_B_C = per_sample_meta_channel_mask_B_C

        # If using input masking for value prediction, mask out the loss for everything except the value prediction
        # This is necessary since otherwise the loss will be computed for all latent frames, not just the value prediction frame
        if (
            self.config.mask_current_state_action_for_value_prediction
            or self.config.mask_future_state_for_qvalue_prediction
        ):
            mask_B_T = torch.ones((B, T), dtype=x0_B_C_T_H_W.dtype, device=sigma_B_T.device)  # All 1s mask
            # Rollout value-function samples (rollout_data_mask == 1 and value_function_sample_mask == 1)
            value_idx_B = (
                ((rollout_data_mask == 1) & (value_function_sample_mask == 1)).to(torch.long).to(sigma_B_T.device)
            )
            if torch.any(value_idx_B):
                value_batch_indices = (
                    torch.nonzero(value_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(sigma_B_T.device)
                )
                # First set the mask to 0 for everything except the value prediction, but only do this for value prediction samples
                mask_B_T[value_batch_indices, :] = 0
                # Then set the mask to 1 for the value prediction
                mask_B_T[value_batch_indices, value_indices[value_batch_indices]] = 1
            final_mask_B_T = final_mask_B_T * mask_B_T

        # Build per-sample mask to select which frames contribute to loss
        # - Demo samples: action + future proprio + value prediction
        # - Rollout world-model samples: future state (proprio, wrist image, primary image) + value prediction
        # - Rollout value-function samples: only value prediction
        if self.config.mask_loss_for_action_future_state_prediction:
            B, T = x0_B_C_T_H_W.shape[0], x0_B_C_T_H_W.shape[2]
            mask_B_T = torch.zeros(
                (B, T), dtype=x0_B_C_T_H_W.dtype, device=sigma_B_T.device
            )  # All 0s mask, to be filled with 1s for the relevant timesteps
            # Demo samples (rollout_data_mask == 0)
            demo_idx_B = (rollout_data_mask == 0).to(torch.long).to(sigma_B_T.device)
            if torch.any(demo_idx_B):
                demo_batch_indices = (
                    torch.nonzero(demo_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(sigma_B_T.device)
                )
                mask_B_T[demo_batch_indices, action_indices[demo_batch_indices]] = 1
                if torch.all(future_proprio_indices != -1):  # -1 indicates future proprio is not used
                    mask_B_T[demo_batch_indices, future_proprio_indices[demo_batch_indices]] = 1
                mask_B_T[demo_batch_indices, value_indices[demo_batch_indices]] = 1
            # Rollout world-model samples (rollout_data_mask == 1 and world_model_sample_mask == 1)
            world_idx_B = (rollout_data_mask == 1) & (world_model_sample_mask == 1).to(torch.long).to(sigma_B_T.device)
            if torch.any(world_idx_B):
                world_batch_indices = (
                    torch.nonzero(world_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(sigma_B_T.device)
                )
                if torch.all(future_image_indices != -1):  # -1 indicates future image is not used
                    mask_B_T[world_batch_indices, future_image_indices[world_batch_indices]] = 1
                if future_image2_indices is not None and torch.all(
                    future_image2_indices != -1
                ):  # -1 indicates secondary image is not used
                    mask_B_T[world_batch_indices, future_image2_indices[world_batch_indices]] = 1
                if torch.all(future_wrist_image_indices != -1):  # -1 indicates future wrist image is not used
                    mask_B_T[world_batch_indices, future_wrist_image_indices[world_batch_indices]] = 1
                if future_wrist_image2_indices is not None and torch.all(
                    future_wrist_image2_indices != -1
                ):  # -1 indicates future wrist image #2 is not used
                    mask_B_T[world_batch_indices, future_wrist_image2_indices[world_batch_indices]] = 1
                if torch.all(future_proprio_indices != -1):  # -1 indicates future proprio is not used
                    mask_B_T[world_batch_indices, future_proprio_indices[world_batch_indices]] = 1
                mask_B_T[world_batch_indices, value_indices[world_batch_indices]] = 1
            # Rollout value-function samples (rollout_data_mask == 1 and value_function_sample_mask == 1)
            value_idx_B = (
                ((rollout_data_mask == 1) & (value_function_sample_mask == 1)).to(torch.long).to(sigma_B_T.device)
            )
            if torch.any(value_idx_B):
                value_batch_indices = (
                    torch.nonzero(value_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(sigma_B_T.device)
                )
                mask_B_T[value_batch_indices, value_indices[value_batch_indices]] = 1
            final_mask_B_T = final_mask_B_T * mask_B_T

        # Build per-sample mask to select which frames contribute to loss
        # - Demo samples: only action prediction + future state prediction
        # - Rollout world-model samples: only future state (proprio, wrist image, primary image)
        # - Rollout value-function samples: N/A (assert that we don't encounter any value function samples here)
        if self.config.mask_value_prediction_loss_for_policy_prediction:
            assert value_function_sample_mask.sum() == 0, (
                "No value function samples should be present when mask_value_prediction_loss_for_policy_prediction==True!"
            )
            B, T = x0_B_C_T_H_W.shape[0], x0_B_C_T_H_W.shape[2]
            mask_B_T = torch.zeros(
                (B, T), dtype=x0_B_C_T_H_W.dtype, device=sigma_B_T.device
            )  # All 0s mask, to be filled with 1s for the relevant timesteps
            # Demo samples (rollout_data_mask == 0)
            demo_idx_B = (rollout_data_mask == 0).to(torch.long).to(sigma_B_T.device)
            if torch.any(demo_idx_B):
                demo_batch_indices = (
                    torch.nonzero(demo_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(sigma_B_T.device)
                )
                mask_B_T[demo_batch_indices, action_indices[demo_batch_indices]] = 1
                if torch.all(future_image_indices != -1):  # -1 indicates future image is not used
                    mask_B_T[demo_batch_indices, future_image_indices[demo_batch_indices]] = 1
                if future_image2_indices is not None and torch.all(
                    future_image2_indices != -1
                ):  # -1 indicates secondary image is not used
                    mask_B_T[demo_batch_indices, future_image2_indices[demo_batch_indices]] = 1
                if torch.all(future_wrist_image_indices != -1):  # -1 indicates future wrist image is not used
                    mask_B_T[demo_batch_indices, future_wrist_image_indices[demo_batch_indices]] = 1
                if future_wrist_image2_indices is not None and torch.all(
                    future_wrist_image2_indices != -1
                ):  # -1 indicates future wrist image #2 is not used
                    mask_B_T[demo_batch_indices, future_wrist_image2_indices[demo_batch_indices]] = 1
                if torch.all(future_proprio_indices != -1):  # -1 indicates future proprio is not used
                    mask_B_T[demo_batch_indices, future_proprio_indices[demo_batch_indices]] = 1
            # Rollout world-model samples (rollout_data_mask == 1 and world_model_sample_mask == 1)
            world_idx_B = (rollout_data_mask == 1) & (world_model_sample_mask == 1).to(torch.long).to(sigma_B_T.device)
            if torch.any(world_idx_B):
                world_batch_indices = (
                    torch.nonzero(world_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(sigma_B_T.device)
                )
                if torch.all(future_image_indices != -1):  # -1 indicates future image is not used
                    mask_B_T[world_batch_indices, future_image_indices[world_batch_indices]] = 1
                if future_image2_indices is not None and torch.all(
                    future_image2_indices != -1
                ):  # -1 indicates secondary image is not used
                    mask_B_T[world_batch_indices, future_image2_indices[world_batch_indices]] = 1
                if torch.all(future_wrist_image_indices != -1):  # -1 indicates future wrist image is not used
                    mask_B_T[world_batch_indices, future_wrist_image_indices[world_batch_indices]] = 1
                if future_wrist_image2_indices is not None and torch.all(
                    future_wrist_image2_indices != -1
                ):  # -1 indicates future wrist image #2 is not used
                    mask_B_T[world_batch_indices, future_wrist_image2_indices[world_batch_indices]] = 1
                if torch.all(future_proprio_indices != -1):  # -1 indicates future proprio is not used
                    mask_B_T[world_batch_indices, future_proprio_indices[world_batch_indices]] = 1
            final_mask_B_T = final_mask_B_T * mask_B_T

        # If applicable, upweight the loss on the action predictions by a factor of `action_loss_multiplier`
        if self.config.action_loss_multiplier != 1:
            # Only upweight the loss on the action indices
            final_mask_B_T[batch_indices, action_indices] = final_mask_B_T[batch_indices, action_indices] * int(
                self.config.action_loss_multiplier
            )

        # Per-frame latent loss weighting for image modalities.
        if self.config.future_image_loss_weight != 1.0:
            _scale_mask_at_indices(future_image_indices, float(self.config.future_image_loss_weight))
        if self.config.future_image2_loss_weight != 1.0:
            _scale_mask_at_indices(future_image2_indices, float(self.config.future_image2_loss_weight))
        if self.config.future_wrist_image_loss_weight != 1.0:
            _scale_mask_at_indices(future_wrist_image_indices, float(self.config.future_wrist_image_loss_weight))
        if self.config.future_wrist_image2_loss_weight != 1.0:
            _scale_mask_at_indices(future_wrist_image2_indices, float(self.config.future_wrist_image2_loss_weight))

        # extra loss mask for each sample, for example, human faces, hands
        pred_mse_B_C_T_H_W = (x0_B_C_T_H_W - model_pred.x0) ** 2
        edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(weights_per_sigma_B_T, "b t -> b 1 t 1 1")

        kendall_loss = edm_loss_B_C_T_H_W

        # Apply the loss mask to the loss
        if (
            self.config.mask_loss_for_action_future_state_prediction
            or self.config.mask_current_state_action_for_value_prediction
            or self.config.mask_future_state_for_qvalue_prediction
            or self.config.action_loss_multiplier != 1
            or self.config.use_single_meta_frame
        ):
            kendall_loss = kendall_loss * rearrange(final_mask_B_T, "b t -> b 1 t 1 1")
            if meta_channel_loss_scale_B_C is not None:
                kendall_loss[batch_indices, :, action_indices, :, :] = (
                    kendall_loss[batch_indices, :, action_indices, :, :]
                    * meta_channel_loss_scale_B_C[:, :, None, None]
                )

        # MetaWAM auxiliary spectral losses on the meta frame (zero in upstream mode).
        frequency_action_slot_huber_loss = torch.tensor(0.0, device=x0_B_C_T_H_W.device, dtype=x0_B_C_T_H_W.dtype)
        frequency_future_proprio_slot_huber_loss = torch.tensor(0.0, device=x0_B_C_T_H_W.device, dtype=x0_B_C_T_H_W.dtype)
        frequency_value_slot_huber_loss = torch.tensor(0.0, device=x0_B_C_T_H_W.device, dtype=x0_B_C_T_H_W.dtype)
        frequency_active_slot_huber_loss = torch.tensor(0.0, device=x0_B_C_T_H_W.device, dtype=x0_B_C_T_H_W.dtype)
        frequency_offslot_sparsity_loss = torch.tensor(0.0, device=x0_B_C_T_H_W.device, dtype=x0_B_C_T_H_W.dtype)
        if self.config.use_single_meta_frame:
            batch_indices = torch.arange(x0_B_C_T_H_W.shape[0], device=x0_B_C_T_H_W.device)
            meta_gt_B_C_H_W = x0_B_C_T_H_W[batch_indices, :, action_indices, :, :]
            meta_pred_B_C_H_W = model_pred.x0[batch_indices, :, action_indices, :, :]

            # The GT meta frame only has row-0 coefficients by construction.
            coeff_gt_row0_B_C_W = _dct2_row0_coeff_per_channel_ortho(meta_gt_B_C_H_W)
            coeff_pred_B_C_H_W = _dct2_per_channel_ortho(meta_pred_B_C_H_W)

            demo_sample_mask_B = (rollout_data_mask == 0).to(device=x0_B_C_T_H_W.device).bool()
            world_sample_mask_B = (
                ((rollout_data_mask == 1) & (world_model_sample_mask == 1)).to(device=x0_B_C_T_H_W.device).bool()
            )
            value_sample_mask_B = (
                ((rollout_data_mask == 1) & (value_function_sample_mask == 1)).to(device=x0_B_C_T_H_W.device).bool()
            )
            demo_or_world_mask_B = demo_sample_mask_B | world_sample_mask_B
            any_sample_mask_B = demo_sample_mask_B | world_sample_mask_B | value_sample_mask_B

            def _masked_average(loss_per_sample_B: torch.Tensor, sample_mask_B: torch.Tensor) -> torch.Tensor:
                sample_mask_f = sample_mask_B.to(device=loss_per_sample_B.device, dtype=loss_per_sample_B.dtype)
                denom = torch.clamp(sample_mask_f.sum(), min=1.0)
                return (loss_per_sample_B * sample_mask_f).sum() / denom

            huber_delta = float(self.config.frequency_active_slot_huber_delta)
            has_future_proprio = torch.all(original_future_proprio_indices != -1)

            # Action slots (demo samples).
            action_gt_flat = coeff_gt_row0_B_C_W[:, _FREQUENCY_V2_ACTION_SLICE, :].reshape(
                coeff_gt_row0_B_C_W.shape[0], -1
            )
            action_pred_flat = coeff_pred_B_C_H_W[:, _FREQUENCY_V2_ACTION_SLICE, 0, :].reshape(
                coeff_pred_B_C_H_W.shape[0], -1
            )
            action_num_elements = action_chunk.reshape(action_chunk.shape[0], -1).shape[1]
            k_action = min(action_num_elements, action_gt_flat.shape[1], action_pred_flat.shape[1])
            if k_action > 0:
                action_huber_B_K = F.huber_loss(
                    action_pred_flat[:, :k_action],
                    action_gt_flat[:, :k_action],
                    reduction="none",
                    delta=huber_delta,
                )
                action_loss_B = action_huber_B_K.mean(dim=1)
                frequency_action_slot_huber_loss = _masked_average(action_loss_B, demo_sample_mask_B)

            # Future proprio slots (demo + world-model samples).
            if has_future_proprio:
                future_gt_flat = coeff_gt_row0_B_C_W[:, _FREQUENCY_V2_FUTURE_PROPRIO_SLICE, :].reshape(
                    coeff_gt_row0_B_C_W.shape[0], -1
                )
                future_pred_flat = coeff_pred_B_C_H_W[:, _FREQUENCY_V2_FUTURE_PROPRIO_SLICE, 0, :].reshape(
                    coeff_pred_B_C_H_W.shape[0], -1
                )
                future_num_elements = future_proprio.reshape(future_proprio.shape[0], -1).shape[1]
                k_future = min(future_num_elements, future_gt_flat.shape[1], future_pred_flat.shape[1])
                if k_future > 0:
                    future_huber_B_K = F.huber_loss(
                        future_pred_flat[:, :k_future],
                        future_gt_flat[:, :k_future],
                        reduction="none",
                        delta=huber_delta,
                    )
                    future_loss_B = future_huber_B_K.mean(dim=1)
                    frequency_future_proprio_slot_huber_loss = _masked_average(
                        future_loss_B, demo_or_world_mask_B
                    )

            # Value slot (all samples).
            value_huber_B_2 = F.huber_loss(
                coeff_pred_B_C_H_W[:, _FREQUENCY_V2_VALUE_SLICE, 0, 0],
                coeff_gt_row0_B_C_W[:, _FREQUENCY_V2_VALUE_SLICE, 0],
                reduction="none",
                delta=huber_delta,
            )
            frequency_value_slot_huber_loss = _masked_average(value_huber_B_2.mean(dim=1), any_sample_mask_B)

            frequency_active_slot_huber_loss = (
                float(self.config.frequency_action_slot_loss_weight) * frequency_action_slot_huber_loss
                + float(self.config.frequency_future_proprio_slot_loss_weight) * frequency_future_proprio_slot_huber_loss
                + float(self.config.frequency_value_slot_loss_weight) * frequency_value_slot_huber_loss
            )

            # Off-slot sparsity: penalize predicted DCT energy outside the active slots.
            if self.config.frequency_offslot_sparsity_weight > 0:
                slot_mask_1_C_H_W = torch.zeros(
                    (1, coeff_pred_B_C_H_W.shape[1], coeff_pred_B_C_H_W.shape[2], coeff_pred_B_C_H_W.shape[3]),
                    device=coeff_pred_B_C_H_W.device,
                    dtype=coeff_pred_B_C_H_W.dtype,
                )
                k_task = min(3, coeff_pred_B_C_H_W.shape[-1])
                slot_mask_1_C_H_W[:, _FREQUENCY_V2_TASK_SLICE, 0, :k_task] = 1.0
                k_current = min(3, coeff_pred_B_C_H_W.shape[-1])
                slot_mask_1_C_H_W[:, _FREQUENCY_V2_CURRENT_PROPRIO_SLICE, 0, :k_current] = 1.0
                slot_mask_1_C_H_W[:, _FREQUENCY_V2_ACTION_SLICE, :, :] = _build_frequency_v2_action_slot_mask(
                    batch_size=1,
                    coeff_device=coeff_pred_B_C_H_W.device,
                    coeff_dtype=coeff_pred_B_C_H_W.dtype,
                    latent_h=coeff_pred_B_C_H_W.shape[2],
                    latent_w=coeff_pred_B_C_H_W.shape[3],
                    num_action_elements=k_action,
                )
                if has_future_proprio:
                    slot_mask_1_C_H_W[:, _FREQUENCY_V2_FUTURE_PROPRIO_SLICE, :, :] = (
                        _build_frequency_v2_vector_slot_mask(
                            batch_size=1,
                            coeff_device=coeff_pred_B_C_H_W.device,
                            coeff_dtype=coeff_pred_B_C_H_W.dtype,
                            latent_h=coeff_pred_B_C_H_W.shape[2],
                            latent_w=coeff_pred_B_C_H_W.shape[3],
                            slot_slice=_FREQUENCY_V2_FUTURE_PROPRIO_SLICE,
                            num_elements=k_future,
                        )
                    )
                slot_mask_1_C_H_W[:, _FREQUENCY_V2_VALUE_SLICE, 0, 0] = 1.0

                offslot_abs_B = (coeff_pred_B_C_H_W.abs() * (1.0 - slot_mask_1_C_H_W)).mean(dim=[1, 2, 3])
                frequency_offslot_sparsity_loss = (
                    float(self.config.frequency_offslot_sparsity_weight)
                    * _masked_average(offslot_abs_B, any_sample_mask_B)
                )

        # Get losses for future third-person image prediction
        if torch.all(future_image_indices != -1):  # -1 indicates future third-person image is not used
            batch_indices = torch.arange(x0_B_C_T_H_W.shape[0], device=x0_B_C_T_H_W.device)
            future_image_diff = (
                x0_B_C_T_H_W[batch_indices, :, future_image_indices, :, :]
                - model_pred.x0[batch_indices, :, future_image_indices, :, :]
            )
            future_image_diff_demo = future_image_diff[rollout_data_mask == 0]
            future_image_diff_world_model = future_image_diff[world_model_sample_mask == 1]
            future_image_diff_value_function = future_image_diff[value_function_sample_mask == 1]

            demo_sample_future_image_mse_loss = (future_image_diff_demo**2).mean()
            demo_sample_future_image_l1_loss = torch.abs(future_image_diff_demo).mean()
            world_model_sample_future_image_mse_loss = (future_image_diff_world_model**2).mean()
            world_model_sample_future_image_l1_loss = torch.abs(future_image_diff_world_model).mean()
            all_samples_future_image_mse_loss = (future_image_diff**2).mean()
            all_samples_future_image_l1_loss = torch.abs(future_image_diff).mean()
        else:
            # If not generating future third-person images, set all future image losses to nan
            demo_sample_future_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            demo_sample_future_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)

        # Get losses for future wrist image prediction
        if torch.all(future_wrist_image_indices != -1):  # -1 indicates future wrist image is not used
            future_wrist_image_diff = (
                x0_B_C_T_H_W[batch_indices, :, future_wrist_image_indices, :, :]
                - model_pred.x0[batch_indices, :, future_wrist_image_indices, :, :]
            )
            future_wrist_image_diff_demo = future_wrist_image_diff[rollout_data_mask == 0]
            future_wrist_image_diff_world_model = future_wrist_image_diff[world_model_sample_mask == 1]
            future_wrist_image_diff_value_function = future_wrist_image_diff[value_function_sample_mask == 1]

            demo_sample_future_wrist_image_mse_loss = (future_wrist_image_diff_demo**2).mean()
            demo_sample_future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff_demo).mean()
            world_model_sample_future_wrist_image_mse_loss = (future_wrist_image_diff_world_model**2).mean()
            world_model_sample_future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff_world_model).mean()
            all_samples_future_wrist_image_mse_loss = (future_wrist_image_diff**2).mean()
            all_samples_future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff).mean()
        else:
            # If not generating future wrist images, set all future wrist image losses to nan
            demo_sample_future_wrist_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            demo_sample_future_wrist_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_wrist_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_wrist_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_wrist_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_wrist_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)

        # Get losses for future proprio prediction
        if torch.all(future_proprio_indices != -1):  # -1 indicates future proprio is not used
            future_proprio_diff = (
                x0_B_C_T_H_W[batch_indices, :, future_proprio_indices, :, :]
                - model_pred.x0[batch_indices, :, future_proprio_indices, :, :]
            )
            future_proprio_diff_demo = future_proprio_diff[rollout_data_mask == 0]
            future_proprio_diff_world_model = future_proprio_diff[world_model_sample_mask == 1]
            future_proprio_diff_value_function = future_proprio_diff[value_function_sample_mask == 1]

            demo_sample_future_proprio_mse_loss = (future_proprio_diff_demo**2).mean()
            demo_sample_future_proprio_l1_loss = torch.abs(future_proprio_diff_demo).mean()
            world_model_sample_future_proprio_mse_loss = (future_proprio_diff_world_model**2).mean()
            world_model_sample_future_proprio_l1_loss = torch.abs(future_proprio_diff_world_model).mean()
            all_samples_future_proprio_mse_loss = (future_proprio_diff**2).mean()
            all_samples_future_proprio_l1_loss = torch.abs(future_proprio_diff).mean()
        else:
            # If not generating future proprio, set all future proprio losses to nan
            demo_sample_future_proprio_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            demo_sample_future_proprio_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_proprio_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_proprio_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_proprio_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_proprio_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)

        # Get losses for action prediction
        action_diff = (
            x0_B_C_T_H_W[batch_indices, :, action_indices, :, :] - model_pred.x0[batch_indices, :, action_indices, :, :]
        )
        action_diff_demo = action_diff[rollout_data_mask == 0]
        action_diff_world_model = action_diff[world_model_sample_mask == 1]
        demo_sample_action_mse_loss = (action_diff_demo**2).mean()
        demo_sample_action_l1_loss = torch.abs(action_diff_demo).mean()
        all_samples_action_mse_loss = (action_diff**2).mean()
        all_samples_action_l1_loss = torch.abs(action_diff).mean()

        # Get losses for value function prediction
        value_diff = (
            x0_B_C_T_H_W[batch_indices, :, value_indices, :, :] - model_pred.x0[batch_indices, :, value_indices, :, :]
        )
        value_diff_demo = value_diff[rollout_data_mask == 0]
        value_diff_world_model = value_diff[world_model_sample_mask == 1]
        value_diff_value_function = value_diff[value_function_sample_mask == 1]
        demo_sample_value_mse_loss = (value_diff_demo**2).mean()
        demo_sample_value_l1_loss = torch.abs(value_diff_demo).mean()
        world_model_sample_value_mse_loss = (value_diff_world_model**2).mean()
        world_model_sample_value_l1_loss = torch.abs(value_diff_world_model).mean()
        value_function_sample_value_mse_loss = (value_diff_value_function**2).mean()
        value_function_sample_value_l1_loss = torch.abs(value_diff_value_function).mean()
        all_samples_value_mse_loss = (value_diff**2).mean()
        all_samples_value_l1_loss = torch.abs(value_diff).mean()

        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigma_B_T,
            "weights_per_sigma": weights_per_sigma_B_T,
            "condition": condition,
            "model_pred": model_pred,
            "mse_loss": pred_mse_B_C_T_H_W.mean(),
            "edm_loss": edm_loss_B_C_T_H_W.mean(),
            "edm_loss_per_frame": torch.mean(edm_loss_B_C_T_H_W, dim=[1, 3, 4]),
            # Demo sample losses
            "demo_sample_action_mse_loss": demo_sample_action_mse_loss,  # Main action loss for policy
            "demo_sample_action_l1_loss": demo_sample_action_l1_loss,  # Main action loss for policy
            "demo_sample_future_proprio_mse_loss": demo_sample_future_proprio_mse_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_proprio_l1_loss": demo_sample_future_proprio_l1_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_wrist_image_mse_loss": demo_sample_future_wrist_image_mse_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_wrist_image_l1_loss": demo_sample_future_wrist_image_l1_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_image_mse_loss": demo_sample_future_image_mse_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_image_l1_loss": demo_sample_future_image_l1_loss,  # Auxiliary future state loss for policy
            "demo_sample_value_mse_loss": demo_sample_value_mse_loss,  # Auxiliary value loss for policy
            "demo_sample_value_l1_loss": demo_sample_value_l1_loss,  # Auxiliary value loss for policy
            # World model sample losses
            "world_model_sample_future_proprio_mse_loss": world_model_sample_future_proprio_mse_loss,  # Main future state loss for world model
            "world_model_sample_future_proprio_l1_loss": world_model_sample_future_proprio_l1_loss,  # Main future state loss for world model
            "world_model_sample_future_wrist_image_mse_loss": world_model_sample_future_wrist_image_mse_loss,  # Main future state loss for world model
            "world_model_sample_future_wrist_image_l1_loss": world_model_sample_future_wrist_image_l1_loss,  # Main future state loss for world model
            "world_model_sample_future_image_mse_loss": world_model_sample_future_image_mse_loss,  # Main future state loss for world model
            "world_model_sample_future_image_l1_loss": world_model_sample_future_image_l1_loss,  # Main future state loss for world model
            "world_model_sample_value_mse_loss": world_model_sample_value_mse_loss,  # Auxiliary value loss for world model
            "world_model_sample_value_l1_loss": world_model_sample_value_l1_loss,  # Auxiliary value loss for world model
            # Value function sample losses
            "value_function_sample_value_mse_loss": value_function_sample_value_mse_loss,  # Main loss for value function
            "value_function_sample_value_l1_loss": value_function_sample_value_l1_loss,  # Main loss for value function
            # MetaWAM auxiliary spectral losses
            "frequency_action_slot_huber_loss": frequency_action_slot_huber_loss,
            "frequency_future_proprio_slot_huber_loss": frequency_future_proprio_slot_huber_loss,
            "frequency_value_slot_huber_loss": frequency_value_slot_huber_loss,
            "frequency_active_slot_huber_loss": frequency_active_slot_huber_loss,
            "frequency_offslot_sparsity_loss": frequency_offslot_sparsity_loss,
        }
        return output_batch, kendall_loss, pred_mse_B_C_T_H_W, edm_loss_B_C_T_H_W

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: float | None = None,
        use_variance_scale: bool = False,
        worker_id: int = 0,
        skip_vae_encoding: bool = False,
        previous_generated_latent: torch.Tensor = None,
        return_orig_clean_latent_frames: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch with Cosmos Policy extensions.

        Extended to support:
        - Variance scaling for increased diversity
        - Autoregressive generation (skip_vae_encoding, previous_generated_latent)
        - Returning original clean latent frames

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"
            use_variance_scale (bool): use variance scale to increase diversity in outputs
            worker_id (int): worker id for random seed
            skip_vae_encoding (bool): Skip VAE encoding if True
            previous_generated_latent (torch.Tensor): Previous generated sample
            return_orig_clean_latent_frames (bool): Whether to return the clean latent frames
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        if return_orig_clean_latent_frames:
            x0_fn, orig_clean_latent_frames = self.get_x0_fn_from_batch(
                data_batch,
                guidance,
                is_negative_prompt=is_negative_prompt,
                skip_vae_encoding=skip_vae_encoding,
                previous_generated_latent=previous_generated_latent,
                return_orig_clean_latent_frames=True,
            )
        else:
            x0_fn = self.get_x0_fn_from_batch(
                data_batch,
                guidance,
                is_negative_prompt=is_negative_prompt,
                skip_vae_encoding=skip_vae_encoding,
                previous_generated_latent=previous_generated_latent,
            )

        # NOTE (user): Add random variance scaling to increase diversity in outputs
        if use_variance_scale:
            torch.manual_seed(seed)
            sigma_max_variance_scale = torch.rand(1).item() * 2.0 + 1.0  # uniform between 1.0 and 3.0
            sigma_min_variance_scale = torch.rand(1).item() * 0.9 + 0.1  # uniform between 0.1 and 1.0
        else:
            sigma_max_variance_scale = 1.0
            sigma_min_variance_scale = 1.0

        # NOTE: FlowUniPC scheduler support (inherited from base, keeping for compatibility)
        if self.config.use_flowunipc_scheduler:
            # Use parent implementation for FlowUniPC
            return super().generate_samples_from_batch(
                data_batch,
                guidance,
                seed,
                state_shape,
                n_sample,
                is_negative_prompt,
                num_steps,
                solver_option,
                x_sigma_max,
                sigma_max,
                **kwargs,
            )

        if x_sigma_max is None:
            x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
                * self.sde.sigma_max
                * sigma_max_variance_scale
            )

        if self.net.is_context_parallel_enabled:
            x_sigma_max = broadcast_split_tensor(
                x_sigma_max, seq_dim=2, process_group=self.get_context_parallel_group()
            )

        if sigma_max is None:
            sigma_max = self.sde.sigma_max
        samples = self.sampler(
            x0_fn,
            x_sigma_max,
            num_steps=num_steps,
            sigma_max=sigma_max * sigma_max_variance_scale,
            sigma_min=self.sde.sigma_min * sigma_min_variance_scale,
            solver_option=solver_option,
        )
        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

        if return_orig_clean_latent_frames:
            return samples, orig_clean_latent_frames
        else:
            return samples
