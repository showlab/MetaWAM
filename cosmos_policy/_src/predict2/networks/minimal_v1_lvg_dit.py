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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms

from cosmos_policy._src.predict2.conditioner import DataType
from cosmos_policy._src.predict2.networks.minimal_v4_dit import MiniTrainDIT, PatchEmbed


class MinimalV1LVGDiT(MiniTrainDIT):
    def __init__(self, *args, timestep_scale: float = 1.0, slot_condition_mask_channels: int = 0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        assert slot_condition_mask_channels >= 0, (
            f"slot_condition_mask_channels must be >= 0, got {slot_condition_mask_channels}"
        )
        self.slot_condition_mask_channels = int(slot_condition_mask_channels)
        kwargs["in_channels"] += 1  # Add 1 for the condition mask
        self.timestep_scale = timestep_scale
        super().__init__(*args, **kwargs)
        # Optional slot-aware condition mask (MetaWAM). x_embedder keeps the base model's single coarse
        # mask channel; the slot-aware mask goes through a separate patch embedder whose output is added
        # residually after x_embedder.
        self._slot_condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None
        if self.slot_condition_mask_channels > 0:
            self.slot_mask_embedder = PatchEmbed(
                spatial_patch_size=self.patch_spatial,
                temporal_patch_size=self.patch_temporal,
                in_channels=self.slot_condition_mask_channels,
                out_channels=self.model_channels,
            )
            # Zero init: training starts from the exact base-model behavior.
            nn.init.zeros_(self.slot_mask_embedder.proj[1].weight)
        else:
            self.slot_mask_embedder = None

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Same as MiniTrainDIT.prepare_embedded_sequence, plus the residual slot-mask embedding.
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.slot_mask_embedder is not None and self._slot_condition_video_input_mask_B_C_T_H_W is not None:
            slot_condition_video_input_mask_B_C_T_H_W = self._slot_condition_video_input_mask_B_C_T_H_W
            assert slot_condition_video_input_mask_B_C_T_H_W.shape[1] == self.slot_condition_mask_channels, (
                "Slot condition mask channel mismatch: "
                f"got {slot_condition_video_input_mask_B_C_T_H_W.shape[1]}, "
                f"expected {self.slot_condition_mask_channels}"
            )
            x_B_T_H_W_D = x_B_T_H_W_D + self.slot_mask_embedder(
                slot_condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)
            ).to(dtype=x_B_T_H_W_D.dtype)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)

        return x_B_T_H_W_D, None, extra_pos_emb

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        slot_condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            if condition_video_input_mask_B_C_T_H_W is None:
                B, _, T, H, W = x_B_C_T_H_W.shape
                condition_video_input_mask_B_C_T_H_W = torch.zeros(
                    (B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device
                )
            else:
                assert condition_video_input_mask_B_C_T_H_W.shape[1] == 1, (
                    f"Condition mask must have 1 channel, got {condition_video_input_mask_B_C_T_H_W.shape[1]}"
                )
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
        if slot_condition_video_input_mask_B_C_T_H_W is None and self.slot_condition_mask_channels > 0:
            # No slot-aware mask from the caller: an all-zero mask reproduces the base model.
            B, _, T, H, W = x_B_C_T_H_W.shape
            slot_condition_video_input_mask_B_C_T_H_W = torch.zeros(
                (B, self.slot_condition_mask_channels, T, H, W),
                dtype=x_B_C_T_H_W.dtype,
                device=x_B_C_T_H_W.device,
            )
        # Consumed by prepare_embedded_sequence during super().forward().
        self._slot_condition_video_input_mask_B_C_T_H_W = slot_condition_video_input_mask_B_C_T_H_W
        try:
            return super().forward(
                x_B_C_T_H_W=x_B_C_T_H_W,
                timesteps_B_T=timesteps_B_T * self.timestep_scale,
                crossattn_emb=crossattn_emb,
                fps=fps,
                padding_mask=padding_mask,
                data_type=data_type,
                intermediate_feature_ids=intermediate_feature_ids,
                img_context_emb=img_context_emb,
            )
        finally:
            self._slot_condition_video_input_mask_B_C_T_H_W = None
