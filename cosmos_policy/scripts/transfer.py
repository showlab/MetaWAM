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
Convert a distributed checkpoint (.distcp) to a single PyTorch checkpoint (.pt).

This script loads a .distcp sharded checkpoint (produced by distributed training)
and saves it as a single .pt file that can be used for single-GPU inference.

Must be run with torchrun (even for single GPU) because torch.distributed.checkpoint
requires a distributed process group.

Usage:
    torchrun --nproc_per_node=1 -m cosmos_policy.scripts.transfer \
        --config <EXPERIMENT_CONFIG_NAME> \
        --distcp_path <PATH_TO_ITER_DIR>/model \
        --output_path <OUTPUT_PATH>.pt \
        --config_file cosmos_policy/config/config.py

Example:
    torchrun --nproc_per_node=1 -m cosmos_policy.scripts.transfer \
        --config metawam_robocasa__inference \
        --distcp_path <RUN_DIR>/checkpoints/iter_000040000/model \
        --output_path <RUN_DIR>/iter_000040000_model.pt \
        --config_file cosmos_policy/config/config.py
"""

import argparse
import importlib
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import misc
from cosmos_policy._src.imaginaire.utils.config_helper import get_config_module, override


def main():
    parser = argparse.ArgumentParser(description="Convert .distcp checkpoint to .pt")
    parser.add_argument("--config", required=True, help="Experiment config name")
    parser.add_argument("--distcp_path", required=True, help="Path to distcp model directory (the 'model/' folder)")
    parser.add_argument("--output_path", required=True, help="Output .pt file path")
    parser.add_argument("--config_file", default="cosmos_policy/config/config.py", help="Config file path")
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Allow checkpoint vs model key mismatch (unsafe). Default: strict; mismatch fails fast.",
    )
    args = parser.parse_args()

    # Initialize distributed (required by torch.distributed.checkpoint)
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    print(f"[convert] Loading config: {args.config}")
    config_module = get_config_module(args.config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={args.config}"])

    # Disable EMA to save memory
    config.model.config.ema.enabled = False
    # Disable FSDP sharding for single-process loading
    config.model.config.fsdp_shard_size = 1

    config.validate()
    config.freeze()

    misc.set_random_seed(seed=0, by_rank=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True

    # Instantiate model (architecture only, no weights yet)
    print("[convert] Instantiating model architecture...")
    with misc.timer("instantiate model"):
        model = instantiate(config.model)
        model.on_train_start()

    # Get the model's state dict as a template for DCP loading
    print(f"[convert] Loading distcp checkpoint from: {args.distcp_path}")
    _state_dict = get_model_state_dict(model)

    # Load the distributed checkpoint into the state dict (strict by default so wrong --config fails fast)
    storage_reader = FileSystemReader(args.distcp_path)
    load_planner = DefaultLoadPlanner(allow_partial_load=args.allow_partial_load)
    dcp.load(
        _state_dict,
        storage_reader=storage_reader,
        planner=load_planner,
    )

    if hasattr(load_planner, "metadata") and load_planner.metadata is not None:
        checkpoint_keys = set(load_planner.metadata.state_dict_metadata.keys())
        model_keys = set(_state_dict.keys())
        missing_keys = sorted(model_keys - checkpoint_keys)
        unexpected_keys = [k for k in sorted(checkpoint_keys - model_keys) if "_extra_state" not in k]
        if not args.allow_partial_load and (missing_keys or unexpected_keys):
            msg = ["[convert] Checkpoint keys do not match model from --config (strict load)."]
            if missing_keys:
                msg.append(f"Missing in checkpoint ({len(missing_keys)}): " + ", ".join(missing_keys[:20]))
                if len(missing_keys) > 20:
                    msg.append(f"... and {len(missing_keys) - 20} more")
            if unexpected_keys:
                msg.append(f"Unexpected in checkpoint ({len(unexpected_keys)}): " + ", ".join(unexpected_keys[:20]))
                if len(unexpected_keys) > 20:
                    msg.append(f"... and {len(unexpected_keys) - 20} more")
            raise RuntimeError("\n".join(msg))
        if args.allow_partial_load:
            if missing_keys:
                print(f"[convert] WARNING: {len(missing_keys)} missing keys (not in checkpoint)")
                for k in missing_keys[:10]:
                    print(f"  - {k}")
            if unexpected_keys:
                print(f"[convert] WARNING: {len(unexpected_keys)} unexpected keys (not in model)")
                for k in unexpected_keys[:10]:
                    print(f"  - {k}")

    # Move all tensors to CPU for saving
    cpu_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in _state_dict.items()}

    # Save as .pt
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    print(f"[convert] Saving .pt checkpoint to: {args.output_path}")
    torch.save(cpu_state_dict, args.output_path)

    file_size_gb = os.path.getsize(args.output_path) / (1024**3)
    print(f"[convert] Done! Saved {len(cpu_state_dict)} keys, file size: {file_size_gb:.2f} GB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
