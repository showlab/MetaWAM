#!/bin/bash
# Convert a sharded training checkpoint (DCP) into a single .pt file for evaluation / deployment.
#
#   bash scripts/convert_checkpoint.sh metawam_robocasa__inference \
#       output/cosmos_policy/metawam/metawam_robocasa/checkpoints/iter_000008400/model \
#       checkpoints/metawam_robocasa_8400.pt
set -euo pipefail

CONFIG="${1:?inference config name, e.g. metawam_robocasa__inference}"
DCP_DIR="${2:?path to checkpoints/iter_XXXXXXXXX/model}"
OUTPUT="${3:?output .pt path}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"
UV_GROUP="${UV_GROUP:-robocasa}"  # use `libero` when converting a LIBERO checkpoint inside the LIBERO environment

uv run --extra cu128 --group "${UV_GROUP}" --python 3.10 \
  torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" -m cosmos_policy.scripts.transfer \
  --config "${CONFIG}" \
  --distcp_path "${DCP_DIR}" \
  --output_path "${OUTPUT}" \
  --config_file cosmos_policy/config/config.py
