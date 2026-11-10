"""Resample a raw on-policy rollout episode onto a fixed-Hz time grid.

Raw rollouts have variable per-frame dt because chunk-boundary inference takes
~1.5s, so the recorded timestamps look like
    [100ms in-chunk] x 7  +  [~1500ms stall]  +  repeat...
The teleop training dataset is post-processed to a strictly uniform 15Hz grid
(dt = 66.67ms). To make the two sources interchangeable for the world-model /
value-function dataloader, this module resamples a raw episode onto the same
uniform grid.

Per-field interpolation rules:
  state[0:3]  (xyz)         -> linear interp
  state[3:6]  (rpy)         -> np.unwrap -> linear interp -> wrap_to_pi
  state[6:8]  (gripper)     -> linear interp
  actions[:]                -> nearest-neighbor. Each raw action is a one-shot
                                delta command sent to the robot at that raw
                                step; linearly interpolating breaks the
                                "state_delta == action" semantic (an interp'd
                                action does not correspond to any command that
                                was actually issued). Nearest preserves the
                                interpretation "this is the raw command in
                                effect at this synthetic time".
  images   (JPEG bytes)     -> nearest-neighbor (the robot literally held
                                position during stalls, so "held" images
                                are physically faithful, not synthetic)

The output parquet schema gets ONE extra column so downstream code can filter
or weigh resampled rows:
  raw_dt_ms (float32): dt between the two raw frames bracketing this output
                       timestamp. In-chunk rows are ~100ms (the raw control
                       period). Rows that fall inside a chunk-boundary
                       inference stall report the full stall dt (~1500ms).
                       In practice ~2/3 of the output frames fall in stalls,
                       so a dataloader can filter `raw_dt_ms < 200` to keep
                       only frames where state was changing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


# Matches wrap_to_pi() in run_franka_eval.py and _wrap_to_pi() in
# realworld_lerobot_dataset.py. Inlined to avoid pulling in their heavy
# imports (cameras, robot interface) for an offline data tool.
def _wrap_to_pi(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return ((arr + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class ResampledEpisode:
    """Output of resample_episode_to_fixed_hz. Mirrors EpisodeBuffer's layout
    but with arrays already stacked + the raw_dt_ms debug column."""
    instruction: str
    images: List[bytes]
    wrist_images: List[bytes]
    left_images: List[bytes]
    states: np.ndarray          # (T_new, STATE_DIM) float32
    actions: np.ndarray         # (T_new, ACTION_DIM) float32
    timestamps: np.ndarray      # (T_new,) float32, strictly uniform target_dt
    raw_dt_ms: np.ndarray       # (T_new,) float32

    @property
    def length(self) -> int:
        return int(self.timestamps.shape[0])


def _nearest_indices(ts_raw: np.ndarray, new_ts: np.ndarray) -> np.ndarray:
    """For each value in new_ts, return the index in ts_raw whose value is
    closest. ts_raw must be sorted ascending."""
    right = np.searchsorted(ts_raw, new_ts, side="left")
    right = np.clip(right, 1, len(ts_raw) - 1)
    left = right - 1
    use_left = np.abs(ts_raw[left] - new_ts) <= np.abs(ts_raw[right] - new_ts)
    return np.where(use_left, left, right)


def _bracketing_dt_ms(ts_raw: np.ndarray, new_ts: np.ndarray) -> np.ndarray:
    """For each new_ts value, return the gap (ms) between the two raw frames
    that bracket it. For new_ts past the last raw point, use the final gap."""
    # right = first raw index strictly greater than new_ts
    right = np.searchsorted(ts_raw, new_ts, side="right")
    right = np.clip(right, 1, len(ts_raw) - 1)
    left = right - 1
    return ((ts_raw[right] - ts_raw[left]) * 1000.0).astype(np.float32)


def resample_episode_to_fixed_hz(
    *,
    instruction: str,
    images: Sequence[bytes],
    wrist_images: Sequence[bytes],
    left_images: Sequence[bytes],
    states: np.ndarray,         # (T_raw, STATE_DIM)
    actions: np.ndarray,        # (T_raw, ACTION_DIM)
    timestamps: np.ndarray,     # (T_raw,) seconds, monotonically increasing, starts at ~0
    target_hz: float = 15.0,
) -> ResampledEpisode:
    """Resample a raw episode onto a uniform target_hz time grid.

    The output grid spans [0, ts_raw[-1]] inclusive (truncated at the last
    full target_dt step, so we never extrapolate beyond the last raw sample).
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    ts_raw = np.asarray(timestamps, dtype=np.float64)

    T_raw = ts_raw.shape[0]
    if T_raw < 2:
        raise ValueError(f"Need at least 2 raw frames to resample, got {T_raw}")
    if states.shape[0] != T_raw or actions.shape[0] != T_raw:
        raise ValueError(
            f"Length mismatch: ts={T_raw}, states={states.shape[0]}, actions={actions.shape[0]}"
        )
    if states.shape[1] < 8 or actions.shape[1] < 7:
        raise ValueError(
            f"Unexpected feature dims: state={states.shape}, action={actions.shape}"
        )
    if not np.all(np.diff(ts_raw) > 0):
        raise ValueError("Raw timestamps must be strictly monotonically increasing")
    if len(images) != T_raw or len(wrist_images) != T_raw or len(left_images) != T_raw:
        raise ValueError(
            f"Image list length mismatch: ts={T_raw}, "
            f"images={len(images)}, wrist={len(wrist_images)}, left={len(left_images)}"
        )

    # Anchor grid at 0 (collect_rollouts already subtracts t0). End at the
    # last raw sample but never extrapolate past it.
    target_dt = 1.0 / float(target_hz)
    t_end = float(ts_raw[-1])
    n_new = int(np.floor(t_end / target_dt)) + 1     # include t=0
    new_ts = np.arange(n_new, dtype=np.float64) * target_dt
    new_ts = new_ts[new_ts <= t_end + 1e-9]
    N = new_ts.shape[0]
    if N < 1:
        raise ValueError(
            f"Resample produced zero output frames (raw span={t_end:.3f}s, target_dt={target_dt:.4f}s)"
        )

    # --- state ---
    new_states = np.empty((N, states.shape[1]), dtype=np.float32)
    # xyz: dims 0..2
    for i in range(3):
        new_states[:, i] = np.interp(new_ts, ts_raw, states[:, i]).astype(np.float32)
    # rpy: dims 3..5 -- unwrap across time so we don't interpolate through
    # ±pi discontinuities, then wrap back to [-pi, pi] which matches the
    # convention the dataloader and policy operate in.
    for i in (3, 4, 5):
        unwrapped = np.unwrap(states[:, i])
        interped = np.interp(new_ts, ts_raw, unwrapped)
        new_states[:, i] = _wrap_to_pi(interped).astype(np.float32)
    # gripper widths: dims 6..7
    for i in (6, 7):
        new_states[:, i] = np.interp(new_ts, ts_raw, states[:, i]).astype(np.float32)

    # --- actions + images: all nearest-neighbor (see module docstring) ---
    nn_idx = _nearest_indices(ts_raw, new_ts)
    new_actions = actions[nn_idx].astype(np.float32)
    # The gripper channel is semantically a command in [0, 1] (close/open
    # convention is handled elsewhere). Policy outputs can overshoot slightly
    # (e.g. -0.03 or 1.03); clamp before writing so normalized rollout actions
    # stay inside the demo statistics bounds.
    new_actions[:, 6] = np.clip(new_actions[:, 6], 0.0, 1.0)
    new_images       = [images[i]       for i in nn_idx]
    new_wrist_images = [wrist_images[i] for i in nn_idx]
    new_left_images  = [left_images[i]  for i in nn_idx]

    raw_dt_ms = _bracketing_dt_ms(ts_raw, new_ts)

    return ResampledEpisode(
        instruction=instruction,
        images=new_images,
        wrist_images=new_wrist_images,
        left_images=new_left_images,
        states=new_states,
        actions=new_actions,
        timestamps=new_ts.astype(np.float32),
        raw_dt_ms=raw_dt_ms,
    )
