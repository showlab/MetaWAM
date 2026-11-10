"""Collect on-policy Franka rollouts in LeRobot parquet format.

Drives the same FrankaPolicyRunner used by run_franka_eval.py, but installs a
per-frame callback that buffers (image, wrist_image, left_image, state, action,
timestamp) for the whole episode. After each rollout the operator labels the
episode (s/f/d) and a parquet file plus meta/episodes.jsonl + meta/tasks.jsonl
entries are appended to the target split.

The output layout matches RealWorldLeRobotDataset:
    <output_root>/<task_name>/
        data/chunk-000/episode_NNNNNN.parquet
        meta/episodes.jsonl    (one record per episode, with extra "success" + "collected_at")
        meta/tasks.jsonl       (task vocabulary)
        meta/info.json         (split-level info)
        vis/episode_NNNNNN_success|failure_wrist.mp4

Parquet schema is the existing teleop schema plus one optional diagnostic
column (`raw_dt_ms`). Downstream code that reads the standard columns
(RealWorldLeRobotDataset, generate_dataset_statistics.py) can ingest the new
episodes without changes. Failed and successful rollouts are both saved; the
"success" flag in episodes.jsonl lets downstream code split them as needed.

Usage:
    python -m cosmos_policy.experiments.robot.franka.collect_rollouts \
        --output-root <OUTPUT_DIR>/lerobot_rollouts \
        --task-name build \
        --default-prompt "Place the orange block on top of the gray block" \
        --nuc-ip <NUC_IP> \
        --external-camera <EXTERNAL_CAMERA_SERIAL> \
        --wrist-camera <WRIST_CAMERA_SERIAL> \
        --left-camera <LEFT_CAMERA_SERIAL> \
        --server-ip 127.0.0.1 --server-port 8777

    # Smoke test with mock hardware:
    python -m cosmos_policy.experiments.robot.franka.collect_rollouts \
        --use-mock-cameras --use-mock-robot \
        --output-root /tmp/test_rollouts --task-name build \
        --default-prompt "Place the orange block on top of the gray block" \
        --max-timesteps 30 --num-rollouts 2
"""

from __future__ import annotations

import datetime
import json
import os
import select
import subprocess
import sys
import termios
import time
import tty
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

from cosmos_policy.experiments.robot.franka import config as cfg
from cosmos_policy.experiments.robot.franka.resample_rollout import (
    ResampledEpisode,
    resample_episode_to_fixed_hz,
)
from cosmos_policy.experiments.robot.franka.run_franka_eval import FrankaPolicyRunner


STATE_DIM = 8
ACTION_DIM = 7
JPEG_QUALITY = 95

# HuggingFace datasets metadata that marks the three image columns as Image
# features. Not required by the cosmos training pipeline (it reads parquet
# directly with pyarrow) but keeps `datasets`/HF viewers compatible.
HF_SCHEMA_METADATA = {
    "info": {
        "features": {
            "image": {"_type": "Image"},
            "wrist_image": {"_type": "Image"},
            "left_image": {"_type": "Image"},
            "state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": STATE_DIM, "_type": "Sequence"},
            "actions": {"feature": {"dtype": "float32", "_type": "Value"}, "length": ACTION_DIM, "_type": "Sequence"},
            "timestamp": {"dtype": "float32", "_type": "Value"},
            "frame_index": {"dtype": "int64", "_type": "Value"},
            "episode_index": {"dtype": "int64", "_type": "Value"},
            "index": {"dtype": "int64", "_type": "Value"},
            "task_index": {"dtype": "int64", "_type": "Value"},
            "raw_dt_ms": {"dtype": "float32", "_type": "Value"},
        }
    }
}

PARQUET_SCHEMA = pa.schema(
    [
        pa.field(
            "image",
            pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
        ),
        pa.field(
            "wrist_image",
            pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
        ),
        pa.field(
            "left_image",
            pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
        ),
        pa.field("state", pa.list_(pa.float32(), STATE_DIM)),
        pa.field("actions", pa.list_(pa.float32(), ACTION_DIM)),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
        # raw control-loop dt at this row. For resampled episodes it carries
        # the bracketing raw dt (so a dataloader can drop stall frames with
        # raw_dt_ms > 200). For raw episodes it's the actual per-step dt.
        pa.field("raw_dt_ms", pa.float32()),
    ],
    metadata={b"huggingface": json.dumps(HF_SCHEMA_METADATA).encode("utf-8")},
)


def _encode_jpeg(image_rgb: np.ndarray) -> bytes:
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode image to JPEG")
    return bytes(buf)


def _image_struct(jpeg_bytes: bytes) -> Dict[str, Any]:
    return {"bytes": jpeg_bytes, "path": None}


@dataclass
class EpisodeBuffer:
    instruction: str
    images: List[bytes] = field(default_factory=list)
    wrist_images: List[bytes] = field(default_factory=list)
    left_images: List[bytes] = field(default_factory=list)
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    t0: Optional[float] = None

    def add(
        self,
        image_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        left_rgb: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        step_time: float,
    ) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if state.shape[0] != STATE_DIM:
            raise ValueError(f"Expected {STATE_DIM}D state, got shape {state.shape}")
        if action.shape[0] != ACTION_DIM:
            raise ValueError(f"Expected {ACTION_DIM}D action, got shape {action.shape}")
        if self.t0 is None:
            self.t0 = float(step_time)

        self.images.append(_encode_jpeg(image_rgb))
        self.wrist_images.append(_encode_jpeg(wrist_rgb))
        self.left_images.append(_encode_jpeg(left_rgb))
        self.states.append(state)
        self.actions.append(action)
        self.timestamps.append(float(step_time) - self.t0)

    @property
    def length(self) -> int:
        return len(self.states)


class LeRobotDatasetWriter:
    """Append-only writer for a LeRobot split. Discovers existing state on disk
    so a session can resume episode/global index numbering after a crash."""

    def __init__(
        self,
        output_root: Path,
        task_name: str,
        save_wrist_vis: bool = True,
        wrist_vis_fps: int = 10,
        target_hz: float = 15.0,
        save_raw_too: bool = False,
    ):
        self.split_root = Path(output_root) / task_name
        self.task_name = task_name
        self.data_dir = self.split_root / "data" / "chunk-000"
        # When resampling is on and the caller asks for it, also keep the raw
        # rollout in a parallel data_raw/ tree so the resample can be
        # re-derived offline if its logic ever changes.
        self.raw_data_dir = self.split_root / "data_raw" / "chunk-000"
        self.meta_dir = self.split_root / "meta"
        self.vis_dir = self.split_root / "vis"
        self.save_wrist_vis = save_wrist_vis
        self.wrist_vis_fps = wrist_vis_fps
        self.target_hz = float(target_hz)
        self.save_raw_too = bool(save_raw_too) and self.target_hz > 0
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        if self.save_wrist_vis:
            self.vis_dir.mkdir(parents=True, exist_ok=True)
        if self.save_raw_too:
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.meta_dir / "episodes.jsonl"
        self.tasks_path = self.meta_dir / "tasks.jsonl"
        self.info_path = self.meta_dir / "info.json"
        self._load_state()

    def _load_state(self) -> None:
        max_ep = -1
        for parquet_path in self.data_dir.glob("episode_*.parquet"):
            try:
                ep_idx = int(parquet_path.stem.split("_")[-1])
            except ValueError:
                continue
            if ep_idx > max_ep:
                max_ep = ep_idx
        self.next_episode_index = max_ep + 1

        self.next_global_index = 0
        if max_ep >= 0:
            last_path = self.data_dir / f"episode_{max_ep:06d}.parquet"
            try:
                table = pq.read_table(last_path, columns=["index"])
                self.next_global_index = int(table["index"].to_pandas().max()) + 1
            except Exception as e:
                print(f"  [WARNING] Failed to read last episode for global index: {e}")

        self.tasks: Dict[str, int] = {}
        if self.tasks_path.exists():
            with self.tasks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self.tasks[str(rec["task"])] = int(rec["task_index"])

    def _ensure_task_index(self, instruction: str) -> int:
        if instruction in self.tasks:
            return self.tasks[instruction]
        new_idx = max(self.tasks.values(), default=-1) + 1
        self.tasks[instruction] = new_idx
        with self.tasks_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"task_index": new_idx, "task": instruction}) + "\n")
        return new_idx

    def write_episode(self, buffer: EpisodeBuffer, success: bool) -> int:
        """Write one rollout episode. If self.target_hz > 0, the buffer is
        resampled onto a uniform grid before going to parquet so the on-disk
        dt matches the teleop training set; raw frames are also dumped to
        data_raw/ when self.save_raw_too is set. Wrist-vis videos are always
        rendered from the raw buffer."""
        if buffer.length == 0:
            raise ValueError("Refusing to write empty episode buffer")

        episode_index = self.next_episode_index
        task_index = self._ensure_task_index(buffer.instruction)

        raw_T = buffer.length
        raw_duration_s = float(buffer.timestamps[-1]) if raw_T > 1 else 0.0

        if self.target_hz > 0 and raw_T >= 2:
            resampled = resample_episode_to_fixed_hz(
                instruction=buffer.instruction,
                images=buffer.images,
                wrist_images=buffer.wrist_images,
                left_images=buffer.left_images,
                states=np.stack(buffer.states),
                actions=np.stack(buffer.actions),
                timestamps=np.asarray(buffer.timestamps, dtype=np.float64),
                target_hz=self.target_hz,
            )
            T = resampled.length
            self._write_parquet(
                self.data_dir / f"episode_{episode_index:06d}.parquet",
                resampled=resampled,
                episode_index=episode_index,
                task_index=task_index,
            )
            if self.save_raw_too:
                self._write_raw_parquet(
                    self.raw_data_dir / f"episode_{episode_index:06d}.parquet",
                    buffer,
                    episode_index=episode_index,
                    task_index=task_index,
                )
        else:
            # target_hz == 0 (or trivial episode): write raw frames straight to data/.
            T = raw_T
            self._write_raw_parquet(
                self.data_dir / f"episode_{episode_index:06d}.parquet",
                buffer,
                episode_index=episode_index,
                task_index=task_index,
            )

        wrist_vis_path = self._write_wrist_vis(buffer, episode_index, success)

        record = {
            "episode_index": episode_index,
            "tasks": [buffer.instruction],
            "length": T,
            "raw_length": raw_T,
            "raw_duration_s": raw_duration_s,
            "resampled_target_hz": self.target_hz if self.target_hz > 0 else None,
            "success": bool(success),
            "collected_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if wrist_vis_path is not None:
            record["wrist_vis_path"] = str(wrist_vis_path.relative_to(self.split_root))
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self._update_info_json()

        self.next_episode_index += 1
        self.next_global_index += T
        return episode_index

    def _write_parquet(
        self,
        out_path: Path,
        *,
        resampled: ResampledEpisode,
        episode_index: int,
        task_index: int,
    ) -> None:
        T = resampled.length
        global_indices = list(range(self.next_global_index, self.next_global_index + T))
        table = pa.table(
            {
                "image":         [_image_struct(b) for b in resampled.images],
                "wrist_image":   [_image_struct(b) for b in resampled.wrist_images],
                "left_image":    [_image_struct(b) for b in resampled.left_images],
                "state":         [s.tolist() for s in resampled.states],
                "actions":       [a.tolist() for a in resampled.actions],
                "timestamp":     pa.array(resampled.timestamps.tolist(), type=pa.float32()),
                "frame_index":   pa.array(list(range(T)), type=pa.int64()),
                "episode_index": pa.array([episode_index] * T, type=pa.int64()),
                "index":         pa.array(global_indices, type=pa.int64()),
                "task_index":    pa.array([task_index] * T, type=pa.int64()),
                "raw_dt_ms":     pa.array(resampled.raw_dt_ms.tolist(), type=pa.float32()),
            },
            schema=PARQUET_SCHEMA,
        )
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, out_path)

    def _write_raw_parquet(
        self,
        out_path: Path,
        buffer: EpisodeBuffer,
        *,
        episode_index: int,
        task_index: int,
    ) -> None:
        T = buffer.length
        global_indices = list(range(self.next_global_index, self.next_global_index + T))
        # raw_dt_ms = actual gap to the next frame; last frame repeats the
        # preceding gap so the column length matches T.
        ts = np.asarray(buffer.timestamps, dtype=np.float64)
        if T >= 2:
            diffs = (np.diff(ts) * 1000.0).astype(np.float32)
            raw_dt_ms = np.concatenate([diffs, diffs[-1:]]).astype(np.float32)
        else:
            raw_dt_ms = np.zeros(T, dtype=np.float32)
        table = pa.table(
            {
                "image":         [_image_struct(b) for b in buffer.images],
                "wrist_image":   [_image_struct(b) for b in buffer.wrist_images],
                "left_image":    [_image_struct(b) for b in buffer.left_images],
                "state":         [s.tolist() for s in buffer.states],
                "actions":       [self._clamp_actions(a).tolist() for a in buffer.actions],
                "timestamp":     pa.array(buffer.timestamps, type=pa.float32()),
                "frame_index":   pa.array(list(range(T)), type=pa.int64()),
                "episode_index": pa.array([episode_index] * T, type=pa.int64()),
                "index":         pa.array(global_indices, type=pa.int64()),
                "task_index":    pa.array([task_index] * T, type=pa.int64()),
                "raw_dt_ms":     pa.array(raw_dt_ms.tolist(), type=pa.float32()),
            },
            schema=PARQUET_SCHEMA,
        )
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, out_path)

    @staticmethod
    def _clamp_actions(action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape[-1] >= 7:
            action[..., 6] = np.clip(action[..., 6], 0.0, 1.0)
        return action

    @staticmethod
    def _decode_wrist_frames(
        wrist_images: List[bytes], width: int, height: int,
    ) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        for jpeg_bytes in wrist_images:
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        return frames

    @staticmethod
    def _write_h264_video_ffmpeg(
        frames: List[np.ndarray], out_path: Path, fps: int, width: int, height: int,
    ) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            assert proc.stdin is not None
            for frame in frames:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
            returncode = proc.wait()
        except Exception:
            proc.kill()
            proc.wait()
            raise

        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {returncode}: {stderr.strip()}")

    @staticmethod
    def _write_mp4v_video_opencv(
        frames: List[np.ndarray], out_path: Path, fps: int, width: int, height: int,
    ) -> None:
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer for {out_path}")

        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

    def _write_wrist_vis(self, buffer: EpisodeBuffer, episode_index: int, success: bool) -> Optional[Path]:
        if not self.save_wrist_vis or not buffer.wrist_images:
            return None

        label = "success" if success else "failure"
        out_path = self.vis_dir / f"episode_{episode_index:06d}_{label}_wrist.mp4"
        tmp_path = self.vis_dir / f"episode_{episode_index:06d}_{label}_wrist.tmp.mp4"

        try:
            first = cv2.imdecode(np.frombuffer(buffer.wrist_images[0], dtype=np.uint8), cv2.IMREAD_COLOR)
            if first is None:
                raise RuntimeError("failed to decode first wrist frame")

            height, width = first.shape[:2]
            fps = max(1, int(self.wrist_vis_fps))
            frames = self._decode_wrist_frames(buffer.wrist_images, width, height)
            if not frames:
                raise RuntimeError("failed to decode wrist frames")

            try:
                self._write_h264_video_ffmpeg(frames, tmp_path, fps, width, height)
            except Exception as ffmpeg_error:
                print(
                    "  [WARNING] H.264 wrist vis encode failed; falling back to mp4v. "
                    f"VSCode preview may not work. Details: {ffmpeg_error}"
                )
                self._write_mp4v_video_opencv(frames, tmp_path, fps, width, height)

            os.replace(tmp_path, out_path)
            print(f"  [Collector] Wrist vis saved: {out_path}")
            return out_path
        except Exception as e:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            print(f"  [WARNING] Failed to write wrist vis video: {e}")
            return None

    def _update_info_json(self) -> None:
        num_episodes = 0
        with self.episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    num_episodes += 1
        info: Dict[str, Any] = {}
        if self.info_path.exists():
            try:
                info = json.loads(self.info_path.read_text(encoding="utf-8"))
            except Exception:
                info = {}
        info.update(
            {
                "generated_by": "collect_rollouts.py",
                "task_name": self.task_name,
                "last_updated": datetime.datetime.now().isoformat(timespec="seconds"),
                "num_episodes": num_episodes,
                # The fps the on-disk parquet is written at -- target_hz when
                # resampling is on, otherwise None (raw rollouts have variable dt).
                "fps": self.target_hz if self.target_hz > 0 else None,
            }
        )
        self.info_path.write_text(json.dumps(info, indent=2))


@contextmanager
def _cbreak_stdin():
    """Put stdin in non-canonical (cbreak) mode so single keypresses are visible
    without waiting for Enter. Falls back to a no-op when stdin is not a TTY
    (e.g. piped input, CI), in which case key polling will return None."""
    if not sys.stdin.isatty():
        yield None
        return
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def _poll_keypress(fd: Optional[int]) -> Optional[str]:
    if fd is None:
        return None
    if select.select([sys.stdin], [], [], 0.0)[0]:
        ch = sys.stdin.read(1)
        return ch if ch else None
    return None


@dataclass
class _RolloutControl:
    """Carried by the keypress callback so the main loop knows what the
    operator asked for during the rollout."""
    stop_requested: bool = False
    auto_label: Optional[Literal["s", "f", "d"]] = None
    cbreak_fd: Optional[int] = None


def _prompt_label() -> Literal["s", "f", "d"]:
    while True:
        ans = input("Label episode? [s]uccess / [f]ailure / [d]iscard: ").strip().lower()
        if ans in ("s", "success"):
            return "s"
        if ans in ("f", "fail", "failure"):
            return "f"
        if ans in ("d", "discard", "drop"):
            return "d"
        print("  Please enter 's', 'f', or 'd'.")


def _resolve_instruction(raw: str, default_prompt: Optional[str]) -> Optional[str]:
    if raw.lower() in ("quit", "exit", "q"):
        return None
    if not raw and default_prompt:
        return default_prompt
    aliases = {
        "d_sweep": "Sweep the green block into the dustpan",
        "d_cup": "Place the cup on the green coaster",
        "d_build": "Place the orange block on top of the gray block",
    }
    if raw.lower() in aliases:
        return aliases[raw.lower()]
    return raw or None


def _parse_home_joints_csv(raw: str) -> np.ndarray:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != 7:
        raise ValueError(f"Expected 7 comma-separated home joint values, got {len(values)}: {raw!r}")
    return np.asarray(values, dtype=np.float64)


def _parse_csv_vector(raw: str, *, expected_len: int, name: str) -> np.ndarray:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated {name} values, got {len(values)}: {raw!r}")
    return np.asarray(values, dtype=np.float64)


def _sample_uniform_range(low: np.ndarray, high: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if low.shape != high.shape:
        raise ValueError(f"Range shape mismatch: low={low.shape}, high={high.shape}")
    if np.any(high < low):
        raise ValueError(f"Range high must be >= low for every axis: low={low}, high={high}")
    return rng.uniform(low=low, high=high).astype(np.float64)


def _move_to_start_joints(
    runner: FrankaPolicyRunner,
    joints: np.ndarray,
    move_time: float,
    settle_time: float,
    label: str = "randomized start",
) -> None:
    if runner.robot is None:
        raise RuntimeError(f"Robot is not initialized; call runner.setup() before moving to {label} joints.")
    joint_csv = ", ".join(f"{value:.9f}" for value in joints)
    print(f"[Collector] Moving to {label} joints: {joint_csv}")
    try:
        runner.robot.terminate_current_policy()
    except Exception:
        pass
    runner.robot.move_to_joint_positions(joints, time_to_go=float(move_time))
    if settle_time > 0:
        time.sleep(float(settle_time))
    # move_to_joint_positions may replace the impedance policy on the robot.
    # Restart the controller expected by the rollout action loop.
    try:
        runner.robot.terminate_current_policy()
    except Exception:
        pass
    runner._start_robot_controller()


def _sample_jittered_joints(
    base_joints: np.ndarray,
    jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    offsets = rng.uniform(-jitter, jitter, size=base_joints.shape)
    return (base_joints + offsets).astype(np.float64)


def main(
    # Output
    output_root: str = "lerobot_rollouts",
    task_name: str = "cup",
    save_wrist_vis: bool = True,
    wrist_vis_fps: int = 15,
    # Resample
    # 15 Hz matches the teleop training dataset's strict 66.67ms grid. Set to 0
    # to skip resampling and write raw variable-dt frames straight to data/.
    target_hz: float = 15.0,
    # When resampling, also dump the unresampled per-step frames under
    # data_raw/ for offline debugging / re-derivation.
    save_raw_too: bool = False,
    # Cameras
    external_camera: Optional[str] = None,
    wrist_camera: Optional[str] = None,
    left_camera: Optional[str] = None,
    use_mock_cameras: bool = False,
    # Robot
    nuc_ip: str = "127.0.0.1",
    nuc_port: int = 4242,
    control_mode: Literal["joint", "eef"] = "eef",
    # 15 Hz matches the teleop training distribution. The actual achievable
    # rate is limited by camera read + server inference latency; chunk-boundary
    # inference takes ~1.5s, so cross-chunk dt blows up regardless of this
    # setting. The resample step downstream rectifies dt to target_hz.
    control_frequency: int = 15,
    use_mock_robot: bool = False,
    # Inference server
    server_ip: str = "127.0.0.1",
    server_port: int = 8777,
    # Rollout loop
    max_timesteps: int = 600,
    num_rollouts: int = -1,
    show_cameras: bool = True,
    default_prompt: Optional[str] = None,
    close_gripper_on_start: bool = False,
    # Policy / control knobs (mirrors run_franka_eval)
    open_loop_horizon: int = 8,
    policy_proprio_format: Literal["eef_8d", "joint_9d"] = "eef_8d",
    policy_gripper_proprio_mode: Literal["measured_width_signed", "zero"] = "measured_width_signed",
    policy_action_mode: Literal["delta_eef_7d", "delta_eef_7d_z_positive_down", "absolute_eef"] = "delta_eef_7d",
    policy_gripper_positive_action: Literal["close", "open"] = "open",
    gripper_release_confirm_steps: Optional[int] = None,
    gripper_chunk_lookahead_release: bool = True,
    gripper_chunk_lookahead_confirm_steps: int = 3,
    gripper_chunk_lookahead_min_offset: Optional[int] = None,
    delta_action_base: Literal["measured", "target"] = "measured",
    lock_rpy_actions: bool = False,
    lock_rpy_action_axes: str = "",
    max_target_position_lead: float = 0.05,
    max_target_vertical_lead: float = 0.02,
    min_target_z: Optional[float] = None,
    closed_gripper_min_target_z: Optional[float] = None,
    build_task_min_target_z: Optional[float] = None,
    build_task_closed_gripper_min_target_z: Optional[float] = None,
    start_joint_low: Optional[str] = None,
    start_joint_high: Optional[str] = None,
    start_joint_move_time: float = 3.0,
    start_joint_settle_time: float = 0.5,
    post_rollout_home_base: Optional[str] = None,
    post_rollout_home_jitter: float = 0.015,
    post_rollout_home_time: float = 5.0,
):
    """Collect on-policy rollouts into a LeRobot-format split.

    By default each rollout is resampled to a strictly uniform target_hz grid
    (15 Hz, matching the teleop training set) before being written to parquet
    -- raw rollouts have variable dt because chunk-boundary inference takes
    ~1.5s. Pass --target-hz 0 to disable resampling and dump raw frames as-is.
    """
    server_url = f"http://{server_ip}:{server_port}"
    start_joint_low_values = None
    start_joint_high_values = None
    start_joint_rng = np.random.default_rng()
    if start_joint_low or start_joint_high:
        if not start_joint_low or not start_joint_high:
            raise ValueError("Provide both start_joint_low and start_joint_high, or neither.")
        if start_joint_move_time <= 0:
            raise ValueError("start_joint_move_time must be > 0")
        if start_joint_settle_time < 0:
            raise ValueError("start_joint_settle_time must be >= 0")
        start_joint_low_values = _parse_csv_vector(start_joint_low, expected_len=7, name="start joint low")
        start_joint_high_values = _parse_csv_vector(start_joint_high, expected_len=7, name="start joint high")
        if np.any(start_joint_high_values < start_joint_low_values):
            raise ValueError("start_joint_high must be >= start_joint_low for every joint.")

    post_rollout_home_joints = None
    post_rollout_home_rng = np.random.default_rng()
    if post_rollout_home_base:
        if post_rollout_home_jitter < 0:
            raise ValueError("post_rollout_home_jitter must be >= 0")
        post_rollout_home_joints = _parse_home_joints_csv(post_rollout_home_base)

    config = cfg.Config(
        camera=cfg.CameraConfig(
            external_camera_serial=external_camera,
            wrist_camera_serial=wrist_camera,
            left_camera_serial=left_camera,
            use_mock_cameras=use_mock_cameras,
        ),
        robot=cfg.RobotConfig(
            nuc_ip=nuc_ip,
            nuc_port=nuc_port,
            control_mode=control_mode,
            control_frequency=control_frequency,
            use_mock_robot=use_mock_robot,
        ),
        policy=cfg.PolicyConfig(
            open_loop_horizon=open_loop_horizon,
            proprio_format=policy_proprio_format,
            gripper_proprio_mode=policy_gripper_proprio_mode,
            action_mode=policy_action_mode,
            gripper_positive_action=policy_gripper_positive_action,
            gripper_release_confirm_steps=gripper_release_confirm_steps,
            gripper_chunk_lookahead_release=gripper_chunk_lookahead_release,
            gripper_chunk_lookahead_confirm_steps=gripper_chunk_lookahead_confirm_steps,
            gripper_chunk_lookahead_min_offset=gripper_chunk_lookahead_min_offset,
            delta_action_base=delta_action_base,
            lock_rpy_actions=lock_rpy_actions,
            lock_rpy_action_axes=lock_rpy_action_axes,
            max_target_position_lead=max_target_position_lead,
            max_target_vertical_lead=max_target_vertical_lead,
            min_target_z=min_target_z,
            closed_gripper_min_target_z=closed_gripper_min_target_z,
            build_task_min_target_z=build_task_min_target_z,
            build_task_closed_gripper_min_target_z=build_task_closed_gripper_min_target_z,
            default_prompt=default_prompt,
        ),
        max_timesteps=max_timesteps,
        show_cameras=show_cameras,
        save_video=False,
        debug_predictions=False,
        close_gripper_on_rollout_start=close_gripper_on_start,
    )

    writer = LeRobotDatasetWriter(
        Path(output_root),
        task_name,
        save_wrist_vis=save_wrist_vis,
        wrist_vis_fps=wrist_vis_fps,
        target_hz=target_hz,
        save_raw_too=save_raw_too,
    )
    print("=" * 70)
    print(f"[Collector] Writing rollouts to: {writer.split_root}")
    print(
        f"[Collector] Resuming at episode_index={writer.next_episode_index}, "
        f"global index={writer.next_global_index}"
    )
    print(f"[Collector] Control frequency: {control_frequency} Hz")
    if target_hz > 0:
        print(f"[Collector] Resampling to {target_hz} Hz before parquet write"
              + ("  (raw copy under data_raw/)" if save_raw_too else ""))
    else:
        print("[Collector] Resampling DISABLED -- writing raw variable-dt frames")
    print(f"[Collector] Server: {server_url}")
    if save_wrist_vis:
        print(f"[Collector] Wrist vis: {writer.vis_dir} ({wrist_vis_fps} FPS)")
    if start_joint_low_values is not None and start_joint_high_values is not None:
        print("[Collector] Randomized pre-rollout joint start enabled:")
        print(f"  low:  {start_joint_low}")
        print(f"  high: {start_joint_high}")
    if post_rollout_home_joints is not None:
        print(
            "[Collector] Interactive randomized home enabled after continue choice: "
            f"+/-{post_rollout_home_jitter:.3f} around {post_rollout_home_base}"
        )
    print("=" * 70)

    runner = FrankaPolicyRunner(config, server_url)
    current_buffer: Optional[EpisodeBuffer] = None
    control = _RolloutControl()

    def on_frame(payload: Dict[str, Any]) -> None:
        if current_buffer is not None:
            current_buffer.add(
                image_rgb=payload["external_img_rgb"],
                wrist_rgb=payload["wrist_img_rgb"],
                left_rgb=payload["left_img_rgb"],
                state=payload["raw_state"],
                action=payload["policy_action"],
                step_time=payload["step_start_time"],
            )
        ch = _poll_keypress(control.cbreak_fd)
        if ch is None:
            return
        c = ch.lower()
        if c in ("s", "f", "d"):
            control.stop_requested = True
            control.auto_label = c  # type: ignore[assignment]
            runner._early_stop_requested = True
            label_name = {"s": "SUCCESS", "f": "FAILURE", "d": "DISCARD"}[c]
            print(f"\n  [Keyboard] '{c}' pressed → stop + label {label_name}")
        elif c == "q":
            control.stop_requested = True
            control.auto_label = None
            runner._early_stop_requested = True
            print("\n  [Keyboard] 'q' pressed → stop (will prompt for label)")

    runner._on_frame = on_frame

    rollouts_saved = 0
    try:
        runner.setup()

        while True:
            if 0 < num_rollouts <= rollouts_saved:
                print(f"[Collector] Reached requested num_rollouts={num_rollouts}; stopping.")
                break

            print("\n" + "=" * 70)
            prompt_suffix = f" [default: {default_prompt}]" if default_prompt else ""
            raw = input(f"Enter task instruction{prompt_suffix} (or 'quit' to exit): ").strip()
            instruction = _resolve_instruction(raw, default_prompt)
            if instruction is None and raw.lower() in ("quit", "exit", "q"):
                break
            if not instruction:
                print("Please enter a valid instruction.")
                continue

            if start_joint_low_values is not None and start_joint_high_values is not None:
                start_joints = _sample_uniform_range(
                    start_joint_low_values,
                    start_joint_high_values,
                    start_joint_rng,
                )
                _move_to_start_joints(
                    runner,
                    start_joints,
                    start_joint_move_time,
                    start_joint_settle_time,
                    label="randomized start",
                )

            current_buffer = EpisodeBuffer(instruction=instruction)
            control.stop_requested = False
            control.auto_label = None
            print(
                "  [Collector] Rollout running. Keys during rollout: "
                "[s]=stop+save success, [f]=stop+save failure, "
                "[d]=stop+discard, [q]=stop and ask."
            )
            try:
                with _cbreak_stdin() as fd:
                    control.cbreak_fd = fd
                    try:
                        runner.run_rollout(instruction)
                    finally:
                        control.cbreak_fd = None
            except KeyboardInterrupt:
                print("\n  [Collector] Rollout interrupted by user (Ctrl+C).")

            if current_buffer.length == 0:
                print("  [Collector] No frames captured; nothing to label or save.")
                current_buffer = None
                continue

            if control.auto_label is not None:
                label: Literal["s", "f", "d"] = control.auto_label
                print(f"  [Collector] Using keypress label: {label}")
            else:
                label = _prompt_label()

            if label == "d":
                print(f"  [Collector] Discarded episode ({current_buffer.length} frames).")
            else:
                success = label == "s"
                ep_idx = writer.write_episode(current_buffer, success=success)
                rollouts_saved += 1
                print(
                    f"  [Collector] Saved episode_{ep_idx:06d} "
                    f"({current_buffer.length} frames, success={success}). "
                    f"Total this session: {rollouts_saved}."
                )

            current_buffer = None

            if num_rollouts <= 0:
                while True:
                    cont = input("\nRun another rollout? (y/n): ").strip().lower()
                    if cont in ("y", "yes"):
                        continue_rollouts = True
                        break
                    if cont in ("n", "no"):
                        continue_rollouts = False
                        break
                    print("Please enter y or n.")

                if post_rollout_home_joints is not None:
                    home_context = "next rollout" if continue_rollouts else "exit"
                    while True:
                        home_choice = input(f"Home before {home_context}? (y/n): ").strip().lower()
                        if home_choice in ("y", "yes", "h", "home"):
                            home_joints = _sample_jittered_joints(
                                post_rollout_home_joints,
                                post_rollout_home_jitter,
                                post_rollout_home_rng,
                            )
                            _move_to_start_joints(
                                runner,
                                home_joints,
                                post_rollout_home_time,
                                0.0,
                                label="randomized home",
                            )
                            break
                        if home_choice in ("n", "no"):
                            break
                        print("Please enter y or n.")

                if not continue_rollouts:
                    break

    finally:
        runner._on_frame = None
        try:
            runner.cleanup()
        except Exception as e:
            print(f"  [WARNING] runner.cleanup() failed: {e}")
        print(
            f"\n[Collector] Session complete. Saved {rollouts_saved} episode(s) "
            f"to {writer.split_root}"
        )


if __name__ == "__main__":
    tyro.cli(main)
