"""Retime raw on-policy rollouts to a fixed-Hz grid without interpolation.

This keeps exactly one row per real executed control step, rewrites
``timestamp = frame_index / target_hz``, clamps the gripper action channel to
``[0, 1]``, and preserves the original wall-clock gap in ``raw_dt_ms`` so the
dataloader can filter windows that crossed inference stalls.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATE_DIM = 8
ACTION_DIM = 7

RETIMED_SCHEMA = pa.schema(
    [
        pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
        pa.field("wrist_image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
        pa.field("left_image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
        pa.field("state", pa.list_(pa.float32(), STATE_DIM)),
        pa.field("actions", pa.list_(pa.float32(), ACTION_DIM)),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("raw_dt_ms", pa.float32()),
    ]
)


def _image_struct(jpeg_bytes: bytes) -> dict:
    return {"bytes": jpeg_bytes, "path": None}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_episodes(path: Path) -> dict[int, dict]:
    return {int(r["episode_index"]): r for r in _load_jsonl(path)}


def _load_tasks(path: Path) -> dict[int, str]:
    return {int(r["task_index"]): str(r["task"]) for r in _load_jsonl(path)}


def _read_raw_episode(path: Path) -> dict:
    table = pq.read_table(path)
    actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
    if actions.shape[-1] >= 7:
        actions[:, 6] = np.clip(actions[:, 6], 0.0, 1.0)
    raw_ts = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    if raw_ts.shape[0] >= 2:
        diffs = (np.diff(raw_ts) * 1000.0).astype(np.float32)
        raw_dt_ms = np.concatenate([diffs, diffs[-1:]]).astype(np.float32)
    else:
        raw_dt_ms = np.zeros((raw_ts.shape[0],), dtype=np.float32)
    return {
        "images": [r["bytes"] for r in table["image"].to_pylist()],
        "wrist_images": [r["bytes"] for r in table["wrist_image"].to_pylist()],
        "left_images": [r["bytes"] for r in table["left_image"].to_pylist()],
        "states": np.asarray(table["state"].to_pylist(), dtype=np.float32),
        "actions": actions,
        "raw_timestamps": raw_ts,
        "raw_dt_ms": raw_dt_ms,
        "task_index": int(table["task_index"][0].as_py()),
    }


def _write_retimed_parquet(
    out_path: Path,
    ep: dict,
    episode_index: int,
    task_index: int,
    global_index_start: int,
    target_hz: float,
) -> None:
    T = ep["states"].shape[0]
    timestamps = (np.arange(T, dtype=np.float32) / float(target_hz)).astype(np.float32)
    table = pa.table(
        {
            "image": [_image_struct(b) for b in ep["images"]],
            "wrist_image": [_image_struct(b) for b in ep["wrist_images"]],
            "left_image": [_image_struct(b) for b in ep["left_images"]],
            "state": [s.tolist() for s in ep["states"]],
            "actions": [a.tolist() for a in ep["actions"]],
            "timestamp": pa.array(timestamps.tolist(), type=pa.float32()),
            "frame_index": pa.array(list(range(T)), type=pa.int64()),
            "episode_index": pa.array([episode_index] * T, type=pa.int64()),
            "index": pa.array(list(range(global_index_start, global_index_start + T)), type=pa.int64()),
            "task_index": pa.array([task_index] * T, type=pa.int64()),
            "raw_dt_ms": pa.array(ep["raw_dt_ms"].tolist(), type=pa.float32()),
        },
        schema=RETIMED_SCHEMA,
    )
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(out_path)


def _process_task(src_task: Path, dst_task: Path, target_hz: float, copy_vis: bool) -> dict:
    src_data = src_task / "data" / "chunk-000"
    if not src_data.exists():
        return {"skipped": True, "reason": f"no data dir at {src_data}"}

    src_eps = _load_episodes(src_task / "meta" / "episodes.jsonl")
    src_tasks = _load_tasks(src_task / "meta" / "tasks.jsonl")
    dst_data = dst_task / "data" / "chunk-000"
    dst_meta = dst_task / "meta"
    dst_vis = dst_task / "vis"
    dst_data.mkdir(parents=True, exist_ok=True)
    dst_meta.mkdir(parents=True, exist_ok=True)

    episode_records = []
    out_tasks: dict[int, str] = {}
    global_index = 0
    total_frames = 0
    n_eps = 0
    n_failed = 0

    for ep_path in sorted(src_data.glob("episode_*.parquet")):
        try:
            ep_idx = int(ep_path.stem.split("_")[-1])
            raw = _read_raw_episode(ep_path)
            task_index = int(raw["task_index"])
            instruction = src_tasks.get(task_index, f"task_{task_index}")
            _write_retimed_parquet(
                dst_data / ep_path.name,
                raw,
                episode_index=ep_idx,
                task_index=task_index,
                global_index_start=global_index,
                target_hz=target_hz,
            )
        except Exception as e:
            print(f"  [skip] {ep_path.name}: {e}")
            n_failed += 1
            continue

        T = int(raw["states"].shape[0])
        rec = dict(src_eps.get(ep_idx, {}))
        rec["episode_index"] = ep_idx
        rec["tasks"] = [instruction]
        rec["length"] = T
        rec["retimed_target_hz"] = float(target_hz)
        rec["raw_length"] = T
        rec["raw_duration_s"] = float(raw["raw_timestamps"][-1] - raw["raw_timestamps"][0]) if T > 1 else 0.0
        rec["retime_note"] = "one row per real executed step; timestamp=frame_index/fps; no interpolation"
        episode_records.append(rec)
        out_tasks[task_index] = instruction

        stall_frac = float((raw["raw_dt_ms"] > 200.0).mean() * 100.0) if T else 0.0
        print(
            f"  ep {ep_idx:3d}: T={T:4d}, raw_dur={rec['raw_duration_s']:.1f}s "
            f"-> retime_dur={(T - 1) / target_hz:.1f}s, stall_rows={stall_frac:.1f}%"
        )
        global_index += T
        total_frames += T
        n_eps += 1

    with (dst_meta / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for rec in sorted(episode_records, key=lambda r: r["episode_index"]):
            f.write(json.dumps(rec) + "\n")

    with (dst_meta / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for ti in sorted(out_tasks):
            f.write(json.dumps({"task_index": ti, "task": out_tasks[ti]}) + "\n")

    info = {
        "generated_by": "retime_existing_rollouts.py",
        "source_root": str(src_task),
        "task_name": dst_task.name,
        "last_updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "fps": float(target_hz),
        "num_episodes": n_eps,
        "total_frames": total_frames,
        "retime_policy": "no interpolation; one row per real executed control step; timestamp=frame_index/fps; gripper action clipped to [0,1]",
    }
    (dst_meta / "info.json").write_text(json.dumps(info, indent=2))

    src_vis = src_task / "vis"
    if copy_vis and src_vis.exists():
        dst_vis.mkdir(parents=True, exist_ok=True)
        for p in src_vis.iterdir():
            if p.is_file():
                shutil.copy2(p, dst_vis / p.name)

    return {"skipped": False, "n_eps": n_eps, "n_failed": n_failed, "total_frames": total_frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-hz", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-vis", action="store_true", default=True)
    parser.add_argument("--no-copy-vis", dest="copy_vis", action="store_false")
    args = parser.parse_args()

    src = args.source_root.resolve()
    dst = args.output_root.resolve()
    if not src.exists():
        raise SystemExit(f"source root does not exist: {src}")
    if src == dst:
        raise SystemExit("source and output roots must differ")
    if dst.exists() and any(dst.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output root is non-empty: {dst}; pass --overwrite")
        print(f"[wipe] {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[source] {src}")
    print(f"[output] {dst}")
    print(f"[target] {args.target_hz} Hz")
    total_eps = 0
    total_frames = 0
    for src_task in [p for p in sorted(src.iterdir()) if p.is_dir()]:
        print(f"\n[task] {src_task.name}")
        result = _process_task(src_task, dst / src_task.name, args.target_hz, args.copy_vis)
        if result.get("skipped"):
            print(f"  [skipped] {result.get('reason')}")
            continue
        total_eps += result["n_eps"]
        total_frames += result["total_frames"]
        if result["n_failed"]:
            print(f"  [warn] failed episodes: {result['n_failed']}")
    print(f"\n[done] wrote {total_eps} episodes / {total_frames} frames to {dst}")


if __name__ == "__main__":
    main()
