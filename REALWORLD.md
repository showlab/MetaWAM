# MetaWAM on a Real Franka Robot

This page covers the real-robot pipeline: collecting data, training, and deploying MetaWAM on a Franka Panda. In the paper we evaluate three tasks, and a single policy is trained jointly on all of them:

| Task | Instruction |
|---|---|
| Place Cup | `Place the cup on the green coaster` |
| Build Block | `Place the orange block on top of the gray block` |
| Sweep | `Sweep the green block into the dustpan` |

The pipeline is:
1. Record teleoperated demonstrations and store them in LeRobot (parquet) format.
2. Optionally deploy a policy and collect on-policy rollouts (successes and failures) for world-model and value learning, then retime them.
3. Assemble the training dataset and precompute the T5 embeddings of the instructions.
4. Train with `metawam_realworld` and convert the checkpoint to a single `.pt` file.
5. Run the policy server on a GPU machine and the robot client on the robot PC.

## Relevant Files

Robot side and deployment (`cosmos_policy/experiments/robot/franka/`)
* `deploy_franka.py`, `run_deploy_franka.sh`: Policy server (GPU machine); FastAPI app with `POST /act` and `GET /health`
* `run_franka_eval.py`, `run_franka_eval.sh`: Robot client; reads the cameras, queries the server and executes the actions
* `collect_rollouts.py`, `run_collect_rollouts.sh`: Robot client that records labeled policy rollouts in LeRobot format
* `retime_existing_rollouts.py`: Rewrites rollout timestamps onto a fixed 15 Hz grid without interpolation
* `resample_rollout.py`: Optional online resampling used by `collect_rollouts.py` when `--target-hz > 0`
* `camera_utils.py`, `franka_interface.py`, `config.py`: RealSense cameras, Franka ZeroRPC client, client configuration
* `requirements_franka.txt`: Extra dependencies for the server and the robot PC

Training
* `cosmos_policy/datasets/realworld_lerobot_dataset.py`: LeRobot-format dataset loader
* `cosmos_policy/datasets/save_t5_text_embeddings_from_texts.py`: Precomputes T5 embeddings for a list of instructions
* `scripts/train_realworld.sh`, `scripts/convert_checkpoint.sh`: Training and checkpoint conversion
* `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`: Experiment configs `metawam_realworld` (training) and `metawam_realworld__inference` (deployment)

## Hardware and System Layout

The code assumes the following setup:
* **Robot:** Franka Panda with the Franka Hand gripper, controlled through [Polymetis](https://github.com/facebookresearch/fairo/tree/main/polymetis) on the robot's real-time PC (NUC). The client connects to a ZeroRPC server on the NUC (default port 4242) that wraps Polymetis and exposes the methods called in `franka_interface.py` (`get_ee_pose`, `get_joint_positions`, `start_cartesian_impedance`, `update_desired_ee_pose`, `move_to_joint_positions`, `get_gripper_position`, `set_gripper_position`, `control_gripper`, ...). This server is not part of the repository.
* **Cameras:** three Intel RealSense cameras connected to the robot PC, captured at 640x480 and resized to 224x224 for the policy:
  * external camera behind the robot, facing the workspace: primary view (parquet column `image`)
  * second third-person camera in front of the robot: secondary view (parquet column `left_image`)
  * wrist camera facing downwards: wrist view (parquet column `wrist_image`)
* **Control:** 15 Hz end-effector control. The policy outputs chunks of 16 actions `[dx, dy, dz, droll, dpitch, dyaw, gripper]` that are accumulated into target poses and tracked with Cartesian impedance control.
* **Machines:** the robot PC runs the client (cameras, robot connection, operator keyboard), and a GPU machine runs the policy server. They communicate over HTTP (default port 8777).

```
robot PC (cameras + client) --ZeroRPC :4242--> Franka NUC (Polymetis)
robot PC (cameras + client) --HTTP    :8777--> GPU machine (policy server)
```

## Installation

**GPU machine.** Follow [SETUP.md](SETUP.md). Training uses the `robocasa` dependency group (as in `scripts/train_realworld.sh`). The policy server uses the `aloha` group: it is inherited from upstream Cosmos Policy (where it served the ALOHA robot) and provides the server dependencies `fastapi`, `uvicorn`, `json_numpy` and `draccus`. `run_deploy_franka.sh` uses it by default (`FRANKA_UV_GROUP`):
```bash
uv sync --extra cu128 --group aloha --python 3.10
```
The `robocasa`, `libero` and `aloha` groups conflict with each other, so uv re-syncs the environment when you switch between them (see [SETUP.md](SETUP.md)). Instead of the `aloha` group, you can also install the server part of `cosmos_policy/experiments/robot/franka/requirements_franka.txt` into an existing environment.

**Robot PC.** Clone this repository; the client scripts are run from the repository root. Then use either uv:
```bash
uv sync --group aloha --python 3.10
uv pip install -r cosmos_policy/experiments/robot/franka/requirements_franka.txt
```
or a plain virtualenv/conda environment:
```bash
pip install numpy torch torchvision tqdm Pillow opencv-python scipy pyarrow tyro \
  -r cosmos_policy/experiments/robot/franka/requirements_franka.txt
```
The robot-side packages in `requirements_franka.txt` are `pyrealsense2` (cameras), `zerorpc` (robot), `requests` (policy server) and `moviepy` (optional rollout videos). The client launchers use `python` from the active conda environment or virtualenv if there is one, otherwise `.venv/bin/python`, otherwise `uv run --group aloha` (`FRANKA_CLIENT_UV_GROUP`). Note that a later `uv sync` removes packages installed with `uv pip`.

List the serial numbers of the connected RealSense cameras with:
```bash
python -m cosmos_policy.experiments.robot.franka.camera_utils
```

## 1. Data

### LeRobot format

Every dataset split read by the loader is a *split root*: a directory with `meta/episodes.jsonl` and `data/`, which contains one parquet file per episode named `episode_<episode_index>.parquet` (e.g. `data/chunk-000/episode_000000.parquet`). Each parquet row is one control step at 15 Hz:

| Column | Type | Content |
|---|---|---|
| `image`, `left_image`, `wrist_image` | struct `{bytes, path}` | JPEG-encoded primary, secondary and wrist images |
| `state` | float32[8] | End-effector pose `[x, y, z, roll, pitch, yaw]` and the gripper width split into `[w/2, -w/2]` |
| `actions` | float32[7] | End-effector delta `[dx, dy, dz, droll, dpitch, dyaw]` and the gripper command in [0, 1] (1 = open) |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | | Standard LeRobot bookkeeping |
| `raw_dt_ms` | float32 | (Rollouts only) wall-clock time between recorded steps |

Each line of `meta/episodes.jsonl` describes one episode, e.g. `{"episode_index": 0, "tasks": ["Place the cup on the green coaster"], "length": 211, "success": true}`. `meta/tasks.jsonl` lists the instructions (`{"task_index": 0, "task": "..."}`) and `meta/info.json` holds the LeRobot dataset info. The files written by `collect_rollouts.py` and `retime_existing_rollouts.py` follow this format.

### Teleoperated demonstrations

The repository does not include a teleoperation tool. Record demonstrations with your own teleoperation setup and export them in the format above, using the same cameras, state and action conventions as the client (`run_franka_eval.py`).

### Policy rollouts (optional)

MetaWAM also learns its world model and value function from policy rollouts. To collect them, train a first policy on the demonstrations (see below), start the policy server, and run the collector on the robot PC:
```bash
export FRANKA_NUC_IP=<NUC_IP>
export FRANKA_EXTERNAL_CAMERA=<SERIAL> FRANKA_LEFT_CAMERA=<SERIAL> FRANKA_WRIST_CAMERA=<SERIAL>
export FRANKA_SERVER_IP=127.0.0.1 FRANKA_SERVER_PORT=8777

FRANKA_TASK_NAME=cup FRANKA_OUTPUT_ROOT=lerobot_rollouts \
  bash cosmos_policy/experiments/robot/franka/run_collect_rollouts.sh
```
During a rollout, single keypresses stop the episode and label it: `s` saves it as a success, `f` saves it as a failure, `d` discards it, and `q` asks for the label interactively. If `FRANKA_MAX_TIMESTEPS` (default 600) is reached first, you are prompted for the label. Episodes are written to `$FRANKA_OUTPUT_ROOT/$FRANKA_TASK_NAME/` with a `success` flag in `meta/episodes.jsonl`.

`FRANKA_TASK_NAME` (`cup`, `build` or `sweep`) also selects a preset: the default instruction (used when you press Enter at the prompt, override with `FRANKA_DEFAULT_PROMPT`), the range from which the start joint configuration is sampled before each rollout (`FRANKA_START_JOINT_LOW`/`FRANKA_START_JOINT_HIGH`), the joint configuration the robot returns to afterwards (`FRANKA_POST_ROLLOUT_HOME_BASE`), and a lower z limit for the end-effector targets (`FRANKA_MIN_TARGET_Z`). Other useful variables are `FRANKA_NUM_ROLLOUTS` (default -1, i.e., until you quit), `FRANKA_OPEN_LOOP_HORIZON` (actions executed per policy query, at most 16) and `FRANKA_CONTROL_FREQUENCY` (default 15 Hz).

By default (`FRANKA_TARGET_HZ=0`), the collector stores one row per executed control step with the real wall-clock timestamps, which include the pauses at chunk boundaries while the server runs inference. Retime the rollouts offline onto the 15 Hz grid of the demonstrations; this keeps every executed step (no interpolation), sets `timestamp = frame_index / 15`, and keeps the measured step duration in `raw_dt_ms` (any environment with numpy and pyarrow works):
```bash
python -m cosmos_policy.experiments.robot.franka.retime_existing_rollouts \
  --source-root lerobot_rollouts \
  --output-root lerobot_rollouts_retimed
```
During training, rollout windows with more than two steps longer than 200 ms are skipped (`rollout_max_raw_dt_ms` and `rollout_max_stalls_per_window` in the dataset config).

### Training dataset

The training dataset `REALWORLD_DATASET_DIR` has two parts:
```
$REALWORLD_DATASET_DIR/
├── success_only/              # teleoperated demonstrations (policy learning)
│   ├── meta/                  # episodes.jsonl, tasks.jsonl, info.json
│   ├── data/
│   └── t5_embeddings.pkl
└── all_episodes/              # world-model and value learning
    ├── success/               # demonstrations and successful rollouts
    └── failed/                # failed rollouts
```
* `success_only/` and `all_episodes/` can each be a split root themselves or contain split roots one level down. For example, you can copy the task folders of the retimed rollouts directly into `all_episodes/`.
* Episode success is read from the `success` field in `meta/episodes.jsonl`. Without it, episodes under a folder named `failed`/`failure` are failures and all others are successes.
* As in Cosmos Policy, the value targets are discounted Monte-Carlo returns (gamma = 0.99, rescaled to [-1, 1]) for successful episodes and -1 at every step for failed episodes.
* `all_episodes/` must exist. Before you have rollouts, it can simply contain the demonstrations.

In the paper, `success_only/` holds 80 teleoperated demonstrations per task, and `all_episodes/` holds the demonstrations plus the retimed policy rollouts.

### T5 embeddings

The instructions are encoded once with T5-11B (downloaded from Hugging Face on first use). Write one instruction per line and save the embeddings next to the demonstrations:
```bash
cat > realworld_tasks.txt << 'EOF'
Place the cup on the green coaster
Place the orange block on top of the gray block
Sweep the green block into the dustpan
EOF

uv run --extra cu128 --group robocasa --python 3.10 \
  python -m cosmos_policy.datasets.save_t5_text_embeddings_from_texts \
    --texts_path realworld_tasks.txt \
    --output_path $REALWORLD_DATASET_DIR/success_only/t5_embeddings.pkl
```
The file must contain every instruction in the dataset. At deployment, type the instructions exactly as they appear in this file; an instruction that is not in the file is encoded with T5-11B on the server at query time.

## 2. Training

```bash
export REALWORLD_DATASET_DIR=/path/to/franka_lerobot_dataset
bash scripts/train_realworld.sh
```

`metawam_realworld` trains one policy on every task in `REALWORLD_DATASET_DIR`. It fine-tunes [nvidia/Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) with the same 8-latent-frame layout as RoboCasa (input meta-frame, wrist, primary and secondary views, output meta-frame, three future views) and 16-step action chunks. The defaults match the paper setup: 4 GPUs, per-GPU batch size 64 and 4 gradient accumulation steps (effective batch size 1024). `NUM_GPUS`, `GRAD_ACCUM`, `MASTER_PORT`, `IMAGINAIRE_OUTPUT_ROOT` and extra config overrides work as described in [ROBOCASA.md](ROBOCASA.md#training-on-robocasa-datasets).

On the first launch, the loader computes the normalization statistics of the demonstrations and writes them to `$REALWORLD_DATASET_DIR/success_only/dataset_statistics.json`. The policy server needs this file.

Checkpoints are saved every 600 iterations under `$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy/metawam/metawam_realworld/checkpoints/iter_XXXXXXXXX/`. Convert the one you want to deploy into a single `.pt` file:
```bash
bash scripts/convert_checkpoint.sh metawam_realworld__inference \
  output/cosmos_policy/metawam/metawam_realworld/checkpoints/iter_XXXXXXXXX/model \
  checkpoints/metawam_realworld.pt
```

## 3. Deployment

### Policy server (GPU machine)

From the repository root:
```bash
export FRANKA_CKPT_PATH=checkpoints/metawam_realworld.pt
export FRANKA_DATASET_STATS_PATH=$REALWORLD_DATASET_DIR/success_only/dataset_statistics.json
export FRANKA_T5_EMBEDDINGS_PATH=$REALWORLD_DATASET_DIR/success_only/t5_embeddings.pkl
bash cosmos_policy/experiments/robot/franka/run_deploy_franka.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `FRANKA_CKPT_PATH`, `FRANKA_DATASET_STATS_PATH`, `FRANKA_T5_EMBEDDINGS_PATH` | (required) | Converted checkpoint, dataset statistics and T5 embeddings of the training run |
| `FRANKA_CONFIG` | `metawam_realworld__inference` | Inference config |
| `FRANKA_SERVER_HOST`, `FRANKA_SERVER_PORT` | `127.0.0.1`, `8777` | Address the server binds to |
| `FRANKA_DEBUG_PREDICTIONS` | `True` | Return the predicted future images, proprio and value along with the actions (a client can also request them per query) |
| `FRANKA_NUM_QUERIES_BEST_OF_N` | `1` | Best-of-N planning: sample N action chunks and execute the one with the highest predicted value |
| `FRANKA_UV_GROUP`, `FRANKA_UV_EXTRA` | `aloha`, `cu128` | uv dependency group / extra (`FRANKA_USE_UV=0` uses `.venv/bin/python` or `python` instead) |

Further planning options (`FRANKA_AR_FUTURE_PREDICTION`, `FRANKA_AR_VALUE_PREDICTION`, `FRANKA_AR_QVALUE_PREDICTION`, `FRANKA_SEARCH_DEPTH`, `FRANKA_PLANNING_MODEL_CONFIG_NAME`, `FRANKA_PLANNING_MODEL_CKPT_PATH`) are documented in `run_deploy_franka.sh`, and extra arguments are forwarded to `deploy_franka.py`.

The server binds to `127.0.0.1` by default. Either forward the port from the robot PC,
```bash
ssh -N -L 8777:localhost:8777 <user>@<gpu-machine>   # on the robot PC; then use FRANKA_SERVER_IP=127.0.0.1
```
or set `FRANKA_SERVER_HOST=0.0.0.0` on a trusted network and use `FRANKA_SERVER_IP=<gpu-machine-ip>` on the robot PC. Check that the server is up with `curl http://127.0.0.1:8777/health`.

### Robot client (robot PC)

From the repository root:
```bash
export FRANKA_NUC_IP=<NUC_IP>
export FRANKA_EXTERNAL_CAMERA=<SERIAL> FRANKA_LEFT_CAMERA=<SERIAL> FRANKA_WRIST_CAMERA=<SERIAL>
export FRANKA_SERVER_IP=127.0.0.1 FRANKA_SERVER_PORT=8777

FRANKA_TASK_NAME=cup bash cosmos_policy/experiments/robot/franka/run_franka_eval.sh
```
The client asks for an instruction (Enter uses the preset's default instruction), moves the robot to a start configuration sampled from the preset's range, and runs the policy until `FRANKA_MAX_TIMESTEPS` (default 600) is reached or you press Ctrl+C; it then asks whether to run another episode. Task success is judged by the operator.

| Variable | Default | Meaning |
|---|---|---|
| `FRANKA_NUC_IP`, `FRANKA_NUC_PORT` | `127.0.0.1`, `4242` | ZeroRPC server on the NUC |
| `FRANKA_EXTERNAL_CAMERA`, `FRANKA_LEFT_CAMERA`, `FRANKA_WRIST_CAMERA` | (empty) | RealSense serial numbers (primary, secondary, wrist) |
| `FRANKA_SERVER_IP`, `FRANKA_SERVER_PORT` | `127.0.0.1`, `8777` | Policy server |
| `FRANKA_TASK_NAME` | `sweep` | Preset (`cup`, `build`, `sweep`): default instruction, start joint range and z limits |
| `FRANKA_OPEN_LOOP_HORIZON` | `16` | Actions executed per policy query (at most the chunk size, 16) |
| `FRANKA_MAX_TIMESTEPS` | `600` | Maximum control steps per episode |
| `FRANKA_SAVE_VIDEO`, `FRANKA_SHOW_CAMERAS` | `False`, `False` | Save a rollout video / show the camera feeds |
| `FRANKA_DEBUG_PREDICTIONS` | `True` | Save the server's predicted future images, proprio and value to `FRANKA_DEBUG_LOG_DIR` (default `franka_debug_predictions`) |

The remaining variables in `run_franka_eval.sh` control the gripper logic and the workspace safety limits (`FRANKA_MIN_TARGET_Z`, `FRANKA_MAX_TARGET_POSITION_LEAD`, ...); check them against your workspace before running on hardware. Extra arguments are forwarded to `run_franka_eval.py`.

To test the server and the client without hardware, use mock cameras and a mock robot:
```bash
bash cosmos_policy/experiments/robot/franka/run_franka_eval.sh --use-mock-cameras --use-mock-robot
```
