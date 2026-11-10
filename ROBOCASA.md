# MetaWAM in the RoboCasa Simulation Benchmark

(The environment, datasets and evaluation protocol are the same as in [Cosmos Policy](https://github.com/NVlabs/cosmos-policy), whose instructions are in turn adapted from [OpenVLA-OFT](https://github.com/moojink/openvla-oft).)

## Relevant Files

Evaluation
* `cosmos_policy/experiments/robot/robocasa/`: RoboCasa eval files
  * `run_robocasa_eval.py`: RoboCasa eval script
  * `robocasa_utils.py`: RoboCasa eval utils
* `cosmos_policy/experiments/robot/`: General eval utils files
  * `cosmos_utils.py`: Model loading, meta-frame readout (actions, future proprio, value) and other eval utils
  * `robot_utils.py`: Other eval utils
* `scripts/test_robocasa.sh`: Runs the full evaluation protocol (24 tasks x 50 trials x 3 seeds)

Training
* `cosmos_policy/scripts/train.py`: Training script (NOTE: Do NOT use other train.py scripts that may exist elsewhere in the codebase for base video models)
* `scripts/train_robocasa.sh`: Launches training with the paper setup
* `scripts/convert_checkpoint.sh`: Converts a sharded training checkpoint into a single `.pt` file for evaluation
* `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`: Experiment configs `metawam_robocasa` (training) and `metawam_robocasa__inference` (evaluation)


## Setup

First, follow the instructions here: [SETUP.md](SETUP.md). Start and enter the Docker container.

Next, inside the Docker container, run the command below to install dependencies needed for RoboCasa:
```bash
uv sync --extra cu128 --group robocasa --python 3.10
```

Then, clone the RoboCasa repo, install the package, and download assets:
```bash
git clone https://github.com/moojink/robocasa-cosmos-policy.git
uv pip install -e robocasa-cosmos-policy
uv run --extra cu128 --group robocasa --python 3.10 robocasa-cosmos-policy/robocasa/scripts/download_kitchen_assets.py
uv run --extra cu128 --group robocasa --python 3.10 robocasa-cosmos-policy/robocasa/scripts/setup_macros.py
```

On a headless machine, render MuJoCo with EGL:
```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Only if EGL does not find the NVIDIA driver (e.g., inside some containers):
export __EGL_VENDOR_LIBRARY_DIRS=$(pwd)/egl_vendor.d
```

### Downloading Modified RoboCasa Datasets
(Optional, if you plan to launch training) MetaWAM is trained on the same RoboCasa data as Cosmos Policy: 24 tasks with ~50 demonstrations per task, plus the policy rollouts used for world-model and value learning. To download it, run the command below. This step is optional since we provide a pretrained MetaWAM checkpoint.
```bash
# Downloads to a local directory called `RoboCasa-Cosmos-Policy`
hf download nvidia/RoboCasa-Cosmos-Policy --repo-type dataset --local-dir RoboCasa-Cosmos-Policy
# Set the current base datasets directory as BASE_DATASETS_DIR (needed for training)
export BASE_DATASETS_DIR=$(pwd)
```

## Launching RoboCasa Evaluations

We trained MetaWAM on the RoboCasa benchmark with 24 tasks and 50 demonstrations per task. Below is the pretrained checkpoint:
* [Ethanliang99/MetaWAM-RoboCasa-Predict2-2B](https://huggingface.co/Ethanliang99/MetaWAM-RoboCasa-Predict2-2B): `MetaWAM-RoboCasa-Predict2-2B.pt`, `robocasa_dataset_statistics.json`, `robocasa_t5_embeddings.pkl`

The dataset statistics and T5 embedding files are identical to the ones in [nvidia/Cosmos-Policy-RoboCasa-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B).

Download it once; the commands below use these local paths:
```bash
hf download Ethanliang99/MetaWAM-RoboCasa-Predict2-2B --local-dir checkpoints/MetaWAM-RoboCasa-Predict2-2B
```

### Full benchmark

`scripts/test_robocasa.sh` runs the evaluation protocol: all 24 tasks, 50 trials per task, seeds {195, 196, 197}. Run it from the repository root:
```bash
bash scripts/test_robocasa.sh
```

It is configured with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `CKPT_DIR` | `checkpoints/MetaWAM-RoboCasa-Predict2-2B` | Local checkpoint folder |
| `CKPT_PATH` | `$CKPT_DIR/MetaWAM-RoboCasa-Predict2-2B.pt` | Model checkpoint; set it to evaluate your own converted `.pt` |
| `DATASET_STATS_PATH`, `T5_EMBEDDINGS_PATH` | files in `$CKPT_DIR` | Dataset statistics and T5 embeddings |
| `TASK_NAME` | `all` | `all`, `pnp` (the 8 pick-and-place tasks), or a single task such as `TurnOffMicrowave` |
| `SEEDS` | `195 196 197` | Seeds to evaluate, one run per seed |
| `NUM_TRIALS_PER_TASK` | `50` | Trials per task |
| `GPU_ID` | `0` | GPU to run on |
| `NUM_PARALLEL_ENVS` | `10` | Simulation workers sharing the model on that GPU |
| `LOG_DIR` | `logs/robocasa` | Where logs and rollout videos are written |
| `RUN_NAME` | `metawam_robocasa` | Prefix of the run id and rollout folder |

For example, to evaluate a single task with a single seed:
```bash
TASK_NAME=TurnOffMicrowave SEEDS=195 bash scripts/test_robocasa.sh
```

### Single evaluation command

The script runs the command below once per seed:
```bash
uv run --extra cu128 --group robocasa --python 3.10 \
  python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
    --config metawam_robocasa__inference \
    --ckpt_path checkpoints/MetaWAM-RoboCasa-Predict2-2B/MetaWAM-RoboCasa-Predict2-2B.pt \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --num_wrist_images 1 \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path checkpoints/MetaWAM-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path checkpoints/MetaWAM-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 32 \
    --num_open_loop_steps 16 \
    --task_name all \
    --num_trials_per_task 50 \
    --run_id_note metawam_robocasa--5stepAct--seed195--deterministic \
    --local_log_dir $(pwd)/logs/robocasa \
    --seed 195 \
    --randomize_seed False \
    --deterministic True \
    --use_variance_scale False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --data_collection False \
    --rollout_prefix metawam_robocasa_seed195 \
    --num_parallel_envs 10 \
    --inference_batch_timeout 0.05
```

To use the local download instead, pass the files directly:
```bash
    --ckpt_path checkpoints/MetaWAM-RoboCasa-Predict2-2B/MetaWAM-RoboCasa-Predict2-2B.pt \
    --dataset_stats_path checkpoints/MetaWAM-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path checkpoints/MetaWAM-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
```

Notes:
* `--ckpt_path` accepts either a Hugging Face repo id (`<org>/<repo>`, the `.pt` file inside is downloaded) or a local `.pt` file. A relative path with exactly two components (e.g. `checkpoints/MetaWAM-RoboCasa-Predict2-2B`) is interpreted as a repo id, so point it at the `.pt` file. `--dataset_stats_path` and `--t5_text_embeddings_path` accept a local file or `<org>/<repo>/<file>`. Set `HF_HOME` to change where downloads are cached.
* The `--seed` and `--deterministic` arguments matter if you want to reproduce our results: we used seeds {195, 196, 197} with `--deterministic True` and report the average over the three seeds (see the [README](README.md) for the numbers). Results may vary slightly with a different PyTorch version or different hardware.
* `--num_parallel_envs N` runs N simulation worker processes that share one model on the same GPU and batches their queries. Use `--num_parallel_envs 1` to run episodes serially. It cannot be combined with best-of-N search, data collection, or autoregressive future/value prediction.
* A relative `--local_log_dir` is resolved against `cosmos_policy/experiments/robot/robocasa/`, not the current directory, so pass an absolute path (as above) or set `COSMOS_POLICY_LOG_DIR`. Rollout videos are saved to `<local_log_dir>/<rollout_prefix>/<TaskName>--<timestamp>/`. To resume an interrupted run, pass its log file via `--resume_log_path <local_log_dir>/ENV_EVAL-....txt`; finished episodes are skipped.
* The evaluation script logs results locally. You can also log results in Weights & Biases by setting `--use_wandb True` and specifying `--wandb_entity <ENTITY>` and `--wandb_project <PROJECT>`.

## Training on RoboCasa Datasets

First, download the RoboCasa datasets following the instructions above.

Then, launch the training script from the repository root:
```bash
# Set BASE_DATASETS_DIR to the directory containing the RoboCasa-Cosmos-Policy dataset
export BASE_DATASETS_DIR=/PATH/TO/BASE/DATASETS/DIRECTORY  # E.g., `/home/user/data/` if `RoboCasa-Cosmos-Policy` is in this directory

bash scripts/train_robocasa.sh
```

This fine-tunes [nvidia/Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) (downloaded automatically from Hugging Face on the first launch) with the `metawam_robocasa` experiment config. The defaults match the paper setup: 4 GPUs, per-GPU batch size 64 and 4 gradient accumulation steps, i.e., an effective batch size of 64 * 4 * 4 = 1024. The script reads these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `NUM_GPUS` | `4` | GPUs per node (`torchrun --nproc_per_node`) |
| `GRAD_ACCUM` | `4` | Gradient accumulation steps (`trainer.grad_accum_iter`) |
| `MASTER_PORT` | `12341` | torchrun master port |
| `IMAGINAIRE_OUTPUT_ROOT` | `./output` | Root directory for checkpoints and logs |

Any extra arguments are appended as config overrides. For example, `bash scripts/train_robocasa.sh dataloader_train.batch_size=32` sets the per-GPU batch size to 32; if you lower it to fit in memory, raise `GRAD_ACCUM` to keep the effective batch size. Training logs to Weights & Biases by default: run `wandb login` first, or append `job.wandb_mode=disabled`.

Checkpoints are saved every 600 iterations as sharded (DCP) checkpoints under `$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy/metawam/metawam_robocasa/checkpoints/iter_XXXXXXXXX/`. Relaunching the same command resumes from the latest one. Training runs until you stop it; the released checkpoint is iteration 8,400. You can modify the experiment in `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py` or override any field on the command line.

To evaluate a checkpoint, first convert it into a single `.pt` file, then pass that file to the evaluation script:
```bash
bash scripts/convert_checkpoint.sh metawam_robocasa__inference \
  output/cosmos_policy/metawam/metawam_robocasa/checkpoints/iter_000008400/model \
  checkpoints/metawam_robocasa_8400.pt

CKPT_PATH=checkpoints/metawam_robocasa_8400.pt bash scripts/test_robocasa.sh
```

The released `robocasa_dataset_statistics.json` applies if you trained on the unmodified dataset. Otherwise, use the statistics written by your training run, `DATASET_STATS_PATH=$BASE_DATASETS_DIR/RoboCasa-Cosmos-Policy/success_only/dataset_statistics.json`. Please be sure to test your policy with the same device/GPU used to train it! Otherwise, performance may drop substantially.
