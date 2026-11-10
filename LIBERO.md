# MetaWAM in the LIBERO Simulation Benchmark

(The environment, datasets and evaluation protocol are the same as in [Cosmos Policy](https://github.com/NVlabs/cosmos-policy), whose instructions are in turn adapted from [OpenVLA-OFT](https://github.com/moojink/openvla-oft).)

## Relevant Files

Evaluation
* `cosmos_policy/experiments/robot/libero/`: LIBERO eval files
  * `run_libero_eval.py`: LIBERO (and LIBERO-Plus) eval script
  * `libero_utils.py`: LIBERO eval utils
  * `summarize_libero_plus.py`: Merges sharded LIBERO-Plus logs into a per-category success table
* `cosmos_policy/experiments/robot/`: General eval utils files
  * `cosmos_utils.py`: Model loading, meta-frame readout (actions, future proprio, value) and other eval utils
  * `robot_utils.py`: Other eval utils
* `cosmos_policy/datasets/save_libero_plus_t5_text_embeddings.py`: Precomputes T5 embeddings of the LIBERO-Plus instructions
* `scripts/test_libero.sh`: Runs the LIBERO evaluation protocol (4 suites x 3 seeds)
* `scripts/test_libero_plus.sh`: Runs the sharded LIBERO-Plus evaluation and its summary

Training
* `cosmos_policy/scripts/train.py`: Training script (NOTE: Do NOT use other train.py scripts that may exist elsewhere in the codebase for base video models)
* `scripts/train_libero.sh`: Launches training with the paper setup
* `scripts/convert_checkpoint.sh`: Converts a sharded training checkpoint into a single `.pt` file for evaluation
* `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`: Experiment configs `metawam_libero` (training) and `metawam_libero__inference` (evaluation)


## Setup

First, follow the instructions here: [SETUP.md](SETUP.md). Start and enter the Docker container.

Then, inside the Docker container, run the command below to install dependencies needed for LIBERO:
```bash
uv sync --extra cu128 --group libero --python 3.10
```

On a headless machine, render MuJoCo with EGL:
```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Only if EGL does not find the NVIDIA driver (e.g., inside some containers):
export __EGL_VENDOR_LIBRARY_DIRS=$(pwd)/egl_vendor.d
```

### Downloading Modified LIBERO Datasets
(Optional, if you plan to launch training) MetaWAM is trained on the same LIBERO data as Cosmos Policy: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal and LIBERO-10 in one superset, plus the rollouts used for world-model and value learning. To download it, run the command below. This step is optional since we provide a pretrained MetaWAM checkpoint.
```bash
# Downloads to a local directory called `LIBERO-Cosmos-Policy`
hf download nvidia/LIBERO-Cosmos-Policy --repo-type dataset --local-dir LIBERO-Cosmos-Policy
# Set the current base datasets directory as BASE_DATASETS_DIR (needed for training)
export BASE_DATASETS_DIR=$(pwd)
```

## Launching LIBERO Evaluations

We trained MetaWAM on four LIBERO task suites altogether in one run: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 (also called LIBERO-Long). Below is the pretrained checkpoint:
* [Ethanliang99/MetaWAM-LIBERO-Predict2-2B](https://huggingface.co/Ethanliang99/MetaWAM-LIBERO-Predict2-2B): `MetaWAM-LIBERO-Predict2-2B.pt`, `libero_dataset_statistics.json`, `libero_t5_embeddings.pkl`

The dataset statistics and T5 embedding files are identical to the ones in [nvidia/Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B).

Download it once; the commands below use these local paths:
```bash
hf download Ethanliang99/MetaWAM-LIBERO-Predict2-2B --local-dir checkpoints/MetaWAM-LIBERO-Predict2-2B
```

### Full benchmark

`scripts/test_libero.sh` evaluates the four suites (10 tasks x 50 trials each) with seeds {195, 196, 197}, one run per suite and seed. Run it from the repository root:
```bash
bash scripts/test_libero.sh
```

It is configured with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `CKPT_DIR` | `checkpoints/MetaWAM-LIBERO-Predict2-2B` | Local checkpoint folder |
| `CKPT_PATH` | `$CKPT_DIR/MetaWAM-LIBERO-Predict2-2B.pt` | Model checkpoint; set it to evaluate your own converted `.pt` |
| `DATASET_STATS_PATH`, `T5_EMBEDDINGS_PATH` | files in `$CKPT_DIR` | Dataset statistics and T5 embeddings |
| `SUITES` | `libero_spatial libero_object libero_goal libero_10` | Suites to evaluate |
| `SEEDS` | `195 196 197` | Seeds to evaluate |
| `NUM_TRIALS_PER_TASK` | `50` | Trials per task |
| `GPU_ID` | `0` | GPU to run on |
| `LOG_DIR` | `logs/libero` | Where logs and rollout videos are written |
| `RUN_NAME` | `metawam_libero` | Prefix of the run id |

### Single evaluation command

The script runs the command below for each suite and seed:
```bash
uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
    --config metawam_libero__inference \
    --ckpt_path checkpoints/MetaWAM-LIBERO-Predict2-2B/MetaWAM-LIBERO-Predict2-2B.pt \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path checkpoints/MetaWAM-LIBERO-Predict2-2B/libero_dataset_statistics.json \
    --t5_text_embeddings_path checkpoints/MetaWAM-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 16 \
    --num_open_loop_steps 16 \
    --task_suite_name libero_10 \
    --num_trials_per_task 50 \
    --local_log_dir logs/libero \
    --run_id_note metawam_libero--5stepAct--seed195--deterministic \
    --seed 195 \
    --randomize_seed False \
    --deterministic True \
    --use_variance_scale False \
    --ar_future_prediction False \
    --ar_value_prediction False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --data_collection False
```

To use the local download instead, pass the files directly:
```bash
    --ckpt_path checkpoints/MetaWAM-LIBERO-Predict2-2B/MetaWAM-LIBERO-Predict2-2B.pt \
    --dataset_stats_path checkpoints/MetaWAM-LIBERO-Predict2-2B/libero_dataset_statistics.json \
    --t5_text_embeddings_path checkpoints/MetaWAM-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
```

Notes:
* `--task_suite_name` is one of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`. Each run evaluates 500 trials (10 tasks x 50 episodes); change the number of trials per task with `--num_trials_per_task`.
* `--ckpt_path` accepts either a Hugging Face repo id (`<org>/<repo>`, the `.pt` file inside is downloaded) or a local `.pt` file. A relative path with exactly two components (e.g. `checkpoints/MetaWAM-LIBERO-Predict2-2B`) is interpreted as a repo id, so point it at the `.pt` file. `--dataset_stats_path` and `--t5_text_embeddings_path` accept a local file or `<org>/<repo>/<file>`. Set `HF_HOME` to change where downloads are cached.
* The `--seed` and `--deterministic` arguments matter if you want to reproduce our results: we used seeds {195, 196, 197} with `--deterministic True` and report the average over the three seeds (see the [README](README.md) for the numbers). Results may vary slightly with a different PyTorch version or different hardware.
* Logs are written to `--local_log_dir` and rollout videos to `--rollout_video_dir` (default: `./rollouts/<date>`). You can also log results in Weights & Biases by setting `--use_wandb True` and specifying `--wandb_entity <ENTITY>` and `--wandb_project <PROJECT>`.

## Launching LIBERO-Plus Evaluations

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) perturbs the four LIBERO suites (camera viewpoint, robot initial state, language, lighting, background, sensor noise and object layout), giving thousands of tasks per suite. We evaluate the same MetaWAM-LIBERO checkpoint without fine-tuning, one episode per task.

**1. Install LIBERO-Plus.** Clone it, install the system packages it needs and download its assets as described in its README. Do not `pip install -e` it: `scripts/test_libero_plus.sh` puts the checkout on `PYTHONPATH`, where it shadows the `libero` package of the `libero` dependency group, and sets `LIBERO_CONFIG_PATH=$HOME/.libero_plus` so that LIBERO-Plus does not overwrite the LIBERO config. Install its extra Python requirements into this environment (again after any `uv sync`, which removes them):
```bash
git clone https://github.com/sylvestf/LIBERO-plus.git
export LIBERO_PLUS_ROOT=$(pwd)/LIBERO-plus
uv pip install -r $LIBERO_PLUS_ROOT/extra_requirements.txt
```

**2. Precompute the T5 embeddings of the LIBERO-Plus instructions.** The released `libero_t5_embeddings.pkl` only covers the 40 original LIBERO instructions; missing instructions would otherwise be encoded with T5-11B during evaluation. The command below writes one cache per suite, `libero_plus_t5_embeddings__<suite>.pkl`, to `--out_dir` (`--gpus` is a comma-separated list of GPUs to split the encoding across):
```bash
PYTHONPATH=$LIBERO_PLUS_ROOT LIBERO_CONFIG_PATH=$HOME/.libero_plus \
uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.datasets.save_libero_plus_t5_text_embeddings \
    --out_dir liberoplus_t5 \
    --gpus 0 \
    --merge_pkl checkpoints/MetaWAM-LIBERO-Predict2-2B/libero_t5_embeddings.pkl
export LIBERO_PLUS_T5_DIR=$(pwd)/liberoplus_t5
```

**3. Run the sharded evaluation and the summary.** Each suite is split into `NUM_SHARDS` shards (`--task_num_shards N --task_shard_index i`) that run in parallel, assigned round-robin to the GPUs in `GPUS`; several shards can share a GPU. Once all suites are done, `summarize_libero_plus.py` merges the shard logs and prints the success rate per perturbation category and suite.
```bash
GPUS="0 1 2 3" NUM_SHARDS=8 bash scripts/test_libero_plus.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `LIBERO_PLUS_ROOT` | (required) | LIBERO-Plus checkout |
| `LIBERO_PLUS_T5_DIR` | (required) | Directory with `libero_plus_t5_embeddings__<suite>.pkl` |
| `CKPT_DIR`, `CKPT_PATH`, `DATASET_STATS_PATH` | as in `test_libero.sh` | Checkpoint and dataset statistics |
| `SUITES` | `libero_spatial libero_object libero_goal libero_10` | Suites to evaluate |
| `SEED` | `195` | Seed |
| `GPUS` | `0` | Space-separated GPU ids |
| `NUM_SHARDS` | `8` | Parallel processes per suite |
| `LOG_DIR` | `logs/libero_plus` | Eval logs (`ENV_EVAL-*.txt`), per-shard stdout (`stdout/`) and rollout videos |
| `RUN_TAG` | `metawam_liberoplus--seed$SEED` | Identifies the run in the log names; the summary only reads logs containing it |
| `RESUME_DIR` | (empty) | Set to `$LOG_DIR/task_ids` to evaluate only the unfinished tasks of a previous run with the same `RUN_TAG` |
| `LIBERO_PLUS_PROMPT_MODE` | `clean_nonlanguage` | Language conditioning, see below |

Notes:
* The summary also writes `$LOG_DIR/$RUN_TAG.json` and, under `$LOG_DIR/task_ids/`, the task ids of each suite that have no result yet. Rerunning with `RESUME_DIR=$LOG_DIR/task_ids` evaluates only those; each task is counted once, so the partial logs of both runs are merged. To rerun the summary alone:
  ```bash
  PYTHONPATH=$LIBERO_PLUS_ROOT uv run --extra cu128 --group libero --python 3.10 \
    python -m cosmos_policy.experiments.robot.libero.summarize_libero_plus \
      --log_dir logs/libero_plus --libero_plus_root $LIBERO_PLUS_ROOT --run_tag metawam_liberoplus--seed195
  ```
* `LIBERO_PLUS_PROMPT_MODE` selects which instruction the model is conditioned on. `clean_nonlanguage` (default, used for the paper) keeps the rewritten instruction for the language perturbations and restores the original LIBERO instruction for the other six categories, so that only the language category perturbs the text condition. `raw` uses the LIBERO-Plus instruction as is for every task. `clean_nonlanguage` reads the original instructions from a local `libero_t5_embeddings.pkl` (`LIBERO_PLUS_CANONICAL_PROMPTS_PATH`, default `$CKPT_DIR/libero_t5_embeddings.pkl`).
* If you run shards by hand, include `--sh<i>of<N>--` in `--run_id_note` (e.g. `--run_id_note metawam_liberoplus--seed195--sh0of8--deterministic`) so that `summarize_libero_plus.py` can merge the logs.
* Every process loads the model and its suite's T5 cache onto the GPU, so choose `NUM_SHARDS` per GPU according to GPU memory.

## Training on LIBERO Datasets

First, download the LIBERO datasets following the instructions above.

Then, launch the training script from the repository root:
```bash
# Set BASE_DATASETS_DIR to the directory containing the LIBERO-Cosmos-Policy dataset
export BASE_DATASETS_DIR=/PATH/TO/BASE/DATASETS/DIRECTORY  # E.g., `/home/user/data/` if `LIBERO-Cosmos-Policy` is in this directory

bash scripts/train_libero.sh
```

This fine-tunes [nvidia/Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) (downloaded automatically from Hugging Face on the first launch) with the `metawam_libero` experiment config. The defaults match the paper setup: 4 GPUs, per-GPU batch size 75 and 4 gradient accumulation steps, i.e., an effective batch size of 75 * 4 * 4 = 1200. The script reads these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `NUM_GPUS` | `4` | GPUs per node (`torchrun --nproc_per_node`) |
| `GRAD_ACCUM` | `4` | Gradient accumulation steps (`trainer.grad_accum_iter`) |
| `MASTER_PORT` | `12341` | torchrun master port |
| `IMAGINAIRE_OUTPUT_ROOT` | `./output` | Root directory for checkpoints and logs |

Any extra arguments are appended as config overrides. For example, `bash scripts/train_libero.sh dataloader_train.batch_size=32` sets the per-GPU batch size to 32; if you lower it to fit in memory, raise `GRAD_ACCUM` to keep the effective batch size. Training logs to Weights & Biases by default: run `wandb login` first, or append `job.wandb_mode=disabled`.

Checkpoints are saved every 600 iterations as sharded (DCP) checkpoints under `$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy/metawam/metawam_libero/checkpoints/iter_XXXXXXXXX/`. Relaunching the same command resumes from the latest one. Training runs until you stop it; the released checkpoint is iteration 10,200. You can modify the experiment in `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py` or override any field on the command line.

To evaluate a checkpoint, first convert it into a single `.pt` file, then pass that file to the evaluation script:
```bash
bash scripts/convert_checkpoint.sh metawam_libero__inference \
  output/cosmos_policy/metawam/metawam_libero/checkpoints/iter_000010200/model \
  checkpoints/metawam_libero_10200.pt

CKPT_PATH=checkpoints/metawam_libero_10200.pt bash scripts/test_libero.sh
```

The released `libero_dataset_statistics.json` applies if you trained on the unmodified dataset. Otherwise, use the statistics written by your training run, `DATASET_STATS_PATH=$BASE_DATASETS_DIR/LIBERO-Cosmos-Policy/success_only/dataset_statistics.json`. Please be sure to test your policy with the same device/GPU used to train it! Otherwise, performance may drop substantially.
