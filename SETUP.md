# Setup Guide

This guide walks you through setting up MetaWAM using Docker. The environment is the same as for Cosmos Policy.

## System Requirements

* NVIDIA GPU with CUDA support
* Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
* Linux x86-64

## Installation

### Build the Docker Image

From the project root, build the image:

```bash
docker build -t cosmos-policy docker
```

### Launch the Docker Container

Start an interactive session:

```bash
docker run \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/home/cosmos/.cache \
  -v $(pwd):/workspace \
  --gpus all \
  --ipc=host \
  -it \
  --rm \
  -w /workspace \
  --entrypoint bash \
  cosmos-policy
```

**Optional arguments:**
* `-v $HOME/.cache:/home/cosmos/.cache`: Reuses host cache for `uv`, `huggingface`, etc.
* `--ipc=host`: Shares host's inter-process communication namespace. Required for PyTorch's parallel data loading to avoid out-of-memory errors (containers get only 64MB shared memory by default). If your security policy doesn't allow this, try using `--shm-size 32g` instead to allocate sufficient isolated shared memory.

### Dependency Groups

Python dependencies are managed with `uv`. Each benchmark has its own dependency group, installed together with the `cu128` extra:
* `robocasa`: RoboCasa evaluation, and training for RoboCasa and the real-world tasks
* `libero`: LIBERO and LIBERO-Plus evaluation and training
* `aloha`: the real-robot policy server (FastAPI); see [REALWORLD.md](REALWORLD.md)

These groups conflict with each other, so `uv` re-syncs the environment whenever you switch groups. If you need two of them at the same time (e.g., training while serving a policy), give each its own environment, for example by setting `UV_PROJECT_ENVIRONMENT=.venv-libero` for one of them.

## Running an Example

Once you are inside the container, you can run the evaluation and training commands for LIBERO, RoboCasa and the real robot (see [LIBERO.md](LIBERO.md), [ROBOCASA.md](ROBOCASA.md) and [REALWORLD.md](REALWORLD.md)).
