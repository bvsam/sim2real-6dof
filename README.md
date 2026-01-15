# sim2real-6dof <!-- omit from toc -->

Sim to Real 6 degrees of freedom (6DOF) category level pose estimation.

Based off the paper [Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation](https://arxiv.org/abs/1901.02970) by H. Wang et al.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
    - [Environment Setup (WSL)](#environment-setup-wsl)
    - [Data Setup](#data-setup)
  - [Running](#running)
    - [Synthetic Data Generation](#synthetic-data-generation)
    - [Model Training](#model-training)

## Usage

### Prerequisites

- Ensure [Docker](https://www.docker.com/) is installed, along with [Docker Compose](https://docs.docker.com/compose/install)

### Setup

#### Environment Setup (WSL)

1. If not already done, run the compose file to start things up

```bash
docker compose up -d
```

2. Enter the running container for development

```bash
docker compose exec blender bash
```

3. Once temporarily done, stop the running container

```bash
docker compose stop
```

4. Resume the running container later using

```bash
docker compose start
```

5. Once completely done, end the running container and remove everything

```bash
docker compose down
```

#### Data Setup

Run the `setup.py` script to download and setup input datasets for data generation

```bash
uv run scripts/setup.py
```

### Running

#### Synthetic Data Generation

Run the `generata_data.py` script with `blenderproc run`

```bash
uv run blenderproc run src/sim2real_6dof/generate_data.py -c configs/config.yaml
```

#### Model Training

Run the `train` project script with `uv run`

```bash
uv run train \
  --repo-id bvsam/sim2real-6dof \
  --output-dir training_output \
  --batch-size 16 \
  --num-workers 2 \
  --prefetch-factor 4 \
  --checkpoint-interval 100 \
  --log-interval 1 \
  --stage-epochs 1 1 10 \
  --stage-lrs 0.008 0.0008 0.0008 \
  --stage-freezes 5 4 3 \
  --device cuda
```
