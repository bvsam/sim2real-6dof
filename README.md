# sim2real-6dof

Sim to Real 6 degrees of freedom (6DOF) pose estimation.

## Running

### Environment Setup (WSL)

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

### Data Setup

Run the `setup.py` script to download and setup input datasets for data generation

```bash
uv run scripts/setup.py
```

### Data Generation

Run the `generata_data.py` script with `blenderproc run`

```bash
uv run blenderproc run src/sim2real_6dof/generate_data.py -c configs/config.yaml
```
