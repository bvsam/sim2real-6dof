# sim2real-6d

Sim to Real 6 degrees of freedom (6DOF) pose estimation.

## Running (WSL)

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
