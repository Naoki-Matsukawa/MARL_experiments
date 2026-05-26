# SMAC CUDA Container

This container is for SMAC experiments on A100/H200 class GPUs. It uses a CUDA
12.1 PyTorch image so the PyTorch build includes newer GPU architectures than
the legacy local `torch==1.10.2+cu102` environment.

Build locally:

```bash
docker build -f Dockerfile.smac -t mat-smac:cuda12 .
```

Run a quick YAML experiment through Docker:

```bash
./mat/scripts/run_yaml_docker.sh inspect_enemy_jitter
```

Run the main experiment:

```bash
./mat/scripts/run_yaml_docker.sh partial_enemy_jitter
```

For Slurm clusters that use Apptainer/Singularity, build a `.sif` from the
Docker image where Docker is available:

```bash
apptainer build mat-smac_cuda12.sif docker-daemon://mat-smac:cuda12
```

Then submit:

```bash
CONTAINER_IMAGE=/path/to/mat-smac_cuda12.sif sbatch mat/scripts/run_yaml_apptainer_slurm.sh partial_enemy_jitter
```

The StarCraft II installation is not copied into the image. The scripts bind
the repository into the container and use:

```bash
SC2PATH=/workspace/Multi-Agent-Transformer/3rdparty/StarCraftII
```
