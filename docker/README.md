# SMAC CUDA Container

This container is for SMAC experiments on A100/H200 class GPUs. It uses a CUDA
12.1 PyTorch image so the PyTorch build includes newer GPU architectures than
the legacy local `torch==1.10.2+cu102` environment.

Build the Docker image where Docker is available:

```bash
docker build -f Dockerfile.smac -t mat-smac:cuda12 .
```

Docker is mainly for producing the image. On Slurm/HPC, prefer Apptainer with
`--nv`; it exposes the GPU allocation given by Slurm instead of using Docker's
`--gpus all`.

Optional local Docker smoke test:

```bash
./mat/scripts/run_yaml_docker.sh inspect_enemy_jitter
```

By default the Docker helper uses one GPU:

```bash
DOCKER_GPUS=device=0 ./mat/scripts/run_yaml_docker.sh inspect_enemy_jitter
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

The Slurm script requests one GPU:

```bash
#SBATCH -G 1
```

Inside the job, `apptainer exec --nv` should only expose the GPU assigned by
Slurm. Do not use Docker `--gpus all` inside Slurm unless the cluster explicitly
requires Docker and scopes devices itself.

The StarCraft II installation is not copied into the image. The scripts bind
the repository into the container and use:

```bash
SC2PATH=/workspace/Multi-Agent-Transformer/3rdparty/StarCraftII
```
