# Per-environment Docker images

Each simulation environment has its own image, dependency lock, and
Dockerfile, so that conflicting dependencies (old `gym`, `jax`, `mujoco-py`,
...) never have to share a single environment:

| Env | Dockerfile | Deps |
|---|---|---|
| SMAC (StarCraft II) | `docker/Dockerfile.smac` | `docker/smac/{pyproject.toml,uv.lock}` |
| Football (gfootball) | `docker/Dockerfile.football` | `docker/football/{pyproject.toml,uv.lock}` |
| MPE | `docker/Dockerfile.mpe` | `docker/mpe/{pyproject.toml,uv.lock}` |
| VMAS | `docker/Dockerfile.vmas` | `docker/vmas/{pyproject.toml,uv.lock}` |
| Robotarium | `docker/Dockerfile.robotarium` | `docker/robotarium/{pyproject.toml,uv.lock}` |
| JaxMARL-Robotarium | `docker/Dockerfile.jaxmarl-robotarium` | `docker/jaxmarl-robotarium/{pyproject.toml,uv.lock}` |
| multiagent_mujoco (mujoco-py) | `docker/Dockerfile.ma-mujoco` | `docker/ma-mujoco/{pyproject.toml,uv.lock}` |

MARBLER and DexterousHandEnvs do not have images yet (MARBLER depends on an
unpublished local patch of `robotarium_python_simulator`; DexterousHandEnvs
needs Isaac Gym, which is EULA-gated and not pip-installable). See
`docs/ci-experiment-management-plan.md` for the status of the wider CI/infra
plan.

## Base image and dependency install

Every image is built `FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` so
the CUDA 12.1 / PyTorch 2.3.1 layer is shared and cached across images. Each
Dockerfile installs its environment's dependencies with:

```dockerfile
COPY docker/<env>/pyproject.toml docker/<env>/uv.lock /tmp/<env>-project/
RUN cd /tmp/<env>-project \
    && uv export --frozen --no-hashes -o requirements.lock.txt \
    && uv pip install --system --no-cache -r requirements.lock.txt
```

`uv pip install --system` installs straight into the base image's existing
conda env instead of creating a separate uv-managed venv (`uv sync` would
otherwise discard the preinstalled torch/CUDA build). Torch itself is
intentionally left out of every `pyproject.toml` so it's never reinstalled —
except `docker/vmas/pyproject.toml`, which pins `torch==2.3.1` explicitly
because `vmas` itself declares torch as a dependency and would otherwise pull
a newer, mismatched CUDA build.

To add or update a dependency: edit the env's `pyproject.toml`, then run
`(cd docker/<env> && uv lock)` to refresh its `uv.lock`, and rebuild.

## Building an image

```bash
docker build -f docker/Dockerfile.<env> -t mat-<env>:cuda12 .
```

Build from the repository root (not `docker/`) — the Dockerfiles `COPY`
paths relative to the repo root, and the SMAC/Robotarium/JaxMARL-Robotarium
images need `3rdparty/` (see below) in the build context.

## `3rdparty/` submodules

`robotarium`, `jaxmarl-robotarium` read `3rdparty/robotarium_python_simulator`
and `3rdparty/JaxMARL-Robotarium` (both git submodules) directly off disk at
runtime, not as pip packages. Initialize them before building or running:

```bash
git submodule update --init 3rdparty/robotarium_python_simulator 3rdparty/JaxMARL-Robotarium
```

`3rdparty/StarCraftII` is a ~5GB binary install, not a submodule — fetch it
with `install_sc2.sh` (or bind-mount an existing install) rather than baking
it into the image; see below.

## Running a container

```bash
docker run --rm --gpus all \
  -v "$(pwd)":/workspace/Multi-Agent-Transformer \
  -w /workspace/Multi-Agent-Transformer/mat/scripts \
  mat-<env>:cuda12 \
  python train/train_<env>.py --env_name <...> ...
```

Or via the YAML launcher: `python mat/scripts/run_yaml.py <config>` inside
the container (see `mat/scripts/configs/README.md`).

SMAC additionally needs `SC2PATH` and a StarCraft II install bind-mounted in:

```bash
docker run --rm --gpus all \
  -e SC2PATH=/workspace/Multi-Agent-Transformer/3rdparty/StarCraftII \
  -v "$(pwd)":/workspace/Multi-Agent-Transformer \
  -w /workspace/Multi-Agent-Transformer \
  mat-smac:cuda12 \
  python mat/scripts/run_yaml.py <config>
```

JaxMARL-Robotarium needs a couple of JAX env vars set (also wired into its
YAML configs under `mat/scripts/configs/marbler/`):

```bash
-e XLA_PYTHON_CLIENT_PREALLOCATE=false -e JAX_PLATFORMS=cuda
```

**Known issue**: JaxMARL-Robotarium's vendored barrier-certificate code
(`3rdparty/JaxMARL-Robotarium/.../rps_jax/utilities/barrier_certificates2.py`)
hits `XlaRuntimeError: cuSolver internal error` on `jnp.linalg.inv` on this
cluster's V100 + driver 570.158.01 + jaxlib 0.4.38 combination. Reproduced
across GPUs, not a memory-contention issue. Training script requires the CUDA
backend (won't fall back to CPU). Unresolved — likely a jaxlib/cuSolver/driver
compatibility issue in the vendored dependency, not in this repo's own code.

## Slurm / Apptainer

Docker is mainly for producing images locally — Slurm compute nodes have no
Docker daemon access, only Apptainer. Apptainer consumes the same Docker
images directly, so nothing needs rebuilding from scratch. Build a `.sif`
from a Docker image on shironagasu (where Docker is available) into
`docker/sif/` (gitignored — these are multi-GB binaries, not repo content):

```bash
apptainer build docker/sif/mat-<env>_cuda12.sif docker-daemon://mat-<env>:cuda12
```

Then submit, e.g. for SMAC:

```bash
CONTAINER_IMAGE=/path/to/mat-smac_cuda12.sif sbatch mat/scripts/run_yaml_apptainer_slurm.sh <run-name>
```

`mat/scripts/run_yaml_apptainer_slurm.sh` requests one GPU (`#SBATCH -G 1`)
and runs `apptainer exec --nv`, which only exposes the GPU Slurm assigned. Do
not use Docker `--gpus all` inside Slurm unless the cluster explicitly
requires Docker and scopes devices itself.

**Verified**: `docker/sif/mat-vmas_cuda12.sif` ran `mat/scripts/configs/_ci_smoke/vmas_smoke.yaml`
to completion on an actual GPU node (`koku`, dgx-a100-80g) via
`apptainer exec --nv`.

**Known gotcha**: Apptainer bind-mounts `$HOME` into the container by
default, so Python's user-site mechanism picks up whatever is installed in
the *host's* `~/.local/lib/python3.10/site-packages` (e.g. a newer numpy)
ahead of the image's own pinned versions — this silently broke wandb's
import (`np.float_` was removed in numpy 2.0) even though the image itself
pins `numpy==1.26.4`. Fixed by passing `--env PYTHONNOUSERSITE=1`, which
`run_yaml_apptainer_slurm.sh` and `.github/workflows/train-on-push.yml` both
set.

To make a `slurm:`-mode YAML config use a container instead of the default
native `.venv-cu121` execution, point it at the Apptainer script:

```yaml
slurm:
  script: mat/scripts/run_yaml_apptainer_slurm.sh
```

`.github/workflows/train-on-push.yml` resolves `CONTAINER_IMAGE` for you
from the config's `script:` the same way it resolves the Docker tag for
direct-mode runs — you don't need to set it by hand for a push-triggered
run, only for a manual `sbatch`.

`mat/scripts/run_yaml_docker.sh` remains a local Docker smoke-test helper
(defaults to the SMAC image; override `IMAGE`/`CONFIG_PATH` for other envs).
