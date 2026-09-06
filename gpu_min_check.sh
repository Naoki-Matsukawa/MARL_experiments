#!/bin/bash
#SBATCH -p dgx-a100-40g
#SBATCH -w makkou
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 00:03:00
#SBATCH -J gpu-min-check
#SBATCH -o dump/gpu-min-check.%j.out
#SBATCH -e dump/gpu-min-check.%j.err

hostname
date

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
echo "SLURM_GPUS=$SLURM_GPUS"

echo "--- nvidia-smi ---"
nvidia-smi
echo "--- nvidia-smi -L ---"
nvidia-smi -L

echo "--- device files ---"
ls -l /dev/nvidia* || true

echo "--- CUDA compiler/runtime ---"
which nvcc || true
nvcc --version || true

echo "--- Python CUDA libs if any ---"
python3 - <<'PY'
import ctypes
for lib in ["libcuda.so", "libcuda.so.1", "libcudart.so", "libnvidia-ml.so.1"]:
    try:
        ctypes.CDLL(lib)
        print(lib, "OK")
    except OSError as e:
        print(lib, "NG", e)
PY
