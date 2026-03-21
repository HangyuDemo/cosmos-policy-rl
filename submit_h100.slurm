#!/bin/bash
#SBATCH --job-name=cosmos-libero-h100
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM job id: ${SLURM_JOB_ID:-unset}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

mkdir -p outputs
mkdir -p .cache
mkdir -p logs

docker build -t cosmos-policy docker

docker run \
  --name "cosmos-policy-${SLURM_JOB_ID}" \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -e BASE_DATASETS_DIR=/workspace/data \
  -e PYTHONUNBUFFERED=1 \
  -v "$HOME/.cache:/home/cosmos/.cache" \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${PROJECT_DIR}/outputs:/workspace/outputs" \
  --gpus all \
  --ipc=host \
  -w /workspace \
  cosmos-policy \
  bash -lc '
    set -euo pipefail

    echo "Inside container at: $(date)"
    nvidia-smi

    uv run --extra cu128 --group libero --python 3.10 \
      torchrun --nproc_per_node=8 --master_port=12341 \
      -m cosmos_policy.scripts.train \
      --config=cosmos_policy/config/config.py -- \
      experiment="cosmos_predict2_2b_480p_libero" \
      trainer.grad_accum_iter=8 \
      job.path_local=/workspace/outputs
  '

echo "Job finished at: $(date)"
