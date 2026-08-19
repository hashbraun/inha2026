#!/bin/bash
#SBATCH --job-name=pareto_batch
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/pareto_batch_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/pareto_batch_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home1/sota/anaconda3/envs/inha2026/bin/python \
  /home1/sota/inha2026/scripts/track_extreme/pareto_batch.py "$@"
