#!/bin/bash
#SBATCH --job-name=p25_diag
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=80G
#SBATCH --time=0:30:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/p25_diag_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/p25_diag_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home1/sota/anaconda3/envs/inha2026/bin/python /home1/sota/inha2026/scripts/predict25_bridge_diag.py
