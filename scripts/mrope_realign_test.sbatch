#!/bin/bash
#SBATCH --job-name=mrope_realign
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/mrope_realign_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/mrope_realign_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home1/sota/anaconda3/envs/inha2026/bin/python /home1/sota/inha2026/scripts/mrope_realign_test.py
