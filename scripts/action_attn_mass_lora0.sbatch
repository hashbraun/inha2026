#!/bin/bash
#SBATCH --job-name=mass_lora0
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=100G
#SBATCH --time=0:20:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/mass_lora0_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/mass_lora0_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home1/sota/anaconda3/envs/inha2026/bin/python /home1/sota/inha2026/scripts/action_attn_mass_lora0.py
