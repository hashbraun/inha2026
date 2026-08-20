#!/bin/bash
#SBATCH --job-name=proj_norm
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=80G
#SBATCH --time=0:15:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/proj_norm_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/proj_norm_%j.err
set -e
export PYTHONUNBUFFERED=1
/home1/sota/anaconda3/envs/inha2026/bin/python /home1/sota/inha2026/scripts/action_proj_norm.py
