#!/bin/bash
#SBATCH --job-name=wan_pilot12
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=100G
#SBATCH --time=1:30:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/wan_pilot_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/wan_pilot_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1
OUT=/home1/sota/inha2026/submission_kit/fresh/input_videos_wan22_pilot
/home1/sota/anaconda3/envs/inha2026/bin/python \
    /home1/sota/inha2026/scripts/track_extreme/wan_pilot.py \
    --samples 12 --out-dir $OUT --steps 30 --guidance 5.0
