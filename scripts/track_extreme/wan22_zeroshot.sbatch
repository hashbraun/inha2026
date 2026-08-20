#!/bin/bash
#SBATCH --job-name=wan22_zs
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-113
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/wan22_zs_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/wan22_zs_%j.err
set -e
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home1/sota/.cache/huggingface
NV_LIB=$(find /home1/sota/anaconda3/envs/inha2026_bw/lib/python3.11/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIB}${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=/home1/sota/inha2026/submission_kit/fresh/input_videos_wan22_zeroshot
/home1/sota/anaconda3/envs/inha2026_bw/bin/python \
    /home1/sota/inha2026/scripts/track_extreme/wan22_zeroshot.py \
    --samples 5 --out-dir $OUT --steps 30 --guidance 5.0
