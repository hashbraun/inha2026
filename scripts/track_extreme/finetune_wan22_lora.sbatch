#!/bin/bash
#SBATCH --job-name=wan22_lora
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-113
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/wan22_lora_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/wan22_lora_%j.err
set -e
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home1/sota/.cache/huggingface
NV_LIB=$(find /home1/sota/anaconda3/envs/inha2026_bw/lib/python3.11/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIB}${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home1/sota/anaconda3/envs/inha2026_bw/bin/python \
    /home1/sota/inha2026/scripts/track_extreme/finetune_wan22_lora.py \
    --steps 500 \
    --lr 1e-4 \
    --rank 16 \
    --save-every 100 \
    --ckpt-dir /home1/sota/inha2026/checkpoints/wan22_lora \
    --max-episodes 500 \
    --seed 0
