#!/bin/bash
#SBATCH --job-name=v5_seed
#SBATCH --nodelist=gpu-113
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/home1/sota/inha2026/logs/seed_%x_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/seed_%x_%j.err

# Multi-seed 앙상블: v5_base + delta_base + gs=5.0 + steps=50, seed만 변경
# 5 seeds (0/1/2/3/4) 전부 만들면 frame-wise median으로 앙상블
#
# 사용:
#   sbatch --job-name=seed1 --export=ALL,SEED=1 scripts/submit_v5base_multiseed.sbatch
#   sbatch --job-name=seed2 --export=ALL,SEED=2 scripts/submit_v5base_multiseed.sbatch
#   sbatch --job-name=seed3 --export=ALL,SEED=3 scripts/submit_v5base_multiseed.sbatch
#   sbatch --job-name=seed4 --export=ALL,SEED=4 scripts/submit_v5base_multiseed.sbatch

set -e
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home1/sota/.cache/huggingface

: "${SEED:?SEED required}"
CKPT=/home1/sota/inha2026/checkpoints/cosmos3_nano_v5_base/ckpt_step040000.pt
OUT=/home1/sota/inha2026/submission_kit/input_videos_v5base_seed${SEED}

echo "[$(date)] multi-seed 추론: seed=$SEED"
/home1/sota/anaconda3/envs/inha2026/bin/python /home1/sota/inha2026/scripts/infer_cosmos3_nano.py \
    --ckpt "$CKPT" --samples 216 --steps 50 --guidance 5.0 \
    --rank 32 --action-repr delta_base --seed "$SEED" \
    --out-dir "$OUT"

echo "[$(date)] 완료 → $OUT"
