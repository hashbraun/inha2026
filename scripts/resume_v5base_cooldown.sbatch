#!/bin/bash
#SBATCH --job-name=v5b_resume
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/sota/inha2026/logs/resume_%x_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/resume_%x_%j.err

# v5_base 40k ckpt에서 이어 학습. 낮은 LR로 SGD noise 감소 (EMA-like cooldown).
#
# 사용:
#   sbatch --job-name=v5b_cooldown_8k --export=ALL,TAG=cooldown_8k,STEPS=8000,LR=1e-5 --nodelist=gpu-108 scripts/resume_v5base_cooldown.sbatch
#   sbatch --job-name=v5b_long_20k --export=ALL,TAG=long_20k,STEPS=20000,LR=5e-5 --nodelist=gpu-109 scripts/resume_v5base_cooldown.sbatch

set -e
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home1/sota/.cache/huggingface

: "${TAG:?TAG 필수}"
: "${STEPS:=8000}"
: "${LR:=1e-5}"
RESUME=/home1/sota/inha2026/checkpoints/cosmos3_nano_v5_base/ckpt_step040000.pt
CKPT_DIR=/home1/sota/inha2026/checkpoints/v5b_${TAG}

echo "[$(date)] v5_base 이어 학습: tag=$TAG steps=$STEPS lr=$LR"
echo "  resume: $RESUME"
echo "  ckpt_dir: $CKPT_DIR"

/home1/sota/anaconda3/envs/inha2026/bin/python /home1/sota/inha2026/scripts/finetune_cosmos3_nano.py \
    --steps "$STEPS" --lr "$LR" --proj-lr-mult 5.0 --rank 32 --grad-acc 4 \
    --max-episodes 11132 --save-every 2000 --action-loss-weight 0 \
    --action-repr delta_base --seed 0 \
    --resume "$RESUME" \
    --ckpt-dir "$CKPT_DIR"

echo "[$(date)] 완료: $TAG"
