#!/bin/bash
#SBATCH --job-name=dmd
#SBATCH --nodelist=gpu-113
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=90G
#SBATCH --time=15:00:00
#SBATCH --output=/home1/sota/inha2026/logs/dmd_%x_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/dmd_%x_%j.err

# DMD (Direct-Metric Distillation) — Agent B Top 1 (10/10)
#
# v5_base 40k에서 warm-start. 채점기 3종(R3D-18, DINOv2, ActionExtractor) 직접 감독:
#   L = L_flow + λ_dino·L_dino + λ_r3d·L_r3d + λ_act·L_action
#
# 이유: 지금까지 실패한 loss들이 홀드아웃 무신호였던 이유는 eval 채점기 자체를 loss로 안 썼기 때문.
# 채점기 3종이 오픈소스이므로 latent flow가 감독 못 하던 픽셀·특징·관절 공간을 직접 감독.
#
# 사용:
#   sbatch --job-name=dmd_screen_8k --export=ALL,TAG=screen_8k,STEPS=8000 scripts/train_dmd.sbatch
#   sbatch --job-name=dmd_prod_40k  --export=ALL,TAG=prod_40k,STEPS=40000 scripts/train_dmd.sbatch

set -e
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home1/sota/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${TAG:?TAG 필수}"
: "${STEPS:=8000}"
: "${LR:=5e-5}"
: "${DINO_W:=0.5}"
: "${R3D_W:=0.5}"
: "${ACTION_W:=1.0}"
RESUME=/home1/sota/inha2026/checkpoints/cosmos3_nano_v5_base/ckpt_step040000.pt
CKPT_DIR=/home1/sota/inha2026/checkpoints/dmd_${TAG}

echo "[$(date)] DMD 학습: tag=$TAG steps=$STEPS lr=$LR"
echo "  loss weights: dino=$DINO_W r3d=$R3D_W action=$ACTION_W"
echo "  resume: $RESUME"
echo "  ckpt: $CKPT_DIR"

/home1/sota/anaconda3/envs/inha2026/bin/python /home1/sota/inha2026/scripts/finetune_cosmos3_nano.py \
    --steps "$STEPS" --lr "$LR" --proj-lr-mult 5.0 --rank 32 --grad-acc 4 \
    --max-episodes 11132 --save-every 2000 \
    --action-repr delta_base --seed 0 \
    --resume "$RESUME" \
    --action-loss-weight "$ACTION_W" \
    --dino-loss-weight "$DINO_W" --r3d-loss-weight "$R3D_W" \
    --perc-sigma-min 0 --perc-sigma-max 0.7 --perc-no-damp \
    --diag-log "/home1/sota/inha2026/diag/dmd_${TAG}_train.json" \
    --diag-every 100 --diag-grad-every 500 \
    --ckpt-dir "$CKPT_DIR"

echo "[$(date)] DMD 학습 완료: $TAG"
