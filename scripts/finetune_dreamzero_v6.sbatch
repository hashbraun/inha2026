#!/bin/bash
#SBATCH --job-name=dz_v6_ft
#SBATCH --nodelist=gpu-112
#SBATCH --gres=gpu:A6000_ada:2
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/home1/sota/inha2026/logs/dz_v6_ft_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/dz_v6_ft_%j.err

# v6 변경사항 (v5 대비):
#   1. Flash decoupled noise (--flash-noise):
#      - video sigma: Beta(7,1) 독립 샘플 → E[σ]≈0.875, 항상 고노이즈
#      - action sigma: U(0,1) 독립 샘플 → 전 범위 균등
#      - 두 값이 서로 무관 → DiT가 video 복원 시 clean action token 의존 강제
#   2. 나머지 (joint loss, SO-101 가중치, sigma 범위, lr, grad_accum)는 v5와 동일
#      - sigma-low/high는 flash 모드에서 video용으로는 무시되고 action 초기화에만 사용
set -e
PYTHON=/home1/sota/anaconda3/envs/inha2026/bin/python
SCRIPT=/home1/sota/inha2026/scripts/finetune_dreamzero_full.py

export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Job: $SLURM_JOB_ID | Node: $(hostname) | $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

mkdir -p /home1/sota/inha2026/logs
mkdir -p /home1/sota/inha2026/checkpoints/dreamzero_v6_lora

cd /home1/sota/inha2026

$PYTHON $SCRIPT \
    --data-root       /home1/sota/inha2026/data/train \
    --local-lora-path /home1/sota/.cache/huggingface/hub/models--Vizuara--dreamzero-so101-lora/snapshots/831baf3c8c04a238c981b77a539e93f7417e48e1 \
    --out-dir         /home1/sota/inha2026/checkpoints/dreamzero_v6_lora \
    --num-steps       10000 \
    --lr              1e-4 \
    --lambda-dit      1.0 \
    --lambda-action   1.0 \
    --sigma-low       0.0 \
    --sigma-high      1.0 \
    --flash-noise \
    --grad-accum      8 \
    --max-clips       20000 \
    --save-every      2000 \
    --embodiment-id   26

echo ""
echo "=== Done: $(date) ==="
