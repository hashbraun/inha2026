#!/bin/bash
#SBATCH --job-name=dz_v5_ft
#SBATCH --nodelist=gpu-112
#SBATCH --gres=gpu:A6000_ada:2
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/home1/sota/inha2026/logs/dz_v5_ft_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/dz_v5_ft_%j.err

# v5 변경사항:
#   1. joint loss: dit path에서 action_noise_pred_joint를 L_action에 포함
#      (이전: standalone pass로 분리 → action_encoder에 간접 gradient만)
#      (이후: joint attention 경로로 direct gradient)
#   2. sigma 전체 범위 (0.0~1.0): shift=5 스케줄이 알아서 저노이즈에 집중
#   3. lr 1e-4: LoRA 표준 학습률
#   4. grad_accum 8: gradient 안정화 (메모리 추가 없음)
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
mkdir -p /home1/sota/inha2026/checkpoints/dreamzero_v5_lora

cd /home1/sota/inha2026

$PYTHON $SCRIPT \
    --data-root       /home1/sota/inha2026/data/train \
    --local-lora-path /home1/sota/.cache/huggingface/hub/models--Vizuara--dreamzero-so101-lora/snapshots/831baf3c8c04a238c981b77a539e93f7417e48e1 \
    --out-dir         /home1/sota/inha2026/checkpoints/dreamzero_v5_lora \
    --num-steps       10000 \
    --lr              1e-4 \
    --lambda-dit      1.0 \
    --lambda-action   1.0 \
    --sigma-low       0.0 \
    --sigma-high      1.0 \
    --grad-accum      8 \
    --max-clips       20000 \
    --save-every      2000 \
    --embodiment-id   26

echo ""
echo "=== Done: $(date) ==="
