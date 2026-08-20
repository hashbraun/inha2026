#!/bin/bash
#SBATCH --job-name=st1_frz_shift
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-113
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/st1_frz_shift_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/st1_frz_shift_%j.err
set -e
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home1/sota/.cache/huggingface
NV_LIB=$(find /home1/sota/anaconda3/envs/inha2026_bw/lib/python3.11/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIB}${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Stage 1 gated: frozen action_proj + timestep shift (중저노이즈 편향) ===
# 목표: v5b_base 20k에서 이어서 학습, 액션 grounding 보존 + 중노이즈 대역 감독 강화
# 종료: 2k step (holdout gt-zero gap 재검증 후 연장 판단)

export FREEZE_ACTION_PATH=1
export SIGMA_MODE=beta
export SIGMA_BETA_A=2.0
export SIGMA_BETA_B=2.0

# Resume from v5b_base step 20000 (Codex 권장 시작점)
/home1/sota/anaconda3/envs/inha2026_bw/bin/python \
    /home1/sota/inha2026/scripts/finetune_cosmos3_nano.py \
    --steps 2500 \
    --lr 5e-5 \
    --rank 32 \
    --action-repr delta_base \
    --resume /home1/sota/inha2026/checkpoints/cosmos3_nano_v5_base/ckpt_step020000.pt \
    --save-every 500 \
    --ckpt-dir /home1/sota/inha2026/checkpoints/st1_frozen_shift \
    --seed 0
