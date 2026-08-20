#!/bin/bash
#SBATCH --job-name=st4_freeze
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-113
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/st4_freeze_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/st4_freeze_%j.err
set -e
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home1/sota/.cache/huggingface
NV_LIB=$(find /home1/sota/anaconda3/envs/inha2026_bw/lib/python3.11/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIB}${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Stage 4: 완전 freeze (LoRA freeze, action_proj만 학습) ===
# 목적: attention 완전 pretrained 유지 → mass 2.07 보존 검증
# 대상: action_proj_in/out + action_modality_embed만 학습
# LoRA는 add되지만 freeze (FREEZE_LORA=1)

export FREEZE_LORA=1
export LORA_TARGETS="to_add_out,mlp_moe_gen.gate_proj,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj"
export SIGMA_MODE=beta
export SIGMA_BETA_A=2.0
export SIGMA_BETA_B=2.0

/home1/sota/anaconda3/envs/inha2026_bw/bin/python \
    /home1/sota/inha2026/scripts/finetune_cosmos3_nano.py \
    --steps 2500 \
    --lr 1e-4 \
    --rank 32 \
    --action-repr delta_base \
    --save-every 500 \
    --ckpt-dir /home1/sota/inha2026/checkpoints/st4_full_freeze \
    --seed 0
