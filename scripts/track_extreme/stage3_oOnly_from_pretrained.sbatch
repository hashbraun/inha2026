#!/bin/bash
#SBATCH --job-name=st3_oOnly
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-113
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/st3_oOnly_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/st3_oOnly_%j.err
set -e
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home1/sota/.cache/huggingface
NV_LIB=$(find /home1/sota/anaconda3/envs/inha2026_bw/lib/python3.11/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIB}${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Stage 3: O-only LoRA + timestep shift, Cosmos3-Nano PRETRAINED에서 시작 ===
# 목적: attention Q/K/V freeze → mass 2.07x 보존 검증
# LoRA scope: to_add_out (O projection) + mlp_moe_gen.* (MLP)
# No --resume → 프리트레인 그대로 시작 (v5b_20k 오염 회피)

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
    --ckpt-dir /home1/sota/inha2026/checkpoints/st3_oOnly \
    --seed 0
