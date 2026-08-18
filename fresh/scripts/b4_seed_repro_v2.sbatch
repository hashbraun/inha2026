#!/bin/bash
#SBATCH --job-name=b4rep43
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-113
#SBATCH --ntasks=1
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/b4rep43_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/b4rep43_%j.err
set -e
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home1/sota/.cache/huggingface
NV_LIB=$(find /home1/sota/anaconda3/envs/inha2026_bw/lib/python3.11/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIB}${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESUME=/home1/sota/inha2026/checkpoints/cosmos3_nano_v5_base/ckpt_step040000.pt
E_CKPT=/home1/sota/inha2026/checkpoints/fresh_invdyn/invdyn_step020000.pt
CKPT_DIR=/home1/sota/inha2026/checkpoints/b4_w005_seed43

mkdir -p $CKPT_DIR
/home1/sota/anaconda3/envs/inha2026_bw/bin/python \
    /home1/sota/inha2026/fresh/training/finetune_v5b_e_loss.py \
    --resume "$RESUME" --ckpt-dir "$CKPT_DIR" \
    --e-invdyn-ckpt "$E_CKPT" \
    --steps 8000 --lr 3e-5 --rank 32 --grad-acc 4 \
    --save-every 500 \
    --action-loss-weight 0.05 \
    --action-repr delta_base --seed 43
echo "[$(date)] b4_seed43 done"
