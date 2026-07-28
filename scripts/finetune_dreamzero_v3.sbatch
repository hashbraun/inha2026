#!/bin/bash
#SBATCH --job-name=dz_v3_ft
#SBATCH --nodelist=gpu-112
#SBATCH --gres=gpu:A6000_ada:2
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/home1/sota/inha2026/logs/dz_v3_ft_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/dz_v3_ft_%j.err

# v3: emb_id bug fix — L_action now uses embodiment_id (26) instead of emb_id_0 (0)
# so L_action trains the same action_encoder row that L_dit uses.
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
mkdir -p /home1/sota/inha2026/checkpoints/dreamzero_v3_lora

cd /home1/sota/inha2026

$PYTHON $SCRIPT \
    --data-root       /home1/sota/inha2026/data/train \
    --local-lora-path /home1/sota/.cache/huggingface/hub/models--Vizuara--dreamzero-so101-lora/snapshots/831baf3c8c04a238c981b77a539e93f7417e48e1 \
    --out-dir         /home1/sota/inha2026/checkpoints/dreamzero_v3_lora \
    --num-steps       10000 \
    --lr              2e-5 \
    --lambda-dit      1.0 \
    --lambda-action   1.0 \
    --sigma-low       0.3 \
    --sigma-high      0.6 \
    --grad-accum      4 \
    --max-clips       20000 \
    --save-every      2000 \
    --embodiment-id   26

echo ""
echo "=== Done: $(date) ==="
