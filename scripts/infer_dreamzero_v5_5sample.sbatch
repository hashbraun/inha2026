#!/bin/bash
#SBATCH --job-name=dz_v5_infer
#SBATCH --nodelist=gpu-112
#SBATCH --gres=gpu:A6000_ada:2
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/home1/sota/inha2026/logs/dz_v5_infer_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/dz_v5_infer_%j.err

set -e
PYTHON=/home1/sota/anaconda3/envs/inha2026/bin/python
SCRIPT=/home1/sota/inha2026/scripts/infer_dreamzero_action_inpaint.py
OUT_DIR=/home1/sota/inha2026/submission_kit/input_videos_v5_5sample
LORA_CKPT=/home1/sota/inha2026/checkpoints/dreamzero_v5_lora/lora_step10000.pt

export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1

echo "=== Job: $SLURM_JOB_ID | Node: $(hostname) | $(date) ==="
echo "LoRA: $LORA_CKPT"
mkdir -p "$OUT_DIR"

cd /home1/sota/inha2026
$PYTHON $SCRIPT \
    --eval-dir        /home1/sota/inha2026/data/eval \
    --out-dir         "$OUT_DIR" \
    --eval-stats      /home1/sota/inha2026/data/eval/eval_action_statistics.json \
    --embodiment-id   26 \
    --local-lora-path /home1/sota/.cache/huggingface/hub/models--Vizuara--dreamzero-so101-lora/snapshots/831baf3c8c04a238c981b77a539e93f7417e48e1 \
    --ae-lora-path    "$LORA_CKPT" \
    --num-samples     5

echo "=== 완료: $(ls $OUT_DIR/*.mp4 2>/dev/null | wc -l)개 | $(date) ==="
