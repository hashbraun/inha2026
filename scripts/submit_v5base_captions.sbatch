#!/bin/bash
#SBATCH --job-name=v5_caps
#SBATCH --nodelist=gpu-113
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/home1/sota/inha2026/logs/caps_%x_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/caps_%x_%j.err

set -e
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home1/sota/.cache/huggingface

CKPT=/home1/sota/inha2026/checkpoints/cosmos3_nano_v5_base/ckpt_step040000.pt
OUT=/home1/sota/inha2026/submission_kit/input_videos_v5base_captions

echo "[$(date)] v5_base + captions 주입 추론"
/home1/sota/anaconda3/envs/inha2026/bin/python /home1/sota/inha2026/scripts/infer_cosmos3_nano.py \
    --ckpt "$CKPT" --samples 216 --steps 50 --guidance 5.0 \
    --rank 32 --action-repr delta_base \
    --use-captions \
    --out-dir "$OUT"

cd /home1/sota/inha2026/submission_kit
/home1/sota/anaconda3/envs/inha2026/bin/python make_submission_csv.py \
    --prediction-root "$OUT" \
    --challenge-root ../data/eval \
    --output-csv submission_v5base_CAPTIONS.csv \
    --action-stats-path ../data/train/so100_action_statistics.json \
    --action-extractor-ckpt checkpoints/action_extractor.ckpt

echo "[$(date)] 완료 → submission_v5base_CAPTIONS.csv"
