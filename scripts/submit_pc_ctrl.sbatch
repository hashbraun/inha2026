#!/bin/bash
#SBATCH --job-name=submit_ctrl
#SBATCH --nodelist=gpu-109
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/home1/sota/inha2026/logs/submit_%x_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/submit_%x_%j.err

# pc_ctrl (absolute_ngrip 8k, seed=0) 216 샘플 생성 → submission CSV.
# 홀드아웃 combined=0.0907로 이전 스크리닝 최고군. 실제 리더보드 검증 필요.
# 회귀식 재교정도 이 제출로.
#
# 사용: sbatch scripts/submit_pc_ctrl.sbatch

set -e
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home1/sota/.cache/huggingface

PYTHON=/home1/sota/anaconda3/envs/inha2026/bin/python
DIR=/home1/sota/inha2026/scripts
SUB=/home1/sota/inha2026/submission_kit
CKPT=/home1/sota/inha2026/checkpoints/perc_ctrl/ckpt_step008000.pt
OUT=$SUB/input_videos_pc_ctrl_gs50
LABEL=pc_ctrl_8k_gs50

echo "[$(date)] Phase 1: 216 샘플 추론 (steps=50, gs=5.0, absolute_ngrip)"
$PYTHON $DIR/infer_cosmos3_nano.py \
    --ckpt "$CKPT" --samples 216 --steps 50 --guidance 5.0 \
    --rank 32 --action-repr absolute_ngrip \
    --out-dir "$OUT"

echo "[$(date)] Phase 2: submission CSV 생성"
cd $SUB
$PYTHON make_submission_csv.py \
    --prediction-root "$OUT" \
    --challenge-root ../data/eval \
    --output-csv "submission_${LABEL}.csv" \
    --action-stats-path ../data/train/so100_action_statistics.json \
    --action-extractor-ckpt checkpoints/action_extractor.ckpt

echo "[$(date)] 완료 → $SUB/submission_${LABEL}.csv"
