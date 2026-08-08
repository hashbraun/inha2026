#!/bin/bash
#SBATCH --job-name=sv5
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/home1/sota/inha2026/logs/sv5_%x_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/sv5_%x_%j.err

# v5_base(delta_base 40k, 리더보드 0.245) 기반 추론 파라미터 개선.
# 리더보드 검증 유일한 정답 경로. 홀드아웃 신뢰 불가 (pc_ctrl 8k 0.09 → 리더보드 0.435).
#
# ⚠️ 2026-08-07 사고: 아래 make_submission_csv.py의 --action-stats-path에 eval stats를 넣어
# 리더보드 점수가 0.19가량 붕괴됐음. 반드시 default(train stats) 유지할 것.
#
# 사용:
#   sbatch --job-name=v5_gs35   --export=ALL,TAG=gs35,GS=3.5,STEPS=50,SAMPLER=default,RESCALE=0 scripts/submit_v5base_variant.sbatch
#   sbatch --job-name=v5_gs70   --export=ALL,TAG=gs70,GS=7.0,STEPS=50,SAMPLER=default,RESCALE=0 scripts/submit_v5base_variant.sbatch
#   sbatch --job-name=v5_s75    --export=ALL,TAG=s75,GS=5.0,STEPS=75,SAMPLER=default,RESCALE=0 scripts/submit_v5base_variant.sbatch
#   sbatch --job-name=v5_s100   --export=ALL,TAG=s100,GS=5.0,STEPS=100,SAMPLER=default,RESCALE=0 scripts/submit_v5base_variant.sbatch

set -e
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home1/sota/.cache/huggingface

: "${TAG:?}" ; : "${GS:=5.0}" ; : "${STEPS:=50}"
: "${SAMPLER:=default}" ; : "${RESCALE:=0}"
CKPT=/home1/sota/inha2026/checkpoints/cosmos3_nano_v5_base/ckpt_step040000.pt
OUT=/home1/sota/inha2026/submission_kit/input_videos_v5_${TAG}
LABEL="v5base_${TAG}"

echo "[$(date)] v5_base variant: tag=$TAG gs=$GS steps=$STEPS sampler=$SAMPLER rescale=$RESCALE"

/home1/sota/anaconda3/envs/inha2026/bin/python /home1/sota/inha2026/scripts/infer_cosmos3_nano.py \
    --ckpt "$CKPT" --samples 216 --steps "$STEPS" --guidance "$GS" \
    --rank 32 --action-repr delta_base \
    --sampler "$SAMPLER" --cfg-rescale "$RESCALE" \
    --out-dir "$OUT"

cd /home1/sota/inha2026/submission_kit
/home1/sota/anaconda3/envs/inha2026/bin/python make_submission_csv.py \
    --prediction-root "$OUT" \
    --challenge-root ../data/eval \
    --output-csv "submission_${LABEL}.csv" \
    --action-stats-path ../data/train/so100_action_statistics.json \
    --action-extractor-ckpt checkpoints/action_extractor.ckpt

echo "[$(date)] 완료 → submission_${LABEL}.csv"
