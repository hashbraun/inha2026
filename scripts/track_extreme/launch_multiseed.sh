#!/bin/bash
# Multi-seed B4 rejection sampling. seed 1~7 (0은 이미 실측 있음).
# 각 sample당 8후보 (0~7) → per-sample E-invdyn + alex gate로 best pick.
set -e
ROOT=/home1/sota/inha2026
PY=/home1/sota/anaconda3/envs/inha2026/bin/python
INFER=$ROOT/scripts/infer_cosmos3_nano.py
SUB=$ROOT/submission_kit/fresh
LOG=$ROOT/logs/fresh
CKPT=$ROOT/checkpoints/v5b_eloss_bw/ckpt_step006000.pt

for SEED in 1 2 3 4 5 6 7; do
    TAG=b4_seed${SEED}
    OUT=$SUB/input_videos_$TAG
    CSV=$SUB/submission_$TAG.csv
    if [ -f "$CSV" ]; then echo "SKIP $TAG (csv exists)"; continue; fi
    SBATCH=/tmp/ms_${TAG}.sbatch
    cat > $SBATCH <<EOF
#!/bin/bash
#SBATCH --job-name=ms_${TAG}
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --output=$LOG/${TAG}_%j.log
#SBATCH --error=$LOG/${TAG}_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT
$PY $INFER --ckpt $CKPT --samples 216 --steps 35 --guidance 6.0 \\
    --rank 32 --action-repr delta_base --seed $SEED --out-dir $OUT
cd $ROOT/submission_kit
$PY make_submission_csv.py \\
    --prediction-root fresh/input_videos_$TAG \\
    --output-csv fresh/submission_$TAG.csv
EOF
    JID=$(sbatch $SBATCH | awk '{print $NF}')
    echo "LAUNCHED $TAG seed=$SEED (job $JID)"
done
