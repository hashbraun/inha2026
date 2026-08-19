#!/bin/bash
set -e
ROOT=/home1/sota/inha2026
PY=/home1/sota/anaconda3/envs/inha2026/bin/python
INFER=$ROOT/scripts/infer_cosmos3_nano.py
SUB=$ROOT/submission_kit/fresh
LOG=$ROOT/logs/fresh
CKPT=$ROOT/checkpoints/b4sw_bw_w0025/ckpt_step012000.pt

for SEED in 1 2 3 4 5; do
    TAG=w0025_seed${SEED}
    OUT=$SUB/input_videos_$TAG
    CSV=$SUB/submission_$TAG.csv
    if [ -f "$CSV" ]; then echo "SKIP $TAG"; continue; fi
    SBATCH=/tmp/msw_${TAG}.sbatch
    cat > $SBATCH <<INNER
#!/bin/bash
#SBATCH --job-name=msw_${TAG}
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
$PY $INFER --ckpt $CKPT --samples 216 --steps 35 --guidance 6.0 --rank 32 --action-repr delta_base --seed $SEED --out-dir $OUT
cd $ROOT/submission_kit
$PY make_submission_csv.py --prediction-root fresh/input_videos_$TAG --output-csv fresh/submission_$TAG.csv
INNER
    JID=$(sbatch $SBATCH | awk '{print $NF}')
    echo "LAUNCHED $TAG seed=$SEED (job $JID)"
done
