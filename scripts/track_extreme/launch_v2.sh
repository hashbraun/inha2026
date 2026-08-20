#!/bin/bash
# 방향성 확장 실험 5개 launch. auto_loop_extreme와 독립.
set -e
ROOT=/home1/sota/inha2026
PY=/home1/sota/anaconda3/envs/inha2026/bin/python
INFER=$ROOT/scripts/infer_cosmos3_nano.py
INTERP=$ROOT/scripts/track_extreme/weight_interp.py
INTERP_DIR=$ROOT/checkpoints/interp
SUB=$ROOT/submission_kit/fresh
LOG=$ROOT/logs/fresh

CKPT_B4=$ROOT/checkpoints/v5b_eloss_bw/ckpt_step006000.pt
CKPT_W0025=$ROOT/checkpoints/b4sw_bw_w0025/ckpt_step012000.pt

# 1) alpha=0.10 (more w0025) — interp only
$PY $INTERP --ckpt-a $CKPT_B4 --ckpt-b $CKPT_W0025 --alpha 0.10 \
    --out $INTERP_DIR/interp_b4_w0025_a10.pt

# 2) alpha=0.05 (nearly pure w0025)
$PY $INTERP --ckpt-a $CKPT_B4 --ckpt-b $CKPT_W0025 --alpha 0.05 \
    --out $INTERP_DIR/interp_b4_w0025_a05.pt

# 3) alpha=0.25 already exists → will use for combo

# CANDIDATES:
declare -a CANDS=(
    "exp2_w0025_cfg30_s35|$CKPT_W0025|3.0|35"
    "exp2_w0025_cfg35_s35|$CKPT_W0025|3.5|35"
    "exp2_interp_a10_cfg60|$INTERP_DIR/interp_b4_w0025_a10.pt|6.0|35"
    "exp2_interp_a05_cfg60|$INTERP_DIR/interp_b4_w0025_a05.pt|6.0|35"
    "exp2_interp_a25_cfg45|$INTERP_DIR/interp_b4_w0025_a25.pt|4.5|35"
    "exp2_interp_a10_cfg45|$INTERP_DIR/interp_b4_w0025_a10.pt|4.5|35"
)

for ENTRY in "${CANDS[@]}"; do
    IFS='|' read -r TAG CKPT CFG STEPS <<< "$ENTRY"
    OUT=$SUB/input_videos_$TAG
    CSV=$SUB/submission_$TAG.csv
    if [ -f "$CSV" ]; then echo "SKIP $TAG (csv exists)"; continue; fi

    SBATCH=/tmp/exp2_${TAG}.sbatch
    cat > $SBATCH <<EOF
#!/bin/bash
#SBATCH --job-name=v2_${TAG:0:14}
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
$PY $INFER --ckpt $CKPT --samples 216 --steps $STEPS --guidance $CFG \\
    --rank 32 --action-repr delta_base --out-dir $OUT
cd $ROOT/submission_kit
$PY make_submission_csv.py \\
    --prediction-root fresh/input_videos_$TAG \\
    --output-csv fresh/submission_$TAG.csv
EOF
    JID=$(sbatch $SBATCH | awk '{print $NF}')
    echo "LAUNCHED $TAG (job $JID, ckpt=$(basename $CKPT), cfg=$CFG steps=$STEPS)"
done
