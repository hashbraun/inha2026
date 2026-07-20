#!/bin/bash
#SBATCH --job-name=inha_lora_finetune
#SBATCH --nodelist=gpu-106
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --output=/home1/sota/inha2026/logs/lora_train_%j.out
#SBATCH --error=/home1/sota/inha2026/logs/lora_train_%j.err

set -e

CHALLENGE_KIT=/home1/sota/inha2026/baseline/challenge_kit
PYTHON=/home1/sota/anaconda3/bin/python

echo "[SBATCH] Job ID: $SLURM_JOB_ID, Node: $SLURMD_NODENAME"
echo "[SBATCH] Start: $(date)"

mkdir -p /home1/sota/inha2026/logs

export PYTHONPATH="${CHALLENGE_KIT}/src:${CHALLENGE_KIT}/libs/dynamicrafter:${PYTHONPATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0
export CUDA_VISIBLE_DEVICES=0

if [ -f "/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" ]; then
    export LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
fi

cd "${CHALLENGE_KIT}"

$PYTHON -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=1 \
    --master_addr=127.0.0.1 --master_port=12354 \
    --node_rank=0 \
    scripts/train_diffusion.py \
    --base configs/train/inha_lora_finetune.yaml \
    --train \
    --devices 1 \
    --lora-rank 4 \
    --lora-alpha 4 \
    --unet-checkpoint ../checkpoints/baseline_diffusion.ckpt \
    lightning.trainer.num_nodes=1

echo "[SBATCH] Done: $(date)"
