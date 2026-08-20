#!/bin/bash
#SBATCH --job-name=cfg_alex
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=40G
#SBATCH --time=0:30:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/cfg_alex_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/cfg_alex_%j.err
set -e
export PYTHONUNBUFFERED=1
/home1/sota/anaconda3/envs/inha2026/bin/python /home1/sota/inha2026/scripts/cfg_alex_check.py
