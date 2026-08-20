"""Predict2.5 백본 진단: Bridge 원본으로 attention mass + gt-rev/nfl 측정.

SO-100 데이터 사용 안 함. Bridge-format action 사용.
이 backbone에서 action conditioning이 살아있는지만 순수하게 검증.

측정:
- gt-zero MAE
- gt-reverse MAE
- gt-gt (seed 다르게 3번) → nfl
- gt-zero / nfl, gt-rev / nfl
- attention mass (있으면)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio
from PIL import Image
from itertools import combinations

# Predict2.5 checkpoint 위치
CKPT_ROOT = "/home1/sota/inha2026/models/predict2.5_action_cond/robot/action-cond"
EVAL_DIR = Path("/home1/sota/inha2026/data/eval")   # first frame 재사용 OK (배경만)

# Predict2.5 spec (IMPLEMENTATION_NOTES에서 확인됨)
# - action_dim = 28 = 7 × 4 chunks (Bridge)
# - 13 frames, 256x320, 10fps
# - 7D per frame = [rel_xyz3, rel_rpy3, gripper1] Bridge scaler [20]×6

OUT_ROOT = Path("/home1/sota/inha2026/submission_kit/predict25_diag")

print("=== Predict2.5 Bridge diagnosis ===")
print(f"CKPT_ROOT: {CKPT_ROOT}")
print(f"OUT: {OUT_ROOT}")

# Check ckpt existence
ckpt_files = list(Path(CKPT_ROOT).rglob("*.pt")) if Path(CKPT_ROOT).exists() else []
print(f"\nckpt files ({len(ckpt_files)}):")
for p in ckpt_files:
    print(f"  {p} ({p.stat().st_size/1e9:.2f} GB)")

# 다운로드된 파일 확인
predict25_root = Path("/home1/sota/inha2026/models/predict2.5_action_cond")
if predict25_root.exists():
    print("\npredict25 dir contents:")
    for p in sorted(predict25_root.rglob("*"))[:40]:
        if p.is_file():
            print(f"  {p.relative_to(predict25_root)} ({p.stat().st_size/1e6:.1f} MB)")
else:
    print(f"\n[ERROR] {predict25_root} not found")

# Attempt to load pipeline (Cosmos-Predict2 style)
try:
    from cosmos_predict2 import Video2WorldActionConditionedPipeline
    print("\n[OK] cosmos_predict2 import successful")
except ImportError as e:
    print(f"\n[FAIL] cosmos_predict2 import: {e}")
    print("→ Predict2.5 진단 하려면 별도 세팅 필요. 오늘 대회와 무관.")
    sys.exit(0)

# 여기까지 오면 pipeline 로드 가능
print("\n[TODO] pipeline load + Bridge action ablation")
