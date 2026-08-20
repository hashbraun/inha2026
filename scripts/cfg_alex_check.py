"""CFG 후보들의 alex_cos vs baseline 분포. GPU."""
import sys, json
sys.path.insert(0, "/home1/sota/inha2026/fresh")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio
from torchvision.models import alexnet, AlexNet_Weights
from data.so100_dataset import TARGET_H, TARGET_W, resize_pad

device = torch.device("cuda:0")
alex = alexnet(weights=AlexNet_Weights.DEFAULT).features.to(device).eval()

def alex_feat(v_uint8):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(v_uint8).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    with torch.no_grad():
        return alex(x).flatten(1).mean(0)

def load(path):
    v = iio.imread(str(path))[:16]
    if v.shape[1] != TARGET_H or v.shape[2] != TARGET_W:
        v = resize_pad(v, TARGET_H, TARGET_W)
    return v

SUB = Path("/home1/sota/inha2026/submission_kit/fresh")
base_dir = SUB / "input_videos_b4_sw_w0025_step12000"
cfg_dirs = [
    ("cfg45",     SUB / "input_videos_w0025_cfg45_s35"),
    ("cfg55",     SUB / "input_videos_w0025_cfg55_s35"),
    ("cfg75",     SUB / "input_videos_w0025_cfg75_s35"),
    ("cfg60_s50", SUB / "input_videos_w0025_cfg60_s50"),
]

print("=== CFG 후보 alex_cos vs baseline 분포 ===")
print(f"{'tag':12s} {'n':>4s} {'mean':>7s} {'min':>7s} {'q10':>7s} {'q50':>7s} {'q90':>7s} {'max':>7s} {'≥0.985 pass':>13s}")
for tag, d in cfg_dirs:
    cos_list = []
    for sid_i in range(216):
        sid = f"sample_{sid_i:06d}"
        try:
            b = load(base_dir / f"{sid}.mp4")
            c = load(d / f"{sid}.mp4")
            fb = alex_feat(b)
            fc = alex_feat(c)
            cos = float(F.cosine_similarity(fb.unsqueeze(0), fc.unsqueeze(0)).item())
            cos_list.append(cos)
        except Exception:
            pass
    arr = np.array(cos_list)
    n_pass = int((arr >= 0.985).sum())
    print(f"{tag:12s} {len(arr):4d} {arr.mean():7.4f} {arr.min():7.4f} {np.quantile(arr,0.1):7.4f} "
          f"{np.median(arr):7.4f} {np.quantile(arr,0.9):7.4f} {arr.max():7.4f} {n_pass}/{len(arr)} ({100*n_pass/len(arr):.0f}%)")

# 참고: seeds / sweep steps 후보 대조
print()
print("=== 대조: seed 후보 alex_cos vs baseline ===")
seeds_dirs = [(f"seed{i}", SUB / f"input_videos_w0025_seed{i}") for i in range(1, 6)]
for tag, d in seeds_dirs:
    cos_list = []
    for sid_i in range(216):
        sid = f"sample_{sid_i:06d}"
        try:
            b = load(base_dir / f"{sid}.mp4")
            c = load(d / f"{sid}.mp4")
            fb = alex_feat(b); fc = alex_feat(c)
            cos = float(F.cosine_similarity(fb.unsqueeze(0), fc.unsqueeze(0)).item())
            cos_list.append(cos)
        except: pass
    arr = np.array(cos_list)
    n_pass = int((arr >= 0.985).sum())
    print(f"{tag:12s} {len(arr):4d} {arr.mean():7.4f} {arr.min():7.4f} {np.quantile(arr,0.1):7.4f} "
          f"{np.median(arr):7.4f} {np.quantile(arr,0.9):7.4f} {arr.max():7.4f} {n_pass}/{len(arr)} ({100*n_pass/len(arr):.0f}%)")

print()
print("=== 대조: sweep step 후보 ===")
step_dirs = [(f"step{s}", SUB / f"input_videos_b4_sw_w0025_step{s}") for s in [4000, 6000, 8000, 10000]]
for tag, d in step_dirs:
    cos_list = []
    for sid_i in range(216):
        sid = f"sample_{sid_i:06d}"
        try:
            b = load(base_dir / f"{sid}.mp4")
            c = load(d / f"{sid}.mp4")
            fb = alex_feat(b); fc = alex_feat(c)
            cos = float(F.cosine_similarity(fb.unsqueeze(0), fc.unsqueeze(0)).item())
            cos_list.append(cos)
        except: pass
    arr = np.array(cos_list)
    n_pass = int((arr >= 0.985).sum())
    print(f"{tag:12s} {len(arr):4d} {arr.mean():7.4f} {arr.min():7.4f} {np.quantile(arr,0.1):7.4f} "
          f"{np.median(arr):7.4f} {np.quantile(arr,0.9):7.4f} {arr.max():7.4f} {n_pass}/{len(arr)} ({100*n_pass/len(arr):.0f}%)")
