"""Wan2.2 zero-shot pilot 5 sample 자체 판독기 평가.
- E-invdyn L1
- AlexNet cosine vs B4 baseline
- 각 sample별 & 평균
"""
import sys, json
sys.path.insert(0, "/home1/sota/inha2026/fresh")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio
from torchvision.models import alexnet, AlexNet_Weights
from data.so100_dataset import TARGET_H, TARGET_W, resize_pad
from models.inverse_dynamics import InverseDynamicsPredictor

TRAIN_STATS = "/home1/sota/inha2026/data/train/so100_action_statistics.json"
EVAL_DIR = Path("/home1/sota/inha2026/data/eval")
INVDYN_CKPT = "/home1/sota/inha2026/checkpoints/fresh_invdyn/invdyn_step020000.pt"

device = torch.device("cuda:0")
alex = alexnet(weights=AlexNet_Weights.DEFAULT).features.to(device).eval()
invdyn = InverseDynamicsPredictor().to(device).eval()
ck = torch.load(INVDYN_CKPT, map_location=device)
invdyn.load_state_dict(ck["model"])

with open(TRAIN_STATS) as f:
    d = json.load(f)
act_mean = torch.tensor(d["mean"], dtype=torch.float32, device=device)
act_std = torch.tensor(d["std"], dtype=torch.float32, device=device)

def load(path):
    v = iio.imread(str(path))[:16]
    if v.shape[1] != TARGET_H or v.shape[2] != TARGET_W:
        v = resize_pad(v, TARGET_H, TARGET_W)
    return v

def alex_feat(v):
    mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    x = torch.from_numpy(v).to(device).permute(0,3,1,2).float()/255.0
    x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
    x = (x - mean)/std
    with torch.no_grad():
        return alex(x).flatten(1).mean(0)

def action_l1(v, gt_a):
    vt = torch.from_numpy(v).float().div(255).mul(2).sub(1).permute(3,0,1,2).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = invdyn(vt)[0]
    pred_dn = pred * act_std + act_mean
    return float(torch.mean(torch.abs(pred_dn - torch.from_numpy(gt_a).to(device))))

WAN_DIR = Path("/home1/sota/inha2026/submission_kit/fresh/input_videos_wan22_zeroshot")
B4_DIR = Path("/home1/sota/inha2026/submission_kit/fresh/input_videos_b4_step6000")

print("=== Wan2.2 zero-shot 5 sample 자체 판독기 ===")
print(f"{'sample':15s} {'wan_l1':>8s} {'b4_l1':>8s} {'delta':>7s} {'alex_cos':>10s} {'ff_l2_wan':>10s} {'ff_l2_b4':>9s}")
for i in range(5):
    sid = f"sample_{i:06d}"
    try:
        wv = load(WAN_DIR / f"{sid}.mp4")
        bv = load(B4_DIR / f"{sid}.mp4")
        gt_a = np.load(EVAL_DIR / "actions" / f"{sid}.npy")[:16].astype(np.float32)
        wan_l1 = action_l1(wv, gt_a)
        b4_l1 = action_l1(bv, gt_a)
        wf = alex_feat(wv)
        bf = alex_feat(bv)
        cos = float(F.cosine_similarity(wf.unsqueeze(0), bf.unsqueeze(0)).item())
        # first frame vs GT eval
        from PIL import Image
        gt_frame = np.array(Image.open(EVAL_DIR/"images"/f"{sid}.png").convert("RGB"))
        if gt_frame.shape != wv[0].shape:
            gt_frame = resize_pad(gt_frame[None], TARGET_H, TARGET_W)[0]
        ff_l2_wan = float(np.linalg.norm((wv[0].astype(np.float32) - gt_frame.astype(np.float32))/255.0))
        ff_l2_b4 = float(np.linalg.norm((bv[0].astype(np.float32) - gt_frame.astype(np.float32))/255.0))
        print(f"{sid:15s} {wan_l1:8.3f} {b4_l1:8.3f} {wan_l1-b4_l1:+7.3f} {cos:10.4f} {ff_l2_wan:10.2f} {ff_l2_b4:9.2f}")
    except Exception as e:
        print(f"{sid} err: {e}")
