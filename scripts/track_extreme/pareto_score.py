"""Pareto gate 자체 판독기 (kit R3D/DINO/GRU 미사용, 규정 준수).

각 candidate mp4 폴더에 대해:
1. E-invdyn action L1 (mean/median) — clean-room predictor
2. AlexNet feature 유사도 vs B4 baseline — visual drift 검출
3. Adjacent-frame L2 mean — motion smoothness proxy
4. First-frame L2 vs GT eval image — anchor 정합성

사용: kit R3D/DINO/GRU 절대 사용 안 함. AlexNet(ImageNet, torchvision) OK.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import alexnet, AlexNet_Weights

sys.path.insert(0, "/home1/sota/inha2026/fresh")
from data.so100_dataset import TARGET_H, TARGET_W, resize_pad
from models.inverse_dynamics import InverseDynamicsPredictor

TRAIN_STATS = "/home1/sota/inha2026/data/train/so100_action_statistics.json"
EVAL_DIR = Path("/home1/sota/inha2026/data/eval")
INVDYN_CKPT = "/home1/sota/inha2026/checkpoints/fresh_invdyn/invdyn_step020000.pt"


def load_stats(device):
    with open(TRAIN_STATS) as f:
        d = json.load(f)
    return (torch.tensor(d["mean"], dtype=torch.float32, device=device),
            torch.tensor(d["std"], dtype=torch.float32, device=device))


def load_mp4(path: Path, device, normalize_gan=True):
    """Return (3, T, H, W) float in [-1,1] if normalize_gan else (T,H,W,3) uint8."""
    v = iio.imread(str(path))[:16]
    if v.shape[1] != TARGET_H or v.shape[2] != TARGET_W:
        v = resize_pad(v, TARGET_H, TARGET_W)
    if not normalize_gan:
        return v
    t = torch.from_numpy(v).float().div(255).mul(2).sub(1)
    return t.permute(3, 0, 1, 2).to(device)  # (3,T,H,W)


def load_alexnet(device):
    m = alexnet(weights=AlexNet_Weights.DEFAULT).to(device).eval()
    return m.features  # conv trunk (5 conv layers), output (B, 256, 6, 6) for 224


def alexnet_feat(frames_uint8, model, device):
    """frames_uint8: (T,H,W,3) uint8 → (T, feat_dim) float feature per frame."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(frames_uint8).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    with torch.no_grad():
        feat = model(x)  # (T, 256, 6, 6)
    return feat.flatten(1)  # (T, 256*6*6)


def score_video(mp4_path, sid, alex_model, invdyn, action_mean, action_std, gt_image_np, device):
    v_uint8 = load_mp4(mp4_path, device, normalize_gan=False)  # (T,H,W,3)
    v_norm = torch.from_numpy(v_uint8).float().div(255).mul(2).sub(1)
    v_norm = v_norm.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1,3,T,H,W)

    # E-invdyn action L1
    with torch.no_grad():
        pred = invdyn(v_norm)[0]  # (T, 6)
    gt_action = np.load(EVAL_DIR / "actions" / f"{sid}.npy")[:16].astype(np.float32)
    gt_action_t = torch.from_numpy(gt_action).to(device)
    pred_denorm = pred * action_std + action_mean
    action_l1 = float(torch.mean(torch.abs(pred_denorm - gt_action_t)))

    # AlexNet feature (per-frame → mean)
    alex_feat = alexnet_feat(v_uint8, alex_model, device).mean(0)  # (feat_dim,)

    # Adjacent-frame L2 (motion smoothness)
    dv = v_uint8.astype(np.float32) / 255.0
    diffs = np.linalg.norm(np.diff(dv, axis=0).reshape(15, -1), axis=1).mean()

    # First-frame vs GT eval image
    gt_frame = np.array(Image.open(EVAL_DIR / "images" / f"{sid}.png").convert("RGB"))
    if gt_frame.shape != v_uint8[0].shape:
        gt_frame = resize_pad(gt_frame[None], TARGET_H, TARGET_W)[0]
    ff_l2 = float(np.linalg.norm((v_uint8[0].astype(np.float32) - gt_frame.astype(np.float32)) / 255.0))

    return {
        "action_l1": action_l1,
        "alex_feat": alex_feat.cpu(),
        "temporal_diff": float(diffs),
        "first_frame_l2": ff_l2,
    }


def summarize(scores):
    if not scores:
        return {}
    al = np.array([s["action_l1"] for s in scores])
    td = np.array([s["temporal_diff"] for s in scores])
    ff = np.array([s["first_frame_l2"] for s in scores])
    return {
        "n": len(scores),
        "action_l1_mean": float(al.mean()),
        "action_l1_median": float(np.median(al)),
        "temporal_diff_mean": float(td.mean()),
        "first_frame_l2_mean": float(ff.mean()),
    }


def alex_cosine_vs_ref(scores, ref_scores):
    """평균 alex feature 벡터 사이의 cosine sim."""
    feats_a = torch.stack([s["alex_feat"] for s in scores])
    feats_b = torch.stack([s["alex_feat"] for s in ref_scores])
    n = min(len(feats_a), len(feats_b))
    fa, fb = feats_a[:n], feats_b[:n]
    fa_n = fa / (fa.norm(dim=-1, keepdim=True) + 1e-8)
    fb_n = fb / (fb.norm(dim=-1, keepdim=True) + 1e-8)
    return float((fa_n * fb_n).sum(dim=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="mp4 폴더")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--samples", type=int, default=216)
    ap.add_argument("--ref-videos", default="/home1/sota/inha2026/submission_kit/fresh/input_videos_b4_step6000",
                    help="AlexNet cosine 기준 (기본: B4 step6000)")
    ap.add_argument("--invdyn-ckpt", default=INVDYN_CKPT)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    alex_model = load_alexnet(device)
    invdyn = InverseDynamicsPredictor().to(device).eval()
    ck = torch.load(args.invdyn_ckpt, map_location=device)
    invdyn.load_state_dict(ck["model"])
    print(f"[invdyn] loaded step {ck['step']}", flush=True)
    action_mean, action_std = load_stats(device)

    videos_dir = Path(args.videos)
    ref_dir = Path(args.ref_videos)

    sample_ids = [f"sample_{i:06d}" for i in range(args.samples)]
    scores = []
    ref_scores = []
    for i, sid in enumerate(sample_ids):
        vp = videos_dir / f"{sid}.mp4"
        if not vp.exists():
            continue
        try:
            s = score_video(vp, sid, alex_model, invdyn, action_mean, action_std, None, device)
            scores.append(s)
        except Exception as e:
            print(f"  {sid} err: {e}", flush=True)
            continue
        if ref_dir.exists():
            rp = ref_dir / f"{sid}.mp4"
            if rp.exists():
                try:
                    rs = score_video(rp, sid, alex_model, invdyn, action_mean, action_std, None, device)
                    ref_scores.append(rs)
                except Exception:
                    pass
        if (i + 1) % 40 == 0:
            print(f"  progress {i+1}/{args.samples}", flush=True)

    summary = summarize(scores)
    if ref_scores:
        summary["alex_cosine_vs_ref"] = alex_cosine_vs_ref(scores, ref_scores)
        summary["ref_videos"] = str(ref_dir)
        summary["ref_summary"] = summarize(ref_scores)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("[summary]", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
