"""Multi-seed rejection sampling picker.

여러 seed로 생성한 216개 sample들에 대해 sample-wise best 선택.
Codex gated policy:
- 기본 B4 유지 (seed 0)
- 후보가 E-invdyn action ≥0.01 개선 AND alex_cos ≥ 0.985 → 교체
- 통과 못하면 원본 유지

출력: mp4 심볼릭 링크 + submission CSV
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import alexnet, AlexNet_Weights

sys.path.insert(0, "/home1/sota/inha2026/fresh")
from data.so100_dataset import TARGET_H, TARGET_W, resize_pad
from models.inverse_dynamics import InverseDynamicsPredictor

TRAIN_STATS = "/home1/sota/inha2026/data/train/so100_action_statistics.json"
EVAL_DIR = Path("/home1/sota/inha2026/data/eval")
INVDYN_CKPT = "/home1/sota/inha2026/checkpoints/fresh_invdyn/invdyn_step020000.pt"


def load_video(path, device):
    v = iio.imread(str(path))[:16]
    if v.shape[1] != TARGET_H or v.shape[2] != TARGET_W:
        v = resize_pad(v, TARGET_H, TARGET_W)
    return v  # uint8 (T,H,W,3)


def action_l1(v_uint8, invdyn, gt_action, mean, std, device):
    v = torch.from_numpy(v_uint8).float().div(255).mul(2).sub(1).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = invdyn(v)[0]
    pred_denorm = pred * std + mean
    return float(torch.mean(torch.abs(pred_denorm - torch.from_numpy(gt_action).to(device))))


def alex_feat(v_uint8, alex, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(v_uint8).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    with torch.no_grad():
        return alex(x).flatten(1).mean(0)  # (feat_dim,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dirs", nargs="+", required=True,
                    help="mp4 폴더들. 첫번째가 기본 (교체 기준)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--samples", type=int, default=216)
    ap.add_argument("--action-improve", type=float, default=1.0,
                    help="E-invdyn L1 개선 최소값 (joint deg). Codex 0.01은 normalized 기준이라 raw로 조정")
    ap.add_argument("--visual-min-cos", type=float, default=0.985)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {device}", flush=True)
    alex = alexnet(weights=AlexNet_Weights.DEFAULT).features.to(device).eval()
    invdyn = InverseDynamicsPredictor().to(device).eval()
    ck = torch.load(INVDYN_CKPT, map_location=device)
    invdyn.load_state_dict(ck["model"])
    print(f"[invdyn] step {ck['step']}", flush=True)

    with open(TRAIN_STATS) as f:
        d = json.load(f)
    act_mean = torch.tensor(d["mean"], dtype=torch.float32, device=device)
    act_std = torch.tensor(d["std"], dtype=torch.float32, device=device)

    pool_dirs = [Path(p) for p in args.pool_dirs]
    print(f"[pool] {len(pool_dirs)} sources:")
    for p in pool_dirs:
        print(f"  {p.name}: {len(list(p.glob('*.mp4')))} mp4s")

    stats = {"kept_base": 0, "replaced": 0, "no_candidate": 0}
    per_sample = []

    for i in range(args.samples):
        sid = f"sample_{i:06d}"
        gt_action = np.load(EVAL_DIR / "actions" / f"{sid}.npy")[:16].astype(np.float32)

        # Load all pool candidates
        candidates = []
        for j, pool in enumerate(pool_dirs):
            vp = pool / f"{sid}.mp4"
            if vp.exists():
                v = load_video(vp, device)
                candidates.append((j, pool, v))

        if not candidates:
            stats["no_candidate"] += 1
            per_sample.append({"sid": sid, "picked": None, "reason": "no_candidate"})
            continue

        # Base (first pool)
        base_j, base_pool, base_v = candidates[0]
        base_l1 = action_l1(base_v, invdyn, gt_action, act_mean, act_std, device)
        base_alex = alex_feat(base_v, alex, device)

        # Score others
        best = (base_j, base_pool, base_v, base_l1, 1.0)  # (j, pool, v, l1, cos)
        for j, pool, v in candidates[1:]:
            cand_l1 = action_l1(v, invdyn, gt_action, act_mean, act_std, device)
            if cand_l1 > base_l1 - args.action_improve:
                continue  # action 개선 부족
            cand_alex = alex_feat(v, alex, device)
            cos = float(F.cosine_similarity(base_alex.unsqueeze(0), cand_alex.unsqueeze(0)).item())
            if cos < args.visual_min_cos:
                continue  # visual 손상
            # 통과: 개선 폭 큰 것 우선
            if cand_l1 < best[3]:
                best = (j, pool, v, cand_l1, cos)

        picked_j, picked_pool, picked_v, picked_l1, picked_cos = best
        if picked_j == base_j:
            stats["kept_base"] += 1
        else:
            stats["replaced"] += 1

        # Copy mp4 to out_dir
        src = picked_pool / f"{sid}.mp4"
        dst = out_dir / f"{sid}.mp4"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy(src, dst)
        per_sample.append({
            "sid": sid, "picked": str(src), "l1": picked_l1, "cos": picked_cos,
            "base_l1": base_l1,
        })
        if (i + 1) % 30 == 0:
            print(f"  progress {i+1}/{args.samples}, kept={stats['kept_base']}, replaced={stats['replaced']}",
                  flush=True)

    # Save report
    report_path = Path(args.out_dir).parent / f"{Path(args.out_dir).name}_report.json"
    with open(report_path, "w") as f:
        json.dump({"stats": stats, "pool_dirs": [str(p) for p in pool_dirs],
                    "per_sample": per_sample}, f, indent=2)
    print(f"\n[stats] kept_base={stats['kept_base']}, replaced={stats['replaced']}, "
          f"no_candidate={stats['no_candidate']}")
    print(f"[report] {report_path}")

    # Generate CSV
    import subprocess
    csv_out = Path(args.out_csv)
    rel_pred = out_dir.relative_to("/home1/sota/inha2026/submission_kit")
    rel_csv = csv_out.relative_to("/home1/sota/inha2026/submission_kit")
    print(f"\n[csv] generating {csv_out} ...")
    r = subprocess.run(
        [
            "/home1/sota/anaconda3/envs/inha2026/bin/python",
            "make_submission_csv.py",
            "--prediction-root", str(rel_pred),
            "--output-csv", str(rel_csv),
        ],
        cwd="/home1/sota/inha2026/submission_kit",
        capture_output=True, text=True,
    )
    print(r.stdout[-500:])
    if r.returncode != 0:
        print(f"[csv ERR] {r.stderr[-500:]}")


if __name__ == "__main__":
    main()
