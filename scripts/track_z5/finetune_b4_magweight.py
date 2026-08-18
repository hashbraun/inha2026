"""B4 + Action-magnitude weighted loss.

전략:
- B4 성공 pattern 유지 (v5b LoRA + action_proj + E invdyn L1 with sigma clamp)
- Sample-level weight: 큰 action 시퀀스일수록 loss weight ↑
- Visual 훼손 방지 위해 base 학습 방식 그대로 (미미 조정)
- 큰 action에 대한 학습 집중 → B4의 특정 sequence 부정확 문제 완화

핵심 수식:
  motion_score = mean(|action[t+1] - action[t]|) over t   # per-clip magnitude
  weight = 1.0 + α × (motion_score - median) / std        # z-score based
          clipped to [0.5, 3.0]  # 극단 방지
  L = weight × (L_flow + (1-σ)·0.05·L_action)

즉 큰 motion clip은 loss ~2-3배, 작은 motion은 ~0.5배.
전체 loss magnitude는 dataset 평균 1.0 근처 유지.
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, "/home1/sota/inha2026/scripts")
sys.path.insert(0, "/home1/sota/inha2026/fresh")

import finetune_cosmos3_nano as base
import numpy as np
import torch
import torch.nn.functional as F
from models.inverse_dynamics import InverseDynamicsPredictor


class ETruncatedTo16(torch.nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, video): return self.inner(video[:, :, :16])


def load_e_invdyn(path: str, map_location):
    m = InverseDynamicsPredictor()
    ck = torch.load(path, map_location=map_location)
    m.load_state_dict(ck["model"])
    return ETruncatedTo16(m)


base.load_so100_action_extractor_checkpoint = load_e_invdyn


# Monkey-patch decode_and_predict_action to force pred_frames=16
_orig_decode = base.decode_and_predict_action


def decode_and_predict_action_16(pipe, action_extractor, latents, pred_v, noisy_mask_v, sigma):
    pred_action, _ = _orig_decode(pipe, action_extractor, latents, pred_v, noisy_mask_v, sigma)
    return pred_action, pred_action.shape[1]


base.decode_and_predict_action = decode_and_predict_action_16


# ---- Action-magnitude weighting monkey-patch ----
# base.train_step signature:
#   train_step(pipe, transformer, batch, device, dtype, scheduler, generator,
#              action_extractor=None, action_mean=None, action_std=None,
#              action_loss_weight=0.0, perc=None, ...)

_orig_train_step = base.train_step


def _compute_motion_weight(joints, alpha, med, std):
    """Per-clip motion score based weight. 큰 motion → weight ↑."""
    a = np.asarray(joints, dtype=np.float32)
    if a.shape[0] < 2:
        return 1.0
    delta = np.abs(a[1:] - a[:-1]).sum(axis=1)  # (T-1,)
    motion_score = float(delta.mean())
    if std < 1e-6:
        return 1.0
    z = (motion_score - med) / std
    # weight = 1.0 + alpha × z, clipped [0.5, 3.0]
    w = 1.0 + alpha * z
    return max(0.5, min(3.0, w))


# Class attributes for weighted-train (init from env)
_MAG_ALPHA = float(os.environ.get("MAG_ALPHA", "0.5"))
_MAG_MEDIAN = float(os.environ.get("MAG_MEDIAN", "7.3"))  # from prior distribution analysis
_MAG_STD = float(os.environ.get("MAG_STD", "9.0"))
print(f"[mag-weighted] alpha={_MAG_ALPHA} median={_MAG_MEDIAN} std={_MAG_STD}", flush=True)


def train_step_magweighted(pipe, transformer, batch, device, dtype, scheduler, generator, **kwargs):
    """Original train_step wrapper: multiplies loss by motion weight."""
    result = _orig_train_step(pipe, transformer, batch, device, dtype, scheduler, generator, **kwargs)
    # base.train_step returns (loss, flow_loss_v, action_loss_v, perc_parts)
    if isinstance(result, tuple) and len(result) == 4:
        loss, flow_v, action_v, perc = result
        # Compute weight from batch joints
        _, joints = batch  # (clip_np, joints)
        w = _compute_motion_weight(joints, _MAG_ALPHA, _MAG_MEDIAN, _MAG_STD)
        loss_weighted = loss * w
        if perc is not None and isinstance(perc, dict):
            perc = {**perc, "mag_weight": w}
        return loss_weighted, flow_v, action_v, perc
    return result


base.train_step = train_step_magweighted


if __name__ == "__main__":
    e_ckpt = None
    filtered_argv = []
    for i, a in enumerate(sys.argv):
        if a == "--e-invdyn-ckpt" and i + 1 < len(sys.argv):
            e_ckpt = sys.argv[i + 1]
        elif i > 0 and sys.argv[i - 1] == "--e-invdyn-ckpt":
            continue
        else:
            filtered_argv.append(a)
    if e_ckpt is None:
        e_ckpt = "/home1/sota/inha2026/checkpoints/fresh_invdyn/invdyn_step020000.pt"
    base.ACTION_EXTRACTOR_CKPT = e_ckpt
    print(f"[mag-b4] E invdyn ckpt: {e_ckpt}", flush=True)
    sys.argv = filtered_argv
    base.main()
