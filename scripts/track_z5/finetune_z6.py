"""Z6-lite: v5_base LoRA + Bounded Cross-attn adapter + L_preserve + clean-room L_action.

Codex 자문 반영 3변경 (Z5 대비):
1. RMS-normalized bounded residual (structurally ≤ α_max × h RMS)
2. Shuffled ranking 제거 → L_preserve = ||v_adapted - stopgrad(v_base)||²
3. weight_decay 1e-4 + grad clip 1.0 + layer stats p95/max

Loss:  L = L_flow + λ_a · L_action_cleanroom + λ_p · L_preserve

Layer stats logged: mean/p95/max residual_rel per step (평균만 보면 함정)
Adaptive:
  residual_p95 > 8%: gate LR = 0
  residual_p95 > 10%: gate freeze
  visual proxy 2회 연속 악화: rollback signal (early exit)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/home1/sota/inha2026/scripts")
sys.path.insert(0, "/home1/sota/inha2026/scripts/track_z5")
sys.path.insert(0, "/home1/sota/inha2026/fresh")
sys.path.insert(0, "/home1/sota/inha2026/submission_kit")

from diffusers import Cosmos3OmniPipeline, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig
from so100_to_bridge_v3 import build_action_features
from finetune_cosmos3_nano import (
    SO100ClipDataset, build_static_segments, LORA_TARGET_MODULES,
    TRAIN_DIR, DOMAIN_ID, CHUNK_SIZE, TARGET_FRAMES,
    RESOLUTION_TIER, MODEL_ID, PROMPT,
    decode_and_predict_action,   # clean-room action MAE via E invdyn
)
from crossattn_adapter_z6 import BoundedCrossAttnAdapter
from models.inverse_dynamics import InverseDynamicsPredictor


def build_action_x0(joints, action_repr, action_dim, device, dtype):
    bridge = build_action_features(joints[:CHUNK_SIZE], action_repr)
    raw = torch.from_numpy(bridge).to(device=device, dtype=dtype)
    pad = torch.zeros(raw.shape[0], action_dim - raw.shape[1], device=device, dtype=dtype)
    return torch.cat([raw, pad], dim=-1)


def compute_encoded_action(tr, adapter, x0_action, device):
    per_token_dom = torch.full((CHUNK_SIZE,), DOMAIN_ID, dtype=torch.long, device=device)
    with torch.no_grad():
        packed = tr.action_proj_in(x0_action, per_token_dom)
        packed = packed + tr.action_modality_embed
    packed = packed.detach().to(dtype=next(adapter.parameters()).dtype)
    return adapter.encode_action(packed)


class ETruncatedTo16(torch.nn.Module):
    """E invdyn expects T=16, decode gives T=17 → slice first 16."""
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, video): return self.inner(video[:, :, :16])


def forward_pred_v(pipe, tr, adapter, joints, action_repr, latents, vision_condition_mask,
                   timestep, device, dtype, adapter_enabled: bool = True):
    """Forward through DiT. adapter_enabled=False → v_base."""
    action_dim = tr.action_dim
    x0_action = build_action_x0(joints, action_repr, action_dim, device, dtype)

    text_seg = build_static_segments(pipe, PROMPT, DOMAIN_ID, device, dtype)
    vision_seg = pipe._prepare_vision_segment(
        input_vision_tokens=latents, has_image_condition=True,
        mrope_offset=text_seg["vision_start_temporal_offset"], vision_fps=6.0,
        curr=text_seg["und_len"], device=device, condition_frame_indexes=[0],
    )
    action_seg = pipe._prepare_action_segment(
        input_action_tokens=x0_action,
        condition_frame_indexes=list(range(CHUNK_SIZE)),
        mrope_offset=text_seg["vision_start_temporal_offset"], action_fps=6.0,
        curr=text_seg["und_len"] + vision_seg["num_vision_tokens"], device=device,
    )
    position_ids = torch.cat([text_seg["text_mrope_ids"], vision_seg["vision_mrope_ids"],
                              action_seg["action_mrope_ids"]], dim=1)
    sequence_length = text_seg["und_len"] + vision_seg["num_vision_tokens"] + action_seg["action_len"]
    vision_timesteps = torch.full((vision_seg["num_noisy_vision_tokens"],), float(timestep), device=device)
    action_domain_id = torch.tensor([DOMAIN_ID], dtype=torch.long, device=device)

    encoded = compute_encoded_action(tr, adapter, x0_action, device)
    num_v = int(vision_seg["num_vision_tokens"])
    adapter.set_context(encoded, num_v)
    adapter.set_enabled(adapter_enabled)
    try:
        preds_vision, _, _ = tr(
            input_ids=text_seg["input_ids"], text_indexes=text_seg["text_indexes"],
            position_ids=position_ids, und_len=text_seg["und_len"], sequence_length=sequence_length,
            vision_tokens=[latents], vision_token_shapes=vision_seg["vision_token_shapes"],
            vision_sequence_indexes=vision_seg["vision_sequence_indexes"],
            vision_mse_loss_indexes=vision_seg["vision_mse_loss_indexes"],
            vision_timesteps=vision_timesteps,
            vision_noisy_frame_indexes=vision_seg["vision_noisy_frame_indexes"],
            action_tokens=[x0_action], action_token_shapes=action_seg["action_token_shapes"],
            action_sequence_indexes=action_seg["action_sequence_indexes"],
            action_mse_loss_indexes=action_seg["action_mse_loss_indexes"],
            action_timesteps=torch.zeros((0,), device=device),
            action_noisy_frame_indexes=action_seg["action_noisy_frame_indexes"],
            action_domain_ids=[action_domain_id],
        )
    finally:
        adapter.clear_context()
        adapter.set_enabled(True)  # 다음 step은 default enabled
    return preds_vision[0], vision_seg, action_seg


def flow_loss_fn(pred_v, velocity_target, vision_condition_mask):
    noisy_mask = (1.0 - vision_condition_mask).to(dtype=torch.float32).expand_as(pred_v)
    sq = (pred_v.float() - velocity_target.float()) ** 2 * noisy_mask
    return sq.sum() / noisy_mask.sum().clamp(min=1.0)


def train_step(pipe, tr, adapter, e_invdyn, batch, action_repr, device, dtype, scheduler,
               generator, action_mean, action_std, action_weight, preserve_weight):
    clip_np, joints = batch
    T_len = clip_np.shape[0]

    cond_clip, action_image_size, _, _ = pipe._prepare_action_video_conditioning(
        [Image.fromarray(f) for f in clip_np], RESOLUTION_TIER, T_len, device=device, dtype=dtype,
    )
    with torch.no_grad():
        x0_vision = pipe._encode_video(cond_clip).contiguous().float()
        x0_vision = pipe._remove_action_video_padding_from_latent(x0_vision, action_image_size)
    latent_t = x0_vision.shape[2]

    idx = random.randrange(len(scheduler.timesteps))
    sigma = scheduler.sigmas[idx].to(device=device, dtype=torch.float32)
    timestep = scheduler.timesteps[idx]

    vision_cond_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=dtype)
    vision_cond_mask[0, 0, 0] = 1.0
    noise_v = torch.randn(x0_vision.shape, generator=generator, device=device, dtype=torch.float32)
    x_t = (1 - sigma) * x0_vision.float() + sigma * noise_v
    latents = (vision_cond_mask.float() * x0_vision.float()
               + (1 - vision_cond_mask.float()) * x_t).to(dtype)
    velocity_target = (noise_v - x0_vision.float()).to(dtype)

    # Forward 1: adapter enabled → v_adapted (gradient flows)
    pred_v_adapted, vision_seg, action_seg = forward_pred_v(
        pipe, tr, adapter, joints, action_repr, latents, vision_cond_mask, timestep,
        device, dtype, adapter_enabled=True,
    )

    # Forward 2: adapter disabled → v_base (no gradient, torch.no_grad optional but fine)
    with torch.no_grad():
        pred_v_base, _, _ = forward_pred_v(
            pipe, tr, adapter, joints, action_repr, latents, vision_cond_mask, timestep,
            device, dtype, adapter_enabled=False,
        )
        pred_v_base = pred_v_base.detach()

    # L_flow (adapted)
    L_flow = flow_loss_fn(pred_v_adapted, velocity_target, vision_cond_mask)

    # L_preserve: ||v_adapted - sg(v_base)||²  (masked to noisy region)
    noisy_mask = (1.0 - vision_cond_mask).to(dtype=torch.float32).expand_as(pred_v_adapted)
    diff = (pred_v_adapted.float() - pred_v_base.float()) ** 2 * noisy_mask
    L_preserve = diff.sum() / noisy_mask.sum().clamp(min=1.0)

    # L_action: clean-room E invdyn on decoded x0 (기존 B4 방식)
    # E invdyn (ETruncatedTo16) returns T=16 but decode_and_predict_action returns pred_frames=17.
    # Trust pred_action.shape[1] over pred_frames (B4 monkey-patch pattern).
    L_action = torch.zeros((), device=device)
    if action_weight > 0:
        pred_action, _pred_frames = decode_and_predict_action(
            pipe, e_invdyn, latents, pred_v_adapted, noisy_mask.mean(-1, keepdim=True),
            sigma,
        )
        actual_T = pred_action.shape[1]
        gt = torch.from_numpy(joints[:actual_T]).to(device=device, dtype=torch.float32)
        gt_norm = ((gt - action_mean) / action_std).unsqueeze(0)
        L_action = F.mse_loss(pred_action.float(), gt_norm)

    total = L_flow + action_weight * L_action + preserve_weight * L_preserve
    return total, L_flow, L_action, L_preserve, float(sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000, help="Pilot 2k, cont 6k+")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--grad-acc", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-episodes", type=int, default=11132)
    ap.add_argument("--action-repr", default="delta_base")
    ap.add_argument("--resume", required=True, help="v5_base 40k ckpt")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--e-invdyn-ckpt",
                    default="/home1/sota/inha2026/checkpoints/fresh_invdyn/invdyn_step020000.pt")
    ap.add_argument("--target-layer-start", type=int, default=20)
    ap.add_argument("--target-layer-end", type=int, default=35)
    ap.add_argument("--film-hidden-size", type=int, default=512)
    ap.add_argument("--attn-hidden-size", type=int, default=512)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--alpha-max", type=float, default=0.08)
    ap.add_argument("--action-weight", type=float, default=0.10)
    ap.add_argument("--preserve-weight", type=float, default=0.20)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--gate-freeze-p95", type=float, default=0.10,
                    help="p95 residual_rel 이 이 값 초과하면 gate 파라미터 freeze")
    ap.add_argument("--gate-slow-p95", type=float, default=0.08,
                    help="p95 초과 시 gate LR 0으로 다운스케일")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = "cuda"
    dtype = torch.bfloat16

    print(f"[z6] resume={Path(args.resume).name} steps={args.steps} lr={args.lr}", flush=True)
    print(f"[z6] BoundedCrossAttn layers=[{args.target_layer_start}, {args.target_layer_end}] "
          f"α_max={args.alpha_max} action_w={args.action_weight} preserve_w={args.preserve_weight} "
          f"wd={args.weight_decay} grad_clip={args.grad_clip}", flush=True)

    # -------- Load pipeline --------
    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
    pipe.to(device)
    tr = pipe.transformer
    tr.requires_grad_(False)

    # -------- Add LoRA (frozen) --------
    tr.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=LORA_TARGET_MODULES,
                              init_lora_weights="gaussian"))
    for n, p in tr.named_parameters():
        if "lora_" in n:
            p.requires_grad_(False)

    # -------- Load v5_base --------
    ck = torch.load(args.resume, map_location=device)
    sd = tr.state_dict()
    for k, v in ck["lora"].items():
        if k in sd:
            sd[k] = v.to(device=device, dtype=sd[k].dtype)
    tr.load_state_dict(sd, strict=False)
    tr.action_proj_in.load_state_dict(ck["action_proj_in"])
    tr.action_proj_out.load_state_dict(ck["action_proj_out"])
    with torch.no_grad():
        tr.action_modality_embed.copy_(ck["action_modality_embed"].to(device))
    print(f"[z6] loaded v5_base step={ck['step']}", flush=True)
    tr.action_proj_in.requires_grad_(False)
    tr.action_proj_out.requires_grad_(False)
    tr.action_modality_embed.requires_grad_(False)

    # -------- Bounded adapter --------
    target_layers = list(range(args.target_layer_start, args.target_layer_end + 1))
    adapter = BoundedCrossAttnAdapter(
        hidden_size=int(tr.config.hidden_size),
        action_hidden_size=int(tr.config.hidden_size),
        film_hidden_size=args.film_hidden_size,
        attn_hidden_size=args.attn_hidden_size,
        num_heads=args.num_heads,
        alpha_max=args.alpha_max,
        target_layers=target_layers,
        chunk_size=CHUNK_SIZE,
    ).to(device=device, dtype=torch.float32)
    adapter.attach_to(tr)
    print(f"[z6] BoundedCrossAttn adapter trainable = {adapter.num_trainable_params()/1e6:.2f}M", flush=True)

    # E invdyn (clean-room predictor)
    e_inner = InverseDynamicsPredictor()
    e_ck = torch.load(args.e_invdyn_ckpt, map_location=device)
    e_inner.load_state_dict(e_ck["model"])
    # E invdyn을 float32로 유지 (기존 B4와 일치, decoded video가 float32라 dtype match)
    e_invdyn = ETruncatedTo16(e_inner).to(device=device).eval()
    for p in e_invdyn.parameters(): p.requires_grad_(False)
    print(f"[z6] E invdyn ckpt loaded step={e_ck.get('step','?')}", flush=True)

    # Two param groups: adapter (main) + gate (with LR gating)
    main_params = [p for n, p in adapter.named_parameters() if "gate_logit" not in n]
    gate_params = [p for n, p in adapter.named_parameters() if "gate_logit" in n]
    optimizer = torch.optim.AdamW(
        [{"params": main_params, "lr": args.lr, "weight_decay": args.weight_decay},
         {"params": gate_params, "lr": args.lr, "weight_decay": 0.0}],
    )

    scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    scheduler.set_timesteps(num_inference_steps=50, device=device)

    # action stats
    import json as _j
    with open("/home1/sota/inha2026/data/train/so100_action_statistics.json") as f:
        stats = _j.load(f)
    action_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    action_std = torch.tensor(stats["std"], dtype=torch.float32, device=device)

    dataset = SO100ClipDataset(TRAIN_DIR, TARGET_FRAMES, max_episodes=args.max_episodes)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    ck_dir = Path(args.ckpt_dir); ck_dir.mkdir(parents=True, exist_ok=True)

    tr.eval(); adapter.train()

    data_idx = list(range(len(dataset)))
    random.shuffle(data_idx)
    di = 0
    t0 = time.time()
    accum_loss = 0.0
    diag_path = ck_dir / "z6_train.jsonl"

    gate_frozen = False
    for step in range(args.steps):
        if di >= len(data_idx):
            random.shuffle(data_idx); di = 0
        batch = dataset[data_idx[di]]; di += 1

        try:
            total, L_flow, L_action, L_preserve, sigma_v = train_step(
                pipe, tr, adapter, e_invdyn, batch, args.action_repr,
                device, dtype, scheduler, generator,
                action_mean, action_std,
                action_weight=args.action_weight, preserve_weight=args.preserve_weight,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[oom] step {step}, skip", flush=True)
                torch.cuda.empty_cache(); continue
            raise

        (total / args.grad_acc).backward()
        accum_loss += float(total.item())

        summary = adapter.get_summary_stats()

        # Adaptive gate control
        if not gate_frozen and summary["p95"] > args.gate_freeze_p95:
            for p in gate_params: p.requires_grad_(False)
            gate_frozen = True
            print(f"[z6] step {step}: gate FROZEN (p95 {summary['p95']:.3f} > {args.gate_freeze_p95})", flush=True)
        elif summary["p95"] > args.gate_slow_p95:
            # gate LR to 0 for this step (실질 정지)
            optimizer.param_groups[1]["lr"] = 0.0
        else:
            optimizer.param_groups[1]["lr"] = args.lr

        if (step + 1) % args.grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in adapter.parameters() if p.requires_grad], args.grad_clip
            )
            optimizer.step()
            optimizer.zero_grad()

        if step % 20 == 0 or step < 5:
            dt = time.time() - t0
            l20 = adapter.get_stats().get(20, {})
            print(f"step {step:05d}/{args.steps} L={float(total):.4f} "
                  f"L_fm={float(L_flow):.4f} L_act={float(L_action):.4f} L_pres={float(L_preserve):.4f} "
                  f"sigma={sigma_v:.3f} "
                  f"rel[mean={summary['mean']:.3f} p95={summary['p95']:.3f} max={summary['max']:.3f}] "
                  f"L20[α={l20.get('alpha_effective',0):.3f}] "
                  f"avg={accum_loss/(step+1):.4f} elapsed={dt:.0f}s",
                  flush=True)

        with open(diag_path, "a") as f:
            rec = {"step": step, "L": float(total), "L_flow": float(L_flow),
                   "L_action": float(L_action), "L_preserve": float(L_preserve), "sigma": sigma_v,
                   "residual_summary": summary,
                   "residual_by_layer": {str(k): v for k, v in adapter.get_stats().items()}}
            f.write(json.dumps(rec) + "\n")

        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            ckpt = {
                "step": step + 1,
                "bounded_crossattn": adapter.state_dict(),
                "target_layers": target_layers,
                "film_hidden_size": args.film_hidden_size,
                "attn_hidden_size": args.attn_hidden_size,
                "num_heads": args.num_heads,
                "alpha_max": args.alpha_max,
                "action_repr": args.action_repr,
                "resume_from": args.resume,
            }
            sp = ck_dir / f"z6_step{step+1:06d}.pt"
            torch.save(ckpt, sp)
            print(f"[ckpt] {sp}", flush=True)

    dt = time.time() - t0
    print(f"[z6] finished {args.steps} steps in {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
