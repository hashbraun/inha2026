"""Z5: v5_base 40k LoRA + Cross-attn adapter + shuffled-action ranking fine-tune.

Video-action alignment을 위한 학습:
  1. Cross-attn adapter (layers 20-35, 82M) — video hidden이 action token에 직접 attend
  2. Shuffled-action ranking loss — 임의 shuffle된 action이 주어졌을 때 L_flow가 커야
  3. Action dropout (선택) — 일부 batch에서 action zero → unconditional balance

Frozen:   Cosmos3-Nano backbone, v5_base 40k LoRA, action_proj_in/out, action_modality_embed
Trainable: Cross-attn adapter (~82M)

Loss:  L = L_flow_gt + λ_rank × max(0, margin + L_flow_gt - L_flow_shuffled)
       (shuffled action에서 flow loss가 커야 하도록 margin ranking)
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
sys.path.insert(0, "/home1/sota/inha2026/submission_kit")

from diffusers import Cosmos3OmniPipeline, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig
from so100_to_bridge_v3 import build_action_features
from finetune_cosmos3_nano import (
    SO100ClipDataset, build_static_segments, LORA_TARGET_MODULES,
    TRAIN_DIR, DOMAIN_ID, CHUNK_SIZE, TARGET_FRAMES,
    RESOLUTION_TIER, MODEL_ID, PROMPT,
)
from crossattn_adapter import CrossAttnAdapter


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


def forward_and_loss(pipe, tr, adapter, joints, action_repr, latents, velocity_target,
                     vision_condition_mask, timestep, device, dtype):
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
    num_vision_tokens = int(vision_seg["num_vision_tokens"])
    adapter.set_context(encoded, num_vision_tokens)
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

    pred_v = preds_vision[0]
    noisy_mask_v = (1.0 - vision_condition_mask).to(dtype=torch.float32).expand_as(pred_v)
    sq_err = (pred_v.float() - velocity_target.float()) ** 2 * noisy_mask_v
    flow_loss = sq_err.sum() / noisy_mask_v.sum().clamp(min=1.0)
    return flow_loss


def train_step(pipe, tr, adapter, batch, action_repr, device, dtype, scheduler, generator,
               shuffled_joints=None, rank_margin=0.05, action_dropout_p=0.0):
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

    vision_condition_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=dtype)
    vision_condition_mask[0, 0, 0] = 1.0
    noise_vision = torch.randn(x0_vision.shape, generator=generator, device=device, dtype=torch.float32)
    x_t_vision = (1 - sigma) * x0_vision.float() + sigma * noise_vision
    latents = (vision_condition_mask.float() * x0_vision.float()
               + (1 - vision_condition_mask.float()) * x_t_vision).to(dtype)
    velocity_target = (noise_vision - x0_vision.float()).to(dtype)

    # Action dropout: 일부 batch에서 action zero (unconditional)
    use_action = (random.random() > action_dropout_p)
    joints_use = joints if use_action else np.zeros_like(joints)

    # Forward 1: GT action (or zeroed if dropout)
    flow_gt = forward_and_loss(pipe, tr, adapter, joints_use, action_repr, latents,
                               velocity_target, vision_condition_mask, timestep, device, dtype)

    rank_loss = torch.zeros((), device=device)
    flow_shuf_v = 0.0
    if shuffled_joints is not None and use_action:
        # Forward 2: shuffled action → pred는 GT video 재현 못해야 → flow loss 커야
        flow_shuf = forward_and_loss(pipe, tr, adapter, shuffled_joints, action_repr, latents,
                                     velocity_target, vision_condition_mask, timestep, device, dtype)
        # Ranking: shuffled >> gt. Margin loss.
        rank_loss = F.relu(rank_margin + flow_gt.detach() - flow_shuf)
        flow_shuf_v = float(flow_shuf.detach())

    return flow_gt, rank_loss, float(sigma), flow_shuf_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--grad-acc", type=int, default=2)
    ap.add_argument("--max-episodes", type=int, default=11132)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--action-repr", default="delta_base")
    ap.add_argument("--resume", required=True, help="v5_base 40k ckpt")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--target-layer-start", type=int, default=20)
    ap.add_argument("--target-layer-end", type=int, default=35, help="inclusive")
    ap.add_argument("--film-hidden-size", type=int, default=512)
    ap.add_argument("--attn-hidden-size", type=int, default=512)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--rank-margin", type=float, default=0.05,
                    help="Shuffled action ranking margin. Set 0 to disable shuffled loss.")
    ap.add_argument("--rank-weight", type=float, default=0.5,
                    help="λ_rank in L = L_flow + λ × rank_loss")
    ap.add_argument("--action-dropout", type=float, default=0.10)
    ap.add_argument("--holdout-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = "cuda"
    dtype = torch.bfloat16

    print(f"[z5] resume={Path(args.resume).name} steps={args.steps} lr={args.lr} rank_margin={args.rank_margin}", flush=True)
    print(f"[z5] CrossAttn layers=[{args.target_layer_start}, {args.target_layer_end}] film_h={args.film_hidden_size} attn_h={args.attn_hidden_size} nheads={args.num_heads}", flush=True)

    # -------- Load pipeline --------
    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
    pipe.to(device)
    tr = pipe.transformer
    tr.requires_grad_(False)

    # -------- Add LoRA (frozen, for v5_base load) --------
    tr.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=LORA_TARGET_MODULES,
                              init_lora_weights="gaussian"))
    for n, p in tr.named_parameters():
        if "lora_" in n:
            p.requires_grad_(False)

    # -------- Load v5_base 40k --------
    ck = torch.load(args.resume, map_location=device)
    sd = tr.state_dict()
    n_loaded = 0
    for k, v in ck["lora"].items():
        if k in sd:
            sd[k] = v.to(device=device, dtype=sd[k].dtype); n_loaded += 1
    tr.load_state_dict(sd, strict=False)
    tr.action_proj_in.load_state_dict(ck["action_proj_in"])
    tr.action_proj_out.load_state_dict(ck["action_proj_out"])
    with torch.no_grad():
        tr.action_modality_embed.copy_(ck["action_modality_embed"].to(device))
    print(f"[z5] loaded v5_base step={ck['step']} (LoRA {n_loaded}/{len(ck['lora'])})", flush=True)
    tr.action_proj_in.requires_grad_(False)
    tr.action_proj_out.requires_grad_(False)
    tr.action_modality_embed.requires_grad_(False)

    # -------- Build & attach Cross-attn adapter --------
    target_layers = list(range(args.target_layer_start, args.target_layer_end + 1))
    adapter = CrossAttnAdapter(
        hidden_size=int(tr.config.hidden_size),
        action_hidden_size=int(tr.config.hidden_size),
        film_hidden_size=args.film_hidden_size,
        attn_hidden_size=args.attn_hidden_size,
        num_heads=args.num_heads,
        target_layers=target_layers,
        chunk_size=CHUNK_SIZE,
    ).to(device=device, dtype=torch.float32)
    adapter.attach_to(tr)
    print(f"[z5] CrossAttn adapter trainable = {adapter.num_trainable_params()/1e6:.2f}M", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in adapter.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )

    scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    scheduler.set_timesteps(num_inference_steps=50, device=device)

    dataset = SO100ClipDataset(TRAIN_DIR, TARGET_FRAMES, max_episodes=args.max_episodes)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    ck_dir = Path(args.ckpt_dir); ck_dir.mkdir(parents=True, exist_ok=True)

    tr.eval()
    adapter.train()

    data_idx = list(range(len(dataset)))
    random.shuffle(data_idx)
    di = 0
    t0 = time.time()
    accum_loss = 0.0
    diag_path = ck_dir / "z5_train.jsonl"
    total_steps = args.steps

    use_shuffled = (args.rank_margin > 0 and args.rank_weight > 0)
    print(f"[z5] shuffled_action_loss={'ON' if use_shuffled else 'OFF'} action_dropout_p={args.action_dropout}", flush=True)

    for step in range(total_steps):
        if di >= len(data_idx):
            random.shuffle(data_idx); di = 0
        batch = dataset[data_idx[di]]
        di += 1

        # Prepare shuffled action (같은 batch의 다른 시간 offset — action shuffle)
        shuffled_joints = None
        if use_shuffled:
            clip_np, joints = batch
            # Simple shuffle: temporal reverse or random permutation
            perm = np.random.permutation(joints.shape[0])
            shuffled_joints = joints[perm].copy()

        try:
            flow_gt, rank_loss, sigma_v, flow_shuf_v = train_step(
                pipe, tr, adapter, batch, args.action_repr, device, dtype,
                scheduler, generator,
                shuffled_joints=shuffled_joints,
                rank_margin=args.rank_margin,
                action_dropout_p=args.action_dropout,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[oom] step {step}, skip", flush=True)
                torch.cuda.empty_cache(); continue
            raise

        loss = flow_gt + args.rank_weight * rank_loss
        (loss / args.grad_acc).backward()
        accum_loss += float(loss.item())

        adapter_stats = adapter.get_stats()

        if (step + 1) % args.grad_acc == 0:
            torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % 20 == 0 or step < 5:
            dt = time.time() - t0
            l20 = adapter_stats.get(20, {})
            l28 = adapter_stats.get(28, {})
            l35 = adapter_stats.get(35, {})
            print(f"step {step:05d}/{total_steps} L={float(loss):.4f} L_fm={float(flow_gt):.4f} "
                  f"L_shuf={flow_shuf_v:.4f} L_rank={float(rank_loss):.4f} sigma={sigma_v:.3f} "
                  f"L20[gate={l20.get('gate',0):.3f}, rel={l20.get('residual_rel_norm',0):.2e}] "
                  f"L28[gate={l28.get('gate',0):.3f}] L35[gate={l35.get('gate',0):.3f}] "
                  f"avg={accum_loss/(step+1):.4f} elapsed={dt:.0f}s",
                  flush=True)

        with open(diag_path, "a") as f:
            rec = {"step": step, "L": float(loss), "L_fm": float(flow_gt),
                   "L_shuf": flow_shuf_v, "L_rank": float(rank_loss), "sigma": sigma_v,
                   "adapter_stats": {str(k): v for k, v in adapter_stats.items()}}
            f.write(json.dumps(rec) + "\n")

        if (step + 1) % args.save_every == 0 or step == total_steps - 1:
            ckpt = {
                "step": step + 1,
                "crossattn_adapter": adapter.state_dict(),
                "target_layers": target_layers,
                "film_hidden_size": args.film_hidden_size,
                "attn_hidden_size": args.attn_hidden_size,
                "num_heads": args.num_heads,
                "action_repr": args.action_repr,
                "resume_from": args.resume,
            }
            save_path = ck_dir / f"z5_step{step+1:06d}.pt"
            torch.save(ckpt, save_path)
            print(f"[ckpt] {save_path}", flush=True)

    total_time = time.time() - t0
    print(f"[z5] finished {total_steps} steps in {total_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
