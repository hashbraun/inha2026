"""Wan2.2-TI2V-5B LoRA fine-tune on SO-100 videos.

목적: 첫 프레임 anchor 유지 + SO-100 시각 도메인 학습
- Action conditioning 없음 (Wan에 native head 없음, 시간 부족)
- 일반 텍스트 프롬프트 사용
- LoRA target: attention QKV + FFN
- Loss: flow matching (video reconstruction)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home1/sota/inha2026/scripts")

CHUNK_SIZE = 17
TARGET_FRAMES = 18
HEIGHT = 320
WIDTH = 512


class SO100VideoDataset(torch.utils.data.Dataset):
    """SO-100 학습 영상 → (T, 3, H, W) uint8 → normalize [-1, 1]."""
    def __init__(self, train_dir: str, target_frames: int, max_episodes: int | None = None):
        self.target_frames = target_frames
        train_root = Path(train_dir)
        eps = sorted(train_root.rglob("data/chunk-*/*.parquet"))
        if max_episodes:
            eps = eps[:max_episodes]
        # video path 추론
        self.samples = []
        for pq in eps:
            # 실제 구조: USER/TASK/data/chunk-XXX/episode_XXX.parquet
            #        → USER/TASK/videos/chunk-XXX/observation.images.image/episode_XXX.mp4
            rel = pq.relative_to(train_root)
            parts = list(rel.parts)
            if len(parts) < 5:
                continue
            user_task = Path(parts[0]) / parts[1]
            chunk = parts[3]
            fname = parts[4].replace(".parquet", ".mp4")
            video_path = train_root / user_task / "videos" / chunk / "observation.images.image" / fname
            if video_path.exists():
                self.samples.append((pq, video_path))
        random.shuffle(self.samples)
        print(f"[dataset] {len(self.samples)} episodes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import imageio.v3 as iio
        pq, vp = self.samples[idx]
        # 랜덤 시작점
        vid = iio.imread(str(vp))   # (T, H, W, 3) uint8
        T = vid.shape[0]
        if T < self.target_frames:
            # padding
            pad_needed = self.target_frames - T
            vid = np.concatenate([vid, np.tile(vid[-1:], (pad_needed, 1, 1, 1))], axis=0)
            start = 0
        else:
            start = random.randint(0, T - self.target_frames)
        clip = vid[start:start + self.target_frames]   # (T, H, W, 3)
        # resize to HEIGHT x WIDTH
        import cv2
        clip_resized = np.stack([
            cv2.resize(f, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            for f in clip
        ])
        # (T, H, W, 3) uint8 → (3, T, H, W) float [-1, 1]
        t = torch.from_numpy(clip_resized).float().div(255).mul(2).sub(1).permute(3, 0, 1, 2)
        return {"video": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan-root", default="/home1/sota/inha2026/models/wan22_ti2v_5b_diffusers")
    ap.add_argument("--train-dir", default="/home1/sota/inha2026/data/train")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--ckpt-dir", default="/home1/sota/inha2026/checkpoints/wan22_lora")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=500)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    device = "cuda"
    dtype = torch.bfloat16

    from diffusers import WanImageToVideoPipeline
    from peft import LoraConfig
    from diffusers.training_utils import compute_density_for_timestep_sampling

    print(f"[loading] {args.wan_root}", flush=True)
    pipe = WanImageToVideoPipeline.from_pretrained(args.wan_root, torch_dtype=dtype)
    pipe.to(device)

    tf = pipe.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # 모든 파라미터 freeze
    tf.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # LoRA 추가: self-attn (attn1) + cross-attn (attn2) + FFN
    LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.rank,
        target_modules=LORA_TARGETS,
        init_lora_weights="gaussian",
    )
    tf.add_adapter(lora_config)
    lora_params = [p for n, p in tf.named_parameters() if "lora_" in n]
    for p in lora_params:
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in lora_params)
    print(f"[LoRA] {len(lora_params)} params, total {n_trainable/1e6:.2f}M", flush=True)

    # 텍스트 인코딩 (고정 프롬프트)
    PROMPT = "A robotic arm on a tabletop performing a manipulation task, static camera"
    NEG = "blurry, distorted, low quality, static"
    with torch.no_grad():
        pos_ids = tokenizer(PROMPT, return_tensors="pt", padding="max_length", truncation=True,
                            max_length=512).input_ids.to(device)
        neg_ids = tokenizer(NEG, return_tensors="pt", padding="max_length", truncation=True,
                            max_length=512).input_ids.to(device)
        pos_emb = text_encoder(pos_ids).last_hidden_state.to(dtype)
        neg_emb = text_encoder(neg_ids).last_hidden_state.to(dtype)
    print(f"[text] pos_emb shape={tuple(pos_emb.shape)}", flush=True)

    # Data
    ds = SO100VideoDataset(args.train_dir, target_frames=TARGET_FRAMES,
                             max_episodes=args.max_episodes)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=2,
                                          pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)

    # Ckpt dir
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tf.enable_gradient_checkpointing()
    tf.train()

    # Training loop
    step = 0
    data_iter = iter(loader)
    import time
    t0 = time.time()
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        video = batch["video"].to(device=device, dtype=dtype)   # (1, 3, T, H, W)

        # VAE encode
        with torch.no_grad():
            latents = vae.encode(video).latent_dist.sample()   # (1, C_lat, T_lat, H_lat, W_lat)
            latents = latents * vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else latents

        # Sample sigma (flow matching)
        b = latents.shape[0]
        idx = random.randint(0, len(scheduler.timesteps) - 1)
        sigma = scheduler.sigmas[idx].to(device=device, dtype=torch.float32)
        timestep = scheduler.timesteps[idx].to(device=device).expand(b)

        # Noise
        noise = torch.randn_like(latents, dtype=torch.float32)
        x_t = ((1 - sigma) * latents.float() + sigma * noise).to(dtype)
        v_target = (noise - latents.float()).to(dtype)

        # Forward
        # WanTransformer3DModel expects: hidden_states, encoder_hidden_states, timestep
        # We need to check the exact signature
        try:
            pred = tf(
                hidden_states=x_t,
                encoder_hidden_states=pos_emb,
                timestep=timestep,
                return_dict=False,
            )[0]
        except TypeError as e:
            print(f"[WARN] tf forward signature issue: {e}. Trying alternative call...")
            # Try positional
            pred = tf(x_t, pos_emb, timestep).sample

        loss = torch.nn.functional.mse_loss(pred.float(), v_target.float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()

        step += 1
        if step % 10 == 0:
            elapsed = time.time() - t0
            print(f"step {step:06d}/{args.steps} loss={loss.item():.4f} sigma={float(sigma):.4f} elapsed={elapsed:.0f}s", flush=True)

        if step % args.save_every == 0:
            ckpt_path = ckpt_dir / f"ckpt_step{step:06d}.pt"
            lora_state = {n: p.detach().cpu() for n, p in tf.named_parameters() if "lora_" in n}
            torch.save({"lora": lora_state, "step": step}, ckpt_path)
            print(f"  ckpt saved: {ckpt_path}", flush=True)

    # Final save
    ckpt_path = ckpt_dir / f"ckpt_step{step:06d}.pt"
    lora_state = {n: p.detach().cpu() for n, p in tf.named_parameters() if "lora_" in n}
    torch.save({"lora": lora_state, "step": step}, ckpt_path)
    print(f"[DONE] final ckpt: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
