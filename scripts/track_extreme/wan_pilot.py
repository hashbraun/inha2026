"""Wan2.2-TI2V-5B zero-shot 12-sample kill pilot.

- diffusers WanImageToVideoPipeline
- 320x512 × 16 frames × 6fps (eval spec)
- 첫 프레임 + 프롬프트 → generate
- Action grounding 없음 (zero-shot)
- Kill 판정용 (E-invdyn L1 + AlexNet cosine)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

WAN_ID = "/home1/sota/inha2026/models/wan22_ti2v_5b"
EVAL_DIR = Path("/home1/sota/inha2026/data/eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    from diffusers import WanImageToVideoPipeline

    device = "cuda"
    dtype = torch.bfloat16
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[loading] Wan2.2-TI2V-5B from {WAN_ID}", flush=True)
    pipe = WanImageToVideoPipeline.from_pretrained(WAN_ID, torch_dtype=dtype)
    pipe.to(device)
    print("[loaded]", flush=True)

    prompt = ("A robotic arm on a tabletop performing a manipulation task, "
               "static camera, realistic lighting, sharp focus")
    neg = "blurry, distorted, low quality, static image, frozen"

    img_paths = sorted((EVAL_DIR / "images").glob("*.png"))[: args.samples]
    generator = torch.Generator(device=device).manual_seed(0)

    for img_path in img_paths:
        sid = img_path.stem
        image = Image.open(img_path).convert("RGB").resize((args.width, args.height), Image.BICUBIC)
        try:
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=neg,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            )
            frames = result.frames[0]  # list of PIL images
            if isinstance(frames, list):
                frames_np = np.stack([np.array(f) for f in frames])
            else:
                frames_np = np.array(frames)
            frames_np = frames_np[: args.num_frames]
            iio.imwrite(out / f"{sid}.mp4", frames_np, fps=args.fps, codec="libx264")
            print(f"  {sid}: {frames_np.shape} saved", flush=True)
        except Exception as e:
            print(f"  {sid} ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
