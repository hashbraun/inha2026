"""Wan2.2-TI2V-5B zero-shot pilot (diffusers-compat).

경쟁팀이 Wan2.2 + hyperparameter/loss 수정으로 성능 냈다는 정보 기반 blind 시도.
1단계: zero-shot inference (action 없이, 첫 프레임 + 텍스트만)로 시각 품질 확인.
Cosmos3-Nano baseline 대비 시각적 improvement 있는지 판단.
"""
import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

WAN_ROOT = "/home1/sota/inha2026/models/wan22_ti2v_5b_diffusers"
EVAL_DIR = Path("/home1/sota/inha2026/data/eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--fps-out", type=int, default=6)
    args = ap.parse_args()

    from diffusers import WanImageToVideoPipeline, WanPipeline

    device = "cuda"
    dtype = torch.bfloat16
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[loading] Wan2.2-TI2V-5B-Diffusers from {WAN_ROOT}", flush=True)
    # 우선 WanImageToVideoPipeline 시도, 안 되면 WanPipeline
    try:
        pipe = WanImageToVideoPipeline.from_pretrained(WAN_ROOT, torch_dtype=dtype)
    except Exception as e:
        print(f"[WanImageToVideoPipeline fail] {e}", flush=True)
        print("[fallback] WanPipeline (T2V) 시도", flush=True)
        pipe = WanPipeline.from_pretrained(WAN_ROOT, torch_dtype=dtype)
    pipe.to(device)
    if hasattr(pipe, "enable_model_cpu_offload"):
        pass  # 필요시 사용 (5B라 여유 있음)
    print("[loaded]", flush=True)

    prompt = ("A robotic arm on a tabletop performing a manipulation task, "
              "static camera, realistic lighting, sharp focus")
    neg = "blurry, distorted, low quality, static image, frozen"

    img_paths = sorted((EVAL_DIR / "images").glob("*.png"))[: args.samples]
    gen = torch.Generator(device=device).manual_seed(0)

    for img_path in img_paths:
        sid = img_path.stem
        image = Image.open(img_path).convert("RGB").resize((args.width, args.height), Image.BICUBIC)
        try:
            call_kwargs = dict(
                prompt=prompt,
                negative_prompt=neg,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=gen,
            )
            # WanImageToVideoPipeline이면 image 인자, WanPipeline이면 없음
            if isinstance(pipe, WanImageToVideoPipeline):
                call_kwargs["image"] = image
            result = pipe(**call_kwargs)
            frames = result.frames[0]
            if isinstance(frames, list):
                frames_np = np.stack([np.array(f) for f in frames])
            else:
                frames_np = np.array(frames)
            frames_np = frames_np[: args.num_frames]
            iio.imwrite(out / f"{sid}.mp4", frames_np, fps=args.fps_out, codec="libx264")
            print(f"  {sid}: {frames_np.shape} saved", flush=True)
        except Exception as e:
            print(f"  {sid} ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
