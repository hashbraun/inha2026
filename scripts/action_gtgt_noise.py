"""gt-gt 노이즈 플로어: 같은 GT action + 같은 프레임, seed만 바꿔 N회 생성.
목적: 두 생성 결과의 픽셀 MAE 분포 → 시드 잡음 floor.
이게 7 근처면 앞선 진단 표 전부 노이즈.
"""
import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home1/sota/inha2026/scripts")
from so100_to_bridge_v3 import build_action_features, make_action_condition
from diffusers import Cosmos3OmniPipeline
from peft import LoraConfig

MODEL_ID = "nvidia/Cosmos3-Nano"
DOMAIN_NAME = "bridge_orig_lerobot"
CHUNK_SIZE = 17
RESOLUTION_TIER = 480
PROMPT = "A robotic arm on a tabletop performing a manipulation task, static camera."
NEG = "blurry, distorted, low quality, static image, frozen"
EVAL_DIR = Path("/home1/sota/inha2026/data/eval")
OUT_ROOT = Path("/home1/sota/inha2026/submission_kit/action_gtgt_noise")

LORA_TARGET = ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj"]

STEPS = 35
GUIDANCE = 6.0
RANK = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="none=pretrained, path=fine-tuned")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--action-repr", default="delta_base")
    args = ap.parse_args()

    out = OUT_ROOT / args.tag
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16
    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
    pipe.to(device)
    tf = pipe.transformer

    if args.ckpt.lower() != "none":
        tf.add_adapter(LoraConfig(r=RANK, lora_alpha=RANK, target_modules=LORA_TARGET))
        ck = torch.load(args.ckpt, map_location=device)
        sd = tf.state_dict()
        for name, tensor in ck["lora"].items():
            if name in sd:
                sd[name] = tensor.to(device=device, dtype=sd[name].dtype)
        tf.load_state_dict(sd, strict=False)
        tf.action_proj_in.load_state_dict(ck["action_proj_in"])
        tf.action_proj_out.load_state_dict(ck["action_proj_out"])
        with torch.no_grad():
            tf.action_modality_embed.copy_(ck["action_modality_embed"].to(device))
        print(f"[loaded] {args.ckpt} step={ck['step']}")
    else:
        print("[pretrained] no LoRA")
    tf.eval()

    sample_ids = [f"sample_{i:06d}" for i in range(args.n_samples)]

    for sid in sample_ids:
        image = Image.open(EVAL_DIR / "images" / f"{sid}.png").convert("RGB")
        joints = np.load(EVAL_DIR / "actions" / f"{sid}.npy")
        bridge_gt = build_action_features(joints[:CHUNK_SIZE], args.action_repr)
        raw = torch.from_numpy(bridge_gt).to(device=device, dtype=dtype)
        cond = make_action_condition(raw, chunk_size=CHUNK_SIZE, domain_name=DOMAIN_NAME,
                                       resolution_tier=RESOLUTION_TIER, image=image)
        for seed in range(args.n_seeds):
            gen = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                res = pipe(prompt=PROMPT, negative_prompt=NEG, action=cond,
                            generator=gen, num_inference_steps=STEPS, guidance_scale=GUIDANCE)
            frames = np.stack([np.array(f) for f in res.video])[:16]
            iio.imwrite(out / f"{sid}_seed{seed}.mp4", frames, fps=6, codec="libx264")
            print(f"  {sid}_seed{seed} saved", flush=True)

    print(f"=== done: {out} ===")


if __name__ == "__main__":
    main()
