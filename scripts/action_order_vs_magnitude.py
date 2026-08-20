"""순서 vs 크기 분리 실험:
- shuffle_time: GT 액션의 시간축 무작위 순서 셔플 (크기 보존, 순서 파괴)
- scale_0.5, scale_2.0: GT × 0.5, × 2.0 (순서 보존, 크기 변경)

(b)만 크면 경로 구조 문제 확정 → gated Stage-1도 ID head도 순위 밀림
(a)와 (b) 모두 GT와 크게 다르면 방향/순서 둘 다 반응하는 정상 FD
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
OUT_ROOT = Path("/home1/sota/inha2026/submission_kit/action_order_vs_mag")
LORA_TARGET = ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj"]
STEPS = 35
GUIDANCE = 6.0
RANK = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--action-repr", default="delta_base")
    ap.add_argument("--seed", type=int, default=0)
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
        print(f"[loaded] {args.ckpt}")
    tf.eval()

    sample_ids = [f"sample_{i:06d}" for i in range(args.n_samples)]
    conditions = ["gt", "shuffle_time", "scale_0.5", "scale_2.0"]

    for cond_name in conditions:
        cd = out / cond_name
        cd.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {cond_name} ---")
        for sid in sample_ids:
            image = Image.open(EVAL_DIR / "images" / f"{sid}.png").convert("RGB")
            joints = np.load(EVAL_DIR / "actions" / f"{sid}.npy")
            bridge_gt = build_action_features(joints[:CHUNK_SIZE], args.action_repr)

            if cond_name == "gt":
                action_np = bridge_gt.copy()
            elif cond_name == "shuffle_time":
                # 시간축만 셔플 (원소 집합 보존, 순서 파괴)
                rng = np.random.RandomState(hash(sid) & 0xFFFFFFFF)
                perm = rng.permutation(bridge_gt.shape[0])
                action_np = bridge_gt[perm].copy()
            elif cond_name == "scale_0.5":
                action_np = (bridge_gt * 0.5).copy()
            elif cond_name == "scale_2.0":
                action_np = (bridge_gt * 2.0).copy()

            raw = torch.from_numpy(action_np).to(device=device, dtype=dtype)
            cond = make_action_condition(raw, chunk_size=CHUNK_SIZE, domain_name=DOMAIN_NAME,
                                           resolution_tier=RESOLUTION_TIER, image=image)
            gen = torch.Generator(device=device).manual_seed(args.seed)
            with torch.no_grad():
                res = pipe(prompt=PROMPT, negative_prompt=NEG, action=cond,
                            generator=gen, num_inference_steps=STEPS, guidance_scale=GUIDANCE)
            frames = np.stack([np.array(f) for f in res.video])[:16]
            iio.imwrite(cd / f"{sid}.mp4", frames, fps=6, codec="libx264")
            print(f"  {sid} saved", flush=True)

    print(f"\n=== done: {out} ===")


if __name__ == "__main__":
    main()
