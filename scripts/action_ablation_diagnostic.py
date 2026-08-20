"""배관 vs 최적화 진단: 다양한 ckpt/action_repr로 action ablation.

각 (ckpt, action_repr) 조합에 대해 3 sample × (GT/zero/randn/reverse) 생성.
결과: submission_kit/action_ablation_diag/{tag}/{cond}/*.mp4
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
OUT_ROOT = Path("/home1/sota/inha2026/submission_kit/action_ablation_diag")

LORA_TARGET_MODULES = [
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
]

N_SAMPLES = 3
STEPS = 35
GUIDANCE = 6.0
RANK = 32


def load_pipe():
    device = "cuda"
    dtype = torch.bfloat16
    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
    pipe.to(device)
    return pipe, device, dtype


def apply_ckpt(pipe, ckpt_path, device, dtype):
    """ckpt가 None이면 원본 Cosmos3-Nano (LoRA 없음)."""
    tf = pipe.transformer
    if ckpt_path is None:
        # LoRA 자체를 추가하지 않음 → base 상태
        print("[ckpt] BASE Cosmos3-Nano (no LoRA, no fine-tune)")
        return
    lora_config = LoraConfig(r=RANK, lora_alpha=RANK, target_modules=LORA_TARGET_MODULES)
    tf.add_adapter(lora_config)
    ck = torch.load(ckpt_path, map_location=device)
    sd = tf.state_dict()
    for name, tensor in ck["lora"].items():
        if name in sd:
            sd[name] = tensor.to(device=device, dtype=sd[name].dtype)
    tf.load_state_dict(sd, strict=False)
    tf.action_proj_in.load_state_dict(ck["action_proj_in"])
    tf.action_proj_out.load_state_dict(ck["action_proj_out"])
    with torch.no_grad():
        tf.action_modality_embed.copy_(ck["action_modality_embed"].to(device))
    print(f"[ckpt] {ckpt_path} step={ck['step']}")


def run_ablation(pipe, device, dtype, action_repr, tag, sample_ids):
    tf = pipe.transformer
    tf.eval()
    for cond_name in ("gt", "zeros", "randn", "reverse"):
        out_dir = OUT_ROOT / tag / cond_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {tag}/{cond_name}", flush=True)
        gen = torch.Generator(device=device).manual_seed(0)
        for sid in sample_ids:
            image = Image.open(EVAL_DIR / "images" / f"{sid}.png").convert("RGB")
            joints = np.load(EVAL_DIR / "actions" / f"{sid}.npy")
            bridge_gt = build_action_features(joints[:CHUNK_SIZE], action_repr)
            if cond_name == "gt":
                action_np = bridge_gt
            elif cond_name == "zeros":
                action_np = np.zeros_like(bridge_gt)
            elif cond_name == "randn":
                rng = np.random.RandomState(hash(sid) & 0xFFFFFFFF)
                action_np = rng.randn(*bridge_gt.shape).astype(bridge_gt.dtype)
            else:
                action_np = bridge_gt[::-1].copy()
            raw = torch.from_numpy(action_np).to(device=device, dtype=dtype)
            cond = make_action_condition(raw, chunk_size=CHUNK_SIZE, domain_name=DOMAIN_NAME,
                                          resolution_tier=RESOLUTION_TIER, image=image)
            with torch.no_grad():
                out = pipe(prompt=PROMPT, negative_prompt=NEG, action=cond,
                           generator=gen, num_inference_steps=STEPS, guidance_scale=GUIDANCE)
            frames = np.stack([np.array(f) for f in out.video])[:16]
            iio.imwrite(out_dir / f"{sid}.mp4", frames, fps=6, codec="libx264")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", required=True,
                    help="format: tag:ckpt:repr (ckpt=none 이면 pretrained)")
    args = ap.parse_args()

    sample_ids = [f"sample_{i:06d}" for i in range(N_SAMPLES)]

    for exp in args.experiments:
        tag, ckpt, repr_name = exp.split(":")
        print(f"\n=== EXPERIMENT: {tag} (ckpt={ckpt}, repr={repr_name}) ===", flush=True)
        pipe, device, dtype = load_pipe()
        ckpt_path = None if ckpt.lower() == "none" else ckpt
        apply_ckpt(pipe, ckpt_path, device, dtype)
        run_ablation(pipe, device, dtype, repr_name, tag, sample_ids)
        del pipe
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("\n=== all experiments done ===")


if __name__ == "__main__":
    main()
