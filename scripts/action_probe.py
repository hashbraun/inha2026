"""진단#3: action 텐서가 실제로 transformer.forward까지 도달하는지 검증.

각 조건 (gt/zero/randn/reverse)에서:
1. make_action_condition 후 cond.raw_actions.norm() 출력
2. pipeline 내부에서 raw_actions.norm() 출력 (monkey-patch)
3. transformer.forward에서 action_tokens.norm() 출력 (monkey-patch)

모두 다른 값이면 텐서 도달 OK. 같으면 파이프라인 어딘가에서 override.
"""
import sys
from pathlib import Path

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
CKPT = Path("/home1/sota/inha2026/checkpoints/v5b_eloss_bw/ckpt_step006000.pt")
LORA_TARGET = ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj"]

device = "cuda"
dtype = torch.bfloat16

print("=== ACTION TENSOR PROBE (B4 ckpt) ===")
print()

pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
pipe.to(device)
tf = pipe.transformer
tf.add_adapter(LoraConfig(r=32, lora_alpha=32, target_modules=LORA_TARGET))
ck = torch.load(CKPT, map_location=device)
sd = tf.state_dict()
for name, tensor in ck["lora"].items():
    if name in sd:
        sd[name] = tensor.to(device=device, dtype=sd[name].dtype)
tf.load_state_dict(sd, strict=False)
tf.action_proj_in.load_state_dict(ck["action_proj_in"])
tf.action_proj_out.load_state_dict(ck["action_proj_out"])
with torch.no_grad():
    tf.action_modality_embed.copy_(ck["action_modality_embed"].to(device))
tf.eval()
print(f"[loaded] B4 step {ck['step']}")

# Monkey-patch transformer forward to log action_tokens
orig_forward = tf.forward
def probe_forward(*args, **kwargs):
    action_tokens = kwargs.get("action_tokens")
    if action_tokens is not None:
        for i, at in enumerate(action_tokens):
            print(f"  [transformer.forward] action_tokens[{i}]: shape={tuple(at.shape)} "
                  f"norm={at.float().norm().item():.4f} "
                  f"mean={at.float().mean().item():.4f} "
                  f"std={at.float().std().item():.4f}")
    return orig_forward(*args, **kwargs)
tf.forward = probe_forward

# Test conditions
sid = "sample_000000"
image = Image.open(EVAL_DIR / "images" / f"{sid}.png").convert("RGB")
joints = np.load(EVAL_DIR / "actions" / f"{sid}.npy")
bridge_gt = build_action_features(joints[:CHUNK_SIZE], "delta_base")

test_conditions = {
    "gt": bridge_gt.copy(),
    "zeros": np.zeros_like(bridge_gt),
    "randn": np.random.RandomState(42).randn(*bridge_gt.shape).astype(bridge_gt.dtype),
    "reverse": bridge_gt[::-1].copy(),
    "gt_2x": (bridge_gt * 2.0).astype(bridge_gt.dtype),
}

for cond_name, action_np in test_conditions.items():
    print(f"\n--- condition: {cond_name} ---")
    print(f"  [input] action_np shape={action_np.shape} "
          f"norm={np.linalg.norm(action_np):.4f} "
          f"mean={action_np.mean():.4f} "
          f"std={action_np.std():.4f}")
    raw = torch.from_numpy(action_np).to(device=device, dtype=dtype)
    print(f"  [tensor] raw shape={tuple(raw.shape)} norm={raw.float().norm().item():.4f}")
    cond = make_action_condition(raw, chunk_size=CHUNK_SIZE, domain_name=DOMAIN_NAME,
                                    resolution_tier=RESOLUTION_TIER, image=image)
    print(f"  [cond] cond.raw_actions norm={cond.raw_actions.float().norm().item():.4f} "
          f"shape={tuple(cond.raw_actions.shape)}")
    # Single-step forward (steps=1, guidance=1.0) to trigger tf.forward once
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        # 매우 짧게 (단순히 tf.forward가 실제 호출되는지만)
        out = pipe(prompt=PROMPT, negative_prompt=NEG, action=cond,
                    generator=gen, num_inference_steps=1, guidance_scale=1.0)
    print(f"  [done] pipe returned video shape={np.array(out.video[0]).shape if out.video else None}")

print("\n=== PROBE COMPLETE ===")
print("판정: 각 조건에서 [transformer.forward] action_tokens norm이 다르면 도달 OK")
print("      모두 같으면 파이프라인에서 override 발생 (진단 전제 뒤집힘)")
