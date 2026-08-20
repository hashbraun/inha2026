"""B4 (v5b_eloss_bw/step6000) 위 action ablation:
같은 seed 로 GT / zeros / randn / reverse 4가지 action 조건 → 픽셀 MAE 비교."""
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from so100_to_bridge_v3 import build_action_features, make_action_condition  # noqa: E402
from diffusers import Cosmos3OmniPipeline  # noqa: E402
from peft import LoraConfig  # noqa: E402

MODEL_ID = "nvidia/Cosmos3-Nano"
DOMAIN_NAME = "bridge_orig_lerobot"
CHUNK_SIZE = 17
RESOLUTION_TIER = 480
PROMPT = "A robotic arm on a tabletop performing a manipulation task, static camera."
NEGATIVE_PROMPT = "blurry, distorted, low quality, static image, frozen"
EVAL_DIR = Path("/home1/sota/inha2026/data/eval")
CKPT = Path("/home1/sota/inha2026/checkpoints/v5b_eloss_bw/ckpt_step006000.pt")   # B4
OUT_ROOT = Path("/home1/sota/inha2026/submission_kit/action_ablation_b4")

LORA_TARGET_MODULES = [
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
]

N_SAMPLES = 3
STEPS = 35
GUIDANCE = 6.0
RANK = 32
ACTION_REPR = "delta_base"

device = "cuda"
dtype = torch.bfloat16

print(f"[B4 ablation] ckpt={CKPT.name}, N={N_SAMPLES}, 4 conditions")

pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
pipe.to(device)
tf = pipe.transformer
lora_config = LoraConfig(r=RANK, lora_alpha=RANK, target_modules=LORA_TARGET_MODULES)
tf.add_adapter(lora_config)
ckpt = torch.load(CKPT, map_location=device)
sd = tf.state_dict()
for name, tensor in ckpt["lora"].items():
    if name in sd:
        sd[name] = tensor.to(device=device, dtype=sd[name].dtype)
tf.load_state_dict(sd, strict=False)
tf.action_proj_in.load_state_dict(ckpt["action_proj_in"])
tf.action_proj_out.load_state_dict(ckpt["action_proj_out"])
with torch.no_grad():
    tf.action_modality_embed.copy_(ckpt["action_modality_embed"].to(device))
tf.eval()
print(f"[loaded] step={ckpt['step']}")

img_paths = sorted((EVAL_DIR / "images").glob("*.png"))[:N_SAMPLES]

for cond_name in ("gt", "zeros", "randn", "reverse"):
    out_dir = OUT_ROOT / cond_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- {cond_name} → {out_dir} ---")
    gen = torch.Generator(device=device).manual_seed(0)
    for img_path in img_paths:
        sid = img_path.stem
        image = Image.open(img_path).convert("RGB")
        joints = np.load(EVAL_DIR / "actions" / f"{sid}.npy")
        bridge_gt = build_action_features(joints[:CHUNK_SIZE], ACTION_REPR)

        if cond_name == "gt":
            action_np = bridge_gt
        elif cond_name == "zeros":
            action_np = np.zeros_like(bridge_gt)
        elif cond_name == "randn":
            rng = np.random.RandomState(hash(sid) & 0xFFFFFFFF)
            action_np = rng.randn(*bridge_gt.shape).astype(bridge_gt.dtype)
        else:  # reverse
            action_np = bridge_gt[::-1].copy()

        raw = torch.from_numpy(action_np).to(device=device, dtype=dtype)
        cond = make_action_condition(raw, chunk_size=CHUNK_SIZE, domain_name=DOMAIN_NAME,
                                       resolution_tier=RESOLUTION_TIER, image=image)
        with torch.no_grad():
            out = pipe(prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
                       action=cond, generator=gen,
                       num_inference_steps=STEPS, guidance_scale=GUIDANCE)
        frames = np.stack([np.array(f) for f in out.video])[:16]
        iio.imwrite(out_dir / f"{sid}.mp4", frames, fps=6, codec="libx264")
        print(f"  {sid}: saved")

print("=== B4 ablation done ===")
