"""Z5 inference wrapper: v5b LoRA + Z5 Cross-attn adapter attached.

infer_cosmos3_nano.py의 로직을 그대로 재사용하고 adapter attach + forward wrap만 추가.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home1/sota/inha2026/scripts")
sys.path.insert(0, "/home1/sota/inha2026/scripts/track_z5")

from so100_to_bridge_v3 import ACTION_REPRS, build_action_features, make_action_condition
from diffusers import Cosmos3OmniPipeline
from peft import LoraConfig
from crossattn_adapter import CrossAttnAdapter

ACTION_REPR = "delta_base"
MODEL_ID = "nvidia/Cosmos3-Nano"
DOMAIN_NAME = "bridge_orig_lerobot"
DOMAIN_ID = 7
CHUNK_SIZE = 17
RESOLUTION_TIER = 480
PROMPT = "A robotic arm on a tabletop performing a manipulation task, static camera."
NEGATIVE_PROMPT = "blurry, distorted, low quality, static image, frozen"
EVAL_DIR = Path("/home1/sota/inha2026/data/eval")
EVAL_CAPTIONS = EVAL_DIR / "eval_captions.json"

LORA_TARGET_MODULES = [
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
]


def to_bridge10(joints):
    return build_action_features(joints, ACTION_REPR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5b-ckpt", required=True)
    ap.add_argument("--z5-ckpt", required=True)
    ap.add_argument("--samples", type=int, default=216)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--action-repr", type=str, default="delta_base", choices=ACTION_REPRS)
    ap.add_argument("--use-captions", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    global ACTION_REPR
    ACTION_REPR = args.action_repr

    device = "cuda"
    dtype = torch.bfloat16
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[z5-inf] v5b={Path(args.v5b_ckpt).name} z5={Path(args.z5_ckpt).name}")
    print(f"[z5-inf] samples={args.samples} steps={args.steps} guidance={args.guidance}")

    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
    pipe.to(device)
    tr = pipe.transformer

    tr.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=LORA_TARGET_MODULES))

    ck = torch.load(args.v5b_ckpt, map_location=device)
    sd = tr.state_dict()
    for name, tensor in ck["lora"].items():
        if name in sd:
            sd[name] = tensor.to(device=device, dtype=sd[name].dtype)
    tr.load_state_dict(sd, strict=False)
    tr.action_proj_in.load_state_dict(ck["action_proj_in"])
    tr.action_proj_out.load_state_dict(ck["action_proj_out"])
    with torch.no_grad():
        tr.action_modality_embed.copy_(ck["action_modality_embed"].to(device))
    print(f"[z5-inf] v5b loaded step={ck['step']}")

    z5_ck = torch.load(args.z5_ckpt, map_location=device)
    target_layers = z5_ck["target_layers"]
    adapter = CrossAttnAdapter(
        hidden_size=int(tr.config.hidden_size),
        action_hidden_size=int(tr.config.hidden_size),
        film_hidden_size=z5_ck["film_hidden_size"],
        attn_hidden_size=z5_ck.get("attn_hidden_size", 512),
        num_heads=z5_ck.get("num_heads", 8),
        target_layers=target_layers,
        chunk_size=CHUNK_SIZE,
    ).to(device=device, dtype=torch.float32)
    adapter.load_state_dict(z5_ck["crossattn_adapter"])
    adapter.attach_to(tr)
    adapter.eval()
    print(f"[z5-inf] Z5 adapter loaded step={z5_ck['step']} layers={target_layers[0]}..{target_layers[-1]}")

    # Forward wrap: encoded_action set_context 자동
    orig_forward = tr.forward
    def wrapped_forward(*a, **kw):
        action_tokens = kw.get("action_tokens", None)
        vision_token_shapes = kw.get("vision_token_shapes", None)
        action_domain_ids = kw.get("action_domain_ids", None)
        if action_tokens and vision_token_shapes and action_domain_ids:
            x0_action = action_tokens[0]
            dom_t = action_domain_ids[0]
            dom_val = int(dom_t.item() if dom_t.numel() == 1 else dom_t[0].item())
            per_tok = torch.full((x0_action.shape[0],), dom_val, dtype=torch.long, device=x0_action.device)
            with torch.no_grad():
                packed = tr.action_proj_in(x0_action, per_tok)
                packed = packed + tr.action_modality_embed
            packed = packed.to(dtype=next(adapter.parameters()).dtype)
            enc = adapter.encode_action(packed)
            num_v = 0
            for sh in vision_token_shapes:
                num_v += int(sh[0]) * int(sh[1]) * int(sh[2])
            adapter.set_context(enc, num_v)
        try:
            return orig_forward(*a, **kw)
        finally:
            adapter.clear_context()
    tr.forward = wrapped_forward
    tr.eval()

    img_paths = sorted((EVAL_DIR / "images").glob("*.png"))[: args.samples]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    captions = None
    if args.use_captions and EVAL_CAPTIONS.exists():
        import json as _json
        with open(EVAL_CAPTIONS) as f:
            captions = _json.load(f)

    n_done = 0
    for img_path in img_paths:
        sid = img_path.stem
        image = Image.open(img_path).convert("RGB")
        joints = np.load(EVAL_DIR / "actions" / f"{sid}.npy")
        bridge10 = to_bridge10(joints[:CHUNK_SIZE])
        raw_actions = torch.from_numpy(bridge10).to(device=device, dtype=dtype)

        cond = make_action_condition(
            raw_actions, chunk_size=CHUNK_SIZE, domain_name=DOMAIN_NAME,
            resolution_tier=RESOLUTION_TIER, image=image,
        )
        prompt = PROMPT
        if captions is not None:
            cap = captions.get(sid, "").strip()
            if cap:
                prompt = f"A robotic arm on a tabletop performing the following task: {cap} Static camera."

        with torch.no_grad():
            out = pipe(
                prompt=prompt, negative_prompt=NEGATIVE_PROMPT, action=cond,
                generator=generator, num_inference_steps=args.steps,
                guidance_scale=args.guidance,
            )
        frames = np.stack([np.array(f) for f in out.video])[:16]
        iio.imwrite(str(out_dir / f"{sid}.mp4"), frames, fps=6, codec="libx264")
        n_done += 1
        if n_done % 20 == 0 or n_done < 3:
            print(f"  [{n_done}/{len(img_paths)}] {sid}: {frames.shape}", flush=True)

    print(f"[z5-inf] done: {n_done} mp4 → {out_dir}")


if __name__ == "__main__":
    main()
