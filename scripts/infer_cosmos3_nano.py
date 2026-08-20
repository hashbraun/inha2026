"""
cosmos3_nano_v1 체크포인트(LoRA + action_proj/action_modality_embed)로 eval 샘플 추론.
"""
import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from so100_to_bridge_v3 import ACTION_REPRS, build_action_features, make_action_condition  # noqa: E402

from diffusers import Cosmos3OmniPipeline  # noqa: E402
from peft import LoraConfig  # noqa: E402

# 학습 때 쓴 표현과 반드시 일치해야 한다 (--action-repr).
ACTION_REPR = "absolute"


def to_bridge10(joints):
    return build_action_features(joints, ACTION_REPR)

MODEL_ID = "nvidia/Cosmos3-Nano"
DOMAIN_NAME = "bridge_orig_lerobot"
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
import os as _os
_lora_targets_env = _os.environ.get("LORA_TARGETS")
if _lora_targets_env:
    LORA_TARGET_MODULES = [s.strip() for s in _lora_targets_env.split(",") if s.strip()]
    print(f"[LORA_TARGETS override] {LORA_TARGET_MODULES}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--guidance", type=float, default=6.0,
                    help="탐색 결과 3.5가 균형점 (Video 최고, Action 15% 개선)")
    ap.add_argument("--action-repr", type=str, default="absolute", choices=ACTION_REPRS)
    # 추론 개선 옵션 (v5_base baseline에 하나씩 적용해 리더보드 실험)
    ap.add_argument("--cfg-rescale", type=float, default=0.0,
                    help="CFG rescale phi (0=비활성, 0.7 권장). 색·노출 왜곡 보정")
    ap.add_argument("--sampler", type=str, default="default", choices=["default", "dpmpp"],
                    help="dpmpp = DPMSolverMultistepScheduler (선명한 픽셀)")
    ap.add_argument("--use-captions", action="store_true",
                    help="eval_captions.json에서 sample별 캡션 주입 (99/216 유효, 46%%)")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch.Generator seed (multi-seed 앙상블용)")
    args = ap.parse_args()

    global ACTION_REPR
    ACTION_REPR = args.action_repr
    print(f"설정: guidance={args.guidance}, steps={args.steps}, action_repr={ACTION_REPR}")

    device = "cuda"
    dtype = torch.bfloat16
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
    pipe.to(device)
    transformer = pipe.transformer

    if args.sampler == "dpmpp":
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print(f"sampler: DPMSolverMultistepScheduler (base: {pipe.scheduler.__class__.__name__})")

    lora_config = LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=LORA_TARGET_MODULES)
    transformer.add_adapter(lora_config)

    ckpt = torch.load(args.ckpt, map_location=device)
    sd = transformer.state_dict()
    missing = []
    for name, tensor in ckpt["lora"].items():
        if name in sd:
            sd[name] = tensor.to(device=device, dtype=sd[name].dtype)
        else:
            missing.append(name)
    transformer.load_state_dict(sd, strict=False)
    transformer.action_proj_in.load_state_dict(ckpt["action_proj_in"])
    transformer.action_proj_out.load_state_dict(ckpt["action_proj_out"])
    with torch.no_grad():
        transformer.action_modality_embed.copy_(ckpt["action_modality_embed"].to(device))
    print(f"체크포인트 로드 완료 (step={ckpt['step']}, LoRA 텐서 {len(ckpt['lora'])}개, 누락 {len(missing)}개)")
    transformer.eval()

    img_paths = sorted((EVAL_DIR / "images").glob("*.png"))[: args.samples]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    captions = None
    if args.use_captions:
        import json as _json
        with open(EVAL_CAPTIONS) as f:
            captions = _json.load(f)
        n_valid = sum(1 for v in captions.values() if v and v.strip())
        print(f"captions loaded: {n_valid}/{len(captions)} 유효 ({100*n_valid//len(captions)}%)")

    for img_path in img_paths:
        sid = img_path.stem
        image = Image.open(img_path).convert("RGB")
        joints = np.load(EVAL_DIR / "actions" / f"{sid}.npy")  # (16,6)
        bridge10 = to_bridge10(joints[:CHUNK_SIZE])
        raw_actions = torch.from_numpy(bridge10).to(device=device, dtype=dtype)

        cond = make_action_condition(
            raw_actions,
            chunk_size=CHUNK_SIZE,
            domain_name=DOMAIN_NAME,
            resolution_tier=RESOLUTION_TIER,
            image=image,
        )
        prompt = PROMPT
        if captions is not None:
            cap = captions.get(sid, "").strip()
            if cap:
                prompt = f"A robotic arm on a tabletop performing the following task: {cap} Static camera."
        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                action=cond,
                generator=generator,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
            )
        frames = np.stack([np.array(f) for f in out.video])[:16]  # (16,H,W,3)
        iio.imwrite(out_dir / f"{sid}.mp4", frames, fps=6, codec="libx264")
        print(f"  {sid}: {frames.shape} 저장 완료")

    print("추론 완료:", out_dir)


if __name__ == "__main__":
    main()
