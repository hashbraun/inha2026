"""LoRA adapter checkpoint interpolation.

두 ckpt를 alpha:beta 비율로 선형 보간하여 새 ckpt 저장.
- lora, action_proj_in, action_proj_out, action_modality_embed 모두 선형 결합.
- 모든 3개 base ckpt는 동일 구조 (r=32, 504 tensors) 확인 완료.
"""
import argparse
from pathlib import Path

import torch


def interp_state_dict(sd_a, sd_b, alpha, beta):
    out = {}
    for k in sd_a:
        if k not in sd_b:
            raise KeyError(f"key mismatch: {k}")
        out[k] = alpha * sd_a[k].float() + beta * sd_b[k].float()
        out[k] = out[k].to(sd_a[k].dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="weight for ckpt-a")
    ap.add_argument("--beta", type=float, default=None, help="weight for ckpt-b (default: 1-alpha)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    beta = args.beta if args.beta is not None else (1.0 - args.alpha)
    print(f"interp: {args.alpha:.3f}*A + {beta:.3f}*B")

    a = torch.load(args.ckpt_a, map_location="cpu", weights_only=False)
    b = torch.load(args.ckpt_b, map_location="cpu", weights_only=False)

    out = {
        "lora": interp_state_dict(a["lora"], b["lora"], args.alpha, beta),
        "action_proj_in": interp_state_dict(a["action_proj_in"], b["action_proj_in"], args.alpha, beta),
        "action_proj_out": interp_state_dict(a["action_proj_out"], b["action_proj_out"], args.alpha, beta),
        "action_modality_embed": args.alpha * a["action_modality_embed"].float()
                                + beta * b["action_modality_embed"].float(),
        "step": f"interp({args.alpha:.2f}*step{a.get('step','?')}+{beta:.2f}*step{b.get('step','?')})",
    }
    out["action_modality_embed"] = out["action_modality_embed"].to(a["action_modality_embed"].dtype)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"저장: {args.out}")
    print(f"  lora: {len(out['lora'])} tensors, action_proj_in.fc: {out['action_proj_in']['fc.weight'].shape}")


if __name__ == "__main__":
    main()
