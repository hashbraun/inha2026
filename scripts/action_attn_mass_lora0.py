"""LoRA 0-step mass 측정.

- add_adapter(LoraConfig)만 하고, weights 로드 안 함
- 두 init 모드 대조:
  * gaussian (finetune script 실제 사용)
  * default (표준 init, lora_B=0 → 수학적으로 pretrained와 동일)

- 각 mode에서 두 scope 시험:
  * O-only + MLP (Stage 3 scope)
  * full attention + MLP (기존 v5b/B4 scope)

- 대조 baseline: pretrained (no adapter)
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/home1/sota/inha2026/scripts")
from so100_to_bridge_v3 import build_action_features, make_action_condition
from diffusers import Cosmos3OmniPipeline
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3AttnProcessor, _rotate_half
from peft import LoraConfig

MODEL_ID = "nvidia/Cosmos3-Nano"
DOMAIN_NAME = "bridge_orig_lerobot"
CHUNK_SIZE = 17
RESOLUTION_TIER = 480
PROMPT = "A robotic arm on a tabletop performing a manipulation task, static camera."
NEG = "blurry, distorted, low quality, static image, frozen"
EVAL_DIR = Path("/home1/sota/inha2026/data/eval")

CONFIGS = [
    ("no_adapter", None, None),
    # O-only scope
    ("Oonly_gaussian", ["to_add_out", "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj"], "gaussian"),
    ("Oonly_default", ["to_add_out", "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj"], True),
    # Full attention scope (v5b/B4에서 쓴 것)
    ("full_gaussian",  ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out", "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj"], "gaussian"),
    ("full_default",   ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out", "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj"], True),
]


def probe(pipe, tag, action_np, vh, ah, ulh):
    device = "cuda"; dtype = torch.bfloat16
    tf = pipe.transformer
    layer_stats = []
    orig_call = Cosmos3AttnProcessor.__call__

    def probe_call(self, attn, und_seq, gen_seq, rotary_emb):
        q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.norm_added_q(q_gen); k_gen = attn.norm_added_k(k_gen); k_und = attn.norm_k(k_und)
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        cos_gen_u = cos_gen.unsqueeze(1); sin_gen_u = sin_gen.unsqueeze(1)
        cos_und_u = cos_und.unsqueeze(1); sin_und_u = sin_und.unsqueeze(1)
        q_gen_r = q_gen * cos_gen_u + _rotate_half(q_gen) * sin_gen_u
        k_gen_r = k_gen * cos_gen_u + _rotate_half(k_gen) * sin_gen_u
        k_und_r = k_und * cos_und_u + _rotate_half(k_und) * sin_und_u
        vi = vh[0]; ai = ah[0]; und_len = ulh[0]
        if vi is not None and ai is not None and len(vi) > 0 and len(ai) > 0:
            all_k = torch.cat([k_und_r, k_gen_r], dim=0)
            n_rep = q_gen_r.shape[1] // all_k.shape[1]
            all_k_rep = all_k.repeat_interleave(n_rep, dim=1)
            q_v = q_gen_r[vi]
            scale = 1.0 / (attn.head_dim ** 0.5)
            scores = torch.einsum('vhd,thd->hvt', q_v.float(), all_k_rep.float()) * scale
            weights = F.softmax(scores, dim=-1)
            action_abs = ai + und_len
            action_mass = weights[:, :, action_abs].sum(dim=-1).mean().item()
            total_len = all_k_rep.shape[0]
            layer_stats.append({"action_mass": action_mass, "action_uniform": len(ai)/total_len})
        return orig_call(self, attn, und_seq, gen_seq, rotary_emb)

    Cosmos3AttnProcessor.__call__ = probe_call

    try:
        sid = "sample_000000"
        image = Image.open(EVAL_DIR / "images" / f"{sid}.png").convert("RGB")
        joints = np.load(EVAL_DIR / "actions" / f"{sid}.npy")
        action_np = build_action_features(joints[:CHUNK_SIZE], "delta_base")
        raw = torch.from_numpy(action_np).to(device=device, dtype=dtype)
        cond = make_action_condition(raw, chunk_size=CHUNK_SIZE, domain_name=DOMAIN_NAME,
                                       resolution_tier=RESOLUTION_TIER, image=image)
        orig_forward = tf.forward
        def hf(*args, **kwargs):
            vs = kwargs.get("vision_sequence_indexes")
            as_ = kwargs.get("action_sequence_indexes")
            und_len = kwargs.get("und_len", 0)
            if vs is not None and as_ is not None:
                vs_np = vs.detach().cpu().numpy() if torch.is_tensor(vs) else np.array(vs)
                if isinstance(as_, list) or (hasattr(as_, '__iter__') and not isinstance(as_, torch.Tensor)):
                    as_np = as_[0].detach().cpu().numpy() if torch.is_tensor(as_[0]) else np.array(as_[0])
                else:
                    as_np = as_.detach().cpu().numpy() if torch.is_tensor(as_) else np.array(as_)
                vh[0] = torch.tensor(vs_np - und_len, device=device, dtype=torch.long)
                ah[0] = torch.tensor(as_np - und_len, device=device, dtype=torch.long)
                ulh[0] = und_len
            return orig_forward(*args, **kwargs)
        tf.forward = hf
        gen = torch.Generator(device=device).manual_seed(0)
        with torch.no_grad():
            _ = pipe(prompt=PROMPT, negative_prompt=NEG, action=cond, generator=gen,
                       num_inference_steps=1, guidance_scale=1.0)
    finally:
        Cosmos3AttnProcessor.__call__ = orig_call

    if not layer_stats:
        return None
    n = len(layer_stats)
    uf = layer_stats[0]['action_uniform']
    am = float(np.mean([s['action_mass'] for s in layer_stats]))
    mid = layer_stats[n//3:2*n//3]
    am_mid = float(np.mean([s['action_mass'] for s in mid])) if mid else am
    return {
        "action_mass_avg": am, "action_mass_mid": am_mid,
        "ratio_avg": am / uf, "ratio_mid": am_mid / uf,
    }


def main():
    device = "cuda"; dtype = torch.bfloat16
    print("=" * 60)
    print("LoRA 0-step mass 진단")
    print("=" * 60)
    print(f"{'config':22s} {'ratio_avg':>9s} {'ratio_mid':>9s} {'note':40s}")
    for tag, targets, init in CONFIGS:
        pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
        pipe.to(device)
        if targets is not None:
            lora_config = LoraConfig(r=32, lora_alpha=32, target_modules=targets, init_lora_weights=init)
            pipe.transformer.add_adapter(lora_config)
            note = f"init={init}, targets={len(targets)}"
        else:
            note = "no LoRA (baseline)"
        pipe.transformer.eval()
        vh = [None]; ah = [None]; ulh = [0]
        r = probe(pipe, tag, None, vh, ah, ulh)
        if r:
            print(f"{tag:22s} {r['ratio_avg']:9.2f} {r['ratio_mid']:9.2f} {note:40s}", flush=True)
        del pipe
        import gc; gc.collect(); torch.cuda.empty_cache()

    print()
    print("=== 해석 ===")
    print("no_adapter mid 2.07x (프리트레인 baseline)")
    print("- Oonly_default이 mid ≈ 2.07 → 학습이 파괴 (0-step OK)")
    print("- Oonly_gaussian이 mid ≪ 2.07 → gaussian init이 0-step에서 파괴 (구조 문제)")
    print("- full_default vs Oonly_default: attention 열려있는 config가 낮으면 attention scope이 문제")


if __name__ == "__main__":
    main()
