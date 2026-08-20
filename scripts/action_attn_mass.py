"""진단#2 v2: vision→action attention MASS 측정.

이전 v1은 17 action tokens 사이 분포 entropy만 봄.
v2는 전체 key에서 action이 차지하는 mass 비율 측정.

핵심 지표:
- vision→vision mass (기준선)
- vision→action mass
- ratio = action_mass / uniform_baseline
  * uniform baseline = 17 / (und_len + gen_len) ≈ 1.1%
  * ratio > 1.5 → 순수 비율 이상, 실제 attention
  * ratio ≈ 1.0 → uniform (구조적으로 무시)
  * ratio < 0.5 → 능동 회피
"""
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
LORA_TARGET = ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj"]

CKPTS = {
    "pretrained": None,
    "v5b_20k": "/home1/sota/inha2026/checkpoints/cosmos3_nano_v5_base/ckpt_step020000.pt",
    "B4_step6000": "/home1/sota/inha2026/checkpoints/v5b_eloss_bw/ckpt_step006000.pt",
}


def load_pipe(ckpt_path):
    device = "cuda"
    dtype = torch.bfloat16
    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
    pipe.to(device)
    tf = pipe.transformer
    if ckpt_path is not None:
        tf.add_adapter(LoraConfig(r=32, lora_alpha=32, target_modules=LORA_TARGET))
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
    tf.eval()
    return pipe, device, dtype


def probe_pipe(pipe, tag, action_np, vision_indexes_holder, action_indexes_holder, und_len_holder):
    device = "cuda"
    dtype = torch.bfloat16
    tf = pipe.transformer

    layer_stats = []
    orig_call = Cosmos3AttnProcessor.__call__

    def probe_call(self, attn, und_seq, gen_seq, rotary_emb):
        # Full attention scores computation (for measurement only)
        q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)
        k_und = attn.norm_k(k_und)
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        cos_gen_u = cos_gen.unsqueeze(1); sin_gen_u = sin_gen.unsqueeze(1)
        cos_und_u = cos_und.unsqueeze(1); sin_und_u = sin_und.unsqueeze(1)
        q_gen_r = q_gen * cos_gen_u + _rotate_half(q_gen) * sin_gen_u
        k_gen_r = k_gen * cos_gen_u + _rotate_half(k_gen) * sin_gen_u
        k_und_r = k_und * cos_und_u + _rotate_half(k_und) * sin_und_u

        vi = vision_indexes_holder[0]
        ai = action_indexes_holder[0]
        und_len = und_len_holder[0]
        if vi is not None and ai is not None and len(vi) > 0 and len(ai) > 0:
            # 실제 전체 key: und + gen (all_k)
            all_k = torch.cat([k_und_r, k_gen_r], dim=0)   # (und_len + gen_len, Hk, D)
            # GQA rep
            n_rep = q_gen_r.shape[1] // all_k.shape[1]
            all_k_rep = all_k.repeat_interleave(n_rep, dim=1)   # (T, H, D)
            # Q at vision positions
            q_v = q_gen_r[vi]     # (V, H, D)
            # scores: (H, V, T)
            scale = 1.0 / (attn.head_dim ** 0.5)
            scores = torch.einsum('vhd,thd->hvt', q_v.float(), all_k_rep.float()) * scale
            weights = F.softmax(scores, dim=-1)   # (H, V, T)
            # T index layout: [0..und_len-1] = und, [und_len..und_len+len(gen)-1] = gen
            # gen 안에서 action indexes = ai (gen-relative)
            action_abs = ai + und_len   # 절대 인덱스
            und_mass = weights[:, :, :und_len].sum(dim=-1).mean().item()          # scalar
            gen_mass = weights[:, :, und_len:].sum(dim=-1).mean().item()
            action_mass = weights[:, :, action_abs].sum(dim=-1).mean().item()
            # vision self-attention mass
            vision_abs = vi + und_len
            vision_mass = weights[:, :, vision_abs].sum(dim=-1).mean().item()
            total_len = all_k_rep.shape[0]
            action_uniform_baseline = len(ai) / total_len
            layer_stats.append({
                "und_mass": und_mass,
                "gen_mass": gen_mass,
                "vision_mass": vision_mass,
                "action_mass": action_mass,
                "action_uniform_baseline": action_uniform_baseline,
                "total_len": total_len,
            })
        return orig_call(self, attn, und_seq, gen_seq, rotary_emb)

    Cosmos3AttnProcessor.__call__ = probe_call

    try:
        sid = "sample_000000"
        image = Image.open(EVAL_DIR / "images" / f"{sid}.png").convert("RGB")
        raw = torch.from_numpy(action_np).to(device=device, dtype=dtype)
        cond = make_action_condition(raw, chunk_size=CHUNK_SIZE, domain_name=DOMAIN_NAME,
                                       resolution_tier=RESOLUTION_TIER, image=image)

        orig_forward = tf.forward
        def hooked_forward(*args, **kwargs):
            vs = kwargs.get("vision_sequence_indexes")
            as_ = kwargs.get("action_sequence_indexes")
            und_len = kwargs.get("und_len", 0)
            if vs is not None and as_ is not None:
                vs_np = vs.detach().cpu().numpy() if torch.is_tensor(vs) else np.array(vs)
                if isinstance(as_, list) or (hasattr(as_, '__iter__') and not isinstance(as_, torch.Tensor)):
                    as_np = as_[0].detach().cpu().numpy() if torch.is_tensor(as_[0]) else np.array(as_[0])
                else:
                    as_np = as_.detach().cpu().numpy() if torch.is_tensor(as_) else np.array(as_)
                vision_indexes_holder[0] = torch.tensor(vs_np - und_len, device=device, dtype=torch.long)
                action_indexes_holder[0] = torch.tensor(as_np - und_len, device=device, dtype=torch.long)
                und_len_holder[0] = und_len
                print(f"  [hook] und_len={und_len}, vision={len(vs_np)}, action={len(as_np)}", flush=True)
            return orig_forward(*args, **kwargs)
        tf.forward = hooked_forward

        gen = torch.Generator(device=device).manual_seed(0)
        with torch.no_grad():
            _ = pipe(prompt=PROMPT, negative_prompt=NEG, action=cond,
                       generator=gen, num_inference_steps=1, guidance_scale=1.0)
        tf.forward = orig_forward
    finally:
        Cosmos3AttnProcessor.__call__ = orig_call

    if layer_stats:
        # Per phase
        n = len(layer_stats)
        early = layer_stats[:n // 3]
        mid = layer_stats[n // 3: 2 * n // 3]
        late = layer_stats[2 * n // 3:]
        print(f"  [{tag}] total_key_len = {layer_stats[0]['total_len']}, "
              f"action_uniform_baseline = {layer_stats[0]['action_uniform_baseline']*100:.3f}%")
        for phase_name, group in [("early", early), ("mid", mid), ("late", late)]:
            if not group: continue
            uf = layer_stats[0]['action_uniform_baseline']
            am = np.mean([s['action_mass'] for s in group])
            vm = np.mean([s['vision_mass'] for s in group])
            um = np.mean([s['und_mass'] for s in group])
            ratio_uf = am / uf if uf > 0 else 0
            ratio_v = am / vm if vm > 0 else 0
            print(f"  [{tag}][{phase_name}] und_mass={um*100:.2f}% vision_mass={vm*100:.2f}% "
                  f"action_mass={am*100:.4f}% (uniform={uf*100:.3f}%, ratio={ratio_uf:.2f}x, /vision={ratio_v*1000:.2f}‰)",
                  flush=True)
        return {
            "action_mass_avg": float(np.mean([s['action_mass'] for s in layer_stats])),
            "action_uniform_baseline": layer_stats[0]['action_uniform_baseline'],
            "vision_mass_avg": float(np.mean([s['vision_mass'] for s in layer_stats])),
        }
    return None


def main():
    action_np = None
    results = {}
    for tag, ckpt in CKPTS.items():
        print(f"\n=== {tag} (ckpt={ckpt}) ===", flush=True)
        pipe, device, dtype = load_pipe(ckpt)
        if action_np is None:
            joints = np.load(EVAL_DIR / "actions" / "sample_000000.npy")
            action_np = build_action_features(joints[:CHUNK_SIZE], "delta_base")
        vh = [None]; ah = [None]; ulh = [0]
        result = probe_pipe(pipe, tag, action_np, vh, ah, ulh)
        if result:
            am = result["action_mass_avg"]
            uf = result["action_uniform_baseline"]
            vm = result["vision_mass_avg"]
            ratio = am / uf if uf > 0 else 0
            print(f"  [{tag}] SUMMARY: action_mass={am*100:.4f}% uniform={uf*100:.3f}% "
                  f"ratio={ratio:.2f}x vision_mass={vm*100:.2f}%")
            results[tag] = result
        del pipe
        import gc; gc.collect()
        torch.cuda.empty_cache()

    print("\n=== 최종 판정 ===")
    for tag, r in results.items():
        am = r["action_mass_avg"]
        uf = r["action_uniform_baseline"]
        ratio = am / uf if uf > 0 else 0
        if ratio > 2.0:
            verdict = "🟢 강한 attention (실질 사용)"
        elif ratio > 1.2:
            verdict = "🟡 약한 signal"
        elif ratio > 0.5:
            verdict = "🔴 uniform 수준 (사실상 무시)"
        else:
            verdict = "⚫ 능동 회피"
        print(f"  {tag}: action_mass {am*100:.4f}% / uniform {uf*100:.3f}% = {ratio:.2f}x → {verdict}")


if __name__ == "__main__":
    main()
