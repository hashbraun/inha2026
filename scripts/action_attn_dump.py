"""진단#2: vision query → action key attention weight 덤프.

Cosmos3PackedMoTAttention의 gen pathway attention에서:
- Q at vision positions × K at action positions만 slice
- softmax 후 각 vision query가 17개 action key에 부여한 weight 분포
- entropy 계산: uniform(=log 17)에 가까우면 bag-of-tokens 확정, sharp하면 기각

Pipeline이 vision_sequence_indexes, action_sequence_indexes를 갖고 있으므로
이걸 forward에 훅으로 접근.

비교 조건:
- pretrained (no LoRA)
- v5b_20k
- B4 (v5b_eloss_bw step 6000)
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
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3AttnProcessor
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


def probe_pipe(pipe, tag, action_np, vision_indexes_holder, action_indexes_holder):
    """1-step forward with instrumented attention to dump vision→action weights."""
    device = "cuda"
    dtype = torch.bfloat16
    tf = pipe.transformer

    # Monkey-patch Cosmos3AttnProcessor.__call__ to capture Q, K
    attn_weights_per_layer = []  # list of (vision_len, action_len) tensors

    orig_call = Cosmos3AttnProcessor.__call__

    def probe_call(self, attn, und_seq, gen_seq, rotary_emb):
        # Do the full computation but ALSO compute Q_vision · K_action.T
        q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        cos_gen_u = cos_gen.unsqueeze(1); sin_gen_u = sin_gen.unsqueeze(1)
        from diffusers.models.transformers.transformer_cosmos3 import _rotate_half
        q_gen_r = q_gen * cos_gen_u + _rotate_half(q_gen) * sin_gen_u
        k_gen_r = k_gen * cos_gen_u + _rotate_half(k_gen) * sin_gen_u

        # Slice: gen_seq indices relative to gen. vision_indexes/action_indexes are indices
        # in full joint seq, need to subtract und_len
        vi = vision_indexes_holder[0]
        ai = action_indexes_holder[0]
        # gen part starts after und tokens. vi/ai are stored as gen-relative positions.
        # Take q at vision positions, k at action positions
        if vi is not None and ai is not None and len(vi) > 0 and len(ai) > 0:
            q_v = q_gen_r[vi]   # (num_vision, H, D)
            k_a = k_gen_r[ai]   # (num_action, Hk, D)
            # GQA: repeat k for num_attention_heads
            n_rep = q_v.shape[1] // k_a.shape[1]
            k_a_rep = k_a.repeat_interleave(n_rep, dim=1)  # (num_action, H, D)
            # attention scores: (H, num_vision, num_action)
            scale = 1.0 / (attn.head_dim ** 0.5)
            scores = torch.einsum('vhd,ahd->hva', q_v.float(), k_a_rep.float()) * scale
            weights = F.softmax(scores, dim=-1)  # (H, V, A)
            # entropy per (H, V): -sum(w * log(w+eps))
            eps = 1e-9
            entropy = -(weights * (weights + eps).log()).sum(dim=-1)  # (H, V)
            attn_weights_per_layer.append({
                "H": weights.shape[0], "V": weights.shape[1], "A": weights.shape[2],
                "entropy_mean": float(entropy.mean().item()),
                "entropy_median": float(entropy.median().item()),
                "log_A_uniform": float(np.log(weights.shape[2])),
                "weight_max_avg": float(weights.max(dim=-1).values.mean().item()),
            })

        # Now do the real full compute (return actual result)
        return orig_call(self, attn, und_seq, gen_seq, rotary_emb)

    Cosmos3AttnProcessor.__call__ = probe_call

    try:
        # Setup one probe call: intercept prepare_inputs to save vision/action indexes
        sid = "sample_000000"
        image = Image.open(EVAL_DIR / "images" / f"{sid}.png").convert("RGB")
        raw = torch.from_numpy(action_np).to(device=device, dtype=dtype)
        cond = make_action_condition(raw, chunk_size=CHUNK_SIZE, domain_name=DOMAIN_NAME,
                                       resolution_tier=RESOLUTION_TIER, image=image)

        # We need vision_sequence_indexes and action_sequence_indexes.
        # Easiest: hook the transformer.forward to capture kwargs before layers run.
        orig_forward = tf.forward
        def hooked_forward(*args, **kwargs):
            vs = kwargs.get("vision_sequence_indexes")
            as_ = kwargs.get("action_sequence_indexes")
            und_len = kwargs.get("und_len", 0)
            if vs is not None and as_ is not None:
                # Convert to gen-relative (subtract und_len)
                vs_np = vs.detach().cpu().numpy() if torch.is_tensor(vs) else np.array(vs)
                as_np = as_.detach().cpu().numpy() if torch.is_tensor(as_) else np.array(as_)
                if hasattr(as_, '__iter__') and not isinstance(as_, torch.Tensor):
                    as_np = as_[0].detach().cpu().numpy() if torch.is_tensor(as_[0]) else np.array(as_[0])
                vision_indexes_holder[0] = torch.tensor(vs_np - und_len, device=device, dtype=torch.long)
                action_indexes_holder[0] = torch.tensor(as_np - und_len, device=device, dtype=torch.long)
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

    if attn_weights_per_layer:
        # aggregate across layers
        # Layer 0 is at the beginning, later layers process more abstracted features
        early = attn_weights_per_layer[:len(attn_weights_per_layer) // 3]
        mid = attn_weights_per_layer[len(attn_weights_per_layer) // 3: 2 * len(attn_weights_per_layer) // 3]
        late = attn_weights_per_layer[2 * len(attn_weights_per_layer) // 3:]
        for phase_name, group in [("early", early), ("mid", mid), ("late", late)]:
            if not group: continue
            avg_ent = np.mean([w["entropy_mean"] for w in group])
            avg_max = np.mean([w["weight_max_avg"] for w in group])
            uniform_ent = group[0]["log_A_uniform"]
            print(f"  [{tag}][{phase_name}] entropy={avg_ent:.4f} / uniform={uniform_ent:.4f} "
                  f"(ratio={avg_ent/uniform_ent:.2%}) weight_max_avg={avg_max:.4f} "
                  f"(uniform=1/{group[0]['A']}={1/group[0]['A']:.4f})", flush=True)
        return {
            "n_layers": len(attn_weights_per_layer),
            "avg_entropy": float(np.mean([w["entropy_mean"] for w in attn_weights_per_layer])),
            "uniform_entropy": attn_weights_per_layer[0]["log_A_uniform"],
        }
    return None


def main():
    action_np = None
    for tag, ckpt in CKPTS.items():
        print(f"\n=== {tag} (ckpt={ckpt}) ===", flush=True)
        pipe, device, dtype = load_pipe(ckpt)
        if action_np is None:
            joints = np.load(EVAL_DIR / "actions" / "sample_000000.npy")
            action_np = build_action_features(joints[:CHUNK_SIZE], "delta_base")
        vh = [None]
        ah = [None]
        result = probe_pipe(pipe, tag, action_np, vh, ah)
        if result:
            ratio = result["avg_entropy"] / result["uniform_entropy"]
            print(f"  [{tag}] SUMMARY: entropy={result['avg_entropy']:.4f} / uniform={result['uniform_entropy']:.4f} "
                  f"= {ratio:.2%} (>95% = uniform bag-of-tokens, <70% = sharp attention)")
        del pipe
        import gc; gc.collect()
        torch.cuda.empty_cache()

    print("\n=== 판정 ===")
    print("- entropy/uniform > 95% → attention이 17 action tokens에 균등 (bag-of-tokens 확정)")
    print("- entropy/uniform 70-95% → 일부 집중")
    print("- entropy/uniform < 70% → sharp attention (bag-of-tokens 기각, 다른 원인)")


if __name__ == "__main__":
    main()
