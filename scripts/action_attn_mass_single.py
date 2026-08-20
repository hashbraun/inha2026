"""단일 ckpt attention mass 측정 (orchestrator에서 env로 ckpt 전달)."""
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

# LORA_TARGETS env
_lora_targets_env = os.environ.get("LORA_TARGETS")
if _lora_targets_env:
    LORA_TARGET = [s.strip() for s in _lora_targets_env.split(",") if s.strip()]
else:
    LORA_TARGET = ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                    "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj"]
print(f"[LORA_TARGET] {LORA_TARGET}")

CKPT = os.environ["MASS_ONLY_CKPT"]
TAG = os.environ.get("MASS_CKPT_TAG", "unknown")

device = "cuda"
dtype = torch.bfloat16
pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
pipe.to(device)
tf = pipe.transformer
if CKPT.lower() != "none":
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
    print(f"[loaded] {CKPT} step={ck.get('step')}")
tf.eval()

layer_stats = []
orig_call = Cosmos3AttnProcessor.__call__
vh = [None]; ah = [None]; ulh = [0]

def probe_call(self, attn, und_seq, gen_seq, rotary_emb):
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
        vision_abs = vi + und_len
        vision_mass = weights[:, :, vision_abs].sum(dim=-1).mean().item()
        total_len = all_k_rep.shape[0]
        layer_stats.append({
            "action_mass": action_mass, "vision_mass": vision_mass,
            "action_uniform": len(ai) / total_len, "total_len": total_len,
        })
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

if layer_stats:
    uf = layer_stats[0]['action_uniform']
    am = float(np.mean([s['action_mass'] for s in layer_stats]))
    vm = float(np.mean([s['vision_mass'] for s in layer_stats]))
    ratio = am / uf if uf > 0 else 0
    # per phase
    n = len(layer_stats)
    early = layer_stats[:n // 3]; mid = layer_stats[n//3:2*n//3]; late = layer_stats[2*n//3:]
    for pn, g in [("early", early), ("mid", mid), ("late", late)]:
        if not g: continue
        a = float(np.mean([s['action_mass'] for s in g]))
        print(f"  [{TAG}][{pn}] action_mass={a*100:.4f}% ratio={a/uf:.2f}x")
    print(f"[{TAG}] SUMMARY: action_mass={am*100:.4f}% uniform={uf*100:.3f}% ratio={ratio:.2f}x vision_mass={vm*100:.2f}%")
