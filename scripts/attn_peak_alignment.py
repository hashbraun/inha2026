"""Peak alignment 검증 — mRoPE 위상이 attention에 실제로 반영되는가.

각 vision t-frame 별로:
- 그 vision frame의 spatial 토큰들이 17개 action key에 주는 attention 분포
- Peak가 vision t와 대응하는 action t (예: vision t=k → action t=[4k, 4k+1, 4k+2, 4k+3])
  근처에 있는가?

수치화:
- expected peak center: vision t의 4 action 그룹 시작 index
- observed peak: attention weight argmax over 17 action tokens
- Alignment error: |observed - expected| (0=완벽, 8=완전 무작위 예상)

- pretrained만 대상 (fine-tuned는 이미 mass 열화 확인됨)
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

MODEL_ID = "nvidia/Cosmos3-Nano"
DOMAIN_NAME = "bridge_orig_lerobot"
CHUNK_SIZE = 17
RESOLUTION_TIER = 480
PROMPT = "A robotic arm on a tabletop performing a manipulation task, static camera."
NEG = "blurry, distorted, low quality, static image, frozen"
EVAL_DIR = Path("/home1/sota/inha2026/data/eval")

device = "cuda"; dtype = torch.bfloat16

pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
pipe.to(device)
tf = pipe.transformer
tf.eval()

# Capture layer-wise attention scores
captured = {"vh": [None], "ah": [None], "ulh": [0], "layers": []}

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
    vi = captured["vh"][0]; ai = captured["ah"][0]; und_len = captured["ulh"][0]
    if vi is not None and ai is not None:
        all_k = torch.cat([k_und_r, k_gen_r], dim=0)
        n_rep = q_gen_r.shape[1] // all_k.shape[1]
        all_k_rep = all_k.repeat_interleave(n_rep, dim=1)
        q_v = q_gen_r[vi]
        scale = 1.0 / (attn.head_dim ** 0.5)
        scores = torch.einsum('vhd,thd->hvt', q_v.float(), all_k_rep.float()) * scale
        weights = F.softmax(scores, dim=-1)   # (H, V, T_total)
        action_abs = ai + und_len
        # 액션 17개에 대한 attention weight만 추출
        action_weights = weights[:, :, action_abs]   # (H, V, 17)
        # head + spatial 평균 → per vision-t 별 action-token 분포
        # vi은 gen-relative index. gen 안에서 vision layout = T×H×W flattened
        # T_lat=4, spatial=(vi_count / 4)
        captured["layers"].append(action_weights.cpu())
    return orig_call(self, attn, und_seq, gen_seq, rotary_emb)

Cosmos3AttnProcessor.__call__ = probe_call

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
        captured["vh"][0] = torch.tensor(vs_np - und_len, device=device, dtype=torch.long)
        captured["ah"][0] = torch.tensor(as_np - und_len, device=device, dtype=torch.long)
        captured["ulh"][0] = und_len
        # 저장: vision 개수, T_lat 추정 (첫 몇 개 같은 t이면 spatial 그룹)
        captured["n_vision"] = len(vs_np)
    return orig_forward(*args, **kwargs)
tf.forward = hf

with torch.no_grad():
    _ = pipe(prompt=PROMPT, negative_prompt=NEG, action=cond,
               generator=torch.Generator(device=device).manual_seed(0),
               num_inference_steps=1, guidance_scale=1.0)

print(f"n_vision_tokens: {captured['n_vision']}")
n_v = captured['n_vision']
# T_lat 추정: 4 frames로 가정 → spatial = n_v / 4
T_lat = 4
spatial_per_t = n_v // T_lat
print(f"assumed T_lat={T_lat}, spatial per t = {spatial_per_t}")

# 층별 분석
n_layers = len(captured["layers"])
print(f"\nn_attention_layers: {n_layers}")

# 각 층 × 각 vision t-frame 별 peak action index
print(f"\n{'layer':>5s} {'v_t':>3s} {'peak':>4s} {'weight':>7s} {'expected':>8s} {'error':>5s}")
alignment_errors = []
for l_idx, aw in enumerate(captured["layers"]):
    # aw: (H, V, 17)
    aw_hmean = aw.mean(dim=0)  # (V, 17)  head averaged
    for v_t in range(T_lat):
        # vision t=v_t의 spatial 토큰 인덱스: v_t * spatial_per_t ~ (v_t+1) * spatial_per_t
        s_start = v_t * spatial_per_t
        s_end = min((v_t + 1) * spatial_per_t, n_v)
        # 이 그룹의 평균 action distribution
        aw_t = aw_hmean[s_start:s_end].mean(dim=0)   # (17,)
        peak = int(aw_t.argmax().item())
        peak_w = float(aw_t.max().item())
        # expected peak = vision t=k에 대응하는 action t = [4k, 4k+1, 4k+2, 4k+3]. center 4k+1.5 rounded
        expected = 4 * v_t + 1   # +1 offset from action t 시작 (mRoPE offset 1)
        expected = min(expected, 16)
        error = abs(peak - expected)
        alignment_errors.append(error)
        if l_idx in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
            print(f"{l_idx:5d} {v_t:3d} {peak:4d} {peak_w:7.4f} {expected:8d} {error:5d}")

print(f"\n=== Alignment 통계 ===")
alignment_errors = np.array(alignment_errors)
print(f"mean alignment error: {alignment_errors.mean():.2f}  (0=완벽, 8=완전무작위 예상)")
print(f"median: {np.median(alignment_errors):.2f}, std: {alignment_errors.std():.2f}")
print(f"error 분포: 0~2 {(alignment_errors <= 2).sum()}, 3~5 {((alignment_errors >=3) & (alignment_errors <= 5)).sum()}, 6+ {(alignment_errors >=6).sum()} / total {len(alignment_errors)}")
print(f"random baseline: uniform 1/17 attention → mean expected error ~4.24")

print(f"\n=== 판정 ===")
mean_err = alignment_errors.mean()
if mean_err < 2.0:
    print(f"🟢 peak alignment 성공 ({mean_err:.2f} < 2.0) → mRoPE 정상, 원인 미상 (다른 원인)")
elif mean_err < 3.5:
    print(f"🟡 부분 alignment ({mean_err:.2f}) → 위상 약하게 반영")
else:
    print(f"🔴 alignment 실패 ({mean_err:.2f} ≥ 3.5, uniform baseline ~4.24) → 위상 정렬 실패 확정")
    print(f"   수정: action t_index step 0.25 → 1.0 (vision과 같은 해상도)")
