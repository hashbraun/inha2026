"""mRoPE 재정렬 테스트 - 4개 매핑 옵션 peak alignment 비교.

Baseline (원래 pipeline):
- Action t_index: [1.0, 2.0, ..., 17.0] (step 1.0)
- Vision t_index: [0.0, 4.0, 8.0, 12.0] (step 4.0)
- 실제 pos_ids: Action step 0.25 (vision과 fps 정규화), Vision step 1.0

옵션들 (모두 vision과 같은 discrete t_index space):
A. GROUP_HARD: action[4k..4k+3] → t=vision[k]. residual (action 16) → t=vision[3]
B. GROUP_CENTER: 각 그룹 중심 (0.375, 1.375, 2.375, 3.375)
C. CONTINUOUS_LERP: action t = k/(N-1) * (V-1) → [0, 0.1875, ..., 3.0]
D. VISION_ALIGN_1TO1: action t = min(k // 4, V-1) → [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,3] (D=A와 유사)

Peak alignment 규칙: 각 vision t=k → expected action peak = 그룹 중심 (4k+1.5).
error = |observed peak (17 space) - expected|
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
from diffusers.pipelines.cosmos import pipeline_cosmos3_omni as p_module
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

# 매핑 옵션들
def make_action_t_indices(mode, action_len=17, vision_T=4):
    """action_len개 토큰에 vision t space (0..V-1)로 t_index 부여."""
    if mode == "baseline":
        # 원래 pipeline이 계산하는 것 (이것은 monkey patch 안 함)
        return None
    if mode == "GROUP_HARD":
        # action[4k..4k+3] → t=k. residual → t=V-1
        idx = []
        for k in range(action_len):
            g = min(k // 4, vision_T - 1)
            idx.append(float(g))
        return np.array(idx, dtype=np.float32)
    if mode == "GROUP_CENTER":
        # 각 그룹 중심 (0.375, 1.375, ..., or per-token evenly)
        # 4 tokens per group, center at group_start + 0.375
        idx = []
        for k in range(action_len):
            g = min(k // 4, vision_T - 1)
            offset_in_group = k - g * 4
            # 그룹 안에서 균등 (0, 0.25, 0.5, 0.75 offset from group start)
            t = g + offset_in_group * 0.25 - 0.375  # 그룹 중심에 대해 상대적
            idx.append(t)
        return np.array(idx, dtype=np.float32)
    if mode == "CONTINUOUS_LERP":
        # 0..(V-1)에 균등 분포 (16 step)
        return np.linspace(0.0, vision_T - 1, action_len).astype(np.float32)
    if mode == "OFFSET_SHIFT":
        # 원래 매핑에 +1 shift 되어 있는 것 제거
        # Original: 15125.25, 15125.5, ..., 즉 first action offset 0.25
        # Try: action t_index = k * 0.25 (start at 0)
        # 이건 vision t와 정확히 정렬 (vision start at 0.0)
        return (np.arange(action_len) * 0.25).astype(np.float32)
    raise ValueError(mode)


# get_3d_mrope_ids_vae_tokens monkey-patch
_orig_get_3d = p_module.get_3d_mrope_ids_vae_tokens
CURRENT_MODE = ["baseline"]

def patched_get_3d(grid_t, grid_h, grid_w, temporal_offset,
                    reset_spatial_indices=True, fps=None, base_fps=24.0,
                    temporal_compression_factor=4, base_temporal_compression_factor=None,
                    start_frame_offset=0):
    # action call은 grid_t=17 (=CHUNK_SIZE) grid_h=grid_w=1
    is_action = (grid_h == 1 and grid_w == 1 and grid_t == 17)
    orig_result = _orig_get_3d(grid_t, grid_h, grid_w, temporal_offset,
                                  reset_spatial_indices=reset_spatial_indices,
                                  fps=fps, base_fps=base_fps,
                                  temporal_compression_factor=temporal_compression_factor,
                                  base_temporal_compression_factor=base_temporal_compression_factor,
                                  start_frame_offset=start_frame_offset)
    mrope_ids, next_offset = orig_result
    if is_action and CURRENT_MODE[0] != "baseline":
        # action mrope_ids 재정의
        # mrope_ids shape: (3, grid_t)  [t, h, w]
        custom_t = make_action_t_indices(CURRENT_MODE[0], action_len=grid_t, vision_T=4)
        if custom_t is not None:
            # temporal offset 유지
            device_ = mrope_ids.device
            dtype_ = mrope_ids.dtype
            new_t = torch.tensor(custom_t + float(temporal_offset), device=device_, dtype=dtype_)
            mrope_ids = mrope_ids.clone()
            mrope_ids[0] = new_t
    return mrope_ids, next_offset

p_module.get_3d_mrope_ids_vae_tokens = patched_get_3d


def probe_peak_alignment(pipe, tag):
    """peak alignment 측정."""
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
            weights = F.softmax(scores, dim=-1)
            action_abs = ai + und_len
            aw = weights[:, :, action_abs]   # (H, V, 17)
            captured["layers"].append(aw.cpu())
        return orig_call(self, attn, und_seq, gen_seq, rotary_emb)
    Cosmos3AttnProcessor.__call__ = probe_call

    tf = pipe.transformer
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
            captured["n_vision"] = len(vs_np)
        return orig_forward(*args, **kwargs)
    tf.forward = hf

    try:
        sid = "sample_000000"
        image = Image.open(EVAL_DIR / "images" / f"{sid}.png").convert("RGB")
        joints = np.load(EVAL_DIR / "actions" / f"{sid}.npy")
        action_np = build_action_features(joints[:CHUNK_SIZE], "delta_base")
        raw = torch.from_numpy(action_np).to(device=device, dtype=dtype)
        cond = make_action_condition(raw, chunk_size=CHUNK_SIZE, domain_name=DOMAIN_NAME,
                                       resolution_tier=RESOLUTION_TIER, image=image)
        with torch.no_grad():
            _ = pipe(prompt=PROMPT, negative_prompt=NEG, action=cond,
                       generator=torch.Generator(device=device).manual_seed(0),
                       num_inference_steps=1, guidance_scale=1.0)
    finally:
        Cosmos3AttnProcessor.__call__ = orig_call
        tf.forward = orig_forward

    if not captured["layers"]:
        return None
    n_v = captured["n_vision"]
    T_lat = 4
    spatial_per_t = n_v // T_lat
    n_layers = len(captured["layers"])

    alignment_errors = []
    for l_idx, aw in enumerate(captured["layers"]):
        aw_hmean = aw.mean(dim=0)   # (V, 17)
        for v_t in range(T_lat):
            s_start = v_t * spatial_per_t
            s_end = min((v_t + 1) * spatial_per_t, n_v)
            aw_t = aw_hmean[s_start:s_end].mean(dim=0)
            peak = int(aw_t.argmax().item())
            expected = min(4 * v_t + 1, 16)
            alignment_errors.append(abs(peak - expected))
    return {"mean": np.mean(alignment_errors), "median": np.median(alignment_errors),
             "high_err_frac": float((np.array(alignment_errors) >= 6).sum() / len(alignment_errors))}


# 5개 모드 테스트
MODES = ["baseline", "GROUP_HARD", "GROUP_CENTER", "CONTINUOUS_LERP", "OFFSET_SHIFT"]
results = {}
for mode in MODES:
    CURRENT_MODE[0] = mode
    print(f"\n=== TEST: {mode} ===", flush=True)
    # action t_indices preview
    if mode != "baseline":
        custom = make_action_t_indices(mode)
        print(f"  action t_indices: {custom.tolist()}")
    r = probe_peak_alignment(pipe, mode)
    if r:
        print(f"  → mean_err={r['mean']:.2f} median={r['median']:.1f} high_err(≥6)={r['high_err_frac']:.1%}", flush=True)
        results[mode] = r

print("\n" + "=" * 60)
print(f"{'mode':22s} {'mean_err':>10s} {'high_err':>10s} {'판정':>10s}")
best_mode, best_err = None, 99
for mode, r in results.items():
    verdict = "🟢 성공" if r['mean'] < 2.0 else ("🟡 부분" if r['mean'] < 3.5 else "🔴 실패")
    if r['mean'] < best_err:
        best_err = r['mean']; best_mode = mode
    print(f"{mode:22s} {r['mean']:10.2f} {r['high_err_frac']:10.1%} {verdict:>10s}")

print(f"\nBEST: {best_mode} (mean_err={best_err:.2f})")
print(f"Baseline: {results.get('baseline', {}).get('mean', 'N/A')}")
