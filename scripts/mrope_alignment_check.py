"""mRoPE t_index 정렬 검증.

action 토큰과 vision latent 토큰의 t_index 실제 값을 dump.
- action: grid_t=17, temporal_compression_factor=1
- vision: grid_t=T_lat=4, temporal_compression_factor=vae_scale (=4)

핵심 질문:
- 두 modality의 t_index가 같은 시간축으로 정렬돼 있는가?
- 4x 스케일 mismatch가 존재하는가?
"""
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, "/home1/sota/inha2026/scripts")
from so100_to_bridge_v3 import build_action_features, make_action_condition
from diffusers import Cosmos3OmniPipeline
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import get_3d_mrope_ids_vae_tokens

MODEL_ID = "nvidia/Cosmos3-Nano"
DOMAIN_NAME = "bridge_orig_lerobot"
CHUNK_SIZE = 17
RESOLUTION_TIER = 480

device = "cuda"
dtype = torch.bfloat16

pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
pipe.to(device)
tf = pipe.transformer
cfg = tf.config
vae_cfg = pipe.vae.config

print("=" * 60)
print("Cosmos3-Nano mRoPE t_index 정렬 진단")
print("=" * 60)
print(f"base_fps: {cfg.base_fps}")
print(f"enable_fps_modulation: {cfg.enable_fps_modulation}")
print(f"vae temporal compression: {vae_cfg.scale_factor_temporal}")
print(f"unified_3d_mrope_reset_spatial_ids: {getattr(cfg, 'unified_3d_mrope_reset_spatial_ids', 'N/A')}")

# ---- Action mRoPE ids (grid_t=17) ----
action_len = CHUNK_SIZE
action_fps = 6.0  # eval spec

action_mrope, action_next = get_3d_mrope_ids_vae_tokens(
    grid_t=action_len, grid_h=1, grid_w=1,
    temporal_offset=0,
    reset_spatial_indices=getattr(cfg, 'unified_3d_mrope_reset_spatial_ids', True),
    fps=action_fps if cfg.enable_fps_modulation else None,
    base_fps=float(cfg.base_fps),
    temporal_compression_factor=1,
    base_temporal_compression_factor=vae_cfg.scale_factor_temporal,
    start_frame_offset=1,
)
print()
print("--- ACTION mRoPE (grid_t=17, temporal_compression=1, fps=6) ---")
print(f"shape: {tuple(action_mrope.shape)}")
print(f"t_index values (row 0): {action_mrope[0].tolist()}")
print(f"h_index (row 1): {action_mrope[1].tolist()}")
print(f"w_index (row 2): {action_mrope[2].tolist()}")

# ---- Vision mRoPE ids (grid_t=T_lat) ----
# 16 pixel frames × 4x compression = 4 latent frames
T_lat = 4
video_fps = 6.0
vae_temporal = vae_cfg.scale_factor_temporal

# vision has grid_h × grid_w latent tokens per t frame (for 320×512 pixel: ~40×64=2560 tokens per frame? need to check)
# For mRoPE calculation, use grid_h=1 grid_w=1 to isolate t axis
vision_mrope_t_only, _ = get_3d_mrope_ids_vae_tokens(
    grid_t=T_lat, grid_h=1, grid_w=1,
    temporal_offset=0,
    reset_spatial_indices=True,
    fps=video_fps if cfg.enable_fps_modulation else None,
    base_fps=float(cfg.base_fps),
    temporal_compression_factor=vae_temporal,
    base_temporal_compression_factor=vae_temporal,
    start_frame_offset=0,
)
print()
print(f"--- VISION mRoPE (grid_t={T_lat}, temporal_compression={vae_temporal}, fps={video_fps}) ---")
print(f"shape: {tuple(vision_mrope_t_only.shape)}")
print(f"t_index values (row 0, spatial 1x1): {vision_mrope_t_only[0].tolist()}")

# ---- 정렬 비교 ----
print()
print("=" * 60)
print("=== t_index 정렬 비교 ===")
print(f"ACTION t_indices (17개): {action_mrope[0].tolist()}")
print(f"VISION t_indices ({T_lat}개): {vision_mrope_t_only[0].tolist()}")

# 시간축 실제 스케일
if cfg.enable_fps_modulation:
    action_step = float(action_mrope[0][1] - action_mrope[0][0])
    vision_step = float(vision_mrope_t_only[0][1] - vision_mrope_t_only[0][0]) if T_lat > 1 else float('nan')
    ratio = vision_step / action_step if action_step != 0 else float('inf')
    print(f"\nAction t_index step: {action_step:.4f}")
    print(f"Vision t_index step: {vision_step:.4f}")
    print(f"Ratio (vision/action): {ratio:.4f}x  (예상 = {vae_temporal}x if scale mismatch)")

# action 범위 vs vision 범위
a_min, a_max = float(action_mrope[0].min()), float(action_mrope[0].max())
v_min, v_max = float(vision_mrope_t_only[0].min()), float(vision_mrope_t_only[0].max())
print(f"\nAction t 범위: [{a_min:.2f}, {a_max:.2f}]  span={a_max-a_min:.2f}")
print(f"Vision t 범위: [{v_min:.2f}, {v_max:.2f}]  span={v_max-v_min:.2f}")
print(f"두 범위 겹침: [{max(a_min, v_min):.2f}, {min(a_max, v_max):.2f}]")

# 실제 pipeline 실행 시 vision과 action의 sequence 배치 확인
print()
print("=== 실제 pipeline 실행에서 vision + action position_ids 축소 확인 ===")
EVAL_DIR = Path("/home1/sota/inha2026/data/eval")
sid = "sample_000000"
image = Image.open(EVAL_DIR / "images" / f"{sid}.png").convert("RGB")
joints = np.load(EVAL_DIR / "actions" / f"{sid}.npy")
action_np = build_action_features(joints[:CHUNK_SIZE], "delta_base")
raw = torch.from_numpy(action_np).to(device=device, dtype=dtype)
cond = make_action_condition(raw, chunk_size=CHUNK_SIZE, domain_name=DOMAIN_NAME,
                                resolution_tier=RESOLUTION_TIER, image=image)

captured = {}
orig_forward = tf.forward
def hooked(*args, **kwargs):
    pos_ids = kwargs.get("position_ids")
    vs = kwargs.get("vision_sequence_indexes")
    as_ = kwargs.get("action_sequence_indexes")
    und_len = kwargs.get("und_len", 0)
    captured["pos_ids"] = pos_ids.detach().cpu() if pos_ids is not None else None
    captured["vision_indexes"] = vs.detach().cpu() if torch.is_tensor(vs) else vs
    captured["action_indexes"] = as_
    captured["und_len"] = und_len
    return orig_forward(*args, **kwargs)
tf.forward = hooked

with torch.no_grad():
    _ = pipe(prompt="test", negative_prompt="", action=cond,
               generator=torch.Generator(device=device).manual_seed(0),
               num_inference_steps=1, guidance_scale=1.0)

if captured:
    pos = captured["pos_ids"]
    vs = captured["vision_indexes"]
    as_ = captured["action_indexes"]
    und = captured["und_len"]
    print(f"und_len: {und}")
    print(f"pos_ids shape: {tuple(pos.shape) if pos is not None else 'None'}")
    if pos is not None:
        print(f"  pos_ids[0] (t axis) at vision indexes (first 10): {pos[0, vs[:10]].tolist() if hasattr(vs, '__len__') else 'N/A'}")
        if hasattr(as_, '__iter__') and not isinstance(as_, torch.Tensor):
            as_np = as_[0].detach().cpu()
        else:
            as_np = as_.detach().cpu() if torch.is_tensor(as_) else torch.tensor(as_)
        print(f"  pos_ids[0] (t axis) at action indexes (all 17): {pos[0, as_np].tolist()}")

print()
print("=" * 60)
print("=== 판정 ===")
print("Ratio ≈ 4x → action-vision t_index 스케일 mismatch (mRoPE 가설 지지)")
print("Ratio ≈ 1x → 정렬 정상, mRoPE 가설 기각 → 원인 더 깊은 곳 (attention 학습 자체 문제)")
