#!/usr/bin/env python3
"""
DreamZero-SO101 Action Inpainting Inference
재학습 없이 DreamZero의 joint denoising loop에서 action latent를 target actions으로 고정.

입력: eval/images/*.png + eval/actions/*.npy (16×6)
출력: 16-frame 512×320 MP4

Flow:
  1. snapshot_download Vizuara/dreamzero-so101-lora → local path
  2. VLA.load_lora(local_path)  (base Wan2.1 auto-downloaded to HF cache)
  3. Tokenize generic text prompt with UMT5-XXL tokenizer
  4. 2 autoregressive calls to lazy_joint_video_action (generates 17 pixel frames)
  5. VAE decode + take first 16 frames + resize 320×176 → 512×320

Action Inpainting:
  At each denoising step, instead of freely denoising the action latent,
  we replace it with the re-noised target action (flow matching interpolation):
    x_noisy = (1 - sigma) * target_clean + sigma * noise
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ──────────────── DreamZero path ────────────────
DREAMZERO_DIR = Path(__file__).resolve().parent.parent / "dreamzero"
sys.path.insert(0, str(DREAMZERO_DIR))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
# Skip torch.compile inside post_initialize() (saves ~2 GB peak + avoids compile OOM)
os.environ["ENABLE_TENSORRT"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch._dynamo
torch._dynamo.config.disable = True


# ──────────────────────────────────────────────────────────────────────────────
# Data utilities
# ──────────────────────────────────────────────────────────────────────────────

def list_eval_samples(eval_dir: Path):
    image_dir = eval_dir / "images"
    action_dir = eval_dir / "actions"
    ids = sorted([p.stem for p in image_dir.glob("*.png")])
    return [(sid, image_dir / f"{sid}.png", action_dir / f"{sid}.npy") for sid in ids]


def load_first_frame(image_path: Path, target_h=320, target_w=512):
    """Load and resize to target resolution (default 320×512 = challenge output size)."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)  # (H, W, 3)


def compute_dynamic_alpha(actions_raw: np.ndarray, base_alpha: float = 0.7) -> float:
    """
    Action magnitude에 따라 inpainting alpha를 동적으로 결정.
    - 정적 샘플(이동량 ≈ 0): alpha 낮춤 → 모델 자유도 높여 비디오 품질 우선
    - 큰 동작 샘플: alpha 높임 → action 신호를 더 강하게 반영
    기준: eval 평균 이동량 144, std 70
    """
    total_movement = float(np.abs(np.diff(actions_raw, axis=0)).sum())
    # 이동량 0 → alpha=0.30, 이동량 144(평균) → alpha≈base_alpha, 이동량 288+ → alpha=0.85
    alpha = np.clip(0.30 + (base_alpha - 0.30) * (total_movement / 144.0), 0.30, 0.85)
    return float(alpha)


def load_and_normalize_actions(action_path: Path, eval_stats_path: Path,
                                action_dim=32, action_horizon=24,
                                base_alpha: float = 0.7):
    """16×6 actions → normalize → pad to (action_horizon, action_dim).
    Returns (tensor, alpha) where alpha is computed from action magnitude."""
    from scipy.ndimage import gaussian_filter1d

    actions = np.load(action_path).astype(np.float32)
    T_orig, D_orig = actions.shape

    # dynamic alpha: compute before normalization (raw joint values)
    alpha = compute_dynamic_alpha(actions, base_alpha=base_alpha)

    with open(eval_stats_path) as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.clip(np.array(stats["std"], dtype=np.float32), 1e-6, None)

    actions_norm = (actions - mean) / std
    actions_norm = np.clip(actions_norm / 3.0, -1.0, 1.0)

    # gripper smoothing (dim 5): open↔close 급격한 전환 완화
    # sigma=1.0 → 약 2~3 프레임에 걸쳐 부드럽게 전환
    if D_orig > 5:
        actions_norm[:, 5] = gaussian_filter1d(actions_norm[:, 5], sigma=1.0)

    # Pad time: repeat last action
    if T_orig < action_horizon:
        pad = np.repeat(actions_norm[-1:], action_horizon - T_orig, axis=0)
        actions_norm = np.concatenate([actions_norm, pad], axis=0)
    else:
        actions_norm = actions_norm[:action_horizon]

    # Pad action dim: 6 → 32
    pad_dim = action_dim - D_orig
    actions_pad = np.concatenate(
        [actions_norm, np.zeros((action_horizon, pad_dim), dtype=np.float32)], axis=1
    )
    return torch.from_numpy(actions_pad), alpha  # (24, 32), float


# ──────────────────────────────────────────────────────────────────────────────
# Text tokenization
# ──────────────────────────────────────────────────────────────────────────────

def build_text_tensors(positive_text: str, negative_text: str, max_length: int = 512,
                       device: str = "cuda"):
    """
    Tokenize text with UMT5-XXL tokenizer (auto-downloaded from HF).
    Returns: (text_ids, text_mask, neg_ids, neg_mask) all (1, max_length) int64.
    """
    from transformers import AutoTokenizer
    print("[text] Loading UMT5-XXL tokenizer ...")
    tok = AutoTokenizer.from_pretrained("google/umt5-xxl")

    def tokenize(text):
        enc = tok(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device, dtype=torch.long)
        mask = enc["attention_mask"].to(device, dtype=torch.long)
        return ids, mask

    pos_ids, pos_mask = tokenize(positive_text)
    neg_ids, neg_mask = tokenize(negative_text)
    print(f"[text] tokenized: pos_len={pos_mask.sum().item()}, neg_len={neg_mask.sum().item()}")
    return pos_ids, pos_mask, neg_ids, neg_mask


# ──────────────────────────────────────────────────────────────────────────────
# Action inpainting patch
# ──────────────────────────────────────────────────────────────────────────────

def install_action_inpaint_patch(action_head, target_actions_clean: torch.Tensor, alpha: float = 0.7):
    """
    Monkeypatch FlowUniPCMultistepScheduler.step so that when called for
    the action latent (shape B×action_horizon×action_dim), it injects the
    re-noised target instead of freely denoising.

    Also patches generate_noise to capture the initial action noise.

    Returns a cleanup function that restores the originals.
    """
    from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import (
        FlowUniPCMultistepScheduler,
    )

    action_horizon = action_head.action_horizon
    action_dim_model = action_head.action_dim

    noise_ref = [None]   # captures the random noise used for action at inference time
    targets_ref = [target_actions_clean]  # (1, 24, 32) on device, bfloat16

    # ── Patch generate_noise to capture action noise ──
    original_gen_noise = action_head.generate_noise

    def patched_gen_noise(shape, seed=None, device="cpu", dtype=torch.float16):
        result = original_gen_noise(shape, seed=seed, device=device, dtype=dtype)
        if len(shape) == 3 and shape[-2] == action_horizon and shape[-1] == action_dim_model:
            noise_ref[0] = result.clone().to(dtype=torch.bfloat16)
        return result

    action_head.generate_noise = patched_gen_noise

    # ── Patch FlowUniPCMultistepScheduler.step at class level ──
    original_step = FlowUniPCMultistepScheduler.step

    def patched_step(self_sched, model_output, timestep, sample,
                     step_index=None, return_dict=True):
        # Detect action scheduler by sample shape
        is_action = (
            len(sample.shape) == 3
            and sample.shape[-2] == action_horizon
            and sample.shape[-1] == action_dim_model
        )
        if is_action and noise_ref[0] is not None:
            # Compute next sigma (noise level after this step)
            if step_index is not None and step_index + 1 < len(self_sched.sigmas):
                next_sigma = self_sched.sigmas[step_index + 1].item()
            else:
                next_sigma = 0.0
            # Flow matching re-noising: x = (1-sigma)*clean + sigma*noise
            clean = targets_ref[0].to(sample.device, dtype=sample.dtype)
            noise = noise_ref[0].to(sample.device, dtype=sample.dtype)
            x_target = (1.0 - next_sigma) * clean + next_sigma * noise
            # alpha=1.0: full inpainting, alpha=0.0: free denoising
            # alpha=0.7: 70% target-constrained, 30% free → accommodates SO-100/SO-101 domain gap
            free_result = original_step(self_sched, model_output, timestep, sample,
                                        step_index=step_index, return_dict=True)
            x_free = free_result["prev_sample"]
            x_noisy = alpha * x_target + (1.0 - alpha) * x_free
            if return_dict:
                return {"prev_sample": x_noisy}
            return (x_noisy,)
        else:
            return original_step(self_sched, model_output, timestep, sample,
                                 step_index=step_index, return_dict=return_dict)

    FlowUniPCMultistepScheduler.step = patched_step

    def cleanup():
        FlowUniPCMultistepScheduler.step = original_step
        action_head.generate_noise = original_gen_noise

    return cleanup


# ──────────────────────────────────────────────────────────────────────────────
# Main inference
# ──────────────────────────────────────────────────────────────────────────────

def run_inference_one_sample(model, first_frame: np.ndarray, target_actions: torch.Tensor,
                              pos_ids, pos_mask, neg_ids, neg_mask,
                              embodiment_id: int = 26, device: str = "cuda",
                              inpaint_alpha: float = 0.7):
    """
    Run DreamZero with action inpainting.
    Two autoregressive calls → 5 latent frames → 17 pixel frames (take first 16).

    Args:
        first_frame: (H=176, W=320, 3) uint8
        target_actions: (action_horizon=24, action_dim=32) normalized to [-1,1]

    Returns:
        frames: (17, 176, 320, 3) uint8
    """
    action_head = model.action_head

    # ── 1. Prepare target actions on device ──
    actions_clean = target_actions.unsqueeze(0).to(device, dtype=torch.bfloat16)  # (1,24,32)

    # ── 2. Prepare image tensor (B=1, T=1, H, W, C) uint8 ──
    img_t = torch.from_numpy(first_frame).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,176,320,3)

    from transformers.feature_extraction_utils import BatchFeature

    def make_action_input(latent_video_hint=None):
        data = {
            "images": img_t,                                          # (1,1,H,W,3) uint8
            "text": pos_ids,                                          # (1, max_len)
            "text_attention_mask": pos_mask,
            "text_negative": neg_ids,
            "text_attention_mask_negative": neg_mask,
            "action": actions_clean,                                  # (1,24,32) - target (will be inpainted)
            "state": torch.zeros(1, 1, 64, device=device, dtype=torch.bfloat16),  # max_state_dim=64
            "embodiment_id": torch.tensor([embodiment_id], device=device, dtype=torch.long),
            "has_real_action": torch.ones(1, dtype=torch.bool, device=device),
            "action_mask": torch.ones(1, action_head.action_horizon, action_head.action_dim,
                                       device=device, dtype=torch.bfloat16),
        }
        return BatchFeature(data=data)

    backbone_output = BatchFeature(data={})

    # ── 3. Install action inpainting patch ──
    cleanup = install_action_inpaint_patch(action_head, actions_clean, alpha=inpaint_alpha)

    try:
        all_latents = []

        # First call: current_start_frame=0 → returns (1, 16, 3, H_lat, W_lat)
        # (first frame latent + 2 generated)
        action_input = make_action_input()
        with torch.no_grad():
            result1 = action_head.lazy_joint_video_action(backbone_output, action_input)
        video_pred_1 = result1["video_pred"]  # (1, 16, 3, 22, 40)
        all_latents.append(video_pred_1)
        print(f"    [chunk1] video_pred shape: {video_pred_1.shape}, "
              f"current_start_frame={action_head.current_start_frame}")

        # Second call: returns (1, 16, 2, H_lat, W_lat) new frames
        action_input = make_action_input()
        with torch.no_grad():
            result2 = action_head.lazy_joint_video_action(
                backbone_output, action_input, latent_video=video_pred_1
            )
        video_pred_2 = result2["video_pred"]  # (1, 16, 2, 22, 40)
        all_latents.append(video_pred_2)
        print(f"    [chunk2] video_pred shape: {video_pred_2.shape}, "
              f"current_start_frame={action_head.current_start_frame}")

    finally:
        cleanup()

    # ── 4. Accumulate latents and VAE decode ──
    # all_latents: [(1,16,3,22,40), (1,16,2,22,40)]  → cat dim=2 → (1,16,5,22,40)
    combined = torch.cat(all_latents, dim=2)
    print(f"    [decode] combined latent shape: {combined.shape}")

    with torch.no_grad():
        # tiled=False: latent (22×40 at 176×320, 40×64 at 320×512) is small enough
        # to decode without spatial tiling, eliminating seam artifacts at tile boundaries
        frames = action_head.vae.decode(
            combined,
            tiled=False,
        )  # (1, 3, T_pix, H, W)

    # frames: (1, 3, T_pix, 176, 320)  in [-1, 1]
    frames = frames[0]                             # (3, T_pix, 176, 320)
    frames = frames.permute(1, 2, 3, 0)           # (T_pix, 176, 320, 3)
    frames = ((frames.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    frames = frames.cpu().numpy()                  # (T_pix, 176, 320, 3) uint8

    print(f"    [decode] pixel frames: {frames.shape}")
    return frames


def reset_model_state(model):
    ah = model.action_head
    ah.current_start_frame = 0
    ah.language = None
    ah.kv_cache1 = None
    ah.kv_cache_neg = None
    ah.crossattn_cache = None
    ah.crossattn_cache_neg = None
    ah.clip_feas = None
    ah.ys = None


def save_video(frames: np.ndarray, out_path: Path, fps: int = 6,
               out_h: int = 320, out_w: int = 512, n_frames: int = 16):
    """Take first n_frames, write H.264 MP4. Resizes only if frame size != (out_h, out_w)."""
    import subprocess
    frames = frames[:n_frames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_path) + ".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (out_w, out_h))
    for frame in frames:
        h, w = frame.shape[:2]
        if (h, w) != (out_h, out_w):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out_path)],
        check=True, capture_output=True,
    )
    tmp.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir",   default="/home1/sota/inha2026/data/eval")
    parser.add_argument("--out-dir",    default="/home1/sota/inha2026/submission_kit/input_videos_dreamzero")
    parser.add_argument("--lora-path",  default="Vizuara/dreamzero-so101-lora",
                        help="HF repo ID or local path for DreamZero LoRA weights")
    parser.add_argument("--eval-stats", default="/home1/sota/inha2026/data/eval/eval_action_statistics.json")
    parser.add_argument("--embodiment-id", type=int, default=26,
                        help="Embodiment ID used during SO-101 training (26, 17, or 32)")
    parser.add_argument("--positive-text", default="robot arm manipulation pick up object",
                        help="Text prompt for video generation (positive); overridden per-sample if --captions-path is set")
    parser.add_argument("--captions-path", default="/home1/sota/inha2026/data/eval/eval_captions.json",
                        help="Per-sample captions JSON {sample_id: caption}; empty entries fall back to --positive-text")
    parser.add_argument("--negative-text", default="poor quality, blurry, low resolution",
                        help="Text prompt for CFG negative")
    parser.add_argument("--num-samples",    type=int,   default=None)
    parser.add_argument("--overwrite",      action="store_true")
    parser.add_argument("--gpu",            type=int,   default=0)
    parser.add_argument("--local-lora-path", default=None,
                        help="If set, skip snapshot_download and use this local directory directly")
    parser.add_argument("--ae-lora-path", default=None,
                        help="Path to AE fine-tuned LoRA checkpoint (.pt) to load on top of base LoRA")
    parser.add_argument("--num-steps",      type=int,   default=50,
                        help="Denoising steps (default 50; model default is 16)")
    parser.add_argument("--inpaint-alpha",  type=float, default=0.7,
                        help="Action inpainting strength 0.0=free 1.0=full (default 0.7)")
    parser.add_argument("--guidance-scale", type=float, default=5.0,
                        help="CFG guidance scale (default 5.0)")
    parser.add_argument("--gen-height",     type=int,   default=320,
                        help="Generation height (default 320 = challenge target)")
    parser.add_argument("--gen-width",      type=int,   default=512,
                        help="Generation width (default 512 = challenge target)")
    args = parser.parse_args()

    # 2-GPU split: cuda:0 = DiT+CLIP+VAE, cuda:1 = T5
    # Do NOT limit to 1 GPU so both are visible.
    device = "cuda:0"

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Download DreamZero LoRA weights locally ──
    if args.local_lora_path:
        local_lora_dir = args.local_lora_path
        print(f"[model] Using local LoRA path: {local_lora_dir}")
    else:
        from huggingface_hub import snapshot_download
        print(f"[model] Downloading {args.lora_path} to local cache ...")
        local_lora_dir = snapshot_download(repo_id=args.lora_path)
        print(f"[model] LoRA weights at: {local_lora_dir}")

    # ── Load model (base Wan2.1 auto-downloaded from HF cache) ──
    print("[model] Loading VLA.load_lora() ...")
    print("        NOTE: Wan2.1-I2V-14B-480P (~82 GB) will be auto-downloaded if not cached.")
    from groot.vla.model.dreamzero.base_vla import VLA
    model = VLA.load_lora(local_lora_dir)
    model.eval()

    # ── 2-GPU 분산 로딩 ──
    # cuda:0: DiT (~33 GB) + CLIP (~4.5 GB) + VAE (~2 GB) = ~39.5 GB
    # cuda:1: T5 UMT5-XXL (~10 GB)
    # 경계에서만 텐서 이동: T5 input→cuda:1, output→cuda:0

    print("[model] post_initialize() — DiT+T5 → cuda:0 (torch.compile skipped) ...")
    model.post_initialize()
    torch.cuda.empty_cache()
    g0 = torch.cuda.memory_allocated(0) / 1024**3
    print(f"[model] cuda:0 after post_initialize: {g0:.1f} GB")

    # ── T5 → cuda:1 (cuda:0 에서 ~10 GB 확보) ──
    print("[model] Moving T5 → cuda:1 ...")
    model.action_head.text_encoder.to("cuda:1")
    torch.cuda.empty_cache()
    g0 = torch.cuda.memory_allocated(0) / 1024**3
    g1 = torch.cuda.memory_allocated(1) / 1024**3
    print(f"[model] cuda:0: {g0:.1f} GB | cuda:1: {g1:.1f} GB")

    # ── CLIP visual → cuda:0 (parameter-level move; open_clip .to() 전파 안됨) ──
    print("[model] Moving CLIP visual params → cuda:0 bf16 ...")
    _clip_vis = model.action_head.image_encoder.model.visual
    for p in _clip_vis.parameters():
        p.data = p.data.to("cuda:0", dtype=torch.bfloat16)
    for b in _clip_vis.buffers():
        b.data = b.data.to("cuda:0", dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    g0 = torch.cuda.memory_allocated(0) / 1024**3
    print(f"[model] cuda:0 after CLIP→cuda:0: {g0:.1f} GB")

    # ── VAE → cuda:0 ──
    print("[model] Moving VAE → cuda:0 ...")
    model.action_head.vae.to("cuda:0")
    model.action_head._vae_device_ready = True  # prevent _ensure_vae_on_device re-move
    torch.cuda.empty_cache()
    g0 = torch.cuda.memory_allocated(0) / 1024**3
    print(f"[model] cuda:0 after VAE→cuda:0: {g0:.1f} GB")

    # ── Load AE fine-tuned LoRA (if provided) ──
    if args.ae_lora_path:
        print(f"[model] Loading AE fine-tuned LoRA from {args.ae_lora_path} ...")
        ae_lora_state = torch.load(args.ae_lora_path, map_location="cpu")
        missing, unexpected = model.action_head.load_state_dict(ae_lora_state, strict=False)
        print(f"[model] AE LoRA loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")

    # ── Hyperparameter overrides ──
    model.action_head.num_inference_steps = args.num_steps
    model.action_head.cfg_scale = args.guidance_scale
    model.action_head.config.target_video_height = args.gen_height
    model.action_head.config.target_video_width  = args.gen_width
    print(f"[config] num_steps={args.num_steps}, guidance_scale={args.guidance_scale}, "
          f"gen={args.gen_height}×{args.gen_width}, inpaint_alpha={args.inpaint_alpha}")

    # ── T5 boundary patch: input ids(cuda:0) → cuda:1, output → cuda:0 ──
    _orig_t5_fwd = model.action_head.text_encoder.forward
    def _t5_fwd_bridge(input_ids, attention_mask=None, **kwargs):
        out = _orig_t5_fwd(
            input_ids.to("cuda:1"),
            attention_mask.to("cuda:1") if attention_mask is not None else None,
            **kwargs,
        )
        return out.to("cuda:0", dtype=torch.bfloat16) if isinstance(out, torch.Tensor) else out
    model.action_head.text_encoder.forward = _t5_fwd_bridge
    print("[patch] T5: input→cuda:1, output→cuda:0")

    # ── encode_image patch: input image(CPU/cuda:0) → CLIP device(cuda:0), output → cuda:0 ──
    import types as _types
    _img_enc = model.action_head.image_encoder

    def _patched_encode_image(self, videos):
        import torch.nn.functional as _F
        size = (self.model.image_size,) * 2
        videos = torch.cat([
            _F.interpolate(u, size=size, mode='bicubic', align_corners=False)
            for u in videos
        ])
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))
        vis_p = next(iter(self.model.visual.parameters()))
        videos = videos.to(device=vis_p.device, dtype=vis_p.dtype)
        out = self.model.visual(videos, use_31_block=True).clone()
        return out.to("cuda:0", dtype=torch.bfloat16)

    _img_enc.encode_image = _types.MethodType(_patched_encode_image, _img_enc)
    print("[patch] encode_image: videos→cuda:0(CLIP), output→cuda:0")
    print("[model] Model ready.")

    # ── Load per-sample captions (optional) ──
    sample_captions = {}
    if args.captions_path:
        import json as _json
        try:
            sample_captions = _json.loads(Path(args.captions_path).read_text())
            filled = sum(1 for v in sample_captions.values() if v and v.strip())
            print(f"[captions] Loaded {filled}/{len(sample_captions)} non-empty captions from {args.captions_path}")
        except Exception as e:
            print(f"[captions] Failed to load {args.captions_path}: {e} — using default positive-text")

    # ── Tokenize default text prompts (used as fallback) ──
    pos_ids, pos_mask, neg_ids, neg_mask = build_text_tensors(
        args.positive_text, args.negative_text, device=device
    )

    # ── Get eval samples ──
    samples = list_eval_samples(eval_dir)
    if args.num_samples is not None:
        samples = samples[:args.num_samples]
    print(f"[infer] Processing {len(samples)} samples → {out_dir}")

    # ── Run inference ──
    for sid, img_path, act_path in samples:
        out_path = out_dir / f"{sid}.mp4"
        if out_path.exists() and not args.overwrite:
            print(f"  [skip] {sid}")
            continue

        print(f"  [{sid}] Loading data ...")
        first_frame = load_first_frame(img_path, target_h=args.gen_height, target_w=args.gen_width)
        target_actions, sample_alpha = load_and_normalize_actions(
            act_path, Path(args.eval_stats), action_dim=32, action_horizon=24,
            base_alpha=args.inpaint_alpha,
        )

        reset_model_state(model)

        # Per-sample caption override
        caption = sample_captions.get(sid, "").strip()
        if caption:
            cur_pos_ids, cur_pos_mask, cur_neg_ids, cur_neg_mask = build_text_tensors(
                caption, args.negative_text, device=device
            )
        else:
            cur_pos_ids, cur_pos_mask, cur_neg_ids, cur_neg_mask = pos_ids, pos_mask, neg_ids, neg_mask

        print(f"  [{sid}] Running action inpainting (alpha={sample_alpha:.2f}, caption={'custom' if caption else 'default'}) ...")
        try:
            frames = run_inference_one_sample(
                model, first_frame, target_actions,
                cur_pos_ids, cur_pos_mask, cur_neg_ids, cur_neg_mask,
                embodiment_id=args.embodiment_id, device=device,
                inpaint_alpha=sample_alpha,
            )
            save_video(frames, out_path, fps=6, out_h=args.gen_height, out_w=args.gen_width, n_frames=16)
            print(f"  [{sid}] Saved → {out_path}")
        except Exception as e:
            print(f"  [{sid}] ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\n[done] {out_dir}")
    print("[next] cd submission_kit && python make_submission_csv.py --input-dir input_videos_dreamzero")


if __name__ == "__main__":
    main()
