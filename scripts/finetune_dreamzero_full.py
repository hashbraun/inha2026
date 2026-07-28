"""
DreamZero LoRA fine-tuning with action_extractor + R3D-18 + DINOv2 perceptual losses.

Loss formula:
  L_total = L_ae + λ_dit * L_dit + λ_r3d * L_r3d + λ_dino * L_dino

  L_ae   = L1(ae_model(pred_video), target_actions)         # action accuracy
  L_dit  = MSE(v_pred, v_target)                           # prevent forgetting
  L_r3d  = 1 - cosine_sim(R3D(pred), R3D(real))           # video motion similarity
  L_dino = 1 - cosine_sim(DINO(pred_frames), DINO(real_frames))  # visual fidelity

Note: TORCHDYNAMO_DISABLE=1 must be set before running (see sbatch).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Must be set before any torch import to prevent VAE torch.compile + einops crash
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent / "dreamzero"))
sys.path.insert(0, str(Path(__file__).parent.parent / "submission_kit"))

# ──────────────────────────────────────────────────────────────────────────────
# Normalization constants (must match feature_csv_utils.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

_KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
_KINETICS_STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def read_video_frames(mp4_path: Path, start: int, length: int) -> np.ndarray:
    """Return (T, H, W, 3) uint8 array."""
    frames = []
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i < start:
                continue
            if i >= start + length:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)  # (T, H, W, 3)


class SO100ClipDataset(Dataset):
    CLIP_LEN_LONG  = 49   # ≥49 frames → DiT joint path (blockwise causal attn)
    CLIP_LEN_SHORT = 16   # 16~48 frames → standalone AE path
    EVAL_MEAN = np.array([4.38, 85.91, 92.29, 63.55, 64.41, 9.14], dtype=np.float32)
    EVAL_STD  = np.array([29.41, 74.75, 63.78, 15.66, 40.38, 9.46], dtype=np.float32)

    def __init__(self, data_root: str, max_clips: int = 5000, filter_eval_dist: bool = False):
        self.clips = []
        self._build_index(Path(data_root), max_clips, filter_eval_dist)
        long_n  = sum(1 for c in self.clips if c[5])
        short_n = len(self.clips) - long_n
        print(f"[dataset] {len(self.clips)} clips indexed  "
              f"(long/DiT={long_n}, short/AE={short_n})")

    def _build_index(self, root: Path, max_clips: int, filter_eval_dist: bool):
        import pandas as pd
        for info_path in sorted(root.glob("*/*/meta/info.json")):
            task_dir = info_path.parent.parent
            try:
                info = json.loads(info_path.read_text())
            except Exception:
                continue
            for chunk_dir in sorted((task_dir / "data").glob("chunk-*")):
                for parquet_path in sorted(chunk_dir.glob("episode_*.parquet")):
                    ep_idx = int(parquet_path.stem.split("_")[1])
                    chunk_idx = int(chunk_dir.name.split("-")[1])
                    mp4_rel = info["video_path"].format(
                        episode_chunk=chunk_idx,
                        episode_index=ep_idx,
                        video_key="observation.images.image",
                    )
                    mp4_path = task_dir / mp4_rel
                    if not mp4_path.exists():
                        continue
                    try:
                        df = pd.read_parquet(parquet_path, columns=["frame_index", "action", "observation.state"])
                    except Exception:
                        continue
                    actions = np.stack(df["action"].values).astype(np.float32)
                    states  = np.stack(df["observation.state"].values).astype(np.float32)
                    n_frames = len(actions)
                    # assign bucket
                    if n_frames >= self.CLIP_LEN_LONG:
                        clip_len, use_dit = self.CLIP_LEN_LONG, True
                    elif n_frames >= self.CLIP_LEN_SHORT:
                        clip_len, use_dit = self.CLIP_LEN_SHORT, False
                    else:
                        continue
                    stride = clip_len // 2
                    for start in range(0, n_frames - clip_len, stride):
                        clip_actions = actions[start:start + clip_len]
                        clip_states  = states[start:start + clip_len]
                        if filter_eval_dist:
                            wr = clip_actions[:, 4]
                            sl = clip_actions[:, 1]
                            if not (30 <= wr.mean() <= 100 and 50 <= sl.mean() <= 120):
                                continue
                        self.clips.append((mp4_path, clip_actions, clip_states, start, clip_len, use_dit))
                        if len(self.clips) >= max_clips:
                            return

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        mp4_path, actions, states, start, clip_len, use_dit = self.clips[idx]
        frames = read_video_frames(mp4_path, start, clip_len)        # (T, H, W, 3)
        actions_norm = (actions - self.EVAL_MEAN) / self.EVAL_STD
        from scipy.ndimage import gaussian_filter1d
        actions_norm[:, 5] = gaussian_filter1d(actions_norm[:, 5], sigma=1.0)
        actions_norm = np.clip(actions_norm, -1.0, 1.0).astype(np.float32)
        states_norm = (states - self.EVAL_MEAN) / self.EVAL_STD
        states_norm = np.clip(states_norm, -1.0, 1.0).astype(np.float32)
        return {
            "frames":      torch.from_numpy(frames),             # (T, H, W, 3) uint8
            "actions_norm": torch.from_numpy(actions_norm),      # (T, 6)
            "states_norm":  torch.from_numpy(states_norm),       # (T, 6)
            "use_dit":      torch.tensor(use_dit, dtype=torch.bool),
        }


# ──────────────────────────────────────────────────────────────────────────────
# x0 recovery from flow-matching velocity prediction
# ──────────────────────────────────────────────────────────────────────────────

def recover_x0(noisy: torch.Tensor, v_pred: torch.Tensor,
               timestep_id: torch.Tensor, scheduler) -> torch.Tensor:
    """Flow matching: x0 = x_t - sigma * v"""
    sigma = scheduler.sigmas[timestep_id.cpu()].to(noisy.device, dtype=noisy.dtype)
    while sigma.dim() < noisy.dim():
        sigma = sigma.unsqueeze(-1)
    return noisy - sigma * v_pred


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────

def frames_to_action_extractor_input(pixel_video: torch.Tensor,
                                      target_h: int = 320,
                                      target_w: int = 512) -> torch.Tensor:
    """pixel_video: (B, T, C, H, W) in [-1, 1] → (B, C, T, target_h, target_w)"""
    B, T, C, H, W = pixel_video.shape
    flat = pixel_video.reshape(B * T, C, H, W)
    flat = F.interpolate(flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return flat.reshape(B, T, C, target_h, target_w).permute(0, 2, 1, 3, 4)


def _resize_pad_square(frames: torch.Tensor, size: int) -> torch.Tensor:
    """Resize-pad to (N, C, size, size). Input: (N, C, H, W) float [0, 1]."""
    _, _, h, w = frames.shape
    scale = min(size / h, size / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    x = F.interpolate(frames, size=(rh, rw), mode="bilinear", align_corners=False)
    pt = (size - rh) // 2
    pl = (size - rw) // 2
    return F.pad(x, (pl, size - rw - pl, pt, size - rh - pt), value=0.0)


def _norm_model_out(output) -> torch.Tensor:
    """Normalize various timm/torchvision model output formats to (N, dim)."""
    if isinstance(output, dict):
        if "x_norm_clstoken" in output:
            output = output["x_norm_clstoken"]
        elif "features" in output:
            output = output["features"]
        else:
            vals = [v for v in output.values() if isinstance(v, torch.Tensor)]
            output = vals[0]
    elif isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 3:
        output = output[:, 0]   # CLS token
    elif output.ndim > 3:
        output = output.flatten(2).mean(-1)
    return output


def prep_for_r3d(pixel_video: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    pixel_video: (B, T, C, H, W) in [-1, 1] → (B, C, T, 112, 112) Kinetics-normalized.
    Matches feature_csv_utils.extract_video_features exactly.
    """
    x = (pixel_video.float() + 1.0) / 2.0          # [0, 1]
    x = x.permute(0, 2, 1, 3, 4)                   # (B, C, T, H, W)
    T_f = x.shape[2]
    x = F.interpolate(x, size=(T_f, 112, 112), mode="trilinear", align_corners=False)
    km = _KINETICS_MEAN.to(device=device, dtype=x.dtype)
    ks = _KINETICS_STD.to(device=device, dtype=x.dtype)
    return (x - km) / ks


def prep_real_for_r3d(frames_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """frames_u8: (T, H, W, 3) uint8 → (1, C, T, 112, 112) Kinetics-normalized."""
    x = frames_u8.float() / 255.0                  # [0, 1]
    x = x.permute(3, 0, 1, 2).unsqueeze(0)         # (1, C, T, H, W)
    T_f = x.shape[2]
    x = F.interpolate(x.to(device), size=(T_f, 112, 112), mode="trilinear", align_corners=False)
    km = _KINETICS_MEAN.to(device=device, dtype=x.dtype)
    ks = _KINETICS_STD.to(device=device, dtype=x.dtype)
    return (x - km) / ks


def prep_for_dino(pixel_video: torch.Tensor, dino_size: int,
                  device: torch.device) -> torch.Tensor:
    """
    pixel_video: (B, N_frames, C, H, W) in [-1, 1] → (B*N, C, dino_size, dino_size) ImageNet-norm.
    Matches feature_csv_utils.extract_dino_features exactly.
    """
    B, N, C, H, W = pixel_video.shape
    x = (pixel_video.float() + 1.0) / 2.0
    x = x.reshape(B * N, C, H, W)
    x = _resize_pad_square(x, dino_size).to(device)
    im = _IMAGENET_MEAN.to(device=device, dtype=x.dtype)
    is_ = _IMAGENET_STD.to(device=device, dtype=x.dtype)
    return (x - im) / is_


def prep_real_for_dino(frames_u8: torch.Tensor, dino_size: int,
                       device: torch.device) -> torch.Tensor:
    """frames_u8: (N, H, W, 3) uint8 → (N, C, dino_size, dino_size) ImageNet-norm."""
    x = frames_u8.float() / 255.0
    x = x.permute(0, 3, 1, 2)                      # (N, C, H, W)
    x = _resize_pad_square(x, dino_size).to(device)
    im = _IMAGENET_MEAN.to(device=device, dtype=x.dtype)
    is_ = _IMAGENET_STD.to(device=device, dtype=x.dtype)
    return (x - im) / is_


def resolve_dino_size(model: torch.nn.Module) -> int:
    """Return the model's expected square input size."""
    pe = getattr(model, "patch_embed", None)
    img_size = getattr(pe, "img_size", None)
    if isinstance(img_size, (tuple, list)) and img_size:
        return int(img_size[0])
    if isinstance(img_size, int):
        return int(img_size)
    return 518  # ViT-S/14 DINOv2 default


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",        default="/home1/sota/inha2026/data/train")
    parser.add_argument("--lora-path",        default="Vizuara/dreamzero-so101-lora")
    parser.add_argument("--local-lora-path",  default=None)
    parser.add_argument("--ae-ckpt",          default="/home1/sota/inha2026/submission_kit/checkpoints/action_extractor.ckpt")
    parser.add_argument("--out-dir",          default="/home1/sota/inha2026/checkpoints/dreamzero_full_lora")
    parser.add_argument("--num-steps",        type=int,   default=500)
    parser.add_argument("--lr",               type=float, default=5e-6)
    parser.add_argument("--lambda-dit",       type=float, default=0.1)
    parser.add_argument("--lambda-action",    type=float, default=1.0,
                        help="Weight for action denoising loss (trains action_encoder)")
    parser.add_argument("--lambda-state",     type=float, default=1.0,
                        help="Weight for state reconstruction loss (trains state_encoder)")
    parser.add_argument("--lambda-r3d",       type=float, default=1.0,
                        help="Weight for R3D-18 cosine loss")
    parser.add_argument("--lambda-dino",      type=float, default=1.0,
                        help="Weight for DINOv2 cosine loss")
    parser.add_argument("--dino-frames",      type=int,   default=4,
                        help="Number of frames to use for DINO loss (saves memory)")
    parser.add_argument("--sigma-low",        type=float, default=0.3,
                        help="Lower bound of sigma range for timestep sampling")
    parser.add_argument("--sigma-high",       type=float, default=0.6,
                        help="Upper bound of sigma range for timestep sampling")
    parser.add_argument("--flash-noise",      action="store_true",
                        help="DreamZero-Flash decoupled noise: video~Beta(7,1), action~U(0,1) independently")
    parser.add_argument("--grad-accum",       type=int,   default=4)
    parser.add_argument("--max-clips",        type=int,   default=5000)
    parser.add_argument("--save-every",       type=int,   default=500)
    parser.add_argument("--embodiment-id",    type=int,   default=26)
    parser.add_argument("--positive-text",    default="robot arm manipulation pick up object")
    args = parser.parse_args()

    device = "cuda:0"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load DreamZero ───────────────────────────────────────────────────────
    print("[model] Loading DreamZero...")
    from groot.vla.model.dreamzero.base_vla import VLA

    if args.local_lora_path:
        local_lora_dir = args.local_lora_path
    else:
        from huggingface_hub import snapshot_download
        local_lora_dir = snapshot_download(repo_id=args.lora_path)

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    model = VLA.load_lora(local_lora_dir)
    model.eval()
    model.post_initialize()

    action_head = model.action_head

    # ── Load SO-101 pretrained action_encoder/decoder ─────────────────────────
    # VLA.load_lora() loads LoRA keys but misses action_encoder/decoder because
    # the safetensors was saved from a PEFT-wrapped model and keys carry the
    # prefix "action_head.model.base_model.model." — they don't match the bare
    # model's key names, so they silently land in "unexpected_keys" and are
    # discarded. We strip the prefix and load them manually.
    from safetensors import safe_open as _safe_open
    _lora_st = Path(local_lora_dir) / "model.safetensors"
    # Strip "action_head.model." prefix so keys match the PEFT-wrapped model's
    # state_dict format, e.g. "base_model.model.action_encoder.W1.W"
    _PREFIX_FULL = "action_head.model.base_model.model."
    _PREFIX_PEFT = "action_head.model."
    if _lora_st.exists():
        with _safe_open(str(_lora_st), framework="pt") as _f:
            _enc_dec = {
                k[len(_PREFIX_PEFT):]: _f.get_tensor(k)
                for k in _f.keys()
                if k.startswith(_PREFIX_FULL) and ("action_encoder" in k or "action_decoder" in k)
            }
        if _enc_dec:
            _missing, _unexpected = action_head.model.load_state_dict(_enc_dec, strict=False)
            _enc_dec_keys = list(_enc_dec.keys())
            _loaded = [k for k in _enc_dec_keys if k not in _unexpected]
            print(f"[model] SO-101 action_encoder/decoder loaded: {len(_loaded)}/{len(_enc_dec_keys)} tensors")
            if _unexpected:
                print(f"[model]   unexpected (not loaded): {_unexpected}")
            else:
                print(f"[model]   all tensors matched: {_enc_dec_keys}")
        else:
            print("[model] WARNING: no action_encoder/decoder tensors found in safetensors")
    else:
        print(f"[model] WARNING: {_lora_st} not found, skipping action_encoder/decoder load")

    # GPU split: cuda:0 = DiT+CLIP+VAE, cuda:1 = T5
    action_head.text_encoder.to("cuda:1")
    _clip_vis = action_head.image_encoder.model.visual
    for p in _clip_vis.parameters():
        p.data = p.data.to("cuda:0", dtype=torch.bfloat16)
    for b in _clip_vis.buffers():
        b.data = b.data.to("cuda:0", dtype=torch.bfloat16)
    action_head.vae.to("cuda:0")
    action_head._vae_device_ready = True
    torch.cuda.empty_cache()

    _orig_t5_forward = action_head.text_encoder.forward
    def _patched_t5_forward(*a, **kw):
        out = _orig_t5_forward(*a, **kw)
        return out.to("cuda:0", dtype=torch.bfloat16) if isinstance(out, torch.Tensor) else out
    action_head.text_encoder.forward = _patched_t5_forward

    _orig_enc_img = action_head.encode_image
    def _patched_encode_image(image, *a, **kw):
        out = _orig_enc_img(image, *a, **kw)
        return tuple(x.to("cuda:0", dtype=torch.bfloat16) if isinstance(x, torch.Tensor) else x for x in out)
    action_head.encode_image = _patched_encode_image

    # ── Enable LoRA + action_encoder + action_decoder gradients ─────────────
    # action_decoder is not used at inference, but unfreezing it gives
    # action_encoder a proper reconstruction gradient (vs. frozen-random-decoder
    # which only passes noise). Both encoder and decoder train together via
    # L_action so that action_encoder produces structured features for L_dit.
    for name, param in action_head.named_parameters():
        is_lora = "lora" in name.lower()
        is_action_enc = "action_encoder" in name
        is_action_dec = "action_decoder" in name
        param.requires_grad_(is_lora or is_action_enc or is_action_dec)

    trainable = sum(p.numel() for p in action_head.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in action_head.parameters())
    print(f"[model] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("[data] Building dataset...")
    dataset = SO100ClipDataset(args.data_root, max_clips=args.max_clips)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=False)
    loader_iter = iter(loader)

    # ── Tokenize text (constant across all steps) ────────────────────────────
    from transformers import AutoTokenizer
    print("[text] Loading UMT5-XXL tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    tok_out = tokenizer(
        args.positive_text, return_tensors="pt", padding="max_length",
        truncation=True, max_length=512,
    )
    pos_ids  = tok_out["input_ids"].to("cuda:1")
    pos_mask = tok_out["attention_mask"].to("cuda:1")

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in action_head.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    action_head.scheduler.set_timesteps(1000, training=True)
    num_train_ts = action_head.scheduler.num_train_timesteps

    # ── Compute t_id range for σ ∈ [sigma_low, sigma_high] ──────────────────
    # scheduler.sigmas[t_id] gives the sigma value for training index t_id.
    # We sample only from the range where x0_pred is most meaningful.
    all_sigmas = action_head.scheduler.sigmas.cpu().float()  # (num_train_ts,) or (+1)
    valid = ((all_sigmas >= args.sigma_low) & (all_sigmas <= args.sigma_high)).nonzero(as_tuple=True)[0]
    if len(valid) == 0:
        print(f"[warn] No σ in [{args.sigma_low},{args.sigma_high}], using full range")
        t_id_min, t_id_max = 0, num_train_ts
    else:
        t_id_min, t_id_max = int(valid.min()), int(valid.max()) + 1
    print(f"[train] σ range [{args.sigma_low},{args.sigma_high}] → t_id [{t_id_min},{t_id_max}] "
          f"({t_id_max - t_id_min}/{num_train_ts} steps)")

    # ── Flash decoupled noise 준비 ───────────────────────────────────────────
    if args.flash_noise:
        _beta_dist = torch.distributions.Beta(
            torch.tensor(7.0), torch.tensor(1.0)
        )
        print("[train] Flash noise enabled: video~Beta(7,1), action~U(0,1) independent")

    # ── Training loop ────────────────────────────────────────────────────────
    print("[train] Starting...")
    log_path = out_dir / "train_log.jsonl"
    optimizer.zero_grad()

    # Frame indices for DINO (uniformly sampled)
    dino_n = min(args.dino_frames, SO100ClipDataset.CLIP_LEN_LONG)

    for step in range(1, args.num_steps + 1):
        t0 = time.time()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        frames       = batch["frames"][0]       # (T, H, W, 3) uint8
        actions_norm = batch["actions_norm"][0]  # (T, 6) float32
        states_norm  = batch["states_norm"][0]   # (T, 6) float32
        use_dit      = batch["use_dit"][0].item()
        T, H, W, C = frames.shape

        # ── Preprocess video for DiT ─────────────────────────────────────
        target_h, target_w = 176, 320
        vid = frames.float().permute(0, 3, 1, 2) / 255.0   # (T, 3, H, W)
        vid = F.interpolate(vid, size=(target_h, target_w), mode="bilinear", align_corners=False)
        vid = (vid - 0.5) * 2.0                             # [-1, 1]
        vid = vid.to("cuda:0", dtype=torch.bfloat16)
        vid_bcthw = vid.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, 3, T, Ht, Wt)

        # ── VAE encode (no grad) ─────────────────────────────────────────
        with torch.no_grad():
            z_clean = action_head.encode_video(
                vid_bcthw, action_head.tiled,
                (action_head.tile_size_height, action_head.tile_size_width),
                (action_head.tile_stride_height, action_head.tile_stride_width),
            )

        # ── CLIP + T5 encode (no grad) ───────────────────────────────────
        image = vid_bcthw[:, :, :1].permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            clip_feas, ys, image_enc = action_head.encode_image(image, T, target_h, target_w)
            clip_feas = clip_feas.to("cuda:0", dtype=torch.bfloat16)
            ys        = ys.to("cuda:0", dtype=torch.bfloat16)
            prompt_embs = action_head.encode_prompt(pos_ids, pos_mask)

        # ── Add noise ────────────────────────────────────────────────────
        z = z_clean.transpose(1, 2)                         # (1, T_lat, C_lat, H_lat, W_lat)
        noise = torch.randn_like(z)
        B, T_lat, C_lat, H_lat, W_lat = z.shape
        tokens_per_frame = (H_lat // 2) * (W_lat // 2)
        seq_len = T_lat * tokens_per_frame
        if step == 1:
            print(f"[debug] T={T} frames → z shape={list(z.shape)}, T_lat={T_lat}, frame_seqlen={tokens_per_frame}")

        if args.flash_noise:
            # ── Flash decoupled noise ────────────────────────────────────
            # Video: Beta(7,1) → E[σ]≈0.875, 항상 고노이즈 바이어스
            sigma_v_raw = _beta_dist.sample((B * T_lat,)).float()  # [0,1]
            # 각 sigma_v_raw에 대해 가장 가까운 t_id를 찾음
            # all_sigmas: (N,), sigma_v_raw: (B*T_lat,) → diff: (B*T_lat, N)
            t_id_flat = torch.argmin(
                (all_sigmas.unsqueeze(0) - sigma_v_raw.unsqueeze(1)).abs(), dim=1
            ).clamp(0, len(all_sigmas) - 2)            # (B*T_lat,)
            t_id = t_id_flat.reshape(B, T_lat)          # (B, T_lat)

            # Action: U(0,1) 독립 샘플
            sigma_a_raw = torch.rand(B).float()          # [0,1]
            t_scalar_a = torch.argmin(
                (all_sigmas.unsqueeze(0) - sigma_a_raw.unsqueeze(1)).abs(), dim=1
            ).clamp(0, len(all_sigmas) - 2)             # (B,)
        else:
            # ── Standard coupled noise (기존) ────────────────────────────
            t_id = torch.randint(t_id_min, t_id_max, (B, T_lat))  # (B, T_lat)
            t_scalar_a = t_id[:, 0]                                 # (B,)

        timestep = action_head.scheduler.timesteps[t_id.cpu()].to("cuda:0")

        noisy_z = action_head.scheduler.add_noise(
            z.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (B, T_lat))

        v_target = action_head.scheduler.training_target(
            z, noise, timestep
        ).transpose(1, 2)

        # ── Build clean action tensor (24×32) for action_encoder training ─
        action_dim = action_head.model.action_dim    # 32
        action_horizon = action_head.action_horizon  # 24
        T_act = min(T, action_horizon)
        clean_action_np = np.zeros((action_horizon, action_dim), dtype=np.float32)
        clean_action_np[:T_act, :6] = actions_norm[:T_act].numpy()
        clean_action = torch.from_numpy(clean_action_np).unsqueeze(0).to("cuda:0", dtype=torch.bfloat16)  # (1, 24, 32)

        # ── Noisy action ─────────────────────────────────────────────────
        timestep_scalar_a = action_head.scheduler.timesteps[t_scalar_a.cpu()].to("cuda:0")  # (B,)
        sigma_a = all_sigmas[t_scalar_a].to("cuda:0", dtype=torch.float32)[:, None, None]
        noise_a = torch.randn_like(clean_action)
        noisy_a = ((1.0 - sigma_a) * clean_action + sigma_a * noise_a).to(dtype=torch.bfloat16)
        v_a_target = (noise_a - clean_action).float()
        ts_expanded = timestep_scalar_a[:, None].expand(-1, action_horizon)  # (B, 24)

        action_head.set_frozen_modules_to_eval_mode()
        action_head.train()
        embodiment_id = torch.tensor([args.embodiment_id], device="cuda:0", dtype=torch.long)
        emb_id_0 = torch.tensor([0], device="cuda:0", dtype=torch.long)

        if use_dit:
            # ── Bucket LONG (T≥49): DiT joint path ───────────────────────────
            # action=noisy_a → joint attention (video ↔ action tokens).
            # action_noise_pred_joint: action velocity predicted via joint attention.
            # Including it in loss gives action_encoder direct gradient through
            # the joint attention path (not just indirect via video token updates).
            state_obs    = states_norm[0].unsqueeze(0).unsqueeze(0).to("cuda:0", dtype=torch.float32)
            state_padded = F.pad(state_obs, (0, 64 - 6)).to(dtype=torch.bfloat16)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_pred, action_noise_pred_joint = action_head.model(
                    noisy_z.transpose(1, 2),
                    timestep=timestep, clip_feature=clip_feas, y=ys,
                    context=prompt_embs, seq_len=seq_len,
                    state=state_padded, embodiment_id=embodiment_id,
                    action=noisy_a, timestep_action=ts_expanded, clean_x=None,
                )
            vt, vp = v_target, v_pred
            if vt.shape != vp.shape:
                vt = vt[..., :vp.shape[-2], :vp.shape[-1]]
            L_dit    = F.mse_loss(vp.float(), vt.float())
            # Joint action loss: action_decoder output from DiT joint attention path
            L_action = F.mse_loss(action_noise_pred_joint.float(), v_a_target)
            L_total  = args.lambda_dit * L_dit + args.lambda_action * L_action
        else:
            # ── Bucket SHORT (T<49): DiT without action (frame_seqlen constraint) ──
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_pred, _ = action_head.model(
                    noisy_z.transpose(1, 2),
                    timestep=timestep, clip_feature=clip_feas, y=ys,
                    context=prompt_embs, seq_len=seq_len,
                    state=torch.zeros(1, 1, 64, device="cuda:0", dtype=torch.bfloat16),
                    embodiment_id=embodiment_id,
                    action=None, timestep_action=None, clean_x=None,
                )
            vt, vp = v_target, v_pred
            if vt.shape != vp.shape:
                vt = vt[..., :vp.shape[-2], :vp.shape[-1]]
            L_dit = F.mse_loss(vp.float(), vt.float())
            # ae path: no joint attention → standalone pass for action_encoder gradient
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                feat_a = action_head.model.action_encoder(noisy_a, ts_expanded, emb_id_0)
                pred_a = action_head.model.action_decoder(feat_a, emb_id_0)
            L_action = F.mse_loss(pred_a.float(), v_a_target)
            L_total  = args.lambda_dit * L_dit + args.lambda_action * L_action

        if torch.isnan(L_total) or torch.isinf(L_total):
            print(f"[warn] NaN/Inf at step {step} "
                  f"(dit={L_dit.item():.4f} act={L_action.item():.4f}), skipping backward")
            optimizer.zero_grad()
            continue

        (L_total / args.grad_accum).backward()

        # ── Gradient flow check (step 1 only) ────────────────────────────
        if step == 1:
            modules = {
                "action_encoder": "action_encoder",
                "action_decoder": "action_decoder",
                "lora":           "lora",
            }
            for label, key in modules.items():
                grads = [p.grad.norm().item() for n, p in action_head.named_parameters()
                         if key in n.lower() and p.grad is not None]
                no_grad = sum(1 for n, p in action_head.named_parameters()
                              if key in n.lower() and p.requires_grad and p.grad is None)
                if grads:
                    print(f"[grad_check] {label}: mean={sum(grads)/len(grads):.6f}  "
                          f"max={max(grads):.6f}  params_with_grad={len(grads)}  no_grad={no_grad}")
                else:
                    print(f"[grad_check] {label}: NO GRADIENT  requires_grad_params="
                          f"{sum(1 for n,p in action_head.named_parameters() if key in n.lower() and p.requires_grad)}")

        # ── Gradient accumulation ─────────────────────────────────────────
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in action_head.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler_lr.step()
            optimizer.zero_grad()

        dt = time.time() - t0
        if step % 10 == 0:
            lr_now = scheduler_lr.get_last_lr()[0]
            path_tag = "dit" if use_dit else "ae"
            print(f"[{step:04d}/{args.num_steps}] "
                  f"L_dit={L_dit.item():.4f}  L_act={L_action.item():.4f}  "
                  f"L={L_total.item():.4f}  lr={lr_now:.2e}  {dt:.1f}s  [{path_tag}]")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step,
                    "L_dit": L_dit.item(),
                    "L_action": L_action.item(),
                    "L_total": L_total.item(), "lr": lr_now,
                    "path": path_tag,
                }) + "\n")

        if step % args.save_every == 0:
            ckpt_path = out_dir / f"lora_step{step:05d}.pt"
            lora_state = {
                k: v for k, v in action_head.state_dict().items()
                if "lora" in k.lower() or any(
                    m in k for m in ["state_encoder", "action_encoder", "action_decoder"]
                )
            }
            torch.save(lora_state, ckpt_path)
            print(f"[ckpt] Saved {ckpt_path} ({len(lora_state)} tensors)")

    # Final save
    final_path = out_dir / "lora_final.pt"
    lora_state = {
        k: v for k, v in action_head.state_dict().items()
        if "lora" in k.lower() or any(
            m in k for m in ["state_encoder", "action_encoder", "action_decoder"]
        )
    }
    torch.save(lora_state, final_path)
    print(f"[done] Final LoRA saved → {final_path}")


if __name__ == "__main__":
    main()
