"""
Cosmos3-Nano forward_dynamics LoRA 파인튜닝 (신규 설계).

파이프라인 내부 헬퍼(_prepare_text_segment/_prepare_vision_segment/_prepare_action_segment/
_encode_video)를 그대로 재사용해 학습용 forward pass를 직접 구성한다 (Cosmos3OmniPipeline.__call__은
@torch.no_grad()라 학습에 재사용 불가).

Flow-matching 컨벤션 (scheduler_config.json: prediction_type=flow_prediction 실측 확인):
  x_t = (1 - sigma) * x0 + sigma * noise
  model_output(target) = noise - x0
  (diffusers UniPCMultistepScheduler.convert_model_output: x0_pred = sample - sigma_t * model_output 로부터 역산)

forward_dynamics 모드에서는 action 전 구간이 condition(clean)이라 action에는 loss가 없다
(action_mse_loss_indexes가 항상 비어 있음 — transformer_cosmos3.py 실측 확인). Video(vision) 토큰 중
condition frame(0번, 첫 프레임)을 제외한 나머지에만 flow-matching MSE loss를 건다.

action_proj_in/out은 nn.Embedding 기반 DomainAwareLinear라 domain_id=7(bridge_orig_lerobot)로만
forward하면 gradient가 자동으로 도메인 7 행에만 흐른다 (다른 도메인 행은 lookup되지 않아 grad=0) —
별도 마스킹 없이 전체 파라미터를 그대로 학습 가능.
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

# ACTION_SIGMA_CLAMP=0 → (1-sigma).clamp weighting 제거 (constant weight)
# 기본값 1 (기존 동작 유지, 하위호환)
_ACTION_SIGMA_CLAMP = os.environ.get("ACTION_SIGMA_CLAMP", "1") == "1"

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from so100_to_bridge_v3 import ACTION_REPRS, build_action_features  # noqa: E402

# action 표현 방식. main()에서 --action-repr 로 설정한다.
#   absolute      : 절대 EE pose 10D (v1~v4). 공식 문서 기준으로는 틀린 표현
#   delta_local   : 이전 프레임 EE 좌표계 기준 delta 10D (Cosmos3 규약, 기준계 후보 1)
#   delta_base    : 로봇 base 좌표계 기준 delta 10D (기준계 후보 2, v5)
#   geom_v3       : delta_base 10D + 절대 EE pose + FK 관절 위치 + 관절각 = 37D
#   geom_v3_local : 위와 같되 delta를 local 기준으로
# 분기는 so100_to_bridge_v3.build_action_features 한 곳에만 둔다 (경로 간 불일치 방지).
ACTION_REPR = "absolute"
# 첫 스텝에서 action 텐서의 shape/값을 실제로 출력한다 (--smoke-test가 켠다).
# 새 표현을 넣을 때 silent failure(전부 0, 폭 불일치, NaN)를 잡기 위한 것.
VERBOSE_FIRST_STEP = False
# 계측: _decode_x0이 이 dict가 None이 아니면 clamp saturation·값 범위를 여기에 기록한다.
# 매 스텝 새로 덮어쓰므로 학습 루프가 diag 주기마다 읽어 sidecar JSON에 append.
_DEC_SAT_LOG = None
# 프레임 가중 flow loss — 첫 프레임 쪽 가중치, 마지막이 1.0. 선형 감소.
# None이면 균등 (기존 동작).
_FLOW_FRAME_WEIGHT = None
# Latent motion loss weight — action 방향성을 직접 감독. 0이면 비활성 (기존).
_MOTION_WEIGHT = 0.0
_MOTION_DAMP = True   # 고노이즈 스텝에서 (1-σ)^0.5로 완화. False면 감쇠 없음.


def to_bridge10(joints):
    return build_action_features(joints, ACTION_REPR)

sys.path.insert(0, "/home1/sota/inha2026/submission_kit")
from action_extractor import load_so100_action_extractor_checkpoint  # noqa: E402

from diffusers import Cosmos3OmniPipeline  # noqa: E402
from peft import LoraConfig  # noqa: E402

ACTION_EXTRACTOR_CKPT = "/home1/sota/inha2026/submission_kit/checkpoints/action_extractor.ckpt"
TRAIN_ACTION_STATS = "/home1/sota/inha2026/data/train/so100_action_statistics.json"

MODEL_ID = "nvidia/Cosmos3-Nano"
DOMAIN_NAME = "bridge_orig_lerobot"
DOMAIN_ID = 7
CHUNK_SIZE = 17  # target_frames = chunk_size+1 = 18 -> VAE 디코드시 17 pixel frame -> [:16] 사용(추론시)
TARGET_FRAMES = CHUNK_SIZE + 1
RESOLUTION_TIER = 480
PROMPT = "A robotic arm on a tabletop performing a manipulation task, static camera."
TRAIN_DIR = "/home1/sota/inha2026/data/train"

LORA_TARGET_MODULES = [
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
]


class SO100ClipDataset(torch.utils.data.Dataset):
    """episode 하나에서 (target_frames 길이의 비디오 클립, 대응 action) 하나를 샘플링."""

    def __init__(self, train_dir: str, target_frames: int, max_episodes: int | None = None,
                 exclude_holdout: bool = True):
        self.target_frames = target_frames
        pq_files = sorted(glob.glob(f"{train_dir}/*/*/data/chunk-*/episode_*.parquet"))
        # 홀드아웃 배제 — pc_ctrl 8k가 리더보드 0.435 (예측 0.271 대비 60% 이탈)로 누출 확정.
        # 신규 학습은 이 200 에피소드를 미포함하여 진짜 미본 홀드아웃 확보.
        if exclude_holdout:
            from holdout_split import filter_training_pool
            pq_files = filter_training_pool(pq_files)
        rng = np.random.default_rng(0)
        if max_episodes and len(pq_files) > max_episodes:
            pq_files = list(rng.choice(pq_files, size=max_episodes, replace=False))
        self.items = []
        for pq in pq_files:
            video = pq.replace("/data/chunk-", "/videos/chunk-").replace(
                "/episode_", "/observation.images.image/episode_"
            ).replace(".parquet", ".mp4")
            if Path(video).exists():
                self.items.append((pq, video))
        print(f"[dataset] usable episodes: {len(self.items)} / {len(pq_files)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pq, video = self.items[idx]
        df = pd.read_parquet(pq, columns=["action"])
        actions = np.stack(df["action"].to_numpy()).astype(np.float32)  # (N,6)
        n = len(actions)
        if n < self.target_frames:
            return self.__getitem__((idx + 1) % len(self))
        start = random.randint(0, n - self.target_frames)

        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.target_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) < self.target_frames:
            return self.__getitem__((idx + 1) % len(self))

        clip = np.stack(frames)  # (T,H,W,3) uint8
        joints = actions[start : start + self.target_frames]  # (T,6)
        return clip, joints


def build_static_segments(pipe, prompt, action_domain_id, device, dtype):
    cond_ids, uncond_ids = pipe.tokenize_prompt(
        prompt=prompt,
        negative_prompt=None,
        num_frames=TARGET_FRAMES,
        height=RESOLUTION_TIER,
        width=RESOLUTION_TIER,
        fps=6.0,
        action_mode="forward_dynamics",
        action_view_point="third_person_view",
    )
    text_seg = pipe._prepare_text_segment(cond_ids, device=device)
    return text_seg


def decode_and_predict_action(pipe, action_extractor, latents, pred_v, noisy_mask_v, sigma):
    """x0_pred(노이즈 latent에서 velocity로 역산) -> VAE 디코드 -> action_extractor로 action 예측.

    flow-matching 역산(x0_pred = sample - sigma*model_output)은 UniPCMultistepScheduler.convert_model_output
    (flow_prediction 분기, scheduling_unipc_multistep.py:805-807 실측)과 동일한 공식이다.
    """
    pred_v_masked = pred_v.float() * noisy_mask_v
    if pred_v_masked.ndim == 4:
        pred_v_masked = pred_v_masked.unsqueeze(0)  # (1,C,T,H,W)
    lat = latents.float()
    if lat.ndim == 4:
        lat = lat.unsqueeze(0)
    x0_pred = lat - sigma * pred_v_masked  # (1,C,T,H,W)

    mean = pipe._vae_latents_mean.to(device=x0_pred.device, dtype=torch.float32)
    inv_std = pipe._vae_latents_inv_std.to(device=x0_pred.device, dtype=torch.float32)
    z_raw = (x0_pred / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)).to(pipe.vae.dtype)
    # 원해상도(480x640) decode 그래프는 ~80GB로 96GB에서도 OOM.
    # latent 공간을 절반으로 줄여 240x320으로 decode(그래프 ~1/4) — action(로봇 팔 위치) 판별에는 충분.
    t_lat, h_lat, w_lat = z_raw.shape[2], z_raw.shape[3], z_raw.shape[4]
    z_small = F.interpolate(
        z_raw.float(), size=(t_lat, h_lat // 2, w_lat // 2), mode="trilinear", align_corners=False
    ).to(pipe.vae.dtype)
    decoded = torch.utils.checkpoint.checkpoint(
        lambda z: pipe.vae.decode(z).sample, z_small, use_reentrant=False
    ).float()  # (1,3,T_pixel,H/2,W/2), 대략 [-1,1]
    decoded = decoded.clamp(-1.0, 1.0)
    # 채점기(make_submission_csv)와 동일한 320x512로 리사이즈 — extractor 학습 분포 일치 + 메모리 절감
    b, c, t_pix, h, w = decoded.shape
    decoded = F.interpolate(
        decoded.permute(0, 2, 1, 3, 4).reshape(b * t_pix, c, h, w),
        size=(320, 512), mode="bilinear", align_corners=False,
    ).reshape(b, t_pix, c, 320, 512).permute(0, 2, 1, 3, 4)
    # eval 모드 RNN은 cudnn backward 불가 -> GRU 경로만 cudnn 비활성으로 실행
    with torch.backends.cudnn.flags(enabled=False):
        pred_action = action_extractor(decoded)  # (1,T_pixel,6)
    return pred_action, t_pix


def _decode_x0(pipe, latents, pred_v, noisy_mask_v, sigma):
    """노이즈 latent에서 x0를 역산해 픽셀로 디코드 (미분 가능).

    `decode_and_predict_action`과 같은 패턴이다 — 원해상도 decode 그래프는 ~80GB라
    96GB에서도 OOM이므로 latent를 공간적으로 절반으로 줄이고 단일 checkpoint로 디코드한다.
    (Cosmos VAE는 3D라 DreamZero처럼 프레임별로 쪼갤 수 없다.)
    """
    pv = pred_v.float() * noisy_mask_v
    if pv.ndim == 4:
        pv = pv.unsqueeze(0)
    lat = latents.float()
    if lat.ndim == 4:
        lat = lat.unsqueeze(0)
    x0 = lat - sigma * pv
    mean = pipe._vae_latents_mean.to(device=x0.device, dtype=torch.float32)
    inv = pipe._vae_latents_inv_std.to(device=x0.device, dtype=torch.float32)
    z = x0 / inv.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
    t_l, h_l, w_l = z.shape[2], z.shape[3], z.shape[4]
    z = F.interpolate(z, size=(t_l, h_l // 2, w_l // 2), mode="trilinear", align_corners=False)
    dec = torch.utils.checkpoint.checkpoint(
        lambda a: pipe.vae.decode(a.to(pipe.vae.dtype)).sample, z, use_reentrant=False
    ).float()
    # clamp saturation 실측 — 픽셀이 [-1,1] 밖으로 얼마나 벗어났는지. gradient가 죽는 영역이 얼마나 되는지 진단.
    if _DEC_SAT_LOG is not None:
        with torch.no_grad():
            _DEC_SAT_LOG["hi"] = float((dec >= 1.0).float().mean())
            _DEC_SAT_LOG["lo"] = float((dec <= -1.0).float().mean())
            _DEC_SAT_LOG["max"] = float(dec.max())
            _DEC_SAT_LOG["min"] = float(dec.min())
    return dec.clamp(-1.0, 1.0)                       # (1,3,T,H,W)


def perceptual_loss(pipe, dino, r3d, dino_size, latents, pred_v, noisy_mask_v, sigma,
                    clip_np, device, w_dino, w_r3d):
    """채점 지표(DINOv2 · R3D-18)를 직접 최적화하는 loss.

    왜 필요한가 (2026-08-05 실측):
      VAE 왕복 바닥값 DINO 0.0102 / Video 0.0014인데 우리 모델은 0.1006 / 0.0510.
      즉 오차의 90~97%가 아키텍처가 아니라 모델 탓이고 회수 가능하다.
      그런데 달성 가능 구간 중 회수율이 DINO 14% / Video 47%다 —
      현재 loss가 latent flow matching이라 픽셀→특징 공간의 채점과 어긋나 있다.
      이 loss가 그 간극을 직접 메운다.
    """
    from feature_csv_utils import (IMAGENET_MEAN, IMAGENET_STD, KINETICS_MEAN, KINETICS_STD,
                                   _resize_pad_frame_batch, _normalize_image_model_output)
    grad_ckpt = torch.utils.checkpoint.checkpoint

    dec = _decode_x0(pipe, latents, pred_v, noisy_mask_v, sigma)     # (1,3,T,h,w) [-1,1]
    T = dec.shape[2]
    n = min(T, len(clip_np))
    dec = dec[:, :, :n]
    h, w = dec.shape[3], dec.shape[4]

    gt = torch.from_numpy(clip_np[:n]).to(device).float().permute(3, 0, 1, 2).unsqueeze(0)
    gt = gt / 255.0 * 2.0 - 1.0
    gt = F.interpolate(gt, size=(n, h, w), mode="trilinear", align_corners=False)

    losses = {}
    total = torch.zeros((), device=device)

    if w_dino > 0 and dino is not None:
        def dino_prep(v):
            x = ((v[0].permute(1, 0, 2, 3) + 1.0) / 2.0) * 255.0      # (T,3,h,w) 0~255
            x = _resize_pad_frame_batch(x, dino_size)                  # 내부에서 /255
            return (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)

        # ⚠ 전 프레임을 하나의 checkpoint로 감싸면 backward 재계산 때 모든 프레임의
        #   활성값이 동시에 살아나 OOM이 난다(gpu-109 44GB에서 실측). 청크로 쪼갠다.
        CH = 4
        xp = dino_prep(dec)
        f_p = torch.cat([_normalize_image_model_output(
            grad_ckpt(dino, xp[i:i + CH], use_reentrant=False)) for i in range(0, xp.shape[0], CH)], 0)
        with torch.no_grad():
            xg = dino_prep(gt)
            f_g = torch.cat([_normalize_image_model_output(dino(xg[i:i + CH]))
                             for i in range(0, xg.shape[0], CH)], 0)
        l = (1.0 - F.cosine_similarity(f_p.float(), f_g.float(), dim=-1)).mean()
        losses["dino"] = float(l.detach())
        total = total + w_dino * l

    if w_r3d > 0 and r3d is not None:
        def prep(v):
            x = (v + 1.0) / 2.0
            x = F.interpolate(x, size=(v.shape[2], 112, 112), mode="trilinear", align_corners=False)
            return (x - KINETICS_MEAN.to(device)) / KINETICS_STD.to(device)
        f_p = _normalize_image_model_output(grad_ckpt(r3d, prep(dec), use_reentrant=False))
        with torch.no_grad():
            f_g = _normalize_image_model_output(r3d(prep(gt)))
        l = (1.0 - F.cosine_similarity(f_p.float(), f_g.float(), dim=-1)).mean()
        losses["r3d"] = float(l.detach())
        total = total + w_r3d * l

    return total, losses


def train_step(pipe, transformer, batch, device, dtype, scheduler, generator, action_extractor=None, action_mean=None, action_std=None, action_loss_weight=0.0,
               perc=None):
    clip_np, joints = batch  # clip_np: (T,H,W,3) uint8, joints: (T,6)
    T_len = clip_np.shape[0]

    # 1) 첫 프레임 PIL -> action video conditioning canvas(패딩 포함) -> VAE 인코딩(x0)
    first_frame = Image.fromarray(clip_np[0])
    cond_clip, action_image_size, h, w = pipe._prepare_action_video_conditioning(
        [Image.fromarray(f) for f in clip_np], RESOLUTION_TIER, T_len, device=device, dtype=dtype
    )
    with torch.no_grad():
        x0_vision = pipe._encode_video(cond_clip).contiguous().float()
        x0_vision = pipe._remove_action_video_padding_from_latent(x0_vision, action_image_size)
    latent_t = x0_vision.shape[2]

    # 2) action -> bridge10D -> action_dim(64)로 zero-pad, chunk_size 길이로 자르기(transition 15개, 여기선 17)
    bridge10 = to_bridge10(joints[:CHUNK_SIZE])  # (chunk_size,10)
    raw_actions = torch.from_numpy(bridge10).to(device=device, dtype=dtype)
    action_dim = transformer.action_dim
    pad = torch.zeros(raw_actions.shape[0], action_dim - raw_actions.shape[1], device=device, dtype=dtype)
    x0_action = torch.cat([raw_actions, pad], dim=-1)  # (chunk_size, action_dim) 전부 clean(condition)

    global VERBOSE_FIRST_STEP
    if VERBOSE_FIRST_STEP:
        VERBOSE_FIRST_STEP = False
        f = x0_action.float()
        nz = int((f.abs().sum(0) > 0).sum())
        print(f"\n[첫 스텝 진단] action_repr={ACTION_REPR}")
        print(f"  bridge feature : {bridge10.shape}  (기대 {ACTION_REPR} → 폭 {bridge10.shape[1]})")
        print(f"  x0_action      : {tuple(x0_action.shape)}  비영 차원 {nz}/{action_dim}")
        print(f"  값 범위        : [{f.min():.3f}, {f.max():.3f}]  평균|x| {f.abs().mean():.3f}")
        print(f"  NaN/Inf        : {int(torch.isnan(f).sum())} / {int(torch.isinf(f).sum())}")
        print(f"  vision latent  : {tuple(x0_vision.shape)}  clean 프레임 1개(index 0)")
        if nz != bridge10.shape[1]:
            print(f"  ⚠ 비영 차원({nz})이 feature 폭({bridge10.shape[1]})과 다름 — 상수 0 차원이 있는지 확인")
        print(flush=True)

    # 3) flow-matching 노이즈: vision은 frame0 제외 전부 noisy, action은 전부 clean(forward_dynamics)
    idx = random.randrange(len(scheduler.timesteps))
    sigma = scheduler.sigmas[idx].to(device=device, dtype=torch.float32)
    timestep = scheduler.timesteps[idx]

    vision_condition_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=dtype)
    vision_condition_mask[0, 0, 0] = 1.0
    noise_vision = torch.randn(x0_vision.shape, generator=generator, device=device, dtype=torch.float32)
    x_t_vision = (1 - sigma) * x0_vision.float() + sigma * noise_vision
    latents = (vision_condition_mask.float() * x0_vision.float() + (1 - vision_condition_mask.float()) * x_t_vision).to(dtype)
    velocity_target_vision = (noise_vision - x0_vision.float()).to(dtype)

    action_latents = x0_action  # 전부 clean

    # 4) 세그먼트 조립 (text + vision + action)
    text_seg = build_static_segments(pipe, PROMPT, DOMAIN_ID, device, dtype)
    vision_seg = pipe._prepare_vision_segment(
        input_vision_tokens=latents,
        has_image_condition=True,
        mrope_offset=text_seg["vision_start_temporal_offset"],
        vision_fps=6.0,
        curr=text_seg["und_len"],
        device=device,
        condition_frame_indexes=[0],
    )
    action_seg = pipe._prepare_action_segment(
        input_action_tokens=action_latents,
        condition_frame_indexes=list(range(CHUNK_SIZE)),  # 전부 clean
        mrope_offset=text_seg["vision_start_temporal_offset"],
        action_fps=6.0,
        curr=text_seg["und_len"] + vision_seg["num_vision_tokens"],
        device=device,
    )
    position_ids = torch.cat([text_seg["text_mrope_ids"], vision_seg["vision_mrope_ids"], action_seg["action_mrope_ids"]], dim=1)
    sequence_length = text_seg["und_len"] + vision_seg["num_vision_tokens"] + action_seg["action_len"]

    vision_timesteps = torch.full((vision_seg["num_noisy_vision_tokens"],), float(timestep), device=device)
    action_domain_id = torch.tensor([DOMAIN_ID], dtype=torch.long, device=device)

    preds_vision, _, preds_action = transformer(
        input_ids=text_seg["input_ids"],
        text_indexes=text_seg["text_indexes"],
        position_ids=position_ids,
        und_len=text_seg["und_len"],
        sequence_length=sequence_length,
        vision_tokens=[latents],
        vision_token_shapes=vision_seg["vision_token_shapes"],
        vision_sequence_indexes=vision_seg["vision_sequence_indexes"],
        vision_mse_loss_indexes=vision_seg["vision_mse_loss_indexes"],
        vision_timesteps=vision_timesteps,
        vision_noisy_frame_indexes=vision_seg["vision_noisy_frame_indexes"],
        action_tokens=[action_latents],
        action_token_shapes=action_seg["action_token_shapes"],
        action_sequence_indexes=action_seg["action_sequence_indexes"],
        action_mse_loss_indexes=action_seg["action_mse_loss_indexes"],
        action_timesteps=torch.zeros((0,), device=device),
        action_noisy_frame_indexes=action_seg["action_noisy_frame_indexes"],
        action_domain_ids=[action_domain_id],
    )

    pred_v, mask_v = preds_vision[0], vision_condition_mask
    noisy_mask_v = (1.0 - mask_v).to(dtype=torch.float32).expand_as(pred_v)
    sq_err = (pred_v.float() - velocity_target_vision.float()) ** 2 * noisy_mask_v

    # 텐서 레이아웃 탐지: x0_vision의 T축이 어디에 있는지 latent_t로 매칭.
    # 각기 다른 pipeline 리비전에서 (B,C,T,H,W) 또는 (B,T,C,H,W) 가능. 한 번만 판별 후 재사용.
    _T_AXIS = None
    for ax in range(pred_v.ndim):
        if pred_v.shape[ax] == latent_t:
            _T_AXIS = ax
            break
    if _T_AXIS is None:
        raise RuntimeError(f"pred_v shape {tuple(pred_v.shape)}에서 T={latent_t} 축 탐지 실패")

    # 프레임 가중 (선택). F 분석에서 프레임 3~6이 취약함이 확인됨.
    if _FLOW_FRAME_WEIGHT is not None:
        w = torch.linspace(_FLOW_FRAME_WEIGHT, 1.0, latent_t, device=pred_v.device, dtype=torch.float32)
        view_shape = [1] * pred_v.ndim; view_shape[_T_AXIS] = latent_t
        w = w.view(*view_shape)
        sq_err = sq_err * w
        mask_w = noisy_mask_v * w
        flow_loss = sq_err.sum() / mask_w.sum().clamp(min=1.0)
    else:
        flow_loss = sq_err.sum() / noisy_mask_v.sum().clamp(min=1.0)

    # Latent motion loss — action 방향성을 직접 감독. flow_loss는 노이즈→클린 방향만 배우고
    # 프레임 간 변화 벡터는 간접 학습. 여기서 x0_hat 프레임 차분을 GT 차분과 맞추도록 강제.
    motion_loss = torch.zeros((), device=device)
    if _MOTION_WEIGHT > 0:
        pv = pred_v.float() * noisy_mask_v
        x0_hat = latents.float() - sigma * pv
        # T축에서 인접 프레임 차분 (index_select로 안전하게)
        idx_next = torch.arange(1, latent_t, device=device)
        idx_prev = torch.arange(0, latent_t - 1, device=device)
        d_hat = x0_hat.index_select(_T_AXIS, idx_next) - x0_hat.index_select(_T_AXIS, idx_prev)
        d_gt  = x0_vision.float().index_select(_T_AXIS, idx_next) - x0_vision.float().index_select(_T_AXIS, idx_prev)
        motion_loss = ((d_hat - d_gt) ** 2).mean()
        # σ 감쇠 옵션: 고노이즈에서 x0_hat 오염되므로 절반은 (1-σ)로 감쇠.
        # perceptual과 달리 x0가 latent이라 오염이 작음 → 완전 감쇠 대신 (1-σ)^0.5 정도 부드럽게.
        damp = float((1.0 - sigma).clamp(min=0.0, max=1.0)) ** 0.5 if _MOTION_DAMP else 1.0
        loss_add_motion = _MOTION_WEIGHT * damp * motion_loss

    action_loss = torch.zeros((), device=device)
    if action_extractor is not None and action_loss_weight > 0:
        pred_action, pred_frames = decode_and_predict_action(
            pipe, action_extractor, latents, pred_v, noisy_mask_v, sigma
        )
        gt = torch.from_numpy(joints[:pred_frames]).to(device=device, dtype=torch.float32)
        gt_norm = ((gt - action_mean) / action_std).unsqueeze(0)  # (1,T,6)
        action_loss = F.mse_loss(pred_action.float(), gt_norm)
        # ACTION_SIGMA_CLAMP=0 시 sigma-independent constant weight.
        # 기본(=1)은 기존 (1-sigma).clamp 유지 — 하위호환.
        if _ACTION_SIGMA_CLAMP:
            weight = float(action_loss_weight) * float((1.0 - sigma).clamp(min=0.0, max=1.0))
        else:
            weight = float(action_loss_weight)
        loss = flow_loss + weight * action_loss
    else:
        loss = flow_loss

    if _MOTION_WEIGHT > 0:
        loss = loss + loss_add_motion

    perc_parts = {"sigma": float(sigma), "motion": float(motion_loss.detach())}
    if perc is not None:
        s = float(sigma)
        # σ gate — 실측: scheduler.sigmas median 0.929로 고노이즈 편중. 대부분 스텝에서
        #   x0_hat=latents−σ·pred_v가 velocity error를 σ배 증폭한 상태라 perceptual gradient가 오염.
        # gate 밖에선 forward도 skip (계산량 절감).
        in_gate = (s <= perc.get("smax", 1.0)) and (s >= perc.get("smin", 0.0))
        if in_gate:
            p_loss, extra = perceptual_loss(
                pipe, perc["dino"], perc["r3d"], perc["dino_size"],
                latents, pred_v, noisy_mask_v, sigma, clip_np, device,
                perc["w_dino"], perc["w_r3d"],
            )
            perc_parts.update(extra)
            if perc.get("damp", True):
                # 기존 (1−σ) 감쇠 — 하위호환. no_damp=True면 사용 안 함.
                loss = loss + float((1.0 - sigma).clamp(min=0.0, max=1.0)) * p_loss
            else:
                loss = loss + p_loss
            perc_parts["p_loss"] = float(p_loss.detach())
        else:
            perc_parts["gated_out"] = 1.0
    return loss, flow_loss.item(), action_loss.item(), perc_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--proj-lr-mult", type=float, default=5.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--grad-acc", type=int, default=4)
    ap.add_argument("--max-episodes", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--ckpt-dir", type=str, default="/home1/sota/inha2026/checkpoints/cosmos3_nano_v1")
    ap.add_argument("--smoke-test", action="store_true", help="1스텝만 실행하고 종료 (forward/backward 검증용)")
    ap.add_argument("--action-loss-weight", type=float, default=0.0, help="0이면 action-following 보조 loss 비활성")
    # 채점 지표 직접 최적화 (docs/plans/metric-targeted/). 0이면 비활성 = 기존 동작.
    ap.add_argument("--dino-loss-weight", type=float, default=0.0)
    ap.add_argument("--r3d-loss-weight", type=float, default=0.0)
    # perceptual σ gating — 1차 실험 결과 (1−σ) 감쇠로도 pc_dino/r3d/both 모두 대조군 대비 악화.
    # scheduler.sigmas 실측 median 0.929로 고노이즈에 편중돼 있음. 그 구간에서 x0_hat이 불안정 →
    # perceptual gradient가 학습 신호가 아니라 잡음으로 흘러 pc_r3d 부호검정 p=0.002로 강한 악화.
    # 대안: hard gate로 신뢰 가능한 σ 구간만 발화시킨다. gate 안에서는 감쇠 없음.
    ap.add_argument("--perc-sigma-max", type=float, default=1.0,
                    help="σ가 이 값 이하일 때만 perceptual loss 계산. 기본 1.0 = 항상")
    ap.add_argument("--perc-sigma-min", type=float, default=0.0,
                    help="σ가 이 값 이상일 때만 perceptual loss 계산. 저노이즈에서 gradient ∂x0_hat/∂v̂=−σ가 작아지는 것 방지")
    ap.add_argument("--perc-no-damp", action="store_true",
                    help="perceptual에 (1−σ) 감쇠를 적용하지 않는다 (hard gate와 함께 사용)")
    # 계측 레이어 — 매 학습 잡이 다음 재설계에 필요한 정보를 부산물로 남긴다.
    # 미지정 시 기존 동작 그대로(오버헤드 0).
    ap.add_argument("--diag-log", type=str, default=None,
                    help="sidecar JSON 경로. 매 학습에서 σ 히스토그램·loss 성분·grad norm·dec saturation을 여기 append.")
    ap.add_argument("--diag-every", type=int, default=50,
                    help="가벼운 기록 주기 (σ, loss 성분, dec saturation). 학습 속도 영향 무시할 수준")
    ap.add_argument("--diag-grad-every", type=int, default=500,
                    help="비싼 grad-norm 분리 주기 (double backward). 500 기본이면 8k에서 16회 → 오버헤드 3~4%")
    # 프레임 가중 flow loss — F 분석에서 프레임 3~6이 취약함이 확인됨.
    # 첫 프레임 쪽 가중치(마지막이 1.0). 1.0이면 균등(기존), 2.0이면 첫 프레임 2배 강조.
    ap.add_argument("--flow-frame-weight", type=float, default=1.0,
                    help="첫 프레임 가중치. 1.0=균등, 2.0=첫 프레임 2배 강조 (마지막은 1.0)")
    # Latent motion loss — action 방향성 직접 감독. flow_loss가 놓치는 프레임 간 변화 벡터를
    # x0_hat 차분과 GT 차분의 MSE로 강제. perceptual과 달리 latent 공간이라 x0 오염이 작음.
    ap.add_argument("--motion-loss-weight", type=float, default=0.0,
                    help="Latent motion loss 가중치. 0=비활성. 권장 시작 0.5")
    ap.add_argument("--motion-no-damp", action="store_true",
                    help="motion loss에 (1-σ)^0.5 감쇠 미적용")
    # 시드. 미지정(None)이면 기존과 동일하게 시드 없음 — 실행 간 잡음 측정(G2)을 위해 기본값을 바꾸지 않는다.
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None, help="이어서 학습할 체크포인트 (rank가 같아야 함)")
    ap.add_argument("--action-repr", type=str, default="absolute", choices=ACTION_REPRS,
                    help="action 표현. Cosmos3 공식 규약은 delta (기술보고서: flange-pose deltas). "
                         "geom_v3는 빈 54채널에 절대 기하를 채운 37D")
    args = ap.parse_args()

    global ACTION_REPR, VERBOSE_FIRST_STEP
    ACTION_REPR = args.action_repr
    VERBOSE_FIRST_STEP = args.smoke_test
    print(f"action 표현: {ACTION_REPR}")

    device = "cuda"
    dtype = torch.bfloat16
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, enable_safety_checker=False)
    pipe.to(device)
    transformer = pipe.transformer
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.rank, target_modules=LORA_TARGET_MODULES, init_lora_weights="gaussian",
    )
    transformer.add_adapter(lora_config)
    lora_params = [p for n, p in transformer.named_parameters() if "lora_" in n]
    # FREEZE_LORA=1 → LoRA freeze, action_proj만 학습 (native path full fine-tune)
    freeze_lora = os.environ.get("FREEZE_LORA", "0") == "1"
    for p in lora_params:
        p.requires_grad_(not freeze_lora)
    if freeze_lora:
        print(f"[FREEZE_LORA=1] LoRA 파라미터 {len(lora_params)}개 freeze — action_proj/embed만 학습", flush=True)

    transformer.action_proj_in.requires_grad_(True)
    transformer.action_proj_out.requires_grad_(True)
    transformer.action_modality_embed.requires_grad_(True)
    proj_params = list(transformer.action_proj_in.parameters()) + list(transformer.action_proj_out.parameters()) + [transformer.action_modality_embed]

    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        sd = transformer.state_dict()
        n_loaded = 0
        for k, v in ck["lora"].items():
            if k in sd:
                sd[k] = v.to(device=device, dtype=sd[k].dtype)
                n_loaded += 1
        transformer.load_state_dict(sd, strict=False)
        transformer.action_proj_in.load_state_dict(ck["action_proj_in"])
        transformer.action_proj_out.load_state_dict(ck["action_proj_out"])
        with torch.no_grad():
            transformer.action_modality_embed.copy_(ck["action_modality_embed"].to(device))
        print(f"이어서 학습: {Path(args.resume).name} (step={ck['step']}, LoRA {n_loaded}/{len(ck['lora'])}개 로드)")
        if n_loaded != len(ck["lora"]):
            raise RuntimeError(f"LoRA 텐서 불일치 — rank가 체크포인트와 다른지 확인 ({n_loaded}/{len(ck['lora'])})")

    n_lora = sum(p.numel() for p in lora_params)
    n_proj = sum(p.numel() for p in proj_params)
    print(f"Trainable: LoRA={n_lora/1e6:.2f}M  action_proj/embed={n_proj/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            {"params": proj_params, "lr": args.lr * args.proj_lr_mult},
        ]
    )

    scheduler = pipe.scheduler
    scheduler.set_timesteps(1000, device=device)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"시드 고정: {args.seed}")

    if args.flow_frame_weight != 1.0:
        globals()["_FLOW_FRAME_WEIGHT"] = args.flow_frame_weight
        print(f"프레임 가중 flow loss: first={args.flow_frame_weight}, last=1.0 (선형)")

    if args.motion_loss_weight > 0:
        globals()["_MOTION_WEIGHT"] = args.motion_loss_weight
        globals()["_MOTION_DAMP"] = not args.motion_no_damp
        print(f"Latent motion loss: weight={args.motion_loss_weight} damp={not args.motion_no_damp}")

    perc = None
    if args.dino_loss_weight > 0 or args.r3d_loss_weight > 0:
        from feature_csv_utils import (load_dino_model, load_video_feature_model,
                                       resolve_dino_image_size)
        d = load_dino_model(device, "vit_small_patch14_dinov2.lvd142m", pretrained=True) \
            if args.dino_loss_weight > 0 else None
        r = load_video_feature_model(device, pretrained=True) if args.r3d_loss_weight > 0 else None
        for mdl in (d, r):
            if mdl is not None:
                mdl.requires_grad_(False)
        perc = dict(dino=d, r3d=r, w_dino=args.dino_loss_weight, w_r3d=args.r3d_loss_weight,
                    dino_size=resolve_dino_image_size(d, requested_size=0) if d is not None else 0,
                    smax=args.perc_sigma_max, smin=args.perc_sigma_min, damp=not args.perc_no_damp)
        print(f"perceptual loss 활성: dino={args.dino_loss_weight} r3d={args.r3d_loss_weight} "
              f"σ∈[{args.perc_sigma_min:.2f}, {args.perc_sigma_max:.2f}] damp={not args.perc_no_damp}")

    action_extractor = None
    action_mean = action_std = None
    if args.action_loss_weight > 0:

        action_extractor = load_so100_action_extractor_checkpoint(ACTION_EXTRACTOR_CKPT, map_location=device)
        action_extractor.to(device=device, dtype=torch.float32)
        action_extractor.eval()
        action_extractor.requires_grad_(False)
        with open(TRAIN_ACTION_STATS) as f:
            stats = json.load(f)
        action_mean = torch.tensor(stats["mean"], device=device, dtype=torch.float32)
        action_std = torch.tensor(stats["std"], device=device, dtype=torch.float32)
        print(f"action-following 보조 loss 활성화 (weight={args.action_loss_weight})")

    dataset = SO100ClipDataset(TRAIN_DIR, TARGET_FRAMES, max_episodes=args.max_episodes)
    generator = torch.Generator(device=device).manual_seed(0)

    # 계측 sidecar — 다음 재설계에 필요한 부산물. 미지정 시 오버헤드 0.
    diag_records = []
    diag_path = None
    if args.diag_log:
        diag_path = Path(args.diag_log)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        globals()["_DEC_SAT_LOG"] = {}  # _decode_x0이 여기 기록
        print(f"계측 로그: {diag_path} (diag_every={args.diag_every}, grad_every={args.diag_grad_every})")

    transformer.train()
    step = 0
    accum_loss = 0.0
    t0 = time.time()
    optimizer.zero_grad()
    data_idx = list(range(len(dataset)))
    random.shuffle(data_idx)
    di = 0
    while step < args.steps:
        if di >= len(data_idx):
            random.shuffle(data_idx)
            di = 0
        batch = dataset[data_idx[di]]
        di += 1

        loss, flow_loss_v, action_loss_v, perc_v = train_step(
            pipe, transformer, batch, device, dtype, scheduler, generator,
            action_extractor=action_extractor, action_mean=action_mean, action_std=action_std,
            action_loss_weight=args.action_loss_weight, perc=perc,
        )
        (loss / args.grad_acc).backward()
        accum_loss += loss.item()

        if (step + 1) % args.grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(lora_params + proj_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % 20 == 0:
            dt = time.time() - t0
            ptxt = "".join(f" {k}={v:.4f}" for k, v in perc_v.items())
            print(f"step {step:06d}/{args.steps} loss={loss.item():.4f} flow={flow_loss_v:.4f} action={action_loss_v:.4f}{ptxt} avg={accum_loss/(step+1):.4f} elapsed={dt:.0f}s", flush=True)

        # 계측 — sidecar에 append. 다음 재설계에서 이 값들을 읽어 별도 잡 없이 판정.
        if diag_path is not None and step % args.diag_every == 0:
            rec = dict(step=step, sigma=perc_v.get("sigma", 0.0),
                       flow_loss=flow_loss_v, action_loss=action_loss_v,
                       loss=loss.item(),
                       dino=perc_v.get("dino", 0.0), r3d=perc_v.get("r3d", 0.0),
                       p_loss=perc_v.get("p_loss", 0.0),
                       gated_out=int(perc_v.get("gated_out", 0.0) > 0))
            # dec saturation (perceptual 활성 스텝만 채워짐)
            if _DEC_SAT_LOG:
                rec.update(dec_sat_hi=_DEC_SAT_LOG.get("hi", 0.0),
                           dec_sat_lo=_DEC_SAT_LOG.get("lo", 0.0),
                           dec_max=_DEC_SAT_LOG.get("max", 0.0),
                           dec_min=_DEC_SAT_LOG.get("min", 0.0))
                _DEC_SAT_LOG.clear()  # 다음 스텝 대비
            # 비싼 grad-norm 분리 — 500스텝마다만
            if perc is not None and step % args.diag_grad_every == 0 and step > 0:
                # 방금 backward가 (loss/grad_acc) 스케일이라 순수 grad 안 됨.
                # 대신 별도 forward+backward 2회로 계산: flow-only, flow+perc.
                # RNG state 저장·복원으로 σ·noise 동일 재현. (diag_perc_grad.py와 동일 원리)
                rng_s = (random.getstate(), np.random.get_state(),
                         torch.get_rng_state(), torch.cuda.get_rng_state(), generator.get_state())
                trainable = lora_params + proj_params
                for p in trainable:
                    if p.grad is not None: p.grad = None
                lf, _, _, _ = train_step(pipe, transformer, batch, device, dtype, scheduler, generator,
                                          action_extractor=None, perc=None)
                lf.backward()
                gf = torch.cat([p.grad.detach().flatten() for p in trainable if p.grad is not None]).cpu().float()
                gnf = float(gf.norm())
                del lf
                for p in trainable:
                    p.grad = None
                torch.cuda.empty_cache()
                random.setstate(rng_s[0]); np.random.set_state(rng_s[1])
                torch.set_rng_state(rng_s[2]); torch.cuda.set_rng_state(rng_s[3])
                generator.set_state(rng_s[4])
                lp, _, _, _ = train_step(pipe, transformer, batch, device, dtype, scheduler, generator,
                                          action_extractor=None, perc=perc)
                lp.backward()
                ga = torch.cat([p.grad.detach().flatten() for p in trainable if p.grad is not None]).cpu().float()
                del lp
                for p in trainable:
                    p.grad = None
                torch.cuda.empty_cache()
                gp = ga - gf
                rec.update(gn_flow=gnf, gn_perc=float(gp.norm()),
                           gn_ratio=float(gp.norm() / max(gnf, 1e-9)),
                           gn_cos=float(torch.nn.functional.cosine_similarity(gf.unsqueeze(0), gp.unsqueeze(0)).item()))
                del gf, ga, gp
            diag_records.append(rec)
            # 매 diag_every마다 flush — 잡이 죽어도 지금까지 자료는 남음
            with open(diag_path, "w") as f:
                json.dump(diag_records, f)

        if args.smoke_test:
            # 규칙: shape·값·gradient를 눈으로 확인한 뒤에만 "성공"으로 본다.
            # 특히 action_proj_in/out은 새 입력 폭을 받는 층이라 gradient가 반드시 흘러야 한다.
            print("[gradient 점검]")
            groups = {"lora": lora_params, "action_proj": proj_params}
            ok = True
            for name, params in groups.items():
                g = [p.grad for p in params if p.grad is not None]
                if not g:
                    print(f"  {name:12s} gradient 없음 ❌")
                    ok = False
                    continue
                norm = torch.sqrt(sum((x.float() ** 2).sum() for x in g)).item()
                nan = sum(int(torch.isnan(x).any()) for x in g)
                print(f"  {name:12s} 파라미터 {len(params):4d}개  grad norm {norm:.4e}  NaN {nan}  "
                      f"{'✓' if norm > 0 and nan == 0 else '❌'}")
                ok = ok and norm > 0 and nan == 0
            print(f"  loss={loss.item():.4f} (finite={bool(torch.isfinite(loss))})")
            print("SMOKE TEST " + ("OK: forward+backward+gradient 모두 정상" if ok else "실패: 위 ❌ 항목 확인"))
            if not ok:
                sys.exit(1)
            return

        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            state = {
                "lora": {n: p.detach().cpu() for n, p in transformer.named_parameters() if "lora_" in n},
                "action_proj_in": transformer.action_proj_in.state_dict(),
                "action_proj_out": transformer.action_proj_out.state_dict(),
                "action_modality_embed": transformer.action_modality_embed.detach().cpu(),
                "step": step + 1,
            }
            torch.save(state, f"{args.ckpt_dir}/ckpt_step{step + 1:06d}.pt")
            print(f"  체크포인트 저장: ckpt_step{step + 1:06d}.pt")

        step += 1

    print("학습 완료")


if __name__ == "__main__":
    main()
