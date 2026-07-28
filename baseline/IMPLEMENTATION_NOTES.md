# DreamZero SO-100 Fine-tuning 구현 노트

> **매 쿼리 시작 시 반드시 이 파일을 읽고 작업에 반영할 것.**

---

## 데이터 구조

- 경로: `/home1/sota/inha2026/data/train/<user>/<task>/data/chunk-*/*.parquet`
- Parquet 컬럼:
  - `action`: (6,) float — 관절 속도/위치 명령
  - `observation.state`: (6,) float — 현재 관절 각도 (action과 같은 차원, 유사 범위)
  - `frame_index`, `episode_index`, `timestamp`, `task_index`
- 영상: `videos/observation.images.image/chunk-*/video_episode_*.mp4`
- 클립 길이: 16프레임, stride 8 (50% overlap)
- 정규화 상수: EVAL_MEAN/EVAL_STD (feature_csv_utils.py와 동일)

---

## 모델 아키텍처 제약

### frame_seqlen=880 문제
- 모델 config: `frame_seqlen=880`
- 우리 영상: T_lat=4 → tokens = 4×220 = **880** (= frame_seqlen)
- `_blockwise_causal_flash_attn` assert: `num_image_blocks == num_action_blocks`
  - `image_blocks_len = total - frame_seqlen - action - state = 880 - 880 - 24 - 1 = -25` (불가)
  - → **action을 DiT에 직접 넣을 수 없음**
  - 최소 필요 프레임: ~48프레임 (video tokens ≥ 2×frame_seqlen)

### Teacher Forcing (TF) 경로
- `clean_x is not None` → `is_tf=True`
- TF 경로에서도 동일 layout check → 동일 crash
- → `clean_x=None` 사용

### 현재 해결책: standalone pass
- DiT forward: `action=None, clean_x=None` (안전)
- action_encoder, state_encoder는 별도 standalone pass로 학습

---

## 학습 모듈별 gradient 현황 (Job 28249 기준)

| 모듈 | gradient | 비고 |
|------|----------|------|
| LoRA (800 params) | ✅ mean=0.017 | DiT forward L_dit |
| action_encoder (6 params) | ✅ mean=0.620 | standalone AE pass |
| action_decoder (4 params) | ✅ mean=2.837 | standalone AE pass |
| state_encoder (4 params) | ✅ (Job 28250~) | standalone state pass |

---

## Loss 구성

```
L_total = L_ae
        + λ_dit   * L_dit       # DiT flow matching (LoRA)
        + λ_action * L_action   # action_encoder/decoder denoising
        + λ_state  * L_state    # state_encoder reconstruction (첫 프레임 state → decoder → 복원)
        + λ_r3d   * L_r3d       # R3D-18 cosine (motion similarity)
        + λ_dino  * L_dino      # DINOv2 cosine (visual fidelity)
```

현재 λ 설정: `dit=0.1, action=1.0, state=1.0, r3d=1.0, dino=1.0`

---

## state_encoder 연결 방식

- `state_encoder = CategorySpecificMLP(input_dim=64, hidden_dim=hidden_size, output_dim=dim)`
- 입력: `(B, 1, 64)` — observation.state (6-dim)을 zero-padding으로 64dim 확장
- standalone pass:
  1. `feat_s = state_encoder(state_padded, emb_id_0)` → `(B, 1, dim)`
  2. `pred_s = action_decoder(feat_s, emb_id_0)` → `(B, 1, 32)`
  3. `L_state = MSE(pred_s[:,:,:6], state_obs)` — 첫 6dim만 비교

---

## action_encoder 연결 방식

- `action_encoder = MultiEmbodimentActionEncoder(action_dim=32, hidden_size=dim, num_embodiments=1)`
- `timestep_action`: `(B, action_horizon=24)` — scalar를 expand
- standalone pass:
  1. noisy_a = (1-σ)*clean_action + σ*noise
  2. `feat_a = action_encoder(noisy_a, ts_expanded, emb_id_0)` → `(B, 24, dim)`
  3. `pred_a = action_decoder(feat_a, emb_id_0)` → `(B, 24, 32)`
  4. `L_action = MSE(pred_a, v_a_target)` — flow matching velocity

---

## 중요 파일 경로

- 학습 스크립트: `/home1/sota/inha2026/scripts/finetune_dreamzero_full.py`
- SLURM sbatch: `/home1/sota/inha2026/scripts/finetune_dreamzero_full.sbatch`
- 체크포인트 저장: `/home1/sota/inha2026/checkpoints/dreamzero_v2_lora/`
- 로그: `/home1/sota/inha2026/logs/dz_v2_ft_<jobid>.log`
- 모델 config: `/home1/sota/.cache/huggingface/hub/models--Vizuara--dreamzero-so101-lora/.../config.json`
  - `frame_seqlen: 880`, `num_action_per_block: 24`, `num_state_per_block: 1`, `num_frame_per_block: 2`, `action_horizon: 24`, `max_state_dim: 64`
- DiT 모듈: `/home1/sota/inha2026/dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
- action_encoder 모듈: `/home1/sota/inha2026/dreamzero/groot/vla/model/n1_5/modules/action_encoder.py`

---

## 과거 실패 이력

| Job | 원인 | 수정 |
|-----|------|------|
| ~28027 | action=None → action_encoder gradient=0 | standalone pass 추가 |
| 28227 | TF layout crash (clean_x≠None + action≠None) | clean_x=None |
| 28244 | timestep shape (B,) → SinusoidalPE expects (B,T) | ts_expanded = scalar[:,None].expand(-1, 24) |
| 28247 | blockwise attn assert (action≠None in DiT) | action=None in DiT 유지 |
