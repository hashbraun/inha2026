# Cosmos3-Nano 구현 기록

> 계획서(PLAN.md) 대비 실제로 구현·검증한 내용. 2026-08-02 ~ 08-03.

## 결과 요약

| 제출 | 모델 | 추론 설정 | 리더보드 |
|---|---|---|---|
| — | baseline (DynamiCrafter 11M) | — | 0.5174 |
| — | DreamZero v5 (Wan2.1-I2V-14B) | step 6 | 0.4500 |
| 1차 | **Cosmos3-Nano v1** | gs=6.0, steps=35 | **0.2934** |
| 2차 | Cosmos3-Nano v4 | gs=3.5, steps=50 | 0.26915 |
| 3차 | **Cosmos3-Nano v4** | **gs=5.0, steps=50** | **0.26809** |

베이스라인 대비 48% 개선.

---

## 1. 아키텍처 실측 (diffusers 0.39.0 소스 + config.json)

`nvidia/Cosmos3-Nano` / `Cosmos3OmniPipeline` / `Cosmos3OmniTransformer`

| 항목 | 값 | 의미 |
|---|---|---|
| `video_temporal_causal` | **false** | 비인과적 어텐션 → 16프레임에서도 action 토큰을 시퀀스에 넣을 수 있음 |
| `num_hidden_layers` / `hidden_size` | 36 / 4096 | |
| `num_attention_heads` / `num_key_value_heads` | 32 / 8 | GQA 4:1 |
| `use_moe` | true | MoT — understanding/generation 경로 분리 |
| `action_dim` / `num_embodiment_domains` | 64 / 32 | |

### action conditioning 경로
`CosmosActionCondition(mode="forward_dynamics", ...)` 네이티브 지원.
`raw_actions [T, 10]` → `action_proj_in`(DomainAwareLinear) → `+ action_modality_embed` → DiT 시퀀스에 패킹.

- `bridge_orig_lerobot` = domain id **7**, `raw_action_dim=10` (9D pose + 1D gripper)
- **domain 7은 사전학습되어 있음** — `action_proj_in.bias` norm 1.48 (미학습 도메인은 정확히 0.0). 재초기화 불필요.
- `action_proj_in/out`은 `nn.Embedding` 기반 `DomainAwareLinear`라 PEFT LoRA로 직접 타겟 불가. 단 domain 7로만 forward하면 gradient가 해당 행에만 흐르므로 전체 파라미터를 그대로 학습해도 됨.

### 프레임 수 관계
`target_frames = chunk_size + 1`. 16프레임 제출을 위해 **`chunk_size=17`** → 디코드 17프레임 → `[:16]` 슬라이스.
`raw_actions`가 chunk_size보다 짧으면 파이프라인이 마지막 값을 자동 반복 패딩.

### Flow matching 규약
`scheduler_config.json`: `prediction_type="flow_prediction"`, `use_flow_sigmas=true`, UniPC solver_order=2.
- `x_t = (1-σ)·x0 + σ·noise`
- target = `noise - x0`  (`convert_model_output`의 `x0_pred = sample - σ·model_output`에서 역산)

---

## 2. 데이터 변환

### v1 (`scripts/so100_to_bridge.py`) — 절대 pose
SO-100 6D 관절각도(degree) → 공식 FK(`baseline/challenge_kit/scripts/so100_fk.py`의 `JOINT_PARAMS`) → EE 변환행렬
→ `[translation 3D, Zhou 6D rotation, gripper]`

**검증**: 공식 `so100_fk()`의 EE 위치와 translation이 오차 0.0으로 완전 일치. 6D 회전도 정규직교(norm 1.0, 내적 1e-8).

### v2 (`scripts/so100_to_bridge_v2.py`) — delta pose
Cosmos3 기술보고서 확인 결과 규약이 다름:
> "end-effector flange-pose **deltas** as effector poses, continuous gripper **open/close** values as grasp states"

- pose: 프레임 간 delta (`frame="local"`: 이전 EE 좌표계 / `frame="base"`: 로봇 base 좌표계)
- gripper: **클립별 min-max [0,1] 정규화** (데이터셋마다 캘리브레이션이 달라 전역 상수 불가 — 어떤 에피소드는 0~0.85, 어떤 건 0~48)

**v1의 문제**: gripper가 관절각 원값(train 최대 119.4)이라 translation(±0.45)·rotation(±1)을 20~50배 압도.
**v2 실측**: translation delta 0.76~1.41cm/프레임, gripper 100% [0,1] 범위.

---

## 3. 학습

`scripts/finetune_cosmos3_nano.py` — 파이프라인 내부 헬퍼(`_prepare_text_segment`, `_prepare_vision_segment`, `_prepare_action_segment`, `_encode_video`)를 재사용해 학습용 forward를 직접 구성. `Cosmos3OmniPipeline.__call__`은 `@torch.no_grad()`라 재사용 불가.

**LoRA 타겟**: `add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out` (attention) + **`mlp_moe_gen.{gate,up,down}_proj`** (generation 전용 MLP)
**추가 학습**: `action_proj_in/out`, `action_modality_embed` (lr 5배)
**loss**: vision 토큰의 flow-matching MSE만. forward_dynamics에서는 action 전 구간이 condition이라 action loss가 없음(`action_mse_loss_indexes`가 항상 빈 텐서).

### 학습 변형

| 버전 | rank | action 표현 | step | episode | 비고 |
|---|---|---|---|---|---|
| v1 | 16 | absolute | 20,000 | 4,000 | 최초 성공 |
| v2 | 16 | absolute | 중단 | — | action 보조 loss 시도 → 무효 판명 후 중단 |
| v3 | 16 | absolute | +30,000 (v1 resume) | 11,132 | rank 16 한계 확인 |
| **v4** | **32** | absolute | 40,000 | 11,132 | **현 최고** |
| v5_local | 32 | delta (EE 좌표계) | 40,000 | 11,132 | |
| v5_base | 32 | delta (base 좌표계) | 40,000 | 11,132 | Action 최저 |
| v6 | 64 | absolute | 40,000 | 11,132 | 용량 포화 확인용 (진행 중) |

### 메모리 이슈 해결
- gradient checkpointing은 `transformer.enable_gradient_checkpointing()` 호출 필요 (`gradient_checkpointing = True` 속성 대입은 무효 — `_gradient_checkpointing_func`가 None으로 남아 `TypeError`)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 44GB(A6000)에서는 OOM, 96GB(RTX PRO 6000) 필요

---

## 4. 평가 체계

### 문제: 기존 로컬 지표가 무효였음
`scripts/eval_cosmos3.py`가 `TARGET_H, TARGET_W = 512, 320`(세로형)이었으나 공식 채점기 `make_submission_csv.py`는 `320×512`(가로형). 480×640 가로 영상이 세로 캔버스에 찌부러져 측정값이 왜곡됨.

동일 영상 측정: **버그 상태 1.29 → 수정 후 0.6436 → 공식 CSV 기준 실제 0.4165**.
→ 2026-08-02 이전의 로컬 Action MAE 수치는 전부 신뢰 불가. 수정 완료.

### 새 지표 (`scripts/eval_holdout.py`)
eval 216샘플은 GT 영상이 없어 로컬 채점 불가 → **학습에 쓰지 않은 train 에피소드**로 생성 후 실제 GT 영상과 비교.
- 특징 추출: 공식 `feature_csv_utils`의 `extract_dino_features` / `extract_video_features` 그대로
- 전처리: 공식과 동일한 320×512, pad=True
- 홀드아웃 선정: 학습 코드의 표집(`rng(0).choice(sorted, N)`)을 재현해 **여집합에서만** 선택 (데이터 오염 차단)
- **`--action-repr`을 학습과 반드시 일치**시켜야 함 (불일치 시 결과 무의미)

### 해석 기준점 (`scripts/eval_holdout_references.py`)

| 기준 | DINO | Video | 평균 |
|---|---|---|---|
| 완벽 (GT=GT) | 0.0000 | 0.0000 | 0.0000 |
| 정지영상 (첫 프레임 반복) | 0.1195 | 0.0982 | 0.1089 |
| 무관 영상 (다른 에피소드) | 0.7010 | 0.3175 | 0.5093 |

### 모델별 결과 (n=40, 동일 홀드아웃)

| 모델 | DINO | Video | 평균 | 리더보드 |
|---|---|---|---|---|
| base (파인튜닝 전) | 0.1496 | 0.0850 | 0.1173 | — |
| v1 (rank16) | 0.1543 | 0.0541 | 0.1042 | 0.2934 |
| v3 (rank16, resume) | 0.1299 | 0.0520 | 0.0909 | — |
| **v4 (rank32)** | 0.1117 | 0.0569 | **0.0843** | **0.26809** |
| v5_local (delta EE) | 0.1102 | 0.0523 | 0.0813 | — |
| v5_base (delta base) | 0.1125 | 0.0508 | 0.0817 | — |

**홀드아웃 지표가 리더보드와 같은 방향으로 움직임을 2회 검증** (v1→v4 모델 개선, gs 3.5→5.0 설정 개선 모두 양쪽이 함께 개선).

### Action MAE는 eval 샘플에서만 측정
train 도메인에서는 `action_extractor`가 GT와 **상관 0.013**(무신호)이므로 홀드아웃에서 재면 안 됨.
실제 eval 216샘플(GT action 보유)에서 공식 경로로 측정: `scripts/sweep_inference.py`.

| 모델 (gs=5.0) | Action MAE |
|---|---|
| v4 | 0.3797 |
| v5_local | 0.4086 |
| **v5_base** | **0.3266** |

Action MAE vs 리더보드 상관 **+0.820** (표본 7). 홀드아웃(DINO/Video)이 구분하지 못한 v5_local vs v5_base를 Action이 명확히 갈랐음.

---

## 5. 추론 설정 탐색 (`scripts/sweep_inference.py`)

v4 기준, steps=50:

| guidance | DINO | Video | Action |
|---|---|---|---|
| 2.0 | 0.1243 | 0.0389 | **0.2377** |
| 3.5 | 0.1307 | 0.0365 | 0.2540 |
| **5.0** | **0.1204** | 0.0342 | 0.2690 |
| 6.0 | 0.1224 | **0.0325** | 0.2878 |

**최적 guidance는 모델마다 다름** — v1에서는 3.5가 균형점이었으나 v4에서는 5.0. 모델을 바꾸면 재탐색 필요.

### fps 검증 (가설 기각)
학습은 `fps=6.0`, 추론 기본값은 `24.0`으로 불일치가 있었으나, 추론을 6.0으로 맞추면 **오히려 DINO/Video가 나빠짐**(0.1112→0.1241, 0.0396→0.0443). 사전학습 `base_fps=24`의 prior와 충돌하는 것으로 보임. 현재 24.0 유지.

### 개선 기여도
| 변경 | 점수 변화 |
|---|---|
| 모델 (v1 → v4) | −0.0243 (8.2%) |
| 추론 설정 (gs 3.5 → 5.0) | −0.0011 (0.4%) |

**모델 개선이 추론 튜닝보다 20배 이상 효과적.**

---

## 6. 폐기한 접근: action-following 보조 loss

velocity 예측 → x0 역산 → VAE 디코드 → `action_extractor` → GT action과 MSE를 보조 loss로 추가하는 방식을 구현했으나 **무효로 판명**:

- 실제 train 영상을 그대로 추출기에 넣어도 MSE 2.4~3.2 (상수 0 예측 0.81~1.31보다 나쁨)
- GT와 평균 상관 **0.013**, 편향 제거 후에도 무정보 기준 미달
- σ, 디코드 해상도, 전처리 방향, 영상-action 정렬(seek vs 순차, 픽셀차 0.00) 전부 원인 아님으로 배제
- eval과 시각적으로 유사한 train 영상(isadev)만 골라도 상관 −0.193

→ v2 학습 중단. 검증 스크립트는 `scripts/diagnose_action_loss.py`, `check_extractor_*.py`에 보존.

**주의**: 이 결과는 **train 도메인 한정**. eval 도메인에서는 Action 지표가 유효함(상관 +0.820).

---

## 7. 산출물

### 스크립트
| 파일 | 역할 |
|---|---|
| `scripts/so100_to_bridge.py` | SO-100 → bridge 10D (절대 pose) |
| `scripts/so100_to_bridge_v2.py` | SO-100 → bridge 10D (delta + gripper 정규화) |
| `scripts/finetune_cosmos3_nano.py` | 학습 (`--action-repr`, `--resume`, `--rank`) |
| `scripts/finetune_cosmos3_nano_v{3,4,5_local,5_base,6_r64}.sbatch` | 변형별 SLURM |
| `scripts/infer_cosmos3_nano.py` | 추론 (`--guidance`, `--action-repr`) |
| `scripts/eval_holdout.py` | **홀드아웃 평가 (주 지표)** |
| `scripts/eval_holdout_references.py` | 해석 기준점 |
| `scripts/sweep_inference.py` | 추론 설정 탐색 + eval Action MAE |
| `scripts/verify_action_conditioning.py` | action conditioning 실효성 검증 |
| `scripts/run_v5base_timed.sh` | 추론 소요시간 측정 |

### 체크포인트 / 제출물
- `checkpoints/cosmos3_nano_v{1,3,4,5_local,5_base,6_r64}/`
- `submission_kit/submission_cosmos3_nano_v1.csv`, `_v4_gs35.csv`, `_v4_gs50.csv`
- `submission_kit/input_videos_cosmos3_nano_v1_full/`, `input_videos_v4_gs35/`, `input_videos_v4_gs50/`

---

## 8. 실행 환경 주의사항

- **gpu-113** (RTX PRO 6000 Blackwell 96GB × 9): 다른 사용자가 상시 5장 사용. `squeue -u`만 보면 여유를 오판하므로 `squeue -w gpu-113`로 확인.
- **gpu-109** (A6000 Ada 44GB × 8): 대체로 idle하나 **cuDNN 버전 충돌로 GRU 모델 로드 불가**. `make_submission_csv.py`, `eval_cosmos3.py`, `sweep_inference.py`(action_extractor 사용)는 여기서 실행 금지. 학습과 `eval_holdout.py`는 정상.
- SLURM 계산 노드는 `/tmp` 아래 경로를 공유하지 않음. 스크립트는 프로젝트 디렉토리에 둘 것.
- `squeue`는 job 이름을 8자로 자름. 감시 루프는 **job ID**로 조건을 걸 것.

---

## 9. 남은 과제

1. **v5_base(delta) 제출 검증** — 홀드아웃 0.0817, Action 0.3266으로 v4보다 두 지표 모두 우세. 216샘플 생성 진행 중.
2. **v6(rank 64)** — rank 16→32에서 DINO가 뚜렷이 개선됐으므로 포화점 확인 필요.
3. **추론 시간** — gpu-113 실측 18.3초/샘플 × 216 ≈ **66분**. 챌린지 규정(1시간)을 초과할 가능성. steps 조정 검토 필요. (단 규정 자체가 리포지토리 공식 파일에 근거가 없어 미확인 상태)
4. **DINO(시각 충실도)** — v4 기준 0.1117로, 정지영상 기준선(0.1195)을 겨우 넘긴 수준. 가장 큰 개선 여지.
