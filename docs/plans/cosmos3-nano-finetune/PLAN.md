# Cosmos3-Nano 파인튜닝 계획서 (신규, 과거 v3~v7 이력 미참조)

## 개요

- **목적**: SO-100 로봇팔 첫 프레임 이미지 + 16스텝 action 시퀀스로 미래 행동 mp4를 생성하는 모델을, `nvidia/Cosmos3-Nano`의 **내장 forward_dynamics action conditioning**을 이용해 파인튜닝
- **범위**: 데이터 변환 파이프라인, action conditioning 검증, LoRA 학습 스크립트, 평가, (조건부) 제출
- **예상 소요**: 5 Phase (사전검증 → 데이터파이프라인 → 학습 → 평가 → 제출)
- **원칙**: 이 계획은 baseline/data/모델 소스코드를 직접 재조사해 처음부터 작성했으며, 과거 cosmos3-lora v3~v7 실험의 설계 판단을 전제로 삼지 않는다.

---

## 현재 상태 분석 (원본 파일/소스코드 직접 확인)

### 태스크 & 데이터 (data/, baseline/challenge_kit/ 실측)
- 입력: 640×480 PNG 첫 프레임 + `(16, 6)` float32 action 시퀀스. **action은 절대 관절각도(degree)** — `action[t] ≈ observation.state[t+1]` (MAE 0.3~1.3) 로 직접 확인, delta 아님
- action 6차원 순서: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` (`meta/info.json` 실측)
- train: 56 네임스페이스 / 128 데이터셋 / 11,132 episode / 총 프레임 약 102만. **fps가 6 또는 10으로 혼재**, 카메라 키도 대부분 `observation.images.image`이나 일부 다름
- eval: 정확히 216개 (`actions/`, `images/` 파일 수 카운트로 검증), GT 영상 없음
- **train/eval 정규화 통계 불일치 확인**: `so100_action_statistics.json`(train, wrist_roll mean=-29.4) vs `eval_action_statistics.json`(eval, wrist_roll mean=+64.4) — 부호까지 반대. 이 축은 단순 정규화로는 못 맞춤 → 각도 wrap-around(예: -180~180 vs 0~360) 가능성 의심, Phase 0에서 원인 확인 필요
- `eval_captions.json`은 대회 공식 배포물이 아니라 참가자가 Qwen3-VL로 생성한 추정치 — 캡션 조건부 접근 시 신뢰 불가

### 제출 규격 (submission_kit/ 실측)
- `submission_kit/input_videos/sample_000000.mp4` ~ `sample_000215.mp4` (216개, 파일명=sample_id)
- `make_submission_csv.py` 강제 검증: **정확히 16프레임** (`temporal_length` 불일치 시 예외 발생). 해상도는 내부 자동 패딩/리사이즈되어 엄격하지 않음
- CSV 컬럼: `sample_id, feature_component, feature_json` — `feature_component` ∈ {DINO, Video Feature, Action} Component
- Action MAE는 `submission_kit/action_extractor.py`(3D CNN + BiGRU)가 생성 영상에서 예측한 action과, **train 통계로 정규화한** eval GT action 간 MAE (`make_submission_csv.py` 기본값 `--action-stats-path .../so100_action_statistics.json`)

### ⚠️ 미확인 (리포지토리 내 근거 없음 — 반드시 대회 공지 원문 재확인 필요)
- 최종 점수 합산 가중치(예: "0.3×DINO+0.3×VideoFeature+0.4×Action")는 공식 README/challenge_kit/submission_kit 어디에도 없음. 참가자가 작성한 노트북/과거 plan 문서에만 존재하며 1차 출처 불명
- "추론 1시간 이내, RTX PRO 6000 96GB, 학습 4일 이내" 제약도 동일하게 공식 파일에서 근거를 찾지 못함
- → **이 두 전제가 틀리면 Phase 2 이후 우선순위(속도 vs 품질)가 달라짐. 계획 승인 전 대회 공지 페이지에서 재확인 요청**

### Cosmos3-Nano 아키텍처 (diffusers 0.39.0 소스 + config.json 직접 확인)
- `Cosmos3OmniTransformer`: hidden_size=4096, 36 layers, 32 attn heads / KV heads 8 (GQA), MoT(Mixture-of-Transformers) 구조로 understanding(`to_q/k/v`) / generation(`add_q/k/v_proj`, `to_add_out`, `mlp_moe_gen`) 경로 분리
- **`video_temporal_causal=false`** — DreamZero(blockwise causal, 16프레임에서 action↔DiT 불가)와 달리 **비인과적 어텐션**이라 16프레임에서 action conditioning이 구조적으로 가능함을 소스로 확인
- action conditioning: `CosmosActionCondition(mode="forward_dynamics", ...)` 네이티브 지원. `raw_actions: [T, raw_action_dim]` → `action_proj_in`(DomainAwareLinear, 32도메인별 독립 가중치) → `+ action_modality_embed`(도메인 공유, 4096차원 벡터) → DiT
- `bridge_orig_lerobot` 도메인(id=7): `raw_action_dim=10` = 9D effector pose(3D translation + 6D rotation) + 1D gripper → SO-100 6D joint angle을 FK로 변환 필요
- **프레임-청크 관계**: `target_frames = chunk_size + 1`. 16프레임 출력 필요 → **chunk_size=15**, `raw_actions`는 15개 transition (16개 eval action 중 15개 사용 또는 pairwise 변환)
- `action_proj_in/out`은 `nn.Linear`가 아닌 배치 구조라 표준 PEFT LoRA로 직접 타겟 불가 → domain-7 슬라이스만 별도 파라미터로 직접 학습 필요 (fine-tune, LoRA 아님)
- generation 경로 LoRA 후보: `add_q_proj/add_k_proj/add_v_proj/to_add_out` (attention) + **`mlp_moe_gen.{gate_proj,up_proj,down_proj}`** (지금까지 미시도, 소스 확인 결과 generation 전용 MLP로 LoRA 타겟 가능)

---

## 구현 계획

### Phase 0: 사전 검증 (필수, 코드 작성 전 — 프로젝트 CLAUDE.md 규칙)
1. `action_proj_in`/`action_modality_embed`가 실제로 gradient를 받는지, `bridge_orig_lerobot`(domain 7) 슬라이스가 로딩되는지 실측 (`state_dict` 키/shape 직접 출력)
2. 동일 첫 프레임 + (GT action / zero action / random action) 3조건으로 forward_dynamics 추론 → 생성 영상 픽셀 MSE 비교 → conditioning이 실제로 결과를 바꾸는지 정량 확인
3. wrist_roll train/eval 부호 반전 원인 확인: 원본 raw 각도 범위를 직접 비교해 wrap-around(모듈로 360) 여부 판단
4. 위 결과에 따라 Phase 2 학습 설계 확정 (conditioning이 약하면 domain-7 proj 학습 비중을 높이고, wrap-around가 원인이면 정규화 전 각도 unwrap 적용)

### Phase 1: 데이터 파이프라인
- SO-100 6D 관절각도 → FK → bridge 10D(pose 9D + gripper 1D) 변환 함수 신규 작성 (챌린지킷 공식 FK 대조 필수)
- 16-action 시퀀스 → 15-transition `raw_actions` 매핑 규칙 확정 (pipeline의 `chunk_size+1=target_frames` 제약에 맞춤)
- train 데이터에서 SO-100→bridge 변환 후 실제 정규화 통계(mean/std 또는 quantile) 추출
- resolution_tier 선택: 640×480 입력 aspect에 가장 가까운 tier를 실측 비교로 결정 (다운스케일만 허용, 업스케일 금지 제약 고려)

### Phase 2: LoRA 파인튜닝
- Phase 0 결과에 따라 두 갈래 중 선택:
  - conditioning이 유효하면: attention LoRA(rank 16~32) + `mlp_moe_gen` LoRA 확장 + domain-7 action_proj 저강도 유지
  - conditioning이 약하면: domain-7 action_proj / action_modality_embed을 높은 lr로 우선 학습, LoRA는 보조
- gpu-109 (idle, A6000 Ada × 8) 사용 예정 — 실행 직전 재확인
- 체크포인트/로그: `checkpoints/cosmos3_nano_v1/`, `logs/cosmos3_nano_v1_<jobid>.log` (기존 `cosmos3_lora_*`, `cosmos3_fd_lora_*` 네이밍과 구분)

### Phase 3: 평가
- 소수 샘플(5개)로 quick eval → Action MAE 확인
- `eval_action_statistics.json`(eval 통계) vs `so100_action_statistics.json`(train 통계, 채점기 기본값) 두 기준 모두로 측정해 실제 채점 조건과 맞춰봄

### Phase 4: 제출 (조건부, 품질 확보 시)
- 216샘플 전체 추론 (정확히 16프레임 강제 확인)
- `make_submission_csv.py` 실행 → 데이콘 업로드

---

## 기술 선택 이유

| 선택 | 이유 |
|---|---|
| `CosmosActionCondition(mode="forward_dynamics")` 네이티브 API 사용 | 과거 DreamZero의 blockwise causal 제약(16프레임 불가) 문제가 Cosmos3엔 없음(`video_temporal_causal=false`, 소스 확인). FK 변환 로직 재사용 이상 가치 있음 |
| Phase 0을 최우선 필수로 배치 | action conditioning이 실제 작동하는지 한 번도 실측된 적 없는 상태에서 대규모 학습에 GPU를 투입하는 것은 낭비 위험 |
| `mlp_moe_gen` LoRA 확장 후보 추가 | 소스 확인 결과 generation 전용 경로이면서 지금까지 아무도 건드리지 않은 파라미터 — 저위험 확장 |
| gpu-109 사용 | sinfo 확인 결과 유일하게 완전 idle한 노드 (A6000 Ada × 8) |

## 리스크

| 리스크 | 확률 | 대응 |
|---|---|---|
| 점수 가중치/시간제약이 리포지토리 기록과 다름 | 중간 | **계획 승인 전 대회 공지 원문 재확인 요청** (아래 승인 요청 시 안내) |
| wrist_roll train/eval 분포 반전이 wrap-around가 아닌 다른 원인 | 중간 | Phase 0에서 원본 각도 히스토그램 직접 확인 후 대응 |
| action conditioning이 Phase 0에서 "무의미"로 판정 | 낮음~중간 | domain-7 proj 중심 학습으로 전환 (계획에 이미 분기 반영) |
| resolution_tier 선택 오류로 화질/conditioning 성능 저하 | 낮음 | Phase 1에서 실측 비교 후 확정 |
