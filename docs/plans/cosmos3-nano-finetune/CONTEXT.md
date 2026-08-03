# Cosmos3-Nano 파인튜닝 맥락 노트 (신규)

## 결정 기록

| 결정 사항 | 선택지 | 최종 선택 | 이유 |
|---|---|---|---|
| 계획 수립 방식 | 과거 v3~v7 이력 기반 이어가기 / 완전 재조사 후 신규 수립 | **완전 재조사 후 신규 수립** | 사용자가 "이전 작업 로그에 영향받고 싶지 않다"고 명시 |
| action conditioning 방식 | 과거처럼 FK만 별도 구현 / Cosmos3 네이티브 `forward_dynamics` API 사용 | 네이티브 API | 소스(`pipeline_cosmos3_omni.py`) 확인 결과 chunk_size/raw_actions 인터페이스가 이미 존재, 재발명 불필요 |
| Phase 0 (사전검증) 필요 여부 | 생략하고 바로 학습 / 먼저 검증 | 먼저 검증 | 프로젝트 CLAUDE.md 규칙(신규 접근법 전 실측 필수) + action conditioning 실효성이 한 번도 측정된 적 없음 |
| 실행 GPU | gpu-109(idle) / gpu-106,108,110-112(mix) / gpu-113(v7 사용중) | gpu-109 | sinfo 실측 결과 유일한 완전 idle 노드 |
| 정규화 기준 | train 통계만 / eval 통계만 / 둘 다 비교 | 둘 다 비교 (Phase 3) | 채점기 기본값은 train 통계(`make_submission_csv.py` 확인)이나 실제 eval 분포와 어긋남을 발견해 둘 다 측정 필요 |

## 참조 자료 (직접 확인한 원본 파일)

### 대회 규칙 / 제출
- `/home1/sota/inha2026/README.md`, `/home1/sota/inha2026/baseline/README.md` — 공식 설명, git 커밋 1개(`3772b43`)로 원본 확인
- `/home1/sota/inha2026/baseline/challenge_kit/scripts/eval/feature_csv_utils.py` — DINO/VideoFeature/Action 세 컴포넌트 계산 로직 (합산 가중치 없음)
- `/home1/sota/inha2026/submission_kit/make_submission_csv.py` — 제출 CSV 생성, 16프레임 강제 검증
- `/home1/sota/inha2026/submission_kit/action_extractor.py` — Action MAE 계산용 3D CNN+BiGRU
- `/home1/sota/inha2026/baseline/challenge_kit/scripts/so100_fk.py`, `src/ldwma/datasets/lerobot_so100.py` — 공식 FK/데이터로더 참조 구현

### 데이터
- `data/train/<ns>/<task>/data/chunk-*/episode_*.parquet` (action, observation.state, 6D degree)
- `data/train/<ns>/<task>/meta/info.json` (fps, action.names, robot_type)
- `data/eval/actions/sample_NNNNNN.npy` (16,6) float32, `data/eval/images/sample_NNNNNN.png` (640×480)
- `data/train/so100_action_statistics.json` (count=974661), `data/eval/eval_action_statistics.json` (count=3456=216×16)

### Cosmos3-Nano 아키텍처 (diffusers 0.39.0, 직접 소스 확인)
- `diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py` — `CosmosActionCondition`, `_EMBODIMENT_TO_DOMAIN_ID`, `_EMBODIMENT_TO_RAW_ACTION_DIM`, `_ACTION_RESOLUTION_BINS`
- `diffusers/models/transformers/transformer_cosmos3.py` — `Cosmos3OmniTransformer`, `DomainAwareLinear`, `Cosmos3PackedMoTAttention`, `Cosmos3VLTextMoTDecoderLayer`
- `/home1/sota/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/.../transformer/config.json` — hidden_size=4096, layers=36, heads=32/8(GQA), action_dim=64, num_embodiment_domains=32, video_temporal_causal=false

## 주요 제약 조건

### 확인된 것
- 제출 mp4는 정확히 16프레임이어야 함 (해상도는 자동 패딩/리사이즈되어 엄격하지 않음)
- Cosmos3 forward_dynamics: `target_frames = chunk_size + 1` → 16프레임 출력 시 `chunk_size=15`
- `bridge_orig_lerobot` 도메인 raw_action_dim=10 (9D pose + 1D gripper), domain id=7
- `action_proj_in/out`은 `DomainAwareLinear`(32도메인 배치 구조)라 표준 LoRA로 직접 타겟 불가, domain-7 슬라이스만 직접 학습 가능
- generation 경로 LoRA 후보: `add_q_proj/k_proj/v_proj`, `to_add_out`, `mlp_moe_gen.{gate_proj,up_proj,down_proj}`
- train action 통계(wrist_roll mean=-29.4)와 eval action 통계(wrist_roll mean=+64.4) 부호 반전 — 원본 JSON 실측으로 재확인

### 미확인 (승인 전 사용자 재확인 필요)
- **점수 산식 가중치** (참가자 노트북에만 "0.3×DINO+0.3×VideoFeature+0.4×Action" 기재, 공식 파일 근거 없음)
- **시간/하드웨어 제약** (참가자 노트북에만 "학습 4일/추론 1시간, RTX PRO 6000 96GB" 기재, 공식 파일 근거 없음)
- eval_captions.json은 비공식(참가자가 Qwen3-VL로 생성) — 원본 대회가 캡션을 제공하는지 불명

## 사용자 요구사항 원문

> "/home1/sota/inha2026 이 프로젝트 /home1/sota/inha2026/baseline, /home1/sota/inha2026/data 잘 확인하고 cosmos3 nano로 파인튜닝해줘"
> (v7이 완료 임박임을 보고한 후) "v7과 무관하게 즉시 병렬로 새 실험 설계"
> (v3~v7 이력 기반 CONTEXT/PLAN 수정 시도를 거부하며) "이전 작업 로그에 영향을 받고 싶지 않은데, 그냥 데이터부터 아키텍처, 대회 규칙까지 세부적으로 다 분석하고 새로 계획했으면 좋겠어"
