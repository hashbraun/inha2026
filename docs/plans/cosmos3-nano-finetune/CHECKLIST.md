# Cosmos3-Nano 파인튜닝 체크리스트

## 진행 상태: 🟡 진행 중 (리더보드 0.26809 달성, 추가 실험 진행 중)

> 실제 구현 내용과 측정값은 [IMPLEMENTATION.md](IMPLEMENTATION.md) 참조.

## 작업 목록

- [x] Phase 0: 사전 검증
  - [x] `action_proj_in`/`action_modality_embed`/domain-7 state_dict 실측 → domain 7 사전학습 확인(bias norm 1.48)
  - [x] GT/zero/random action 3조건 추론 비교 → **conditioning 유효** (GT-zero MSE > random-zero MSE, 3/3 샘플)
  - [x] wrist_roll train/eval 부호 반전 원인 → wrap-around 아님, eval이 특정 태스크 편중된 실제 분포 차이
  - [x] 프레임수 공식 확인 → `chunk_size=17` → 17프레임 → `[:16]`

- [x] Phase 1: 데이터 파이프라인
  - [x] SO-100 → bridge 10D FK 변환 (`so100_to_bridge.py`) + 공식 FK와 오차 0.0 대조 검증
  - [x] delta 변환 (`so100_to_bridge_v2.py`) — Cosmos3 공식 규약 반영
  - [x] gripper 정규화 (클립별 min-max) — 전역 상수는 데이터셋 캘리브레이션 차이로 부적합
  - [x] resolution_tier 확정 (480, 네이티브 해상도 보존)

- [x] Phase 2: 학습
  - [x] `finetune_cosmos3_nano.py` 작성 (파이프라인 헬퍼 재사용 + flow-matching loss)
  - [x] v1 (rank16) → 리더보드 0.2934
  - [x] v3 (rank16 resume) / **v4 (rank32)** → rank 32 우세 확인
  - [x] v5_local / v5_base (delta 표현) → delta가 근소 우세, base 좌표계가 Action에서 명확히 우세
  - [ ] v6 (rank 64) — 진행 중, 용량 포화점 확인

- [x] Phase 3: 평가 체계 재정립
  - [x] `eval_cosmos3.py` 방향 버그 수정 (512×320 → 320×512)
  - [x] `eval_holdout.py` 신규 작성 (train 홀드아웃 + 공식 특징 추출기)
  - [x] 해석 기준점 산출 (정지영상 0.1089, 무관 영상 0.5093)
  - [x] 홀드아웃 지표가 리더보드와 같은 방향임을 2회 검증
  - [x] Action MAE는 eval 샘플에서만 측정하도록 분리 (train 도메인 무신호 확인)

- [x] Phase 4: 추론 설정 탐색
  - [x] guidance 탐색 (v1 기준 3.5, v4 기준 5.0 — 모델마다 다름)
  - [x] steps 탐색 (35 vs 50 vs 70)
  - [x] fps 가설 검증 → 기각 (학습값 6.0으로 맞추면 오히려 악화)

- [x] Phase 5: 제출
  - [x] v1 → 0.2934
  - [x] v4 gs=3.5 → 0.26915
  - [x] **v4 gs=5.0 → 0.26809 (현 최고)**
  - [ ] v5_base gs=5.0 — 216샘플 생성 진행 중

## 폐기한 접근
- [x] action-following 보조 loss (v2) — train 도메인 추출기가 무신호(상관 0.013)로 확인되어 중단

## 남은 과제
- [ ] v5_base 제출 및 검증
- [ ] v6(rank64) 결과 확인
- [ ] 추론 시간 66분 → 1시간 제약 대응 (steps 조정 검토)
- [ ] DINO(시각 충실도) 개선 — 최대 약점

## 변경 로그

| 시간 | 내용 |
|---|---|
| 2026-08-02 | 계획 수립, Phase 0~2 구현, v1 학습 및 제출(0.2934) |
| 2026-08-02 | 평가 체계 재정립 (방향 버그 수정, 홀드아웃 지표 신설) |
| 2026-08-02 | v3/v4 학습, v4 제출(0.26915 → 0.26809) |
| 2026-08-03 | delta 표현 구현(공식 문서 근거), v5 학습 및 평가 |
