# Handoff — 2026-08-19 (auto loop v2 continuous mode)

## ⚡ 다음 세션 즉시 재개 프롬프트

```
inha2026 D-day 임박. 자동화 v2 실행 중 (사용자 KILL까지 무한 loop).
아래 순서 확인:
1. /home1/sota/inha2026/HANDOFF_2026-08-19_root.md (이 파일)
2. squeue -u sota (활성 job)
3. tail -50 /home1/sota/inha2026/logs/fresh/auto_loop_v2.log
4. cat /home1/sota/inha2026/logs/fresh/auto_loop_v2_state.json | head -80
5. ls /home1/sota/inha2026/docs/plans/fresh_restart/codex_pivot_tick*.txt (최신 codex 자문)
6. 최신 CSV: ls -lt /home1/sota/inha2026/submission_kit/fresh/submission_b4_sw*.csv
```

## 대회 확정 사실
- Dacon 236736, 마감 **2026-08-20 18:00 KST**
- 채점 `0.3·(1-DINO_cos) + 0.3·(1-Video_cos R3D-18) + 0.4·Action_MAE` (lower better)

## 실측 확정 자산
| CSV | Action mean | 실측 LB | 상태 |
|---|---|---|---|
| **v5b_ANCHORED_soft.csv** | 0.3886 | **0.24524** | 안전본 필수 제출 |
| **submission_b4_step6000.csv** | 0.3798 | **0.2407** | 도전본 (안전본 -0.0045 개선) |
| e_select_ANCH.csv | 0.3625 | 0.263 | 폐기 (visual 손상) |
| tier1_seed1 | ? | 0.270 | 참고 |
| tto_v5b_ANCH_dino16.csv | 0.078 | 0.04 | **규정 위반, 절대 지정 금지** |

## 실패 이력 (재시도 금지)
- **B4v2** (constant weight 0.15): LB 0.2508 (visual +0.010 손상)
- **Z6-B4 hybrid step15000**: LB 0.2499 (82M adapter + 장기학습 visual 손상)
- **Z5 shuffled ranking**: 예상 0.2478
- **Z6-lite step2000**: 예상 0.2415 (B4 미달)
- **NAP step8000**: 예상 0.2418 (LoRA freeze 표현력 부족)
- **Cosmos-Predict2.5**: G5 falsification FAIL 2회 (backbone action-agnostic)

## Auto loop v2 (사용자 KILL까지 무한 실행)

### 실행 정보
- PID: **794036**
- Log: `/home1/sota/inha2026/logs/fresh/auto_loop_v2.log`
- State: `/home1/sota/inha2026/logs/fresh/auto_loop_v2_state.json`
- Tick: 10분
- **종료 조건**: 사용자 KILL만 (`kill 794036`)

### 자동 동작
1. **각 candidate 학습 완료** → 여러 ckpt (4k/6k/8k/10k/12k) auto inference
2. **Inference 완료** → CSV Action mean 계산
3. **판정**:
   - `Action < 0.35`: **big_success** → 사용자 실측 slot 요청 (loop 계속)
   - `Action < 0.3798`: marginal (loop 계속)
   - `Action ≥ 0.3798`: failed (loop 계속)
4. **각 pivot event** → **Codex 자문 자동 launch** (로그+state embed, `--sandbox danger-full-access`)
   - 파일: `docs/plans/fresh_restart/codex_pivot_tick{N}.txt`
5. **모든 candidate settled** → 다음 후보 자동 queue launch (seed reproduction 등, 무한 continue)

### 활성 학습 chain (Blackwell sequential)
```
30749: B4 sweep w=0.025, 12k [RUNNING]
 → 30757: B4 sweep w=0.05, 12k (Codex 권장 성공 basin 재현)
 → 30758: B4 sweep w=0.075, 12k (조건부)
 → auto: b4_w005_seed43 (queue, 다음 후보)
 → auto: b4_w005_seed44 (queue)
```

### B4 원본 세부 ckpt sweep (A6000_ada, 병렬)
- 30759: step 500 inference
- 30760: step 1000 inference
- 30761: step 1500 inference

## Codex 최신 자문 (`codex_log_review.txt`) — 로그 분석 반영

**한 줄**: "30749 계속 + 4k/6k/8k 먼저 평가 + w=0.05 재현으로 전환. 12k endpoint·magweight·새 hybrid보다 checkpoint selection이 기대값·안전성 최고."

**핵심 통찰**:
1. **B4v2 실증**: auxiliary가 flow objective 압도 → visual 파괴
2. **Z6-B4 hybrid preserve loss**는 flow의 0.05%로 사실상 무효
3. **Action mean 기반 예상 LB 안전영역 밖에서 무효** (B4v2, hybrid 두 번 예측 실패)
4. **Training loss 최저점 ≠ LB 최적점** → checkpoint selection이 훨씬 중요
5. Auto loop v1 bug: pending state ckpt_dir 없어 KeyError (v2에서 수정)

**적용 완료**:
- ✅ Magweight (30756) cancelled
- ✅ Chain swap (30757 w=0.05 → 30758 w=0.075, Codex 권장 우선순위)
- ✅ B4 원본 세부 ckpt sweep (30759-30761)
- ✅ auto_loop v1 kill → v2로 교체 (codex 자문 자동 통합)

## Pivot Queue (settled 후 자동 launch)

### 1단계 (실행 중)
- B4 sweep chain: w=0.025 → w=0.05 → w=0.075
- B4 원본 세부 (step 500/1000/1500 inference)

### 2단계 (queue, auto launch)
- **b4_w005_seed43**: `b4_seed_repro_v2.sbatch`, w=0.05 seed 43 재현
- **b4_w005_seed44**: `b4_seed_repro_v3.sbatch`, w=0.05 seed 44 재현
- Codex 조언: "성공한 정확한 w=0.05 step6000 재현 - variance 확인"

### 3단계 (Codex 조언 시 추가 launch)
- Action-only mild weighting (clip [0.75, 1.5], mean-normalized) — Codex 재설계 안전 형태
- Large-motion stratified sampling
- B4 + ANCH prediction blend (매우 보수적, 1개만)

### 4단계 (마감 후로 미룸)
- E/invdyn ensemble 재학습
- 새 backbone (Wan2.2, DreamZero 14B)
- RAFT/DPO/RL

## 규정
- 금지: submission_kit action_extractor(GRU), R3D-18, DINOv2 학습/selection/mp4 수정 사용
- 허용: 외부 pretrained + 자체 clean-room predictor (E invdyn)
- **최종 지정 절대 금지**: TTO CSV 3개 (실격)

## 자원
- Blackwell gpu-113: `inha2026_bw` env (torch 2.15dev), LD_LIBRARY_PATH nvidia sub-package 필수
- A6000_ada gpu-108/109/110/112: `inha2026` env
- **B4 계열은 Blackwell 필수** (A6000 44GB OOM)

## 다음 세션 대응 시나리오

### A. Auto loop 정상 진행 중
- 최신 codex_pivot_tick*.txt 읽고 조언 확인
- 사용자 결정 필요한 이벤트 (big_success)만 응답

### B. Auto loop 프로세스 죽음
```
nohup /home1/sota/anaconda3/envs/inha2026/bin/python \
  /home1/sota/inha2026/scripts/track_z5/auto_loop_v2.py \
  > /home1/sota/inha2026/logs/fresh/auto_loop_v2_stdout.log 2>&1 &
```
State 파일 그대로 이어서 진행 (json 유지)

### C. Big success (Action < 0.35) 발견
- 해당 CSV 실측 slot 사용 여부 사용자 결정 요청
- 실측 결과 좋으면 새 도전본 지정

### D. 마감 임박 (< 10h)
- 신규 학습 launch 금지 (Codex 강력 권장)
- `kill 794036` (auto loop 정지)
- 진행 중 학습 kill 검토
- 검증/제출 buffer 확보 최우선

### E. 최종 제출 (D-day)
- 안전본: `submission_ANCHORED_soft.csv` (0.24524)
- 도전본: 최선 실측 CSV (없으면 `submission_b4_step6000.csv` 0.2407)
- TTO CSV 3개 절대 지정 금지

## Git 상태
- Branch: `yunjae`
- 최근 commit: `d8ea392` (D-1 handoff + mag-weighted + auto loop)
- 이번 세션 새 파일: `auto_loop_v2.py`, `b4_seed_repro_v{2,3}.sbatch`, 이 HANDOFF 업데이트

## Key insight (뼈아픈 교훈)
- **Pattern 1**: E invdyn(자체) vs kit GRU 부호 반전. 어떤 auxiliary도 tight coupling 불가.
- **Pattern 2**: 예상 LB 회귀식 `0.4·A + 0.089` 안전영역 밖에서 실패 (B4v2 +0.013, hybrid +0.010 오차).
- **B4 성공 = 미미 조정 + 짧은 학습 + 우연**. 더 세게/오래/크게 = 필패.
- **Codex 최선**: B4 근방 촘촘 탐색 + checkpoint selection이 유일한 30% 문턱 방식.
