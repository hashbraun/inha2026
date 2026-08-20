# AUTO DECISION LOG (사용자 취침 중 자율 진행)

**시작**: 2026-08-20 01:45 KST
**마감**: 2026-08-20 18:00 KST (16h 15m)
**사용자 취침 예상 종료**: 06:00~08:00 KST

## 자율 실행 원칙 (사용자 승인 하에)

### 반드시 유지 (SAFE)
- **안전본**: `v5b_ANCHORED_soft.csv` LB 0.24524 → `AUTO_ready_safe_v5b_anch.csv`
- **도전본 기본**: `submission_b4_step6000.csv` LB 0.2407 → `AUTO_ready_challenge_b4.csv`
- 이 두 개는 **어떤 실험이 실패해도 방어됨**

### 새 후보 제출 조건 (모두 만족 시)
1. Pareto gate 통과: E-invdyn L1 개선 AND alex_cos vs B4 ≥ 0.985
2. Order 반응 확인: gt-shuffle / noise_floor > 1.5 (순서 정보 사용)
3. Self-consistency: Action mean ≤ 0.36

### 새 candidate 실측 slot 사용
- 아침 사용자 확인 후 결정 (자동 실측 안 함)
- 대신 `AUTO_ready_new_best.csv`로 준비만 완료

### 시간 기준 auto-stop
- 08:00 KST: 모든 학습 kill 결정 시점
- 12:00 KST: 새 inference launch 마지막 시점
- 15:00 KST: 최종 CSV 확정
- 17:00 KST: 사용자에게 결과 대기

---

## Timeline

### [2026-08-20T01:42:04] Auto Orchestrator START
- [2026-08-20T01:42:04] Stage 1 job: 30955, ckpt dir: /home1/sota/inha2026/checkpoints/st1_frozen_shift

### [2026-08-20T01:47] 순서 vs 크기 실험 완료 (B4)
| sample | gt-shuffle | gt-x0.5 | gt-x2.0 |
|---|---|---|---|
| 000000 | 4.71 | 8.15 | 7.95 |
| 000001 | 10.56 | 7.44 | 5.69 |
| 000002 | 5.07 | 4.15 | 4.43 |
| **AVG** | **6.78** | **6.58** | **6.02** |

- 세 조건 모두 유사한 MAE (~6-7)
- gt-gt noise floor 대기 → 판정 최종
- 만약 noise floor ≈ 6-7이면: 어떤 action 조작도 유의미하지 않음 (action grounding 약함)
- 만약 noise floor ≈ 2-3이면: 순서/크기 모두 반응 (정상 FD)

### 대기 사항
- gtgt_b4 (30952): 6~10분 남음
- Stage 1 학습 (30955): ~1h 진행 중, 첫 checkpoint step 500 예상
- 오케스트레이터가 자동으로 checkpoint별 ablation → best selection → 216 inf → Pareto → AUTO_ready_new_best 준비

### [2026-08-20T02:00+] gt-gt 노이즈 플로어 (B4) 확정
- noise_floor mean = **8.829** (median 8.391)
  - sample_000000: 10.085
  - sample_000001: 8.161
  - sample_000002: 8.242

### 정규화 진단 (gt-zero / noise_floor)
- A0_pretrained: gt-zero/nfl = **0.98x**
- A2_v5b_4k: gt-zero/nfl = **0.57x**
- A3_v5b_20k: gt-zero/nfl = **2.93x**
- A5_B4: gt-zero/nfl = **0.80x**

### 순서/크기 재해석
- shuffle: MAE=6.780 = **0.77x noise_floor**
- x0.5: MAE=6.580 = **0.75x noise_floor**
- x2.0: MAE=6.022 = **0.68x noise_floor**

### [2026-08-20T01:47] 🚨 결정적 재해석
**noise floor = 8.83 (mean), 8.39 (median)**

앞 진단 재해석 (gt-zero / noise_floor):
| ckpt | 정규화 | 판정 |
|---|---|---|
| A0 pretrained | 0.98x | 노이즈 |
| A2 v5b_4k | 0.57x | 노이즈 이하 |
| **A3 v5b_20k** | **2.93x** | **유일한 signal** |
| A5 B4 | 0.80x | 노이즈 |

순서/크기 (B4): 모두 0.6~0.8x noise → 반응 없음

**결론**: B4 학습이 v5b_20k의 action grounding을 파괴함. Stage 1 목표 = v5b_20k 2.93x 유지 또는 개선.

### 판정 기준 재조정
- Stage 1 checkpoint 판정: gt-zero / noise_floor > **1.5x** (약한 signal)
- Best 후보: > **2.0x** (검증됨)
- 이하는 kill 또는 skip

### [2026-08-20T01:46:27] Auto Orchestrator START
- [2026-08-20T01:46:27] Stage 1 job: 30955, ckpt dir: /home1/sota/inha2026/checkpoints/st1_frozen_shift
- [2026-08-20T01:46:27] 신규 checkpoint 감지: ckpt_step000500.pt
- [2026-08-20T01:46:27]   Ablation job 30956 launched (tag=st1_step000500)
- [2026-08-20T01:52:30]   Ablation ckpt_step000500.pt: gt-zero=28.671 gt-rev=7.753 zero-randn=26.477
- [2026-08-20T01:54:30] 신규 checkpoint 감지: ckpt_step001000.pt
- [2026-08-20T01:54:30]   Ablation job 30957 launched (tag=st1_step001000)
- [2026-08-20T02:00:33]   Ablation ckpt_step001000.pt: gt-zero=30.495 gt-rev=7.006 zero-randn=28.930
- [2026-08-20T02:02:33] 신규 checkpoint 감지: ckpt_step001500.pt
- [2026-08-20T02:02:33]   Ablation job 30958 launched (tag=st1_step001500)
- [2026-08-20T02:08:36]   Ablation ckpt_step001500.pt: gt-zero=31.863 gt-rev=7.946 zero-randn=30.088
- [2026-08-20T02:10:36] 신규 checkpoint 감지: ckpt_step002000.pt
- [2026-08-20T02:10:36]   Ablation job 30959 launched (tag=st1_step002000)
- [2026-08-20T02:16:39]   Ablation ckpt_step002000.pt: gt-zero=34.943 gt-rev=7.971 zero-randn=32.712
- [2026-08-20T02:18:39] Stage 1 (job 30955) 종료됨
- [2026-08-20T02:18:39] 신규 checkpoint 감지: ckpt_step002500.pt
- [2026-08-20T02:18:39]   Ablation job 30960 launched (tag=st1_step002500)
- [2026-08-20T02:18:39]   ckpt_step002000.pt: gt-zero/nfl=3.96x rev/nfl=0.90x score=3.96
- [2026-08-20T02:18:39]   ckpt_step001500.pt: gt-zero/nfl=3.61x rev/nfl=0.90x score=3.61
- [2026-08-20T02:18:39]   ckpt_step001000.pt: gt-zero/nfl=3.45x rev/nfl=0.79x score=3.45
- [2026-08-20T02:18:39]   ckpt_step000500.pt: gt-zero/nfl=3.25x rev/nfl=0.88x score=3.25

### [2026-08-20T02:18:39] Best Stage 1 ckpt: ckpt_step002000.pt score=3.957 (gt-zero 3.96x, rev 0.90x)
- [2026-08-20T02:18:39] 216 inference launched: job 30961, tag=st1_best_002000
- [2026-08-20T02:24:42]   Ablation ckpt_step002500.pt: gt-zero=34.465 gt-rev=8.156 zero-randn=33.385
- [2026-08-20T03:42:43]   st1_best_002000 inf 완료: Action=0.3965 pred_LB=0.2476
- [2026-08-20T03:42:43]   Pareto job 30963 launched
- [2026-08-20T03:44:43]   st1_best_002000 Pareto: e-inv=41.699 alex_cos=0.9490
- [2026-08-20T03:44:43]   ❌ Pareto gate 불통과: Action=0.3965, alex_cos=0.9490, e-inv=41.699
- [2026-08-20T03:44:43] 모든 후속 작업 완료. Orchestrator 정상 exit.

### [2026-08-20T07:40] 🎉 rej_w0025 실측 = 0.2340
- Predicted: 0.2342, Actual: 0.2340, 오차 -0.0002 (안전영역 정확)
- 이전 도전본 B4 (0.2407) 대비 **-0.0067 개선** (약 3.4σ, 유의미)
- **새 도전본 확정**: `submission_rej_w0025.csv`
- **Rejection picker 접근 실제 작동 확인**

### 새 도전본 지정
- CHALLENGE 갱신: `AUTO_ready_challenge_rej_w0025.csv` (LB 0.2340)
- 기존 `AUTO_ready_challenge_b4_step6000.csv` (0.2407)는 fallback으로 유지

### [2026-08-20T09:00] LoRA 0-step 진단 완료
- 5 config 모두 ratio_avg=1.46 / ratio_mid=2.07 → LoRA 삽입 자체 무해
- **학습이 원인 확정** — Stage 3(500 step)에서 1.14x로 파괴됨

### [2026-08-20T09:03] 사용자 계획 실행
- ✅ CSV 확정 제출 완료 (safe v5b_ANCH 0.24524 + challenge rej_w0025 0.2340)
- ✅ Stage 4 launch (job 30984, FREEZE_LORA=1, action_proj만 학습)
- ✅ Seeds 6-15 launch (jobs 30985-30994, 10잡 병렬)
- 판정 threshold: 0.004 (σ=0.002 × 2)
