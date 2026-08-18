# Handoff — 2026-08-19 (D-1)

## 대회
- **Dacon 236736**, 인하 AI 챌린지 World Model Challenge
- 마감 **2026-08-20 18:00 KST** (남은 ~28-33h 기준 08-19 기준)
- 채점: `0.3·(1-DINO_cos) + 0.3·(1-Video_cos R3D-18) + 0.4·Action_MAE`

## ⚡ 즉시 재개 프롬프트 (다음 세션)
```
inha2026 D-day 24-30h 남음. 다음 파일 순서로 읽고 자동 loop 이어서:
1. docs/plans/fresh_restart/HANDOFF_2026-08-19.md  (이 파일)
2. logs/fresh/auto_loop.log  (auto_loop 최신 tick)
3. logs/fresh/orchestrator_state.json  (state)
4. squeue -u sota  (진행 job)
5. 필요 시 새 candidate launch 후 auto_loop 재시작
```

## 확정 자산 (실측 검증)
| CSV | Action mean | 실측 LB | 상태 |
|---|---|---|---|
| **v5b_ANCHORED_soft** | 0.3886 | **0.24524** | 안전본 (필수 제출) |
| **B4 step6000** | 0.3798 | **0.2407** | 도전본 (안전본 -0.0045 개선) |
| e_select_ANCH | 0.3625 | 0.263 | 폐기 (visual 손상) |
| tier1_seed1 | ? | 0.270 | 참고 |
| tto_v5b_ANCH_dino16 | 0.078 | 0.04 | **규정 위반, 절대 제출 금지** |

## 실패 이력 (재시도 금지)
| 시도 | Action mean | LB | 원인 |
|---|---|---|---|
| B4v2 (constant weight 0.15) | 0.3735 | 0.2508 | Visual +0.010 손상 |
| Z5 (shuffled ranking + cross-attn) | 0.3970 | 0.2478 예상 | Objective misalign, visual 훼손 |
| Z6-lite step2000 (bounded α=0.08) | 0.3813 | 0.2415 예상 | B4 미달 |
| Z6-B4 hybrid step15000 | 0.3780 | **0.2499 실측** | 82M adapter + 장기학습 → visual 손상 |
| NAP (LoRA freeze) step8000 | 0.3819 | 0.2418 예상 | LoRA freeze로 표현력 부족 |
| Cosmos-Predict2.5 falsification | - | - | G5 FAIL (eff_action=0.64) 2회 재현 |
| Fresh DiT scratch | 0.956 | 나쁨 | Video prior 부재 |
| Track E Wan2.1-Fun | - | 0.70+ | 통합 실패 |

## 진행 중 학습 (2026-08-19 아침)
| Job | 내용 | GPU | 상태 |
|---|---|---|---|
| **30749** | B4 sweep weight=0.025, 12k step | Blackwell gpu-113 | pending → running |
| **30750** | B4 sweep weight=0.075, 12k step | Blackwell (depends 30749) | pending |
| **30751** | B4 sweep weight=0.05, 12k step (control) | Blackwell (depends 30750) | pending |

각 job ~4-5h. Total sequential ~12-15h.

## Auto loop (PID 431820)
- 10분 tick
- 각 candidate 학습 완료 → 자동 inference launch (여러 ckpt) → CSV Action mean 계산
- Action < 0.375 → success (사용자 실측 slot 확인 요청)
- Action 0.375-0.379 → marginal
- Action ≥ 0.379 → failed → 다음 pivot

State: `logs/fresh/auto_loop_state.json`

## 자동 Pivot Queue (실패 시 자동 진행)

### 1단계: B4 sweep 결과 판정 (진행 중, ~12-15h 후)
- 최선 ckpt Action mean 확인
- < 0.375: 사용자 실측 slot 사용 요청
- ≥ 0.375: 다음 단계

### 2단계: **Action-magnitude weighted B4** (준비 완료, 미실행)
- 파일: `scripts/track_z5/finetune_b4_magweight.py`
- 큰 action 시퀀스만 loss weight ↑ (기존 B4 성능 유지 + 큰 움직임 학습 집중)
- Env: `MAG_ALPHA=0.5 MAG_MEDIAN=7.3 MAG_STD=9.0`
- Launch:
```bash
# Blackwell sbatch (기존 b4_v5b_e_loss_bw.sbatch 참조)
export MAG_ALPHA=0.5
sbatch --nodelist=gpu-113 <새 sbatch>  # finetune_b4_magweight.py 실행
```
- Motion score 분포 (측정됨):
  - median 7.3, mean 9.3, p90 22.8, max 50.0
  - alpha=0.5로 z-score 기반 [0.5, 3.0] clip

### 3단계: B4 checkpoint sweep 세밀 (Codex 권장)
- B4 6000 base → weight variation 3개 × ckpts 4개 = 12개 조합
- 이미 30749/30750/30751로 시작

### 4단계 (최후): RAFT/RL with E invdyn ensemble
- E invdyn seed 3개 학습 → ensemble reward
- Diffusion sample rank + fine-tune
- 시간 20-30h (D-day 마감 위험)
- **규정 확인 필수** (Codex v5: "각 CSV score 보고 selection도 kit 정보 활용 소지" 있음)

### 5단계 (실패 확정 시): B4 step6000 최종 확정
- 안전본 v5b_ANCHORED_soft + 도전본 B4 step6000
- 최소 안전본 대비 -0.0045 개선 확정

## Codex 최종 권장 (`docs/plans/fresh_restart/codex_break_b4.txt`)

**핵심**: "B4는 우연히 sweet spot. 더 세게·오래보다 근방 촘촘 탐색이 유일한 30% 문턱 전략"

**확률 순위**:
1. B4 continuation + 다중 ckpt + IDM ensemble selection: **40-55%**
2. B4 + preserve-only 6-12k sweep: **35-45%**
3. Z6-B4 hybrid + checkpoint selection: 35-45% (실측 실패 확정)
4. Z6-B4 hybrid 단발 15k: 25-35% (실측 확정)
5. B4 longer flat/decay: 25-35%
6. NAP + differential LR: 20-30%
7. RAFT/DPO/RL 현 predictor: 10-20%
8. 새 backbone: 5-15%

## 코드 자산 (커밋됨 commit 8c31cec)

- `scripts/finetune_cosmos3_nano.py`: ACTION_SIGMA_CLAMP env var + FREEZE_LORA env var 지원
- `scripts/track_z5/crossattn_adapter.py`: Z5 adapter (실패)
- `scripts/track_z5/crossattn_adapter_z6.py`: Z6 bounded adapter (실패)
- `scripts/track_z5/finetune_z5.py`: Z5 학습 (실패)
- `scripts/track_z5/finetune_z6.py`: Z6-lite/hybrid (실패)
- `scripts/track_z5/infer_z5.py`: Z5/Z6 inference (성공)
- `scripts/track_z5/finetune_b4_magweight.py`: **NEW** action-magnitude weighted B4 (준비, 미실행)
- `scripts/track_z5/orchestrator_z6_p25.py`: Z6+P25 orchestrator (P25 폐기 후 종료)
- `scripts/track_z5/auto_loop.py`: continuous auto loop (진행 중 PID 431820)

## 규정 재확인
- 금지: `submission_kit/checkpoints/action_extractor.ckpt` (GRU), R3D-18, DINOv2 학습/selection/mp4 수정 사용
- 허용: 외부 pretrained (Cosmos3, Wan 등), 자체 clean-room predictor (E invdyn)
- 애매: TTO/RL sample rank (Codex v5 우려)
- **최종 지정 절대 금지**: TTO CSV 3개 (실격)

## Key Insight (뼈아픈 교훈)
1. **Pattern 1**: 우리 자체 predictor (E invdyn) vs kit GRU 부호 반전 (판독기 v2 실측)
   - 어떤 auxiliary loss도 kit GRU와 tight coupling 못함
2. **Pattern 2**: 예상 LB 회귀식 (0.4·Action + 0.089) **좋은 영역 밖에서 실패**
   - B4v2: 예상 0.238 → 실측 0.251 (+0.013)
   - Z6-B4: 예상 0.240 → 실측 0.250 (+0.010)
   - Action mean 개선 ≠ LB 개선 (visual 손상 fold)
3. **B4의 성공 = "미미 조정 + 짧은 학습 + 우연"**
   - 더 세게 (weight ↑) → visual 손상
   - 더 오래 (step ↑) → visual 손상
   - 더 크게 (adapter 추가) → visual 손상
   - 근방 촘촘 탐색만이 답
4. **판독기 v2 부호 반전** = 근본 misalign 증거
   - TTO(kit LB 0.04) → E invdyn L1=0.96 (나쁨)
   - e_select_ANCH(LB 0.263) → E invdyn L1=0.86 (좋음)
   - **자체 predictor로 kit GRU 완벽 근사 불가능**

## 자원 및 환경
- Blackwell gpu-113 (95GB × 1, sm_120): `inha2026_bw` env (torch 2.15dev, cu130)
  - `LD_LIBRARY_PATH`에 nvidia sub-package `/lib` 필수
- A6000_ada gpu-108/109/110/112 (44GB × 8): `inha2026` env
- **B4 계열은 Blackwell 필요** (A6000 44GB에서 OOM)
- Wan2.2 다운로드 완료: `models/wan22_ti2v_5b/` (32GB, 미사용)
- Cosmos-Predict2.5 ckpt: `models/predict2.5_action_cond/` (4.25GB, G5 실패 확정)

## 최종 제출 전략 (D-day)

**시나리오 A** (Sweep에서 성공 candidate 발견):
- 안전본: v5b_ANCHORED_soft.csv
- 도전본: 최선 Action mean CSV (실측 확인 후)

**시나리오 B** (Sweep 실패):
- Action-magnitude weighted B4 시도 (~6h)
- 성공 시 새 도전본

**시나리오 C** (모두 실패):
- 안전본: v5b_ANCHORED_soft.csv (0.24524)
- 도전본: **B4 step6000 (0.2407)** ← 확정 -0.0045 개선

**어느 시나리오든 최소 도전본 = B4 step6000 (실측 확정)**.

## 규정 위반 CSV 폐기 확인
`submission_kit/fresh/submission_tto_*.csv` 3개는 절대 지정 금지. 파일 자체는 남겨도 최종 제출 시 절대 선택 X.
