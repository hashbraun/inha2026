# 주간 논문 리뷰 세미나 계획 — Cosmos 3-Nano + Inha 2026 실전 적용

**시간**: 20분 (Q&A 별도)
**청중**: 내부 연구자
**논문**: NVIDIA Cosmos 3 Technical Report v2 (arXiv 2606.02800)
**컨셉**: 표지 0.5분 · 논문 9분 · 실전 8분 · 마무리 2분 (총 13장, 19.5분)
**인용 버전 고정**: arXiv:2606.02800v2 (2026 발행분 기준, 열람일 2026-08-18)
**논문 flow 원칙**: tech report 본문 순서 (§1 Intro → §2 Model → §4 Training → §6 Results) 유지. §3 Data, §5 Infra, §7 Related Work는 발표에서 skip.

---

## 슬라이드 구성

### [30초] Slide 1 — 표지
- 제목: 논문 제목 ("Cosmos 3: ...")
- 부제목: 발표자
- 하단: 소속 협회 + 논문 PDF 주소
- *배경 이미지: Fig. 1 (Cosmos 3 개념도) 흐리게 깔기*
- **출처**: 자체

---

## Part A. 논문 (9분 · 6장) — tech report §1→§2→§4→§6 순

### [1.5분] Slide 2 — [§1 Introduction] 월드모델 배경 + Cosmos 3 위치
- 월드모델 정의 (한 줄): "환경 관측·행동 → 미래 관측·행동을 예측하는 모델"
- **World Action Model 3분류** (alphaxiv 2606.20781 §3.1~§3.3) — *발표자가 붙이는 좌표계, Cosmos 논문 자체 taxonomy는 아님*:
  - Render-and-Decode: 픽셀 미래 생성 후 행동 복원 (UniPi, DreamZero, GR-1)
  - Latent-Only: 픽셀 디코드 전 latent에서 행동 (VPP, Genie Envisioner, UWM)
  - Video-Generation-Free: 비시각 표현만 (FLARE, DUST, PointWorld)
  - 결합 방식 세부 축은 §4 four-axis anatomy 참고
- **Cosmos 3의 위치 (발표자 해석)**: Render-and-Decode에 가장 가깝지만, action을 영상에서 사후 복원하지 않고 **공동 생성**하는 **경계적·혼합 사례**. FD/ID/Policy 세 모드를 한 net에 통합
- Cosmos 3 자기 정의: 언어·이미지·비디오·오디오·**action** 이해·생성을 통합한 omnimodal WFM, Physical AI 겨냥, 2026 최신
- 스케일: Nano **16B** / Edge 4B / Super 64B (dense × 2 tower)
- 이 발표는 Nano
- *시각자료: WAM 3분류 도식 (직접 제작) + Cosmos 3 위치 표시*
- **출처**: WAM survey (arXiv 2606.20781 §3.1~§3.3, §4) + Cosmos 3 report §1 (Introduction)

### [2분] Slide 3 — [§2 Model] Dual-tower MoT 구조 (Fig. 2)
- 한 layer 안에 AR/DM 각자 LayerNorm·MLP·attention 파라미터 별도
- **AR Reasoner tower** = causal self-attn, 언어 + ViT vision token 처리
- **Diffusion Generator tower** = full attn, 자기 K/V + AR K/V 함께 참조
- 정보 흐름 = **AR → DM 단방향** (DM token은 AR로 역류 X, 파라미터는 공동학습)
- Nano backbone = Qwen3-VL-8B (36 layer, hidden 4096, 32 heads)
- 발표 포인트: "왜 두 tower로 쪼갰나" — reasoning(causal)과 generation(bidirectional)의 attention 성질 차이
- *png 시각자료: Fig. 2 dual-tower attention mask 도식*
- *gif 생성: AR K/V가 DM에 흘러가는 애니메이션*
- **출처**: Cosmos 3 report §2 (Model), Fig. 2 (dual-tower MoT)
- ⚠️ 세부 소섹션 번호는 원문 재확인 후 확정 (codex 지적: §2.1은 Encoders라 별도)

### [1.5분] Slide 4 — [§2.1 Encoders] Modality encoder & Token arrangement
- Video 이해 (§2.1.1): patch 16×16 → 2×2 merge + DeepStack, ViT는 backbone과 공동학습
- Video 생성 (§2.1.1): **frozen Wan2.2-TI2V-5B VAE**, 시간 4×·공간 32×32 압축
- Audio (§2.1.2): 별도 encoder
- Action (§2.1.3): embodiment별 input/output linear projection → 공유 hidden space (별도 action Transformer 없음), **Fig. 3에 unified action representation** 도식
- Token 배치: 항상 `[AR sub][DM sub]`, DM 내부는 `clean cond → noisy target`, modality 순서 vision → audio → action
- *png/gif 시각자료: Fig. 3 + token stream 예시*
- **출처**: Cosmos 3 report §2.1.1 (Image/Video), §2.1.2 (Audio), §2.1.3 (Action), Fig. 3

### [1.5분] Slide 5 — [§2 Model] 생성 모드 FD·ID·Policy 통합 (Fig. 4)
- 같은 네트워크에서 **어떤 token을 clean/noisy로 지정하냐**만 바꿈 (loss 자체는 diffusion objective)
- Forward dynamics = clean action → noisy future video
- Inverse dynamics = clean video transition → noisy action
- Policy = video + action 둘 다 noisy target (행동과 그 결과 동시 생성)
- 이게 논문의 진짜 지분: 3개 task를 하나로 재정의
- *png/gif 시각자료: Fig. 4 인용 (clean/noisy 배치 3종)*
- **출처**: Cosmos 3 report §2 (Model, Generation Modes), Fig. 4

### [1.5분] Slide 6 — [§4 Training] 3-stage recipe
- Pretraining (§4.1): Nano 기준 **31.05T processed token, 1,024 GB200**
- **Action mid-training (§4.2.2)**: 61.3K hours / 8.4M episodes, robotics + AV + camera + egocentric
- Robot-policy post-training (§4.2.5): DROID 등 task별 full post-training. LoRA는 논문에 언급 없음
- 발표 포인트: "action이 mid-training 단계에서 대규모 데이터로 심어진 modality"라는 점 → 뒤에 나올 우리 파이프라인의 gap 설명 근거
- **출처**: Cosmos 3 report §4.1 (Pretraining), §4.2.2 (Mid-training), §4.2.5 (Robot-policy post-training)
- ⚠️ 이전 초안의 "action loss 10× 가중"은 원문에서 확인 안 됨 → 삭제 (codex 지적)

### [1분] Slide 7 — [§6 Results] 벤치마크 요약 + 저자 3대 기여
- *png 시각자료: 논문 Table 19 / Table 20 인용*
- RoboLab-120 **Overall–Specific 39.7%** (Table 19, 논문 시점 1위) — *전체 성공률로 말하지 말 것*
- LIBERO-10 MT-init > PT-init: 2k iter, checkpoint당 500 rollouts 기준 **97.4% vs 95.2%** (Table 20) — mid-training 가치 정량화
- Video-action consistency PSNR: **left third-person camera 23.19 dB** / wrist 17.33 dB
- 저자 3대 기여 (§1 재인용):
  (a) omnimodal MoT 통합
  (b) action = 일급 modality, FD/ID/policy 전이 가능
  (c) infra·checkpoint·데이터 공개
- **출처**: Cosmos 3 report §6 (Results), Table 19 (RoboLab), Table 20 (LIBERO), §1 (Contributions)

---

## Part B. 실전 적용 (8분 · 5장)

### [1.5분] Slide 8 — Inha 2026 챌린지 셋업
- Task: first frame 1장 + action seq → 3초 로봇 팔 영상 생성
- 채점식: `0.3·(1-DINO_cos) + 0.3·(1-Video_cos R3D18) + 0.4·Action_MAE`
- 규정: kit 안의 GRU/R3D/DINO를 학습·selection에 사용 금지
- 최고 실측 = **0.2406** (목표 미달, 이걸 안고 발표)
- **출처**: Dacon 236736 규정 + 자체 실측

### [2분] Slide 9 — 우리 파이프라인 (논문 recipe와의 gap 명시)
- 백본: Cosmos 3-Nano LoRA 40k step (base I2V post-train checkpoint 위)
- 자체 붙인 것 = **LoRA + action FiLM 주입** (둘 다 **논문엔 없는 recipe**)
- 후처리 = first-frame anchoring (ANCHORED_soft)
- 정직한 gap 명시: 논문은 full post-train + mid-training으로 action 심음. 우린 그 예산 없어서 LoRA + FiLM으로 우회한 셈
- *시각자료: 우리 파이프라인 diagram (LoRA/FiLM 주입 위치 표시, draw.io)*
- **출처**: 자체 파이프라인 + Cosmos 3 report §4 대비

### [1.5분] Slide 10 — 관찰 ①: 채점식 회귀 (탐색적)
- 제출 표본 **n=5** 회귀:
  `LB ≈ 0.24 − 0.15·vid_l2 + 0.008·dino_l2 + 2.804·act_mae + 0.79·act_diff`
- act_mae 계수가 큰 건 사실. 다만:
  - n=5, 자유도 부족 → **표준화 계수·LOO·단변량 산점도로 재확인 필요**
  - `act_mae`는 채점식에 이미 들어있음 → 강한 관계는 부분적으로 자명
- 결론 톤: "act_mae 항이 LB 변동의 상당 부분을 설명하는 것으로 **관찰**" (단정 X)
- *시각자료: n=5 산점도 (자체 제작)*
- **출처**: 자체 제출 실측

### [1.5분] Slide 11 — 관찰 ②: 접근별 성패 패턴

| 접근 | act_mean (GT 0.389) | 실측/예측 LB | 결과 |
|---|---|---|---|
| Cosmos 3-Nano LoRA (자연 diffusion) | 0.39 | 0.2406 실측 | ✅ |
| DMD 8k (채점기 감독) | 0.658 | +0.755 예측 | ❌ |
| Robot-Factored WM (compositing) | 0.681 | 0.767 예측 | ❌ |
| Wan-Fun + FK skeleton (warp) | 0.33 | 0.701 예측 | ❌ |
| still216 (정지) | 낮음 | 0.428 실측 | ❌ |

- 공통 패턴: motion 분포가 자연 diffusion과 다르면 GRU가 페널티 부여
- 교란 요인 명시: 모델·후처리·화질이 함께 변함 → 순수하게 GRU 탓만 못함
- *시각자료: 성공/실패 영상 각 1개 (총 2개)*
- **출처**: 자체 제출 실측 + Codex v5~v7 조사 기록

### [1.5분] Slide 12 — 관찰 → 가설 → 검증할 실험
- **관찰**: 자연 diffusion 계열만 낮은 act_mae 받음
- **가설**: GRU action extractor가 motion naturalness에 강하게 반응 (action semantic이 아니라)
- **검증에 필요한 통제 실험** (아직 안 함, 발표에서 정직하게 밝힘):
  - 같은 영상 temporal shuffle → act_mae 변화
  - Action seq만 바꿔치기 → act_mae 반응
  - Frame freeze / warp 강도 sweep
  - 자체 재훈련 GRU vs 원본 GRU 출력 비교
- 결론 톤: "확정 아님, 다음 세션 우선순위"
- **출처**: 자체 관찰 + 후속 실험 설계

---

## Part C. 마무리 (2분)

### [1.5분] Slide 13 — Takeaways
1. Cosmos 3의 진짜 기여 = **action 일급화 + FD/ID/policy 통합 formulation** (아키텍처 자체보다 이쪽)
2. Nano 스케일 downstream = 논문 recipe 그대로 못 씀. LoRA + FiLM 같은 우회가 현실
3. 자동 metric은 human eval과 다른 걸 재는 경우가 많음. 챌린지 벤치마크에선 특히 조심할 것

### [0.5분] Discussion 프롬프트 (Q&A로 넘기며)
- "action mid-training 없이 LoRA만으로 FD prior 심을 수 있나?"
- "Nano의 dual-tower 정보 흐름이 downstream fine-tuning에 어떤 제약을 주나?"
- "GRU가 naturalness detector라는 가설, 어떻게 반증할까?"

---

## 시간 배분

| Part | 시간 | 슬라이드 |
|---|---|---|
| Intro | 0.5분 | 1 |
| Part A 논문 | 9분 | 2-7 |
| Part B 실전 | 8분 | 8-12 (1.5+2+1.5+1.5+1.5) |
| 마무리 | 2분 | 13 |
| **총** | **19.5분** (버퍼 +0.5분) | **13장** |

## 논문 flow ↔ 슬라이드 매핑

| Slide | 논문 section | 근거 링크 |
|---|---|---|
| 2 (Cosmos 3 소개) | tech report §1 (Introduction) | [alphaXiv 2606.02800](https://www.alphaxiv.org/abs/2606.02800) |
| 2 (WAM 3분류) | WAM survey §3.1~§3.3, §4 | [alphaXiv 2606.20781](https://www.alphaxiv.org/abs/2606.20781) |
| 3 | tech report §2 (Model), Fig. 2 | 위 alphaxiv |
| 4 | tech report §2.1.1/§2.1.2/§2.1.3, Fig. 3 | 위 alphaxiv |
| 5 | tech report §2 (Generation Modes), Fig. 4 | 위 alphaxiv |
| 6 | tech report §4.1, §4.2.2, §4.2.5 | 위 alphaxiv |
| 7 | tech report §6 (Results), Table 19, Table 20 | 위 alphaxiv |
| 8~12 | 자체 실측·파이프라인·Codex v5~v7 조사 | 로컬 |

## 준비물

- [ ] Cosmos 3 tech report Fig. 1 / **2** / 3 / 4 캡처 (Fig. 2 = dual-tower MoT 핵심)
- [ ] Nano 스펙 표 (16B, Qwen3-VL-8B backbone)
- [ ] WAM 3분류 도식 (직접 제작)
- [ ] 우리 파이프라인 diagram (LoRA + FiLM 주입 위치)
- [ ] LB 회귀 산점도 (n=5 명시)
- [ ] 접근별 성패 표
- [ ] 성공/실패 영상 각 1개 (총 2개)
- [ ] 백업: LoRA rank/lr, 학습 curve, action FiLM 구현 상세

## 주의 (codex 1차·2차 검토 반영)

- 회귀식 관련: "압도적"·"결정" 같은 단정형 표현 금지. "관찰·시사·설명력이 있음" 톤 유지
- GRU=naturalness는 **가설**로 명시. 통제 실험 없이 결론 내지 말 것
- "실패 접근 = 성공적 발견" 프레이밍 자제. 목표 미달과 관찰 가치 분리
- Nano의 LoRA/FiLM은 **논문에 없는 자체 recipe**임을 슬라이드에서 명시 (오해 방지)
- Data (§3) / Infrastructure (§5) / Related Work (§7)는 발표에서 skip. 질문 나오면 백업 슬라이드 준비
- **WAM 3분류로 Cosmos 3 위치 지정한 것은 발표자 해석** — survey가 직접 분류한 게 아님을 슬라이드에서 명시
- **"action loss 10× 가중" 원문 미확인 → 삭제 완료** (2차 검토 지적)
- **Fig 번호 정정**: dual-tower MoT는 Fig. 2, unified action representation은 Fig. 3
- **§ 번호 정정**: mid-training §4.2.2, robot-policy post-training §4.2.5
- **RoboLab 39.7%는 Table 19 Overall–Specific**, 전체 성공률로 말하지 말 것
- **PSNR 23.19 dB는 left third-person camera** (그냥 third-person 아님)
- 남은 약점 (2차 검토): n=5 회귀식이 과도하게 정밀해 보임, 목표 미달 원인 (LoRA/FiLM vs 데이터 vs metric) 분리 안 됨 → Slide 12 이후 Q&A에서 다룰 것
