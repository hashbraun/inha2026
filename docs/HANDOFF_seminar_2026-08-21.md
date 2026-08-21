# Handoff · Cosmos 3-Nano 세미나 슬라이드 · 2026-08-21

## 📁 파일 · 서버

- **HTML** · `/home1/sota/inha2026/docs/seminar_cosmos_nano.html`
- **에셋** · `/home1/sota/inha2026/docs/imgs/` (Notion 원본 image 1~12), `/home1/sota/inha2026/docs/imgs/p03/` (포트폴리오 자산 · eval_samples · dataset_eda · good_bad_grid · good_sample_085/086/052.gif)
- **서버** · `python3 -m http.server 9203 --bind 0.0.0.0 --directory /home1/sota/inha2026/docs` (PID 193911)
- **URL** · `http://192.168.110.106:9203/seminar_cosmos_nano.html`
- **강제 새로고침** · Cmd/Ctrl + Shift + R
- **참고 스타일 원본** · `http://192.168.110.107:9102/briefing/index.html` · `http://192.168.110.107:9102/vision/index.html` (내부망)

## 📚 참고 자료

- **논문** · Cosmos 3 Technical Report v2 (arXiv:2606.02800) · https://www.alphaxiv.org/abs/2606.02800
- **WAM survey** · 2606.20781 (분류 taxonomy 참고)
- **Notion export** · `/tmp/cosmos_notion/` (HTML + image 1~12)
- **사용자 포트폴리오 로컬** · `/home1/sota/cv/yunjae/portfolio.html`, `/home1/sota/cv/yunjae/projects/project-03.html`
- **포트폴리오 GitHub Pages** · https://hashbraun.github.io/projects/project-03.html
- **인하 2026 챌린지 자산** · `/home1/sota/inha2026/HANDOFF_2026-08-19_root.md`

## 🎯 발표 컨텍스트

- **발표자** · 조윤재 · HDC Labs · AI Lab
- **발표일** · 2026-08-21 (오늘)
- **시간** · 20분 + Q&A
- **총 슬라이드** · **28장** (본편 25 + Appendix 3) · 19p 흡수·삭제, 결론 3장 분리 반영

## 🗂️ 챕터 · 슬라이드 매핑 (현재 상태)

| 페이지 | 챕터 | 제목 · 요약 |
|---|---|---|
| 1 | 개요 | 표지 · Cosmos 3 (저자 99명 나열) |
| 2 | Ch.1 배경 | 기존 패러다임 (5가지 문제) |
| 3 | Ch.1 배경 | Solution · VLM+Video+VLA 통합 |
| 4 | Ch.1 배경 | WFM 정의 · aside |
| 5 | Ch.3 WFM·WAM | WFM → WAM 궤적 통합 |
| 6 | Ch.3 WFM·WAM | 세 가지 WAM 모드 (FD/ID/Policy · unified MoT SVG) |
| 7 | Ch.4 아키텍처 | MoT 전체 flow + 입력 토큰 형태 + Layer pathway |
| 8 | Ch.5 학습 | Pre-training · 아키텍처 재사용 GIF |
| 9 | Ch.5 학습 | Mid-training · 3 모드 활성화 |
| 10 | Ch.5 학습 | Post-training · specialist 분화 |
| 11 | Ch.6 추론 | Inference pipeline · sampling loop ×N |
| 12 | Ch.7 실험 | Overview · 5 questions |
| 13 | Ch.7 실험 | 실험 1 · Robotics FD (image 8) |
| 14 | Ch.7 실험 | 실험 2 · Camera Motion FD (image 9) |
| 15 | Ch.7 실험 | 실험 3 · Inverse Dynamics (image 10) |
| 16 | Ch.7 실험 | 실험 4 · Robot Policy (image 11) |
| **17** | Ch.8 실전 | 챌린지 Task · 데이터 · Constraints |
| **18** | Ch.8 실전 | EDA · 발견 · 방향성 |
| **19** | Ch.8 실전 | 여러 방법론 실험 · 방법론 시각 3-card (❌ 삭제 예정) |
| **20** | Ch.8 실전 | 최종 전략 6축 + 파이프라인 SVG |
| **21** | Ch.9 정리 | 결과 · Ablation · 회귀 |
| **22** | Ch.9 정리 | Good rollout 3-gif + 환경·Model·Data |
| **23** | Ch.9 정리 | 결론 · Paper vs Experience + 한계 + 실전 의의 |
| 24 | Ch.9 정리 | End · 감사합니다·Q&A |
| 25-27 | Appendix | Cosmos 3 오픈소스 · 모델 스펙 · Two-way Flat Attention |

## 🛠 진행 이력 · 남은 작업

### 2026-08-21 세션 (완료)
- [x] 17p · Train/Eval 이미지 중복 → **동일 형식 1장**으로 통합 (eval_samples.png · max-height 20vmin)
- [x] 17p · 우측 gif 축소 (44vmin → 32vmin, padding 축소)
- [x] 17p · 좌측 폰트·Constraints 확대 (1.55→1.85vmin body, cb-item 1.45vmin)
- [x] 18p · "확정 아닌 검증 대상" 문구 삭제, 세 발견에 각 방법론 명시 (① EE-Delta · ② ANCHORED_soft · ③ Action Injection)
- [x] 18p 하단 · 방법론 시각 3-card 흡수 (SVG 3개 · 세로 6.5vmin)
- [x] 19p · 슬라이드 자체 삭제 (18p에 흡수 완료)
- [x] 20p(→19p) · 파이프라인 SVG 확대 (44vmin → 58vmin, split 0.82:1.18)
- [x] 21p(→20p) · Ablation 표 폰트↑ (1.5→1.85vmin), padding↓ (.85→.45vmin) · `.ablation-tight` 클래스 추가
- [x] 22p(→21p) · gif 확대 (24→36vmin), 3-card 축소 (padding .7vmin, font 1.15vmin), 가운데 gif 중앙 정렬 (margin auto), gif 간격 축소 (1→0.35vmin)
- [x] 23p(→22p) · CORE MESSAGE 강조 배너 추가 (좌측 5px ok stripe, 2.15vmin heading), "6일 스프린트" 제거, 현실적 한계로 재작성 (full training 데이터 · 예산 · 채점기 접근 · 정량화 미완)

### 세션 산출물
- **발표 스크립트** · `/home1/sota/inha2026/docs/SCRIPT_seminar_2026-08-21.md`
  - 슬라이드별 발화 초 배분 (표지 30s ~ 결론 90s · 총 24.8분 · 20분 컷 우선순위 명시)
  - Q&A 예상 질문 5선
  - 시간 배분 표 · 리허설 체크리스트

### 2026-08-21 세션 (추가 개선)
- [x] 17p · Task 프레이밍을 **Action-conditioned Video Generation = WAM (Forward Dynamics)**로 강조
- [x] 17p · train (영상+action) vs eval (첫 프레임+action) **데이터 형식 차이 SVG 대비 시각화**
- [x] 17p · Output gif 카드 여백 완전 제거 (border+radius만) · gif 크기 36vmin으로 확대
- [x] 20p · Ablation 표 **진짜 확대** — `.ablation-big` 강제 스타일 (본문 2.2vmin, num 2.35vmin, padding .55vmin)
- [x] 21p · 하단 3-card 폰트 1.55vmin, padding·line-height 자연화
- [x] 22p 결론 1장 → **3장 분리** (Paper vs Experience · 근본적 한계 · 실전 의의)
  - Paper vs Experience · 논문 리서치(agent) 반영 · 재현 성공 2축 + 재현 갭 3축 표
  - 근본적 한계 · 실험 로그 리서치(agent) 반영 · 4개 근본 (action 신호 약함 · 채점기 saturation · 도메인 gap · I2V 정보 한계)
  - 실전 의의 · 실측 로봇 · embodiment 이식 · latency · 방법론 이식 4카드
- [x] 스크립트 업데이트 · 결론 3장 발화 배분 재작성 · 24.8분 오버 대응 컷 우선순위 명시

## 🎨 CSS 참고 (수정 시)

- 팔레트 · `--acc #4c8dff` · `--ok #5ec97a` · `--warn #e8a355` (3색만)
- 폰트 시스템 · `--h1 6.4vmin` · `--h2 3.4vmin` · `--h2-mega 5.6vmin` · `--h3 2vmin` · `--body 1.65vmin` · `--small 1.35vmin` · `--tiny 1.15vmin`
- 이미지 여백 · `.fig-tight` 클래스 (padding 0.4vmin · img padding 0)
- 실험 슬라이드 통일 · `.exp-slide .exp-split .exp-card`
- 프로세스 리스트 · `.process-head .process-list` (좌 [과정] 텍스트 스타일)

## 🐛 알려진 이슈

- 17p의 train/eval 이미지가 동일 (같은 파일 사용 · placeholder 상태)
- Ch.7 실험 4장은 SVG bar chart 대신 이미지(imgs/image 8~12.png) 사용 중
- Two-way Flat Attention 슬라이드는 Appendix에 이동됨 (본편 시간 확보용)

## 📝 사용자 스타일 · 톤 규칙

- 응답 · **한국어**, 명사형 종결 선호
- AI스러운 톤 지양
- 툴 실행 승인 요청 시 · **왜 실행하는지 한 문장** 먼저 설명 (예: "SAM2 config 확인하기 위해 컨테이너 탐색합니다")
- 확인 없이 알아서 진행 · 되돌리기 쉬운 작업은 승인 없이 실행
- 사용자 자기 자료 (포트폴리오 · 노션 export) · 요약·인용 자유
- Codex 검토 · 실험/구현 결정 전 codex CLI 자문 (확증편향 완화)

## 🚀 다음 세션 시작 프롬프트 (권장)

```
docs/HANDOFF_seminar_2026-08-21.md 읽고 이어서.
17p부터 순차 진행 · 남은 작업 우선순위 그대로.
서버 죽어 있으면 재기동 (포트 9203, --directory docs).
브라우저 강제 새로고침 안내 필수.
```
