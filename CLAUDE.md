# DreamZero 프로젝트 규칙

## 매 쿼리 시작 시 필수 수행
1. `/home1/sota/inha2026/baseline/IMPLEMENTATION_NOTES.md` 를 읽는다.
2. 노트의 내용을 기반으로 작업한다.
3. 새로운 발견(데이터 구조, 모델 제약, 실패 이력 등)이 생기면 즉시 노트를 업데이트한다.

## 코드 작성 전 필수 검증 (이 단계를 건너뛰면 안 됨)

코드를 한 줄이라도 작성하기 전에 아래를 **직접 실행해서** 확인한다.
가정하거나 추론으로 때우는 것은 금지. 확인이 귀찮아도 반드시 한다.

### 모델/가중치 관련
- state_dict 키 이름: `list(model.state_dict().keys())[:20]`
- 실제 텐서 shape: `{k: v.shape for k, v in model.state_dict().items() if 'target' in k}`
- load_state_dict 결과: `missing`, `unexpected` 양쪽 모두 출력해서 실제 로딩 성공 확인
- PEFT/LoRA 래핑 여부: 키에 `base_model.model.` prefix 있는지 확인

### 데이터 관련
- parquet 컬럼, 영상 경로, 텐서 shape 등 가정 금지
- 확인 방법: 환경 Python `/home1/sota/anaconda3/envs/inha2026/bin/python` 사용

### 코드 수정 후
- 수정한 코드가 실제로 의도한 경로로 실행되는지 로그로 확인
- `missing`/`unexpected` 같은 silent failure가 없는지 반드시 체크
- 첫 번째 스텝 로그에서 shape, 값, 그래디언트 유무를 확인한 뒤 "성공"으로 보고
