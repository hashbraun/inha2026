# Baseline

이 폴더는 베이스라인 모델 실행 예시입니다.

베이스라인은 선택 사항입니다. 자체 모델로 영상을 생성하는 참가자는 이 폴더를 사용하지 않아도 됩니다.

## 실행 방법

`baseline.ipynb`를 열고 위에서부터 차례대로 실행합니다.

Python 3.8-3.12 환경을 지원합니다.

노트북은 다음 과정을 수행합니다.

1. 데이터 경로 확인
2. 베이스라인 실행 환경 설치
3. 베이스라인에 필요한 backbone checkpoint 다운로드
4. 평가 입력으로 mp4 영상 생성
5. `../submission_kit`을 사용해 제출 CSV 생성

## 출력 위치

베이스라인으로 생성된 mp4 영상은 아래 폴더에 저장됩니다.

```text
../submission_kit/input_videos/
```

제출 CSV는 아래 경로에 생성됩니다.

```text
../submission_kit/submission_features.csv
```

## Checkpoint

제공된 베이스라인 checkpoint는 아래 파일입니다.

```text
checkpoints/baseline_diffusion.ckpt
```

노트북 실행 중 `checkpoints/backbone.ckpt`가 없으면 자동으로 다운로드합니다.

`xformers`는 선택 사항입니다. 설치되어 있으면 더 빠를 수 있지만, 설치되어 있지 않아도 기본 attention으로 실행됩니다.
