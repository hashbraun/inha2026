# Submission Kit

이 폴더는 참가자가 생성한 mp4 영상을 제출 CSV로 변환하는 도구입니다.

## 1. 영상 넣기

생성한 mp4 파일을 `input_videos/` 폴더에 넣습니다.

```text
input_videos/sample_000000.mp4
input_videos/sample_000001.mp4
...
input_videos/sample_000215.mp4
```

파일명은 반드시 평가 샘플 ID와 같아야 합니다.

## 2. 설치

Python 3.8-3.12 환경을 지원합니다.

```bash
python -m pip install -r requirements.txt
```

이미 호환되는 PyTorch / torchvision이 설치된 환경이라면 그대로 사용해도 됩니다.

## 3. CSV 생성

```bash
python make_submission_csv.py
```

성공하면 아래 파일이 생성됩니다.

```text
submission_features.csv
```

이 파일을 데이콘에 제출합니다.

## 참고

- `input_videos/`에 필요한 mp4 파일이 없으면 누락된 파일명을 알려 줍니다.
- `checkpoints/action_extractor.ckpt`는 CSV 생성에 필요한 파일이므로 삭제하거나 수정하지 마세요.
