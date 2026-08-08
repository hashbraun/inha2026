"""각 eval 첫 프레임을 16번 이어붙여 정지영상 mp4 216개 생성."""
import imageio.v3 as iio
import numpy as np
from pathlib import Path
from PIL import Image

EVAL_DIR = Path("/home1/sota/inha2026/data/eval/images")
OUT_DIR = Path("/home1/sota/inha2026/submission_kit/input_videos_still216")
OUT_DIR.mkdir(parents=True, exist_ok=True)

paths = sorted(EVAL_DIR.glob("*.png"))
print(f"이미지 {len(paths)}개, out_dir={OUT_DIR}")

for p in paths:
    img = np.array(Image.open(p).convert("RGB"))
    frames = np.repeat(img[None], 16, axis=0)
    iio.imwrite(OUT_DIR / f"{p.stem}.mp4", frames, fps=6, codec="libx264")

print(f"완료: {len(paths)}개 mp4 저장")
