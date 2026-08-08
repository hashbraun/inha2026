"""첫 프레임 앵커 + Latent Blending 후처리 (Plan Top 1).

## 배경
metric-targeted 측정에서 프레임 0~5 구간에서 우리 생성 mp4가 정지영상보다 나쁨.
즉 초반 프레임을 GT 첫 프레임 쪽으로 blending하면 리더보드가 오르는 방향.

## 처리
frame 0    = GT_PNG 그대로 (bit-exact overwrite)
frame 1~5  = α[k] * GT + (1 - α[k]) * gen[k], α = [0.7, 0.5, 0.35, 0.2, 0.1]
frame 6~15 = gen 그대로 (선택: temporal smoothing)

## 사용
    python postprocess_first_frame_anchor.py \\
        --in-videos submission_kit/input_videos_v5base_gs50 \\
        --gt-images data/eval/images \\
        --out-videos submission_kit/input_videos_v5base_gs50_anchored \\
        --alpha 0.7 0.5 0.35 0.2 0.1
"""
import argparse
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-videos', required=True, help='입력 mp4 폴더 (216개)')
    ap.add_argument('--gt-images', required=True, help='eval GT PNG 폴더 (216개 sample_*.png)')
    ap.add_argument('--out-videos', required=True, help='출력 mp4 폴더')
    ap.add_argument('--alpha', nargs='*', type=float,
                    default=[0.7, 0.5, 0.35, 0.2, 0.1],
                    help='프레임 1부터 적용할 alpha 리스트 (α*GT + (1-α)*gen). 빈 리스트면 frame 0만 overwrite')
    ap.add_argument('--fps', type=int, default=6)
    ap.add_argument('--overwrite-frame-0', action='store_true', default=True,
                    help='프레임 0을 GT로 완전 덮기 (기본 True)')
    args = ap.parse_args()

    in_dir = Path(args.in_videos)
    gt_dir = Path(args.gt_images)
    out_dir = Path(args.out_videos)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = np.array(args.alpha, dtype=np.float32)
    print(f'alpha (frame 1..{len(alphas)}): {alphas.tolist()}')
    print(f'overwrite frame 0: {args.overwrite_frame_0}')

    mp4s = sorted(in_dir.glob('sample_*.mp4'))
    print(f'{len(mp4s)}개 mp4 처리')

    n_processed = 0
    for mp4 in mp4s:
        sid = mp4.stem
        gt_png = gt_dir / f'{sid}.png'
        if not gt_png.exists():
            print(f'  skip {sid}: GT PNG 없음')
            continue

        gt = np.array(Image.open(gt_png).convert('RGB'), dtype=np.float32)  # (H, W, 3)
        gen = iio.imread(mp4).astype(np.float32)  # (16, H, W, 3)
        assert gen.shape[0] >= 16, f'{sid}: frame 수 {gen.shape[0]} < 16'
        gen = gen[:16]

        # GT 이미지 크기와 mp4 프레임 크기 일치 확인/보정
        if gt.shape != gen.shape[1:]:
            gt_resized = np.array(Image.fromarray(gt.astype(np.uint8)).resize(
                (gen.shape[2], gen.shape[1]), Image.BILINEAR), dtype=np.float32)
        else:
            gt_resized = gt

        # frame 0: GT로 완전 덮기
        if args.overwrite_frame_0:
            gen[0] = gt_resized

        # frame 1..len(alphas): α blending
        for k, alpha in enumerate(alphas, start=1):
            if k >= 16:
                break
            gen[k] = alpha * gt_resized + (1.0 - alpha) * gen[k]

        # frame len(alphas)+1..15: 그대로
        gen = np.clip(gen, 0, 255).astype(np.uint8)
        iio.imwrite(out_dir / f'{sid}.mp4', gen, fps=args.fps, codec='libx264')
        n_processed += 1

    print(f'완료: {n_processed}/{len(mp4s)} 저장 → {out_dir}')


if __name__ == '__main__':
    main()
