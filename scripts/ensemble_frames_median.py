"""여러 mp4 세트(같은 sample_id)를 프레임별 median으로 앙상블.

사용:
    python ensemble_frames_median.py \\
        --in-dirs submission_kit/input_videos_v5base_gs50 \\
                  submission_kit/input_videos_v5base_seed1 \\
                  submission_kit/input_videos_v5base_seed2 \\
                  submission_kit/input_videos_v5base_seed3 \\
                  submission_kit/input_videos_v5base_seed4 \\
        --out-dir submission_kit/input_videos_v5base_median5

median이 mean보다 outlier 저감에 강하고 blur가 덜 심함 (No-training Top 1 근거).
"""
import argparse
from pathlib import Path
import imageio.v3 as iio
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dirs', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--fps', type=int, default=6)
    ap.add_argument('--mode', choices=['median', 'mean'], default='median')
    args = ap.parse_args()

    in_dirs = [Path(d) for d in args.in_dirs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_mp4s = sorted(in_dirs[0].glob('sample_*.mp4'))
    print(f'{len(ref_mp4s)}개 sample, {len(in_dirs)} 세트 앙상블 ({args.mode})')
    print(f'입력 세트: {[d.name for d in in_dirs]}')

    n_ok = 0
    for ref in ref_mp4s:
        sid = ref.stem
        # 모든 세트에 존재 확인
        paths = [d / f'{sid}.mp4' for d in in_dirs]
        if not all(p.exists() for p in paths):
            missing = [i for i, p in enumerate(paths) if not p.exists()]
            print(f'  skip {sid}: missing in {missing}')
            continue

        stack = []
        for p in paths:
            v = iio.imread(p).astype(np.float32)[:16]
            stack.append(v)
        stack = np.stack(stack, axis=0)  # (N, 16, H, W, 3)

        if args.mode == 'median':
            out = np.median(stack, axis=0)
        else:
            out = np.mean(stack, axis=0)
        out = np.clip(out, 0, 255).astype(np.uint8)
        iio.imwrite(out_dir / f'{sid}.mp4', out, fps=args.fps, codec='libx264')
        n_ok += 1

    print(f'완료: {n_ok}/{len(ref_mp4s)} 저장 → {out_dir}')


if __name__ == '__main__':
    main()
