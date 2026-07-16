from __future__ import annotations

import argparse
from pathlib import Path

import torch

from feature_csv_utils import list_challenge_sample_ids, write_feature_csv


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert generated mp4 videos into the required submission CSV.")
    parser.add_argument("--prediction-root", default="input_videos")
    parser.add_argument("--challenge-root", default="../data/eval")
    parser.add_argument("--output-csv", default="submission_features.csv")
    parser.add_argument("--action-stats-path", default="../data/train/so100_action_statistics.json")
    parser.add_argument("--action-extractor-ckpt", default="checkpoints/action_extractor.ckpt")
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--temporal-length", type=int, default=16)
    parser.add_argument("--target-height", type=int, default=320)
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--pad", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-pad", dest="pad", action="store_false")
    parser.add_argument("--feature-precision", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_ids = list_challenge_sample_ids(Path(args.challenge_root))
    prediction_root = Path(args.prediction_root)
    missing = [sample_id for sample_id in sample_ids if not (prediction_root / f"{sample_id}.mp4").exists()]
    if missing:
        preview = ", ".join(f"{sample_id}.mp4" for sample_id in missing[:10])
        raise FileNotFoundError(
            f"{prediction_root}에 제출 변환 대상 mp4가 부족합니다. "
            f"총 {len(missing)}개 누락. 예: {preview}"
        )
    expected = set(sample_ids)
    extra = sorted(path.name for path in prediction_root.glob("*.mp4") if path.stem not in expected)
    if extra:
        preview = ", ".join(extra[:10])
        print(f"[warning] 평가 sample_id와 매칭되지 않는 mp4 {len(extra)}개는 무시됩니다. 예: {preview}")

    write_feature_csv(
        video_root=prediction_root,
        sample_ids=sample_ids,
        output_csv=Path(args.output_csv),
        source="submission",
        device=device,
        temporal_length=args.temporal_length,
        target_height=args.target_height,
        target_width=args.target_width,
        pad=args.pad,
        feature_batch_size=args.feature_batch_size,
        precision=args.feature_precision,
        action_extractor_ckpt=args.action_extractor_ckpt,
        challenge_root=Path(args.challenge_root),
        action_stats_path=args.action_stats_path,
    )
    print(f"[submission csv] saved to {args.output_csv}")


if __name__ == "__main__":
    main()
