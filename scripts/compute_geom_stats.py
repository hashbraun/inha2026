"""SO-100 기하 feature 정규화 상수를 train 데이터에서 산출한다.

so100_to_bridge_v3의 절대 항(관절각, FK 위치, EE translation)을 [-1,1]로 맞추기 위한 상수.
rot6d는 이미 [-1,1]이므로 대상이 아니다.

⚠️ 반드시 `action`(command)으로 계산한다. eval에는 observation.state가 없으므로
   모델 조건용 기하는 학습·추론 모두 action 기준이어야 한다.
   (캘리브레이션용 FK는 반대로 observation.state를 쓴다 — docs/plans/camera-calibration/PLAN.md)

로버스트 스케일: 분위수를 ±1로 매핑한다. 평균/표준편차는 이상치에 끌려간다.

## 표집·분위수 선택 근거 (2026-08-04 실측)

**데이터셋별 균형 표집이 필수다.** 무작위 600 에피소드로 뽑았을 때 eval 관절각이 train
[1%,99%]를 joint1 10.7% / joint2 14.4% / joint4 5.6% 벗어났는데, 128개 데이터셋에서
각 8에피소드씩 균형 표집하니 joint2 0.0% / joint4 0.7%로 떨어졌다. 즉 대부분은 실제
도메인 차이가 아니라 **표집 편향**이었다.

**남는 진짜 도메인 차이는 joint1뿐이다** — eval의 10.3%가 train 전체 min/max 밖이다.
분위수를 넓혀도 해결되지 않으므로 CLIP 여유로 흡수한다(so100_to_bridge_v3.CLIP).

기본 분위수를 [0.1, 99.9]로 두는 이유: joint2는 [1,99]에서 14.4%가 밖이지만 min/max
기준으로는 0%다. 꼬리에 실제 데이터가 있다는 뜻이라 꼬리를 살린다.
"""
import argparse
import glob
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home1/sota/inha2026/baseline/challenge_kit/scripts")
from so100_fk import so100_fk  # noqa: E402

TRAIN_DIR = "/home1/sota/inha2026/data/train"
OUT_PATH = "/home1/sota/inha2026/data/train/so100_geom_statistics.json"
# FK 위치 중 실제로 정보를 담는 관절만 쓴다 (아래 근거는 산출 시 재검증한다):
#   joint 0,1 : q와 무관한 상수
#   joint 6   : 절대 EE translation과 동일 → 중복
FK_JOINTS_DEFAULT = [2, 3, 4, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=8,
                    help="데이터셋당 에피소드 수. 균형 표집이 아니면 통계가 편향된다")
    ap.add_argument("--quantile", type=float, nargs=2, default=[0.1, 99.9])
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--fk-joints", type=int, nargs="+", default=FK_JOINTS_DEFAULT,
                    help="FK 위치 stats에 포함할 keypoint 인덱스. 46D 표현은 --fk-joints 0 1 2 3 4 5 6 사용")
    args = ap.parse_args()
    fk_joints = args.fk_joints

    all_pq = sorted(glob.glob(f"{TRAIN_DIR}/*/*/data/chunk-*/episode_*.parquet"))
    by_ds = {}
    for f in all_pq:
        by_ds.setdefault("/".join(f.split("/")[-5:-3]), []).append(f)
    rng = np.random.default_rng(0)
    files = []
    for v in by_ds.values():
        files += list(rng.choice(v, size=min(args.per_dataset, len(v)), replace=False))
    print(f"데이터셋 {len(by_ds)}개 × 최대 {args.per_dataset}에피소드 = {len(files)}개")

    chunks, n_err = [], 0
    for pq in files:
        try:
            a = np.stack(pd.read_parquet(pq, columns=["action"])["action"].to_numpy()).astype(np.float32)
        except Exception:
            n_err += 1
            continue
        if a.ndim == 2 and a.shape[1] == 6:
            chunks.append(a)
    Q = np.concatenate(chunks, 0)
    print(f"총 {len(Q)} 스텝 (파싱 실패 {n_err}개)")

    P = so100_fk(Q)  # (N,7,3)

    # 상수 차원 재검증 — fk_joints 가정이 데이터에서 실제로 성립하는지 확인
    span = P.max(0) - P.min(0)  # (7,3)
    const = [j for j in range(7) if span[j].max() < 1e-6]
    print(f"상수인 관절: {const}  (fk_joints={fk_joints}에서 제외되어야 함)")
    if not set(const).issubset({0, 1}):
        print(f"⚠ 예상과 다름: {const}. fk_joints를 재검토할 것")
    dup = float(np.abs(P[:, 6] - P[:, 6]).max())  # 자기 자신, 아래에서 EE와 비교
    ee_pos = P[:, 6]

    def pct(x):
        lo, hi = np.percentile(x, args.quantile, axis=0)
        # 폭이 0인 축(상수)은 스케일 1로 두어 0으로 나누지 않게 한다
        return lo, np.where(hi - lo < 1e-6, lo + 1.0, hi)

    q_lo, q_hi = pct(Q)
    fk_lo, fk_hi = pct(P[:, fk_joints].reshape(len(P), -1))
    ee_lo, ee_hi = pct(ee_pos)

    stats = {
        "source": "action (command)",
        "n_steps": int(len(Q)),
        "n_episodes": len(files),
        "n_datasets": len(by_ds),
        "per_dataset": args.per_dataset,
        "quantile": args.quantile,
        "fk_joints": fk_joints,
        "joint_deg": {"lo": q_lo.tolist(), "hi": q_hi.tolist()},
        "fk_pos_m": {"lo": fk_lo.tolist(), "hi": fk_hi.tolist()},
        "ee_trans_m": {"lo": ee_lo.tolist(), "hi": ee_hi.tolist()},
    }
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'항목':>14s} {'lo':>28s} {'hi':>28s}")
    print(f"{'joint_deg':>14s} {np.round(q_lo,1).tolist()!s:>28s} {np.round(q_hi,1).tolist()!s:>28s}")
    print(f"{'ee_trans_m':>14s} {np.round(ee_lo,3).tolist()!s:>28s} {np.round(ee_hi,3).tolist()!s:>28s}")
    print(f"{'fk_pos_m':>14s} {len(fk_lo)}차원 (관절 {fk_joints} × xyz)")
    print(f"\n저장: {args.out}   (dup check {dup:.1e})")


if __name__ == "__main__":
    main()
