"""회귀식(Agent 1, R²=0.993)으로 CSV의 리더보드 점수 예측.

LB ≈ 0.2404 − 0.1511·vid_l2 + 0.00766·dino_l2 + 2.804·act_mae + 0.788·act_diff

verify(baseline)와 diff 계산 후 대입. 리더보드 제출 전 사전 필터링.

사용:
    python predict_lb.py submission_kit/submission_XXX.csv
    python predict_lb.py submission_kit/submission_XXX.csv submission_kit/submission_YYY.csv ...
"""
import csv, json, sys
from pathlib import Path
import numpy as np

VERIFY = "submission_kit/submission_v5_gs50verify_TRAINSTATS.csv"  # 재현 baseline

# Agent 1 회귀 계수 (R²=0.993, n=6)
COEF_INTERCEPT = 0.2404
COEF_VID_L2 = -0.1511
COEF_DINO_L2 = 0.00766
COEF_ACT_MAE = 2.804
COEF_ACT_DIFF = 0.788


def load(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[(row['sample_id'], row['feature_component'])] = np.array(json.loads(row['feature_json']), dtype=np.float32)
    return rows


def compute_features(csv_path, verify_data):
    d = load(csv_path)
    sids = sorted({k[0] for k in verify_data})

    # verify 대비 diff (per-sample)
    vid_l2s, dino_l2s, act_maes = [], [], []
    for sid in sids:
        v_vid = verify_data[(sid, 'Video Feature Component')].flatten()
        v_din = verify_data[(sid, 'DINO Component')].flatten()
        v_act = verify_data[(sid, 'Action Component')].flatten()

        c_vid = d[(sid, 'Video Feature Component')].flatten()
        c_din = d[(sid, 'DINO Component')].flatten()
        c_act = d[(sid, 'Action Component')].flatten()

        vid_l2s.append(np.linalg.norm(c_vid - v_vid))
        dino_l2s.append(np.linalg.norm(c_din - v_din))
        act_maes.append(np.abs(c_act - v_act).mean())

    vid_l2 = float(np.mean(vid_l2s))
    dino_l2 = float(np.mean(dino_l2s))
    act_mae = float(np.mean(act_maes))
    # act_diff = mean Action Component 값의 verify 대비 diff (부호 방향)
    v_mean = float(np.mean([verify_data[(sid, 'Action Component')].mean() for sid in sids]))
    c_mean = float(np.mean([d[(sid, 'Action Component')].mean() for sid in sids]))
    act_diff = c_mean - v_mean

    predicted_lb = (COEF_INTERCEPT
                    + COEF_VID_L2 * vid_l2
                    + COEF_DINO_L2 * dino_l2
                    + COEF_ACT_MAE * act_mae
                    + COEF_ACT_DIFF * act_diff)

    return {
        'vid_l2': vid_l2,
        'dino_l2': dino_l2,
        'act_mae': act_mae,
        'act_diff': act_diff,
        'act_mean': c_mean,
        'predicted_lb': predicted_lb,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    verify_data = load(VERIFY)
    verify_act_mean = float(np.mean([verify_data[k].mean() for k in verify_data if k[1] == 'Action Component']))

    print(f"{'CSV':60s} {'pred_LB':>9s} {'act_mae':>9s} {'act_diff':>9s} {'vid_l2':>7s} {'dino_l2':>8s} {'act_mean':>9s}")
    print(f"{'verify (baseline)':60s} {'0.2493':>9s} {'0':>9s} {'0':>9s} {'0':>7s} {'0':>8s} {verify_act_mean:>9.4f}")
    print('-' * 120)

    for csv_path in sys.argv[1:]:
        r = compute_features(csv_path, verify_data)
        marker = '  ← 개선 예측' if r['predicted_lb'] < 0.2493 else ('  ← 악화 예측' if r['predicted_lb'] > 0.25 else '  ← 미미')
        print(f"{Path(csv_path).name:60s} {r['predicted_lb']:>9.4f} {r['act_mae']:>9.4f} {r['act_diff']:>+9.4f} {r['vid_l2']:>7.3f} {r['dino_l2']:>8.3f} {r['act_mean']:>9.4f}{marker}")


if __name__ == "__main__":
    main()
