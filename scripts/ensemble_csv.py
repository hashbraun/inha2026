"""여러 CSV의 feature_json 값을 평균해서 앙상블 CSV 생성.

사용:
    python ensemble_csv.py --inputs a.csv b.csv --output ensemble.csv [--weights 0.6 0.4]
"""
import argparse, csv, json
from pathlib import Path
import numpy as np


def load(path):
    rows = {}
    header_order = []
    with open(path) as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        for row in r:
            key = (row['sample_id'], row['feature_component'])
            rows[key] = np.array(json.loads(row['feature_json']), dtype=np.float32)
            header_order.append((row['sample_id'], row['feature_component']))
    # dedupe order
    seen = set()
    order = [k for k in header_order if not (k in seen or seen.add(k))]
    return rows, order, fieldnames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--weights', nargs='+', type=float, default=None,
                    help='입력별 가중치. 미지정 시 균등')
    ap.add_argument('--precision', type=int, default=6)
    args = ap.parse_args()

    n = len(args.inputs)
    weights = args.weights if args.weights else [1.0 / n] * n
    assert len(weights) == n, f'weights {len(weights)} != inputs {n}'
    s = sum(weights)
    weights = [w / s for w in weights]  # 정규화
    print(f'입력 {n}개, 가중치 {weights}')

    datas = [load(p) for p in args.inputs]
    ref_rows, order, fieldnames = datas[0]
    for i, (rows, _, _) in enumerate(datas[1:], 1):
        for key in ref_rows:
            if key not in rows:
                raise KeyError(f'{args.inputs[i]}에 {key} 없음')

    print(f'{len(order)} rows × {n} inputs 앙상블 시작')

    with open(args.output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for key in order:
            merged = np.zeros_like(datas[0][0][key])
            for (rows, _, _), wt in zip(datas, weights):
                merged += wt * rows[key]
            # feature_json 다시 저장
            arr = merged.astype(np.float32)
            fmt = f'%.{args.precision}f'
            # 원래 shape 유지: 리스트 in 리스트
            v = json.dumps(arr.tolist(), separators=(',', ':'))
            row = [key[0], key[1], v]
            w.writerow(row)

    print(f'저장: {args.output}')


if __name__ == '__main__':
    main()
