"""Continuous watcher for multi-seed / Wan pilot / new candidate results.

무한 loop, 사용자 KILL까지:
- 매 60s 폴링
- Multi-seed 후보 완료 감지 → rejection_picker 자동 실행 → CSV 생성 → Action mean 리포트
- Wan pilot 완료 감지 → E-invdyn L1 계산 → kill 판정
- 결과를 auto_watcher.log에 기록
"""
from __future__ import annotations

import datetime
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home1/sota/inha2026")
LOG = ROOT / "logs/fresh/auto_watcher.log"
SUB = ROOT / "submission_kit/fresh"

B4_SEEDS = [f"submission_b4_seed{i}.csv" for i in range(1, 8)]
W0025_SEEDS = [f"submission_w0025_seed{i}.csv" for i in range(1, 6)]

B4_BASELINE = "submission_b4_step6000.csv"
W0025_BASELINE = "submission_b4_sw_w0025_step12000.csv"

PICKER = ROOT / "scripts/track_extreme/rejection_picker.py"

STATE = ROOT / "logs/fresh/auto_watcher_state.json"


def now(): return datetime.datetime.now().isoformat(timespec="seconds")


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {msg}\n"
    print(line, end="", flush=True)
    with open(LOG, "a") as f:
        f.write(line)


def load_state():
    if STATE.exists():
        return json.load(open(STATE))
    return {"started": now(), "done_events": []}


def save_state(s):
    with open(STATE, "w") as f:
        json.dump(s, f, indent=2, default=str)


def action_mean(csv):
    p = SUB / csv
    if not p.exists():
        return None
    try:
        import pandas as pd, numpy as np
        df = pd.read_csv(p)
        act = df[df["feature_component"] == "Action Component"]
        arr = np.array([json.loads(s) for s in act["feature_json"]]).squeeze()
        return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                "pred_lb": float(0.4 * arr.mean() + 0.089)}
    except Exception as e:
        return {"error": str(e)}


def run_rejection_picker(pool_dirs, out_dir, out_csv, tag):
    cmd = [
        "sbatch", "--parsable",
        "--gres=gpu:A6000_ada:1", "--exclude=gpu-113", "--mem=40G", "--time=1:00:00",
        f"--job-name=rp_{tag[:12]}",
        f"--output=/home1/sota/inha2026/logs/fresh/rp_{tag}_%j.log",
        f"--error=/home1/sota/inha2026/logs/fresh/rp_{tag}_%j.err",
        f"--wrap=/home1/sota/anaconda3/envs/inha2026/bin/python {PICKER} "
        f"--pool-dirs {' '.join(str(p) for p in pool_dirs)} "
        f"--out-dir {out_dir} --out-csv {out_csv}"
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"[picker ERR] {tag}: {r.stderr[:300]}")
        return -1
    try:
        return int(r.stdout.strip())
    except Exception:
        return -1


def check_multiseed(name, baseline_csv, seed_csvs, state):
    """모든 seed CSV 완료 시 rejection picker 자동 실행."""
    event_key = f"multiseed_{name}"
    if event_key in state["done_events"]:
        return
    if not (SUB / baseline_csv).exists():
        return
    missing = [s for s in seed_csvs if not (SUB / s).exists()]
    if missing:
        return  # 아직 대기

    # 모든 CSV 완료. Action mean 계산
    log(f"[multiseed_{name}] all seeds ready")
    all_csvs = [baseline_csv] + seed_csvs
    action_means = {}
    for c in all_csvs:
        am = action_mean(c)
        if am and "error" not in am:
            action_means[c] = am
            log(f"  {c}: Action={am['mean']:.4f} pred_LB={am['pred_lb']:.4f}")

    # 통합 rejection picker 실행 → sample당 best
    pool_dirs = []
    for c in all_csvs:
        d = SUB / c.replace("submission_", "input_videos_").replace(".csv", "")
        if d.exists():
            pool_dirs.append(d)
    out_dir = SUB / f"input_videos_rej_{name}"
    out_csv = SUB / f"submission_rej_{name}.csv"
    if not out_csv.exists():
        jid = run_rejection_picker(pool_dirs, out_dir, out_csv, name)
        log(f"[rejection_picker] {name} launched job {jid}")
    state["done_events"].append(event_key)


def check_wan_pilot(state):
    if "wan_pilot_scored" in state["done_events"]:
        return
    pilot_dir = SUB / "input_videos_wan22_pilot"
    if not pilot_dir.exists():
        return
    mp4s = list(pilot_dir.glob("*.mp4"))
    if len(mp4s) < 12:
        return
    log(f"[wan_pilot] {len(mp4s)} mp4s ready → scoring")

    # E-invdyn L1 + AlexNet feature 계산
    b4_dir = SUB / "input_videos_b4_step6000"
    scorer = ROOT / "scripts/track_extreme/pareto_score.py"
    out_json = ROOT / "logs/fresh/pareto/wan22_pilot.json"
    cmd = [
        "sbatch", "--parsable",
        "--gres=gpu:A6000_ada:1", "--exclude=gpu-113", "--mem=40G", "--time=0:30:00",
        "--job-name=wan_pareto",
        "--output=/home1/sota/inha2026/logs/fresh/wan_pareto_%j.log",
        "--error=/home1/sota/inha2026/logs/fresh/wan_pareto_%j.err",
        f"--wrap=/home1/sota/anaconda3/envs/inha2026/bin/python {scorer} "
        f"--videos {pilot_dir} --out-json {out_json} --samples 12 --ref-videos {b4_dir}"
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        jid = r.stdout.strip()
        log(f"[wan_pareto] launched job {jid}")
        state["done_events"].append("wan_pilot_scored")


def check_pareto_summary(state):
    """pareto 결과가 오면 로그에 정리."""
    d = ROOT / "logs/fresh/pareto"
    if not d.exists():
        return
    for p in d.glob("*.json"):
        if p.name == "SUMMARY.json":
            continue
        marker = f"pareto_read_{p.stem}"
        if marker in state["done_events"]:
            continue
        try:
            s = json.load(open(p))
            log(f"[pareto] {p.stem}: action_l1={s.get('action_l1_mean',0):.3f} "
                f"alex_cos={s.get('alex_cosine_vs_ref','n/a')} "
                f"n={s.get('n',0)}")
            state["done_events"].append(marker)
        except Exception:
            pass


def main():
    log("=" * 60)
    log("Auto watcher START")
    state = load_state()
    tick = 0
    while True:
        tick += 1
        try:
            check_multiseed("b4", B4_BASELINE, B4_SEEDS, state)
            check_multiseed("w0025", W0025_BASELINE, W0025_SEEDS, state)
            check_wan_pilot(state)
            check_pareto_summary(state)
        except Exception as e:
            log(f"[tick {tick} err] {e}")
        save_state(state)
        time.sleep(60)


if __name__ == "__main__":
    main()
