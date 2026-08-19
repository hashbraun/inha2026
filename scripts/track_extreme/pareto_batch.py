"""Auto loop EXTREME 결과 CSV 폴더들을 순차로 pareto_score 처리.

- auto_loop_extreme_state.json 읽어 각 candidate 의 mp4 폴더 찾기
- 이미 pareto json 있는 것은 skip
- 하나씩 순차 처리 (같은 GPU 재사용)
- 결과: /home1/sota/inha2026/logs/fresh/pareto_{name}.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home1/sota/inha2026")
STATE = ROOT / "logs/fresh/auto_loop_extreme_state.json"
SUB_DIR = ROOT / "submission_kit/fresh"
OUT_DIR = ROOT / "logs/fresh/pareto"
SCORE_SCRIPT = ROOT / "scripts/track_extreme/pareto_score.py"
REF_VIDEOS = SUB_DIR / "input_videos_b4_step6000"  # AlexNet cosine 기준 (도전본)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="이미 있는 json도 덮어쓰기")
    ap.add_argument("--samples", type=int, default=216)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE.exists():
        print(f"NO state: {STATE}")
        return 1
    state = json.load(open(STATE))

    todo = []
    for name in state["candidates"]:
        videos = SUB_DIR / f"input_videos_{name}"
        if not videos.exists():
            continue
        n_mp4 = len(list(videos.glob("*.mp4")))
        if n_mp4 < args.samples * 0.9:
            print(f"[skip] {name}: only {n_mp4}/{args.samples} mp4")
            continue
        out_json = OUT_DIR / f"{name}.json"
        if out_json.exists() and not args.force:
            print(f"[done] {name} → {out_json.name}")
            continue
        todo.append((name, videos, out_json))

    # Reference baseline 먼저 추가
    ref_json = OUT_DIR / "b4_step6000.json"
    if not ref_json.exists() or args.force:
        todo.insert(0, ("b4_step6000", REF_VIDEOS, ref_json))

    print(f"[queue] {len(todo)} candidates to score")
    for name, videos, out_json in todo:
        cmd = [
            "/home1/sota/anaconda3/envs/inha2026/bin/python",
            str(SCORE_SCRIPT),
            "--videos", str(videos),
            "--out-json", str(out_json),
            "--samples", str(args.samples),
            "--ref-videos", str(REF_VIDEOS),
        ]
        print(f"[run] {name}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            print(f"[FAIL] {name}: {r.stderr[-400:]}", flush=True)
            continue
        try:
            with open(out_json) as f:
                s = json.load(f)
            print(f"  action_l1={s['action_l1_mean']:.3f} temp={s['temporal_diff_mean']:.2f} "
                  f"ff={s['first_frame_l2_mean']:.2f} alex_vs_ref={s.get('alex_cosine_vs_ref','n/a')}",
                  flush=True)
        except Exception as e:
            print(f"[parse err] {name}: {e}", flush=True)

    # 요약 표 저장
    summary_path = OUT_DIR / "SUMMARY.json"
    summary = {}
    for p in sorted(OUT_DIR.glob("*.json")):
        if p.name == "SUMMARY.json":
            continue
        try:
            summary[p.stem] = json.load(open(p))
        except Exception:
            pass
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY] {summary_path}")
    # 정렬해서 CLI에도 print
    ranked = sorted(
        summary.items(),
        key=lambda kv: kv[1].get("action_l1_mean", 1e9)
    )
    print(f"\n{'name':30s} {'action_l1':>10s} {'temp_diff':>10s} {'ff_l2':>8s} {'alex_vs_ref':>12s}")
    for name, s in ranked:
        print(f"{name:30s} {s.get('action_l1_mean',0):10.3f} "
              f"{s.get('temporal_diff_mean',0):10.3f} "
              f"{s.get('first_frame_l2_mean',0):8.3f} "
              f"{s.get('alex_cosine_vs_ref','n/a'):>12}")


if __name__ == "__main__":
    sys.exit(main())
