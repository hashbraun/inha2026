"""Auto loop v2: Codex 자문 통합 + 개선될 때까지 pivot.

기능:
1. 여러 candidate 학습/inference 감시 (SLURM jobs)
2. 학습 완료 → 자동 inference launch (다중 ckpt)
3. Inference 완료 → CSV Action mean 계산
4. B4 (0.2407 실측)보다 좋은 Action mean 발견 → 사용자 실측 slot 요청
5. **각 candidate settle 시 codex 자문 자동 launch** — 로그+state embed
6. Codex 조언 기반 pivot 자동 (또는 pre-defined queue)

**중단 조건**:
- Video+Action 종합적으로 크게 개선된 candidate 발견 (예: 실측 LB < 0.235)
- 사용자 kill

state: /home1/sota/inha2026/logs/fresh/auto_loop_v2_state.json
log:   /home1/sota/inha2026/logs/fresh/auto_loop_v2.log
codex 자문: /home1/sota/inha2026/docs/plans/fresh_restart/codex_pivot_{tick}.txt
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import time
from pathlib import Path

LOG = Path("/home1/sota/inha2026/logs/fresh/auto_loop_v2.log")
STATE = Path("/home1/sota/inha2026/logs/fresh/auto_loop_v2_state.json")
CODEX_DIR = Path("/home1/sota/inha2026/docs/plans/fresh_restart")

B4_ACTION_MEAN = 0.3798
B4_LB = 0.2407
BASELINE_LB = 0.24524
SUCCESS_ACTION = 0.375     # Codex 예상 LB 무효 → 실측 필요, action mean 기준만 활용
BIG_SUCCESS_ACTION = 0.35  # 이것 발견 시 사용자 실측 확정 요청

# Candidate config
CANDIDATES = {
    "b4_sw_w0025": {
        "job_id": 30749, "ckpt_dir": "/home1/sota/inha2026/checkpoints/b4sw_bw_w0025",
        "ckpt_pattern": "ckpt_step*.pt", "adapter_type": "b4",
        "inf_steps": [4000, 6000, 8000, 10000, 12000],
        "description": "B4 sweep weight=0.025",
    },
    "b4_sw_w005": {
        "job_id": 30757, "ckpt_dir": "/home1/sota/inha2026/checkpoints/b4sw_bw_w005",
        "ckpt_pattern": "ckpt_step*.pt", "adapter_type": "b4",
        "inf_steps": [4000, 6000, 8000, 10000, 12000],
        "description": "B4 sweep weight=0.05 (=B4 성공 setting 재현)",
    },
    "b4_sw_w0075": {
        "job_id": 30758, "ckpt_dir": "/home1/sota/inha2026/checkpoints/b4sw_bw_w0075",
        "ckpt_pattern": "ckpt_step*.pt", "adapter_type": "b4",
        "inf_steps": [4000, 6000, 8000, 10000, 12000],
        "description": "B4 sweep weight=0.075",
    },
    "b4_original_500": {"job_id": 30759, "inf_only": True, "adapter_type": "b4",
                        "csv": "/home1/sota/inha2026/submission_kit/fresh/submission_b4_step500.csv"},
    "b4_original_1000": {"job_id": 30760, "inf_only": True, "adapter_type": "b4",
                         "csv": "/home1/sota/inha2026/submission_kit/fresh/submission_b4_step1500.csv"},
    "b4_original_1500": {"job_id": 30761, "inf_only": True, "adapter_type": "b4",
                         "csv": "/home1/sota/inha2026/submission_kit/fresh/submission_b4_step1500.csv"},
}


def now(): return datetime.datetime.now().isoformat(timespec="seconds")


def log(msg: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {msg}\n"
    print(line, end="", flush=True)
    with open(LOG, "a") as f:
        f.write(line)


def load_state():
    if STATE.exists():
        return json.load(open(STATE))
    return {
        "started": now(),
        "candidates": {n: {**c, "phase": "training", "results": {}, "inf_jobs": {}}
                       for n, c in CANDIDATES.items()},
        "best_challenger": {"csv": "submission_b4_step6000.csv", "action": B4_ACTION_MEAN, "lb": B4_LB},
        "success_candidates": [],
        "codex_reviews": [],
    }


def save_state(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(s, f, indent=2, ensure_ascii=False, default=str)


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def job_running(job_id):
    return len(sh(f"squeue -j {job_id} -h -o '%T' 2>/dev/null")) > 0


def compute_action_mean(csv_path):
    if not Path(csv_path).exists():
        return None
    try:
        import pandas as pd, numpy as np
        df = pd.read_csv(csv_path)
        act = df[df["feature_component"] == "Action Component"]
        arr = np.array([json.loads(s) for s in act["feature_json"]]).squeeze()
        return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                "predicted_lb": float(0.4 * arr.mean() + 0.089)}
    except Exception as e:
        return {"error": str(e)}


def launch_inference(ckpt_path, out_dir, csv_out, adapter_type, tag):
    if adapter_type == "b4":
        script = "/home1/sota/inha2026/scripts/infer_cosmos3_nano.py"
        args = f"--ckpt {ckpt_path} --samples 216 --steps 35 --guidance 6.0 --rank 32 --action-repr delta_base --out-dir {out_dir}"
    else:
        log(f"unknown adapter_type: {adapter_type}")
        return -1
    sbatch = f"""#!/bin/bash
#SBATCH --job-name=inf_{tag[:15]}
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/inf_{tag}_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/inf_{tag}_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p {out_dir}
/home1/sota/anaconda3/envs/inha2026/bin/python {script} {args}
cd /home1/sota/inha2026/submission_kit
/home1/sota/anaconda3/envs/inha2026/bin/python make_submission_csv.py \\
    --prediction-root {Path(out_dir).relative_to(Path('/home1/sota/inha2026/submission_kit'))} \\
    --output-csv {Path(csv_out).relative_to(Path('/home1/sota/inha2026/submission_kit'))}
"""
    p = Path(f"/tmp/inf_{tag}.sbatch")
    p.write_text(sbatch)
    out = sh(f"sbatch {p}")
    try:
        return int(out.split()[-1])
    except Exception:
        return -1


def launch_codex_pivot(tick: int, state: dict, event: str):
    """Codex 자문: pivot 시점, 로그+state embed."""
    out_file = CODEX_DIR / f"codex_pivot_tick{tick}.txt"
    tmp = Path(f"/tmp/codex_pivot_tick{tick}.txt")

    # State snapshot
    state_str = json.dumps(state, indent=2, ensure_ascii=False, default=str)[:8000]
    # Recent log
    recent_log = ""
    if LOG.exists():
        recent_log = "\n".join(LOG.read_text().splitlines()[-80:])

    prompt = f"""CONTEXT: inha2026 D-day 임박. Auto loop v2 tick {tick} pivot 시점.

## 이벤트
{event}

## 실측 확정
- v5b_ANCH: 0.24524 (안전본)
- B4 step6000: 0.2407 (도전본)
- B4v2: 0.2508 (실패)
- Z6-B4 hybrid: 0.2499 (실패)

## Auto loop 최근 로그
```
{recent_log}
```

## State snapshot
```json
{state_str}
```

## 물음 (roundtable)
1. 이 pivot 시점에서 다음 후보 우선순위?
2. 현재 결과 중 개선 신호 있는가 (Action + Visual 종합)?
3. 실측 slot 사용할 만한 candidate?
4. 다음 학습 launch 권장? 어떤 것?
5. 마감 안전 관점에서 위험한 시도가 있는가?

brutally honest. 이전 자문 (`codex_break_b4.txt`, `codex_plan_review.txt`, `codex_log_review.txt`)의 조언 재확인. 짧게 (300 words 이내).
"""
    tmp.write_text(prompt)
    cmd = f"nohup bash -c 'cat {tmp} | /home1/sota/anaconda3/bin/codex exec --sandbox danger-full-access --skip-git-repo-check' > {out_file} 2>&1 &"
    subprocess.Popen(cmd, shell=True)
    log(f"[codex] pivot tick{tick} launched → {out_file}")
    state.setdefault("codex_reviews", []).append({"tick": tick, "file": str(out_file), "event": event})


def tick_candidate(name: str, cand: dict, state: dict) -> str | None:
    """각 candidate phase 진행. 이벤트 리턴 (pivot signal)."""
    phase = cand.get("phase", "training")

    # Inference-only candidate (B4 원본 세부 ckpt)
    if cand.get("inf_only"):
        job_id = cand["job_id"]
        if job_running(job_id):
            return None
        csv = cand.get("csv")
        if csv and Path(csv).exists():
            am = compute_action_mean(csv)
            if am and "error" not in am:
                cand["result"] = am
                cand["phase"] = "done"
                delta_vs_b4 = am["mean"] - B4_ACTION_MEAN
                log(f"[{name}] Action={am['mean']:.4f} (vs B4 {delta_vs_b4:+.4f})")
                if am["mean"] < BIG_SUCCESS_ACTION:
                    state["success_candidates"].append({"name": name, **am})
                    return f"BIG SUCCESS: {name} Action={am['mean']:.4f}"
                elif am["mean"] < B4_ACTION_MEAN:
                    return f"marginal: {name} Action={am['mean']:.4f}"
        return None

    # Training candidate
    if phase == "training":
        if job_running(cand["job_id"]):
            return None
        # 학습 완료 → inference launch
        ckpt_dir = Path(cand["ckpt_dir"])
        if not ckpt_dir.exists():
            log(f"[{name}] ckpt_dir missing → wait")
            return None
        launched = 0
        for step in cand["inf_steps"]:
            ckpt_name = f"ckpt_step{step:06d}.pt"
            ckpt_p = ckpt_dir / ckpt_name
            if not ckpt_p.exists():
                continue
            tag = f"{name}_step{step}"
            out_dir = Path(f"/home1/sota/inha2026/submission_kit/fresh/input_videos_{tag}")
            csv_out = Path(f"/home1/sota/inha2026/submission_kit/fresh/submission_{tag}.csv")
            if csv_out.exists():
                cand["inf_jobs"][ckpt_name] = {"done": True, "csv": str(csv_out)}
                continue
            info = cand["inf_jobs"].get(ckpt_name, {})
            if info.get("job_id") and job_running(info["job_id"]):
                continue
            jid = launch_inference(ckpt_p, out_dir, csv_out, cand["adapter_type"], tag)
            cand["inf_jobs"][ckpt_name] = {"job_id": jid, "csv": str(csv_out)}
            log(f"[{name}] {ckpt_name} inf job {jid} launched")
            launched += 1
        if launched == 0 and cand["inf_jobs"] and all(
            v.get("done") or (v.get("job_id") and not job_running(v["job_id"]))
            for v in cand["inf_jobs"].values()
        ):
            cand["phase"] = "evaluate"
        return None

    if phase == "evaluate":
        best = None
        for ckpt_name, info in cand["inf_jobs"].items():
            csv = info.get("csv")
            am = compute_action_mean(csv) if csv else None
            if not am or "error" in am:
                continue
            cand["results"][ckpt_name] = am
            log(f"[{name}] {ckpt_name}: Action={am['mean']:.4f}")
            if best is None or am["mean"] < best["mean"]:
                best = {**am, "ckpt": ckpt_name}
        if best is None:
            cand["phase"] = "failed"
            return f"{name} failed (no valid results)"
        cand["best"] = best
        delta = best["mean"] - B4_ACTION_MEAN
        if best["mean"] < BIG_SUCCESS_ACTION:
            state["success_candidates"].append({"name": name, **best})
            cand["phase"] = "big_success"
            return f"BIG SUCCESS: {name} best {best['mean']:.4f}"
        elif best["mean"] < B4_ACTION_MEAN:
            cand["phase"] = "marginal"
            return f"marginal: {name} best {best['mean']:.4f} (vs B4 {delta:+.4f})"
        else:
            cand["phase"] = "failed"
            return f"{name} failed (best {best['mean']:.4f})"


def main():
    log("=" * 60)
    log("Auto loop v2 START (codex pivot integration)")
    state = load_state()
    save_state(state)

    tick = 0
    pivot_events = []
    while True:
        tick += 1
        log(f"--- tick {tick} ---")
        events_this_tick = []
        for name, cand in state["candidates"].items():
            if cand.get("phase") in ("done", "failed", "big_success", "marginal"):
                continue
            try:
                ev = tick_candidate(name, cand, state)
                if ev:
                    events_this_tick.append(ev)
                    log(f"[event] {ev}")
            except Exception as e:
                log(f"[{name}] tick error: {e}")

        save_state(state)

        # 새 pivot 이벤트 → Codex 자문 launch
        if events_this_tick:
            event_str = " | ".join(events_this_tick)
            pivot_events.append(event_str)
            launch_codex_pivot(tick, state, event_str)
            save_state(state)

        # Big success 알림 (계속 진행)
        if state["success_candidates"]:
            log(f"BIG SUCCESS candidates: {state['success_candidates']}")
            log(">>> 사용자 실측 slot 확인 필요. Loop 계속 유지.")

        # 모든 candidate settled → 새 후보 자동 launch (loop 유지)
        active = [n for n, c in state["candidates"].items()
                  if c.get("phase") not in ("done", "failed", "big_success", "marginal")]
        if not active:
            log(f"모든 candidate settled: {[(n, c.get('phase')) for n, c in state['candidates'].items()]}")
            # 새 후보 자동 추가 (queue)
            added = maybe_add_next_candidate(state)
            if added:
                log(f"[queue] 새 후보 추가: {added}. Loop 계속.")
            else:
                log("[queue] 대기 상태. 5분 후 재확인.")

        time.sleep(600)   # 10분 tick — 사용자 KILL로만 종료

    log("Auto loop v2 END (should never reach)")


# 후보 queue (settled 후 자동 launch될 추가 후보 sbatch 정의)
_ADDITIONAL_CANDIDATE_QUEUE = [
    {
        "name": "b4_w005_seed43",
        "description": "w=0.05 성공 basin seed 43 reproduction (Codex 권장 #2)",
        "sbatch": "/home1/sota/inha2026/fresh/scripts/b4_seed_repro_v2.sbatch",
        "ckpt_dir": "/home1/sota/inha2026/checkpoints/b4_w005_seed43",
        "ckpt_pattern": "ckpt_step*.pt", "adapter_type": "b4",
        "inf_steps": [4000, 6000, 8000],
    },
    {
        "name": "b4_w005_seed44",
        "description": "w=0.05 성공 basin seed 44 reproduction",
        "sbatch": "/home1/sota/inha2026/fresh/scripts/b4_seed_repro_v3.sbatch",
        "ckpt_dir": "/home1/sota/inha2026/checkpoints/b4_w005_seed44",
        "ckpt_pattern": "ckpt_step*.pt", "adapter_type": "b4",
        "inf_steps": [4000, 6000, 8000],
    },
]


def maybe_add_next_candidate(state: dict) -> str | None:
    """대기 queue에서 다음 후보 launch. 이미 있는 것은 skip."""
    for tmpl in _ADDITIONAL_CANDIDATE_QUEUE:
        name = tmpl["name"]
        if name in state["candidates"]:
            continue
        sbatch = tmpl.get("sbatch")
        if not sbatch or not Path(sbatch).exists():
            log(f"[queue] sbatch missing: {sbatch}")
            continue
        # Blackwell 여유 확인 (동시 1개만)
        blackwell_busy = "gpu-113" in sh("squeue -u sota -h -o '%R' 2>/dev/null")
        if blackwell_busy:
            log(f"[queue] Blackwell busy — {name} defer")
            return None
        out = sh(f"sbatch {sbatch}")
        try:
            jid = int(out.split()[-1])
        except Exception:
            log(f"[queue] {name} launch failed: {out}")
            continue
        state["candidates"][name] = {
            "job_id": jid, "phase": "training", "results": {}, "inf_jobs": {},
            "ckpt_dir": tmpl["ckpt_dir"], "ckpt_pattern": tmpl["ckpt_pattern"],
            "adapter_type": tmpl["adapter_type"], "inf_steps": tmpl["inf_steps"],
            "description": tmpl["description"],
        }
        return f"{name} (job {jid})"
    return None


if __name__ == "__main__":
    main()
