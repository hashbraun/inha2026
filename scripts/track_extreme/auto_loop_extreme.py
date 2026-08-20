"""Auto loop EXTREME: codex 승인 트랙 (weight interpolation + CFG/steps sweep).

Codex tick40 조언 반영:
- P2 후처리 폐기 (mp4 수정 판정, 규정 위반 위험)
- P3 zero-shot backbone 폐기 (upside 없음, action grounding 부재)
- 놓친 시도 = weight interpolation → 최우선
- CFG/steps 소형 sweep (200-step/sampler 교체 금지)
- Pareto gate 자체 판독기 (E-invdyn + clean-room visual + action sensitivity + schema)

기존 auto_loop_v2와 격리 (별도 state/log/candidate queue).
"""
from __future__ import annotations

import datetime
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home1/sota/inha2026")
LOG = ROOT / "logs/fresh/auto_loop_extreme.log"
STATE = ROOT / "logs/fresh/auto_loop_extreme_state.json"
CODEX_DIR = ROOT / "docs/plans/fresh_restart"
INTERP_DIR = ROOT / "checkpoints/interp"
SUB_DIR = ROOT / "submission_kit/fresh"
INFER_SCRIPT = ROOT / "scripts/infer_cosmos3_nano.py"
INTERP_SCRIPT = ROOT / "scripts/track_extreme/weight_interp.py"
MAKECSV = ROOT / "submission_kit/make_submission_csv.py"

CKPT_V5B = ROOT / "checkpoints/cosmos3_nano_v5_base/ckpt_step020000.pt"     # 안전본 backbone (LB 0.24524)
CKPT_B4 = ROOT / "checkpoints/v5b_eloss_bw/ckpt_step006000.pt"              # 도전본 원본 (LB 0.2407)
CKPT_W0025 = ROOT / "checkpoints/b4sw_bw_w0025/ckpt_step012000.pt"          # 신규 후보 (Action 0.3647)

B4_ACTION_MEAN = 0.3798
B4_LB = 0.2407
BASELINE_LB = 0.24524
BIG_SUCCESS_ACTION = 0.35

# ---- Candidate queue ------------------------------------------------------

# 각 candidate: interp_from(ckpt_a, ckpt_b, alpha) OR cfg_sweep(ckpt, guidance, steps)
CANDIDATES = {
    # === Weight interpolation (5) — Codex 최우선 놓친 시도 ===
    "interp_b4_w0025_a25": {"type": "interp", "a": CKPT_B4, "b": CKPT_W0025, "alpha": 0.25,
                             "desc": "0.25*B4 + 0.75*w0025 (w0025 우세)"},
    "interp_b4_w0025_a50": {"type": "interp", "a": CKPT_B4, "b": CKPT_W0025, "alpha": 0.50,
                             "desc": "0.5*B4 + 0.5*w0025 (균형)"},
    "interp_b4_w0025_a75": {"type": "interp", "a": CKPT_B4, "b": CKPT_W0025, "alpha": 0.75,
                             "desc": "0.75*B4 + 0.25*w0025 (B4 우세)"},
    "interp_v5b_w0025_a50": {"type": "interp", "a": CKPT_V5B, "b": CKPT_W0025, "alpha": 0.50,
                              "desc": "0.5*v5b + 0.5*w0025 (softer w0025)"},
    "interp_v5b_b4_a50": {"type": "interp", "a": CKPT_V5B, "b": CKPT_B4, "alpha": 0.50,
                          "desc": "0.5*v5b + 0.5*B4 (softer B4)"},

    # === CFG/steps sweep on w0025/12k (가장 유망한 신규) ===
    "w0025_cfg45_s35": {"type": "cfg", "ckpt": CKPT_W0025, "guidance": 4.5, "steps": 35,
                        "desc": "w0025 낮은 CFG"},
    "w0025_cfg55_s35": {"type": "cfg", "ckpt": CKPT_W0025, "guidance": 5.5, "steps": 35,
                        "desc": "w0025 CFG 5.5"},
    "w0025_cfg75_s35": {"type": "cfg", "ckpt": CKPT_W0025, "guidance": 7.5, "steps": 35,
                        "desc": "w0025 높은 CFG"},
    "w0025_cfg60_s50": {"type": "cfg", "ckpt": CKPT_W0025, "guidance": 6.0, "steps": 50,
                        "desc": "w0025 baseline CFG 더 많은 step"},

    # === CFG/steps sweep on B4 (기존 도전본 최적화) ===
    "b4_cfg45_s35": {"type": "cfg", "ckpt": CKPT_B4, "guidance": 4.5, "steps": 35,
                     "desc": "B4 낮은 CFG"},
    "b4_cfg55_s35": {"type": "cfg", "ckpt": CKPT_B4, "guidance": 5.5, "steps": 35,
                     "desc": "B4 CFG 5.5"},
    "b4_cfg75_s35": {"type": "cfg", "ckpt": CKPT_B4, "guidance": 7.5, "steps": 35,
                     "desc": "B4 높은 CFG"},
    "b4_cfg60_s50": {"type": "cfg", "ckpt": CKPT_B4, "guidance": 6.0, "steps": 50,
                     "desc": "B4 baseline CFG 더 많은 step"},
}

MAX_PARALLEL_JOBS = 6  # 다른 사용자 배려


# ---- utilities ----

def now(): return datetime.datetime.now().isoformat(timespec="seconds")


def log(msg: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {msg}\n"
    print(line, end="", flush=True)
    with open(LOG, "a") as f:
        f.write(line)


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        return f"__err__:{e}"


def job_running(job_id):
    return len(sh(f"squeue -j {job_id} -h -o '%T' 2>/dev/null")) > 0


def my_running_extreme_jobs(state):
    n = 0
    for c in state["candidates"].values():
        jid = c.get("job_id")
        if jid and job_running(jid):
            n += 1
    return n


def load_state():
    if STATE.exists():
        return json.load(open(STATE))
    return {
        "started": now(),
        "candidates": {n: {**c, "phase": "pending", "ckpt": str(c.get("ckpt", "")),
                           "a": str(c.get("a", "")), "b": str(c.get("b", ""))}
                        for n, c in CANDIDATES.items()},
        "best_challenger": {"csv": "submission_b4_step6000.csv", "action": B4_ACTION_MEAN, "lb": B4_LB},
        "success": [],
        "codex_reviews": [],
    }


def save_state(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(s, f, indent=2, ensure_ascii=False, default=str)


def compute_action_mean(csv_path):
    if not Path(csv_path).exists():
        return None
    try:
        import pandas as pd
        import numpy as np
        df = pd.read_csv(csv_path)
        act = df[df["feature_component"] == "Action Component"]
        arr = np.array([json.loads(s) for s in act["feature_json"]]).squeeze()
        return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                "predicted_lb": float(0.4 * arr.mean() + 0.089)}
    except Exception as e:
        return {"error": str(e)}


def build_interp_ckpt(name, cand):
    """weight_interp.py 로 interp ckpt 생성. 이미 있으면 skip."""
    out = INTERP_DIR / f"{name}.pt"
    if out.exists():
        return out
    alpha = cand["alpha"]
    a = cand["a"] if isinstance(cand["a"], str) else str(cand["a"])
    b = cand["b"] if isinstance(cand["b"], str) else str(cand["b"])
    cmd = (f"/home1/sota/anaconda3/envs/inha2026/bin/python {INTERP_SCRIPT} "
           f"--ckpt-a {a} --ckpt-b {b} --alpha {alpha} --out {out}")
    log(f"[{name}] interp build: alpha={alpha}")
    r = sh(cmd, timeout=180)
    if not out.exists():
        log(f"[{name}] interp FAILED: {r[:200]}")
        return None
    return out


def launch_inference(name, ckpt, guidance, steps, tag):
    """A6000_ada inference sbatch launch."""
    out_dir = SUB_DIR / f"input_videos_{tag}"
    csv_out = SUB_DIR / f"submission_{tag}.csv"
    if csv_out.exists():
        return -2  # already done
    args = (f"--ckpt {ckpt} --samples 216 --steps {steps} --guidance {guidance} "
            f"--rank 32 --action-repr delta_base --out-dir {out_dir}")
    sbatch = f"""#!/bin/bash
#SBATCH --job-name=ex_{tag[:15]}
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/ex_{tag}_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/ex_{tag}_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p {out_dir}
/home1/sota/anaconda3/envs/inha2026/bin/python {INFER_SCRIPT} {args}
cd /home1/sota/inha2026/submission_kit
/home1/sota/anaconda3/envs/inha2026/bin/python make_submission_csv.py \\
    --prediction-root fresh/input_videos_{tag} \\
    --output-csv fresh/submission_{tag}.csv
"""
    p = Path(f"/tmp/ex_{tag}.sbatch")
    p.write_text(sbatch)
    out = sh(f"sbatch {p}")
    try:
        return int(out.split()[-1])
    except Exception:
        log(f"[{name}] launch fail: {out[:200]}")
        return -1


# ---- codex integration ----

def launch_codex_pivot(tick, state, event):
    out_file = CODEX_DIR / f"codex_extreme_tick{tick}.txt"
    tmp = Path(f"/tmp/codex_extreme_tick{tick}.txt")
    state_str = json.dumps(state, indent=2, ensure_ascii=False, default=str)[:6000]
    recent_log = ""
    if LOG.exists():
        recent_log = "\n".join(LOG.read_text().splitlines()[-60:])
    prompt = f"""CONTEXT: inha2026 auto loop EXTREME tick {tick} pivot 시점.
Codex 이전 자문(codex_extreme_plan.txt) 반영해서 weight interpolation + CFG/steps sweep 진행 중.

## 이벤트
{event}

## 실측 확정
- v5b_ANCH: 0.24524 (안전본)
- B4 step6000: 0.2407 (도전본)

## Auto loop EXTREME 최근 로그
```
{recent_log}
```

## State snapshot
```json
{state_str}
```

## 물음 (300 words 이내, brutally honest)
1. 이 pivot 시점에서 실측 slot 사용할 candidate?
2. Action mean만으로 판단하는 게 위험한 후보(예: interp 결과)?
3. 남은 candidate 중 kill/skip 권장?
4. 다음 tick까지 추가 launch 후보 있나?
5. Pareto gate (E-invdyn + clean-room LPIPS)를 다음 tick 전에 어떤 후보에 적용해야 하나?
"""
    tmp.write_text(prompt)
    cmd = (f"nohup bash -c 'cat {tmp} | /home1/sota/anaconda3/bin/codex exec "
           f"--sandbox danger-full-access --skip-git-repo-check' > {out_file} 2>&1 &")
    subprocess.Popen(cmd, shell=True)
    log(f"[codex] extreme tick{tick} launched → {out_file}")
    state.setdefault("codex_reviews", []).append({"tick": tick, "file": str(out_file), "event": event})


# ---- main tick logic ----

def tick_candidate(name, cand, state):
    phase = cand.get("phase", "pending")

    if phase in ("done", "failed", "big_success", "marginal"):
        return None

    # Pending → interp build (only for interp type)
    if phase == "pending":
        if cand["type"] == "interp":
            path = build_interp_ckpt(name, cand)
            if path is None:
                cand["phase"] = "failed"
                return f"{name}: interp build failed"
            cand["ckpt"] = str(path)
        cand["phase"] = "ready"
        return None

    # Ready → check parallel budget, launch inference
    if phase == "ready":
        running = my_running_extreme_jobs(state)
        if running >= MAX_PARALLEL_JOBS:
            return None
        ckpt = cand["ckpt"]
        guidance = cand.get("guidance", 6.0)
        steps = cand.get("steps", 35)
        jid = launch_inference(name, ckpt, guidance, steps, name)
        if jid == -2:
            cand["phase"] = "inf_done"
            cand["csv"] = str(SUB_DIR / f"submission_{name}.csv")
            log(f"[{name}] CSV already exists → inf_done")
            return None
        if jid <= 0:
            cand["phase"] = "failed"
            return f"{name}: launch failed"
        cand["job_id"] = jid
        cand["phase"] = "inference"
        log(f"[{name}] inf job {jid} launched (CFG={guidance}, steps={steps})")
        return None

    # Inference → wait, then compute Action mean
    if phase == "inference":
        if job_running(cand["job_id"]):
            return None
        csv = SUB_DIR / f"submission_{name}.csv"
        if not csv.exists():
            cand["phase"] = "failed"
            return f"{name}: CSV missing after inf job {cand['job_id']}"
        cand["csv"] = str(csv)
        cand["phase"] = "inf_done"

    # inf_done → compute score
    if phase == "inf_done" or cand.get("phase") == "inf_done":
        csv = cand.get("csv")
        am = compute_action_mean(csv)
        if am is None or "error" in am:
            cand["phase"] = "failed"
            return f"{name}: action mean calc failed"
        cand["result"] = am
        cand["phase"] = "done"
        delta_vs_b4 = am["mean"] - B4_ACTION_MEAN
        log(f"[{name}] Action={am['mean']:.4f} (vs B4 {delta_vs_b4:+.4f}), pred_LB={am['predicted_lb']:.4f}")
        if am["mean"] < BIG_SUCCESS_ACTION:
            state["success"].append({"name": name, **am})
            return f"BIG SUCCESS: {name} Action={am['mean']:.4f}"
        if am["mean"] < B4_ACTION_MEAN:
            return f"marginal: {name} Action={am['mean']:.4f} (vs B4 {delta_vs_b4:+.4f})"
        return None

    return None


def main():
    log("=" * 60)
    log("Auto loop EXTREME START (codex 승인 트랙)")
    log(f"MAX_PARALLEL_JOBS={MAX_PARALLEL_JOBS}, candidates={len(CANDIDATES)}")
    state = load_state()
    save_state(state)

    tick = 0
    while True:
        tick += 1
        log(f"--- tick {tick} ---")
        events = []
        for name, cand in state["candidates"].items():
            try:
                ev = tick_candidate(name, cand, state)
                if ev:
                    events.append(ev)
                    log(f"[event] {ev}")
            except Exception as e:
                log(f"[{name}] tick error: {e}")
        save_state(state)

        if events:
            launch_codex_pivot(tick, state, " | ".join(events))
            save_state(state)

        # 전부 settled?
        settled = all(c.get("phase") in ("done", "failed", "big_success", "marginal")
                       for c in state["candidates"].values())
        if settled:
            log(f"모든 candidate settled. Success: {len(state['success'])}")
            for n, c in state["candidates"].items():
                if "result" in c:
                    log(f"  {n}: Action={c['result']['mean']:.4f}, pred_LB={c['result']['predicted_lb']:.4f}")
            log(">>> 사용자 판단 필요. 5분마다 재확인.")
            time.sleep(300)
            continue

        time.sleep(300)  # 5분 tick


if __name__ == "__main__":
    main()
