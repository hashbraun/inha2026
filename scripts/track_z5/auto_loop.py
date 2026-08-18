"""Continuous automation loop: 개선될 때까지 자동 pivot.

Current challenger: B4 step6000 (LB 0.2407)
Goal: B4를 이기는 CSV Action mean 찾을 때까지 반복.

Loop:
  1. 활성 학습 job 감시 (z6b4 hybrid, b4 longer, ...)
  2. 학습 완료 → auto inference launch (여러 ckpt)
  3. Inference 완료 → CSV Action mean 계산
  4. B4 (0.3798) 대비:
     - Action mean < 0.375 → 사용자 실측 slot 요청 (성공 후보)
     - Action mean 0.375-0.385 → 미미, 다음 시도로 pivot
     - Action mean > 0.385 → 실패, 다음 시도
  5. 성공 candidate 없으면 다음 후보 launch (queue 있음)

Pivot queue (Codex 자문 기반):
  1. Z6-B4 hybrid (진행 중)
  2. B4 longer 20k (진행 중)
  3. Z6-B4 hybrid v2 (다른 α, weight)
  4. NAP + LoRA 병행 (다음)
  5. RAFT/RL fine-tune (마지막)

state: /home1/sota/inha2026/logs/fresh/auto_loop_state.json
log:   /home1/sota/inha2026/logs/fresh/auto_loop.log
"""
from __future__ import annotations

import datetime
import json
import subprocess
import time
from pathlib import Path

LOG = Path("/home1/sota/inha2026/logs/fresh/auto_loop.log")
STATE = Path("/home1/sota/inha2026/logs/fresh/auto_loop_state.json")

B4_ACTION_MEAN = 0.3798
B4_LB = 0.2407
BASELINE_LB = 0.24524

# 미해결 후보 pool. 각 진입 시 초기 sbatch launch, 완료 후 inference chain.
CANDIDATE_QUEUE = [
    # 진행 중이면 job_id 지정하여 skip launch
    {"name": "z6b4_hybrid", "job_id": 30738,
     "ckpt_dir": "/home1/sota/inha2026/checkpoints/z6b4_hybrid",
     "ckpt_pattern": "z6_step*.pt",
     "adapter_type": "bounded_z6",
     "inference_ckpts": ["z6_step004000.pt", "z6_step008000.pt", "z6_step012000.pt", "z6_step015000.pt"]},
    {"name": "b4_longer", "job_id": 30739,
     "ckpt_dir": "/home1/sota/inha2026/checkpoints/b4_longer",
     "ckpt_pattern": "ckpt_step*.pt",
     "adapter_type": "b4",
     "inference_ckpts": ["ckpt_step006000.pt", "ckpt_step010000.pt", "ckpt_step015000.pt", "ckpt_step020000.pt"]},
    # 추가 후보 (실행 대기)
    {"name": "z6b4_hybrid_v2", "job_id": None,
     "sbatch_template": "z6b4_hybrid_v2",
     "params": {"alpha_max": 0.03, "action_weight": 0.03, "preserve_weight": 0.15}},
    {"name": "nap_plus_lora", "job_id": None,
     "sbatch_template": "nap_plus_lora"},
]


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
        "candidates": {c["name"]: {**c, "phase": "training", "inf_jobs": {}, "results": {}}
                       for c in CANDIDATE_QUEUE},
        "best_challenger": {"csv": "submission_b4_step6000.csv", "action_mean": B4_ACTION_MEAN, "lb": B4_LB},
        "success_candidates": [],   # Action mean < 0.375
    }


def save_state(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(s, f, indent=2, ensure_ascii=False, default=str)


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
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


def launch_inference(ckpt_path, out_dir, csv_out, adapter_type):
    """Adapter 타입별 inference sbatch launch."""
    if adapter_type == "b4":
        script = "/home1/sota/inha2026/scripts/infer_cosmos3_nano.py"
        args = f"--ckpt {ckpt_path} --samples 216 --steps 35 --guidance 6.0 --rank 32 --action-repr delta_base --out-dir {out_dir}"
    elif adapter_type in ("z5", "bounded_z6"):
        v5b = "/home1/sota/inha2026/checkpoints/cosmos3_nano_v5_base/ckpt_step040000.pt"
        script = "/home1/sota/inha2026/scripts/track_z5/infer_z5.py"
        args = f"--v5b-ckpt {v5b} --z5-ckpt {ckpt_path} --samples 216 --steps 35 --guidance 6.0 --rank 32 --action-repr delta_base --out-dir {out_dir}"
    else:
        log(f"unknown adapter_type: {adapter_type}")
        return -1
    tag = Path(csv_out).stem.replace("submission_", "")
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
    sbatch_path = Path(f"/tmp/inf_{tag}.sbatch")
    sbatch_path.write_text(sbatch)
    out = sh(f"sbatch {sbatch_path}")
    try:
        return int(out.split()[-1])
    except Exception:
        log(f"launch failed: {out}")
        return -1


def tick_candidate(name, cand, state):
    """각 candidate의 phase 진행."""
    phase = cand["phase"]

    # Training 진행 or 완료
    if phase == "training":
        if cand.get("job_id") and job_running(cand["job_id"]):
            return  # 학습 중
        # 학습 완료 or job_id 없음
        # 완료된 ckpt 확인
        ckpt_dir = Path(cand["ckpt_dir"])
        if not ckpt_dir.exists():
            log(f"[{name}] ckpt_dir not found → still queued or failed")
            return
        pattern = cand.get("ckpt_pattern", "*.pt")
        available = sorted(ckpt_dir.glob(pattern))
        if not available:
            log(f"[{name}] no ckpts yet")
            return
        # 학습 완료. Inference launch (지정된 ckpts 또는 마지막 3개)
        target = cand.get("inference_ckpts") or [p.name for p in available[-3:]]
        launched = 0
        for ckpt_name in target:
            ckpt_p = ckpt_dir / ckpt_name
            if not ckpt_p.exists():
                continue
            tag = f"{name}_{ckpt_name.replace('.pt','')}"
            out_dir = Path(f"/home1/sota/inha2026/submission_kit/fresh/input_videos_{tag}")
            csv_out = Path(f"/home1/sota/inha2026/submission_kit/fresh/submission_{tag}.csv")
            if csv_out.exists():
                log(f"[{name}] {ckpt_name} CSV exists → skip")
                cand["inf_jobs"][ckpt_name] = {"done": True, "csv": str(csv_out)}
                continue
            if cand["inf_jobs"].get(ckpt_name, {}).get("job_id"):
                jid = cand["inf_jobs"][ckpt_name]["job_id"]
                if job_running(jid):
                    log(f"[{name}] {ckpt_name} inf {jid} still running")
                    continue
            # Launch inference
            jid = launch_inference(ckpt_p, out_dir, csv_out, cand.get("adapter_type", "b4"))
            cand["inf_jobs"][ckpt_name] = {"job_id": jid, "csv": str(csv_out)}
            log(f"[{name}] {ckpt_name} inf job {jid} launched")
            launched += 1
        if launched == 0 and all(v.get("done") or (v.get("job_id") and not job_running(v["job_id"]))
                                  for v in cand["inf_jobs"].values()):
            # All inferences done → check results
            cand["phase"] = "evaluate"
        return

    if phase == "evaluate":
        # 각 CSV Action mean 계산 + 비교
        best_action = None
        for ckpt_name, info in cand["inf_jobs"].items():
            csv = info.get("csv")
            if not csv or not Path(csv).exists():
                continue
            am = compute_action_mean(csv)
            if not am or "error" in am:
                continue
            cand["results"][ckpt_name] = am
            log(f"[{name}] {ckpt_name}: Action mean={am['mean']:.4f} pred_LB={am['predicted_lb']:.4f}")
            if best_action is None or am["mean"] < best_action["mean"]:
                best_action = {**am, "ckpt": ckpt_name}
        if best_action is None:
            log(f"[{name}] no evaluable results → failed")
            cand["phase"] = "failed"
            return
        cand["best"] = best_action
        # Compare to B4
        if best_action["mean"] < 0.375:
            log(f"[{name}] SUCCESS candidate (Action {best_action['mean']:.4f} < 0.375)")
            state["success_candidates"].append({"name": name, **best_action})
            cand["phase"] = "success"
        elif best_action["mean"] < B4_ACTION_MEAN:
            log(f"[{name}] marginal (better than B4 but > 0.375)")
            cand["phase"] = "marginal"
        else:
            log(f"[{name}] FAILED (Action {best_action['mean']:.4f} >= B4 {B4_ACTION_MEAN})")
            cand["phase"] = "failed"
        return


def main():
    log("=" * 60)
    log("Auto loop start")
    state = load_state()
    save_state(state)

    tick = 0
    while True:
        tick += 1
        log(f"--- tick {tick} ---")
        for name, cand in state["candidates"].items():
            if cand["phase"] in ("success", "marginal", "failed"):
                continue
            try:
                tick_candidate(name, cand, state)
            except Exception as e:
                log(f"[{name}] error: {e}")
        save_state(state)

        # 종료 조건 1: 성공 후보 있음
        if state["success_candidates"]:
            log(f"SUCCESS: {state['success_candidates']}")
            log("모든 성공 후보 실측 slot 사용 검토 필요 (사용자 확인)")

        # 종료 조건 2: 모든 활성 candidate settled
        active = [n for n, c in state["candidates"].items() if c["phase"] not in ("success", "marginal", "failed")]
        if not active:
            log(f"모든 candidate settled: {[c['phase'] for c in state['candidates'].values()]}")
            # Success 있으면 유지, 없으면 다음 후보 launch 로직 필요 (TODO)
            break

        time.sleep(600)   # 10분 간격
    log("Auto loop END")


if __name__ == "__main__":
    main()
