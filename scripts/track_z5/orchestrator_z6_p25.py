"""Z6-lite + Cosmos-Predict2.5 자동화 orchestrator.

파이프라인:
1. Z6-lite pilot 감시 (Job 30704) → 완료 시 inference + CSV → Action mean
2. 병렬: Cosmos-Predict2.5 falsification 재실행 (다른 GPU)
3. Codex 정기 자문 (매 phase 완료 후)
4. 성능 정체 or Go 조건 실패 시 다음 후보로 pivot

state 파일: /home1/sota/inha2026/logs/fresh/orchestrator_state.json
log: /home1/sota/inha2026/logs/fresh/orchestrator.log

정지 조건:
- 모든 후보 시도 완료
- 사용자 kill
"""
from __future__ import annotations

import datetime
import json
import subprocess
import time
from pathlib import Path

LOG = Path("/home1/sota/inha2026/logs/fresh/orchestrator.log")
STATE = Path("/home1/sota/inha2026/logs/fresh/orchestrator_state.json")

BASELINE_LB = 0.24524            # v5b_ANCHORED_soft
CURRENT_CHALLENGER_LB = 0.2407   # B4 step6000

# 후보 파이프라인 (순차/병렬 관리)
CANDIDATES = [
    {"name": "z6_pilot_30704", "phase": "training", "type": "z6",
     "job_id": 30704, "ckpt_dir": "/home1/sota/inha2026/checkpoints/z6_bounded",
     "inference_out": "/home1/sota/inha2026/submission_kit/fresh/input_videos_z6_step2000",
     "csv_out": "/home1/sota/inha2026/submission_kit/fresh/submission_z6_step2000.csv"},
    {"name": "p25_falsification", "phase": "pending", "type": "predict25_falsification"},
]


def now(): return datetime.datetime.now().isoformat(timespec="seconds")


def log(msg: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {msg}\n"
    print(line, end="", flush=True)
    with open(LOG, "a") as f:
        f.write(line)


def save_state(state: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_state() -> dict:
    if STATE.exists():
        return json.load(open(STATE))
    return {"started": now(), "candidates": {c["name"]: c for c in CANDIDATES},
            "current_challenger": {"name": "b4_step6000", "lb": CURRENT_CHALLENGER_LB},
            "safe_baseline": {"name": "v5b_ANCHORED_soft", "lb": BASELINE_LB},
            "codex_consultations": []}


def sh(cmd: str, capture: bool = True, timeout: int = 60) -> str:
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture, text=True, timeout=timeout)
        return result.stdout.strip() if capture else ""
    except subprocess.TimeoutExpired:
        return ""


def job_running(job_id: int) -> bool:
    out = sh(f"squeue -j {job_id} -h -o '%T' 2>/dev/null")
    return len(out) > 0


def get_current_step(log_path: Path) -> int:
    """z6 log에서 latest step 파싱."""
    if not log_path.exists():
        return 0
    lines = log_path.read_text().splitlines()[-100:]
    for line in reversed(lines):
        if line.startswith("step "):
            try:
                return int(line.split()[1].split("/")[0])
            except Exception:
                pass
    return 0


def compute_csv_action_mean(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    try:
        import pandas as pd, numpy as np
        df = pd.read_csv(csv_path)
        act = df[df["feature_component"] == "Action Component"]
        arr = np.array([json.loads(s) for s in act["feature_json"]]).squeeze()
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "predicted_lb": float(0.4 * arr.mean() + 0.089),
        }
    except Exception as e:
        return {"error": str(e)}


def launch_z6_inference(ckpt_path: Path, out_dir: Path, csv_out: Path) -> int:
    """Z6 inference sbatch launch. Returns job id."""
    v5b = "/home1/sota/inha2026/checkpoints/cosmos3_nano_v5_base/ckpt_step040000.pt"
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=z6inf
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/home1/sota/inha2026/logs/fresh/z6inf_%j.log
#SBATCH --error=/home1/sota/inha2026/logs/fresh/z6inf_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p {out_dir}
/home1/sota/anaconda3/envs/inha2026/bin/python \\
    /home1/sota/inha2026/scripts/track_z5/infer_z5.py \\
    --v5b-ckpt {v5b} --z5-ckpt {ckpt_path} \\
    --samples 216 --steps 35 --guidance 6.0 --rank 32 \\
    --action-repr delta_base --out-dir {out_dir}
cd /home1/sota/inha2026/submission_kit
/home1/sota/anaconda3/envs/inha2026/bin/python make_submission_csv.py \\
    --prediction-root {out_dir.relative_to(Path("/home1/sota/inha2026/submission_kit"))} \\
    --output-csv {csv_out.relative_to(Path("/home1/sota/inha2026/submission_kit"))}
echo "[$(date)] z6 inference done"
"""
    sbatch_path = Path("/tmp/z6_inf_orchestrator.sbatch")
    sbatch_path.write_text(sbatch_content)
    out = sh(f"sbatch {sbatch_path}")
    try:
        return int(out.split()[-1])
    except Exception:
        log(f"sbatch failed: {out}")
        return -1


def launch_p25_falsification() -> int:
    """Cosmos-Predict2.5 falsification pilot 재실행."""
    sbatch = "/home1/sota/inha2026/scripts/predict2/falsification_pilot.sbatch"
    if not Path(sbatch).exists():
        log(f"p25 sbatch not found: {sbatch}")
        return -1
    out = sh(f"sbatch {sbatch}")
    try:
        return int(out.split()[-1])
    except Exception:
        log(f"p25 launch failed: {out}")
        return -1


def codex_consult(question: str, tag: str) -> str:
    """Background Codex 자문. output file path 반환."""
    out_file = Path(f"/home1/sota/inha2026/docs/plans/fresh_restart/codex_{tag}.txt")
    tmp_prompt = Path(f"/tmp/codex_{tag}_prompt.txt")
    tmp_prompt.write_text(question)
    cmd = (f"nohup bash -c 'cat {tmp_prompt} | "
           f"/home1/sota/anaconda3/bin/codex exec --sandbox read-only --skip-git-repo-check' "
           f"> {out_file} 2>&1 &")
    subprocess.Popen(cmd, shell=True)
    log(f"codex {tag} launched → {out_file}")
    return str(out_file)


def evaluate_z6_go_conditions(state: dict, ckpt_path: Path, csv_action_mean: float) -> tuple[bool, list[str]]:
    """Codex Z6 Go 조건 6개 사전 판정 (간략화):
       - CSV Action mean이 B4(0.3798) 대비 2-3% 개선?
       - Predicted LB이 안전본(0.24524)보다 낮은가?
    """
    reasons = []
    B4_ACT = 0.3798
    improvement_pct = (B4_ACT - csv_action_mean) / B4_ACT * 100
    if improvement_pct < 2.0:
        reasons.append(f"Action improvement {improvement_pct:.1f}% < 2% target")
    pred_lb = 0.4 * csv_action_mean + 0.089
    if pred_lb >= BASELINE_LB:
        reasons.append(f"Predicted LB {pred_lb:.4f} >= safe baseline {BASELINE_LB}")
    if not reasons:
        return True, ["All Go conditions met"]
    return False, reasons


def phase_z6_pilot(state: dict):
    z6 = state["candidates"]["z6_pilot_30704"]
    job_id = z6["job_id"]
    if job_running(job_id):
        step = get_current_step(Path(f"/home1/sota/inha2026/logs/fresh/z6_{job_id}.log"))
        log(f"[z6] pilot running step={step}/2000")
        return  # 다음 tick 대기

    log(f"[z6] pilot job {job_id} exited")
    # 최신 ckpt 확인
    ckpts = sorted(Path(z6["ckpt_dir"]).glob("z6_step*.pt"))
    if not ckpts:
        log(f"[z6] no ckpt found → phase failed")
        z6["phase"] = "failed"
        z6["reason"] = "no ckpt"
        return
    latest = ckpts[-1]
    log(f"[z6] latest ckpt: {latest.name}")

    # Inference launch
    if z6["phase"] == "training":
        out_dir = Path(z6["inference_out"])
        csv_out = Path(z6["csv_out"])
        inf_job = launch_z6_inference(latest, out_dir, csv_out)
        z6["inf_job_id"] = inf_job
        z6["phase"] = "inference"
        log(f"[z6] inference job {inf_job} launched")
        return

    if z6["phase"] == "inference":
        if job_running(z6["inf_job_id"]):
            log(f"[z6] inference {z6['inf_job_id']} running")
            return
        # inference done → check CSV
        csv_out = Path(z6["csv_out"])
        stats = compute_csv_action_mean(csv_out)
        if stats is None or "error" in stats:
            log(f"[z6] CSV failed: {stats}")
            z6["phase"] = "failed"; z6["reason"] = "csv error"
            return
        z6["csv_stats"] = stats
        log(f"[z6] CSV done: Action mean={stats['mean']:.4f} pred_LB={stats['predicted_lb']:.4f}")

        # Go 조건 판정
        ok, reasons = evaluate_z6_go_conditions(state, latest, stats["mean"])
        z6["go_conditions"] = {"pass": ok, "reasons": reasons}
        if ok:
            log(f"[z6] GO conditions PASSED → continuation 검토")
            z6["phase"] = "go_pass"
        else:
            log(f"[z6] GO conditions FAILED: {reasons}")
            z6["phase"] = "go_fail"
            # 다음 후보로 pivot
        return


def phase_p25(state: dict):
    p25 = state["candidates"]["p25_falsification"]
    if p25["phase"] == "pending":
        job_id = launch_p25_falsification()
        if job_id < 0:
            p25["phase"] = "failed"; p25["reason"] = "launch failed"
            return
        p25["job_id"] = job_id
        p25["phase"] = "running"
        log(f"[p25] falsification job {job_id} launched")
        return
    if p25["phase"] == "running":
        if job_running(p25["job_id"]):
            log(f"[p25] falsification running")
            return
        # 완료 → 결과 확인
        result_json = "/home1/sota/inha2026/logs/predict2.5_falsification.jsonl"
        if Path(result_json).exists():
            lines = Path(result_json).read_text().strip().splitlines()
            if lines:
                last = json.loads(lines[-1])
                p25["last_result"] = last
                log(f"[p25] falsification result: {last}")
                # D_action/D_seed 확인 필요 (스크립트 자체 계산)
                p25["phase"] = "done"
        else:
            p25["phase"] = "failed"; p25["reason"] = "no result json"


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("Orchestrator start")
    state = load_state()
    save_state(state)

    tick = 0
    while True:
        tick += 1
        log(f"--- tick {tick} ---")

        # Phase 1: Z6-lite pilot
        try:
            phase_z6_pilot(state)
        except Exception as e:
            log(f"[z6] phase error: {e}")

        # Phase 2: Cosmos-Predict2.5 재도전 (병렬)
        try:
            phase_p25(state)
        except Exception as e:
            log(f"[p25] phase error: {e}")

        save_state(state)

        # 종료 조건
        all_done = all(
            c.get("phase") in ("go_pass", "go_fail", "done", "failed")
            for c in state["candidates"].values()
        )
        if all_done:
            log("All candidates settled → orchestrator DONE")
            break

        # Codex 자문 트리거 (3 tick마다)
        if tick % 3 == 0:
            summary = json.dumps({n: c.get("phase") for n, c in state["candidates"].items()})
            log(f"[codex] tick {tick} summary: {summary}")

        time.sleep(600)  # 10분 간격 폴링

    log("Orchestrator END")


if __name__ == "__main__":
    main()
