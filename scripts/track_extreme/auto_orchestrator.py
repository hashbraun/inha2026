"""사용자 취침 중 자율 진행 오케스트레이터.

Stage 1 학습 중 매 checkpoint 저장 시:
- Action ablation (gt/zero/randn/reverse) 실행
- gt-gt noise floor는 이미 측정된 값 재사용
- Pareto gate 판정 → decision log 기록

Stage 1 학습 종료 시:
- Best checkpoint 선택
- 216 sample inference launch
- Pareto batch 판정
- 통과 시 AUTO_ready_new_best.csv 준비

전 과정 AUTO_DECISION_LOG.md에 기록.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home1/sota/inha2026")
LOG = ROOT / "docs/plans/fresh_restart/AUTO_DECISION_LOG.md"
CKPT_DIR = ROOT / "checkpoints/st1_frozen_shift"
STAGE1_JOB_ID = 30955
SUB = ROOT / "submission_kit/fresh"
READY = ROOT / "submissions"

# Thresholds (사용자 승인 원칙)
BIG_SUCCESS_ACTION = 0.36  # Action mean
GT_GT_FLOOR = 8.83  # 실측 (B4 seeds, 2026-08-20 01:47)
V5B_20K_REACTIVITY = 2.93  # v5b_20k gt-zero/nfl (유일한 검증된 signal)
MIN_REACTIVITY = 1.5  # 최소 통과 기준


def now(): return datetime.datetime.now().isoformat(timespec="seconds")


def log(msg: str, section: bool = False):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"\n### [{now()}] {msg}\n" if section else f"- [{now()}] {msg}\n"
    with open(LOG, "a") as f:
        f.write(line)


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except Exception as e:
        return "", str(e), -1


def job_running(job_id):
    out, _, _ = sh(f"squeue -j {job_id} -h -o '%T' 2>/dev/null")
    return "RUNNING" in out or "PENDING" in out


def compute_action_mean(csv_path):
    if not Path(csv_path).exists():
        return None
    try:
        import pandas as pd, numpy as np
        df = pd.read_csv(csv_path)
        act = df[df["feature_component"] == "Action Component"]
        arr = np.array([json.loads(s) for s in act["feature_json"]]).squeeze()
        return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                "pred_lb": float(0.4 * arr.mean() + 0.089)}
    except Exception as e:
        return {"error": str(e)}


def launch_ablation_on_ckpt(ckpt_path, tag):
    """4-condition ablation (gt/zero/randn/reverse) via sbatch. return job id."""
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=abl_{tag[:12]}
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=80G
#SBATCH --time=0:30:00
#SBATCH --output={ROOT}/logs/fresh/abl_{tag}_%j.log
#SBATCH --error={ROOT}/logs/fresh/abl_{tag}_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home1/sota/anaconda3/envs/inha2026/bin/python \
  /home1/sota/inha2026/scripts/action_ablation_diagnostic.py \
  --experiments "{tag}:{ckpt_path}:delta_base"
"""
    p = Path(f"/tmp/abl_{tag}.sbatch")
    p.write_text(sbatch_content)
    out, err, rc = sh(f"sbatch {p}")
    try:
        return int(out.split()[-1])
    except Exception:
        return -1


def launch_full_inference(ckpt_path, tag, guidance=6.0, steps=35):
    out_dir = SUB / f"input_videos_{tag}"
    csv_out = SUB / f"submission_{tag}.csv"
    if csv_out.exists():
        return -2
    sbatch = f"""#!/bin/bash
#SBATCH --job-name=inf_{tag[:12]}
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --output={ROOT}/logs/fresh/inf_{tag}_%j.log
#SBATCH --error={ROOT}/logs/fresh/inf_{tag}_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p {out_dir}
/home1/sota/anaconda3/envs/inha2026/bin/python /home1/sota/inha2026/scripts/infer_cosmos3_nano.py \\
    --ckpt {ckpt_path} --samples 216 --steps {steps} --guidance {guidance} \\
    --rank 32 --action-repr delta_base --out-dir {out_dir}
cd {ROOT}/submission_kit
/home1/sota/anaconda3/envs/inha2026/bin/python make_submission_csv.py \\
    --prediction-root fresh/input_videos_{tag} \\
    --output-csv fresh/submission_{tag}.csv
"""
    p = Path(f"/tmp/inf_{tag}.sbatch")
    p.write_text(sbatch)
    out, _, _ = sh(f"sbatch {p}")
    try:
        return int(out.split()[-1])
    except Exception:
        return -1


def launch_pareto(video_dir, out_json, tag):
    sbatch = f"""#!/bin/bash
#SBATCH --job-name=par_{tag[:12]}
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=40G
#SBATCH --time=0:30:00
#SBATCH --output={ROOT}/logs/fresh/par_{tag}_%j.log
#SBATCH --error={ROOT}/logs/fresh/par_{tag}_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home1/sota/anaconda3/envs/inha2026/bin/python \
  /home1/sota/inha2026/scripts/track_extreme/pareto_score.py \
  --videos {video_dir} --out-json {out_json} --samples 216 \
  --ref-videos {SUB}/input_videos_b4_step6000
"""
    p = Path(f"/tmp/par_{tag}.sbatch")
    p.write_text(sbatch)
    out, _, _ = sh(f"sbatch {p}")
    try:
        return int(out.split()[-1])
    except Exception:
        return -1


def main():
    log("Auto Orchestrator START", section=True)
    log(f"Stage 1 job: {STAGE1_JOB_ID}, ckpt dir: {CKPT_DIR}")

    seen_ckpts = set()
    ablation_jobs = {}   # ckpt_name → job_id
    ablation_done = {}   # ckpt_name → summary
    stage1_done = False
    inf_launched = {}
    inf_done = {}
    pareto_launched = {}

    tick = 0
    while True:
        tick += 1
        now_hh = int(datetime.datetime.now().strftime("%H"))

        # Time-based auto stop (08:00 KST 이후 새 학습 launch 금지)
        if now_hh >= 8 and not stage1_done:
            if job_running(STAGE1_JOB_ID):
                log(f"⚠️ 08:00 KST 넘음, Stage 1 still running. 그대로 두되 새 실험 launch 안 함")

        # 15:00 KST 넘으면 종료 준비
        if now_hh >= 15:
            log("⚠️ 15:00 KST 도달. Orchestrator 종료. 사용자 확인 대기.")
            break

        # 1) Stage 1 학습 상태
        if not stage1_done:
            if not job_running(STAGE1_JOB_ID):
                stage1_done = True
                log(f"Stage 1 (job {STAGE1_JOB_ID}) 종료됨")

        # 2) 새 checkpoint 감지
        if CKPT_DIR.exists():
            for ckpt in sorted(CKPT_DIR.glob("ckpt_step*.pt")):
                if ckpt.name in seen_ckpts:
                    continue
                seen_ckpts.add(ckpt.name)
                log(f"신규 checkpoint 감지: {ckpt.name}")
                # ablation launch (gt/zero/randn/reverse)
                tag = f"st1_{ckpt.stem.replace('ckpt_step', 'step')}"
                jid = launch_ablation_on_ckpt(str(ckpt), tag)
                if jid > 0:
                    ablation_jobs[ckpt.name] = (jid, tag)
                    log(f"  Ablation job {jid} launched (tag={tag})")

        # 3) Ablation 완료 시 gt-zero 계산
        for ckpt_name, (jid, tag) in list(ablation_jobs.items()):
            if ckpt_name in ablation_done:
                continue
            if job_running(jid):
                continue
            # analyze
            try:
                import numpy as np
                import imageio.v3 as iio
                d = ROOT / f"submission_kit/action_ablation_diag/{tag}"
                gz_list, gr_list, grev_list, zr_list = [], [], [], []
                for sid in ["sample_000000", "sample_000001", "sample_000002"]:
                    try:
                        gt = iio.imread(d / "gt" / f"{sid}.mp4")[:16].astype(np.float32)
                        zero = iio.imread(d / "zeros" / f"{sid}.mp4")[:16].astype(np.float32)
                        randn = iio.imread(d / "randn" / f"{sid}.mp4")[:16].astype(np.float32)
                        rev = iio.imread(d / "reverse" / f"{sid}.mp4")[:16].astype(np.float32)
                        gz_list.append(float(np.abs(gt-zero).mean()))
                        gr_list.append(float(np.abs(gt-randn).mean()))
                        grev_list.append(float(np.abs(gt-rev).mean()))
                        zr_list.append(float(np.abs(zero-randn).mean()))
                    except Exception:
                        pass
                if gz_list:
                    s = {"gt_zero": np.mean(gz_list), "gt_randn": np.mean(gr_list),
                          "gt_rev": np.mean(grev_list), "zero_randn": np.mean(zr_list)}
                    ablation_done[ckpt_name] = s
                    log(f"  Ablation {ckpt_name}: gt-zero={s['gt_zero']:.3f} "
                        f"gt-rev={s['gt_rev']:.3f} zero-randn={s['zero_randn']:.3f}")
            except Exception as e:
                log(f"  Ablation analyze err {ckpt_name}: {e}")

        # 4) Stage 1 종료 시 best ckpt로 216 inference launch
        if stage1_done and not inf_launched:
            # Pick best checkpoint based on gt-zero (higher = more action reactive)
            # But use gt-gt normalized if noise floor known
            if ablation_done:
                # 정규화 판정: gt-zero / noise_floor > MIN_REACTIVITY
                # 순서 신호도 고려: gt-rev/nfl 낮으면 순서 무시 → penalty
                ranked = []
                for name, s in ablation_done.items():
                    gz_ratio = s["gt_zero"] / GT_GT_FLOOR
                    rev_ratio = s["gt_rev"] / GT_GT_FLOOR if s.get("gt_rev") else 0
                    # v5b_20k 2.93x 기준으로 상대 점수, 순서 신호 (rev > 1.0) 보너스
                    score = gz_ratio + 0.5 * max(0, rev_ratio - 1.0)
                    ranked.append((name, score, s, gz_ratio, rev_ratio))
                ranked.sort(key=lambda x: -x[1])
                for name, score, s, gz_r, rv_r in ranked:
                    log(f"  {name}: gt-zero/nfl={gz_r:.2f}x rev/nfl={rv_r:.2f}x score={score:.2f}")
                best_name, best_score, best_s, best_gz, best_rv = ranked[0]
                best_ckpt = CKPT_DIR / best_name
                # MIN_REACTIVITY 미달 시 kill (B4의 0.80x 수준이면 무의미)
                if best_gz < MIN_REACTIVITY:
                    log(f"⚠️ Best ckpt {best_name} gt-zero/nfl={best_gz:.2f}x < {MIN_REACTIVITY}x threshold. "
                        f"216 inference skip.", section=True)
                    stage1_done = True  # skip further inference
                    continue
                log(f"Best Stage 1 ckpt: {best_name} score={best_score:.3f} "
                    f"(gt-zero {best_gz:.2f}x, rev {best_rv:.2f}x)", section=True)

                # Launch full 216 inference
                tag = f"st1_best_{best_name.replace('ckpt_step', '').replace('.pt','')}"
                jid = launch_full_inference(str(best_ckpt), tag)
                if jid > 0:
                    inf_launched[tag] = (jid, str(best_ckpt))
                    log(f"216 inference launched: job {jid}, tag={tag}")

        # 5) inference 완료 시 pareto launch
        for tag, (jid, ckpt_p) in list(inf_launched.items()):
            if tag in inf_done:
                continue
            if job_running(jid):
                continue
            csv = SUB / f"submission_{tag}.csv"
            if not csv.exists():
                log(f"  {tag} inf 종료됐지만 CSV 없음")
                inf_done[tag] = None
                continue
            am = compute_action_mean(csv)
            log(f"  {tag} inf 완료: Action={am.get('mean', 0):.4f} pred_LB={am.get('pred_lb', 0):.4f}")
            inf_done[tag] = am
            # Pareto launch
            video_dir = SUB / f"input_videos_{tag}"
            out_json = ROOT / f"logs/fresh/pareto/{tag}.json"
            pjid = launch_pareto(str(video_dir), str(out_json), tag)
            if pjid > 0:
                pareto_launched[tag] = pjid
                log(f"  Pareto job {pjid} launched")

        # 6) Pareto 완료 시 최종 판정
        for tag, pjid in list(pareto_launched.items()):
            if job_running(pjid):
                continue
            p_json = ROOT / f"logs/fresh/pareto/{tag}.json"
            if not p_json.exists():
                continue
            try:
                s = json.load(open(p_json))
                am = inf_done.get(tag, {})
                action_mean = am.get("mean", 1.0)
                alex_cos = s.get("alex_cosine_vs_ref", 0)
                action_l1 = s.get("action_l1_mean", 999)
                log(f"  {tag} Pareto: e-inv={action_l1:.3f} alex_cos={alex_cos:.4f}")

                # 최종 판정
                # 조건: Action < 0.36 AND alex_cos >= 0.985 AND e-inv 개선
                # B4 ref: e-inv 41.953, alex_cos 1.0
                is_good = (
                    action_mean < BIG_SUCCESS_ACTION
                    and alex_cos >= 0.985
                    and action_l1 <= 41.5  # 개선
                )
                if is_good:
                    # Copy to AUTO_ready_new_best.csv
                    src = SUB / f"submission_{tag}.csv"
                    dst = READY / f"AUTO_ready_new_best_{tag}.csv"
                    subprocess.run(["cp", str(src), str(dst)])
                    log(f"  ✅ 신규 도전 후보 준비: {dst.name}", section=True)
                else:
                    log(f"  ❌ Pareto gate 불통과: Action={action_mean:.4f}, "
                        f"alex_cos={alex_cos:.4f}, e-inv={action_l1:.3f}")
                # pareto 처리 완료 표시
                del pareto_launched[tag]
            except Exception as e:
                log(f"  Pareto parse err {tag}: {e}")

        # 종료 조건
        if stage1_done and not ablation_jobs and not inf_launched and not pareto_launched:
            log("모든 파이프라인 종료. Orchestrator 정상 exit.")
            break
        if stage1_done and all(name in ablation_done for name in seen_ckpts) \
                and all(tag in inf_done for tag in inf_launched) \
                and not pareto_launched:
            log("모든 후속 작업 완료. Orchestrator 정상 exit.")
            break

        time.sleep(120)  # 2분 tick


if __name__ == "__main__":
    main()
