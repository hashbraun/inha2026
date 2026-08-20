"""Stage 3 orchestrator: 각 ckpt 저장 시 attention mass + gt-rev/nfl 자동 측정.

조기 중단 규칙 (사용자 승인):
- step 500 checkpoint의 mass ratio < 1.5x → scancel Stage 3 job → 알림 로그

각 ckpt (500/1000/1500/2000/2500):
1. attention mass (vision Q → action K, 우리 자체 script 재사용)
2. gt/zero/randn/reverse ablation
3. gt-rev/nfl 정규화 (nfl=8.83 B4 기준)

모든 결과 → /home1/sota/inha2026/docs/plans/fresh_restart/STAGE3_LOG.md 축적
"""
from __future__ import annotations

import datetime
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home1/sota/inha2026")
LOG = ROOT / "docs/plans/fresh_restart/STAGE3_LOG.md"
CKPT_DIR = ROOT / "checkpoints/st4_full_freeze"
GT_GT_FLOOR = 8.83   # B4 seed 기반 실측
MIN_MASS_500 = 1.5   # 500 step에서 이 이하면 조기 kill
STAGE3_JOB = None    # 첫 tick에서 확정


def now(): return datetime.datetime.now().isoformat(timespec="seconds")


def log(msg, section=False):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"\n### [{now()}] {msg}\n" if section else f"- [{now()}] {msg}\n"
    with open(LOG, "a") as f:
        f.write(line)
    print(line, end="", flush=True)


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except Exception as e:
        return f"__err__:{e}", -1


def find_stage3_job():
    out, _ = sh("squeue -u sota -h -n st4_full_freeze -o '%i'")
    if out:
        try:
            return int(out.split()[0])
        except Exception:
            return None
    return None


def job_running(job_id):
    out, _ = sh(f"squeue -j {job_id} -h -o '%T' 2>/dev/null")
    return "RUNNING" in out or "PENDING" in out


def launch_attn_mass_on_ckpt(ckpt_path, tag):
    """attention mass 측정 sbatch."""
    sb = f"""#!/bin/bash
#SBATCH --job-name=mass_{tag[:10]}
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=100G
#SBATCH --time=0:15:00
#SBATCH --output={ROOT}/logs/fresh/mass_{tag}_%j.log
#SBATCH --error={ROOT}/logs/fresh/mass_{tag}_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LORA_TARGETS="to_add_out,mlp_moe_gen.gate_proj,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj"
# 커스텀: 이 ckpt 하나만 측정
/home1/sota/anaconda3/envs/inha2026/bin/python <<PYEOF
import sys, os
os.environ["MASS_ONLY_CKPT"] = "{ckpt_path}"
os.environ["MASS_CKPT_TAG"] = "{tag}"
exec(open("/home1/sota/inha2026/scripts/action_attn_mass_single.py").read())
PYEOF
"""
    p = Path(f"/home1/sota/inha2026/scripts/track_extreme/tmp_mass_{tag}.sbatch")
    p.write_text(sb)
    out, _ = sh(f"sbatch {p}")
    try:
        return int(out.split()[-1])
    except Exception:
        return -1


def launch_ablation_on_ckpt(ckpt_path, tag):
    """gt/zero/randn/reverse ablation."""
    sb = f"""#!/bin/bash
#SBATCH --job-name=ab3_{tag[:10]}
#SBATCH --gres=gpu:A6000_ada:1
#SBATCH --exclude=gpu-113
#SBATCH --mem=80G
#SBATCH --time=0:20:00
#SBATCH --output={ROOT}/logs/fresh/ab3_{tag}_%j.log
#SBATCH --error={ROOT}/logs/fresh/ab3_{tag}_%j.err
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LORA_TARGETS="to_add_out,mlp_moe_gen.gate_proj,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj"
/home1/sota/anaconda3/envs/inha2026/bin/python \
  /home1/sota/inha2026/scripts/action_ablation_diagnostic.py \
  --experiments "st3_{tag}:{ckpt_path}:delta_base"
"""
    p = Path(f"/home1/sota/inha2026/scripts/track_extreme/tmp_ab3_{tag}.sbatch")
    p.write_text(sb)
    out, _ = sh(f"sbatch {p}")
    try:
        return int(out.split()[-1])
    except Exception:
        return -1


def parse_ablation(tag):
    d = ROOT / f"submission_kit/action_ablation_diag/st3_{tag}"
    if not (d / "reverse" / "sample_000002.mp4").exists():
        return None
    import numpy as np
    import imageio.v3 as iio
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
    if not gz_list:
        return None
    return {
        "gt_zero": float(np.mean(gz_list)),
        "gt_randn": float(np.mean(gr_list)),
        "gt_rev": float(np.mean(grev_list)),
        "zero_randn": float(np.mean(zr_list)),
        "gt_rev_nfl": float(np.mean(grev_list) / GT_GT_FLOOR),
        "gt_zero_nfl": float(np.mean(gz_list) / GT_GT_FLOOR),
    }


def parse_mass(tag):
    """mass log에서 pretrained format 결과 parsing."""
    logs = list(ROOT.glob(f"logs/fresh/mass_{tag}_*.log"))
    if not logs:
        return None
    txt = logs[-1].read_text()
    if "SUMMARY" not in txt:
        return None
    # pretrained 형식: "action_mass=X.XXXX% uniform=X.XXX% ratio=X.XXx"
    import re
    m = re.search(r"SUMMARY:\s+action_mass=([\d.]+)%\s+uniform=([\d.]+)%\s+ratio=([\d.]+)x", txt)
    if m:
        return {"action_mass": float(m.group(1)), "uniform": float(m.group(2)), "ratio": float(m.group(3))}
    return None


def main():
    global STAGE3_JOB
    log("Stage 3 Orchestrator START (O-only LoRA + timestep shift, from pretrained)", section=True)
    log(f"CKPT dir: {CKPT_DIR}")
    log(f"Early kill rule: step 500 mass < {MIN_MASS_500}x → scancel")
    log(f"NFL (B4 seed): {GT_GT_FLOOR}")

    seen_ckpts = set()
    mass_jobs = {}       # ckpt_name → (jid, tag)
    mass_done = {}       # ckpt_name → summary
    ab_jobs = {}
    ab_done = {}
    killed = False

    tick = 0
    while True:
        tick += 1
        # Stage 3 job 찾기
        if STAGE3_JOB is None:
            STAGE3_JOB = find_stage3_job()
            if STAGE3_JOB:
                log(f"Stage 3 job detected: {STAGE3_JOB}")

        # 새 checkpoint 감지
        if CKPT_DIR.exists():
            for ckpt in sorted(CKPT_DIR.glob("ckpt_step*.pt")):
                if ckpt.name in seen_ckpts:
                    continue
                seen_ckpts.add(ckpt.name)
                log(f"신규 checkpoint: {ckpt.name}", section=True)
                tag = ckpt.stem.replace("ckpt_step", "step")
                # Launch mass + ablation
                m_jid = launch_attn_mass_on_ckpt(str(ckpt), tag)
                a_jid = launch_ablation_on_ckpt(str(ckpt), tag)
                mass_jobs[ckpt.name] = (m_jid, tag)
                ab_jobs[ckpt.name] = (a_jid, tag)
                log(f"  mass job {m_jid}, ablation job {a_jid} launched")

        # Mass 결과 확인 + 조기 kill 판정
        for ckpt_name, (jid, tag) in list(mass_jobs.items()):
            if ckpt_name in mass_done:
                continue
            if job_running(jid):
                continue
            m = parse_mass(tag)
            if m:
                mass_done[ckpt_name] = m
                log(f"[{ckpt_name}] MASS: action={m['action_mass']:.4f}% uniform={m['uniform']:.3f}% ratio={m['ratio']:.2f}x")
                # 조기 kill: step 500 이 threshold 이하면
                if ckpt_name == "ckpt_step000500.pt" and not killed and m['ratio'] < MIN_MASS_500:
                    log(f"⚠️ 조기 kill 조건 발동: step500 mass ratio {m['ratio']:.2f}x < {MIN_MASS_500}x", section=True)
                    if STAGE3_JOB:
                        sh(f"scancel {STAGE3_JOB}")
                        log(f"scancel {STAGE3_JOB} 실행")
                    killed = True

        # Ablation 결과 확인
        for ckpt_name, (jid, tag) in list(ab_jobs.items()):
            if ckpt_name in ab_done:
                continue
            if job_running(jid):
                continue
            a = parse_ablation(tag)
            if a:
                ab_done[ckpt_name] = a
                log(f"[{ckpt_name}] ABL: gt-zero/nfl={a['gt_zero_nfl']:.2f}x gt-rev/nfl={a['gt_rev_nfl']:.2f}x "
                    f"(gt-rev={a['gt_rev']:.2f}, gt-zero={a['gt_zero']:.2f}, zero-randn={a['zero_randn']:.2f})")

        # 종료 조건
        stage3_gone = STAGE3_JOB and not job_running(STAGE3_JOB)
        if stage3_gone and not any(job_running(j) for j, _ in list(mass_jobs.values()) + list(ab_jobs.values())):
            log("=" * 40, section=True)
            log("모든 job 종료. 최종 요약:")
            for ckpt_name in sorted(seen_ckpts):
                m = mass_done.get(ckpt_name, {})
                a = ab_done.get(ckpt_name, {})
                log(f"  {ckpt_name}: mass ratio={m.get('ratio','N/A')} gt-rev/nfl={a.get('gt_rev_nfl','N/A')}")
            # Best ckpt selection: mass > 1.5 AND gt-rev/nfl > 1.0
            best = None
            best_score = -1
            for ckpt_name in seen_ckpts:
                m = mass_done.get(ckpt_name, {}).get("ratio", 0)
                r = ab_done.get(ckpt_name, {}).get("gt_rev_nfl", 0)
                if m > 1.5 and r > 1.0:
                    score = m + r
                    if score > best_score:
                        best_score = score
                        best = ckpt_name
            if best:
                log(f"BEST candidate: {best} — 216 inference 대상", section=True)
            else:
                log("PASS 후보 없음. Stage 3 실패. rej_w0025 유지.", section=True)
            break

        time.sleep(60)


if __name__ == "__main__":
    main()
