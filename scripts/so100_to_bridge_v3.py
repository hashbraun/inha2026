"""SO-100 관절각도 → Cosmos3 action 토큰 37D (v3: 절대 기하 복원).

## 왜 v3인가

v2(delta 10D)는 Cosmos3 공식 규약을 따르지만 **두 가지 정보를 잃는다**:

1. **절대 앵커 없음** — delta 시퀀스만으로는 팔의 절대 configuration이 미결정이다.
2. **팔 전체 형상 없음** — 10D는 EE flange만 담는다. eval 216샘플 실측으로 중간 링크가
   EE의 51~93%만큼 움직인다(j3 92mm / j4 144mm / j5 166mm / j6 179mm). DINO·Video 지표는
   프레임 전체 픽셀을 보므로 이 링크들이 화면에서 차지하는 비중이 작지 않다.

action 토큰은 64차원인데 v2는 10차원만 쓰고 54차원이 zero-pad다
(`finetune_cosmos3_nano.py:194-195`). 그 빈 채널에 절대 기하를 넣으면
**transformer 구조·토큰 수·시퀀스 길이·FLOPs 변화 없이** 정보가 추가된다.

핵심 기대효과는 **암묵적 캘리브레이션**이다. 모델은 프레임 0에서 팔의 픽셀을 보면서
동시에 그 팔의 3D 좌표를 받게 되므로 2D-3D 대응이 성립한다. delta만으로는 이 대응이 없다.

## 레이아웃 (37D, 나머지 27채널은 의도적으로 비움)

```
[ 0:10]  v2 delta 10D                  (대조 유지, 기존과 완전히 동일)
[10:13]  절대 EE translation 3D        정규화
[13:19]  절대 EE rot6d 6D              이미 ±1이라 그대로
[19:31]  FK 관절 2~5 위치 12D          정규화
[31:37]  관절각 6D                     per-joint 정규화
[37:64]  예약 (Phase 4의 투영 keypoint 자리)
```

### 왜 FK가 21D가 아니라 12D인가 (실측, 54531 스텝)
- **joint 0·1은 상수** — `positions[i+1]=T[:3,3]`이라 관절 회전이 위치를 안 바꾼다. 6D 낭비
- **joint 6은 절대 EE translation과 동일** — 3D 중복
→ 정보를 담는 건 joint 2~5의 12D뿐이다.

### 왜 관절각 6D를 따로 넣는가 (FK와 중복 아님)
FK 위치 배열은 **마지막 관절에 눈이 멀어 있다**. joint5를 +30° 돌려도 위치 변화 0.00mm이고
rot6d만 0.481 바뀐다. 손목 회전 정보는 rot6d/관절각에만 존재한다.

## 시간 정렬

실측(α=0.958, 픽셀↔관절속도 상관 lag+1)으로 **`pose(action[t])`는 frame t+1의 자세**다.
따라서 토큰 t에 `pose(joints[t])`를 그대로 넣으면 토큰 0~14가 frame 1~15를 덮는다 — 시프트 불필요.
반면 delta 항은 `pose(a[t+1])−pose(a[t])`라 전이 t+1→t+2를 가리킨다(v2 그대로 유지).

## 정규화

`data/train/so100_geom_statistics.json` ([1%,99%] → ±1). `scripts/compute_geom_stats.py`로 재생성.
**반드시 `action`(command) 기준**이다. eval에 observation.state가 없으므로 학습도 action이어야 한다.
(캘리브레이션용 FK는 반대로 state를 쓴다 — `docs/plans/camera-calibration/PLAN.md`)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home1/sota/inha2026/baseline/challenge_kit/scripts")
from so100_fk import so100_fk  # noqa: E402

sys.path.insert(0, "/home1/sota/inha2026/scripts")
from so100_to_bridge import joint_deg_to_bridge10  # noqa: E402  (절대 EE pose)
from so100_to_bridge_v2 import joint_deg_to_bridge10_delta  # noqa: E402

STATS_PATH = Path("/home1/sota/inha2026/data/train/so100_geom_statistics.json")
FEATURE_DIM = 37
# 정규화 후 허용 범위. 이상치 폭주는 막되 정상적인 분포 밖 값은 살린다.
# 실측(eval 216샘플): 클리핑 없는 최대 |x| = 1.86, |x|>2 는 0.000%.
# joint1은 eval의 10.3%가 train min/max 밖이라 여유가 필요하다 — 2.5면 실제로 절단이 일어나지 않는다.
CLIP = 2.5

_STATS = None


def _stats():
    global _STATS
    if _STATS is None:
        if not STATS_PATH.exists():
            raise FileNotFoundError(
                f"{STATS_PATH} 없음. 먼저 실행: python scripts/compute_geom_stats.py"
            )
        with open(STATS_PATH) as f:
            s = json.load(f)
        _STATS = {
            k: (np.asarray(s[k]["lo"], np.float32), np.asarray(s[k]["hi"], np.float32))
            for k in ("joint_deg", "fk_pos_m", "ee_trans_m")
        }
        _STATS["fk_joints"] = s["fk_joints"]
    return _STATS


def _norm(x, lo, hi):
    """[lo,hi] → [-1,1], CLIP으로 절단."""
    return np.clip(2.0 * (x - lo) / (hi - lo) - 1.0, -CLIP, CLIP).astype(np.float32)


def joint_deg_to_bridge_v3(joint_angles_deg: np.ndarray, delta_frame: str = "base") -> np.ndarray:
    """
    joint_angles_deg: (T, 6) degree
    returns: (T, 37) float32
    """
    q = np.asarray(joint_angles_deg, dtype=np.float32)
    assert q.ndim == 2 and q.shape[1] == 6, q.shape
    st = _stats()
    T = q.shape[0]

    out = np.zeros((T, FEATURE_DIM), dtype=np.float32)

    # [0:10] v2 delta — 기존 표현을 그대로 보존한다(대조군과 동일한 채널)
    out[:, 0:10] = joint_deg_to_bridge10_delta(q, frame=delta_frame)

    # [10:19] 절대 EE pose. joint_deg_to_bridge10은 [trans3, rot6d, gripper(원값)] 순이며
    #         gripper 원값은 delta 항의 정규화 그리퍼와 중복이므로 버린다.
    abs10 = joint_deg_to_bridge10(q)
    lo, hi = st["ee_trans_m"]
    out[:, 10:13] = _norm(abs10[:, 0:3], lo, hi)
    out[:, 13:19] = np.clip(abs10[:, 3:9], -CLIP, CLIP)      # rot6d는 이미 ±1

    # [19:31] FK 관절 2~5 위치.
    # fk_pos[2](=joint2의 z)는 이 로봇 구조상 상수다. 입력 상수축은 projection의 bias와
    # 동치라 무해하므로, 인덱스 관리를 복잡하게 만들면서까지 빼지 않는다.
    P = so100_fk(q)[:, st["fk_joints"], :].reshape(T, -1)     # (T,12)
    lo, hi = st["fk_pos_m"]
    out[:, 19:31] = _norm(P, lo, hi)

    # [31:37] 관절각 per-joint 정규화 (joint5 = 그리퍼의 절대 개도도 여기 포함)
    lo, hi = st["joint_deg"]
    out[:, 31:37] = _norm(q, lo, hi)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 표현 분기 + 파이프라인 입력 생성 — 학습/추론/홀드아웃이 갈라지지 않도록 여기 한 곳에 둔다.
# (`--action-repr` 불일치는 이 프로젝트에서 반복된 사고 지점이다. CLAUDE.md 참조)
# ─────────────────────────────────────────────────────────────────────────────

# 스크리닝 사다리 — B1→B3→B4로 정보량이 단조 증가하고, B2는 v4/v5 교란을 분리한다.
#   delta_base     10D  B1 대조군 (= v5_base 레시피)
#   absolute_ngrip 10D  B2 절대 EE pose + 정규화 그리퍼.
#                       v4는 절대 표현이면서 그리퍼가 원값(0~119)이라 나머지 9차원을 20~50배
#                       압도했다. v5의 개선이 delta 때문인지 그리퍼 정규화 때문인지 분리한다.
#   geom_ee        19D  B3 delta + 절대 EE pose. "절대 앵커"만의 효과
#   geom_v3        37D  B4 위 + FK 관절 위치 + 관절각. "팔 전체 형상"까지
#   delta_base_shift 10D  B5 정렬만 수정. geom_v3의 개선이 "절대 기하" 때문인지
#                         "정렬 수정" 때문인지 분리한다 (geom_v3는 절대 항이 이미 정렬돼 있어
#                         두 요인이 섞여 있다)
ACTION_REPRS = ["absolute", "absolute_ngrip", "delta_local", "delta_base",
                "delta_base_shift", "delta_local_shift", "geom_ee", "geom_v3", "geom_v3_local",
                "geom_v4_46d", "geom_v4_46d_local"]

FEATURE_DIMS = {"absolute": 10, "absolute_ngrip": 10, "delta_local": 10, "delta_base": 10,
                "delta_base_shift": 10, "delta_local_shift": 10,
                "geom_ee": 19, "geom_v3": 37, "geom_v3_local": 37,
                "geom_v4_46d": 46, "geom_v4_46d_local": 46}

# 46D 전용 stats path (7관절 FK 포함, compute_geom_stats.py --fk-joints 0 1 2 3 4 5 6로 생성)
STATS_PATH_46D = Path("/home1/sota/inha2026/data/train/so100_geom_statistics_46d.json")
FEATURE_DIM_46D = 46
_STATS_46D = None


def _stats_46d():
    """46D 표현용 stats. fk_pos_m이 21D (7관절 × xyz)."""
    global _STATS_46D
    if _STATS_46D is None:
        if not STATS_PATH_46D.exists():
            raise FileNotFoundError(
                f"{STATS_PATH_46D} 없음. 먼저 실행: "
                f"python scripts/compute_geom_stats.py --fk-joints 0 1 2 3 4 5 6 --out {STATS_PATH_46D}"
            )
        with open(STATS_PATH_46D) as f:
            s = json.load(f)
        _STATS_46D = {
            k: (np.asarray(s[k]["lo"], np.float32), np.asarray(s[k]["hi"], np.float32))
            for k in ("joint_deg", "fk_pos_m", "ee_trans_m")
        }
    return _STATS_46D


def joint_deg_to_bridge46(joint_angles_deg: np.ndarray, delta_frame: str = "local") -> np.ndarray:
    """(T, 6) degree → (T, 46) float32.

    ## 레이아웃
    ```
    [ 0:10]  delta 10D (delta_frame='local' 또는 'base')
    [10:13]  절대 EE translation 3D  ([1%,99%] normalized)
    [13:19]  절대 EE rot6d 6D        (±1 clip)
    [19:40]  FK 7관절 위치 21D       (base + 6 joints, [1%,99%] normalized)
    [40:46]  관절각 per-joint 6D     ([1%,99%] normalized)
    ```

    ## v3(37D) 대비 확장
    - FK: 4관절(12D) → 7관절 전체(21D). joint 0/1은 상수 채널이지만 인덱스 관리 위해 유지
      (projection의 bias에 흡수, 무해). joint 6은 EE translation과 중복이지만
      학습이 자동으로 중복성 처리 가능.
    - 시간 축은 v3와 동일 (`pose(a[t])`는 frame t+1 자세, shift 불필요)

    ## delta_frame 선택
    - 'local': 이전 프레임 EE 좌표계 기준 (v5_base와 다른 축 — 실험 독립성)
    - 'base': 로봇 base 좌표계 기준 (v5_base와 동일)
    """
    q = np.asarray(joint_angles_deg, dtype=np.float32)
    assert q.ndim == 2 and q.shape[1] == 6, q.shape
    st = _stats_46d()
    T = q.shape[0]

    out = np.zeros((T, FEATURE_DIM_46D), dtype=np.float32)

    # [0:10] delta (local 또는 base)
    out[:, 0:10] = joint_deg_to_bridge10_delta(q, frame=delta_frame)

    # [10:19] 절대 EE pose. joint_deg_to_bridge10은 [trans3, rot6d, gripper] 순
    abs10 = joint_deg_to_bridge10(q)
    lo, hi = st["ee_trans_m"]
    out[:, 10:13] = _norm(abs10[:, 0:3], lo, hi)
    out[:, 13:19] = np.clip(abs10[:, 3:9], -CLIP, CLIP)      # rot6d는 이미 ±1

    # [19:40] FK 7관절 위치 (base + 6 joints)
    P = so100_fk(q).reshape(T, -1)     # (T, 21)  — 7 keypoints × xyz
    lo, hi = st["fk_pos_m"]
    out[:, 19:40] = _norm(P, lo, hi)

    # [40:46] 관절각 per-joint 정규화
    lo, hi = st["joint_deg"]
    out[:, 40:46] = _norm(q, lo, hi)

    return out


def joint_deg_to_bridge10_absolute_ngrip(joint_angles_deg: np.ndarray) -> np.ndarray:
    """절대 EE pose 9D + 클립 내 min-max 정규화 그리퍼 1D.

    v2와 동일한 그리퍼 정규화를 절대 표현에 적용한 것 — 오직 그 차이만 갖는다.
    """
    from so100_to_bridge_v2 import _normalize_gripper

    q = np.asarray(joint_angles_deg, dtype=np.float32)
    out = joint_deg_to_bridge10(q).copy()
    out[:, 9] = _normalize_gripper(q[:, 5])
    return out


def joint_deg_to_bridge10_delta_shift(joint_angles_deg: np.ndarray, frame: str = "base") -> np.ndarray:
    """시간 정렬을 고친 delta 10D.

    기존 delta는 `out[t] = pose(a[t+1]) − pose(a[t])`인데, 실측(α=0.958, 상관 lag+1)으로
    `pose(a[t])`가 frame t+1의 자세이므로 이건 **전이 t+1→t+2**를 가리킨다.
    토큰 t가 가리켜야 할 것은 전이 t→t+1 = `pose(a[t]) − pose(a[t−1])`이다.

    a[−1]은 eval에 없으므로 관절 공간에서 뒤로 외삽한다: `a[−1] ≈ 2·a[0] − a[1]`.
    **학습에서도 동일하게 외삽한다.** 학습 클립은 이전 프레임 action을 실제로 갖고 있지만,
    그걸 쓰면 추론과 불일치가 생긴다(이 프로젝트에서 반복된 사고 유형).

    이 표현의 목적은 오직 하나 — geom_v3의 개선이 "절대 기하" 때문인지
    "정렬 수정" 때문인지 분리하는 것.
    """
    q = np.asarray(joint_angles_deg, dtype=np.float32)
    assert q.ndim == 2 and q.shape[1] == 6, q.shape
    if len(q) < 2:
        return joint_deg_to_bridge10_delta(q, frame=frame)
    q_prev = (2.0 * q[0] - q[1])[None]                     # 외삽한 a[-1]
    ext = np.concatenate([q_prev, q], axis=0)              # (T+1, 6)
    return joint_deg_to_bridge10_delta(ext, frame=frame)[: len(q)]


def build_action_features(joints: np.ndarray, repr_name: str) -> np.ndarray:
    """(T,6) degree → (T,D). D는 표현에 따라 10 / 19 / 37."""
    if repr_name == "absolute":
        return joint_deg_to_bridge10(joints)
    if repr_name == "absolute_ngrip":
        return joint_deg_to_bridge10_absolute_ngrip(joints)
    if repr_name.endswith("_shift"):
        return joint_deg_to_bridge10_delta_shift(joints, frame=repr_name[len("delta_"):-len("_shift")])
    if repr_name.startswith("delta_"):
        return joint_deg_to_bridge10_delta(joints, frame=repr_name[len("delta_"):])
    if repr_name == "geom_ee":
        return joint_deg_to_bridge_v3(joints)[:, :19]          # delta 10 + 절대 EE pose 9
    if repr_name.startswith("geom_v3"):
        return joint_deg_to_bridge_v3(joints, delta_frame="local" if repr_name.endswith("_local") else "base")
    if repr_name.startswith("geom_v4_46d"):
        return joint_deg_to_bridge46(joints, delta_frame="local" if repr_name.endswith("_local") else "base")
    raise ValueError(f"알 수 없는 action 표현: {repr_name!r} (가능: {ACTION_REPRS})")


def make_action_condition(raw_actions, *, chunk_size, domain_name, resolution_tier, image,
                          view_point="third_person_view", mode="forward_dynamics"):
    """CosmosActionCondition 생성. 10D가 아닌 표현도 통과시킨다.

    파이프라인은 `__post_init__`에서 `raw_actions.shape[1] == raw_action_dim(=10)`을 검증하지만
    frozen dataclass가 아니라 **사후 대입은 통과**한다(실측). 이후 파이프라인 862행이
    `shape[-1] < action_dim(64)`이면 zero-pad하므로 37D도 그대로 쓰인다.

    ※ 이 우회는 diffusers와 우리 코드에만 적용된다. `submission_kit`은 규칙 4에 따라 수정 금지.
    """
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition

    width = raw_actions.shape[-1]
    placeholder = raw_actions[:, :10] if width > 10 else raw_actions
    cond = CosmosActionCondition(
        mode=mode, chunk_size=chunk_size, domain_name=domain_name,
        resolution_tier=resolution_tier, raw_actions=placeholder,
        image=image, view_point=view_point,
    )
    if width > 10:
        cond.raw_actions = raw_actions          # 검증 우회 (사후 대입)
    return cond


if __name__ == "__main__":
    import glob

    ev = sorted(glob.glob("/home1/sota/inha2026/data/eval/actions/*.npy"))
    X = np.stack([joint_deg_to_bridge_v3(np.load(f)) for f in ev])
    print(f"eval {len(ev)}샘플 → {X.shape}")
    names = [("delta", 0, 10), ("ee_trans", 10, 13), ("ee_rot6d", 13, 19),
             ("fk_pos", 19, 31), ("joint_deg", 31, 37)]
    print(f"\n{'블록':>10s} {'차원':>5s} {'min':>8s} {'max':>8s} {'평균|x|':>8s} {'상수축':>7s}")
    for n, a, b in names:
        blk = X[:, :, a:b]
        span = blk.reshape(-1, b - a).ptp(0)
        print(f"{n:>10s} {b-a:5d} {blk.min():8.3f} {blk.max():8.3f} "
              f"{np.abs(blk).mean():8.3f} {int((span < 1e-6).sum()):7d}")
    sat = float((np.abs(X) >= CLIP - 1e-6).mean())
    print(f"\nCLIP({CLIP}) 포화 비율: {sat*100:.3f}%   (높으면 통계 재산출 필요)")
    print(f"전 차원 상수 여부: {int((X.reshape(-1, FEATURE_DIM).ptp(0) < 1e-6).sum())}개 상수축")
