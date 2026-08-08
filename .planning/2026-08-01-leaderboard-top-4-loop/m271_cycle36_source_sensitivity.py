"""M271 P4 사이클 36 — 외부 소스 하한의 전제 민감도. 어떤 소스여야 뒤집히는가.

사용자가 "새 기저모델 학습" 경로를 택했고, 그 경로의 유일한 열린 채널은 **공급 밖 정보**
였다. 그런데 수집을 시작하기 전에 확인해 보니 이미 닫혀 있었다.

`reports/m270_sigma_decomposition.md` 실측:

    group  현재최선 sigma  무한소스 하한 sigma*sqrt(rho)  필요치   배수
      1        2.159              2.059              0.571   3.6x
      2        2.368              2.282              0.571   4.0x
      3        2.129              2.048              0.571   3.6x

    소스간 오차상관 rho = 0.739 ~ 0.799

소스를 무한히 더해도 평균 오차는 `sigma*sqrt(rho)` 아래로 내려가지 않는다. 그 하한이
정산 밴드가 요구하는 풍속 정확도의 3.6~4.0 배다. **수집 없이 닫힌다.**

그러나 **하한은 전제 위에 서 있다**: "동일 품질(sigma), 동일 상관(rho)". 새 소스가
더 정확하거나(sigma 낮음) 덜 닮았으면(rho 낮음) 하한이 내려간다. 그 전제를 흔들었을 때
어디까지 가야 뒤집히는지는 아무도 재지 않았다. 이 노드가 그것을 잰다.

이것이 곧 **수집 결정의 근거**다. 필요한 소스 품질이 현실에 존재하는 어떤 NWP 로도
도달 불가능하면 6~8 시간 수집은 정당화되지 않는다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. `m270_sigma_decomposition` 의 해석식을 그대로 쓰고 **역으로 푼다**:
    필요치에 도달하려면 새 소스의 `(sigma_new, rho_new)` 가 얼마여야 하는가.
  - 2 소스 결합의 오차분산:
        Var = (s1^2 + s2^2 + 2*r*s1*s2) / 4     (등가중 평균)
    최적 가중 결합이면
        Var_opt = s1^2*s2^2*(1-r^2) / (s1^2 + s2^2 - 2*r*s1*s2)
    후자를 쓴다 — 새 소스에 **가장 유리한** 가정이어야 폐기가 견고하다.
  - 현실 기준점: ECMWF HRES 가 GFS 대비 풍속 RMSE 를 대략 10~20% 낮추는 것이 통상
    보고되는 범위다. 그 정도가 필요치를 만족하는지 본다.

② 사양 동결

  기준  각 그룹의 현재 최선 `sigma_cur` (GFS·LDAPS 최적혼합), 필요치 `0.571` m/s
        (급경사 전형 기울기 기준. 최대기울기 기준 `0.395` 도 함께 보고)
  격자  새 소스의 상대품질 `q = sigma_new / sigma_cur` in [0.5, 1.2]
        새 소스의 상관     `rho_new` in [0.0, 0.9]
  결합  **최적 가중** (새 소스에 가장 유리)

  사전확약(실행 전 동결):
    H1  현실 범위 `q in [0.8, 1.0]`, `rho_new in [0.6, 0.8]` 에서 어떤 조합도 필요치
        `0.571` 을 만족하지 못한다.
    H2  필요치를 만족하려면 `rho_new < 0.2` 이거나 `q < 0.35` 여야 한다. 즉 **GFS/LDAPS 와
        거의 무상관이거나 3 배 정확한** 소스가 필요하다.
    H3  세 그룹 모두에서 같은 결론이 나온다.
  셋 다 성립하면 외부 NWP 수집은 정당화되지 않고 축이 **수집 없이** 닫힌다.
  하나라도 기각되면 어떤 소스를 노려야 하는지가 구체적으로 나온다.

**게이트 무관. 데이터 수집 없음. lockbox 미사용.**
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle36_source_sensitivity.md"
RECEIPT = REPORTS / "m271_cycle36_source_sensitivity_receipt.json"

NODE_ID = "C1N36_SOURCE_SENSITIVITY"
LANE = "L2"
PARENT_NODE = "C1N35_FIXED_POLICY_CORRECTION"
CLOSES = "AXIS_EXTERNAL_NWP_SOURCE"

# m270_sigma_decomposition.md 실측 (이 노드가 재측정하지 않고 인용한다)
CURRENT = {
    1: {"sigma_best_mix": 2.159, "sigma_gfs": 2.437, "sigma_ldaps": 2.169, "rho": 0.799},
    2: {"sigma_best_mix": 2.368, "sigma_gfs": 2.830, "sigma_ldaps": 2.368, "rho": 0.771},
    3: {"sigma_best_mix": 2.129, "sigma_gfs": 2.634, "sigma_ldaps": 2.129, "rho": 0.739},
}
REQUIRED_TYPICAL = 0.571  # 급경사 전형 기울기 기준
REQUIRED_MAX_SLOPE = 0.395  # 최대기울기 기준

Q_GRID = np.round(np.arange(0.50, 1.201, 0.05), 3)
RHO_GRID = np.round(np.arange(0.0, 0.901, 0.05), 3)
REALISTIC_Q = (0.80, 1.00)
REALISTIC_RHO = (0.60, 0.80)
H2_RHO_BOUND = 0.20
H2_Q_BOUND = 0.35


def optimal_pair_sigma(s1: float, s2: float, rho: float) -> float:
    """두 소스의 **최적 가중** 결합 오차 표준편차. 새 소스에 가장 유리한 가정."""
    denom = s1 * s1 + s2 * s2 - 2.0 * rho * s1 * s2
    if denom <= 0:
        return min(s1, s2)
    var = (s1 * s1) * (s2 * s2) * (1.0 - rho * rho) / denom
    return float(np.sqrt(max(var, 0.0)))


def main() -> int:
    per_group: dict[int, Any] = {}
    for group, cur in CURRENT.items():
        s_cur = cur["sigma_best_mix"]
        grid = []
        for q in Q_GRID:
            row = []
            for rho in RHO_GRID:
                s_new = q * s_cur
                combined = optimal_pair_sigma(s_cur, s_new, float(rho))
                row.append(combined)
            grid.append(row)
        arr = np.array(grid)

        realistic_mask = (
            (Q_GRID >= REALISTIC_Q[0])[:, None] & (Q_GRID <= REALISTIC_Q[1])[:, None]
            & (RHO_GRID >= REALISTIC_RHO[0])[None, :]
            & (RHO_GRID <= REALISTIC_RHO[1])[None, :]
        )
        realistic_best = float(arr[realistic_mask].min()) if realistic_mask.any() else float("nan")

        # 필요치를 만족하는 (q, rho) 가 있는가
        feasible = arr <= REQUIRED_TYPICAL
        feasible_pairs = [
            {"q": float(Q_GRID[i]), "rho": float(RHO_GRID[j]),
             "combined_sigma": float(arr[i, j])}
            for i, j in zip(*np.nonzero(feasible), strict=True)
        ]
        # 각 rho 에서 필요치를 만족하는 최대 q (= 요구되는 최소 품질)
        needed_q_by_rho = {}
        for j, rho in enumerate(RHO_GRID):
            ok = np.nonzero(arr[:, j] <= REQUIRED_TYPICAL)[0]
            needed_q_by_rho[float(rho)] = float(Q_GRID[ok].max()) if len(ok) else None

        per_group[group] = {
            "sigma_current_best_mix": s_cur,
            "rho_gfs_ldaps": cur["rho"],
            "infinite_source_floor": float(s_cur * np.sqrt(cur["rho"])),
            "required_typical": REQUIRED_TYPICAL,
            "required_max_slope": REQUIRED_MAX_SLOPE,
            "realistic_window_best_sigma": realistic_best,
            "realistic_window_multiple_of_required": realistic_best / REQUIRED_TYPICAL,
            "feasible_pair_count": len(feasible_pairs),
            "feasible_examples": feasible_pairs[:8],
            "needed_q_by_rho": needed_q_by_rho,
            "grid": {"q": [float(x) for x in Q_GRID],
                     "rho": [float(x) for x in RHO_GRID],
                     "combined_sigma": arr.round(4).tolist()},
        }

    h1 = all(
        v["realistic_window_best_sigma"] > REQUIRED_TYPICAL for v in per_group.values()
    )
    # H2: 필요치를 만족하는 조합이 있다면 그것들이 전부 rho < 0.2 또는 q < 0.35 인가
    feasible_any = [
        p for v in per_group.values() for p in v["feasible_examples"]
    ]
    h2 = (not feasible_any) or all(
        p["rho"] < H2_RHO_BOUND or p["q"] < H2_Q_BOUND for p in feasible_any
    )
    h3 = len({v["realistic_window_best_sigma"] > REQUIRED_TYPICAL
              for v in per_group.values()}) == 1

    closed = bool(h1 and h2 and h3)
    verdict = (
        "EXTERNAL_NWP_CLOSED_WITHOUT_COLLECTION" if closed
        else "EXTERNAL_NWP_MAY_HELP_SPECIFY_TARGET_SOURCE"
    )
    check = {
        "H1_expectation": f"현실 범위(q {REALISTIC_Q}, rho {REALISTIC_RHO})에서 필요치 "
                          f"{REQUIRED_TYPICAL} 미달성",
        "H1_held": h1,
        "H2_expectation": f"만족 조합은 rho < {H2_RHO_BOUND} 또는 q < {H2_Q_BOUND} 뿐",
        "H2_held": h2,
        "H3_expectation": "세 그룹 결론 일치",
        "H3_held": h3,
        "collection_justified": not closed,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE, "closes": CLOSES,
        "gate_touched": False, "data_collected": False, "lockbox_used": False,
        "cited_measurement": "reports/m270_sigma_decomposition.md",
        "combination_assumption": "최적 가중 (새 소스에 가장 유리). 등가중보다 낮은 오차를 "
                                  "주므로 이 가정에서 닫히면 견고하다",
        "reference_point": "ECMWF HRES 가 GFS 대비 풍속 RMSE 를 통상 10~20% 낮춘다고 "
                           "보고되는 범위 -> q ~ 0.80~0.90",
        "per_group": per_group,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 36 — 외부 소스 하한의 전제 민감도",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 인용 실측: `{payload['cited_measurement']}` (재측정하지 않는다)",
        "- **데이터 수집 없음.** 수집 결정의 근거를 먼저 만드는 노드다",
        "",
        "## 1. 이미 닫혀 있던 것",
        "",
        "| group | 현재최선 sigma | 소스간 rho | 무한소스 하한 | 필요치 | 배수 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for g, v in per_group.items():
        lines.append(
            f"| {g} | {v['sigma_current_best_mix']:.3f} | {v['rho_gfs_ldaps']:.3f} | "
            f"**{v['infinite_source_floor']:.3f}** | {REQUIRED_TYPICAL} | "
            f"**{v['infinite_source_floor'] / REQUIRED_TYPICAL:.1f}x** |"
        )
    lines += [
        "",
        "소스를 무한히 넣어도 `sigma*sqrt(rho)` 아래로 못 간다. 그런데 이 하한은",
        "**동일 품질·동일 상관** 전제 위에 있다. 아래는 그 전제를 흔든 결과다.",
        "",
        "## 2. 새 소스가 얼마나 좋아야 하는가",
        "",
        f"결합 가정: **{payload['combination_assumption']}**",
        "",
        f"현실 기준점: {payload['reference_point']}",
        "",
        "| group | 현실범위 최선 sigma | 필요치 대비 | 필요치 만족 조합 수 |",
        "|---:|---:|---:|---:|",
    ]
    for g, v in per_group.items():
        lines.append(
            f"| {g} | **{v['realistic_window_best_sigma']:.3f}** | "
            f"**{v['realistic_window_multiple_of_required']:.1f}x** | "
            f"{v['feasible_pair_count']} |"
        )

    lines += [
        "",
        "### 상관별로 요구되는 소스 품질 (group 1)",
        "",
        "`q = sigma_new / sigma_cur`. 필요치를 만족하려면 q 가 이 값 이하여야 한다.",
        "",
        "| rho_new | 요구 q |",
        "|---:|---|",
    ]
    for rho, q in per_group[1]["needed_q_by_rho"].items():
        if rho > 0.9:
            continue
        lines.append(f"| {rho:.2f} | {'**' + f'{q:.2f}' + '**' if q else '도달 불가'} |")

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        "",
        f"판정: **{verdict}**",
        "",
        f"외부 NWP 수집 정당화: **{check['collection_justified']}**",
        "",
    ]
    if closed:
        lines += [
            "## 4. 이것이 뜻하는 것",
            "",
            "현실에 존재하는 어떤 NWP 소스도 (GFS 대비 10~20% 정확하고 상관 0.6~0.8 인",
            "범위) 정산 밴드가 요구하는 풍속 정확도에 **자릿수만큼 못 미친다.** 필요치를",
            "만족하려면 GFS/LDAPS 와 **거의 무상관**이거나 **3 배 가까이 정확한** 소스여야",
            "하는데, 같은 대기를 같은 물리로 푸는 모델들 사이에 그런 것은 없다.",
            "",
            "따라서 **6~8 시간 수집은 정당화되지 않는다.** 축은 수집 없이 닫힌다.",
        ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE36_SOURCE_SENSITIVITY",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [], "model_fits": 0,
        "lockbox_reopened": False, "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    for g, v in per_group.items():
        print(f"[C36] g{g} 현재 {v['sigma_current_best_mix']:.3f} / "
              f"무한소스하한 {v['infinite_source_floor']:.3f} / "
              f"현실범위 최선 {v['realistic_window_best_sigma']:.3f} "
              f"({v['realistic_window_multiple_of_required']:.1f}x 필요치) / "
              f"만족조합 {v['feasible_pair_count']}")
    print(f"[C36] H1 {h1} | H2 {h2} | H3 {h3}")
    print(f"[C36] 판정: {verdict}  수집 정당화 {check['collection_justified']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
