"""M271 P4 사이클 47 — 외부 NWP 축 부활 (C9). 사이클 36 의 요구치가 틀렸다.

사이클 46 이 반증에 성공했다. 사이클 36 은 외부 NWP 를 "물리적으로 불가" 로 닫았는데,
그 판정은 **요구치를 잘못 잡은** 것이었다.

  사이클 36 이 쓴 요구치 `0.571 m/s`
      = 급경사 구간에서 **점별로** `|출력오차| <= 6% 용량` 을 만족하기 위한 풍속 정확도
  사이클 46 이 역산한 요구치 `1.817 m/s`
      = 상위권 FICR(로컬환산 0.45564)에 도달하기 위한 풍속 정확도

**3.2 배 차이**다. FICR 은 발전량가중 평균이라 **평탄 구간에서도 점수를 준다.** 급경사에서
점별로 밴드에 들지 못해도 전체 FICR 은 오를 수 있다. 나는 "점별로 못 들면 FICR 도 못 올린다"
고 암묵 가정했고, 그것이 틀렸다.

정정된 요구치로 사이클 36 의 민감도 격자를 다시 돌린다. 사이클 36 의 하한 계산 자체는
건드리지 않는다 — 틀린 것은 요구치뿐이다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 36 의 `optimal_pair_sigma` 를 그대로 재사용하고 **요구치만** 바꾼다.
    한 변수만 움직여야 귀속이 된다.
  - 요구치는 사이클 46 이 **실측 예측에서 역산**한 값이다. 파워커브 기울기에서 유도한
    이론값이 아니라 이 데이터의 실제 오차분포에서 나온 것이므로 더 직접적이다.

② 사양 동결

  요구치  `1.817` m/s (상위권 FICR 로컬환산), 부차적으로 `1.871` (로컬 Total 0.66)
          비교용으로 사이클 36 의 `0.571` 도 함께 보고
  격자    사이클 36 과 동일: `q = sigma_new/sigma_cur in [0.5, 1.2]`,
          `rho_new in [0.0, 0.9]`, 최적 가중 결합
  현실범위 `q in [0.8, 1.0]`, `rho in [0.6, 0.8]` (ECMWF 급)

  사전확약(실행 전 동결):
    H1  정정된 요구치(1.817) 하에서 **현실범위 안에 만족 조합이 존재**한다.
    H2  세 그룹 모두에서 존재한다.
    H3  로컬 Total 0.66 요구치(1.871) 하에서도 존재한다.
    H4  사이클 36 의 요구치(0.571) 하에서는 여전히 **존재하지 않는다**.
        (사이클 36 의 계산 자체는 옳았고 요구치만 틀렸음을 확인)

  H1·H2 가 성립하면 `EXTERNAL_NWP_PHYSICALLY_INSUFFICIENT` 전제가 **뒤집히고** 축이
  부활한다. 그러면 외부 NWP 수집이 처음으로 정당화된다.

**새 학습·수집·게이트 수정 없음.** 이 노드는 재계산과 전제 판정이다.
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

from m271_cycle36_source_sensitivity import (
    CURRENT,
    Q_GRID,
    REALISTIC_Q,
    REALISTIC_RHO,
    RHO_GRID,
    optimal_pair_sigma,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle47_external_nwp_revived.md"
RECEIPT = REPORTS / "m271_cycle47_external_nwp_revived_receipt.json"

NODE_ID = "C1N47_EXTERNAL_NWP_REVIVED"
LANE = "L2"
PARENT_NODE = "C1N46_CLOSURE_FALSIFICATION"
REVIVES = "AXIS_EXTERNAL_NWP_SOURCE"
FLIPS_PREMISE = "EXTERNAL_NWP_PHYSICALLY_INSUFFICIENT"

REQUIREMENTS = {
    "leader_ficr_local_equiv": 1.817,
    "local_total_066": 1.871,
    "cycle36_pointwise": 0.571,
}


def feasibility(sigma_cur: float, required: float) -> dict[str, Any]:
    grid = np.array(
        [[optimal_pair_sigma(sigma_cur, q * sigma_cur, float(r)) for r in RHO_GRID]
         for q in Q_GRID]
    )
    feasible = grid <= required
    realistic = (
        (Q_GRID >= REALISTIC_Q[0])[:, None] & (Q_GRID <= REALISTIC_Q[1])[:, None]
        & (RHO_GRID >= REALISTIC_RHO[0])[None, :] & (RHO_GRID <= REALISTIC_RHO[1])[None, :]
    )
    realistic_best = float(grid[realistic].min()) if realistic.any() else float("nan")
    realistic_feasible = bool((feasible & realistic).any())
    pairs = [
        {"q": float(Q_GRID[i]), "rho": float(RHO_GRID[j]), "sigma": float(grid[i, j])}
        for i, j in zip(*np.nonzero(feasible & realistic), strict=True)
    ]
    return {
        "required": required,
        "realistic_best_sigma": realistic_best,
        "margin": required - realistic_best,
        "feasible_anywhere": bool(feasible.any()),
        "feasible_in_realistic_window": realistic_feasible,
        "realistic_feasible_count": len(pairs),
        "examples": pairs[:6],
    }


def main() -> int:
    per_group: dict[int, Any] = {}
    for group, cur in CURRENT.items():
        s = cur["sigma_best_mix"]
        per_group[group] = {
            "sigma_current": s,
            "rho_gfs_ldaps": cur["rho"],
            **{name: feasibility(s, req) for name, req in REQUIREMENTS.items()},
        }

    leader_ok = [per_group[g]["leader_ficr_local_equiv"] for g in per_group]
    total_ok = [per_group[g]["local_total_066"] for g in per_group]
    point_ok = [per_group[g]["cycle36_pointwise"] for g in per_group]

    h1 = bool(leader_ok[0]["feasible_in_realistic_window"])
    h2 = bool(all(v["feasible_in_realistic_window"] for v in leader_ok))
    h3 = bool(all(v["feasible_in_realistic_window"] for v in total_ok))
    h4 = bool(not any(v["feasible_in_realistic_window"] for v in point_ok))

    revived = bool(h1 and h2)
    verdict = (
        "EXTERNAL_NWP_AXIS_REVIVED_COLLECTION_JUSTIFIED" if revived
        else "STILL_CLOSED_UNDER_CORRECTED_REQUIREMENT"
    )
    check = {
        "H1_expectation": f"정정 요구치 {REQUIREMENTS['leader_ficr_local_equiv']} 하에 "
                          "현실범위 만족 조합 존재 (그룹1)",
        "H1_held": h1,
        "H2_expectation": "세 그룹 모두 존재",
        "H2_held": h2,
        "H3_expectation": f"로컬 Total 0.66 요구치 "
                          f"{REQUIREMENTS['local_total_066']} 하에도 존재",
        "H3_held": h3,
        "H4_expectation": f"사이클 36 요구치 {REQUIREMENTS['cycle36_pointwise']} 하에는 "
                          "여전히 부존재 (36 의 계산은 옳았음을 확인)",
        "H4_held": h4,
        "premise_flipped": revived,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "revives": REVIVES, "flips_premise": FLIPS_PREMISE,
        "error_found": "사이클 36 은 요구치를 급경사 **점별** 밴드 적중 기준(0.571 m/s)으로 "
                       "잡았다. FICR 은 발전량가중 평균이라 평탄 구간에서도 점수를 주므로 "
                       "실제 요구치는 사이클 46 역산으로 1.817 m/s — **3.2 배 관대**하다",
        "requirements": REQUIREMENTS,
        "per_group": per_group,
        "no_training": True, "no_collection": True, "gate_touched": False,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 47 — 외부 NWP 축 부활",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 부활 대상: `{REVIVES}` / 뒤집는 전제: `{FLIPS_PREMISE}`",
        "- **새 학습·수집 없음.** 사이클 36 의 계산을 요구치만 바꿔 다시 돌린다",
        "",
        "## 1. 무엇이 틀렸는가",
        "",
        payload["error_found"] + ".",
        "",
        "| 요구치 출처 | 값 (m/s) |",
        "|---|---:|",
        f"| 사이클 36 — 급경사 점별 밴드 적중 | {REQUIREMENTS['cycle36_pointwise']} |",
        f"| **사이클 46 — 상위권 FICR 역산** | **{REQUIREMENTS['leader_ficr_local_equiv']}** |",
        f"| 사이클 46 — 로컬 Total 0.66 역산 | {REQUIREMENTS['local_total_066']} |",
        "",
        "## 2. 정정된 요구치 하의 실현가능성",
        "",
        "결합 가정은 사이클 36 과 동일한 **최적 가중**, 현실범위는 "
        f"`q {REALISTIC_Q}` x `rho {REALISTIC_RHO}` (ECMWF 급).",
        "",
        "| group | 현재 sigma | 현실범위 최선 | 요구치 1.817 | 여유 | 만족 조합 |",
        "|---:|---:|---:|:---:|---:|---:|",
    ]
    for g, v in per_group.items():
        f = v["leader_ficr_local_equiv"]
        lines.append(
            f"| {g} | {v['sigma_current']:.3f} | {f['realistic_best_sigma']:.3f} | "
            f"{'**가능**' if f['feasible_in_realistic_window'] else '불가'} | "
            f"{f['margin']:+.3f} | {f['realistic_feasible_count']} |"
        )
    lines += [
        "",
        "| group | 요구치 1.871 (Total 0.66) | 요구치 0.571 (사이클 36) |",
        "|---:|:---:|:---:|",
    ]
    for g, v in per_group.items():
        a = v["local_total_066"]["feasible_in_realistic_window"]
        b = v["cycle36_pointwise"]["feasible_in_realistic_window"]
        lines.append(
            f"| {g} | {'**가능**' if a else '불가'} | {'가능' if b else '불가 (36 재현)'} |"
        )

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}**",
        "",
        f"판정: **{verdict}**",
        "",
        f"전제 `{FLIPS_PREMISE}` 뒤집힘: **{revived}**",
        "",
    ]
    if revived:
        lines += [
            "## 4. 이것이 여는 것",
            "",
            "외부 NWP 소스 추가가 **처음으로 정당화된다.** 필요한 것은 오차 16% 감소이고,",
            "사이클 36 자신의 계산이 ECMWF 급 소스(q 0.8~1.0, rho 0.6~0.8)와의 최적 결합으로",
            f"{per_group[1]['leader_ficr_local_equiv']['realistic_best_sigma']:.3f} m/s "
            "(22% 감소)가 가능하다고 했다.",
            "",
            "다만 남는 조건 둘을 명시한다.",
            "1. **요구치 1.817 은 국소 선형 근사**로 환산한 값이다(출력오차 ∝ 풍속오차).",
            "2. 사이클 10 이 GFS+LDAPS **블렌딩**의 실측 이득을 0.10~0.37% 로 쟀다. 그것은",
            "   두 공급 소스 간이고, 새 소스는 rho 가 더 낮을 수 있다는 것이 이 계산의 전제다.",
            "   **rho 가 실제로 낮은지는 수집 전에는 알 수 없다.**",
        ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE47_EXTERNAL_NWP_REVIVED",
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
        f = v["leader_ficr_local_equiv"]
        print(f"[C47] g{g} 현재 {v['sigma_current']:.3f} / 현실범위 최선 "
              f"{f['realistic_best_sigma']:.3f} / 요구 1.817 -> "
              f"{'가능' if f['feasible_in_realistic_window'] else '불가'} "
              f"(여유 {f['margin']:+.3f}, 조합 {f['realistic_feasible_count']})")
    print(f"[C47] H1 {h1} | H2 {h2} | H3 {h3} | H4 {h4}")
    print(f"[C47] 판정: {verdict}  전제 뒤집힘 {revived}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
