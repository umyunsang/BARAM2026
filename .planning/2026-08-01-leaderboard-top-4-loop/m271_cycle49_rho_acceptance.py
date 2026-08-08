"""M271 P4 사이클 49 — 새 소스의 합격선을 **재기 전에** 확정한다.

사이클 47 이 외부 NWP 축을 열었고 48 이 그 근거를 검증했다. 다음 단계는 새 소스의 오차상관
`rho` 를 재는 것인데, **재고 나서 "이 정도면 되나" 를 판단하면 그게 사후 기준**이다. 이
프로젝트가 반복해서 걸린 함정이 정확히 그것이다.

이 노드는 사이클 47 의 격자를 **역으로 풀어** 수용 영역을 미리 확정한다. 새 소스를 재면
그 값이 영역 안인지 밖인지 **즉시 판정**된다. 협상의 여지가 없다.

수용 조건
    optimal_pair_sigma(sigma_cur, q * sigma_cur, rho) <= 요구치
    요구치 = 1.871 m/s (로컬 Total 0.66, 사이클 46 실측 역산)
    보조   = 1.817 m/s (상위권 FICR)

**새 소스를 수집하지 않는다.** 규칙 조항(`AGENTS.md` 5 행)이 아직 외부 데이터를 금지하고
있으므로 이 노드는 수집 없이 판정 기준만 만든다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 36 의 결합식을 rho 에 대해 역으로 푼다.
  - 사이클 48 이 그 식을 실측으로 검증했다(오차 +0.026~+0.048, 보수적 방향).

② 사양 동결

  요구치  주 `1.871` (로컬 Total 0.66) / 보조 `1.817` (상위권 FICR)
  격자    `q = sigma_new / sigma_cur in [0.6, 1.3]` 간격 0.05
  출력    각 q 에서 **허용되는 최대 rho** (그보다 크면 불합격)
  판정    세 그룹 **모두** 만족해야 합격. 한 그룹이라도 벗어나면 불합격

  사전확약(실행 전 동결):
    H1  `q = 1.0`(새 소스가 현재와 동등 품질)에서 허용 rho 가 존재한다.
    H2  ECMWF 급 가정 `q = 0.85` 에서 허용 rho 가 **0.6 이상**이다.
        (현실적으로 관측되는 모델간 오차상관이 0.7~0.8 이므로, 허용선이 0.6 미만이면
         사실상 도달 불가다)
    H3  세 그룹 중 제약이 가장 빡빡한 그룹을 특정한다 (사이클 47 에서 g2 였다).

  **이 노드가 정한 수용선은 이후 변경하지 않는다.** 새 소스 측정 결과를 보고 완화하면
  기준이 무의미해진다.

**수집·학습·게이트 수정 없음.**
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

from m271_cycle36_source_sensitivity import CURRENT, optimal_pair_sigma

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle49_rho_acceptance.md"
RECEIPT = REPORTS / "m271_cycle49_rho_acceptance_receipt.json"

NODE_ID = "C1N49_RHO_ACCEPTANCE"
LANE = "L2"
PARENT_NODE = "C1N48_MODEL_VALIDATION"

REQUIREMENT_PRIMARY = 1.871  # 로컬 Total 0.66
REQUIREMENT_SECONDARY = 1.817  # 상위권 FICR
Q_GRID = np.round(np.arange(0.60, 1.3001, 0.05), 4)
RHO_SCAN = np.round(np.arange(0.0, 0.9901, 0.005), 4)
H2_Q = 0.85
H2_MIN_ALLOWED_RHO = 0.60


def max_rho_for(sigma_cur: float, q: float, required: float) -> float | None:
    """이 q 에서 요구치를 만족하는 **최대** rho. 없으면 None."""
    allowed = [
        r for r in RHO_SCAN
        if optimal_pair_sigma(sigma_cur, q * sigma_cur, float(r)) <= required
    ]
    return float(max(allowed)) if allowed else None


def main() -> int:
    frontier: dict[str, Any] = {}
    for name, required in (
        ("primary_local_066", REQUIREMENT_PRIMARY),
        ("secondary_leader_ficr", REQUIREMENT_SECONDARY),
    ):
        per_q = []
        for q in Q_GRID:
            per_group = {
                g: max_rho_for(c["sigma_best_mix"], float(q), required)
                for g, c in CURRENT.items()
            }
            values = [v for v in per_group.values() if v is not None]
            binding = (
                min(per_group, key=lambda g: (per_group[g] is None, per_group[g] or -1))
                if per_group else None
            )
            per_q.append(
                {
                    "q": float(q),
                    "max_rho_by_group": per_group,
                    "all_groups_feasible": bool(len(values) == len(CURRENT)),
                    "binding_max_rho": float(min(values)) if len(values) == len(CURRENT)
                    else None,
                    "binding_group": int(binding) if binding is not None else None,
                }
            )
        frontier[name] = per_q

    primary = frontier["primary_local_066"]
    at_one = next(r for r in primary if abs(r["q"] - 1.0) < 1e-9)
    at_h2 = next(r for r in primary if abs(r["q"] - H2_Q) < 1e-9)
    h1 = bool(at_one["all_groups_feasible"])
    h2 = bool(
        at_h2["all_groups_feasible"]
        and at_h2["binding_max_rho"] is not None
        and at_h2["binding_max_rho"] >= H2_MIN_ALLOWED_RHO
    )
    binding_counts: dict[int, int] = {}
    for r in primary:
        if r["binding_group"] is not None:
            binding_counts[r["binding_group"]] = binding_counts.get(r["binding_group"], 0) + 1
    tightest = max(binding_counts, key=binding_counts.get) if binding_counts else None
    h3 = tightest is not None

    verdict = (
        "ACCEPTANCE_REGION_FROZEN_MEASUREMENT_CAN_PROCEED" if (h1 and h2)
        else "ACCEPTANCE_REGION_TOO_TIGHT_COLLECTION_NOT_WORTH_IT"
    )
    check = {
        "H1_expectation": "q=1.0 에서 허용 rho 존재",
        "H1_held": h1, "H1_max_rho": at_one["binding_max_rho"],
        "H2_expectation": f"q={H2_Q} 에서 허용 rho >= {H2_MIN_ALLOWED_RHO}",
        "H2_held": h2, "H2_max_rho": at_h2["binding_max_rho"],
        "H3_expectation": "제약이 가장 빡빡한 그룹 특정",
        "H3_held": h3, "H3_binding_group": tightest,
        "frozen": "이 수용선은 새 소스 측정 결과를 보고 변경하지 않는다",
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "purpose": "새 소스 rho 측정의 합격선을 **재기 전에** 확정한다",
        "no_collection": True, "no_training": True, "gate_touched": False,
        "requirements": {
            "primary_local_066": REQUIREMENT_PRIMARY,
            "secondary_leader_ficr": REQUIREMENT_SECONDARY,
            "source": "사이클 46 오차 축소 역산, 사이클 48 이 검증",
        },
        "current_sigma": {g: c["sigma_best_mix"] for g, c in CURRENT.items()},
        "acceptance_frontier": frontier,
        "predeclared_check": check,
        "blocking_rule": "AGENTS.md 5 행이 외부 데이터를 금지하고 있어 수집은 하지 않는다",
    }

    lines = [
        "# M271 P4 사이클 49 — 새 소스 합격선 (측정 전 확정)",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- **수집 없음.** {payload['blocking_rule']}",
        "",
        "## 1. 왜 먼저 정하는가",
        "",
        "재고 나서 \"이 정도면 되나\" 를 판단하면 그게 **사후 기준**이다. 이 프로젝트가",
        "반복해서 걸린 함정이 정확히 그것이므로, 수용 영역을 먼저 동결한다.",
        "",
        f"요구치: 주 **{REQUIREMENT_PRIMARY} m/s** (로컬 Total 0.66) / "
        f"보조 {REQUIREMENT_SECONDARY} (상위권 FICR).",
        f"출처: {payload['requirements']['source']}.",
        "",
        "현재 sigma: " + ", ".join(
            f"g{g} {v:.3f}" for g, v in payload["current_sigma"].items()
        ),
        "",
        "## 2. 수용 영역 — 각 q 에서 허용되는 **최대 rho**",
        "",
        "`q = sigma_new / sigma_cur`. 새 소스의 오차상관이 이 값 **이하**여야 합격이다.",
        "세 그룹 모두 만족해야 하므로 구속선은 가장 빡빡한 그룹이 정한다.",
        "",
        "| q | g1 | g2 | g3 | **구속선** | 구속 그룹 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in primary:
        cells = []
        for g in (1, 2, 3):
            v = r["max_rho_by_group"][g]
            cells.append(f"{v:.3f}" if v is not None else "—")
        binding = (
            f"**{r['binding_max_rho']:.3f}**" if r["binding_max_rho"] is not None else "**불가**"
        )
        bg = r["binding_group"] if r["binding_group"] is not None else "—"
        lines.append(f"| {r['q']:.2f} | {cells[0]} | {cells[1]} | {cells[2]} | {binding} | {bg} |")

    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** "
        f"(허용 rho {at_one['binding_max_rho']})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}** "
        f"(허용 rho {at_h2['binding_max_rho']})",
        f"- H3 `{check['H3_expectation']}` -> **{h3}** (그룹 {tightest})",
        "",
        f"판정: **{verdict}**",
        "",
        f"**{check['frozen']}.**",
        "",
        "## 4. 측정 절차 (규칙 해제 후)",
        "",
        "1. 새 소스의 **예보** 아카이브에서 짧은 표본(60~200 일)을 받는다.",
        "   재분석(ERA5 등)은 공식 규칙이 금지하므로 반드시 예보여야 한다.",
        "2. 그 소스의 풍속을 현재 최선 혼합과 같은 방식으로 정렬해 `sigma_new` 와",
        "   `rho(오차_new, 오차_cur)` 를 잰다.",
        "3. `q = sigma_new / sigma_cur` 를 구하고 위 표에서 그 q 의 구속선과 대조한다.",
        "4. `rho_measured <= 구속선` 이면 전량 수집이 정당화된다. 아니면 축이 **측정으로**",
        "   닫힌다.",
        "",
        "**후보 주의**: GEFS 는 GFS 와 같은 모델계열(FV3, 같은 자료동화)이라 rho 가 0.9 대일",
        "수 있다. 다른 계열(ECMWF IFS/AIFS, DWD ICON, 캐나다 GEM)이 rho 를 낮출 가능성이 높다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE49_RHO_ACCEPTANCE",
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

    print(f"[C49] 요구치 주 {REQUIREMENT_PRIMARY} / 보조 {REQUIREMENT_SECONDARY}")
    for r in primary:
        if abs(r["q"] - round(r["q"] * 10) / 10) < 1e-9 and int(r["q"] * 100) % 10 == 0:
            b = r["binding_max_rho"]
            print(f"[C49]   q={r['q']:.2f}  구속 rho <= "
                  f"{b if b is None else format(b, '.3f')}  (그룹 {r['binding_group']})")
    print(f"[C49] H1 {h1} (q=1.0 허용 rho {at_one['binding_max_rho']}) | "
          f"H2 {h2} (q={H2_Q} 허용 rho {at_h2['binding_max_rho']}) | "
          f"H3 {h3} (구속 그룹 {tightest})")
    print(f"[C49] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
