"""M271 P4 사이클 50 — 결합 모형의 음수가중 결함 교정과 수용 영역 재도출.

사이클 49 의 수용 영역이 **비단조**로 나왔다: q=0.60 에서 허용 rho 0.990, q=1.00 에서
0.245, q=1.20 에서 다시 0.990. 진단 결과 원인이 특정됐다.

    사이클 36 의 `optimal_pair_sigma` 는 **제약 없는** 최적가중이라 `rho > q` 에서
    소스 1 의 가중이 **음수**가 된다. rho=0.99, q=0.8 에서 w1 = -2.71, 결합 sigma = 1.03.
    이는 두 소스를 **차분해 오차를 상쇄**한다는 외삽이며 실제로 배포할 수 없다.

**사이클 25 에서 인용한 Breiman (1996) 의 논거가 그대로 적용된다**: 비음 가중이라야 결합이
보간적(min <= 결합 <= max)이고 실제로 쓸 수 있다. 사이클 36 에서 "새 소스에 가장 유리한
가정" 이라며 최적가중을 골랐는데, 유리한 정도가 아니라 **물리적으로 불가능한 영역까지**
열어준 것이다.

영향 범위
  - 사이클 36 (요구치 0.571 에서 만족 0 개): 음수가중은 상황을 **좋게** 만드는 방향이므로
    그 결론은 **보수적이라 유지**된다.
  - 사이클 47 (현실범위 q 0.8~1.0, rho 0.6~0.8): 음수가중 경계가 `rho = q` 이므로 그 창은
    경계 안쪽이다. **재확인이 필요하되 유지될 가능성이 높다.**
  - 사이클 49 (rho 를 0.99 까지 스캔): **오염됐다. 이 노드가 대체한다.**

① 방법 리서치 (실행 전)
  - Breiman (1996) *Stacked Regressions* — 비음·합1 제약이 결합을 보간적으로 만든다.
    사이클 25 가 이미 인용했고 같은 원리다.
  - 제약 결합:
        sigma^2(w) = w^2 s1^2 + (1-w)^2 s2^2 + 2 w (1-w) rho s1 s2,  w in [0, 1]
    비제약 최적해를 [0,1] 로 clip 한 것이 최소다(2 차식이므로).

② 사양 동결

  요구치  주 `1.871` (로컬 Total 0.66) / 보조 `1.817` (상위권 FICR) — 사이클 46 실측
  격자    `q in [0.6, 1.3]` 간격 0.05, `rho in [0, 0.99]` 간격 0.005
  결합    **비음 가중 제약** `w in [0, 1]`
  판정    세 그룹 모두 만족해야 합격

  사전확약(실행 전 동결):
    H1  비제약식이 `rho > q` 에서 음수 가중을 낸다 (진단 확인).
    H2  제약식의 허용 rho 가 q 에 대해 **단조**다 (비단조 artifact 제거 확인).
    H3  사이클 47 의 현실범위 결론(로컬 0.66 기준 세 그룹 실현가능)이 **제약식에서도
        유지**된다.
    H4  `q = 0.85` 에서 허용 rho 가 `0.6` 이상이다 (사이클 49 의 H2 를 제약식으로 재판정).

  H3 이 기각되면 사이클 47 의 축 부활이 음수가중 artifact 였다는 뜻이고, 외부 NWP 축을
  다시 닫아야 한다. **이 수용선은 새 소스 측정 결과를 보고 변경하지 않는다.**

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
REPORT_MD = REPORTS / "m271_cycle50_nonnegative_weights.md"
RECEIPT = REPORTS / "m271_cycle50_nonnegative_weights_receipt.json"

NODE_ID = "C1N50_NONNEGATIVE_WEIGHTS"
LANE = "L2"
PARENT_NODE = "C1N49_RHO_ACCEPTANCE"
SUPERSEDES = "C1N49_RHO_ACCEPTANCE"

REQUIREMENT_PRIMARY = 1.871
REQUIREMENT_SECONDARY = 1.817
Q_GRID = np.round(np.arange(0.60, 1.3001, 0.05), 4)
RHO_SCAN = np.round(np.arange(0.0, 0.9901, 0.005), 4)
REALISTIC_Q = (0.80, 1.00)
REALISTIC_RHO = (0.60, 0.80)
H4_Q = 0.85
H4_MIN_RHO = 0.60


def constrained_pair_sigma(s1: float, s2: float, rho: float) -> tuple[float, float]:
    """비음 가중 제약 `w in [0,1]` 하의 최소 결합 sigma 와 그 가중."""
    denom = s1 * s1 + s2 * s2 - 2.0 * rho * s1 * s2
    w = 0.5 if denom <= 0 else (s2 * s2 - rho * s1 * s2) / denom
    w = float(np.clip(w, 0.0, 1.0))
    var = w * w * s1 * s1 + (1 - w) ** 2 * s2 * s2 + 2 * w * (1 - w) * rho * s1 * s2
    return float(np.sqrt(max(var, 0.0))), w


def max_rho_constrained(sigma_cur: float, q: float, required: float) -> float | None:
    allowed = [
        r for r in RHO_SCAN
        if constrained_pair_sigma(sigma_cur, q * sigma_cur, float(r))[0] <= required
    ]
    return float(max(allowed)) if allowed else None


def main() -> int:
    # --- H1 진단: 비제약식의 음수가중 경계
    diagnosis = []
    s_ref = CURRENT[1]["sigma_best_mix"]
    for q in (0.7, 0.8, 0.9, 1.0):
        s2 = q * s_ref
        first_negative = None
        for r in RHO_SCAN:
            denom = s_ref * s_ref + s2 * s2 - 2 * r * s_ref * s2
            w = (s2 * s2 - r * s_ref * s2) / denom if denom > 0 else 0.5
            if w < 0:
                first_negative = float(r)
                break
        diagnosis.append(
            {
                "q": q, "first_negative_weight_rho": first_negative,
                "matches_rho_equals_q": bool(
                    first_negative is not None and abs(first_negative - q) <= 0.01
                ),
                "unconstrained_sigma_at_rho099": optimal_pair_sigma(s_ref, s2, 0.99),
                "constrained_sigma_at_rho099": constrained_pair_sigma(s_ref, s2, 0.99)[0],
            }
        )
    h1 = all(d["matches_rho_equals_q"] for d in diagnosis)

    # --- 제약식 수용 영역
    frontier = []
    for q in Q_GRID:
        per_group = {
            g: max_rho_constrained(c["sigma_best_mix"], float(q), REQUIREMENT_PRIMARY)
            for g, c in CURRENT.items()
        }
        values = [v for v in per_group.values() if v is not None]
        feasible = len(values) == len(CURRENT)
        binding_group = (
            min(per_group, key=lambda g: per_group[g] if per_group[g] is not None else 9)
            if feasible else None
        )
        frontier.append(
            {
                "q": float(q), "max_rho_by_group": per_group,
                "all_groups_feasible": bool(feasible),
                "binding_max_rho": float(min(values)) if feasible else None,
                "binding_group": int(binding_group) if binding_group is not None else None,
            }
        )

    feasible_rows = [r for r in frontier if r["all_groups_feasible"]]
    rhos = [r["binding_max_rho"] for r in feasible_rows]
    h2 = bool(len(rhos) >= 2 and (all(np.diff(rhos) >= -1e-9) or all(np.diff(rhos) <= 1e-9)))

    # --- H3 사이클 47 재확인 (현실범위, 제약식)
    revalidation = {}
    for g, c in CURRENT.items():
        s = c["sigma_best_mix"]
        best, best_at = float("inf"), None
        for q in np.round(np.arange(REALISTIC_Q[0], REALISTIC_Q[1] + 1e-9, 0.05), 4):
            for r in np.round(np.arange(REALISTIC_RHO[0], REALISTIC_RHO[1] + 1e-9, 0.05), 4):
                v, w = constrained_pair_sigma(s, float(q) * s, float(r))
                if v < best:
                    best, best_at = v, {"q": float(q), "rho": float(r), "w1": w}
        revalidation[g] = {
            "best_sigma_constrained": best, "at": best_at,
            "requirement": REQUIREMENT_PRIMARY,
            "feasible": bool(best <= REQUIREMENT_PRIMARY),
            "margin": REQUIREMENT_PRIMARY - best,
        }
    h3 = all(v["feasible"] for v in revalidation.values())

    at_h4 = next(r for r in frontier if abs(r["q"] - H4_Q) < 1e-9)
    h4 = bool(
        at_h4["all_groups_feasible"]
        and at_h4["binding_max_rho"] is not None
        and at_h4["binding_max_rho"] >= H4_MIN_RHO
    )

    axis_survives = h3
    verdict = (
        "AXIS_SURVIVES_UNDER_NONNEGATIVE_WEIGHTS" if axis_survives
        else "AXIS_REVIVAL_WAS_NEGATIVE_WEIGHT_ARTIFACT_RECLOSE"
    )
    check = {
        "H1_expectation": "비제약식이 rho > q 에서 음수가중",
        "H1_held": h1,
        "H2_expectation": "제약식 허용 rho 가 q 에 단조",
        "H2_held": h2,
        "H3_expectation": "사이클 47 결론이 제약식에서도 유지",
        "H3_held": h3,
        "H4_expectation": f"q={H4_Q} 에서 허용 rho >= {H4_MIN_RHO}",
        "H4_held": h4, "H4_measured": at_h4["binding_max_rho"],
        "frozen": "이 수용선은 새 소스 측정 결과를 보고 변경하지 않는다",
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE, "supersedes": SUPERSEDES,
        "defect": "사이클 36 의 optimal_pair_sigma 는 제약 없는 최적가중이라 rho > q 에서 "
                  "음수 가중을 낸다. 두 소스를 차분해 오차를 상쇄한다는 외삽이며 배포 불가",
        "impact": {
            "cycle36": "결론(요구치 0.571 에서 만족 0 개)은 음수가중이 상황을 좋게 만드는 "
                       "방향이므로 보수적이라 유지",
            "cycle47": "현실범위가 음수가중 경계(rho=q) 안쪽이라 재확인 대상 — 이 노드 H3",
            "cycle49": "rho 를 0.99 까지 스캔해 오염. 이 노드가 대체",
        },
        "no_collection": True, "no_training": True, "gate_touched": False,
        "requirements": {"primary": REQUIREMENT_PRIMARY, "secondary": REQUIREMENT_SECONDARY},
        "negative_weight_diagnosis": diagnosis,
        "acceptance_frontier_constrained": frontier,
        "cycle47_revalidation": revalidation,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 50 — 음수가중 결함 교정과 수용 영역 재도출",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / 대체 `{SUPERSEDES}`",
        "- **수집·학습 없음**",
        "",
        "## 1. 결함",
        "",
        payload["defect"] + ".",
        "",
        "| q | 음수가중 시작 rho | rho=q 와 일치 | 비제약 sigma(rho=0.99) | 제약 sigma |",
        "|---:|---:|:---:|---:|---:|",
    ]
    for d in diagnosis:
        fn = d["first_negative_weight_rho"]
        lines.append(
            f"| {d['q']:.2f} | {fn if fn is None else format(fn, '.3f')} | "
            f"{'O' if d['matches_rho_equals_q'] else 'X'} | "
            f"{d['unconstrained_sigma_at_rho099']:.4f} | "
            f"{d['constrained_sigma_at_rho099']:.4f} |"
        )
    lines += [
        "",
        "영향 범위:",
        f"- 사이클 36 — {payload['impact']['cycle36']}",
        f"- 사이클 47 — {payload['impact']['cycle47']}",
        f"- 사이클 49 — {payload['impact']['cycle49']}",
        "",
        "## 2. 사이클 47 재확인 (H3)",
        "",
        f"현실범위 `q {REALISTIC_Q}` x `rho {REALISTIC_RHO}`, **비음 가중 제약**.",
        "",
        "| group | 제약식 최선 sigma | 최적점 (q, rho, w1) | 요구치 대비 | 실현가능 |",
        "|---:|---:|---|---:|:---:|",
    ]
    for g, v in revalidation.items():
        a = v["at"]
        lines.append(
            f"| {g} | **{v['best_sigma_constrained']:.3f}** | "
            f"q={a['q']:.2f}, rho={a['rho']:.2f}, w1={a['w1']:.3f} | "
            f"{v['margin']:+.3f} | {'**O**' if v['feasible'] else 'X'} |"
        )

    lines += [
        "",
        "## 3. 수용 영역 (비음 가중 제약)",
        "",
        "`q = sigma_new / sigma_cur`. 새 소스의 오차상관이 구속선 **이하**여야 합격.",
        "",
        "| q | g1 | g2 | g3 | **구속선** | 구속 그룹 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in frontier:
        cells = [
            (f"{r['max_rho_by_group'][g]:.3f}" if r["max_rho_by_group"][g] is not None else "—")
            for g in (1, 2, 3)
        ]
        b = (
            f"**{r['binding_max_rho']:.3f}**" if r["binding_max_rho"] is not None
            else "**불가**"
        )
        bg = r["binding_group"] if r["binding_group"] is not None else "—"
        lines.append(
            f"| {r['q']:.2f} | {cells[0]} | {cells[1]} | {cells[2]} | {b} | {bg} |"
        )

    lines += [
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}** ({at_h4['binding_max_rho']})",
        "",
        f"판정: **{verdict}**",
        "",
        f"**{check['frozen']}.**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE50_NONNEGATIVE_WEIGHTS",
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

    for d in diagnosis:
        print(f"[C50] q={d['q']:.2f} 음수가중 시작 rho={d['first_negative_weight_rho']} "
              f"| 비제약 sigma(0.99)={d['unconstrained_sigma_at_rho099']:.4f} "
              f"-> 제약 {d['constrained_sigma_at_rho099']:.4f}")
    for g, v in revalidation.items():
        print(f"[C50] 재확인 g{g} 제약식 최선 {v['best_sigma_constrained']:.3f} "
              f"(요구 {REQUIREMENT_PRIMARY}, 여유 {v['margin']:+.3f}) -> {v['feasible']}")
    print(f"[C50] H1 {h1} | H2 {h2} | H3 {h3} | H4 {h4} "
          f"(q={H4_Q} 허용 rho {at_h4['binding_max_rho']})")
    print(f"[C50] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
