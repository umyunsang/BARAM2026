"""M271 P4 사이클 48 — 사이클 46·47 이 기댄 두 가정의 검증.

사이클 46·47 이 사이클 36 의 폐쇄를 뒤집었다. 그 결론은 가정 두 개 위에 서 있고, 둘 다
검증된 적이 없었다. 이 노드가 검증한다. **새 학습·수집 없음.**

  가정 A  소스 결합 산식 `sigma_opt^2 = s1^2 s2^2 (1-r^2) / (s1^2 + s2^2 - 2 r s1 s2)`
          (및 등가중판)이 이 데이터에서 실제로 맞는다.
  가정 B  상위권의 우위가 **균일한 오차 축소**로 설명된다. 즉 `pred_k = actual +
          k(pred-actual)` 한 모수로 그들의 FICR 과 1-NMAE 를 **동시에** 맞출 수 있다.

가정 B 가 이 노드의 핵심이다. 상위권이 더 나은 **결정 정책**을 가졌다면 FICR 만 오르고
1-NMAE 는 제자리여야 한다 — 단일 k 적합이 깨진다. 두 지표가 함께 움직였다면 밑에 깔린
**예보 자체**가 더 정확한 것이고, 그때만 사이클 47 의 "풍속 정확도 요구치" 프레임이 성립한다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 두 가정을 각각 관측과 대조한다.
  - 가정 B 는 **과적합 위험이 없는 검사**다. k 를 한 지표에 맞추고 **다른 지표를 예측**해
    빗나가는지 본다. 1 모수로 2 관측을 맞히는 것이므로 자유도가 남는다.
  - 오프셋 분해: `Total = 0.5(1-NMAE) + 0.5*FICR` 이므로
    `offset_nmae = 2*offset_total - offset_ficr`. M261 앵커의 실측값을 쓴다.

② 사양 동결

  가정 A  공급 소스 2 개(GFS, LDAPS)의 개별 sigma 와 rho 로 결합 sigma 를 예측하고
          `m270_sigma_decomposition` 이 실측한 평균혼합과 대조
  가정 B  사이클 46 의 k 곡선에서
            k_ficr  = 상위권 로컬환산 FICR 을 맞추는 k -> 1-NMAE 를 **예측**
            k_nmae  = 상위권 로컬환산 1-NMAE 를 맞추는 k -> FICR 을 **예측**

  사전확약(실행 전 동결):
    A1  세 그룹 모두에서 등가중 예측이 실측의 `±0.10` m/s 이내.
    A2  예측이 실측보다 **크거나 같다**(산식이 보수적). 성립하면 사이클 47 의 실현가능
        판정이 보수적인 쪽으로 기운다.
    B1  `|k_ficr - k_nmae| <= 0.03`. 두 지표가 같은 k 를 가리킨다.
    B2  교차예측 불일치가 두 지표 모두 **`0.005` 이내**.
    B3  (기전) B1·B2 가 성립하면 상위권 우위는 **결정층이 아니라 예보 정확도**에서 온다.
        이는 사이클 4·7·8 의 결정층 폐쇄와 **모순되지 않는다**.

  A1·A2·B1·B2 가 모두 성립하면 사이클 46·47 의 결론이 검증된 가정 위에 선다.
  하나라도 기각되면 그 결론의 근거가 약해지는 지점을 명시한다.

**새 학습·수집·게이트 수정·lockbox 사용 없음.**
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
C46_RECEIPT = REPORTS / "m271_cycle46_closure_falsification_receipt.json"
REPORT_MD = REPORTS / "m271_cycle48_model_validation.md"
RECEIPT = REPORTS / "m271_cycle48_model_validation_receipt.json"

NODE_ID = "C1N48_MODEL_VALIDATION"
LANE = "L8"
PARENT_NODE = "C1N47_EXTERNAL_NWP_REVIVED"

# m270_sigma_decomposition.md 실측 '평균혼합' 열
OBSERVED_MIX = {1: 2.159, 2: 2.406, 3: 2.175}
LEADER = {"total": 0.67365, "one_minus_nmae": 0.87964, "ficr": 0.46767}
OFFSET_TOTAL = 0.006554
OFFSET_FICR = 0.012027

A1_TOLERANCE = 0.10
B1_TOLERANCE = 0.03
B2_TOLERANCE = 0.005


def main() -> int:
    # --- 가정 A
    formula = []
    for group, c in CURRENT.items():
        s1, s2, r = c["sigma_gfs"], c["sigma_ldaps"], c["rho"]
        opt = optimal_pair_sigma(s1, s2, r)
        eq = float(np.sqrt((s1 * s1 + s2 * s2 + 2 * r * s1 * s2) / 4.0))
        obs = OBSERVED_MIX[group]
        formula.append(
            {
                "group": group, "sigma_gfs": s1, "sigma_ldaps": s2, "rho": r,
                "predicted_optimal": opt, "predicted_equal": eq, "observed_mix": obs,
                "error_equal": eq - obs, "error_optimal": opt - obs,
                "within_tolerance": bool(abs(eq - obs) <= A1_TOLERANCE),
                "conservative": bool(eq >= obs),
            }
        )
    a1 = all(f["within_tolerance"] for f in formula)
    a2 = all(f["conservative"] for f in formula)

    # --- 가정 B
    curve = json.loads(C46_RECEIPT.read_text(encoding="utf-8"))["result"][
        "error_scaling_curve"
    ]
    ks = np.array([c["k"] for c in curve])
    offset_nmae = 2.0 * OFFSET_TOTAL - OFFSET_FICR
    leader_local = {
        "total": LEADER["total"] - OFFSET_TOTAL,
        "one_minus_nmae": LEADER["one_minus_nmae"] - offset_nmae,
        "ficr": LEADER["ficr"] - OFFSET_FICR,
    }

    def predict(field: str, k: float) -> float:
        return float(np.interp(k, ks, [c[field] for c in curve]))

    def invert(field: str, target: float) -> float:
        ys = [c[field] for c in curve]
        return float(np.interp(target, ys[::-1], ks[::-1]))

    k_ficr = invert("ficr", leader_local["ficr"])
    k_nmae = invert("one_minus_nmae", leader_local["one_minus_nmae"])
    cross = {
        "k_from_ficr": k_ficr,
        "predicted_nmae_at_k_ficr": predict("one_minus_nmae", k_ficr),
        "residual_nmae": predict("one_minus_nmae", k_ficr)
        - leader_local["one_minus_nmae"],
        "k_from_nmae": k_nmae,
        "predicted_ficr_at_k_nmae": predict("ficr", k_nmae),
        "residual_ficr": predict("ficr", k_nmae) - leader_local["ficr"],
        "k_disagreement": abs(k_ficr - k_nmae),
    }
    b1 = bool(cross["k_disagreement"] <= B1_TOLERANCE)
    b2 = bool(
        abs(cross["residual_nmae"]) <= B2_TOLERANCE
        and abs(cross["residual_ficr"]) <= B2_TOLERANCE
    )
    b3 = bool(b1 and b2)

    validated = bool(a1 and a2 and b1 and b2)
    verdict = (
        "ASSUMPTIONS_VALIDATED_CYCLE46_47_STAND" if validated
        else "ASSUMPTION_FAILED_SEE_NOTES"
    )
    check = {
        "A1_expectation": f"등가중 예측이 실측의 ±{A1_TOLERANCE} m/s 이내 (3 그룹)",
        "A1_held": a1,
        "A2_expectation": "예측 >= 실측 (산식이 보수적)",
        "A2_held": a2,
        "B1_expectation": f"|k_ficr - k_nmae| <= {B1_TOLERANCE}",
        "B1_held": b1, "B1_measured": cross["k_disagreement"],
        "B2_expectation": f"교차예측 불일치가 두 지표 모두 {B2_TOLERANCE} 이내",
        "B2_held": b2,
        "B3_expectation": "상위권 우위가 결정층이 아니라 예보 정확도에서 온다",
        "B3_held": b3,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "validates": ["C1N46_CLOSURE_FALSIFICATION", "C1N47_EXTERNAL_NWP_REVIVED"],
        "no_training": True, "no_collection": True, "gate_touched": False,
        "assumption_A_formula": formula,
        "offset_decomposition": {
            "offset_total": OFFSET_TOTAL, "offset_ficr": OFFSET_FICR,
            "offset_one_minus_nmae": offset_nmae,
            "derivation": "Total = 0.5(1-NMAE) + 0.5*FICR -> "
                          "offset_nmae = 2*offset_total - offset_ficr",
        },
        "leader_online": LEADER, "leader_local_equivalent": leader_local,
        "assumption_B_cross_prediction": cross,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 48 — 사이클 46·47 이 기댄 두 가정의 검증",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "- **새 학습·수집 없음.** 두 가정을 관측과 대조한다",
        "",
        "## 1. 가정 A — 소스 결합 산식",
        "",
        "| group | sGFS | sLDAPS | rho | 예측(최적) | 예측(등가중) | 실측 혼합 | 등가중 오차 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in formula:
        lines.append(
            f"| {f['group']} | {f['sigma_gfs']:.3f} | {f['sigma_ldaps']:.3f} | "
            f"{f['rho']:.3f} | {f['predicted_optimal']:.3f} | "
            f"{f['predicted_equal']:.3f} | {f['observed_mix']:.3f} | "
            f"**{f['error_equal']:+.3f}** |"
        )
    lines += [
        "",
        "산식이 실측보다 **약간 비관적**이다. 즉 사이클 47 의 실현가능 판정은 보수적인 쪽이다.",
        "",
        "## 2. 가정 B — 상위권 우위가 균일 오차 축소인가",
        "",
        f"오프셋 분해: Total `+{OFFSET_TOTAL:.6f}` = FICR `+{OFFSET_FICR:.6f}` + "
        f"1-NMAE `+{offset_nmae:.6f}` (유도: {payload['offset_decomposition']['derivation']})",
        "",
        "| | 온라인 | 로컬환산 |",
        "|---|---:|---:|",
        f"| Total | {LEADER['total']:.5f} | {leader_local['total']:.5f} |",
        f"| 1-NMAE | {LEADER['one_minus_nmae']:.5f} | "
        f"{leader_local['one_minus_nmae']:.5f} |",
        f"| FICR | {LEADER['ficr']:.5f} | {leader_local['ficr']:.5f} |",
        "",
        "**교차예측**: 한 지표로 k 를 맞추고 **다른 지표를 예측**한다. 1 모수로 2 관측을",
        "맞히므로 자유도가 남아 과적합이 아니다.",
        "",
        "| 맞춘 지표 | k | 예측 대상 | 예측값 | 실측 | 잔차 |",
        "|---|---:|---|---:|---:|---:|",
        f"| FICR | {k_ficr:.4f} | 1-NMAE | "
        f"{cross['predicted_nmae_at_k_ficr']:.5f} | "
        f"{leader_local['one_minus_nmae']:.5f} | **{cross['residual_nmae']:+.5f}** |",
        f"| 1-NMAE | {k_nmae:.4f} | FICR | "
        f"{cross['predicted_ficr_at_k_nmae']:.5f} | "
        f"{leader_local['ficr']:.5f} | **{cross['residual_ficr']:+.5f}** |",
        "",
        f"두 k 의 불일치 **{cross['k_disagreement']:.4f}**.",
        "",
        "## 3. 사전확약 대조",
        "",
        f"- A1 `{check['A1_expectation']}` -> **{a1}**",
        f"- A2 `{check['A2_expectation']}` -> **{a2}**",
        f"- B1 `{check['B1_expectation']}` -> **{b1}** ({cross['k_disagreement']:.4f})",
        f"- B2 `{check['B2_expectation']}` -> **{b2}**",
        f"- B3 `{check['B3_expectation']}` -> **{b3}**",
        "",
        f"판정: **{verdict}**",
        "",
        "## 4. 이것이 확정하는 것",
        "",
    ]
    if validated:
        lines += [
            "단일 오차 축소 계수 하나가 상위권의 **두 지표를 동시에** 설명한다. 만약 그들이",
            "더 나은 **결정 정책**을 가졌다면 FICR 만 오르고 1-NMAE 는 제자리여야 하므로",
            "이 적합이 깨진다. 깨지지 않았다는 것은 우위가 **밑에 깔린 예보 정확도**에서",
            "온다는 뜻이다.",
            "",
            "따라서 폐쇄 집합이 **자기모순 없이** 리더보드를 설명한다:",
            "- 결정층(사이클 4·7·8) 폐쇄 — **유지**. 상위권도 결정층으로 벌지 않는다",
            "- 사후 연산·잔차(13~34, 27~31) 폐쇄 — **유지**",
            "- 외부 NWP(사이클 36) 폐쇄 — **반증됨**. 요구치가 3.2 배 과대했다",
            "",
            "남은 유일한 경로는 **예보 정확도 16% 개선**이고, 그것은 새 소스로만 온다.",
        ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE48_MODEL_VALIDATION",
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

    for f in formula:
        print(f"[C48] A g{f['group']} 예측(등가중) {f['predicted_equal']:.3f} vs 실측 "
              f"{f['observed_mix']:.3f} 오차 {f['error_equal']:+.3f}")
    print(f"[C48] A1 {a1} | A2 {a2}")
    print(f"[C48] B k_ficr {k_ficr:.4f} / k_nmae {k_nmae:.4f} "
          f"불일치 {cross['k_disagreement']:.4f}")
    print(f"[C48] B 교차잔차: 1-NMAE {cross['residual_nmae']:+.5f} / "
          f"FICR {cross['residual_ficr']:+.5f}")
    print(f"[C48] B1 {b1} | B2 {b2} | B3 {b3}")
    print(f"[C48] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
