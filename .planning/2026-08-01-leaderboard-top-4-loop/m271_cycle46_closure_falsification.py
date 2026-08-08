"""M271 P4 사이클 46 — 폐쇄 집합이 리더보드를 설명하는가 (반증 시도).

45 사이클이 축을 닫았다. 그 폐쇄들이 참이라면 **관측 가능한 함의**가 있다: 상위권이 내는
FICR 이 우리 폐쇄 집합 하에서 도달 가능한 범위 안에 있어야 한다. 벗어나면 폐쇄 중 하나가
틀린 것이다.

    1위 연식2   온라인 Total 0.67365  1-NMAE 0.87964  **FICR 0.46767**
    우리 M261   온라인 Total 0.63653  1-NMAE 0.85789  FICR 0.41517

내 폐쇄들이 말하는 것
  - 사이클 36: 풍속 정확도는 `sigma*sqrt(rho)` 하한에 막힌다. 현실 범위 최선 1.676 m/s,
    필요치 0.571 m/s 의 2.9 배. **외부 소스로도 개선 불가.**
  - 사이클 4·8: 고정 분포에서 사후 결정 사상의 천장은 격차의 8.8%, 일반화 조건화는 16.5%.
  - 사이클 27·28·30·31: 잔차는 공급 NWP 로 설명되지 않는다(2x2 네 칸 R^2 음수).

셋이 다 참이면 상위권의 FICR 우위는 **어디서도 올 수 없다.** 그런데 관측된다. 모순이다.

이 노드는 그 모순을 정량화한다. 방법은 **오차 축소 역산**이다.

    pred_k = actual + k * (pred - actual)

`k` 는 오차 배율이다(k=1 이 현재, k<1 이 더 정확한 예보). k 를 훑어 상위권 FICR 에
도달하는 `k*` 를 찾고, 그것을 풍속 정확도로 환산해 사이클 36 의 하한과 대조한다.

  - 필요 sigma = `k* x 2.159` (그룹1 현재 최선 혼합. 출력오차가 풍속오차에 비례한다는
    국소 선형 근사)
  - 사이클 36 의 현실범위 달성가능 최선 = **1.676 m/s**

  필요 sigma > 1.676  ->  상위권 성적이 **더 나은 풍속**으로 설명된다. 사이클 36 의
                          "물리적으로 불가" 프레임이 흔들린다.
  필요 sigma < 1.676  ->  풍속으로는 설명 불가. 상위권은 **다른 층**에서 얻고 있으며
                          결정층·표현 폐쇄가 흔들린다.

어느 쪽이든 폐쇄 집합에 대한 정보다. **둘 다 아니면** 폐쇄 집합이 일관된 것이다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 반증 설계다: 폐쇄 집합의 관측 가능한 함의를 뽑아 외부 사실과 대조한다.
  - **로컬/온라인 표면 차이를 보정한다.** 상위권 수치는 온라인(2025), 우리 측정은
    로컬(2023)이다. M261 앵커가 잰 FICR 오프셋 `+0.012027` 로 상위권 FICR 을 로컬 척도로
    환산해 비교한다. 오프셋은 방법군 의존이므로 **보정 전후를 모두 보고**한다.

② 사양 동결

  기준   배포 `M269@T0.5_G1.5` 예측 (고정정책, 로컬 2023 Q2-Q4)
  격자   `k in [0.05, 1.00]` 간격 0.05
  목표   (a) 상위권 FICR 0.46767 (온라인) 및 로컬 환산 0.45564
         (b) 로컬 Total 0.66

  사전확약(실행 전 동결):
    H1  `k in (0, 1]` 안에 상위권 로컬환산 FICR 에 도달하는 해가 있다.
    H2  그 `k*` 로 환산한 필요 sigma 가 사이클 36 의 달성가능 최선 **1.676 m/s 보다 크다**.
        성립하면 상위권은 더 나은 풍속으로 설명 가능하고 사이클 36 프레임이 흔들린다.
    H3  로컬 Total 0.66 에 필요한 `k` 도 격자 안에 있다.
    H4  (일관성) H2 가 기각되면 상위권 우위는 풍속이 아닌 층에서 오므로, 결정층 폐쇄
        (오라클 8.8%)와 **모순**된다. 그 모순을 명시 기록한다.

**새 학습·게이트 수정·lockbox 사용 없음.** 이 노드는 반증 시도이지 후보 생성이 아니다.
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

from m270_monthly_validation import load_predictions
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle46_closure_falsification.md"
RECEIPT = REPORTS / "m271_cycle46_closure_falsification_receipt.json"

NODE_ID = "C1N46_CLOSURE_FALSIFICATION"
LANE = "L8"
PARENT_NODE = "C1N45_SUBMISSION_DECISION"
DEPLOYED = "T0.5_G1.5"

LEADER = {"team": "연식2", "total": 0.67365, "one_minus_nmae": 0.87964, "ficr": 0.46767}
OURS_ONLINE = {"model": "M261", "total": 0.6365274327, "ficr": 0.41517}
FICR_OFFSET = 0.012027  # m270_second_anchor: 로컬->온라인 FICR 오프셋
LOCAL_TARGET_TOTAL = 0.66

SIGMA_CURRENT_G1 = 2.159  # 사이클 36 인용 (그룹1 최선 혼합)
SIGMA_ACHIEVABLE_FLOOR = 1.676  # 사이클 36 현실범위 최선 (최적가중, q 0.8~1.0, rho 0.6~0.8)
SIGMA_REQUIRED_POINTWISE = 0.571  # 급경사 전형 기울기 기준

K_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 4)


def main() -> int:
    base = load_predictions(DEPLOYED)
    actual = base["actual_kwh"].to_numpy(dtype="float64")
    pred = base["prediction_kwh"].to_numpy(dtype="float64")

    curve = []
    for k in K_GRID:
        frame = base.loc[
            :, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "month"]
        ].copy()
        frame["prediction_kwh"] = actual + k * (pred - actual)
        score = official(frame)
        curve.append({"k": float(k), **score})

    leader_local_ficr = LEADER["ficr"] - FICR_OFFSET

    def solve(field: str, target: float) -> float | None:
        """단조 감소(k 증가 -> 성능 하락) 가정 하에 선형보간으로 해를 찾는다."""
        xs = [c["k"] for c in curve]
        ys = [c[field] for c in curve]
        for i in range(len(xs) - 1):
            lo, hi = ys[i], ys[i + 1]
            if (lo - target) * (hi - target) <= 0 and lo != hi:
                w = (target - lo) / (hi - lo)
                return float(xs[i] + w * (xs[i + 1] - xs[i]))
        return None

    k_leader_online = solve("ficr", LEADER["ficr"])
    k_leader_local = solve("ficr", leader_local_ficr)
    k_total_066 = solve("total", LOCAL_TARGET_TOTAL)

    def sigma_for(k: float | None) -> float | None:
        return None if k is None else float(k * SIGMA_CURRENT_G1)

    sigma_leader_local = sigma_for(k_leader_local)
    sigma_leader_online = sigma_for(k_leader_online)
    sigma_total_066 = sigma_for(k_total_066)

    h1 = bool(k_leader_local is not None)
    h2 = bool(sigma_leader_local is not None and sigma_leader_local > SIGMA_ACHIEVABLE_FLOOR)
    h3 = bool(k_total_066 is not None)
    h4 = bool(h1 and not h2)  # 풍속으로 설명 불가 -> 결정층 폐쇄와 모순

    if not h1:
        verdict = "LEADER_FICR_UNREACHABLE_EVEN_WITH_PERFECT_FORECAST"
    elif h2:
        verdict = "LEADER_EXPLAINED_BY_BETTER_WIND_CYCLE36_FRAME_WEAKENED"
    else:
        verdict = "LEADER_NOT_EXPLAINED_BY_WIND_DECISION_CLOSURE_CONTRADICTED"

    check = {
        "H1_expectation": "격자 안에 상위권 로컬환산 FICR 해가 있다",
        "H1_held": h1, "H1_k": k_leader_local,
        "H2_expectation": f"필요 sigma > 달성가능 최선 {SIGMA_ACHIEVABLE_FLOOR}",
        "H2_held": h2, "H2_sigma": sigma_leader_local,
        "H3_expectation": f"로컬 Total {LOCAL_TARGET_TOTAL} 해가 격자 안에 있다",
        "H3_held": h3, "H3_k": k_total_066,
        "H4_expectation": "H2 기각시 결정층 폐쇄와 모순됨을 명시",
        "H4_held": h4,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "design": "반증 시도. 폐쇄 집합의 관측 가능한 함의를 리더보드와 대조한다",
        "no_training": True, "gate_touched": False, "lockbox_used": False,
        "leader": LEADER, "ours_online": OURS_ONLINE,
        "ficr_offset_used": FICR_OFFSET,
        "leader_local_equivalent_ficr": leader_local_ficr,
        "baseline": {"policy": DEPLOYED, **official(base)},
        "error_scaling_curve": curve,
        "solutions": {
            "k_for_leader_ficr_online": k_leader_online,
            "k_for_leader_ficr_local_equiv": k_leader_local,
            "k_for_local_total_066": k_total_066,
        },
        "sigma_translation": {
            "sigma_current_g1": SIGMA_CURRENT_G1,
            "sigma_achievable_floor": SIGMA_ACHIEVABLE_FLOOR,
            "sigma_required_pointwise": SIGMA_REQUIRED_POINTWISE,
            "sigma_for_leader_local": sigma_leader_local,
            "sigma_for_leader_online": sigma_leader_online,
            "sigma_for_total_066": sigma_total_066,
            "approximation": "출력오차가 풍속오차에 비례한다는 국소 선형 근사",
        },
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 46 — 폐쇄 집합이 리더보드를 설명하는가",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "- **반증 시도**다. 후보를 만들지 않는다. 새 학습·게이트 수정 없음",
        "",
        "## 1. 모순의 형태",
        "",
        "| | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
        f"| 1위 `{LEADER['team']}` (온라인) | {LEADER['total']:.5f} | "
        f"{LEADER['one_minus_nmae']:.5f} | **{LEADER['ficr']:.5f}** |",
        f"| 우리 `{OURS_ONLINE['model']}` (온라인) | {OURS_ONLINE['total']:.5f} | — | "
        f"{OURS_ONLINE['ficr']:.5f} |",
        "",
        "내 폐쇄들이 참이면 상위권의 FICR 우위는 어디서도 올 수 없다. 그런데 관측된다.",
        "",
        f"로컬 척도 환산: FICR 오프셋 `+{FICR_OFFSET:.6f}` (M261 앵커) 를 빼면 "
        f"상위권 로컬환산 FICR = **{leader_local_ficr:.5f}**.",
        "",
        "## 2. 오차 축소 역산",
        "",
        "`pred_k = actual + k * (pred - actual)`. k=1 이 현재, k<1 이 더 정확한 예보다.",
        "",
        "| k | Total | 1-NMAE | FICR |",
        "|---:|---:|---:|---:|",
    ]
    for c in curve[::2]:
        lines.append(
            f"| {c['k']:.2f} | {c['total']:.6f} | {c['one_minus_nmae']:.6f} | "
            f"{c['ficr']:.6f} |"
        )

    s = payload["sigma_translation"]
    lines += [
        "",
        "## 3. 풍속 정확도로 환산",
        "",
        f"근사: {s['approximation']}. 필요 sigma = `k x {SIGMA_CURRENT_G1}`.",
        "",
        "| 목표 | 필요 k | 필요 sigma (m/s) | 달성가능 최선 대비 |",
        "|---|---:|---:|---|",
    ]
    for label, kk, ss in (
        ("상위권 FICR (로컬환산)", k_leader_local, sigma_leader_local),
        ("상위권 FICR (온라인 원값)", k_leader_online, sigma_leader_online),
        (f"로컬 Total {LOCAL_TARGET_TOTAL}", k_total_066, sigma_total_066),
    ):
        if kk is None:
            lines.append(f"| {label} | 격자 밖 | — | — |")
        else:
            rel = "**초과 (달성 가능)**" if ss > SIGMA_ACHIEVABLE_FLOOR else "미만 (달성 불가)"
            lines.append(f"| {label} | {kk:.4f} | **{ss:.3f}** | {rel} |")
    lines += [
        "",
        f"사이클 36 의 현실범위 달성가능 최선 = **{SIGMA_ACHIEVABLE_FLOOR} m/s**, "
        f"점별 필요치 = {SIGMA_REQUIRED_POINTWISE} m/s.",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}**",
        "",
        f"판정: **{verdict}**",
        "",
        "## 5. 읽는 법",
        "",
    ]
    if h2:
        lines += [
            "필요 sigma 가 달성가능 최선을 **넘는다**. 즉 상위권 성적은 **우리보다 나은",
            "풍속 예보**로 설명될 수 있고, 사이클 36 이 쓴 '물리적으로 불가' 프레임은",
            "**점별 밴드 적중 요구치(0.571 m/s)를 기준으로 한 것이지 FICR 자체의 요구치가",
            "아니었다.** 발전량가중 FICR 은 평탄 구간에서도 점수를 주므로 요구치가 더 낮다.",
            "",
            "-> 사이클 36 의 폐쇄를 **재검해야 한다.**",
        ]
    elif h1:
        lines += [
            "필요 sigma 가 달성가능 최선보다 **낮다**. 즉 상위권 우위는 풍속 정확도로",
            "설명되지 않는다. 그렇다면 결정층·표현층에서 오는 것인데, 사이클 4·8 이 그",
            "천장을 격차의 8.8%(일반화 16.5%)로 쟀다. **둘 다 참일 수 없다.**",
            "",
            "-> 결정층 폐쇄를 **재검해야 한다.**",
        ]
    else:
        lines += [
            "완벽한 예보(k->0)로도 상위권 FICR 에 도달하지 못한다. 이는 상위권의 온라인",
            "수치가 우리 로컬 표면과 근본적으로 다른 모집단임을 뜻하며, 오프셋 보정만으로는",
            "비교할 수 없다는 결론이다.",
        ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE46_CLOSURE_FALSIFICATION",
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

    print(f"[C46] 기준 {DEPLOYED} FICR {payload['baseline']['ficr']:.6f} "
          f"Total {payload['baseline']['total']:.6f}")
    print(f"[C46] 상위권 FICR 온라인 {LEADER['ficr']:.5f} -> "
          f"로컬환산 {leader_local_ficr:.5f}")
    for label, kk, ss in (
        ("상위권(로컬환산)", k_leader_local, sigma_leader_local),
        ("상위권(온라인)", k_leader_online, sigma_leader_online),
        ("로컬 Total 0.66", k_total_066, sigma_total_066),
    ):
        if kk is None:
            print(f"[C46]   {label:<18} 격자 밖")
        else:
            print(f"[C46]   {label:<18} k={kk:.4f}  필요 sigma={ss:.3f} "
                  f"(달성가능 {SIGMA_ACHIEVABLE_FLOOR})")
    print(f"[C46] H1 {h1} | H2 {h2} | H3 {h3} | H4 {h4}")
    print(f"[C46] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
