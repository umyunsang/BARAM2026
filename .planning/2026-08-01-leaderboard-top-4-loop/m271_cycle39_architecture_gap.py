"""M271 P4 사이클 39 — 누출 철회와 아키텍처 격차 측정. 밴드 손실은 왜 잘못 놓였는가.

사이클 38 의 V1 가드가 발화했다(CONTROL 이 배포보다 -0.0869). 원인을 가르려고 전체 피처
풀로 L1 회귀기를 돌렸더니 **0.656158** 이 나왔다 — 배포 대비 +0.0276, 로컬 목표 0.66 에서
0.0038. 보고하기 전에 검사했고, **누출이었다.**

  `scada_ws` = 관측 나셀풍속. `actual_kwh` 와 상관 **0.9266**, 2023 년 결측률 0.01%.
  평가기간(2025)에는 SCADA 가 없다(A5 확정). 배포 불가능한 수치이므로 **철회한다.**

`_surface()` 는 `_scada_wind()` 로 이 컬럼 하나를 별도 병합한다. 그것만 빼고 다시 재면:

  전체 1,347 피처 L1 회귀기  0.542403
  M115 의 87 피처 L1 회귀기  0.541677

**피처 커버리지는 원인이 아니다.** 87 개든 1,347 개든 같다. 격차는 **구조**다.

  직접 점회귀        ~0.542
  분류기 + 결정정책  0.628605 (배포) ~ 0.630310 (M115@T0.6_G0.2)
  차이               **약 0.087**

즉 사이클 37·38 의 밴드 손실 실험은 **0.087 뒤처진 아키텍처 위에** 손실을 얹은 것이다.
손실이 아무리 좋아도 따라잡을 수 없다. 실험이 잘못 놓였지 손실 축이 닫힌 게 아니다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 이 노드는 **이미 실행된 진단을 기록**하고 누출을 명시 철회한다.
  - 추론 가용성 판정 기준을 데이터로 고정한다: `test_features.parquet` 에 존재하거나
    grid/geometric 유래여야 하고, **SCADA 병합 컬럼은 학습기간 전용이므로 제외**한다.

② 사양 동결 — 이 노드는 기록과 판정만 한다

  사전확약:
    H1  누출 제거 후 전체피처 회귀기가 87 피처판과 **0.005 이내**다.
        (성립하면 피처 커버리지가 원인이 아니라는 뜻)
    H2  누출 제거 후 회귀기가 배포 대비 **-0.05 보다 나쁘다**.
        (성립하면 격차가 구조에서 온다는 뜻)
    H3  누출판(0.656158)이 비누출판(0.542403)보다 **0.10 이상** 높다.
        (성립하면 그 수치가 전적으로 누출임이 확정)

  셋 다 성립하면 판정은 `BAND_LOSS_MISFRAMED_ARCHITECTURE_GAP_DOMINATES` 이며,
  손실 축은 **여전히 미판정**이다. 올바른 실험은 분류기 틀 **안에서** 밴드 인지 학습이다.

**게이트 무관. 이 노드는 새 학습을 하지 않는다.** 2024 행·lockbox 미사용.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle39_architecture_gap.md"
RECEIPT = REPORTS / "m271_cycle39_architecture_gap_receipt.json"

NODE_ID = "C1N39_ARCHITECTURE_GAP"
LANE = "L3"
PARENT_NODE = "C1N38_BAND_LOSS_FIXED"
RETRACTS = "세션 중 관측된 전체피처 회귀기 0.656158"

# 실행된 진단의 결과. 이 노드는 재실행하지 않고 기록한다.
MEASURED = {
    "leaked_full_features": {
        "total": 0.656158, "features": 1348, "includes_scada_ws": True,
        "status": "RETRACTED",
        "reason": "scada_ws 는 관측 나셀풍속. actual_kwh 와 상관 0.9266, 평가기간 부재",
    },
    "clean_full_features": {
        "total": 0.542403, "one_minus_nmae": 0.826922, "ficr": 0.257885,
        "features": 1347, "includes_scada_ws": False,
    },
    "clean_87_features": {"total": 0.541677, "features": 87, "includes_scada_ws": False},
    "deployed_classifier_policy": {"total": 0.628605, "name": "M269_PROBE_TOP100@T0.5_G1.5"},
    "best_fixed_policy": {"total": 0.630310, "name": "M115_XGBOOST@T0.6_G0.2"},
}
H1_TOL = 0.005
H2_MIN_DEFICIT = 0.05
H3_MIN_LEAK = 0.10


def main() -> int:
    clean_full = MEASURED["clean_full_features"]["total"]
    clean_87 = MEASURED["clean_87_features"]["total"]
    deployed = MEASURED["deployed_classifier_policy"]["total"]
    leaked = MEASURED["leaked_full_features"]["total"]

    coverage_delta = abs(clean_full - clean_87)
    structural_gap = deployed - clean_full
    leak_size = leaked - clean_full

    h1 = bool(coverage_delta <= H1_TOL)
    h2 = bool(structural_gap >= H2_MIN_DEFICIT)
    h3 = bool(leak_size >= H3_MIN_LEAK)
    verdict = (
        "BAND_LOSS_MISFRAMED_ARCHITECTURE_GAP_DOMINATES" if (h1 and h2 and h3)
        else "DIAGNOSIS_INCONCLUSIVE"
    )

    check = {
        "H1_expectation": f"전체피처 vs 87피처 차이 <= {H1_TOL}",
        "H1_held": h1, "H1_measured": coverage_delta,
        "H2_expectation": f"회귀기가 배포 대비 {H2_MIN_DEFICIT} 이상 열세",
        "H2_held": h2, "H2_measured": structural_gap,
        "H3_expectation": f"누출판 - 비누출판 >= {H3_MIN_LEAK}",
        "H3_held": h3, "H3_measured": leak_size,
        "loss_axis_status": "UNJUDGED — 올바른 실험은 분류기 틀 안에서의 밴드 인지 학습",
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "retracts": RETRACTS,
        "retraction_reason": MEASURED["leaked_full_features"]["reason"],
        "admissibility_rule": "test_features.parquet 에 존재하거나 grid/geometric 유래여야 "
                              "하고, SCADA 병합 컬럼(scada_ws)은 학습기간 전용이라 제외",
        "measured": MEASURED,
        "coverage_delta": coverage_delta,
        "structural_gap": structural_gap,
        "leak_size": leak_size,
        "gate_touched": False, "new_training": False, "lockbox_used": False,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 39 — 누출 철회와 아키텍처 격차",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "- 이 노드는 새 학습을 하지 않는다. 이미 실행된 진단을 기록·판정한다",
        "",
        "## 1. 철회",
        "",
        f"**{RETRACTS}** 를 철회한다.",
        "",
        f"사유: {payload['retraction_reason']}.",
        "",
        f"허용 기준(이제 고정): {payload['admissibility_rule']}.",
        "",
        "## 2. 측정",
        "",
        "| 대상 | 피처 | Total | 비고 |",
        "|---|---:|---:|---|",
        f"| 전체피처 회귀기 (누출) | {MEASURED['leaked_full_features']['features']:,} | "
        f"~~{leaked:.6f}~~ | **철회** |",
        f"| 전체피처 회귀기 (정상) | "
        f"{MEASURED['clean_full_features']['features']:,} | **{clean_full:.6f}** | "
        f"1-NMAE {MEASURED['clean_full_features']['one_minus_nmae']:.6f} / "
        f"FICR {MEASURED['clean_full_features']['ficr']:.6f} |",
        f"| 87피처 회귀기 (정상) | {MEASURED['clean_87_features']['features']} | "
        f"{clean_87:.6f} | 사이클 38 CONTROL |",
        f"| 배포 분류기+정책 | — | **{deployed:.6f}** | "
        f"`{MEASURED['deployed_classifier_policy']['name']}` |",
        f"| 최선 고정정책 | — | {MEASURED['best_fixed_policy']['total']:.6f} | "
        f"`{MEASURED['best_fixed_policy']['name']}` |",
        "",
        f"- 피처 커버리지 효과 **{coverage_delta:+.6f}** (87 -> 1,347)",
        f"- **구조 격차 {structural_gap:+.6f}** (직접 점회귀 -> 분류기+정책)",
        f"- 누출 크기 **{leak_size:+.6f}**",
        "",
        "## 3. 이것이 뜻하는 것",
        "",
        "이 지표에서 **46-class 분포 표현 + 결정정책은 직접 점회귀보다 약 0.087 우월하다.**",
        "프로젝트가 그 구조로 수렴한 이유가 처음으로 측정됐다.",
        "",
        "따라서 사이클 37·38 의 밴드 손실 실험은 **0.087 뒤처진 아키텍처 위에** 손실을 얹은",
        "것이다. 손실이 아무리 좋아도 따라잡을 수 없다. **실험이 잘못 놓였지 손실 축이 닫힌",
        "것이 아니다.** 올바른 실험은 분류기 틀 **안에서** 밴드 인지 학습이다.",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** ({coverage_delta:.6f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}** ({structural_gap:.6f})",
        f"- H3 `{check['H3_expectation']}` -> **{h3}** ({leak_size:.6f})",
        "",
        f"판정: **{verdict}**",
        "",
        f"손실 축 상태: **{check['loss_axis_status']}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE39_ARCHITECTURE_GAP",
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

    print(f"[C39] 철회: 전체피처 회귀기 {leaked:.6f} (scada_ws 누출)")
    print(f"[C39] 정상 전체피처 {clean_full:.6f} / 87피처 {clean_87:.6f} "
          f"-> 커버리지 효과 {coverage_delta:.6f}")
    print(f"[C39] 배포 분류기+정책 {deployed:.6f} -> 구조 격차 {structural_gap:.6f}")
    print(f"[C39] H1 {h1} | H2 {h2} | H3 {h3}")
    print(f"[C39] 판정: {verdict}")
    print(f"[C39] 손실 축: {check['loss_axis_status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
