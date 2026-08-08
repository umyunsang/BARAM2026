"""M271 P4 사이클 32 — lockbox 2 차 소비: 결합 연산자의 연도 전이 검증.

**되돌릴 수 없는 행위다.** 2024 lockbox 는 2026-08-01 22:11 에 이미 1 차 소진됐고
(freeze `f25afd46...`, run `lockbox-2024`, 챔피언 `tree-calibrated` @ 0.627605),
이 노드는 사용자의 명시적 승인 하에 **2 차 소비**를 기록한다.

왜 챔피언 자체가 아니라 연산자를 검증하는가
---------------------------------------------
챔피언 `M271_MEDIAN4` 의 멤버 4 개는 2024 예측이 없다. 만들려면 러너를 새로 써야 하는데
receipt 에 완전한 재현 사양이 없다(M113·M115 는 `feature_count` 만, M244 는 Q4 JSON 부재).
재구성해서 얻은 2024 점수는 **내 재구성물의 점수**이고, 나빠졌을 때 "2024 가 어려웠다" 와
"내가 다르게 만들었다" 를 구분할 수 없다. 해석 불가능한 숫자에 일회용 카드를 쓰지 않는다.

대신 챔피언의 **핵심 기전**을 검증한다. 사이클 17 의 주장은 이것이다:

    FICR 은 `|err| <= 6% 용량` 지시함수이므로 계단 손실이다. 이상치 멤버 하나가 평균을
    밴드 밖으로 끌어내지만(영향력 무한) 중앙값은 끌리지 않는다(영향력 유한).
    따라서 median 이 mean 을 이긴다.

기전이 진짜라면 **다른 해, 다른 모델군** 에서도 재현돼야 한다. lockbox 3 개 모델
(`tree-calibrated`, `catboost-shared`, `control-random_forest`)의 2024 예측이 이미
디스크에 있고, 이들은 챔피언 멤버가 **아니다** — 즉 독립 복제다. 2024 는 12 개월이라
동결 게이트 검정력도 2023(9 개월)보다 높다.

**이 노드가 확립하는 것**: 결합 연산자의 연도 전이.
**이 노드가 확립하지 않는 것**: `M271_MEDIAN4` 자체의 2024 점수. 그건 여전히 미측정이다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 17 의 대조를 그대로 다른 연도·다른 모델군에 옮긴다.
  - 복제(replication)가 기전 주장의 표준 검증이라는 것이 유일한 방법론적 근거다.
    사이클 25 가 fold 간 재현 실패로 lambda 를 기각했고, 사이클 29 가 같은 잣대로
    챔피언을 통과시켰다. 여기서는 **연도 간** 재현을 본다.
  - 멤버 3 개는 1 차 freeze 가 정한 것이며 내가 고르지 않았다. 선택 편향 없음.

② 사양 동결 (점수를 읽기 **전에** 확정)

  사전확약:
    H1  2024 에서 `median(3) > mean(3)`.
    H2  `median(3)` 이 `mean(3)` 을 부모로 한 **동결 게이트를 통과**한다 (12 개월).
    H3  `median(3)` 이 **최고 단일 모델**보다 높다 (Breiman 의 스태킹 주장).
    H4  이득이 **FICR 쪽에서** 나온다: FICR 델타의 Total 기여가 1-NMAE 델타 기여보다 크다.
        기전이 "밴드 밖으로 끌려나가는 것을 막는다" 이므로 FICR 이 주도해야 한다.

  H1·H2 가 성립하면 결합 연산자가 연도를 넘어 전이한다. H4 까지 성립하면 전이하는 것이
  **주장된 그 기전** 임이 확인된다. 기각되면 챔피언의 2023 이득이 연도 특이일 위험을
  명시 기록한다.

  **게이트를 수정하지 않는다.** 읽기만 한다. 정책도 바꾸지 않는다.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, evaluate_gate
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
LOCKBOX_DIR = ROOT / "artifacts" / "backtests" / "lockbox" / "lockbox-2024"
LOCK_PATH = ROOT / "artifacts" / "locks" / "lockbox-2024.consumed.json"
SECOND_USE_RECORD = ROOT / "artifacts" / "locks" / "lockbox-2024.second-use.json"
REPORT_MD = REPORTS / "m271_cycle32_lockbox_operator.md"
RECEIPT = REPORTS / "m271_cycle32_lockbox_operator_receipt.json"

NODE_ID = "C1N32_LOCKBOX_OPERATOR"
LANE = "L4"
PARENT_NODE = "C1N29_CHAMPION_AUDIT"
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]

FIRST_USE = {
    "when_utc": "2026-08-01T13:11:03Z (파일 mtime 22:11 KST)",
    "freeze_id": "f25afd46c98560e7280d4625e3e7aa283c508eaa3d7966de42ad6c43299de35f",
    "run_id": "lockbox-2024",
    "candidates": ["control-random_forest", "tree-calibrated",
                   "catboost-shared-b1e92b1d786fa958"],
    "champion": "tree-calibrated",
    "champion_total_2024": 0.6276054065728833,
}

MEMBERS = (
    "tree-calibrated",
    "catboost-shared-b1e92b1d786fa958",
    "control-random_forest",
)


def load_members() -> pd.DataFrame:
    base: pd.DataFrame | None = None
    for name in MEMBERS:
        path = LOCKBOX_DIR / f"{name}.parquet"
        frame = pd.read_parquet(path)
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
        piece = frame.loc[:, [*KEYS, "prediction_kwh"]].rename(
            columns={"prediction_kwh": name}
        )
        if base is None:
            base = frame.loc[:, [*KEYS, "actual_kwh"]].copy()
        before = len(base)
        base = base.merge(piece, on=KEYS, how="inner")
        assert len(base) == before, f"{name} 조인에서 행이 바뀌었다"
    assert base is not None
    base["month"] = base["forecast_kst_dtm"].dt.to_period("M").astype(str)
    return base


def combined(base: pd.DataFrame, how: str) -> pd.DataFrame:
    arr = base.loc[:, list(MEMBERS)].to_numpy(dtype="float64")
    out = base.loc[:, [*KEYS, "actual_kwh", "month"]].copy()
    out["prediction_kwh"] = np.median(arr, axis=1) if how == "median" else arr.mean(axis=1)
    return out


def single(base: pd.DataFrame, name: str) -> pd.DataFrame:
    out = base.loc[:, [*KEYS, "actual_kwh", "month"]].copy()
    out["prediction_kwh"] = base[name]
    return out


def main() -> int:
    assert LOCK_PATH.exists(), "1 차 소비 락이 없다. 전제가 틀렸으므로 중단한다"
    first_lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert first_lock.get("state") == "CONSUMED_BEFORE_SCORING"

    # 2 차 소비를 **채점 전에** 기록한다. 1 차 락은 건드리지 않는다.
    spec_source = Path(__file__).read_bytes()
    record = {
        "schema_version": 1,
        "state": "SECOND_USE_CONSUMED_BEFORE_SCORING",
        "lockbox_year": 2024,
        "authorised_by": "user (explicit, 2026-08-04)",
        "first_use": FIRST_USE,
        "first_lock_sha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "second_use_node": NODE_ID,
        "second_use_question": "결합 연산자(median vs mean)의 연도 전이",
        "second_use_does_not_establish": "M271_MEDIAN4 자체의 2024 점수",
        "members": list(MEMBERS),
        "predeclaration_sha256": hashlib.sha256(spec_source).hexdigest(),
        "recorded_utc": datetime.now(UTC).isoformat(),
    }
    SECOND_USE_RECORD.write_text(
        json.dumps(record, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    base = load_members()
    med = combined(base, "median")
    avg = combined(base, "mean")
    med_score = official(med)
    avg_score = official(avg)
    singles = {name: official(single(base, name)) for name in MEMBERS}
    best_name = max(singles, key=lambda n: singles[n]["total"])
    best = singles[best_name]

    gate = evaluate_gate(med, avg)
    stats = gate.evidence

    delta_total = med_score["total"] - avg_score["total"]
    delta_ficr_contrib = 0.5 * (med_score["ficr"] - avg_score["ficr"])
    delta_nmae_contrib = 0.5 * (
        med_score["one_minus_nmae"] - avg_score["one_minus_nmae"]
    )

    h1 = bool(delta_total > 0)
    h2 = bool(gate.passed)
    h3 = bool(med_score["total"] > best["total"])
    h4 = bool(delta_ficr_contrib > delta_nmae_contrib)

    # 쌍별 스프레드 — 기전이 걸려면 멤버가 실제로 흩어져 있어야 한다.
    arr = base.loc[:, list(MEMBERS)].to_numpy(dtype="float64")
    from baram.constants import CAPACITIES_KWH

    cap = base["group_id"].map(CAPACITIES_KWH).astype(float).to_numpy()
    elig = base["actual_kwh"].to_numpy(dtype="float64") >= 0.10 * cap
    spread = ((arr.max(axis=1) - arr.min(axis=1)) / cap)[elig]
    pair_corr = {
        f"{a}~{b}": float(np.corrcoef(base[a], base[b])[0, 1])
        for a, b in itertools.combinations(MEMBERS, 2)
    }

    monthly = {}
    for m in sorted(base["month"].unique()):
        cm = med.loc[med["month"] == m]
        ca = avg.loc[avg["month"] == m]
        monthly[m] = {
            "median": official(cm)["total"],
            "mean": official(ca)["total"],
            "delta": official(cm)["total"] - official(ca)["total"],
        }
    positive_months = sum(1 for v in monthly.values() if v["delta"] > 0)

    verdict = (
        "OPERATOR_TRANSFERS_ACROSS_YEARS" if (h1 and h2)
        else ("OPERATOR_POSITIVE_BUT_GATE_REJECTED" if h1
              else "OPERATOR_DOES_NOT_TRANSFER")
    )
    check = {
        "H1_expectation": "2024 에서 median(3) > mean(3)",
        "H1_held": h1, "H1_measured": delta_total,
        "H2_expectation": "median 이 mean 부모로 동결 게이트 통과 (12 개월)",
        "H2_held": h2,
        "H3_expectation": "median 이 최고 단일 모델보다 높다",
        "H3_held": h3,
        "H4_expectation": "이득이 FICR 쪽에서 나온다 (기전 확인)",
        "H4_held": h4,
        "mechanism_confirmed": bool(h1 and h4),
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "gate_version": GATE_VERSION, "gate_modified": False,
        "lockbox_reopened": True,
        "lockbox_use_ordinal": 2,
        "second_use_record": str(SECOND_USE_RECORD.relative_to(ROOT)),
        "first_use": FIRST_USE,
        "scope_established": "결합 연산자의 연도 전이",
        "scope_not_established": "M271_MEDIAN4 자체의 2024 점수 (여전히 미측정)",
        "members": list(MEMBERS),
        "rows": len(base),
        "eligible_rows": int(elig.sum()),
        "member_diversity": {
            "spread_median_cap": float(np.median(spread)),
            "spread_p90_cap": float(np.quantile(spread, 0.90)),
            "pairwise_prediction_correlation": pair_corr,
        },
        "scores_2024": {
            "median": med_score, "mean": avg_score,
            "singles": {k: v for k, v in singles.items()},
            "best_single": best_name,
            "delta_median_minus_mean": delta_total,
            "delta_median_minus_best_single": med_score["total"] - best["total"],
            "ficr_contribution": delta_ficr_contrib,
            "nmae_contribution": delta_nmae_contrib,
        },
        "gate": {
            "passed": h2,
            "flags": {la.split()[0]: bool(ok) for la, ok in gate.conditions.items()},
            "positive_months": int(stats["positive_months"]),
            "months_scored": int(stats["months_scored"]),
            "sign_test_p": float(stats["sign_test_p_greater"]),
            "median_delta": float(stats["median_total_delta"]),
            "bootstrap_q05": float(stats["block_bootstrap_q05"]),
            "min_delta": float(stats["min_total_delta"]),
        },
        "monthly": monthly,
        "positive_months_raw": positive_months,
        "predeclared_check": check,
    }

    s = payload["scores_2024"]
    flags = "".join("O" if payload["gate"]["flags"].get(x) else "-"
                    for x in ("G1", "G2", "G3", "G4"))
    lines = [
        "# M271 P4 사이클 32 — lockbox 2 차 소비: 결합 연산자의 연도 전이",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "- **lockbox 2 차 소비.** 사용자 명시 승인. 되돌릴 수 없다",
        f"- 1 차 소비: {FIRST_USE['when_utc']}, freeze `{FIRST_USE['freeze_id'][:16]}...`, "
        f"챔피언 `{FIRST_USE['champion']}` @ {FIRST_USE['champion_total_2024']:.6f}",
        f"- 2 차 기록: `{payload['second_use_record']}` (채점 **전에** 기록)",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함)",
        "",
        "## 0. 범위",
        "",
        f"- **확립하는 것**: {payload['scope_established']}",
        f"- **확립하지 않는 것**: {payload['scope_not_established']}",
        "",
        "챔피언 멤버 4 개는 2024 예측이 없고 receipt 에 완전한 재현 사양도 없다. 재구성해서",
        "얻은 숫자는 재구성물의 점수이지 챔피언의 점수가 아니므로, 일회용 카드를 거기에",
        "쓰지 않았다. 대신 챔피언의 **핵심 기전** 을 다른 해·다른 모델군에서 복제한다.",
        "",
        "## 1. 멤버 (1 차 freeze 가 정한 것. 내가 고르지 않았다)",
        "",
        f"행 {len(base):,} / 유효행 {int(elig.sum()):,}",
        "",
        "| 모델 | 2024 Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
    ]
    for name in MEMBERS:
        v = singles[name]
        mark = " **(최고)**" if name == best_name else ""
        lines.append(
            f"| `{name}`{mark} | {v['total']:.6f} | {v['one_minus_nmae']:.6f} | "
            f"{v['ficr']:.6f} |"
        )
    lines += [
        "",
        f"멤버 스프레드 중앙값 {payload['member_diversity']['spread_median_cap']:.4f} 용량, "
        f"p90 {payload['member_diversity']['spread_p90_cap']:.4f}",
        "",
        "| 쌍 | 예측 상관 |",
        "|---|---:|",
    ]
    for k, v in pair_corr.items():
        lines.append(f"| `{k}` | {v:+.4f} |")

    lines += [
        "",
        "## 2. 결합 연산자 (H1 · H3 · H4)",
        "",
        "| 연산 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|",
        f"| `mean(3)` | {avg_score['total']:.6f} | {avg_score['one_minus_nmae']:.6f} | "
        f"{avg_score['ficr']:.6f} |",
        f"| **`median(3)`** | **{med_score['total']:.6f}** | "
        f"{med_score['one_minus_nmae']:.6f} | {med_score['ficr']:.6f} |",
        f"| 최고 단일 `{best_name}` | {best['total']:.6f} | "
        f"{best['one_minus_nmae']:.6f} | {best['ficr']:.6f} |",
        "",
        f"- median - mean = **{delta_total:+.6f}**",
        f"- median - 최고단일 = **{s['delta_median_minus_best_single']:+.6f}**",
        f"- 기여 분해: FICR **{delta_ficr_contrib:+.6f}** / "
        f"1-NMAE {delta_nmae_contrib:+.6f}",
        "",
        "## 3. 동결 게이트 — 12 개월 (H2)",
        "",
        f"`{flags}` {payload['gate']['positive_months']}/"
        f"{payload['gate']['months_scored']}월 "
        f"p={payload['gate']['sign_test_p']:.4f} "
        f"중앙값 {payload['gate']['median_delta']:+.6f} "
        f"q05={payload['gate']['bootstrap_q05']:+.6f} "
        f"최소월 {payload['gate']['min_delta']:+.6f} -> "
        f"**{'통과' if h2 else '기각'}**",
        "",
        "| 월 | mean | median | 차이 |",
        "|---|---:|---:|---:|",
    ]
    for m, v in monthly.items():
        lines.append(
            f"| {m} | {v['mean']:.6f} | {v['median']:.6f} | **{v['delta']:+.6f}** |"
        )
    lines += [
        "",
        f"양수월 {positive_months}/{len(monthly)}",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** ({delta_total:+.6f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}**",
        "",
        f"기전 확인 (H1 ∧ H4): **{check['mechanism_confirmed']}**",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE32_LOCKBOX_OPERATOR",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(spec_source).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [], "model_fits": 0,
        "lockbox_reopened": True,
        "lockbox_use_ordinal": 2,
        "new_2024_evaluation": True,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[C32] lockbox 2 차 소비 기록 -> {SECOND_USE_RECORD.name}")
    for name in MEMBERS:
        print(f"[C32]   {name:>38} {singles[name]['total']:.6f}")
    print(f"[C32] mean(3)   {avg_score['total']:.6f}")
    print(f"[C32] median(3) {med_score['total']:.6f}  "
          f"(mean 대비 {delta_total:+.6f}, 최고단일 대비 "
          f"{s['delta_median_minus_best_single']:+.6f})")
    print(f"[C32] 기여: FICR {delta_ficr_contrib:+.6f} / 1-NMAE {delta_nmae_contrib:+.6f}")
    print(f"[C32] 게이트 [{flags}] {payload['gate']['positive_months']}/"
          f"{payload['gate']['months_scored']}월 "
          f"p={payload['gate']['sign_test_p']:.4f} "
          f"q05={payload['gate']['bootstrap_q05']:+.6f}")
    print(f"[C32] H1 {h1} | H2 {h2} | H3 {h3} | H4 {h4}  -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
