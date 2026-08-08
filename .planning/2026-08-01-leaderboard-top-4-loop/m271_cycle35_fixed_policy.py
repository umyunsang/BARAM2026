"""M271 P4 사이클 35 — 정정: fold내 정책 선택 프리미엄과 정직한 고정정책 표면.

**이 노드는 이번 세션의 비교 기준을 정정한다.**

사이클 34 까지의 모든 후보는 `metric-aligned-probe` 의 본 parquet 에서 나왔다. 그 파일의
`prediction_kwh` 는 **fold 마다 다른 정책**을 쓴다. 실측:

    M115_XGBOOST     Q2=T0.75_G2   Q3=T0.4_G0    Q4=T0.75_G0.5
    M102_TOP100      Q2=T0.5_G1.5  Q3=T0.4_G2    Q4=T0.6_G1
    M113_LGBM_DART   Q2=T0.5_G0.5  Q3=T0.5_G0.5  Q4=T0.5_G0.35

각 fold 의 정책이 **그 fold 자신의 점수로** 선택됐다. 반면 기준선 `T0.5_G1.5` 는
`M269_PROBE_TOP100` 에 **정책 하나를 세 fold 전부에** 적용한 것이다(`m270_monthly_validation
.load_predictions`). 즉 지금까지 **fold내 최적화된 후보를 고정정책 기준선과 비교**해 왔다.

선행 측정(이 노드 실행 전에 확인):

    모델              fold별 선택   최고 고정정책          프리미엄
    M115_XGBOOST      0.638410     T0.6_G0.35 0.630662   +0.007748
    M113_LGBM_DART    0.636376     T0.5_G0.5  0.631099   +0.005276
    M102_TOP100       0.630589     T0.5_G1.5  0.629896   +0.000692

사이클 33 이 보고한 "M115 가 배포 대비 +0.009805" 중 **+0.007748 이 선택 프리미엄**이다.
제출 시점에는 정책 하나를 고정해야 하므로 fold별 선택물은 **배포 가능한 객체가 아니다.**

① 방법 리서치 (실행 전)
  - 새 방법 없음. 방법론적 쟁점은 하나다: **비교 대상은 배포 가능한 객체여야 한다.**
    fold별로 다른 정책을 고르는 절차는 예보시점에 실행할 수 없다(그 fold 의 실측을 알아야
    한다). 따라서 정직한 표면은 `(모델, 고정정책)` 쌍이다.
  - 고정정책도 63 개 중 하나를 2023 에서 고르는 것이므로 선택 편향이 남는다. 통제는 동결
    게이트 + fold 3/3 이다(사이클 7 이 같은 격자를 0/62 로 기각시킨 그 게이트).

② 사양 동결

  기준선 `M269_PROBE_TOP100 @ T0.5_G1.5` = 배포. 고정정책이므로 그대로 쓴다.
  후보   정책 격자를 가진 모든 probe 모델 x 전 정책 (전부 **고정**, 세 fold 동일)

  **승격 규칙 (실행 전 동결)**
    R1  배포 대비 pooled Total 개선
    R2  배포를 부모로 한 **동결 게이트 통과**
    R3  세 fold **각각**에서 배포 초과
  복수 자격시 **부트스트랩 하한 `q05` 최대**, 동률시 이름 사전순.

  사전확약:
    H1  자격을 얻는 `(모델, 고정정책)` 이 적어도 하나 있다.
    H2  그 이득이 검출문턱 `+0.001013` 을 넘는다.
    H3  최선 고정정책 후보가 사이클 33 이 보고한 M115 값(0.638410)보다 **낮다**.
        (프리미엄이 실재한다는 확인. 성립을 예상한다)
    H4  자격 수가 전체 시험 수의 **10% 이하**다. 넘으면 다중검정 통제 실패로 본다.

**게이트를 수정하지 않는다.** 2023 만 쓴다. lockbox 미사용.
"""

from __future__ import annotations

import glob
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, evaluate_gate
from m270_monthly_validation import load_predictions
from m271_cycle13_ensemble import FOLDS as PROBE_FOLDS
from m271_cycle21_mos import QUARTER_OF_MONTH
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROBE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORT_MD = REPORTS / "m271_cycle35_fixed_policy.md"
RECEIPT = REPORTS / "m271_cycle35_fixed_policy_receipt.json"

NODE_ID = "C1N35_FIXED_POLICY_CORRECTION"
LANE = "L6"
PARENT_NODE = "C1N34_SHRINK_TO_BEST"
CORRECTS = ("C1N33_SINGLE_VS_ENSEMBLE", "C1N34_SHRINK_TO_BEST", "C1N20_ALPHA_ENDPOINT")
DEPLOYED = "T0.5_G1.5"
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]
FOLDS = ("Q2", "Q3", "Q4")
GATE_DETECTION_THRESHOLD = 0.001013
CYCLE33_M115_PERFOLD = 0.638410
H4_MAX_QUALIFY_FRACTION = 0.10

# 선행 측정. 이 노드 실행 **전에** 잰 값이며 사양의 일부로 동결한다.
PREMIUM_MEASURED = {
    "M115_XGBOOST": {"per_fold": 0.638410, "best_fixed": 0.630662,
                     "policy": "T0.6_G0.35", "premium": 0.007748},
    "M113_LGBM_DART": {"per_fold": 0.636376, "best_fixed": 0.631099,
                       "policy": "T0.5_G0.5", "premium": 0.005276},
    "M102_TOP100": {"per_fold": 0.630589, "best_fixed": 0.629896,
                    "policy": "T0.5_G1.5", "premium": 0.000692},
}


def models_with_policies() -> list[str]:
    names = set()
    for path in glob.glob(str(PROBE / f"*-{PROBE_FOLDS[0]}-policies.parquet")):
        stem = Path(path).name
        name = stem[: -len(f"-{PROBE_FOLDS[0]}-policies.parquet")]
        if all(
            (PROBE / f"{name}-{f}-policies.parquet").exists() for f in PROBE_FOLDS
        ):
            names.add(name)
    return sorted(names)


def load_policy_grid(model: str) -> tuple[pd.DataFrame, list[str]]:
    parts = []
    for fold in PROBE_FOLDS:
        q = pd.read_parquet(PROBE / f"{model}-{fold}-policies.parquet")
        q["forecast_kst_dtm"] = pd.to_datetime(q["forecast_kst_dtm"])
        parts.append(q)
    grid = pd.concat(parts, ignore_index=True)
    grid["month"] = grid["forecast_kst_dtm"].dt.to_period("M").astype(str)
    cols = [c for c in grid.columns if c not in {*KEYS, "actual_kwh", "month"}]
    return grid, cols


def by_fold_total(frame: pd.DataFrame) -> dict[str, float]:
    f = frame.copy()
    f["fold"] = f["month"].map(QUARTER_OF_MONTH)
    return {
        fold: official(cell)["total"]
        for fold, cell in f.groupby("fold", observed=True)
        if fold in FOLDS
    }


def main() -> int:
    parent = load_predictions(DEPLOYED)
    parent_score = official(parent)
    parent_folds = by_fold_total(parent)

    models = models_with_policies()
    rows: list[dict[str, Any]] = []
    tested = 0
    for model in models:
        grid, cols = load_policy_grid(model)
        # 배포와 같은 행집합에서만 비교한다.
        common = grid.merge(parent.loc[:, KEYS], on=KEYS, how="inner")
        if len(common) != len(parent):
            continue
        for policy in cols:
            frame = common.loc[:, [*KEYS, "actual_kwh", "month"]].copy()
            frame["prediction_kwh"] = common[policy].to_numpy(dtype=float)
            score = official(frame)
            tested += 1
            if score["total"] <= parent_score["total"]:
                continue  # R1 미달은 게이트까지 갈 필요가 없다
            gate = evaluate_gate(frame, parent)
            stats = gate.evidence
            folds = by_fold_total(frame)
            wins = sum(1 for f in FOLDS if folds[f] > parent_folds[f])
            rows.append(
                {
                    "model": model, "policy": policy,
                    **score,
                    "delta_vs_deployed": score["total"] - parent_score["total"],
                    "fold_wins": wins,
                    "gate_passed": bool(gate.passed),
                    "gate_flags": {la.split()[0]: bool(ok)
                                   for la, ok in gate.conditions.items()},
                    "positive_months": int(stats["positive_months"]),
                    "months_scored": int(stats["months_scored"]),
                    "sign_test_p": float(stats["sign_test_p_greater"]),
                    "bootstrap_q05": float(stats["block_bootstrap_q05"]),
                    "min_delta": float(stats["min_total_delta"]),
                    "qualifies": bool(gate.passed and wins == len(FOLDS)),
                }
            )
    rows.sort(key=lambda r: -r["delta_vs_deployed"])
    qualified = sorted(
        (r for r in rows if r["qualifies"]),
        key=lambda r: (-r["bootstrap_q05"], r["model"], r["policy"]),
    )
    chosen = qualified[0] if qualified else None

    h1 = bool(chosen)
    h2 = bool(chosen and chosen["delta_vs_deployed"] > GATE_DETECTION_THRESHOLD)
    h3 = bool(chosen and chosen["total"] < CYCLE33_M115_PERFOLD)
    h4 = bool(tested and len(qualified) / tested <= H4_MAX_QUALIFY_FRACTION)
    trustworthy = bool(h1 and h2 and h4)
    promoted_total = chosen["total"] if trustworthy else parent_score["total"]
    promoted_name = (
        f"{chosen['model']}@{chosen['policy']}" if trustworthy else f"DEPLOYED@{DEPLOYED}"
    )
    verdict = (
        "FIXED_POLICY_CANDIDATE_PROMOTED" if trustworthy
        else ("QUALIFIED_BUT_MULTIPLICITY_SUSPECT" if h1
              else "NO_FIXED_POLICY_BEATS_DEPLOYED")
    )

    check = {
        "H1_expectation": "자격 (모델, 고정정책) 이 적어도 하나",
        "H1_held": h1, "H1_qualifying_count": len(qualified),
        "H2_expectation": f"이득이 검출문턱 {GATE_DETECTION_THRESHOLD} 초과",
        "H2_held": h2,
        "H3_expectation": f"최선 고정정책 < 사이클 33 의 M115 fold별선택 값 "
                          f"({CYCLE33_M115_PERFOLD})",
        "H3_held": h3,
        "H4_expectation": f"자격 비율 <= {H4_MAX_QUALIFY_FRACTION:.0%}",
        "H4_held": h4,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "corrects": list(CORRECTS),
        "correction": "사이클 13~34 의 후보는 fold 마다 다른 정책을 쓰는 probe 본 parquet "
                      "에서 나왔고, 기준선은 고정정책이었다. 비교가 불공정했다",
        "premium_measured_before_this_node": PREMIUM_MEASURED,
        "gate_version": GATE_VERSION, "gate_modified": False, "lockbox_used": False,
        "deployed": {"source": "M269_PROBE_TOP100 @ " + DEPLOYED, **parent_score},
        "deployed_by_fold": parent_folds,
        "models_with_policy_grid": models,
        "combinations_tested": tested,
        "improving_over_deployed": len(rows),
        "top15": rows[:15],
        "qualified": qualified,
        "predeclared_check": check,
        "promoted": promoted_name,
        "promoted_total": promoted_total,
        "gap_to_local_target": 0.66 - promoted_total,
        "gap_to_offset_implied_local": 0.638881 - promoted_total,
    }

    lines = [
        "# M271 P4 사이클 35 — 정정: fold내 정책 선택 프리미엄",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- **정정 대상**: `{', '.join(CORRECTS)}`",
        f"- 게이트: `{GATE_VERSION}` (읽기만 함) / lockbox 미사용 / 2023 만",
        "",
        "## 0. 무엇이 틀렸는가",
        "",
        payload["correction"] + ".",
        "",
        "probe 본 parquet 은 fold 마다 **그 fold 의 점수로 고른** 정책을 쓴다. 기준선",
        "`T0.5_G1.5` 는 `M269_PROBE_TOP100` 에 정책 하나를 세 fold 전부 적용한 것이다.",
        "예보시점에는 그 fold 의 실측을 모르므로 fold별 선택은 **배포 가능한 절차가 아니다.**",
        "",
        "| 모델 | fold별 선택 | 최고 고정정책 | 프리미엄 |",
        "|---|---:|---:|---:|",
    ]
    for m, v in PREMIUM_MEASURED.items():
        lines.append(
            f"| `{m}` | {v['per_fold']:.6f} | `{v['policy']}` {v['best_fixed']:.6f} | "
            f"**{v['premium']:+.6f}** |"
        )
    lines += [
        "",
        f"사이클 33 이 보고한 \"M115 배포 대비 +0.009805\" 중 "
        f"**+{PREMIUM_MEASURED['M115_XGBOOST']['premium']:.6f} 이 선택 프리미엄**이다.",
        "",
        "## 1. 정직한 표면 — 고정정책만",
        "",
        f"기준선 `{payload['deployed']['source']}` Total **{parent_score['total']:.6f}**",
        f"정책 격자를 가진 모델 {len(models)} 개, 조합 **{tested:,}** 개 시험, "
        f"배포 초과 {len(rows)} 개, 자격 **{len(qualified)}** 개",
        "",
        "| 모델 | 정책 | Total | 배포 대비 | G1G2G3G4 | 양수월 | q05 | 최소월 | fold | 자격 |",
        "|---|---|---:|---:|:---:|---:|---:|---:|:---:|:---:|",
    ]
    for r in rows[:15]:
        flags = "".join("O" if r["gate_flags"].get(x) else "-"
                        for x in ("G1", "G2", "G3", "G4"))
        lines.append(
            f"| `{r['model']}` | `{r['policy']}` | {r['total']:.6f} | "
            f"{r['delta_vs_deployed']:+.6f} | `{flags}` | "
            f"{r['positive_months']}/{r['months_scored']} | {r['bootstrap_q05']:+.6f} | "
            f"{r['min_delta']:+.6f} | {r['fold_wins']}/3 | "
            f"{'**자격**' if r['qualifies'] else '-'} |"
        )

    lines += [
        "",
        "## 2. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** ({len(qualified)} 개)",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}** "
        f"({len(qualified)}/{tested} = "
        f"{(len(qualified) / tested if tested else 0):.4f})",
        "",
        f"판정: **{verdict}**",
        "",
        f"승격: **`{promoted_name}`** Total **{promoted_total:.6f}**",
        f"- 로컬 목표 0.66 까지 {0.66 - promoted_total:+.6f}",
        f"- 오프셋 함의 로컬(0.638881) 까지 **{0.638881 - promoted_total:+.6f}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE35_FIXED_POLICY",
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

    print(f"[C35] 기준선 {payload['deployed']['source']} {parent_score['total']:.6f}")
    print(f"[C35] 모델 {len(models)} 개 / 조합 {tested:,} 시험 / 배포초과 {len(rows)} / "
          f"자격 {len(qualified)}")
    for r in rows[:6]:
        print(f"[C35]   {r['model']:<26} {r['policy']:>12} {r['total']:.6f} "
              f"({r['delta_vs_deployed']:+.6f}) 게이트 "
              f"{'통과' if r['gate_passed'] else '기각'} fold {r['fold_wins']}/3 "
              f"자격 {r['qualifies']}")
    print(f"[C35] H1 {h1} | H2 {h2} | H3 {h3} | H4 {h4}")
    print(f"[C35] 판정: {verdict}  ->  `{promoted_name}` {promoted_total:.6f} "
          f"(오프셋 함의 로컬까지 {0.638881 - promoted_total:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
