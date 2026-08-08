"""M271 P4 사이클 26 — 멤버 풀 폭의 Q3 전용 선별 (C9 전제 재검).

사이클 24 가 "멤버 수는 4 개가 12 개보다 낫다" 로 축을 닫았다. 그런데 그 12 개는
**3 개 fold 를 전부 가진 모델** 이라는 조건에서 나온 수다. probe 디렉터리 실측:

    Q3 단독 124 개 / Q4 53 개 / Q2 16 개 / 3-fold 전부 13 개

즉 폐기 근거가 실은 **평가 표면의 제약** 이었다. 현재 쓰는 4 개의 31 배 풀이 Q3 에 있다.

**이 노드는 승격하지 않는다.** 동결 게이트는 9 개월이 필요하고 Q3 는 3 개월뿐이다.
Q2/Q4 백필에 계산자원을 쓸 가치가 있는지만 판정하는 **선별** 이다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 17 의 결합자(median)와 사이클 24 의 사다리 설계를 그대로 쓰고
    **멤버 풀만** 넓힌다.
  - 선택 편향 방어가 유일한 방법론적 쟁점이다. 124 개 중 좋은 것을 고르면 same-fold
    선택이다. 따라서 멤버 집합은 **성능과 무관한 규칙**으로만 만든다.
      P4    사이클 13 이 동결한 E3 네 개 (대조군)
      P12   3-fold 전부 가진 12 개 (사이클 13 의 E1)
      P124  Q3 에 예측이 있는 **전부**. 선택 없음
    P124 는 나쁜 멤버도 전부 포함한다 — 그게 요점이다. median 의 붕괴점이 50% 이므로
    나쁜 멤버가 절반 미만이면 견딘다는 주장을 정면으로 시험한다.

② 사양 동결

  평가는 **Q3(2023-07/08/09) 전용**. 공식 산식은 그대로 쓰되 모집단이 다르므로 다른
  사이클의 pooled 수치와 직접 비교하지 않는다. 모든 비교는 같은 Q3 안에서만 한다.

  사전확약(실행 전 동결):
    H1  Q3 에서 `median(P124) > median(P4)`.
    H2  그 차이가 게이트 검출문턱 `0.001013` 을 넘는다.
    H3  Q3 세 달 **전부**에서 P124 가 P4 보다 높다.
        (3 개월은 동결 게이트에 부족하다. **게이트 대용이 아니며 승격 근거가 아니다.**)
    H4  P124 봉투의 커버리지가 P4 봉투보다 크다. 넓은 풀이 실제로 실측을 감싸는가.

  셋(H1·H2·H3)이 모두 성립하면 Q2/Q4 백필이 정당화된다. 하나라도 기각되면 사이클 24 의
  폐기가 풀 폭에 대해서도 유효한 것으로 확정된다.

**게이트를 수정하지 않는다. 승격하지 않는다.**

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import glob
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_cycle13_ensemble import ALL_MODELS, ENSEMBLES
from m271_evaluate_candidate import official

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROBE = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
REPORT_MD = REPORTS / "m271_cycle26_pool_screen.md"
RECEIPT = REPORTS / "m271_cycle26_pool_screen_receipt.json"

NODE_ID = "C1N26_POOL_WIDTH_SCREEN"
LANE = "L7"
PARENT_NODE = "C1N20_ALPHA_ENDPOINT"
REOPENS = "C1N24_MEMBER_COUNT_REOPEN"
FOLD = "dev-2023-Q3"
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]
NEEDED = {*KEYS, "actual_kwh", "prediction_kwh"}
GATE_DETECTION_THRESHOLD = 0.001013
ELIGIBLE_THRESHOLD = 0.10
MIN_MONTH_ROWS = 100  # fold 경계 흘림 진단용. 사전확약 H3 자체는 이 필터를 쓰지 않는다


def pool_paths() -> list[Path]:
    return sorted(
        Path(p) for p in glob.glob(str(PROBE / f"*-{FOLD}.parquet"))
        if "policies" not in Path(p).name
    )


def model_name(path: Path) -> str:
    return path.name[: -len(f"-{FOLD}.parquet")]


def load_pool(paths: list[Path]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """멤버를 나란히 놓는다. 성능과 무관한 유효성 검사만 한다."""
    base: pd.DataFrame | None = None
    cols: list[str] = []
    dropped: list[str] = []
    for path in paths:
        frame = pd.read_parquet(path)
        name = model_name(path)
        if not NEEDED <= set(frame.columns):
            dropped.append(f"{name}: 컬럼 부족")
            continue
        frame = frame.loc[:, sorted(NEEDED)].copy()
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
        if frame["prediction_kwh"].isna().any():
            dropped.append(f"{name}: NaN 예측")
            continue
        if frame["prediction_kwh"].nunique() <= 1:
            dropped.append(f"{name}: 상수 예측")
            continue
        piece = frame.loc[:, [*KEYS, "prediction_kwh"]].rename(
            columns={"prediction_kwh": name}
        )
        if base is None:
            base = frame.loc[:, [*KEYS, "actual_kwh"]].copy()
        before = len(base)
        base = base.merge(piece, on=KEYS, how="inner")
        if len(base) != before:
            dropped.append(f"{name}: 조인 행 불일치 {before}->{len(base)}")
            base = base.drop(columns=[name])
            continue
        cols.append(name)
    assert base is not None, "풀이 비었다"
    return base, cols, dropped


def score(base: pd.DataFrame, members: list[str]) -> dict[str, Any]:
    arr = base.loc[:, members].to_numpy(dtype="float64")
    frame = base.loc[:, [*KEYS, "actual_kwh"]].copy()
    frame["prediction_kwh"] = np.median(arr, axis=1)
    frame["month"] = frame["forecast_kst_dtm"].dt.to_period("M").astype(str)
    result = official(frame)
    cap = frame["group_id"].map(CAPACITIES_KWH).astype(float).to_numpy()
    actual = frame["actual_kwh"].to_numpy(dtype="float64")
    lo, hi = arr.min(axis=1), arr.max(axis=1)
    elig = actual >= ELIGIBLE_THRESHOLD * cap
    inside = (actual >= lo) & (actual <= hi)
    monthly = {}
    for m, cell in frame.groupby("month", observed=True):
        monthly[m] = official(cell)["total"]
    return {
        "n_members": len(members),
        **result,
        "monthly_total": monthly,
        "coverage_eligible": float(inside[elig].mean()),
        "median_envelope_width_cap": float(np.median(((hi - lo) / cap)[elig])),
    }


def main() -> int:
    paths = pool_paths()
    base, available, dropped = load_pool(paths)

    p4 = [m for m in ENSEMBLES["E3_FOUR_FAMILY"] if m in available]
    p12 = [m for m in ALL_MODELS if m in available]
    p_all = list(available)
    assert len(p4) == len(ENSEMBLES["E3_FOUR_FAMILY"]), f"E3 멤버 누락: {p4}"

    sets = {"P4_E3_FOUR_FAMILY": p4, "P12_FULL_FOLD_COVERAGE": p12, "P_ALL_Q3": p_all}
    results = {name: score(base, members) for name, members in sets.items()}

    # 개별 멤버 품질 분포 — 나쁜 멤버가 얼마나 있는지
    singles = []
    for name in p_all:
        frame = base.loc[:, [*KEYS, "actual_kwh"]].copy()
        frame["prediction_kwh"] = base[name]
        singles.append({"model": name, "total": official(frame)["total"]})
    singles.sort(key=lambda x: -x["total"])
    totals = np.array([s["total"] for s in singles])

    a, b = results["P4_E3_FOUR_FAMILY"], results["P_ALL_Q3"]
    delta = b["total"] - a["total"]
    months = sorted(a["monthly_total"])
    monthly_delta = {m: b["monthly_total"][m] - a["monthly_total"][m] for m in months}

    h1 = bool(delta > 0)
    h2 = bool(abs(delta) > GATE_DETECTION_THRESHOLD and delta > 0)
    h3 = bool(all(v > 0 for v in monthly_delta.values()))
    h4 = bool(b["coverage_eligible"] > a["coverage_eligible"])
    backfill = bool(h1 and h2 and h3)

    # 사양 결함 기록. H3 에 최소 월 크기를 명시하지 않았는데 fold 경계가 2023-10 으로
    # 3 행 흘러넘쳤다. **사전확약은 완화하지 않는다.** 다만 판정이 그 결함에 좌우되는지는
    # 밝혀야 하므로 얇은 달을 뺀 읽기를 함께 기록한다.
    month_rows = (
        base.assign(month=base["forecast_kst_dtm"].dt.to_period("M").astype(str))
        .groupby("month", observed=True)
        .size()
        .to_dict()
    )
    thick_months = [m for m in months if month_rows.get(m, 0) >= MIN_MONTH_ROWS]
    h3_thick = bool(
        thick_months and all(monthly_delta[m] > 0 for m in thick_months)
    )
    defect = {
        "issue": "H3 에 최소 월 크기 미명시. fold 경계가 얇은 달로 흘러넘쳤다",
        "month_rows": {m: int(month_rows.get(m, 0)) for m in months},
        "thin_months_excluded": [m for m in months if m not in thick_months],
        "H3_on_thick_months_only": h3_thick,
        "verdict_changes_if_corrected": bool(h3 != h3_thick and h3_thick),
    }

    check = {
        "H1_expectation": "Q3 에서 median(P_ALL) > median(P4)",
        "H1_held": h1, "H1_measured": delta,
        "H2_expectation": f"차이가 검출문턱 {GATE_DETECTION_THRESHOLD} 초과",
        "H2_held": h2,
        "H3_expectation": "Q3 세 달 전부에서 우세",
        "H3_held": h3, "H3_measured": monthly_delta,
        "H4_expectation": "P_ALL 봉투 커버리지 > P4 봉투 커버리지",
        "H4_held": h4,
        "spec_defect": defect,
        "promotes_candidate": False,
        "justifies_q2_q4_backfill": backfill,
        "verdict": (
            "POOL_WIDTH_MATERIAL_BACKFILL_JUSTIFIED" if backfill
            else "POOL_WIDTH_IMMATERIAL_C24_CONFIRMED"
        ),
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE, "reopens": REOPENS,
        "fold": FOLD, "gate_modified": False, "promotes": False,
        "evaluation_surface_note": "Q3 3 개월 전용. 동결 게이트(9 개월)를 대신하지 않는다",
        "pool": {
            "files_found": len(paths),
            "usable": len(available),
            "dropped": dropped,
            "fold_coverage_census": {"Q3_only_pool": len(available), "full_fold": len(p12)},
        },
        "sets": {k: v for k, v in results.items()},
        "single_model_quality": {
            "n": len(singles),
            "best": singles[:5],
            "worst": singles[-5:],
            "median_total": float(np.median(totals)),
            "frac_below_p4_ensemble": float((totals < a["total"]).mean()),
        },
        "comparison": {
            "delta_total_all_vs_p4": delta,
            "monthly_delta": monthly_delta,
            "detection_threshold": GATE_DETECTION_THRESHOLD,
        },
        "spec_defect": defect,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 26 — 멤버 풀 폭의 Q3 전용 선별",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 재검 대상: `{REOPENS}` — 사이클 24 는 3-fold 전부 가진 12 개 안에서 판정했다",
        f"- 평가 표면: **{FOLD} 전용 (3 개월)**. 동결 게이트를 대신하지 않으며 "
        "**이 노드는 승격하지 않는다**",
        f"- 풀: 파일 {len(paths)} 개 -> 사용 가능 {len(available)} 개 "
        f"(제외 {len(dropped)} 개, 성능과 무관한 유효성 사유만)",
        "",
        "## 1. 멤버 집합별 Q3 성적",
        "",
        "집합은 **성능과 무관한 규칙**으로만 만들었다. `P_ALL_Q3` 는 나쁜 멤버도 전부 포함한다.",
        "",
        "| 집합 | 멤버 | Total | 1-NMAE | FICR | 봉투 커버리지 | 봉투폭 중앙값 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| `{name}` | {r['n_members']} | **{r['total']:.6f}** | "
            f"{r['one_minus_nmae']:.6f} | {r['ficr']:.6f} | "
            f"{r['coverage_eligible']:.4f} | {r['median_envelope_width_cap']:.4f} |"
        )

    lines += [
        "",
        "## 2. 월별 대조 (H3)",
        "",
        "| 월 | P4 | P_ALL | 차이 |",
        "|---|---:|---:|---:|",
    ]
    for m in months:
        lines.append(
            f"| {m} | {a['monthly_total'][m]:.6f} | {b['monthly_total'][m]:.6f} | "
            f"**{monthly_delta[m]:+.6f}** |"
        )

    sq = payload["single_model_quality"]
    lines += [
        "",
        "## 3. 개별 멤버 품질 (풀에 나쁜 모델이 얼마나 있는가)",
        "",
        f"- 단일 모델 Q3 Total 중앙값 **{sq['median_total']:.6f}**",
        f"- P4 앙상블({a['total']:.6f})보다 낮은 단일 모델 비율 "
        f"**{sq['frac_below_p4_ensemble']:.1%}**",
        "",
        "| 최상위 5 | Total | | 최하위 5 | Total |",
        "|---|---:|---|---|---:|",
    ]
    for good, bad in zip(sq["best"], sq["worst"], strict=True):
        lines.append(
            f"| `{good['model']}` | {good['total']:.6f} | | `{bad['model']}` | "
            f"{bad['total']:.6f} |"
        )

    lines += [
        "",
        "## 3b. 사양 결함 기록",
        "",
        "H3 에 최소 월 크기를 명시하지 않았는데 fold 경계가 얇은 달로 흘러넘쳤다.",
        "**사전확약은 완화하지 않는다.** 다만 판정이 그 결함에 좌우되는지는 밝혀야 한다.",
        "",
        "| 월 | 행 | P_ALL - P4 |",
        "|---|---:|---:|",
    ]
    for m in months:
        thin = "" if month_rows.get(m, 0) >= MIN_MONTH_ROWS else " **(얇음)**"
        lines.append(
            f"| {m}{thin} | {int(month_rows.get(m, 0)):,} | {monthly_delta[m]:+.6f} |"
        )
    lines += [
        "",
        f"얇은 달({', '.join(defect['thin_months_excluded']) or '없음'})을 뺀 H3 = "
        f"**{h3_thick}**. 판정이 결함에 좌우되는가: "
        f"**{defect['verdict_changes_if_corrected']}**",
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** (차이 {delta:+.6f})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}** "
        f"({a['coverage_eligible']:.4f} -> {b['coverage_eligible']:.4f})",
        "",
        f"판정: **{check['verdict']}**",
        "",
        f"Q2/Q4 백필 정당화: **{backfill}**",
        "",
        "## 5. 이 노드가 하지 않은 것",
        "",
        "Q3 3 개월 결과다. 동결 게이트는 9 개월을 요구하므로 **여기서 아무것도 승격되지",
        "않는다.** Q2/Q4 에 예측이 없는 모델을 백필해 3-fold pooled 평가를 회복한 뒤에야",
        "게이트에 올릴 수 있다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE26_POOL_SCREEN",
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

    print(f"[C26] 풀 {len(paths)} 파일 -> 사용가능 {len(available)} (제외 {len(dropped)})")
    for name, r in results.items():
        print(f"[C26] {name:>24} k={r['n_members']:>3}  Q3 Total {r['total']:.6f}  "
              f"커버리지 {r['coverage_eligible']:.4f}")
    print(f"[C26] P_ALL - P4 = {delta:+.6f}  월별 "
          + " ".join(f"{m[-2:]}={v:+.5f}" for m, v in monthly_delta.items()))
    print(f"[C26] H1 {h1} | H2 {h2} | H3 {h3} | H4 {h4}")
    print(f"[C26] 판정: {check['verdict']}  백필 정당화 {backfill}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
