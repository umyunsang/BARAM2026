"""M271 후보 평가 하네스 — 어떤 후보든 한 번에 판정한다.

12 사이클의 진단을 재사용 가능한 도구로 바꾼다. 남은 유일한 경로가 모델 작업인데, 그 작업의
**검증 비용**을 낮추는 것이 이 파일의 목적이다.

후보 하나를 받아 다음을 한 번에 낸다.

  1. 공식 산식 점수 (pooled, fold 별, 그룹별)
  2. **동결 월별 게이트** 판정 (`m270_gate.py`, 재동결 금지 — 읽기만)
  3. 게이트 검출력 대비 위치 — 사이클 9 가 잰 문턱 `+0.001013`
  4. **결손 원장 차분** — 어느 셀이 나아지고 어느 셀이 나빠졌는가, 회수가능질량 기준
  5. 사이클 1~12 가 닫은 축과의 대조 — 이 후보가 이미 닫힌 축에 기대고 있는가

마지막 항목이 중요하다. 이 프로젝트의 실패 모드는 같은 축을 반복해 파는 것이었다. 후보가
어떤 폐기 전제에 기대고 있으면 그 전제가 뒤집혔는지부터 확인해야 한다.

사용:
    python m271_evaluate_candidate.py <정책명>
    python m271_evaluate_candidate.py --parquet <예측 parquet 경로>

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import argparse
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
from m271_n0_deficit_init import Y_BAND_EDGES, build_ledger

from baram.constants import CAPACITIES_KWH, METRIC_COLUMNS
from baram.evaluation.official import evaluate_official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

DEPLOYED = "T0.5_G1.5"
TARGET = 0.66
GATE_DETECTION_THRESHOLD = 0.001013  # 사이클 9 실측
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")

# 사이클 1~12 가 닫은 축. 후보가 여기 기대면 전제부터 확인해야 한다.
CLOSED_AXES = {
    "decision_policy": "오라클 상한 8.8%, 동결 게이트 0/62 (사이클 4·7)",
    "policy_selection_conditioning": "일반화 조건화 오라클 16.5% (사이클 8)",
    "wind_sector": "240도 초과비율 0.968 — 손실이 발전량에 비례 (사이클 3)",
    "gfs_spatial_interpolation": "nearest 가 IDW 보다 나쁨 3/3 (사이클 5)",
    "wind50_midpoint_feature": "중점이 10m·max 보다 상관 낮음 3/3 (사이클 6)",
    "unused_nwp_columns": "진짜미사용 평균 MI 가 선언 컬럼의 0.41배 (A2)",
    "turbine_availability": "기전 확인 15~23 sigma 이나 예보시점 미지 (사이클 2)",
    "coarser_representation": "v2 7분위가 46-bin 대비 FICR -0.0139 (사이클 11)",
    "external_nwp_blend": "GFS+LDAPS 실측 감소 0.10~0.37%, 국소성 1.1 sigma (사이클 10)",
    "g3_training_history": "이력 3->9개월에 열세 무감소 (사이클 12)",
}


def load_candidate(policy: str | None, parquet: Path | None) -> pd.DataFrame:
    if policy:
        return load_predictions(policy)
    frame = pd.read_parquet(parquet)
    required = {"forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "prediction_kwh"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"candidate parquet is missing columns: {sorted(missing)}")
    out = frame.loc[:, sorted(required)].copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
    return out


def official(frame: pd.DataFrame) -> dict[str, Any]:
    metric = frame.loc[:, sorted(METRIC_COLUMNS)]
    r = evaluate_official(metric, CAPACITIES_KWH)
    return {
        "total": r.total,
        "one_minus_nmae": r.one_minus_nmae,
        "ficr": r.ficr,
        "group_ficr": {int(k): float(v) for k, v in r.group_ficr.items()},
        "group_nmae": {int(k): float(v) for k, v in r.group_nmae.items()},
    }


def ledger_diff(candidate: pd.DataFrame, parent: pd.DataFrame) -> dict[str, Any]:
    """어느 셀이 나아졌는가. 회수가능질량 기준으로 본다."""
    def cells(frame: pd.DataFrame) -> pd.DataFrame:
        annotated = frame.copy()
        annotated["capacity"] = annotated["group_id"].map(CAPACITIES_KWH).astype(float)
        annotated["y"] = annotated["actual_kwh"] / annotated["capacity"]
        annotated["abs_err_rate"] = (
            (annotated["prediction_kwh"] - annotated["actual_kwh"]).abs()
            / annotated["capacity"]
        )
        from baram.evaluation.official import settlement_unit

        annotated["unit"] = settlement_unit(annotated["abs_err_rate"].to_numpy(float))
        annotated["month"] = annotated["forecast_kst_dtm"].dt.to_period("M").astype(str)
        annotated["y_band"] = pd.cut(
            annotated["y"], bins=list(Y_BAND_EDGES), right=True
        ).astype(str)
        eligible = annotated.loc[
            annotated["actual_kwh"] >= 0.10 * annotated["capacity"]
        ].reset_index(drop=True)
        table, _ = build_ledger(eligible, ["month", "y_band"])
        table["key"] = (
            "group_id=" + table["group_id"].astype(str)
            + "|month=" + table["month"] + "|y_band=" + table["y_band"]
        )
        return table.set_index("key")

    left, right = cells(candidate), cells(parent)
    shared = left.index.intersection(right.index)
    delta = (right.loc[shared, "total_loss"] - left.loc[shared, "total_loss"]).sort_values(
        ascending=False
    )
    return {
        "cells_compared": len(shared),
        "cells_improved": int((delta > 0).sum()),
        "cells_worsened": int((delta < 0).sum()),
        "loss_removed": float(delta[delta > 0].sum()),
        "loss_added": float(-delta[delta < 0].sum()),
        "net_loss_removed": float(delta.sum()),
        "top_improved": [
            {"cell": k, "loss_removed": float(v)} for k, v in delta.head(5).items()
        ],
        "top_worsened": [
            {"cell": k, "loss_added": float(-v)} for k, v in delta.tail(5).items()
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="M271 후보 평가 하네스")
    parser.add_argument("policy", nargs="?", help="probe parquet 의 정책 컬럼명")
    parser.add_argument("--parquet", type=Path, help="예측 parquet 경로")
    parser.add_argument("--leans-on", nargs="*", default=[],
                        help=f"기대는 축 (닫힌 축: {', '.join(sorted(CLOSED_AXES))})")
    parser.add_argument("--name", default=None, help="후보 이름 (리포트용)")
    args = parser.parse_args()

    if not args.policy and not args.parquet:
        parser.error("정책명 또는 --parquet 중 하나가 필요하다")

    name = args.name or args.policy or args.parquet.stem
    candidate = load_candidate(args.policy, args.parquet)
    parent = load_predictions(DEPLOYED)

    cand_score = official(candidate)
    parent_score = official(parent)
    delta_total = cand_score["total"] - parent_score["total"]

    gate = evaluate_gate(candidate, parent)
    gate_flags = {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()}
    stats = gate.evidence

    diff = ledger_diff(candidate, parent)

    # 이 후보가 닫힌 축에 기대는가.
    leaning = {a: CLOSED_AXES[a] for a in args.leans_on if a in CLOSED_AXES}
    unknown = [a for a in args.leans_on if a not in CLOSED_AXES]

    verdict = (
        "GATE_PASSED"
        if gate.passed
        else ("BELOW_GATE_DETECTION" if abs(delta_total) < GATE_DETECTION_THRESHOLD
              else "GATE_REJECTED")
    )

    payload = {
        "candidate": name,
        "gate_version": GATE_VERSION,
        "gate_modified": False,
        "official": {"candidate": cand_score, "parent": parent_score,
                     "delta_total": delta_total},
        "target": TARGET,
        "gap_remaining": TARGET - cand_score["total"],
        "gate": {
            "passed": bool(gate.passed),
            "flags": gate_flags,
            "positive_months": int(stats["months_scored"] and stats["positive_months"]),
            "months_scored": int(stats["months_scored"]),
            "sign_test_p": float(stats["sign_test_p_greater"]),
            "median_delta": float(stats["median_total_delta"]),
            "bootstrap_q05": float(stats["block_bootstrap_q05"]),
            "min_delta": float(stats["min_total_delta"]),
            "detection_threshold": GATE_DETECTION_THRESHOLD,
            "above_detection_threshold": bool(abs(delta_total) >= GATE_DETECTION_THRESHOLD),
        },
        "ledger_diff": diff,
        "leans_on_closed_axes": leaning,
        "unknown_axes": unknown,
        "verdict": verdict,
    }

    print(f"=== {name} ===")
    print(f"공식 Total {cand_score['total']:.6f}  (배포 {parent_score['total']:.6f}, "
          f"델타 {delta_total:+.6f})")
    print(f"  1-NMAE {cand_score['one_minus_nmae']:.6f}  FICR {cand_score['ficr']:.6f}")
    print(f"  목표 {TARGET} 까지 {payload['gap_remaining']:+.6f}")
    print()
    flags = "".join("O" if gate_flags.get(k) else "-" for k in ("G1", "G2", "G3", "G4"))
    print(f"동결 게이트 [{flags}] {stats['positive_months']}/{stats['months_scored']}월 "
          f"p={stats['sign_test_p_greater']:.4f} q05={stats['block_bootstrap_q05']:+.6f}")
    print(f"  검출 문턱 {GATE_DETECTION_THRESHOLD:+.6f} 대비 "
          f"{'초과' if payload['gate']['above_detection_threshold'] else '미달'}")
    print()
    print(f"결손 원장 차분: 개선 {diff['cells_improved']} / 악화 {diff['cells_worsened']} 셀")
    print(f"  손실 제거 {diff['loss_removed']:.5f} / 추가 {diff['loss_added']:.5f} "
          f"/ 순 {diff['net_loss_removed']:+.5f}")
    for c in diff["top_improved"][:3]:
        print(f"    + {c['cell']}  {c['loss_removed']:+.5f}")
    for c in diff["top_worsened"][:3]:
        print(f"    - {c['cell']}  {-c['loss_added']:+.5f}")
    if leaning:
        print()
        print("!! 이 후보가 기대는 축 중 이미 닫힌 것:")
        for axis, reason in leaning.items():
            print(f"    {axis}: {reason}")
        print("   해당 폐기 전제가 뒤집혔는지 먼저 확인할 것.")
    if unknown:
        print(f"   (미등록 축: {unknown})")
    print()
    print(f"판정: {verdict}")

    out = REPORTS / f"m271_candidate_{name.replace('/', '_')}.json"
    payload["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload["decided_utc"] = datetime.now(UTC).isoformat()
    payload["dacon_upload"] = False
    payload["model_fits"] = 0
    payload["lockbox_reopened"] = False
    payload["new_2024_evaluation"] = False
    out.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"receipt -> {out}")
    return 0 if gate.passed else 1


if __name__ == "__main__":
    sys.exit(main())
