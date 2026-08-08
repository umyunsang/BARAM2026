"""M271 P4 사이클 1 — 최대 결손 셀의 기전 진단.

C15 가 뿌린 상위 결손 셀 `group_id=3 | 2023-11 | y in (0.7, 1.1]` (손실 0.01211) 을 표적으로
리서치 사이클 5 단계를 돈다.

역설: y > 0.7 은 정격 근처라 파워커브가 평탄하다(A5 포화비 0.990~0.998, m270 기울기
0.0530/(m/s) 로 급경사 구간의 1/3). 평탄하면 풍속 오차가 출력 오차로 증폭되지 않으므로
쉬워야 하는데 평균 정산단위가 1.923/4 이고 관측 평균오차는 0.1245 로 밴드 반폭 0.06 의
2 배다.

방향 리서치(①)가 낸 두 기전:
  M1 정격 구간의 출력제한·가용성 — 개별 터빈 curtailment 가 집합 파워커브 정격 구간의
     분산을 키운다. 예보로는 알 수 없으므로 회수 불가 후보.
  M2 공간 이질성 — 복잡지형의 흐름·전단·후류로 터빈별 풍속과 파워커브가 달라진다.
     A3 가 그룹3 만 두 단지에 걸쳐 있고 터빈간 거리가 가장 넓음을 이미 측정했다.
     피처화 가능하므로 회수 가능 후보.

이 노드는 **둘 중 무엇이 얼마나 설명하는지** 를 SCADA 로 직접 잰다. 모델을 적합하지 않는다.

사전확약(실행 전 동결):
  기대 부호 +1 — 고발전 구간에서 터빈 가용성 편차와 출력 이질성이 |예측오차| 와 양의
  관계를 가질 것으로 예상한다. 부호가 반대면 C5(anomaly) 로 라우팅된다.
  중단 조건 — 두 기전의 설명력이 각각 산출되면 종료.

읽기 전용. 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m271_excavation_graph as xg
import m271_materialize as mat
import m271_research as rs
from m271_deficit import DeficitLedger
from m271_n0_common import load_tables
from m271_n0_deficit_init import annotate, load_deployed
from m271_n0_scada import scada_long

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle1_toprate.md"
RECEIPT = REPORTS / "m271_cycle1_toprate_receipt.json"

NODE_ID = "C1N1_TOPRATE_MECHANISM"
LANE = "L2"
TARGET_BAND = "(0.7, 1.1]"
PREDECLARED_SIGN = 1
RATED_KWH_10MIN = {"VESTAS": 600.0, "UNISON": 700.0}
PRODUCING_FRACTION = 0.05  # 정격의 5% 미만이면 사실상 미생산으로 본다


# ---------------------------------------------------------------- ① 방향 (기록)
SOURCES = [
    rs.Source(
        source_id="src::curtailment_rated_variance",
        lane=LANE,
        title="Equivalent power curve / aggregate farm curve — rated-region variance",
        origin="ScienceDirect / ResearchGate (aggregate power curve literature)",
        source_class="peer_reviewed",
        applicability="near_match_only",
        claim=(
            "개별 터빈의 curtailment 가 집합 파워커브의 **정격 구간** 분산을 키운다."
        ),
        conditions_differ=(
            "문헌은 대형 단지 대상. 우리는 그룹당 5~6 기이므로 한 기가 멈추면 집합 출력이 "
            "16~20% 떨어져 영향이 훨씬 크다."
        ),
    ),
    rs.Source(
        source_id="src::spatial_heterogeneity",
        lane=LANE,
        title="The impact of wind field spatial heterogeneity and variability on short-term "
        "wind power forecast errors",
        origin="AIP J. Renewable and Sustainable Energy 11(3) 033304",
        source_class="peer_reviewed",
        applicability="near_match_only",
        claim=(
            "복잡지형의 흐름·전단·후류로 터빈 위치별 풍속과 파워커브가 달라지며, 그 공간 "
            "이질성이 예측오차를 만든다."
        ),
        conditions_differ=(
            "문헌은 넓은 단지 대상. 우리 단지는 2 km 안이지만 869~1001 m 산악 능선이라 "
            "고도 편차가 130 m 다."
        ),
    ),
]

DIRECTION = rs.Direction(
    direction_id="dir::C15::explain::L2::toprate",
    lane=LANE,
    mechanism="정격 근처 집합 출력 분산 = 터빈 가용성 + 공간 이질성",
    why_this_deficit=(
        "파워커브가 평탄한 구간에서 밴드 2 배 오차가 나는 역설은 풍속 예보 오차로 설명되지 "
        "않는다. 집합 수준에서만 생기는 기전이어야 한다."
    ),
    deficit_cell=None,  # 실행 시 주입
    applicability="near_match_only",
    source_ids=[s.source_id for s in SOURCES],
)

SPEC = rs.Specification(
    spec_id="spec::toprate_mechanism",
    direction_id=DIRECTION.direction_id,
    lane=LANE,
    method_name="aggregate_variance_attribution",
    settings={
        "band": TARGET_BAND,
        "producing_threshold_of_rated": PRODUCING_FRACTION,
        "heterogeneity_metric": "cv_across_turbines",
    },
    required_inputs=["scada_vestas_train", "scada_unison_train", "train_labels", "predictions"],
    within_official_rules=True,  # 전부 공급 데이터. SCADA 는 진단 전용.
    reported_performance="문헌은 정성적 기전만 제시. 정량 벤치마크 없음.",
    our_conditions_differ="그룹당 5~6 기 소규모, 2 km 산악 능선, 운전로그 없음",
    expected_effect_range="미상 — 이 노드가 설명력을 처음 측정한다",
    source_ids=[s.source_id for s in SOURCES],
)


# ---------------------------------------------------------------- ④ 실험
def turbine_hourly(tables: Any) -> pd.DataFrame:
    """터빈별 시간 집계. 가용성과 이질성을 재기 위한 표면."""
    scada = scada_long(tables.scada_vestas, tables.scada_unison)
    key = tables.turbines.loc[:, ["manufacturer", "turbine_number", "group_id"]]
    joined = scada.merge(key, on=["manufacturer", "turbine_number"], how="inner")
    joined["hour"] = joined["kst_dtm"].dt.floor("h")
    joined["rated"] = joined["manufacturer"].map(RATED_KWH_10MIN)
    joined["load_factor"] = joined["power_kw"] / joined["rated"]
    return joined


def group_hour_features(turbines: pd.DataFrame) -> pd.DataFrame:
    """시간·그룹별 가용성과 이질성."""
    turbines = turbines.assign(
        producing=(turbines["load_factor"] >= PRODUCING_FRACTION).astype(float)
    )
    agg = turbines.groupby(["group_id", "hour"]).agg(
        turbines_seen=("load_factor", "size"),
        producing_fraction=("producing", "mean"),
        mean_load=("load_factor", "mean"),
        std_load=("load_factor", "std"),
        min_load=("load_factor", "min"),
        max_load=("load_factor", "max"),
    )
    agg = agg.reset_index()
    # 이질성: 터빈간 출력 변동계수. 평균이 0 에 가까우면 정의되지 않으므로 하한을 둔다.
    agg["cv_load"] = agg["std_load"] / agg["mean_load"].clip(lower=0.05)
    agg["load_spread"] = agg["max_load"] - agg["min_load"]
    return agg


def build_surface(tables: Any) -> pd.DataFrame:
    deployed = annotate(load_deployed())
    features = group_hour_features(turbine_hourly(tables))
    merged = deployed.merge(
        features, left_on=["group_id", "forecast_kst_dtm"], right_on=["group_id", "hour"],
        how="inner",
    )
    return merged.loc[merged["y_band"] == TARGET_BAND].reset_index(drop=True)


def _partial(y: pd.Series, x: pd.Series, control: pd.Series) -> float:
    """통제변수에 대해 2 차 다항 잔차화한 뒤 상관."""
    cv = control.to_numpy(dtype=float)
    ry = y.to_numpy(dtype=float) - np.polyval(np.polyfit(cv, y.to_numpy(dtype=float), 2), cv)
    rx = x.to_numpy(dtype=float) - np.polyval(np.polyfit(cv, x.to_numpy(dtype=float), 2), cv)
    return float(np.corrcoef(ry, rx)[0, 1])


def measure(surface: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"band": TARGET_BAND, "rows": len(surface), "by_group": []}
    for group, part in surface.groupby("group_id"):
        if len(part) < 100:
            continue
        se = float(1.0 / np.sqrt(max(len(part) - 3, 1)))
        entry = {
            "group": int(group),
            "rows": len(part),
            "approx_se": se,
            "mean_abs_err_rate": float(part["abs_err_rate"].mean()),
            "mean_producing_fraction": float(part["producing_fraction"].mean()),
            "fraction_of_hours_with_an_idle_turbine": float(
                (part["producing_fraction"] < 1.0).mean()
            ),
            "mean_cv_load": float(part["cv_load"].mean()),
            "mean_load_spread": float(part["load_spread"].mean()),
            # 발전량 수준(y)을 통제한 편상관. 둘 다 y 에 딸려 움직이는 교란을 제거한다.
            "partial_corr_availability": _partial(
                part["abs_err_rate"], part["producing_fraction"], part["y"]
            ),
            "partial_corr_heterogeneity": _partial(
                part["abs_err_rate"], part["cv_load"], part["y"]
            ),
            "partial_corr_spread": _partial(
                part["abs_err_rate"], part["load_spread"], part["y"]
            ),
        }
        entry["availability_sigma_multiple"] = abs(entry["partial_corr_availability"]) / se
        entry["heterogeneity_sigma_multiple"] = abs(entry["partial_corr_heterogeneity"]) / se
        out["by_group"].append(entry)

    # 사전확약 판정: 가용성은 음의 관계(생산 터빈이 많을수록 오차 작음), 이질성은 양의 관계.
    avail = [e["partial_corr_availability"] for e in out["by_group"]]
    hetero = [e["partial_corr_heterogeneity"] for e in out["by_group"]]
    out["predeclared_check"] = {
        "expectation": "가용성 편차와 출력 이질성이 |오차| 와 양의 관계",
        "availability_partials": avail,
        "heterogeneity_partials": hetero,
        # 가용성은 '생산 비율' 이므로 오차와 **음** 의 관계가 기대 방향이다.
        "availability_sign_as_expected": all(a < 0 for a in avail),
        "heterogeneity_sign_as_expected": all(h > 0 for h in hetero),
    }
    return out


def main() -> int:
    ledger = DeficitLedger.from_a7()
    top = ledger.top(1)[0]
    DIRECTION.deficit_cell = top["key"]

    # ③ MATERIALIZE — 발굴 그래프에 새 노드를 만들고 계보와 사전확약을 건다.
    graph = xg.ExcavationGraph()
    graph.register_premise(xg.Premise("P", "부트스트랩", lambda s: True))
    graph.add_node("A4_error", node_type="MEASURE", lane=LANE, status=xg.LIVE)
    node_id, predeclaration = mat.materialize(
        graph, SPEC,
        node_id=NODE_ID, parent_node="A4_error", deficit_cell=top["key"],
        direction_id=DIRECTION.direction_id,
        source_ids=[s.source_id for s in SOURCES],
        expected_sign=PREDECLARED_SIGN,
        expected_effect=0.0,  # 진단 노드. 점수를 직접 올리지 않는다.
        gate_version="DIAGNOSTIC_NO_GATE",
        stop_condition="두 기전의 설명력이 각각 산출되면 종료",
        parent_candidate="T0.5_G1.5",
    )
    lineage = mat.lineage_of(graph, node_id)

    # ④ 1회 실행
    tables, input_hashes = load_tables()
    surface = build_surface(tables)
    result = measure(surface)

    check = result["predeclared_check"]
    payload: dict[str, Any] = {
        "target_cell": {k: top[k] for k in ("key", "loss_share", "gen_weight", "mean_unit")},
        "direction": {
            "direction_id": DIRECTION.direction_id,
            "mechanism": DIRECTION.mechanism,
            "applicability": DIRECTION.applicability,
            "sources": [
                {"id": s.source_id, "claim": s.claim, "conditions_differ": s.conditions_differ,
                 "applicability": s.applicability}
                for s in SOURCES
            ],
        },
        "spec": {
            "spec_id": SPEC.spec_id, "method": SPEC.method_name, "settings": SPEC.settings,
            "within_official_rules": SPEC.within_official_rules,
        },
        "predeclaration": {
            "node_id": predeclaration.node_id,
            "expected_sign": predeclaration.expected_sign,
            "digest": predeclaration.digest(),
        },
        "lineage": lineage,
        "measurement": result,
        "input_hashes": input_hashes,
    }

    lines = [
        "# M271 P4 사이클 1 — 최대 결손 셀의 기전 진단",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 표적 셀: `{top['key']}`",
        f"- 손실 {top['loss_share']:.5f} / 발전비중 {top['gen_weight']:.1%} / 평균단위 "
        f"{top['mean_unit']:.3f}",
        f"- 노드: `{node_id}` / 사전확약 해시 `{predeclaration.digest()[:16]}`",
        "",
        "## 0. 역설",
        "",
        "y > 0.7 은 정격 근처라 파워커브가 평탄하다(A5 포화비 0.990~0.998, 그 구간 기울기는",
        "급경사 구간의 1/3). 평탄하면 풍속 오차가 출력 오차로 증폭되지 않으므로 **쉬워야**",
        "한다. 그런데 평균 정산단위가 1.923/4 이고 관측 평균오차는 밴드 반폭의 2 배다.",
        "",
        "풍속 예보 오차로 설명되지 않으므로 집합 수준에서만 생기는 기전이어야 한다.",
        "",
        "## 1. ① 방향 리서치",
        "",
        f"기전: **{DIRECTION.mechanism}**",
        "",
        "| 소스 | 주장 | 우리 조건과의 차이 | 태그 |",
        "|---|---|---|---|",
    ]
    for s in SOURCES:
        lines.append(
            f"| `{s.source_id}` | {s.claim} | {s.conditions_differ} | `{s.applicability}` |"
        )

    lines += [
        "",
        "`directly_supported` 가 없다. 문헌은 대형 단지 대상이고 우리는 그룹당 5~6 기다.",
        "그래서 문헌 수치를 요구사항으로 쓰지 않고 **기전만** 가져와 직접 측정한다.",
        "",
        "## 2. ③ 계보",
        "",
        f"- `derived_from`: {lineage['derived_from']}",
        f"- `addresses`: {lineage['addresses']}",
        f"- `instantiates`: {lineage['instantiates']}",
        "",
        "## 3. ④ 측정 결과",
        "",
        f"대역 `{TARGET_BAND}` 의 결합 행 **{result['rows']:,}**",
        "",
        "| 그룹 | 행수 | 평균\\|오차\\| | 생산비율 | 유휴터빈 있는 시간 | 이질성 CV | "
        "가용성 편상관 | 이질성 편상관 | SE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for e in result["by_group"]:
        lines.append(
            f"| {e['group']} | {e['rows']:,} | {e['mean_abs_err_rate']:.4f} | "
            f"{e['mean_producing_fraction']:.3f} | "
            f"{e['fraction_of_hours_with_an_idle_turbine']:.1%} | {e['mean_cv_load']:.3f} | "
            f"**{e['partial_corr_availability']:+.4f}** "
            f"({e['availability_sigma_multiple']:.1f}x) | "
            f"**{e['partial_corr_heterogeneity']:+.4f}** "
            f"({e['heterogeneity_sigma_multiple']:.1f}x) | {e['approx_se']:.4f} |"
        )

    lines += [
        "",
        "편상관은 발전량 수준 `y` 에 대해 2 차 다항 잔차화한 뒤 계산했다. 가용성·이질성과",
        "오차가 둘 다 y 에 딸려 움직이는 교란을 제거하기 위함이다.",
        "",
        "## 4. ⑤ 사전확약 대조",
        "",
        f"동결된 기대: *{check['expectation']}. 부호가 반대면 C5(anomaly) 로 라우팅된다.*",
        "",
        "생산비율은 '생산 중인 터빈의 비율' 이므로 오차와 **음** 의 관계가 기대 방향이다.",
        "",
        f"- 가용성 부호 기대대로: **{check['availability_sign_as_expected']}** "
        f"(`{[round(a, 4) for a in check['availability_partials']]}`)",
        f"- 이질성 부호 기대대로: **{check['heterogeneity_sign_as_expected']}** "
        f"(`{[round(h, 4) for h in check['heterogeneity_partials']]}`)",
        "",
        "## 5. 읽는 법",
        "",
        "가용성이 오차를 설명하면 그 부분은 **회수 불가**다 — 예보 시점에 어느 터빈이 멈출지",
        "알 수 없고 평가기간 SCADA 도 없다. 이질성이 설명하면 **회수 가능** 후보다 —",
        "공간 구조는 격자·지형·풍향에서 피처로 만들 수 있다.",
        "",
        "이 노드는 설명력만 잰다. 회수 실험은 별도 노드이며 여기서 하지 않는다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE1_TOPRATE",
        "node": NODE_ID,
        "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False,
        "external_actions": ["web search (read-only, direction research)"],
        "model_fits": 0,
        "lockbox_reopened": False,
        "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[C1] 표적 셀 {top['key']} (손실 {top['loss_share']:.5f})")
    print(f"[C1] 대역 {TARGET_BAND} 결합 행 {result['rows']:,}")
    for e in result["by_group"]:
        idle = e["fraction_of_hours_with_an_idle_turbine"]
        print(
            f"[C1] g{e['group']} n={e['rows']:,} 유휴시간={idle:.1%} "
            f"가용성r={e['partial_corr_availability']:+.4f}"
            f"({e['availability_sigma_multiple']:.1f}x) "
            f"이질성r={e['partial_corr_heterogeneity']:+.4f}"
            f"({e['heterogeneity_sigma_multiple']:.1f}x)"
        )
    print(f"[C1] 사전확약 가용성={check['availability_sign_as_expected']} "
          f"이질성={check['heterogeneity_sign_as_expected']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
