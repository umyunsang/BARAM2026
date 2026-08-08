"""M271 P4 사이클 2 — 부호 역전의 기전 (C5 anomaly).

사이클 1 은 최대 결손 셀에서 두 기전(가용성·공간이질성)을 재려다 **사전확약 부호가 모두
뒤집혔다**. 라우터가 C5(anomaly)로 보냈고, 이 노드는 그 역전의 기전을 찾는다.

방향 리서치(①)가 낸 기전: **collider bias (Berkson's paradox)**.

    nwp_wind  --+
                +--> y (실제 발전량)
    가용성    --+

`y` 는 풍속과 가용성의 **공통 결과** 다. 공통 결과를 통제하면 그 원인들 사이에 없던 연관이
생기며, 전형적으로 가짜 음의 상관이 나타난다. 사이클 1 은 두 번 통제했다.

    1. `y_band == (0.7, 1.1]` 로 대역을 자른 것 자체가 y 조건화
    2. `y` 에 대한 2 차 다항 잔차화

결손 원장의 y 대역 축은 문제없다 — 손실을 쪼개는 **회계**이지 인과 주장이 아니다. 문제는
대역 **안에서** 연관을 재는 순간이며 사이클 1 이 정확히 그것을 했다.

교정: collider 인 `y` 대신 **원인**인 NWP 풍속으로 통제한다.

사전확약(실행 전 동결):
  H1  고정 NWP 풍속에서 생산 터빈 비율이 낮을수록 |오차| 가 크다  -> 편상관 **음수**
  H2  고정 NWP 풍속에서 생산 터빈 비율이 낮을수록 부호오차(예측-실제)가 크다
      (유휴 터빈이 있으면 실제가 예측보다 낮으므로 예측이 초과)      -> 편상관 **음수**
  둘 다 음수면 가용성 기전이 살아난다. 여전히 뒤집히면 기전이 실제로 틀린 것이다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
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

import m271_research as rs
from m271_cycle1_toprate import group_hour_features, turbine_hourly
from m271_deficit import DeficitLedger
from m271_n0_common import load_tables
from m271_n0_deficit_init import annotate, load_deployed

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle2_collider.md"
RECEIPT = REPORTS / "m271_cycle2_collider_receipt.json"

NODE_ID = "C1N2_COLLIDER_CORRECTED"
LANE = "L2"
PREDECLARED_SIGN = -1  # H1/H2 모두 음의 편상관을 기대한다
MIN_ROWS = 200

SOURCES = [
    rs.Source(
        source_id="src::collider_bias",
        lane=LANE,
        title="Collider bias / Berkson's paradox — conditioning on a common effect",
        origin="Nature Communications 11:5749 (2020); Berkson (1946)",
        source_class="peer_reviewed",
        applicability="directly_supported",
        claim=(
            "공통 결과(collider)를 통제하면 그 원인들 사이에 없던 연관이 생기며, 전형적으로 "
            "가짜 음의 상관이 나타난다."
        ),
        conditions_differ=(
            "문헌 예시는 역학·유전이지만 기전은 도메인 무관한 통계적 사실이다. 우리 구조는 "
            "wind -> y <- availability 로 정확히 collider 형태다."
        ),
    ),
]

DIRECTION = rs.Direction(
    direction_id="dir::C5::anomaly::L2::collider",
    lane=LANE,
    mechanism="사이클 1 의 부호 역전은 y 를 통제한 collider bias 다",
    why_this_deficit=(
        "y 는 풍속과 가용성의 공통 결과이므로 통제 대상이 아니라 통제하면 안 되는 변수다. "
        "원인인 NWP 풍속으로 통제하면 기전이 살아나는지 갈린다."
    ),
    deficit_cell=None,
    applicability="directly_supported",
    source_ids=[s.source_id for s in SOURCES],
)


def nwp_wind_hourly(ldaps: pd.DataFrame) -> pd.DataFrame:
    u = ldaps["heightAboveGround_10_10u"].to_numpy(dtype=float)
    v = ldaps["heightAboveGround_10_10v"].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {"forecast_kst_dtm": ldaps["forecast_kst_dtm"], "nwp_ws10": np.hypot(u, v)}
    )
    return frame.groupby("forecast_kst_dtm", as_index=False).mean()


def _partial(y: pd.Series, x: pd.Series, control: pd.Series) -> float:
    cv = control.to_numpy(dtype=float)
    ry = y.to_numpy(dtype=float) - np.polyval(np.polyfit(cv, y.to_numpy(dtype=float), 2), cv)
    rx = x.to_numpy(dtype=float) - np.polyval(np.polyfit(cv, x.to_numpy(dtype=float), 2), cv)
    return float(np.corrcoef(ry, rx)[0, 1])


def build_surface(tables: Any) -> pd.DataFrame:
    deployed = annotate(load_deployed())
    deployed["signed_err_rate"] = (
        deployed["prediction_kwh"] - deployed["actual_kwh"]
    ) / deployed["capacity"]
    features = group_hour_features(turbine_hourly(tables))
    wind = nwp_wind_hourly(tables.ldaps_train)
    merged = deployed.merge(
        features, left_on=["group_id", "forecast_kst_dtm"], right_on=["group_id", "hour"],
        how="inner",
    )
    return merged.merge(wind, on="forecast_kst_dtm", how="inner").reset_index(drop=True)


def measure(surface: pd.DataFrame) -> dict[str, Any]:
    """collider 인 y 대신 원인인 NWP 풍속으로 통제해 재측정한다.

    **대역을 자르지 않는다.** 대역 자르기 자체가 y 조건화이기 때문이다. 대신 전 구간을 쓰고
    풍속으로 통제한다. 비교를 위해 사이클 1 의 y 통제 값도 함께 낸다.
    """
    out: dict[str, Any] = {"rows": len(surface), "by_group": []}
    for group, part in surface.groupby("group_id"):
        if len(part) < MIN_ROWS:
            continue
        se = float(1.0 / np.sqrt(max(len(part) - 3, 1)))
        entry = {
            "group": int(group),
            "rows": len(part),
            "approx_se": se,
            "mean_producing_fraction": float(part["producing_fraction"].mean()),
            "idle_hour_fraction": float((part["producing_fraction"] < 1.0).mean()),
            # 교정: 원인(NWP 풍속)으로 통제
            "H1_abs_err_given_wind": _partial(
                part["abs_err_rate"], part["producing_fraction"], part["nwp_ws10"]
            ),
            "H2_signed_err_given_wind": _partial(
                part["signed_err_rate"], part["producing_fraction"], part["nwp_ws10"]
            ),
            # 대조: collider(y)로 통제한 사이클 1 방식
            "collider_abs_err_given_y": _partial(
                part["abs_err_rate"], part["producing_fraction"], part["y"]
            ),
            "collider_signed_err_given_y": _partial(
                part["signed_err_rate"], part["producing_fraction"], part["y"]
            ),
        }
        entry["H1_sigma"] = abs(entry["H1_abs_err_given_wind"]) / se
        entry["H2_sigma"] = abs(entry["H2_signed_err_given_wind"]) / se
        out["by_group"].append(entry)

    h1 = [e["H1_abs_err_given_wind"] for e in out["by_group"]]
    h2 = [e["H2_signed_err_given_wind"] for e in out["by_group"]]
    collider_h1 = [e["collider_abs_err_given_y"] for e in out["by_group"]]
    out["predeclared_check"] = {
        "H1_expectation": "고정 NWP 풍속에서 생산비율과 |오차| 는 음의 관계",
        "H2_expectation": "고정 NWP 풍속에서 생산비율과 부호오차는 음의 관계",
        "H1_partials": h1,
        "H2_partials": h2,
        "H1_all_negative": all(v < 0 for v in h1),
        "H2_all_negative": all(v < 0 for v in h2),
        "collider_comparison_h1": collider_h1,
        "sign_flipped_by_control": [
            bool(np.sign(a) != np.sign(b)) for a, b in zip(h1, collider_h1, strict=True)
        ],
        "verdict": (
            "AVAILABILITY_MECHANISM_SURVIVES"
            if all(v < 0 for v in h1) and all(v < 0 for v in h2)
            else "MECHANISM_STILL_REJECTED"
        ),
    }
    return out


def main() -> int:
    ledger = DeficitLedger.from_a7()
    DIRECTION.deficit_cell = ledger.top(1)[0]["key"]

    tables, input_hashes = load_tables()
    surface = build_surface(tables)
    result = measure(surface)
    check = result["predeclared_check"]

    payload = {
        "direction": {
            "direction_id": DIRECTION.direction_id,
            "mechanism": DIRECTION.mechanism,
            "applicability": DIRECTION.applicability,
            "sources": [
                {"id": s.source_id, "claim": s.claim, "applicability": s.applicability,
                 "origin": s.origin}
                for s in SOURCES
            ],
        },
        "predeclaration": {
            "H1": "partial(|err|, producing | nwp_ws10) < 0",
            "H2": "partial(signed_err, producing | nwp_ws10) < 0",
            "expected_sign": PREDECLARED_SIGN,
        },
        "measurement": result,
        "input_hashes": input_hashes,
    }

    lines = [
        "# M271 P4 사이클 2 — 부호 역전의 기전 (collider bias)",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `C1N1_TOPRATE_MECHANISM`",
        "- 라우팅 근거: C5(anomaly) — 사이클 1 의 사전확약 부호가 뒤집혔다",
        "",
        "## 1. ① 방향 리서치",
        "",
        f"기전: **{DIRECTION.mechanism}**  (적용성 `{DIRECTION.applicability}`)",
        "",
        "```",
        "nwp_wind  --+",
        "            +--> y (실제 발전량)      y 는 두 원인의 **공통 결과** = collider",
        "가용성    --+",
        "```",
        "",
        f"{SOURCES[0].claim}",
        "",
        f"출처: {SOURCES[0].origin}",
        "",
        "사이클 1 은 y 를 **두 번** 통제했다. 대역 자르기(`y_band`)와 다항 잔차화. 둘 다",
        "collider 조건화다. 결손 원장의 y 대역 축은 손실 **회계**이므로 문제없지만, 대역",
        "**안에서** 연관을 재는 순간 편향이 들어온다.",
        "",
        "## 2. ④ 교정 측정",
        "",
        "collider 인 `y` 대신 원인인 NWP 풍속으로 통제하고, 대역을 자르지 않는다.",
        "",
        "| 그룹 | 행수 | 유휴시간 | **H1** \\|오차\\|\\|풍속 | **H2** 부호오차\\|풍속 | "
        "(대조) \\|오차\\|\\|y | 부호 뒤집힘 |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for e, flipped in zip(
        result["by_group"], check["sign_flipped_by_control"], strict=True
    ):
        lines.append(
            f"| {e['group']} | {e['rows']:,} | {e['idle_hour_fraction']:.1%} | "
            f"**{e['H1_abs_err_given_wind']:+.4f}** ({e['H1_sigma']:.1f}x) | "
            f"**{e['H2_signed_err_given_wind']:+.4f}** ({e['H2_sigma']:.1f}x) | "
            f"{e['collider_abs_err_given_y']:+.4f} | {'예' if flipped else '아니오'} |"
        )

    lines += [
        "",
        "## 3. ⑤ 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> 전 그룹 음수: **{check['H1_all_negative']}** "
        f"(`{[round(v, 4) for v in check['H1_partials']]}`)",
        f"- H2 `{check['H2_expectation']}` -> 전 그룹 음수: **{check['H2_all_negative']}** "
        f"(`{[round(v, 4) for v in check['H2_partials']]}`)",
        "",
        f"판정: **{check['verdict']}**",
        "",
        "## 4. 읽는 법",
        "",
        "H1·H2 가 모두 음수면 가용성 기전이 살아나고, 사이클 1 의 역전은 통제변수 선택의",
        "산물이었다는 뜻이다. 여전히 뒤집히면 기전이 실제로 틀린 것이다.",
        "",
        "어느 쪽이든 **가용성은 예보 시점에 알 수 없다** — 평가기간 SCADA 가 없다. 기전이",
        "살아나도 그 부분은 회수 불가로 분류되며, 회수 가능한 것은 '평균적으로 얼마나 자주",
        "유휴가 발생하는가' 같은 정적 통계뿐이다. 이 노드는 그 판단을 하지 않는다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE2_COLLIDER",
        "node": NODE_ID,
        "lane": LANE,
        "parent_node": "C1N1_TOPRATE_MECHANISM",
        "routed_by": "C5_anomaly",
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False,
        "external_actions": ["web search (read-only, anomaly research)"],
        "model_fits": 0,
        "lockbox_reopened": False,
        "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[C2] 결합 행 {result['rows']:,} (대역 자르지 않음)")
    for e in result["by_group"]:
        print(
            f"[C2] g{e['group']} H1={e['H1_abs_err_given_wind']:+.4f}({e['H1_sigma']:.1f}x) "
            f"H2={e['H2_signed_err_given_wind']:+.4f}({e['H2_sigma']:.1f}x) "
            f"| 대조(y통제)={e['collider_abs_err_given_y']:+.4f}"
        )
    print(f"[C2] H1 전부음수={check['H1_all_negative']} H2 전부음수={check['H2_all_negative']}")
    print(f"[C2] 판정: {check['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
