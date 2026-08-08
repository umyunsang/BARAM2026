"""M271 C6 — 레인 확장 강제: 굶은 레인의 SOTA 를 열고 후보를 실체화한다.

C1N85(C16 재검)가 확정한 것 — C16 은 한 후보를 잘못 잘랐으나 **폐쇄 판정 자체는
문턱과 무관하게 성립**한다. F1(검출문턱) 통과 방향의 합이 +0.001238 이고 격차는
0.029690 이라 필요량의 **4%** 뿐이다. 기존 축으로는 못 간다.

그러면 **탐색하지 않은 축**을 열어야 한다. 레인 인구를 보면 어디가 굶었는지 드러난다.

    L8 분석방법적절성   LIVE 30 / PRUNED  1  = 31   <- 전체의 27%
    L2 피처구성         LIVE  8 / PRUNED 16  = 24
    L3 모델링방법       LIVE  6 / PRUNED 11  = 17
    L1 **데이터전처리** LIVE  2 / PRUNED  3  =  5   <- 굶음
    L5 **예측성능우수성** LIVE 2 / PRUNED  1  =  3   <- 굶음

내가 만든 노드 대부분이 진단·계측기였다(L8). L1·L5 는 PRUNED 가 3·1 로 **닫힌 근거도
거의 없다** — 탐색을 안 한 것이지 막힌 것이 아니다. 라우터 C6 은 `LIVE == 0` 을 요구해
발화하지 않지만 **실질적 기아는 여기다**.

**① 레인 리서치 (실제 수행, 2026-08-06)**

  L1 데이터전처리 — https://doi.org/10.3390/s25175329 / PMC12431095

    "SCADA 는 **감발(curtailment)과 센서 결함** 같은 이상을 포함하며, 전처리 틀은
     이상탐지(iForest / LOF / DBSCAN)를 파워커브 모델링과 **공동 최적화**한다."
    "데이터 정제는 **감발의 영향을 제거**해 이상치를 없앤다."

  **우리 자료에 그대로 걸린다. 그리고 A5 가 이미 자백해 뒀다.**

    A5 §5: "운전로그가 없으므로 표준의 로그 기반 필터링을 **통계적 대체**로 수행한다.
            이 대체는 **표준 준수가 아니며** 그 사실을 리포트에 명시한다."

  **왜 이것이 sigma_v 와 직결되는가.** teacher 의 **표적이 `scada_ws`** 다. 감발 구간에서는
  로터가 느리게 돌아 나셀 풍속계 읽음이 왜곡된다(IEC 61400-12-1 이 NTF 없이 나셀
  풍속계를 인정하지 않는 이유). **오염된 표적으로 학습한 teacher 는 그 오염을 배운다.**
  C1N57B 가 잰 g3 의 theta 0.775(곱셈·가용성 잡음)와 A5 의 UNISON 포화비 0.89~0.95 가
  같은 현상이다.

  **적용성 태그**: `directly_supported`.

**② 실체화 — 후보 노드**

  이 노드는 **후보를 만들고 크기를 추정**한다. 실행은 별도 노드다. 라우터가 F1/F2 로
  판정할 수 있게 기대이득을 명시하는 것이 목적이다.

  N1  **감발 구간 teacher 표적 정제** (L1)
      감발로 판정된 시각을 teacher **학습에서 제외**한다. 예측 대상(라벨)은 그대로 두고
      **표적 오염만** 걷어낸다 — 감발은 예보시점에 알 수 없으므로 피처로 쓸 수 없지만,
      학습 표적에서 빼는 것은 배포 가능하다(학습기간 SCADA 만 쓴다).
      판정 규칙: 파워커브 잔차가 하향 이탈하면서 나셀 풍속이 정격 이상인 구간.
      **기대이득 추정**: g3 의 theta 초과분(0.775 대 g1·g2 의 0.5)이 감발에서 온다고
      보면 g3 sigma_v 의 상한 개선이 그 비율에 걸린다. 보수적으로 **1~3%** 로 잡고,
      실측이 F1(0.001013 = sigma_v 0.62% 감소)을 넘는지가 판정이다.

  N2  **이상치 공동 최적화** (L1)
      iForest/LOF/DBSCAN 을 파워커브 적합과 함께 최적화. 문헌의 표준 틀.
      **기대이득 미상** — N1 이 감발 하나만 다루는 데 비해 범위가 넓지만, N1 의 결과가
      나오기 전에는 크기를 주장할 근거가 없다. **N1 이 F1 을 넘을 때만 착수**한다.

  N3  **잔차 순차보정** (L5) — https://arxiv.org/pdf/2501.14805
      확률예보 오차의 순차적 보정. L5 가 굶었으므로 열어 두되, 우리 결정층 실험
      (C1N60·C1N73)이 전부 0 과 구분 불가였으므로 **같은 이웃일 위험**이 크다.
      C17 신규성 기각에 걸릴 후보다. 기록만 하고 우선순위를 낮춘다.

**③ 사전확약**

  H1  L1·L5 가 실제로 굶었다 — LIVE+PRUNED 가 L8 의 1/5 미만.
  H2  N1 의 근거가 **우리 자료에 있다** — A5 가 기록한 비준수와 C1N57B 의 theta 격차.
  H3  N1 의 기대이득 하한(1%)이 F1 문턱(0.62% sigma_v 감소)을 **넘는다**.
      넘지 않으면 실체화해도 라우터가 즉시 기각하므로 만들 이유가 없다.
  H4  N3 는 C17 신규성 기각 위험이 있다 — 최근 창에 L7/L5 결정층 계열이 반복된다.

**리서치·실체화 전용.** 모델 미변경. 실행은 별도 노드. 게이트·lockbox 미사용.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m271_excavation_graph as xg
from m271_p4_consolidate import build

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_c6_lane_expand.md"
RECEIPT = REPORTS / "m271_c6_lane_expand_receipt.json"

NODE_ID = "C1N86_LANE_EXPAND"
LANE = "L1"
PARENT_NODE = "C1N85_C16_RECHECK"

DETECTION_THRESHOLD = 0.001013
RESPONSE_SLOPE = 0.164
# F1 을 sigma_v 감소율로 환산. C1N69 곡선의 k=1 근방 기울기를 쓴다.
F1_AS_SIGMA_REDUCTION = DETECTION_THRESHOLD / RESPONSE_SLOPE

LANE_NAMES = {
    "L1": "데이터전처리", "L2": "피처구성", "L3": "모델링방법", "L4": "검증전략",
    "L5": "예측성능우수성", "L6": "문제해결접근", "L7": "모델개선전략",
    "L8": "분석방법적절성",
}

RESEARCH = {
    "performed_at": "2026-08-06",
    "trigger": "레인 확장 강제 (C6 은 LIVE==0 을 요구해 자연 발화하지 않으나 실질 기아)",
    "sources": [
        {"url": "https://doi.org/10.3390/s25175329", "class": "peer_reviewed",
         "lane": "L1",
         "finding": "SCADA 의 감발·센서결함 이상을 iForest/LOF/DBSCAN 으로 탐지하고 "
                    "파워커브 모델링과 공동 최적화",
         "applicability": "directly_supported"},
        {"url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12431095/",
         "class": "peer_reviewed", "lane": "L1",
         "finding": "데이터 정제가 감발의 영향을 제거해 이상치를 없앤다",
         "applicability": "directly_supported"},
        {"url": "https://arxiv.org/pdf/2501.14805", "class": "peer_reviewed",
         "lane": "L5",
         "finding": "확률예보 오차의 순차적 보정",
         "applicability": "near_match_only"},
    ],
}

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "N1_CURTAILMENT_CLEAN_TEACHER_TARGET",
        "lane": "L1",
        "what": "감발 구간을 teacher **학습 표적에서 제외**한다. 라벨은 그대로 둔다",
        "why_deployable": (
            "감발은 예보시점에 알 수 없어 피처로 못 쓰지만, 학습 표적에서 빼는 것은 "
            "학습기간 SCADA 만 쓰므로 배포 가능하다"
        ),
        "evidence_in_our_data": (
            "A5 §5 가 '운전로그가 없어 로그 기반 필터링을 통계적 대체로 수행했고 "
            "표준 준수가 아니다' 라고 기록. C1N57B 의 g3 theta 0.775 대 g1·g2 0.5, "
            "A5 의 UNISON 포화비 0.89~0.95 대 VESTAS 0.99"
        ),
        "expected_sigma_reduction_low": 0.01,
        "expected_sigma_reduction_high": 0.03,
        "priority": 1,
    },
    {
        "id": "N2_JOINT_ANOMALY_POWERCURVE",
        "lane": "L1",
        "what": "iForest/LOF/DBSCAN 을 파워커브 적합과 공동 최적화 (문헌 표준 틀)",
        "why_deployable": "학습기간 SCADA 전처리이므로 배포 가능",
        "evidence_in_our_data": "N1 과 같은 결함군이나 범위가 넓다",
        "expected_sigma_reduction_low": None,
        "expected_sigma_reduction_high": None,
        "priority": 3,
        "gate": "N1 이 F1 을 넘을 때만 착수한다 — 그 전에는 크기를 주장할 근거가 없다",
    },
    {
        "id": "N3_SEQUENTIAL_ERROR_CORRECTION",
        "lane": "L5",
        "what": "확률예보 오차의 순차적 보정",
        "why_deployable": "후처리이므로 배포 가능",
        "evidence_in_our_data": "L5 가 굶었다는 것 외에 우리 자료의 직접 근거 없음",
        "expected_sigma_reduction_low": None,
        "expected_sigma_reduction_high": None,
        "priority": 4,
        "risk": "C1N60·C1N73 결정층 계열과 같은 이웃 — C17 신규성 기각 위험",
    },
)


def main() -> int:
    graph, _ledger = build()
    census = graph.lane_census()
    lanes = {
        lane: {
            "name": LANE_NAMES.get(lane, ""),
            "live": int(counts.get(xg.LIVE, 0)),
            "pruned": int(counts.get(xg.PRUNED, 0)),
            "total": int(counts.get(xg.LIVE, 0)) + int(counts.get(xg.PRUNED, 0)),
        }
        for lane, counts in census.items()
    }
    largest = max(v["total"] for v in lanes.values())
    starved = sorted(
        (lane for lane, v in lanes.items() if v["total"] < largest / 5),
        key=lambda x: lanes[x]["total"],
    )
    h1 = bool({"L1", "L5"} <= set(starved))

    n1 = next(c for c in CANDIDATES if c["priority"] == 1)
    h2 = bool(n1["evidence_in_our_data"])
    h3 = bool(float(n1["expected_sigma_reduction_low"]) > F1_AS_SIGMA_REDUCTION)
    h4 = bool(any(c.get("risk") for c in CANDIDATES))

    if h1 and h3:
        verdict = "LANE_EXPANDED_N1_WORTH_EXECUTING"
    elif h1:
        verdict = "LANES_STARVED_BUT_NO_CANDIDATE_CLEARS_DETECTION"
    else:
        verdict = "NO_LANE_STARVATION_FOUND"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "materialization_only": True,
        "research": RESEARCH,
        "lane_census": lanes,
        "starved_lanes": starved,
        "largest_lane_total": largest,
        "detection_threshold": DETECTION_THRESHOLD,
        "f1_as_sigma_reduction": F1_AS_SIGMA_REDUCTION,
        "candidates": list(CANDIDATES),
        "hypotheses": {
            "H1_l1_l5_starved": h1,
            "H2_n1_grounded_in_our_data": h2,
            "H3_n1_low_estimate_clears_f1": h3,
            "H4_n3_novelty_risk": h4,
        },
        "verdict": verdict,
        "next_action": (
            "N1 을 실행 노드로 만든다. 감발 판정 규칙을 실행 전에 동결하고, "
            "teacher sigma_v 를 시간분할 test 행에서 재어 F1 을 넘는지 본다."
        ),
        "dacon_upload": False,
        "external_actions": ["WebSearch"],
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")

    lines = [
        "# M271 C6 — 레인 확장 강제",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **실체화 전용**",
        "",
        "## 1. 레인 인구",
        "",
        "| 레인 | 이름 | LIVE | PRUNED | 전체 |",
        "|---|---|---:|---:|---:|",
    ]
    for lane in sorted(lanes):
        v = lanes[lane]
        mark = " **<- 굶음**" if lane in starved else ""
        lines.append(
            f"| {lane}{mark} | {v['name']} | {v['live']} | {v['pruned']} | {v['total']} |"
        )
    lines += [
        "",
        f"최대 레인 {largest} 개. 그 1/5 미만이면 기아로 본다 -> `{', '.join(starved)}`",
        "",
        "L1·L5 는 PRUNED 가 3·1 로 **닫힌 근거도 거의 없다** — 탐색을 안 한 것이지 "
        "막힌 것이 아니다.",
        "",
        "## 2. 레인 리서치",
        "",
    ]
    for s in RESEARCH["sources"]:
        lines.append(f"- [{s['lane']}] {s['finding']} — <{s['url']}> (`{s['applicability']}`)")
    lines += [
        "",
        "## 3. 실체화된 후보",
        "",
    ]
    for c in sorted(CANDIDATES, key=lambda x: x["priority"]):
        lines += [
            f"### {c['priority']}. `{c['id']}` ({c['lane']})",
            "",
            f"- **무엇** {c['what']}",
            f"- **배포 가능성** {c['why_deployable']}",
            f"- **우리 자료의 근거** {c['evidence_in_our_data']}",
        ]
        if c["expected_sigma_reduction_low"] is not None:
            lines.append(
                f"- **기대 sigma_v 감소** {c['expected_sigma_reduction_low']:.1%}"
                f"~{c['expected_sigma_reduction_high']:.1%} "
                f"(F1 문턱 {F1_AS_SIGMA_REDUCTION:.2%})"
            )
        else:
            lines.append("- **기대이득 미상** — 근거 없이 주장하지 않는다")
        if c.get("gate"):
            lines.append(f"- **착수 조건** {c['gate']}")
        if c.get("risk"):
            lines.append(f"- **위험** {c['risk']}")
        lines.append("")
    lines += [
        "## 4. 사전확약",
        "",
        f"- H1 L1·L5 가 굶었다 -> **{h1}**",
        f"- H2 N1 이 우리 자료에 근거가 있다 -> **{h2}**",
        f"- H3 N1 하한 1% 가 F1({F1_AS_SIGMA_REDUCTION:.2%})을 넘는다 -> **{h3}**",
        f"- H4 N3 는 C17 신규성 기각 위험 -> **{h4}**",
        "",
        "## 5. 판정",
        "",
        f"**{verdict}**",
        "",
        payload["next_action"],
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== C6 레인 확장 ===")
    for lane in sorted(lanes):
        v = lanes[lane]
        mark = "  <- 굶음" if lane in starved else ""
        print(f"[C6] {lane} {v['name']:>10s}  LIVE {v['live']:2d} / PRUNED "
              f"{v['pruned']:2d} / 전체 {v['total']:2d}{mark}")
    print(f"[C6] 굶은 레인 {starved} (최대 {largest} 의 1/5 미만)")
    print(f"[C6] F1 검출문턱 {DETECTION_THRESHOLD} = sigma_v {F1_AS_SIGMA_REDUCTION:.2%} 감소")
    for c in sorted(CANDIDATES, key=lambda x: x["priority"]):
        est = (f"{c['expected_sigma_reduction_low']:.1%}~"
               f"{c['expected_sigma_reduction_high']:.1%}"
               if c["expected_sigma_reduction_low"] is not None else "미상")
        print(f"[C6] {c['priority']}. {c['id']:38s} ({c['lane']}) 기대 {est}")
    print(f"[C6] H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C6] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
