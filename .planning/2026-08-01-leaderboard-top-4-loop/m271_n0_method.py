"""M271 N0 내부 루프 ①② — 방법 리서치 결과와 동결 사양.

N0 은 발굴 그래프의 루트 노드 "데이터 정밀 분석 및 특성 정밀 파악"이다. 계획의 내부
루프에 따라 실행 전에 방법 리서치를 먼저 수행했고, 이 파일이 그 산출(①)과 그로부터
동결한 사양(②)을 기록한다.

여기서 찾은 것은 **수행 방법**이지 도메인 결론이 아니다. "무엇이 결손을 설명하는가"는
방향 리서치이며 A1~A7 증거가 나온 뒤 라우터가 발화시킨다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_n0_method.md"
RECEIPT = REPORTS / "m271_n0_method_receipt.json"

# --------------------------------------------------------------------------
# ① 방법 리서치 — 사전확약대로 4개 하위질의를 독립 수행한 결과.
# 적용성 태그: directly_supported | near_match_only | contradicts_premise | insufficient
# --------------------------------------------------------------------------
METHODS: list[dict[str, Any]] = [
    {
        "sub_query": "a_forecast_verification",
        "serves_nodes": ["A4", "A7"],
        "lane": "L8",
        "method": "층화 예보검증(stratified verification) + Murphy 분해",
        "standard": "Murphy(1973) Brier score -> reliability + resolution + uncertainty. "
        "예보검증의 표준 분해.",
        "frameworks_considered": {
            "xskillscore==0.0.29": {"license": "Apache-2.0", "adopted": False},
            "scores==2.6.0": {"license": "Apache-2.0", "adopted": False},
            "properscoring==0.1": {"license": "Apache-2.0", "adopted": False},
        },
        "tag": "near_match_only",
        "why": (
            "세 라이브러리 모두 xarray 기반 격자 예보를 대상으로 한다. 우리 표면은 3그룹 "
            "시간별 tabular 이고 공식 지표(NMAE/FICR)는 이미 정확히 구현돼 있다. 필요한 것은 "
            "지표 구현이 아니라 **층화 규약**이므로 라이브러리를 도입할 이유가 없다. "
            "dask/xarray/xhistogram 의존을 추가하는 비용만 남는다."
        ),
        "adopt": "방법만 채택. 신규 의존성 0. 층화 축은 리드타임/시각/계절/풍향섹터.",
    },
    {
        "sub_query": "b_power_curve",
        "serves_nodes": ["A5"],
        "lane": "L1",
        "method": "IEC 61400-12-1 풍속 bin 방법 + 공기밀도 정규화, SCADA 이상치 제거",
        "standard": "IEC 61400-12-1:2022 — 단일 풍력터빈 출력성능 계측 절차. "
        "데이터 품질검사(가용성/출력제한/고장 제외)를 명시적으로 요구한다.",
        "frameworks_considered": {},
        "tag": "near_match_only",
        "why": (
            "표준은 기상탑을 갖춘 통제된 단일터빈 계측 프로토콜이다. 우리는 로터 뒤 나셀 "
            "풍속계, 10분 데이터, **운전로그 없음**, 그리고 예측 대상은 그룹 합산 출력이다. "
            "따라서 운전로그 기반 필터링은 부적용이고, **풍속 bin 구성과 공기밀도 정규화**는 "
            "그대로 적용된다. 로그가 없으므로 출력제한/정지 구간은 통계적으로 식별해야 한다."
        ),
        "adopt": (
            "bin 방법 + 밀도 정규화 채택. 로그 대신 통계적 이상치 제거(3-sigma + 사분위 "
            "조합, Mahalanobis 거리)로 대체하고 그 대체 사실을 리포트에 명시한다."
        ),
    },
    {
        "sub_query": "c_feature_relevance",
        "serves_nodes": ["A2"],
        "lane": "L2",
        "method": "KSG(Kraskov-Stoegbauer-Grassberger) kNN 상호정보량",
        "standard": "Kraskov et al. 2004. 연속변수 MI 추정의 표준 kNN 방법.",
        "frameworks_considered": {
            "scikit-learn==1.9.0 (mutual_info_regression)": {
                "license": "BSD-3-Clause",
                "adopted": True,
                "note": "이미 설치됨. 신규 의존성 0.",
            }
        },
        "tag": "near_match_only",
        "why": (
            "두 가지 한계가 A2 의 실제 질문과 어긋난다. (1) 주변 MI 는 기존 12개 컬럼 대비 "
            "**추가** 정보를 재지 못한다. A2 가 알고 싶은 것은 조건부 관련성이다. "
            "(2) kNN 추정량은 편향되며 편향이 표본수와 k 에 의존하는데, 우리 데이터는 강한 "
            "자기상관을 가져 **유효표본수가 n 보다 훨씬 작다**. 절대값을 신뢰할 수 없다."
        ),
        "adopt": (
            "MI 는 **선별 스크린으로만** 쓰고 순위 해석에 그친다. 채택 판정은 시간순 안전 "
            "홀드아웃에서의 추가이득으로 별도 측정한다. 자기상관 보정 없이 유의성을 주장하지 "
            "않는다."
        ),
    },
    {
        "sub_query": "d_timeseries_characterisation",
        "serves_nodes": ["A1"],
        "lane": "L1",
        "method": "표준 특성 피처셋",
        "standard": "catch22 (Lubba et al. 2019) — hctsa 4791개에서 선별한 22개. "
        "분류성능 90% 를 1000배 빠르게 재현.",
        "frameworks_considered": {
            "pycatch22==0.4.5": {"license": "GPL-3.0-or-later", "adopted": False},
            "tsfresh==0.21.2": {"license": "MIT", "adopted": False},
        },
        "tag": "contradicts_premise",
        "why": (
            "**pycatch22 는 GPLv3+ 카피레프트다.** 상금이 걸리고 2차평가에 코드 일체를 "
            "제출해야 하는 대회에서 파생물 라이선스 부담을 지는 것은 정당화되지 않는다. "
            "공식 규칙 5조는 사전학습 *가중치*에 상업이용 허용을 요구하는데, 라이브러리에 "
            "직접 적용되지는 않더라도 같은 취지에서 회피한다. tsfresh 는 MIT 이지만 분류용 "
            "대량 피처 생성기이고 A1 이 필요한 것은 라벨 3계열의 분포/주기/램프 특성이라 "
            "목적이 다르며, statsmodels/stumpy/pywavelets 의존을 추가한다."
        ),
        "adopt": "둘 다 미채택. A1 은 명시적 통계량을 직접 계산한다. 신규 의존성 0.",
    },
]

# --------------------------------------------------------------------------
# ① 부수 발견 — A7 사양을 바꾸므로 별도로 기록한다.
# --------------------------------------------------------------------------
DECOMPOSITION_FINDING = {
    "claim": "공식 Total 은 임의의 행 분할에 대해 정확히 가법 분해된다.",
    "derivation": [
        "FICR_g = sum_i(a_i * u_i) / sum_i(4 * a_i) 이므로, 행 분할 {C} 에 대해",
        "FICR_g = sum_C w_C * (ubar_C / 4),  w_C = sum_{i in C} a_i / sum_all a_i,  sum_C w_C = 1",
        "즉 FICR 은 셀에 대한 볼록결합이다.",
        "NMAE_g = mean_i(|e_i|) 도 유효행에 대한 평균이므로 동일하게 분해된다.",
        "Total = 0.5*(1 - mean_g NMAE_g) + 0.5*mean_g FICR_g 는 둘의 선형결합이다.",
        "유효행 판정(actual >= 0.1*capacity)은 **실측값에만** 의존하므로 후보를 바꿔도 "
        "행 집합과 가중치 w_C 가 변하지 않는다. 분할이 고정이다.",
    ],
    "consequence": (
        "계획의 R9('FICR 이 발전량 가중 비선형이라 가법 분해가 정확히 망라적이지 않을 수 "
        "있음')는 과대평가로 보인다. A7 은 근사 귀속이 아니라 **정확한 회계**를 목표로 할 수 "
        "있고, 잔차 셀이 0 이어야 한다."
    ),
    "status": "DERIVED_NOT_YET_VERIFIED",
    "verification_owner": "A7 (④ 단계에서 수치로 확인. 잔차가 부동소수 오차 범위를 넘으면 "
    "이 주장은 철회되고 R9 가 복구된다.)",
}

# --------------------------------------------------------------------------
# ② 동결 사양 — ① 결과가 각 자식 노드의 사양을 결정한다.
# 실행 전에 동결한다. spec_hash 는 이 구조체의 내용해시다.
# --------------------------------------------------------------------------
SPECS: dict[str, dict[str, Any]] = {
    "A1_labels": {
        "generation": 1,
        "lane": "L1",
        "depends_on": [],
        "method_from": "d_timeseries_characterisation",
        "task": "라벨 3계열의 분포/결측/유효행 비중/주기/램프 특성을 명시 통계량으로 계산",
        "outputs": [
            "그룹별 결측·0·저출력 비중",
            "유효행(actual >= 0.1*capacity) 비중 — 월별/그룹별",
            "발전량 분포 분위수와 y대역 경계 후보",
            "시각/월 주기 구조",
            "시간당 램프 |dy| 분포",
            "그룹간 동시 상관",
            "그룹3 2022 부재가 학습 표면에 미치는 영향",
        ],
        "new_dependencies": [],
        "predeclared_expectation": "없음. 이 노드는 가설 검정이 아니라 특성 기술이다.",
        "stop_condition": "위 7개 산출이 모두 생성되면 종료.",
    },
    "A2_columns": {
        "generation": 1,
        "lane": "L2",
        "depends_on": [],
        "method_from": "c_feature_relevance",
        "task": "공급 NWP 컬럼 전수 인벤토리와 라벨 관련성 스크린",
        "outputs": [
            "GFS/LDAPS 전 컬럼 목록과 configs/features/spatial_v2.yaml 실사용분 대조",
            "미사용 컬럼별 mutual_info_regression 점수 (순위 해석 전용)",
            "자기상관으로 인한 유효표본 축소 경고를 수치와 함께 기록",
        ],
        "new_dependencies": [],
        "predeclared_expectation": (
            "MI 절대값은 신뢰하지 않는다. 순위만 본다. 상위 컬럼이라도 조건부 추가이득은 "
            "별도 측정 전까지 주장하지 않는다."
        ),
        "stop_condition": "전 컬럼에 MI 점수가 붙고 사용/미사용 구분이 완료되면 종료.",
        "router_note": "MI 가 임계 미만이면 C8(PRUNE), 이상이면 C1 로 L2 방향리서치 발화.",
    },
    "A3_spatial": {
        "generation": 1,
        "lane": "L2",
        "depends_on": [],
        "method_from": None,
        "task": "격자-터빈 기하 구조 규명",
        "outputs": [
            "info.xlsx 17터빈의 좌표/허브고/로터경/제작사/모델/그룹 매핑",
            "LDAPS 16격자·GFS 9격자와 각 터빈의 거리 행렬",
            "그룹별 공간 배치와 격자 커버리지",
            "surface_0_h(지형고도), surface_0_lsm(육해 마스크)의 격자별 값",
        ],
        "new_dependencies": [],
        "predeclared_expectation": "없음. 기하 기술이다.",
        "stop_condition": "매핑과 거리 행렬이 산출되면 종료.",
    },
    "A5_scada": {
        "generation": 1,
        "lane": "L3",
        "depends_on": [],
        "method_from": "b_power_curve",
        "task": "IEC 61400-12-1 bin 방법으로 터빈별 경험 파워커브 추정",
        "outputs": [
            "터빈 17기별 풍속 bin 파워커브 (0.5 m/s bin)",
            "공기밀도 정규화 적용 전후 대조",
            "통계적 이상치 제거(3-sigma + 사분위) 전후 대조와 제거율",
            "나셀풍속 대 NWP 풍속의 그룹별 사상 관계",
        ],
        "new_dependencies": [],
        "predeclared_expectation": (
            "운전로그가 없으므로 표준의 로그 기반 필터링을 통계적 대체로 수행한다. "
            "이 대체는 표준 준수가 아니며 그 사실을 리포트에 명시한다."
        ),
        "stop_condition": "17기 파워커브와 제거율이 산출되면 종료.",
        "constraint": "SCADA 는 진단 전용. 추론 피처를 만들지 않는다(평가기간 부재).",
    },
    "A6_timing": {
        "generation": 1,
        "lane": "L1",
        "depends_on": [],
        "method_from": None,
        "task": "시간 규약 감사 — 공급 데이터 가용시각 대 공식 예측기준시점",
        "outputs": [
            "data_available_kst_dtm 실측 분포와 명세서 기술(13:00 KST)의 일치 확인",
            "행별 공식 예측기준시점(대상일 전일 14:00 KST) 계산",
            "행별 여유 = 공식기준시점 - 공급가용시각. 00:00 행의 특수성 확인",
            "리드타임(예보 초기화 09:00 KST 기준) 분포",
        ],
        "new_dependencies": [],
        "predeclared_expectation": (
            "명세서상 모든 행이 13:00 가용이고 공식 기준시점은 14:00 이므로 최소 1시간 "
            "여유가 예상된다. 00:00 행은 대상일이 하루 뒤라 여유가 25시간으로 예상된다. "
            "실측이 이와 다르면 그 자체가 발견이다."
        ),
        "stop_condition": "행별 여유 분포가 산출되면 종료.",
    },
    "A4_error": {
        "generation": 2,
        "lane": "L2",
        "depends_on": ["A1_labels", "A3_spatial"],
        "method_from": "a_forecast_verification",
        "task": "층화 예보검증 — 예보-실측 오차 구조를 축별로 분해",
        "outputs": [
            "리드타임(16~39h)별 오차",
            "시각(01~24)별 오차",
            "월/계절별 오차",
            "풍향 섹터별 오차",
            "LDAPS 50MUmax/min·50MVmax/min 의 max-min 과 오차의 관계",
        ],
        "new_dependencies": [],
        "predeclared_expectation": (
            "max-min 스프레드가 오차와 양의 관계를 가질 것으로 예상한다. 부호가 반대면 "
            "C5(anomaly)로 라우팅된다."
        ),
        "stop_condition": "5개 층화 축이 모두 산출되면 종료.",
    },
    "A7_deficit_init": {
        "generation": 3,
        "lane": "L8",
        "depends_on": ["A1_labels", "A4_error"],
        "method_from": "a_forecast_verification",
        "task": "결손 원장 초기화 — 목표까지의 격차를 셀별로 정확 회계",
        "outputs": [
            "셀 키 (group x 월 x y대역 x 정산단위) 별 w_C 와 ubar_C",
            "현재 로컬 Total 과 목표 0.66 의 격차",
            "격차의 셀별 가법 분해와 **잔차**",
        ],
        "new_dependencies": [],
        "predeclared_expectation": (
            "DECOMPOSITION_FINDING 에 따라 잔차가 부동소수 오차 범위(1e-9) 안이어야 한다. "
            "이를 넘으면 그 주장을 철회하고 계획 R9 를 복구한다. 이 기대는 결과를 보기 전에 "
            "동결한다."
        ),
        "stop_condition": "셀별 분해와 잔차가 산출되면 종료.",
    },
}


def spec_hash() -> str:
    payload = json.dumps(SPECS, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_report() -> None:
    tag_counts: dict[str, int] = {}
    for m in METHODS:
        tag_counts[m["tag"]] = tag_counts.get(m["tag"], 0) + 1

    lines = [
        "# M271 N0 — 방법 리서치(①)와 동결 사양(②)",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        "- 노드: N0 루트 — 데이터 정밀 분석 및 특성 정밀 파악",
        f"- `spec_hash`: `{spec_hash()}`",
        "",
        "내부 루프 ① 은 **수행 방법**을 찾는 단계다. 도메인 결론(무엇이 결손을 설명하는가)은",
        "방향 리서치이며 A1~A7 증거가 나온 뒤 라우터가 발화시킨다.",
        "",
        "## ① 방법 리서치 결과",
        "",
        "| 하위질의 | 대상 노드 | 채택 방법 | 태그 | 신규 의존성 |",
        "|---|---|---|---|---|",
    ]
    for m in METHODS:
        nodes = ", ".join(m["serves_nodes"])
        lines.append(f"| `{m['sub_query']}` | {nodes} | {m['method']} | `{m['tag']}` | 0 |")

    lines += [
        "",
        f"태그 분포: {tag_counts}",
        "",
        "**표준 방법은 확립돼 있으나, 검토한 프레임워크 중 이 과제 조건에 직접 맞는 것은 없다.**",
        "네 하위질의 모두 `directly_supported` 가 아니다. 그래서 방법은 채택하되 라이브러리는",
        "도입하지 않는다. 신규 의존성 총 0개.",
        "",
    ]

    for m in METHODS:
        lines += [
            f"### {m['sub_query']} → {', '.join(m['serves_nodes'])} (레인 {m['lane']})",
            "",
            f"- 표준: {m['standard']}",
        ]
        if m["frameworks_considered"]:
            lines.append("- 검토한 프레임워크:")
            for fw, meta in m["frameworks_considered"].items():
                mark = "채택" if meta.get("adopted") else "미채택"
                extra = f" — {meta['note']}" if meta.get("note") else ""
                lines.append(f"  - `{fw}` (라이선스 {meta['license']}) — **{mark}**{extra}")
        lines += [
            f"- 적용성 태그: `{m['tag']}`",
            f"- 판단: {m['why']}",
            f"- 채택: {m['adopt']}",
            "",
        ]

    lines += [
        "## ① 부수 발견 — 공식 Total 의 정확 가법 분해",
        "",
        f"**주장**: {DECOMPOSITION_FINDING['claim']}",
        "",
        "```",
    ]
    lines += DECOMPOSITION_FINDING["derivation"]
    lines += [
        "```",
        "",
        f"**귀결**: {DECOMPOSITION_FINDING['consequence']}",
        "",
        f"**상태**: `{DECOMPOSITION_FINDING['status']}`",
        "",
        f"검증 소유: {DECOMPOSITION_FINDING['verification_owner']}",
        "",
        "## ② 동결 사양",
        "",
        "① 결과가 각 자식 노드의 사양을 결정한다. 실행 전에 동결하며, 결과를 본 뒤",
        "재해석하지 않는다.",
        "",
        "| 노드 | 세대 | 레인 | 의존 | 방법 출처 | 신규 의존성 |",
        "|---|---|---|---|---|---|",
    ]
    for name, spec in SPECS.items():
        dep = ", ".join(spec["depends_on"]) or "-"
        src = spec["method_from"] or "-"
        lines.append(
            f"| `{name}` | {spec['generation']} | {spec['lane']} | {dep} | `{src}` | "
            f"{len(spec['new_dependencies'])} |"
        )

    lines += [
        "",
        "세대 1(A1·A2·A3·A5·A6)은 서로 독립이므로 병렬 실행한다. 세대 2(A4)는 A1·A3 를,",
        "세대 3(A7)은 A1·A4 를 fan-in 한다.",
        "",
    ]
    for name, spec in SPECS.items():
        lines += [
            f"### `{name}`",
            "",
            f"- 과업: {spec['task']}",
            "- 산출:",
        ]
        lines += [f"  - {o}" for o in spec["outputs"]]
        lines += [f"- 사전확약 기대: {spec['predeclared_expectation']}"]
        if spec.get("constraint"):
            lines.append(f"- 제약: {spec['constraint']}")
        if spec.get("router_note"):
            lines.append(f"- 라우팅: {spec['router_note']}")
        lines += [f"- 중단 조건: {spec['stop_condition']}", ""]

    lines += [
        "## 남는 한계",
        "",
        "1. 네 하위질의 중 `directly_supported` 가 하나도 없다. 표준 방법을 이 과제 표면으로",
        "   옮겨 구현하므로, 구현 자체가 새 실패지점이다.",
        "2. A2 의 MI 는 주변 관련성만 재고 조건부 추가이득을 재지 못한다. 자기상관 때문에",
        "   절대값도 신뢰할 수 없다. 순위 스크린으로만 쓴다.",
        "3. A5 는 운전로그가 없어 IEC 표준의 로그 기반 필터링을 통계적으로 대체한다.",
        "   표준 준수가 아니다.",
        "4. 가법 분해 주장은 **유도했을 뿐 아직 검증되지 않았다**(A7 ④에서 확인).",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    write_report()
    receipt = {
        "schema_version": 1,
        "stage": "M271_N0_METHOD_AND_SPEC",
        "node": "N0",
        "inner_loop_steps": ["1_method_research", "2_spec_freeze"],
        "decided_utc": datetime.now(UTC).isoformat(),
        "spec_hash": spec_hash(),
        "methods": METHODS,
        "decomposition_finding": DECOMPOSITION_FINDING,
        "specs": SPECS,
        "new_dependencies_total": sum(len(s["new_dependencies"]) for s in SPECS.values()),
        "applicability_tags": sorted({m["tag"] for m in METHODS}),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False,
        "external_actions": ["web search (read-only, method research)"],
        "model_fits": 0,
        "lockbox_reopened": False,
        "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"spec_hash = {spec_hash()}")
    print(f"신규 의존성 = {receipt['new_dependencies_total']}")
    print(f"적용성 태그 = {receipt['applicability_tags']}")
    print(f"report  -> {REPORT_MD}")
    print(f"receipt -> {RECEIPT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
