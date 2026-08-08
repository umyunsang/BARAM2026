"""M271 P4 사이클 77 — 소스를 앞단에서 뭉개지 않는다: NWP 소스별 모델 + 스태킹.

**이 노드는 실제 딥리서치에서 나왔다.** 사이클 56~76 의 "① 방법 리서치" 절은 전부 기억에서
인용한 것이었고 웹 조사를 한 번도 하지 않았다. 그 결과 같은 이웃(0.005 규모 효과)만 20 여
사이클 팠고, 격차 0.037 에는 자릿수가 모자랐다.

**① 방법 리서치 (실제 수행, 2026-08-05)**

  조사 질의 — "복수 NWP 소스를 쓸 때 앞단에서 결합하는가, 소스별 모델을 스태킹하는가"

  - HEFTCom2024(3.6GW 풍력·태양광 포트폴리오, 2024) **우승팀 SVK** 는 CatBoost 를
    **NWP 소스별로(DWD / GFS / MEPS) 각각 적합**하고 풍력·태양광도 분리해
    MultiQuantile 손실로 학습했다. 앞단 결합이 아니다.
    https://arxiv.org/pdf/2505.10367
  - 복수 NWP 소스 결합의 오차 감소는 **8~30%** 로 보고된다(직접·간접 예측 각각).
    https://www.sciencedirect.com/science/article/pii/S0360544222027797
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10637996/
  - 같은 문헌군이 "GFS 와 DWD 각각으로 **모델 계열 두 벌**을 학습한 뒤 출력을
    **스태킹**" 하는 구성을 명시한다.
  - **적용성 태그**: `directly_supported`. 우리 자료도 소스가 분리돼 있다
    (`geom__gfs__` 41 / `geom__ldaps__` 34 / `geom__align__` 7 / `sitewind__` 13).

  **왜 우리 구조가 이것과 어긋나는가**

    현재   GFS + LDAPS -> 공간결합 -> 학습형 teacher -> `sitewind__*` 13 개 -> 모델 1 개
    문헌   GFS -> 모델 A,  LDAPS -> 모델 B,  A·B 출력을 스태킹

  우리는 **소스 다양성을 앞단에서 파괴**한다. C1N71 이 잰 teacher 상관 0.89~0.91 은
  feature 부분집합 둘 사이의 값이지 소스 둘 사이가 아니다 — 소스 다양성은 그보다 앞에서
  이미 사라진다. C1N68 이 잰 "모형 스택 전체가 커브 직독 대비 +0.017" 도 같은 증상이다.

  크기도 맞는다. C1N46·C1N48 이 상위권 우위를 **출력오차 16% 감소**(k~0.84)로 설명했고,
  문헌의 8~30% 가 그 범위를 덮는다.

**② 사양 동결**

  하네스   C1N56·C1N60 과 동일한 학습 설정(teacher 복원, generic 기저, leaves 15,
           lr 0.1, 200 rounds, 46 구간 soft target). **바뀌는 것은 피처 분할과
           결합 지점뿐**이라 처리효과가 거기에 귀속된다.
  팔 넷
    POOLED       현행. 전 피처 100 개로 모델 1 개.               <- V1 대조군
    GFS_ONLY     `geom__gfs__*` + `gfs__*` + `cal__*` + sitewind 중 **GFS 유래만**
    LDAPS_ONLY   `geom__ldaps__*` + `cal__*` + sitewind 중 **LDAPS 유래만**
    STACK        GFS_ONLY 와 LDAPS_ONLY 의 **확률행렬**을 결합한 뒤 Bayes 결정
  결합     확률 공간에서 `w * P_gfs + (1-w) * P_ldaps`. `w` 는 **fold-외** 선택
           (격자 0.0~1.0, 0.1 간격, 실행 전 동결). C1N44 이래의 절차.
  sitewind 분할  `sitewind__legacy` 계열은 legacy 보조컬럼(LDAPS 중심), `sitewind__allweather`
           계열은 전 기상컬럼. **`sitewind__mean` 계열은 두 소스를 섞으므로 두 팔 모두에서
           제외한다** — 그것이 이 노드가 없애려는 바로 그 결합이다.
  결정층   C1N60 GLOBAL fold-외 T. 네 팔에 **같은 절차**를 적용한다.

  **타당성 가드**
    V1  POOLED 가 C1N56·C1N60 의 대조군 **0.604043 을 ±0.0005 로 재현**.
        벗어나면 하네스가 바뀐 것이고 나머지 판정을 버린다.
    V2  GFS_ONLY 와 LDAPS_ONLY 의 피처 집합이 **서로소**이고(cal 제외) 각각 20 개 이상.
        겹치면 "소스 분리" 가 아니다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  STACK > POOLED.  **핵심.** 소스를 앞단에서 뭉개지 않는 것이 값을 한다.
    H2  STACK > max(GFS_ONLY, LDAPS_ONLY). 결합이 최선 단일을 이긴다.
        **부호 예단 없음** — C1N34 가 다른 후보군에서 `NOTHING_TO_ADD_TO_BEST_SINGLE`
        을 이미 봤다.
    H3  두 단일 팔의 **확률행렬 상관이 sitewind 상관(0.89~0.91)보다 낮다.**
        낮아야 "앞단 결합이 다양성을 파괴했다" 는 진단이 성립한다. 이것이
        기전 검정이고 H1 의 성패와 무관하게 정보를 준다.
    H4  STACK 이 POOLED 대비 **동결 게이트 통과**.
    H5  이득이 FICR 쪽에서 우세. 상위권 격차의 71% 가 FICR 이므로 방향이 맞아야 한다.

  H1 이 참이면 이 축이 열리고 다음은 소스별 **하이퍼파라미터 분리**·**손실 분리**로
  간다(HEFTCom 우승팀은 MultiQuantile 을 썼다). H1 이 거짓이면 소스 분리가
  이 자료에서는 값을 못 한다는 뜻이고 그 자체가 20 사이클치보다 큰 정보다.

게이트 미수정. lockbox·외부데이터·2024 행·`scada_ws` 예측피처 미사용. 제출 없음.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, evaluate_gate
from m271_cycle37_band_loss import KEYS, PROBE, fold_rows
from m271_cycle40_band_classifier import (
    CLASS_WIDTH,
    N_CLASS,
    PARAMS,
    ROUNDS,
    bayes_decision,
    make_objective,
    one_hot_targets,
)
from m271_cycle42_teacher_restored import all_weather_columns, teach
from m271_cycle44_sharpened_decision import TEMPERATURES, sharpen
from m271_cycle56_measured_powercurve import add_sitewind_with_basis, measured_curves
from m271_evaluate_candidate import official
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle77_per_source_stack.md"
RECEIPT = REPORTS / "m271_cycle77_per_source_stack_receipt.json"

NODE_ID = "C1N77_PER_SOURCE_STACK"
LANE = "L3"
PARENT_NODE = "C1N68_EMPIRICAL_DECOMPOSITION"

CONTROL = 0.604043
TOLERANCE = 0.0005
WEIGHT_GRID = tuple(round(0.1 * i, 1) for i in range(11))
MIN_FEATURES = 20
SOURCES = ("gfs", "ldaps")

RESEARCH = {
    "performed_at": "2026-08-05",
    "query": "복수 NWP 소스: 앞단 결합 vs 소스별 모델 스태킹",
    "sources": [
        {"url": "https://arxiv.org/pdf/2505.10367",
         "class": "peer_reviewed",
         "finding": "HEFTCom2024 우승팀 SVK — CatBoost 를 NWP 소스별(DWD/GFS/MEPS) 분리 적합",
         "applicability": "directly_supported"},
        {"url": "https://www.sciencedirect.com/science/article/pii/S0360544222027797",
         "class": "peer_reviewed",
         "finding": "복수 NWP 소스 결합이 예측오차를 8~30% 감소",
         "applicability": "directly_supported"},
        {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10637996/",
         "class": "peer_reviewed",
         "finding": "소스별 모델 계열 학습 후 출력 스태킹 구성",
         "applicability": "directly_supported"},
    ],
    "decision_impact": "소스 결합 지점을 앞단(sitewind)에서 뒷단(확률 스태킹)으로 옮긴다",
    "stop_condition": "H1 이 거짓이면 소스 분리 축을 닫는다",
}


def split_features(columns: list[str]) -> dict[str, list[str]]:
    """소스별로 나눈다. 두 소스를 섞는 컬럼(`sitewind__mean`, `geom__align__`)은 뺀다."""
    out: dict[str, list[str]] = {s: [] for s in SOURCES}
    shared = [c for c in columns if c.startswith("cal__")]
    for column in columns:
        if column.startswith("cal__"):
            continue
        if column.startswith("geom__gfs__") or column.startswith("gfs"):
            out["gfs"].append(column)
        elif column.startswith("geom__ldaps__") or column.startswith("ldaps"):
            out["ldaps"].append(column)
        # geom__align__ 과 sitewind__mean 계열은 두 소스를 섞으므로 어느 팔에도 넣지 않는다.
    for source in SOURCES:
        out[source] = [*out[source], *shared]
    return out


def main() -> int:
    curves = measured_curves()
    surface, _base, auxiliary = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface["rate"] = surface["actual_kwh"] / surface["capacity"]
    surface = surface.loc[surface["rate"].notna()].reset_index(drop=True)

    wanted = json.loads(
        (PROBE / "M115_XGBOOST-dev-2023-Q3.json").read_text(encoding="utf-8")
    )["selected_feature_names"]
    base_features = [c for c in wanted if c in surface.columns and c != "scada_ws"]
    aux_cols = [c for c in auxiliary if c in surface.columns and c != "scada_ws"]
    aw_cols = all_weather_columns(surface)
    split = split_features(base_features)

    # sitewind 파생을 소스별로 배정한다. `legacy` 는 보조컬럼(LDAPS 중심),
    # `allweather` 는 전 기상컬럼. `mean` 계열은 두 소스를 섞으므로 양쪽에서 뺀다.
    sitewind_of = {
        "gfs": lambda n: "allweather" in n,
        "ldaps": lambda n: "legacy" in n,
    }

    store: dict[str, dict[str, Any]] = {}
    fits = 0
    for probe_fold, meta in fold_rows().items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]].copy()
        test = surface.loc[
            np.array([
                (fid, gid) in meta["keys"]
                for fid, gid in zip(surface["forecast_id"], surface["group_id"],
                                    strict=True)
            ])
        ].copy()
        legacy_tr, legacy_te = teach(train, test, aux_cols)
        aw_tr, aw_te = teach(train, test, aw_cols)

        rate = np.clip(train["rate"].to_numpy(dtype="float64"), 0.0, None)
        label = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
        target = one_hot_targets(rate)

        names = add_sitewind_with_basis(train, legacy_tr, aw_tr, "generic", curves)
        add_sitewind_with_basis(test, legacy_te, aw_te, "generic", curves)

        entry: dict[str, Any] = {
            "meta": test.loc[:, [*KEYS, "actual_kwh"]].copy(),
            "capacity": test["capacity"].to_numpy(dtype="float64"),
            "group": test["group_id"].to_numpy(),
        }
        arms = {
            "pooled": [*base_features, *names],
            "gfs": [*split["gfs"], *[n for n in names if sitewind_of["gfs"](n)]],
            "ldaps": [*split["ldaps"], *[n for n in names if sitewind_of["ldaps"](n)]],
        }
        entry["feature_counts"] = {k: len(v) for k, v in arms.items()}
        for arm, features in arms.items():
            dataset = lgb.Dataset(
                train.loc[:, features].astype("float32"), label=label,
                free_raw_data=False,
            )
            params = dict(PARAMS)
            params["objective"] = make_objective(target)
            booster = lgb.train(params, dataset, num_boost_round=ROUNDS)
            fits += 1
            raw = np.asarray(
                booster.predict(test.loc[:, features].astype("float32"))
            ).reshape(len(test), N_CLASS)
            exp = np.exp(raw - raw.max(axis=1, keepdims=True))
            entry[arm] = exp / exp.sum(axis=1, keepdims=True)
        store[probe_fold] = entry

    folds = sorted(store)
    disjoint = not (set(split["gfs"]) & set(split["ldaps"])
                    - {c for c in base_features if c.startswith("cal__")})
    v2 = bool(
        disjoint
        and min(len(split[s]) for s in SOURCES) >= MIN_FEATURES
    )

    def probability(fold: str, arm: str, weight: float | None = None) -> np.ndarray:
        e = store[fold]
        if arm != "stack":
            return e[arm]
        return weight * e["gfs"] + (1.0 - weight) * e["ldaps"]

    def scored(fold: str, arm: str, temperature: float,
               weight: float | None = None) -> pd.DataFrame:
        e = store[fold]
        out = e["meta"].copy()
        out["prediction_kwh"] = (
            bayes_decision(sharpen(probability(fold, arm, weight), temperature))
            * e["capacity"]
        )
        out["group_id"] = e["group"]
        out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
        return out

    results: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    chosen: dict[str, dict[str, Any]] = {}
    for arm in ("pooled", "gfs", "ldaps", "stack"):
        chosen[arm] = {}
        pieces = []
        for held in folds:
            others = [f for f in folds if f != held]
            best, best_score = None, -np.inf
            grid = (
                [(t, None) for t in TEMPERATURES] if arm != "stack"
                else [(t, w) for t in TEMPERATURES for w in WEIGHT_GRID]
            )
            for temperature, weight in grid:
                frame = pd.concat(
                    [scored(f, arm, temperature, weight) for f in others],
                    ignore_index=True,
                )
                score = official(frame)["total"]
                if score > best_score:
                    best, best_score = (temperature, weight), score
            chosen[arm][held] = {"temperature": best[0], "weight": best[1]}
            pieces.append(scored(held, arm, best[0], best[1]))
        frames[arm] = pd.concat(pieces, ignore_index=True)
        results[arm] = official(frames[arm])

    v1 = bool(abs(results["pooled"]["total"] - CONTROL) <= TOLERANCE)

    h1 = bool(results["stack"]["total"] > results["pooled"]["total"])
    best_single = max(results["gfs"]["total"], results["ldaps"]["total"])
    h2 = bool(results["stack"]["total"] > best_single)

    correlations = []
    for fold in folds:
        a = store[fold]["gfs"].ravel()
        b = store[fold]["ldaps"].ravel()
        correlations.append(float(np.corrcoef(a, b)[0, 1]))
    source_rho = float(np.mean(correlations))
    h3 = bool(source_rho < 0.89)

    gate = evaluate_gate(frames["stack"], frames["pooled"])
    gd = gate.evidence
    h4 = bool(gate.passed)
    flags = {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()}
    signature = "[" + "".join("O" if flags[g] else "-" for g in sorted(flags)) + "]"

    ficr_contrib = 0.5 * (results["stack"]["ficr"] - results["pooled"]["ficr"])
    nmae_contrib = 0.5 * (
        results["stack"]["one_minus_nmae"] - results["pooled"]["one_minus_nmae"]
    )
    h5 = bool(ficr_contrib > nmae_contrib)

    if not v1 or not v2:
        verdict = "HARNESS_OR_SPLIT_GUARD_FAILED_RESULT_VOID"
    elif h1 and h4:
        verdict = "PER_SOURCE_STACKING_WINS_AND_PASSES_GATE"
    elif h1:
        verdict = "PER_SOURCE_STACKING_WINS_GATE_REJECTS"
    else:
        verdict = "FRONT_END_FUSION_IS_NOT_THE_BOTTLENECK"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "gate_version": GATE_VERSION,
        "research": RESEARCH,
        "model_fits": fits,
        "feature_counts": {k: len(v) for k, v in split.items()},
        "arm_feature_counts": store[folds[0]]["feature_counts"],
        "arms": results,
        "chosen": chosen,
        "source_probability_rho": source_rho,
        "sitewind_rho_reference": 0.89,
        "checks": {"V1_pooled_reproduces_control": v1,
                   "V1_gap": abs(results["pooled"]["total"] - CONTROL),
                   "V2_sources_disjoint_and_large": v2},
        "hypotheses": {
            "H1_stack_beats_pooled": h1,
            "H2_stack_beats_best_single": h2,
            "H3_source_rho_below_sitewind_rho": h3,
            "H4_gate_passed": h4,
            "H5_ficr_dominant": h5,
        },
        "contributions": {"ficr": float(ficr_contrib), "nmae": float(nmae_contrib)},
        "gate": {
            "signature": signature, "flags": flags,
            "positive_months": int(gd["positive_months"]),
            "months_scored": int(gd["months_scored"]),
            "sign_test_p": float(gd["sign_test_p_greater"]),
            "median_delta": float(gd["median_total_delta"]),
            "bootstrap_q05": float(gd["block_bootstrap_q05"]),
            "min_delta": float(gd["min_total_delta"]),
        },
        "verdict": verdict,
        "dacon_upload": False,
        "external_actions": ["WebSearch", "WebFetch"],
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# M271 P4 사이클 77 — NWP 소스별 모델 + 확률 스태킹",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        "**실제 딥리서치에서 나온 노드다.**",
        "",
    ]
    for s in RESEARCH["sources"]:
        lines.append(f"- {s['finding']} — <{s['url']}> (`{s['applicability']}`)")
    lines += [
        "",
        "## 1. 팔",
        "",
        "| 팔 | 피처 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|---:|",
    ]
    counts = store[folds[0]]["feature_counts"]
    for arm in ("pooled", "gfs", "ldaps", "stack"):
        r = results[arm]
        n = counts.get(arm, "-")
        lines.append(
            f"| {arm} | {n} | {r['total']:.6f} | {r['one_minus_nmae']:.6f} | "
            f"{r['ficr']:.6f} |"
        )
    lines += [
        "",
        f"두 소스 확률행렬 상관 **{source_rho:.4f}** (sitewind 상관 0.89~0.91 대비)",
        "",
        "## 2. 타당성 가드",
        "",
        f"- V1 POOLED {results['pooled']['total']:.6f} vs 대조군 {CONTROL} -> **{v1}**",
        f"- V2 소스 서로소 + 각 {MIN_FEATURES} 개 이상 "
        f"(gfs {len(split['gfs'])} / ldaps {len(split['ldaps'])}) -> **{v2}**",
        "",
        "## 3. 사전확약",
        "",
        f"- H1 STACK > POOLED -> **{h1}** "
        f"({results['stack']['total'] - results['pooled']['total']:+.6f})",
        f"- H2 STACK > 최선단일 -> **{h2}**",
        f"- H3 소스 상관 < 0.89 -> **{h3}** ({source_rho:.4f})",
        f"- H4 게이트 통과 -> **{h4}** {signature} "
        f"({gd['positive_months']}/{gd['months_scored']} 월)",
        f"- H5 FICR 우세 -> **{h5}** "
        f"(FICR {ficr_contrib:+.6f} / 1-NMAE {nmae_contrib:+.6f})",
        "",
        "## 4. 판정",
        "",
        f"**{verdict}**",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(f"[C77] 적합 {fits} / 피처 gfs {len(split['gfs'])} ldaps {len(split['ldaps'])}")
    for arm in ("pooled", "gfs", "ldaps", "stack"):
        r = results[arm]
        print(f"[C77] {arm:7s} {r['total']:.6f} (1-NMAE {r['one_minus_nmae']:.6f} / "
              f"FICR {r['ficr']:.6f})")
    print(f"[C77] STACK - POOLED {results['stack']['total'] - results['pooled']['total']:+.6f}")
    print(f"[C77] 소스 확률 상관 {source_rho:.4f}")
    print(f"[C77] V1 {v1} / V2 {v2} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4} {signature} / H5 {h5}")
    print(f"[C77] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
