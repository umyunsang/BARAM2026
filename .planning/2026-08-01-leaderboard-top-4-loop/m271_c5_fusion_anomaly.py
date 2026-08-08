"""M271 C5 — 부호 역전의 기전: 융합 시점이 아니라 정보량을 쟀다.

**이 노드는 내가 고른 것이 아니다.** `m271_p4_route.py` 가 C1N77 의 증거 서명을 읽고
C5(`kind=anomaly`, `lane=L3`, "부호 역전의 기전")로 보냈다. C5 는 라우터 표에서
`voi=0.40` 으로 **정보가치가 가장 높은** 조건이다 — 사전확약이 빗나간 곳이 가장 많이
배우는 곳이기 때문이다.

C1N77 은 H1(STACK > POOLED)을 실행 전에 양수로 동결했고 실측은 **-0.000921** 이었다.

**① 방향 리서치 (실제 수행, 2026-08-06)**

  문헌은 조건을 명시한다 — https://dl.acm.org/doi/pdf/10.1145/3589335.3652504

    "모드 간 상관이 **클 때**는 early fusion 이 late fusion 보다 우월하고,
     각 모드가 크게 관련되지 않을 때 late fusion 이 적합하다"
    "두 모델을 따로 학습하면 **피처 간 의존성 일부가 소실된다**"

  우리 소스 확률행렬 상관은 C1N77 이 **0.8550** 으로 쟀다. GFS 와 LDAPS 는 **같은 대기**를
  보는 두 관측이므로 early fusion 조건에 해당한다. HEFTCom2024 의 DWD/GFS/MEPS 는 서로
  다른 모델군이라 상관이 낮고 late fusion 이 통했을 것이다 — 문헌의 8~30% 는 우리
  소스쌍에 그대로 전이되지 않는다.

  **그런데 '의존성 소실' 이 내 경우엔 은유가 아니라 문자 그대로다.**

**② 실제 기전 — 내 설계 결함**

  C1N77 의 V2 가드가 "두 소스 피처가 서로소" 를 요구했고, 그것을 만족시키려고 나는
  어느 쪽 접두사에도 안 걸리는 컬럼을 **양쪽에서 다 버렸다**. 버려진 20 개는 이렇다.

    sitewind__disagreement / sitewind__delta        <- **두 소스의 불일치**를 재는 신호
    geom__align__gfs10_ldaps10__cos                 <- GFS x LDAPS **교차** 정렬
    sitewind__mean / mean2 / mean3 / mean_powercurve <- 두 소스 결합 추정
    sitewind__legacy* / allweather*                  <- 팔에 배정됐어야 했으나 규칙에서 누락
    geom__align__ldaps50min_ldaps50max__cos 등       <- **LDAPS 내부** 정렬인데 접두사가
                                                       `geom__align__` 이라 LDAPS 팔에도
                                                       안 들어갔다. 순수 손실이다.

    POOLED 101 피처  vs  STACK 팔 합계 88 (cal 중복 제외)

  즉 C1N77 은 **융합 시점(early vs late)이 아니라 정보량 차이**를 쟀다. -0.000921 은
  "앞단 융합이 낫다" 가 아니라 "소스 불일치 신호를 포함해 13 개를 뺐다" 이다.
  따라서 C1N77 의 판정 `FRONT_END_FUSION_IS_NOT_THE_BOTTLENECK` 은 **그 실험이
  지지하지 못한다**. 철회한다.

**③ 교정된 사양 — 정보량을 맞추고 융합 시점만 바꾼다**

  하네스   C1N77 과 동일(teacher 복원, generic 기저, leaves 15, lr 0.1, 200 rounds).
  팔 넷
    POOLED        전 피처 101. 현행이자 V1 대조군.
    GFS_PLUS      `geom__gfs__*` + `gfs*` + `cal__*` + **공유 피처 전부**
    LDAPS_PLUS    `geom__ldaps__*` + `cal__*` + **공유 피처 전부**
    STACK_FAIR    위 둘의 확률행렬을 fold-외 가중으로 결합

  **공유 피처**  = 소스 배타적이지 않은 20 개(교차 정렬·불일치·결합 sitewind).
  두 팔에 **모두** 준다. 그러면 두 팔의 합집합 = POOLED 의 피처집합이 되고, 남는 차이는
  **한 모델이 보느냐 두 모델이 나눠 보느냐** 하나뿐이다. 그것이 융합 시점의 정의다.

  V2 를 바꾸는 것이 아니라 **질문을 바꾼다.** "서로소 분할" 은 이 질문에 필요하지 않았고,
  그것을 요구한 것이 결함이었다.

  **타당성 가드**
    V1  POOLED 가 C1N56·C1N60·C1N77 의 대조군 **0.604043 을 ±0.0005 로 재현**.
    V2  두 팔의 피처 **합집합이 POOLED 와 같다**. 정보량이 같아야 융합 시점만 비교된다.
        (교집합이 비어야 한다는 요구는 **철회**한다 — 그것이 C1N77 의 결함이었다.)
    V3  각 팔이 POOLED 보다 피처가 **적다**. 아니면 분할이 아니다.

  사전확약 (V1~V3 통과시에만 판정):
    H1  STACK_FAIR > POOLED.  정보량이 같을 때도 늦은 융합이 값을 하는가.
        **부호 예단 없음** — 문헌이 상관 0.855 에서는 early 가 우월하다고 하므로
        음수일 근거가 오히려 강하다. 그렇다면 그것이 답이고, C1N77 과 달리 이번엔
        **그 답을 지지하는 실험**이 된다.
    H2  STACK_FAIR > max(GFS_PLUS, LDAPS_PLUS). 결합이 최선 단일을 이긴다.
    H3  두 팔의 확률행렬 상관이 C1N77 의 0.8550 **보다 높다**. 공유 피처를 양쪽에
        주면 다양성이 줄어야 앞뒤가 맞는다 — 이것이 기전 검정이다.
    H4  `GFS_PLUS - gfs(C1N77 의 마른 팔)` 이 양수. 버린 피처가 실제로 정보를 갖고
        있었음을 직접 확인한다.

  H1 이 거짓이고 H3·H4 가 참이면 결론이 확정된다: **상관 0.855 의 소스쌍에서는 early
  fusion 이 옳고, C1N77 의 음수는 정보 제거 탓이었다.** 두 진술이 분리된다.

게이트 미수정. lockbox·외부데이터·`scada_ws` 예측피처 미사용. 제출 없음.
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
C77_RECEIPT = REPORTS / "m271_cycle77_per_source_stack_receipt.json"
REPORT_MD = REPORTS / "m271_c5_fusion_anomaly.md"
RECEIPT = REPORTS / "m271_c5_fusion_anomaly_receipt.json"

NODE_ID = "C1N80_FUSION_ANOMALY"
LANE = "L3"
PARENT_NODE = "C1N77_PER_SOURCE_STACK"

CONTROL = 0.604043
TOLERANCE = 0.0005
WEIGHT_GRID = tuple(round(0.1 * i, 1) for i in range(11))

RESEARCH = {
    "performed_at": "2026-08-06",
    "trigger": "라우터 C5 (부호 역전, voi=0.40 으로 표 최고)",
    "kind": "anomaly",
    "lane": "L3",
    "sources": [
        {
            "url": "https://dl.acm.org/doi/pdf/10.1145/3589335.3652504",
            "class": "peer_reviewed",
            "finding": "모드 간 상관이 클 때 early fusion 우월. 따로 학습하면 피처 간 "
                       "의존성이 소실된다",
            "applicability": "directly_supported",
        },
        {
            "url": "https://www.nature.com/articles/s41698-025-00917-6",
            "class": "peer_reviewed",
            "finding": "과적합 위험이 높은 상황에서는 late fusion 과 단순 피처선택이 적합",
            "applicability": "near_match_only",
        },
    ],
    "decision_impact": "공유 피처를 두 팔에 모두 주어 정보량을 맞추고 융합 시점만 비교",
}


def split_features(columns: list[str]) -> dict[str, list[str]]:
    """소스 배타 / 공유로 나눈다. **공유는 버리지 않고 두 팔에 모두 준다.**"""
    gfs, ldaps, shared = [], [], []
    for column in columns:
        if column.startswith(("geom__gfs__", "gfs")):
            gfs.append(column)
        elif column.startswith(("geom__ldaps__", "ldaps")):
            ldaps.append(column)
        else:
            # cal / geom__align__ / 그밖에 소스 배타가 아닌 전부.
            shared.append(column)
    return {"gfs": gfs, "ldaps": ldaps, "shared": shared}


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

        # sitewind 파생은 두 소스가 섞인 추정이므로 **공유**다. 양쪽에 모두 준다.
        arms = {
            "pooled": [*base_features, *names],
            "gfs_plus": [*split["gfs"], *split["shared"], *names],
            "ldaps_plus": [*split["ldaps"], *split["shared"], *names],
        }
        entry: dict[str, Any] = {
            "meta": test.loc[:, [*KEYS, "actual_kwh"]].copy(),
            "capacity": test["capacity"].to_numpy(dtype="float64"),
            "group": test["group_id"].to_numpy(),
            "feature_counts": {k: len(v) for k, v in arms.items()},
        }
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
    pooled_set = set(store[folds[0]]["feature_counts"])
    union = set(split["gfs"]) | set(split["ldaps"]) | set(split["shared"])
    v2 = bool(union == set(base_features))
    counts = store[folds[0]]["feature_counts"]
    v3 = bool(counts["gfs_plus"] < counts["pooled"] and counts["ldaps_plus"] < counts["pooled"])
    del pooled_set

    def probability(fold: str, arm: str, weight: float | None = None) -> np.ndarray:
        e = store[fold]
        if arm != "stack_fair":
            return e[arm]
        return weight * e["gfs_plus"] + (1.0 - weight) * e["ldaps_plus"]

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
    for arm in ("pooled", "gfs_plus", "ldaps_plus", "stack_fair"):
        chosen[arm] = {}
        pieces = []
        for held in folds:
            others = [f for f in folds if f != held]
            grid = (
                [(t, None) for t in TEMPERATURES] if arm != "stack_fair"
                else [(t, w) for t in TEMPERATURES for w in WEIGHT_GRID]
            )
            best, best_score = None, -np.inf
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

    h1 = bool(results["stack_fair"]["total"] > results["pooled"]["total"])
    best_single = max(results["gfs_plus"]["total"], results["ldaps_plus"]["total"])
    h2 = bool(results["stack_fair"]["total"] > best_single)

    rho = float(np.mean([
        np.corrcoef(store[f]["gfs_plus"].ravel(), store[f]["ldaps_plus"].ravel())[0, 1]
        for f in folds
    ]))
    c77 = json.loads(C77_RECEIPT.read_text(encoding="utf-8"))
    rho_lean = float(c77["source_probability_rho"])
    h3 = bool(rho > rho_lean)

    lean_gfs = float(c77["arms"]["gfs"]["total"])
    h4 = bool(results["gfs_plus"]["total"] > lean_gfs)

    gate = evaluate_gate(frames["stack_fair"], frames["pooled"])
    gd = gate.evidence
    flags = {label.split()[0]: bool(ok) for label, ok in gate.conditions.items()}
    signature = "[" + "".join("O" if flags[g] else "-" for g in sorted(flags)) + "]"

    if not (v1 and v2 and v3):
        verdict = "GUARD_FAILED_RESULT_VOID"
    elif h1:
        verdict = "LATE_FUSION_WINS_AT_EQUAL_INFORMATION"
    elif h3 and h4:
        verdict = "EARLY_FUSION_CORRECT_C77_NEGATIVE_WAS_INFORMATION_LOSS"
    else:
        verdict = "EARLY_FUSION_WINS_MECHANISM_UNCONFIRMED"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "triggered_by": "라우터 C5 (anomaly)",
        "gate_version": GATE_VERSION,
        "research": RESEARCH,
        "retracts": {
            "node": "C1N77_PER_SOURCE_STACK",
            "verdict": "FRONT_END_FUSION_IS_NOT_THE_BOTTLENECK",
            "why": (
                "V2 가 '서로소 분할' 을 요구해 교차·불일치 피처 20 개를 양쪽에서 버렸다. "
                "POOLED 101 vs STACK 팔 합계 88 이라 융합 시점이 아니라 정보량을 쟀다."
            ),
        },
        "model_fits": fits,
        "feature_split": {k: len(v) for k, v in split.items()},
        "arm_feature_counts": counts,
        "arms": results,
        "chosen": chosen,
        "probability_rho": rho,
        "probability_rho_c77_lean": rho_lean,
        "checks": {"V1_pooled_reproduces": v1, "V2_union_equals_pooled": v2,
                   "V3_arms_smaller_than_pooled": v3},
        "hypotheses": {
            "H1_late_fusion_wins_at_equal_info": h1,
            "H2_stack_beats_best_single": h2,
            "H3_rho_higher_than_lean_split": h3,
            "H4_shared_features_carried_information": h4,
        },
        "gate": {
            "signature": signature, "flags": flags,
            "positive_months": int(gd["positive_months"]),
            "months_scored": int(gd["months_scored"]),
            "bootstrap_q05": float(gd["block_bootstrap_q05"]),
        },
        "verdict": verdict,
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
        "# M271 C5 — 융합 부호 역전의 기전 (라우터가 지시한 노드)",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        "**내가 고른 노드가 아니다.** 라우터가 C1N77 의 부호 역전을 읽고 C5 로 보냈다 "
        "(`voi=0.40`, 표에서 정보가치 최고).",
        "",
        "## 1. C1N77 판정 철회",
        "",
        f"`{payload['retracts']['verdict']}` 를 철회한다. {payload['retracts']['why']}",
        "",
        "버려졌던 피처에는 `sitewind__disagreement`, `sitewind__delta`, "
        "`geom__align__gfs10_ldaps10__cos` 처럼 **두 소스의 불일치를 재는 신호**가 들어 "
        "있었다. 소스 결합의 핵심을 빼고 결합을 시험한 셈이다.",
        "",
        "## 2. 방향 리서치",
        "",
    ]
    for s in RESEARCH["sources"]:
        lines.append(f"- {s['finding']} — <{s['url']}> (`{s['applicability']}`)")
    lines += [
        "",
        "## 3. 교정된 비교 — 정보량을 맞춘다",
        "",
        f"소스 배타 gfs {len(split['gfs'])} / ldaps {len(split['ldaps'])} / "
        f"**공유 {len(split['shared'])}** (두 팔에 모두 준다)",
        "",
        "| 팔 | 피처 | Total | 1-NMAE | FICR |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("pooled", "gfs_plus", "ldaps_plus", "stack_fair"):
        r = results[arm]
        lines.append(
            f"| {arm} | {counts.get(arm, '-')} | {r['total']:.6f} | "
            f"{r['one_minus_nmae']:.6f} | {r['ficr']:.6f} |"
        )
    lines += [
        "",
        f"확률행렬 상관 **{rho:.4f}** (C1N77 마른 분할 {rho_lean:.4f})",
        "",
        "## 4. 사전확약",
        "",
        f"- V1 POOLED 재현 -> **{v1}**",
        f"- V2 두 팔 합집합 = POOLED -> **{v2}**",
        f"- V3 각 팔 < POOLED -> **{v3}**",
        f"- H1 STACK_FAIR > POOLED -> **{h1}** "
        f"({results['stack_fair']['total'] - results['pooled']['total']:+.6f})",
        f"- H2 결합 > 최선단일 -> **{h2}**",
        f"- H3 상관이 마른 분할보다 높다 -> **{h3}**",
        f"- H4 공유 피처가 정보를 가졌다 (gfs_plus {results['gfs_plus']['total']:.6f} > "
        f"C1N77 gfs {lean_gfs:.6f}) -> **{h4}**",
        "",
        "## 5. 판정",
        "",
        f"**{verdict}**",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== C5 완료 ===")
    print(f"[C5] 피처 배타 gfs {len(split['gfs'])} / ldaps {len(split['ldaps'])} / "
          f"공유 {len(split['shared'])}")
    print(f"[C5] 팔 피처수 {counts}")
    for arm in ("pooled", "gfs_plus", "ldaps_plus", "stack_fair"):
        r = results[arm]
        print(f"[C5] {arm:11s} {r['total']:.6f} (1-NMAE {r['one_minus_nmae']:.6f} / "
              f"FICR {r['ficr']:.6f})")
    print(f"[C5] STACK_FAIR - POOLED "
          f"{results['stack_fair']['total'] - results['pooled']['total']:+.6f}")
    print(f"[C5] 상관 {rho:.4f} (C77 마른분할 {rho_lean:.4f})")
    print(f"[C5] V1 {v1} / V2 {v2} / V3 {v3} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C5] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
