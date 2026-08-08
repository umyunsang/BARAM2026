"""M271 P4 사이클 71 — teacher 편성에 0.5 가 박혀 있다: 최적 가중을 재고 값을 매긴다.

사이클 70 이 V2 가드로 스스로 무효가 됐고, 그 발화가 결함을 드러냈다.

    C54  teacher(allweather) 잔차   1.373 / 1.502 / **1.466**
    C70  `sitewind__mean` 잔차       1.496 / 1.600 / **1.657**

`sitewind__mean` 은 `m271_cycle42_teacher_restored.py` 에서 이렇게 만들어진다.

    matrix["sitewind__mean"] = (legacy + allweather) / 2.0

**단순 50/50 평균이다.** 두 추정량의 오차 크기가 다르고 상관이 1 이 아니면 0.5 는
최적이 아니다. 그리고 이 프로젝트는 최적 가중식을 이미 갖고 있다 — 사이클 50 이
Breiman(1996) 의 비음 제약을 걸어 확정했다.

    w = clip( (s2^2 - rho*s1*s2) / (s1^2 + s2^2 - 2*rho*s1*s2), 0, 1 )

C50 은 그 식을 **외부 소스 결합**에만 썼고 정작 내부 teacher 편성에는 안 썼다.
같은 세션에서 확립한 도구를 바로 옆에서 안 쓴 것이다.

C69 의 반응곡선이 값을 매긴다 — k=1 근방 기울기 0.164/단위이므로, g3 의 13% 격차가
그대로 회수되면 Total **+0.019** 규모다. 이번 세션에서 잰 어떤 것보다 크다.

**① 방법 리서치**

  - 새 방법 없음. C50 의 비음 제약 쌍결합을 **같은 형태로** 내부 teacher 에 적용한다.
    Breiman(1996) 의 stacked regression 이 비음 제약을 요구하는 이유(외삽 방지)가
    여기서도 같다.
  - 가중 선택의 **표본외 규율**이 관건이다. 같은 행에서 rho·sigma 를 재고 그 행에서
    평가하면 표본내 최적이라 이득이 부풀려진다. C44 가 온도 선택에서 쓴
    **fold-외 선택**을 그대로 쓴다 — 보류 fold 의 가중은 나머지 두 fold 에서 고른다.
  - **채택**: fold-외 비음 제약 최적가중 + C69 반응곡선 환산.

**② 사양 동결**

  입력   확률면 캐시 **v3** (`sitewind_legacy`, `sitewind_allweather`, `sitewind`,
         `scada_ws`). 네 값이 모두 있는 행만.
  잔차   `scada_ws - 추정` 을 그룹별로. 선형보정 없이 **원잔차**를 쓴다
         (teacher 는 이미 `scada_ws` 를 표적으로 학습했으므로 보정이 불필요하고,
         보정하면 배포 불가능한 자유도를 넣는 셈이다).
  가중   그룹별로 `s_legacy`, `s_allweather`, `rho` 를 **나머지 두 fold**에서 재고
         C50 식으로 `w*` 를 구해 **보류 fold**에 적용. 실제 배포에서도 학습기간
         잔차로 고를 수 있으므로 실현 가능한 절차다.
  팔     MEAN(현행 0.5) / OPTIMAL(fold-외 w*) / ALLWEATHER(w=0) / LEGACY(w=1)
         마지막 둘은 경계 대조군이다.

  **타당성 가드**
    V1  MEAN 팔의 잔차가 `sitewind` 컬럼의 잔차와 **1e-9 이내** 일치.
        `(legacy+allweather)/2` 재구성이 캐시된 값과 같아야 한다. 어긋나면
        내가 편성 식을 잘못 읽은 것이다.
    V2  C70 이 잰 MEAN 잔차(1.496/1.600/1.657)를 ±0.01 이내로 재현.

  사전확약 (V1·V2 통과시에만 판정):
    H1  세 그룹 모두 `s_allweather < s_legacy`. C54 가 allweather 를 teacher 로 쓴
        근거이고, 참이면 최적 w* 가 0.5 미만(allweather 쪽)이어야 한다.
    H2  fold-외 OPTIMAL 이 MEAN 보다 잔차가 **작다**(세 그룹). 핵심.
    H3  OPTIMAL 이 ALLWEATHER 보다도 작다. 즉 결합이 최선단일을 이긴다.
        **부호 예단 없음** — C34 가 다른 후보군에서 "결합이 최고단일에 더할 것 없음"
        을 이미 봤다. 여기서도 그러면 답은 "그냥 allweather 를 써라" 가 된다.
    H4  g3 의 개선이 가장 크다. C70 이 격차를 g3 에서 가장 크게 쟀다.
    H5  MEAN 대비 OPTIMAL 의 감소율을 C69 곡선으로 환산한 Total 이득이
        **검출문턱 0.001013 을 넘는다**.

  H2·H5 가 참이면 배포 후보가 되고, 그때는 **별도 사이클에서 실제로 학습·채점**해
  동결 게이트에 건다. 이 노드는 **풍속 잔차 층에서만** 재며 Total 이득은 환산 추정이다.
  환산은 커브기계 면에서 모형 면으로의 이전을 가정하므로(C33·C45 에서 두 번 틀린
  가정) **이 노드로 승격 판정을 하지 않는다.**

**진단 전용.** 후보 아님(승격은 별도 사이클). `scada_ws` 는 학습기간 전용이나
**가중 선택에만** 쓰이고 예측 피처가 되지 않는다 — 배포 시에도 학습기간 잔차로
같은 절차를 밟을 수 있다. 게이트 미수정. 제출 없음.
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

from m271_cycle50_nonnegative_weights import constrained_pair_sigma
from m271_decision_surface import load_surface

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C69_RECEIPT = REPORTS / "m271_cycle69_skill_response_receipt.json"
C70_RECEIPT = REPORTS / "m271_cycle70_baseline_correction_receipt.json"
REPORT_MD = REPORTS / "m271_cycle71_teacher_weight.md"
RECEIPT = REPORTS / "m271_cycle71_teacher_weight_receipt.json"

NODE_ID = "C1N71_TEACHER_WEIGHT"
LANE = "L2"
PARENT_NODE = "C1N70_BASELINE_CORRECTION"

DETECTION_THRESHOLD = 0.001013
IDENTITY_TOLERANCE = 1e-9
REPRO_TOLERANCE = 0.01
ARMS = ("mean", "optimal", "allweather", "legacy")


def main() -> int:
    store, info = load_surface()
    c69 = json.loads(C69_RECEIPT.read_text(encoding="utf-8"))
    c70 = json.loads(C70_RECEIPT.read_text(encoding="utf-8"))

    folds = sorted(store)
    frames: dict[str, pd.DataFrame] = {}
    for fold in folds:
        entry = store[fold]
        frame = pd.DataFrame({
            "group_id": entry["group"],
            "scada_ws": entry["scada_ws"],
            "mean": entry["sitewind"],
            "legacy": entry["sitewind_legacy"],
            "allweather": entry["sitewind_allweather"],
        })
        frames[fold] = frame.dropna(
            subset=["scada_ws", "mean", "legacy", "allweather"]
        ).reset_index(drop=True)

    pooled = pd.concat(frames.values(), ignore_index=True)
    rebuilt = (pooled["legacy"] + pooled["allweather"]) / 2.0
    identity = float((rebuilt - pooled["mean"]).abs().max())
    v1 = bool(identity <= IDENTITY_TOLERANCE)

    def residual(frame: pd.DataFrame, weight: float) -> np.ndarray:
        blended = weight * frame["legacy"] + (1.0 - weight) * frame["allweather"]
        return (frame["scada_ws"] - blended).to_numpy(dtype="float64")

    per_group: dict[str, Any] = {}
    chosen: dict[str, dict[str, float]] = {}
    for group in (1, 2, 3):
        block = {f: frames[f].loc[frames[f]["group_id"] == group] for f in folds}
        whole = pd.concat(block.values(), ignore_index=True)
        s_legacy = float(np.std(residual(whole, 1.0), ddof=1))
        s_allweather = float(np.std(residual(whole, 0.0), ddof=1))
        rho = float(np.corrcoef(residual(whole, 1.0), residual(whole, 0.0))[0, 1])

        chosen[str(group)] = {}
        pieces: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
        for held in folds:
            others = pd.concat(
                [block[f] for f in folds if f != held], ignore_index=True
            )
            s1 = float(np.std(residual(others, 1.0), ddof=1))
            s2 = float(np.std(residual(others, 0.0), ddof=1))
            r = float(np.corrcoef(residual(others, 1.0), residual(others, 0.0))[0, 1])
            _sigma, w = constrained_pair_sigma(s1, s2, r)
            chosen[str(group)][held] = float(w)
            pieces["mean"].append(residual(block[held], 0.5))
            pieces["optimal"].append(residual(block[held], w))
            pieces["allweather"].append(residual(block[held], 0.0))
            pieces["legacy"].append(residual(block[held], 1.0))

        sigmas = {
            arm: float(np.std(np.concatenate(parts), ddof=1))
            for arm, parts in pieces.items()
        }
        per_group[str(group)] = {
            "rows": int(len(whole)),
            "sigma_legacy_pooled": s_legacy,
            "sigma_allweather_pooled": s_allweather,
            "rho_pooled": rho,
            "sigma": sigmas,
            "reduction_optimal_vs_mean": 1.0 - sigmas["optimal"] / sigmas["mean"],
            "reduction_allweather_vs_mean": 1.0 - sigmas["allweather"] / sigmas["mean"],
            "chosen_weight_on_legacy": chosen[str(group)],
        }

    c70_mean = {g: c70["per_group"][g]["sigma_teacher_2023"] for g in ("1", "2", "3")}
    repro = max(
        abs(per_group[g]["sigma"]["mean"] - c70_mean[g]) for g in ("1", "2", "3")
    )
    v2 = bool(repro <= REPRO_TOLERANCE)

    h1 = bool(all(
        per_group[g]["sigma_allweather_pooled"] < per_group[g]["sigma_legacy_pooled"]
        for g in ("1", "2", "3")
    ))
    h2 = bool(all(
        per_group[g]["sigma"]["optimal"] < per_group[g]["sigma"]["mean"]
        for g in ("1", "2", "3")
    ))
    h3 = bool(all(
        per_group[g]["sigma"]["optimal"] < per_group[g]["sigma"]["allweather"]
        for g in ("1", "2", "3")
    ))
    best_group = max(
        ("1", "2", "3"), key=lambda g: per_group[g]["reduction_optimal_vs_mean"]
    )
    h4 = bool(best_group == "3")

    response = pd.DataFrame(c69["response"])
    near = response.loc[response["k"].isin([0.9, 1.0])].sort_values("k")
    slope = float(
        (near["total"].iloc[0] - near["total"].iloc[1])
        / (near["k"].iloc[1] - near["k"].iloc[0])
    )
    mean_reduction = float(np.mean([
        per_group[g]["reduction_optimal_vs_mean"] for g in ("1", "2", "3")
    ]))
    allweather_reduction = float(np.mean([
        per_group[g]["reduction_allweather_vs_mean"] for g in ("1", "2", "3")
    ]))
    gain = mean_reduction * slope
    allweather_gain = allweather_reduction * slope
    h5 = bool(gain > DETECTION_THRESHOLD)

    if not v1:
        verdict = "MEAN_RECONSTRUCTION_FAILED_RESULT_VOID"
    elif not v2:
        verdict = "C70_MEAN_NOT_REPRODUCED_RESULT_VOID"
    elif h2 and h5 and h3:
        verdict = "OPTIMAL_WEIGHT_WORTH_PROMOTING"
    elif h2 and h5:
        verdict = "USE_ALLWEATHER_ALONE_COMBINATION_ADDS_NOTHING"
    elif h2:
        verdict = "OPTIMAL_BEATS_MEAN_BUT_BELOW_DETECTION"
    else:
        verdict = "FIFTY_FIFTY_MEAN_IS_ADEQUATE"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "method": "NONNEGATIVE_PAIR_COMBINATION (Breiman 1996; C1N50) with fold-out selection",
        "surface": info,
        "per_group": per_group,
        "checks": {
            "V1_mean_identity_max": identity, "V1_pass": v1,
            "V2_c70_repro_max": repro, "V2_pass": v2,
        },
        "response_slope_near_k1": slope,
        "mean_reduction_optimal": mean_reduction,
        "mean_reduction_allweather": allweather_reduction,
        "estimated_total_gain_optimal": gain,
        "estimated_total_gain_allweather": allweather_gain,
        "detection_threshold": DETECTION_THRESHOLD,
        "hypotheses": {
            "H1_allweather_better_than_legacy": h1,
            "H2_optimal_beats_mean": h2,
            "H3_optimal_beats_allweather": h3,
            "H4_g3_largest_improvement": h4,
            "H5_gain_above_detection": h5,
        },
        "limitation": (
            "Total 이득은 C69 반응곡선(커브기계 면)으로 환산한 **추정**이다. 모형 면으로의 "
            "이전을 가정하며 C33·C45 에서 두 번 틀린 가정이므로, 이 노드로 승격 판정을 "
            "하지 않는다. 승격하려면 별도 사이클에서 실제로 학습·채점해 동결 게이트에 건다."
        ),
        "verdict": verdict,
        "no_collection": True,
        "no_training": True,
        "dacon_upload": False,
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# M271 P4 사이클 71 — teacher 편성 가중",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        f"확률면 spec `{info['spec_digest']}` / prob `{info['probability_digest']}`",
        "",
        "## 1. 성분과 상관",
        "",
        "| 그룹 | 행 | s(legacy) | s(allweather) | rho |",
        "|---:|---:|---:|---:|---:|",
    ]
    for group in ("1", "2", "3"):
        b = per_group[group]
        lines.append(
            f"| {group} | {b['rows']:,} | {b['sigma_legacy_pooled']:.4f} | "
            f"{b['sigma_allweather_pooled']:.4f} | {b['rho_pooled']:.4f} |"
        )
    lines += [
        "",
        "## 2. 팔별 잔차 (fold-외 가중)",
        "",
        "| 그룹 | MEAN(0.5) | OPTIMAL | ALLWEATHER | LEGACY | OPT 감소 | AW 감소 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("1", "2", "3"):
        b = per_group[group]
        s = b["sigma"]
        lines.append(
            f"| {group} | {s['mean']:.4f} | **{s['optimal']:.4f}** | "
            f"{s['allweather']:.4f} | {s['legacy']:.4f} | "
            f"{b['reduction_optimal_vs_mean']:.2%} | "
            f"{b['reduction_allweather_vs_mean']:.2%} |"
        )
    lines += [
        "",
        "## 3. 선택된 가중 (legacy 쪽, fold-외)",
        "",
        "```",
        json.dumps(chosen, indent=1, ensure_ascii=False),
        "```",
        "",
        "## 4. C69 환산",
        "",
        f"- 기울기 **{slope:.4f}** Total/단위 k",
        f"- OPTIMAL 평균 감소 **{mean_reduction:.2%}** -> Total **{gain:+.6f}**",
        f"- ALLWEATHER 평균 감소 **{allweather_reduction:.2%}** -> Total "
        f"**{allweather_gain:+.6f}**",
        f"- 검출문턱 {DETECTION_THRESHOLD}",
        "",
        "## 5. 사전확약",
        "",
        f"- V1 `mean` 재구성 최대차 {identity:.2e} -> **{v1}**",
        f"- V2 C70 재현 최대차 {repro:.4f} -> **{v2}**",
        f"- H1 allweather < legacy -> **{h1}**",
        f"- H2 OPTIMAL < MEAN -> **{h2}**",
        f"- H3 OPTIMAL < ALLWEATHER -> **{h3}**",
        f"- H4 g3 개선 최대 (최대 그룹 {best_group}) -> **{h4}**",
        f"- H5 이득 > 검출문턱 -> **{h5}**",
        "",
        "## 6. 판정",
        "",
        f"**{verdict}**",
        "",
        payload["limitation"],
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(f"[C71] V1 {v1} (최대차 {identity:.2e}) / V2 {v2} (최대차 {repro:.4f})")
    for group in ("1", "2", "3"):
        b = per_group[group]
        s = b["sigma"]
        print(f"[C71] g{group}  legacy {b['sigma_legacy_pooled']:.4f} / allweather "
              f"{b['sigma_allweather_pooled']:.4f} / rho {b['rho_pooled']:.4f}")
        print(f"[C71]      MEAN {s['mean']:.4f} -> OPTIMAL {s['optimal']:.4f} "
              f"({b['reduction_optimal_vs_mean']:+.2%})  |  ALLWEATHER {s['allweather']:.4f} "
              f"({b['reduction_allweather_vs_mean']:+.2%})")
    print(f"[C71] 환산  OPTIMAL {mean_reduction:.2%} -> Total {gain:+.6f}  |  "
          f"ALLWEATHER {allweather_reduction:.2%} -> Total {allweather_gain:+.6f}")
    print(f"[C71] H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4} / H5 {h5}")
    print(f"[C71] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
