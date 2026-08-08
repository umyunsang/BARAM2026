"""M271 P4 사이클 70 — 외부 소스를 **약한 기준선**에 대고 쟀다: 결합 산술을 다시 건다.

사이클 69 가 스킬 반응곡선을 실측해 요구 감소율 **27.3%** 를 냈고(C46 의 역산 13.3% 의
2 배), 곡선 기울기로 보면 C52 가 잰 3.9~6.1% 감소는 Total **+0.0064~+0.0099** 에 해당한다.
검출문턱의 6~10 배라 외부 축을 다시 열 근거로 보였다.

그런데 C52 의 기준선을 확인하니 그게 아니다.

    C51 `sigma_cur` = GFS·LDAPS **두 원시 컬럼**을 선형보정해 최적결합한 것
                      2024 기준 1.917 / 2.096 / 1.912
    C66 teacher      = 학습형 GBM teacher 의 OOF 잔차
                      2023 dev fold 기준 **1.5866**

우리 파이프라인이 실제로 쓰는 것은 teacher 다. C52 는 외부 소스를 **우리가 쓰지 않는
약한 기준선**에 대고 쟀고, 약한 기준선은 새 소스를 실제보다 좋아 보이게 만든다
(`q = sigma_new / sigma_cur` 의 분모가 부풀려지므로).

C54 가 이미 그 격차를 쟀다 — 학습형 teacher 가 IDW 단일컬럼 대비 24.7~29.8% 낮다.
1.917 x (1 - 0.27) = 1.40 대의 값이 나오고 C66 의 1.5866 과 같은 자리다.

**기준선을 고치면 결합 이득이 어느 쪽으로 가는가.** 그것만 재면 축의 개폐가 정해진다.

**① 방법 리서치**

  - 새 방법 없음. C50 이 확정한 **비음 제약 최적 쌍결합**을 그대로 쓴다.
        `w = clip((s2^2 - rho*s1*s2) / (s1^2 + s2^2 - 2*rho*s1*s2), 0, 1)`
    C36 의 비제약식이 `rho > q` 에서 음수 가중(외삽)을 내는 결함을 C50 이 고쳤고,
    Breiman(1996) 의 비음 제약 스태킹 근거를 따른다.
  - 바뀌는 것은 `s1` 하나다 — 원시 2 컬럼 결합에서 **teacher** 로.
  - **연도 교란**을 C52 와 같은 방식으로 처리한다. teacher σ 는 2023 dev fold 에서,
    ECMWF σ 는 2024 에서 쟀으므로 절대값을 직접 섞을 수 없다. 대신 **같은 2023 행에서
    두 기준선의 비** `r = sigma_teacher / sigma_blend` 를 재고, 그 비를 2024 에
    적용한다. 비는 방법 차이를 재고 연도 차이는 약분된다.
  - **채택**: 같은 행 기준선 비 + C50 결합 + C69 반응곡선 환산.

**② 사양 동결**

  2023 기준선  확률면 캐시 v2 의 `sitewind`(teacher) 와 `scada_ws`(truth) 로 teacher σ.
               같은 행에서 `gfs_spatial__idw__wind100_speed` 와
               `ldaps_spatial__idw__wind50max_speed` 를 C51 과 **동일한 `calibrate`**
               로 보정해 C50 결합 -> blend σ. 그룹별.
  비           `r_g = sigma_teacher_g / sigma_blend_g` (2023, 같은 행)
  2024 이전    `sigma_teacher_2024_g = r_g * sigma_cur_2024_g` (C51 receipt)
  결합         `sigma_ecmwf_2024_g` 는 C51 receipt 의 값. rho 는 **teacher 잔차와
               ECMWF 잔차의 상관을 모르므로** C51 이 잰 blend-ECMWF rho 를 쓰되,
               민감도로 rho ± 0.05 도 본다. 이 대입이 이 노드의 최대 약점이며 명시한다.
  환산         C69 반응곡선의 k=1 근방 기울기로 Total 변화 추정.

  **타당성 가드**
    V1  2023 같은 행에서 blend σ 가 teacher σ 보다 **크다**(세 그룹 모두).
        작으면 teacher 가 원시결합보다 나쁘다는 뜻이고 C54 와 모순이라 결과를 버린다.
    V2  비 `r_g` 가 C54 가 잰 감소율 24.7~29.8% 와 정합 — `r_g` in [0.68, 0.80].
        벗어나면 두 계측 중 하나가 틀린 것이다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  교정 기준선에서 `q = sigma_ecmwf / sigma_teacher` 가 **1 을 넘는다**(세 그룹).
        즉 ECMWF 가 우리 teacher 보다 나쁘다.
    H2  교정 결합 감소율이 C52 의 3.9~6.1% **미만**이다. 기준선을 강하게 하면
        새 소스가 더할 것이 줄어든다.
    H3  교정 감소율을 C69 곡선으로 환산한 Total 이득이 **검출문턱 0.001013 미만**.
        참이면 외부 NWP 축이 **교정 후에도, 그리고 더 강하게** 닫힌다.
    H4  rho ± 0.05 민감도에서도 H3 이 유지된다.

  H1~H3 가 참이면 C52 의 판정은 **옳았으나 근거가 약했고**, 이 노드가 더 강한 근거로
  대체한다. H3 가 거짓이면 수집을 진행할 근거가 생긴다.

**진단 전용.** 후보 아님. **수집 없음** — C51 캐시의 기록값만 쓴다. `actual_kwh` 미사용.
게이트 미수정. 제출 없음.
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
from m271_cycle51_external_source_probe import GFS_COL, LDAPS_COL, calibrate
from m271_decision_surface import load_surface
from run_sequence_classifier import _surface

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C51_RECEIPT = REPORTS / "m271_cycle51_external_source_probe_receipt.json"
C69_RECEIPT = REPORTS / "m271_cycle69_skill_response_receipt.json"
REPORT_MD = REPORTS / "m271_cycle70_baseline_correction.md"
RECEIPT = REPORTS / "m271_cycle70_baseline_correction_receipt.json"

NODE_ID = "C1N70_BASELINE_CORRECTION"
LANE = "L2"
PARENT_NODE = "C1N69_SKILL_RESPONSE"

SOURCE = "ecmwf_ifs025"
DETECTION_THRESHOLD = 0.001013
RATIO_RANGE = (0.68, 0.80)
RHO_SENSITIVITY = (-0.05, 0.0, 0.05)
C52_REDUCTION = {1: 0.061, 2: 0.051, 3: 0.039}


def main() -> int:
    store, info = load_surface()
    c51 = json.loads(C51_RECEIPT.read_text(encoding="utf-8"))
    c51 = c51.get("result", c51)
    c69 = json.loads(C69_RECEIPT.read_text(encoding="utf-8"))

    # --- 2023 같은 행에서 두 기준선 ------------------------------------------
    surface, _base, _aux = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    keep = ["forecast_id", "group_id", GFS_COL, LDAPS_COL]
    lookup = surface.loc[:, keep].copy()

    parts: list[pd.DataFrame] = []
    for _fold, entry in store.items():
        frame = entry["meta"].loc[:, ["forecast_id"]].copy()
        frame["group_id"] = entry["group"]
        frame["sitewind"] = entry["sitewind"]
        frame["scada_ws"] = entry["scada_ws"]
        parts.append(frame)
    cache = pd.concat(parts, ignore_index=True)
    joined = cache.merge(lookup, on=["forecast_id", "group_id"], how="left")
    joined = joined.dropna(subset=["scada_ws", "sitewind", GFS_COL, LDAPS_COL])

    per_group: dict[str, Any] = {}
    for group in (1, 2, 3):
        block = joined.loc[joined["group_id"] == group]
        truth = block["scada_ws"].to_numpy(dtype="float64")
        sigma_teacher = float(
            np.std(truth - block["sitewind"].to_numpy(dtype="float64"), ddof=1)
        )
        sig_gfs, res_gfs = calibrate(block[GFS_COL].to_numpy(dtype="float64"), truth)
        sig_ldaps, res_ldaps = calibrate(block[LDAPS_COL].to_numpy(dtype="float64"), truth)
        rho_supplied = float(np.corrcoef(res_gfs, res_ldaps)[0, 1])
        _formula, w = constrained_pair_sigma(sig_gfs, sig_ldaps, rho_supplied)
        sigma_blend = float(np.std(w * res_gfs + (1.0 - w) * res_ldaps, ddof=1))
        per_group[str(group)] = {
            "rows_2023": int(len(block)),
            "sigma_teacher_2023": sigma_teacher,
            "sigma_blend_2023": sigma_blend,
            "sigma_gfs_2023": sig_gfs,
            "sigma_ldaps_2023": sig_ldaps,
            "ratio": sigma_teacher / sigma_blend,
        }

    v1 = bool(all(
        per_group[str(g)]["sigma_blend_2023"] > per_group[str(g)]["sigma_teacher_2023"]
        for g in (1, 2, 3)
    ))
    v2 = bool(all(
        RATIO_RANGE[0] <= per_group[str(g)]["ratio"] <= RATIO_RANGE[1]
        for g in (1, 2, 3)
    ))

    # --- 2024 로 이전해 결합 재계산 -------------------------------------------
    c51_groups = c51["per_group"]
    for group in (1, 2, 3):
        raw = c51_groups[str(group)]
        sigma_cur = float(raw["sigma_cur_realised"])
        source = raw["sources"][SOURCE]
        sigma_new = float(source["sigma_new"])
        rho = float(source["rho_vs_current"])
        ratio = per_group[str(group)]["ratio"]
        sigma_teacher_2024 = ratio * sigma_cur

        naive_sigma, naive_w = constrained_pair_sigma(sigma_cur, sigma_new, rho)
        rows = {}
        for delta in RHO_SENSITIVITY:
            r = float(np.clip(rho + delta, -0.99, 0.99))
            corrected_sigma, corrected_w = constrained_pair_sigma(
                sigma_teacher_2024, sigma_new, r
            )
            reduction = 1.0 - corrected_sigma / sigma_teacher_2024
            rows[f"rho{delta:+.2f}"] = {
                "rho": r,
                "weight_on_new": corrected_w,
                "combined_sigma": corrected_sigma,
                "reduction": reduction,
            }
        per_group[str(group)].update({
            "sigma_cur_2024": sigma_cur,
            "sigma_teacher_2024": sigma_teacher_2024,
            "sigma_ecmwf_2024": sigma_new,
            "q_against_blend": sigma_new / sigma_cur,
            "q_against_teacher": sigma_new / sigma_teacher_2024,
            "rho_reported": rho,
            "naive_reduction": 1.0 - naive_sigma / sigma_cur,
            "naive_weight_on_new": naive_w,
            "c52_reduction": C52_REDUCTION[group],
            "sensitivity": rows,
        })

    h1 = bool(all(per_group[str(g)]["q_against_teacher"] > 1.0 for g in (1, 2, 3)))
    h2 = bool(all(
        per_group[str(g)]["sensitivity"]["rho+0.00"]["reduction"]
        < per_group[str(g)]["c52_reduction"]
        for g in (1, 2, 3)
    ))

    # --- C69 반응곡선으로 환산 -------------------------------------------------
    response = pd.DataFrame(c69["response"])
    near = response.loc[response["k"].isin([0.9, 1.0])].sort_values("k")
    slope = float(
        (near["total"].iloc[0] - near["total"].iloc[1])
        / (near["k"].iloc[1] - near["k"].iloc[0])
    )
    mean_reduction = float(np.mean([
        per_group[str(g)]["sensitivity"]["rho+0.00"]["reduction"] for g in (1, 2, 3)
    ]))
    gain = mean_reduction * slope
    h3 = bool(gain < DETECTION_THRESHOLD)

    worst_gain = max(
        float(np.mean([
            per_group[str(g)]["sensitivity"][key]["reduction"] for g in (1, 2, 3)
        ])) * slope
        for key in ("rho-0.05", "rho+0.00", "rho+0.05")
    )
    h4 = bool(worst_gain < DETECTION_THRESHOLD)

    if not v1:
        verdict = "TEACHER_NOT_BETTER_THAN_BLEND_CONTRADICTS_C54_VOID"
    elif not v2:
        verdict = "RATIO_INCONSISTENT_WITH_C54_VOID"
    elif h3 and h4:
        verdict = "EXTERNAL_NWP_CLOSED_HARDER_AGAINST_CORRECT_BASELINE"
    elif h1:
        verdict = "SOURCE_WORSE_THAN_TEACHER_BUT_GAIN_ABOVE_THRESHOLD"
    else:
        verdict = "CORRECTED_BASELINE_STILL_LEAVES_ROOM"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "method": "NONNEGATIVE_PAIR_COMBINATION (Breiman 1996; C1N50) on corrected baseline",
        "surface": info,
        "source": SOURCE,
        "per_group": per_group,
        "response_slope_near_k1": slope,
        "mean_corrected_reduction": mean_reduction,
        "estimated_total_gain": gain,
        "worst_case_gain": worst_gain,
        "detection_threshold": DETECTION_THRESHOLD,
        "checks": {"V1_blend_worse_than_teacher": v1, "V2_ratio_matches_c54": v2},
        "hypotheses": {
            "H1_source_worse_than_teacher": h1,
            "H2_reduction_below_c52": h2,
            "H3_gain_below_detection": h3,
            "H4_robust_to_rho": h4,
        },
        "limitation": (
            "rho 를 blend-ECMWF 상관으로 대입했다. teacher 잔차와 ECMWF 잔차의 상관은 "
            "직접 재지 않았고, 그러려면 2024 에 teacher 를 돌려야 한다. teacher 가 "
            "blend 보다 정교하므로 실제 rho 는 더 **낮을** 수 있고 그러면 결합 이득이 "
            "커진다 — 민감도 rho-0.05 가 그 방향을 본다."
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
        "# M271 P4 사이클 70 — 외부 소스를 교정된 기준선에 다시 대다",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용, 수집 없음**",
        "",
        "## 1. 2023 같은 행에서 두 기준선",
        "",
        "| 그룹 | 행 | teacher sigma | blend sigma | 비 r |",
        "|---:|---:|---:|---:|---:|",
    ]
    for group in (1, 2, 3):
        b = per_group[str(group)]
        lines.append(
            f"| {group} | {b['rows_2023']:,} | **{b['sigma_teacher_2023']:.4f}** | "
            f"{b['sigma_blend_2023']:.4f} | {b['ratio']:.4f} |"
        )
    lines += [
        "",
        "## 2. 2024 로 이전한 결합",
        "",
        "| 그룹 | sigma_cur (C51) | sigma_teacher | sigma_ECMWF | q(blend) | "
        "**q(teacher)** | C52 감소 | **교정 감소** |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in (1, 2, 3):
        b = per_group[str(group)]
        corrected = b["sensitivity"]["rho+0.00"]["reduction"]
        lines.append(
            f"| {group} | {b['sigma_cur_2024']:.3f} | {b['sigma_teacher_2024']:.3f} | "
            f"{b['sigma_ecmwf_2024']:.3f} | {b['q_against_blend']:.3f} | "
            f"**{b['q_against_teacher']:.3f}** | {b['c52_reduction']:.1%} | "
            f"**{corrected:.2%}** |"
        )
    lines += [
        "",
        "## 3. rho 민감도",
        "",
        "| 그룹 | rho-0.05 | rho | rho+0.05 |",
        "|---:|---:|---:|---:|",
    ]
    for group in (1, 2, 3):
        b = per_group[str(group)]["sensitivity"]
        lines.append(
            f"| {group} | {b['rho-0.05']['reduction']:.2%} | "
            f"{b['rho+0.00']['reduction']:.2%} | {b['rho+0.05']['reduction']:.2%} |"
        )
    lines += [
        "",
        "## 4. C69 반응곡선 환산",
        "",
        f"- `k=1` 근방 기울기 **{slope:.4f}** Total/단위 k",
        f"- 교정 평균 감소율 **{mean_reduction:.2%}**",
        f"- 추정 Total 이득 **{gain:+.6f}** (검출문턱 {DETECTION_THRESHOLD})",
        f"- rho 민감도 최선 **{worst_gain:+.6f}**",
        "",
        "## 5. 사전확약",
        "",
        f"- V1 blend sigma > teacher sigma -> **{v1}**",
        f"- V2 비 r in {RATIO_RANGE} (C54 정합) -> **{v2}**",
        f"- H1 q(teacher) > 1 -> **{h1}**",
        f"- H2 교정 감소 < C52 감소 -> **{h2}**",
        f"- H3 이득 < 검출문턱 -> **{h3}**",
        f"- H4 rho 민감도에서도 유지 -> **{h4}**",
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
    for group in (1, 2, 3):
        b = per_group[str(group)]
        print(f"[C70] g{group} 2023  teacher {b['sigma_teacher_2023']:.4f} / blend "
              f"{b['sigma_blend_2023']:.4f} / 비 {b['ratio']:.4f}")
    for group in (1, 2, 3):
        b = per_group[str(group)]
        print(f"[C70] g{group} 2024  q(blend) {b['q_against_blend']:.3f} -> q(teacher) "
              f"{b['q_against_teacher']:.3f}  |  C52 {b['c52_reduction']:.1%} -> 교정 "
              f"{b['sensitivity']['rho+0.00']['reduction']:.2%}")
    print(f"[C70] V1 {v1} / V2 {v2} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C70] 기울기 {slope:.4f} x 감소 {mean_reduction:.2%} = Total {gain:+.6f} "
          f"(문턱 {DETECTION_THRESHOLD})")
    print(f"[C70] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
