"""M271 P4 사이클 53 — 공급 데이터에서 풍속을 얼마나 더 뽑을 수 있는가.

사이클 36~52 의 sigma 사슬 전체가 하나의 양에 걸려 있다.

    sigma_cur = `gfs_spatial__idw__wind100_speed` / `ldaps_spatial__idw__wind50max_speed`
                **단일 컬럼을 선형 보정한** 잔차 (2024: 1.917 / 2.096 / 1.912)

그런데 공급 데이터에는 GFS 9 격자 x 약 34 변수, LDAPS 16 격자 x 약 29 변수가 있다.
**학습된 사상**(NWP 장 -> 나셀 풍속)이 IDW 단일 컬럼보다 훨씬 잘 뽑을 수 있다. 그렇다면
나는 외부 소스를 **공급 데이터의 약한 표현**과 겨뤄온 것이고, 사이클 51·52 의 판정 기준이
잘못 잡힌 것이 된다.

이 노드는 그것을 잰다. 사이클 42 가 복원한 teacher 를 그대로 쓴다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 42 의 `teach()`(KFold OOF, 그룹별 LightGBM)를 재사용한다.
  - **OOF 여야 한다.** 인샘플 예측으로 sigma 를 재면 과소평가된다.
  - 비교 기준은 사이클 51 이 쓴 것과 **같은 행·같은 truth**(2024, `scada_ws`)여야 한다.
    표면이 다르면 비교가 성립하지 않는다.

② 사양 동결

  표면    2024, truth `scada_ws`, 사이클 51 과 같은 유효행 정의
  기준선  IDW 두 컬럼의 비음가중 최적 결합 (사이클 51 의 `sigma_cur_realised`)
  학습형  teacher 두 프로파일(`legacy` = auxiliary 컬럼, `allweather` = 전 수치 컬럼)의
          **OOF 예측**. 사이클 42 와 동일한 파라미터·시드
  절차    각각 그룹별 선형 보정 후 잔차 표준편차

  사전확약(실행 전 동결):
    H1  학습형 teacher 의 sigma 가 IDW 기준선보다 **낮다** (세 그룹 모두).
    H2  그 감소율이 **13.3% 이상**이다 (사이클 46 이 잰 필요 감소율).
        성립하면 **공급 데이터만으로 이미 요구 풍속 정확도에 도달**한 것이고,
        사이클 36~52 의 sigma 사슬이 잘못된 기준(약한 표현)에 걸려 있었다는 뜻이다.
    H3  (귀결) H2 가 성립하면 병목은 풍속 정확도가 아니라 **풍속 -> 출력 사상**이다.
        그 경우 외부 NWP 폐쇄(사이클 52)는 유지되지만 이유가 바뀐다 — "풍속이 부족해서"가
        아니라 "풍속은 이미 충분한데 다른 데서 샌다".
    H4  teacher 두 프로파일 중 `allweather`(전 컬럼)가 `legacy`(auxiliary)보다 낮다.
        성립하면 격자·변수 폭이 실제로 기여한다는 뜻이다.

**게이트 무관. `actual_kwh` 미사용. 외부 데이터 미사용.**
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

from m271_cycle42_teacher_restored import all_weather_columns, teach
from m271_cycle50_nonnegative_weights import constrained_pair_sigma
from run_sequence_classifier import _surface

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C51_RECEIPT = REPORTS / "m271_cycle51_external_source_probe_receipt.json"
REPORT_MD = REPORTS / "m271_cycle53_supplied_extraction.md"
RECEIPT = REPORTS / "m271_cycle53_supplied_extraction_receipt.json"

NODE_ID = "C1N53_SUPPLIED_EXTRACTION"
LANE = "L2"
PARENT_NODE = "C1N52_EXTERNAL_CLOSURE"
YEAR = 2024
GFS_COL = "gfs_spatial__idw__wind100_speed"
LDAPS_COL = "ldaps_spatial__idw__wind50max_speed"
REQUIRED_REDUCTION = 1.0 - 0.8667  # 13.3%, 사이클 46


def calibrate(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100:
        return float("nan"), np.full(len(x), np.nan)
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    resid = np.full(len(x), np.nan)
    resid[ok] = y[ok] - (slope * x[ok] + intercept)
    return float(np.nanstd(resid[ok], ddof=1)), resid


def main() -> int:
    surface, _base, auxiliary = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    aux_cols = [c for c in auxiliary if c in surface.columns and c != "scada_ws"]
    aw_cols = all_weather_columns(surface)
    assert "scada_ws" not in aux_cols and "scada_ws" not in aw_cols

    # teacher 는 전 기간에서 OOF 로 학습한다(사이클 42 와 같은 절차). 그 뒤 2024 행만 평가.
    labelled = surface.loc[surface["scada_ws"].notna()].reset_index(drop=True)
    empty = labelled.iloc[0:0]
    legacy_oof, _ = teach(labelled, empty, aux_cols)
    aw_oof, _ = teach(labelled, empty, aw_cols)
    labelled["teacher_legacy"] = legacy_oof
    labelled["teacher_allweather"] = aw_oof
    labelled["teacher_mean"] = (legacy_oof + aw_oof) / 2.0

    year_rows = labelled.loc[labelled["forecast_kst_dtm"].dt.year == YEAR]

    c51 = json.loads(C51_RECEIPT.read_text(encoding="utf-8"))["result"]["per_group"]

    per_group: dict[int, Any] = {}
    for group in (1, 2, 3):
        rows = year_rows.loc[year_rows["group_id"] == group]
        needed = [
            "scada_ws", GFS_COL, LDAPS_COL,
            "teacher_legacy", "teacher_allweather", "teacher_mean",
        ]
        usable = rows.dropna(subset=needed)
        truth = usable["scada_ws"].to_numpy(dtype="float64")

        sig_gfs, res_gfs = calibrate(usable[GFS_COL].to_numpy(dtype="float64"), truth)
        sig_ldaps, res_ldaps = calibrate(usable[LDAPS_COL].to_numpy(dtype="float64"), truth)
        rho = float(np.corrcoef(res_gfs, res_ldaps)[0, 1])
        _, w = constrained_pair_sigma(sig_gfs, sig_ldaps, rho)
        res_idw = w * res_gfs + (1.0 - w) * res_ldaps
        sigma_idw = float(np.nanstd(res_idw, ddof=1))

        learned = {}
        for name in ("teacher_legacy", "teacher_allweather", "teacher_mean"):
            sig, _res = calibrate(usable[name].to_numpy(dtype="float64"), truth)
            learned[name] = {
                "sigma": sig,
                "reduction_vs_idw": 1.0 - sig / sigma_idw,
                "meets_required": bool(1.0 - sig / sigma_idw >= REQUIRED_REDUCTION),
            }
        best = min(learned, key=lambda n: learned[n]["sigma"])
        per_group[group] = {
            "rows_usable": len(usable),
            "sigma_idw_mix": sigma_idw,
            "sigma_idw_from_cycle51": c51[str(group)]["sigma_cur_realised"],
            "sigma_gfs": sig_gfs, "sigma_ldaps": sig_ldaps, "rho_gfs_ldaps": rho,
            "learned": learned,
            "best_learned": best,
            "best_sigma": learned[best]["sigma"],
            "best_reduction": learned[best]["reduction_vs_idw"],
        }

    h1 = all(v["best_sigma"] < v["sigma_idw_mix"] for v in per_group.values())
    h2 = all(v["best_reduction"] >= REQUIRED_REDUCTION for v in per_group.values())
    h3 = h2
    h4 = all(
        v["learned"]["teacher_allweather"]["sigma"] < v["learned"]["teacher_legacy"]["sigma"]
        for v in per_group.values()
    )

    if h2:
        verdict = "SUPPLIED_DATA_ALREADY_MEETS_WIND_REQUIREMENT_BOTTLENECK_ELSEWHERE"
    elif h1:
        verdict = "LEARNED_EXTRACTION_HELPS_BUT_SHORT_OF_REQUIREMENT"
    else:
        verdict = "IDW_BASELINE_WAS_ALREADY_NEAR_BEST"

    check = {
        "H1_expectation": "학습형 teacher sigma < IDW 기준선 (세 그룹)",
        "H1_held": h1,
        "H2_expectation": f"감소율 >= {REQUIRED_REDUCTION:.1%}",
        "H2_held": h2,
        "H3_expectation": "H2 성립시 병목은 풍속이 아니라 풍속->출력 사상",
        "H3_held": h3,
        "H4_expectation": "allweather 가 legacy 보다 낮다 (격자·변수 폭이 기여)",
        "H4_held": h4,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "question": "사이클 36~52 의 sigma 사슬이 공급 데이터의 **약한 표현**에 걸려 "
                    "있었는가",
        "surface": {"year": YEAR, "truth": "scada_ws", "uses_actual_kwh": False,
                    "uses_external_data": False},
        "required_reduction": REQUIRED_REDUCTION,
        "teacher_profiles": {"legacy": len(aux_cols), "allweather": len(aw_cols)},
        "per_group": per_group,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 53 — 공급 데이터에서 풍속을 얼마나 더 뽑을 수 있는가",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "- **외부 데이터·`actual_kwh` 미사용.** 공급 데이터만",
        f"- teacher 프로파일: `legacy` {len(aux_cols)} 피처 / "
        f"`allweather` {len(aw_cols)} 피처",
        "",
        "## 1. 질문",
        "",
        payload["question"] + ".",
        "",
        f"필요 감소율 **{REQUIRED_REDUCTION:.1%}** (사이클 46 실측).",
        "",
        "## 2. 측정",
        "",
        "| group | 유효행 | IDW 혼합 sigma | 학습형 최선 | **감소율** | 요구 충족 |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for g, v in per_group.items():
        lines.append(
            f"| {g} | {v['rows_usable']:,} | {v['sigma_idw_mix']:.3f} | "
            f"**{v['best_sigma']:.3f}** ({v['best_learned'].replace('teacher_','')}) | "
            f"**{v['best_reduction']:.1%}** | "
            f"{'**O**' if v['best_reduction'] >= REQUIRED_REDUCTION else 'X'} |"
        )
    lines += [
        "",
        "| group | legacy | allweather | mean |",
        "|---:|---:|---:|---:|",
    ]
    for g, v in per_group.items():
        lines.append(
            f"| {g} | {v['learned']['teacher_legacy']['sigma']:.3f} | "
            f"{v['learned']['teacher_allweather']['sigma']:.3f} | "
            f"{v['learned']['teacher_mean']['sigma']:.3f} |"
        )
    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}**",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    if h2:
        lines += [
            "## 4. 이것이 뒤집는 것",
            "",
            "공급 데이터만으로 학습한 사상이 요구 풍속 정확도에 **이미 도달**한다. 그렇다면",
            "사이클 36~52 가 기준으로 삼은 `IDW 단일 컬럼 선형보정` 은 **공급 데이터의 약한",
            "표현**이었고, 외부 소스를 그것과 겨룬 것은 잘못된 대조였다.",
            "",
            "외부 NWP 폐쇄(사이클 52)는 결론이 유지되지만 **이유가 바뀐다**: 풍속이 부족해서가",
            "아니라 **풍속은 이미 충분한데 출력 예측에서 샌다**. 병목이 `NWP -> 풍속` 이 아니라",
            "`풍속 -> 출력` 이라는 뜻이며, 후자는 파워커브·가용성·후류처럼 **공급 데이터 안**",
            "에서 다뤄야 할 문제다.",
        ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE53_SUPPLIED_EXTRACTION",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [], "model_fits": "teacher OOF",
        "lockbox_reopened": False, "new_2024_evaluation": False,
        "reads_2024_scada_not_labels": True,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    for g, v in per_group.items():
        lg = v["learned"]
        print(f"[C53] g{g} 행 {v['rows_usable']:,}  IDW {v['sigma_idw_mix']:.3f}  "
              f"legacy {lg['teacher_legacy']['sigma']:.3f}  "
              f"allweather {lg['teacher_allweather']['sigma']:.3f}  "
              f"mean {lg['teacher_mean']['sigma']:.3f}  "
              f"-> 최선 {v['best_sigma']:.3f} 감소 {v['best_reduction']:.1%}")
    print(f"[C53] H1 {h1} | H2 {h2} | H3 {h3} | H4 {h4}")
    print(f"[C53] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
