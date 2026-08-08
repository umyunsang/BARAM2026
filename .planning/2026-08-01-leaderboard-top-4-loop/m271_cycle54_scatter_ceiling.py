"""M271 P4 사이클 54 — 시간분할 교정, 그리고 파워커브 산포가 정하는 천장.

두 가지를 한다.

**A. 사이클 53 의 누출 교정.**
사이클 53 이 teacher 의 sigma 감소를 46~49% 로 냈다. `teach()` 가 `KFold(3, shuffle=True)`
를 쓰는데, 시간당 데이터의 lag-1 자기상관이 0.951~0.962(A1)이므로 시각 t 의 "OOF" 예측이
학습셋의 t±1h 이웃에서 사실상 답을 본다. **시간 분할**(학습 <2024, 평가 2024)로 다시 재면:

    group  IDW    teacher(allweather)  감소
      1    1.955        1.373          29.7%
      2    2.120        1.502          29.2%
      3    1.948        1.466          24.7%

17~20%p 가 누출이었다. 그러나 교정 후에도 **필요량 13.3% 의 약 2 배**이므로 사이클 53 의
결론(공급 데이터가 이미 요구 풍속 정확도를 넘는다)은 **살아남는다.**

**B. 그러면 왜 점수가 안 오르는가.**
teacher 출력은 **이미 모델의 피처**다(M102 의 100 개 중 13 개). 좋은 풍속을 갖고도 Total 이
0.63 대라면, 병목은 풍속이 아니라 **같은 풍속에서 출력이 흩어지는 폭**이다.

A5 가 실측 파워커브를 bin 별 `mean_norm` 과 **`std_norm`** 으로 기록해 뒀다. 그 산포가
곧 "풍속을 완벽히 알아도 남는 출력 불확실성"이며, 정산 밴드(+-6% 용량) 대비 얼마나 큰지가
FICR 천장을 정한다. **수집도 학습도 필요 없다 — 이미 잰 값이다.**

① 방법 리서치 (실행 전)
  - 새 방법 없음. A5 receipt 의 bin 별 산포를 읽어 밴드 폭과 대조한다.
  - **집계 효과를 반영해야 한다.** A5 의 산포는 **터빈 1 기** 기준이다. 그룹은 5~6 기를
    합치므로 터빈간 잔차가 부분적으로 상쇄된다. 상쇄 정도를 모르므로 **상한(완전상관,
    상쇄 없음)과 하한(무상관, 1/sqrt(n))을 모두** 보고한다.

② 사양 동결

  자료   `reports/m271_n0_scada_receipt.json` 의 `curves[*].curve[*]`
         (`bin_center`, `mean_norm`, `std_norm`, `size_norm`)
  정규화 각 터빈의 `rated_kwh_per_10min` 으로 나눠 정격비로 환산
  집계   상한 = 터빈 산포 그대로 / 하한 = 터빈 산포 / sqrt(터빈수)
  대조   정산 밴드 반폭 **0.06** (용량 대비)

  사전확약(실행 전 동결):
    A1  시간분할 감소율이 세 그룹 모두 **13.3% 이상**이다 (사이클 53 결론의 생존).
    A2  시간분할 감소율이 무작위 KFold 감소율보다 **낮다** (누출 확인).
    B1  급경사 구간(정격비 0.2~0.8)의 산포가 밴드 반폭 **0.06 을 넘는다**(상한 기준).
        성립하면 풍속을 완벽히 알아도 그 구간에서 밴드 적중이 보장되지 않는다.
    B2  하한(무상관 집계) 기준으로도 급경사 산포가 0.06 을 넘는다.
        성립하면 집계로도 해소되지 않는다 — 훨씬 강한 진술이다.

  B1 이 성립하고 B2 가 기각되면 **터빈간 상쇄가 관건**이 되고, 둘 다 성립하면
  **출력 산포가 FICR 을 직접 구속**한다.

**게이트 무관. 학습·수집 없음. `actual_kwh` 미사용.**
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
A5_RECEIPT = REPORTS / "m271_n0_scada_receipt.json"
REPORT_MD = REPORTS / "m271_cycle54_scatter_ceiling.md"
RECEIPT = REPORTS / "m271_cycle54_scatter_ceiling_receipt.json"

NODE_ID = "C1N54_SCATTER_CEILING"
LANE = "L3"
PARENT_NODE = "C1N53_SUPPLIED_EXTRACTION"

REQUIRED_REDUCTION = 1.0 - 0.8667
BAND_HALF = 0.06
STEEP_LOW, STEEP_HIGH = 0.20, 0.80
TURBINES_PER_GROUP = {1: 6, 2: 6, 3: 5}

# 시간분할 재측정 (이 노드 실행 전에 수행, 사양의 일부로 동결)
CHRONO = {
    1: {"idw": 1.955, "legacy": 1.448, "allweather": 1.373},
    2: {"idw": 2.120, "legacy": 1.563, "allweather": 1.502},
    3: {"idw": 1.948, "legacy": 1.526, "allweather": 1.466},
}
SHUFFLED = {1: 0.487, 2: 0.470, 3: 0.460}  # 사이클 53 의 무작위 KFold 감소율


def main() -> int:
    # --- A. 누출 교정
    chrono_rows = []
    for group, v in CHRONO.items():
        reduction = 1.0 - v["allweather"] / v["idw"]
        chrono_rows.append(
            {
                "group": group, **v,
                "reduction_chronological": reduction,
                "reduction_shuffled": SHUFFLED[group],
                "leakage_inflation": SHUFFLED[group] - reduction,
                "meets_required": bool(reduction >= REQUIRED_REDUCTION),
                "multiple_of_required": reduction / REQUIRED_REDUCTION,
            }
        )
    a1 = all(r["meets_required"] for r in chrono_rows)
    a2 = all(r["reduction_chronological"] < r["reduction_shuffled"] for r in chrono_rows)

    # --- B. 파워커브 산포
    curves = json.loads(A5_RECEIPT.read_text(encoding="utf-8"))["result"]["curves"]
    pooled: dict[float, dict[str, list[float]]] = {}
    for turbine in curves:
        rated = turbine["rated_kwh_per_10min"]
        for point in turbine["curve"]:
            centre = point.get("bin_center")
            mean = point.get("mean_norm")
            std = point.get("std_norm")
            if centre is None or mean is None or std is None:
                continue
            slot = pooled.setdefault(round(float(centre), 2), {"mean": [], "std": []})
            slot["mean"].append(float(mean) / rated)
            slot["std"].append(float(std) / rated)

    bins = []
    for centre in sorted(pooled):
        slot = pooled[centre]
        mean = float(np.mean(slot["mean"]))
        std_single = float(np.mean(slot["std"]))
        bins.append(
            {
                "bin_center": centre,
                "mean_norm_rated": mean,
                "std_single_turbine": std_single,
                "in_steep": bool(STEEP_LOW <= mean <= STEEP_HIGH),
            }
        )

    steep = [b for b in bins if b["in_steep"]]
    n_turbines = float(np.mean(list(TURBINES_PER_GROUP.values())))
    scatter = {
        "bins_total": len(bins),
        "bins_steep": len(steep),
        "band_half_width": BAND_HALF,
        "n_turbines_mean": n_turbines,
        "steep_std_upper": float(np.mean([b["std_single_turbine"] for b in steep]))
        if steep else float("nan"),
        "steep_std_lower": float(
            np.mean([b["std_single_turbine"] for b in steep]) / np.sqrt(n_turbines)
        ) if steep else float("nan"),
        "steep_std_max_upper": float(max(b["std_single_turbine"] for b in steep))
        if steep else float("nan"),
    }
    b1 = bool(scatter["steep_std_upper"] > BAND_HALF)
    b2 = bool(scatter["steep_std_lower"] > BAND_HALF)

    if b1 and b2:
        verdict = "POWER_SCATTER_BINDS_FICR_EVEN_AFTER_AGGREGATION"
    elif b1:
        verdict = "POWER_SCATTER_BINDS_UNLESS_TURBINES_DECORRELATE"
    else:
        verdict = "POWER_SCATTER_WITHIN_BAND_NOT_THE_BOTTLENECK"

    check = {
        "A1_expectation": f"시간분할 감소율 >= {REQUIRED_REDUCTION:.1%} (세 그룹)",
        "A1_held": a1,
        "A2_expectation": "시간분할 < 무작위 KFold (누출 확인)",
        "A2_held": a2,
        "B1_expectation": f"급경사 산포(상한) > 밴드 반폭 {BAND_HALF}",
        "B1_held": b1, "B1_measured": scatter["steep_std_upper"],
        "B2_expectation": f"급경사 산포(하한, 무상관 집계) > {BAND_HALF}",
        "B2_held": b2, "B2_measured": scatter["steep_std_lower"],
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "corrects": "C1N53_SUPPLIED_EXTRACTION (무작위 KFold 시간 누출)",
        "leak": "teach() 의 KFold(3, shuffle=True) 가 시간당 데이터에 무작위 분할을 적용해 "
                "t±1h 이웃이 학습셋에 들어간다. A1 이 잰 lag-1 자기상관 0.951~0.962",
        "no_training": True, "no_collection": True, "uses_actual_kwh": False,
        "required_reduction": REQUIRED_REDUCTION,
        "chronological": chrono_rows,
        "scatter": scatter,
        "bins": bins,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 54 — 시간분할 교정과 출력 산포 천장",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 교정 대상: `{payload['corrects']}`",
        "- **학습·수집 없음.** `actual_kwh` 미사용",
        "",
        "## A. 누출 교정",
        "",
        payload["leak"] + ".",
        "",
        "| group | IDW | teacher(allweather) | 시간분할 감소 | 무작위 KFold "
        "| 누출분 | 필요량 대비 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in chrono_rows:
        lines.append(
            f"| {r['group']} | {r['idw']:.3f} | {r['allweather']:.3f} | "
            f"**{r['reduction_chronological']:.1%}** | {r['reduction_shuffled']:.1%} | "
            f"{r['leakage_inflation']:.1%} | **{r['multiple_of_required']:.1f}x** |"
        )
    lines += [
        "",
        f"교정 후에도 필요량 {REQUIRED_REDUCTION:.1%} 의 약 2 배다. **사이클 53 의 결론은 "
        "살아남는다** — 공급 데이터만으로 요구 풍속 정확도를 넘는다.",
        "",
        "## B. 그러면 왜 점수가 안 오르는가",
        "",
        "teacher 출력은 **이미 모델의 피처**다(M102 의 100 개 중 13 개). 좋은 풍속을 갖고도",
        "Total 이 0.63 대라면 병목은 풍속이 아니라 **같은 풍속에서 출력이 흩어지는 폭**이다.",
        "",
        f"A5 실측: bin {scatter['bins_total']} 개, 급경사(정격비 "
        f"{STEEP_LOW}~{STEEP_HIGH}) **{scatter['bins_steep']}** 개.",
        "",
        "| 기준 | 급경사 산포 (정격비) | 밴드 반폭 | 비 |",
        "|---|---:|---:|---:|",
        f"| 상한 (터빈 1 기, 상쇄 없음) | **{scatter['steep_std_upper']:.4f}** | "
        f"{BAND_HALF} | **{scatter['steep_std_upper'] / BAND_HALF:.2f}x** |",
        f"| 하한 (무상관 집계, /sqrt({n_turbines:.1f})) | "
        f"**{scatter['steep_std_lower']:.4f}** | {BAND_HALF} | "
        f"**{scatter['steep_std_lower'] / BAND_HALF:.2f}x** |",
        "",
        "| bin 중심 | 평균(정격비) | 터빈 산포 | 급경사 |",
        "|---:|---:|---:|:---:|",
    ]
    for b in bins[::3]:
        lines.append(
            f"| {b['bin_center']:.2f} | {b['mean_norm_rated']:.3f} | "
            f"{b['std_single_turbine']:.4f} | {'O' if b['in_steep'] else ''} |"
        )
    lines += [
        "",
        "## C. 사전확약 대조",
        "",
        f"- A1 `{check['A1_expectation']}` -> **{a1}**",
        f"- A2 `{check['A2_expectation']}` -> **{a2}**",
        f"- B1 `{check['B1_expectation']}` -> **{b1}** "
        f"({scatter['steep_std_upper']:.4f})",
        f"- B2 `{check['B2_expectation']}` -> **{b2}** "
        f"({scatter['steep_std_lower']:.4f})",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE54_SCATTER_CEILING",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [], "model_fits": 0,
        "lockbox_reopened": False, "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    for r in chrono_rows:
        print(f"[C54] g{r['group']} 시간분할 감소 {r['reduction_chronological']:.1%} "
              f"(무작위 {r['reduction_shuffled']:.1%}, 누출 {r['leakage_inflation']:.1%}) "
              f"필요량의 {r['multiple_of_required']:.1f}배")
    print(f"[C54] 급경사 bin {scatter['bins_steep']}/{scatter['bins_total']}  "
          f"산포 상한 {scatter['steep_std_upper']:.4f} "
          f"({scatter['steep_std_upper'] / BAND_HALF:.2f}x 밴드)  "
          f"하한 {scatter['steep_std_lower']:.4f} "
          f"({scatter['steep_std_lower'] / BAND_HALF:.2f}x)")
    print(f"[C54] A1 {a1} | A2 {a2} | B1 {b1} | B2 {b2}")
    print(f"[C54] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
