"""M271 P4 사이클 62 — 수준별 온도의 이득은 왜 월별로 갈리는가.

사이클 60 의 처리효과는 크고(+0.008990) 기전 예측이 전부 맞았는데 게이트가 기각했다.
`[-O-O]` 5/9 월, 중앙값 +0.00048, 부트스트랩 q05 -0.00264, 최악월 -0.00637.

파라미터 불안정이 아니다. 선택된 표는 fold 를 가로질러 거의 동일했다
(Q2 {2.2,2.2,0.3} / Q3 {2.2,2.2,0.4} / Q4 {2.2,2.2,0.4}). 그러면 남는 설명은 처리가
**적용되는 방식**이 달마다 다르다는 것이다.

단서는 음수 달이 **0 이 아니라 음수**라는 점이다. 처리가 그 달에 무해하게 비껴가는 게
아니라 손해를 끼친다. 수준 배정이 `yhat`(예비 Bayes 결정) 기반이므로, 고출력 질량이
적은 달에는 수준 2 로 배정된 행 상당수가 **위양성**(예측만 높고 실제는 낮음)일 수 있다.
수준 2 의 T=0.3~0.4 는 분포를 극단적으로 날카롭게 만드니, 틀린 모드에 더 세게 커밋한다.

**① 방법 리서치**

  - 이건 처리효과의 **이질성**(treatment effect heterogeneity) 문제다. 표준 도구는
    사전에 지정한 조절변수(moderator)에 대한 효과 분해다. 사후에 조절변수를 찾으면
    다중검정이 되므로 **실행 전에 후보를 동결**한다(Assmann et al. 2000 의 하위군
    분석 규율, Rothwell 2005 의 사전지정 요건).
  - 조절변수 후보는 기전에서 유도한다. 데이터를 보고 고르지 않는다.
      M1  수준 2 질량   — 처리가 적용되는 행의 비율. 크면 효과가 커야 한다.
      M2  수준 2 위양성률 — 수준 2 로 배정됐으나 실제 rate <= 0.70 인 비율.
                          크면 손해가 나야 한다. **이것이 주 가설.**
      M3  월 평균 rate  — 단순 계절 대리변수. M1 과 공선일 것이므로 대조용.
  - 상관은 월 9 개뿐이라 **순위상관(Spearman)** 으로 보고, 유의성을 주장하지 않는다.
    부호와 크기만 본다. 이 노드는 방향을 고르는 진단이지 판정이 아니다.
  - **채택**: 사전지정 조절변수 3 개에 대한 월별 효과 분해. 적합 없음(캐시 사용).

**② 사양 동결**

  입력   `m271_decision_surface` 캐시 (prob digest 로 C60 과 동일함을 확인).
  팔     GLOBAL = C60 의 전역 T, LEVEL = C60 의 수준별 T. **C60 의 선택표를 그대로 쓴다**
         — 재선택하지 않는다. 재선택하면 다른 실험이 된다.
  월별   `paired_monthly_delta` (게이트가 쓰는 것과 같은 함수) 로 월별 Total 델타.

  사전확약:
    H1  M2(위양성률) 와 월별 델타의 Spearman 상관이 **음수**이고 |rho| >= 0.5.
        주 가설. 위양성이 많은 달이 손해를 본다.
    H2  M1(수준2 질량) 와 델타의 상관이 **양수**. 처리가 많이 적용될수록 이득.
    H3  |rho(M2)| > |rho(M1)|. 손해의 설명력이 이득의 설명력보다 크다 —
        게이트를 떨어뜨리는 것은 이득의 부재가 아니라 **손해의 존재**다.
    H4  음수 달들의 수준 2 위양성률 평균이 양수 달들보다 높다.

  H1·H3 이 함께 참이면 다음 수순이 정해진다: 수준 배정을 위치(`yhat`)가 아니라
  **신뢰도**로 바꾸는 것. H1 이 거짓이면 이 기전이 아니고 조절변수를 다시 찾아야 한다.

**진단 전용.** 후보 아님. 점수를 주장하지 않는다. 게이트 미수정. 제출 없음.
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

from m270_monthly_validation import paired_monthly_delta
from m271_cycle40_band_classifier import bayes_decision
from m271_cycle44_sharpened_decision import sharpen
from m271_cycle60_level_temperature import LEVEL_EDGES, level_of, sharpen_by_row
from m271_decision_surface import load_surface

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle62_month_mechanism.md"
RECEIPT = REPORTS / "m271_cycle62_month_mechanism_receipt.json"
C60_RECEIPT = REPORTS / "m271_cycle60_level_temperature_receipt.json"

NODE_ID = "C1N62_MONTH_MECHANISM"
LANE = "L8"
PARENT_NODE = "C1N60_LEVEL_TEMPERATURE"
TOP_LEVEL = len(LEVEL_EDGES)  # 마지막 수준 인덱스
RHO_FLOOR = 0.5


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def main() -> int:
    store, info = load_surface()
    c60 = json.loads(C60_RECEIPT.read_text(encoding="utf-8"))
    assert info["probability_digest"] == c60["probability_digest"], (
        f"확률면 불일치: 캐시 {info['probability_digest']} vs C60 "
        f"{c60['probability_digest']}"
    )
    chosen = c60["chosen"]

    pieces: dict[str, list[pd.DataFrame]] = {"global": [], "level": []}
    moderators: list[dict[str, Any]] = []
    for fold, entry in store.items():
        preliminary = bayes_decision(sharpen(entry["probability"], 1.0))
        level = level_of(preliminary)
        meta = entry["meta"].copy()
        meta["group_id"] = entry["group"]
        meta["capacity"] = entry["capacity"]
        meta["month"] = meta["forecast_kst_dtm"].dt.to_period("M").astype(str)
        meta["level"] = level
        meta["rate"] = meta["actual_kwh"] / meta["capacity"]

        g_table = float(chosen["global"][fold])
        l_table = {int(k): float(v) for k, v in chosen["level"][fold].items()}
        for arm, temperature in (
            ("global", np.full(len(level), g_table)),
            ("level", np.asarray([l_table[int(v)] for v in level], dtype="float64")),
        ):
            out = meta.loc[:, ["forecast_id", "forecast_kst_dtm", "group_id",
                               "actual_kwh", "month"]].copy()
            out["prediction_kwh"] = (
                bayes_decision(sharpen_by_row(entry["probability"], temperature))
                * entry["capacity"]
            )
            pieces[arm].append(out)

        top = meta["level"].to_numpy() == TOP_LEVEL
        for month, block in meta.groupby("month"):
            is_top = block["level"].to_numpy() == TOP_LEVEL
            n_top = int(is_top.sum())
            moderators.append({
                "month": str(month),
                "rows": int(len(block)),
                "M1_top_mass": float(n_top / len(block)),
                "M2_top_false_positive": float(
                    (block.loc[is_top, "rate"] <= LEVEL_EDGES[-1]).mean()
                ) if n_top else float("nan"),
                "M3_mean_rate": float(block["rate"].mean()),
            })
        del top

    frames = {arm: pd.concat(parts, ignore_index=True) for arm, parts in pieces.items()}
    stats = paired_monthly_delta(frames["level"], frames["global"])
    monthly = pd.DataFrame(
        [
            {"month": month, "delta_total": part["total"],
             "delta_ficr": part["ficr"], "delta_nmae": part["one_minus_nmae"]}
            for month, part in stats["per_month"].items()
        ]
    )

    mod = pd.DataFrame(moderators).groupby("month", as_index=False).agg(
        rows=("rows", "sum"),
        M1_top_mass=("M1_top_mass", "mean"),
        M2_top_false_positive=("M2_top_false_positive", "mean"),
        M3_mean_rate=("M3_mean_rate", "mean"),
    )
    delta_col = "delta_total"
    merged = monthly.merge(mod, on="month", how="inner").sort_values("month")
    d = merged[delta_col].to_numpy(dtype="float64")

    rho = {
        "M1_top_mass": spearman(merged["M1_top_mass"].to_numpy(float), d),
        "M2_top_false_positive": spearman(
            merged["M2_top_false_positive"].to_numpy(float), d
        ),
        "M3_mean_rate": spearman(merged["M3_mean_rate"].to_numpy(float), d),
    }

    h1 = bool(rho["M2_top_false_positive"] < 0 and
              abs(rho["M2_top_false_positive"]) >= RHO_FLOOR)
    h2 = bool(rho["M1_top_mass"] > 0)
    h3 = bool(abs(rho["M2_top_false_positive"]) > abs(rho["M1_top_mass"]))
    negative = merged.loc[d < 0]
    positive = merged.loc[d >= 0]
    fp_neg = float(negative["M2_top_false_positive"].mean()) if len(negative) else float("nan")
    fp_pos = float(positive["M2_top_false_positive"].mean()) if len(positive) else float("nan")
    h4 = bool(fp_neg > fp_pos)

    if h1 and h3:
        verdict = "LOSS_DRIVEN_BY_TOP_LEVEL_FALSE_POSITIVES"
    elif h1:
        verdict = "FALSE_POSITIVE_MODERATES_BUT_MASS_EXPLAINS_MORE"
    else:
        verdict = "PREDECLARED_MODERATORS_DO_NOT_EXPLAIN"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "surface": info,
        "method": "PRESPECIFIED_MODERATOR_DECOMPOSITION (Assmann 2000; Rothwell 2005)",
        "level_edges": list(LEVEL_EDGES),
        "months": merged.to_dict(orient="records"),
        "spearman": rho,
        "false_positive_mean": {"negative_months": fp_neg, "positive_months": fp_pos},
        "hypotheses": {
            "H1_false_positive_negatively_correlated": h1,
            "H2_mass_positively_correlated": h2,
            "H3_false_positive_dominates": h3,
            "H4_negative_months_higher_false_positive": h4,
        },
        "verdict": verdict,
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
        "# M271 P4 사이클 62 — 수준별 온도 이득의 월별 이질성",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        f"확률면 prob `{info['probability_digest']}` — C60 과 동일함을 확인",
        "",
        "## 1. 월별",
        "",
        "| 월 | 델타 | M1 수준2 질량 | M2 위양성률 | M3 평균 rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in merged.itertuples():
        lines.append(
            f"| {row.month} | {getattr(row, delta_col):+.6f} | {row.M1_top_mass:.3f} | "
            f"{row.M2_top_false_positive:.3f} | {row.M3_mean_rate:.3f} |"
        )
    lines += [
        "",
        "## 2. 사전지정 조절변수 (Spearman, n=9, 유의성 미주장)",
        "",
        "| 조절변수 | rho |",
        "|---|---:|",
        f"| M1 수준2 질량 | {rho['M1_top_mass']:+.3f} |",
        f"| M2 수준2 위양성률 | **{rho['M2_top_false_positive']:+.3f}** |",
        f"| M3 월 평균 rate | {rho['M3_mean_rate']:+.3f} |",
        "",
        f"위양성률 평균 — 음수 달 {fp_neg:.3f} / 양수 달 {fp_pos:.3f}",
        "",
        "## 3. 사전확약",
        "",
        f"- H1 위양성률이 음의 상관 (|rho| >= {RHO_FLOOR}) -> **{h1}**",
        f"- H2 수준2 질량이 양의 상관 -> **{h2}**",
        f"- H3 위양성 설명력 > 질량 설명력 -> **{h3}**",
        f"- H4 음수 달의 위양성률이 더 높다 -> **{h4}**",
        "",
        "## 4. 판정",
        "",
        f"**{verdict}**",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(f"[C62] Spearman  M1 질량 {rho['M1_top_mass']:+.3f} / "
          f"M2 위양성 {rho['M2_top_false_positive']:+.3f} / "
          f"M3 평균rate {rho['M3_mean_rate']:+.3f}")
    print(f"[C62] 위양성률  음수달 {fp_neg:.3f} / 양수달 {fp_pos:.3f}")
    print(f"[C62] H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C62] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
