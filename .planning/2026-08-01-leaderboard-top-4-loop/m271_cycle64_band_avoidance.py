"""M271 P4 사이클 64 — 예측이 중간대를 비켜 가는가.

사이클 63 이 원장을 물리 천장에 재기준화하자 라우팅이 뒤집혔고, 새 1 위가 이것이다.

    group 2 | 2023-11 | y_band (0.45, 0.7]     실현 0.136   천장 0.916
    group 2 | 2023-10 | y_band (0.45, 0.7]     실현 0.226   천장 0.916
    group 2 | 2023-07 | y_band (0.45, 0.7]     실현 0.221   천장 0.916

g2 는 세 그룹 중 물리가 가장 좋다(theta 0.495, 변동계수 0.075, 천장 0.9108). 그런데
**중간대에서만** 천장의 15~27% 밖에 못 낸다. 같은 g2 의 고출력대는 0.646~0.685 대
0.880 으로 훨씬 가깝다.

셀은 **실제** 출력으로 묶여 있다. 그러니 실현 0.136 은 *실제가 중간대일 때 우리 예측이
거의 항상 ±6% 밖* 이라는 뜻이다. 가능한 기전 하나가 눈에 띈다 — 예측 분포가 저출력
모드와 고출력 모드로 갈려 있으면 Bayes argmax 는 **모드를 고르고 중간을 비켜 간다.**
정산단위 계단손실은 중간에 어정쩡하게 두느니 한쪽 모드에 거는 쪽을 보상하므로,
결정규칙이 그 비켜감을 **적극적으로** 만들 수도 있다.

이건 예보 정확도 문제가 아니라 결정층 인공물일 수 있고, 그러면 고칠 수 있다.

**① 방법 리서치**

  - 예보 검증에서 이 질문의 표준 도구는 **분할표(contingency table)** 다. 범주형
    예보-관측 결합분포를 놓고 주변분포와 대각선을 본다(Jolliffe & Stephenson 2012,
    Wilks 2011 의 다범주 검증).
  - 비켜감의 표준 이름은 **예보 부족분산(under-dispersion)** 이 아니라 그 반대인
    **과분산/양극화**다. 진단 지표로는 주변분포 비교가 쓰인다 —
    `P(예측 in B)` 대 `P(실제 in B)` 의 비. 1 보다 작으면 그 범주를 덜 예보한다.
    Murphy & Winkler(1987) 의 분포지향 검증틀이 정확히 이 결합분포를 본다.
  - 편향과 산포를 가르는 표준은 **오차의 부호 대칭성**이다. 실제가 중간대인 행에서
    오차가 한쪽으로 쏠리면 이동으로 고칠 수 있고, 대칭이면 더 나은 예보가 필요하다.
  - **채택**: 그룹별 4x4 대역 분할표 + 주변분포 비 + 실제-중간대 행의 오차 부호 분해.
    적합 없음(캐시). 새 런타임 없음.

**② 사양 동결**

  입력   `m271_decision_surface` 캐시. 결정은 C60 이 fold-외로 고른 **전역 T**
         (GLOBAL 팔)를 쓴다 — 배포 계열에 가장 가까운 단일 규칙이고, 수준별 T 는
         C62 가 2 개월 집중으로 배포 불가 판정했으므로 여기 기준으로 쓰지 않는다.
  대역   원장과 동일 경계 (0.10, 0.25, 0.45, 0.70, 1.10). 유효행(실제 >= 10% 용량)만.
  지표   분할표 `P(예측대역 | 실제대역)`, 주변분포 비 `P(예측=B)/P(실제=B)`,
         실제-중간대 행의 오차 부호 분해와 |오차| 분포.

  **타당성 가드**
    V1  캐시 면에서도 g2 중간대의 **실현/천장** 비가 세 그룹 중간대 중 최저이거나
        최저에 준한다(하위 2 위 이내). 아니면 원장 면과 캐시 면이 달라서 C63 의
        라우팅을 이 면으로 옮길 수 없고, 그 사실만 보고한다.

  사전확약 (V1 통과시에만 판정):
    H1  g2 에서 실제가 (0.45,0.70] 인 행 중 **예측도 그 대역**인 비율이 0.5 미만.
        주 가설 — 비켜감.
    H2  중간대 주변분포 비 `P(예측 mid)/P(실제 mid)` 가 세 그룹 모두 1 미만.
        예측이 중간대를 **구조적으로 덜 쓴다**.
    H3  그 비가 g2 에서 가장 낮다. C63 의 여유 순위와 맞아야 한다.
    H4  실제-중간대 행의 오차 부호가 **대칭에 가깝다**(한쪽 비율 0.35~0.65).
        참이면 단순 이동으로 못 고치고 산포 문제다. 거짓이면 **방향성 편향**이고
        이동 가능성이 열린다.

  H1·H2 가 참이면 비켜감이 실재한다. H4 가 방향을 정한다 — 편향이면 다음 노드는
  이동 보정, 대칭이면 결정규칙의 모드 선택 자체를 봐야 한다.

  **부호를 예단하지 않는 곳**: H4 다. 계단손실이 모드 커밋을 보상하므로 대칭일
  이유도 편향일 이유도 있다. 예측하는 것은 비켜감의 **존재**(H1·H2)이지 그 원인의
  방향이 아니다.

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

from m271_cycle40_band_classifier import bayes_decision
from m271_cycle44_sharpened_decision import sharpen
from m271_cycle60_level_temperature import sharpen_by_row
from m271_decision_surface import load_surface

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C60_RECEIPT = REPORTS / "m271_cycle60_level_temperature_receipt.json"
C57B_RECEIPT = REPORTS / "m271_cycle57b_variance_law_receipt.json"
REPORT_MD = REPORTS / "m271_cycle64_band_avoidance.md"
RECEIPT = REPORTS / "m271_cycle64_band_avoidance_receipt.json"

NODE_ID = "C1N64_BAND_AVOIDANCE"
LANE = "L8"
PARENT_NODE = "C1N63_CEILING_REBASE"

EDGES = (0.10, 0.25, 0.45, 0.70, 1.10)
LABELS = ("(0.10,0.25]", "(0.25,0.45]", "(0.45,0.70]", "(0.70,1.10]")
MID = 2  # (0.45,0.70]
ELIGIBLE = 0.10
BAND_HIT = 0.06
SYMMETRIC_LOW, SYMMETRIC_HIGH = 0.35, 0.65


def band_of(rate: np.ndarray) -> np.ndarray:
    """유효행 대역 인덱스. 0.10 이하와 1.10 초과는 -1 로 뺀다."""
    out = np.full(len(rate), -1, dtype=int)
    for i in range(len(EDGES) - 1):
        out[(rate > EDGES[i]) & (rate <= EDGES[i + 1])] = i
    return out


def main() -> int:
    store, info = load_surface()
    c60 = json.loads(C60_RECEIPT.read_text(encoding="utf-8"))
    assert info["probability_digest"] == c60["probability_digest"], "확률면 불일치"
    c57b = json.loads(C57B_RECEIPT.read_text(encoding="utf-8"))

    ceiling_mid = {
        int(g): next(
            b["unit_over_4"] for b in blk["bands"] if b["band"] == "(0.45,0.70]"
        )
        for g, blk in c57b["per_group"].items()
    }

    parts: list[pd.DataFrame] = []
    for fold, entry in store.items():
        temperature = np.full(
            len(entry["capacity"]), float(c60["chosen"]["global"][fold])
        )
        prediction = bayes_decision(sharpen_by_row(entry["probability"], temperature))
        frame = entry["meta"].copy()
        frame["group_id"] = entry["group"]
        frame["capacity"] = entry["capacity"]
        frame["actual_rate"] = frame["actual_kwh"] / frame["capacity"]
        frame["pred_rate"] = prediction
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    data = data.loc[data["actual_rate"] >= ELIGIBLE].reset_index(drop=True)
    data["actual_band"] = band_of(data["actual_rate"].to_numpy(float))
    data["pred_band"] = band_of(np.clip(data["pred_rate"].to_numpy(float), 0.0, 1.10))
    data = data.loc[data["actual_band"] >= 0].reset_index(drop=True)
    data["err"] = data["pred_rate"] - data["actual_rate"]
    data["hit"] = data["err"].abs() <= BAND_HIT

    per_group: dict[str, Any] = {}
    for group in (1, 2, 3):
        block = data.loc[data["group_id"] == group]
        table = np.zeros((len(LABELS), len(LABELS)), dtype=float)
        for actual in range(len(LABELS)):
            row = block.loc[block["actual_band"] == actual]
            if not len(row):
                continue
            for predicted in range(len(LABELS)):
                table[actual, predicted] = float(
                    (row["pred_band"] == predicted).mean()
                )
        actual_share = np.array(
            [float((block["actual_band"] == b).mean()) for b in range(len(LABELS))]
        )
        pred_share = np.array(
            [float((block["pred_band"] == b).mean()) for b in range(len(LABELS))]
        )
        mid_rows = block.loc[block["actual_band"] == MID]
        under = float((mid_rows["err"] < 0).mean()) if len(mid_rows) else float("nan")
        per_group[str(group)] = {
            "rows": int(len(block)),
            "contingency": table.tolist(),
            "actual_share": actual_share.tolist(),
            "pred_share": pred_share.tolist(),
            "marginal_ratio": [
                float(p / a) if a > 0 else float("nan")
                for p, a in zip(pred_share, actual_share, strict=True)
            ],
            "mid_diagonal": float(table[MID, MID]),
            "mid_rows": int(len(mid_rows)),
            "mid_hit_rate": float(mid_rows["hit"].mean()) if len(mid_rows) else float("nan"),
            "mid_ceiling": float(ceiling_mid[group]),
            "mid_realised_over_ceiling": (
                float(mid_rows["hit"].mean() / ceiling_mid[group])
                if len(mid_rows) else float("nan")
            ),
            "mid_under_fraction": under,
            "mid_abs_err_median": float(mid_rows["err"].abs().median())
            if len(mid_rows) else float("nan"),
        }

    ratio_over_ceiling = {
        g: per_group[g]["mid_realised_over_ceiling"] for g in per_group
    }
    ranked = sorted(ratio_over_ceiling, key=lambda g: ratio_over_ceiling[g])
    v1 = bool(ranked.index("2") <= 1)

    h1 = bool(per_group["2"]["mid_diagonal"] < 0.5)
    mid_ratio = {g: per_group[g]["marginal_ratio"][MID] for g in per_group}
    h2 = bool(all(mid_ratio[g] < 1.0 for g in mid_ratio))
    h3 = bool(min(mid_ratio, key=lambda g: mid_ratio[g]) == "2")
    under2 = per_group["2"]["mid_under_fraction"]
    h4 = bool(SYMMETRIC_LOW <= under2 <= SYMMETRIC_HIGH)

    if not v1:
        verdict = "SURFACE_MISMATCH_LEDGER_ROUTING_DOES_NOT_TRANSFER"
    elif h1 and h2 and not h4:
        verdict = "MID_BAND_AVOIDED_WITH_DIRECTIONAL_BIAS"
    elif h1 and h2:
        verdict = "MID_BAND_AVOIDED_ERROR_SYMMETRIC"
    elif h2:
        verdict = "MID_BAND_UNDERUSED_BUT_DIAGONAL_HOLDS"
    else:
        verdict = "NO_MID_BAND_AVOIDANCE"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "method": "CONTINGENCY_TABLE (Jolliffe & Stephenson 2012; Murphy & Winkler 1987)",
        "surface": info,
        "decision": "C60 GLOBAL fold-out T",
        "bands": list(LABELS),
        "per_group": per_group,
        "checks": {"V1_g2_mid_worst_or_second": v1,
                   "mid_realised_over_ceiling": ratio_over_ceiling},
        "hypotheses": {
            "H1_g2_mid_diagonal_below_half": h1,
            "H2_mid_marginal_ratio_below_one_all_groups": h2,
            "H3_g2_lowest_mid_ratio": h3,
            "H4_error_symmetric_in_mid": h4,
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
        "# M271 P4 사이클 64 — 중간대 비켜감",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        f"확률면 prob `{info['probability_digest']}` / 결정 C60 GLOBAL fold-외 T",
        "",
        "## 1. 분할표  `P(예측대역 | 실제대역)`",
        "",
    ]
    for group in ("1", "2", "3"):
        blk = per_group[group]
        lines += [
            f"**group {group}** ({blk['rows']} 행)",
            "",
            "| 실제 \\ 예측 | " + " | ".join(LABELS) + " |",
            "|---|" + "---:|" * len(LABELS),
        ]
        for i, label in enumerate(LABELS):
            cells = " | ".join(f"{v:.3f}" for v in blk["contingency"][i])
            mark = " **<-**" if i == MID else ""
            lines.append(f"| {label}{mark} | {cells} |")
        lines.append("")
    lines += [
        "## 2. 주변분포 비  `P(예측=B) / P(실제=B)`",
        "",
        "| 그룹 | " + " | ".join(LABELS) + " |",
        "|---|" + "---:|" * len(LABELS),
    ]
    for group in ("1", "2", "3"):
        cells = " | ".join(f"{v:.3f}" for v in per_group[group]["marginal_ratio"])
        lines.append(f"| {group} | {cells} |")
    lines += [
        "",
        "## 3. 중간대 상세",
        "",
        "| 그룹 | 행 | 대각선 | 적중률 | 천장 | 실현/천장 | 과소예측 비율 | |오차| 중앙값 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("1", "2", "3"):
        b = per_group[group]
        lines.append(
            f"| {group} | {b['mid_rows']} | {b['mid_diagonal']:.3f} | "
            f"{b['mid_hit_rate']:.3f} | {b['mid_ceiling']:.3f} | "
            f"{b['mid_realised_over_ceiling']:.3f} | {b['mid_under_fraction']:.3f} | "
            f"{b['mid_abs_err_median']:.4f} |"
        )
    lines += [
        "",
        "## 4. 사전확약",
        "",
        f"- V1 g2 중간대 실현/천장이 최저 또는 2 위 -> **{v1}** "
        f"(순위 {' < '.join('g'+g for g in ranked)})",
        f"- H1 g2 중간대 대각선 {per_group['2']['mid_diagonal']:.3f} < 0.5 -> **{h1}**",
        f"- H2 세 그룹 모두 중간대 주변비 < 1 -> **{h2}** "
        f"(g1 {mid_ratio['1']:.3f} / g2 {mid_ratio['2']:.3f} / g3 {mid_ratio['3']:.3f})",
        f"- H3 g2 가 최저 -> **{h3}**",
        f"- H4 오차 부호 대칭 (과소 {under2:.3f}, 대칭구간 "
        f"{SYMMETRIC_LOW}~{SYMMETRIC_HIGH}) -> **{h4}**",
        "",
        "## 5. 판정",
        "",
        f"**{verdict}**",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    for group in ("1", "2", "3"):
        b = per_group[group]
        print(f"[C64] g{group} 중간대  대각선 {b['mid_diagonal']:.3f} / 적중 "
              f"{b['mid_hit_rate']:.3f} / 천장 {b['mid_ceiling']:.3f} / "
              f"실현천장비 {b['mid_realised_over_ceiling']:.3f} / 주변비 "
              f"{b['marginal_ratio'][MID]:.3f} / 과소 {b['mid_under_fraction']:.3f}")
    print(f"[C64] V1 {v1} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C64] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
