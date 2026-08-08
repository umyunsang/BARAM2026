"""M271 P4 사이클 63 — 결손 원장을 동류 평균이 아니라 물리 천장에 재기준화한다.

원장이 루프의 연료다. 프론티어가 어느 셀을 팔지 여기서 나온다. 그런데 회수가능질량의
기준선이 **전 셀 통합 평균 하나**다(`m271_deficit.py:181`).

    avg_ficr_per_gen = 전체 FICR 손실 / 전체 발전량가중
    expected_i       = w_gen_i * avg_ficr_per_gen + ...
    excess_i         = loss_i / expected_i
    recoverable_i    = loss_i * (1 - 1/excess_i)

"그 셀이 **평균 셀만큼** 했다면 사라질 손실" 이다. 이 기준은 셀마다 물리 난이도가
같다고 가정한다. 그런데 사이클 57b 가 그 가정을 깼다.

    조건부 산포의 분산법칙   theta = g1 +0.534 / g2 +0.495 / g3 **+0.775**
    변동계수                 g1 0.114 / g2 0.075 / g3 **0.170**

g3 는 물리적으로 더 어렵다. 동류 평균과 비교하면 **그 어려움 전부가 '회수 가능'으로
계상된다.** 실제로 현재 원장은 상위 20 셀 회수가능질량의 **70% 를 g3 에 배정**한다.
사이클 59 는 정반대를 쟀다 — 풍속을 완벽히 알아도 g3 는 g1·g2 의 **절반만**
회수된다(+0.149 대 +0.288/+0.287). 두 계측이 어긋난다.

**① 방법 리서치**

  - 예보 검증의 표준은 **기준 대비 스킬스코어**다. `SS = (S - S_ref) / (S_perf - S_ref)`
    (Murphy 1988; Jolliffe & Stephenson 2012). 분모에 들어가는 것은 **완전예보**이지
    동류 평균이 아니다. 동류 평균을 완전성의 대리로 쓰면 "더 어려운 셀"과
    "더 못한 예보"가 섞인다 — 지금 원장이 그렇다.
  - Murphy(1988) 의 분해는 **잠재 스킬**(예측가능성이 정하는 상한)과 **실현 스킬**을
    나눈다. 셀별로 필요한 양은 잠재 - 실현이다.
  - 이 문제의 잠재 상한은 이미 있다. 사이클 57 이 그룹×구간별 경험적 FICR 천장을
    쟀고(10 분->시간 집계 + 경험적 분포 + 발전량 가중), 57b 가 그것을 y 대역으로
    묶어 `unit_over_4` 로 저장했다.
  - **채택**: 셀별 여유 = `(1/6) * w_gen * (천장 - ubar/4)`. 원장의 손실 항등식과
    **같은 단위**다(`ficr_loss = (1/6) * w_gen * (1 - ubar/4)` 를 재구성으로 확인).

**② 사양 동결**

  입력   A7 receipt 의 전 셀 108 개 (`ubar`, `w_gen`, `group_id`, `y_band`)
         C57b receipt 의 그룹×대역 천장 (`unit_over_4`)
         **재계산·학습·수집 없음.** 두 산출물의 순수 파생이다.
  매핑   A7 의 `(0.7, 1.1]` <-> C57b 의 `(0.70,1.10]`. 경계값이 같으므로 수치로 맞춘다.
  단위   여유_i = (1/6) * w_gen_i * (ceiling(g,band) - ubar_i/4)

  **타당성 가드**
    V1  손실 항등식 재구성이 전 셀에서 1e-9 이내로 일치. 어긋나면 원장 해석이 틀린 것.
    V2  천장을 **넘는** 셀이 20% 미만. 천장은 SCADA 유래 면에서 쟀고 원장은 모델 예측
        면에서 쟀으므로 전이 가정이 있다. 많이 넘으면 그 전이가 무효이고,
        여유값을 보고하지 않고 그 사실만 보고한다.

  사전확약 (V1·V2 통과시에만 판정):
    H1  두 순위가 실질적으로 다르다 — Spearman(회수가능, 여유) < 0.7.
        같으면 기준선 편향이 라우팅을 바꾸지 않았다는 뜻이고 이 노드는 무해한 확인이다.
    H2  g3 의 여유 점유율이 회수가능 점유율(0.70)보다 **낮다**.
        주 가설. 천장이 낮은 그룹은 어려움이 회수 가능으로 잘못 계상돼 있었다.
    H3  총 여유 >= 격차 0.029690. 거짓이면 **모든 셀이 물리 천장에 도달해도 목표에
        못 미친다**는 뜻이고, 그건 이 아키텍처에서 목표 도달 불가라는 중대한 결론이다.
    H4  여유 최대 대역이 여전히 `(0.45, 0.7]`. 참이면 대역 축에서는 기존 기준선이
        옳았고 **그룹 축에서만** 틀렸다는 국소적 결함이 된다.

  H2·H4 가 함께 참이면 결함이 좁게 특정된다. H3 가 거짓이면 목표 자체를 다시 봐야 한다.

**진단 전용.** 후보 아님. 점수를 주장하지 않는다. 게이트·원장 코드 미수정 —
이 노드는 재기준화를 **측정**하지 원장을 고치지 않는다. 고칠지는 결과를 보고 정한다.
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

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
A7_RECEIPT = REPORTS / "m271_n0_deficit_init_receipt.json"
C57B_RECEIPT = REPORTS / "m271_cycle57b_variance_law_receipt.json"
REPORT_MD = REPORTS / "m271_cycle63_ceiling_rebase.md"
RECEIPT = REPORTS / "m271_cycle63_ceiling_rebase_receipt.json"

NODE_ID = "C1N63_CEILING_REBASE"
LANE = "L8"
PARENT_NODE = "C1N57B_VARIANCE_LAW"

CHAMPION_LOCAL = 0.630310
TARGET = 0.66
GAP = TARGET - CHAMPION_LOCAL
IDENTITY_TOLERANCE = 1e-9
OVERSHOOT_LIMIT = 0.20
SPEARMAN_CEILING = 0.7
G3_RECOVERABLE_SHARE = 0.70


def band_bounds(label: str) -> tuple[float, float]:
    lo, hi = label.strip("(]").split(",")
    return round(float(lo), 2), round(float(hi), 2)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def main() -> int:
    a7 = json.loads(A7_RECEIPT.read_text(encoding="utf-8"))["result"]
    c57b = json.loads(C57B_RECEIPT.read_text(encoding="utf-8"))

    ceilings: dict[tuple[int, tuple[float, float]], float] = {}
    for group, block in c57b["per_group"].items():
        for band in block["bands"]:
            ceilings[(int(group), band_bounds(band["band"]))] = float(band["unit_over_4"])

    cells = a7["cells"]

    # V1 — 원장의 손실 항등식을 전 셀에서 재구성한다.
    identity_max = 0.0
    for cell in cells:
        rebuilt = (1.0 / 6.0) * float(cell["w_gen"]) * (1.0 - float(cell["ubar"]) / 4.0)
        identity_max = max(identity_max, abs(rebuilt - float(cell["ficr_loss"])))
    v1 = bool(identity_max <= IDENTITY_TOLERANCE)

    # 동류 평균 기준(현행 원장) 재현.
    gen_weight = sum(float(c["w_gen"]) for c in cells)
    row_weight = sum(float(c["w_rows"]) for c in cells)
    gen_loss = sum(float(c["ficr_loss"]) for c in cells)
    row_loss = sum(float(c["nmae_loss"]) for c in cells)
    avg_ficr = gen_loss / gen_weight
    avg_nmae = row_loss / row_weight if row_weight > 0 else 0.0

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in cells:
        group = int(cell["group_id"])
        bounds = band_bounds(str(cell["y_band"]))
        ceiling = ceilings.get((group, bounds))
        if ceiling is None:
            missing.append(f"g{group}{cell['y_band']}")
            continue
        loss = float(cell["total_loss"])
        expected = float(cell["w_gen"]) * avg_ficr + float(cell["w_rows"]) * avg_nmae
        excess = loss / expected if expected > 0 else float("nan")
        recoverable = loss * (1.0 - 1.0 / excess) if excess and excess > 0 else 0.0
        realised = float(cell["ubar"]) / 4.0
        headroom = (1.0 / 6.0) * float(cell["w_gen"]) * (ceiling - realised)
        rows.append({
            "key": f"group_id={group}|month={cell['month']}|y_band={cell['y_band']}",
            "group_id": group,
            "month": str(cell["month"]),
            "y_band": str(cell["y_band"]),
            "w_gen": float(cell["w_gen"]),
            "realised_unit": realised,
            "ceiling_unit": ceiling,
            "over_ceiling": bool(realised > ceiling),
            "recoverable_peer": max(recoverable, 0.0),
            "headroom_ceiling": headroom,
        })
    if missing:
        raise RuntimeError(f"천장 매핑 실패: {sorted(set(missing))}")

    frame = pd.DataFrame(rows)
    overshoot = float(frame["over_ceiling"].mean())
    v2 = bool(overshoot < OVERSHOOT_LIMIT)

    positive = frame.loc[frame["headroom_ceiling"] > 0]
    total_headroom = float(positive["headroom_ceiling"].sum())
    total_recoverable = float(frame["recoverable_peer"].sum())

    rho = spearman(
        frame["recoverable_peer"].to_numpy(float),
        frame["headroom_ceiling"].to_numpy(float),
    )
    h1 = bool(rho < SPEARMAN_CEILING)

    by_group_head = positive.groupby("group_id")["headroom_ceiling"].sum()
    by_group_rec = frame.groupby("group_id")["recoverable_peer"].sum()
    g3_head_share = float(by_group_head.get(3, 0.0) / by_group_head.sum())
    g3_rec_share = float(by_group_rec.get(3, 0.0) / by_group_rec.sum())
    h2 = bool(g3_head_share < g3_rec_share)

    h3 = bool(total_headroom >= GAP)

    by_band_head = positive.groupby("y_band")["headroom_ceiling"].sum().sort_values()
    top_band = str(by_band_head.index[-1])
    h4 = bool(top_band == "(0.45, 0.7]")

    if not v1:
        verdict = "LEDGER_IDENTITY_NOT_REPRODUCED_RESULT_VOID"
    elif not v2:
        verdict = "CEILING_TRANSFER_INVALID_TOO_MANY_CELLS_EXCEED"
    elif not h3:
        verdict = "CEILING_HEADROOM_BELOW_GAP_TARGET_UNREACHABLE"
    elif h1 and h2:
        verdict = "PEER_BASELINE_MISATTRIBUTES_HARD_PHYSICS_AS_RECOVERABLE"
    elif h1:
        verdict = "RANKINGS_DIFFER_BUT_NOT_BY_GROUP_DIFFICULTY"
    else:
        verdict = "PEER_BASELINE_ADEQUATE"

    top_head = positive.nlargest(12, "headroom_ceiling")
    top_rec = frame.nlargest(12, "recoverable_peer")

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "method": "SKILL_VS_POTENTIAL (Murphy 1988; Jolliffe & Stephenson 2012)",
        "sources": {
            "a7": A7_RECEIPT.name,
            "c57b": C57B_RECEIPT.name,
            "a7_total": float(a7["official"]["total"]),
        },
        "checks": {
            "V1_identity_reproduced": v1,
            "V1_max_residual": identity_max,
            "V2_overshoot_fraction": overshoot,
            "V2_transfer_valid": v2,
        },
        "totals": {
            "recoverable_peer": total_recoverable,
            "headroom_ceiling": total_headroom,
            "gap_to_target": GAP,
        },
        "spearman_recoverable_vs_headroom": rho,
        "group_share": {
            "headroom": {int(k): float(v / by_group_head.sum())
                         for k, v in by_group_head.items()},
            "recoverable": {int(k): float(v / by_group_rec.sum())
                            for k, v in by_group_rec.items()},
        },
        "band_headroom": {str(k): float(v) for k, v in by_band_head.items()},
        "top_headroom": top_head.to_dict(orient="records"),
        "top_recoverable": top_rec.to_dict(orient="records"),
        "hypotheses": {
            "H1_rankings_differ": h1,
            "H2_g3_share_lower_under_ceiling": h2,
            "H3_headroom_covers_gap": h3,
            "H4_top_band_still_mid": h4,
        },
        "verdict": verdict,
        "no_training": True,
        "no_collection": True,
        "dacon_upload": False,
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# M271 P4 사이클 63 — 결손 원장의 물리 천장 재기준화",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용 (후보 아님)**",
        "",
        f"입력 `{A7_RECEIPT.name}` (108 셀) + `{C57B_RECEIPT.name}` (그룹x대역 천장). "
        "재계산·학습·수집 없음.",
        "",
        "## 1. 타당성 가드",
        "",
        f"- V1 손실 항등식 재구성 최대잔차 {identity_max:.3e} -> **{v1}**",
        f"- V2 천장 초과 셀 비율 {overshoot:.3f} (한계 {OVERSHOOT_LIMIT}) -> **{v2}**",
        "",
        "## 2. 총량",
        "",
        f"- 동류평균 기준 회수가능질량 **{total_recoverable:.5f}**",
        f"- 물리천장 기준 여유 **{total_headroom:.5f}**",
        f"- 격차 {GAP:.5f}",
        "",
        "## 3. 그룹 점유율",
        "",
        "| 그룹 | 동류평균 회수가능 | 물리천장 여유 |",
        "|---:|---:|---:|",
    ]
    for group in (1, 2, 3):
        rec = float(by_group_rec.get(group, 0.0) / by_group_rec.sum())
        head = float(by_group_head.get(group, 0.0) / by_group_head.sum())
        lines.append(f"| {group} | {rec:.3f} | {head:.3f} |")
    lines += [
        "",
        "## 4. 대역별 여유",
        "",
        "| 대역 | 여유 |",
        "|---|---:|",
    ]
    for band, value in by_band_head.sort_values(ascending=False).items():
        lines.append(f"| {band} | {value:.5f} |")
    lines += [
        "",
        "## 5. 상위 셀 — 물리천장 기준",
        "",
        "| 셀 | 실현 단위/4 | 천장 단위/4 | 발전가중 | 여유 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in top_head.itertuples():
        lines.append(
            f"| {row.key} | {row.realised_unit:.3f} | {row.ceiling_unit:.3f} | "
            f"{row.w_gen:.4f} | {row.headroom_ceiling:.5f} |"
        )
    lines += [
        "",
        "## 6. 사전확약",
        "",
        f"- H1 두 순위가 다르다 (Spearman {rho:+.3f} < {SPEARMAN_CEILING}) -> **{h1}**",
        f"- H2 g3 여유 점유율 {g3_head_share:.3f} < 회수가능 점유율 {g3_rec_share:.3f} "
        f"-> **{h2}**",
        f"- H3 총 여유 {total_headroom:.5f} >= 격차 {GAP:.5f} -> **{h3}**",
        f"- H4 여유 최대 대역이 `(0.45, 0.7]` (실제 `{top_band}`) -> **{h4}**",
        "",
        "## 7. 판정",
        "",
        f"**{verdict}**",
        "",
        "천장은 SCADA 유래 면에서, 원장은 모델 예측 면에서 쟀다. V2 가 그 전이의 "
        "타당성을 초과 셀 비율로 잰다. 이 노드는 원장 코드를 **고치지 않는다** — "
        "재기준화의 효과를 측정할 뿐이고, 고칠지는 이 결과로 정한다.",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(f"[C63] V1 항등식 {v1} (잔차 {identity_max:.2e}) / V2 초과 {overshoot:.3f} -> {v2}")
    print(f"[C63] 총량  동류평균 {total_recoverable:.5f} / 물리천장 {total_headroom:.5f} "
          f"/ 격차 {GAP:.5f}")
    print(f"[C63] g3 점유  회수가능 {g3_rec_share:.3f} -> 여유 {g3_head_share:.3f}")
    print(f"[C63] Spearman {rho:+.3f} / 여유최대 대역 {top_band}")
    print(f"[C63] H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C63] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
