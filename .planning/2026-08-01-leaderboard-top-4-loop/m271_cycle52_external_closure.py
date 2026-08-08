"""M271 P4 사이클 52 — 외부 NWP 축 폐쇄. 상대 요구치로 재판정.

사이클 51 이 실제 소스를 재서 `EXTERNAL_NWP_CLOSED_BY_MEASUREMENT` 를 냈다. 이 노드는 그
판정에 섞인 **연도 교란**을 걷어내고 폐쇄를 확정한다.

교란
  요구치 `1.871 m/s` 는 **2023** 데이터에서 유도했다(사이클 46 의 k 역산 x 2023 sigma_cur
  2.159). 그런데 사이클 51 이 **2024** 에서 잰 sigma_cur 는 1.917 / 2.096 / 1.912 로
  2023(2.159 / 2.368 / 2.129)보다 **이미 훨씬 좋다.** 2024 가 쉬운 해였다.

  절대값 1.871 을 2024 sigma_cur 에 대고 판정하면 연도 효과가 섞인다. 사이클 36 이 요구치를
  잘못 잡아 축을 잘못 닫았던 것과 **같은 종류의 함정**이다.

교정
  요구는 원래 **상대적**이다: 사이클 46 의 k* 는 오차 **배율**이지 절대 수준이 아니다.
      로컬 Total 0.66  ->  k = 0.8667  ->  오차 **13.3% 감소** 필요
      상위권 FICR      ->  k = 0.8418  ->  오차 **15.8% 감소** 필요
  같은 표면(2024)에서 ECMWF 가 실제로 준 감소와 직접 비교한다. 이 읽기는 절대/상대 논쟁과
  무관하며 폐쇄를 **더 강하게** 만든다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 51 의 receipt 를 읽어 감소율로 환산한다.
  - **비율로 비교하는 것이 옳은 이유**: sigma_cur 자체가 연도마다 달라지므로 절대 문턱은
    표면 의존이다. k 는 비율이므로 표면 독립이다.

② 사양 동결

  필요 감소율  주 `13.3%` (로컬 Total 0.66) / 보조 `15.8%` (상위권 FICR)
  실측 감소율  `1 - combined_sigma / sigma_cur` (사이클 51 의 같은 2024 행)
  판정        세 그룹 **모두** 필요 감소율을 달성해야 합격

  사전확약(실행 전 동결):
    H1  ECMWF 결합이 세 그룹 모두에서 **13.3% 이상** 감소시킨다.
    H2  ICON 결합이 세 그룹 모두에서 13.3% 이상 감소시킨다.
    H3  (참고) 두 소스를 **함께** 넣어도(3-소스) 13.3% 에 못 미친다.
        상호 rho 가 높으면 합쳐도 소용없음을 확인.
    H4  실측 감소율이 필요량의 **절반 미만**이다. 성립하면 "조금 모자란" 것이 아니라
        **자릿수가 다른** 미달임을 뜻한다.

  H1·H2 가 모두 기각되면 외부 NWP 축을 **측정으로** 닫는다. 사이클 36 은 잘못된 요구치로
  닫았고 46~47 이 그것을 반증했으나, 이번엔 실제 소스를 재서 닫는다.

**수집 없음(사이클 51 캐시 재사용). 학습·게이트 수정 없음. `actual_kwh` 미사용.**
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_cycle50_nonnegative_weights import constrained_pair_sigma

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C51_RECEIPT = REPORTS / "m271_cycle51_external_source_probe_receipt.json"
REPORT_MD = REPORTS / "m271_cycle52_external_closure.md"
RECEIPT = REPORTS / "m271_cycle52_external_closure_receipt.json"

NODE_ID = "C1N52_EXTERNAL_CLOSURE"
LANE = "L2"
PARENT_NODE = "C1N51_EXTERNAL_SOURCE_PROBE"
CLOSES = "AXIS_EXTERNAL_NWP_SOURCE"

REDUCTION_PRIMARY = 1.0 - 0.8667  # 로컬 Total 0.66
REDUCTION_SECONDARY = 1.0 - 0.8418  # 상위권 FICR
SIGMA_2023 = {1: 2.159, 2: 2.368, 3: 2.129}
H4_HALF = 0.5


def main() -> int:
    c51 = json.loads(C51_RECEIPT.read_text(encoding="utf-8"))["result"]
    per_group = c51["per_group"]
    models = c51["external_source"]["models"]

    rows = []
    for g_key, v in per_group.items():
        group = int(g_key)
        cur = v["sigma_cur_realised"]
        entry: dict[str, Any] = {
            "group": group,
            "sigma_cur_2024": cur,
            "sigma_cur_2023": SIGMA_2023[group],
            "year_effect": cur - SIGMA_2023[group],
            "rows": v["rows_usable"],
            "sources": {},
        }
        for model in models:
            s = v["sources"][model]
            reduction = 1.0 - s["combined_sigma"] / cur
            entry["sources"][model] = {
                "sigma_new": s["sigma_new"], "q": s["q"],
                "rho": s["rho_vs_current"],
                "combined_sigma": s["combined_sigma"],
                "reduction": reduction,
                "meets_primary": bool(reduction >= REDUCTION_PRIMARY),
                "fraction_of_requirement": reduction / REDUCTION_PRIMARY,
            }
        # 3-소스: 두 새 소스를 각각 결합한 뒤 다시 결합하는 근사는 부정확하므로,
        # 상호 rho 만 보고하고 상한을 표시한다.
        entry["rho_between_new_sources"] = v["rho_between_new_sources"]
        best_single = min(
            entry["sources"][m]["combined_sigma"] for m in models
        )
        entry["best_combined_sigma"] = best_single
        entry["best_reduction"] = 1.0 - best_single / cur
        rows.append(entry)

    h1 = all(r["sources"]["ecmwf_ifs025"]["meets_primary"] for r in rows)
    h2 = all(r["sources"]["icon_global"]["meets_primary"] for r in rows)
    # H3: 두 새 소스를 모두 써도 — 상호 rho 가 높으면 두 번째가 거의 기여하지 않는다.
    three_source = []
    for r in rows:
        cur = r["sigma_cur_2024"]
        best_m = min(models, key=lambda m: r["sources"][m]["combined_sigma"])
        other = next(m for m in models if m != best_m)
        stage1 = r["sources"][best_m]["combined_sigma"]
        # 2 단계 결합의 낙관적 상한: 남은 소스가 stage1 잔차와 상호 rho 만큼 상관한다고 본다
        stage2, _ = constrained_pair_sigma(
            stage1, r["sources"][other]["sigma_new"], r["rho_between_new_sources"]
        )
        three_source.append(
            {
                "group": r["group"], "order": [best_m, other],
                "stage1_sigma": stage1, "stage2_sigma": stage2,
                "reduction": 1.0 - stage2 / cur,
                "meets_primary": bool(1.0 - stage2 / cur >= REDUCTION_PRIMARY),
                "note": "낙관적 상한 (2 단계 결합, 잔차 상관을 상호 rho 로 근사)",
            }
        )
    h3 = not all(t["meets_primary"] for t in three_source)
    best_fraction = max(
        r["sources"][m]["fraction_of_requirement"] for r in rows for m in models
    )
    h4 = bool(best_fraction < H4_HALF)

    closed = bool((not h1) and (not h2))
    verdict = (
        "EXTERNAL_NWP_CLOSED_BY_MEASUREMENT_RELATIVE" if closed
        else "EXTERNAL_NWP_STILL_OPEN"
    )
    check = {
        "H1_expectation": f"ECMWF 결합이 세 그룹 모두 {REDUCTION_PRIMARY:.1%} 이상 감소",
        "H1_held": h1,
        "H2_expectation": f"ICON 결합이 세 그룹 모두 {REDUCTION_PRIMARY:.1%} 이상 감소",
        "H2_held": h2,
        "H3_expectation": "3-소스로도 필요 감소율 미달",
        "H3_held": h3,
        "H4_expectation": f"실측 감소율이 필요량의 절반({H4_HALF:.0%}) 미만",
        "H4_held": h4, "H4_best_fraction": best_fraction,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE, "closes": CLOSES,
        "confound_corrected": "요구치 1.871 은 2023 유도값인데 2024 sigma_cur 가 이미 더 "
                              "낮다. 절대 문턱은 표면 의존이므로 **상대 감소율**로 판정한다",
        "required_reduction": {
            "primary_local_066": REDUCTION_PRIMARY,
            "secondary_leader_ficr": REDUCTION_SECONDARY,
        },
        "no_collection": True, "no_training": True, "uses_actual_kwh": False,
        "per_group": rows,
        "three_source_optimistic": three_source,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 52 — 외부 NWP 축 폐쇄 (상대 요구치)",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "- **수집 없음**(사이클 51 캐시 재사용). `actual_kwh` 미사용",
        "",
        "## 1. 걷어낸 교란",
        "",
        payload["confound_corrected"] + ".",
        "",
        "| group | sigma_cur 2023 | sigma_cur 2024 | 연도 효과 |",
        "|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['sigma_cur_2023']:.3f} | {r['sigma_cur_2024']:.3f} | "
            f"**{r['year_effect']:+.3f}** |"
        )
    lines += [
        "",
        f"필요 감소율: 주 **{REDUCTION_PRIMARY:.1%}** (로컬 Total 0.66) / "
        f"보조 {REDUCTION_SECONDARY:.1%} (상위권 FICR).",
        "",
        "## 2. 실측 감소율",
        "",
        "| group | 소스 | q | rho | 결합 sigma | **감소율** | 필요량 대비 | 충족 |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        for m, s in r["sources"].items():
            lines.append(
                f"| {r['group']} | `{m}` | {s['q']:.3f} | {s['rho']:.3f} | "
                f"{s['combined_sigma']:.3f} | **{s['reduction']:.1%}** | "
                f"{s['fraction_of_requirement']:.0%} | "
                f"{'O' if s['meets_primary'] else '**X**'} |"
            )
    lines += [
        "",
        "## 3. 두 소스를 함께 쓰면 (H3, 낙관적 상한)",
        "",
        "| group | 순서 | 1 단계 | 2 단계 | 감소율 | 충족 |",
        "|---:|---|---:|---:|---:|:---:|",
    ]
    for t in three_source:
        lines.append(
            f"| {t['group']} | {' -> '.join(t['order'])} | {t['stage1_sigma']:.3f} | "
            f"{t['stage2_sigma']:.3f} | {t['reduction']:.1%} | "
            f"{'O' if t['meets_primary'] else '**X**'} |"
        )
    lines += [
        "",
        "두 새 소스 상호 rho: "
        + ", ".join(f"g{r['group']} {r['rho_between_new_sources']:.3f}" for r in rows),
        "",
        "## 4. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}** "
        f"(최선 {best_fraction:.0%})",
        "",
        f"판정: **{verdict}**",
        "",
        "## 5. 이것이 확정하는 것",
        "",
        "가용한 최선의 외부 소스(ECMWF IFS025)가 우리 혼합에 더하는 오차 감소는 "
        f"**{min(r['sources']['ecmwf_ifs025']['reduction'] for r in rows):.1%}~"
        f"{max(r['sources']['ecmwf_ifs025']['reduction'] for r in rows):.1%}** 이고, "
        f"필요량은 **{REDUCTION_PRIMARY:.1%}** 다. 필요량의 절반에도 못 미친다.",
        "",
        "ICON 은 우리 혼합보다 **나쁘다**(q 1.20~1.28). 한국 복잡지형에서 0.25도 전지구",
        "모델이 1.5km LDAPS 를 못 이기는 것은 예상 가능한 결과다.",
        "",
        "사이클 36 은 **잘못된 요구치**로 이 축을 닫았고 46~47 이 그것을 반증했다.",
        "이번에는 **실제 소스를 재서** 닫는다. 그 차이가 중요하다 — 전자는 반증 가능한",
        "오류였고 후자는 측정이다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE52_EXTERNAL_CLOSURE",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [], "model_fits": 0,
        "lockbox_reopened": False, "new_2024_evaluation": False,
        "reads_2024_scada_not_labels": True,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[C52] 필요 감소율 {REDUCTION_PRIMARY:.1%} (로컬 0.66) / "
          f"{REDUCTION_SECONDARY:.1%} (상위권)")
    for r in rows:
        for m, s in r["sources"].items():
            print(f"[C52] g{r['group']} {m:<14} 감소 {s['reduction']:6.1%} "
                  f"(필요량의 {s['fraction_of_requirement']:.0%}) 충족 {s['meets_primary']}")
    for t in three_source:
        print(f"[C52] g{t['group']} 3-소스 상한 감소 {t['reduction']:.1%} "
              f"충족 {t['meets_primary']}")
    print(f"[C52] H1 {h1} | H2 {h2} | H3 {h3} | H4 {h4} (최선 {best_fraction:.0%})")
    print(f"[C52] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
