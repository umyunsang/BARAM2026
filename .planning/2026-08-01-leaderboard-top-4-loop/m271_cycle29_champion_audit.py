"""M271 P4 사이클 29 — 챔피언의 선택편향 감사와 오프셋 투영.

두 가지를 한 노드에서 한다. 둘 다 lockbox 를 열지 않는다.

**A. 남은 선택편향 노출.** 사이클 17 에서 내가 직접 적었다 — 결합자 셋(mean/median/
modal_window) 중 median 을 고른 것은 **같은 fold 선택**이고 그 값어치가 `+0.0018` 이다.
사이클 20 의 alpha 끝점 승격도 pooled 사다리에서 났다. pooled 인공물인지 실재하는
우열인지는 **fold 를 쪼개면** 드러난다. 세 fold 각각에서 같은 순서가 나오면 인공물이 아니다.

**B. 오프셋이 함의하는 로컬 목표.** 완료 기준은 로컬 `0.66` 인데 그 값은 로컬-온라인
오프셋을 **0 으로 놓은** 것이다. 실측 앵커 둘이 있다.

    M261  로컬 0.629973 -> 온라인 0.636527   오프셋 +0.006554
    M252  로컬 0.605760 -> 온라인 0.626878   오프셋 +0.021119

온라인 0.66 이 요구하는 로컬 값을 두 앵커로 각각 환산해 지금 위치를 표시한다.
**오프셋은 방법군 의존이고 두 앵커가 3.2 배 벌어진다** — 투영이지 예측이 아니다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. A 는 사이클 17·20 의 사다리를 fold 별로 다시 도는 것이고, B 는 이미
    receipt 에 있는 앵커 둘의 산술이다.
  - 방법론적 쟁점은 하나: **fold 별 재현이 선택편향의 표준 방어**라는 것. 사이클 25 가
    같은 검사(fold 간 순위 재현)로 lambda 를 잡음으로 판정했다. 같은 잣대를 챔피언 자신에게
    들이대지 않으면 이중잣대다.

② 사양 동결

  사전확약(실행 전 동결):
    H1  `median` 이 **세 fold 전부**에서 `mean` 보다 높다.
    H2  `alpha=1.0` 이 **세 fold 전부**에서 `alpha=0.5` 보다 높다.
    H3  챔피언이 **세 fold 전부**에서 배포 정책 `T0.5_G1.5` 보다 높다.
  셋 다 성립해야 챔피언의 선택이 pooled 인공물이 아니라고 말할 수 있다. 하나라도
  기각되면 해당 선택의 근거를 pooled 한 번의 측정으로 낮춰 기록한다.

  B 는 가설이 아니라 **보고되는 측정**이다. 사전확약하지 않는다.

**게이트를 수정하지 않는다. lockbox 를 열지 않는다.** 2024 행 미사용.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_monthly_validation import load_predictions
from m271_cycle13_ensemble import ENSEMBLES
from m271_cycle14_shrinkblend import blend
from m271_cycle17_combiner import combine, stack_members
from m271_cycle21_mos import QUARTER_OF_MONTH
from m271_evaluate_candidate import official

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_cycle29_champion_audit.md"
RECEIPT = REPORTS / "m271_cycle29_champion_audit_receipt.json"

NODE_ID = "C1N29_CHAMPION_AUDIT"
LANE = "L4"
PARENT_NODE = "C1N20_ALPHA_ENDPOINT"
CHAMPION = "M271_MEDIAN4"
BASE_ENSEMBLE = "E3_FOUR_FAMILY"
DEPLOYED = "T0.5_G1.5"
FOLDS = ("Q2", "Q3", "Q4")
ONLINE_TARGET = 0.66
LOCAL_TARGET = 0.66

ANCHORS = (
    {"name": "M261", "local": 0.629973, "online": 0.6365274327,
     "note": "배포 계열. 같은 혈통이라 오프셋이 작다"},
    {"name": "M252", "local": 0.6057596789867304, "online": 0.6268784092,
     "note": "독립 앵커. 제출 0 회 계열이라 오프셋이 크다"},
)


def by_fold(frame: pd.DataFrame) -> dict[str, float]:
    f = frame.copy()
    f["fold"] = f["month"].map(QUARTER_OF_MONTH)
    return {
        fold: official(cell)["total"]
        for fold, cell in f.groupby("fold", observed=True)
        if fold in FOLDS
    }


def main() -> int:
    members = ENSEMBLES[BASE_ENSEMBLE]
    k = len(members)
    stacked = stack_members(members)
    parent = load_predictions(DEPLOYED)

    variants: dict[str, pd.DataFrame] = {}
    for operator in ("mean", "median", "modal_window"):
        variants[f"combiner={operator}"] = combine(stacked, k, operator)
    ensemble = combine(stacked, k, "median")
    for alpha in (0.50, 1.00):
        variants[f"alpha={alpha:.2f}"] = blend(parent, ensemble, alpha)
    variants["deployed"] = parent.copy()

    pooled = {name: official(f)["total"] for name, f in variants.items()}
    folds = {name: by_fold(f) for name, f in variants.items()}

    h1_rows = [
        {
            "fold": fold,
            "mean": folds["combiner=mean"][fold],
            "median": folds["combiner=median"][fold],
            "modal_window": folds["combiner=modal_window"][fold],
            "median_beats_mean": bool(
                folds["combiner=median"][fold] > folds["combiner=mean"][fold]
            ),
            "median_beats_all": bool(
                folds["combiner=median"][fold] > folds["combiner=mean"][fold]
                and folds["combiner=median"][fold] > folds["combiner=modal_window"][fold]
            ),
        }
        for fold in FOLDS
    ]
    h1 = all(r["median_beats_mean"] for r in h1_rows)
    h1_strict = all(r["median_beats_all"] for r in h1_rows)

    h2_rows = [
        {
            "fold": fold,
            "alpha_050": folds["alpha=0.50"][fold],
            "alpha_100": folds["alpha=1.00"][fold],
            "endpoint_wins": bool(folds["alpha=1.00"][fold] > folds["alpha=0.50"][fold]),
        }
        for fold in FOLDS
    ]
    h2 = all(r["endpoint_wins"] for r in h2_rows)

    h3_rows = [
        {
            "fold": fold,
            "deployed": folds["deployed"][fold],
            "champion": folds["combiner=median"][fold],
            "delta": folds["combiner=median"][fold] - folds["deployed"][fold],
            "champion_wins": bool(
                folds["combiner=median"][fold] > folds["deployed"][fold]
            ),
        }
        for fold in FOLDS
    ]
    h3 = all(r["champion_wins"] for r in h3_rows)

    champion_local = pooled["combiner=median"]
    projection = []
    for a in ANCHORS:
        offset = a["online"] - a["local"]
        projection.append(
            {
                "anchor": a["name"],
                "anchor_local": a["local"],
                "anchor_online": a["online"],
                "offset": offset,
                "note": a["note"],
                "projected_online_for_champion": champion_local + offset,
                "local_required_for_online_target": ONLINE_TARGET - offset,
                "champion_shortfall_vs_required_local": (ONLINE_TARGET - offset)
                - champion_local,
            }
        )
    projection.sort(key=lambda p: p["offset"])
    band = {
        "champion_local": champion_local,
        "projected_online_low": min(p["projected_online_for_champion"] for p in projection),
        "projected_online_high": max(p["projected_online_for_champion"] for p in projection),
        "local_required_low": min(p["local_required_for_online_target"] for p in projection),
        "local_required_high": max(p["local_required_for_online_target"] for p in projection),
        "current_online_best": 0.6365274327,
        "beats_current_online_best_under_both_anchors": bool(
            min(p["projected_online_for_champion"] for p in projection) > 0.6365274327
        ),
        "user_local_trigger": LOCAL_TARGET,
        "trigger_implies_online_low": LOCAL_TARGET + min(p["offset"] for p in projection),
        "trigger_implies_online_high": LOCAL_TARGET + max(p["offset"] for p in projection),
    }

    check = {
        "H1_expectation": "median 이 세 fold 전부에서 mean 보다 높다",
        "H1_held": h1, "H1_strict_beats_all_three_combiners": h1_strict,
        "H2_expectation": "alpha=1.0 이 세 fold 전부에서 alpha=0.5 보다 높다",
        "H2_held": h2,
        "H3_expectation": "챔피언이 세 fold 전부에서 배포 정책보다 높다",
        "H3_held": h3,
        "selection_not_pooled_artifact": bool(h1 and h2 and h3),
        "verdict": (
            "CHAMPION_CHOICES_REPLICATE_PER_FOLD" if (h1 and h2 and h3)
            else "CHAMPION_CHOICES_PARTLY_POOLED_ONLY"
        ),
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE, "champion": CHAMPION,
        "gate_modified": False, "lockbox_opened": False,
        "pooled_totals": pooled,
        "per_fold_totals": folds,
        "H1_combiner_per_fold": h1_rows,
        "H2_alpha_per_fold": h2_rows,
        "H3_champion_vs_deployed_per_fold": h3_rows,
        "anchors": list(ANCHORS),
        "offset_projection": projection,
        "band": band,
        "predeclared_check": check,
        "caveat": "오프셋은 방법군 의존이고 두 앵커가 3.2 배 벌어진다. 챔피언은 분류기와 "
                  "analog 를 섞으므로 어느 쪽을 따를지 알 수 없다. 투영이지 예측이 아니다",
    }

    lines = [
        "# M271 P4 사이클 29 — 챔피언의 선택편향 감사와 오프셋 투영",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 챔피언 `{CHAMPION}` pooled Total **{champion_local:.6f}**",
        "- **lockbox 를 열지 않는다.** 2024 행 미사용",
        "",
        "## A. fold 를 쪼개면 같은 순서가 나오는가",
        "",
        "사이클 25 는 fold 간 순위 재현 실패로 `lambda` 를 잡음이라 판정했다. 같은 잣대를",
        "챔피언 자신에게 들이댄다.",
        "",
        "### A-1. 결합자 (H1)",
        "",
        "| fold | mean | **median** | modal_window | median>mean | median 최고 |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    for r in h1_rows:
        lines.append(
            f"| {r['fold']} | {r['mean']:.6f} | **{r['median']:.6f}** | "
            f"{r['modal_window']:.6f} | {'O' if r['median_beats_mean'] else '**X**'} | "
            f"{'O' if r['median_beats_all'] else '**X**'} |"
        )
    lines += [
        "",
        "### A-2. alpha 끝점 (H2)",
        "",
        "| fold | alpha=0.50 | **alpha=1.00** | 끝점 우세 |",
        "|---|---:|---:|:---:|",
    ]
    for r in h2_rows:
        lines.append(
            f"| {r['fold']} | {r['alpha_050']:.6f} | **{r['alpha_100']:.6f}** | "
            f"{'O' if r['endpoint_wins'] else '**X**'} |"
        )
    lines += [
        "",
        "### A-3. 챔피언 vs 배포 (H3)",
        "",
        "| fold | 배포 | **챔피언** | 차이 | 우세 |",
        "|---|---:|---:|---:|:---:|",
    ]
    for r in h3_rows:
        lines.append(
            f"| {r['fold']} | {r['deployed']:.6f} | **{r['champion']:.6f}** | "
            f"{r['delta']:+.6f} | {'O' if r['champion_wins'] else '**X**'} |"
        )

    lines += [
        "",
        "## B. 오프셋이 함의하는 로컬 목표 (보고, 사전확약 아님)",
        "",
        "완료 기준 로컬 `0.66` 은 로컬-온라인 오프셋을 **0 으로 놓은** 값이다. 실측 앵커 둘로",
        "환산하면 다르다.",
        "",
        "| 앵커 | 앵커 로컬 | 앵커 온라인 | 오프셋 | 챔피언 투영 온라인 "
        "| 온라인 0.66 이 요구하는 로컬 | 현재 부족분 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in projection:
        lines.append(
            f"| {p['anchor']} | {p['anchor_local']:.6f} | {p['anchor_online']:.6f} | "
            f"**{p['offset']:+.6f}** | **{p['projected_online_for_champion']:.6f}** | "
            f"{p['local_required_for_online_target']:.6f} | "
            f"**{p['champion_shortfall_vs_required_local']:+.6f}** |"
        )
    lines += [
        "",
        f"- 챔피언 투영 온라인 **{band['projected_online_low']:.6f} ~ "
        f"{band['projected_online_high']:.6f}**",
        f"- 현재 온라인 최고점 `{band['current_online_best']:.6f}` 를 두 앵커 **모두**에서 "
        f"넘는가: **{band['beats_current_online_best_under_both_anchors']}**",
        f"- 사용자의 로컬 트리거 `{LOCAL_TARGET}` 은 온라인 "
        f"**{band['trigger_implies_online_low']:.6f} ~ "
        f"{band['trigger_implies_online_high']:.6f}** 를 함의한다 "
        "(리더보드 1 위 0.67365 권역)",
        "",
        f"**주의**: {payload['caveat']}",
        "",
        "## C. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}** "
        f"(셋 다 이기는가: {h1_strict})",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        "",
        f"판정: **{check['verdict']}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE29_CHAMPION_AUDIT",
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

    for r in h1_rows:
        print(f"[C29] {r['fold']} 결합자  mean {r['mean']:.6f} / "
              f"median {r['median']:.6f} / modal {r['modal_window']:.6f}  "
              f"-> median>mean {r['median_beats_mean']}")
    for r in h2_rows:
        print(f"[C29] {r['fold']} alpha   0.50 {r['alpha_050']:.6f} / "
              f"1.00 {r['alpha_100']:.6f}  -> 끝점 {r['endpoint_wins']}")
    for r in h3_rows:
        print(f"[C29] {r['fold']} 챔피언  {r['delta']:+.6f} vs 배포  "
              f"-> {r['champion_wins']}")
    print(f"[C29] H1 {h1} (엄격 {h1_strict}) | H2 {h2} | H3 {h3} -> {check['verdict']}")
    print(f"[C29] 챔피언 로컬 {champion_local:.6f} -> 투영 온라인 "
          f"{band['projected_online_low']:.6f}~{band['projected_online_high']:.6f}")
    print(f"[C29] 온라인 0.66 이 요구하는 로컬 "
          f"{band['local_required_low']:.6f}~{band['local_required_high']:.6f}  "
          f"(현재 {champion_local:.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
