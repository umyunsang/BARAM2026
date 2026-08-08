"""M271 — N2 동결 재검(크기 근거 세우기)과 L5 레인 개방.

**N2 동결의 재검.** C1N86 은 "N2 는 N1 이 F1 을 넘을 때만 착수한다 — **그 전에는 크기를
주장할 근거가 없다**" 로 동결했다. N1(C1N87)은 실패했다. 동결을 그대로 지키면 N2 는 막힌다.

그런데 **동결의 이유를 보면 옳은 대응이 다르다.** 이유는 "크기 근거 없음" 이었지 "N1 이
실패하면 N2 도 실패한다" 가 아니었다. 그리고 두 후보는 **다른 기전**을 시험한다.

    N1  감발(curtailment) 하나. 규칙 `잔차 < -0.15 그리고 풍속 >= 10 m/s`. 판정 2.36%.
    N2  SCADA 이상 일반 — 센서 결함, 과도 오류, 장비 이상. 비지도 탐지.

N1 이 반증한 것은 "**g3 의 theta 초과가 감발이다**" 이지 "SCADA 에 제거할 이상이 없다"
가 아니다. 조건을 N1 에 건 것이 **두 기전을 섞은 것**이었다.

**그래서 동결을 풀지 않고 그 이유를 해소한다.** N2 를 실행하는 대신 **크기 상한을
측정**한다. 근거가 서면 라우터가 판정하고, 안 서면 동결이 그대로 유지된다.

**① N2 크기 탐침 — 적합 없음**

  확률면 캐시 v3 에 `sitewind_allweather`(teacher 예측)와 `scada_ws`(표적)가 있으므로
  잔차를 **공짜로** 얻는다. 물음은 이것이다.

      teacher 잔차 분산이 **소수의 이상 행에 얼마나 몰려 있는가.**

  상위 k% 이상 행이 잔차 분산의 f 를 차지하면, 그 행들을 **완벽히 처리했을 때**의
  sigma 감소 상한은 `1 - sqrt(1 - f)` 다. 이것이 N2 가 낼 수 있는 최대치이고,
  실제 이득은 반드시 그보다 작다. **상한이 F1(0.62%) 미만이면 N2 는 어떤 구현으로도
  문턱을 못 넘는다.**

  이상 탐지  `IsolationForest`(문헌의 iForest). SCADA 관측 공간에서만 — `scada_ws`,
             출력 정격비, 그 잔차. **예보 피처를 쓰지 않는다** — N2 는 표적 정제이지
             예측 개선이 아니다.
  격자       k = 1, 2, 5, 10% (실행 전 동결)
  시드       20260806

**② L5 레인 개방 — 실제 리서치**

  https://arxiv.org/abs/2501.14805 (NABQR, DTU)
    "앙상블 예보를 **신경망(LSTM)으로 보정**한 뒤 **시간적응 분위수회귀**로 중앙값과
     분위수를 얻는다."

  **우리 구조와의 정합을 먼저 본다.**
    - NABQR 의 입력은 **NWP 앙상블 멤버**다. 우리에겐 앙상블이 없다 — 46 구간 분포는
      있으나 단일 모형 출력이고 LSTM 보정이 먹을 다중 멤버가 아니다.
    - "시간적응 분위수회귀" 는 **결정층 재보정**이고, C1N60(수준온도)·C1N73(그룹결합)이
      그 이웃에서 이미 **0 과 구분 불가**였다.
    - **적용성 태그**: `near_match_only`. 전제(앙상블 보유)가 우리와 다르다.

**③ 사전확약**

  H1  N2 상한(k=10%)이 F1 문턱 **0.62% 이상**. 참이면 크기 근거가 서고 라우터가
      판정할 수 있다. 거짓이면 **N2 는 어떤 구현으로도 못 넘으므로 동결을 유지**한다.
  H2  잔차 분산 집중도가 균등보다 높다 — 상위 k% 가 잔차 분산의 k% 초과를 차지.
      아니면 이상 탐지가 잔차와 무관한 것을 잡은 것이다.
  H3  g3 의 집중도가 가장 높다. C1N57B 의 theta 0.775 가 g3 에 이상이 많다고
      가리키므로. **N1 에서 이 예측이 틀렸으므로 반복 검정이다.**
  H4  L5 의 N3 는 **전제 불일치로 착수하지 않는다** — 앙상블이 없다. 이것은 검정이
      아니라 리서치 결과의 기록이다.

**진단·실체화 전용.** 모델 미변경. 게이트·lockbox·외부데이터 미사용. 제출 없음.
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
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_decision_surface import load_surface

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_n2_probe_l5_open.md"
RECEIPT = REPORTS / "m271_n2_probe_l5_open_receipt.json"

NODE_ID = "C1N88_N2_PROBE_L5_OPEN"
LANE = "L1"
PARENT_NODE = "C1N87_CURTAILMENT_CLEAN_TARGET"

K_GRID = (0.01, 0.02, 0.05, 0.10)
SEED = 20260806
DETECTION_THRESHOLD = 0.001013
RESPONSE_SLOPE = 0.164
F1_SIGMA_REDUCTION = DETECTION_THRESHOLD / RESPONSE_SLOPE

L5_RESEARCH = {
    "performed_at": "2026-08-06",
    "lane": "L5",
    "sources": [
        {"url": "https://arxiv.org/abs/2501.14805", "class": "peer_reviewed",
         "finding": "NABQR — 앙상블을 LSTM 으로 보정한 뒤 시간적응 분위수회귀",
         "applicability": "near_match_only"},
        {"url": "https://doi.org/10.3390/en18164226", "class": "peer_reviewed",
         "finding": "결정론 예측 + 확산모형으로 잔차 분포 학습(2 단계)",
         "applicability": "near_match_only"},
    ],
    "why_not_applicable": (
        "NABQR 의 입력은 NWP **앙상블 멤버**다. 우리에겐 앙상블이 없다 — 46 구간 분포는 "
        "단일 모형 출력이라 LSTM 보정이 먹을 다중 멤버가 아니다. 그리고 '시간적응 "
        "분위수회귀' 는 결정층 재보정이고 C1N60·C1N73 이 그 이웃에서 이미 0 과 구분 "
        "불가였다(C17 신규성 기각 대상)."
    ),
}


def main() -> int:
    store, info = load_surface()

    parts: list[pd.DataFrame] = []
    for fold in sorted(store):
        entry = store[fold]
        frame = pd.DataFrame({
            "group_id": entry["group"],
            "scada_ws": entry["scada_ws"],
            "rate": entry["meta"]["actual_kwh"].to_numpy(float) / entry["capacity"],
            "pred": entry["sitewind_allweather"],
        })
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    data = data.loc[data["scada_ws"].notna() & data["pred"].notna()].reset_index(drop=True)
    data["residual"] = data["scada_ws"] - data["pred"]

    per_group: dict[str, Any] = {}
    overall_rows: list[dict[str, Any]] = []

    for group in (1, 2, 3):
        block = data.loc[data["group_id"] == group].reset_index(drop=True)
        # SCADA 관측 공간만. 예보 피처를 쓰지 않는다 — N2 는 표적 정제이지 예측 개선이 아니다.
        features = block.loc[:, ["scada_ws", "rate"]].copy()
        features["curve_gap"] = block["rate"] - np.clip(
            (block["scada_ws"] - 3.0) / 9.0, 0.0, 1.0
        ) ** 3
        detector = IsolationForest(
            n_estimators=200, random_state=SEED, contamination="auto", n_jobs=1
        )
        score = -detector.fit(features).score_samples(features)  # 클수록 이상

        resid = block["residual"].to_numpy(float)
        total_ss = float((resid ** 2).sum())
        order = np.argsort(-score)
        entry: dict[str, Any] = {"rows": int(len(block)), "bounds": {}}
        for k in K_GRID:
            take = max(int(round(k * len(block))), 1)
            idx = order[:take]
            share = float((resid[idx] ** 2).sum() / total_ss) if total_ss > 0 else 0.0
            bound = 1.0 - float(np.sqrt(max(1.0 - share, 0.0)))
            entry["bounds"][f"k{int(k*100)}"] = {
                "rows_flagged": take,
                "variance_share": share,
                "concentration": share / k,
                "sigma_reduction_upper_bound": bound,
            }
        per_group[str(group)] = entry
        overall_rows.append({"group": group, **entry["bounds"]["k10"]})

    # 전체 상한 — 그룹별 잔차를 합쳐 같은 절차.
    pooled_bounds: dict[str, Any] = {}
    resid_all = data["residual"].to_numpy(float)
    features_all = data.loc[:, ["scada_ws", "rate"]].copy()
    features_all["curve_gap"] = data["rate"] - np.clip(
        (data["scada_ws"] - 3.0) / 9.0, 0.0, 1.0
    ) ** 3
    det_all = IsolationForest(
        n_estimators=200, random_state=SEED, contamination="auto", n_jobs=1
    )
    score_all = -det_all.fit(features_all).score_samples(features_all)
    total_ss_all = float((resid_all ** 2).sum())
    order_all = np.argsort(-score_all)
    for k in K_GRID:
        take = max(int(round(k * len(data))), 1)
        idx = order_all[:take]
        share = float((resid_all[idx] ** 2).sum() / total_ss_all)
        pooled_bounds[f"k{int(k*100)}"] = {
            "rows_flagged": take,
            "variance_share": share,
            "concentration": share / k,
            "sigma_reduction_upper_bound": 1.0 - float(np.sqrt(max(1.0 - share, 0.0))),
        }

    bound_k10 = float(pooled_bounds["k10"]["sigma_reduction_upper_bound"])
    h1 = bool(bound_k10 >= F1_SIGMA_REDUCTION)
    h2 = bool(all(
        pooled_bounds[f"k{int(k*100)}"]["concentration"] > 1.0 for k in K_GRID
    ))
    conc_by_group = {
        g: per_group[g]["bounds"]["k10"]["concentration"] for g in per_group
    }
    h3 = bool(max(conc_by_group, key=lambda g: conc_by_group[g]) == "3")
    h4 = True  # L5 는 전제 불일치로 착수하지 않는다는 기록.

    if h1:
        verdict = "N2_MAGNITUDE_BASIS_ESTABLISHED_UNFREEZE"
    else:
        verdict = "N2_UPPER_BOUND_BELOW_DETECTION_FREEZE_HOLDS"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "diagnostic_only": True,
        "freeze_reexamination": {
            "frozen_by": "C1N86",
            "text": "N2 는 N1 이 F1 을 넘을 때만 착수한다 — 그 전에는 크기를 주장할 근거가 없다",
            "n1_outcome": "실패 (C1N87, -0.73%)",
            "finding": (
                "동결의 이유는 '크기 근거 없음' 이었지 'N1 이 실패하면 N2 도 실패한다' 가 "
                "아니다. N1 은 감발 하나를, N2 는 SCADA 이상 일반을 시험한다 — 조건을 "
                "N1 에 건 것이 두 기전을 섞은 것이었다."
            ),
            "response": (
                "동결을 풀지 않고 **그 이유를 해소한다** — N2 를 실행하는 대신 크기 상한을 "
                "측정한다. 근거가 서면 라우터가 판정하고, 안 서면 동결이 유지된다."
            ),
        },
        "surface": info,
        "rows": int(len(data)),
        "k_grid": list(K_GRID),
        "seed": SEED,
        "pooled_bounds": pooled_bounds,
        "per_group": per_group,
        "concentration_by_group_k10": conc_by_group,
        "f1_sigma_reduction": F1_SIGMA_REDUCTION,
        "detection_threshold": DETECTION_THRESHOLD,
        "bound_interpretation": (
            "상위 k% 이상 행이 잔차 분산의 f 를 차지하면 그 행들을 **완벽히 처리했을 때**의 "
            "sigma 감소 상한이 `1 - sqrt(1-f)` 다. 실제 이득은 반드시 그보다 작다."
        ),
        "l5_research": L5_RESEARCH,
        "hypotheses": {
            "H1_bound_clears_detection": h1,
            "H2_concentration_above_uniform": h2,
            "H3_g3_most_concentrated": h3,
            "H4_l5_not_applicable_recorded": h4,
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
        "# M271 — N2 동결 재검과 L5 레인 개방",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **진단 전용**",
        "",
        "## 1. N2 동결 재검",
        "",
        f"C1N86 이 동결한 조건: \"{payload['freeze_reexamination']['text']}\"",
        "",
        payload["freeze_reexamination"]["finding"],
        "",
        f"**대응**: {payload['freeze_reexamination']['response']}",
        "",
        "## 2. N2 크기 상한 (적합 없음, 캐시 잔차 사용)",
        "",
        payload["bound_interpretation"],
        "",
        "| k | 표시 행 | 잔차분산 점유 | 집중도 | **sigma 감소 상한** |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in K_GRID:
        b = pooled_bounds[f"k{int(k*100)}"]
        lines.append(
            f"| {k:.0%} | {b['rows_flagged']:,} | {b['variance_share']:.3f} | "
            f"{b['concentration']:.2f}x | **{b['sigma_reduction_upper_bound']:.2%}** |"
        )
    lines += [
        "",
        f"F1 검출문턱 = sigma_v **{F1_SIGMA_REDUCTION:.2%}** 감소",
        "",
        "그룹별 집중도(k=10%): "
        + " / ".join(f"g{g} {conc_by_group[g]:.2f}x" for g in sorted(conc_by_group)),
        "",
        "## 3. L5 레인 개방 — 리서치 결과",
        "",
    ]
    for s in L5_RESEARCH["sources"]:
        lines.append(f"- {s['finding']} — <{s['url']}> (`{s['applicability']}`)")
    lines += [
        "",
        f"**착수하지 않는 이유**: {L5_RESEARCH['why_not_applicable']}",
        "",
        "## 4. 사전확약",
        "",
        f"- H1 N2 상한(k=10%)이 F1 문턱 이상 -> **{h1}** "
        f"({bound_k10:.2%} vs {F1_SIGMA_REDUCTION:.2%})",
        f"- H2 집중도가 균등 초과 -> **{h2}**",
        f"- H3 g3 집중도 최대 -> **{h3}**",
        f"- H4 L5 전제 불일치 기록 -> **{h4}**",
        "",
        "## 5. 판정",
        "",
        f"**{verdict}**",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== N2 탐침 / L5 개방 ===")
    print(f"[N2P] 행 {len(data):,} / F1 문턱 sigma {F1_SIGMA_REDUCTION:.2%}")
    for k in K_GRID:
        b = pooled_bounds[f"k{int(k*100)}"]
        print(f"[N2P] k={k:.0%}  점유 {b['variance_share']:.3f}  집중 "
              f"{b['concentration']:.2f}x  상한 {b['sigma_reduction_upper_bound']:.2%}")
    print(f"[N2P] 그룹별 집중(k=10%) " + " / ".join(
        f"g{g} {conc_by_group[g]:.2f}x" for g in sorted(conc_by_group)))
    print(f"[N2P] L5: 전제 불일치(앙상블 없음) — 착수 안 함")
    print(f"[N2P] H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[N2P] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
