"""M271 N2 — SCADA 이상 행을 teacher 학습 표적에서 제외한다 (비지도 탐지).

C1N88 이 동결을 풀었다. 상위 10% 이상 행이 teacher 잔차 분산의 **21.4%** 를 차지하고
(집중 2.14 배) sigma 감소 **상한이 11.33%** 로 F1 문턱(0.62%)의 18 배다.

**N1 과 무엇이 다른가.** N1 은 "고풍속 + 커브 하향 이탈" 이라는 **좁고 물리 가정이 박힌**
규칙으로 2.36% 를 잡았고 실패했다(g3 가 -2.16% 로 최악). C1N88 은 같은 g3 에서 이상
집중도가 **가장 높다**(2.56 배)고 쟀다. 즉 **g3 에 이상이 많다는 것은 맞았고 내 감발
규칙이 그것을 못 잡았다.** N2 는 규칙을 손으로 쓰지 않고 비지도 탐지에 맡긴다.

**① 방법 리서치 (C1N86 에서 수행)**

  https://doi.org/10.3390/s25175329 / PMC12431095
    "SCADA 의 감발·센서결함 이상을 **iForest / LOF / DBSCAN** 으로 탐지하고 파워커브
     모델링과 **공동 최적화**한다."

  이 노드는 그 틀의 **첫 단계**만 한다 — 탐지로 표적을 정제한다. 공동 최적화(커브까지
  같이 적합)는 이것이 F1 을 넘을 때의 다음 단계다.

**② 사양 동결**

  탐지   `IsolationForest`(n=200, seed 20260806). **SCADA 관측 공간만** —
         `scada_ws`, 출력 정격비, 커브 이탈. 예보 피처를 쓰지 않는다(표적 정제이지
         예측 개선이 아니다). C1N88 과 **동일한 특징·시드**.
  제외율 k = **5%** 와 **10%** 두 팔. C1N88 이 상한을 6.43% / 11.33% 로 쟀고, 둘 다
         F1 을 크게 넘으므로 어느 쪽이 나은지는 실측이 정한다.
         **실행 전에 두 값만 정하고 결과를 보고 늘리지 않는다.**
  적용   그룹별로 학습행에서 상위 k% 를 제외. **예측 대상은 전 test 행** 그대로.
  분할   C1N84·C1N87 과 **동일** — fold 시작 이전 행으로만 학습, fold test 행에 예측.
         내부 KFold 없음.
  지표   `std(scada_ws - 예측)` on 전 test 행(주) / 비이상 test 행(보조).

  **탐지기는 학습 구간에서만 적합한다.** fold 마다 그 시점 이전 행으로 IsolationForest 를
  새로 적합해야 미래를 보지 않는다. C1N88 은 전 구간에서 적합했으나 그것은 **상한
  측정**이었고, 여기서는 배포 가능한 절차여야 한다.

  **타당성 가드**
    V1  `base` 의 그룹별 sigma 가 C1N84 와 ±0.01 이내.
    V2  제외 비율이 목표 k 의 ±1%p 이내.
    V3  탐지기가 fold 별로 학습 구간에서만 적합됐다(미래 미사용).

  사전확약 (V1~V3 통과시에만 판정):
    H1  두 팔 중 하나라도 `base` 보다 sigma 가 낮다.
    H2  최선 팔의 감소율이 F1 **0.62% 이상**.
    H3  **g3 개선이 최대.** C1N88 이 g3 집중도 2.56 배로 쟀으므로. N1 에서 이 예측이
        틀렸으므로 반복 검정이고, 여기서도 틀리면 "g3 이상 집중" 해석 자체를 버린다.
    H4  실측 감소율이 C1N88 의 상한(k=5% 6.43% / k=10% 11.33%) **미만**.
        넘으면 상한 계산이 틀린 것이다.

  H2 가 참이면 **F1 을 통과한 두 번째 후보**이고 C1N71 과 합산 대상이 된다.
  거짓이면 L1 레인이 상한은 있으나 도달 수단이 없다는 뜻이고, 그것으로 이 레인을 닫는다.

게이트 미수정. lockbox·외부데이터 미사용. `scada_ws` 는 teacher 표적으로만(C1N39).
제출 없음.
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
from lightgbm import LGBMRegressor
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_cycle37_band_loss import fold_rows
from m271_cycle42_teacher_restored import TEACHER_PARAMS, all_weather_columns
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C84_RECEIPT = REPORTS / "m271_c7c_chronological_receipt.json"
C88_RECEIPT = REPORTS / "m271_n2_probe_l5_open_receipt.json"
REPORT_MD = REPORTS / "m271_n2_anomaly_clean.md"
RECEIPT = REPORTS / "m271_n2_anomaly_clean_receipt.json"

NODE_ID = "C1N89_ANOMALY_CLEAN_TARGET"
LANE = "L1"
PARENT_NODE = "C1N88_N2_PROBE_L5_OPEN"

SEED = 20260806
K_ARMS = (0.05, 0.10)
V1_TOLERANCE = 0.01
DETECTION_THRESHOLD = 0.001013
RESPONSE_SLOPE = 0.164
F1_SIGMA_REDUCTION = DETECTION_THRESHOLD / RESPONSE_SLOPE


def anomaly_score(frame: pd.DataFrame, fitted: IsolationForest | None = None):
    """C1N88 과 **동일한 특징**. SCADA 관측 공간만 쓴다."""
    features = frame.loc[:, ["scada_ws", "rate"]].copy()
    features["curve_gap"] = frame["rate"] - np.clip(
        (frame["scada_ws"] - 3.0) / 9.0, 0.0, 1.0
    ) ** 3
    if fitted is None:
        fitted = IsolationForest(
            n_estimators=200, random_state=SEED, contamination="auto", n_jobs=1
        ).fit(features)
    return fitted, -fitted.score_samples(features)


def main() -> int:
    c84 = json.loads(C84_RECEIPT.read_text(encoding="utf-8"))
    c88 = json.loads(C88_RECEIPT.read_text(encoding="utf-8"))

    surface, _base, _aux = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface = surface.loc[surface["actual_kwh"].notna()].reset_index(drop=True)
    surface["rate"] = surface["actual_kwh"] / surface["capacity"]
    columns = all_weather_columns(surface)

    arms = ["base", *[f"clean{int(k*100)}" for k in K_ARMS]]
    residuals: dict[str, dict[int, list[np.ndarray]]] = {a: {} for a in arms}
    residuals_normal: dict[str, dict[int, list[np.ndarray]]] = {a: {} for a in arms}
    excluded_rates: list[float] = []
    fits = 0
    test_rows = 0

    for _probe_fold, meta in fold_rows().items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]]
        test = surface.loc[
            np.array([
                (fid, gid) in meta["keys"]
                for fid, gid in zip(surface["forecast_id"], surface["group_id"],
                                    strict=True)
            ])
        ]
        for group in (1, 2, 3):
            tr = train.loc[
                (train["group_id"] == group) & train["scada_ws"].notna()
            ].reset_index(drop=True)
            te = test.loc[
                (test["group_id"] == group) & test["scada_ws"].notna()
            ].reset_index(drop=True)
            if len(tr) < 200 or len(te) < 50:
                continue
            test_rows += len(te)

            # 탐지기는 **학습 구간에서만** 적합한다. 미래를 보지 않는다.
            detector, tr_score = anomaly_score(tr)
            _, te_score = anomaly_score(te, detector)
            te_normal = te_score < np.quantile(tr_score, 0.90)

            x_te = te.loc[:, columns].astype("float32")
            y_te = te["scada_ws"].to_numpy(dtype="float64")

            for arm in arms:
                if arm == "base":
                    keep = np.ones(len(tr), dtype=bool)
                else:
                    k = int(arm.removeprefix("clean")) / 100.0
                    cutoff = np.quantile(tr_score, 1.0 - k)
                    keep = tr_score < cutoff
                    excluded_rates.append(1.0 - float(keep.mean()))
                sub = tr.loc[keep]
                model = LGBMRegressor(**TEACHER_PARAMS)
                model.fit(sub.loc[:, columns].astype("float32"),
                          sub["scada_ws"].to_numpy(dtype="float64"))
                fits += 1
                pred = model.predict(x_te)
                residuals[arm].setdefault(group, []).append(y_te - pred)
                residuals_normal[arm].setdefault(group, []).append(
                    (y_te - pred)[te_normal]
                )

    def summarise(store: dict[str, dict[int, list[np.ndarray]]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for arm, per_group in store.items():
            entry: dict[str, float] = {}
            pooled = []
            for group, parts in per_group.items():
                joined = np.concatenate(parts)
                entry[f"g{group}"] = float(np.std(joined, ddof=1))
                pooled.append(joined)
            entry["overall"] = float(np.std(np.concatenate(pooled), ddof=1))
            out[arm] = entry
        return out

    primary = summarise(residuals)
    secondary = summarise(residuals_normal)

    c84_base = c84["arms"]["base"]
    v1 = bool(all(
        abs(primary["base"][f"g{g}"] - float(c84_base[f"g{g}"])) <= V1_TOLERANCE
        for g in (1, 2, 3)
    ))
    v2 = bool(all(
        min(abs(r - k) for k in K_ARMS) <= 0.01 for r in excluded_rates
    ))
    v3 = True  # 탐지기를 fold 별 학습 구간에서만 적합했다(위 구현이 강제).

    reductions = {
        arm: 1.0 - primary[arm]["overall"] / primary["base"]["overall"]
        for arm in arms if arm != "base"
    }
    best_arm = max(reductions, key=lambda a: reductions[a])
    best_reduction = reductions[best_arm]
    implied_total = best_reduction * RESPONSE_SLOPE
    per_group_reduction = {
        g: 1.0 - primary[best_arm][f"g{g}"] / primary["base"][f"g{g}"] for g in (1, 2, 3)
    }

    h1 = bool(best_reduction > 0.0)
    h2 = bool(best_reduction >= F1_SIGMA_REDUCTION)
    h3 = bool(max(per_group_reduction, key=lambda g: per_group_reduction[g]) == 3)
    bound_key = f"k{int(float(best_arm.removeprefix('clean')))}"
    upper = float(c88["pooled_bounds"][bound_key]["sigma_reduction_upper_bound"])
    h4 = bool(best_reduction < upper)

    if not (v1 and v2):
        verdict = "GUARD_FAILED_RESULT_VOID"
    elif h2:
        verdict = "ANOMALY_CLEANING_CLEARS_DETECTION_THRESHOLD"
    elif h1:
        verdict = "ANOMALY_CLEANING_HELPS_BUT_BELOW_DETECTION"
    else:
        verdict = "ANOMALY_CLEANING_DOES_NOT_HELP_LANE_CLOSES"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "unfrozen_by": "C1N88 (상한 11.33% > F1 0.62%)",
        "detector": {"kind": "IsolationForest", "n_estimators": 200, "seed": SEED,
                     "features": ["scada_ws", "rate", "curve_gap"],
                     "fit_scope": "fold 별 학습 구간만 — 미래 미사용"},
        "k_arms": list(K_ARMS),
        "excluded_rate_observed": excluded_rates,
        "model_fits": fits,
        "test_rows": test_rows,
        "sigma_primary": primary,
        "sigma_secondary_normal_rows": secondary,
        "reductions": reductions,
        "best_arm": best_arm,
        "best_reduction": best_reduction,
        "reduction_by_group": per_group_reduction,
        "implied_total_gain": implied_total,
        "c88_upper_bound": upper,
        "f1_sigma_reduction": F1_SIGMA_REDUCTION,
        "checks": {"V1_base_matches_c84": v1, "V2_exclusion_rate": v2,
                   "V3_detector_train_only": v3},
        "hypotheses": {
            "H1_helps": h1, "H2_clears_detection": h2,
            "H3_g3_improves_most": h3, "H4_below_upper_bound": h4,
        },
        "verdict": verdict,
        "deployable": True,
        "deployability_note": (
            "탐지기와 제외는 **학습기간 SCADA 만** 쓴다. 평가기간에는 아무것도 필요하지 "
            "않으므로 배포 가능하다."
        ),
        "dacon_upload": False,
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")

    lines = [
        "# M271 N2 — 비지도 이상 탐지로 teacher 표적 정제",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "",
        f"C1N88 이 동결을 풀었다 — 상한 {upper:.2%} 대 F1 문턱 {F1_SIGMA_REDUCTION:.2%}.",
        "",
        "**N1 과의 차이**: N1 은 '고풍속 + 커브 하향 이탈' 이라는 좁고 물리 가정이 박힌 "
        "규칙이었고 실패했다. 여기서는 규칙을 손으로 쓰지 않고 비지도 탐지에 맡긴다.",
        "",
        "## 1. sigma_v — 시간분할 test 행",
        "",
        "| 팔 | 전체 | g1 | g2 | g3 | 감소율 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        r = primary[arm]
        red = f"{reductions[arm]:+.2%}" if arm != "base" else "—"
        lines.append(
            f"| {arm} | **{r['overall']:.4f}** | {r['g1']:.4f} | {r['g2']:.4f} | "
            f"{r['g3']:.4f} | {red} |"
        )
    lines += [
        "",
        f"최선 **{best_arm}** 감소 **{best_reduction:+.2%}** -> 환산 Total "
        f"**{implied_total:+.6f}** / F1 문턱 {F1_SIGMA_REDUCTION:.2%} "
        f"({DETECTION_THRESHOLD})",
        "",
        f"그룹별: g1 {per_group_reduction[1]:+.2%} / g2 {per_group_reduction[2]:+.2%} / "
        f"g3 {per_group_reduction[3]:+.2%}",
        "",
        f"보조(비이상 test 행): base {secondary['base']['overall']:.4f} -> "
        f"{best_arm} {secondary[best_arm]['overall']:.4f}",
        "",
        "## 2. 사전확약",
        "",
        f"- V1 base 가 C1N84 와 ±{V1_TOLERANCE} 이내 -> **{v1}**",
        f"- V2 제외 비율이 목표 ±1%p -> **{v2}**",
        f"- H1 개선이 있다 -> **{h1}**",
        f"- H2 F1 문턱 통과 -> **{h2}**",
        f"- H3 g3 개선 최대 -> **{h3}**",
        f"- H4 상한 미만 -> **{h4}**",
        "",
        "## 3. 판정",
        "",
        f"**{verdict}**",
        "",
        payload["deployability_note"],
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== N2 완료 ===")
    print(f"[N2] 적합 {fits} / test 행 {test_rows:,} / 제외율 "
          f"{[round(r,4) for r in excluded_rates[:3]]}...")
    for arm in arms:
        r = primary[arm]
        red = f"  감소 {reductions[arm]:+.2%}" if arm != "base" else ""
        print(f"[N2] {arm:8s} sigma {r['overall']:.4f} (g1 {r['g1']:.4f} / "
              f"g2 {r['g2']:.4f} / g3 {r['g3']:.4f}){red}")
    print(f"[N2] 최선 {best_arm} {best_reduction:+.2%} -> Total {implied_total:+.6f} "
          f"(F1 {F1_SIGMA_REDUCTION:.2%} / 상한 {upper:.2%})")
    print(f"[N2] 그룹별 g1 {per_group_reduction[1]:+.2%} / g2 "
          f"{per_group_reduction[2]:+.2%} / g3 {per_group_reduction[3]:+.2%}")
    print(f"[N2] V1 {v1} / V2 {v2} / H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[N2] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
