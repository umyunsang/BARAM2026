"""M269 T1 Stage D: representation ceiling of the 46-bin conditional distribution.

Replaces the classifier's own (over-dispersed) probability vector with the EMPIRICAL
distribution of actual generation conditional on the model's own summary state, then takes
the exact official argmax on the champion 0.25%-capacity action grid.

This answers: if this representation's probabilities were perfectly calibrated - to the
extent its own conditioning allows - what FICR could the decision layer reach?

Two protocols:
  * strict prequential - conditional distribution estimated on strictly preceding folds only
  * same-fold oracle   - estimated on the fold being scored (hard upper bound)

Rows sharing a (group, state) share one conditional distribution and therefore one optimal
action, so the argmax is solved once per state rather than per row.

Read-only inputs. No model is fitted here, no 2024 row is read, no submission is built.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official

ROOT = Path(__file__).resolve().parents[2]
DUMPS = ROOT / "artifacts" / "backtests" / "m269-probe"
REPORTS = ROOT / "reports"

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")

# Frozen Stage D contract, declared before any Stage D score is inspected.
ACTIONS = np.round(np.arange(0.075, 1.0760001, 0.0025), 6)   # exact champion action grid
STATE_WIDTH = 0.02                                           # summary-state bin on E[y|x]
NEIGHBOURS = 400                                             # kNN support in E[y|x] space
QUERY_QUANTUM = 0.0025                                       # query cache resolution
ELIGIBLE = 0.10
BAND_4 = 0.06
BAND_3 = 0.08
DEPLOYED_TEMPERATURE = 0.5
DEPLOYED_GAMMA = 1.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fold(fold: str) -> dict[str, np.ndarray]:
    with np.load(DUMPS / f"M269_PROBE-{fold}-probability.npz") as data:
        payload = {key: data[key] for key in data.files}
    capacity = np.asarray([CAPACITIES_KWH[int(g)] for g in payload["group_id"]], dtype=float)
    payload["capacity"] = capacity
    payload["normalized_actual"] = payload["actual_kwh"] / capacity
    payload["expected"] = payload["probability"] @ payload["centers"]
    payload["state"] = np.floor(np.clip(payload["expected"], 0.0, 1.0749) / STATE_WIDTH).astype(int)
    return payload


def units(error: np.ndarray) -> np.ndarray:
    return np.select([error <= BAND_4, error <= BAND_3], [4.0, 3.0], default=0.0)


def deployed_actions(
    payload: dict[str, np.ndarray], mean_generation: dict[int, float]
) -> np.ndarray:
    """Exact champion _fixed_actions rule, recomputed from the dumped probabilities."""
    centers = payload["centers"]
    error = np.abs(ACTIONS[:, None] - centers[None, :])
    unit = units(error)
    calibrated = payload["probability"] ** (1.0 / DEPLOYED_TEMPERATURE)
    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    chosen = np.empty(len(calibrated), dtype=float)
    for group in CAPACITIES_KWH:
        mask = payload["group_id"] == group
        if not mask.any():
            continue
        probability = calibrated[mask]
        utility = -(probability @ error.T) + DEPLOYED_GAMMA * (
            probability @ (centers[None, :] * unit).T
        ) / (4.0 * mean_generation[group])
        chosen[mask] = ACTIONS[np.argmax(utility, axis=1)]
    return chosen


def empirical_action(samples: np.ndarray, mean_generation: float) -> float:
    """Exact official argmax against an empirical conditional sample of normalized generation."""
    error = np.abs(ACTIONS[:, None] - samples[None, :])
    eligible = samples >= ELIGIBLE
    count = max(int(eligible.sum()), 1)
    expected_nmae = (error * eligible[None, :]).sum(axis=1) / count
    settlement = (units(error) * (samples * eligible)[None, :]).sum(axis=1) / count
    utility = -expected_nmae + settlement / (4.0 * mean_generation)
    return float(ACTIONS[int(np.argmax(utility))])


def ceiling_actions(
    target: dict[str, np.ndarray],
    history: dict[str, np.ndarray],
    mean_generation: dict[int, float],
) -> np.ndarray:
    """Condition on the model's own E[y|x] via a fixed-size nearest-neighbour window.

    A kNN window guarantees identical support everywhere, so no row is silently demoted to
    a group-marginal constant action. Queries are cached on a 0.25%-capacity quantum.
    """
    chosen = np.empty(len(target["expected"]), dtype=float)
    for group in CAPACITIES_KWH:
        history_mask = history["group_id"] == group
        target_mask = target["group_id"] == group
        if not target_mask.any() or not history_mask.any():
            continue
        order = np.argsort(history["expected"][history_mask], kind="stable")
        sorted_expected = history["expected"][history_mask][order]
        sorted_actual = history["normalized_actual"][history_mask][order]
        window = min(NEIGHBOURS, len(sorted_actual))
        cache: dict[int, float] = {}
        for index in np.flatnonzero(target_mask):
            key = round(float(target["expected"][index]) / QUERY_QUANTUM)
            if key not in cache:
                position = int(
                    np.searchsorted(sorted_expected, key * QUERY_QUANTUM)
                )
                start = int(np.clip(position - window // 2, 0, len(sorted_actual) - window))
                cache[key] = empirical_action(
                    sorted_actual[start : start + window], mean_generation[group]
                )
            chosen[index] = cache[key]
    return chosen


def score(payload: dict[str, np.ndarray], normalized_action: np.ndarray) -> dict[str, object]:
    frame = pd.DataFrame(
        {
            "forecast_id": np.arange(len(normalized_action)).astype(str),
            "forecast_kst_dtm": pd.to_datetime(payload["forecast_kst_dtm"]),
            "group_id": payload["group_id"].astype(int),
            "actual_kwh": payload["actual_kwh"],
            "prediction_kwh": normalized_action * payload["capacity"],
        }
    )
    result = evaluate_official(frame, CAPACITIES_KWH)
    return {
        "total": result.total,
        "one_minus_nmae": result.one_minus_nmae,
        "ficr": result.ficr,
        "group_ficr": dict(result.group_ficr),
        "tier_counts": {g: dict(c) for g, c in result.settlement_tier_counts.items()},
    }


def mean_eligible(payload: dict[str, np.ndarray]) -> dict[int, float]:
    means: dict[int, float] = {}
    for group in CAPACITIES_KWH:
        values = payload["normalized_actual"][payload["group_id"] == group]
        eligible = values[values >= ELIGIBLE]
        means[group] = float(eligible.mean()) if len(eligible) else ELIGIBLE
    return means


def concat(payloads: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = (
        "group_id",
        "actual_kwh",
        "capacity",
        "normalized_actual",
        "expected",
        "forecast_kst_dtm",
    )
    return {key: np.concatenate([p[key] for p in payloads]) for key in keys}


def main() -> None:
    inputs = {
        f"M269_PROBE-{fold}-probability.npz": sha256_file(
            DUMPS / f"M269_PROBE-{fold}-probability.npz"
        )
        for fold in FOLDS
    }
    payloads = {fold: load_fold(fold) for fold in FOLDS}

    deployed: dict[str, dict[str, object]] = {}
    oracle: dict[str, dict[str, object]] = {}
    prequential: dict[str, dict[str, object]] = {}
    actions_store: dict[str, dict[str, np.ndarray]] = {}

    for index, fold in enumerate(FOLDS):
        payload = payloads[fold]
        own_mean = mean_eligible(payload)
        deployed_a = deployed_actions(payload, own_mean)
        oracle_a = ceiling_actions(payload, payload, own_mean)
        if index == 0:
            prequential_a = deployed_a
        else:
            history = concat([payloads[f] for f in FOLDS[:index]])
            history_mean = mean_eligible(history)
            prequential_a = ceiling_actions(payload, history, history_mean)
        actions_store[fold] = {
            "deployed": deployed_a,
            "oracle": oracle_a,
            "prequential": prequential_a,
        }
        deployed[fold] = score(payload, deployed_a)
        oracle[fold] = score(payload, oracle_a)
        prequential[fold] = score(payload, prequential_a)

    pooled_payload = {
        key: np.concatenate([payloads[f][key] for f in FOLDS])
        for key in ("group_id", "actual_kwh", "capacity", "forecast_kst_dtm")
    }
    for label, block in (
        ("deployed", deployed),
        ("oracle", oracle),
        ("prequential", prequential),
    ):
        merged = np.concatenate([actions_store[f][label] for f in FOLDS])
        block["pooled"] = score(pooled_payload, merged)

    lines: list[str] = []
    lines.append("# M269 T1 Stage D — 46-bin 표현 천장\n")
    lines.append("- 모델 확률을 **자기 상태 조건부 경험적 분포**로 교체 후 정확-지표 argmax")
    lines.append(f"- 행동 격자: 챔피언과 동일 {len(ACTIONS)}점 (0.25% 용량 간격)")
    lines.append(f"- 조건부: E[y|x] 기준 최근접 {NEIGHBOURS}개 이웃 창 (폴백 없음)")
    lines.append("- 신규 모델 적합 없음, 2024 접근 없음, 제출물 없음\n")

    lines.append("## 1. FICR\n")
    lines.append("| fold | 배포 | 시간순안전 천장 | 동일fold 오라클 천장 |")
    lines.append("|---|---:|---:|---:|")
    for fold in (*FOLDS, "pooled"):
        lines.append(
            f"| {fold} | {deployed[fold]['ficr']:.6f} | "
            f"{prequential[fold]['ficr']:.6f} "
            f"({prequential[fold]['ficr'] - deployed[fold]['ficr']:+.6f}) | "
            f"{oracle[fold]['ficr']:.6f} "
            f"({oracle[fold]['ficr'] - deployed[fold]['ficr']:+.6f}) |"
        )

    lines.append("\n## 2. Total\n")
    lines.append("| fold | 배포 | 시간순안전 천장 | 동일fold 오라클 천장 |")
    lines.append("|---|---:|---:|---:|")
    for fold in (*FOLDS, "pooled"):
        lines.append(
            f"| {fold} | {deployed[fold]['total']:.6f} | "
            f"{prequential[fold]['total']:.6f} "
            f"({prequential[fold]['total'] - deployed[fold]['total']:+.6f}) | "
            f"{oracle[fold]['total']:.6f} "
            f"({oracle[fold]['total'] - deployed[fold]['total']:+.6f}) |"
        )

    lines.append("\n## 3. 그룹별 FICR (pooled)\n")
    lines.append("| 프로토콜 | g1 | g2 | g3 |")
    lines.append("|---|---:|---:|---:|")
    for label, block in (("배포", deployed), ("시간순안전", prequential), ("오라클", oracle)):
        g = block["pooled"]["group_ficr"]
        lines.append(f"| {label} | {g[1]:.6f} | {g[2]:.6f} | {g[3]:.6f} |")

    (REPORTS / "m269_stage_d_ceiling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "stage": "M269_T1_STAGE_D_REPRESENTATION_CEILING",
        "action_grid_points": len(ACTIONS),
        "neighbours": NEIGHBOURS,
        "query_quantum": QUERY_QUANTUM,
        "inputs_sha256": inputs,
        "deployed": deployed,
        "strict_prequential_ceiling": prequential,
        "same_fold_oracle_ceiling": oracle,
        "model_fits": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": [],
    }
    (REPORTS / "m269_stage_d_ceiling_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    for fold in (*FOLDS, "pooled"):
        print(
            f"{fold:>12} deployed FICR={deployed[fold]['ficr']:.6f} | "
            f"prequential ceiling={prequential[fold]['ficr']:.6f} | "
            f"oracle ceiling={oracle[fold]['ficr']:.6f}"
        )


if __name__ == "__main__":
    main()
