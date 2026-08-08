"""Build a chronology-safe champion from fixed M68 and M72 policy families."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_sequence_classifier import CAPACITIES, OUTPUT, _score, _sha256

from baram.evaluation.official import evaluate_group_component

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
FAMILY_FILES = {
    "dev-2023-Q2": {
        "M68": "M68_SITEWIND_CLASS_ITER-dev-2023-Q2-policies.parquet",
        "M72": "M72_BIN020-dev-2023-Q2-policies.parquet",
    },
    "dev-2023-Q3": {
        "M68": "M68_SITEWIND_CLASS_ITER-dev-2023-Q3-policies.parquet",
        "M72": "M72F_BIN020_ITER40-dev-2023-Q3-policies.parquet",
    },
    "dev-2023-Q4": {
        "M68": "M68_SITEWIND_CLASS_ITER-dev-2023-Q4-policies.parquet",
        "M72": "M72_BIN020-dev-2023-Q4-policies.parquet",
    },
}


def _candidate_predictions(
    families: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    return {
        f"{family}:{column}": frame[column].to_numpy(dtype=float)
        for family, frame in families.items()
        for column in frame.columns
        if column not in BASE_COLUMNS
    }


def _group_total(frame: pd.DataFrame, group_id: int) -> float:
    component = evaluate_group_component(frame, group_id, CAPACITIES[group_id])
    return 0.5 * (1.0 - component.nmae) + 0.5 * component.ficr


def main() -> None:
    candidate_id = "M95_STRICT_FAMILY"
    families = {
        fold: {
            family: pd.read_parquet(OUTPUT / filename)
            for family, filename in FAMILY_FILES[fold].items()
        }
        for fold in FOLDS
    }
    candidates = {
        fold: _candidate_predictions(fold_families)
        for fold, fold_families in families.items()
    }
    q2 = pd.read_parquet(OUTPUT / "M50_GEOM_CLASS_TEMP075_G05-oof.parquet")
    output_parts = [
        q2.loc[q2["fold_id"].eq(FOLDS[0]), [*BASE_COLUMNS, "prediction_kwh"]]
        .copy()
        .assign(fold_id=FOLDS[0], model_id=candidate_id)
    ]
    selections: dict[str, dict[str, object]] = {}
    fold_scores = {FOLDS[0]: _score(output_parts[0])}
    for fold_index, fold in enumerate(FOLDS[1:], start=1):
        history_folds = FOLDS[:fold_index]
        history = pd.concat(
            [families[item]["M68"][BASE_COLUMNS] for item in history_folds],
            ignore_index=True,
        )
        common = set(candidates[fold])
        for history_fold in history_folds:
            common.intersection_update(candidates[history_fold])
        application = families[fold]["M68"][BASE_COLUMNS].copy()
        prediction = np.empty(len(application), dtype=float)
        fold_selection: dict[str, object] = {}
        for group_id in CAPACITIES:
            history_mask = history["group_id"].eq(group_id).to_numpy()
            application_mask = application["group_id"].eq(group_id).to_numpy()
            scored: list[tuple[float, str]] = []
            for name in sorted(common):
                history_prediction = np.concatenate(
                    [candidates[item][name] for item in history_folds]
                )
                diagnostic = history.loc[history_mask, BASE_COLUMNS].copy()
                diagnostic["prediction_kwh"] = history_prediction[history_mask]
                scored.append((_group_total(diagnostic, group_id), name))
            selected_score, selected_name = max(scored)
            prediction[application_mask] = candidates[fold][selected_name][
                application_mask
            ]
            fold_selection[str(group_id)] = {
                "policy": selected_name,
                "preceding_score": selected_score,
                "history_folds": list(history_folds),
            }
        application["prediction_kwh"] = prediction
        application["fold_id"] = fold
        application["model_id"] = candidate_id
        output_parts.append(application)
        selections[fold] = fold_selection
        fold_scores[fold] = _score(application)
    output = pd.concat(output_parts, ignore_index=True)
    output_path = OUTPUT / f"{candidate_id}-oof.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "scope": "Q2 fixed; Q3 selected on Q2; Q4 selected on pooled Q2-Q3",
        "parent_families": FAMILY_FILES,
        "selections": selections,
        "fold_scores": fold_scores,
        "pooled": _score(output),
        "prediction_path": str(output_path.relative_to(Path.cwd())),
        "prediction_sha256": _sha256(output_path),
        "observed_scada_feature_count": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{candidate_id}-oof.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
