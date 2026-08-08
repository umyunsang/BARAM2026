"""Select one feature-screened policy on Q2 and apply it unchanged later."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from run_sequence_classifier import OUTPUT, _score, _sha256

BASE_COLUMNS = ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
PARENT_PATHS = {
    FOLDS[0]: "M102_TOP100-dev-2023-Q2-policies.parquet",
    FOLDS[1]: "M102_TOP100_I60-dev-2023-Q3-policies.parquet",
    FOLDS[2]: "M102_TOP100-dev-2023-Q4-policies.parquet",
}


def main() -> None:
    candidate_id = "M103_STRICT_TOP100"
    policies = {
        fold: pd.read_parquet(OUTPUT / PARENT_PATHS[fold]) for fold in FOLDS
    }
    candidate_columns = sorted(
        set(policies[FOLDS[0]].columns).difference(BASE_COLUMNS)
    )
    scored: list[tuple[float, str, dict[str, float]]] = []
    for column in candidate_columns:
        candidate = policies[FOLDS[0]][BASE_COLUMNS].copy()
        candidate["prediction_kwh"] = policies[FOLDS[0]][column]
        score = _score(candidate)
        scored.append((score["total"], column, score))
    _, selected_policy, selection_score = max(scored)
    q2_control = pd.read_parquet(OUTPUT / "M50_GEOM_CLASS_TEMP075_G05-oof.parquet")
    output_parts = [
        q2_control.loc[
            q2_control["fold_id"].eq(FOLDS[0]),
            [*BASE_COLUMNS, "prediction_kwh"],
        ]
        .copy()
        .assign(fold_id=FOLDS[0], model_id=candidate_id)
    ]
    fold_scores = {FOLDS[0]: _score(output_parts[0])}
    for fold in FOLDS[1:]:
        output = policies[fold][BASE_COLUMNS].copy()
        output["prediction_kwh"] = policies[fold][selected_policy]
        output["fold_id"] = fold
        output["model_id"] = candidate_id
        output_parts.append(output)
        fold_scores[fold] = _score(output)
    output = pd.concat(output_parts, ignore_index=True)
    output_path = OUTPUT / f"{candidate_id}-oof.parquet"
    output.to_parquet(output_path, index=False)
    receipt = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "scope": "Q2 control; one top-100 policy selected on Q2 and fixed for Q3-Q4",
        "parent_candidate_ids": ["M102_TOP100", "M102_TOP100_I60"],
        "selected_iteration": 60,
        "selected_policy": selected_policy,
        "q2_selection_score": selection_score,
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
