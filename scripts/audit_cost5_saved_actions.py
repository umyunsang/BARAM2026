"""Audit frozen COST5 action artifacts without fitting or scoring.

This is the hard G0 gate for S17-N6.  It reads only artifact keys and existing
receipt metadata.  It never imports the official scorer or accesses 2024/test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
OUTER = ("dev-2023-Q3", "dev-2023-Q4")
DEPENDENT = ("M102_TOP100", "M113_LGBM_DART", "M115_XGBOOST")
KEYS = ("group_id", "forecast_kst_dtm")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _operating_day(timestamp: pd.Series) -> pd.Series:
    return (pd.to_datetime(timestamp) - timedelta(hours=1)).dt.normalize()


def _basis(day: pd.Timestamp) -> pd.Timestamp:
    return day - timedelta(days=1) + timedelta(hours=14)


def _label_available(day: pd.Timestamp) -> pd.Timestamp:
    return day + timedelta(days=1)


def _read_keys(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(KEYS))
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    return frame


def _json_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "candidate_id": payload.get("candidate_id"),
        "fold_id": payload.get("fold_id"),
        "selected_iteration": payload.get("selected_iteration"),
        "has_fit_max_time": "fit_max_time" in payload,
        "has_selection_max_time": "selection_max_time" in payload,
        "has_predeclaration_sha256": "predeclaration_sha256" in payload,
        "selected_policy_fields": sorted(
            key for key in payload if "policy" in key and key != "policy_path"
        ),
    }


def run(repo: Path, predeclaration: Path, output: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    declared = frozen["input_bundle"]["files"]
    hashes = {
        relative: {"declared": digest, "observed": _sha256(repo / relative)}
        for relative, digest in declared.items()
    }
    hashes_pass = all(row["declared"] == row["observed"] for row in hashes.values())

    d_all = pd.read_parquet(
        repo / "research/nodes/S7-N8_D_keys.parquet",
        columns=["fold_id", *KEYS],
    )
    d_all["forecast_kst_dtm"] = pd.to_datetime(d_all["forecast_kst_dtm"])
    d_source = (repo / "research/nodes/s7_savemembers.py").read_text()
    d_valid_time_cutoff = all(
        token in d_source
        for token in ("tr=np.asarray(idx<a)", "mu.fit", "clf.fit")
    )

    fit_rows: list[dict[str, Any]] = []
    completeness: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    iteration_by_model: dict[str, dict[str, Any]] = {}

    for model in DEPENDENT:
        metadata[model] = {}
        iteration_by_model[model] = {}
        for fold in FOLDS:
            receipt_path = (
                repo
                / "artifacts/backtests/metric-aligned-probe"
                / f"{model}-{fold}.json"
            )
            record = _json_metadata(receipt_path)
            metadata[model][fold] = record
            iteration_by_model[model][fold] = record["selected_iteration"]

    for outer_fold, prior_fold in zip(OUTER, FOLDS[:2], strict=True):
        test_d = d_all.loc[d_all["fold_id"].eq(outer_fold), list(KEYS)].copy()
        test_min_day = _operating_day(test_d["forecast_kst_dtm"]).min()
        test_basis = _basis(test_min_day)
        prior_d = d_all.loc[d_all["fold_id"].eq(prior_fold), list(KEYS)].copy()
        max_valid = prior_d["forecast_kst_dtm"].max()
        implied_fit_day = _operating_day(pd.Series([max_valid])).iloc[0]
        implied_label_time = _label_available(implied_fit_day)
        fit_rows.append(
            {
                "model_id": "D",
                "test_fold": outer_fold,
                "source_cutoff": "idx < held valid-time start",
                "implied_fit_max_valid_time": max_valid.isoformat(),
                "implied_fit_max_label_available_time": implied_label_time.isoformat(),
                "test_min_basis_time": test_basis.isoformat(),
                "strictly_before_basis": bool(implied_label_time < test_basis),
                "source_pattern_verified": d_valid_time_cutoff,
            }
        )
        for model in DEPENDENT:
            record = metadata[model][outer_fold]
            fit_rows.append(
                {
                    "model_id": model,
                    "test_fold": outer_fold,
                    "fit_max_time": None,
                    "selection_max_time": None,
                    "test_min_basis_time": test_basis.isoformat(),
                    "strictly_before_basis": False,
                    "reason": (
                        "existing receipt supplies no fit/selection timestamp "
                        "or predeclaration proof"
                    ),
                    "receipt_fields": record,
                }
            )

        keyed = test_d.drop_duplicates(list(KEYS)).set_index(list(KEYS))
        source_counts = {"D": len(keyed)}
        for model in DEPENDENT:
            action = _read_keys(
                repo
                / "artifacts/backtests/metric-aligned-probe"
                / f"{model}-{outer_fold}-policies.parquet"
            ).drop_duplicates(list(KEYS))
            source_counts[model] = len(action)
            keyed = keyed.join(
                action.set_index(list(KEYS)).assign(**{f"has_{model}": True}),
                how="inner",
            )
        common = keyed.reset_index()
        common["operating_day"] = _operating_day(common["forecast_kst_dtm"])
        per_day = common.groupby("operating_day").size()
        incomplete = {
            day.isoformat(): int(count) for day, count in per_day.items() if count != 72
        }
        completeness[outer_fold] = {
            "source_rows": source_counts,
            "common_rows": len(common),
            "complete_days": int((per_day == 72).sum()),
            "incomplete_days": incomplete,
            "mechanical_action": "drop incomplete days before any conditional evaluation",
        }

    provenance_pass = all(row["strictly_before_basis"] for row in fit_rows)
    m102_iterations = iteration_by_model["M102_TOP100"]
    same_fold_selection_evidence = len(set(m102_iterations.values())) > 1
    selection_pass = (
        all(
            record["has_selection_max_time"] and record["has_predeclaration_sha256"]
            for model in metadata.values()
            for record in model.values()
        )
        and not same_fold_selection_evidence
    )
    g0 = {
        "G0a_fit_label_chronology": provenance_pass,
        "G0b_selection_chronology": selection_pass,
        "G0c_complete_atoms_after_mechanical_exclusion": all(
            row["complete_days"] > 0 for row in completeness.values()
        ),
        "G0d_hashes_and_forbidden_inputs": hashes_pass,
    }
    overall = all(g0.values())
    result = {
        "schema_version": 1,
        "node_id": "S17-N6_COST5_SPO_PLUS_PREQUENTIAL",
        "audit_kind": "saved_action_input_provenance_no_fit_no_score",
        "predeclaration": {
            "path": str(predeclaration.relative_to(repo)),
            "sha256": _sha256(predeclaration),
        },
        "input_hashes": hashes,
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "fit_chronology": fit_rows,
        "selection_metadata": metadata,
        "selected_iterations": iteration_by_model,
        "m102_fold_varying_selected_iteration": same_fold_selection_evidence,
        "common_atom_audit": completeness,
        "gate": {**g0, "overall": overall},
        "action": "CONTINUE_TO_SPO_PLUS" if overall else "STOP_BEFORE_FIT_OR_SCORE",
        "forbidden_path_checks": {
            "ecmwf_rejected_read": False,
            "lockbox_2024_read": False,
            "test_period_read": False,
            "official_score_calls": 0,
            "model_fits": 0,
            "dacon_actions": [],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=Path("reports/s17_n6_cost5_spo_plus_predeclaration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/backtests/s17_n6_cost5_spo_plus/input_provenance_audit.json"
        ),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    output = args.output
    if not output.is_absolute():
        output = repo / output
    result = run(repo, predeclaration, output)
    print(json.dumps({"gate": result["gate"], "action": result["action"]}, indent=2))


if __name__ == "__main__":
    main()
