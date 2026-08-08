"""Recover and evaluate frozen S17-N8 COST5 predictions without any refit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official
from baram.evaluation.prequential import (
    chronological_outer_splits,
    run_prequential_protocol,
    validate_issuance_atoms,
)
from baram.loop.events import EventStore

KEYS = ("fold_id", "group_id", "forecast_kst_dtm")
ACTIONS = ("D", "M102_TOP100", "M113_LGBM_DART", "M115_XGBOOST", "CHAMPION")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _vector_hash(frame: pd.DataFrame, column: str, fold: str) -> str:
    part = frame.loc[frame["fold_id"].eq(fold)].sort_values(list(KEYS), kind="stable")
    key_bytes = part[list(KEYS)].astype(str).agg("|".join, axis=1).str.cat(sep="\n").encode()
    values = np.ascontiguousarray(part[column].to_numpy(dtype="<f8")).tobytes()
    return hashlib.sha256(key_bytes + b"\n" + values).hexdigest()


def _load_n8_module(repo: Path) -> Any:
    path = repo / "scripts/run_cost5_spo_plus.py"
    specification = importlib.util.spec_from_file_location("frozen_n8_cost5", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load frozen N8 implementation")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _verify_inputs(repo: Path, predeclaration: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N9 frozen input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N9 frozen bundle digest mismatch")
    partial = json.loads(
        (repo / "artifacts/backtests/s17_n8_cost5_spo_plus/PARTIAL_PRE_SCORE.json").read_text()
    )
    if partial["status"] != "PARTIAL_PRE_SCORE_FROZEN_NO_FAMILY_NO_EVALUATION":
        raise RuntimeError("N8 partial status mismatch")
    for name, digest in partial["outputs"].items():
        path = repo / "artifacts/backtests/s17_n8_cost5_spo_plus" / name
        if _sha256(path) != digest:
            raise RuntimeError(f"N8 partial mutation: {name}")
    return frozen


def _standardize(raw: np.ndarray, parameters: pd.DataFrame) -> np.ndarray:
    medians = parameters["median"].to_numpy(dtype=float)
    means = parameters["mean"].to_numpy(dtype=float)
    scales = parameters["scale"].to_numpy(dtype=float)
    filled = np.where(np.isfinite(raw), raw, medians[None, :])
    return (filled - means[None, :]) / scales[None, :]


def materialize(repo: Path, predeclaration: Path, output_dir: Path) -> dict[str, Any]:
    frozen = _verify_inputs(repo, predeclaration)
    n8 = _load_n8_module(repo)
    partial_dir = repo / "artifacts/backtests/s17_n8_cost5_spo_plus"
    predictions = pd.read_parquet(partial_dir / "predictions.parquet")
    predictions["forecast_kst_dtm"] = pd.to_datetime(predictions["forecast_kst_dtm"])
    features_frame = pd.read_parquet(partial_dir / "features.parquet")
    coefficients_frame = pd.read_parquet(partial_dir / "coefficients.parquet")
    preprocessing_frame = pd.read_parquet(partial_dir / "preprocessing.parquet")
    actions = pd.read_parquet(
        repo / "artifacts/backtests/s17_n7_strict_actions/actions.parquet"
    )
    actions["forecast_kst_dtm"] = pd.to_datetime(actions["forecast_kst_dtm"])
    if not actions[list(KEYS)].equals(predictions[list(KEYS)]):
        raise RuntimeError("N7/N8 prediction key mismatch")
    if not actions[list(KEYS)].equals(features_frame[list(KEYS)]):
        raise RuntimeError("N7/N8 feature key mismatch")
    checked = validate_issuance_atoms(actions)
    splits = chronological_outer_splits(checked, burn_in_folds=1)
    feature_names = [column for column in features_frame if column not in KEYS]
    raw_features = features_frame[feature_names].to_numpy(dtype=float)

    verification: dict[str, Any] = {}
    max_prediction_error = 0.0
    for split in splits:
        parameters = preprocessing_frame.loc[
            preprocessing_frame["test_fold"].eq(split.test_fold)
        ].copy()
        if parameters["feature"].tolist() != feature_names:
            raise RuntimeError(f"{split.test_fold}: preprocessing feature order mismatch")
        coefficients_part = coefficients_frame.loc[
            coefficients_frame["test_fold"].eq(split.test_fold)
        ].copy()
        expected_rows = [*feature_names, "__intercept__"]
        if coefficients_part["feature"].tolist() != expected_rows:
            raise RuntimeError(f"{split.test_fold}: coefficient feature order mismatch")
        coefficients = coefficients_part[list(ACTIONS)].to_numpy(dtype=float)
        train_features = _standardize(
            raw_features[split.train_index.to_numpy(dtype=int)], parameters
        )
        test_features = _standardize(
            raw_features[split.test_index.to_numpy(dtype=int)], parameters
        )
        selected = n8.predict_action(coefficients, test_features)
        test_actions = actions.loc[split.test_index, list(ACTIONS)].to_numpy(dtype=float)
        reproduced = test_actions[np.arange(len(selected)), selected]
        stored = predictions.loc[
            split.test_index, "COST5_SPO_PLUS"
        ].to_numpy(dtype=float)
        prediction_error = float(np.max(np.abs(reproduced - stored)))
        max_prediction_error = max(max_prediction_error, prediction_error)
        stored_choice = predictions.loc[split.test_index, "selected_action"].to_numpy()
        expected_choice = np.asarray(ACTIONS, dtype=object)[selected]
        if prediction_error > 1e-12 or not np.array_equal(stored_choice, expected_choice):
            raise RuntimeError(f"{split.test_fold}: frozen prediction reproduction failed")

        cost, active = n8.official_training_cost(actions.loc[split.train_index])
        active_features = train_features[active]
        active_cost = cost[active]
        design = np.column_stack(
            [active_features, np.ones(len(active_features), dtype=float)]
        )
        zero = np.zeros(coefficients.size, dtype=float)
        initial, _ = n8.spo_plus_value_gradient(zero, design, active_cost, n8.L2)
        final, gradient = n8.spo_plus_value_gradient(
            coefficients.ravel(), design, active_cost, n8.L2
        )
        finite_descent = bool(
            np.isfinite(coefficients).all()
            and np.isfinite(final)
            and np.isfinite(gradient).all()
            and final <= initial + 1e-12
        )
        if not finite_descent:
            raise RuntimeError(f"{split.test_fold}: optimizer verification failed")
        verification[split.test_fold] = {
            "prediction_max_abs_error": prediction_error,
            "choice_exact": True,
            "train_rows": len(split.train_index),
            "train_valid_metric_rows": int(active.sum()),
            "test_rows": len(split.test_index),
            "initial_objective": float(initial),
            "final_objective": float(final),
            "final_gradient_inf_norm": float(np.max(np.abs(gradient))),
            "finite_descent_acceptance": finite_descent,
            "optimizer_reinvoked": False,
            "original_scipy_status": "unavailable_after_post-fit_serialization_exception",
            "choice_counts": {
                action: int(np.count_nonzero(selected == offset))
                for offset, action in enumerate(ACTIONS)
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.parquet"
    shutil.copyfile(partial_dir / "predictions.parquet", prediction_path)
    if _sha256(prediction_path) != _sha256(partial_dir / "predictions.parquet"):
        raise RuntimeError("prediction byte-copy mismatch")
    verification_path = output_dir / "optimizer_verification.json"
    verification_payload = {
        "schema_version": 1,
        "node_id": "S17-N9_COST5_NO_REFIT_MATERIALIZATION_RECOVERY",
        "folds": verification,
        "max_prediction_abs_error": max_prediction_error,
        "optimizer_invocations": 0,
        "assessment_score_calls": 0,
    }
    verification_path.write_text(
        json.dumps(verification_payload, ensure_ascii=False, indent=2) + "\n"
    )

    n7_provenance = pd.read_parquet(
        repo
        / "artifacts/backtests/s17_n7_strict_actions/procedure_provenance.parquet"
    )
    provenance_rows: list[dict[str, Any]] = []
    predeclaration_hash = _sha256(predeclaration)
    for split in splits:
        champion = n7_provenance.loc[
            n7_provenance["model_id"].eq("CHAMPION")
            & n7_provenance["test_fold"].eq(split.test_fold)
        ]
        if len(champion) != 1:
            raise RuntimeError("missing N7 champion provenance")
        champion_row = champion.iloc[0].to_dict()
        champion_row["fit_max_time"] = pd.to_datetime(champion_row["fit_max_time"])
        champion_row["selection_max_time"] = pd.to_datetime(
            champion_row["selection_max_time"]
        )
        provenance_rows.append(champion_row)
        provenance_rows.append(
            {
                "model_id": "COST5_SPO_PLUS",
                "test_fold": split.test_fold,
                "fit_max_time": split.train_max_label_time,
                "selection_max_time": split.train_max_label_time,
                "policy_id": "LINEAR_SPO_PLUS_L2_0p001_ONE_SHOT_N8_FROZEN",
                "predeclaration_sha256": predeclaration_hash,
                "prediction_sha256": _vector_hash(
                    predictions, "COST5_SPO_PLUS", split.test_fold
                ),
                "weights_fit": "past_only_expanding",
            }
        )
    provenance = pd.DataFrame(provenance_rows).sort_values(
        ["test_fold", "model_id"], kind="stable"
    )
    for column in ("fit_max_time", "selection_max_time"):
        provenance[column] = pd.to_datetime(provenance[column])
    provenance_path = output_dir / "procedure_provenance.parquet"
    provenance.to_parquet(provenance_path, index=False)

    output_hashes = {
        path.name: _sha256(path)
        for path in (prediction_path, provenance_path, verification_path)
    }
    family = {
        "schema_version": 1,
        "node_id": "S17-N9_COST5_NO_REFIT_MATERIALIZATION_RECOVERY",
        "predeclaration_sha256": predeclaration_hash,
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "code_sha256": _sha256(Path(__file__)),
        "frozen_n8_code_sha256": _sha256(repo / "scripts/run_cost5_spo_plus.py"),
        "family": ["CHAMPION", "COST5_SPO_PLUS"],
        "incumbent": "CHAMPION",
        "action_order": list(ACTIONS),
        "outputs": output_hashes,
        "optimizer_verification": verification,
        "prediction_vectors": {
            model: {
                split.test_fold: _vector_hash(predictions, model, split.test_fold)
                for split in splits
            }
            for model in ("CHAMPION", "COST5_SPO_PLUS")
        },
        "materialization_forbidden_access": {
            "optimizer_or_model_fits": 0,
            "q3_q4_metric_or_component_calls": 0,
            "lockbox_2024": False,
            "test_period": False,
            "rejected_ecmwf": False,
            "dacon_actions": [],
        },
    }
    family_path = output_dir / "family_manifest.json"
    family_path.write_text(json.dumps(family, ensure_ascii=False, indent=2) + "\n")
    return {
        "family_manifest_sha256": _sha256(family_path),
        "outputs": output_hashes,
        "max_prediction_abs_error": max_prediction_error,
        "optimizer_invocations": 0,
        "assessment_score_calls": 0,
    }


def _metric_frame(frame: pd.DataFrame, prediction: str) -> pd.DataFrame:
    metric = frame[["forecast_kst_dtm", "group_id", "actual_kwh", prediction]].copy()
    metric.insert(0, "forecast_id", np.arange(len(metric), dtype=np.int64))
    return metric.rename(columns={prediction: "prediction_kwh"})


def _score_to_json(score: Any) -> dict[str, Any]:
    return {
        "total": float(score.total),
        "one_minus_nmae": float(score.one_minus_nmae),
        "ficr": float(score.ficr),
        "group_nmae": {str(key): float(value) for key, value in score.group_nmae.items()},
        "group_ficr": {str(key): float(value) for key, value in score.group_ficr.items()},
    }


def evaluate(
    repo: Path,
    predeclaration: Path,
    output_dir: Path,
    expected_family_sha256: str,
) -> dict[str, Any]:
    _verify_inputs(repo, predeclaration)
    family_path = output_dir / "family_manifest.json"
    if _sha256(family_path) != expected_family_sha256:
        raise RuntimeError("N9 family freeze hash mismatch")
    family = json.loads(family_path.read_text())
    for name, digest in family["outputs"].items():
        if _sha256(output_dir / name) != digest:
            raise RuntimeError(f"post-freeze N9 output mutation: {name}")
    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    predictions["forecast_kst_dtm"] = pd.to_datetime(
        predictions["forecast_kst_dtm"]
    )
    provenance = pd.read_parquet(output_dir / "procedure_provenance.parquet")
    event_store = EventStore(
        repo, repo / "artifacts/registry/loop_events_s17.sqlite"
    )
    protocol = run_prequential_protocol(
        predictions,
        prediction_columns=["CHAMPION", "COST5_SPO_PLUS"],
        incumbent="CHAMPION",
        capacities=CAPACITIES_KWH,
        procedure_provenance=provenance,
        family_manifest_sha256=expected_family_sha256,
        comparison_index=1,
        event_store=event_store,
        n_rep=4999,
        seed=20260808,
        block_lengths=(3, 7, 14),
        margin_total=0.001635,
    )
    outer = predictions.loc[~predictions["fold_id"].eq("dev-2023-Q2")].copy()
    incumbent = evaluate_official(_metric_frame(outer, "CHAMPION"), CAPACITIES_KWH)
    candidate = evaluate_official(
        _metric_frame(outer, "COST5_SPO_PLUS"), CAPACITIES_KWH
    )
    scores = {
        "CHAMPION": _score_to_json(incumbent),
        "COST5_SPO_PLUS": _score_to_json(candidate),
    }
    delta = float(candidate.total - incumbent.total)
    protocol_delta = float(
        protocol["blocks"]["7"]["joint_max_t"]["candidates"]
        ["COST5_SPO_PLUS"]["observed_delta_total"]
    )
    if abs(delta - protocol_delta) >= 1e-12:
        raise RuntimeError("N9 point/protocol delta disagreement")
    promoted = bool(
        delta >= 0.001635
        and protocol["promotion_stable_all_blocks"]["COST5_SPO_PLUS"]
        and protocol["inference"] == "SUPPORTED"
    )
    result = {
        "schema_version": 1,
        "node_id": "S17-N9_COST5_NO_REFIT_MATERIALIZATION_RECOVERY",
        "family_manifest_sha256": expected_family_sha256,
        "comparison_index": 1,
        "scores": scores,
        "delta_total": delta,
        "target_total": 0.66,
        "candidate_reaches_target_point": bool(candidate.total >= 0.66),
        "promotion_supported": promoted,
        "protocol": protocol,
        "score_calls": {
            "outer_point_official": 2,
            "strict_prequential_protocol": 1,
        },
        "model_fits_or_optimizer_invocations": 0,
        "evidence_label": (
            "retrospective_chronology_repaired_multiplicity_aware_not_fresh_holdout"
        ),
        "forbidden_access": {
            "lockbox_2024": False,
            "test_period": False,
            "rejected_ecmwf": False,
            "dacon_actions": [],
        },
    }
    evaluation_path = output_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("materialize", "evaluate"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=Path("reports/s17_n9_cost5_recovery_predeclaration.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/backtests/s17_n9_cost5_recovery"),
    )
    parser.add_argument("--family-sha256", default="")
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    if args.phase == "materialize":
        result = materialize(repo, predeclaration, output_dir)
    else:
        if len(args.family_sha256) != 64:
            raise RuntimeError("evaluate requires frozen --family-sha256")
        result = evaluate(
            repo,
            predeclaration,
            output_dir,
            args.family_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
