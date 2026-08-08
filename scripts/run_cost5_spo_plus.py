"""Fit and evaluate the one-shot S17-N8 linear SPO+ action selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official, settlement_unit
from baram.evaluation.prequential import (
    chronological_outer_splits,
    run_prequential_protocol,
    validate_issuance_atoms,
)
from baram.loop.events import EventStore

KEYS = ("fold_id", "group_id", "forecast_kst_dtm")
BASE_ACTIONS = ("D", "M102_TOP100", "M113_LGBM_DART", "M115_XGBOOST")
ACTIONS = (*BASE_ACTIONS, "CHAMPION")
STATE = (
    "atm__hub_consensus",
    "ldaps_spatial__idw__wind50max_speed",
    "gfs_spatial__idw__wind100_speed",
    "ldaps_spatial__idw__etc_0_blh",
    "atm__alpha_100_80",
    "atm__theta850_minus_t2",
    "g2__l50x__rng",
    "g2__l50x__std",
    "atm__gust_factor",
    "cal__hour_sin",
    "cal__hour_cos",
    "cal__doy_sin",
    "cal__doy_cos",
)
L2 = 0.001
MAXITER = 500
FTOL = 1e-12
GTOL = 1e-7


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


def _verify_frozen_inputs(repo: Path, frozen: dict[str, Any]) -> None:
    observed = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("frozen N8 input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("frozen N8 bundle digest mismatch")


def _surface_state(repo: Path, actions: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, str(repo / "research/nodes"))
    from harness import surface

    surface_frame, _, _ = surface(("G2", "DROP:grid__"))
    missing = sorted(set(STATE) - set(surface_frame.columns))
    if missing:
        raise RuntimeError(f"missing frozen state features: {missing}")
    surface_index = pd.MultiIndex.from_arrays(
        [surface_frame["grp"].to_numpy(), surface_frame.index],
        names=["group_id", "forecast_kst_dtm"],
    )
    if surface_index.has_duplicates:
        raise RuntimeError("state surface has duplicate group/time keys")
    action_index = pd.MultiIndex.from_arrays(
        [actions["group_id"], pd.to_datetime(actions["forecast_kst_dtm"])],
        names=["group_id", "forecast_kst_dtm"],
    )
    return pd.DataFrame(
        surface_frame[list(STATE)].to_numpy(dtype=float),
        index=surface_index,
        columns=list(STATE),
    ).reindex(action_index)


def build_features(repo: Path, actions: pd.DataFrame) -> pd.DataFrame:
    capacity = actions["group_id"].map(CAPACITIES_KWH).to_numpy(dtype=float)
    base_cf = actions[list(BASE_ACTIONS)].to_numpy(dtype=float) / capacity[:, None]
    champion_cf = actions["CHAMPION"].to_numpy(dtype=float) / capacity
    features = pd.DataFrame(
        {f"a_{action}": base_cf[:, offset] for offset, action in enumerate(BASE_ACTIONS)}
    )
    features["a_CHAMPION"] = champion_cf
    features["a_mean"] = base_cf.mean(axis=1)
    features["a_sd"] = base_cf.std(axis=1)
    features["a_rng"] = base_cf.max(axis=1) - base_cf.min(axis=1)

    d_keys = pd.read_parquet(repo / "research/nodes/S7-N8_D_keys.parquet")
    probability = np.load(repo / "research/nodes/S7-N8_D_prob.npy")
    if len(d_keys) != len(probability):
        raise RuntimeError("D probability alignment length mismatch")
    centers = (np.arange(probability.shape[1]) + 0.5) * 0.04
    d_mean = probability @ centers
    d_spread = np.sqrt(
        (probability * (centers[None, :] - d_mean[:, None]) ** 2).sum(axis=1)
    )
    moments = d_keys[[*list(KEYS), "mean_gen_g"]].copy()
    moments["d_mean"] = d_mean
    moments["d_spread"] = d_spread
    aligned = actions[list(KEYS)].merge(moments, on=list(KEYS), how="left", validate="one_to_one")
    if aligned[["d_mean", "d_spread", "mean_gen_g"]].isna().any().any():
        raise RuntimeError("D moments failed action alignment")
    features["d_mean"] = aligned["d_mean"].to_numpy(dtype=float)
    features["d_spread"] = aligned["d_spread"].to_numpy(dtype=float)
    features["mean_gen"] = aligned["mean_gen_g"].to_numpy(dtype=float)
    group = actions["group_id"].to_numpy(dtype=int)
    for group_id in (1, 2, 3):
        features[f"g{group_id}"] = (group == group_id).astype(float)
    state = _surface_state(repo, actions)
    for column in STATE:
        features[column] = state[column].to_numpy(dtype=float)
    expected = 5 + 3 + 2 + 1 + 3 + len(STATE)
    if features.shape != (len(actions), expected):
        raise RuntimeError(f"feature shape mismatch: {features.shape}")
    return features.astype("float64")


def preprocess_fold(
    features: pd.DataFrame,
    train_index: pd.Index,
    test_index: pd.Index,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train = features.loc[train_index].to_numpy(dtype=float)
    test = features.loc[test_index].to_numpy(dtype=float)
    finite_train = np.where(np.isfinite(train), train, np.nan)
    medians = np.nanmedian(finite_train, axis=0)
    if not np.isfinite(medians).all():
        raise RuntimeError("all-nonfinite training feature")
    train = np.where(np.isfinite(train), train, medians[None, :])
    test = np.where(np.isfinite(test), test, medians[None, :])
    means = train.mean(axis=0)
    scales = train.std(axis=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    train = (train - means[None, :]) / scales[None, :]
    test = (test - means[None, :]) / scales[None, :]
    parameters = pd.DataFrame(
        {
            "feature": list(features.columns),
            "median": medians,
            "mean": means,
            "scale": scales,
        }
    )
    return train, test, parameters


def official_training_cost(
    frame: pd.DataFrame,
    action_columns: tuple[str, ...] = ACTIONS,
) -> tuple[np.ndarray, np.ndarray]:
    actual = frame["actual_kwh"].to_numpy(dtype=float)
    group = frame["group_id"].to_numpy(dtype=int)
    action = frame[list(action_columns)].to_numpy(dtype=float)
    cost = np.zeros_like(action)
    active = np.zeros(len(frame), dtype=bool)
    for group_id in (1, 2, 3):
        capacity = float(CAPACITIES_KWH[group_id])
        valid = (group == group_id) & (actual >= 0.1 * capacity)
        if not valid.any():
            raise RuntimeError(f"empty training metric group {group_id}")
        active |= valid
        y = actual[valid]
        error = np.abs(action[valid] - y[:, None]) / capacity
        units = settlement_unit(error)
        count = int(valid.sum())
        actual_sum = float(y.sum())
        cost[valid] = error / (6.0 * count) - y[:, None] * units / (
            24.0 * actual_sum
        )
    cost *= len(frame)
    return cost, active


def spo_plus_value_gradient(
    coefficients: np.ndarray,
    design: np.ndarray,
    cost: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    n_rows, n_features = design.shape
    n_actions = cost.shape[1]
    matrix = coefficients.reshape(n_features, n_actions)
    predicted = design @ matrix
    true_choice = np.argmin(cost, axis=1)
    adversary = np.argmax(cost - 2.0 * predicted, axis=1)
    rows = np.arange(n_rows)
    loss = (
        cost[rows, adversary]
        - 2.0 * predicted[rows, adversary]
        + 2.0 * predicted[rows, true_choice]
        - cost[rows, true_choice]
    )
    regularized = matrix.copy()
    regularized[-1] = 0.0
    value = float(loss.mean() + 0.5 * l2 * np.square(regularized).sum())
    prediction_gradient = np.zeros_like(predicted)
    np.add.at(prediction_gradient, (rows, adversary), -2.0 / n_rows)
    np.add.at(prediction_gradient, (rows, true_choice), 2.0 / n_rows)
    gradient = design.T @ prediction_gradient + l2 * regularized
    return value, gradient.ravel()


def fit_spo_plus(features: np.ndarray, cost: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    design = np.column_stack([features, np.ones(len(features), dtype=float)])
    initial = np.zeros(design.shape[1] * cost.shape[1], dtype=float)
    initial_value, _ = spo_plus_value_gradient(initial, design, cost, L2)

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        return spo_plus_value_gradient(coefficients, design, cost, L2)

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": MAXITER, "ftol": FTOL, "gtol": GTOL},
    )
    final_value, final_gradient = objective(result.x)
    acceptable = bool(
        np.isfinite(result.x).all()
        and np.isfinite(final_value)
        and np.isfinite(final_gradient).all()
        and final_value <= initial_value + 1e-12
    )
    if not acceptable:
        raise RuntimeError("frozen optimizer acceptance gate failed")
    return result.x.reshape(design.shape[1], cost.shape[1]), {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "initial_objective": initial_value,
        "final_objective": final_value,
        "final_gradient_inf_norm": float(np.max(np.abs(final_gradient))),
        "finite_descent_acceptance": acceptable,
    }


def predict_action(coefficients: np.ndarray, features: np.ndarray) -> np.ndarray:
    design = np.column_stack([features, np.ones(len(features), dtype=float)])
    return np.argmin(design @ coefficients, axis=1)


def _save_fit_outputs(
    repo: Path,
    predeclaration: Path,
    output_dir: Path,
) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    _verify_frozen_inputs(repo, frozen)
    actions = pd.read_parquet(
        repo / "artifacts/backtests/s17_n7_strict_actions/actions.parquet"
    )
    actions["forecast_kst_dtm"] = pd.to_datetime(actions["forecast_kst_dtm"])
    checked = validate_issuance_atoms(actions)
    splits = chronological_outer_splits(checked, burn_in_folds=1)
    features = build_features(repo, actions)
    if not actions.index.equals(features.index):
        raise RuntimeError("action/feature index mismatch")

    candidate = actions["CHAMPION"].to_numpy(dtype=float).copy()
    choice = np.full(len(actions), "CHAMPION", dtype=object)
    optimizer: dict[str, Any] = {}
    coefficient_parts: list[pd.DataFrame] = []
    preprocess_parts: list[pd.DataFrame] = []
    for split in splits:
        train_features, test_features, parameters = preprocess_fold(
            features, split.train_index, split.test_index
        )
        train_frame = actions.loc[split.train_index]
        cost, active = official_training_cost(train_frame)
        coefficients, fit_receipt = fit_spo_plus(
            train_features[active], cost[active]
        )
        selected = predict_action(coefficients, test_features)
        test_actions = actions.loc[split.test_index, list(ACTIONS)].to_numpy(dtype=float)
        candidate[split.test_index] = test_actions[
            np.arange(len(split.test_index)), selected
        ]
        choice[split.test_index] = np.asarray(ACTIONS, dtype=object)[selected]
        fit_receipt.update(
            {
                "train_rows": len(split.train_index),
                "train_valid_metric_rows": int(active.sum()),
                "test_rows": len(split.test_index),
                "train_max_label_time": split.train_max_label_time.isoformat(),
                "test_min_basis_time": split.test_min_basis_time.isoformat(),
                "choice_counts": {
                    action: int(np.count_nonzero(selected == offset))
                    for offset, action in enumerate(ACTIONS)
                },
            }
        )
        optimizer[split.test_fold] = fit_receipt
        coefficient_parts.append(
            pd.DataFrame(
                coefficients,
                index=[*list(features.columns), "__intercept__"],
                columns=list(ACTIONS),
            )
            .rename_axis("feature")
            .reset_index()
            .assign(test_fold=split.test_fold)
        )
        preprocess_parts.append(parameters.assign(test_fold=split.test_fold))

    predictions = actions[[*list(KEYS), "actual_kwh", "CHAMPION"]].copy()
    predictions["COST5_SPO_PLUS"] = candidate
    predictions["selected_action"] = choice
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    features_path = output_dir / "features.parquet"
    pd.concat([actions[list(KEYS)], features], axis=1).to_parquet(
        features_path, index=False
    )
    coefficients_path = output_dir / "coefficients.parquet"
    pd.concat(coefficient_parts, ignore_index=True).to_parquet(
        coefficients_path, index=False
    )
    preprocessing_path = output_dir / "preprocessing.parquet"
    pd.concat(preprocess_parts, ignore_index=True).to_parquet(
        preprocessing_path, index=False
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
        provenance_rows.append(champion_row)
        candidate_hash = _vector_hash(
            predictions, "COST5_SPO_PLUS", split.test_fold
        )
        provenance_rows.append(
            {
                "model_id": "COST5_SPO_PLUS",
                "test_fold": split.test_fold,
                "fit_max_time": split.train_max_label_time,
                "selection_max_time": split.train_max_label_time,
                "policy_id": "LINEAR_SPO_PLUS_L2_0p001_ONE_SHOT",
                "predeclaration_sha256": predeclaration_hash,
                "prediction_sha256": candidate_hash,
                "weights_fit": "past_only_expanding",
            }
        )
    provenance = pd.DataFrame(provenance_rows).sort_values(
        ["test_fold", "model_id"], kind="stable"
    )
    provenance_path = output_dir / "procedure_provenance.parquet"
    provenance.to_parquet(provenance_path, index=False)

    output_paths = {
        path.name: _sha256(path)
        for path in (
            predictions_path,
            features_path,
            coefficients_path,
            preprocessing_path,
            provenance_path,
        )
    }
    family = {
        "schema_version": 1,
        "node_id": "S17-N8_COST5_SPO_PLUS_STRICT_PREQUENTIAL_COMPARISON",
        "predeclaration_sha256": predeclaration_hash,
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "code_sha256": _sha256(Path(__file__)),
        "family": ["CHAMPION", "COST5_SPO_PLUS"],
        "incumbent": "CHAMPION",
        "feature_names": list(features.columns),
        "action_order": list(ACTIONS),
        "optimizer": optimizer,
        "outputs": output_paths,
        "prediction_vectors": {
            model: {
                split.test_fold: _vector_hash(predictions, model, split.test_fold)
                for split in splits
            }
            for model in ("CHAMPION", "COST5_SPO_PLUS")
        },
        "fit_phase_forbidden_access": {
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
        "outputs": output_paths,
        "optimizer": optimizer,
        "assessment_metric_calls": 0,
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


def _evaluate(
    repo: Path,
    predeclaration: Path,
    output_dir: Path,
    expected_family_sha256: str,
) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    _verify_frozen_inputs(repo, frozen)
    family_path = output_dir / "family_manifest.json"
    if _sha256(family_path) != expected_family_sha256:
        raise RuntimeError("family manifest freeze hash mismatch")
    family = json.loads(family_path.read_text())
    for name, digest in family["outputs"].items():
        if _sha256(output_dir / name) != digest:
            raise RuntimeError(f"post-freeze output mutation: {name}")
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
    scores = {"CHAMPION": _score_to_json(incumbent), "COST5_SPO_PLUS": _score_to_json(candidate)}
    delta = float(candidate.total - incumbent.total)
    protocol_delta = float(
        protocol["blocks"]["7"]["joint_max_t"]["candidates"]
        ["COST5_SPO_PLUS"]["observed_delta_total"]
    )
    if abs(delta - protocol_delta) >= 1e-12:
        raise RuntimeError("point score and protocol delta disagree")
    promoted = bool(
        delta >= 0.001635
        and protocol["promotion_stable_all_blocks"]["COST5_SPO_PLUS"]
        and protocol["inference"] == "SUPPORTED"
    )
    result = {
        "schema_version": 1,
        "node_id": "S17-N8_COST5_SPO_PLUS_STRICT_PREQUENTIAL_COMPARISON",
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
    parser.add_argument("phase", choices=("fit", "evaluate"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=Path("reports/s17_n8_cost5_spo_plus_predeclaration.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/backtests/s17_n8_cost5_spo_plus"),
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
    if args.phase == "fit":
        result = _save_fit_outputs(repo, predeclaration, output_dir)
    else:
        if len(args.family_sha256) != 64:
            raise RuntimeError("evaluate requires frozen --family-sha256")
        result = _evaluate(
            repo,
            predeclaration,
            output_dir,
            args.family_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
