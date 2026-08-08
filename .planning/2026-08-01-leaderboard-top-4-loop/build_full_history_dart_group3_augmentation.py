"""Build the deployment-only full-history realization of promoted M265.

Groups 1/2 remain exact parsed M263. Group 3 is the frozen Q2-selected M114
blend: 0.60 times M263 (M107 lineage for group 3) plus 0.40 times a 140-tree
M113 LightGBM-DART expected-utility action. No score is computed here.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from build_full_history_strict_temporal_champion import (
    CACHE,
    FEATURE_SOURCE_HASHES,
    FROZEN_HELPER_HASHES,
    REPO,
    SITEWIND_ITERATIONS,
    SUBMISSIONS,
    _add_sitewind_features,
    _all_weather_columns,
    _allweather_sitewind,
    _array_sha256,
    _artifact_kib,
    _build_test_weather_derivatives,
    _legacy_sitewind,
    _surface,
    _test_surface,
    _validate_frozen_evidence,
)
from build_full_history_strict_temporal_champion import (
    FROZEN_HASHES as M261_FROZEN_HASHES,
)
from lightgbm import LGBMClassifier
from run_sequence_classifier import BASELINE_SHA, CAPACITIES, OPEN_SHA

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.submission.build import build_submission
from baram.submission.validate import validate_submission

CANDIDATE_ID = "M266_FULL_HISTORY_DART_GROUP3_AUGMENTATION"
CANDIDATE_PATH = SUBMISSIONS / "submission_M266.csv"
RECEIPT_PATH = SUBMISSIONS / "submission_M266.receipt.json"
EVIDENCE = REPO / "artifacts" / "backtests" / "metric-aligned-probe"
M113_RECEIPT = EVIDENCE / "M113_LGBM_DART-dev-2023-Q2.json"
M114_RECEIPT = EVIDENCE / "M114_STRICT_DART_BLEND-oof.json"
M265_RECEIPT = EVIDENCE / "M265_DART_GROUP3_AUGMENTATION.json"
M265_PREDICTION = EVIDENCE / "M265_DART_GROUP3_AUGMENTATION-oof.parquet"
M263_SUBMISSION = SUBMISSIONS / "submission_M263.csv"
M263_RECEIPT = SUBMISSIONS / "submission_M263.receipt.json"
M261_BUILDER = Path(__file__).parent / "build_full_history_strict_temporal_champion.py"
FROZEN_HASHES = {
    "m113_receipt": "cd449c73b0761032aa2a84a2a1dbbc2fe70537130d6ce7425cd86cbf38cdea63",
    "m114_receipt": "ac12b47fde57bc6f4b553c17f05e7f65fb91524c2665d9378df963513d729d93",
    "m265_receipt": "fd936854d340ba41462cfd1562cc7c0fd2da54af5795326bc01b673e81ffbd75",
    "m265_prediction": "123c0a4f8a4d42fa2e2bb164016e4d6f13c7bc11948700dbd2166c579c616b51",
    "m263_submission": "5a1d701660a83291105a7172c03f6e1ae71250b67e7ac0c2378d7ca01d335819",
    "m263_receipt": "d8b6916a3b2239c773fac7f9d4e4d2f201d7707cab09e07ee42d9eb4a77706b1",
    "m261_builder": "f6b95481a3551256dec9458a3bfb270ca9056efbbc45d29d6d2929f5a9737959",
}
CLASS_WIDTH = 0.02
DART_ITERATIONS = 140
ACTION_TEMPERATURE = 0.6
ACTION_GAMMA = 0.5
PARENT_WEIGHT = 0.6
DART_WEIGHT = 0.4


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _direct_score_calls() -> list[str]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden = {"evaluate_official", "_score"}
    result: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        if name in forbidden:
            result.append(f"{name}:{node.lineno}")
    return result


def _validate_promoted_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    m102, _, _ = _validate_frozen_evidence()
    paths = {
        "m113_receipt": M113_RECEIPT,
        "m114_receipt": M114_RECEIPT,
        "m265_receipt": M265_RECEIPT,
        "m265_prediction": M265_PREDICTION,
        "m263_submission": M263_SUBMISSION,
        "m263_receipt": M263_RECEIPT,
        "m261_builder": M261_BUILDER,
    }
    for name, path in paths.items():
        observed = sha256_file(path)
        if observed != FROZEN_HASHES[name]:
            raise RuntimeError(f"M266 frozen {name} hash changed: {observed}")
    if _direct_score_calls():
        raise RuntimeError(f"M266 contains direct score calls: {_direct_score_calls()}")

    m113 = _json(M113_RECEIPT)
    m114 = _json(M114_RECEIPT)
    m265 = _json(M265_RECEIPT)
    m263 = _json(M263_RECEIPT)
    selected_features = m113.get("selected_feature_names")
    if (
        m113.get("candidate_id") != "M113_LGBM_DART"
        or m113.get("fold_id") != "dev-2023-Q2"
        or m113.get("architecture") != "lgbm_dart_multiclass_fixed_m102_features"
        or m113.get("selected_iteration") != DART_ITERATIONS
        or m113.get("feature_count") != 100
        or m113.get("sitewind_feature_count") != 14
        or selected_features != m102.get("selected_feature_names")
        or m113.get("new_2024_evaluation")
        or m113.get("lockbox_reopened")
        or m113.get("external_actions")
    ):
        raise RuntimeError("M266 M113 frozen architecture changed")
    selection = m114.get("selections", {}).get("3", {})
    if (
        m114.get("candidate_id") != "M114_STRICT_DART_BLEND"
        or m114.get("selection_fold") != "dev-2023-Q2"
        or m114.get("selected_iteration") != DART_ITERATIONS
        or selection.get("policy") != "T0.6_G0.5"
        or abs(float(selection.get("parent_weight", -1.0)) - PARENT_WEIGHT) > 1e-12
        or m114.get("new_2024_evaluation")
        or m114.get("lockbox_reopened")
        or m114.get("external_actions")
    ):
        raise RuntimeError("M266 M114 frozen blend changed")
    if (
        m265.get("candidate_id") != "M265_DART_GROUP3_AUGMENTATION"
        or m265.get("state") != "LOCAL_DART_GROUP3_AUGMENTATION_PROMOTED_NO_TEST_BUILD"
        or not m265.get("promotion", {}).get("promoted")
        or m265.get("prediction_sha256") != FROZEN_HASHES["m265_prediction"]
        or m265.get("new_2024_evaluation")
        or m265.get("lockbox_reopened")
        or m265.get("external_actions")
    ):
        raise RuntimeError("M266 M265 promotion boundary changed")
    if (
        m263.get("candidate_id") != "M263_FULL_HISTORY_INDEPENDENT_LINEAGE_ENSEMBLE"
        or m263.get("state")
        != "LOCAL_FROZEN_M263_FULL_HISTORY_ENSEMBLE_BUILT_NOT_UPLOADED"
        or m263.get("online_score") is not None
        or m263.get("new_2024_evaluation")
        or m263.get("lockbox_reopened")
        or m263.get("external_actions")
    ):
        raise RuntimeError("M266 M263 deployment boundary changed")
    return m102, m113


def _fixed_dart_actions(
    probability: np.ndarray,
    centers: np.ndarray,
    groups: np.ndarray,
    mean_generation: dict[int, float],
) -> np.ndarray:
    actions = np.arange(0.075, 1.076, 0.0025)
    error = np.abs(actions[:, None] - centers[None, :])
    units = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
    calibrated = probability ** (1.0 / ACTION_TEMPERATURE)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    chosen = np.empty(len(probability), dtype=float)
    for group_id in CAPACITIES:
        mask = groups == group_id
        group_probability = calibrated[mask]
        utility = -(group_probability @ error.T) + ACTION_GAMMA * (
            group_probability @ (centers[None, :] * units).T
        ) / (4.0 * mean_generation[group_id])
        chosen[mask] = actions[np.argmax(utility, axis=1)]
    return chosen


def _fit_dart(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected_features: list[str],
    train_legacy: np.ndarray,
    test_legacy: np.ndarray,
    train_allweather: np.ndarray,
    test_allweather: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_features = [name for name in selected_features if not name.startswith("sitewind__")]
    missing_train = sorted(set(base_features).difference(train.columns))
    missing_test = sorted(set(base_features).difference(test.columns))
    if missing_train or missing_test:
        raise RuntimeError(
            f"M266 selected base features missing: train={missing_train}, test={missing_test}"
        )
    train_matrix = train[base_features].astype("float32")
    test_matrix = test[base_features].astype("float32")
    train_sitewind = _add_sitewind_features(train_matrix, train_legacy, train_allweather)
    test_sitewind = _add_sitewind_features(test_matrix, test_legacy, test_allweather)
    if train_sitewind != test_sitewind or len(train_sitewind) != 14:
        raise RuntimeError("M266 train/test site-wind feature contract changed")
    missing = sorted(set(selected_features).difference(train_matrix.columns))
    if missing:
        raise RuntimeError(f"M266 frozen selected features missing: {missing}")
    train_matrix = train_matrix[selected_features]
    test_matrix = test_matrix[selected_features]
    train_nonfinite_count = int(
        (~np.isfinite(train_matrix.to_numpy(dtype="float32"))).sum()
    )
    test_nonfinite_count = int(
        (~np.isfinite(test_matrix.to_numpy(dtype="float32"))).sum()
    )

    normalized = train["actual_kwh"] / train["group_id"].map(CAPACITIES)
    training = train["actual_kwh"].notna().to_numpy() & normalized.ge(0.10).to_numpy()
    raw_bins = np.floor((normalized.clip(0.10, 1.074999) - 0.10) / CLASS_WIDTH).astype(
        "Int64"
    )
    active_bins = [
        int(value)
        for value in sorted(raw_bins.loc[training].dropna().astype(int).unique())
    ]
    if active_bins != list(range(46)):
        raise RuntimeError(f"M266 full-history active bins changed: {active_bins}")
    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    classes = raw_bins.map(bin_to_class).astype("Int64")
    centers = np.asarray(
        [
            normalized.loc[training & classes.eq(class_id)].mean()
            for class_id in range(len(active_bins))
        ],
        dtype=float,
    )
    if not np.isfinite(centers).all() or (centers < 0.10).any():
        raise RuntimeError("M266 class centers changed eligibility")
    params = {
        "objective": "multiclass",
        "num_class": len(active_bins),
        "n_estimators": DART_ITERATIONS,
        "learning_rate": 0.025,
        "num_leaves": 15,
        "min_child_samples": 80,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "boosting_type": "dart",
        "drop_rate": 0.05,
        "skip_drop": 0.5,
        "random_state": 20260802,
        "n_jobs": 6,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    classifier = LGBMClassifier(**params)
    sample_weight = normalized.loc[training].clip(lower=0.10).to_numpy(dtype=float)
    classifier.fit(
        train_matrix.loc[training],
        classes.loc[training].astype(int),
        sample_weight=sample_weight,
    )
    if classifier.classes_.tolist() != list(range(46)):
        raise RuntimeError("M266 fitted classifier class order changed")
    probability = classifier.predict_proba(test_matrix, num_iteration=DART_ITERATIONS)
    mean_generation = {
        group_id: float(normalized.loc[training & train["group_id"].eq(group_id)].mean())
        for group_id in CAPACITIES
    }
    chosen = _fixed_dart_actions(
        probability,
        centers,
        test["group_id"].to_numpy(dtype=int),
        mean_generation,
    )
    output = test[["forecast_id", "forecast_kst_dtm", "group_id"]].copy()
    output["prediction_normalized"] = chosen
    output["prediction_kwh"] = chosen * output["group_id"].map(CAPACITIES).to_numpy(
        dtype=float
    )
    diagnostics = {
        "training_rows": int(training.sum()),
        "training_rows_by_group": {
            str(group): int((training & train["group_id"].eq(group).to_numpy()).sum())
            for group in CAPACITIES
        },
        "final_fit_operating_years": sorted(
            int(value)
            for value in (
                train.loc[training, "forecast_kst_dtm"] - pd.Timedelta(hours=1)
            ).dt.year.unique()
        ),
        "active_bins": active_bins,
        "class_centers_sha256": _array_sha256(centers),
        "mean_generation": {str(group): value for group, value in mean_generation.items()},
        "classifier_probability_sha256": _array_sha256(probability),
        "dart_action_sha256": _array_sha256(chosen),
        "selected_feature_count": len(selected_features),
        "selected_feature_sha256": canonical_sha256(selected_features),
        "train_matrix_nonfinite_count": train_nonfinite_count,
        "test_matrix_nonfinite_count": test_nonfinite_count,
    }
    return output, diagnostics


def _parent_submission() -> pd.DataFrame:
    parent = pd.read_csv(M263_SUBMISSION, encoding="utf-8-sig")
    expected = [
        "forecast_id",
        "forecast_kst_dtm",
        "kpx_group_1",
        "kpx_group_2",
        "kpx_group_3",
    ]
    if parent.columns.tolist() != expected or len(parent) != 8760:
        raise RuntimeError("M266 M263 submission schema changed")
    parent["forecast_kst_dtm"] = pd.to_datetime(parent["forecast_kst_dtm"], errors="raise")
    if parent[["forecast_id", "forecast_kst_dtm"]].duplicated().any():
        raise RuntimeError("M266 M263 submission keys are not unique")
    return parent


def _assemble_candidate(parent: pd.DataFrame, dart: pd.DataFrame) -> pd.DataFrame:
    group3 = dart.loc[
        dart["group_id"].eq(3),
        ["forecast_id", "forecast_kst_dtm", "prediction_kwh"],
    ].rename(columns={"prediction_kwh": "dart_group3_kwh"})
    if len(group3) != 8760 or group3[["forecast_id", "forecast_kst_dtm"]].duplicated().any():
        raise RuntimeError("M266 DART group-3 key contract changed")
    candidate = parent.merge(
        group3,
        on=["forecast_id", "forecast_kst_dtm"],
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    if not candidate["_merge"].eq("both").all() or candidate["dart_group3_kwh"].isna().any():
        raise RuntimeError("M266 DART group-3 key alignment changed")
    candidate["kpx_group_3"] = (
        PARENT_WEIGHT * candidate["kpx_group_3"].to_numpy(dtype=float)
        + DART_WEIGHT * candidate["dart_group3_kwh"].to_numpy(dtype=float)
    )
    return candidate[
        [
            "forecast_id",
            "forecast_kst_dtm",
            "kpx_group_1",
            "kpx_group_2",
            "kpx_group_3",
        ]
    ]


def _self_test() -> None:
    m102, m113 = _validate_promoted_evidence()
    probability = np.asarray([[0.25, 0.75], [0.75, 0.25]], dtype=float)
    centers = np.asarray([0.25, 0.75], dtype=float)
    groups = np.asarray([1, 3], dtype=int)
    means = {1: 0.5, 2: 0.5, 3: 0.5}
    actions = _fixed_dart_actions(probability, centers, groups, means)
    if actions.shape != (2,) or not np.isfinite(actions).all():
        raise RuntimeError("M266 deterministic action self-test failed")
    print(
        json.dumps(
            {
                "state": "M266_SELF_TEST_PASS",
                "m102_feature_count": len(m102["selected_feature_names"]),
                "m113_feature_count": len(m113["selected_feature_names"]),
                "direct_score_calls": _direct_score_calls(),
                "synthetic_action_sha256": _array_sha256(actions),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return

    m102, m113 = _validate_promoted_evidence()
    train, base_columns, auxiliary_columns = _surface()
    grid, geometric, weather_repair = _build_test_weather_derivatives()
    test = _test_surface(grid, geometric)
    del grid, geometric
    gc.collect()
    allweather_columns = _all_weather_columns(train)
    for profile, columns in (
        ("legacy", auxiliary_columns),
        ("allweather", allweather_columns),
        ("windgeom", base_columns),
    ):
        missing = sorted(set(columns).difference(test.columns))
        if missing:
            raise RuntimeError(f"M266 test {profile} feature parity failed: {missing}")
    print(
        json.dumps(
            {
                "stage": "feature_contract_pass",
                "train_rows": len(train),
                "test_rows": len(test),
                "selected_features": len(m113["selected_feature_names"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    train_legacy, test_legacy = _legacy_sitewind(train, test, auxiliary_columns)
    print(
        json.dumps(
            {
                "stage": "legacy_sitewind_complete",
                "train_sha256": _array_sha256(train_legacy),
                "test_sha256": _array_sha256(test_legacy),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    train_allweather, test_allweather = _allweather_sitewind(train, test, allweather_columns)
    print(
        json.dumps(
            {
                "stage": "allweather_sitewind_complete",
                "train_sha256": _array_sha256(train_allweather),
                "test_sha256": _array_sha256(test_allweather),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    selected_features = [str(value) for value in m113["selected_feature_names"]]
    dart, fit_diagnostics = _fit_dart(
        train,
        test,
        selected_features,
        train_legacy,
        test_legacy,
        train_allweather,
        test_allweather,
    )
    print(
        json.dumps(
            {
                "stage": "dart_fit_complete",
                "probability_sha256": fit_diagnostics["classifier_probability_sha256"],
                "action_sha256": fit_diagnostics["dart_action_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    parent = _parent_submission()
    wide = _assemble_candidate(parent, dart)
    sample = pd.read_parquet(CACHE / "submission_keys.parquet")
    policy = {
        "architecture": "full_history_m265_dart_group3_augmentation",
        "parent_submission": "submission_M263.csv",
        "groups_1_2": "exact_parsed_M263",
        "group_3": {
            "parent_weight": PARENT_WEIGHT,
            "dart_weight": DART_WEIGHT,
            "dart_architecture": "lgbm_dart_multiclass_fixed_m102_features",
            "dart_iterations": DART_ITERATIONS,
            "class_width": CLASS_WIDTH,
            "selected_feature_names": selected_features,
            "action_policy": "T0.6_G0.5",
            "temporal_transform": None,
        },
        "classifier_parameters": {
            "learning_rate": 0.025,
            "num_leaves": 15,
            "min_child_samples": 80,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "boosting_type": "dart",
            "drop_rate": 0.05,
            "skip_drop": 0.5,
            "random_state": 20260802,
            "n_jobs": 6,
            "deterministic": True,
            "force_col_wise": True,
        },
        "generation_weight_power": 1.0,
        "sitewind_iterations": {str(group): value for group, value in SITEWIND_ITERATIONS.items()},
        "sitewind_crossfit": "three_fold_kfold_original_seeds",
        "sitewind_test_fit": "full_supplied_history",
        "final_fit_label_years": [2022, 2023, 2024],
        "observed_scada_inference_features": [],
        "selection_after_final_fit": False,
    }
    policy_sha = canonical_sha256(policy)
    csv_sha = build_submission(sample, wide, CANDIDATE_PATH)
    validation = validate_submission(
        CANDIDATE_PATH,
        sample,
        candidate_id=CANDIDATE_ID,
        source_sha256=OPEN_SHA,
        champion_policy_sha256=policy_sha,
        cap_modes={1: "nonnegative_only", 2: "nonnegative_only", 3: "nonnegative_only"},
    )
    if validation.csv_sha256 != csv_sha:
        raise RuntimeError("M266 build and validation hashes differ")
    parsed = pd.read_csv(CANDIDATE_PATH, encoding="utf-8-sig")
    parsed_parent = pd.read_csv(M263_SUBMISSION, encoding="utf-8-sig")
    for column in ("kpx_group_1", "kpx_group_2"):
        if not np.array_equal(
            parsed[column].to_numpy(dtype=float),
            parsed_parent[column].to_numpy(dtype=float),
        ):
            raise RuntimeError(f"M266 {column} changed from parsed M263")
    expected_group3 = (
        PARENT_WEIGHT * parsed_parent["kpx_group_3"].to_numpy(dtype=float)
        + DART_WEIGHT
        * dart.loc[dart["group_id"].eq(3), "prediction_kwh"].to_numpy(dtype=float)
    )
    group3_roundtrip_error = np.abs(
        parsed["kpx_group_3"].to_numpy(dtype=float) - expected_group3
    )
    if float(group3_roundtrip_error.max()) > 1e-10:
        raise RuntimeError("M266 group-3 CSV round-trip formula changed")

    values = parsed[["kpx_group_1", "kpx_group_2", "kpx_group_3"]].to_numpy(
        dtype=float
    )
    receipt = {
        "schema_version": 1,
        "state": "LOCAL_FROZEN_M266_FULL_HISTORY_DART_GROUP3_BUILT_NOT_UPLOADED",
        "candidate_id": CANDIDATE_ID,
        "candidate_path": str(CANDIDATE_PATH.relative_to(REPO)),
        "policy": policy,
        "policy_sha256": policy_sha,
        "fit_diagnostics": fit_diagnostics,
        "sitewind_diagnostics": {
            "legacy_train_sha256": _array_sha256(train_legacy),
            "legacy_test_sha256": _array_sha256(test_legacy),
            "allweather_train_sha256": _array_sha256(train_allweather),
            "allweather_test_sha256": _array_sha256(test_allweather),
            "train_scada_observed_rows": int(train["scada_ws"].notna().sum()),
            "test_scada_observed_rows": 0,
            "observed_scada_feature_count": 0,
        },
        "test_weather_repair": weather_repair,
        "prediction_diagnostics": {
            "row_count": len(parsed),
            "minimum_kwh": float(values.min()),
            "maximum_kwh": float(values.max()),
            "groups_1_2_exact_parsed_parent": True,
            "group3_max_formula_roundtrip_error_kwh": float(
                group3_roundtrip_error.max()
            ),
            "dart_group3_prediction_sha256": _array_sha256(
                dart.loc[dart["group_id"].eq(3), "prediction_kwh"].to_numpy(dtype=float)
            ),
        },
        "submission_receipt": asdict(validation),
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            **FROZEN_HASHES,
            "m261_frozen_input_sha256": M261_FROZEN_HASHES,
            "m261_helper_source_sha256": FROZEN_HELPER_HASHES,
            "feature_submission_source_sha256": FEATURE_SOURCE_HASHES,
            "builder_code_sha256": sha256_file(Path(__file__)),
        },
        "source_evidence": {
            "m102_feature_names_sha256": canonical_sha256(
                m102["selected_feature_names"]
            ),
            "m113_feature_names_sha256": canonical_sha256(
                m113["selected_feature_names"]
            ),
            "m114_group3_policy": "T0.6_G0.5",
            "m114_group3_parent_weight": PARENT_WEIGHT,
            "m265_promoted": True,
        },
        "evaluation_contract": {
            "target_total_strictly_greater_than": 0.66,
            "direct_score_calls": _direct_score_calls(),
            "score_function_calls": 0,
            "metrics_computed_on_2024": False,
            "2024_slice_or_comparison_created": False,
            "selection_after_final_fit": False,
            "parameter_search_after_final_fit": False,
            "local_score": None,
            "online_score": None,
            "target_status": "UNVERIFIED_REQUIRES_EXTERNAL_DACON_RESULT",
        },
        "artifact_budget": {
            "limit_kib": 6 * 1024 * 1024,
            "candidate_bytes": CANDIDATE_PATH.stat().st_size,
            "status": "PASS_AT_BUILD",
        },
        "online_score": None,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "no_external_upload": True,
        "external_actions": [],
    }
    if _artifact_kib() > receipt["artifact_budget"]["limit_kib"]:
        raise RuntimeError("M266 artifact budget exceeded")
    RECEIPT_PATH.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
