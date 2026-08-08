"""Build M252 by final-fitting the frozen M245 policy on 2022-2024.

This is a deployment-only final fit.  It reuses the exact M244 recipes and the
exact M245 scope decision, exposes every official training row to the frozen
daily-profile loader, and predicts the unlabeled 2025 test set.  It deliberately
contains no official-score call, 2024 slice, policy search, or selection step.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from build_q1_group12_rare_event_challenger import (
    Q1_ANALOG_GROUPS,
    Q1_FALLBACK_GROUPS,
    SUPPORTED_QUARTERS,
    _enforce_group3_q1_fallback,
    _verified_m245,
)
from build_rare_event_corrected_analog_challenger import _verified_m244
from build_robust_analog_test_challenger import (
    _test_parent_long,
    _test_surface,
    _wide_submission,
)
from build_spread_shrunk_analog_challenger import _verified_parent
from build_supported_season_analog_challenger import _operating_support
from build_v2_transfer_sequence_challenger import (
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
)
from run_conditional_daily_analog_profile import Recipe, _combine, _feature_sets
from run_rare_event_corrected_analog_transfer import _rare_event_profile
from run_recency_spread_analog_transfer import _composed_profile
from run_spread_shrunk_analog_transfer import (
    _apply_spread_recipe,
    _spread_adjusted_profile,
)

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.submission.build import build_submission
from baram.submission.validate import validate_submission

ROOT = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).resolve().parent
CACHE = ROOT / "artifacts" / "cache" / OPEN_SHA
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
SUBMISSIONS = ROOT / "artifacts" / "submissions"

PARENT = SUBMISSIONS / "E0_V2_SEQUENCE_TRANSFER-a5111cee22bc.csv"
M244_CANDIDATE = SUBMISSIONS / "E0_RARE_EVENT_ANALOG-5f6a9679a463.csv"
M244_CANDIDATE_SHA = (
    "1c9c0a509a71ee9ebe2eb0e15bdc37eadfed42687644b9ef0423b89de20a3ab3"
)
M244_RECEIPT = OUTPUT / "M244_RARE_EVENT_CORRECTED_ANALOG_Q234.json"
M244_RECEIPT_SHA = (
    "8489f2759ce39fe4abb7c5179386cde0e1a39d22a5d07673a782b4163410a371"
)
M245_SCOPE_RECEIPT = OUTPUT / "M245_Q1_GROUP12_RARE_EVENT_AUDIT.json"
M245_SCOPE_RECEIPT_SHA = (
    "568a18c5476eae5d90c13cfcbe254c18d292bdd581523f7a16b7d85ab1c9733d"
)
M245_CANDIDATE = (
    SUBMISSIONS / "E0_Q1_GROUP12_RARE_EVENT_ANALOG-bea349ead945.csv"
)
M245_CANDIDATE_SHA = (
    "daafce4b61bb81265d3b48346290c3a7e3273127940eebf0c51e9c32433f16e7"
)
M245_CANDIDATE_RECEIPT = M245_CANDIDATE.with_suffix(".receipt.json")
M245_CANDIDATE_RECEIPT_SHA = (
    "c5f7927ecef3aa3d5bb6459a64de4b307014a241e38da6b3446fb156622b1e0a"
)
LOCKBOX_RECEIPT = ROOT / "artifacts" / "locks" / "lockbox-2024.consumed.json"
LOCKBOX_RECEIPT_SHA = (
    "866f22dcd88c8bcbb1841b55d989a9af43b6f9e133606b47bb7378b1b97ace1f"
)

TRAIN_FEATURES = CACHE / "train_features.parquet"
TEST_FEATURES = CACHE / "test_features.parquet"
LABELS = CACHE / "labels_long.parquet"
FINAL_FIT_YEARS = (2022, 2023, 2024)
TEST_OPERATING_YEAR = 2025
EXPECTED_TRAIN_ROWS = 78_912
EXPECTED_TEST_ROWS = 26_280
TARGET_TOTAL_STRICTLY_GREATER_THAN = 0.66

FROZEN_HELPER_HASHES = {
    "build_q1_group12_rare_event_challenger.py": (
        "4296af997f5645edfacaafdfa00d7631c9de7036ded23e0437df3713f96e9143"
    ),
    "build_rare_event_corrected_analog_challenger.py": (
        "17cba2145471c9004ec45f22c3429acc7ff7b8be3d9445a8ce25ca24d9f3fd84"
    ),
    "build_robust_analog_test_challenger.py": (
        "0ab19ca15c49cd61b68fc43a3ca83b2a983e179e1176f116d871d4580b121dc1"
    ),
    "build_spread_shrunk_analog_challenger.py": (
        "a527cf26e34ab39c2e778831ccfc2908f20201afa2bce4ee6c01646ff0b85269"
    ),
    "build_supported_season_analog_challenger.py": (
        "9ae084c74b5f862ce13cc7c546d915ca4ac22f7b043e65805dc1c0cd30809439"
    ),
    "run_conditional_daily_analog_profile.py": (
        "fae7328f11a520fa8765ca741116c53c50ebf5fc0fcccd4b6ecb69cee367ea82"
    ),
    "run_rare_event_corrected_analog_transfer.py": (
        "92dd10d3f1124ce3aa2776d7408b1d9d6a92d57dd8b35ffbca547e5ea1717c1b"
    ),
    "run_recency_spread_analog_transfer.py": (
        "df2ec81406d4ed46a63e85def97e6f28aff901bb431d15cbdb94691661418a11"
    ),
    "run_spread_shrunk_analog_transfer.py": (
        "44e723a8fe434cdb19cd4f8d1103616f929466fc2e87b7f3fae83f6a9435aa29"
    ),
}


def _verify_frozen_helpers() -> dict[str, str]:
    observed = {
        name: sha256_file(PLAN / name) for name in FROZEN_HELPER_HASHES
    }
    if observed != FROZEN_HELPER_HASHES:
        changed = {
            name: {"expected": FROZEN_HELPER_HASHES[name], "observed": value}
            for name, value in observed.items()
            if value != FROZEN_HELPER_HASHES[name]
        }
        raise RuntimeError(f"M252 frozen helper changed: {changed}")
    return observed


def _direct_score_calls() -> list[str]:
    """Return direct score-like calls in this builder's AST.

    Frozen imported helpers are hash-pinned and limited to retrieval, blending,
    topology, and submission construction.  This guard separately proves that
    the M252 orchestration itself cannot call a scoring API.
    """

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden = {
        "_group_score",
        "_score",
        "evaluate",
        "official_score",
        "score",
    }
    calls: list[str] = []
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
            calls.append(name)
    return sorted(calls)


def _verified_m245_candidate() -> dict[str, object]:
    if sha256_file(M245_CANDIDATE) != M245_CANDIDATE_SHA:
        raise RuntimeError("M245 candidate hash mismatch")
    if sha256_file(M245_CANDIDATE_RECEIPT) != M245_CANDIDATE_RECEIPT_SHA:
        raise RuntimeError("M245 candidate receipt hash mismatch")
    receipt = json.loads(M245_CANDIDATE_RECEIPT.read_text(encoding="utf-8"))
    if (
        receipt["state"]
        != "LOCAL_Q1_GROUP12_RARE_EVENT_CHALLENGER_BUILT_NOT_UPLOADED"
        or receipt["submission_receipt"]["csv_sha256"] != M245_CANDIDATE_SHA
        or receipt["policy"]["m244_policy_sha256"]
        != "2975e719dc2dfeb4c3ba41639eedad5f10b2ec04a128c2a95494e848ac90f3eb"
        or receipt["policy"]["m245_scope_policy_sha256"]
        != "36f82aaaaf96904ef51e94da6cc67703b2c6241c421c42fd482d531926ec1485"
    ):
        raise RuntimeError("M245 candidate lineage changed")
    if (
        receipt.get("online_score") is not None
        or receipt.get("new_2024_evaluation")
        or receipt.get("lockbox_reopened")
        or not receipt.get("no_external_upload")
    ):
        raise RuntimeError("M245 evidence boundary changed")
    return receipt


def _resolved_feature_sets(recipes: dict[int, Recipe]) -> dict[str, list[str]]:
    train_schema = list(pq.ParquetFile(TRAIN_FEATURES).schema.names)
    test_schema = set(pq.ParquetFile(TEST_FEATURES).schema.names)
    feature_sets = _feature_sets(train_schema)
    if len(feature_sets["core"]) != 25:
        raise RuntimeError("M252 core feature count changed")
    if any(name not in test_schema for name in feature_sets["core"]):
        raise RuntimeError("M252 core feature is absent from test features")
    if any(not recipe.representation.startswith("core_") for recipe in recipes.values()):
        raise RuntimeError("M252 frozen recipe unexpectedly requires a non-core feature set")
    return feature_sets


def _full_history_surface(
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    keys = ["forecast_id", "forecast_kst_dtm", "group_id", "operating_year"]
    features = pd.read_parquet(
        TRAIN_FEATURES,
        columns=[*keys, "data_available_kst_dtm", *feature_names],
    )
    labels = pd.read_parquet(LABELS, columns=[*keys, "actual_kwh"])
    for frame in (features, labels):
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    features["data_available_kst_dtm"] = pd.to_datetime(
        features["data_available_kst_dtm"]
    )
    if features.duplicated(keys[:3]).any() or labels.duplicated(keys[:3]).any():
        raise RuntimeError("M252 full-history inputs contain duplicate forecast keys")
    surface = features.merge(labels, on=keys, validate="one_to_one")
    if len(features) != EXPECTED_TRAIN_ROWS or len(surface) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("M252 full-history row count changed")
    if tuple(sorted(surface["operating_year"].unique())) != FINAL_FIT_YEARS:
        raise RuntimeError("M252 final-fit operating-year contract changed")
    if surface["forecast_kst_dtm"].min() != pd.Timestamp("2022-01-01 01:00:00"):
        raise RuntimeError("M252 training start changed")
    if surface["forecast_kst_dtm"].max() != pd.Timestamp("2025-01-01 00:00:00"):
        raise RuntimeError("M252 operating-year boundary row changed")
    finite_targets = surface["actual_kwh"].dropna().to_numpy(dtype=float)
    if not np.isfinite(finite_targets).all() or np.any(finite_targets < 0.0):
        raise RuntimeError("M252 official labels violate the numeric contract")

    daily = (
        surface.groupby(
            ["group_id", "operating_year", "data_available_kst_dtm"],
            sort=True,
        )
        .agg(rows=("forecast_id", "size"), labeled_rows=("actual_kwh", "count"))
        .reset_index()
    )
    daily["eligible_complete_day"] = daily["rows"].eq(24) & daily[
        "labeled_rows"
    ].eq(24)
    row_support: dict[str, dict[str, dict[str, int]]] = {}
    complete_support: dict[str, dict[str, int]] = {}
    for group_id in (1, 2, 3):
        row_support[str(group_id)] = {}
        complete_support[str(group_id)] = {}
        for year in FINAL_FIT_YEARS:
            rows = surface.loc[
                surface["group_id"].eq(group_id)
                & surface["operating_year"].eq(year),
                "actual_kwh",
            ]
            row_support[str(group_id)][str(year)] = {
                "source_rows": len(rows),
                "labeled_rows": int(rows.notna().sum()),
            }
            complete_support[str(group_id)][str(year)] = int(
                daily.loc[
                    daily["group_id"].eq(group_id)
                    & daily["operating_year"].eq(year),
                    "eligible_complete_day",
                ].sum()
            )
    if any(complete_support[str(group_id)]["2024"] <= 0 for group_id in (1, 2, 3)):
        raise RuntimeError("M252 did not expose 2024 complete days to final fit")
    diagnostics = {
        "source_rows": len(surface),
        "operating_years": list(FINAL_FIT_YEARS),
        "rows_by_group_and_operating_year": row_support,
        "eligible_complete_days_by_group_and_operating_year": complete_support,
        "feature_count": len(feature_names),
        "target_use": "fit_only_no_evaluation",
    }
    return surface, diagnostics


def _test_contract(test: pd.DataFrame) -> dict[str, object]:
    metadata = pd.read_parquet(
        TEST_FEATURES,
        columns=[
            "forecast_id",
            "forecast_kst_dtm",
            "data_available_kst_dtm",
            "group_id",
            "operating_year",
        ],
    )
    metadata["forecast_kst_dtm"] = pd.to_datetime(metadata["forecast_kst_dtm"])
    metadata["data_available_kst_dtm"] = pd.to_datetime(
        metadata["data_available_kst_dtm"]
    )
    if len(metadata) != EXPECTED_TEST_ROWS or len(test) != EXPECTED_TEST_ROWS:
        raise RuntimeError("M252 test row count changed")
    if set(metadata["operating_year"].unique()) != {TEST_OPERATING_YEAR}:
        raise RuntimeError("M252 test operating-year contract changed")
    if not test["actual_kwh"].isna().all():
        raise RuntimeError("M252 test surface unexpectedly contains a target")
    if metadata.duplicated(["forecast_id", "forecast_kst_dtm", "group_id"]).any():
        raise RuntimeError("M252 test metadata contains duplicate keys")
    return {
        "source_rows": len(metadata),
        "operating_year": TEST_OPERATING_YEAR,
        "actual_column_materialized_as_nan": True,
        "target_values_present": False,
        "first_forecast_kst_dtm": str(metadata["forecast_kst_dtm"].min()),
        "last_forecast_kst_dtm": str(metadata["forecast_kst_dtm"].max()),
        "first_data_available_kst_dtm": str(
            metadata["data_available_kst_dtm"].min()
        ),
    }


def _strict_training_day_contract(
    surface: pd.DataFrame,
    group_id: int,
    query_issuances: np.ndarray,
) -> dict[str, object]:
    group = surface.loc[surface["group_id"].eq(group_id)]
    daily = (
        group.groupby(
            ["operating_year", "data_available_kst_dtm"],
            sort=True,
        )
        .agg(
            rows=("forecast_id", "size"),
            labeled_rows=("actual_kwh", "count"),
            day_end=("forecast_kst_dtm", "max"),
        )
        .reset_index()
    )
    complete = daily["rows"].eq(24) & daily["labeled_rows"].eq(24)
    cutoff = pd.Timestamp(np.min(query_issuances))
    strict = complete & daily["day_end"].lt(cutoff)
    by_year = {
        str(year): int(
            strict.loc[daily["operating_year"].eq(year)].sum()
        )
        for year in FINAL_FIT_YEARS
    }
    return {
        "query_cutoff": str(cutoff),
        "eligible_complete_days": int(complete.sum()),
        "strictly_available_training_days": int(strict.sum()),
        "strictly_available_days_by_operating_year": by_year,
        "complete_days_excluded_at_query_boundary": int(
            (complete & ~daily["day_end"].lt(cutoff)).sum()
        ),
        "boundary_rule": "complete target day_end strictly before query issuance",
    }


def _difference_diagnostics(
    candidate_path: Path,
    support: pd.DataFrame,
) -> dict[str, object]:
    parent = pd.read_csv(PARENT, encoding="utf-8-sig")
    prior = pd.read_csv(M245_CANDIDATE, encoding="utf-8-sig")
    candidate = pd.read_csv(candidate_path, encoding="utf-8-sig")
    for frame in (parent, prior, candidate):
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    keys = ["forecast_id", "forecast_kst_dtm"]
    if not parent[keys].equals(prior[keys]) or not parent[keys].equals(
        candidate[keys]
    ):
        raise RuntimeError("M252 keys differ from a frozen parent")
    flags = candidate[keys].merge(
        support[[*keys, "operating_quarter"]],
        on=keys,
        validate="one_to_one",
    )
    diagnostics: dict[str, object] = {
        "reference_candidate_id": "E0_Q1_GROUP12_RARE_EVENT_ANALOG-bea349ead945",
        "reference_candidate_sha256": M245_CANDIDATE_SHA,
        "by_group_and_operating_quarter": {},
    }
    changed_any = False
    for group_id in (1, 2, 3):
        column = f"kpx_group_{group_id}"
        candidate_values = candidate[column].to_numpy(dtype=float)
        prior_values = prior[column].to_numpy(dtype=float)
        delta = candidate_values - prior_values
        group_diagnostics: dict[str, object] = {}
        for quarter in range(1, 5):
            mask = flags["operating_quarter"].eq(quarter).to_numpy()
            quarter_delta = delta[mask]
            changed_rows = int(np.count_nonzero(quarter_delta))
            changed_any = changed_any or changed_rows > 0
            group_diagnostics[str(quarter)] = {
                "rows": int(mask.sum()),
                "changed_rows_vs_m245": changed_rows,
                "mean_absolute_delta_kwh_vs_m245": float(
                    np.mean(np.abs(quarter_delta))
                ),
                "max_absolute_delta_kwh_vs_m245": float(
                    np.max(np.abs(quarter_delta))
                ),
            }
        diagnostics["by_group_and_operating_quarter"][str(group_id)] = (
            group_diagnostics
        )
    q1 = flags["operating_quarter"].eq(1).to_numpy()
    if not np.array_equal(
        candidate.loc[q1, "kpx_group_3"].to_numpy(dtype=float),
        parent.loc[q1, "kpx_group_3"].to_numpy(dtype=float),
    ):
        raise RuntimeError("M252 group-3 Q1 differs from the exact M231 fallback")
    if not changed_any:
        raise RuntimeError("M252 full-history final fit did not change M245")
    diagnostics["group3_q1_exact_m231_parent"] = True
    diagnostics["candidate_differs_from_m245"] = True
    return diagnostics


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if sha256_file(LOCKBOX_RECEIPT) != LOCKBOX_RECEIPT_SHA:
        raise RuntimeError("consumed lockbox receipt hash mismatch before M252")
    if _direct_score_calls():
        raise RuntimeError(f"M252 contains direct score calls: {_direct_score_calls()}")
    helper_hashes = _verify_frozen_helpers()
    if sha256_file(M244_CANDIDATE) != M244_CANDIDATE_SHA:
        raise RuntimeError("M244 test candidate hash mismatch")
    if sha256_file(M244_RECEIPT) != M244_RECEIPT_SHA:
        raise RuntimeError("M244 development receipt hash mismatch")
    if sha256_file(M245_SCOPE_RECEIPT) != M245_SCOPE_RECEIPT_SHA:
        raise RuntimeError("M245 scope receipt hash mismatch")

    parent_receipt = _verified_parent()
    m244 = _verified_m244()
    m245_scope = _verified_m245()
    m245_candidate_receipt = _verified_m245_candidate()
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m244["policy"]["recipes"].items()
    }
    if set(recipes) != {1, 2, 3}:
        raise RuntimeError("M252 frozen recipe groups changed")

    feature_sets = _resolved_feature_sets(recipes)
    surface, training_diagnostics = _full_history_surface(feature_sets["core"])
    test = _test_surface(feature_sets["core"])
    test_diagnostics = _test_contract(test)
    if surface["data_available_kst_dtm"].max() >= test[
        "data_available_kst_dtm"
    ].min():
        raise RuntimeError("M252 training availability overlaps test issuance")
    train_columns = [
        "forecast_id",
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "group_id",
        "actual_kwh",
        *feature_sets["core"],
    ]
    analog_surface = pd.concat(
        [surface[train_columns], test[train_columns]],
        ignore_index=True,
    )

    sample, support, support_diagnostics = _operating_support()
    all_issuances = np.sort(support["issuance"].unique())
    supported_issuances = np.sort(
        support.loc[support["analog_supported"], "issuance"].unique()
    )
    if len(all_issuances) != 365 or len(supported_issuances) != 275:
        raise RuntimeError("M252 test issuance support changed")

    test_parent = _test_parent_long(test)
    replacements: list[pd.DataFrame] = []
    retrieval: dict[str, object] = {}
    strict_fit_by_group: dict[str, object] = {}
    for group_id, recipe in recipes.items():
        query_issuances = (
            all_issuances if group_id in Q1_ANALOG_GROUPS else supported_issuances
        )
        corrected_profile, corrected_diagnostics = _rare_event_profile(
            analog_surface,
            feature_sets,
            group_id,
            query_issuances,
            recipe,
        )
        spread_profile, spread_diagnostics = _spread_adjusted_profile(
            analog_surface,
            feature_sets,
            group_id,
            query_issuances,
            recipe,
        )
        strict_fit = _strict_training_day_contract(
            surface,
            group_id,
            query_issuances,
        )
        expected_training_days = strict_fit["strictly_available_training_days"]
        if (
            corrected_diagnostics["training_days"] != expected_training_days
            or spread_diagnostics["training_days"] != expected_training_days
        ):
            raise RuntimeError(
                f"M252 group-{group_id} final-fit day exposure changed: "
                f"expected={expected_training_days}, "
                f"rare={corrected_diagnostics['training_days']}, "
                f"spread={spread_diagnostics['training_days']}"
            )
        strict_fit_by_group[str(group_id)] = strict_fit
        profile = _composed_profile(corrected_profile, spread_profile)
        replacements.append(
            _apply_spread_recipe(test_parent, profile, group_id, recipe)
        )
        retrieval[str(group_id)] = {
            "query_issuance_days": len(query_issuances),
            "final_fit_training_days": expected_training_days,
            "rare_event": corrected_diagnostics,
            "spread": spread_diagnostics,
        }
    training_diagnostics["strict_fit_by_group"] = strict_fit_by_group
    test_output = _combine(test_parent, replacements)
    test_output = _enforce_group3_q1_fallback(test_parent, test_output, support)
    wide = _wide_submission(test_output)

    policy = {
        "architecture": "frozen_m245_policy_full_official_history_final_fit",
        "base_deployment_policy_sha256": m245_candidate_receipt["policy_sha256"],
        "m244_policy_sha256": m244["policy_sha256"],
        "m245_scope_policy_sha256": m245_scope["policy_sha256"],
        "recipes": {
            str(group_id): asdict(recipe) for group_id, recipe in recipes.items()
        },
        "q1_group12_action": "apply_frozen_M244_on_full_history",
        "q1_group3_action": "retain_M231_parent_exactly",
        "q2_q3_q4_action": "apply_frozen_M244_on_full_history",
        "q1_analog_groups": list(Q1_ANALOG_GROUPS),
        "q1_fallback_groups": list(Q1_FALLBACK_GROUPS),
        "supported_operating_quarters": list(SUPPORTED_QUARTERS),
        "final_fit_operating_years": list(FINAL_FIT_YEARS),
        "final_fit_change": "analog_training_history_only",
        "daily_profile_requirement": "24_finite_labels_per_issuance_group",
        "training_cutoff": "strictly_before_first_requested_test_issuance",
        "selection_after_final_fit": False,
        "parameter_search_after_final_fit": False,
        "metric_after_final_fit": False,
        "helper_source_sha256": helper_hashes,
        "parent_csv_sha256": parent_receipt["submission_receipt"]["csv_sha256"],
    }
    policy_sha = canonical_sha256(policy)
    candidate_id = f"E0_FULL_HISTORY_Q1_GROUP12_ANALOG-{policy_sha[:12]}"
    candidate_path = SUBMISSIONS / f"{candidate_id}.csv"
    csv_sha = build_submission(sample, wide, candidate_path)
    validation = validate_submission(
        candidate_path,
        sample,
        candidate_id=candidate_id,
        source_sha256=OPEN_SHA,
        champion_policy_sha256=policy_sha,
        cap_modes={1: "capacity", 2: "capacity", 3: "capacity"},
    )
    if validation.csv_sha256 != csv_sha:
        raise RuntimeError("M252 build and validation hashes differ")
    difference_diagnostics = _difference_diagnostics(candidate_path, support)
    lockbox_after = sha256_file(LOCKBOX_RECEIPT)
    if lockbox_after != LOCKBOX_RECEIPT_SHA:
        raise RuntimeError("consumed lockbox receipt changed during M252")

    receipt = {
        "schema_version": 1,
        "state": "LOCAL_FROZEN_POLICY_FULL_HISTORY_FINAL_FIT_BUILT_NOT_UPLOADED",
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_path.relative_to(ROOT)),
        "policy": policy,
        "policy_sha256": policy_sha,
        "training_diagnostics": training_diagnostics,
        "test_diagnostics": test_diagnostics,
        "support_diagnostics": support_diagnostics,
        "retrieval_diagnostics": retrieval,
        "difference_diagnostics": difference_diagnostics,
        "submission_receipt": asdict(validation),
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "train_features_sha256": sha256_file(TRAIN_FEATURES),
            "test_features_sha256": sha256_file(TEST_FEATURES),
            "labels_long_sha256": sha256_file(LABELS),
            "m244_candidate_sha256": M244_CANDIDATE_SHA,
            "m244_receipt_sha256": M244_RECEIPT_SHA,
            "m245_scope_receipt_sha256": M245_SCOPE_RECEIPT_SHA,
            "m245_candidate_sha256": M245_CANDIDATE_SHA,
            "m245_candidate_receipt_sha256": M245_CANDIDATE_RECEIPT_SHA,
            "lockbox_receipt_sha256_before": LOCKBOX_RECEIPT_SHA,
            "lockbox_receipt_sha256_after": lockbox_after,
            "builder_code_sha256": sha256_file(Path(__file__)),
        },
        "evaluation_contract": {
            "target_total_strictly_greater_than": (
                TARGET_TOTAL_STRICTLY_GREATER_THAN
            ),
            "direct_score_calls": _direct_score_calls(),
            "score_function_calls": 0,
            "metrics_computed_on_2024": False,
            "2024_slice_or_comparison_created": False,
            "selection_after_final_fit": False,
            "local_score": None,
            "online_score": None,
            "target_status": "UNVERIFIED_REQUIRES_EXTERNAL_DACON_RESULT",
        },
        "online_score": None,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "no_external_upload": True,
        "external_actions": [],
    }
    receipt_path = candidate_path.with_suffix(".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
