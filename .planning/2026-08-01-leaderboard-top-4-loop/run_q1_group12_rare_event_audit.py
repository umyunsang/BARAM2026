"""Audit exact M244 on a strict 2022-to-2023-Q1 group-1/2 surface."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_v2_transfer_sequence_challenger import (
    EXPECTED_SELECTIONS,
    _apply_long,
    _group_score,
)
from run_conditional_daily_analog_profile import Recipe, _combine, _feature_sets
from run_group12_analog_transfer import (
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
    SEED,
    _model_contract,
)
from run_rare_event_corrected_analog_transfer import _rare_event_profile
from run_recency_spread_analog_transfer import _composed_profile
from run_spread_shrunk_analog_transfer import (
    _apply_spread_recipe,
    _spread_adjusted_profile,
)
from strict_dev_surface import DEV_CUTOFF, development_surface

from baram.contracts.hashing import canonical_sha256, sha256_file
from baram.models.baselines import predict_bundle
from baram.models.lightgbm import fit_lgbm_bundle

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
M244_RECEIPT = OUTPUT / "M244_RARE_EVENT_CORRECTED_ANALOG_Q234.json"
M244_RECEIPT_SHA = (
    "8489f2759ce39fe4abb7c5179386cde0e1a39d22a5d07673a782b4163410a371"
)
MODEL_ID = "M245_Q1_GROUP12_RARE_EVENT_AUDIT"
GROUP12 = (1, 2)
N_JOBS = 6
EXPECTED_VALIDATION_DAYS = 90


def _q1_surface(surface: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    group1 = surface.loc[surface["group_id"].eq(1)].copy()
    daily = group1.groupby("data_available_kst_dtm", sort=True).agg(
        rows=("forecast_id", "size"),
        first_forecast=("forecast_kst_dtm", "min"),
        last_forecast=("forecast_kst_dtm", "max"),
    )
    q1 = daily.loc[
        daily["rows"].eq(24)
        & daily["first_forecast"].dt.year.eq(2023)
        & daily["first_forecast"].dt.quarter.eq(1)
    ]
    if len(q1) != EXPECTED_VALIDATION_DAYS:
        raise RuntimeError(f"M245 Q1 validation-day count changed: {len(q1)}")
    validation_issuances = q1.index.to_numpy()
    cutoff = pd.Timestamp(np.min(validation_issuances))
    batch_end = surface.groupby("issuance_batch", sort=True)[
        "forecast_kst_dtm"
    ].max()
    train_batches = batch_end.loc[batch_end.lt(cutoff)].index
    train = surface.loc[
        surface["issuance_batch"].isin(train_batches)
        & surface["group_id"].isin(GROUP12)
        & surface["actual_kwh"].notna()
    ].reset_index(drop=True)
    valid = surface.loc[
        surface["data_available_kst_dtm"].isin(validation_issuances)
        & surface["group_id"].isin(GROUP12)
        & surface["actual_kwh"].notna()
    ].reset_index(drop=True)
    if len(valid) != EXPECTED_VALIDATION_DAYS * 24 * len(GROUP12):
        raise RuntimeError("M245 Q1 validation-row count changed")
    if (
        train.empty
        or train["forecast_kst_dtm"].max() >= cutoff
        or valid["data_available_kst_dtm"].min() != cutoff
        or valid["forecast_kst_dtm"].min() != pd.Timestamp("2023-01-01 01:00:00")
        or valid["forecast_kst_dtm"].max() != pd.Timestamp("2023-04-01 00:00:00")
    ):
        raise RuntimeError("M245 strict Q1 chronology changed")
    diagnostics = {
        "training_rows": len(train),
        "training_batches": int(train["issuance_batch"].nunique()),
        "training_max_forecast": str(train["forecast_kst_dtm"].max()),
        "validation_rows": len(valid),
        "validation_days": len(q1),
        "validation_first_issuance": str(cutoff),
        "validation_first_forecast": str(valid["forecast_kst_dtm"].min()),
        "validation_last_forecast": str(valid["forecast_kst_dtm"].max()),
    }
    return train, valid, diagnostics


def _fit_parents(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: tuple[str, ...],
    contracts: dict[str, dict[str, object]],
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    shared_features = (*feature_names, "group_id", "capacity_kwh")
    outputs: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, object] = {}
    for family, params in contracts.items():
        target = train["actual_kwh"] / train["capacity_kwh"]
        bundle = fit_lgbm_bundle(
            train[list(shared_features)],
            target,
            train["issuance_batch"],
            shared_features,
            "dev-2023-Q1-group12",
            None,
            1.0,
            params,
            SEED,
            N_JOBS,
        )
        normalized = predict_bundle(
            bundle,
            valid[list(shared_features)].reset_index(drop=True),
            "dev-2023-Q1-group12",
        )
        parent = valid[
            [
                "forecast_id",
                "forecast_kst_dtm",
                "group_id",
                "actual_kwh",
                "data_available_kst_dtm",
            ]
        ].copy()
        capacities = valid["capacity_kwh"].to_numpy(dtype=float)
        parent["prediction_kwh"] = np.clip(normalized * capacities, 0.0, capacities)
        parent = _apply_long(
            parent,
            {group_id: EXPECTED_SELECTIONS[group_id] for group_id in GROUP12},
        )
        outputs[family] = parent.reset_index(drop=True)
        diagnostics[family] = {
            "training_rows": len(train),
            "validation_rows": len(parent),
            "selected_iteration": bundle.estimator.n_estimators_,
            "model_manifest": asdict(bundle.manifest),
        }
    return outputs, diagnostics


def _group12_score(frame: pd.DataFrame) -> dict[str, float]:
    scores = [_group_score(frame, group_id) for group_id in GROUP12]
    one_minus_nmae = float(np.mean([score["one_minus_nmae"] for score in scores]))
    ficr = float(np.mean([score["ficr"] for score in scores]))
    return {
        "total": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _paired_bootstrap(
    parent: pd.DataFrame,
    corrected: pd.DataFrame,
    replicates: int = 2_000,
) -> dict[str, float | int | str]:
    issuances = parent["data_available_kst_dtm"].drop_duplicates().to_numpy()
    positions = {
        issuance: np.flatnonzero(
            parent["data_available_kst_dtm"].eq(issuance).to_numpy()
        )
        for issuance in issuances
    }
    random = np.random.default_rng(20260803)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = random.choice(issuances, size=len(issuances), replace=True)
        rows = np.concatenate([positions[issuance] for issuance in sampled])
        deltas[index] = _group12_score(corrected.iloc[rows])["total"] - _group12_score(
            parent.iloc[rows]
        )["total"]
    return {
        "unit": "issuance_day_group12",
        "seed": 20260803,
        "replicates": replicates,
        "mean": float(np.mean(deltas)),
        "q025": float(np.quantile(deltas, 0.025)),
        "q05": float(np.quantile(deltas, 0.05)),
        "median": float(np.quantile(deltas, 0.50)),
        "q95": float(np.quantile(deltas, 0.95)),
        "q975": float(np.quantile(deltas, 0.975)),
        "positive_fraction": float(np.mean(deltas > 0.0)),
    }


def main() -> None:
    if sha256_file(OPEN) != OPEN_SHA or sha256_file(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    if sha256_file(M244_RECEIPT) != M244_RECEIPT_SHA:
        raise RuntimeError("M244 promoted receipt hash mismatch")
    m244 = json.loads(M244_RECEIPT.read_text(encoding="utf-8"))
    recipes = {
        int(group_id): Recipe(**raw)
        for group_id, raw in m244["policy"]["recipes"].items()
        if int(group_id) in GROUP12
    }

    surface, numeric = development_surface()
    if surface["forecast_kst_dtm"].ge(DEV_CUTOFF).any():
        raise RuntimeError("lockbox row reached M245 Q1 audit")
    feature_sets = _feature_sets(numeric)
    train, valid, surface_diagnostics = _q1_surface(surface)
    feature_names, contracts = _model_contract()
    missing = sorted(set(feature_names) - set(surface))
    if missing:
        raise RuntimeError(f"M245 frozen parent feature missing: {missing[:5]}")
    parents, model_diagnostics = _fit_parents(
        train,
        valid,
        feature_names,
        contracts,
    )
    query_issuances = np.sort(valid["data_available_kst_dtm"].unique())

    scores: dict[str, object] = {}
    corrected_outputs: dict[str, pd.DataFrame] = {}
    retrieval: dict[str, object] = {}
    all_group_deltas: list[float] = []
    combined_deltas: list[float] = []
    for family, parent in parents.items():
        replacements: list[pd.DataFrame] = []
        family_retrieval: dict[str, object] = {}
        group_scores: dict[str, object] = {}
        for group_id, recipe in recipes.items():
            corrected_profile, corrected_diagnostics = _rare_event_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances,
                recipe,
            )
            spread_profile, spread_diagnostics = _spread_adjusted_profile(
                surface,
                feature_sets,
                group_id,
                query_issuances,
                recipe,
            )
            profile = _composed_profile(corrected_profile, spread_profile)
            replacement = _apply_spread_recipe(parent, profile, group_id, recipe)
            replacements.append(replacement)
            before = _group_score(parent, group_id)
            after = _group_score(replacement, group_id)
            delta = after["total"] - before["total"]
            all_group_deltas.append(delta)
            group_scores[str(group_id)] = {
                "parent": before,
                "corrected": after,
                "total_delta": delta,
            }
            family_retrieval[str(group_id)] = {
                "rare_event": corrected_diagnostics,
                "spread": spread_diagnostics,
            }
        corrected = _combine(parent, replacements)
        corrected_outputs[family] = corrected
        parent_score = _group12_score(parent)
        corrected_score = _group12_score(corrected)
        combined_delta = corrected_score["total"] - parent_score["total"]
        combined_deltas.append(combined_delta)
        scores[family] = {
            "parent": parent_score,
            "corrected": corrected_score,
            "total_delta": combined_delta,
            "groups": group_scores,
            "paired_bootstrap": _paired_bootstrap(parent, corrected),
        }
        retrieval[family] = family_retrieval

    scope_pass = all(delta > 0.0 for delta in all_group_deltas) and all(
        delta > 0.0 for delta in combined_deltas
    )
    parent_output = pd.concat(
        [frame.assign(parent_family=family) for family, frame in parents.items()],
        ignore_index=True,
    )
    corrected_output = pd.concat(
        [
            frame.assign(parent_family=family, model_id=MODEL_ID)
            for family, frame in corrected_outputs.items()
        ],
        ignore_index=True,
    )
    parent_path = OUTPUT / f"{MODEL_ID}-parents.parquet"
    prediction_path = OUTPUT / f"{MODEL_ID}-predictions.parquet"
    parent_output.to_parquet(parent_path, index=False)
    corrected_output.to_parquet(prediction_path, index=False)

    policy = {
        "architecture": "strict_2022_to_2023_q1_group12_exact_m244_scope_audit",
        "groups": list(GROUP12),
        "validation_days": EXPECTED_VALIDATION_DAYS,
        "parent_families": sorted(contracts),
        "parent_sequence_policy": {
            str(group_id): list(EXPECTED_SELECTIONS[group_id])
            for group_id in GROUP12
        },
        "m244_policy_sha256": m244["policy_sha256"],
        "scope_gate": (
            "positive M244 Total delta for each group under both parents and "
            "positive combined group12 Total delta under both parents"
        ),
        "group3_policy": "unconditional_M231_fallback",
        "parameter_search": False,
        "parent_search": False,
        "date_search": False,
        "group_exception_search": False,
    }
    receipt = {
        "schema_version": 1,
        "candidate_id": MODEL_ID,
        "state": (
            "LOCAL_Q1_GROUP12_SCOPE_PASS_TEST_EXTENSION_ELIGIBLE"
            if scope_pass
            else "LOCAL_Q1_GROUP12_SCOPE_REJECTED"
        ),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "surface_diagnostics": surface_diagnostics,
        "model_diagnostics": model_diagnostics,
        "scores": scores,
        "retrieval_diagnostics": retrieval,
        "scope_pass": scope_pass,
        "test_extension_eligible": scope_pass,
        "parent_path": str(parent_path.relative_to(ROOT)),
        "parent_sha256": sha256_file(parent_path),
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "inputs": {
            "open_zip_sha256": OPEN_SHA,
            "baseline_ipynb_sha256": BASELINE_SHA,
            "m244_receipt_sha256": sha256_file(M244_RECEIPT),
        },
        "online_score": None,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "no_external_upload": True,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{MODEL_ID}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
