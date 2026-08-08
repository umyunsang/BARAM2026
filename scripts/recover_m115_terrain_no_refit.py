"""Typed-provenance-only recovery and assessment for S17-N22A."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official
from baram.evaluation.prequential import run_prequential_protocol
from baram.loop.events import EventStore

FAMILY = (
    "CHAMPION",
    "M115_REFIT_ZERO",
    "TERRAIN_SX300_H8_M115_REPLACED",
)
OUTER = ("dev-2023-Q3", "dev-2023-Q4")
KEYS = ("fold_id", "group_id", "forecast_kst_dtm")
PROVENANCE_COLUMNS = (
    "model_id",
    "test_fold",
    "fit_max_time",
    "selection_max_time",
    "policy_id",
    "predeclaration_sha256",
    "prediction_sha256",
    "weights_fit",
)
SOURCE_PREDICTION_SHA256 = "ccb1ae4fe0fcf6f1828ee0b7db128d487b733f81f9da80d4e6e67b9827a30e25"
SOURCE_DETAILS_SHA256 = "6c2d73829f47546106747d6ee537192d70b5a5471fe56768a1b781903f17f454"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    material = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(material).hexdigest()


def verify_inputs(repo: Path, predeclaration: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: sha256_path(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N22A frozen input hash mismatch")
    if canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N22A input bundle mismatch")
    spec_path = repo / frozen["recovery_spec"]["path"]
    if sha256_path(spec_path) != frozen["recovery_spec"]["sha256"]:
        raise RuntimeError("N22A recovery specification mutation")
    return frozen


def vector_hash(frame: pd.DataFrame, column: str, fold: str) -> str:
    part = frame.loc[frame["fold_id"].eq(fold)].sort_values(list(KEYS), kind="stable")
    key_bytes = part[list(KEYS)].astype(str).agg("|".join, axis=1).str.cat(sep="\n").encode()
    values = np.ascontiguousarray(part[column].to_numpy(dtype="<f8")).tobytes()
    return hashlib.sha256(key_bytes + b"\n" + values).hexdigest()


def iso_string(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def provenance_table(rows: list[dict[str, str]]) -> pa.Table:
    arrays = [
        pa.array([row[column] for row in rows], type=pa.string())
        for column in PROVENANCE_COLUMNS
    ]
    return pa.Table.from_arrays(arrays, names=list(PROVENANCE_COLUMNS))


def recover(
    repo: Path,
    predeclaration: Path,
    output_dir: Path,
) -> dict[str, Any]:
    frozen = verify_inputs(repo, predeclaration)
    source_dir = repo / "artifacts/backtests/s17_n22_m115_terrain"
    source_prediction = source_dir / "predictions.parquet"
    source_details = source_dir / "materialization_details.json"
    if sha256_path(source_prediction) != SOURCE_PREDICTION_SHA256:
        raise RuntimeError("N22A source prediction mutation")
    if sha256_path(source_details) != SOURCE_DETAILS_SHA256:
        raise RuntimeError("N22A source detail mutation")
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_prediction = output_dir / "predictions.parquet"
    copied_details = output_dir / "materialization_details.json"
    shutil.copyfile(source_prediction, copied_prediction)
    shutil.copyfile(source_details, copied_details)
    if sha256_path(copied_prediction) != SOURCE_PREDICTION_SHA256:
        raise RuntimeError("N22A prediction byte-copy mismatch")
    if sha256_path(copied_details) != SOURCE_DETAILS_SHA256:
        raise RuntimeError("N22A details byte-copy mismatch")
    details = json.loads(copied_details.read_text())
    required_detail_gates = (
        details["all_guards_pass"] is True
        and all(details["guards"].values())
        and details["fit_count"] == 6
        and details["predict_calls"] == 6
        and details["assessment_actual_values_read"] == 0
        and details["control_max_abs_kwh"] == 0.0
        and details["zero_outer_max_abs_kwh"] == 0.0
    )
    if not required_detail_gates:
        raise RuntimeError("N22A inherited materialization guard failed")
    predictions = pd.read_parquet(copied_prediction)
    predictions["forecast_kst_dtm"] = pd.to_datetime(predictions["forecast_kst_dtm"])
    required_columns = {
        *KEYS,
        *FAMILY,
        "M115_XGBOOST",
        "M115_CONTROL",
        "M115_TERRAIN",
    }
    if set(predictions) != required_columns:
        raise RuntimeError("N22A frozen prediction columns changed")
    if len(predictions) != 19_440 or predictions.duplicated(list(KEYS)).any():
        raise RuntimeError("N22A frozen prediction key contract failed")
    if not np.isfinite(predictions[list(FAMILY)].to_numpy(dtype=float)).all():
        raise RuntimeError("N22A frozen family nonfinite")
    control_error = float(
        np.max(
            np.abs(
                predictions["M115_CONTROL"].to_numpy(dtype=float)
                - predictions["M115_XGBOOST"].to_numpy(dtype=float)
            )
        )
    )
    outer = predictions["fold_id"].isin(OUTER)
    zero_error = float(
        np.max(
            np.abs(
                predictions.loc[outer, "M115_REFIT_ZERO"].to_numpy(dtype=float)
                - predictions.loc[outer, "CHAMPION"].to_numpy(dtype=float)
            )
        )
    )
    q2 = predictions["fold_id"].eq("dev-2023-Q2")
    q2_errors = {
        model: float(
            np.max(
                np.abs(
                    predictions.loc[q2, model].to_numpy(dtype=float)
                    - predictions.loc[q2, "CHAMPION"].to_numpy(dtype=float)
                )
            )
        )
        for model in FAMILY[1:]
    }
    if control_error != 0.0 or zero_error != 0.0 or any(q2_errors.values()):
        raise RuntimeError("N22A exact numerical recovery gate failed")
    inherited = pd.read_parquet(
        repo / "artifacts/backtests/s17_n7_strict_actions/procedure_provenance.parquet"
    )
    rows: list[dict[str, str]] = []
    for fold in OUTER:
        champion = inherited.loc[
            inherited["model_id"].eq("CHAMPION") & inherited["test_fold"].eq(fold)
        ]
        if len(champion) != 1:
            raise RuntimeError(f"N22A inherited Champion provenance missing: {fold}")
        row = champion.iloc[0]
        rows.append(
            {
                "model_id": str(row["model_id"]),
                "test_fold": str(row["test_fold"]),
                "fit_max_time": iso_string(row["fit_max_time"]),
                "selection_max_time": iso_string(row["selection_max_time"]),
                "policy_id": str(row["policy_id"]),
                "predeclaration_sha256": str(row["predeclaration_sha256"]),
                "prediction_sha256": vector_hash(predictions, "CHAMPION", fold),
                "weights_fit": str(row["weights_fit"]),
            }
        )
        fit_time = str(details["fits"][fold]["fit_label_available_max"])
        for model, policy in (
            ("M115_REFIT_ZERO", "FIXED_T0.75_G2_M115_REFIT_ZERO"),
            (
                "TERRAIN_SX300_H8_M115_REPLACED",
                "FIXED_T0.75_G2_TERRAIN_MEAN16_REPLACE_M115",
            ),
        ):
            rows.append(
                {
                    "model_id": model,
                    "test_fold": fold,
                    "fit_max_time": iso_string(fit_time),
                    "selection_max_time": "2023-07-01T00:00:00",
                    "policy_id": policy,
                    "predeclaration_sha256": sha256_path(predeclaration),
                    "prediction_sha256": vector_hash(predictions, model, fold),
                    "weights_fit": "past_only_expanding",
                }
            )
    provenance_path = output_dir / "procedure_provenance.parquet"
    pq.write_table(provenance_table(rows), provenance_path)
    recovered = pq.read_table(provenance_path)
    if recovered.schema != pa.schema(
        [pa.field(column, pa.string()) for column in PROVENANCE_COLUMNS]
    ):
        raise RuntimeError("N22A typed provenance schema roundtrip failed")
    if recovered.to_pylist() != rows:
        raise RuntimeError("N22A typed provenance value roundtrip failed")
    recovery = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "predeclaration_sha256": sha256_path(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "source_hashes": {
            "predictions.parquet": SOURCE_PREDICTION_SHA256,
            "materialization_details.json": SOURCE_DETAILS_SHA256,
        },
        "copy_hashes": {
            "predictions.parquet": sha256_path(copied_prediction),
            "materialization_details.json": sha256_path(copied_details),
            "procedure_provenance.parquet": sha256_path(provenance_path),
        },
        "guards": {
            "source_hashes_exact": True,
            "predictions_byte_copy_exact": True,
            "details_byte_copy_exact": True,
            "inherited_materialization_all_pass": True,
            "new_fits_zero": True,
            "new_predict_calls_zero": True,
            "action_max_abs_change_zero": True,
            "rows_19440_unique": True,
            "family_finite": True,
            "control_max_abs_kwh_zero": control_error == 0.0,
            "zero_outer_max_abs_kwh_zero": zero_error == 0.0,
            "q2_family_equal": all(value == 0.0 for value in q2_errors.values()),
            "typed_provenance_roundtrip_exact": True,
            "assessment_actual_values_read_zero": True,
            "score_calls_zero": True,
        },
        "vector_hashes": {
            f"{fold}/{model}": vector_hash(predictions, model, fold)
            for fold in OUTER
            for model in FAMILY
        },
        "new_fits": 0,
        "new_predict_calls": 0,
        "new_policy_calls": 0,
        "new_score_calls": 0,
        "assessment_actual_values_read": 0,
        "comparison_index_consumed": False,
        "forbidden_access": {
            "2024_values": False,
            "test": False,
            "rejected_ecmwf": False,
            "quarantined_n10": False,
            "external_requests": 0,
            "dependency_changes": False,
            "dacon_actions": [],
        },
    }
    if not all(recovery["guards"].values()):
        raise RuntimeError("N22A recovery guard failed")
    recovery_path = output_dir / "typed_recovery.json"
    recovery_path.write_text(json.dumps(recovery, ensure_ascii=False, indent=2) + "\n")
    family = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "parent_node_id": "S17-N22_M115_TERRAIN_SX300_H8_STRICT_PREQUENTIAL_COMPARISON",
        "predeclaration_sha256": sha256_path(predeclaration),
        "recovery_spec_sha256": frozen["recovery_spec"]["sha256"],
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "family": list(FAMILY),
        "comparison_index_if_assessed": 4,
        "inherited_fits": 6,
        "new_fits": 0,
        "new_predict_calls": 0,
        "maximum_action_change": 0.0,
        "assessment_actual_values_read": 0,
        "materialization_guards_pass": True,
        "output_hashes": {
            "predictions.parquet": sha256_path(copied_prediction),
            "materialization_details.json": sha256_path(copied_details),
            "procedure_provenance.parquet": sha256_path(provenance_path),
            "typed_recovery.json": sha256_path(recovery_path),
        },
    }
    family_path = output_dir / "family_manifest.json"
    family_path.write_text(json.dumps(family, ensure_ascii=False, indent=2) + "\n")
    return family


def metric_score(frame: pd.DataFrame, column: str) -> dict[str, float]:
    metric = frame[["forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    metric.insert(0, "forecast_id", np.arange(len(metric), dtype=np.int64))
    metric["prediction_kwh"] = frame[column].to_numpy(dtype=float)
    result = evaluate_official(metric, CAPACITIES_KWH)
    return {
        "total": float(result.total),
        "one_minus_nmae": float(result.one_minus_nmae),
        "ficr": float(result.ficr),
    }


def evaluate(
    repo: Path,
    predeclaration: Path,
    output_dir: Path,
    expected_family_sha256: str,
) -> dict[str, Any]:
    frozen = verify_inputs(repo, predeclaration)
    family_path = output_dir / "family_manifest.json"
    if sha256_path(family_path) != expected_family_sha256:
        raise RuntimeError("N22A family freeze mismatch")
    family = json.loads(family_path.read_text())
    for name, digest in family["output_hashes"].items():
        if sha256_path(output_dir / name) != digest:
            raise RuntimeError(f"N22A frozen output mutation: {name}")
    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    predictions["forecast_kst_dtm"] = pd.to_datetime(predictions["forecast_kst_dtm"])
    actual = pd.read_parquet(
        repo / "artifacts/backtests/s17_n7_strict_actions/actions.parquet",
        columns=["fold_id", "group_id", "forecast_kst_dtm", "actual_kwh"],
    )
    actual["forecast_kst_dtm"] = pd.to_datetime(actual["forecast_kst_dtm"])
    assessment = predictions.merge(
        actual,
        on=list(KEYS),
        how="left",
        validate="one_to_one",
    )
    if len(assessment) != 19_440 or assessment["actual_kwh"].isna().any():
        raise RuntimeError("N22A assessment actual alignment failed")
    provenance = pd.read_parquet(output_dir / "procedure_provenance.parquet")
    protocol_family = ("CHAMPION", "TERRAIN_SX300_H8_M115_REPLACED")
    provenance = provenance.loc[provenance["model_id"].isin(protocol_family)].copy()
    store = EventStore(repo, repo / "artifacts/registry/loop_events_s17.sqlite")
    protocol = run_prequential_protocol(
        assessment,
        prediction_columns=list(protocol_family),
        incumbent="CHAMPION",
        capacities=CAPACITIES_KWH,
        procedure_provenance=provenance,
        family_manifest_sha256=expected_family_sha256,
        comparison_index=4,
        event_store=store,
        n_rep=4999,
        seed=20260808,
        block_lengths=(3, 7, 14),
        margin_total=0.001635,
    )
    outer = assessment.loc[assessment["fold_id"].isin(OUTER)].copy()
    scores = {model: metric_score(outer, model) for model in FAMILY}
    deltas = {
        model: scores[model]["total"] - scores["CHAMPION"]["total"]
        for model in FAMILY
        if model != "CHAMPION"
    }
    candidate = "TERRAIN_SX300_H8_M115_REPLACED"
    protocol_delta = protocol["blocks"]["7"]["joint_max_t"]["candidates"][candidate][
        "observed_delta_total"
    ]
    if abs(protocol_delta - deltas[candidate]) >= 1e-12:
        raise RuntimeError("N22A point/protocol delta disagreement")
    promotion = bool(
        deltas[candidate] >= 0.001635
        and protocol["promotion_stable_all_blocks"][candidate]
        and protocol["inference"] == "SUPPORTED"
    )
    goal_reached = bool(promotion and scores[candidate]["total"] >= 0.66)
    result = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "family_manifest_sha256": expected_family_sha256,
        "comparison_index": 4,
        "scores": scores,
        "deltas_vs_champion": deltas,
        "promotion_supported": promotion,
        "goal_total_0p66_reached": goal_reached,
        "protocol": protocol,
        "mcs_deduplication": {
            "excluded_identical_column": "M115_REFIT_ZERO",
            "identity_proof": "outer action vector is byte/value identical to CHAMPION",
            "reason": "S17-N3 mcs_tr requires identical loss columns to be deduplicated",
            "model_or_policy_choice": False,
        },
        "assessment_attempts": {
            "attempt_1": (
                "Failed deterministically at MCS identical-loss guard after actual "
                "access; no result artifact was written."
            ),
            "attempt_2": (
                "Same immutable family/comparison index with proven-identical zero "
                "column excluded only from MCS/protocol."
            ),
        },
        "score_calls": {
            "outer_official": len(FAMILY),
            "strict_protocol_successful": 1,
            "failed_protocol_attempt_before_mcs": 1,
            "materialization_or_policy_selection": 0,
        },
        "assessment_actual_values_read": len(assessment),
        "inherited_fits": 6,
        "new_fits": 0,
        "new_predict_calls": 0,
        "comparison_consumed": True,
        "verdict": (
            "SUPPORTED_GOAL_REACHED"
            if goal_reached
            else ("SUPPORTED_BELOW_GOAL" if promotion else "REFUTED")
        ),
        "next_handoff": (
            ["S17_TARGET_DELIVERY"]
            if goal_reached
            else ["S17-N23_POST_TERRAIN_FRONTIER_RESEARCH_INTAKE"]
        ),
        "forbidden_access": {
            "2024_values": False,
            "test": False,
            "rejected_ecmwf": False,
            "quarantined_n10": False,
            "external_requests": 0,
            "dependency_changes": False,
            "dacon_actions": [],
        },
    }
    path = output_dir / "evaluation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("recover", "evaluate"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=Path("reports/s17_n22a_m115_terrain_no_refit_predeclaration.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/backtests/s17_n22a_m115_terrain_recovery"),
    )
    parser.add_argument("--family-sha256")
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    output_dir = args.output_dir
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    if args.mode == "recover":
        result = recover(repo, predeclaration, output_dir)
    else:
        if not args.family_sha256:
            raise RuntimeError("N22A evaluate requires --family-sha256")
        result = evaluate(
            repo,
            predeclaration,
            output_dir,
            args.family_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
