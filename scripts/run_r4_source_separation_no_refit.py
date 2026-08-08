"""Materialize and evaluate the frozen no-refit S17 R4 family."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official
from baram.evaluation.prequential import run_prequential_protocol
from baram.loop.events import EventStore

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
OUTER = ("dev-2023-Q3", "dev-2023-Q4")
KEYS = ("fold_id", "group_id", "forecast_kst_dtm")
SOURCE_TAGS = ("D", "DL", "DG")
FAMILY = (
    "CHAMPION",
    "R4_CONCAT_D",
    "R4_LDAPS_ONLY",
    "R4_GFS_ONLY",
    "R4_SEP_EQUAL",
    "R4_SEP_Q2_WEIGHTED",
)
TREATMENTS = ("R4_SEP_EQUAL", "R4_SEP_Q2_WEIGHTED")
TEMPERATURES = (0.4, 0.5, 0.6, 0.75, 0.85, 1.0, 1.2)
GAMMAS = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
WEIGHTS = tuple(index / 20 for index in range(21))
EXPECTED_DAYS = {
    "dev-2023-Q2": 90,
    "dev-2023-Q3": 89,
    "dev-2023-Q4": 91,
}


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


def _operating_day(timestamp: pd.Series) -> pd.Series:
    return (pd.to_datetime(timestamp) - timedelta(hours=1)).dt.normalize()


def _metric_total(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    metric = frame[["forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    metric.insert(0, "forecast_id", np.arange(len(metric), dtype=np.int64))
    metric["prediction_kwh"] = np.asarray(prediction, dtype=float)
    return float(evaluate_official(metric, CAPACITIES_KWH).total)


def _metric_frame(frame: pd.DataFrame, prediction: str) -> pd.DataFrame:
    metric = frame[["forecast_kst_dtm", "group_id", "actual_kwh", prediction]].copy()
    metric.insert(0, "forecast_id", np.arange(len(metric), dtype=np.int64))
    return metric.rename(columns={prediction: "prediction_kwh"})


def _score_json(score: Any) -> dict[str, Any]:
    return {
        "total": float(score.total),
        "one_minus_nmae": float(score.one_minus_nmae),
        "ficr": float(score.ficr),
        "group_nmae": {
            str(key): float(value) for key, value in score.group_nmae.items()
        },
        "group_ficr": {
            str(key): float(value) for key, value in score.group_ficr.items()
        },
    }


def _policy_frames(
    repo: Path, probability: np.ndarray, keys: pd.DataFrame
) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(repo / "research/nodes"))
    from loop_lib import utility_frames

    raw = utility_frames(
        probability,
        keys,
        temps=list(TEMPERATURES),
        gammas=list(GAMMAS),
    )
    return {
        f"T{temperature:g}_G{gamma:g}": action
        for (temperature, gamma), action in raw.items()
    }


def _q2_mask(keys: pd.DataFrame) -> np.ndarray:
    day = _operating_day(keys["forecast_kst_dtm"])
    is_q2 = keys["fold_id"].eq("dev-2023-Q2")
    first = day.loc[is_q2].min()
    return np.asarray(is_q2 & day.gt(first))


def _select_policy(
    q2: pd.DataFrame,
    q2_indices: np.ndarray,
    frames: dict[str, np.ndarray],
) -> tuple[str, int]:
    ranked = [
        (_metric_total(q2, action[q2_indices]), policy)
        for policy, action in frames.items()
    ]
    ranked.sort(key=lambda record: (-record[0], record[1]))
    return ranked[0][1], len(ranked)


def _drop_first_and_incomplete(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate = frame.copy()
    candidate["operating_day"] = _operating_day(candidate["forecast_kst_dtm"])
    first = candidate.groupby("fold_id")["operating_day"].transform("min")
    candidate = candidate.loc[candidate["operating_day"].gt(first)].copy()
    counts = candidate.groupby(["fold_id", "operating_day"]).size()
    complete = set(counts.loc[counts.eq(72)].index)
    keep = [
        (fold, day) in complete
        for fold, day in zip(
            candidate["fold_id"], candidate["operating_day"], strict=True
        )
    ]
    candidate = candidate.loc[keep].copy()
    retained = {
        str(key): int(value)
        for key, value in candidate.groupby("fold_id")["operating_day"]
        .nunique()
        .items()
    }
    if retained != EXPECTED_DAYS:
        raise RuntimeError(f"R4 retained day mismatch: {retained}")
    if not candidate.groupby(["fold_id", "operating_day"]).size().eq(72).all():
        raise RuntimeError("R4 incomplete issuance atom survived")
    excluded = {
        f"{fold}/{day.date().isoformat()}": int(count)
        for (fold, day), count in counts.items()
        if count != 72
    }
    fold_order = {fold: index for index, fold in enumerate(FOLDS)}
    candidate["_fold_order"] = candidate["fold_id"].map(fold_order)
    candidate = candidate.sort_values(
        ["_fold_order", "group_id", "forecast_kst_dtm"], kind="stable"
    ).drop(columns=["_fold_order", "operating_day"])
    return candidate.reset_index(drop=True), {
        "retained_days": retained,
        "retained_rows": len(candidate),
        "post_shift_incomplete_days": excluded,
    }


def _vector_hash(frame: pd.DataFrame, column: str, fold: str) -> str:
    part = frame.loc[frame["fold_id"].eq(fold)].sort_values(list(KEYS), kind="stable")
    material = part[list(KEYS)].astype(str).agg("|".join, axis=1).str.cat(sep="\n").encode()
    values = np.ascontiguousarray(part[column].to_numpy(dtype="<f8")).tobytes()
    return hashlib.sha256(material + b"\n" + values).hexdigest()


def _verify_inputs(repo: Path, predeclaration: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N17 input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N17 input bundle mismatch")
    if frozen["comparison_index"] != 3:
        raise RuntimeError("N17 comparison index mismatch")
    return frozen


def materialize(repo: Path, predeclaration: Path, output_dir: Path) -> dict[str, Any]:
    frozen = _verify_inputs(repo, predeclaration)
    keys = pd.read_parquet(repo / "research/nodes/S7-N8_D_keys.parquet")
    keys["forecast_kst_dtm"] = pd.to_datetime(keys["forecast_kst_dtm"])
    keys["actual_kwh"] = (
        keys["cf"].to_numpy(dtype=float)
        * keys["group_id"].map(CAPACITIES_KWH).to_numpy(dtype=float)
    )
    q2_mask = _q2_mask(keys)
    q2_indices = np.flatnonzero(q2_mask)
    q2 = keys.loc[q2_mask].copy()
    if len(q2) != 90 * 72:
        raise RuntimeError("N17 Q2 calibration atom mismatch")
    actions = keys[[*list(KEYS), "actual_kwh"]].copy()
    selected_policies: dict[str, str] = {}
    inner_calls = 0
    for tag in SOURCE_TAGS:
        tag_keys = pd.read_parquet(repo / f"research/nodes/S7-N8_{tag}_keys.parquet")
        if not tag_keys.equals(pd.read_parquet(repo / "research/nodes/S7-N8_D_keys.parquet")):
            raise RuntimeError(f"N17 {tag} key mismatch")
        probability = np.load(repo / f"research/nodes/S7-N8_{tag}_prob.npy")
        frames = _policy_frames(repo, probability, keys)
        policy, calls = _select_policy(q2, q2_indices, frames)
        selected_policies[tag] = policy
        inner_calls += calls
        actions[tag] = frames[policy]
    actions, atom_audit = _drop_first_and_incomplete(actions)
    n7_path = repo / "artifacts/backtests/s17_n7_strict_actions/actions.parquet"
    n7 = pd.read_parquet(n7_path)
    n7["forecast_kst_dtm"] = pd.to_datetime(n7["forecast_kst_dtm"])
    aligned = actions.merge(
        n7[[*list(KEYS), "actual_kwh", "D", "CHAMPION"]].rename(
            columns={"actual_kwh": "n7_actual", "D": "n7_D"}
        ),
        on=list(KEYS),
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != len(actions) or len(aligned) != len(n7):
        raise RuntimeError("N17 N7 row alignment mismatch")
    actual_error = float(np.max(np.abs(aligned["actual_kwh"] - aligned["n7_actual"])))
    concat_error = float(np.max(np.abs(aligned["D"] - aligned["n7_D"])))
    if actual_error > 1e-9 or concat_error > 1e-12:
        raise RuntimeError("N17 failed N7 control reproduction")
    actions = aligned[[*list(KEYS), "actual_kwh", "D", "DL", "DG", "CHAMPION"]]
    actions = actions.rename(
        columns={
            "D": "R4_CONCAT_D",
            "DL": "R4_LDAPS_ONLY",
            "DG": "R4_GFS_ONLY",
        }
    )
    actions["R4_SEP_EQUAL"] = 0.5 * (
        actions["R4_LDAPS_ONLY"] + actions["R4_GFS_ONLY"]
    )
    q2_final = actions.loc[actions["fold_id"].eq("dev-2023-Q2")].copy()
    weight_ranked = []
    for weight in WEIGHTS:
        prediction = (
            weight * q2_final["R4_LDAPS_ONLY"].to_numpy(dtype=float)
            + (1.0 - weight) * q2_final["R4_GFS_ONLY"].to_numpy(dtype=float)
        )
        weight_ranked.append((_metric_total(q2_final, prediction), weight))
    weight_ranked.sort(key=lambda record: (-record[0], record[1]))
    ldaps_weight = float(weight_ranked[0][1])
    inner_calls += len(weight_ranked)
    if inner_calls != frozen["materialization"]["inner_q2_official_calls_expected"]:
        raise RuntimeError("N17 inner score-call count mismatch")
    actions["R4_SEP_Q2_WEIGHTED"] = (
        ldaps_weight * actions["R4_LDAPS_ONLY"]
        + (1.0 - ldaps_weight) * actions["R4_GFS_ONLY"]
    )
    predictions = actions[[*list(KEYS), "actual_kwh", *list(FAMILY)]].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    n7_receipt = json.loads(
        (repo / "reports/s17_n7_strict_action_reconstruction_receipt.json").read_text()
    )
    fit_max = n7_receipt["chronology"]["fit_label_available_max"]
    selection_max = n7_receipt["chronology"]["selection_label_max"]
    n7_predeclaration = n7_receipt["predeclaration"]["sha256"]
    policy_ids = {
        "CHAMPION": "FIXED_0p30_D_0p70_DEPAVG",
        "R4_CONCAT_D": selected_policies["D"],
        "R4_LDAPS_ONLY": selected_policies["DL"],
        "R4_GFS_ONLY": selected_policies["DG"],
        "R4_SEP_EQUAL": "FIXED_EQUAL_DL_DG_AFTER_Q2_POLICIES",
        "R4_SEP_Q2_WEIGHTED": f"Q2_ONLY_WLDAPS_{ldaps_weight:.2f}",
    }
    provenance_rows: list[dict[str, Any]] = []
    prediction_vectors: dict[str, dict[str, str]] = {}
    for model in FAMILY:
        prediction_vectors[model] = {}
        for fold in OUTER:
            digest = _vector_hash(predictions, model, fold)
            prediction_vectors[model][fold] = digest
            provenance_rows.append(
                {
                    "model_id": model,
                    "test_fold": fold,
                    "fit_max_time": fit_max[fold],
                    "selection_max_time": selection_max,
                    "policy_id": policy_ids[model],
                    "predeclaration_sha256": (
                        n7_predeclaration
                        if model == "CHAMPION"
                        else _sha256(predeclaration)
                    ),
                    "prediction_sha256": digest,
                    "weights_fit": (
                        "fixed" if model == "CHAMPION" else "past_only_expanding"
                    ),
                }
            )
    provenance = pd.DataFrame(provenance_rows).sort_values(
        ["test_fold", "model_id"], kind="stable"
    )
    provenance_path = output_dir / "procedure_provenance.parquet"
    provenance.to_parquet(provenance_path, index=False)
    details = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "selected_q2_policies": selected_policies,
        "q2_ldaps_weight": ldaps_weight,
        "source_weight_grid": list(WEIGHTS),
        "inner_q2_official_calls": inner_calls,
        "q3_q4_assessment_score_calls": 0,
        "model_or_optimizer_fits": 0,
        "inherited_outer_model_fits": 18,
        "atom_audit": atom_audit,
        "n7_control_reproduction": {
            "actual_max_abs_error": actual_error,
            "concat_D_max_abs_error": concat_error,
        },
        "known_outcome_exposure": frozen["known_outcome_exposure"],
    }
    details_path = output_dir / "materialization_details.json"
    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    output_hashes = {
        path.name: _sha256(path)
        for path in (predictions_path, provenance_path, details_path)
    }
    family = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "predeclaration_sha256": _sha256(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "code_sha256": _sha256(Path(__file__)),
        "family": list(FAMILY),
        "incumbent": "CHAMPION",
        "mechanism_control": "R4_CONCAT_D",
        "treatments": list(TREATMENTS),
        "outage_fallbacks": frozen["outage_fallbacks"],
        "comparison_index": 3,
        "output_hashes": output_hashes,
        "prediction_vectors": prediction_vectors,
        "selected_q2_policies": selected_policies,
        "q2_ldaps_weight": ldaps_weight,
        "action_accounting": {
            "model_or_optimizer_fits": 0,
            "inherited_outer_model_fits": 18,
            "inner_q2_official_calls": inner_calls,
            "q3_q4_assessment_score_calls": 0,
        },
        "forbidden_access": {
            "2024_operating_day_features_or_labels": False,
            "test_period": False,
            "rejected_ecmwf": False,
            "dacon_actions": [],
        },
    }
    family_path = output_dir / "family_manifest.json"
    family_path.write_text(json.dumps(family, ensure_ascii=False, indent=2) + "\n")
    return {
        "family_manifest_sha256": _sha256(family_path),
        "output_hashes": output_hashes,
        "selected_q2_policies": selected_policies,
        "q2_ldaps_weight": ldaps_weight,
        "model_or_optimizer_fits": 0,
        "q3_q4_assessment_score_calls": 0,
    }


def evaluate(
    repo: Path,
    predeclaration: Path,
    output_dir: Path,
    expected_family_sha256: str,
) -> dict[str, Any]:
    frozen = _verify_inputs(repo, predeclaration)
    family_path = output_dir / "family_manifest.json"
    if _sha256(family_path) != expected_family_sha256:
        raise RuntimeError("N17 family freeze mismatch")
    family = json.loads(family_path.read_text())
    for name, digest in family["output_hashes"].items():
        if _sha256(output_dir / name) != digest:
            raise RuntimeError(f"N17 post-freeze artifact mutation: {name}")
    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    predictions["forecast_kst_dtm"] = pd.to_datetime(predictions["forecast_kst_dtm"])
    provenance = pd.read_parquet(output_dir / "procedure_provenance.parquet")
    store = EventStore(repo, repo / "artifacts/registry/loop_events_s17.sqlite")
    protocol = run_prequential_protocol(
        predictions,
        prediction_columns=list(FAMILY),
        incumbent="CHAMPION",
        capacities=CAPACITIES_KWH,
        procedure_provenance=provenance,
        family_manifest_sha256=expected_family_sha256,
        comparison_index=3,
        event_store=store,
        n_rep=4999,
        seed=20260808,
        block_lengths=(3, 7, 14),
        margin_total=0.001635,
    )
    outer = predictions.loc[~predictions["fold_id"].eq("dev-2023-Q2")].copy()
    scores = {
        model: _score_json(
            evaluate_official(_metric_frame(outer, model), CAPACITIES_KWH)
        )
        for model in FAMILY
    }
    deltas_vs_champion = {
        model: scores[model]["total"] - scores["CHAMPION"]["total"]
        for model in FAMILY
        if model != "CHAMPION"
    }
    deltas_vs_concat = {
        model: scores[model]["total"] - scores["R4_CONCAT_D"]["total"]
        for model in TREATMENTS
    }
    promotion: dict[str, bool] = {}
    for model in TREATMENTS:
        protocol_delta = protocol["blocks"]["7"]["joint_max_t"]["candidates"][model][
            "observed_delta_total"
        ]
        if abs(protocol_delta - deltas_vs_champion[model]) >= 1e-12:
            raise RuntimeError(f"N17 point/protocol delta disagreement: {model}")
        promotion[model] = bool(
            deltas_vs_champion[model]
            >= frozen["evaluation"]["margin_vs_champion_total"]
            and deltas_vs_concat[model]
            >= frozen["evaluation"]["minimum_treatment_minus_concat_total"]
            and protocol["promotion_stable_all_blocks"][model]
            and protocol["inference"] == "SUPPORTED"
        )
    goal_reached = any(
        promotion[model] and scores[model]["total"] >= 0.66 for model in TREATMENTS
    )
    result = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "family_manifest_sha256": expected_family_sha256,
        "comparison_index": 3,
        "scores": scores,
        "deltas_vs_champion": deltas_vs_champion,
        "treatment_deltas_vs_concat": deltas_vs_concat,
        "promotion_supported": promotion,
        "goal_total_0p66_reached": goal_reached,
        "outage_fallback_scores": {
            "gfs_missing": scores["R4_LDAPS_ONLY"],
            "ldaps_missing": scores["R4_GFS_ONLY"],
        },
        "protocol": protocol,
        "score_calls": {
            "outer_point_official": len(FAMILY),
            "strict_prequential_protocol": 1,
        },
        "model_fits_during_materialization_or_evaluation": 0,
        "inherited_outer_model_fits": 18,
        "known_outcome_exposure": frozen["known_outcome_exposure"],
        "evidence_label": frozen["evaluation"]["evidence_label"],
        "forbidden_access": family["forbidden_access"],
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
        default=Path("reports/s17_n17_r4_no_refit_source_separation_predeclaration.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/backtests/s17_n17_r4_source_separation"),
    )
    parser.add_argument("--family-sha256", default="")
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    output_dir = args.output_dir
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    if args.phase == "materialize":
        result = materialize(repo, predeclaration, output_dir)
    else:
        if len(args.family_sha256) != 64:
            raise RuntimeError("evaluate requires frozen --family-sha256")
        result = evaluate(repo, predeclaration, output_dir, args.family_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
