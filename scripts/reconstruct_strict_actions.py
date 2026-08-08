"""Build the S17-N7 shifted, Q2-frozen action cube without assessment scoring."""

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

FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
OUTER = ("dev-2023-Q3", "dev-2023-Q4")
KEYS = ("fold_id", "group_id", "forecast_kst_dtm")
MODELS = ("D", "M102_TOP100", "M113_LGBM_DART", "M115_XGBOOST")
D_TEMPERATURES = (0.4, 0.5, 0.6, 0.75, 0.85, 1.0, 1.2)
D_GAMMAS = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
EXPECTED_DAYS = {
    "dev-2023-Q2": 90,
    "dev-2023-Q3": 89,
    "dev-2023-Q4": 91,
}
STEMS = {
    "M102_TOP100": {
        "dev-2023-Q2": "M102_TOP100",
        "dev-2023-Q3": "M102_TOP100_I60",
        "dev-2023-Q4": "M102_TOP100",
    },
    "M113_LGBM_DART": {fold: "M113_LGBM_DART" for fold in FOLDS},
    "M115_XGBOOST": {fold: "M115_XGBOOST" for fold in FOLDS},
}
ITERATIONS = {
    "M102_TOP100": 60,
    "M113_LGBM_DART": 140,
    "M115_XGBOOST": 100,
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


def _basis(day: pd.Timestamp) -> pd.Timestamp:
    return day - timedelta(days=1) + timedelta(hours=14)


def _label_available(day: pd.Timestamp) -> pd.Timestamp:
    return day + timedelta(days=1)


def _metric_total(
    frame: pd.DataFrame,
    prediction: np.ndarray | pd.Series,
) -> float:
    metric = frame[["forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    metric.insert(0, "forecast_id", np.arange(len(metric), dtype=np.int64))
    metric["prediction_kwh"] = np.asarray(prediction, dtype=float)
    return float(evaluate_official(metric, CAPACITIES_KWH).total)


def _select_q2_policy(
    q2: pd.DataFrame,
    candidates: dict[str, np.ndarray],
) -> tuple[str, int]:
    ranked = [
        (_metric_total(q2, prediction), policy)
        for policy, prediction in candidates.items()
    ]
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return ranked[0][1], len(ranked)


def _load_d(repo: Path) -> tuple[pd.DataFrame, np.ndarray]:
    keys = pd.read_parquet(repo / "research/nodes/S7-N8_D_keys.parquet")
    keys["forecast_kst_dtm"] = pd.to_datetime(keys["forecast_kst_dtm"])
    probability = np.load(repo / "research/nodes/S7-N8_D_prob.npy")
    if len(keys) != len(probability):
        raise RuntimeError("D key/probability length mismatch")
    return keys, probability


def _d_frames(repo: Path, probability: np.ndarray, keys: pd.DataFrame) -> dict[str, np.ndarray]:
    nodes = repo / "research/nodes"
    sys.path.insert(0, str(nodes))
    from loop_lib import utility_frames

    raw = utility_frames(
        probability,
        keys,
        temps=list(D_TEMPERATURES),
        gammas=list(D_GAMMAS),
    )
    return {f"T{temperature:g}_G{gamma:g}": value for (temperature, gamma), value in raw.items()}


def _load_dependent(repo: Path, model: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    base = repo / "artifacts/backtests/metric-aligned-probe"
    for fold in FOLDS:
        stem = STEMS[model][fold]
        part = pd.read_parquet(base / f"{stem}-{fold}-policies.parquet")
        part["forecast_kst_dtm"] = pd.to_datetime(part["forecast_kst_dtm"])
        part.insert(0, "fold_id", fold)
        parts.append(part)
        metadata = json.loads((base / f"{stem}-{fold}.json").read_text())
        if int(metadata["selected_iteration"]) != ITERATIONS[model]:
            raise RuntimeError(f"{model}/{fold}: fixed iteration mismatch")
    return pd.concat(parts, ignore_index=True)


def _drop_first_and_incomplete(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate = frame.copy()
    candidate["operating_day"] = _operating_day(candidate["forecast_kst_dtm"])
    first = candidate.groupby("fold_id")["operating_day"].transform("min")
    candidate = candidate.loc[candidate["operating_day"].gt(first)].copy()
    counts = candidate.groupby(["fold_id", "operating_day"]).size()
    complete_index = set(counts.loc[counts.eq(72)].index)
    keep = [
        (fold, day) in complete_index
        for fold, day in zip(
            candidate["fold_id"], candidate["operating_day"], strict=True
        )
    ]
    candidate = candidate.loc[keep].copy()
    retained = candidate.groupby("fold_id")["operating_day"].nunique().to_dict()
    retained = {str(key): int(value) for key, value in retained.items()}
    if retained != EXPECTED_DAYS:
        raise RuntimeError(f"retained operating-day mismatch: {retained}")
    final_counts = candidate.groupby(["fold_id", "operating_day"]).size()
    if not final_counts.eq(72).all():
        raise RuntimeError("non-72-cell atom survived")
    excluded = {
        f"{fold}/{day.date().isoformat()}": int(count)
        for (fold, day), count in counts.items()
        if count != 72
    }
    return candidate, {
        "retained_days": retained,
        "retained_rows": len(candidate),
        "post_shift_incomplete_days": excluded,
    }


def _vector_hash(frame: pd.DataFrame, column: str, fold: str) -> str:
    part = frame.loc[frame["fold_id"].eq(fold)].sort_values(list(KEYS), kind="stable")
    material = part[list(KEYS)].astype(str).agg("|".join, axis=1).str.cat(sep="\n").encode()
    values = np.ascontiguousarray(part[column].to_numpy(dtype="<f8")).tobytes()
    return hashlib.sha256(material + b"\n" + values).hexdigest()


def run(repo: Path, predeclaration: Path, output_dir: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed_hashes = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed_hashes != frozen["input_bundle"]["files"]:
        raise RuntimeError("frozen input hash mismatch")
    if _canonical_hash(observed_hashes) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("input bundle digest mismatch")

    d_keys, probability = _load_d(repo)
    d_keys["actual_kwh"] = (
        d_keys["cf"].to_numpy(dtype=float)
        * d_keys["group_id"].map(CAPACITIES_KWH).to_numpy(dtype=float)
    )
    d_frames = _d_frames(repo, probability, d_keys)
    d_q2 = d_keys.loc[d_keys["fold_id"].eq("dev-2023-Q2")].copy()
    d_q2["operating_day"] = _operating_day(d_q2["forecast_kst_dtm"])
    d_q2 = d_q2.loc[d_q2["operating_day"].gt(d_q2["operating_day"].min())]
    d_candidates = {
        policy: action[d_q2.index.to_numpy(dtype=int)]
        for policy, action in d_frames.items()
    }
    d_policy, d_calls = _select_q2_policy(d_q2, d_candidates)
    master = d_keys[[*list(KEYS), "actual_kwh"]].copy()
    master["D"] = d_frames[d_policy]

    selected_policies = {"D": d_policy}
    inner_calls = d_calls
    actual_disagreement: dict[str, float] = {}
    for model in MODELS[1:]:
        dependent = _load_dependent(repo, model)
        policy_columns = sorted(column for column in dependent if column.startswith("T"))
        q2 = dependent.loc[dependent["fold_id"].eq("dev-2023-Q2")].copy()
        q2["operating_day"] = _operating_day(q2["forecast_kst_dtm"])
        q2 = q2.loc[q2["operating_day"].gt(q2["operating_day"].min())]
        policy, calls = _select_q2_policy(
            q2,
            {column: q2[column].to_numpy(dtype=float) for column in policy_columns},
        )
        selected_policies[model] = policy
        inner_calls += calls
        chosen = dependent[[*list(KEYS), "actual_kwh", policy]].rename(
            columns={"actual_kwh": f"actual_{model}", policy: model}
        )
        master = master.merge(chosen, on=list(KEYS), how="inner", validate="one_to_one")
        disagreement = np.max(
            np.abs(master["actual_kwh"] - master[f"actual_{model}"])
        )
        actual_disagreement[model] = float(disagreement)
        if disagreement > 1e-9:
            raise RuntimeError(f"{model}: actual alignment mismatch")
        master = master.drop(columns=f"actual_{model}")
    if inner_calls != 245:
        raise RuntimeError(f"inner evaluation count mismatch: {inner_calls}")

    master["CHAMPION"] = 0.30 * master["D"] + 0.70 * master[
        list(MODELS[1:])
    ].mean(axis=1)
    actions, atom_audit = _drop_first_and_incomplete(master)
    fold_order = {fold: offset for offset, fold in enumerate(FOLDS)}
    actions["_fold_order"] = actions["fold_id"].map(fold_order)
    actions = actions.sort_values(
        ["_fold_order", "group_id", "forecast_kst_dtm"], kind="stable"
    ).drop(columns=["_fold_order", "operating_day"])
    actions = actions.reset_index(drop=True)
    if len(actions) != 270 * 72:
        raise RuntimeError("final action row count mismatch")

    train_fit_max: dict[str, str] = {}
    shifted_basis: dict[str, str] = {}
    harness = repo / "artifacts/cache/harness/6e2a1e3f3bb5782ec35a2454"
    for fold in FOLDS:
        train = pd.read_parquet(
            harness / f"{fold}__train.parquet", columns=["forecast_kst_dtm"]
        )
        fit_max = pd.to_datetime(train["forecast_kst_dtm"]).max()
        first_day = _operating_day(
            actions.loc[actions["fold_id"].eq(fold), "forecast_kst_dtm"]
        ).min()
        basis = _basis(first_day)
        label_time = _label_available(_operating_day(pd.Series([fit_max])).iloc[0])
        if not label_time < basis:
            raise RuntimeError(f"{fold}: shifted fit chronology failed")
        train_fit_max[fold] = label_time.isoformat()
        shifted_basis[fold] = basis.isoformat()

    output_dir.mkdir(parents=True, exist_ok=True)
    action_path = output_dir / "actions.parquet"
    actions.to_parquet(action_path, index=False)
    action_hash = _sha256(action_path)

    selection_max = pd.Timestamp("2023-07-01 00:00:00")
    provenance_rows: list[dict[str, Any]] = []
    vector_hashes: dict[str, dict[str, str]] = {}
    for model in (*MODELS, "CHAMPION"):
        vector_hashes[model] = {}
        for fold in OUTER:
            vector_digest = _vector_hash(actions, model, fold)
            vector_hashes[model][fold] = vector_digest
            provenance_rows.append(
                {
                    "model_id": model,
                    "test_fold": fold,
                    "fit_max_time": train_fit_max[fold],
                    "selection_max_time": selection_max,
                    "policy_id": (
                        selected_policies.get(model, "FIXED_0p30_D_0p70_DEPAVG")
                    ),
                    "predeclaration_sha256": _sha256(predeclaration),
                    "prediction_sha256": vector_digest,
                    "weights_fit": "fixed",
                }
            )
    provenance = pd.DataFrame(provenance_rows).sort_values(
        ["test_fold", "model_id"], kind="stable"
    )
    provenance_path = output_dir / "procedure_provenance.parquet"
    provenance.to_parquet(provenance_path, index=False)
    provenance_hash = _sha256(provenance_path)

    manifest = {
        "schema_version": 1,
        "node_id": "S17-N7_STRICT_ACTION_PROVENANCE_RECONSTRUCTION",
        "predeclaration_sha256": _sha256(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "selected_policies": selected_policies,
        "fixed_iterations": ITERATIONS,
        "m102_q3_source": "M102_TOP100_I60",
        "atom_audit": atom_audit,
        "fit_label_available_max": train_fit_max,
        "shifted_test_min_basis": shifted_basis,
        "actual_alignment_max_abs": actual_disagreement,
        "inner_selection": {
            "fold": "dev-2023-Q2",
            "official_total_calls": inner_calls,
            "assessment_score_calls": 0,
            "tie_break": "lexicographically_smallest_policy_id",
        },
        "action_vector_sha256": vector_hashes,
        "outputs": {
            "actions.parquet": action_hash,
            "procedure_provenance.parquet": provenance_hash,
        },
        "forbidden_access": {
            "q3_q4_score_or_component": False,
            "model_fit": False,
            "lockbox_2024": False,
            "test_period": False,
            "rejected_ecmwf": False,
            "dacon_actions": [],
        },
    }
    manifest_path = output_dir / "family_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {
        "selected_policies": selected_policies,
        "atom_audit": atom_audit,
        "action_sha256": action_hash,
        "provenance_sha256": provenance_hash,
        "family_manifest_sha256": _sha256(manifest_path),
        "inner_q2_calls": inner_calls,
        "assessment_score_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--predeclaration",
        type=Path,
        default=Path("reports/s17_n7_strict_action_reconstruction_predeclaration.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/backtests/s17_n7_strict_actions"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    print(json.dumps(run(repo, predeclaration, output_dir), indent=2))


if __name__ == "__main__":
    main()
