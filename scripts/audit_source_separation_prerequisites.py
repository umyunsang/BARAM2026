"""Audit frozen source-separated artifacts for strict no-refit reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baram.loop.events import EventStore

TAGS = ("D", "DL", "DG")
CROSS_SOURCE_MARKERS = (
    "source_disagreement__",
    "atm__hub_consensus",
    "atm__hub_disagree",
    "atm__dewpoint_depression",
    "atm__rh_deficit",
)


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


def _operating_day(frame: pd.DataFrame) -> pd.Series:
    forecast = pd.to_datetime(frame["forecast_kst_dtm"])
    return (forecast - timedelta(hours=1)).dt.normalize()


def run(repo: Path, predeclaration: Path, output: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N16 input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N16 input bundle mismatch")
    node_dir = repo / "research/nodes"
    key_paths = {tag: node_dir / f"S7-N8_{tag}_keys.parquet" for tag in TAGS}
    prob_paths = {tag: node_dir / f"S7-N8_{tag}_prob.npy" for tag in TAGS}
    key_hashes = {tag: _sha256(path) for tag, path in key_paths.items()}
    byte_identical_keys = len(set(key_hashes.values())) == 1
    keys = pd.read_parquet(key_paths["D"])
    probability_audit: dict[str, Any] = {}
    probability_gate = True
    for tag, path in prob_paths.items():
        values = np.load(path, mmap_mode="r")
        finite = bool(np.isfinite(values).all())
        min_value = float(values.min())
        max_value = float(values.max())
        max_sum_error = float(np.max(np.abs(np.asarray(values.sum(axis=1)) - 1.0)))
        shape_match = bool(values.shape[0] == len(keys) and values.ndim == 2)
        passes = bool(
            finite
            and shape_match
            and min_value >= 0.0
            and max_value <= 1.0
            and max_sum_error <= 1e-6
        )
        probability_gate &= passes
        probability_audit[tag] = {
            "path": str(path.relative_to(repo)),
            "sha256": _sha256(path),
            "shape": list(values.shape),
            "finite": finite,
            "min": min_value,
            "max": max_value,
            "max_probability_sum_abs_error": max_sum_error,
            "passes": passes,
        }
    columns = json.loads((node_dir / "S12-N3_columns.json").read_text())
    ldaps_columns = columns["ldaps_cols"]
    gfs_columns = columns["gfs_cols"]
    ldaps_forbidden = [
        column
        for column in ldaps_columns
        if column.startswith(("gfs__", "gfs_spatial__", "geom__gfs__", "g2__g"))
        or column.startswith(CROSS_SOURCE_MARKERS)
        or column.startswith(("phys__", "phys_v2__"))
    ]
    gfs_forbidden = [
        column
        for column in gfs_columns
        if column.startswith(("ldaps__", "ldaps_spatial__", "geom__ldaps__", "g2__l"))
        or column.startswith(CROSS_SOURCE_MARKERS)
    ]
    source_purity_gate = not ldaps_forbidden and not gfs_forbidden
    script_text = (node_dir / "s12_n3_source_split.py").read_text()
    architecture_markers = {
        "single_build_function_for_both_sources": all(
            marker in script_text for marker in ("build('DL'", "build('DG'", "def build(")
        ),
        "outer_past_only_mask": "tr = np.asarray(idx < a)" in script_text,
        "same_regressor_config": "lgb.LGBMRegressor(**MU)" in script_text,
        "same_classifier_config": "lgb.LGBMClassifier(**DART_CLF)" in script_text,
        "source_column_split_only": "cld, cgf, excl = split_columns(COLS)" in script_text,
    }
    architecture_gate = all(architecture_markers.values())
    days = _operating_day(keys)
    retained_days: dict[str, int] = {}
    first_days: dict[str, str] = {}
    for fold_id, indices in keys.groupby("fold_id").groups.items():
        fold_days = days.loc[indices]
        first_day = fold_days.min()
        first_days[str(fold_id)] = str(first_day.date())
        retained_days[str(fold_id)] = int(fold_days[fold_days > first_day].nunique())
    expected_retained = {
        "dev-2023-Q2": 90,
        "dev-2023-Q3": 91,
        "dev-2023-Q4": 91,
    }
    # Q3 has two incomplete days removed only after exact cross-family intersection.
    raw_shift_gate = retained_days == expected_retained
    n7_receipt = json.loads(
        (repo / "reports/s17_n7_strict_action_reconstruction_receipt.json").read_text()
    )
    strict_inequalities = n7_receipt["chronology"]["strict_inequalities_all_pass"]
    same_outer_masks = architecture_markers["outer_past_only_mask"]
    chronology_gate = bool(raw_shift_gate and strict_inequalities and same_outer_masks)
    store = EventStore(repo, repo / "artifacts/registry/loop_events_s17.sqlite")
    store.verify_chain()
    snapshot = store.snapshot()
    prior_strict_r4 = [
        node
        for node in snapshot["closed"]
        if "R4" in node and "SOURCE_SEPARATED" in node
    ]
    no_prior_strict_comparison_gate = not prior_strict_r4
    legacy_manifest = json.loads(
        (repo / "reports/s17_legacy_cycle_manifest.json").read_text()
    )
    legacy_non_score_bearing = bool(
        legacy_manifest["classification"] == "LEGACY_UNRECONSTRUCTABLE"
        and not legacy_manifest["score_bearing"]
    )
    gates = {
        "byte_identical_key_tables": byte_identical_keys,
        "probability_integrity": probability_gate,
        "matched_architecture": architecture_gate,
        "source_purity": source_purity_gate,
        "reconstructible_outer_chronology": chronology_gate,
        "legacy_non_score_bearing": legacy_non_score_bearing,
        "no_prior_exact_s17_r4_comparison": no_prior_strict_comparison_gate,
    }
    verdict = (
        "READY_NO_REFIT_STRICT_R4_MATERIALIZATION"
        if all(gates.values())
        else "BLOCKED"
    )
    result = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "predeclaration_sha256": _sha256(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "known_outcome_exposure": frozen["known_outcome_exposure"],
        "keys": {
            "hashes": key_hashes,
            "byte_identical": byte_identical_keys,
            "rows": len(keys),
            "columns": list(keys.columns),
        },
        "probabilities": probability_audit,
        "source_columns": {
            "ldaps_count": len(ldaps_columns),
            "gfs_count": len(gfs_columns),
            "excluded_count": len(columns["excluded"]),
            "ldaps_forbidden": ldaps_forbidden,
            "gfs_forbidden": gfs_forbidden,
        },
        "architecture_markers": architecture_markers,
        "chronology": {
            "first_operating_days": first_days,
            "raw_days_after_first_day_drop": retained_days,
            "expected_before_complete_day_intersection": expected_retained,
            "n7_strict_inequalities": strict_inequalities,
            "complete_day_intersection_still_required": True,
        },
        "prior_strict_r4_nodes": prior_strict_r4,
        "gates": gates,
        "verdict": verdict,
        "handoff": frozen["handoff_if_ready"] if verdict.startswith("READY") else [],
        "actions": {
            "model_fits": 0,
            "action_or_policy_calculations": 0,
            "official_score_calls": 0,
            "comparison_index": None,
            "test_or_2024_access": False,
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
        default=Path("reports/s17_n16_r4_source_separation_prerequisite_predeclaration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/s17_n16_r4_source_separation_prerequisite.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    output = args.output
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    if not output.is_absolute():
        output = repo / output
    print(
        json.dumps(
            run(repo, predeclaration, output), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
