"""No-refit typed-provenance recovery for the frozen N13 R2 family."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official
from baram.evaluation.prequential import run_prequential_protocol
from baram.loop.events import EventStore

KEYS = ("fold_id", "group_id", "forecast_kst_dtm")
FAMILY = ("CHAMPION", "R2_CONTROL_ZERO4", "R2_VERTICAL_PCA2X2")
RECOVERED_MODELS = set(FAMILY[1:])
OLD_ENUM = "past_only_expanding_with_past_calibration_tail"
NEW_ENUM = "past_only_expanding"


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


def _verify_inputs(repo: Path, predeclaration: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N13A input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N13A input bundle mismatch")
    if frozen["comparison_index"] != 2:
        raise RuntimeError("N13A comparison index mismatch")
    original_family = repo / "artifacts/backtests/s17_n13_r2_vertical_pca/family_manifest.json"
    if _sha256(original_family) != frozen["frozen_partial"]["family_manifest_sha256"]:
        raise RuntimeError("N13 original family mutation")
    return frozen


def materialize(repo: Path, predeclaration: Path, output_dir: Path) -> dict[str, Any]:
    frozen = _verify_inputs(repo, predeclaration)
    original_dir = repo / "artifacts/backtests/s17_n13_r2_vertical_pca"
    original_family = json.loads((original_dir / "family_manifest.json").read_text())
    for name, digest in original_family["output_hashes"].items():
        if _sha256(original_dir / name) != digest:
            raise RuntimeError(f"N13 original output mutation: {name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_names = ("predictions.parquet", "fit_details.json", "pca_loadings.json")
    for name in copied_names:
        shutil.copyfile(original_dir / name, output_dir / name)
        if _sha256(output_dir / name) != _sha256(original_dir / name):
            raise RuntimeError(f"N13A byte-copy mismatch: {name}")

    original_provenance = pd.read_parquet(original_dir / "procedure_provenance.parquet")
    recovered = original_provenance.copy(deep=True)
    selected = recovered["model_id"].isin(RECOVERED_MODELS)
    if int(selected.sum()) != 4:
        raise RuntimeError("N13A expected four candidate/control provenance rows")
    if not recovered.loc[selected, "weights_fit"].eq(OLD_ENUM).all():
        raise RuntimeError("N13A old enum mismatch")
    if not recovered.loc[~selected, "weights_fit"].eq("fixed").all():
        raise RuntimeError("N13A champion enum mismatch")
    recovered.loc[selected, "weights_fit"] = NEW_ENUM
    for column in recovered.columns:
        if column == "weights_fit":
            continue
        if not original_provenance[column].equals(recovered[column]):
            raise RuntimeError(f"N13A modified forbidden provenance column: {column}")
    provenance_path = output_dir / "procedure_provenance.parquet"
    recovered.to_parquet(provenance_path, index=False)
    recovery_path = output_dir / "recovery_verification.json"
    recovery_payload = {
        "schema_version": 1,
        "node_id": "S17-N13A_R2_NO_REFIT_TYPED_PROVENANCE_RECOVERY",
        "original_family_manifest_sha256": _sha256(original_dir / "family_manifest.json"),
        "prediction_byte_copy": True,
        "fit_details_byte_copy": True,
        "pca_loadings_byte_copy": True,
        "changed_column": "weights_fit",
        "changed_rows": int(selected.sum()),
        "old_enum": OLD_ENUM,
        "new_enum": NEW_ENUM,
        "all_other_provenance_cells_equal": True,
        "model_or_optimizer_fits": 0,
        "official_score_calls": 0,
    }
    recovery_path.write_text(
        json.dumps(recovery_payload, ensure_ascii=False, indent=2) + "\n"
    )
    output_hashes = {
        path.name: _sha256(path)
        for path in (
            output_dir / "predictions.parquet",
            provenance_path,
            output_dir / "fit_details.json",
            output_dir / "pca_loadings.json",
            recovery_path,
        )
    }
    family = {
        "schema_version": 1,
        "node_id": "S17-N13A_R2_NO_REFIT_TYPED_PROVENANCE_RECOVERY",
        "predeclaration_sha256": _sha256(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "code_sha256": _sha256(Path(__file__)),
        "family": list(FAMILY),
        "incumbent": "CHAMPION",
        "control": "R2_CONTROL_ZERO4",
        "candidate": "R2_VERTICAL_PCA2X2",
        "comparison_index": 2,
        "output_hashes": output_hashes,
        "prediction_vectors": original_family["prediction_vectors"],
        "recovery": {
            "model_or_optimizer_fits": 0,
            "official_score_calls": 0,
            "inherited_original_model_fits": 12,
            "prediction_vectors_unchanged": True,
            "typed_enum_only": True,
        },
        "forbidden_access": original_family["forbidden_access"],
    }
    family_path = output_dir / "family_manifest.json"
    family_path.write_text(json.dumps(family, ensure_ascii=False, indent=2) + "\n")
    return {
        "family_manifest_sha256": _sha256(family_path),
        "outputs": output_hashes,
        "model_or_optimizer_fits": 0,
        "official_score_calls": 0,
    }


def _metric_frame(frame: pd.DataFrame, prediction: str) -> pd.DataFrame:
    metric = frame[["forecast_kst_dtm", "group_id", "actual_kwh", prediction]].copy()
    metric.insert(0, "forecast_id", np.arange(len(metric), dtype=np.int64))
    return metric.rename(columns={prediction: "prediction_kwh"})


def _score_json(score: Any) -> dict[str, Any]:
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
    frozen = _verify_inputs(repo, predeclaration)
    family_path = output_dir / "family_manifest.json"
    if _sha256(family_path) != expected_family_sha256:
        raise RuntimeError("N13A family freeze mismatch")
    family_manifest = json.loads(family_path.read_text())
    for name, digest in family_manifest["output_hashes"].items():
        if _sha256(output_dir / name) != digest:
            raise RuntimeError(f"N13A post-freeze artifact mutation: {name}")
    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    predictions["forecast_kst_dtm"] = pd.to_datetime(predictions["forecast_kst_dtm"])
    provenance = pd.read_parquet(output_dir / "procedure_provenance.parquet")
    event_store = EventStore(repo, repo / "artifacts/registry/loop_events_s17.sqlite")
    protocol = run_prequential_protocol(
        predictions,
        prediction_columns=list(FAMILY),
        incumbent="CHAMPION",
        capacities=CAPACITIES_KWH,
        procedure_provenance=provenance,
        family_manifest_sha256=expected_family_sha256,
        comparison_index=2,
        event_store=event_store,
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
    delta_incumbent = (
        scores["R2_VERTICAL_PCA2X2"]["total"] - scores["CHAMPION"]["total"]
    )
    delta_control = (
        scores["R2_VERTICAL_PCA2X2"]["total"]
        - scores["R2_CONTROL_ZERO4"]["total"]
    )
    protocol_delta = protocol["blocks"]["7"]["joint_max_t"]["candidates"][
        "R2_VERTICAL_PCA2X2"
    ]["observed_delta_total"]
    if abs(delta_incumbent - protocol_delta) >= 1e-12:
        raise RuntimeError("N13A point/protocol candidate delta disagreement")
    stable = protocol["promotion_stable_all_blocks"]["R2_VERTICAL_PCA2X2"]
    promotion_supported = bool(
        delta_incumbent >= 0.001635
        and delta_control >= 0.00357236259
        and stable
        and protocol["inference"] == "SUPPORTED"
    )
    goal_reached = bool(
        promotion_supported and scores["R2_VERTICAL_PCA2X2"]["total"] >= 0.66
    )
    result = {
        "schema_version": 1,
        "node_id": "S17-N13A_R2_NO_REFIT_TYPED_PROVENANCE_RECOVERY",
        "family_manifest_sha256": expected_family_sha256,
        "comparison_index": 2,
        "scores": scores,
        "candidate_delta_incumbent_total": delta_incumbent,
        "candidate_delta_control_total": delta_control,
        "promotion_supported": promotion_supported,
        "goal_total_0p66_reached": goal_reached,
        "protocol": protocol,
        "score_calls": {
            "outer_point_official": 3,
            "strict_prequential_protocol": 1,
        },
        "model_fits_during_recovery_or_evaluation": 0,
        "inherited_original_model_fits": 12,
        "evidence_label": frozen["evaluation"]["evidence_label"],
        "forbidden_access": family_manifest["forbidden_access"],
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
        default=Path("reports/s17_n13a_r2_provenance_recovery_predeclaration.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/backtests/s17_n13a_r2_provenance_recovery"),
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
        result = evaluate(repo, predeclaration, output_dir, args.family_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
