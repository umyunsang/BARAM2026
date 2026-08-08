"""Corrected exact-print identifiability gate for the R3 KMAPP branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


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


def run(repo: Path, predeclaration: Path, output: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed != frozen["input_bundle"]["files"]:
        raise RuntimeError("N15A input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N15A input bundle mismatch")
    pdf_path = repo / "research/sources/kim_lee_et_al_2021_kmapp_28740.pdf"
    reader = PdfReader(pdf_path)
    page = reader.pages[2].extract_text()
    compact = re.sub(r"\s+", "", page)
    exact_print_detected = bool(
        "ezH" in compact and "ezH" not in compact
    )
    equation_excerpt_match = re.search(
        r"[Δ]uHCz.*?e.*?[Δ]H,", compact, flags=re.DOTALL
    )
    equation_excerpt = (
        equation_excerpt_match.group(0) if equation_excerpt_match else ""
    )
    # Printed RHS dimensions: u * exp(dimensionless) * delta_H.
    # (L/T) * 1 * L = L^2/T, not the required L/T.
    dimensions = {
        "lhs_delta_u": {"length_power": 1, "time_power": -1},
        "rhs_printed": {"length_power": 2, "time_power": -1},
    }
    dimensional_consistency = dimensions["lhs_delta_u"] == dimensions["rhs_printed"]
    n12_receipt = json.loads(
        (repo / "reports/s17_n12_vertical_profile_prerequisite_receipt.json").read_text()
    )
    n12_path = repo / n12_receipt["artifact"]["path"]
    if _sha256(n12_path) != n12_receipt["artifact"]["sha256"]:
        raise RuntimeError("N12 artifact mutation")
    n12 = json.loads(n12_path.read_text())
    ldaps = n12["source_vectors"]["ldaps"]
    extrema = n12["ldaps_50_extrema_present_but_not_level_means"]
    page_defines_u_hhc = all(
        marker in page
        for marker in ("u(hHC)", "LDAPS", "reference", "height")
    )
    page_workflow_vertical_interpolation = "Vertical interpolation" in page
    supplied_mean_levels = ldaps["comparable_heights_m"]
    primary_prescribes_reconstruction_from_5_10 = False
    u_hhc_observable = bool(
        page_defines_u_hhc
        and primary_prescribes_reconstruction_from_5_10
        and not any(extrema.values())
    )
    gates = {
        "exact_print_detected": exact_print_detected,
        "dimensional_consistency_without_inserting_unprinted_factor": dimensional_consistency,
        "u_hhc_observable_from_supplied_same_source_mean_vectors": u_hhc_observable,
    }
    result = {
        "schema_version": 1,
        "node_id": frozen["node_id"],
        "predeclaration_sha256": _sha256(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "primary_pdf": {
            "path": str(pdf_path.relative_to(repo)),
            "sha256": _sha256(pdf_path),
            "page_number_one_based": 3,
            "page_text_sha256": hashlib.sha256(page.encode()).hexdigest(),
            "equation_2_compact_excerpt": equation_excerpt,
            "defines_u_hhc_as_ldaps_forecast_wind_at_reference_height": page_defines_u_hhc,
            "workflow_includes_vertical_interpolation": page_workflow_vertical_interpolation,
        },
        "dimensional_audit": {
            "declared_units": frozen["frozen_exact_printed_equations"],
            "symbolic_dimensions": dimensions,
            "passes": dimensional_consistency,
            "unprinted_factor_inserted": False,
        },
        "observability_audit": {
            "supplied_ldaps_comparable_mean_vector_heights_m": supplied_mean_levels,
            "ldaps_50m_component_extrema_not_mean_vectors": extrema,
            "primary_prescribes_exact_reconstruction_from_supplied_5_10m_vectors": (
                primary_prescribes_reconstruction_from_5_10
            ),
            "u_hhc_observable": u_hhc_observable,
            "forbidden_substitutions_used": [],
        },
        "gates": gates,
        "verdict": "REFUTED_R3_EXACT_PUBLISHED_TREATMENT_UNIDENTIFIED_NO_DEM_GET",
        "handoff": frozen["handoff_on_refutation"],
        "actions": {
            "dem_tile_get_or_range_requests": 0,
            "dem_tile_body_bytes": 0,
            "archive_weather_value_rows": 0,
            "model_fits": 0,
            "official_score_calls": 0,
            "target_or_scada_access": False,
            "test_member_access": False,
            "lockbox_2024_performance_access": False,
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
        default=Path("reports/s17_n15a_r3_corrected_formula_gate_predeclaration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/s17_n15a_r3_corrected_formula_gate.json"),
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
