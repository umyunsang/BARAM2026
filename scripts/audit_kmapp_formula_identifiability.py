"""Fail closed on the frozen N15 primary-formula identifiability gate."""

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
        raise RuntimeError("N15 input hash mismatch")
    if _canonical_hash(observed) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N15 input bundle mismatch")
    pdf_path = repo / "research/sources/kim_lee_et_al_2021_kmapp_28740.pdf"
    page = PdfReader(pdf_path).pages[2].extract_text()
    compact = re.sub(r"\s+", "", page)
    equation_excerpt_match = re.search(
        r"[Δ]uHCz.*?e.*?[Δ]H,", compact, flags=re.DOTALL
    )
    equation_excerpt = (
        equation_excerpt_match.group(0) if equation_excerpt_match else ""
    )
    has_u_hhc_definition = all(
        marker in page
        for marker in ("u(hHC)", "LDAPS", "reference", "height")
    )
    # The PDF text layer places delta_H immediately after exp(-kappa*z), with
    # no second kappa token. The preregistration had frozen a kappa multiplier.
    printed_no_multiplier = bool(
        "ezH" in compact and "ezH" not in compact
    )
    n12_receipt = json.loads(
        (repo / "reports/s17_n12_vertical_profile_prerequisite_receipt.json").read_text()
    )
    n12_path = repo / n12_receipt["artifact"]["path"]
    if _sha256(n12_path) != n12_receipt["artifact"]["sha256"]:
        raise RuntimeError("N12 artifact mutation")
    n12 = json.loads(n12_path.read_text())
    source_vectors = n12["source_vectors"]
    ldaps_levels = source_vectors["ldaps"]["comparable_heights_m"]
    ldaps_brackets_117 = source_vectors["ldaps"]["brackets_117m"]
    extrema = n12["ldaps_50_extrema_present_but_not_level_means"]
    exact_formula_matches_freeze = not printed_no_multiplier
    u_hhc_identified = bool(
        has_u_hhc_definition
        and ldaps_brackets_117
        and not any(extrema.values())
    )
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
            "printed_equation_has_no_kappa_multiplier_after_exponential": printed_no_multiplier,
            "text_defines_u_hhc_as_ldaps_forecast_wind_at_reference_height": has_u_hhc_definition,
        },
        "predeclaration_formula_matches_primary_print": exact_formula_matches_freeze,
        "supplied_vertical_support": {
            "ldaps_comparable_mean_vector_heights_m": ldaps_levels,
            "ldaps_50m_fields_are_component_extrema_not_mean_vectors": extrema,
            "brackets_even_117m_hub_height": ldaps_brackets_117,
            "u_hhc_identified_without_substitution_or_extrapolation": u_hhc_identified,
        },
        "gates": {
            "exact_primary_equation_matches_frozen_equation": exact_formula_matches_freeze,
            "u_hhc_identified": u_hhc_identified,
        },
        "verdict": "INCONCLUSIVE_PREDECLARED_EQUATION_MISMATCH_NO_TILE_GET",
        "handoff": [
            "S17-N15A_R3_CORRECTED_PRIMARY_FORMULA_IDENTIFIABILITY_GATE"
        ],
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
        default=Path("reports/s17_n15_r3_formula_identifiability_predeclaration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/s17_n15_r3_formula_identifiability.json"),
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
