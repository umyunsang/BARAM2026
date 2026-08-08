"""Replace only validation site-wind predictions in an existing OOF-safe cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_sequence_classifier import (
    BASELINE,
    BASELINE_SHA,
    OPEN,
    OPEN_SHA,
    OUTPUT,
    _sha256,
    _surface,
)
from run_site_wind_classifier import FOLDS
from run_site_wind_teacher import _validation_mask

BASE_CACHE_ID = "M64B_ALLWEATHER_SITEWIND_CLASS"
KEYS = ["forecast_id", "forecast_kst_dtm", "group_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold", choices=FOLDS, default=FOLDS[-1])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--column", required=True)
    args = parser.parse_args()
    if _sha256(OPEN) != OPEN_SHA or _sha256(BASELINE) != BASELINE_SHA:
        raise RuntimeError("immutable competition input hash mismatch")
    surface, _, _ = _surface()
    validation = _validation_mask(surface, args.fold)
    source = pd.read_parquet(args.source)
    required = {*KEYS, args.column}
    if required.difference(source.columns):
        missing = sorted(required.difference(source.columns))
        raise RuntimeError(f"missing override columns: {missing}")
    expected = surface.loc[validation, KEYS].reset_index(drop=True)
    observed = source[KEYS].reset_index(drop=True)
    if not expected.equals(observed):
        raise RuntimeError("site-wind validation key order changed")
    values = source[args.column].to_numpy(dtype="float32")
    if not np.isfinite(values).all():
        raise RuntimeError("site-wind override contains non-finite values")

    base_path = OUTPUT / f"{BASE_CACHE_ID}-{args.fold}-sitewind-features.npz"
    base = np.load(base_path)
    legacy = base["legacy"].copy()
    allweather = base["allweather"].copy()
    allweather[validation] = values
    output_path = OUTPUT / f"{args.candidate_id}-{args.fold}-sitewind-features.npz"
    np.savez_compressed(
        output_path,
        legacy=legacy,
        allweather=allweather,
        iterations=base["iterations"],
    )
    receipt = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "fold_id": args.fold,
        "scope": "validation-only NWP-to-site-wind representation screen",
        "base_cache_path": str(base_path.relative_to(Path.cwd())),
        "base_cache_sha256": _sha256(base_path),
        "source_path": str(args.source),
        "source_sha256": _sha256(args.source),
        "source_column": args.column,
        "override_count": int(validation.sum()),
        "cache_path": str(output_path.relative_to(Path.cwd())),
        "cache_sha256": _sha256(output_path),
        "observed_validation_scada_used_for_power_prediction": False,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "external_actions": [],
    }
    receipt_path = OUTPUT / f"{args.candidate_id}-{args.fold}-sitewind-features.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
