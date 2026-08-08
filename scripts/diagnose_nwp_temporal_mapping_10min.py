"""Corrected four-mapping NWP versus 10-minute SCADA wind diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAPPINGS = ("t", "t_minus_1", "mean_t_minus_1_t", "mean_t_t_plus_1")
CURRENT = "t"
NWP_SPECS = {
    "gfs": {
        "member": "train/gfs_train.csv",
        "rows": 157_680,
        "winds": {
            "gfs_10m": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
            "gfs_80m": ("heightAboveGround_80_u", "heightAboveGround_80_v"),
            "gfs_100m": (
                "heightAboveGround_100_100u",
                "heightAboveGround_100_100v",
            ),
        },
    },
    "ldaps": {
        "member": "train/ldaps_train.csv",
        "rows": 280_320,
        "winds": {
            "ldaps_10m": (
                "heightAboveGround_10_10u",
                "heightAboveGround_10_10v",
            ),
        },
    },
}
SCADA_SPECS = {
    "vestas": {
        "member": "train/scada_vestas_train.csv",
        "columns": [f"vestas_wtg{index:02d}_ws" for index in range(1, 13)],
        "minimum_turbines": 6,
        "first_hour": pd.Timestamp("2022-01-01 01:00:00"),
        "hour_rule": "ceil",
    },
    "unison": {
        "member": "train/scada_unison_train.csv",
        "columns": [f"unison_wtg{index:02d}_ws" for index in range(1, 6)],
        "minimum_turbines": 3,
        "first_hour": pd.Timestamp("2023-01-01 01:00:00"),
        "hour_rule": "floor_plus_one",
    },
}
END_HOUR = pd.Timestamp("2024-01-01 00:00:00")
ALL_DAYS = pd.date_range("2022-01-01", "2023-12-31", freq="D")


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


def _mapped_hour(timestamp: pd.Timestamp, rule: str) -> pd.Timestamp:
    if rule == "ceil":
        return timestamp.ceil("h")
    if rule == "floor_plus_one":
        return timestamp.floor("h") + timedelta(hours=1)
    raise ValueError(f"unknown hour rule: {rule}")


def _scada_prefix_rows(
    archive: zipfile.ZipFile,
    member: str,
    first_hour: pd.Timestamp,
    rule: str,
) -> tuple[int, str, str]:
    """Count only the timestamp prefix; do not parse any SCADA value field."""
    count = 0
    raw_min: str | None = None
    raw_max: str | None = None
    with archive.open(member) as binary:
        stream = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
        header = stream.readline().rstrip("\r\n").split(",")
        if not header or header[0] != "kst_dtm":
            raise RuntimeError(f"{member}: timestamp must be first")
        for line in stream:
            raw_timestamp = line.split(",", 1)[0]
            mapped = _mapped_hour(pd.Timestamp(raw_timestamp), rule)
            if mapped > END_HOUR:
                break
            if mapped < first_hour:
                raise RuntimeError(f"{member}: unexpected pre-scope row")
            raw_min = raw_timestamp if raw_min is None else raw_min
            raw_max = raw_timestamp
            count += 1
    if not count or raw_min is None or raw_max is None:
        raise RuntimeError(f"{member}: no in-scope SCADA rows")
    return count, raw_min, raw_max


def _load_scada(
    archive: zipfile.ZipFile,
    farm: str,
    spec: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    prefix_rows, raw_min, raw_max = _scada_prefix_rows(
        archive,
        spec["member"],
        spec["first_hour"],
        spec["hour_rule"],
    )
    selected_columns = ["kst_dtm", *spec["columns"]]
    with archive.open(spec["member"]) as stream:
        frame = pd.read_csv(
            stream,
            encoding="utf-8-sig",
            nrows=prefix_rows,
            usecols=selected_columns,
        )
    if list(frame.columns) != selected_columns:
        raise RuntimeError(f"{farm}: projected SCADA schema mismatch")
    timestamp = pd.to_datetime(frame["kst_dtm"], errors="raise")
    if not timestamp.is_monotonic_increasing:
        raise RuntimeError(f"{farm}: raw SCADA timestamps not monotone")
    if spec["hour_rule"] == "ceil":
        hour = timestamp.dt.ceil("h")
    else:
        hour = timestamp.dt.floor("h") + timedelta(hours=1)
    if hour.min() != spec["first_hour"] or hour.max() != END_HOUR:
        raise RuntimeError(f"{farm}: projected SCADA hour boundary mismatch")
    values = frame[spec["columns"]].apply(pd.to_numeric, errors="coerce")
    values = values.where((values >= 0.0) & (values <= 40.0))
    valid_turbines = values.notna().sum(axis=1)
    row_median = values.median(axis=1, skipna=True).where(
        valid_turbines >= spec["minimum_turbines"]
    )
    hourly_frame = pd.DataFrame({"hour": hour, "wind": row_median})
    grouped = hourly_frame.groupby("hour", sort=True)["wind"]
    slot_count = grouped.count()
    hourly = grouped.mean().where(slot_count >= 4).rename(f"scada_{farm}")
    expected_hours = pd.date_range(spec["first_hour"], END_HOUR, freq="h")
    hourly = hourly.reindex(expected_hours)
    report = {
        "member": spec["member"],
        "timestamp_only_prefix_pass": True,
        "projected_columns": selected_columns,
        "power_columns_materialized": False,
        "prefix_rows": prefix_rows,
        "raw_timestamp_min": raw_min,
        "raw_timestamp_max": raw_max,
        "mapped_hour_min": hourly.index.min().isoformat(),
        "mapped_hour_max": hourly.index.max().isoformat(),
        "hour_count": len(hourly),
        "finite_hour_count": int(hourly.notna().sum()),
        "coverage": float(hourly.notna().mean()),
        "minimum_turbines_per_row": spec["minimum_turbines"],
        "minimum_ten_minute_rows_per_hour": 4,
        "hour_rule": spec["hour_rule"],
    }
    return hourly, report


def _load_nwp(
    archive: zipfile.ZipFile,
    source: str,
    spec: dict[str, Any],
) -> pd.DataFrame:
    wind_columns = [column for pair in spec["winds"].values() for column in pair]
    selected = [
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        *wind_columns,
    ]
    with archive.open(spec["member"]) as stream:
        frame = pd.read_csv(
            stream,
            encoding="utf-8-sig",
            nrows=int(spec["rows"]),
            usecols=selected,
        )
    if len(frame) != spec["rows"]:
        raise RuntimeError(f"{source}: NWP prefix row mismatch")
    valid = pd.to_datetime(frame.pop("forecast_kst_dtm"), errors="raise")
    available = pd.to_datetime(frame.pop("data_available_kst_dtm"), errors="raise")
    operating_day = (valid - timedelta(hours=1)).dt.normalize()
    if operating_day.min() != ALL_DAYS.min() or operating_day.max() != ALL_DAYS.max():
        raise RuntimeError(f"{source}: NWP operating-day scope mismatch")
    result = pd.DataFrame({"available": available, "valid": valid})
    for name, (u_column, v_column) in spec["winds"].items():
        u = pd.to_numeric(frame[u_column], errors="coerce")
        v = pd.to_numeric(frame[v_column], errors="coerce")
        result[name] = np.hypot(u, v)
    grouped = result.groupby(["available", "valid"], sort=True).mean(numeric_only=True)
    expected_grid_count = 9 if source == "gfs" else 16
    counts = result.groupby(["available", "valid"], sort=True).size()
    if not bool((counts == expected_grid_count).all()):
        raise RuntimeError(f"{source}: incomplete grid field")
    if not np.isfinite(grouped.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{source}: nonfinite NWP wind")
    return grouped.reset_index()


def _mapping_frame(nwp: pd.DataFrame) -> pd.DataFrame:
    wind_columns = list(NWP_SPECS["gfs"]["winds"]) + list(
        NWP_SPECS["ldaps"]["winds"]
    )
    ordered = nwp.sort_values(["available", "valid"], kind="stable").reset_index(drop=True)
    issue_size = ordered.groupby("available")["valid"].transform("size")
    if not bool((issue_size == 24).all()):
        raise RuntimeError("NWP issue does not contain exactly 24 valid hours")
    position = ordered.groupby("available").cumcount()
    prior = ordered.groupby("available", sort=False)[wind_columns].shift(1)
    following = ordered.groupby("available", sort=False)[wind_columns].shift(-1)
    common = (position >= 1) & (position <= 22)
    base = ordered.loc[common, ["available", "valid"]].copy()
    base["operating_day"] = (base["valid"] - timedelta(hours=1)).dt.normalize()
    if set(base["valid"].dt.hour.unique()) != set(range(2, 24)):
        raise RuntimeError("common mapping support must be valid hours 02 through 23")
    for column in wind_columns:
        current = ordered.loc[common, column].to_numpy(dtype=float)
        previous = prior.loc[common, column].to_numpy(dtype=float)
        next_value = following.loc[common, column].to_numpy(dtype=float)
        base[f"{column}__t"] = current
        base[f"{column}__t_minus_1"] = previous
        base[f"{column}__mean_t_minus_1_t"] = 0.5 * (previous + current)
        base[f"{column}__mean_t_t_plus_1"] = 0.5 * (current + next_value)
    if not np.isfinite(base.filter(regex="__(?:t|mean)").to_numpy(dtype=float)).all():
        raise RuntimeError("nonfinite mapped NWP wind")
    return base


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def _fisher(correlation: float | np.ndarray) -> float | np.ndarray:
    return np.arctanh(np.clip(correlation, -0.999999, 0.999999))


def _stationary_indices(
    days: int,
    replicates: int,
    mean_block: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.empty((replicates, days), dtype=np.int16)
    indices[:, 0] = rng.integers(0, days, size=replicates)
    for offset in range(1, days):
        restart = rng.random(replicates) < (1.0 / mean_block)
        continuation = (indices[:, offset - 1].astype(int) + 1) % days
        replacement = rng.integers(0, days, size=replicates)
        indices[:, offset] = np.where(restart, replacement, continuation)
    return indices


def _correlation_from_stats(stats: np.ndarray) -> np.ndarray:
    n, sx, sy, sxx, syy, sxy = np.moveaxis(stats, -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = sxy - sx * sy / n
        variance_x = sxx - sx * sx / n
        variance_y = syy - sy * sy / n
        result = covariance / np.sqrt(variance_x * variance_y)
    return np.clip(result, -0.999999, 0.999999)


def run(repo: Path, predeclaration: Path, output: Path) -> dict[str, Any]:
    frozen = json.loads(predeclaration.read_text())
    observed_hashes = {
        relative: _sha256(repo / relative)
        for relative in frozen["input_bundle"]["files"]
    }
    if observed_hashes != frozen["input_bundle"]["files"]:
        raise RuntimeError("N11A input hash mismatch")
    if _canonical_hash(observed_hashes) != frozen["input_bundle"]["sha256"]:
        raise RuntimeError("N11A input bundle mismatch")

    archive_path = repo / "inputs/competition/open_wind_236727.zip"
    with zipfile.ZipFile(archive_path) as archive:
        nwp_sources = {
            source: _load_nwp(archive, source, spec)
            for source, spec in NWP_SPECS.items()
        }
        scada_loaded = {
            farm: _load_scada(archive, farm, spec)
            for farm, spec in SCADA_SPECS.items()
        }
    nwp = nwp_sources["gfs"].merge(
        nwp_sources["ldaps"],
        on=["available", "valid"],
        how="inner",
        validate="one_to_one",
    )
    if len(nwp) != 730 * 24:
        raise RuntimeError("cross-source NWP key mismatch")
    mapped = _mapping_frame(nwp)
    scada_reports: dict[str, Any] = {}
    for farm, (hourly, report) in scada_loaded.items():
        mapped = mapped.merge(
            hourly.rename_axis("valid").reset_index(),
            on="valid",
            how="left",
            validate="many_to_one",
        )
        scada_reports[farm] = report

    winds = list(NWP_SPECS["gfs"]["winds"]) + list(NWP_SPECS["ldaps"]["winds"])
    panels = [(wind, farm) for wind in winds for farm in SCADA_SPECS]
    panel_names = [f"{wind}__{farm}" for wind, farm in panels]
    correlations: dict[str, dict[str, float]] = {mapping: {} for mapping in MAPPINGS}
    pooled_fisher: dict[str, dict[str, float]] = {mapping: {} for mapping in MAPPINGS}
    coverage_records: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []

    for wind, farm in panels:
        panel = f"{wind}__{farm}"
        y = mapped[f"scada_{farm}"].to_numpy(dtype=float)
        x_values = {
            mapping: mapped[f"{wind}__{mapping}"].to_numpy(dtype=float)
            for mapping in MAPPINGS
        }
        common = np.isfinite(y)
        for values in x_values.values():
            common &= np.isfinite(values)
        for mapping, values in x_values.items():
            correlation = _correlation(values[common], y[common])
            correlations[mapping][panel] = correlation
            pooled_fisher[mapping][panel] = float(_fisher(correlation))
        available_years = (2022, 2023) if farm == "vestas" else (2023,)
        for year in available_years:
            year_mask = mapped["operating_day"].dt.year.to_numpy() == year
            mask = common & year_mask
            expected_rows = int(year_mask.sum())
            record: dict[str, Any] = {
                "panel": panel,
                "source_family": wind.split("_", 1)[0],
                "farm": farm,
                "year": year,
                "rows": int(mask.sum()),
                "expected_rows": expected_rows,
                "coverage": float(mask.sum() / expected_rows),
                "correlations": {},
            }
            z_values: dict[str, float] = {}
            for mapping, values in x_values.items():
                correlation = _correlation(values[mask], y[mask])
                record["correlations"][mapping] = correlation
                z_values[mapping] = float(_fisher(correlation))
            best_value = max(z_values.values())
            winners = [name for name, value in z_values.items() if abs(value - best_value) <= 1e-12]
            record["winner"] = winners[0] if len(winners) == 1 else "TIE"
            strata.append(record)
            coverage_records.append(
                {
                    "panel": panel,
                    "year": year,
                    "rows": int(mask.sum()),
                    "expected_rows": expected_rows,
                    "coverage": float(mask.sum() / expected_rows),
                }
            )

    aggregate = {
        mapping: float(np.mean(list(pooled_fisher[mapping].values())))
        for mapping in MAPPINGS
    }
    delta = {mapping: aggregate[mapping] - aggregate[CURRENT] for mapping in MAPPINGS}
    family_delta: dict[str, dict[str, float]] = {}
    for family in ("gfs", "ldaps"):
        family_panels = [name for name in panel_names if name.startswith(f"{family}_")]
        family_delta[family] = {
            mapping: float(
                np.mean([pooled_fisher[mapping][name] for name in family_panels])
                - np.mean([pooled_fisher[CURRENT][name] for name in family_panels])
            )
            for mapping in MAPPINGS
        }
    strata_wins = {
        mapping: sum(record["winner"] == mapping for record in strata)
        for mapping in MAPPINGS
    }

    # Per-day sufficient statistics permit a joint stationary bootstrap without
    # repeatedly materializing hourly observations.
    day_lookup = {day: index for index, day in enumerate(ALL_DAYS)}
    daily_stats = np.zeros((len(ALL_DAYS), len(panels), len(MAPPINGS), 6), dtype=float)
    operating_days = mapped["operating_day"]
    for panel_index, (wind, farm) in enumerate(panels):
        y_all = mapped[f"scada_{farm}"].to_numpy(dtype=float)
        for mapping_index, mapping in enumerate(MAPPINGS):
            x_all = mapped[f"{wind}__{mapping}"].to_numpy(dtype=float)
            finite = np.isfinite(x_all) & np.isfinite(y_all)
            for day, positions in mapped.loc[finite].groupby(operating_days[finite]).groups.items():
                indices = np.asarray(list(positions), dtype=int)
                x = x_all[indices]
                y = y_all[indices]
                daily_stats[day_lookup[pd.Timestamp(day)], panel_index, mapping_index] = (
                    len(x),
                    x.sum(),
                    y.sum(),
                    np.dot(x, x),
                    np.dot(y, y),
                    np.dot(x, y),
                )
    bootstrap = frozen["metric"]["bootstrap"]
    indices = _stationary_indices(
        len(ALL_DAYS),
        int(bootstrap["replicates"]),
        int(bootstrap["mean_block_days"]),
        int(bootstrap["seed"]),
    )
    weights = np.zeros((len(indices), len(ALL_DAYS)), dtype=np.int16)
    for replicate, sampled in enumerate(indices):
        weights[replicate] = np.bincount(sampled, minlength=len(ALL_DAYS))
    totals = np.einsum("bd,dpmk->bpmk", weights, daily_stats, optimize=True)
    bootstrap_r = _correlation_from_stats(totals)
    if not np.isfinite(bootstrap_r).all():
        raise RuntimeError("nonfinite bootstrap correlation")
    bootstrap_aggregate = np.mean(_fisher(bootstrap_r), axis=1)
    bootstrap_delta = bootstrap_aggregate - bootstrap_aggregate[:, [0]]
    delta_vector = np.array([delta[mapping] for mapping in MAPPINGS])
    alternative_indices = np.arange(1, len(MAPPINGS))
    centered_error = delta_vector[alternative_indices] - bootstrap_delta[:, alternative_indices]
    max_error = np.max(centered_error, axis=1)
    critical = float(np.quantile(max_error, 0.95))
    simultaneous_lower = {
        mapping: float(delta[mapping] - critical)
        for mapping in MAPPINGS[1:]
    }
    bootstrap_intervals = {
        mapping: {
            "p05": float(np.quantile(bootstrap_delta[:, index], 0.05)),
            "p50": float(np.quantile(bootstrap_delta[:, index], 0.50)),
            "p95": float(np.quantile(bootstrap_delta[:, index], 0.95)),
            "simultaneous_one_sided_95_lower": simultaneous_lower[mapping],
        }
        for index, mapping in enumerate(MAPPINGS[1:], start=1)
    }

    largest = max(aggregate.values())
    winners = [mapping for mapping, value in aggregate.items() if abs(value - largest) <= 1e-12]
    candidate = winners[0] if len(winners) == 1 else "TIE"
    coverage_gate = all(record["coverage"] >= 0.95 for record in coverage_records)
    clear_checks: dict[str, Any] = {
        "unique_largest": candidate != "TIE",
        "noncurrent": candidate not in {"TIE", CURRENT},
        "delta_at_least_0_01": False,
        "simultaneous_lower_gt_zero": False,
        "strata_wins_at_least_9_of_12": False,
        "positive_in_gfs_and_ldaps": False,
        "coverage_each_panel_year_at_least_0_95": coverage_gate,
    }
    if candidate not in {"TIE", CURRENT}:
        clear_checks.update(
            {
                "delta_at_least_0_01": delta[candidate] >= 0.01,
                "simultaneous_lower_gt_zero": simultaneous_lower[candidate] > 0.0,
                "strata_wins_at_least_9_of_12": strata_wins[candidate] >= 9,
                "positive_in_gfs_and_ldaps": all(
                    family_delta[family][candidate] > 0.0 for family in family_delta
                ),
            }
        )
    clear_winner = all(clear_checks.values())
    if clear_winner:
        result_status = "SUPPORTED_CLEAR_NONCURRENT"
        handoff = [f"S17-N12_{candidate.upper()}_STRICT_PREQUENTIAL_OOF"]
    elif candidate == CURRENT:
        result_status = "REFUTED_CURRENT_T_BEST"
        handoff = ["S17-N12_VERTICAL_PROFILE_TERRAIN_PREREQUISITE_AUDIT"]
    else:
        result_status = "INCONCLUSIVE_NO_CLEAR_WINNER"
        handoff = ["S17-N12_VERTICAL_PROFILE_TERRAIN_PREREQUISITE_AUDIT"]

    result = {
        "schema_version": 1,
        "node_id": "S17-N11A_FOUR_MAPPING_10MIN_SCADA_CORRECTED_DIAGNOSTIC",
        "predeclaration_sha256": _sha256(predeclaration),
        "input_bundle_sha256": frozen["input_bundle"]["sha256"],
        "scope": {
            "operating_days": 730,
            "common_hours_per_day": 22,
            "2024_mapped_hour_wind_materialized": False,
            "power_columns_materialized": False,
        },
        "scada": scada_reports,
        "mapping_order": list(MAPPINGS),
        "nwp_panel_definition": "mean across grids of per-grid vector speed",
        "scada_panel_definition": "turbine median per 10-minute row then hourly mean",
        "aggregate_fisher_z": aggregate,
        "delta_fisher_z_vs_t": delta,
        "pooled_correlations": correlations,
        "family_delta_fisher_z_vs_t": family_delta,
        "strata": strata,
        "strata_wins": strata_wins,
        "coverage": coverage_records,
        "bootstrap": {
            "scheme": "stationary operating-day",
            "mean_block_days": int(bootstrap["mean_block_days"]),
            "replicates": int(bootstrap["replicates"]),
            "seed": int(bootstrap["seed"]),
            "centered_max_critical_95": critical,
            "intervals": bootstrap_intervals,
        },
        "candidate": candidate,
        "clear_winner_checks": clear_checks,
        "clear_noncurrent_winner": clear_winner,
        "result_status": result_status,
        "handoff": handoff,
        "actions": {
            "model_fits": 0,
            "training_scada_wind_access": True,
            "power_columns_or_labels_access": False,
            "official_score_calls": 0,
            "comparison_index": None,
            "lockbox_2024_performance_access": False,
            "test_data_access": False,
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
        default=Path("reports/s17_n11a_four_mapping_10min_scada_predeclaration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/s17_n11a_four_mapping_wind.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    predeclaration = args.predeclaration
    if not predeclaration.is_absolute():
        predeclaration = repo / predeclaration
    output = args.output
    if not output.is_absolute():
        output = repo / output
    result = run(repo, predeclaration, output)
    print(
        json.dumps(
            {
                "candidate": result["candidate"],
                "delta": result["delta_fisher_z_vs_t"],
                "checks": result["clear_winner_checks"],
                "status": result["result_status"],
                "handoff": result["handoff"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
