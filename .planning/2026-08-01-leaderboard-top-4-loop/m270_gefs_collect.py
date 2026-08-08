"""M270 finding 2: parallel GEFS ensemble-spread collector.

Reusable for both the expanded probe and, if the signal survives, bulk collection.

DESIGN
HTTP fetches run on a thread pool because they are I/O bound and the serial probe measured
about 1.9 seconds per request. GRIB decoding stays SERIAL in the main thread: the ecCodes C
library's thread safety is not something to assume, and decoding is cheap next to transfer.

RESUMABILITY
Already-collected `(cycle_date, lead)` pairs are skipped, so an interrupted run resumes
without refetching. This matters under the single-root-session constraint.

AVAILABILITY-TIME COMPLIANCE
Only the previous day's 18Z cycle is used, issued about 10 hours before the 04:00 UTC
reference instant. The cycle is part of the S3 key, so availability is provable from the
path rather than asserted.

Read-only public HTTPS. No model is fitted, no 2024 row is read.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "artifacts" / "backtests" / "metric-aligned-probe"
STORE = PROBE_DIR / "M270_GEFS_SPREAD_PROBE.parquet"

BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
PRODUCT = "atmos/pgrb2sp25"
CYCLE = "18"
LEADS = (24, 30, 36, 42)
WANTED = ("UGRD:10 m above ground", "VGRD:10 m above ground", "GUST:surface")
GRID_POINTS = [(lat, lon) for lat in (37.50, 37.25, 37.00) for lon in (128.75, 129.00, 129.25)]

TIMEOUT = 60
RETRIES = 3


def _stem(date: str, lead: int) -> str:
    return f"{BUCKET}/gefs.{date}/{CYCLE}/{PRODUCT}/gespr.t{CYCLE}z.pgrb2s.0p25.f{lead:03d}"


def _get(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    last: Exception | None = None
    for _ in range(RETRIES):
        try:
            request = urllib.request.Request(url)
            if byte_range is not None:
                request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return response.read()
        except Exception as exc:  # transient S3/network failures are expected at this volume
            last = exc
    raise RuntimeError(f"fetch failed after {RETRIES} attempts: {url}") from last


def fetch_messages(date: str, lead: int) -> dict[str, bytes] | None:
    """Fetch only the wanted GRIB messages via byte ranges parsed from the .idx sidecar."""
    stem = _stem(date, lead)
    try:
        text = _get(stem + ".idx").decode("utf-8", errors="replace")
    except Exception:
        return None
    rows = []
    for line in text.strip().split("\n"):
        parts = line.split(":")
        if len(parts) >= 5:
            rows.append((int(parts[1]), ":".join(parts[3:5])))
    out: dict[str, bytes] = {}
    for i, (offset, description) in enumerate(rows):
        if description not in WANTED:
            continue
        end = rows[i + 1][0] - 1 if i + 1 < len(rows) else offset + 4_000_000
        try:
            out[description.split(":")[0]] = _get(stem, (offset, end))
        except Exception:
            return None
    return out if len(out) == len(WANTED) else None


def decode(messages: dict[str, bytes]) -> dict[str, float]:
    """Serial decode. Mean ensemble standard deviation over the nine supplied grid points."""
    import eccodes

    values: dict[str, float] = {}
    for name, raw in messages.items():
        handle = eccodes.codes_new_from_message(raw)
        try:
            points = [
                eccodes.codes_grib_find_nearest(handle, lat, lon)[0]["value"]
                for lat, lon in GRID_POINTS
            ]
        finally:
            eccodes.codes_release(handle)
        values[name] = float(np.mean(points))
    return values


def collect(days: list[str], workers: int) -> pd.DataFrame:
    existing = pd.read_parquet(STORE) if STORE.exists() else pd.DataFrame()
    done: set[tuple[str, int]] = set()
    if not existing.empty:
        done = {
            (str(r.cycle_date), int(r.lead))
            for r in existing.itertuples(index=False)
        }
    todo = [(d, lead) for d in days for lead in LEADS if (d, lead) not in done]
    print(f"already collected={len(done)} todo={len(todo)} workers={workers}", flush=True)

    records: list[dict[str, object]] = []
    failures = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(fetch_messages, d, lead): (d, lead) for d, lead in todo}
        for index, future in enumerate(futures.as_completed(pending), start=1):
            date, lead = pending[future]
            try:
                messages = future.result()
            except Exception:
                messages = None
            if messages is None:
                failures += 1
                continue
            values = decode(messages)  # serial by construction
            valid_kst = (
                datetime.strptime(date, "%Y%m%d") + timedelta(hours=18 + lead + 9)
            )
            records.append(
                {
                    "cycle_date": date,
                    "lead": lead,
                    "forecast_kst_dtm": pd.Timestamp(valid_kst),
                    "spread_u10": values["UGRD"],
                    "spread_v10": values["VGRD"],
                    "spread_gust": values["GUST"],
                }
            )
            if index % 100 == 0:
                print(f"  {index}/{len(todo)} fetched, failures={failures}", flush=True)

    fresh = pd.DataFrame(records)
    combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    if combined.empty:
        raise RuntimeError("no spread rows collected")
    combined["spread_vec"] = np.hypot(combined["spread_u10"], combined["spread_v10"])
    combined = combined.drop_duplicates(["cycle_date", "lead"]).sort_values(
        ["cycle_date", "lead"], kind="stable"
    )
    combined.to_parquet(STORE, index=False)
    print(f"new={len(fresh)} failures={failures} total={len(combined)}", flush=True)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start", default="2023-04-01")
    parser.add_argument("--end", default="2023-12-29")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")

    first = datetime.strptime(args.start, "%Y-%m-%d")
    last = datetime.strptime(args.end, "%Y-%m-%d")
    span = (last - first).days
    # Target days are sampled evenly; the cycle is always two days earlier, because the
    # reference time for targets on day T is 13:00 KST on T-1 and the cycle is 18Z on T-2.
    targets = [first + timedelta(days=round(i * span / (args.days - 1))) for i in range(args.days)]
    cycles = sorted({(t - timedelta(days=2)).strftime("%Y%m%d") for t in targets})
    collect(cycles, args.workers)


if __name__ == "__main__":
    main()
