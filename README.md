# BARAM2026 Wind Forecast Pipeline

This repository implements a leakage-safe, exact-metric forecasting workflow for Dacon competition 236727. It reads the supplied ZIP in place, derives deterministic content-addressed features, evaluates only chronology-safe OOF predictions, consumes the 2024 operating-year lockbox once, and builds a locally verified 2025 submission candidate.

The official local objective is:

`0.5 × (1 - mean(group NMAE)) + 0.5 × mean(group FICR)`

Only rows with actual generation at least 10% of group capacity are scored. FICR settlement is 4 units at capacity-relative absolute error up to 6%, 3 units above 6% through 8%, and 0 beyond 8%.

## Boundaries

- Source files are read-only and hash-checked.
- The 24 forecasts from 01:00 through the following 00:00 form one operating day.
- Development uses operating-year 2023 Q2–Q4; operating year 2024 is the one-time lockbox.
- No external data, remote model API, Dacon upload, or leaderboard claim is made here.

## Operator workflow

Synchronize the approved local runtime with `uv sync --extra dev --extra challenger`, then use `uv run python -m baram.cli --help`. The closed workflow is audit, prepare, split-build, backtest, select, lockbox, fit-final, build-submission, and reproduce. The challenger extra is only exercised when the post-LightGBM activation gate passes. Every mutating stage writes hashes and receipts under `reports/` or `artifacts/`.

The development stages are `controls`, `lightgbm`, `ablation`, and conditionally `challengers`. Run `select` after all activated development stages to freeze at most three candidates. Do not invoke `lockbox` until the frozen manifest, source hashes, tests, and configuration budget have been independently checked.

The development ladder must finish and write `artifacts/manifests/candidate_freeze.json` before `lockbox` is called. The lockbox command atomically consumes its one-use local receipt before reading 2024 labels; a failed or interrupted attempt is still consumed. `build-submission` only creates and validates a local UTF-8-BOM CSV. It never opens a browser or uploads to Dacon.
