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


## SK@v5 Hierarchical Discovery Workflow (2026-08-09 snapshot)

### Status

| Item | Value |
|---|---|
| Champion Total | 0.63148273 |
| Target | 0.660000 |
| Gap | 0.0285173 (≈15% MAE reduction needed) |
| EventStore sequence | 180 |
| Comparison count | 5 |
| Next comparison index | 6 |
| Lockbox 2024 | CONSUMED, unavailable |
| Fresh holdout | None |

### Recent adversarial review

A root-authored adversarial review identified that the workflow spends ∼70% of cycles on governance
and ∼20% on perturbing a known-capped model member, and only ∼10% on the representation problem
that gates the target. The review is at `.planning/2026-08-01-leaderboard-top-4-loop/adversarial_review.md`.

### Quick setup for a new VM or work environment

```bash
git clone https://github.com/umyunsang/BARAM2026.git
cd BARAM2026
# Python 3.12 required
uv sync --extra dev --extra challenger --extra experiment --extra graph
# Place the competition ZIP at inputs/competition/open_wind_236727.zip
# Verify its SHA-256:
uv run python -c "import hashlib;print(hashlib.sha256(open('inputs/competition/open_wind_236727.zip','rb').read()).hexdigest())"
# Should print: 920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b

# Build the prepared feature cache (one-time, ∼2 min, reads the ZIP):
uv run python -m baram.cli prepare

# Rebuild the seq__ and geom__ feature blocks (one-time, reads grid pivot):
uv run python scripts/build_s17_n36_seq_geom.py

# Run a quick diagnostic to verify the setup:
uv run python -m pytest tests -q

# Pre-declared comparison script (index 6):
# uv run python scripts/run_s17_n37_geom_strict.py

# Install additional packages for experiments (dependency freeze lifted 2026-08-09):
uv add torch xgboost catboost statsmodels  # or any package via uv
```

### Key directory layout

```
BARAM2026/
├── src/baram/          # Pipeline modules (data, features, models, evaluation, loop)
├── configs/            # YAML configuration (features, models, splits)
├── tests/              # Pytest suite
├── scripts/            # Experiment scripts (N29–N51)
├── research/           # Research lanes, lane outputs, node diagnostics
│   └── nodes/          # Frozen foundation maps and capability constraints
├── reports/            # Node predeclarations, closures, adjudication receipts
├── artifacts/
│   ├── registry/       # EventStore (SQLite), node specs, capability ontology
│   ├── manifests/      # Prepare manifest, candidate freezes
│   ├── features/       # Generated feature blocks (s17_n36 seq/geom)
│   └── backtests/      # Frozen stricter backtest artifacts (N7 actions)
├── inputs/
│   ├── competition/    # Place open_wind_236727.zip here (NOT committed)
│   ├── notebooks/      # Baseline IPYNB, official metric IPYNB
│   └── rules/          # Official competition rules snapshot
└── .planning/          # SK, DS, IP documents, adversarial review, task plans
```

### SK@v5 workflow authority

The workflow uses a three-gate approval cascade (SK → DS → IP), which the
adversarial review recommends collapsing. Currently approved:

- `SK@v5`: hierarchical SOTA-to-local pipeline discovery skeleton
- `DS@v5`: synthesized design specifying EventStore, NodeSpec v3, event v2, evaluation hierarchy
- `IP@v3`: 14-package implementation plan, dependency installation authorized, no 2024 lockbox

Every score-bearing comparison is predeclared and closed through the EventStore
(`artifacts/registry/loop_events_s17.sqlite`). Non-score-bearing diagnostics also
use the same governance.

### What is tracked vs not tracked

- Tracked: source code, tests, configs, scripts, research outputs, reports, planning docs,
  EventStore SQLite, registry JSON, manifests, feature block scripts
- NOT tracked: `.venv/`, `artifacts/cache/` (generated parquet files, ∼1.2 GB),
  `inputs/competition/open_wind_236727.zip` (competition data, 111 MB), `.pyc`,
  backtest artifacts, model checkpoints, submission CSVs
