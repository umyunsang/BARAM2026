# Implementation Plan — BARAM 2026 Top-20 Forecasting Pipeline

## Metadata

- Plan ID: `IP`
- Version: `v2`
- Status: `IMPLEMENTATION_APPROVED_AND_ACTIVE(IP@v2)`
- Date: 2026-08-01 (Asia/Seoul)
- Approved design base: `DS@v3`
- Research base: `SK@v3`, `reports/top20_module_research.md`
- Repository baseline: Git `383265af810eee24129de2d7ce99457bb8a757ef`
- Planned execution run ID: `baram-v2-20260801-01`
- Nearest authority: `/Users/um-yunsang/BARAM2026/AGENTS.md`
- Requested approval phrase: `IP@v2 승인`

## 1. Authority and outcome

After exact `IP@v2 승인`, implement and execute a **single-root, competition-data-only, chronology-safe local experiment loop** that extends the existing `src/baram` pipeline. The loop will test supplied-grid spatial features, robust point models, predictive distributions, expected official-utility decisions, and at most three residual-diverse ensemble parents. It will freeze and reproduce the best supported local 2025 candidate without uploading it.

`IP@v2` approval will authorize the local source edits, tests, bounded model training, generated local artifacts, final fitting, local prediction/CSV construction, and reproduction steps specified here. It will not authorize Dacon upload/account mutation, external data or weights, remote inference/compute, remote Git, system-package changes, deep/TFT/NGBoost installation or execution, or any new 2024 validation score.

## 2. Immutable and runtime preconditions

Every full stage starts with these fail-closed checks:

1. `open.zip` SHA-256 equals `920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b`.
2. `baseline.ipynb` SHA-256 equals `712b26f4d2748860c94cff1e0100c23810468c983173f8e9ef8d009fe82df48c`.
3. Python is project-local `>=3.12,<3.13`; no system Python or package mutation.
4. `n_jobs` is within `1..6`; every estimator receives at most six workers.
5. `artifacts/locks/lockbox-2024.consumed.json` exists and remains byte-identical.
6. No development split, scorer, calibration policy, ensemble weight, or promotion rule contains operating year 2024.
7. Generated artifacts remain below the existing 10 GiB budget and stay untracked.
8. No subagent, worktree, background process, remote compute, browser mutation, or Dacon action is started.

Failure of items 1-6 is an immediate pipeline block. Storage excess pauses before writing the next artifact. No automatic deletion of prior user artifacts is authorized.

## 3. Frozen development and inference contract

- Development/selection horizon: existing issuance-batch folds inside 2023 only.
- Fold unit: complete daily NWP issuance batch, never individual rows.
- Label availability: groups 1/2/3 only where the existing fold contract marks them eligible and complete.
- 2024: no new score, slice, ablation, calibration, policy fit, or selection. Prior receipts are historical diagnostics only.
- Final fit: after the v2 candidate policy is frozen, training may consume all competition-supplied 2022-2024 labels as ordinary official training rows without calculating a 2024 metric. This is not a second lockbox evaluation: no 2024 score, slice, comparison, policy fit, or selection is produced, and no final-fit output may feed back into selection.
- Test inference: supplied 2025 GFS/LDAPS and static `info.xlsx` metadata only; no SCADA/target lag/realized observation.
- Official metric: `src/baram/evaluation/official.py` remains the scoring oracle and is not redefined by model modules.
- External target guide: Total `0.66200`, `1-NMAE 0.87500`, FICR `0.44900`; it remains a Dacon-derived guide, not an offline PASS threshold.

## 4. Experiment accounting

### 4.1 Definition

A full-chronology candidate is one frozen feature/model/distribution/policy/ensemble specification evaluated across every eligible 2023 development fold. Quantiles belonging to one frozen distribution bundle count as one candidate. Seed repeats of a finalist are recorded separately and cannot create a new configuration.

### 4.2 Hard ceilings

| Stage | Full-chronology candidates | Notes |
| --- | ---: | --- |
| Contract/parity | 2 | Existing pipeline parity and repeated hash |
| Spatial/physical feature families | 8 | Includes current-feature control |
| Deterministic point models | 10 | Conditional CatBoost rows are skipped, not replaced, if unavailable/gated out |
| Distribution bundles | 8 | Residual control, LightGBM quantiles, dependency-free QRF implementations |
| Expected-utility policies | 10 | Median/coarse controls included |
| Ensemble candidates | 6 | Two or three parents only |
| Final reproduction | 2 | Identical config/input runs |
| **Maximum** | **46** | Stop earlier when evidence is sufficient |

No stage may borrow unused slots from another stage without a revised approved plan. Cached, hash-identical reruns do not consume a slot; a changed config does.

## 5. Promotion thresholds fixed before training

The prior three-seed finalist ranges were `0.001128`, `0.003014`, and `0.002511`. Therefore:

| Threshold | Fixed value | Application |
| --- | ---: | --- |
| Material pooled Total improvement | `>= +0.0035` | Feature, point, decision, ensemble promotion |
| Finalist three-seed Total range | `<= 0.0035` | Point/distribution-derived finalist stability |
| Evaluation-fold Total delta | `> 0` on every later fold | No lucky-fold promotion |
| Pooled `1-NMAE` guardrail | `>= -0.0010` versus parent | Decision/ensemble may trade only a bounded amount |
| Per-group component-Total guardrail | `>= -0.0010` for every group | Prevent hidden group collapse |
| Group-3 component-Total delta | `>= 0` | Mandatory for decision/ensemble promotion |
| Decision-layer FICR improvement | `>= +0.0070` | Matches `+0.0035` Total when NMAE is flat |
| Residual-diversity absolute correlation | `< 0.995` | Existing conservative parent-admission threshold |

The material Total threshold is above the largest observed seed range (`0.003014`). It rejects the prior blend gain (`+0.000907`) while accepting the prior chronology-safe calibration gain (`+0.020432`). Values are evaluated without display rounding.

Distribution-specific gates:

- quantile levels exactly `(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)`;
- post-repair crossing rate `0` and finite keyed output coverage `100%`;
- mean normalized pinball loss across those levels `<= 0.99 ×` the preceding-fold empirical-residual distribution control;
- q10-q90 empirical coverage in `[0.72, 0.88]` pooled and `[0.65, 0.92]` for every group;
- q50 official Total no worse than the frozen point parent by more than `0.0010`.

If no distribution passes, the pipeline retains the point champion and current coarse calibration; it does not fabricate an expected-utility PASS.

## 6. Planned file and interface changes

### T0 — Gate, baseline, and immutable receipt (`M0`, `M9`)

**Modify**

- `src/baram/workflows.py`: add `run_v2_preflight`, stage accounting, closed-lockbox guard, and v2 receipt routing.
- `src/baram/cli.py`: expose only approved v2 local stages; keep existing commands compatible.
- `src/baram/contracts/types.py`: add frozen v2 metadata/policy manifest value objects.
- `src/baram/experiments/registry.py`: record stage slot consumption and parent hashes atomically.

**Add**

- `configs/v2/promotion.yaml`: the exact thresholds in section 5.
- `configs/v2/budget.yaml`: per-stage slot ceilings and 10 GiB limit.
- `tests/integration/test_v2_preflight.py`.
- `tests/experiments/test_v2_budget.py`.

**Required behavior**

- preflight verifies both immutable source hashes, Python version, worker cap, artifact budget, current Git/code identity, exact metric hash, and consumed-lock byte hash;
- any attempt to route a v2 development command to `run_lockbox` raises `ContractError` before reading labels or predictions;
- each stage manifest records input/config/code/output hashes, slots used/remaining, runtime, seed, worker count, and lockbox byte hash.

**PASS**: contract/parity tests pass twice with identical manifest hashes and the consumed-lock file remains byte-identical.

### T1 — Turbine metadata contract (`M0`, `M1`)

**Add**

- `src/baram/data/turbines.py` with:
  - `parse_dms_coordinate`;
  - `parse_turbine_workbook`;
  - `validate_turbine_topology`;
  - `group_static_metadata`.
- `tests/data/test_turbines.py` using an in-memory workbook fixture and immutable-archive integration assertion.

**Modify**

- `src/baram/data/canonical.py`: add the validated turbine table to `CanonicalTables` without extracting `info.xlsx`.
- `src/baram/data/quality.py`: receipt rows for 17 turbines, group counts `(6,6,5)`, group capacities `(21.6,21.6,21.0 MW)`, hub 117 m, rotor/fleet values, and decimal-coordinate bounds.
- `tests/data/test_canonical.py`, `tests/data/test_quality.py`.

**Contract**

- open workbook bytes through `BytesIO`/`openpyxl` in read-only, data-only mode;
- forward-fill workbook group cells only after validating the group boundary rows;
- reject malformed DMS, duplicate turbine identity, missing group/fleet/capacity, count/capacity drift, or coordinates outside the supplied grid region;
- do not write or normalize the workbook itself.

**Receipt**: `artifacts/manifests/v2_turbines.json` plus reviewer-facing `reports/v2_data_contract.json`.

### T2 — Group-aware supplied-NWP features (`M2`)

**Add**

- `src/baram/features/spatial.py` with:
  - `haversine_km`;
  - `build_group_grid_weights`;
  - `aggregate_group_weather`;
  - `add_source_disagreement_features`.
- `configs/features/spatial_v2.yaml`: core supplied variable allowlists and feature-family definitions.
- `tests/features/test_spatial.py`.

**Modify**

- `src/baram/features/weather.py`: merge group-aware spatial rows while retaining the current global aggregation control.
- `src/baram/features/physics.py`: bounded supplied-level shear, 117 m hub speed, density, `rho × v³`, rotor swept-area/fleet proxies, and invalid-input flags.
- `src/baram/features/pipeline.py`: freeze `spatial_mode`, variable allowlist hash, grid-weight hash, and fold-fitted median lineage.
- `tests/features/test_weather.py`, `test_physics.py`, `test_pipeline.py`.

**Deterministic geometry**

- haversine distance in kilometers;
- inverse-distance weight `1 / max(distance_km, 0.1)^2`, normalized per turbine and averaged within group;
- nearest-grid feature with stable `grid_id` tie-break;
- vector components aggregated before speed/direction;
- source disagreement only for semantically aligned supplied variables;
- no external terrain, wakes, roughness, power curves, or observations.

**Bounded physics**

- supplied 80/100 m shear exponent `log(v100/v80)/log(100/80)` when both speeds exceed 0.1 m/s;
- exponent clipped to `[-0.2, 0.6]`, otherwise fallback `0.2` plus an explicit flag;
- final decision action support never exceeds the preceding-data bound defined in T6.

**Eight feature candidates**

1. `S0_GLOBAL_SELECTED` — current selected F1 control;
2. `S1_IDW_WIND`;
3. `S2_NEAREST_WIND`;
4. `S3_IDW_NEAREST_WIND`;
5. `S4_VECTOR_SPREAD`;
6. `S5_SOURCE_DISAGREEMENT`;
7. `S6_PHYSICS_FLEET`;
8. `S7_PROMOTED_COMBINATION` — union only of preceding families that individually clear component guardrails.

Every candidate uses the same frozen point-model control to isolate feature effect. A failed family is removed from S7.

**Receipt**: `reports/v2_spatial_feature_ablation.json`, feature parquet hashes under `artifacts/backtests/v2-spatial/baram-v2-20260801-01/`.

### T3 — Robust deterministic point champion (`M3`)

**Add**

- `configs/models/point_v2.yaml` with ten named candidate rows and no Cartesian expansion.
- `tests/models/test_point_v2.py`.

**Modify**

- `src/baram/models/lightgbm.py`: allowlist objective-specific Huber parameters while preserving deterministic L1 behavior and inner issuance-batch stopping.
- `src/baram/models/oof.py`: accept named v2 point specifications without changing keyed point-output schema.
- `src/baram/workflows.py`: `backtest --stage point-v2` full-fold runner and three-seed finalist stability.
- existing LightGBM/OOF/integration tests.

**Ten rows maximum**

- frozen shared-L1 control;
- group-specific L1;
- shared and group-specific Huber (`alpha=0.90`);
- four predeclared LightGBM leaf/min-child/column variants around the frozen control;
- shared and group-specific CatBoost MAE challengers only if the optional dependency is already installed and their first honest OOF residual correlation passes `<0.995`.

CatBoost rows are skipped rather than replaced when unavailable or gated out. No XGBoost expansion and no dependency installation occur.

**Selection**

- rank by three-seed mean Total after all guardrails;
- require seed range `<=0.0035`;
- keep one point champion and at most one residual-diverse challenger.

**Receipt**: `reports/v2_point_models.json` and hashed OOF parquets.

### T4 — Predictive distribution bundles (`M4`)

**Add**

- `src/baram/models/quantile.py`: deterministic LightGBM quantile bundles and monotone repair.
- `src/baram/models/qrf.py`: dependency-free Meinshausen-style leaf-cooccurrence sample weights and weighted quantiles on scikit-learn forests.
- `src/baram/models/distribution_oof.py`: keyed long-form distribution OOF contract.
- `src/baram/evaluation/probabilistic.py`: pinball, coverage, width, crossing, group/lead/season diagnostics.
- `configs/models/distribution_v2.yaml`.
- `tests/models/test_quantile.py`, `test_qrf.py`, `test_distribution_oof.py`.
- `tests/evaluation/test_probabilistic.py`.

**Output schema**

`forecast_id`, `forecast_kst_dtm`, `group_id`, `fold_id`, `model_id`, `quantile`, `prediction_kwh`, parent feature/model hashes. Exactly seven monotone quantiles per key.

**Eight bundles maximum**

- one preceding-fold empirical-residual distribution control;
- four LightGBM quantile bundles across shared/group-specific and two frozen complexity choices;
- two QRF bundles (`128` trees, minimum leaf `40` or `80`, maximum six workers);
- one chronological conformalized-quantile diagnostic adjustment of the best preceding distribution.

No exchangeability guarantee is claimed. Conformal adjustment is evaluated chronologically and is retained only if it passes section 5.

**Receipt**: `reports/v2_distribution_models.json`, OOF distribution parquet/hash, coverage and pinball tables.

### T5 — Cross-fitted expected official-utility decisions (`M6`)

**Add**

- `src/baram/decisions/expected_utility.py` with:
  - `ExpectedUtilityPolicy` frozen contract;
  - `quantile_quadrature`;
  - `fit_expected_utility_policy`;
  - `apply_expected_utility_policy`;
  - `cross_fit_expected_utility`.
- `configs/decisions/utility_v2.yaml`.
- `tests/decisions/test_expected_utility.py`.

**Modify**

- `src/baram/contracts/types.py`: distribution/policy lineage hashes and state hierarchy.
- `src/baram/decisions/calibrate.py`: preserve the current method as the coarse control; no expanded grid.
- `src/baram/evaluation/slices.py`: threshold-distance and policy-state diagnostics.
- `src/baram/workflows.py`: `backtest --stage decision-v2` and final frozen policy application.

**Deterministic expected utility**

- reconstruct a monotone conditional quantile curve;
- integrate on 19 fixed probability midpoints, not random Monte Carlo;
- candidate actions are all predicted quantiles plus median capacity offsets `(-0.04,-0.02,-0.01,-0.005,0,0.005,0.01,0.02,0.04)`;
- lower action bound is zero;
- upper bound is `min(1.10 × capacity, max(1.01 × capacity, preceding-label q99.9))`, learned only from preceding policy-training rows and frozen;
- expected NMAE and generation-weighted 4/3/0 settlement contributions use the unchanged official thresholds;
- validation labels never enter distribution fitting, action selection, or state support.

**Ten policies maximum**

1. q50 median control;
2. current coarse group scale/offset control;
3. global expected utility;
4. group hierarchy;
5. group + lead;
6. group + season;
7. group + wind regime;
8. group + lead + wind hierarchy;
9. threshold-proximity hierarchy;
10. one refined action grid around the best passing policy.

State cells below 500 preceding rows use the parent policy only. Supported children shrink toward the parent with weight `n / (n + 1000)`. Promotion requires all section-5 decision thresholds.

**Receipt**: `reports/v2_decision_layer.json`, policy JSON/hash, cross-fit point predictions, state support and threshold-migration tables.

### T6 — Residual-diverse constrained ensemble (`M5`)

**Modify**

- `src/baram/decisions/blend.py`:
  - add generic `fit_convex_blend`/`apply_convex_blend` for two or three parents;
  - keep `fit_two_model_blend` behavior and hashes backward-compatible;
  - deterministic simplex step `0.05` and simpler-parent tie-break.
- `src/baram/experiments/promotion.py`: require residual diversity plus the v2 material/component/fold gates.
- `src/baram/workflows.py`: preceding-fold blend fit and later-fold evaluation.
- `tests/decisions/test_blend.py`, `tests/experiments/test_promotion.py`.

**Candidate parents**

- point champion;
- expected-utility distribution champion;
- at most one admitted residual-diverse point/QRF challenger.

At most six predeclared parent sets are evaluated. No same-fold weights, public-score weights, unconstrained stacking, or more than three parents.

**Receipt**: `reports/v2_ensemble.json`, residual correlation matrix, group weights, parent/prediction hashes.

### T7 — Self-evaluation and promotion (`M8`)

**Add**

- `src/baram/evaluation/v2_report.py`.
- `configs/v2/report_slices.yaml`.
- `tests/evaluation/test_v2_report.py`.

**Modify**

- `src/baram/evaluation/failure_slices.py`, `slices.py`: group, fold, month, season, target hour, lead, wind, NWP missingness, generation band, ramp, settlement tier, and 6%/8% boundary-migration slices.
- `src/baram/experiments/promotion.py`: pure `decide_v2_*` functions parameterized only by the frozen threshold config.
- `src/baram/workflows.py`: one promotion ledger linking every candidate to its control and failure evidence.

**Required report**

- exact pooled/fold/group Total, `1-NMAE`, FICR and deltas;
- seed range, feature/model/distribution/policy/ensemble lineage;
- quantile diagnostics and support;
- predicted-value support and cap hits;
- settlement tier migrations versus parent;
- failure slices with row counts;
- configuration slots and artifact bytes;
- narrow PASS/FAIL reason for every gate.

**Receipt**: `reports/v2_self_evaluation.json` and `reports/v2_promotion_ledger.jsonl`.

### T8 — Candidate freeze, final fit, local CSV, and reproduction (`M9`)

**Modify**

- `src/baram/workflows.py`: v2 candidate freeze, final fit, distribution/utility inference, and reproduction routing.
- `src/baram/inference/final.py`: load frozen spatial/distribution/decision/ensemble lineage.
- `src/baram/submission/build.py`, `validate.py`: preserve exact sample keys/encoding and record new parent hashes.
- `tests/inference/test_final.py`, `tests/submission/test_build.py`, `test_validate.py`.
- `tests/integration/test_tiny_pipeline.py`: include spatial feature, quantile, expected-utility, and candidate hashes in two fresh processes.

**Freeze rules**

- candidate choices and all policy parameters freeze before full 2022-2024 final fitting;
- no 2024 metric call exists on the v2 final-fit path;
- final training may use competition-supplied 2024 labels only after freeze and only for fitting;
- two complete final runs must reproduce model/policy/prediction/CSV hashes;
- output CSV remains local and untracked;
- no upload, browser, account, or team action is callable from this pipeline.

**Receipts**

- `reports/v2_candidate_freeze.json`;
- `reports/v2_final_model.json`;
- `reports/v2_reproduction.json`;
- local untracked `artifacts/submissions/<candidate_id>.csv` and its hash receipt.

## 7. Test and command sequence after approval

Commands are sequential in the single root session. The approved run ID is `baram-v2-20260801-01` and becomes immutable at the first stage write.

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m ruff check src tests
./.venv/bin/python -m baram.cli audit --config configs/default.yaml
./.venv/bin/python -m baram.cli v2-preflight --config configs/default.yaml --run-id baram-v2-20260801-01
./.venv/bin/python -m baram.cli prepare --config configs/default.yaml --run-id baram-v2-20260801-01
./.venv/bin/python -m baram.cli split-build --config configs/default.yaml --run-id baram-v2-20260801-01
./.venv/bin/python -m baram.cli backtest --config configs/default.yaml --run-id baram-v2-20260801-01 --stage spatial-v2
./.venv/bin/python -m baram.cli backtest --config configs/default.yaml --run-id baram-v2-20260801-01 --stage point-v2
./.venv/bin/python -m baram.cli backtest --config configs/default.yaml --run-id baram-v2-20260801-01 --stage distribution-v2
./.venv/bin/python -m baram.cli backtest --config configs/default.yaml --run-id baram-v2-20260801-01 --stage decision-v2
./.venv/bin/python -m baram.cli backtest --config configs/default.yaml --run-id baram-v2-20260801-01 --stage ensemble-v2
./.venv/bin/python -m baram.cli select --config configs/default.yaml --run-id baram-v2-20260801-01
./.venv/bin/python -m baram.cli fit-final --config configs/default.yaml --run-id baram-v2-20260801-01 --champion-receipt reports/v2_candidate_freeze.json
./.venv/bin/python -m baram.cli build-submission --config configs/default.yaml --run-id baram-v2-20260801-01 --model-receipt reports/v2_final_model.json
./.venv/bin/python -m baram.cli reproduce --config configs/default.yaml --candidate-receipt reports/v2_candidate_freeze.json
```

The `lockbox` command is intentionally absent. Full tests run before the first training stage and again after each source-edit batch; targeted tests run immediately after their owning task.

## 8. Rollback, failure, and stop rules

### Hard block

- source/notebook hash mismatch;
- archive/schema/availability/scorer parity failure;
- worker count above six;
- any 2024 development/evaluation read or lock-file mutation;
- external data/weight/API/remote-compute path;
- non-finite/duplicate/misaligned distribution or point keys;
- artifact budget exceeded before the next write;
- reproduction mismatch after one clean, identical retry.

On hard block: stop the pipeline, retain the failing receipt, do not delete prior artifacts, and request a revised authority or correction.

### Module fallback

- spatial failure -> current selected global feature control;
- point failure -> frozen current LightGBM workhorse;
- distribution failure -> point champion plus current coarse calibration;
- expected-utility failure -> q50 or accepted coarse calibration;
- ensemble failure -> single decision champion;
- conditional CatBoost unavailable/fails diversity -> skip, no installation;
- TFT/NGBoost -> remain deferred, never auto-activate.

### Loop stop

Stop a module when its slot ceiling is reached or every untried candidate is ruled out by a parent gate. Stop the full local loop when:

1. G0-G6 and two-run reproduction pass for one candidate; or
2. all 46 allowed slots are exhausted; or
3. a hard block occurs.

The first outcome is `PASS_LOCAL_V2_REPRODUCED_CANDIDATE`, not top-20 proof. Actual score/rank confirmation requires a later, separately authorized Dacon result. This plan never uploads.

## 9. Session and workspace discipline

- one root process lane only;
- no subagents, recursive delegation, worktrees, background training, or orphaned terminals;
- after each stage, ensure the CLI has exited, record artifact/runtime counts, release in-memory model references, and verify no stage-owned process remains;
- preserve unrelated untracked/dirty files;
- never use `git add .`; any later staging uses explicit reviewer-facing paths only;
- no destructive cleanup; generated artifacts are removed only with a separate exact target authorization.

## 10. Approval ledger and next boundary

| Artifact | Status | Authority |
| --- | --- | --- |
| `SK@v3` | Approved | bounded research only |
| `DS@v3` | Approved | prepare `IP@v2` only |
| `IP@v2` | Approved and active | exact user message `IP@v2 승인`, received 2026-08-02 |

Approval of this plan authorizes only the local implementation and bounded execution enumerated above. It does not authorize Dacon upload/account mutation, external data/weights, remote inference/compute/Git, system dependency changes, TFT/NGBoost/deep execution, a new lockbox score, or deletion of existing artifacts.

## Approval record

- Exact user message: `IP@v2 승인`
- Received: 2026-08-02 (Asia/Seoul)
- Authorized next action: execute T0-T8 sequentially under this plan's budgets, gates, and exclusions.
