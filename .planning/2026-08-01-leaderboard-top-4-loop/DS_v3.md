# Design Specification — BARAM 2026 Top-20 Forecasting Pipeline

## Metadata

- Design ID: `DS`
- Version: `v3`
- Status: `DESIGN_APPROVED_FOR_IP(DS@v3)`
- Date: 2026-08-01 (Asia/Seoul)
- Approval message: `DS@v3 승인`
- Approved research base: `SK@v3`
- Supersedes: target, promotion, architecture-priority, and stopping fields of `DS@v2`
- Research report: `reports/top20_module_research.md`
- Nearest authority: `/Users/um-yunsang/BARAM2026/AGENTS.md`

## 1. Outcome contract

Design an official-data-only, leakage-safe, reproducible forecasting loop whose external operating guide is safely above the saved top-20 finals cutoff. The pipeline must improve both deterministic accuracy and settlement-value capture while keeping the consumed 2024 lockbox closed.

This design does not claim that an offline score proves top-20 placement. Actual Dacon placement is measurable only by a separately authorized competition result; Dacon upload remains outside scope.

## 2. Target contract

| Level | Total | 1-NMAE | FICR | Role |
| --- | ---: | ---: | ---: | --- |
| Hard snapshot minimum | 0.65971 | 0.87991 | 0.43952 | Rank-20 reference; not a safe operating target |
| Intermediate snapshot buffer | 0.66033 | 0.87740 | 0.44326 | Rank-15 reference; only 0.00062 total above rank 20 |
| Recommended operating guide | **0.66200** | **0.87500** | **0.44900** | Movement-safe balanced component guide |

The public target is a guide, not a local fold pass threshold. Local promotion uses paired chronological improvement, component/group guardrails, and reproducibility; no unsupported mapping from local scores to leaderboard rank will be fitted from two submissions.

## 3. Non-negotiable boundaries

- Immutable inputs: `open.zip` and `baseline.ipynb` with frozen hashes.
- Modeling data: competition-supplied data only.
- Forbidden: external weather/terrain/power curves, test-period observations, pretrained weights, remote inference/API models, remote compute, remote Git, and Dacon account mutation/upload.
- Runtime for later approved work: project-local Python 3.12, at most six model workers.
- 2024 lockbox: already consumed once; never reopen, rescore, or use for selection.
- Execution: one root only; no subagents, worktrees, or background sessions.
- Approval scope: `DS@v3` approval authorizes only preparation of `IP@v2`, not implementation or training.

## 4. Recommended architecture

```text
immutable competition archive
  -> M0 contract/canonical keys
  -> M1 quality and availability masks
  -> M2 turbine/group-aware supplied-NWP + physical features
  -> M3 deterministic LightGBM workhorse
  -> M4 cross-fitted quantiles / QRF distribution benchmark
  -> M6 expected official-utility point decision
  -> M5 residual-diverse constrained ensemble (only if additive)
  -> M8 exact chronological self-evaluation
  -> M9 reproducible candidate/receipt orchestration
  -> local candidate only
```

M7 TFT is a conditional challenger outside the primary route until a later value/compute/dependency gate is approved.

## 5. Module contracts

### M0 — Contract and reproducibility engineer

- Owns hashes, schema, timestamp/group keys, issue/valid-time availability, deterministic seeds, worker cap, and artifact identity.
- Must reproduce current prepare/split/scorer contracts before new experiments.
- Blocks all downstream work on row/key drift, future availability, or scorer mismatch.

### M1 — Data-quality and EDA engineer

- Maps missingness by source/grid/variable/time, label availability, capacity support, season, group, fleet, lead, and wind regime.
- Treats group-3 missing labels as unavailable, never zero.
- Keeps SCADA quarantined because test SCADA is absent.
- Produces only diagnostics and deterministic imputation specifications.

### M2 — Feature engineer

- Uses supplied turbine coordinates and LDAPS/GFS grid coordinates for group/turbine distance-weighted and nearest-grid features.
- Preserves wind vectors before deriving speed/direction; adds spatial dispersion/gradient and LDAPS-GFS disagreement.
- Uses supplied 117 m hub, rotor, capacity, pressure/temperature/humidity variables for bounded shear, density, and kinetic-power proxies.
- Adds issue/lead/valid-time and source-missingness signals.
- Does not use external terrain, wake models, manufacturer curves, or target/SCADA lags.

### M3 — Deterministic AI/ML engineer

- Primary: capacity-normalized LightGBM with robust point objectives.
- Benchmarks: current RF and existing tree adapters under identical folds.
- CatBoost remains a challenger only if OOF residual diversity and group robustness justify it.
- Compares shared/group-indicator, shared-plus-calibration, and group-specific forms.

### M4 — Probabilistic AI/ML engineer

- Primary: LightGBM quantile family using identical chronology.
- Benchmark: Quantile Regression Forests.
- Conditional: NGBoost only if its distributional diagnostics and decision utility justify complexity.
- Repairs quantile crossing deterministically and evaluates held-out pinball/coverage by group/lead/season/wind.

### M5 — Ensemble engineer

- Uses only complete preceding-fold OOF predictions.
- Requires measurable residual diversity before a model enters the blend.
- Learns constrained weights on earlier data and evaluates on later folds.
- Caps the promoted ensemble at three parents; rejects public-score weighting and unlimited stacking.

### M6 — FICR decision and calibration engineer

- Uses cross-fitted predictive distributions to choose a bounded per-row point maximizing expected official composite utility.
- Conditions hierarchically on group, lead, season, wind regime, and 6%/8% threshold proximity, with shrinkage for sparse cells.
- Benchmarks median point, current scale/offset calibration, and expected-utility policy.
- Must improve total/FICR without breaching the predeclared NMAE or group guardrails.

### M7 — Temporal/deep challenger engineer

- Candidate: TFT using the 24-hour known-future NWP sequence and static group/fleet context.
- Activation requires a later classical residual-value finding plus explicit dependency/compute approval.
- Retires if a predeclared material gain is not demonstrated; no deep dependency is installed under this design.

### M8 — Self-evaluation engineer

- Runs the unmodified official metric and paired fold comparisons.
- Reports total, `1-NMAE`, FICR, group, month, lead, wind regime, predicted-support, and settlement-band slices.
- Audits calibration leakage, quantile coverage, residual diversity, failure concentration, and reproducibility.
- Labels every PASS by its exact surface; never equates local PASS with finals qualification.

### M9 — Orchestration engineer

- Owns stage budgets, deterministic cache/config keys, candidate hashes, receipts, stop conditions, and cleanup.
- Prevents lockbox reads and Dacon actions.
- Requires two identical final local reproductions before a candidate can be handed off.

## 6. Validation and leakage contract

1. Preserve 24-hour NWP issuance batches.
2. Use expanding/prequential development folds within the existing 2023 development horizon.
3. Fit every transform, imputer, model, quantile repair, calibration rule, utility policy, and ensemble weight using training or strictly preceding OOF data.
4. Keep 2024 closed; prior receipts are diagnostic history only.
5. Run the exact official scorer over complete held-out folds and retain group/regime components.
6. Any feature must be identically available from the supplied 2025 forecast package at decision time.

## 7. Alternatives and decision

| Alternative | Tradeoff | Disposition |
| --- | --- | --- |
| A. Current deterministic pipeline + spatial features + coarse calibration | Fastest and lowest risk, but weak threshold-uncertainty handling | Mandatory benchmark, not the target architecture |
| B. Spatial classical + quantile/QRF distribution + expected-utility decision | Best direct fit to score structure with manageable data/compute; requires strict cross-fitting | **Selected primary architecture** |
| C. TFT-first temporal stack | Relevant 24-hour sequence structure, but high cost and uncertain incremental value | Conditional later challenger |

## 8. Promotion gates

| Gate | Pass condition | Failure action |
| --- | --- | --- |
| G0 Contract | Exact schema/key/hash/scorer parity and no availability violation | Block pipeline |
| G1 Features | Stable positive paired total effect with no material worst-fold/group collapse | Drop family or simplify |
| G2 Deterministic | Stable total/NMAE improvement and valid output support | Retain prior workhorse |
| G3 Distribution | Better held-out distribution diagnostics and positive downstream decision value | Retain point model only |
| G4 Decision | Total/FICR lift with NMAE and group guardrails | Fall back to simpler calibration |
| G5 Ensemble | Improvement over champion on every later fold plus residual diversity | Use champion alone |
| G6 Reproduction | Two identical approved runs and candidate/hash parity | Do not hand off candidate |

The exact material-delta and run-variance thresholds must be specified in `IP@v2` before training. They will not be invented from the one consumed lockbox score.

## 9. Later experiment budget ceiling

Proposed maximum full-chronology candidates, subject to later IP approval:

- contract/parity: 2;
- spatial/physical feature families: 8;
- deterministic model/objective: 10;
- quantile/distribution: 8;
- decision policy: 10;
- ensemble: 6;
- final reproduction: 2.

Deep/TFT work is excluded from this ceiling until separately activated. Stages stop early when the evidence is sufficient or a gate fails.

## 10. Evidence and presentation package

Retain per-run config/hash/runtime, feature manifest, fold membership, exact metrics, component/group/regime tables, paired deltas, calibration/coverage diagnostics, residual-correlation matrix, decision-policy complexity, prediction support, failure cases, and two-run reproduction receipts. These artifacts serve both model governance and the official 100-point offline rubric.

## 11. Known limitations

- Saved leaderboard values are a dated public snapshot, not current/private/final standings.
- The saved HTML exposes the authenticated best summary and submission count, not two per-submission histories.
- Local folds cannot be calibrated reliably to public score from two submissions.
- Group 3 has shorter history and lower prior FICR, so hierarchical sharing is necessary and its risk remains high.
- Expected-utility decisions depend on honest distribution calibration; leakage can manufacture gains.
- TFT benefit is unknown.

## 12. Completion condition for this design phase

`DS@v3` was approved by the exact user message `DS@v3 승인` on 2026-08-01. The next allowed artifact is `IP@v2`, specifying implementation tasks, tests, budgets, receipts, rollback/stop rules, and distinct future approvals.

No code implementation, training, dependency change, Dacon upload/account mutation, external data/weights, remote compute, remote Git, or 2024 lockbox reuse is authorized by this document.

## Approval record

- Exact message: `DS@v3 승인`
- Status: `APPROVED`
- Authorized next action: prepare `IP@v2` only.
- Still excluded: code implementation, model training, dependency changes, Dacon upload/account mutation, external data/weights, remote compute/Git, and any reuse of the consumed 2024 lockbox.
