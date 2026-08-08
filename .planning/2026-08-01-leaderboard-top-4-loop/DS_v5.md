# DS@v5 — Synthesized design for hierarchical pipeline discovery

## Metadata

| Field | Value |
|---|---|
| Status | `SYNTHESIZED_DESIGN_PENDING_APPROVAL(DS@v5)` |
| Authorizing skeleton | `.planning/2026-08-01-leaderboard-top-4-loop/SK_v5.md` SHA-256 `c2c813475ab5f0a741e6adbceab196d5b38447f2aeb02f9716a8f5be8973c820` |
| Approval receipt | `reports/sk_v5_approval_receipt.json` SHA-256 `e827b324a23913346598d274f2ded496ffc3134243de5636033ee6a3ac387173` |
| Wave-A manifest | `reports/sk_v5_wave_a_research_manifest.json` SHA-256 `4690826f4f2f629220c085ffc87883c1ec2fa8990f7f92430483ff4f975ae312` |
| Frozen foundation map | `research/nodes/sk5_foundation_map.json` SHA-256 `393ee74bf53251037547fc52bc92c554a4aa207e2288f194766d836675512b35` |
| Exact approval phrase | `DS@v5 승인` |
| Next gate after approval | `IP@v3` (implementation plan; still no fit or score until `IP@v3` is separately approved) |

### Wave-A inputs adjudicated by the root

| Lane | Artifact | SHA-256 | Root adjudication | Disposition |
|---|---|---|---|---|
| V5-ENGINE | `research/lanes/v5_engine_audit.md` | `7d16ba4def18ef21b879bd9cac3901157c0739261e696839a1d0e69ac9aed6b7` | `reports/sk_v5_wave_a_engine_adjudication.json` `af7935fcd1b3116f8d09c1afcfca28cb6ba3a9e6e3b1f0f727ae7023804f4201` | ACCEPT_WITH_SCOPE_DOWNGRADE |
| V5-FOUNDATION | `research/lanes/v5_foundation_deficit.md` | `b94ddbd2d61a4f03958bd917b121cce288ad1f68807b46fecdc16dff0af77c35` | `reports/sk_v5_wave_a_foundation_adjudication.json` `11dc402ae0dfbcd7bd254017ee1f636bd22c4a2870db1113a17da5f38b66ba7f` | ACCEPT_DIRECT_CLAIMS |
| V5-DISCOVERY-SCIENCE | `research/lanes/v5_discovery_science.md` | `67aa1b3c0bafc17cd1f7999f534b33f6ea9ac4cfac915c5430434a192e7b5657` | `reports/sk_v5_wave_a_discovery_adjudication.json` `3bf28ec448b38336fc4024b58d2385a2810e4bcfc89d086783fd8fa0442fc2b6` | ACCEPT_WITH_SCOPE_DOWNGRADE |
| V5-VALIDATION | `research/lanes/v5_validation_research.md` | `64241ec77181d7647c837d91fc14c03eaa45f6cb1732f1981d77e377e58578e5` | `reports/sk_v5_wave_a_validation_adjudication.json` `f92b80a530cd43fe7475268f7725da8bf009ab2530b26d8e619fccd4b487888c` | ACCEPT_WITH_SCOPE_DOWNGRADE |

The root independently re-read every design-critical local line and fetched every design-critical primary source; adjudication receipts record the verbatim checks. No lane conclusion is inherited from narrative alone.

---

## 1. Problem statement this design must solve

1. The current engine is stage-linear: `depth` is the S1–S11 ordinal, one mutable record exists per stage, only `directly_supported` findings expand, results update telemetry without creating descendants, re-entry is contradiction-only and destructive, one score can deliver, and below-target exhaustion can terminate.
2. The foundation is partial: S1 purpose and S4 formula are freezable, while S2 time-axis semantics and S3 diagnostic lineage are not.
3. The evaluation surface is exposed: no fresh holdout exists, so a strict replay is necessary but cannot be called confirmation.
4. The literature supports scientific-lineage mechanics and warns against automatic transfer, greedy scalar search, subjective ranking, and uncalibrated cheap screens.

DS@v5 therefore designs an **exposure-aware, append-only, node-depth discovery engine** whose experiments are governed by an honest evaluation hierarchy, and whose first work is repairing the S2/S3 foundation rather than proposing a model.

---

## 2. Architectural decision A0 (freeze first)

**The EventStore SQLite transaction is the sole authoritative node/lineage/lifecycle write path. `SpecRegistry` becomes a query facade plus a derived deterministic JSON export.**

Rationale (`directly_supported`): event append and registry JSON replacement are separately atomic today, so a crash can leave `CLOSED` committed while the registry still exposes a live candidate, and no transaction can create a parent closure plus its result children together.

Constraints:
- Existing `events` rows, envelope bytes, and chain hashes stay byte-identical; the chain verifier is unchanged for v1 bodies.
- New tables are additive; `PRAGMA user_version` (currently `0`) drives an idempotent migration.
- The live registry file is `schema_version` **1** with **47** specs while `SpecRegistry.save()` emits **2**; the importer must accept both, preserve every raw field in `legacy_payload`, and never synthesize ancestry.

---

## 3. Node contract v3

```text
identity:      schema_version=3, node_id, contract_sha256, node_type, origin,
               stage_visit_id, stage_id, stage_depth, node_depth, subcapability
lineage:       parent_ids[], parent_edges[{parent_id, relation, causal_event_sha256?}]
               relations: DERIVED_FROM | RESULT_RESIDUAL | ADVANCES_STAGE |
                          REENTERS_UPSTREAM | REVISES | REPRODUCES | REVIVES
deficit:       local_deficit_id, diagnostic_artifact{path,sha256},
               research_claim_ids[], basis_local_evidence_ids[]
hypothesis:    migration_contract, mechanism, falsifier, inconclusive_conditions,
               hypothesis_status = PROVISIONAL | ADMISSIBLE | REJECTED
experiment:    treatment, control, degrees_of_freedom, candidate_family,
               trial_budget, estimated_compute, preconditions, reference, arguments,
               score_bearing, evaluation_tier, governance_checks
evaluation:    metric, min_effect, component_gates, stability_gate, multiplicity_gate,
               row_alignment_keys, resampling_unit, block_rule, assumption_checklist
exposure:      surface_id, surface_rows_hash, surface_exposure_status, exposure_ledger_ids,
               primary_estimand, comparison_family_id, historical_unknown_trials,
               independent_confirmation_count
closure:       child_generation_rule, closure_key, closure_premise, revival_premise
projection:    status, verdict, disposition, evidence_status, maturity, event hashes
```

Invariants:
- `node_id` is a deterministic hash of the canonical immutable contract; `node_depth = 0` for a root and `1 + max(parent.node_depth)` otherwise; forged depths, missing parents, cycles, and `parent_ids`/`parent_edges` mismatches fail closed.
- `stage_depth` is the S1–S11 ordinal and is independent of `node_depth`. Upstream re-entry lowers `stage_depth` and raises `node_depth`.
- Lifecycle is projected from committed events, never edited in JSON.
- `surface_exposure_status` defaults to `UNKNOWN_NOT_FRESH`, never `FRESH`.

### Node kinds

`FOUNDATION_SNAPSHOT`, `STAGE_DEFICIT`, `RESEARCH`, `MIGRATION_HYPOTHESIS`, `CONTROLLED_EXPERIMENT`, `ADJUDICATION`, `RESIDUAL_DEFICIT`, `APPROPRIATENESS_AUDIT`.

### Evidence-to-hypothesis rule (unchanged meanings)

`near_match_only` may navigate to a `PROVISIONAL` hypothesis and can never be dispatched. Admission to `ADMISSIBLE` requires a directly supported source mechanism **or** a directly supported local diagnostic, plus a complete scope-gap matrix (`population, geography, horizon, issue_time, inputs, target, metric, resolution, topology, compute, licence`), exact local mapping, DOF/budget, falsifier, and inconclusive conditions. No tag is ever upgraded by admission.

---

## 4. Event contract v2 (additive)

| Event | Required additions | Transaction rule |
|---|---|---|
| `NODE_DECLARED` | canonical `node_contract`, `node_contract_sha256`, `lineage_complete` | node rows + parent edges + event in one commit |
| `PREDECLARED` v2 | v1 body plus `node_contract_sha256`, `attempt_id`, `child_generation_rule_sha256`, evaluation-tier and exposure fields | event + node projection in one commit |
| `STARTED` | `predeclared_event_sha256`, `attempt_id`, `runner_reference`, `runner_contract_sha256` | committed before dispatch; `STARTED` without `CLOSED` resumes as recovery pause, never blind rerun |
| `CLOSED` v2 | v1 body plus tri-state result, `residual_deficit`, `selected_disposition`, `spawned_child_ids`, `evidence_status`, `reproduction_key`/`output_digest` when applicable | closure + parent lifecycle + every child declared in one commit; `spawned_child_ids` must equal inserted child rows |
| `NODE_INVALIDATED` | prior closure hash, causal evidence hash, reason, replacement child id | old rows never cleared |
| `WORKFLOW_PAUSED` | reason enum, resume requirements, open frontier, authority/budget evidence | typed, resumable, non-success |
| `SUCCESS_DECLARED` | two distinct qualifying `CLOSED` hashes, one reproduction key, output identity/tolerance, pipeline-lineage digest | only exit |
| `REGISTRY_MIGRATED` | old schema/hash, imported/conflicted counts, migration-code hash, projection digest | appended only after a fully committed import |

v1 bodies keep v1 validation and canonical serialization.

---

## 5. Transition machine

```text
STAGE_DEFICIT → RESEARCH → {MIGRATION_HYPOTHESIS(PROVISIONAL) | RESIDUAL_DEFICIT | APPROPRIATENESS_AUDIT}
MIGRATION_HYPOTHESIS(ADMISSIBLE) → CONTROLLED_EXPERIMENT → ADJUDICATION (atomic close_and_spawn)
ADJUDICATION(SUPPORTED|REFUTED|INCONCLUSIVE) → RESIDUAL_DEFICIT → APPROPRIATENESS_AUDIT
APPROPRIATENESS_AUDIT → exactly one of DEEPEN | BRANCH | ADVANCE | REENTER | FAIL_CLOSED_NODE | RESEARCH_PAUSE
qualifying reproduction #1 → REPRODUCES child ; qualifying reproduction #2 → SUCCESS_DECLARED
```

Re-entry reasons: `PREMISE_CONTRADICTION`, `LEAKAGE`, `DATA_QUALITY`, `FEATURE_INSUFFICIENCY`, `REGIME_COLLAPSE`, `CALIBRATION_FAILURE`, `VALIDATION_INSTABILITY`. Prior stage visits are immutable; affected snapshots become `STALE`.

Routing order: mandatory causal/re-entry child → descendant of the active path → greatest `node_depth` → declared priority → stable `node_id`. Parent completion, typed prerequisites, trial budget, and stage-visit validity are hard gates. Unknown ontology subcapabilities fail closed.

---

## 6. Stage maturity replaces boolean completion

`M0 INVENTORIED → M1 FOUNDATION → M2 RESEARCHED → M3 MIGRATION_READY → M4 EXPERIMENTED → M5 ADJUDICATED → M6 REPRODUCED_RESIDUALIZED`.

A stage snapshot reports the maturity distribution of its material subcapabilities plus a deficit artifact, migration matrix, adjudication, residual child, and appropriateness disposition. One `M6` path may not hide `M0`/`M1` material deficits. `explicit_unknown` alone can never complete a stage.

**Closure key** — never close a family from one node:

```text
(mechanism, treatment, data_snapshot, chronology/fit_surface,
 validation_surface, policy, metric/components, alignment_keys, budget)
```

Every closure states what was not tested, viable siblings, residual uncertainty, and a frozen revival predicate. Revival requires a predicate recorded **before** any rescued result is viewed.

---

## 7. Evaluation hierarchy (exposure-aware)

| Tier | Surface | Allowed language | Gate |
|---|---|---|---|
| T0 `EXPLORATORY_DIAGNOSTIC` | exposed history, no fit or a cheap fit | mechanism, deficit, magnitude sketch | never "significant"/"validated" |
| T1 `BOUNDED_SCREEN` | inner prequential origins only | ranking/pruning within a frozen candidate cap | screen score is optimization information |
| T2 confirmatory-form comparison | full strict prequential, exact Total | `RETROSPECTIVE_CORROBORATION` or, for a complete recorded family, `RETROSPECTIVE_SELECTION_ADJUSTED_CORROBORATION` | `CONFIRMATION_PENDING` always attached |
| T3 `FRESH_CONFIRMATION` | genuinely unexposed chronological outcomes | confirmation | unavailable today; `independent_confirmation_count = 0` |

Mandatory rules:
- Nest the **entire** selection procedure (preprocessing → features → model → hyperparameters → calibration → blend → action policy) inside past-only inner origins; the evaluated object is the selector, not a hindsight winner.
- The indivisible comparison atom is the 72-cell issuance day; resample whole days and recompute exact `Total`/components inside every draw.
- Dependent inference is assumption-gated: if the differential fails stability/mixing diagnostics or block sensitivity disagrees, the result becomes descriptive-only or `ASSUMPTION_FAILED`, never a favorable-block selection.
- RC/MCS apply only to one aligned, complete, recorded family; `historical_unknown_trials=true` restricts every claim to that recorded family.
- Alpha-investing is ineligible unless each incoming p-value is conditionally valid; it controls mFDR, not FWER.
- Two deterministic reproductions prove reproducibility, not two independent confirmations.

A claim linter blocks the words *fresh*, *independent holdout*, *unbiased*, *best*, and *future superiority* whenever the corresponding gate fails.

---

## 8. Acquisition policy

Deterministic lexicographic queue: confirmation obligation → causal leverage on the frozen deficit list → expected information per cost → diversity/similarity cluster → age. Ties are logged. A small fixed exploration reserve prevents incumbent lock-in.

Blocked pending local calibration: Hyperband-style brackets, BOHB-style learned acquisition, freeze-thaw information-gain revival, tournament/Elo promotion, and greedy best-score child selection. Cheap screens may **park**, never permanently falsify, a family until local rank concordance and false-prune behavior are measured.

---

## 9. Wave-B entry: the first work is foundation repair, not a model

Ordered by the frozen deficit list (`research/nodes/sk5_foundation_map.json`):

| Order | Node | Deficit | Tier | Output | Fit? |
|---|---|---|---|---|---|
| 1 | `N29_ISSUANCE_CUBE_KEYS` | FD1 | T0 | source × reference × available_at × valid_time × variable × level × grid key/schema audit, uniqueness, lead equation, cross-source join coverage | none |
| 2 | `N30_ACTIVE_LINEAGE_GRAPH` | FD6 | T0 | builder→config→manifest prefix invocation graph and exact set differences | none |
| 3 | `N31_MISSINGNESS_STATE` | FD4 | T0 | presence/null-mask coverage cube and deterministic fallback transition table | none |
| 4 | `N32_LABEL_SUPPORT_CUBE` | FD2 | T0 | group × month × hour/lead presence and official-eligibility counts, values suppressed | none |
| 5 | `N33_DIAGNOSTIC_INDEX` | FD3 | T0 | score-free diagnostic registry with evidence class and contamination flags | none |
| 6 | `N34_COMPARISON_RECEIPT_SCHEMA` | FD5 | T0 | contract test forcing component/group/lead/regime/boundary/tier/alignment/policy fields | none |
| 7 | `N35_EXPOSURE_LEDGER` | FD7 | T0 | append-only candidate/choice/family/freshness ledger | none |

Only after these does Wave-B research open S5/S6/S7 hypotheses. The next score-bearing comparison index remains **5**.

---

## 10. Test matrix (pure contract tests, `tmp_path`, synthetic envelopes)

33 named tests across `tests/loop/test_registry.py`, `test_research_contract.py`, `test_stages.py`, `test_router.py`, `test_events.py`, `test_evaluation_contract.py`, `test_graph_contract.py`, plus 15 evaluation gates `V5VT-01…15`, plus the retained governance regressions (predeclaration, chain tamper, receipt hash, comparison monotonicity, chronology/alignment, lockbox and Dacon prohibition, runner fail-closed). Only the intentionally superseded single-success and exhaustion assertions change. Required command after implementation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/loop -q
```

---

## 11. Explicitly not authorized by DS@v5

Source edits, model fits, score-bearing runs, external data-body acquisition, Dacon/account actions, remote compute, any 2024 read, and Wave-B execution. Those require `IP@v3`.

**Amended 2026-08-09.** Dependency installation is no longer prohibited: the user authorised installing any package or framework into the project `.venv` provided the official rules hold (`inputs/rules/official_rules_2026-08-09.md` sha256 `6dcececbfb33df6761c87220ea8ce15d875d6b7afc9d90c5afe5dd827ad54ee5`). Consequently (a) `BLOCKED_DEPENDENCY` is retired as a rejection reason, (b) the pretrained-weight axis reopens for weights released on or before 2026-07-05 under a licence permitting use, modification, distribution, redistribution and commercial use, (c) external **data** is governed by per-row basis-time availability rather than any blanket year cutoff, and (d) remote API model inference stays forbidden, so weights must run locally.

---

## 12. Open questions this design deliberately leaves to `IP@v3`

1. Reproduction identity: byte-identical output digest versus a stated numeric tolerance.
2. Multi-parent `node_depth` rule confirmation (`1 + max`).
3. Practical minimum effect, component safety gates, block-length grid, and trial budgets.
4. Migration dry-run thresholds for legacy nodes whose stage/parent/event mapping cannot be verified.
5. Whether the legacy `build_graph` is adapted or retired.

---

## 13. Approval request

Approve this design with the exact phrase **`DS@v5 승인`**. Approval authorizes writing `IP@v3` only; implementation, fits, and scores remain blocked until `IP@v3` is separately approved.
