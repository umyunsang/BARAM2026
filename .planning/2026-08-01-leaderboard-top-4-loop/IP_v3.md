# IP@v3 — Implementation plan for the DS@v5 discovery engine

## Metadata

| Field | Value |
|---|---|
| Status | `IMPLEMENTATION_PLAN_PENDING_APPROVAL(IP@v3)` |
| Approved design | `.planning/2026-08-01-leaderboard-top-4-loop/DS_v5.md` SHA-256 `7d49fe2c02b8b4c9275e4ba6b73c8ad20cc6ac9de2377482a4902ce98c8cc38f` |
| Design approval receipt | `reports/ds_v5_approval_receipt.json` SHA-256 `bcc331389b3e4076516d0cf870045298328e4f77f6c19d2b6962e24c2fd623f7` |
| Skeleton | `SK_v5.md` SHA-256 `c2c813475ab5f0a741e6adbceab196d5b38447f2aeb02f9716a8f5be8973c820` |
| Frozen foundation map | `research/nodes/sk5_foundation_map.json` SHA-256 `393ee74bf53251037547fc52bc92c554a4aa207e2288f194766d836675512b35` |
| Exact approval phrase | `IP@v3 승인` |
| Baseline EventStore | `artifacts/registry/loop_events_s17.sqlite` SHA-256 `5881b48f39ad8e8a334f11dd30abee1f893547da34aed55dc1979f1e476f0e12`, sequence 158, comparison count 4 |
| Baseline registry | `artifacts/registry/loop_node_specs.json` SHA-256 `88402768b5b1276c953d44d1b7333e2c93c39e86421f82f7a930c509f307e190`, `schema_version` 1, 47 specs |

## What approval of IP@v3 authorizes

1. Root-owned source edits inside `src/baram/loop/`, `src/baram/evaluation/` (schema/receipt fields only), `tests/loop/`, and a new versioned structure-only ontology artifact.
1b. **Dependency installation into the project `.venv`** (authorised 2026-08-09). Each install is pinned, recorded in the node receipt, reproducible from submitted code, and never used for remote inference. `BLOCKED_DEPENDENCY` is no longer a valid rejection.
2. Execution of the pure contract test suite.
3. A **no-mutation** migration dry run, then the one-time migration of the live EventStore and registry.
4. Execution of the seven T0 no-fit foundation diagnostics `N29`–`N35`.
5. T1 bounded screens and, afterwards, T2 score-bearing strict prequential comparisons **only** through the implemented predeclare/close governance, resuming at comparison index **5**.

## What remains forbidden regardless of IP@v3

Any 2024 lockbox read; Dacon, account, team or upload actions; remote compute, **remote API model inference** or git push; reanalysis products; post-hoc corrected observations; test-period observations; `git add .`; rewriting or renumbering any existing EventStore row; and declaring success on anything other than two deterministic strict reproductions at Total >= 0.660000.

**Eligibility gates, corrected 2026-08-09 against `inputs/rules/official_rules_2026-08-09.md` sha256 `6dcececbfb33df6761c87220ea8ce15d875d6b7afc9d90c5afe5dd827ad54ee5`:**
- External **data**: no blanket year cutoff. Each datum must be created/published/finalised before that row's D-1 14:00 KST basis time, publicly accessible to anyone, licence/privacy/ToS compliant, and reproducible from the submission with source, collection method, collection time, usage period, variables, licence and preprocessing code recorded.
- Pretrained **weights**: released on or before 2026-07-05 under a licence permitting use, modification, distribution, redistribution and commercial use; loaded and executed locally only.
- Dependencies: installable and pre-approved; pin the version.

---

## Work packages

Each package lists files, acceptance criteria, and rollback. Packages run in order; WP1–WP11 touch no live store.

### WP0 — Baseline freeze and migration dry run (read-only)
- **Do:** record SHA-256 of every file to be touched; produce `reports/ip_v3_migration_dryrun.json` containing, for the live registry and EventStore, counts by schema version, node IDs with and without a verifiable `arguments.stage`, node IDs with and without a matching `PREDECLARED`/`CLOSED` event, duplicate IDs, cycles, and proposed defaults. Values of `outcome` are summarised by presence and note class only; no score is transcribed.
- **Accept:** the dry run runs against read-only copies and mutates nothing; every conflict is listed, never silently defaulted.
- **Rollback:** none needed; nothing is written outside `reports/`.

### WP1 — `events.py`: additive v2 schema and transactions
- **Do:** add `PRAGMA user_version` migration creating `node_specs` and `node_parents` with foreign keys; add event types `NODE_DECLARED`, `STARTED`, `NODE_INVALIDATED`, `WORKFLOW_PAUSED`, `SUCCESS_DECLARED`, `REGISTRY_MIGRATED`; add v2 body validators dispatched by `node_contract_version`; add `declare_batch`, `predeclare_node`, and `close_and_spawn`, each validating every row before a single commit.
- **Accept:** v1 envelope bytes, canonical serialization, chain hashes and the existing sequence are unchanged; `test_v1_event_chain_prefix_survives_v5_open_and_migration` passes; every failpoint test shows all-or-nothing.
- **Rollback:** the migration is additive only; dropping the new tables restores the prior schema without touching `events`.

### WP2 — `registry.py`: node contract v3 and derived facade
- **Do:** implement `NodeSpecV3` with the DS@v5 identity, lineage, deficit, hypothesis, experiment, evaluation, exposure and closure fields; derive `node_depth`; reject missing parents, cycles, forged depth and `parent_ids`/`parent_edges` mismatch; make lifecycle a projection of committed events; make `save()` a deterministic derived export; implement an importer that accepts `schema_version` **1 and 2**, preserves every raw field in `legacy_payload`, and sets `lineage_complete=false` rather than inventing ancestry; keep read aliases `id`, `arguments['stage']`, `candidate`, `retired`; replace mutable `invalidate` with an event-backed supersede that requires a replacement node ID.
- **Accept:** importing the live file loses no field; unresolvable lineage is honestly flagged; no closed node ID can be reused.
- **Rollback:** the original JSON is copied to `artifacts/registry/loop_node_specs.v1.backup.json` before any write.

### WP3 — `research.py`: hypothesis layer and node-scoped artifacts
- **Do:** keep `Finding` and the four tag meanings unchanged; add `MigrationHypothesis` with the scope-gap matrix, mechanism, treatment, control, DOF, budget, falsifier and inconclusive conditions; add `derive_provisional_hypotheses` (a `near_match_only` finding may produce only `PROVISIONAL`) and `admit_experiment` (requires a directly supported source mechanism or local diagnostic plus a complete contract); move request/findings paths to `research/nodes/{stage_visit_id}/{node_id}/`, reading the legacy stage path only as a hash-recorded fallback.
- **Accept:** a provisional hypothesis is never dispatchable; no tag is upgraded by admission; two visits to one stage never collide.

### WP4 — `stages.py`: append-only stage visits and maturity
- **Do:** add `StageVisit` and `StageSnapshot(status=ACTIVE|PROVISIONALLY_COMPLETE|STALE)` with the M0–M6 maturity vector; replace destructive `reopen_from` with `open_stage_visit(target_stage, causal_result_event, reason, affected_snapshots)`; require the six completeness components before a snapshot may be provisionally complete; keep `done` as a projection.
- **Accept:** `explicit_unknown` alone cannot complete a stage; re-entry never deletes a prior visit.

### WP5 — `router.py` and `ontology.py`: lineage-aware depth-first routing
- **Do:** eligibility requires a declared node, satisfied typed parent edges, a valid stage visit, remaining trial budget and satisfied ontology preconditions; ordering is mandatory causal/re-entry child, then active-path descendant, then greatest `node_depth`, then declared priority, then stable ID; unknown subcapability IDs fail closed; add a versioned structure-only ontology with explicit `stage_id` and `materiality` without overwriting v1.
- **Accept:** selection is deterministic under shuffled input; an unknown capability never dispatches.

### WP6 — `evaluation.py`: tri-state, pause and two-reproduction success
- **Do:** preserve `SUPPORTED|REFUTED|INCONCLUSIVE` downstream instead of collapsing to a boolean; replace single-score delivery with `success_eligibility(event_projection, target)` requiring two distinct qualifying closures that share one reproduction key and complete lineage; add the DS@v5 evaluation-tier, exposure and assumption fields; keep `best_score` as telemetry only.
- **Accept:** telemetry above target alone never delivers; one qualifying closure yields a reproduction child, two yield `SUCCESS_DECLARED`.

### WP7 — `graph_staged.py`: the transition machine
- **Do:** implement deficit → research → hypothesis → experiment → adjudication → residual → appropriateness audit; make closure plus adjudication plus children one `close_and_spawn` transaction; implement the seven re-entry reasons; replace generic halt with typed `RESEARCH_PAUSE`, `FAIL_CLOSED_NODE` and `SUCCESS_EXIT`, keeping a named v1 compatibility adapter for `halted` and `HALT_ALLOWED_EXHAUSTED`; reconcile the frontier from EventStore at every `observe`, honouring the existing dict-merge and `operator.add` state reducers so replay never duplicates additive lists.
- **Accept:** every tri-state result produces an adjudication and at least one allowed child in one commit; restart after a commit but before checkpoint refresh causes no redispatch and no duplicate child.

### WP8 — Evaluation hierarchy fields and claim linter
- **Do:** add `evaluation_tier`, `primary_estimand`, `surface_exposure_status` (default `UNKNOWN_NOT_FRESH`), exposure ledger IDs, selector hash, family/trial accounting, `historical_unknown_trials`, block rule and sensitivity grid, assumption checklist, `evidence_status` and `independent_confirmation_count` to the comparison receipt; implement the `V5VT-01…15` gates; implement a claim linter that blocks *fresh*, *independent holdout*, *unbiased*, *best* and *future superiority* when the matching gate fails.
- **Accept:** a strict receipt missing component, group, fold, lead, regime, boundary, tier, count, alignment, policy or fit-surface fields fails closed.

### WP9 — Legacy `graph.py` disposition
- **Do:** decide explicitly between adapting `build_graph` behind the compatibility adapter or retiring it; whichever is chosen is recorded in a receipt and covered by tests.
- **Accept:** no unaudited second mutating graph remains reachable.

### WP10 — Test matrix
- **Do:** implement the 33 named contract tests plus the 15 evaluation gates over `tmp_path` and synthetic envelopes; retain every governance regression and change only the intentionally superseded single-success and exhaustion assertions.
- **Verify:**
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/loop tests/evaluation -q
.venv/bin/python -m ruff check .
```

### WP11 — Live migration
- **Do:** back up the registry JSON; run the WP0 dry run again against the live stores and abort on any conflict; execute the additive DB migration and registry import in one transaction; append `REGISTRY_MIGRATED`; re-verify the event chain and confirm that sequence 1–158 hashes are byte-identical.
- **Accept:** comparison count stays 4; the pre-migration tail hash `1f92bedddaff46aec17470ef69634967cabfbdb4ce30f57dc54a4d31ec566eee` is still reachable and unchanged.
- **Rollback:** restore the backup JSON and drop the added tables; `events` is never modified, so the chain cannot be damaged.

### WP12 — T0 foundation diagnostics `N29`–`N35`
Run in the frozen deficit order, each predeclared and closed as a non-score-bearing node, each consuming **no** comparison index:

| Node | Deficit | Output | Data touched |
|---|---|---|---|
| `N29_ISSUANCE_CUBE_KEYS` | FD1 | source × reference × available_at × valid_time × variable × level × grid key audit, uniqueness, lead equation, join coverage | keys and schema only |
| `N30_ACTIVE_LINEAGE_GRAPH` | FD6 | builder→config→manifest invocation graph and exact prefix set differences | source and manifest hashes only |
| `N31_MISSINGNESS_STATE` | FD4 | presence/null-mask coverage cube and fallback transition table | presence masks only |
| `N32_LABEL_SUPPORT_CUBE` | FD2 | group × month × hour/lead presence and official-eligibility counts | counts only, values suppressed |
| `N33_DIAGNOSTIC_INDEX` | FD3 | score-free diagnostic registry with evidence class and contamination flags | metadata only |
| `N34_COMPARISON_RECEIPT_SCHEMA` | FD5 | contract test forcing the full diagnostic receipt schema | synthetic only |
| `N35_EXPOSURE_LEDGER` | FD7 | append-only candidate/choice/family/freshness ledger | metadata only |

**Accept:** no fit, no metric call, no label value, no prediction body, and no comparison index consumed. Each closes with a residual deficit and an appropriateness decision.

### WP13 — First hypothesis cycle
Only after WP12: admit the highest-leverage migration hypothesis produced by Wave-B research and the repaired foundation, run a T0 diagnostic, then a T1 bounded screen on inner origins, then, if the screen survives, one T2 strict prequential comparison predeclared at comparison index **5**. The full selection procedure is nested inside past-only origins; the resampling unit is the 72-cell issuance day; exact Total and components are recomputed inside every draw; the outcome carries `RETROSPECTIVE_CORROBORATION` plus `CONFIRMATION_PENDING`, never a fresh-confirmation label.

---

## Global verification after every package

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests -q
.venv/bin/python -m ruff check .
```
plus: immutable input hashes re-checked, EventStore chain verified, comparison count re-derived from the ledger, and every touched file's SHA-256 recorded in the package receipt.

## Stop conditions

- Any governance test regression, chain mismatch, or migration conflict → halt the package, roll back, and record a `FAIL_CLOSED_NODE`.
- Bounded exhaustion below target → `RESEARCH_PAUSE` with resume requirements, never success.
- Two deterministic strict reproductions at Total >= 0.660000 with complete lineage → `SUCCESS_DECLARED`.

## Approval request

Approve with the exact phrase **`IP@v3 승인`**.
