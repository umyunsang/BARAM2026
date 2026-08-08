# Project Skeleton — Hierarchical SOTA-to-Local Pipeline Discovery

## Metadata

- Skeleton ID: `SK`
- Version: `v5`
- Stable locator: `.planning/2026-08-01-leaderboard-top-4-loop/SK_v5.md`
- Status: `SKELETON_APPROVAL_PENDING(SK@v5)`
- Created: `2026-08-09T04:39:08.147938+09:00`
- Base authority: `/Users/um-yunsang/BARAM2026/AGENTS.md`
- Approved historical base: `SK@v3`, `DS@v3`, `IP@v2`
- Supersedes if approved: the withdrawn proposal `SK@v4` and the exact-treatment-only discovery interpretation used in S17-N18 through N28
- Revision trigger: the user explicitly distinguishes **method discovery and local migration** from searching for an already published exact BARAM treatment.

## Raw request

> 우리 해커톤 문제는 이미 있는 경로를 찾는 문제가 아니라 가장 정합한 데이터분석을 바탕으로 다음노드(데이터 전처리) 방법들을 논문자료와 실험,연구자료를 딥리서치해서 방향을 잡고 sota자료와 벤치마크자료들로 방법을 적용한 다음 깊이의 노드들을 만들어서 각 노드별로 실험,연구, 실행을 해서 또 그결과를 바탕으로 다음노드(피처구성)방법들을 논문자료와 실험,연구자료를 딥리서치해서 방향을 잡고 sota자료와 벤치마크자료들로 방법을 적용한 다음 깊이의 노드들을 만들어서 각 노드별로 실험,연구, 실행을 하는거야 이런식으로 깊이를 내려가며 예측문제의 목적파악->데이터특성->데이터정밀분석-> 평가지표의 이해-> 데이터전처리->피처 구성-> 모델링 방법-> 검증 전략 및 예측 성능평가 -> 문제 해결 접근 방식(워크플로우) -> 모델 개선 전략(워크플로우) -> 분석방법의 적절성(워크플로우) 이런식으로 발굴작업을 해야하는데, 우리 워크플로우는 내려가며 발굴하도로 구축되어 있는지 확인하고, 워크플로우를 이용하여 루프 발굴, 개발 작업 진행해 탈출조건은 로컬평가 목표스코어 이상의 고도화된 예측모델 개발, 발굴 이야.
> 
> 결국 고도화되고 고정밀한 예측모델을 개발, 발굴을 한다는건, 정확한 데이터분석을 바탕으로 예측모델 파이프라인을 구축한다는거랑 동일하다는거야 각 파이프라인에 해당하고 구성하는 단계에서 가장 벤치마크하고 sota한 자료들로만 파이프라인을 구축하면 목표스코어 달성은 물론 해커톤 리더보드 1등도 가능한거야, 따라서 파이프라인구축을 위한 워크플로우가 더 중요해, 각 파이프라인을 얼마나 세분화하는것도 중요하고, 세분화된 영역에 대한 sota자료와 벤치마크자료 딥리서치도 중요하고, 리서치한 결과를 바탕으로 추론하여 해커톤 데이터구조와, 구성에 맞게 마이그레이션 하는것도 중요한거야 이에 파이프라인 확장이 파운데이션부터 시작해서 앞서 설명한 루프를 통해 지속적인 확장과 발굴, 정확한 평가,판단도 중요한거야.

## Intake normalization

| Field | Normalized content |
|---|---|
| Outcome | Audit and repair the workflow so it recursively decomposes the forecasting pipeline, performs stage-specific SOTA/benchmark research, derives explicit local migration hypotheses, executes controlled experiments, and creates result-conditioned child nodes until a reproducible strict local official `Total >= 0.660000` model is developed. |
| Target/context | BARAM2026 day-ahead wind-power prediction; 72-cell issuance days; supplied GFS/LDAPS, turbine/group metadata and labels; official discontinuous NMAE/FICR action metric; append-only S17 EventStore. |
| Deliverable | A verified hierarchical discovery workflow, stage/child-node ledger, research and migration dossiers, controlled root-owned experiments, strict prequential adjudications, residual-deficit descendants, and a reproducible promoted pipeline. |
| Scope included | Bounded local workflow audit; pre-cutoff primary/SOTA/benchmark/winning-solution research; inference and migration to BARAM data; approved local preprocessing/feature/model/validation experiments; eligible public external inputs/weights under `AGENTS.md`. |
| Scope excluded | Further 2024 score/slice/selection; test-period observations; reanalysis; remote API inference; Dacon/account/team mutation; remote deployment/Git; unlicensed or post-cutoff deployed inputs. |
| Must | Start from exact local data/metric diagnostics; preserve strict issuance chronology; distinguish source evidence, transfer hypotheses and local experimental evidence; predeclare material degrees of freedom; root owns writes/fits; at most six model workers; append every comparison; reproduce the winner. |
| Avoid | Requiring a paper to prescribe BARAM's exact field/grid/formula before a hypothesis can be tested; adopting “SOTA” by label; treating a near match as proof of transfer; post-hoc variant rescue; same-fold selection; 2024 reuse; mixed-policy artifacts; false terminal success below target. |
| Provisional premise | “Using SOTA at every module guarantees `0.66` or leaderboard first” is a motivating hypothesis, not evidence. End-to-end compatibility and local gains must be demonstrated. |
| Done signal | Two deterministic reproductions of a strict chronology-safe local official pooled `Total >= 0.660000`, with exact inputs/features/model/policy/alignment lineage, multiplicity-aware adjudication, and no new 2024 use. |
| Open item | `none`; the architecture direction and local exit threshold are explicit. Exact implementation details will be synthesized in `DS@v5` after approved research. |

## Authority split

### Authoritative

1. The workflow is the primary product; model families are nodes inside it.
2. Discovery must descend in this order while permitting evidence-driven re-entry:
   `purpose → data characteristics → precision analysis → metric/action → preprocessing → feature construction → modelling → validation/evaluation → problem-solving workflow → improvement workflow → appropriateness audit`.
3. Each stage must be decomposed into child questions, researched with SOTA/benchmarks, migrated to the actual BARAM structure, experimentally tested, adjudicated, and followed by a residual-deficit child.
4. Literature need not contain an exact BARAM-ready recipe. Sound inference and a falsifiable local migration experiment are required.
5. The only success exit is a reproducible strict local official `Total >= 0.660000` model.

### Provisional and evidence-bearing

- Which stage currently contains the largest actionable deficit.
- Which SOTA/benchmark operation transfers to the supplied horizon, grid, target and action metric.
- Whether a module improves the current end-to-end pipeline rather than its source benchmark only.
- Whether extensive adaptive 2023 search can support the claimed deployment generalization.
- Whether external public inputs or pretrained weights clear availability, licence and symmetry gates.

## Bounded local workflow audit

| Claim | Exact local evidence | Tag | Consequence |
|---|---|---|---|
| The requested S1–S11 stage names already exist in the right semantic order. | `src/baram/loop/stages.py` SHA `d7cdbe242330deb00a3021dcd5e7dac3fb137419fc2544da65dab96aa28403ab`; `STAGES` lists the eleven requested categories. | `directly_supported` | Reuse the taxonomy rather than renaming it. |
| Current “depth” is only the ordinal stage index. | `graph_staged.py:59-81` sets `depth = index(stage)+1`; `stages.py:77-84` returns only the first incomplete linear stage. | `directly_supported` | It does not represent recursive child depth within preprocessing, features or modelling. |
| Current expansion admits only exact `directly_supported` findings. | `research.py` SHA `b8085f28c9a444eeeb4f6ac2c202f31758e2587d9c422870b275767be11416d7`; `ACTIONABLE_TAGS={"directly_supported"}`, `expand` at lines 112-130. | `directly_supported` | This collapses research into searching for an existing exact path and cannot represent a provisional migration hypothesis. |
| Near matches can close a stage without generating an experiment. | `research.py:102-109` marks all-near-match/insufficient findings `explicit_unknown`; `stages.py:107-124` accepts that when no nodes were emitted. | `directly_supported` | The loop can declare stage completion before testing a bounded research-informed migration. |
| Node contracts lack parent, child depth, local deficit, migration gaps, degrees of freedom and falsifier. | `registry.py:18-39` `NodeSpec` fields. | `directly_supported` | Result-conditioned descendants and audit-ready transfer reasoning are not first-class. |
| Downstream results do not normally create new research descendants or re-enter an earlier stage. | `graph_staged.py:337-454` only accepts/rejects and updates stagnation; `reopen_from` is contradiction-driven. | `directly_supported` | The graph executes a pass; it does not perform the requested scientific deepening loop. |
| Below-target all-stage closure can halt as evidence-exhausted. | `graph_staged.py:83-149` combines `termination_action` with `halt_eligibility`; EventStore sequence 158 currently reports `HALT_ALLOWED_EXHAUSTED` at Champion `0.6314827308346854`. | `directly_supported` | Exhaustion must become a pause/research-reseed state, never the requested success exit. |
| The ontology is broad but coverage is not equivalent to deep research. | `capability_ontology_v1.json` SHA `e8cdb044f6cdd64d832a292870a298e52ceb76b519385fd67a88b52851854c0f` has 55 subcapabilities; all six preprocessing entries have research status `없음`, and all six feature entries are `취소(L3)`. | `directly_supported` | Binary stage coverage overstated pipeline discovery maturity. |
| Reusable governance foundations exist. | PREDECLARED/CLOSED EventStore, strict prequential evaluator, root-only execution, immutable hashes and sequence-158 verified chain. | `directly_supported` | Repair semantics and schemas; do not discard the engine. |

### Audit conclusion

**The current workflow has the correct stage labels and strong governance plumbing, but it is not yet built for the user's requested recursive discovery.** It is linear at stage level, exact-treatment-gated at research expansion, shallow at child-node lineage, and able to stop below target after explicit unknowns. This is a material architecture defect, not a new model-candidate question.

## Canonical hierarchical loop

```text
FOUNDATION SNAPSHOT
  local purpose/data/metric/chronology contract
      ↓
STAGE DEFICIT NODE
  exact local diagnostic + leverage estimate + uncertainty
      ↓
BOUNDED DEEP RESEARCH
  primary papers + SOTA benchmarks + winning/open implementations
      ↓
MIGRATION HYPOTHESES (siblings)
  source claim | scope gaps | local mapping | DOF | expected mechanism | falsifier
      ↓
CONTROLLED EXPERIMENT NODES
  cheap diagnostic → bounded screen → full strict prequential comparison
      ↓
ADJUDICATION
  score/components/stability/multiplicity/reproducibility
      ↓
RESIDUAL DEFICIT CHILD
  explain what remains, deepen same stage, advance, branch, or re-enter upstream
      ↺
SUCCESS ONLY WHEN strict reproduced Total >= 0.660000
```

`stage_depth` and `node_depth` are separate. Moving S5→S6 increments stage depth; an S5 preprocessing result that produces a new missingness or transformation question increments node depth inside S5.

## Evidence-to-hypothesis contract

The four evidence tags retain their established meanings. The repair is **not** to relabel near evidence as direct evidence.

1. `directly_supported` confirms only the exact source/local factual claim it cites.
2. `near_match_only` remains insufficient to claim transfer or expected gain, but may navigate toward a clearly labeled `provisional_migration_hypothesis`.
3. A migration hypothesis becomes experiment-admissible only when all are frozen:
   - directly supported source mechanism/benchmark **or** directly supported local diagnostic;
   - population, geography, horizon, issue time, inputs, target, metric, resolution, topology, compute and licence gaps;
   - exact local transform/model/policy and deployment-symmetric inputs;
   - parent/control, material degrees of freedom, trial budget and row keys;
   - expected effect channel and a result that would falsify it;
   - strict chronology and no-2024 proof.
4. Analyst choices are allowed when motivated, explicitly frozen and counted. They do not become “provider-prescribed”; local experiments, not rhetoric, adjudicate them.
5. Local experiment results are a distinct evidence layer. A successful local comparison can support promotion; a paper's effect size cannot.
6. A local-data-derived hypothesis with no direct paper recipe is admissible when the diagnostic is reproducible, the operation is mathematically complete, the family is bounded, and SOTA/benchmark research documents alternatives.

This separates **evidence quality** from **hypothesis generation** and corrects the N27/N28 exact-path failure mode without weakening promotion evidence.

## Required node schema for `DS@v5`

Every node must eventually carry at least:

- `node_id`, `parent_ids`, `stage_id`, `stage_depth`, `node_depth`, `node_type`;
- `local_deficit_id` and exact diagnostic artifact;
- `research_claim_ids` with evidence tags;
- `migration_contract` and unresolved scope gaps;
- `hypothesis`, `mechanism`, `treatment`, `control`, `falsifier`;
- `degrees_of_freedom`, candidate family/trial budget, estimated compute;
- data availability/chronology/licence/deployment checks;
- metric, minimum effect, component/stability/multiplicity gates;
- status, result, residual deficit, child-generation rule and closure/revival premise.

## Stage completeness and re-entry

A stage is not complete merely because one research file exists or no exact paper recipe was found. A provisional stage snapshot requires:

1. a local foundation/deficit artifact;
2. bounded research coverage of its material subcapabilities;
3. a migration matrix, including rejected alternatives;
4. at least one controlled benchmark/migrated experiment when a feasible falsifiable hypothesis exists;
5. adjudication and residual-deficit children;
6. an appropriateness audit explaining advance, deeper descent, branch, upstream re-entry or genuine authorization/budget pause.

Any downstream result may re-enter an earlier stage for a recorded result-driven reason such as data leakage, feature insufficiency, regime collapse, calibration failure or validation instability. Contradictions remain a mandatory special case, not the only re-entry trigger.

## Evaluation hierarchy to research and synthesize

- Keep 72-cell issuance-day atomicity and strictly preceding fit/selection/label availability.
- Separate exploratory diagnostics, bounded screening and confirmatory full comparisons.
- Count all material choices and preserve append-only comparison index 5 onward.
- Use exact Total plus NMAE, FICR, group/fold/lead/regime effects and failure concentration.
- Research nested/prequential selection, sequential multiplicity and dependent uncertainty suitable for the limited 2022–2023 surface.
- Require fixed-policy reproduction and exact alignment/policy/fit-surface receipts.
- Do not claim an independent holdout where none remains.

Exact thresholds and budgets belong in the later `DS@v5`/`IP@v3`, not this skeleton.

## Post-approval research manifest

Approval authorizes only read-only research lanes and bounded local audit. Lanes may write only their own files under `research/`; root performs synthesis.

### Wave A — workflow and foundation

| Lane | Exclusive question | Bound | Output |
|---|---|---|---|
| `V5-ENGINE` | Which code/state/test changes are required for true child depth, hypothesis nodes, result-driven re-entry, pause-vs-success and atomic lineage? | ≤30 local files; no web; 60 min | engine invariant and test matrix |
| `V5-FOUNDATION` | What exact purpose/data/precision/metric deficits remain after S17, without reading 2024/test outcomes? | fixed local schema/code/allowed development diagnostics only; 60 min | S1–S4 foundation/deficit map |
| `V5-DISCOVERY-SCIENCE` | SOTA scientific/AutoML discovery workflows for hierarchical hypothesis generation, adaptive experiment allocation and falsification in time series. | ≤10 primary papers/official benchmark docs, pre-cutoff; 90 min | S9–S11 migration ledger |
| `V5-VALIDATION` | Nested/prequential evaluation and multiplicity control for repeated dependent time-series model development. | ≤10 primary papers/standards, pre-cutoff; 90 min | validation design alternatives |

### Wave B — pipeline modules

Run only after the root freezes the Wave-A foundation map. Independent lanes may research in parallel, but execution descends stage-by-stage.

| Lane | Exclusive question | Bound | Output |
|---|---|---|---|
| `V5-PREPROCESS` | SOTA NWP issue/valid alignment, QC, missingness, bias correction, anomaly/curtailment handling and spatial preprocessing that can be migrated to BARAM. | ≤14 primary/benchmark sources | S5 method→migration matrix |
| `V5-FEATURES` | SOTA spatial-temporal, multiscale, physical and representation-learning features for issued NWP wind-power forecasting. | ≤16 primary/benchmark sources | S6 sibling hypotheses |
| `V5-MODELS` | SOTA point, distributional, sequence, graph, mixture/expert and decision-focused models under small nonstationary wind datasets. | ≤16 primary/benchmark sources | S7 model ladder |
| `V5-BENCHMARKS` | Reproducible winning/benchmark pipelines and open implementations, including ablations, data scale, licence and compute. | ≤12 official repositories/papers | end-to-end benchmark migration table |
| `V5-EXTERNAL` | Eligible public data/weights only: pre-cutoff publication, commercial licence, D-1 14:00 availability, anonymous archive and deployment symmetry. | ≤10 official source/license/archive audits | eligibility ledger; no data body download |

Every lane must state what is source fact, what is transfer hypothesis, what remains unknown, and what smallest local experiment would discriminate alternatives.

## Synthesis and implementation boundary

After approved research, root will create a separately fingerprinted `DS@v5` containing:

- repaired graph/state schemas and tests;
- frozen foundation and active deficit;
- ranked migration-hypothesis portfolio;
- experiment ladder, budgets and multiplicity contract;
- stage completion/re-entry/pause semantics;
- the first stage-specific execution batch.

`DS@v5` requires exact user approval before implementation planning. `IP@v3` then requires exact approval before source edits or fits outside unchanged previously approved interfaces. This skeleton never authorizes Dacon, 2024 evaluation, dependency changes, bulk external retrieval or remote inference.

## Exit and pause semantics

- `SUCCESS_EXIT`: only two reproduced strict local official `Total >= 0.660000` runs with full lineage.
- `RESEARCH_PAUSE`: missing approval, source/access/licence block, compute/budget ceiling, or bounded evidence exhaustion. It is resumable and never marks the user goal achieved.
- `FAIL_CLOSED_NODE`: one node/family is rejected; it does not terminate the pipeline while viable sibling/upstream hypotheses remain.
- Current sequence-158 `HALT_ALLOWED_EXHAUSTED` becomes historical evidence of the old semantics, not proof that the revised pipeline is exhausted.

## Approval request

Requested exact approval: **`SK@v5 승인`**.

Approval authorizes only the bounded Wave-A and Wave-B read-only research manifest and bounded local workflow audit. It does not authorize source implementation, model fitting, dependency installation, external data-body acquisition, Dacon/account actions, remote inference/compute/Git, or any 2024 evaluation. Research will be synthesized into `DS@v5` for separate exact approval.
