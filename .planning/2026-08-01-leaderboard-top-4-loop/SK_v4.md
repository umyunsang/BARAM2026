# Project Skeleton — Recursive Evidence-to-Experiment Pipeline Discovery

## Metadata

- Skeleton ID: `SK`
- Version: `v4`
- Stable locator: `.planning/2026-08-01-leaderboard-top-4-loop/SK_v4.md`
- Status: `SKELETON_APPROVAL_PENDING(SK@v4)`
- Created: `2026-08-08` (Asia/Seoul)
- Base authority: `/Users/um-yunsang/BARAM2026/AGENTS.md`
- Existing approved base: `SK@v3`, `DS@v3`, `IP@v2`
- Revision trigger: recursive stage-depth discovery and strict local-OOF `Total >= 0.66` are now the explicit outcome.

## Raw request

> 결국 고도화되고 고정밀한 예측모델을 개발, 발굴을 한다는건, 정확한 데이터분석을 바탕으로 예측모델 파이프라인을 구축한다는거랑 동일하다는거야 각 파이프라인에 해당하고 구성하는 단계에서 가장 벤치마크하고 sota한 자료들로만 파이프라인을 구축하면 0.66 스코어 달성은 물론 해커톤 리더보드 1등도 가능한거야, 따라서 파이프라인구축을 위한 워크플로우가 더 중요해, 각 파이프라인을 얼마나 세분화하는것도 중요하고, 세분화된 영역에 대한 sota자료와 벤치마크자료 딥리서치도 중요하고, 리서치한 결과를 바탕으로 추론하여 해커톤 데이터구조와, 구성에 맞게 마이그레이션 하는것도 중요한거야 이에 파이프라인 확장이 파운데이션부터 시작해서 앞서 설명한 루프를 통해 지속적인 확장과 발굴, 정확한 평가,판단도 중요한거야. 우리 해커톤 문제는 이미 있는 경로를 찾는 문제가 아니라 가장 정합한 데이터분석을 바탕으로 다음노드(데이터 전처리) 방법들을 논문자료와 실험,연구자료를 딥리서치해서 방향을 잡고 sota자료와 벤치마크자료들로 방법을 적용한 다음 깊이의 노드들을 만들어서 각 노드별로 실험,연구, 실행을 해서 또 그결과를 바탕으로 다음노드(피처구성)방법들을 논문자료와 실험,연구자료를 딥리서치해서 방향을 잡고 sota자료와 벤치마크자료들로 방법을 적용한 다음 깊이의 노드들을 만들어서 각 노드별로 실험,연구, 실행을 하는거야 이런식으로 깊이를 내려가며 예측문제의 목적파악->데이터특성->데이터정밀분석-> 평가지표의 이해-> 데이터전처리->피처 구성-> 모델링 방법-> 검증 전략 및 예측 성능평가 -> 문제 해결 접근 방식(워크플로우) -> 모델 개선 전략(워크플로우) -> 분석방법의 적절성(워크플로우) 이런식으로 발굴작업을 해야하는데, 우리 워크플로우는 내려가며 발굴하도로 구축되어 있는지 확인하고, 워크플로우를 이용하여 루프 발굴, 개발 작업 진행해 탈출조건은 로컬평가 0.66스코어 이상의 고도화된 예측모델 개발, 발굴 이야.

## Intake normalization

| Field | Normalized content |
|---|---|
| Outcome | Audit whether the current workflow truly descends from foundation through fine-grained pipeline nodes, then run a research → migration → experiment → adjudication → next-node loop until a reproducible strict chronology-safe local official `Total >= 0.66` model exists. |
| Target/context | BARAM2026 wind-power forecasting competition; the existing S1-S11 stage taxonomy, M270/M271 graph/router, experiment registry, and S12-S16 evidence lineage. |
| Deliverable | (1) workflow/coverage audit, (2) revised typed recursive graph and evidence ledger, (3) stage-specific research dossiers and migration contracts, (4) root-executed experiments and receipts, (5) reproducible promoted local model. |
| Scope boundary | Include bounded local audit, primary-source/SOTA/benchmark research, eligible public external data and commercial-use-compatible pretrained weights, local source edits and model fits after later design/implementation approval. Exclude Dacon upload/account mutation, remote API inference, reanalysis, test-period observations, and any further 2024 lockbox score/slice/selection. |
| Must | Preserve immutable hashes; use official metric; use strict issuance-batch chronology; at most six model workers; every research claim gets an evidence tag; every migration states exact data/metric/topology mismatch; every candidate is predeclared; root owns all repository writes/fits. |
| Avoid | SOTA-by-label adoption, same-fold/post-hoc selection, stale closure reuse, local→online offset transfer across method classes, mixed-policy artifacts, uncontrolled multi-DOF blending, cosmetic variants, and claiming leaderboard rank from local evidence. |
| Done signal | Final: official strict chronology-safe OOF pooled `Total >= 0.660000`, exact key/policy lineage, no 2024 reuse, multiplicity-aware promotion, and deterministic reproduction. Research phase: a supported/unknown status for every material workflow field and an approval-ready `DS@v4`. |
| Open item | `none`; use strict `>= 0.660000` (not rounded display score) as the requested local exit threshold. |

## Authority split

### Authoritative

- The workflow itself, not a single model family, must be developed and audited.
- Discovery proceeds through: purpose → data characteristics → precision analysis → metric → preprocessing → features → modelling → validation/evaluation → problem-solving workflow → improvement workflow → appropriateness audit.
- Each stage must be decomposed into deeper nodes, researched against SOTA/benchmarks, migrated to this competition, executed, and used to choose the next node.
- The local exit condition is strict official-score `Total >= 0.66`.
- Existing project restrictions in `AGENTS.md` remain binding.

### Provisional premises requiring evidence

- “Using SOTA at every stage is sufficient for `0.66` or leaderboard first” is a hypothesis, not a success claim. Individually strong modules may be incompatible with the data horizon, availability boundary, step-reward action, or one another.
- The current workflow is not assumed complete. Preliminary bounded local evidence shows it has a real graph/router but uneven depth coverage and ledger defects.
- A single pooled OOF score is not automatically an unbiased deployment estimate after extensive adaptive search. The validation design must quantify selection multiplicity and block/time stability.
- A globally published benchmark effect is transferable only after an exact migration audit of population, forecast horizon, input availability, target, metric, model topology, and compute/license constraints.

## Canonical skeleton

> Build a stage-aware recursive evidence-to-experiment graph for BARAM2026. At every S1-S11 node, diagnose the binding local deficit; perform bounded primary-source research; translate only exact-scope-supported operations through an explicit migration contract; predeclare and root-execute the smallest discriminating experiment; adjudicate with strict chronological, multiplicity-aware official scoring; update the deficit/closure ledger; then descend, branch, or re-enter an earlier stage. Continue until a deterministic, leakage-safe strict local OOF `Total >= 0.660000` model is reproduced.

## Preliminary bounded local audit (not yet synthesized design)

| Claim | Evidence | Tag | Implication |
|---|---|---|---|
| A graph/router/checkpoint foundation exists. | `reports/m271_framework_gate_receipt.json`: Network/conditional-edge, deterministic parallel reduction, dynamic fanout, checkpoint fork tests all pass. | `directly_supported` | Do not replace the engine merely for novelty; audit its research/decision semantics. |
| The loop ledger previously failed to track most cycles. | `reports/m271_close_cycles_receipt.json`: 88 unrecorded cycles and stale `stall_counter=97`. | `directly_supported` | A node cannot close/revive reliably unless every cycle atomically updates the ledger. |
| Stage-depth research is uneven. | `research/nodes/S12-N21_workflow_audit.json`: deep research only at S3/S5/S6/S9; S7/S8/S10/S11 lacked dedicated deep-research lanes; S11 remained open. | `directly_supported` | The current workflow is only partially descending; stage coverage must become an enforced invariant. |
| Later loop work already reached a stronger strict local champion but remains below target. | `research/nodes/S16-N11_delivery.json`: champion `Total=0.6361842494`; target gap `0.0238157506`. | `directly_supported` | Micro-tuning is insufficient; next-node selection must target materially sized, information-adding mechanisms. |
| A large oracle member-selection gap exists but learned gates failed. | `S16-N2_verify_oracle.json` oracle `0.72333` vs champion `0.63618`; S16 N3-N9 gates stay below champion. | `directly_supported` | “Diversity exists” and “deployable selector exists” must be separate nodes; selector evidence cannot be inferred from oracle reachability. |

## Applicable existing project gates (names retained)

The research phase must audit and compose—not rename or silently replace—the currently applicable local gates:

1. `G0-G6` from approved `DS@v3`/`IP@v2` (contract, features, deterministic, distribution, decision, ensemble, reproduction).
2. `M270_MONTHLY_GATE_v1_frozen_2026-08-04` (sign-test, positive median, block-bootstrap q05, worst-month floor) where its frozen scope matches.
3. `M271_ROUTER_v3_frozen_2026-08-06`, including `C16` magnitude floor and `C17` novelty rejection.
4. S15/S16 block-bootstrap arbitration and sequential multiplicity adjustment.
5. Standing blend receipt fields: input policy, fitting surface (in-sample/fold-outside), and row-alignment keys.

Whether these gates conflict, duplicate, or leave gaps is a research question for `DS@v4`; no threshold is changed before evidence and exact approval.

## Proposed post-approval research manifest

Every delegated lane is read-only except that it may write its own deliverable below `research/`. It performs no model fit, touches no lockbox, and makes no shared-state or external account change. Root validates and synthesizes all claims.

| Lane | Exclusive questions | Sources / bound | Explicit exclusions | Output / stop |
|---|---|---|---|---|
| `RWA-LOCAL` | Does the engine enforce stage descent, child-node depth, atomic ledger updates, re-entry, closure-premise invalidation, and promotion separation? | At most 35 targeted local files; 90 min. | No web, fit, source edit, score call, or broad artifact dump. | `research/rwa_local_audit.md` with stage coverage matrix and evidence tags; stop when S1-S11 and all invariants are classified. |
| `RWA-EVAL` | SOTA/benchmark practice for adaptive time-series evaluation, repeated model search, block uncertainty, nested/prequential selection, and forecast-action metrics. | Up to 10 primary papers/official docs, published by search date; 90 min. | No model recommendation outside evaluation scope. | Evidence ledger + exact migration constraints; stop at supported/unknown for each validation gap. |
| `RWA-DATA` | Best-supported operations for NWP availability alignment, missingness, bias/curtailment/anomaly treatment, spatial interpolation, hub-height reconstruction, and multi-source weather fusion. | Up to 14 primary sources/standards; 2 h. | No reanalysis/test observations; no generic blog evidence. | Stage S2-S6 evidence/migration matrix. |
| `RWA-MODEL` | SOTA/benchmarks for probabilistic wind-power forecasts, mixture-of-experts/member selection, decision-focused/modal-interval learning, domain adaptation, and ensemble diversity under step rewards. | Up to 16 primary sources/official benchmark reports; 2 h. | No fit, dependency install, or claim based only on leaderboard snippets. | Stage S7/S10 candidate operations with exact non-duplication tests. |
| `RWA-LOOP` | SOTA/benchmarks for scientific discovery graphs, novelty/magnitude acquisition, falsification, stage coverage, and closure/revival semantics. | Up to 8 primary papers/official docs; 75 min. | No wholesale framework replacement unless exact defect requires it. | S9-S11 workflow delta and deterministic invariants. |
| `RWA-EXT` | Which public external inputs/weights are actually eligible and available by D-1 14:00 KST, commercially licensed, and likely to add independent information? | Up to 12 official data/licence/archive sources; 2 h. | No download/bulk collection, API inference, reanalysis, or test observations. | Eligibility/value-of-information ledger; stop on exact eligibility + archive proof or `insufficient`. |

All material claims use exactly one tag: `directly_supported`, `contradicts_premise`, `near_match_only`, or `insufficient`. Any exact contradiction triggers `REVISION_REQUIRED(SK@v4 -> SK@v5)` for affected fields.

## Approval boundary

Requested approval: `SK@v4 승인`.

Approval authorizes only the bounded research manifest and the bounded local workflow audit above. It does **not** authorize source implementation, model fitting, dependency installation, bulk external-data retrieval, Dacon/account actions, remote inference/compute/Git, or any 2024 evaluation. After research, root will synthesize a separately fingerprinted `DS@v4` for exact approval; implementation will require a written `IP@v3` and its exact approval.
