# Project Skeleton — Dacon Top-20 Finals Target

## Metadata

- Skeleton ID: `SK`
- Version: `v3`
- Stable locator: `.planning/2026-08-01-leaderboard-top-4-loop/SK_v3.md`
- Status: `SKELETON_APPROVED_FOR_RESEARCH(SK@v3)`
- Created: `2026-08-01` (Asia/Seoul)
- Supersedes target fields in: `SK@v2` and `DS@v2`
- Nearest project authority: `/Users/um-yunsang/BARAM2026/AGENTS.md`
- Source snapshot SHA-256: `f1328f9b68772da6a91a2cd715d649193e6f76252b47c5dafd36ece04567257c`

## Raw target revision

> 아 추가로 상위 20팀만 본선에 진출할수있어, 현재 시각기준으로 2%에서 3%이내에는 들어가야해

## Authority split

### Authoritative intent

- Outcome: build a local forecasting pipeline aimed at qualifying for the finals, not merely entering the visible top-4% band.
- Hard placement target: only the top 20 teams advance; the operating objective must therefore correspond to approximately the top 2-3% at the current snapshot.
- Must: use the supplied Dacon leaderboard as the current numeric guide, maintain a safety buffer above rank 20, and keep all prior official-data, lockbox, runtime, privacy, and no-upload constraints.
- Avoid: treating the old rank-35/0.66000 target as sufficient, claiming finals qualification from local validation, or expanding approval to training/upload/remote compute.

### Provisional premises

- The official competition rules state that only 20 teams advance to the second-round presentation; this must be verified on an official Dacon rule/overview surface after skeleton approval.
- The D-0 public cutoff can move after the saved timestamp and can differ from the final/private leaderboard.
- Offline scores are only proxy signals and may not map one-to-one to Dacon test-period scores.
- Candidate implementation choices remain provisional until bounded repository, literature, and benchmark research is complete.

## Normalized direction

- Outcome: replace the top-4% target contract with a finals-qualification-oriented top-20 contract.
- Target/context: Dacon competition `236727`, saved public D-0 leaderboard, ranks 1-20 as the qualification reference envelope.
- Deliverable: a revised evidence receipt, `DS@v3`, and later `IP@v2` specifying a leakage-safe local experiment loop against the top-20 proxy target.
- Scope included after approval: official Dacon rule verification; supplied leaderboard analysis; read-only audit of current local code/runtime; primary-source research for wind forecasting, tree/time-series ensembles, probabilistic calibration, and decision-focused forecasting.
- Scope excluded: source implementation, training, dependency installation, Dacon upload or account mutation, external training data, pretrained weights, remote inference/compute, remote Git, and reuse of the consumed 2024 lockbox.
- Must / avoid: single root only; no subagents, worktrees, or background sessions; at most six model workers in any later approved run; preserve narrow PASS labels.
- Done signal for research: the rank-20 minimum, movement-safe operating target, component guardrails, module choices, experiment budget, and limitations are supported or explicitly unknown in `DS@v3`.
- Open item: `none`; use the conservative operating target below unless revised by exact official evidence.

### Canonical skeleton

> For Dacon competition 236727, use the supplied D-0 public leaderboard and bounded official/primary-source research to design a single-root, official-data-only forecasting loop whose local proxy aims safely above the top-20 finals cutoff, while preserving the consumed 2024 lockbox and performing no training, upload, account mutation, dependency change, or remote compute before later approvals.

## Supplied-snapshot evidence

All values below are directly parsed from the immutable user-supplied HTML; team identities are excluded.

| Reference | Rank | Total | 1-NMAE | FICR | Submissions | Evidence tag |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Top-1% visible endpoint | 10 | 0.66139 | 0.87790 | 0.44488 | 110 | `directly_supported` |
| Inner safety reference | 15 | 0.66033 | 0.87740 | 0.44326 | 83 | `directly_supported` |
| Finals minimum snapshot cutoff | 20 | 0.65971 | 0.87991 | 0.43952 | 41 | `directly_supported` |
| First outside top 20 | 21 | 0.65948 | 0.87261 | 0.44634 | 23 | `directly_supported` |
| Prior top-4% endpoint | 35 | 0.65686 | 0.87213 | 0.44158 | 34 | `directly_supported` |
| Current authenticated summary | 689 | 0.62366 | 0.87305 | 0.37426 | 2 | `directly_supported` |

Snapshot top-20 distribution:

- Total: minimum `0.65971`, median `0.66132`, maximum `0.67365`.
- 1-NMAE: minimum `0.87297`, median `0.87804`, maximum `0.89047`.
- FICR: minimum `0.43952`, median `0.446715`, maximum `0.46767`.
- Submission count: minimum `5`, median `83`, maximum `128`.

## Target alternatives

| Approach | Proxy target | Benefit | Risk | Decision |
| --- | --- | --- | --- | --- |
| A — Exact rank-20 cutoff | Total `>= 0.65971` | Lowest immediate hurdle | No protection from D-0 movement or public/private shift | Reject as operating target; retain as hard minimum |
| B — Rank-15 snapshot buffer | Total `>= 0.66033` | Small placement buffer | Only `0.00062` above rank 20 | Retain as intermediate gate |
| C — Movement-safe score buffer | Total `>= 0.66200`, 1-NMAE `>= 0.87500`, FICR `>= 0.44900` | Clears the snapshot rank-10 score and balances both components | Harder local proxy; offline-to-board mapping remains uncertain | Recommended operating target |

At the current visible 1-NMAE `0.87305`, rank 20 requires FICR `0.44637`; the recommended total `0.66200` requires FICR `0.45095`. The current summary therefore needs total `+0.03834` and FICR approximately `+0.07669` if 1-NMAE is unchanged. Unlike the rank-35 target, rank 20 also favors a stronger 1-NMAE envelope, so `DS@v3` must keep FICR as the primary bottleneck while adding a meaningful NMAE improvement lane.

## Project-local authority

| Rule | Required state | Current implication |
| --- | --- | --- |
| Single-root execution | No subagents, worktrees, or background sessions | All research remains one sequential root lane |
| Official-data boundary | Competition data only for modeling | Literature may guide methods but may not supply training data or pretrained artifacts |
| Runtime | Project-local Python 3.12; <=6 workers | Verify live runtime before `IP@v2` |
| 2024 lockbox | One use only | Never reopen for target tuning |
| External action | Separate explicit approval | Dacon upload/account mutation remains prohibited |
| Narrow PASS | State exact verification surface | Local proxy cannot prove finals placement |

## Revision transition

- Prior state: `SK@v2` and approved `DS@v2` targeted visible rank 35/top 4%, with safe total `0.66000`.
- Trigger: the user added the hard finals constraint of top 20 teams.
- Affected fields: outcome, target rank, score target, component guardrails, promotion gate, stop condition, and interpretation of success.
- Unaffected fields retained provisionally: official metric implementation, single-root execution, official-data-only modeling, FICR-aware calibration, chronology safety, module boundaries, no-upload rule, and one-use lockbox protection.
- State transition: `REVISION_REQUIRED(SK@v2 -> SK@v3)`; the target/promotion portions of `DS@v2` are invalidated.
- Paused work: external SOTA research, synthesis, `IP@v2`, implementation, training, and all Dacon actions.
- Allowed before approval: this skeleton, supplied-file arithmetic, and local approval-ledger maintenance.

## Proposed post-approval research manifest

The nearest project rule forbids delegation; the root will execute the following sequentially.

| Lane | Questions | Allowed sources | Bound | Output | Stop condition |
| --- | --- | --- | --- | --- | --- |
| `R1-OFFICIAL-FINALS` | Does the official rule confirm top-20 presentation advancement and what public/private caveats apply? | Official Dacon competition pages only | Up to 3 official pages | Tagged evidence rows | Stop on exact rule or `insufficient` |
| `R2-LIVE-REPO` | Which current modules/configs/tests can implement the revised target without reopening 2024? | Current local repository and immutable receipts | Targeted files only | Gap map | Stop when every DS module maps to a live interface |
| `R3-PRIMARY-SOURCES` | Which methods directly support wind forecasting, probabilistic calibration, ensemble diversity, and decision-focused point selection? | Primary papers, standards, and official library docs | Up to 12 sources | Research matrix | Stop when each nontrivial module has direct or insufficient evidence |
| `R4-SYNTHESIS` | What revised architecture, budget, and promotion gates follow? | R1-R3 evidence only | One synthesis pass | `DS@v3` | Stop at approval-ready design |

`R3-PRIMARY-SOURCES` covers the complete competition-data pipeline in this order: data availability/contract, data quality and EDA, weather/physical feature construction, chronology-safe validation, deterministic tree models, probabilistic/distributional models, temporal challengers, ensemble diversity, FICR-aware point decisions, self-evaluation, and reproducibility. External sources may justify methods only; they may not contribute data, pretrained weights, forecasts, or test-period observations.

## Approval ledger

| Artifact | Scope | Message | Status |
| --- | --- | --- | --- |
| `SK@v2` | Saved top-4% leaderboard research | `SK@v2 승인` | `APPROVED_PRIOR_SCOPE` |
| `DS@v2` | Top-4% design and preparation of research/`IP@v2` | `DS@v2 승인` | `APPROVED_THEN_TARGET_SUPERSEDED` |
| `SK@v3` | Revised top-20/finals research only, including module-by-module research from competition-data feature construction through model development | Exact message `SK@v3 승인, sota/벤치마크 조사는 대회데이터를 바탕으로 피처구성부터 시작해서 예측모델 개발까지 각 모듈별로 조사하는거맞지?` | `APPROVED` |

## Approval request

Approved by the user on 2026-08-01 with the exact message recorded above. Approval authorizes only the bounded read-only lanes and the clarified end-to-end module research scope. It does not authorize code changes, training, dependency installation, Dacon upload/account mutation, external datasets, pretrained weights, remote compute, or remote Git.
