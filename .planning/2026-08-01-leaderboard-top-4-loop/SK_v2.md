# Project Skeleton — Dacon Read-Only Benchmark and Top-4% Target

## Metadata

- Skeleton ID: `SK`
- Version: `v2`
- Stable locator: `.planning/2026-08-01-leaderboard-top-4-loop/SK_v2.md`
- Status: `APPROVED_THEN_TARGET_SUPERSEDED(SK@v2)`
- Created: `2026-08-01` (Asia/Seoul)
- Base scope: previously approved `SK@v1`, `DS@v1`, and `IP@v1` local pipeline
- Nearest project authority: `/Users/um-yunsang/BARAM2026/AGENTS.md`
- Project-local template used: `none`; adapted from the preprocessing fallback skeleton

## Raw request and clarification

> https://dacon.io/competitions/official/236727/leaderboard 리더보드를 보면 알겠지만 이정도 점수로 절때 제출할수 없어 적어도 상위 4% 성적이 나올때 까지 루프 실험과 고도화작업을 통해 루프해

> 데이콘 업로드를 하라는게 아니라 로그인된 dacon 계정에서 점수와 순위확인만해 리더보드에 각 제출물 별로 1-NMAE과 FICR 점수가 같이 공지되니깐 최소 가이드들을 알고 작업하라는 말이야

## Authority split

### Authoritative intent

- Outcome: establish live Dacon performance guideposts, then use them to shape a local model-improvement loop aimed at top-4% performance.
- Explicit target/context: competition `236727`, its live leaderboard, and the already logged-in user's existing submission-history surface.
- Must: read score, rank, `1-NMAE`, and `FICR` where Dacon exposes them; calculate and record a top-4% boundary; preserve the completed local candidate and one-use 2024 lockbox.
- Avoid: any new upload, submission, form entry, account/team/settings change, remote Git action, or claim that a local score is a live leaderboard score.
- Permissions already granted: bounded navigation and observation in the already authenticated Chrome/Dacon session, limited to leaderboard and existing submission records.
- Explicit prior approvals already recorded: `SK@v1`, `DS@v1`, and `IP@v1` apply only to the completed local pipeline and do not authorize this revised research or any new implementation.

### Provisional items

- Factual premises: current participant/team count, exact top-4% rank boundary, cutoff total score, component-score distribution, the user's visible prior scores/ranks, and public/private leaderboard status.
- Success claims: no current artifact is confirmed to meet the live top-4% boundary.
- Inferred scope: numerical account history may be recorded, but account/team names and other identifiers will not be copied into the research receipt.
- Candidate solutions: metric-aware calibration, chronology-safe validation improvements, official-data-only feature engineering, diversified tree ensembles, and conditional temporal challengers.
- Assumed approvals: none for training, implementation, new dependencies, GPU/remote compute, or Dacon submission.

## Normalized direction

- Outcome: turn live leaderboard and existing-submission evidence into concrete minimum total-score, `1-NMAE`, and `FICR` targets for an offline improvement pipeline.
- Target/context: Dacon competition `236727` as visible in the user's logged-in Chrome session on the observation date.
- Deliverable: a timestamped read-only evidence receipt containing leaderboard population, top-4% rank calculation, cutoff score/component guide, existing-submission score/rank history visible to the user, unknowns, and exact source locators.
- Scope included: official Dacon leaderboard; existing submissions/results already present in the logged-in account; relevant leaderboard filters/tabs; read-only inspection of the official scoring/rules page only when needed to interpret displayed fields.
- Scope excluded: upload or resubmission; typing into Dacon forms; account/team/settings mutation; deleting or downloading submissions; messaging; browser extension/plugin installation; external datasets; model training or source-code changes during this research gate.
- Must / avoid: use official Dacon surfaces as primary evidence; timestamp observations; distinguish team-level leaderboard rows from submission-level history; preserve unknowns; do not expose personal identifiers.
- Provisional premises to verify: whether component scores are exposed per leaderboard row, per submission, or both; public/private split; leaderboard population definition; tie handling; and whether `ceil(0.04 × N)` is the correct operational top-4% boundary.
- Candidate solutions: after research, compare three offline-loop designs—metric-balanced tree ensemble, NMAE-first predictor with FICR-aware calibration, and regime/temporal challenger—and retain only evidence-supported options.
- Done signal: exact Dacon-derived target values and current numerical gap are recorded with timestamp and locators, with no external mutation performed.
- Open item: `none`

### Canonical skeleton

> For Dacon competition 236727 in the user's authenticated Chrome session, produce a timestamped read-only benchmark receipt that quantifies the live top-4% total and component-score targets plus the user's existing-submission gap; use only official Dacon evidence, preserve privacy and the consumed lockbox, and perform no upload, account mutation, code change, or training.

## Project-local authority

| Local gate or rule | Exact criterion | Current state | Evidence required | Authority locator |
| --- | --- | --- | --- | --- |
| Single-root execution | No subagents, worktrees, or background agent sessions | Active | Live agent/session audit | `AGENTS.md:3` |
| Official-data boundary | Competition-supplied data only; no external data, pretrained weights, remote inference, reanalysis, or test-period observations | Active | Source manifest and experiment receipts | `AGENTS.md:5` |
| Runtime boundary | Project-local Python 3.12 and at most six model workers | Active for later implementation | Runtime receipt | `AGENTS.md:6` |
| 2024 lockbox | One use only; prior consumption cannot become an iterative oracle | Consumed and preserved | Existing lockbox receipt | `AGENTS.md:7` |
| Dacon external action | Separate explicit authority required for upload or browser/account mutation | Upload prohibited; read-only navigation explicitly allowed | This user clarification plus browser action log | `AGENTS.md:8` |
| Narrow PASS | Claims must state whether they prove research observation, local validation, reproduction, or leaderboard result | Active | Scoped receipts | `AGENTS.md:10` |

## Approval ledger

| State | Artifact ID/version/locator | Decision and scope | Approving authority | Message/turn locator | Date | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Prior skeleton/design/plan | `SK@v1`, `DS@v1`, `IP@v1` | Completed local pipeline only | User | Prior messages in this task | 2026-08-01 | `APPROVED_PRIOR_SCOPE_ONLY` |
| Skeleton approval | `SK@v2`; this file's stable locator; pre-approval SHA-256 `9b976ec8141efced10a058f43d47b606d677238dda881d37dc95ca6c3d7f8699` | Bounded read-only Dacon research only; no upload or account mutation | User | Exact message `SK@v2 승인` in the next user turn | 2026-08-01 | `APPROVED` |
| Synthesized-design approval | `DS@v2`; `.planning/2026-08-01-leaderboard-top-4-loop/DS_v2.md` | Prepare bounded primary-source research and `IP@v2` only; no training, dependencies, upload, account mutation, or remote compute | User | Exact message `DS@v2 승인` | 2026-08-01 | `APPROVED` |
| Implementation authorization | future `IP@v2` | Approved local experiment plan only; Dacon upload remains excluded unless separately authorized | User | none | — | `PENDING` |

## Research manifest

The nearest project rule forbids subagents, so the root will execute one bounded sequential lane.

| Lane ID | Exclusive questions | Exclusions | Source/time bound | Output | Stop/escalate condition | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `L1-DACON-READONLY` | Visible rank boundary; cutoff total/component scores; authenticated user summary; board semantics | No upload, typing, account/team mutation, non-Dacon research, or personal identifiers | User-supplied saved HTML of the exact official Dacon leaderboard URL | Evidence-ledger rows and timestamped read-only receipt | Stop after source integrity, cutoff, current gap, and limitations are recorded | `COMPLETE_SNAPSHOT_EVIDENCE` |

## Evidence ledger

Use exactly one evidence tag: `directly_supported`, `contradicts_premise`, `near_match_only`, or `insufficient`.

| Claim ID | Skeleton field / local gate | Finding | Evidence tag | Source locator/date | Scope match | Implication | Proposed delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `LOCAL-001` | Current leaderboard status | The completed candidate has only local reproduction and lockbox receipts, not a Dacon leaderboard receipt. | `directly_supported` | `reports/final_audit.json`, inspected 2026-08-01 | Exact current repository scope | Live cutoff and user gap still require approved observation. | none |
| `LOCAL-002` | 2024 lockbox | The 2024 lockbox was already consumed once. | `directly_supported` | `reports/lockbox_receipt.json`, inspected 2026-08-01 | Exact project lockbox | Do not reopen or tune against it. | none |
| `USER-001` | External-action boundary | The user explicitly permits score/rank inspection and explicitly excludes Dacon upload. | `directly_supported` | Current user clarification, 2026-08-01 | Exact competition and requested action | Browser work must remain read-only. | none |
| `DACON-001` | Top-4% target | The saved D-0 public page marks rank 35 as the top-4% endpoint with total 0.65686. | `directly_supported` | `reports/leaderboard_readonly_receipt.json`, source hash recorded 2026-08-01 | Exact competition snapshot | Establishes a timestamped minimum guide, not a final-board guarantee. | Set safe total target to 0.66000. |
| `DACON-002` | Current numerical gap | The redacted user summary is rank 689, total 0.62366, 1-NMAE 0.87305, FICR 0.37426. | `directly_supported` | Same receipt and supplied HTML | Exact authenticated summary | FICR, not 1-NMAE, is the primary visible gap. | Use FICR-first design with NMAE guardrail. |
| `DACON-003` | Per-submission history premise | The saved file exposes submission count 2 but no two individual result rows. | `contradicts_premise` | Same receipt; HTML structure/content search | Exact supplied file | Per-submission comparison remains unknown. | Do not infer; request another artifact only if later necessary. |

## Contradiction and revision

- Current state: no exact-scope contradiction recorded before external research.
- Affected premise: none.
- Exact evidence: none.
- Downstream dependencies: leaderboard targets and later model design depend on `L1-DACON-READONLY`.
- Paused work: browser research, synthesis, implementation, training, and all Dacon submission actions.
- Unaffected work allowed: local planning-file maintenance only.
- Revision version: none.
- Field-level diff: `SK@v2` narrows the new external scope to read-only Dacon observation and removes inferred upload authority.
- Fresh approval required: `SK@v2 승인` before opening authenticated Dacon pages.

## Synthesis

- Design ID: `DS`
- Design version: `v2`
- Design fingerprint or stable locator: `.planning/2026-08-01-leaderboard-top-4-loop/DS_v2.md`
- Base skeleton ID/version: `SK@v2`
- Supported fields: saved-snapshot cutoff, current redacted summary, component gap, displayed total formula, page band semantics
- Explicit unknowns or blocked fields: per-submission history, full participant population, final/private leaderboard, future cutoff movement
- Resolved contradictions: none
- Pending contradictions: the supplied file contradicts the premise that it contains per-submission history; this does not block the top-4% target guide
- Alternatives retained: classical ensemble, hierarchical calibration, distributional utility challenger, and deferred temporal/deep challenger
- Limitations: without new uploads, later work can target a live threshold but cannot verify a newly trained model's actual leaderboard rank.
- Recommended design: FiCR-first cross-fitted hierarchical calibration and utility-aware decision layer, pending exact `DS@v2` approval.

## Next action

- Current state: `REVISION_REQUIRED(SK@v2 -> SK@v3)` after the user added a hard top-20 finals constraint.
- Authorized action: maintain historical receipts and present `SK@v3` for approval.
- Prohibited action: official/SOTA research, repository gap audit, synthesis, code, training, submission, account mutation, and external communication before revised-skeleton approval.
- Next checkpoint: exact user approval `SK@v3 승인`.
