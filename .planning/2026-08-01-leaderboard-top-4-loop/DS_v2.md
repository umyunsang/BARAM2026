# Synthesized Design — FiCR-first Offline Improvement Loop

## Metadata

- Design ID: `DS`
- Version: `v2`
- Stable locator: `.planning/2026-08-01-leaderboard-top-4-loop/DS_v2.md`
- Status: `APPROVED_THEN_TARGET_CONTRACT_INVALIDATED(DS@v2)`
- Base skeleton: `SK@v2`
- Evidence receipt: `reports/leaderboard_readonly_receipt.json`
- Scope: local design and later implementation planning only
- Explicit exclusions: Dacon upload/account mutation, external data, pretrained weights, remote inference, reusing the consumed 2024 lockbox as an iterative oracle

## Evidence-backed target contract

The supplied D-0 public-leaderboard snapshot establishes rank 35 as the visible top-4% endpoint.

| Contract | Total | 1-NMAE | FICR | Meaning |
| --- | ---: | ---: | ---: | --- |
| Current visible summary | 0.62366 | 0.87305 | 0.37426 | Authenticated user summary in the saved HTML |
| Snapshot minimum | 0.65686 | >= 0.87300 | >= 0.44067 at current 1-NMAE | Match visible rank-35 total |
| Safe proxy target | >= 0.66000 | >= 0.87300 | >= 0.44695 at current 1-NMAE | Buffer above the D-0 rank-35 cutoff |

The current 1-NMAE already exceeds rank 35 by 0.00092, while FICR trails it by 0.06732. The design therefore treats FICR as the primary bottleneck and 1-NMAE as a guardrail.

## Pipeline architecture

### M0 — Contract and Reproducibility Engineer

- Freeze official source hashes, runtime, seeds, feature schema, split IDs, configuration, and receipts per run.
- Use only project-local Python 3.12 and at most six model workers.
- Keep generated outputs untracked and preserve the current candidate as an immutable baseline.
- Refuse promotion if target alignment, row order, bounds, or capacity constraints fail.

### M1 — Data Analysis Engineer

- Build chronology-safe out-of-fold diagnostics by group, forecast lead hour, month/season, wind-speed regime, predicted-power band, and weather-change regime.
- Report normalized absolute-error distributions, group contribution to aggregate score, and capacity/value-weighted FICR contribution.
- Count migrations across the official settlement boundaries: `<= 6%`, `(6%, 8%]`, and `> 8%` capacity error.
- Separate zero/low-production rows from valid settlement rows using the official `actual >= 10% capacity` rule.

### M2 — Feature Engineer

- Reuse only competition-supplied observations and forecasts.
- Add auditable circular wind-direction transforms, vector components, multi-height shear/veer, stability proxies, forecast deltas/ramps, lead-time encodings, diurnal/seasonal harmonics, group interactions, and missingness/availability flags.
- Fit all learned transforms inside each training fold; prohibit future timestamps and target-derived rolling leakage.
- Promote feature families by ablation across chronology folds, not by one split.

### M3 — AI/ML Engineer: classical ensemble lane

- Recommended first lane: diversified LightGBM/CatBoost residual ensemble with shared and group-specific models.
- Optimize exact official metrics on out-of-fold predictions; retain NMAE-strong members and FICR-diverse members.
- Extend the existing two-model group blend to constrained multi-model blending with fine weights and stability penalties.
- Prefer this lane first because the existing classical baseline is reproducible and the missing score is primarily decision/calibration related.

### M4 — Settlement Decision Engineer

- Replace the current coarse global/group scale-offset grid with cross-fitted hierarchical calibration by `group × lead-hour × predicted-power/weather regime`.
- Shrink sparse segments toward group and global fallbacks; never fit a segment calibrator on its own validation labels.
- Optimize exact official total score while enforcing the 1-NMAE floor.
- Add boundary-migration diagnostics that quantify actual production value moved into the 4-unit and 3-unit settlement tiers.

### M5 — Distributional/Utility Challenger

- Train quantile or residual-distribution models from official data only.
- At prediction time, select the point forecast maximizing expected settlement utility, using estimated probability mass inside the `±6%` and `±8%` capacity bands, subject to the NMAE guardrail.
- Compare against median/mean point forecasts under identical folds and features.
- Stop this lane if calibration error or fold instability erases its exact-metric gain.

### M6 — Temporal/Deep Challenger

- Keep temporal/deep architectures inactive by default.
- Activate only if classical ensemble plus decision calibration plateaus, and only under a separately approved compute/dependency budget.
- Require a clear incremental gain after inference cost, seed variance, and leakage audit; do not use pretrained weights.

### M7 — Self-Evaluation Engineer

- Use rolling issuance-batch chronology folds with identical fold IDs for all candidates.
- Record mean, median, worst-fold, per-group 1-NMAE/FICR, tier counts, seed variance, and paired deltas against the frozen baseline.
- Enforce physical bounds and group-specific capacity clipping.
- Do not reopen the consumed 2024 lockbox. The Dacon target is an external guide, not a claim that offline and public-board distributions are identical.

### M8 — Experiment Orchestrator

- Run stages in order: data contract → baseline reproduction → diagnostics → feature ablations → ensemble → hierarchical calibration → utility challenger → final reproduction.
- Keep a fixed, approved configuration budget; prune candidates that fail the 1-NMAE floor, group stability, or reproducibility checks.
- Freeze the selected candidate and produce a submission file locally only after all local gates pass. Upload remains excluded.

## Offline loop and promotion gates

1. Reproduce the frozen baseline on unchanged chronology folds.
2. Diagnose settlement-tier misses and identify the highest-value group/lead/regime cells.
3. Run bounded feature and ensemble ablations.
4. Cross-fit segment calibration and utility-aware decisions.
5. Promote only configurations that meet all gates:
   - exact local proxy total `>= 0.66000` across the approved aggregation;
   - 1-NMAE `>= 0.87300` and no material regression versus baseline;
   - FICR `>= 0.44695` at the target aggregate, with stable group/fold behavior;
   - no fold dominated by a single seed or segment;
   - deterministic reproduction, schema/bounds checks, and complete receipts.
6. Label any pass `PASS_LOCAL_PROXY_TARGET`; never call it a live top-4% result without a later Dacon score.

## Bounded experiment budget proposed for IP@v2

- Stage A diagnostics and reproduction: no more than 6 deterministic jobs.
- Stage B feature-family ablations: no more than 24 configurations.
- Stage C ensemble/blend candidates: no more than 18 configurations.
- Stage D calibration/utility policies: no more than 36 cross-fitted configurations after pruning.
- Stage E final seed/reproduction audit: at most 3 seeds for the final two candidates.
- Maximum parallel model workers: 6; no background agents or recursive delegation.

Exact wall-time and hardware ceilings must be written in `IP@v2` after design approval and a live runtime inventory.

## Alternatives and trade-offs

| Option | Expected value | Main risk | Decision |
| --- | --- | --- | --- |
| NMAE-only model search | Small because current 1-NMAE already clears rank-35 value | Cannot close the observed total gap at current FICR | Reject as primary strategy |
| Coarse global/group calibration only | Cheap and reproducible | Misses lead/regime-specific bias and threshold migration | Retain as control |
| Hierarchical cross-fitted calibration | Directly targets the observed FICR bottleneck with controlled complexity | Sparse-segment overfit | Recommended |
| Distributional utility decision | Aligns point decisions with settlement tiers | Probability calibration instability | Challenger after hierarchical calibration |
| Deep temporal model first | May capture sequence structure | Compute/dependency cost and weak evidence that representation is the main bottleneck | Defer |

## Research requirement for implementation planning

Before `IP@v2` is approved, each nontrivial model/decision module must receive a bounded primary-source literature and benchmark review covering wind-power forecasting, gradient-boosting/time-series ensembles, probabilistic calibration, and decision-focused forecasting. That research must not introduce external training data or pretrained artifacts. Citations support design choices; they do not override the competition-data boundary.

## Known limitations

- The supplied file is a D-0 public snapshot; the cutoff can move and does not prove the final/private leaderboard.
- It contains one best user summary and submission count, not the two individual submission-history rows.
- Offline scores may not map perfectly to Dacon because labels for the competition test period are unavailable locally.
- Reaching a local proxy threshold is not proof of top-4% placement.

## Approval request

Approved by the user with the exact message `DS@v2 승인` on 2026-08-01. This authorizes preparation of a detailed `IP@v2` and the bounded primary-source research needed for that plan. It does not authorize training, new dependencies, Dacon upload, account mutation, or remote compute.

After approval, the user added a hard top-20 finals constraint. This invalidates the rank-35 target and promotion thresholds in this design. The architecture remains a provisional candidate only; work transitions to `REVISION_REQUIRED(SK@v2 -> SK@v3)` and requires fresh `SK@v3` approval before research resumes.
