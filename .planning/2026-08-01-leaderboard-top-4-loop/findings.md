# Findings & Decisions: Leaderboard Top-20 Finals Loop

## Requirements

- Requested outcome: use the live Dacon leaderboard and the user's existing submissions as minimum guides, then iterate local modeling toward top-4% performance.
- Explicit target: Dacon competition 236727 leaderboard supplied by the user.
- Explicit permission: inspect existing score/rank/component scores in the already logged-in Dacon account.
- Explicit exclusion: do not upload any submission.
- Must preserve the user's prior operational preference: no subagents and immediate session cleanup.
- Must preserve project rules: official data only, local Python 3.12, six workers, immutable inputs, and no remote Git.

## Directly Supported Local Findings

| Claim | Tag | Evidence | Implication |
|---|---|---|---|
| The prior result is local, not a leaderboard score. | directly_supported | `reports/final_audit.json` status and scope | A live leaderboard cutoff and an actual submitted score remain unknown. |
| The 2024 lockbox was consumed exactly once. | directly_supported | `reports/final_audit.json`, `artifacts/locks/lockbox-2024.consumed.json` | It cannot be reopened or repeatedly optimized against. |
| The prior IP scope excluded Dacon upload and leaderboard claims. | directly_supported | `AGENTS.md`, `reports/final_audit.json`, memory index of IP@v1 boundary | A new external-action approval is required. |
| The current reproducible candidate is calibrated LightGBM with local 2024 total 0.627605. | directly_supported | `reports/lockbox_receipt.json` | This score cannot be compared directly with the leaderboard until the live competition surface is verified. |
| Repository state was clean at commit `383265a` before creating this planning area. | directly_supported | live `git status`, `git log` | Existing implementation can be treated as a frozen baseline. |
| The user authorizes Dacon score/rank inspection and expressly excludes uploads. | directly_supported | current user clarification | The research lane may navigate/read authenticated Dacon pages after `SK@v2` approval, but may not submit or mutate account state. |

## Provisional Premises

- The live leaderboard cutoff for top 4% is unknown until approved read-only browser research.
- The user's visible existing-submission scores, component scores, and ranks are unknown until approved read-only Chrome inspection.
- The current request does not authorize uploads; a later local model cannot be described as achieving a live rank without separate external evidence.
- It is unknown whether the validation gap is primarily calibration, temporal regime shift, feature insufficiency, architecture, or metric optimization.
- Deep/GPU and expanded compute remain outside current authority until explicitly included in a later approved artifact.

## Candidate Directions — Not Yet Decisions

1. Strengthen leakage-safe multi-year rolling validation and leaderboard proxy calibration.
2. Add official-data-only weather-regime, spatial, direction-sector, ramp, and availability-aware features.
3. Build diversified residual ensembles across tree families and group/shared architectures.
4. Add objective-aware calibration/threshold optimization using cross-fitting only.
5. Consider deep temporal challengers only under a separately approved compute/dependency design.

## Resolved Material Question

The user permits authenticated read-only inspection of existing Dacon scores/ranks and explicitly prohibits uploads. No material intake question remains before `SK@v2` approval.

## Resources

- `/Users/um-yunsang/BARAM2026/AGENTS.md`
- `/Users/um-yunsang/BARAM2026/reports/final_audit.json`
- `/Users/um-yunsang/BARAM2026/reports/lockbox_receipt.json`
- User-supplied leaderboard URL (not opened before approval)
- `.planning/2026-08-01-leaderboard-top-4-loop/SK_v2.md`

## Live Research Log

| Claim | Tag | Evidence | Implication |
|---|---|---|---|
| An open Chrome tab exactly matching the official competition leaderboard URL is present. | directly_supported | Chrome `openTabs()` result, 2026-08-01, URL `https://dacon.io/competitions/official/236727/leaderboard` | The requested authenticated surface exists and no new tab/navigation is needed. |
| The first attempt to claim/read that tab was denied because the browser security policy check was temporarily unavailable. | insufficient | Chrome browser-control response, 2026-08-01 | No leaderboard or account values were accessed; retry once through the same documented Chrome surface without bypass. |
| Immediate and delayed retries reached the same pre-page security-policy denial. | insufficient | Three bounded Chrome claim attempts on 2026-08-01 | The authenticated page remains unread; no score, rank, `1-NMAE`, or `FICR` value may be asserted yet. Stop and preserve the exact Chrome/no-fallback boundary. |
| The Chrome research session was finalized after the repeated denial. | directly_supported | Browser finalization response, 2026-08-01 | No agent-created tab remains and the user's original Dacon tab was not closed or changed. |
| After the user refreshed the Dacon tab, Chrome again listed the exact refreshed URL but the policy gate still denied page access. | insufficient | Chrome `openTabs()` timestamp `2026-08-01T13:43:47.169Z` followed by policy denial | Refresh/login state is not the blocker; the browser-control policy verifier remains unavailable. Do not retry or bypass in this turn. |
| A further retry after the user reconnected Chrome again found a newly active Dacon tab but failed at the identical pre-page policy gate. | insufficient | Chrome `openTabs()` timestamp `2026-08-01T13:46:59.991Z` followed by policy denial | The authenticated-Chrome lane is genuinely blocked; repeated retries no longer add evidence. A user-authorized public-page lookup and user-supplied submission screenshot are the narrow fallback. |
| The user supplied a saved HTML of the exact official leaderboard URL; its SHA-256 is `f1328f9b68772da6a91a2cd715d649193e6f76252b47c5dafd36ece04567257c`. | directly_supported | Supplied Desktop HTML, local save timestamp 2026-08-01 22:48:41 KST | The blocked Chrome lane can be completed from a bounded, immutable read-only snapshot. |
| The saved page contains public ranks 1-100 plus one authenticated user summary row. | directly_supported | Parsed HTML table; structural checks recorded in `reports/leaderboard_readonly_receipt.json` | It supports a public cutoff and current best summary, but not full population or per-submission history. |
| The page's legend/classes place ranks 11-35 in the top-4% band and rank 36 in the top-10% band. | directly_supported | `second-group`/`third-group` row classes and page legend | The snapshot top-4% endpoint is rank 35, total 0.65686. |
| The authenticated user summary is rank 689, total 0.62366, 1-NMAE 0.87305, FICR 0.37426, with two submissions. | directly_supported | Redacted authenticated summary row in the supplied HTML | Total gap to rank 35 is 0.03320; no account identity is retained. |
| The current 1-NMAE exceeds rank 35 by 0.00092, but FICR trails by 0.06732. | directly_supported | Arithmetic from the parsed rows; displayed total formula validated to rounding tolerance | FICR is the evidence-backed primary bottleneck; preserve 1-NMAE as a guardrail. |
| At the current 1-NMAE, FICR 0.44067 matches the cutoff total and FICR 0.44695 reaches a 0.66000 safety target. | directly_supported | `FICR = 2 × target_total - current_(1-NMAE)` | Use total >=0.66000, 1-NMAE >=0.873, and FICR >=0.44695 as the safe local proxy contract. |
| The saved HTML does not contain the two individual submission-history rows. | contradicts_premise | Only one occurrence of the user score tuple; no submission-history labels/table; submission count only | Do not claim per-submission scores. A separate saved history page or screenshot would be required. |

## Synthesized Decision

- Design: `.planning/2026-08-01-leaderboard-top-4-loop/DS_v2.md`
- Status: approved by exact user message `DS@v2 승인` on 2026-08-01, then `APPROVED_THEN_TARGET_CONTRACT_INVALIDATED(DS@v2)` by the later top-20 requirement
- Recommended direction: FiCR-first cross-fitted hierarchical calibration and settlement-utility decisions on top of a diversified classical ensemble.
- Guardrail: 1-NMAE must remain at least 0.873 and cannot be traded away for unstable FICR gains.
- Safe local proxy: total >=0.66000 and FICR >=0.44695 under the approved chronology-safe evaluation.
- Claim boundary: a local proxy pass is not a Dacon top-4% result.

## Active Research Boundary

- Authorized: read-only inspection of the live repository and bounded primary-source research needed to prepare `IP@v2`.
- Not authorized: code implementation, model training, dependency installation, Dacon upload/account mutation, external datasets, pretrained weights, or remote compute.
- Memory evidence is navigation-only where it describes older state; the live repository and current receipts must override it.

## Top-20 Target Revision

| Claim | Tag | Evidence | Implication |
| --- | --- | --- | --- |
| Only the top 20 teams advance to the finals. | directly_supported for user intent; official-rule verification pending | Exact current user message | Replace the top-4% success target and verify the official rule after `SK@v3` approval. |
| Snapshot rank 20 is total 0.65971, 1-NMAE 0.87991, FICR 0.43952. | directly_supported | User-supplied HTML SHA-256 `f1328f9b...67257c` | The old 0.66000 safe target has only 0.00029 total-score margin. |
| Snapshot rank 21 is total 0.65948. | directly_supported | Same supplied HTML | Rank-20/21 separation is only 0.00023, so exact-cutoff targeting is fragile. |
| Top-20 median is total 0.66132, 1-NMAE 0.87804, FICR 0.446715. | directly_supported | Same supplied HTML, ranks 1-20 | A balanced score near 0.662 is a more defensible operating target. |
| At current 1-NMAE, total 0.662 requires FICR 0.45095. | directly_supported | Display formula arithmetic | FICR remains the largest gap, but NMAE can no longer be treated as already sufficient for every top-20 profile. |

- Revision state: `REVISION_REQUIRED(SK@v2 -> SK@v3)`.
- Revised skeleton: `.planning/2026-08-01-leaderboard-top-4-loop/SK_v3.md`.
- Recommended provisional operating target: total >=0.66200, 1-NMAE >=0.87500, FICR >=0.44900.
- Approval: exact `SK@v3 승인` received with clarification that research spans every competition-data module from feature construction through prediction-model development.
- Active: official-rule verification, live-repository/data audit, bounded primary-source module research, and `DS@v3` synthesis.
- Still excluded: `IP@v2`, implementation, training, dependencies, uploads/account mutation, external data/pretrained weights, and remote compute until later gates.

## Live Repository Audit for DS@v3

| Claim | Tag | Evidence | Design implication |
| --- | --- | --- | --- |
| The repository already has explicit modules for canonicalization, data quality, chronology/leakage guards, weather/physics features, tree challengers, official scoring, calibration, blending, experiment promotion, inference, and submission validation. | directly_supported | Live `src/baram/**` and `tests/**` inventory at commit `383265a` | DS@v3 should extend the current modular pipeline rather than replace it wholesale. |
| The pinned runtime already supports LightGBM and scikit-learn; CatBoost and XGBoost are optional challengers. | directly_supported | `pyproject.toml` | The primary next design can stay classical-first without adding a deep-learning dependency. |
| The current feature surface is very wide and already includes spatial aggregates, quantiles, derived wind speed/direction, calendar cycles, and lead hour for GFS and LDAPS. | directly_supported | `reports/development_controls.json` feature manifest | Generic “add wind speed/direction” is not a useful recommendation; research must identify missing interactions, calibration structure, target transforms, or distributional features. |
| The frozen candidate is a reproducible local artifact, but its one-use 2024 lockbox total is 0.627605 with FICR 0.380652. | directly_supported | `reports/final_audit.json` | The new target gap is dominated by settlement performance; 2024 cannot be reopened to choose new decisions. |
| The worktree contains only the current planning area and two leaderboard research reports as untracked reviewer-facing artifacts. | directly_supported | Live `git status --short` | No prior source edit is being mistaken for SK@v3 work. |

The first broad receipt query produced output truncation because one manifest embeds the entire feature list. It did not change state. Subsequent receipt inspection must use targeted `jq` projections rather than printing whole manifests.

### Module Gap Map from Current Code

| Module | Current live implementation | Evidence-backed research question |
| --- | --- | --- |
| Data contract | Exact ZIP membership/hash, timestamp parsing, availability lead, 9-grid GFS/16-grid LDAPS cardinality, train/test schema equality | Preserve fail-closed contract; add drift/regime reports without weakening availability rules. |
| Data quality | Missing-cell counts, label sign/capacity checks, SCADA-gap audit; SCADA power fields quarantined | Quantify missingness and regime coverage by fold/source/group; keep quarantined fields out unless a later contract proves inference availability. |
| Spatial/weather features | Per-timestamp global mean/std/min/max/q10/q50/q90; derived u/v speed and direction; `spatial_mode=global_only` | Test fixed spatial summaries, grid-relative contrasts, source disagreement, vector-resultant concentration, and group-specific interactions from supplied grids. |
| Physics | Fixed 117 m extrapolation with alpha 0.2, 100-80 m shear, dry-air density, rho-v-cubed proxy | Derive data-supported shear exponent, stability/density interactions, source-to-source physics residuals, and physically bounded monotone effects. |
| Validation | Whole 24-hour issuance batches; 2023 Q2/Q3/Q4 expanding folds; label cutoffs; 2024 one-use lockbox | Add nested/cross-fitted decision evaluation and fold/regime stability without reopening 2024. |
| Deterministic models | Capacity-normalized L1 LightGBM plus bounded XGBoost/CatBoost challengers, chronology-safe early stopping | Diversify loss/architecture/group sharing only where OOF residual diversity and stability justify it. |
| Ensemble | Exactly two parents; independent group weights on a 0.05 convex grid | Support more parents only after cross-fitted residual-diversity screening; avoid same-fold weight selection. |
| Calibration/decision | Per-group scale/offset/cap grid; cross-fitting exists; residual utility uses coarse median shifts by group/lead/wind state | Replace point-only residual shifts with conditional predictive distributions and direct threshold-hit expected utility, with 1-NMAE guardrails. |
| Official evaluation | Exact capacity-normalized error, actual >=10% capacity admission, 6%/8% settlement tiers, equal group averaging | Optimize the discontinuous FICR component using honest OOF distributions; report group and boundary slices, especially group 3. |

The combined source read for model/decision modules was partially truncated after roughly 11k tokens. It exposed the relevant contracts but not every line; any exact implementation claim in DS@v3 will be rechecked with narrow symbol-level reads.

### Reviewed Baseline Numbers

- 2023 pooled control scores: climatology total `0.396027`, physics proxy `0.468463`, random forest `0.566609`.
- One-use 2024 lockbox: calibrated tree total `0.627605`, `1-NMAE 0.874558`, `FICR 0.380652`; CatBoost total `0.607703`; random-forest control total `0.590336`.
- The calibrated tree's group FICR is `0.419939 / 0.423500 / 0.298518` for groups 1/2/3, confirming group 3 as the most severe settlement-utility weakness in the only consumed lockbox evidence.
- The selected lockbox calibration hit the edge of the coarse positive-shift grid for groups 1 and 3 (`scale 1.04`, `offset +0.02 capacity`), and positive offset for group 2 (`scale 1.02`, `offset +0.02`). This is a strong design signal that a richer conditional calibration/decision family should be investigated, but it is not permission to tune on 2024.

A second receipt projection accidentally retained the champion's embedded feature manifest and was truncated. Future projections must explicitly reconstruct only scalar champion fields; avoid selecting whole nested objects.

## Official Dacon Verification (2026-08-01)

| Official claim | Tag | Primary source | Consequence |
| --- | --- | --- | --- |
| Score is `0.5 × (1-NMAE) + 0.5 × FICR`; only rows with actual generation >=10% of group capacity are evaluated; group metrics are equally averaged. | directly_supported | Dacon evaluation page and official metric code-share page | The local evaluator's central contract is aligned; every decision experiment must preserve both components and the admission rule. |
| Public uses a sampled 40% and Private uses the remaining 60%; first-stage ranking is Private Score 100%. | directly_supported | Dacon evaluation page | Public rank is a guide, not proof of finals qualification; stability and distribution-shift controls matter. |
| Private top 30, including 10 reserves, submit artifacts; after code validation, top 20 proceed to the offline second-stage evaluation. | directly_supported | Dacon evaluation page | The user's top-20 requirement is official, while top-30 is the artifact-submission pool. The operating target should seek a buffer, not rank-20 equality. |
| Final second-stage score is 50% Private leaderboard and 50% offline qualitative evaluation; top 10 win. | directly_supported | Dacon evaluation page | Reproducibility, technical rationale, problem solving, and practical applicability must be designed alongside raw score. |
| Training and inference code must be separated, versions declared, and the submitted code must reproduce Private Score within tolerance. | directly_supported | Dacon evaluation page | M0/M8 reproducibility is a finals requirement, not optional project hygiene. |
| Feature availability is governed by when information became available, not the forecasted timestamp; later forecasts, observations, corrected data, or reanalysis are prohibited. | directly_supported | Dacon rules page | Keep issuance-batch cutoffs and forbid post-issuance features in every module. |
| Official test data contains only GFS and LDAPS; SCADA appears only in train. | directly_supported | Dacon data page | Raw SCADA lags/features cannot appear in test-time inference. It may only inform training-time structure if transformed into inference-available static knowledge under a later explicit design, and the current safer recommendation is to keep it quarantined. |
| Remote model inference APIs are forbidden; pretrained weights have license/date rules. | directly_supported | Dacon rules page | The existing competition-data-only/no-pretrained/no-remote boundary is stricter and simpler to reproduce; retain it. |

Primary URLs recorded for the research report:

- `https://dacon.io/competitions/official/236727/overview/evaluation`
- `https://dacon.io/competitions/official/236727/overview/rules`
- `https://dacon.io/competitions/official/236727/data`
- `https://dacon.io/competitions/official/236727/codeshare/14035`

## Primary-Literature Research Log (bounded)

The literature is used only to choose methods. No paper dataset, forecast, pretrained weight, or external feature enters the competition pipeline.

| Source | Primary contribution | BARAM applicability | Guard / non-applicability |
| --- | --- | --- | --- |
| Landry et al. (2016), *Probabilistic gradient boosting machines for GEFCom2014 wind forecasting* | Winning wind-track method fit quantile GBMs independently by zone and quantile, with smoothing of the dominant weather signal and cross-sectional information. | Strong support for capacity/group-specific quantile boosting and cross-group/source features from supplied GFS/LDAPS. | It is benchmark transfer, not a claimed score comparison; do not copy external data or assume its competition metric matches FICR. |
| Ke et al. (2017), *LightGBM* | Efficient GBDT for high-dimensional data. | Supports keeping LightGBM as the deterministic and quantile workhorse for the very wide supplied-weather table. | The paper does not establish BARAM superiority or threshold utility. |
| Prokhorenkova et al. (2018), *CatBoost* | Ordered boosting reduces prediction shift from target-dependent procedures. | Retain as a diversity challenger, especially if categorical regime/site encodings are added. | Current BARAM inputs are mostly numeric and CatBoost lost on the consumed lockbox; it is not the default champion. |
| Meinshausen (2006), *Quantile Regression Forests* | Nonparametric conditional quantiles/full conditional distribution from forests. | Useful low-complexity distributional baseline for expected threshold-hit calculations. | High-dimensional runtime and tail calibration must be measured; not automatically superior to boosted quantiles. |
| Duan et al. (2020), *NGBoost* | Fits a conditional distribution via natural gradients and proper scoring rules. | Conditional option if boosted quantiles are poorly calibrated and an approved dependency/compute budget exists. | A parametric distribution can be misspecified for zero-bounded/capacity-limited wind power; defer behind quantile trees. |
| Gneiting (2011), *Making and Evaluating Point Forecasts* | The optimal point forecast depends on the evaluation loss; a forecast functional must match the scoring rule. | Direct conceptual basis for deriving the point action from a predictive distribution under the Dacon total/FICR utility instead of defaulting to a mean/median. | The Dacon score is discontinuous and composite, so the Bayes action must be computed numerically and validated cross-fitted. |
| Elmachtoub & Grigas (2021/2022), *Smart Predict, then Optimize* | Decision structure can be incorporated into prediction objectives; SPO+ is a tractable surrogate for suitable optimization problems. | Supports a decision-aware layer and measuring downstream utility rather than prediction error alone. | Dacon FICR is not the paper's linear-objective setting; use as design motivation, not a direct SPO+ implementation claim. |

Current source count: 7 primary research papers. Five slots remain under the SK@v3 cap of 12.

| Source | Primary contribution | BARAM applicability | Guard / non-applicability |
| --- | --- | --- | --- |
| Fang & Chiang (2016), *Improving supervised wind power forecasting models using extended numerical weather variables and unlabelled data* | Regional and environmental NWP variables can supplement target-site meteorology in supervised wind-power models. | Supports retaining supplied-grid/source context, spatial contrasts, and broader physical variables instead of collapsing every field to one site-like mean. | Their use of unlabelled external/extra data is not adopted; only the supplied train/test NWP tables may be used. |
| Bergmeir, Hyndman & Koo (2018), *A note on the validity of cross-validation for evaluating autoregressive time series prediction* | Studies when CV is valid for autoregression and compares evaluation schemes. | Reinforces that dependence assumptions must be explicit and that chronological/issuance-batch extrapolation remains the relevant BARAM evaluation surface. | The paper does not authorize random K-fold for this forecast-availability problem; current expanding issuance-batch folds remain the stricter choice. |
| Bates & Granger (1969), *The Combination of Forecasts* | Forecast combinations can improve error when weights use past forecast errors. | Supports blending genuinely diverse, independently validated parents and learning weights only from preceding OOF errors. | It does not support unlimited stacking or same-fold weight fitting; correlation and stability screening are required. |
| Lim et al. (2021), *Temporal Fusion Transformers* | Multi-horizon architecture handles known-future inputs, historical exogenous inputs, gating, and interpretable attention. | Structurally compatible with 24-hour forecast batches and known-future NWP, so it is the most defensible deep challenger. | Current local evidence shows no approved GPU/value case; dependency and compute costs keep it conditional and behind classical/distributional modules. |
| Romano, Patterson & Candès (2019), *Conformalized Quantile Regression* | Adaptive prediction intervals combine quantile regression with split conformal calibration under exchangeability. | Use foldwise coverage and interval-width diagnostics to calibrate/check conditional quantiles before expected-utility decisions. | Time dependence/regime shift breaks a simple exchangeability guarantee; never claim exact coverage on Private data and keep chronological calibration splits. |

Primary research source cap reached: 12/12. Additional papers found during search are excluded from design authority unless `SK@v3` is revised.

### Literature-Supported Synthesis

1. **Feature module:** preserve supplied NWP spatial/source information with fixed, reproducible contrasts and vector summaries; avoid indiscriminate feature multiplication.
2. **Model module:** keep LightGBM as deterministic/quantile workhorse; retain CatBoost only for measured residual diversity; add a nonparametric distributional baseline before any new deep dependency.
3. **Uncertainty module:** cross-fit conditional quantiles/distributions and diagnose calibration by group, lead, season, wind regime, and threshold proximity.
4. **Decision module:** for each row, numerically select the point prediction that maximizes expected `0.5 × (1-NMAE) + 0.5 × FICR` under the estimated conditional distribution, then apply hard group/fold stability guardrails.
5. **Ensemble module:** only blend parents whose preceding-OOF residuals are sufficiently diverse; all weights and decision policies must be trained on earlier folds and evaluated on later unseen folds.
6. **Deep module:** TFT is a conditional challenger, not the main path. It activates only if classical residuals retain stable batch/lead temporal structure and a later plan approves dependencies/compute.

Primary-page verification succeeded for LightGBM, CatBoost, Quantile Regression Forests, NGBoost, Smart Predict-then-Optimize, Fang & Chiang, Bergmeir et al., TFT, and Conformalized Quantile Regression. Publisher opens for Landry et al. and Bates & Granger returned internal fetch errors, and the Gneiting publisher page returned HTTP 403; their primary publisher search abstracts/metadata remain sufficient for the bounded claim, and no further retry is needed.

The official evaluation page labels an “offline presentation evaluation items and points” section, but its table is embedded as an image and omitted from text extraction. The surrounding official prose confirms the four qualitative dimensions from the description page—task understanding, technical excellence, problem solving, and applicability—but exact per-item point weights remain unverified until the embedded official image is decoded once.

### Official Offline Rubric — Embedded Image Verified

| Item | Points | Official criterion |
| --- | ---: | --- |
| Task understanding | 20 | Understanding of the wind-generation forecasting objective, data characteristics, and evaluation metrics |
| Technical excellence | 30 | Excellence of preprocessing, feature construction, modeling, validation strategy, and predictive performance |
| Problem-solving | 15 | Appropriateness of the problem-solving approach, model-improvement strategy, and analysis method |
| Applicability | 20 | Practical utility for real wind-generation forecasting work and subsequent research |
| Presentation completeness | 15 | Structure, clarity of explanation, and Q&A response |
| **Total** | **100** | Official embedded evaluation rubric |

This resolves the earlier exact-weight uncertainty. The rubric makes the user's module-by-module research request strategically aligned with the highest-weight item: technical excellence (30), while reproducible problem framing, utility analysis, and report quality address the remaining 70 points.

The official settlement image also verifies the step function used locally: capacity-relative absolute error <=6% earns 4 won/kWh, >6% and <=8% earns 3 won/kWh, and >8% earns zero.

## Data-Quality and Prior-Decision Evidence

| Finding | Tag | Evidence | DS@v3 implication |
| --- | --- | --- | --- |
| Prepared rows are 78,912 train feature rows, 26,280 test feature rows, and 8,760 submission timestamps. | directly_supported | `artifacts/manifests/prepare.json` | Preserve exact row/key contracts; group-expanded inference must remain 3 x 8,760. |
| GFS has no blank cells; LDAPS test has 752 blank cells across 48 grid rows at exactly three forecast timestamps. | directly_supported | Same prepare manifest | Add source-specific missingness and GFS-vs-LDAPS fallback/disagreement features; imputation must remain fold-fitted and deterministic. |
| Group 3 labels have 8,766 nulls while groups 1/2 have about 100 each; group 3 also has 38 above-capacity observations. | directly_supported | Same prepare manifest | Shared models need label-availability masks and group-age/regime awareness; prediction clipping to nominal capacity is not automatically valid. |
| SCADA series are complete internally but power fields remain explicitly quarantined because no SCADA exists in test. | directly_supported | Prepare quality receipt + official data page | Do not add target-lag/SCADA features to inference. |
| Chronology-safe coarse calibration lifted 2023 Q3-Q4 pooled total from `0.569738` to `0.590169`, mainly by FICR `0.289892 -> 0.326750`; two-parent blending reached only `0.570645`. | directly_supported | `reports/decision_layer.json` | Conditional calibration/utility is the first implementation priority; ensemble breadth is secondary unless residual diversity is proven. |
| Prior residual utility did not activate; the old method only supported coarse statewise median shifts. | directly_supported | `reports/decision_layer.json` + live utility module | Build the next decision module from predictive distributions and expected official utility, not by merely enlarging the median-shift grid. |
| Prior deep tier was deferred under IP@v1; no GPU/PyTorch evidence was collected. | directly_supported | `reports/deep_tier_decision.json` | TFT remains conditional and requires a later approved value/compute gate. |

## Immutable Data and Baseline Audit

- Reverified immutable hashes: `open.zip` = `920be0c...f720b`; `baseline.ipynb` = `712b26f...f48c`.
- The official train horizon is 2022-2024 and test horizon is all 8,760 hours of 2025. Each daily forecast batch is initialized at 09:00 KST, available at 13:00, and covers the next 01:00-00:00 24-hour operating day.
- LDAPS provides 16 grids at about 1.5 km resolution; GFS provides 9 grids at about 0.25 degree resolution. Their coordinates, grid identities, and issue timestamps are competition-supplied inference-available structure.
- `info.xlsx` contains turbine coordinates, KPX group membership, manufacturer/model, 117 m hub height, 126 m rotor diameter, individual capacity, and group capacity. The current feature code uses a fixed 117 m proxy but does not yet exploit group/turbine coordinates to construct distance-weighted supplied-grid summaries.
- The official baseline notebook globally averages each weather variable by timestamp, adds simple calendar features, median-imputes, trains independent Random Forests, clips to capacity, and uses `n_jobs=-1`. It has no chronological validation, official metric loop, availability guard, spatial/group physics, uncertainty model, or FICR-aware decision layer.
- The current repository is already a major engineering advance over the notebook: exact availability batches, local worker cap, broader aggregates/physics, chronology folds, exact metric, calibration, receipts, and reproducibility. DS@v3 must therefore compare against the repository baseline, not the tutorial notebook.

### Turbine and group topology verified from the immutable archive

- `info.xlsx` contains 17 turbines: VESTAS V126 units 1-12 at 3.6 MW each and UNISON U136 units 1-5 at 4.2 MW each.
- KPX groups 1 and 2 each contain six VESTAS units with nominal group capacity 21.6 MW; group 3 contains five UNISON units with nominal group capacity 21.0 MW.
- All turbines have 117 m hub height; rotor diameters differ by fleet (126 m versus 136 m). Coordinates form a compact but non-identical spatial footprint.
- This directly supports group/turbine-distance-weighted LDAPS/GFS summaries, fleet/model indicators, rotor-aware density/power proxies, and source-disagreement features. It does not authorize external terrain, weather, or turbine-power-curve data.

## DS@v3 synthesis decision

- Selected primary route: competition-supplied spatial geometry and physics features -> capacity-normalized LightGBM point/quantile models -> QRF distribution benchmark -> cross-fitted expected official-utility decision -> residual-diverse constrained ensemble only if additive.
- The strongest local historical signal is the chronology-safe calibration gain (`+0.020431` total on pooled 2023 Q3-Q4); the prior two-parent blend was marginal. Decision calibration therefore precedes ensemble breadth.
- TFT is methodologically relevant to the daily 24-hour known-future NWP sequence but remains a conditional challenger behind a later residual-value, dependency, and compute approval gate.
- External target ladder: hard snapshot minimum `0.65971`, intermediate `0.66033`, operating guide `0.66200 / 0.87500 / 0.44900`. Local folds cannot honestly be mapped to these values from two public summaries.
- `reports/top20_module_research.md` and `DS_v3.md` contain the full module contracts, alternatives, later bounded experiment ceiling, exact leakage controls, limitations, and approval boundary.

### DS@v3 approval transition

- Exact user message `DS@v3 승인` received on 2026-08-01.
- Authority now permits preparation of `IP@v2` only.
- It does not authorize implementation, training, dependency changes, Dacon upload/account mutation, external data/weights, remote compute/Git, or reopening the consumed 2024 lockbox.
- Live source audit confirms the plan must extend the existing `src/baram` modules and workflow/CLI rather than create a replacement repository.

## IP@v2 live implementation surface

- Exact `IP@v2 승인` was received on 2026-08-02. The approved execution surface is local T0-T8 only; all plan exclusions remain binding.
- Live preflight matched the frozen inputs, consumed-lock identity, Python 3.12 runtime, artifact budget, and repository baseline. This is authority to begin T0, not evidence that any model or local candidate has passed.

- `data/archive.py` already reads `info.xlsx` bytes but does not parse turbine metadata; add a deterministic in-memory parser and typed turbine/group table without extracting the workbook.
- `features/weather.py` currently produces global grid statistics and expands them identically across groups. Spatial aggregation belongs in a new narrow feature module and a controlled `build_weather_features` integration, not a rewrite of canonical weather contracts.
- `features/physics.py` uses a fixed 0.2 shear exponent and global GFS means. Extend it with bounded supplied-level shear/fleet/rotor proxies while retaining current features as the mandatory ablation baseline.
- `features/pipeline.py` records `spatial_mode="global_only"`; the manifest must carry the frozen spatial feature mode/config and keep fold-fitted medians.
- `models/oof.py` supports point families only. Distributional OOF output needs an explicit keyed quantile/distribution contract rather than overloading `prediction_kwh` silently.
- Existing dependencies already support LightGBM quantiles and a QRF-style tree-distribution benchmark via scikit-learn; no new dependency is required for the primary path. NGBoost and TFT remain out of scope.
- `decisions/calibrate.py` is a coarse scale/offset grid and `utility.py` is median-residual shifting. A new expected-utility policy must consume preceding-fold predictive distributions and keep the exact scorer unchanged.
- `decisions/blend.py` enforces exactly two parents. Any extension to up to three parents must preserve keyed inputs, convex weights, lineage hashes, and the existing two-parent behavior/tests.
- `experiments/promotion.py` currently accepts tiny positive deltas and majority folds. IP@v2 must add predeclared materiality/run-variance, NMAE/FICR, worst-fold, and worst-group guardrails before training.
- `workflows.py` and `cli.py` already orchestrate audit/prepare/split/backtest/select/lockbox/final/reproduce. The v2 implementation should add bounded stages through these surfaces while hard-disabling the already consumed lockbox path.
- The current LightGBM stage evaluates a 16-config grid across two architectures on the first fold, promotes six to full folds, and repeats three finalists across three seeds. IP@v2 must replace/route this broad old search with the DS@v3 stage ceilings rather than stack another unbounded loop on top.
- Current `run_select` cross-fits coarse calibration and a two-parent control/tree blend, then freezes at most three candidates. The v2 plan should preserve that lineage/freeze discipline while adding distribution and expected-utility evidence as separately hashed parents/policies.
- The CLI currently exposes backtest stages `controls`, `lightgbm`, `ablation`, and `challengers`. New `spatial`, `distribution`, and `decision-v2` stages (or exact equivalents) need explicit routing and integration tests; no lockbox command may be invoked.
- Existing tests already enforce exclusive lockbox consumption and a tiny-process hash reproduction path. V2 must extend these tests with a hard closed-lockbox preflight and distribution/policy lineage checks.

### IP@v2 threshold evidence

- Observed three-seed Total ranges for the prior finalists are `0.0011279162`, `0.0030142387`, and `0.0025112677`; the largest is `0.0030142387`.
- IP@v2 fixes both minimum material Total lift and maximum finalist seed range at `0.0035`, above that observed variation before any new training.
- The prior blend delta `+0.0009070327` would fail; the prior cross-fit calibration delta `+0.0204318018` would pass.
- Decision promotion additionally requires FICR `+0.0070`, pooled `1-NMAE >= -0.0010`, every later fold positive, every group component above `-0.0010`, and group 3 nonnegative.
- The primary distribution path uses installed LightGBM/scikit-learn only. TFT, NGBoost, installation, GPU, and Colab remain excluded.

## IP@v2 implementation checkpoint — 2026-08-02

- T0-T8 interfaces now exist locally. The v2 workflow is separated from the legacy lockbox path and accepts only `baram-v2-20260801-01`.
- Distribution selection compares seven fixed monotone quantiles against the chronological empirical-residual control and the frozen point parent. Failed probabilistic gates fall back to q50/coarse calibration without claiming expected-utility promotion.
- Decision policies fit calibration, state support, and physical upper bounds on preceding OOF rows only; application does not read validation actuals. The final policy freezes from development OOF evidence before the 2022-2024 fit.
- Ensemble weights fit on preceding common OOF folds and evaluate on later folds; two/three-parent candidates must also pass residual-diversity and group-3 guardrails.
- Final fitting may consume supplied 2024 labels only as ordinary training rows. Its receipt explicitly records `metric_calls=[]` and `new_2024_evaluation=false`; no final-fit evidence flows back into selection.
- Current verification is implementation-only: 149 tests and Ruff pass. No new local model score, Dacon score, rank, upload, or top-20 proof exists yet.

### Frozen v2 execution lineage

- Repeated preflight summary SHA-256: `65c4eeb9b67df78f117651d082e3df494721265fb56ed168d2cecac3ccce7f52`.
- Source-tree SHA-256: `4149b4471ee503eaaa43f25eee67a3a3c21297140489a3cde1cf2a6ebc2adf7a`; all subsequent modeling stages must match it.
- Prepared v2 surface: 820 numeric features, 78,912 training rows, 26,280 test rows, and exact three-group expansion.
- Development folds are `dev-2023-Q2`, `dev-2023-Q3`, and `dev-2023-Q4`; their train/validation batch counts are 454/91, 545/92, and 637/92. No development fold contains 2024.

## Active local loop checkpoint: M50-M96

- The dominant inference bottleneck is competition-supplied NWP to site-wind uncertainty. A diagnostic using observed validation SCADA wind reached Q4 Total `0.839888`, but SCADA remains quarantined because it is unavailable in test.
- Fixed all-weather site-wind regression, CatBoost, sequence features, recency weighting, broader meteorology profiles, leaf-count sweeps, generation-weight variants, and direct conditional quantiles did not beat the fixed 2%-bin classifier consistently.
- The chronology-safe M95 family selector is the current reproducible strict champion at Total `0.6236629962`, 1-NMAE `0.8537210690`, and FICR `0.3936049235`. Q2 is fixed; Q3 selects only from Q2; Q4 selects only from pooled Q2-Q3.
- M93 conditional power quantiles alone were rejected. A Q4-only 75:25 blend reached `0.6403567`, but preceding-fold selection did not retain the blend, so it was not promoted.
- M96 cumulative ordinal classification preserves generation-bin order and is the strongest same-fold architecture screen: Q2 `0.6242568`, Q3 `0.5970005`, Q4 `0.6409572`; pooled same-fold diagnostic `0.6270291`. It still requires fixed-iteration chronology-safe integration before promotion.
- Preceding-fold piecewise residual calibration reduced the M95 pooled score to at most `0.6206848`; reject this calibration family.
- The user changed the final completion gate to chronology-safe local Total strictly greater than `0.66000`. No Dacon upload, 2024 re-evaluation, external data, remote compute, or pretrained weight use occurred.

## Active local loop checkpoint: M97-M141

- M107 is the current reproducible chronology-safe champion at Total `0.6268876959`, 1-NMAE `0.8549435658`, and FICR `0.3988318261`. Its fold totals are Q2 `0.620962459`, Q3 `0.597601241`, and Q4 `0.642891756`; Q3 is the main temporal bottleneck.
- Same-fold architecture screens continue to show an information ceiling below the target: M115 XGBoost scored Q2 `0.642766`, Q3 `0.608356`, and Q4 `0.647698`; M138's 2.5%-wide bins raised Q4 only to `0.648482`.
- Group 3 is the main cross-sectional bottleneck. M131's compact, preceding-only pseudo labels raised Q3 group 3 to `0.596533`, but the resulting full-fold diagnostic was only `0.603652`.
- The weather-transfer pseudo-label mapper improved its internal chronological holdout from compact `0.679568` to `0.756541`, but shifted the missing-2022 pseudo-target mean too far and reduced downstream performance. Simple rich/compact anchoring and uncertainty weighting did not fix the extrapolation gap.
- Same-season training helped Q3 group 2 slightly (M137 `0.624559` versus M115 `0.620788`), supporting a bounded experiment that redistributes fixed pseudo-label mass toward the validation season without exposing validation labels.
- Strict cross-model policy selection, sequence residuals, daily multi-output, physics-only, cross-group context, direct metric-Q regression, analog ensembles, richer pseudo labels, and separate group-specific XGBoost did not improve the strict champion.
- Official turbine-power aggregation is highly correlated with actual generation, especially group 3, but it is contemporaneous with the target horizon and unavailable at test inference. It remains diagnostic-only and is not used as a feature.
- The next bounded lane is season-weighted compact pseudo labeling followed by chronology-safe probability/decision calibration. The active completion gate remains strict pooled 2023 OOF Total `>0.66000`; components are diagnostic only.

## Active local loop checkpoint: M142-M163

- Season-weighted compact pseudo labels helped the missing-history group but did not close the gap: M142 reached Q3 group 3 `0.603108` and full Q3 `0.605843`.
- Source-rank distributions were the strongest new same-fold family: M149-M151 scored Q2/Q3/Q4 `0.644575 / 0.617771 / 0.649404`. Strict M152-M154 integrations failed to beat M107 and peaked at `0.624908`.
- M161's inference-safe group-3 pseudo-distribution mixture improved Q3 to `0.621313`; output blending peaked at group-3 `0.613999`. The gain is real but too small for a policy-only route to `0.66000`.
- Supervised daily PLS, ExtraTrees distributions, turbine decomposition, CatBoost multi-quantiles, target smoothing, cross-group mapping, and label-free density-ratio adaptation all failed unseen-Q3 screening.
- Q4 diagnostics expose the remaining information bottleneck: no archived group-3 candidate exceeds `0.612167`, and no Q4 candidate exceeds `1-NMAE=0.854269`. New predictive representations are required; incremental scale/shift/smoothing searches are not a sufficient primary lane.
- M107 remains the reproducible chronology-safe champion: Total `0.6268876959`, `1-NMAE=0.8549435658`, `FICR=0.3988318261`. Completion requires a reproducible chronology-safe Total strictly greater than `0.66000`; 2024 remains closed.

## Active local loop checkpoint: M164-M169

- Direct XGBoost MAE regression (M164) improved Q3 `1-NMAE` to `0.861935` after blending but scored only `0.605127`; regression-error reduction alone does not recover the settlement tiers.
- A wider/deeper source-rank model (M165) improved Q3 from M149's `0.617771` to `0.620085`, with group totals `0.624236 / 0.635697 / 0.600323`. The information gain is too small for full-fold expansion.
- Cross-fitted empirical site-wind residual classes (M166), CatBoost multiclass (M167), supplied-geometry wake sectors (M168), and action-conditional direct official utility (M169) peaked at Q3 `0.605868 / 0.607579 / 0.608777 / 0.606138` and were rejected.
- M169's direct utility target used 781,460 expanded training actions but still produced `FICR=0.352039`; the failure is not caused by an insufficient action grid. Better conditional weather/source uncertainty is required.
- M107 remains the only strict champion at `0.6268876959`. The next bounded lane calibrates GFS-only and LDAPS-only site wind from preceding SCADA via cross-fitting, then exposes only their inference-time predictions and disagreement to the power classifier.

## Active local loop checkpoint: strict audit and M170-M178

- M107 is no longer accepted as a chronology-safe champion. Its historical score is retained only as a legacy diagnostic because the prior fold mask could admit an incomplete issuance batch and its Q2 and Q3/Q4 predictions came from different recipes. The honest corrected pooled champion is unknown until a homogeneous strict rebuild completes.
- The canonical strict training condition is batch-level: the maximum forecast timestamp of every included issuance batch must be earlier than the minimum `data_available_kst_dtm` of the validation batch. Row-wise `forecast_kst_dtm < validation_start` checks are insufficient.
- The expected-utility objective must value settlement revenue using the official group-level eligible-generation mean. Replacing the row-wise denominator fixes a material objective mismatch while leaving the official evaluator unchanged.
- M170-M178 did not produce a promotable Q3 candidate. The calibrated ordinal CDF (M177) reached only `0.590591`; the inference-safe per-turbine NWP-to-wind-to-power stack (M178) reached only `0.584654`, despite strong observed SCADA-to-label agreement.
- Seven archived Q3 candidates have highly correlated predictions (`0.967-0.995`) and absolute errors (`0.864-0.979`). Even a same-fold groupwise convex oracle reached only `0.626260`, so static blending cannot bridge the target gap.
- M173's apparent SCADA-pseudo result used zero pseudo rows. It remains a source-rank/pseudo diagnostic, not evidence for SCADA pseudo-label value.
- Q3 group 3 remains the clearest bottleneck: historical M107 group-3 Total was `0.571448`, FICR `0.289595`, and actual-weighted 6% coverage only `23.2%`. High-output rows are systematically underpredicted.
- The next bounded route is an uncertainty-aware group-3 pseudo-label distribution trained only from preceding group 1/2 and available group-3 labels, followed by proper-score calibration and the corrected group-denominator Bayes action. All outer-fold policy choices must be frozen from earlier inner folds.
- A read-only audit once materialized the full labels parquet only to count dates/rows; it computed no 2024 value or score. Treat this as a protocol incident and require a physical pre-2024 predicate for every future development loader.
- M179 used the new physical pre-2024 loader and selected its width/family/action on a preceding 60-day inner holdout. Coarse settlement-band classification did not transfer: inner `0.617408` became outer-Q3 `0.588787`. Direct hard-band simplification is rejected; retain calibrated fine-grained distributions instead.

## Active local loop checkpoint: M180-M184 and representation probes

- M180-M184 all kept Q3 labels outside model fitting and configuration selection, but none exceeded `0.600582`. The 25-grid expert stack was especially unstable: its inner selection score was `0.599410`, while outer Q3 fell to `0.576452`.
- M183's boosted-leaf conditional distribution is the strongest new fully strict Q3 candidate at `0.600582 / 0.859021 / 0.342143`, still `0.059418` below the Total gate. Leaf-neighborhood density estimation therefore does not solve the conditional-settlement defect by itself.
- M165's validation-label oracle smoothing and transition-HMM diagnostics reach only about `0.632` and `0.625` respectively. Even optimistic structured post-processing cannot bridge the gap, so future work must add predictive information rather than tune output trajectories.
- The Q3 wind signal itself remains present: leading supplied wind-speed features retain Pearson correlations around `0.73-0.80` across groups. However linear-leaf, MLP, correntropy, monotone, and aggressive generation-weight probes all scored below `0.597`; the bottleneck is not fixed by generic smoothness, robustness, or high-output emphasis.
- The active completion condition is still one reproducible, complete-issuance-batch chronology-safe Total strictly greater than `0.66000`. No 2024 row was materialized by these new runners, and no Dacon, external-data, remote-compute, or pretrained-model action occurred.

## Active local loop checkpoint: M185-M186 and daily-profile probe

- M185's raw-grid Fourier representation selected many spectral variables in every group, but its Q3 Total was only `0.580314`. This closes fixed low-pass/high-pass summaries as a standalone route; the cited geometric-Fourier literature uses learned cross-channel operators, which the fixed summaries do not reproduce.
- M186's conditional Gaussian mixture did not rescue weak point forecasts. Group 2 improved from `0.592009` to `0.604019`, while group 1 fell from `0.607355` to `0.600992` and group 3 fell from `0.544794` to `0.542059`. The overall `0.582357` result closes residual-density smoothing as the primary route.
- The random-issuance Q3 point-regression diagnostic reached only `0.583521` despite seeing other Q3 batches during training. The weakness is not explained solely by a pre-Q3/Q3 season boundary; metric-aware target-distribution modeling remains necessary.
- Actual group-1/2 generation contains enough contemporaneous information to map group 3 above `0.71` Total, but substituting forecasted group-1/2 outputs collapses the same mapper below the direct group-3 model. Cross-group coupling is a useful shared-training signal, not an inference shortcut.
- The high-dimensional 24-hour CatBoost variant was stopped after more than 30 minutes without a completed result. A bounded latent version retained 77.54% of input variance and 97.43% of target variance, but its best Ridge/CatBoost blend reached only `0.585086 / 0.862675 / 0.307496`. Daily profile compression is therefore closed as a primary route.
- The next experiment targets a concrete extrapolation defect rather than another post-processing search: raw `operating_year`, `day_of_year`, and `month` fields are removed while cyclic calendar encodings remain, and distribution/policy settings are selected from preceding same-season data before a single untouched-Q3 evaluation.
- M187 disproved that route as configured. It scored only `0.578459 / 0.840071 / 0.316847`; group totals were `0.557261 / 0.607443 / 0.570674`. Every earlier-window selector preferred pure expected-absolute-error action (`gamma=0`), so past seasonal settlement structure did not transfer to Q3 2023. Restore raw calendar fields for later models and do not repeat this policy-transfer design.
- XGBoost's experimental vector-leaf representation did share useful information: M188 beat its otherwise matched independent-output control by `+0.006362` Total. The absolute score was still only `0.583855`, and group 3 fell to `0.561964`, so shared multi-target splits do not supply the missing predictive information. Do not expand this experimental branch.
- Bounded-target transformations do not repair the defect. Across six fixed transforms, the strongest point result was only `0.584494` from a raw/square-root half blend, while raw itself was `0.582322`. The missing settlement coverage is not primarily a variance-stabilization problem.
- M189 establishes the cleanest current source-rank baseline: `0.606307 / 0.851853 / 0.360762`, with group totals `0.606432 / 0.624829 / 0.587660`. Its own same-fold policy oracle is only `0.609457`, and a wider pair-blend oracle is only `0.619098`; neither policy search nor static diversity can bridge the gate.
- Group-3 turbine output curves remain broadly stable across 2023 months, while the Q3 wind distribution shifts lower. Same-season shared NWP-to-SCADA weighting worsened site-wind MAE, but a CatBoost/ExtraTrees mean reduced the fixed group-3 site-wind diagnostic MAE to `1.064574`. This small teacher improvement merits one strict downstream screen, not a broad family expansion.
- M190 completed that strict downstream screen and decisively rejected the hypothesis. Its group-3 Total was `0.372765`, the same-fold group-3 oracle was only `0.447533`, and substituting it into M189 produced `0.534675` overall. A small site-wind MAE gain is therefore not aligned with the conditional generation distribution required by the official settlement metric; do not expand this teacher family.
- M191 shows that changing tree partitions to a standardized neural representation is insufficient. Its low validation classification accuracy was still converged and seed-averaged, yet the official Total was only `0.574304`; a fixed M189 blend also lost to M189. The next neural attempt, if any, must encode target order or settlement geometry rather than repeat plain multiclass cross-entropy.
- M192's graph spectrum was structurally valid and selected frequently, so the null result is not caused by unused features. It helped group 2 but sharply hurt group 3, confirming that generic spatial-field compression does not repair the short-history group-3 transfer problem. Retain the existing turbine-weighted geometric features and do not expand graph/DCT modes.
- M193 encoded generation order explicitly through a shared cumulative survival network, but its FICR fell to `0.284495`. Together with M191, this closes shallow neural representations: neither unordered cross-entropy nor ordinal survival learned a useful conditional settlement distribution from the strict history.
- M194 is the first new representation in this wave to exceed M189: row-level PLS latent transfer raises Total by `+0.005412` and group-3 Total by `+0.014576`. Shared axes dominate selection, but donor12 axes appear in every source model. Because calibration and blend oracles remain below `0.64`, future work must improve the latent conditional representation itself.
- M195 improves the overall score by another `+0.001641`, but its group-3-specific PLS axes transfer their gain to groups 1/2 rather than group 3. The next PLS variant should retain group-1 and group-2 donor targets as a simultaneous vector instead of collapsing them into one stacked scalar response.
- M196 confirms that simultaneous donor structure is useful but site-specific: 11/21/28 new features were selected, group 2 gained strongly, and group 3 regressed. A fixed M195/M196 probability-output blend raises Total to `0.617212`, but the remaining `0.042788` gap cannot be closed by treating the donor axes as globally shared features alone.
- M198 rejects a group-3-only nonlinear head despite selecting 55/71/88 PLS features. Direct latent PLS forecasts retain correlation around `0.77-0.80` on group 3, but their best Total is only `0.544355`; both the independent classifier and linear point path fail to turn that covariance into settlement coverage.
- M199 shows that swapping the XGBoost source-rank head for LightGBM/CatBoost multiclass heads does not add useful error diversity. Both raw boosters stayed below `0.598`, their fixed average reduced FICR to `0.336926`, and even a predeclared half-M197 blend reached only `0.606812`. The next bounded test changes the target representation itself by imposing cumulative ordering over the PLS axes.
- M200's threshold-conditioned learner is not a fair rejection of shared ordinal modeling because eight of twelve threshold tasks per row were selected from the observed target class. That creates a target-dependent sampling distribution and explains the severe probability-location bias (`1-NMAE=0.796585`). A corrected threshold sample must be independent of the label and should fit ordinary, unweighted conditional probabilities before the official expected-utility action is applied.
- M201 provides the corrected rejection: independent uniform threshold sampling repairs point location but not settlement coverage. Its raw group-3 Total stayed at `0.553541` and the half-parent group-3 Total at `0.568179`. The next representation test moves multi-threshold supervision into the PLS dimension-reduction stage while retaining the stronger XGBoost source-rank head.
- M202 proves that the target-basis axes contain supervised covariance but mostly restate information already present in the continuous PLS axes; selecting dozens of them does not improve the fixed score. A fixed PLS-space analog distribution also stayed below `0.591`, so neither target-basis expansion nor nearest-neighbor density is a primary route.
- M203 confirms that group imbalance is real but not the dominant information bottleneck. Equal weighting raises the half-blend group-3 Total by only about `0.00233` while reducing both donor groups; arbitrary group-weight tuning is not justified as a primary route.
- M204 confirms that stochastic XGBoost variance is negligible relative to structural error. Even an optimistic groupwise source/policy oracle remains below `0.620`, so neither more seeds nor nested policy tuning is a viable route to `0.66000`.
- M205 shows that omitted nonlinear atmospheric-state summaries were not the core bottleneck. They carry enough training signal to survive feature screening, but degrade outer-Q3 generalization, especially group 3. The next bounded family changes distribution estimation from class probabilities to direct conditional quantiles.
- M206 separates point location from settlement performance: its median has the strongest recent point accuracy (`1-NMAE=0.864188`) but very poor FICR (`0.303053`). Direct quantile integration does not calibrate the narrow 6%/8% settlement tiers. A cross-fitted conditional residual classifier is the remaining non-neural way to anchor a distribution on the stronger point location.
- M207 rejects that anchored residual route: even honest batch-grouped OOF residual classes do not transfer into Q3, and group 3 falls to `0.550078` before blending. The next new function family is a fixed RBF kernel over compact supervised PLS coordinates rather than another tree-derived distribution.
- M208 rejects the compact RBF-kernel route. Its support-vector fractions were `0.723 / 0.732 / 0.751`, yet raw Total collapsed to `0.550691` and the half-parent result to `0.591571`; nonlinear similarity in the fixed PLS coordinates is not the missing information. The next bounded test partitions the already strongest source-rank learner by four fixed six-hour forecast-lead regimes, addressing horizon-dependent NWP error without introducing target lags or same-fold calibration.
- M209 shows that horizon-dependent specialization is not enough: all 12 source/regime experts had 4,078-5,358 strict training rows, but the fixed half-parent score was only `0.611373`. Even an optimistic group-by-regime oracle across 16 recent candidates reached only `0.632960`, so neither static nor coarse conditional blending can reach the goal. M188's earlier vector-leaf gain motivates one new use of shared multi-output trees: regress a smooth ordered generation distribution directly on the PLS surface instead of fitting independent hard classes or quantiles.
- M210 rejects vector-leaf soft distributions as configured. Jointly regressing all 49 Gaussian-smoothed bins raised point location relative to several distribution candidates but produced only `FICR=0.341905` raw and `0.361614` after blending. Ordered-output sharing is not enough; the next bounded target formulation predicts the two settlement events and absolute-error term directly for each candidate action using the stronger PLS features.
- M211 rejects PLS-conditioned direct event learning: the half-parent blend raised `1-NMAE` to `0.861728` but FICR remained only `0.342834`, confirming that action-target reformulations are not recovering the missing conditional information. A feature audit exposes a more concrete defect: correlated-feature tree-gain screening retained only isolated representatives while omitting most raw LDAPS 10 m/50 m extrema and GFS 850 hPa wind-speed cells. Training-only per-group Pearson ranking yields a 99-feature union, so the next screen preserves that full correlated wind family rather than relying on gain-based sparsification.
- M212 validates a small but real group-3 information gain from preserving the correlated wind family: group-3 FICR rises to `0.361048` and Total to `0.602629`. M213 freezes that gain without disturbing stronger group-1/2 parents, setting a new strict high of `0.618337`. The next controlled interaction applies exact group balancing to the same feature contract; prior M203 evidence suggests only a small lift, so failure closes weight adjustment rather than triggering a sweep.
- M214 confirms the expected small group-balance effect: group 3 gains another `+0.000995` Total while donor groups regress. M215 isolates the gain and raises the strict high to `0.618668`. The shared tree still consumes depth to express site-specific thresholds on weather fields that are identical across group rows, so M216 adds explicit group-3-masked wind copies; this preserves donor training while allowing separate group-3 splits without a standalone short-history head.
- M216 disproves that explicit interaction hypothesis: doubling the correlation additions from 99 to 198 reduces raw group-3 FICR to `0.349995` and adds no donor-group value. Do not expand site masks. The next single-factor test changes only class width to 2.5%, the sole coarser resolution with positive prior Q4 evidence; any failure closes bin-resolution changes on the correlation representation.
- M217 does not win alone, but confirms class-resolution diversity: a second fixed half blend with M215 raises group 1 to `0.621165` and group 3 to `0.607801`, producing M218's `0.620295`. This is a development-selected local artifact, not independent promotion evidence. M219 removes group balancing under the same 2.5% contract to determine whether the diversity comes from bin width itself or its interaction with reweighting.
- M219 shows the effects separate by site: removing group balance helps group 1 but sharply harms group 3. M220's groupwise combination reaches `0.621450`, still `0.038550` below the goal. The next bounded expansion doubles the training-only correlation rank from 80 to 160 per group under the balanced 2.5% contract; it tests whether useful lower-ranked wind levels were excluded without changing the target or decision layer.
- M221 rejects that expansion: the 186-feature union lowers every retained group, so top-K correlation growth is closed. A six-parent groupwise simplex oracle is only `0.622445`; static blending is also closed as the primary route.
- M222 shows a complementary point-location signal can still add a small amount: a fixed 5% M206 median anchor raises groups 1/2 while leaving group 3 unchanged, reaching local-Q3 `0.624706`. Because the weight was selected on Q3, this is a candidate-construction artifact rather than independent validation evidence. The next new representation supplies an honestly cross-fitted raw-weather point estimate as an input to the actual-generation class model, rather than using a post-hoc output blend.
- M223 rejects that representation: honest raw-only OOF point MAE is `0.129904`, and the Q3 point forecast is accurate enough to reach `1-NMAE=0.862982`, but feeding it to actual-generation classes lowers every retained group. Existing probability-model pair blends also stay below `0.626`, so the next target change attacks a different discretization defect: a half-bin-shifted target partition is trained and combined with the original partition before the frozen Bayes action.
- M224 rejects separate shifted partitions, but a cheap M212 probability convolution shows that explicit target-order regularization is useful: 20% neighbor mass raises the raw fixed-policy score to `0.616104`. M225 isolates the only retained gain in group 2 and reaches local-Q3 `0.625649`. The next bounded test moves this smoothing into classifier training by duplicating each row across its own and adjacent ordered bins with fixed `0.2/0.6/0.2` weights.
- M226 shows that the inference convolution gain is not reproduced by training label smoothing; conflicting duplicated labels weaken all three source heads. A cross-fitted uncertainty gate also cannot learn the large per-row point/action oracle, especially for group 3. The remaining official-data route with demonstrated group-3 diversity is to rebuild the 2022 donor-based pseudo-label head on the physically pre-2024 strict PLS/correlation surface and freeze its probability mix before Q3 scoring.
- M227 proves the pseudo mapper's high target-to-target holdout score is not usable through NWP; its inferred-weather head is worse than the short observed-only parent. Split-conformal correction of M206 quantiles also fails. The next classical representation is sliced inverse regression (SIR): PCA-whitened weather directions are supervised by pooled, donor-1/2, and group-3 target slices, adding nonlinear inverse-regression coordinates not present in covariance-only PLS.
- M228 rejects SIR despite capturing high-variance supervised directions. M229's gain instead comes from known-future 24-hour structure, but M230 shows that exposing 792 explicit neighboring-NWP features to the same source-rank heads lowers all three retained groups. The useful sequence signal is therefore in output-profile regularization rather than more rowwise tree inputs. The next route estimates a conditional daily output profile from complete preceding issuance batches and freezes all profile choices before Q3 scoring.
- The sequence result transfers across model and time surfaces. M229's generic recipe improved the old v2 parent's Q3 by `+0.001315` and Q4 by `+0.001784`. A v2-lineage recipe selected on Q3 alone improved Q4 by `+0.002547`, with both 1-NMAE and FICR increasing. This supports keeping a sequence-smoothed final challenger even though the exact online gain is unknowable without an upload.
- The transfer is probable rather than certain: the paired Q4 issuance bootstrap is positive in `91.3%` of replicates but its two-sided 95% interval includes zero, and October is slightly negative. The arithmetic `0.65971 + 0.00254678 = 0.66225678` is therefore only a conditional proxy if the user-reported online score belongs to the same parent behavior; it is not a leaderboard result.

## Active local loop checkpoint: M232-M234 daily analog transfer

- Daily analog matching contains real retrospective signal but is highly selection-sensitive. M232 improved Q3 by `+0.008990` and M233 improved Q2/Q3 by `+0.007582/+0.007947`, yet their frozen Q4 deltas were `-0.001701` and `-0.002557`. Those routes are rejected rather than post-hoc retuned.
- A deterministic three-fold stability rule found M234 recipes with positive group-level deltas on Q2, Q3, and Q4. The resulting overall deltas were `+0.005277/+0.002619/+0.002683`; Q4 improved both components to `1-NMAE=0.853090` and `FICR=0.372819`.
- M234 is not independent forward evidence because Q4 participated in the stability screen. It is a secondary development challenger, while M231 remains the primary candidate whose transform was chosen on Q3 and transferred once to Q4.
- Conditional arithmetic against the user-reported online baseline is `0.65971 + 0.00254678 = 0.66225678` for M231 and `0.65971 + 0.00254678 + 0.00268261 = 0.66493939` for M231 plus M234. These values are prioritization proxies only; no upload occurred and no live DACON score is claimed.
- M234's official test construction uses only complete 2022-2023 target days as analog references and official test NWP as queries. The three final 2026 boundary rows are validated as part of the supplied test contract, not interpreted as training labels.

## Active local loop checkpoint: M235 untouched-2022 selection

- The group-1/2-only 2022 diagnostic folds provide a genuinely earlier selection surface. M235 required one recipe to improve two parent families across Q2, Q3, and Q4, so each selected recipe survived six independent model/time comparisons before any 2023 transfer result was read.
- Group 1 transferred across every 2023 fold, but its Q4 effect decayed to `+0.000624` at group level and only `+0.000208` overall. A group-1-only paired bootstrap was positive in `54.1%` of replicates, so the evidence is too weak to create another test artifact.
- Group 2 demonstrates a year-regime failure: despite large positive 2022 deltas, the frozen recipe reduced 2023 Q3 and Q4. This rules out interpreting retrospective analog stability as automatically invariant across years.
- M235 is rejected as a whole. It strengthens the qualitative case that daily analog information exists, but M231 remains the cleanest transferable candidate and M234 remains the higher-upside, selection-biased development challenger.
- The revised `0.66000` threshold needs only `+0.00029` above the user-reported `0.65971`. On paired Q4 issuance-day resamples, M231 clears that delta in `88.10%` of replicates. M234-on-M231 clears it in `95.95%`, but because M234 used Q4 for recipe selection, that percentage must not be treated as calibrated online success probability.

## Active local loop checkpoint: M236 deployment-support constraint

- The official-source audit found no hidden or unused inference surface. SCADA and labels are train-only; GFS, LDAPS, turbine metadata, and submission keys are already consumed by the pipeline. Further model work cannot honestly claim a new official information source without changing the approved method family.
- M234 had a narrower evidence domain than deployment domain: all analog recipe checks were Q2-Q4, while its test construction applied the correction to Q1 as well. This is a support mismatch even though it does not violate row or leakage contracts.
- M236 is a deterministic hedge rather than a newly tuned model. It equals M231 on every Q1 row and M234 on every Q2-Q4 row, with boundaries defined per complete operating-day issuance batch. The first and last operating days are `2025-01-01 01:00` and `2025-12-31 01:00`; the final `2026-01-01 00:00` forecast row correctly remains inside the last Q4 batch.
- The candidate portfolio now separates evidence and risk: M231 is conservative and independently transferred, M236 is the risk-adjusted score candidate, and M234 is the aggressive full-year analog candidate. Their respective conditional point proxies are `0.66225678`, `0.66427792`, and `0.66493939`; all remain unverified online.

## Active local loop checkpoint: M236 quarter-support audit

- A deterministic, label-free support audit compared every official-test query day with the leave-one-out neighbor-distance distribution of the same historical analog library and fitted PCA representation. It loaded only the physical pre-2024 development surface plus official test NWP; no 2024 label or score was materialized.
- Q1 is not a feature-distribution outlier. Across groups, its median nearest-neighbor LOO percentile is `0.4017` and its recipe-kth percentile is `0.4259`, versus Q2 `0.5374/0.5000`, Q3 `0.3906/0.4398`, and Q4 `0.3885/0.3613`. Groupwise Q1 shares above the training 95th percentile are only `3.33%/2.22%/0.00%`, within the observed Q2-Q4 validation range.
- The M231-to-M234 Q1 correction exposure is nontrivial but comparable with supported quarters: mean absolute deltas are `347.73/271.86/362.75 kWh` for groups 1/2/3, or `1.61%/1.26%/1.73%` of capacity. This confirms that Q1 fallback is a material hedge, not a cosmetic CSV difference.
- Retrieval support does not validate forecast benefit. Group 3 has no pre-Q1 historical target library for a full 2023-Q1 chronology-safe analog evaluation, and M234's recipe still used Q2-Q4 during selection. Therefore M236 remains the risk-adjusted first candidate; M234 remains the higher-upside candidate whose Q1 application is distribution-supported but outcome-unverified.
- The audit report SHA-256 is `f295668fb2c661b0fd4fce6b99e7c85b5f0b027eaef3e5a8ce57a3571203a58e`; a second full run reproduced it byte-for-byte.

## Active local loop checkpoint: M237 predeclared support gate

- Hypothesis fixed before score inspection: M234 analog corrections may be less reliable on group-days whose nearest or recipe-kth distance exceeds the historical training-day leave-one-out 95th percentile.
- Single allowed rule: retain the frozen M234 recipe only when both percentiles are `<=0.95`; otherwise retain M231 for that complete group-day. Q1 remains an unconditional M231 fallback, matching M236's evidence boundary.
- No alternate percentile, soft weight, nearest-only rule, kth-only rule, quarter-specific threshold, or post-result rescue is part of this experiment. Promotion requires positive Q2/Q3/Q4 deltas over M231 and improved worst-fold or Q4 robustness relative to full M234.
- Result: M237 kept Q2 unchanged from M234, improved Q3 by `+0.000339`, but reduced Q4 by `-0.000714` versus M234. Its Q2/Q3/Q4 deltas over M231 were `+0.005277/+0.002957/+0.001969`, so the worst-fold delta fell below M234's `+0.002619` minimum.
- Q4 paired-bootstrap positivity fell from `93.65%` for full M234 to `87.70%` for M237; the mean incremental delta fell from `+0.002667` to `+0.001944`. The fixed gate removed five group-1 days, five group-2 days, and three group-3 days in Q4, and those fallbacks were harmful in aggregate.
- M237 is rejected by its predeclared rule. No threshold rescue and no test CSV are allowed. The result supports retaining M236's quarter-level evidence hedge rather than adding per-day analog-distance gating.
- A live inventory check confirms that sequential tree residual correction is not a fresh lane: M120 already trained on earlier OOF quarters and selected its policy before each later application. It reduced Q4 from the M107 parent to `0.637618` and pooled Total to `0.624055`; repeating the same residual-meta architecture on a lightly smoothed lineage is not justified as a new primary experiment.
- The approved local runtime has no PyTorch, TensorFlow, Keras, or Lightning installation. Scikit-learn MLP distribution and ordinal MLP screens already failed, while CatBoost/XGBoost/LightGBM are installed and extensively screened. A TFT-style run would require a dependency/scope expansion rather than being an unused current-runtime option.
- SCADA-derived turbine and site-wind routes are also closed by direct evidence: observed turbine power nearly reproduces labels, but inference-safe NWP-to-turbine/site-wind error collapses performance. M178 scored `0.584654` and the stricter M190 group-3 downstream screen collapsed to `0.372765`; contemporaneous SCADA cannot appear at inference.
- Label-free target-period adaptation is not an unused route either. The existing domain classifier excluded calendar fields, estimated clipped history-to-target density ratios from NWP only, and reweighted the source-rank learner; it was already rejected on unseen Q3. Repeating transductive covariate weighting would duplicate a closed family.
- Direct nonlinear 24-hour profile learning has already been bounded: daily Ridge/PLS/ExtraTrees/RandomForest were rejected, a high-dimensional CatBoost MultiRMSE run was stopped after 30 minutes, and the completed 64-input/8-target PCA CatBoost screen reached only `0.585086` after blending. The useful sequence evidence comes from output smoothing/analog correction, not a fresh daily-regression route.

## Active local loop checkpoint: M238 predeclared NWP phase alignment

- This is distinct from M155: M155 learned supervised offsets for parent prediction rows, whereas M238 does not fit a phase classifier and does not shift the parent. It changes only the hour alignment of each historical analog target profile before the frozen M234 head is computed.
- Frozen retrieval contract: M234 representation, neighbor count, distance order, kernel weights, head, target transform, and blend weight are unchanged for every group. Phase scoring uses all core features, medians and scales fitted on chronology-safe historical training days only, and per-query/per-neighbor standardized trajectory MSE.
- The sole lag set and deterministic tie order are `(0, -1, +1, -2, +2)`. Query weather at hour `t` is compared with neighbor weather at `t+lag`; the corresponding historical target at `t+lag` is used with edge replication. No circular wrap, target-aware lag choice, lag-range sweep, feature sweep, or post-result group exception is permitted.
- Promotion is fixed before execution: all Q2/Q3/Q4 Total deltas over their M231-lineage parents must be positive, and either the minimum fold delta must strictly exceed M234's minimum or both the Q4 delta and Q4 paired-bootstrap positive fraction must strictly exceed M234. A rejected run creates no submission CSV.
- Result: phase alignment raised Q2 from M234's `0.595751` to `0.597512`, but reduced Q3 from `0.598989` to `0.597283` and Q4 from `0.612955` to `0.610199`. The Q4 delta over M231 became `-0.000074`, so the all-positive requirement failed.
- Q4 paired-bootstrap positivity fell from `93.65%` to `47.05%` and mean incremental Total from `+0.002667` to `-0.000141`. The fixed rule is rejected; no test CSV or post-result phase rescue is allowed.
- The selected phase distribution drifted materially: zero-lag shares were roughly `35%` in Q2 but only `22-25%` in Q4, while mean absolute lag rose from about `1.03-1.08` hours to `1.32-1.38` hours across groups. Better NWP trajectory fit at a shifted index did not translate into stable target-profile alignment.

## Next-lane audit: learned analog similarity

- The frozen M234 core already contains 25 inference-safe trajectory variables: GFS/LDAPS wind speeds, wind-direction sine/cosine pairs, source-disagreement speeds, and two hub-height speed estimates. A new direction-only retrieval is therefore duplicate scope, not a fresh feature lane.
- Existing M121 and M157 used supervised daily models as direct 24-hour predictions or classifier features. They did not replace the analog-neighbor metric with a label-supervised latent distance, so a learned similarity space remains technically distinct from the rejected direct daily-regression family.
- Hu et al., *Weather Analogs with a Machine Learning Similarity Metric for Renewable Resource Forecasting* (`https://arxiv.org/abs/2103.04530`), identifies Euclidean feature weighting as a core analog-ensemble weakness and reports that a learned latent similarity metric can improve wind-speed/solar analog selection and transfer across locations. This directly supports testing learned retrieval rather than another output correction.
- Alessandrini et al., *A novel application of an analog ensemble for short-term wind power forecasting* (`https://doi.org/10.1016/j.renene.2014.11.061`), reports wind speed plus direction, a short temporal window, and about 20 analogs as its best configuration. M234 already covers those variables and comparable neighbor counts; the untested axis is similarity learning, not another hand-weight sweep.
- Any M239 test must remain low-capacity and chronology-safe because the daily sample count is only hundreds. A linear supervised NWP-to-target latent projection is the bounded local analogue of a learned similarity metric; it uses labels only before each query cutoff and uses query NWP alone at application time.
- Final M239 contract uses eight-component PLS rather than a new neural dependency: flatten the frozen core raw+delta trajectory, fit train-only median imputation and standardization, learn NWP X-scores against the 24-hour normalized generation profile, standardize those train X-scores, append the frozen `2.5` cyclic-season coordinates, and use Euclidean distance in that learned space. This is low-capacity, deterministic, and directly comparable with M234.
- All analog recipes remain fixed. The experiment is a metric ablation, not a second recipe search. Component count `8`, feature set, seasonal coordinates, and promotion rule cannot change after results are read.
- Result: M239 scored `0.595265/0.594828/0.609642` on Q2/Q3/Q4. Relative to M231-lineage parents the deltas were `+0.004791/-0.001542/-0.000630`; relative to M234 they were `-0.000486/-0.004161/-0.003312`.
- Q4 paired-bootstrap positivity was `37.10%`, versus `93.65%` for M234. The learned metric is rejected under the all-positive requirement and creates no test CSV.
- The model is numerically converged, but its direct-profile diagnostic exposes regime instability rather than a solver failure: group-1/2 Q3 predicted means were about `0.181/0.202` against training means `0.317/0.350`, and group 3 had only `89/180/272` training days for Q2/Q3/Q4. Retuning PLS complexity on these folds would be selection rescue, not new evidence.

## Next-lane audit: M240 analog-target spread

- Shahriari et al., *Using the analog ensemble method as a proxy measurement for wind power predictability* (`https://doi.org/10.1016/j.renene.2019.06.132`), reports analog-ensemble spread as an appropriate proxy for wind-power forecast difficulty. This supports reliability modulation from neighbor outcomes rather than another NWP-distance rule.
- Repository search found NWP vector-spread input features and M237's nearest/kth retrieval-distance gate, but no use of the retrieved neighbors' historical target-profile dispersion to modulate M234. M240 is therefore a new uncertainty diagnostic, not a renamed support gate.
- Frozen rule: compute weighted per-hour standard deviation across the exact M234 neighbor targets; derive an hour-specific reference from train-day leave-one-out spread at quantile `0.75`; set shrinkage to `1` below the reference and `reference/spread` above it. The M234 analog head and target transformation are unchanged, while only its effective per-row blend is multiplied by this shrinkage.
- Quantile, hourly aggregation, continuous inverse-excess rule, and promotion test are fixed before result inspection. No hard gate, daily aggregate, alternative percentile, or group-specific rescue is allowed.
- Result: M240 scored `0.595286/0.598934/0.613121` on Q2/Q3/Q4, corresponding to `+0.004812/+0.002564/+0.002849` over M231-lineage parents. Relative to M234 it changed Total by `-0.000465/-0.000055/+0.000166`.
- The minimum fold delta did not improve, but the fixed alternative promotion branch passed: Q4 incremental Total and paired-bootstrap positivity both rose, the latter from `93.65%` to `93.75%`. This is a narrow promotion, not evidence of a large effect.
- Shrinkage was mild and targeted: median multiplier remained `1.0` in every group/fold; mean multipliers ranged from `0.959` to `0.996`. Q4 affected about `31.8%/27.8%/28.7%` of group-1/2/3 rows, with minimum multipliers `0.709/0.802/0.559`.
- Deployment remains evidence-bounded: M240 is validated only on Q2-Q4, so any test build must copy M231 exactly on Q1 rather than extrapolating the spread policy into an outcome-unvalidated quarter.
- The final no-upload candidate is `artifacts/submissions/E0_SPREAD_SHRUNK_ANALOG-ead6308128ce.csv`. It has 8,760 rows, UTF-8 BOM, exact sample-key ordering, CSV SHA-256 `e4d69edeb3221272856eac838065510291d4adb5597d6d278ce840e3005c3e90`, and receipt SHA-256 `58bf8dc0b51370156f15e461c88798d6033a274365c1ae49345cd09adafe68ca`.
- Q1 exactness is proven after explicitly restoring M231 values in the final assembly: all 6,480 group-row comparisons have zero difference. The initial failure was only a `1.82e-12 kWh` divide/multiply round-trip, not a support-boundary leak.
- The candidate's conditional proxy is `0.66440299`, but its receipt labels that value `heuristic_prioritization_only_not_online_or_local_score`. The actual chronology-safe Q4 development Total is `0.613121`, so the final local `>0.66000` gate remains open.

## Next-lane audit: M241 analog recency

- Repository search found a rejected recency-weighted row classifier but no temporal decay on M234's retrieved historical target members. The scope is distinct: M241 changes only within-neighbor analog weights after the frozen NWP search.
- M234 already appends cyclic day-of-year coordinates at weight `2.5`, so it captures season but not inter-year concept drift. M235's group-2 reversal between 2022 selection and 2023 transfer is direct local evidence that equal year weighting can be risky.
- *Refining the Selection of Historical Period in Analog Ensemble Technique* (`https://www.mdpi.com/1996-1073/16/22/7630`) studies the analog-history window as a material forecast-design choice. It does not prescribe this dataset's optimum, so M241 uses one conventional, non-searched one-year half-life rather than claiming a literature-optimal value.
- Frozen rule: multiply each exact M234 kernel weight by `2^(-age_days/365)` where age is query issuance minus historical-neighbor issuance, then renormalize and use the same head/transform/blend. No hard window, half-life sweep, M240 spread combination, or group exception is permitted.
- Result: M241 scored `0.595775/0.598838/0.613592` on Q2/Q3/Q4, equal to `+0.005300/+0.002468/+0.003320` over M231. Relative to M234 it changed Total by `+0.000024/-0.000151/+0.000637`.
- Q4 paired-bootstrap positivity improved from `93.65%` to `96.25%`; mean delta rose from `+0.002667` to `+0.003309`, and the 5% delta quantile crossed from `-0.000149` to `+0.000300`. This satisfies the fixed Q4-robustness promotion branch more convincingly than M240.
- The one-year decay is materially active for groups 1/2, lowering their mean effective Q4 neighbor age by `29.7/35.3` days, while group 3 changes only `7.4` days because its history is shorter. No half-life tuning is needed to explain the observed effect.
- The final no-upload candidate is `artifacts/submissions/E0_RECENCY_ANALOG-58cce9ff45a6.csv`, CSV SHA-256 `39a9b275f684146599900baf567963cb7bcd6911e7e2f8ddda3bbad6d31be355`, receipt SHA-256 `60a6194c98fd68ef2d5ff22fe18218b39470534a1487838767d64064e4b3f915`. It has 8,760 rows, exact M231 Q1 fallback, and fixed M241 Q2-Q4 output.
- Its conditional proxy is `0.66475822`, higher than M240's proxy, but remains explicitly non-local/non-online. The actual Q4 development Total `0.613592` keeps the completion gate open.

## Next-lane audit: M242 lead-local AnEn

- M130 is a genuine row-local analog implementation and its same-fold Q3 sweep reached only `0.597071`, with group 3 at `0.562316`. It searched current-row features, 20-160 neighbors, three kernels, and settlement gamma, so repeating that design is closed.
- The non-duplicate gap is narrow but concrete: M130 did not construct the `t-1,t,t+1` predictor window reported as best in Alessandrini et al., and it did not freeze one configuration across Q2-Q4. M242 tests only that missing formulation rather than reopening M130's sweep.
- Frozen M242 contract: 25 core features at previous/current/next within the complete issuance, edge replication, same lead-hour historical candidates, train-only scaling, cyclic season coordinates at `2.5`, 20 uniform neighbors, and M234 group head/transform/blend. No recency or spread combination is included.
- Result: M242 scored `0.593469/0.596442/0.611498` on Q2/Q3/Q4. Relative to M231 the deltas were `+0.002994/+0.000072/+0.001226`, but relative to M234 they were `-0.002282/-0.002546/-0.001457`.
- Q4 paired-bootstrap positivity was `79.95%` with mean incremental Total `+0.001196`, both below M234's `93.65%` and `+0.002667`. The minimum-fold improvement also fell to `+0.000072`, so neither predeclared robustness branch passed.
- M242 is rejected without a test CSV. The result closes the exact lead-local 3-hour-window formulation; changing its neighbor count, kernel, window, feature subset, or combining it with M241 after observing these folds would be rescue selection.

## Next-lane audit: M243 orthogonal analog composition

- Repository search found no prior composition of M240's target-spread blend multiplier with M241's recency-weighted analog head. The two promoted policies act at different points and were each frozen before result inspection, so a single exact composition is distinct from retuning either policy.
- Frozen M243 rule: compute `analog_normalized` exactly as M241, compute `analog_blend_multiplier` exactly as M240 from the unmodified M234 neighbor ensemble and train-LOO 75th-percentile hourly references, then apply the M240 effective-blend equation. M234 retrieval, recipes, transforms, half-life, spread reference, and parent predictions are unchanged.
- There is no half-life, spread quantile, multiplier exponent, mixture weight, group exception, or M242 input. Promotion is fixed relative to M241: all fold deltas over M231 must be positive and either the worst-fold delta must strictly improve or both Q4 delta and Q4 paired-bootstrap positivity must strictly improve.
- Result: M243 scored `0.595216/0.598973/0.613765` on Q2/Q3/Q4, equal to `+0.004742/+0.002603/+0.003492` over M231. Relative to M241 it changed Total by `-0.000559/+0.000135/+0.000172`.
- The minimum fold delta improved from M241's `+0.002468` to `+0.002603`. Q4 bootstrap positivity rose from `96.25%` to `96.55%`, with the 5% delta quantile rising from `+0.000300` to `+0.000383`; both fixed promotion branches pass despite the Q2 tradeoff.
- M243 is promoted narrowly for one supported-season test build. The result does not authorize half-life, spread-threshold, interaction-strength, or group-specific searches, and its actual Q4 Total remains far below the `0.66000` local completion gate.
- The final no-upload candidate is `artifacts/submissions/E0_RECENCY_SPREAD_ANALOG-bc88b89a9c71.csv`, CSV SHA-256 `8af6c8a0670e9fa0f9936b59fc8fe3876b88d581301be816df3ef00e4d5e183d`, receipt SHA-256 `307ab229c6611a07573d6f8547a0f813ea548c1c97ca9e67ecb0e80c436fd9eb`. It has 8,760 rows, exact M231 Q1 fallback, and 6,600 M243 rows.
- Its conditional proxy is `0.66488810`, but the receipt explicitly classifies it as heuristic-only. The verified chronology-safe Q4 Total is `0.613765`; no claim of reaching the final target or beating the live leaderboard is permitted.

## Next-lane audit: M244 rare-event analog MOS

- The existing M106 KNN residual utility model uses row-level parent predictions and generic residual neighbors. No runner adjusts M234/M243 daily analog members by the query-to-neighbor NWP change, so an AnEn regression correction is not a duplicate of the closed residual lane.
- Alessandrini, Sperati, and Delle Monache, *Improving the Analog Ensemble Wind Speed Forecasts for Rare Events* (`https://doi.org/10.1175/MWR-D-19-0006.1`), identifies increasing conditional underprediction in the right tail and reports reliability/CRPS improvement from a linear-regression bias correction. The BARAM adaptation must remain low-capacity because it predicts power, not the paper's wind-speed target.
- Frozen rule: use the supplied `phys_v2__hub117_speed` only; fit one nonnegative OLS slope per group on strict pre-query rows; activate only above that group's train-only 90th percentile; cap slope at `0.20` normalized power per m/s; adjust each M241-recency neighbor member by query-minus-neighbor speed; and retain M240's unmodified spread multiplier. No lead-specific slope, tail quantile, wind proxy, cap, or group exception may be selected after results.
- Promotion is fixed relative to M243: all Q2/Q3/Q4 deltas over M231 must remain positive and either the worst-fold delta must strictly improve or both Q4 delta and paired-bootstrap positivity must strictly improve. A rejected experiment produces no submission CSV.
- Result: M244 scored `0.596395/0.598171/0.614707` on Q2/Q3/Q4, equal to `+0.005921/+0.001801/+0.004435` over M231. Relative to M243 it changed Total by `+0.001179/-0.000802/+0.000943`.
- Q4 paired-bootstrap positivity rose from `96.55%` to `99.55%`; the mean incremental Total rose from `+0.003477` to `+0.004401`, and the 2.5% quantile crossed from `-0.000242` to `+0.000804`. This passes the fixed Q4 robustness branch despite the Q3 tradeoff.
- Fitted slopes were stable and below the cap: roughly `0.051-0.067` normalized power per m/s. Tail activation varied from about `2.5%` in Q3 to `15.5-18.9%` in Q4, explaining why the correction is most active in the later high-wind quarter. No threshold or cap retuning is justified.
- M244 is promoted narrowly for one Q1-fallback test build. Its verified Q4 Total `0.614707` remains below the local `0.66000` gate; the result is evidence of transferable tail correction, not completion.
- The final no-upload candidate is `artifacts/submissions/E0_RARE_EVENT_ANALOG-5f6a9679a463.csv`, CSV SHA-256 `1c9c0a509a71ee9ebe2eb0e15bdc37eadfed42687644b9ef0423b89de20a3ab3`, receipt SHA-256 `6d0d37b0e0eaf2058c81f434dce92a0c6aa49b1645bd74fb5453d41475e6b485`. It has 8,760 rows, exact M231 Q1 fallback, and 6,600 M244 rows.
- Its conditional proxy is `0.66559822`, now the highest portfolio heuristic, but remains explicitly non-local/non-online. It cannot close the target without an authorized external evaluation, and no upload was performed.

## Next-lane audit: M245 Q1 group-1/2 scope

- The physical development surface confirms groups 1/2 have labels from `2022-01-01`, while group 3 starts only at `2023-01-01`. This permits a strict 2022-to-2023-Q1 evaluation for groups 1/2 but provides no honest group-3 Q1 analog validation.
- Frozen validation topology: 90 complete 2023-Q1 operating-day batches for groups 1/2; training batches must have their maximum forecast timestamp strictly before the first validation issuance availability time. The two parent contracts are the unchanged M235 shared L1 and shared Q50 models, followed by the exact M231 sequence transform.
- Exact M244 is applied without refitting any recipe or development parameter. Deployment scope expands to Q1 groups 1/2 only if each of four group-by-parent Total deltas and both combined-parent Total deltas are positive. Group 3 remains an unconditional M231 fallback even if the audit passes.
- This is a scope-support audit, not another Q1 recipe search and not a full official Total because group 3 is intentionally absent. Failure leaves the existing M244 Q1 fallback candidate unchanged.
- Result: every frozen scope surface improved. Under shared-L1, group-1/group-2/combined Total deltas were `+0.009227/+0.001467/+0.005347`; under shared-Q50 they were `+0.008346/+0.004046/+0.006196`.
- Combined paired-bootstrap positivity was `99.75%` for shared-L1 and `99.95%` for shared-Q50. Their 2.5% delta quantiles were `+0.001886` and `+0.002645`, so the scope pass is not driven by a single day or parent family.
- The strict surface used 17,282 group-1/2 training rows from 362 complete pre-cutoff batches. The training maximum forecast was `2022-12-31 00:00`, while the 90-day validation surface ran from `2023-01-01 01:00` through `2023-04-01 00:00`; no 2024 lockbox row was touched.
- The scope-extended no-upload candidate is `artifacts/submissions/E0_Q1_GROUP12_RARE_EVENT_ANALOG-bea349ead945.csv`. CSV SHA-256 is `daafce4b61bb81265d3b48346290c3a7e3273127940eebf0c51e9c32433f16e7`; receipt SHA-256 is `c5f7927ecef3aa3d5bb6459a64de4b307014a241e38da6b3446fb156622b1e0a`.
- Candidate invariants hold after an exact rebuild: Q1 groups 1/2 each change 2,160 rows, Q1 group 3 has zero changed rows, and every Q2-Q4 value equals M244. Eighteen submission/budget tests and the no-upload/no-lockbox receipt assertions pass.
- Its conditional proxy rises to `0.66647715`, but that number combines prior online and local uplifts and is explicitly not a chronology-safe or online score. The highest comparable full Q4 development Total remains M244's `0.614707`, so the final local `>0.66000` gate remains open.

## Next-lane audit: M246 final-power ensemble EMOS

- Repository search found no Ensemble Model Output Statistics calibration of the analog power members. M240 uses their raw weighted spread only to reduce the M234 blend on high-dispersion rows; it never learns a power-member mean or variance correction. The generic residual-model family predicts point residuals or action distributions from parent features and is not the same member-level operation.
- Phipps et al., *Evaluating Ensemble Post-Processing for Wind Power Forecasts* (`https://doi.org/10.1002/we.2736`; open manuscript `https://arxiv.org/abs/2009.14127`), evaluates raw, weather-only, final-power-only, and two-step EMOS. Across synthetic benchmarks and Swedish bidding zones, strategies that post-process the final wind-power ensemble improve calibration and sharpness most consistently; weather-only correction can be neutral or harmful.
- The paper's EMOS mean is a regression on the ensemble mean and its variance is a nonnegative function of ensemble spread. M246 adapts exactly that low-capacity moment structure, but it does not claim the paper's CRPS result transfers automatically to Dacon Total; promotion still depends on the frozen local official-metric contract.
- Frozen calibration topology: after 80 initial complete days, partition the remaining outer-training days into contiguous 90-day blocks, matching the natural quarter-scale validation horizon. Predict every block using all and only complete days strictly before the block. Each pair uses the exact M241 recency weights and the exact M244 rare-event threshold/slope fitted from that earlier reference set. Outer validation labels never enter retrieval, calibration, or policy fit; block length is fixed once and is not searched.
- Fit one group-level, nonnegative-slope affine regression from weighted corrected-member mean to normalized generation. Fit one group-level nonnegative least-squares variance equation `residual^2 = c + d * raw_member_variance`. Map the outer-query members to the fitted mean and variance, clip to `[0,1]`, and recompute the unchanged M244 recipe head.
- Exact M244 representation, neighbor count, kernel, recency half-life, rare-event rule, head, transform, blend, and M240 raw-spread multiplier remain frozen. There is no hourly/lead calibration, minimum-history search, rolling-window search, distribution family, regularization, clipping-bound, component mixture, or group exception.
- Promotion is fixed relative to M244: all Q2/Q3/Q4 Total deltas over their M231-lineage parents must remain positive, and either the minimum fold delta must strictly exceed M244's or both the Q4 delta and Q4 paired-bootstrap positive fraction must strictly exceed M244. Failure produces no submission CSV.
- Result: M246 scored `0.592036/0.595334/0.612677` on Q2/Q3/Q4. Its deltas over M231 were `+0.001562/-0.001036/+0.002405`, versus M244's `+0.005921/+0.001801/+0.004435`; every fold regressed and Q3 crossed below the parent.
- Q4 paired-bootstrap positivity fell from M244's `99.55%` to `87.60%`, with mean delta falling from `+0.004401` to `+0.002398` and the 2.5% quantile becoming negative at `-0.001714`.
- The calibration fit diagnoses a transfer problem rather than an implementation failure. Group-1/2 mean intercepts were often about `-0.10` to `-0.13`, and the short group-3 Q2 calibration fitted a mean slope of only `0.359`. Nonzero variance intercepts also produced very large formal scale ratios when raw member variance approached zero, even though final members were clipped.
- M246 is rejected without a test CSV. A slope floor, zero variance intercept, block-length change, longer minimum history, per-hour fit, or group-specific exception would be post-result rescue selection and is forbidden. The exact rerun reproduced receipt SHA-256 `c1110789b48a73f14f0732b10997f8e52fbb1f8f4c5172b30112512cdce0857f` and prediction SHA-256 `c034bb5461df315bb8da7737beaa8cc2c9cd4f365b7d66e73f6ca70c7002eab6`.

## Next-lane audit: M247 source-separated multi-model AnEn

- Pappa et al., *Analog versus multi-model ensemble forecasting: A comparison for renewable energy resources* (`https://doi.org/10.1016/j.renene.2023.01.030`; JRC record `https://publications.jrc.ec.europa.eu/repository/handle/JRC130644`), compares deterministic NWP, analog ensembles, unweighted and analytically weighted multi-model ensembles, and a hybrid. It reports that analog and multi-model errors are complementary across wind-speed regimes and that the hybrid systematically improves wind-power skill by up to about 2% over the better component.
- The local full-core analog is not the paper's multi-model construction: M234 concatenates nine GFS wind variables, twelve LDAPS wind variables, two disagreement variables, and two GFS-derived hub proxies into one standardized PCA distance, then retrieves one shared neighbor set. Repository search found no independent GFS and LDAPS analog neighbor pools.
- Frozen M247 source definitions are exhaustive within M234's inference-safe core: GFS uses all nine `gfs_spatial__idw__wind*` variables plus `phys__hub117_speed` and `phys_v2__hub117_speed`; LDAPS uses all twelve `ldaps_spatial__idw__wind*` variables. Disagreement features are excluded because they belong to neither independent source.
- Each source independently uses raw+delta PCA24, season weight `2.5`, and the group's unchanged M244 neighbor count and kernel. M247 concatenates the two historical target-member sets and assigns exactly half of total probability mass to each source. There is no source-skill fit, source-weight search, best-source selection, or mixture with the full-core neighbor set.
- The exact one-year recency decay and M244 group slope/90th-percentile rare-event adjustment are applied to each source's members. The unchanged M244 head is then computed on the pooled distribution, and exact M240 full-core raw-spread shrinkage remains the final blend multiplier. Parent, transform, and base blend weight do not change.
- Promotion is fixed relative to M244: all Q2/Q3/Q4 Total deltas over M231 must remain positive, and either the minimum fold delta must strictly exceed M244's or both Q4 delta and Q4 paired-bootstrap positivity must strictly exceed M244. A failure creates no test CSV and cannot be rescued with source weights or group exceptions.
- Result: M247 scored `0.598936/0.595124/0.614306` on Q2/Q3/Q4, for deltas of `+0.008462/-0.001246/+0.004034` over M231. Relative to M244 it changed Total by `+0.002541/-0.003047/-0.000401`; the all-positive gate fails on Q3.
- Q4 paired-bootstrap positivity fell from `99.55%` to `96.45%`, with a negative 2.5% quantile of `-0.000314`. The Q4 mean delta remained useful at `+0.004024`, but it did not exceed M244's `+0.004401`.
- Independent retrieval did create genuine diversity: mean GFS/LDAPS neighbor overlap varied by group/fold from about `26%` to `46%`, with some group-3 days sharing no neighbors. The failure is therefore unstable conversion of diversity into official-metric gain, not accidental duplication of the full-core set.
- M247 is rejected without a test CSV. Source-weight fitting, group-specific source selection, mixing the pooled and full-core distributions, or activating it only in Q2 would all be post-result rescue selection. The exact rerun reproduced receipt SHA-256 `b1d867294408aa612166a31169e3dab1636fa33d7a2f53475c91d0d977baa6e7` and prediction SHA-256 `4aeac7994249deb9c65175d558857773b431d2fe21da12f315d4043541825d42`.

## Next-lane audit: M248 official-eligibility-aware analog Bayes head

- The exact M234/M244 `utility` head has a concrete metric-contract mismatch: it includes analog targets below `0.10` capacity in both expected absolute error and settlement reward, and it divides settlement by the mean of all historical generation. The official evaluator excludes those rows entirely, while `src/baram/decisions/expected_utility.py` conditions action loss on eligible samples and persists the group mean of eligible generation.
- This is not another distribution or action-grid search. M248 preserves the exact M244 neighbor identities, recency weights, rare-event member corrections, M240 spread multiplier, action grid, per-group selected head, transform, blend, and parent predictions. Only groups 1/2 change because their frozen recipe head is `utility`; group 3's frozen `median` head must reproduce M244 exactly.
- For each query/hour, define eligible members by the official fixed threshold `target >= 0.10`. Compute expected normalized absolute error and expected settlement-weighted generation conditional on the eligible member mass, divide the latter by four times the strict pre-query group mean of eligible normalized generation, and select the unchanged action with maximum equal-weight composite. If eligible mass is numerically zero, retain the exact M244 legacy action for that row rather than introduce a learned fallback.
- Promotion is frozen relative to M244: every Q2/Q3/Q4 Total delta over M231 must remain positive, and either the minimum fold delta must strictly exceed M244's or both Q4 delta and paired-bootstrap positive fraction must strictly exceed M244. No eligibility boundary, mean definition, action grid, fallback, group exception, blend, or head assignment can change after scores are read.
- Result: M248 scored `0.597384/0.597203/0.614747` on Q2/Q3/Q4. Relative to M244 the changes were `+0.000988/-0.000969/+0.000039`; all deltas over M231 remained positive, but the worst-fold delta fell to `+0.000833` and therefore failed the first promotion branch.
- Q4 paired-bootstrap positivity was `99.45%`, just below M244's `99.55%`, so the second branch also failed despite a slightly higher Q4 point Total. The corrected algebra materially changed `26.5%/29.7%` of group-1/2 Q4 utility actions, while group 3 remained exactly unchanged and no row needed the zero-eligible-mass fallback.
- M248 is rejected without a test CSV. Selecting it only for Q2, changing the eligibility boundary/denominator, or mixing its decisions with M244 after observing these folds would be rescue selection. The exact rerun reproduced receipt SHA-256 `ddf9663542cd60413117bceab805f93c840a3c5b97eb7636ac0cb9143e84ec68` and prediction SHA-256 `fbd454e0f585608405635f0571765b8724346082ec608a8e6fe42bfc4517089f`.

## Next-lane audit: M249 minimal physical-state analog retrieval

- The PCWG benchmark (`https://doi.org/10.5194/wes-5-199-2020`) treats hub-height wind speed and air density as the conventional power-curve baseline and documents systematic power deviations with shear and other atmospheric states. Its cross-dataset results also warn that correction effectiveness is data-dependent, so local promotion remains mandatory rather than assumed from the paper.
- M244's frozen 25-variable core contains GFS/LDAPS wind vectors, source disagreement, and two hub-speed proxies, but no shear exponent, density, or energy-flux proxy. Those three variables already exist deterministically in the official-data feature surface as `phys_v2__shear_alpha_100_80`, `phys_v2__air_density`, and `phys_v2__rho_v3`; no external measurement or new feature formula is needed.
- M205's rejected 200-feature direct atmospheric classifier does not answer this narrower retrieval question. M249 appends exactly the three physical-state variables to M244's core and changes only the train-fitted raw+delta/PCA24 neighbor geometry. The same recipes, recency half-life, hub-speed rare-event correction, spread rule, parents, heads, transforms, and blends are recomputed without parameter changes.
- Promotion is frozen relative to M244: all Q2/Q3/Q4 deltas over M231 must stay positive, and either the worst-fold delta must strictly improve or both Q4 delta and Q4 paired-bootstrap positivity must strictly improve. No physical-feature subset, scaling/weight, nonlinear transform, PCA count, neighbor, recipe, group, or quarter exception can be introduced after scores are read.
- Result: M249 scored `0.597444/0.597970/0.614660` on Q2/Q3/Q4. Relative to M244 the changes were `+0.001049/-0.000202/-0.000047`; all deltas over M231 remained positive, but the worst-fold delta fell from `+0.001801` to `+0.001599`.
- Q4 paired-bootstrap positivity fell from M244's `99.55%` to `97.95%`, despite a still-positive 2.5% quantile of `+0.000152`. The augmented retrieval materially changed `43.6%/41.5%/47.7%` of group-1/2/3 Q4 analog profile rows, confirming that the physical variables affected neighbor geometry but did not improve forward robustness.
- M249 is rejected without a test CSV. Selecting only the Q2 result or searching subsets/weights/transforms among the three physical variables would be post-result rescue. The exact rerun reproduced receipt SHA-256 `fbc86d6fd76f89f21d83d7ebbebdf98e1ca8627fdd087faa4e29eec7d7c014aa` and prediction SHA-256 `7e702a5e3d0247bd21347d086067ccea843fa2092755f858e6ea83e3fb0e5309`.

## Next-lane audit: M250 member-rank empirical quantile mapping

- Kakimoto et al., *Quantile mapping correction of analog ensemble forecast for solar irradiance* (`https://doi.org/10.1016/j.solener.2022.03.015`), reports that finite AnEn libraries deform the forecast PDF increasingly for members with larger neighbor order and that member-wise quantile mapping improves reliability and CRPS. This is a probabilistic benchmark transfer, not evidence that it will improve Dacon Total.
- M246 fitted low-capacity affine ensemble means and variance equations; M250 is a different nonparametric distribution-shape correction. It reuses M246's fixed strict-prequential topology only to obtain honest calibration pairs: 80 initial complete days followed by fixed 90-day expanding-history blocks, always predicting a block from strictly earlier days.
- For each group and distance rank, pool all 24 lead hours, sort the prequential corrected-member values and their verifying normalized actual values independently, and map each outer-query member through the exact empirical percentile. There is no probability grid or fitted smoother; physical clipping to `[0,1]` is the only bound. The exact M244 weights/head and exact M240 raw-member spread multiplier are retained.
- Promotion is frozen relative to M244: all Q2/Q3/Q4 deltas over M231 must remain positive, and either the worst-fold delta must strictly improve or both Q4 delta and Q4 paired-bootstrap positivity must strictly improve. No lead/hour mapping, correction cap, constraint, shrinkage, rank pooling, group exception, or post-result variant is allowed.
- Result: M250 scored `0.593016/0.594281/0.614489` on Q2/Q3/Q4, changing M244 by `-0.003379/-0.003890/-0.000218`. Q3 became `-0.002089` below its M231 parent, immediately failing the all-positive gate.
- Q4 paired-bootstrap positivity fell from `99.55%` to `91.90%`, and the 2.5% quantile became negative at `-0.001407`. The map was materially active: it changed `96.4%/95.7%/87.4%` of group-1/2/3 Q4 members and shifted them downward on average by roughly `0.106/0.119/0.155` normalized generation.
- M250 is rejected without a test CSV. The source paper notes that constraints can stabilize quantile mapping, but choosing a correction cap or shrinkage after observing this excessive downward shift would violate the frozen contract. The exact rerun reproduced receipt SHA-256 `3db12e6da85455e2dbcb0d8ddfddf971e0eee7396998d97a4223ad28dba10b5d` and prediction SHA-256 `d93cef0e2f71dd8330d6918eee7442950f10985a883a7bb46f4caac78f45ecd5`.

## Post-M250 lane audit: weather-pattern-aware ramp forecasting

- Okada et al., *Wind power ramp forecasting based on the optimal configuration of numerical weather prediction models* (`https://doi.org/10.1002/we.2774`), classifies weather patterns with PCA and chooses among multiple NWP configurations for ramp forecasting. The transferable principle is weather-regime-aware model choice; its experimental intervention depends on selectable dynamical NWP configurations and observed farm data that are not present in the official competition package.
- The local analog pipeline already uses train-fitted PCA over each complete daily GFS/LDAPS trajectory, so it already performs a weather-pattern similarity operation. M238 tested per-neighbor phase alignment, M230 tested neighboring-NWP/ramp context, M229 tested within-issuance output smoothing, and M205 tested direct atmospheric-state modeling. A further ramp/weather-pattern variant would either duplicate those completed lanes or introduce unavailable inputs.
- No M251 experiment is launched. This is an evidence-based lane closure, not a claim that ramp forecasting is generally unhelpful.
- `E0_Q1_GROUP12_RARE_EVENT_ANALOG-bea349ead945.csv` remains the first no-upload candidate. Its `0.66647715` conditional proxy is only a prioritization heuristic; the external `>0.66000` gate remains unverified until the user uploads it and Codex performs read-only score/rank inspection.

## Next-lane audit: M251 M244-to-M107 parent-invariance transfer

- M107 is still the reproducible strict local champion at pooled `0.62688770 / 0.85494357 / 0.39883183`, and its Q2/Q3/Q4 prediction surface is complete. The promoted M244 analog correction was developed only over the v2/M231 lineage, so its ability to add information to an independently trained higher-scoring parent has not been tested.
- M251 applies the exact M244 corrected analog profiles to M107. Retrieval representation, neighbor counts, kernels, one-year recency half-life, 90th-percentile hub-speed correction, slopes, spread reference, heads, transforms, and blend weights are identical to M244. Only the parent prediction changes.
- This differs from the closed static-ensemble lane: there is no convex parent mixture, shift, action grid, or weight search. It tests one already-frozen exogenous weather-analog correction against an independently trained parent.
- Promotion is fixed before execution: every Q2/Q3/Q4 Total delta over M107 must be positive; pooled Total, `1-NMAE`, and FICR must all strictly improve; and the paired Q4 issuance-day bootstrap must have positive fraction above `0.50`. Failure creates no test CSV and closes M244 parent-transfer rescue.
- The first run stopped before scoring because M107's Q4 artifact includes three `2024-01-01 00:00` rows belonging to the final `2023-12-30 13:00` issuance, while the strict feature surface ends at `2023-12-31 23:00`. M251 resolves only the issuance key from the 23 preceding pre-cutoff rows and leaves all 24 predictions of that boundary issuance exactly at M107. No post-cutoff feature or target is materialized.
- Result: Q2/Q3/Q4 Total deltas were `-0.000161/+0.001380/+0.002232`. Pooled Total, `1-NMAE`, and FICR improved from `0.626888/0.854944/0.398832` to `0.628111/0.855712/0.400510`, so the frozen analog contains complementary information, but it is not stable across all three folds.
- The Q4 paired issuance-day bootstrap had mean delta `+0.002248`, positive fraction `55.65%`, and 5%-95% interval `[-0.024650,+0.028881]`. The all-fold promotion branch fails, and the bootstrap shows much weaker robustness than M244 on its original parent.
- M251 is rejected without a test CSV. Selecting only Q3/Q4, using only the groups that improved, shrinking the blend, or substituting an M240/M241/M243 subset after this result would be prohibited parent-transfer rescue.
- Exact rerun reproduced prediction SHA-256 `a86b018bff7c20bfe0cccdf184ef47fd83d3c74498d4208f1fe85eb55438412a` and receipt SHA-256 `b440d8268ce4e7a7d904155fd5c1fd19e504f24614ce25e029eea3739b586a99`. Ruff, compile, 18 submission/budget tests, no-M251-CSV, and no-process checks pass.

## Next deployment lane: M252 frozen-policy 2022-2024 final fit

- IP@v2 line 43 and T8 distinguish final fitting from lockbox evaluation: after policy freeze, all supplied 2022-2024 labels may be consumed as ordinary training rows, provided no 2024 metric, slice, comparison, policy fit, or selection is produced. The existing v2 parent follows this contract and records `training_years=[2022,2023,2024]` with `final_fit_horizon="competition supplied 2022-2024 labels without scoring"`.
- The current M245 builder instead constructs its analog history from `development_surface()`, which physically stops before 2024. Its v2 parent is full-history, but its M244 analog neighbor targets, recency weights, rare-event fit, and spread reference see only 2022-2023. This is a deployment final-fit horizon mismatch, not a new model-family hypothesis.
- M252 freezes the exact M245 scope and M244 recipes before accessing 2024 label values. It may refit only parameters implied by those frozen formulas: PCA/scaler on the full history, analog member library, one-year recency weights, group 90th-percentile hub threshold and OLS slope, and train-LOO 75th-percentile spread references.
- M252 must never score 2024, compare a 2024 prediction with its label, inspect a 2024 slice, or feed final-fit output back into selection. It must preserve Q1 group-3 exact M231 fallback, use M244 for Q1 groups 1/2 and all Q2-Q4 groups, create no external action, and keep the consumed-lock receipt byte-identical.

## M252 result: full-history final-fit candidate built

- `build_full_history_q1_group12_rare_event_challenger.py` hash-pins every frozen helper and M244/M245 evidence artifact, statically rejects direct scoring calls, loads only the 25 frozen core inference features plus official labels, and keeps all test targets as `NaN`.
- The final-fit surface contains all 78,912 official 2022-2024 rows. The frozen daily-profile algorithm admits only 24-hour blocks with 24 finite labels. Under the additional availability rule `target day_end < query issuance`, groups 1/2 use 1,086 days each (`359/364/363` by operating year) and group 3 uses 729 days (`0/365/364`).
- The first run stopped before candidate output because the audit expected 1,087 group-1/2 days. The omitted day ends at `2025-01-01 00:00`, after the first test issuance at `2024-12-31 13:00`; excluding it is necessary to avoid using an outcome unavailable at prediction time. Only the audit expectation was aligned to the already-frozen chronology rule.
- Built `artifacts/submissions/E0_FULL_HISTORY_Q1_GROUP12_ANALOG-1741a964e30b.csv`. It has 8,760 rows, exact sample keys and five-column schema, finite nonnegative capacity-bounded values, and exact M231 fallback for all 2,160 group-3 Q1 rows. It differs materially from M245 in every other group-quarter scope, confirming that the 2024 final-fit history is active.
- CSV SHA-256 is `06fe135a22b2eff0b66303b87090f43a2b3c99ec688f4eb84cadac0c874e0730`; receipt SHA-256 is `ac783f2ec153e84dc4db62864f791f502559e4aa75e51bd375a859ac8ddf75f2`. A second complete run reproduced both hashes exactly.
- The receipt records zero score calls, no 2024 metric/slice/comparison, `local_score=null`, `online_score=null`, no selection after final fit, no upload, and the unchanged consumed-lock hash `866f22dcd88c8bcbb1841b55d989a9af43b6f9e133606b47bb7378b1b97ace1f`. Ruff, compile, and all 18 submission/budget tests pass.
- M252 is now first for user-side score verification because it resolves the parent/analog final-fit horizon mismatch. It is not proven better than M245 and does not satisfy the Total `>0.66000` gate until Dacon returns an external score.

## Post-M252 continuation audit

- The external score gate is still open, so local work may continue only through a predeclared lane that is independent of M252's unscored 2024 final-fit output. Any selection evidence must remain on the physical pre-2024 surface.
- A broad repository search for remaining ensemble/residual/calibration lanes expanded feature-heavy JSON manifests and was truncated. It is not usable evidence about duplication or opportunity. The next audit must inventory filenames and scalar receipt fields first, then inspect only the exact candidate families needed for a decision.
- The compact strict-OOF inventory confirms M107 is still the strongest chronology-safe parent: Q2/Q3/Q4 Total `0.620962/0.597601/0.642892`, pooled `0.626888`. The closest strict alternatives remain below it: M103 pooled `0.625439`, M114 `0.626267`, M154 `0.624908`, while generic library ensemble M134, KNN residual M106, sequential residual M120, and source-rank M152-M154 do not improve the full strict surface.
- Therefore another generic parent blend, library ensemble, KNN residual correction, or source-rank reuse would duplicate negative evidence. The next research question must target a genuinely different forecast-decision mechanism and retain M107 as the strict comparator.
+
### 2026-08-03 M253 primary-literature decision audit

- Primary-source search was restricted to forecasting-to-decision papers; no external data, pretrained weights, or competition score information was used.
- Muñoz, Morales & Pineda (arXiv:1907.07580, v3) formulates renewable forecasting/trading as a feature-driven data-driven newsvendor problem and reports that decision-aligned forecasts can improve both forecast error and balancing cost. This supports optimizing the actual settlement utility, but it overlaps conceptually with the repository's existing quantile expected-utility module and is not sufficient novelty by itself.
- Bruninx et al. (arXiv:2505.05153, v3) treats wind bidding as stochastic optimization under uncertainty with explicit risk constraints/certificates; its main transferable lesson is to constrain risky utility-seeking actions rather than maximizing noisy empirical utility without a trust region.
- The analog-ensemble paper result page was inaccessible (publisher 403), so no claim from its full text is adopted. The local M244/M248 analog-utility lane remains closed by live experiment evidence.
- Candidate M253 hypothesis: strict prequential empirical residual scenarios around the frozen M107 point forecast, exact official 6%/8% settlement utility, and a fixed risk/trust constraint. This is potentially distinct from the existing seven-quantile expected-utility path, but implementation is authorized only if the local novelty audit finds no equivalent scenario optimizer.
+
- Local novelty audit rejects the proposed M253 empirical-residual Bayes action as a duplicate. `run_crossfit_residual_pls.py` already cross-fits a point model, learns a residual probability mass, evaluates the exact official 6%/8% utility over a dense fixed action grid, and applies the maximizing action. `run_conditional_residual_gmm.py` already supplies the simpler group/lead/point-conditioned residual-density variant.
- Direct event learning is also already covered by `run_action_event_classifier.py` (P[error<=6%], P[error<=8%], expected error for each action), while `run_ordinal_cdf.py`, `run_coarse_settlement_classifier.py`, and the analog/GMM paths cover calibrated CDF, band-classification, and scenario variants. Therefore neither finer threshold mass nor empirical residual scenarios constitute a genuinely new lane.
- M253 is closed before execution. No candidate, score call, 2024 evaluation, selection, upload, or new external artifact was created. The literature insight is retained only as rationale for conservative action constraints, not as an experiment license.


### 2026-08-03 live Dacon read-only checkpoint

- Authenticated Chrome showed public rank 703 and exactly two historical submissions. Both rows have Total `0.6236623936`, 1-NMAE `0.8730574493`, and FICR `0.3742673379`; no M252 or other new user-side upload is present.
- The current rank-20 row is Total `0.66010`, 1-NMAE `0.87345`, FICR `0.44675`; rank 21 is Total `0.66006`. Thus Total `>0.66000` is the formal goal, while `0.66010` is the current practical finals cutoff and may move.
- Browser actions were read-only. No file upload, memo entry, submission click, final-save click, or account mutation occurred. M252 remains an unscored first-priority candidate.


### 2026-08-03 deployment-proxy integrity audit

- The historical values `0.66225678`, `0.66493939`, and `0.66647715` were computed by adding local development uplifts to `0.65971`. Live Dacon evidence proves `0.65971` was a rank-24 competitor row, not the authenticated user's baseline; the user's two actual submissions both score `0.6236623936`.
- Therefore those values are not candidate score estimates and must not rank candidates by expected online Total. They remain only arithmetic sensitivity examples: “if an unknown deployment parent scored 0.65971 and the local uplift transferred exactly.” Neither premise is established for the user account.
- M252 remains reproducible and deployment-aligned, but its first-priority status is now based only on frozen-policy/full-history construction and its local development evidence, not the invalid conditional proxy. No current artifact proves or credibly estimates Dacon Total above 0.66000.
### 2026-08-03 official code-share discovery audit

- Official-domain search for competition `236727` surfaced the evaluation-formula share (`codeshare/14035`) but did not expose participant modeling code shares.
- A direct read-only open of `https://dacon.io/competitions/official/236727/codeshare` was rejected by the web reader as an unsafe/unopenable URL; this is a discovery-surface limitation, not evidence that no code shares exist.
- The search cache contained an older leaderboard snapshot, so it must not replace the authenticated live cutoff evidence (rank 20 Total 0.66010).
- No new modeling intervention was established from this pass; no code, data, weights, submission, or account state was changed.

### 2026-08-03 repository re-orientation for the next strict lane

- The durable implementation is modularized under `src/baram/{features,models,decisions,evaluation,inference,submission}` with contract, leakage, prediction-registry, and promotion tests.
- The repository already contains multiple frozen submission variants (sequence, analog, recency, spread, rare-event, and M252 full-history Q1 group-1/2 analog), so a new lane must establish a non-duplicate decision rule or representation before execution.
- Standard tracked-file discovery did not surface the historical `run_*` screening scripts; those live in the hidden planning/experiment workspace and must be located explicitly before claiming novelty.

### 2026-08-03 hidden experiment inventory and operating constraints

- Current project authority reconfirms: one root session only, official supplied data only, local Python 3.12, at most six workers, and no additional use of the 2024 lockbox.
- Explicit hidden-file inventory located the complete experiment surface under `.planning/2026-08-01-leaderboard-top-4-loop/`, including analog, source-rank, daily profile/PCA, spectral, sequence, quantile/distribution, classifier, and exact-utility families.
- Because the inventory already covers daily multi-output, spectral-grid, sequence-wind, residual utility, ordinal CDF, quantile utility, and metric-aligned training, frequency-aware/profile work is presumptively duplicate unless its objective and strict validation pathway differ materially.

### 2026-08-03 target-aligned metric and opportunity recheck

- The exact local evaluator confirms `Total = 0.5 * (1-NMAE) + 0.5 * FICR`, with equal averaging over the three groups and evaluation only where actual generation is at least 10% of group capacity.
- The user's revised completion criterion is Total strictly above `0.66000`; 1-NMAE and FICR remain mandatory diagnostics but are not independent completion gates. Older plan text imposing separate component thresholds must not override this revision.
- Settlement utility is discontinuous: normalized absolute error `<=6%` earns four units, `(6%, 8%]` earns three, and `>8%` earns zero. The existing code already contains median residual shifts and quantile-derived Bayes-action machinery, while the hidden experiment inventory covers denser exact-utility variants.
- M107 remains the strongest strict pre-2024 pooled comparator at Total `0.626888`; M252 remains reproducible but unscored. No inspected local result currently proves or credibly predicts Total `>0.66000`.

### 2026-08-03 baseline and manifest audit

- The supplied baseline averages every weather grid by timestamp, adds basic calendar cycles, median-imputes, and fits three independent `RandomForestRegressor` models. The current repository already subsumes and materially extends this baseline with source-specific spatial summaries, physical/vector features, chronological validation, group-aware models, distributions, and decision layers.
- Therefore re-running or lightly tuning the baseline is not a credible `>0.66000` lane.
- The first scalar projection of `v2_data_contract.json` used keys that are absent at the document root, while the point-model projection accidentally expanded the embedded feature manifest. No claim is taken from null fields or truncated output; future reads must query exact schema paths or use compact scalar extraction.

### 2026-08-03 evidence-lineage consistency checkpoint

- The durable v2 point-model report is an earlier surface: `P0_SHARED_L1` pooled Total `0.575716`, with no accepted challenger. It is not the later M107/M252 evidence lineage and cannot rank those candidates.
- Hidden progress records show that M107 was temporarily invalidated after a complete-issuance-batch audit, then later reused as the reproducible strict comparator after boundary/lineage work. Before any new candidate inherits M107, its exact receipt and rebuilt strict-surface provenance must be verified rather than relying on the label alone.
- M107-dependent experiment families are extensive (daily PCA/multi-output, sequence, spectral, wake/turbine/site-wind, booster, utility/classifier, source-rank, analog transfer). A new intervention must state both why it is not one of these families and which exact strict artifact it compares against.

### 2026-08-03 strict-parent receipt and active stop-gate check

- M107's exact prediction and receipt are byte-pinned by the M251 runner (`3539cada...e537d` and `0167aac1...3ea`) and guarded against 2024 evaluation, lockbox reopen, and external actions. The final three Q4 rows are handled as one already-issued boundary batch retained exactly from M107.
- The active file plan correctly records the revised stop gate: continue score-feedback iterations until Dacon Total is strictly greater than `0.66000`; component scores are diagnostics only. It also correctly prohibits Codex-side upload and requires user-side upload before authenticated read-only verification.
- M252 is therefore a measurement candidate, not a completion result. Additional local experiments can improve evidence quality but cannot establish the online stop gate without an external Dacon score.

### 2026-08-03 experiment-density and autoregressive-feature audit

- The metric-aligned probe directory contains 666 files and covers at least M1-M251 across many family/quarter variants. This confirms that novelty must be judged against artifacts, not only source-module names.
- Target/power/SCADA lag search found no inference-safe implementation because official test inputs contain GFS and LDAPS only; SCADA and targets are train-only. A lagged-generation lane would require unavailable test-period observations and is therefore invalid, not merely untried.
- The useful remaining search space is restricted to transformations of supplied forecast-time GFS/LDAPS plus static training-derived knowledge that does not require rolling test outcomes.

### 2026-08-03 decision-family receipt schema audit

- Metric-aligned receipts use family-specific scalar fields rather than a common `score` key. The initial generic projection returned null and is not evidence of performance.
- Exact-utility coverage is confirmed by receipt schemas: M169/M171 store raw and oracle action policies; M177/M179 store fold scores and strict inner selections; M186 stores conditional-GMM fold score; M207 stores point/raw/final scores with cross-fitted residual PLS; M211 stores raw/final action-event scores.
- The next comparison must extract those exact fields and distinguish raw deployable scores from oracle blend diagnostics; oracle fields cannot promote a candidate.

### 2026-08-03 exact-utility performance and algebra checkpoint

- On unseen Q3, deployable decision-family scores remained sub-target: M169 raw `0.595950`, M171 raw `0.600560`, M177 `0.590591`, M179 `0.588787`, M186 `0.582357`, M207 fixed-parent blend `0.598183`, and M211 fixed-parent blend `0.602281`.
- M211's direct event head predicts unweighted `P(|y-a|<=6%)`, `P(|y-a|<=8%)`, and expected absolute error, then multiplies settlement probability by an action-independent point estimate divided by a group mean. That is not generally equal to the official conditional numerator `E[y * settlement_unit(|y-a|)] / (4 E[y])` because generation and the hit event are correlated.
- This is only a candidate algebra gap until M169 and related metric-aligned runners are inspected: if any already learns the actual-weighted reward directly under a strict outer fold, the proposed correction is duplicate.

### 2026-08-03 actual-weighted reward novelty decision

- M169 directly regresses the exact action utility target `-|y-a| + y * settlement_unit(|y-a|)/(4*group_mean)` on an expanded action grid. Therefore direct actual-weighted settlement reward is already implemented and the proposed algebra correction is not novel.
- M171 trains 6%/8% event classifiers with generation-weighted samples and multiplies their weighted event probabilities by a conditional point estimate over the group mean; this is an approximation to the same official numerator, not a wholly unweighted event formulation.
- M211's later unweighted event form is weaker algebraically, but correcting only M211 would repeat M169/M171's already-negative unseen-Q3 evidence. Close the actual-weighted action-regression lane rather than launching M254 from it.

### 2026-08-03 metric-native Q-function duplication check

- M128 separately fits two action-value regressors for expected absolute error and the exact generation-weighted settlement target `y * units / (4 * group_mean)` over target-independent action samples, then evaluates a dense 0.005 action grid.
- This independently confirms that exact actual-weighted reward modeling is already represented beyond M169/M171. M128 is explicitly a same-fold representation screen and selects iteration/gamma on its validation fold, so its selected score cannot be treated as strict promotion evidence.
- No browser or account interaction was needed for this audit; the official code-share discovery limitation does not justify weakening the local novelty gate.

### 2026-08-03 metric-native screen result and leaderboard schema note

- M128's optimistically selected same-fold Q4 result was only Total `0.624155`, 1-NMAE `0.851897`, FICR `0.396412`; it remains below M107's Q4 `0.642892` despite selecting 300 iterations and gamma 1.0 on Q4 itself.
- This closes direct action-value regression as a primary improvement path on the existing feature surface.
- The live leaderboard receipt stores values under `authenticated_user`, `top_rank_landmarks`, `submission_history`, and `result`; a generic `user/cutoff/rows` projection returned null and must not be interpreted as missing live evidence.

### 2026-08-03 official code-share listing resolved in Chrome

- A read-only Chrome snapshot of the official competition code-share listing resolved the discovery ambiguity. The listing currently contains only two competition entries: the official RandomForest baseline (`codeshare/14031`) and the official evaluation-formula post (`codeshare/14035`).
- No participant modeling code share is available on that official listing, so there is no omitted public competition-specific SOTA recipe to import or benchmark from this surface.
- No upload, code post, login, agreement, submission, account, or team action was taken. The temporary public listing tab will be closed immediately after recording this finding.

### 2026-08-03 frequency/ramp novelty checkpoint

- Existing code covers output ramp smoothing, neighboring-NWP context, daily phase alignment, spatial weather gradients, high-output hurdle/tail heads, spectral-grid summaries, and target-magnitude sample weighting.
- Search did not find a model trained with target temporal-gradient/frequency error weighting itself. That narrow training-objective idea is not textually duplicated by M175 high-output modeling or M229 post-hoc smoothing.
- Novelty alone is insufficient: the ICLR frequency-informed method must be checked at its primary source to determine whether a fixed classical sample-weight adaptation is technically justified and can be evaluated without another broad hyperparameter sweep.

### 2026-08-03 ICLR 2026 frequency-informed applicability audit

- The primary OpenReview record identifies three coupled contributions: a gradient-penalized loss, a physics-embedded/Navier-Stokes-style model, and explicit frequency separation/reweighting for spatial wind-velocity fields. Its reported target is extreme wind-field RMSE, not wind-power FICR.
- The paper attributes extreme-wind underestimation to high-frequency amplitude shrinkage under spatial pattern shifts. This supports the general concern but does not establish that target-gradient sample weights in a tabular power model reproduce the proposed method.
- A classical row-weight proxy would discard the paper's differentiable gradient penalty, spatial field structure, and frequency-separated architecture; launching it as “SOTA-based” would overstate transfer validity. Exact formula/ablation details should be checked before any one-shot adaptation, and a new dependency or deep architecture remains outside the current plan.

### 2026-08-03 frequency-informed lane decision

- Direct OpenReview PDF access remained behind its browser-verification challenge. Indexed primary PDF text confirms only the coupled gradient-loss/physics-backbone/frequency-separation intervention and its spatial extreme-wind ablation, not a tabular sample-weight recipe.
- The bounded exact-formula search did not surface a source-backed classical reduction. Because M175, M185, M205, M229, and M230 already cover the available official-data high-tail, spectral, atmospheric, smoothing, and neighboring-sequence operations, an ad hoc target-ramp weight would be a weak proxy rather than faithful SOTA transfer.
- Close the proposed frequency-weighted M254 before execution. No model, score, candidate, test CSV, 2024 evaluation, or external Dacon action was created.

### 2026-08-03 automatic-continuation target reconciliation

- The automatic goal payload still contains the superseded simultaneous component thresholds. The latest explicit user instruction and active file plan remain authoritative: completion is Dacon Total strictly above `0.66000`; `1-NMAE` and FICR are diagnostics.
- M252 is still awaiting external measurement, but the goal continuation permits further official-data-only pre-2024 work. The next audit will test whether nonlinear within-day time warping is genuinely distinct from M238's single global lag and M242's local lead-window retrieval before authorizing any M254 runner.

### 2026-08-03 M255 nonlinear time-warp novelty audit

- Correct candidate numbering is M255 because M254 was already assigned to and closed for the frequency-weighted proposal.
- M238 keeps M234 day-level neighbors and chooses one constant lag per query-neighbor pair from `0, -1, +1, -2, +2`, then edge-shifts the entire historical target profile.
- M242 abandons day-level neighbors and independently retrieves 20 historical values at each lead from a fixed previous/current/next three-hour feature window.
- A monotone constrained DTW path would be structurally distinct: it retains each frozen M234 day neighbor while allowing local within-day phase variation and maps that neighbor's target profile onto query hours through the NWP alignment path. Primary-source evidence and a leakage-safe fixed path contract are required before implementation.

### 2026-08-03 M255 primary-source bounded research

- Alessandrini et al. (Renewable Energy, 2015) reports that its wind-power Analog Ensemble compares predictor trajectories over a time window and found wind speed/direction, a three-hour matching window, and 20 analogs effective in its dataset. M242 already tested the direct three-hour/same-lead transfer and failed, so this paper alone does not justify repetition.
- Martínez-Álvarez et al. (Renewable Energy, 2021) proposes dynamic time-scan analog forecasting for multi-step wind speed using pattern matching and dynamic similarity. It supports nonlinear temporal matching as a legitimate analog family, but its observed univariate history setup does not directly establish warping of NWP-selected historical power profiles.
- Constrained DTW is a genuine non-duplicate hypothesis, not a benchmark-backed expected win. The publisher pages returned 403 and the Stanford GDTW PDF timed out; only indexed primary abstracts are adopted, and no uninspected formula/result is claimed.

### 2026-08-03 M255 integration target decision

- M244 is the correct comparator rather than legacy M234. Its frozen policy uses exact M243 recency/spread logic plus a train-only high-hub-speed member correction, with group recipes `(20, exponential, utility)`, `(40, exponential, utility)`, and `(5, uniform, median)` and small fixed parent blend weights.
- A useful DTW test must warp both each neighbor's normalized target and hub-speed trajectory onto query hours before M244's high-wind correction; recency weights, slopes, thresholds, heads, transforms, and parent blend weights must remain frozen.
- Spread reliability must be recomputed from the warped corrected members, otherwise combining DTW members with the old unwarped spread gate would create an internally inconsistent policy. The exact M243 spread helper contract will be inspected before the runner is authorized.

### 2026-08-03 M255 staged experiment contract

- Full direct M244 integration would require DTW-consistent query and leave-one-out reference spread plus warped hub-speed correction. That is a larger coupled change and would obscure whether temporal alignment itself carries signal.
- M255 will therefore be a screening experiment on the exact M234 surface, matching M238 except for one factor: replace the single constant lag with an endpoint-anchored Sakoe-Chiba DTW path of radius two over the same train-standardized 25-core NWP trajectories.
- Each frozen M234 day neighbor remains unchanged; only its 24-hour historical target profile is mapped to query hours by the monotone DTW path, averaging multiple aligned neighbor hours per query hour. Day retrieval, neighbor counts, kernels, heads, transforms, parent surfaces, and blend weights remain frozen.
- Fixed path contract: local cost is mean squared standardized NWP difference, allowed moves are diagonal/vertical/horizontal with diagonal-first tie order, endpoints are `(0,0)` and `(23,23)`, and `|query_hour-neighbor_hour| <= 2`. No band, cost, move, feature, neighbor, group, or fold search.
- Promotion uses M238's exact rule against M234: positive parent deltas in Q2/Q3/Q4 and either a strictly better worst-fold delta than M234 or both higher Q4 delta and paired-bootstrap positive fraction. M255 creates no test CSV; only a pass would authorize separate M256 integration with M244.

### 2026-08-03 M255 result

- M255 passed the basic all-fold sign check: Total deltas versus the parent were `+0.005578` (Q2), `+0.000419` (Q3), and `+0.000735` (Q4).
- It did not beat the full M234 correction on the required robustness surfaces. M234's Q3/Q4 deltas were `+0.002619/+0.002683`, while constrained DTW delivered only `+0.000419/+0.000735`; M255's worst-fold delta was therefore much smaller.
- Q4 issuance-day bootstrap positivity was `64.25%` for M255 versus `93.65%` for full M234. The constrained warp weakened, rather than stabilized, the correction.
- State is `LOCAL_CONSTRAINED_DTW_REJECTED_NO_TEST_BUILD`; receipt SHA-256 is `cf3bfac6762d3f743858a0c2103029b083d5294d4cf5dc2464ff66068a0257f2`, prediction SHA-256 is `511cca27a484e68b6562d7fe632c170f48674d532c6d790aee9bf4a10bb4419a`, and the receipt confirms no 2024 evaluation, lockbox reopen, online score, external action, or submission build.

### 2026-08-03 post-M255 remaining-lane inventory

- The implementation inventory already spans point/tree/boosting, ordinal and distributional heads, utility/Q-function policies, residual and pseudo-label paths, daily multi-output/sequence/spectral models, spatial/site-wind and wake features, cross-group stacks, analog retrieval variants, source separation, quantile mapping, EMOS, and constrained temporal alignment.
- Artifact filenames also confirm broad hyperparameter, recency, weather-feature, source, settlement-action, and ensemble screens. A next lane must therefore be justified at the operation level, not merely by renaming a known model family.
- The filename inventory is navigation evidence only. Candidate novelty and benchmark support will be checked against exact code/receipt fields before any new run is authorized.
- The approved design and implementation contracts remain modular around data quality, supplied-NWP features, deterministic/probabilistic models, expected-utility decisions, ensembles, temporal challengers, and self-evaluation. Any new lane must stay within IP@v2's frozen chronology, experiment-accounting, promotion, final-fit, and no-upload boundaries.
- Exact M244 evidence confirms its remaining correction is narrowly defined: a 365-day recency-weighted analog ensemble, a high-hub-speed train-only OLS member correction, and fixed per-group utility/median heads blended lightly into the v2 parent. A next candidate should target a missing source of conditional error rather than refit those knobs.
- M107's receipt uses an older schema and cannot be compared via the M24x scalar projection; inspect its exact keys and companion runner before making any cross-family claim.
- M107 is not a new feature model: it is a Q2-selected, per-group fixed temporal blend of M103 with shifted predictions. Frozen selections are group 1 `(70% original, -1h shift)`, group 2 `(80%, -2h)`, and group 3 `(80%, -2h)`. Its pooled Q2-Q4 Total is `0.626888`, with Q4 group 2 strongest at `0.671912` and group 3 weakest at `0.607095`.
- Because dynamic phase selection already has a dedicated runner, a conditional-shift idea is only novel if its exact implementation evidence shows a missing conditioning signal; do not reopen fixed or generic dynamic phase blending from M107 alone.
- The existing dynamic phase selector already trains a weather-conditional LightGBM multiclass model over offsets `-3..+3`, includes raw shifted-candidate shape features, screens supplied-NWP features, evaluates shared/group architectures, and applies confidence-floor soft probabilities. It was a legacy source-rank diagnostic trained on Q2 and evaluated on Q3, so generic weather-conditional phase selection is already covered.
- A new phase lane would require a specific missing contract such as constrained probabilistic timing uncertainty on the current parent, but that would overlap M107/M151 plus the rejected M251 transfer unless supported by unusually strong residual evidence. It is not the default next candidate.
- The repository's durable implementation already exposes canonical archive/data modules and structured weather, physics, spatial, geometric, sequence, climatology, deterministic, probabilistic, expected-utility, ensemble, evaluation, and final-inference components. Raw inspection should go through those approved surfaces rather than assume a top-level `data/` directory.
- The original module research leaves TFT and NGBoost only as conditional challengers, but IP@v2 explicitly excludes their execution and any new dependency. The current lockfile has LightGBM, scikit-learn, CatBoost, and XGBoost only; deep/TFT/NGBoost therefore cannot be used as the next lane under the active authority.
- Existing scikit-learn MLP distribution/ordinal screens already failed, so a nominally neural label without TFT's architecture would neither satisfy the research hypothesis nor add an unused family.
- Physics-monotone boosting, empirical/isotonic power curves, turbine-level wind teachers, high-output tails, and multiple power-curve proxies are already implemented and rejected or subsumed. A new “physical power curve” label alone is not novel.
- Label-free test-period adaptation is also closed: the existing domain classifier already excluded calendar fields, estimated clipped NWP-only history-to-target density ratios, and failed unseen Q3. CORAL/MMD-style transductive relabeling has no local positive premise and cannot be opened as a cosmetic variant.
- Exact Python-only search found no Tweedie objective implementation. Current first-party LightGBM allows only L1/Huber/quantile; the separate turbine-wind stack screens L1/Huber/L2; XGBoost screens absolute, pseudo-Huber, squared, and quantile losses. Tweedie is therefore an operation-level gap, but novelty alone is insufficient: primary-source suitability and a fixed variance-power contract are required before execution.

### 2026-08-03 Tweedie and local-GP research checkpoint

- Official LightGBM 4.7 documentation confirms a built-in Tweedie regression objective with a log link, but describes total insurance loss as the motivating use case. The initial primary-source search did not find a direct wind-power Tweedie benchmark or a supported variance-power choice for this task.
- The search instead recovered Yan et al. (IEEE TSTE 2016, DOI `10.1109/TSTE.2015.2472963`), which uses a moving-window Gaussian process on forecast errors and reports improved wind-power accuracy at two farms. Its transferable operation is temporally local residual modeling, not Tweedie loss.
- Temporally local residual learning is already substantially represented by recency windows/weights, RBF-SVR on supervised projections, conditional residual distributions, analog retrieval, and strict prequential residual models. A full GP would also face cubic scaling without an approved sparse approximation. Neither finding yet authorizes M256.
- The hidden-inventory synthesis reconfirms that no inspected local result credibly predicts `>0.66000`, and any new model must transform only supplied forecast-time GFS/LDAPS plus static train-derived knowledge. This sharply limits “new” ideas that depend on rolling target or SCADA observations.
- Generic residual bias/MOS calibration is already present through pooled/group/group-lead median shifts, expected-utility calibration, KNN residual utility, EMOS moments, and M244's train-only high-wind correction. Another lead/hour residual table would be duplicate unless driven by a new causal conditioning variable.
- Daily-profile methods already predict complete 24-hour target vectors and explicitly test level, mean-centered shape, and mean-scaled analog transforms. A naive “daily total plus hourly shape” proposal overlaps these operations; exact supervised target-basis code must be checked before treating factorization as novel.
- The supervised target-basis PLS runner operates row-wise over nonlinear target basis functions, while the donor multi-output PLS models simultaneous donor-group targets. Neither creates an independently trained daily aggregate forecast or reconciles hourly forecasts to that aggregate.
- Exact repository search found no forecast-reconciliation, MinT, temporal-hierarchy, daily-aggregate model, or coherent hourly/daily constraint implementation. A daily-energy reconciliation layer is therefore a genuine operation-level gap, distinct from direct 24-output prediction and analog profile transforms.

### 2026-08-03 temporal-reconciliation primary research

- Athanasopoulos et al. (EJOR 2017) defines temporal hierarchies through non-overlapping aggregation and combines independently generated forecasts into coherent predictions; the method is model-agnostic and is reported to improve robustness when model uncertainty is high.
- Jeon, Panagiotelis, and Petropoulos (EJOR 2019, DOI `10.1016/j.ejor.2019.05.020`) evaluates probabilistic temporal reconciliation on wind power and electric load at hourly-to-daily frequencies and reports gains in point as well as probabilistic accuracy.
- Sharma, Bhakar, and Jain (Energy Conversion and Management 2024, DOI `10.1016/j.enconman.2023.118053`) directly studies temporal reconciliation for a wind farm. The publisher open failed, so only indexed primary abstract claims are adopted.
- Abolghasemi, Girolimetto, and Di Fonzo (arXiv `2412.11153`) argues for validation-error rather than in-sample covariance estimation in wind cross-temporal reconciliation and evaluates both accuracy and penalty cost. This directly matches the competition's split accuracy/settlement objective, while also warning that hierarchy and covariance choices matter.
- A bounded temporal-reconciliation candidate should therefore use one daily level plus the 24 hourly leaves and a closed-form validation-error reconciliation—not ratio tuning or a reconciliation-method sweep. Exact parent/inner-OOF feasibility remained to be verified before authorization.

### 2026-08-03 M257 parent-feasibility checkpoint

- The M234 lineage reads frozen v2 parent OOF predictions only for Q2/Q3/Q4; it does not expose an already-built pre-Q2 hourly OOF block suitable for estimating a 25-node reconciliation covariance before Q2.
- Building nested inner M234 forecasts solely to estimate a full MinT covariance would add a second complex model layer and weak-identifiability risk. It would no longer be the bounded one-factor test justified by the literature.
- A simpler model-independent top-down screen remains feasible: independently predict each group/day normalized mean from the 24-hour supplied-NWP trajectory, then project the unchanged hourly parent profile onto the nonnegative bounded set whose mean equals that daily forecast. This uses no covariance or blend grid and makes the daily/hourly hierarchy exactly coherent.
- Exact daily model/features/box-projection and promotion rules must be frozen before execution. The comparator should be the same exact M234 surface used for the prior analog screens, with Q2/Q3/Q4 trained only on preceding complete days.
- The prior M121 daily multi-output screen is negative as a standalone model (reported latent daily peak about `0.585086`), but its Q4 same-fold oracle blend reached `0.644884` versus M107 Q4 `0.642892`, using only a 10% direct-model contribution for group 2 and zero for groups 1/3. This is not promotable evidence, but it shows daily-level forecast diversity concentrated by group.
- M121's selected Q4 direct family was strongly regularized Ridge (`20` features, `alpha=1000`). Reusing that fixed regularization on a scalar daily mean is a defensible no-search prior; the new operation is chronology-safe validation-learned daily combination plus coherent projection, not another direct 24-output sweep.
- M107 is the correct M257 comparator because M121's diversity premise was measured against it and its exact Q2/Q3/Q4 OOF surface is byte-pinned. Prediction SHA-256 is `3539cada59f88a16d4b4181f5aff3c76ff8e9a94954f67f4204ccd09ac8e537d`; receipt SHA-256 is `0167aac129b2afd1a004a3612d32bda7d0916757fc20a38a805950a8d92b93ea`.
- The earlier provisional M234 comparator is superseded for M257. M234 remains relevant to the analog lineage, but temporal reconciliation should first test the stronger strict parent that already has complete Q2-Q4 hourly OOF predictions.
- M107 OOF contains only the key, actual, prediction, fold, and model columns; availability metadata must be merged one-to-one from the physical pre-2024 surface. Calendar-quarter boundaries split some issuance batches, so M257 must reconcile only exact 24-row `(group, issuance)` blocks and leave every partial boundary block byte-identical to M107.
- The existing `_complete_group_days` helper already enforces 24 chronological hours and returns a `days x 24 x features` tensor plus normalized targets. M257 can reuse this topology with the frozen 25-core supplied-NWP feature set and no new data surface.

### 2026-08-03 frozen M257 experiment contract

- Parent: exact byte-pinned `M107_STRICT_TEMPORAL_TOP100` Q2/Q3/Q4 OOF predictions. Q2 output is the parent unchanged and exists only as the first validation-error calibration block.
- Daily base: for each group/fold, flatten the exact 24x25 M234-core supplied-NWP trajectory; median-impute, standardize, and fit `Ridge(alpha=1000)` to the normalized 24-hour target mean. Sample weights are the training daily mean clipped below at `0.10`, matching the prior M121 daily weighting. Training days must be complete, finite-target, and end strictly before the first complete query issuance.
- Sequential reconciliation weights: for Q3/group, fit `w=clip(sum((d-h)*(y-h))/sum((d-h)^2),0,1)` on Q2 daily triples; for Q4/group, fit the identical formula on pooled Q2+Q3 triples. Here `d` is daily-model mean, `h` parent hourly mean, and `y` actual daily mean. Zero denominator deterministically gives `w=0`. No intercept, regularizer, grid, or group exception.
- Coherence: reconciled daily mean is `h + w*(d-h)`. For each complete 24-hour group/issuance block, find the unique additive shift whose componentwise clipping to `[0,1.075]` has exactly that mean. This is the Euclidean projection that minimally changes the M107 hourly shape. Partial boundary blocks remain exactly unchanged.
- Promotion: Q3 and Q4 Total deltas must both be strictly positive, pooled Q2-Q4 Total must strictly improve, and Q4 issuance-day paired-bootstrap positivity must exceed `0.50`. `1-NMAE` and FICR are recorded diagnostics, not independent gates. No test artifact is produced by M257; only a pass can authorize a separately staged deployment integration.
- Numbering: M256 stays unused/reserved because M255 failed the prerequisite for its planned DTW-to-M244 integration. The unrelated temporal-reconciliation lane is M257.
- Prohibited: alternate feature sets, alpha/model/objective, ratio versus additive reconciliation, covariance/MinT variants, clipping bounds, weight pooling, quarter/group exceptions, target smoothing, post-result blending, 2024 evaluation, and any upload.
- Implementation can reuse M251's verified M107 loader, pooled-frame helper, official scorer, and issuance bootstrap without inheriting its analog correction. This preserves the exact boundary fallback and parent-score reproduction guards while keeping M257 independent of M244.
- 2026-08-03 user override: the final target is DACON `Total > 0.66000` only. Component scores remain diagnostics; any older simultaneous-component objective is superseded.
- M257 shows that improving daily aggregate error is not sufficient for this competition objective. Its sequential daily weights were positive for every group, yet pooled FICR fell `0.071215` and Total fell `0.036997`; the hourly settlement threshold geometry is highly sensitive to uniform profile shifts.
- M258 is a bounded likelihood-diversity test, not a claim that Tweedie is established SOTA for this dataset. Novelty is exact-source confirmed; the fixed variance power `1.5` comes from the LightGBM default, and the maximum 20% blend is a predeclared trust region because direct XGBoost L1 was weak but showed small-model diversity in same-fold diagnostics.
- M258 confirms another objective conflict: group-3 Tweedie blending improved Q3/Q4 `1-NMAE` by `+0.000397/+0.001800` but reduced FICR by `-0.002689/-0.011512`. A smooth positive likelihood moves predictions toward conditional means and away from the narrow 6%/8% settlement bands; likelihood-family substitutions are not justified as a rescue.
- Latest authenticated read-only leaderboard state: user rank 704, two submissions, displayed `0.62366 / 0.87305 / 0.37426`; no M252 upload is present. Rank 20 is `0.66026 / 0.87358 / 0.44694`, rank 21 is `0.66010`, and rank 22 is `0.66006`. The practical finals cutoff has moved above the formal `>0.66000` goal.

## M252 official online result and loop reactivation

- The user supplied two DACON screenshots for submission `1509527`, filename `submission_M252.csv`, memo `m252`, submitted `2026-08-03 14:47:25 KST`. The row reports Total `0.6268784092`, 1-NMAE `0.8659033246`, FICR `0.3878534938`; the leaderboard row reports rank `657` with three submissions.
- The displayed Total exactly reconciles as the arithmetic mean of the two components within floating precision. Relative to the prior selected row (`0.6236623936 / 0.8730574493 / 0.3742673379`), M252 changes Total by `+0.0032160156`, 1-NMAE by `-0.0071541247`, and FICR by `+0.0135861559`. The gain is therefore a settlement-score improvement partly offset by worse point accuracy.
- The formal target gap is `0.0331215908`. Holding M252's 1-NMAE fixed would require FICR `0.4540966754`; restoring the prior 1-NMAE would still require FICR `0.4469425507`. This rules out treating another tiny analog blend improvement as sufficient evidence for completion.
- The two supplied images are preserved at SHA-256 `60d567...d2f2c` and `45f6cc...6c665`; the score-feedback receipt is `reports/dacon_m252_online_2026-08-03_receipt.json`. The user performed the upload. Codex performed no upload, submission selection, memo edit, or account mutation.
- M252 remains reproducible but is now externally rejected as a completion candidate. The goal is active again and the next fixed local screen is M259; any later submission candidate still requires user-side upload and a fresh external score.

## M259 predeclaration: D1 predictive-width shrink of M244

- Hypothesis: M252's online tradeoff suggests the analog correction improves threshold capture but can over-correct point predictions. Existing D1 quantiles provide a forecast-time uncertainty signal aligned to the M244 v2/M231 lineage. M259 shrinks only the exact M244-minus-parent correction when D1's normalized `q90-q10` width exceeds a chronology-safe reference; it does not alter the parent, analog retrieval, members, heads, transforms, or rare-event rule.
- Width is `(q90-q10)/capacity`. Q3 uses the group-by-target-hour 75th percentile estimated from Q2 widths; Q4 uses the same statistic from pooled Q2+Q3 widths. The multiplier is exactly `min(1, reference/max(width, tiny))`. Q2 remains exact M244 and is not used as an application-score claim.
- Promotion is fixed before code execution: Q3 and Q4 Total deltas versus exact M244 must both be strictly positive, pooled Q2-Q4 Total must improve, and the Q4 paired-issuance bootstrap positive fraction must exceed `0.50`. Component deltas are diagnostics. No quantile-pair, reference-quantile, exponent, floor, threshold, group/hour exception, blend weight, parent, fold, or test-build rescue may be selected after seeing the result.

## M259 result: rejected

- Q2 is byte-equivalent to exact M244 by contract. Frozen application reduced Q3 Total by `0.0006802209`, Q4 by `0.0004914872`, and pooled Q2-Q4 by `0.0003635833`. The associated Q3/Q4/pooled `1-NMAE` deltas were `-0.0000082371/-0.0001804850/-0.0000774608`; FICR deltas were `-0.0013522048/-0.0008024894/-0.0006497058`.
- Q4 paired-issuance bootstrap positivity was `0.221`, far below the frozen `>0.50` gate. Predictive-width shrinkage therefore did not recover M252's point-accuracy loss and actively weakened settlement capture on the strict development folds.
- Exact rerun reproduced OOF SHA-256 `357f1e47989a42a710fffeac0705a8248bb6c677183c46f9b60a8f1a30217b5d` and receipt SHA-256 `280542fec890b9b6c9c04189a21b5f6437972ad22242eb52e4a58e6178ca7498`. State is `LOCAL_D1_WIDTH_SHRUNK_M244_REJECTED_NO_TEST_BUILD`; no test CSV, 2024 evaluation, upload, account action, remote compute, or dependency change occurred.
- The next audit is FICR-directed rather than another analog micro-adjustment. Existing source-rank classifiers show materially higher same-fold FICR (`0.4269` Q2 and `0.4449` Q4 in their strongest oracle screens) but weak chronology-safe transfer, so any deployment candidate must explicitly separate those evidence surfaces and freeze policy selection before a new run.

## Source-rank deployment audit: closed

- Same-fold M150/M151 scores are representation ceilings, not forward evidence: their policies and parent blends were chosen on the scored Q2/Q4 labels. The corresponding chronology-safe policy transfer is materially weaker. M154 leaves Q2 at M107, then scores `0.598263` on Q3 and `0.638341` on Q4, for pooled Total `0.624908`, below M107's `0.626888`.
- M189 removes the more subtle site-wind meta-feature leakage by generating NWP-to-SCADA proxies prequentially before fitting the source-rank classifier. Its frozen Q3 Total is `0.606307`; even the diagnostic outer-label oracle is `0.609457`. This shows that the high same-fold FICR does not survive strict feature/model chronology.
- A 2025 full-history source-rank build would additionally require a new cross-fitted site-wind training surface and full-history test teachers. That implementation is feasible with supplied inputs, but the strict evidence says it is not the highest-value use of the remaining external score checks. The lane is closed without code, test CSV, 2024 evaluation, or external action.

## M260 predeclaration: multivariate analog regression correction

- Hypothesis: exact M243 retrieves the correct daily weather regime but copies each neighbor's normalized output without correcting the remaining local physical-state difference. M244 corrects only high-hub-speed rows with one group-level scalar slope. M260 instead uses a standard analog-regression operation: train-only group/hour ridge coefficients translate every neighbor from its physical state to the query state before the frozen analog head.
- The four inputs are fixed before execution: `phys_v2__hub117_speed`, `phys_v2__shear_alpha_100_80`, `phys_v2__air_density`, and `phys_v2__fleet_power_proxy_w`. Within each group/hour they are standardized from strictly preceding complete days; a fixed ridge penalty of `10.0` is applied to the four slopes while the intercept is unpenalized. Member adjustment is `beta dot (query_x - neighbor_x)` and the adjusted normalized target is clipped to `[0,1]`.
- Exact M243 retrieval, neighbor counts, kernels, 365-day recency weights, heads, transforms, blend weights, parents, and M240 spread multiplier remain frozen. Promotion requires positive parent-relative deltas on Q2/Q3/Q4 and either a strictly better worst-fold delta than exact M244 or both a strictly better Q4 delta and Q4 paired-bootstrap positive fraction. No coefficient cap, feature substitution, penalty search, hour pooling, correction blending, group/fold exception, or post-result test build is allowed.

## M260 result: rejected

- Parent-relative Total deltas are `+0.0031819755` on Q2, `-0.0009929830` on Q3, and `+0.0000094054` on Q4. Relative to exact M244, M260 loses `0.0027387267/0.0027941187/0.0044255963` on Q2/Q3/Q4. Q4 paired-bootstrap positivity is `0.514`, versus `0.9955` for M244.
- The regression changed analog members too aggressively: group diagnostics show mean absolute adjustments around `0.17` normalized generation in representative folds and material `[0,1]` clipping. Point accuracy remained close in Q4, but FICR fell from M244 `0.376088` to `0.367283`; broad linear state translation is not a safe settlement-score correction.
- Exact rerun reproduced OOF SHA-256 `e799e18127ca0e40e94bfa8a22738325f7a195f308926429098a96d028b1da01` and receipt SHA-256 `6d0cfb3455fab11c7beaa5d9a2d424fc5bf508e940c587f9e2eac840f7d97feb`. No test CSV, 2024 metric, upload, account action, remote compute, or dependency change occurred.

## M261 predeclaration: full-history deployment of M107

- M107 remains the highest strict pooled local candidate and belongs to an independent metric-aware classifier lineage, whereas the user-scored M252 belongs to the v2-plus-analog lineage. M261 therefore targets model-family diversity rather than another analog adjustment.
- Freeze exact Q2 evidence: M102's 100 named features, `0.02` normalized class width, 60 trees, original deterministic LightGBM parameters, generation weight power `1.0`, `T0.5_G1.5` action policy, and Q2 site-wind iterations `[395,170,164]`. Freeze exact M107 group temporal policies: `(-1,0.7)`, `(-2,0.8)`, `(-2,0.8)` for groups 1/2/3.
- Final fit may use every supplied 2022-2024 generation and SCADA row. Site-wind values for classifier training are three-fold cross-fitted under the original seeds, while test values come from models fitted to all supplied SCADA; raw observed SCADA is never passed to the power classifier or test inference. Test raw-grid/geometric features are reconstructed only from supplied GFS/LDAPS and turbine metadata with train/test column parity assertions.
- No 2024 score, time slice, policy selection, feature selection, iteration selection, blend, or calibration is allowed after final fit. The output remains externally unverified until a user-side DACON upload reports its components and Total.

## M261 result: reproducible full-history score-feedback candidate

- The supplied test NWP required one deployment-only data repair before geometric feature construction. GFS had no missing numeric value cells. LDAPS had exactly 752 missing cells across 19 variables at three forecast times; some wind-vector fields were missing on all 16 grids. M261 fills only those cells by linear interpolation over adjacent lead hours within the same issuance and grid, with bidirectional edge fill. This uses no labels, earlier online scores, or external data, and the receipt records the complete per-variable counts and zero unresolved gaps.
- The final surface contains 78,912 supplied training group rows and 26,280 test group rows. Legacy and all-weather SCADA-to-wind teachers generated training proxies by original-seed three-fold cross-fitting and test proxies by full-history fitting. Observed SCADA is never a power-classifier or test-inference feature. The selected test matrix has 100 frozen features and zero non-finite values.
- The final classifier consumes 41,220 official rows meeting the frozen normalized-generation eligibility rule, covers all 46 frozen `0.02` bins, uses exactly 60 deterministic LightGBM trees, and applies `T0.5_G1.5` followed by frozen group temporal transforms. No 2024 prediction-label comparison or score was created; class centers and group means are final-fit parameters, not validation metrics.
- `artifacts/submissions/submission_M261.csv` has 8,760 rows in exact sample order, UTF-8 BOM, no missing values, no duplicate keys, no negative values, and no capacity exceedance. Predictions span normalized `0.1300` to `0.95625`. CSV SHA-256 is `fb937f77dfe501b4d7f8e52da098f07f79b1924dace347d3020da405b660773b`; receipt SHA-256 is `0bbf2aae103ad1115ed5809d5747e965732b827d01ee4c381352eaec0622961c`.
- A complete second run reproduced the legacy train/test hashes, all-weather train/test hashes, classifier probability hash, prediction-array hash, CSV hash, and receipt hash exactly. Ruff, compilation, self-test, independent CSV schema checks, and the 6 GiB artifact budget all pass. The PASS is limited to deterministic construction, lineage, no-leakage-at-inference, and submission contract.
- M261 has no local or online test score and therefore provides no evidence yet that it improves M252's official `0.6268784092` or reaches `Total > 0.66000`. DACON upload remains user-only. The active loop cannot promote, reject, blend, or calibrate M261 from its 2025 predictions before external component feedback is observed.
### 2026-08-03 M262 exact-prior-year profile result

- M262 used only 2022 finite labels for groups 1/2 and retained exact M107 for group 3, incomplete 24-hour profiles, and the Q4 cutoff boundary. The first attempt stopped before scoring because 190 lookup labels were non-finite; excluding those rows activated the predeclared incomplete-profile fallback without imputation or policy change.
- The transfer was active on 4,368 Q2 rows, 4,272 Q3 rows, and 4,176 Q4 rows, but no nonzero mass improved both Q2 and Q3. Group 1's `0.025` mass produced Q2/Q3 deltas `-0.002840/+0.000886`; group 2 produced `-0.006489/-0.000911`.
- The frozen selection is all-zero, so Q4 and pooled scores remain exact M107 and no M262 test CSV is authorized. Very large daily rescaling factors on low prior-year-energy days explain part of the instability, but the predeclared no-rescue rule closes scale caps, alternate transforms, gates, and group/fold exceptions.
- Prediction SHA-256 is `553f62c33a760f7ed79478ce2177b05b235308a34507801be953f16f01586066`; receipt SHA-256 is `4f36bd7e55ddb02b7a38b027d18f5fbab51a70fcf92f14f6049724f90c8c7ffc`. No 2024 score, upload, browser/account mutation, external data, or remote compute occurred.
### 2026-08-03 M263 independent-lineage ensemble and deployment result

- The M244 full-prediction lineage is genuinely complementary to M107: it has better `1-NMAE` on every strict quarter while M107 has materially better FICR. The older M134 library did not contain M244 because it predates the analog lineage.
- Stability selection retained only group 2 with M244 mass `0.10`; groups 1/3 stayed exact M107. Q2/Q3/Q4 Total deltas were `+0.002311/+0.001787/+0.000462`; pooled Total/`1-NMAE`/FICR reached `0.628240/0.856023/0.400457`, improvements of `+0.001352/+0.001079/+0.001625`. Q4 bootstrap positivity was `74.9%`, so every predeclared gate passed.
- The full-history deployment is exactly group 2 `0.90*M261 + 0.10*M252`, with groups 1/3 equal to M261. Two builds reproduced CSV SHA-256 `5a1d701660a83291105a7172c03f6e1ae71250b67e7ac0c2378d7ca01d335819` and receipt SHA-256 `d8b6916a3b2239c773fac7f9d4e4d2f201d7707cab09e07ee42d9eb4a77706b1`.
- Submission validation passed for 8,760 exact keys, BOM, schema/order, finite nonnegative values, and all capacity caps; 17 targeted submission/budget tests passed. The maximum text round-trip blend error was only `3.64e-12 kWh` under the explicit `1e-10 kWh` guard.
- `submission_M263.csv` is built-not-uploaded with zero score calls, no 2024 metric/slice/selection, no post-fit tuning, and no browser/account/external action. Only a user-side DACON upload and read-only result can establish whether the formal `Total > 0.66000` target is met.

### 2026-08-03 M264 predeclaration: strict-family group-1 augmentation

- The authenticated read-only refresh still shows the user's M252 row (`0.62687 / 0.86590 / 0.38785`, three submissions); M261/M263 has not been uploaded or scored. The browser session was finalized without account or submission mutation.
- A bounded full-OOF inventory exposed one material lineage omission after M263: M95's group-1 output is chronology-selected from fixed M68/M72 classifier families, yet neither M134 nor M263 used this family as a group-level component. M95 group 1 is identical to M263 on Q2, improves Q3 with `M72:T0.6_G0.5` selected on Q2, and improves Q4 with `M68:T0.6_G0.5` selected on pooled Q2+Q3. The Q4 policy therefore has a genuine untouched-quarter transfer result.
- M264 is frozen before aggregate construction: use byte-pinned M263 everywhere, replace only group 1 on Q3/Q4 with byte-pinned M95, and leave Q2 byte-identical. Promotion requires positive Q3, Q4, and pooled Total deltas versus M263 plus Q4 paired-issuance bootstrap positivity above `0.50`. Components are diagnostics. No blend mass, family/policy choice, shift, snap, temporal transform, group/quarter exception, or post-result rescue is permitted.
- If and only if the OOF gate passes, deployment freezes the already pre-Q4-selected group-1 `M68:T0.6_G0.5` policy: all 643 wind-geometry plus 14 cross-fitted site-wind features, 2.5% classes, 60 LightGBM trees, temperature `0.6`, settlement gamma `0.5`, no M107 temporal shift, and full supplied 2022-2024 labels for final fit. Groups 2/3 remain exact M263. No 2024 score or selection is allowed.

### 2026-08-03 M264 result and M265 predeclaration

- The preliminary group table that motivated M264 was invalid because it duplicated incorrect capacity constants instead of importing the canonical evaluator. It is discarded evidence. M264 itself used the canonical official scorer and exact byte-pinned parents, so its negative result is valid.
- M264 kept Q2 exact but changed group 1 on Q3/Q4. Q3/Q4/pooled Total deltas were `-0.0036856140/-0.0002620583/-0.0009582366`; pooled `1-NMAE` and FICR also fell `-0.0002596006/-0.0016568727`. Q4 paired-issuance bootstrap positivity was `0.467`. The lane is rejected without a final fit or test CSV.
- Exact M264 rerun reproduced prediction SHA-256 `bca791a865086a7cb6e865a3a3f0d2896e3b752e3e78711e1fa2fa8b119fdac7` and receipt SHA-256 `4532a87166cce281d61c84f606437b3971098aa723676c52f44fcc10991228c0`.
- A fresh inventory using only the imported official scorer finds one archived full-OOF lineage with a stable direct advantage versus M263: M114 group 3. M114 selected 140-tree M113 DART policy `T0.6_G0.5` and 0.60 M107 parent mass on Q2 only, then froze that exact policy for Q3/Q4. Its group-3 Total deltas versus M263 are `+0.008669` on Q3 and `+0.001339` on Q4; Q2 is exact parent.
- M265 is frozen before aggregate construction: exact M263 everywhere except group 3 on Q3/Q4, where exact M114 is used. Promotion requires positive Q3/Q4/pooled Total and Q4 paired-issuance bootstrap positivity above `0.50`; Q2 must remain byte-equivalent. No family/policy/iteration/parent-weight search, blending beyond M114's frozen 0.60 parent mass, temporal transform, calibration, group/quarter exception, or post-result rescue is allowed.

### 2026-08-03 M265 result and M266 deployment contract

- M265 passes every frozen gate. Q2 is exactly M263. Q3/Q4/pooled Total deltas are `+0.0028897165/+0.0004462557/+0.0008103992`; pooled `1-NMAE` rises `+0.0001214848` and FICR rises `+0.0014993136`, reaching pooled `0.629050 / 0.856144 / 0.401956`.
- Q4 paired-issuance bootstrap positivity is `0.7525` with mean delta `+0.0004389`. Exact rerun reproduced prediction SHA-256 `123c0a4f8a4d42fa2e2bb164016e4d6f13c7bc11948700dbd2166c579c616b51` and receipt SHA-256 `fd936854d340ba41462cfd1562cc7c0fd2da54af5795326bc01b673e81ffbd75`.
- M266 deployment is frozen before implementation: groups 1/2 are parsed exactly from byte-pinned M263; group 3 is `0.60*M263 + 0.40*DART`. The DART branch uses the exact M113 Q2 contract: M102's identical 100 feature names, 2% normalized classes, all eligible supplied rows, generation-power-one weights, 140 trees, LightGBM DART (`drop_rate=0.05`, `skip_drop=0.5`, seed `20260802`), and `T0.6_G0.5` expected-utility actions. No temporal smoothing is applied to the raw DART action before the fixed blend.
- The final fit may use all supplied 2022-2024 labels and the same cross-fitted/full-history site-wind teachers and label-free LDAPS interpolation already verified for M261. It must not compute a 2024 metric, slice, comparison, or selection; online score remains null and DACON upload remains user-only.

### 2026-08-03 M266 full-history deployment result

- The first build stopped before DART fitting because the new wrapper incorrectly rejected the 113,984 non-finite training feature cells that the exact M113 runner and M261 deployment intentionally pass to LightGBM's native missing-value handling. Removing only that extra guard restored the frozen training contract; the test matrix has zero non-finite cells. No feature imputation, model parameter, action policy, blend weight, or selection rule changed.
- Both complete builds reproduced the legacy/all-weather site-wind arrays, DART probability SHA-256 `3487b024e266a1269629d70691cbbc4c462fd0c87e8e394089b0e7b712786d49`, DART action SHA-256 `a08aaf4bdf7dc806559dc25ea58d7d51970c0b5e77274686ab3caee1fd2e0752`, CSV SHA-256 `10955b934b035e273317f8a25f98652775ca73bad3854ab683869fe862a84cb3`, and receipt SHA-256 `f70ca20ed5c4ea534ecdb83f53fa3d9b1c436ad8808bc3c845738df8b754c34d`.
- `submission_M266.csv` has 8,760 exact sample keys with UTF-8 BOM. Groups 1/2 are exactly equal to parsed M263; group 3 satisfies `0.60*M263 + 0.40*DART` with maximum CSV round-trip error `3.64e-12 kWh`. Submission validation and 16 targeted tests passed.
- The receipt records zero score calls, no 2024 metric/slice/comparison, no post-fit search or selection, and no upload/account/external action. M266 is therefore the highest-priority unverified score-feedback candidate, not evidence that DACON Total exceeds `0.66000`.

### 2026-08-03 M267 predeclaration: frozen cross-lineage sequence transfer

- A Q2/Q3-only full-OOF substitution audit found no promotable unintegrated direct lineage: the apparent source-rank group-3 gain is excluded by its already documented same-fold/strict-prequential failure, and the apparent M257 group-1 gain is excluded by the frozen no-group-rescue decision after temporal reconciliation failed.
- Q3-only local candidates contain large same-fold selected scores and cannot be treated as independent promotion evidence. The one reusable operation with an actual later-fold transfer is M231: recipes selected on the older v2 Q3 surface improved v2 Q4 from `0.607725` to `0.610272`, raised both `1-NMAE` and FICR, and had Q4 paired-bootstrap positivity `0.913`.
- M267 is frozen before reading its M265 fold results. Apply exactly group 1 `mean5` at mass `0.475`, group 2 `median5` at mass `0.50`, and group 3 `median5` at mass `0.325` within each complete 24-hour issuance/group block; clip only to official `[0, capacity]` bounds and retain every incomplete block exactly. Parent is byte-pinned M265.
- Promotion requires strictly positive Total deltas on Q2, Q3, Q4, and pooled OOF plus Q4 paired-issuance bootstrap positive fraction above `0.50`. Component metrics are mandatory diagnostics, not separate gates. No recipe, window, mass, clipping, group, fold, boundary, or post-result rescue may be selected after this result. M268 test construction is forbidden unless all gates pass.

### 2026-08-03 M267 result and M268 deployment contract

- M267 passed every frozen gate. Q2/Q3/Q4 Total deltas over exact M265 are `+0.0005216538/+0.0005106203/+0.0008319778`; pooled Total rises from `0.6290500323` to `0.6297291534` (`+0.0006791212`).
- Pooled `1-NMAE` rises from `0.8561441701` to `0.8575657624` (`+0.0014215924`), while pooled FICR changes from `0.4019558944` to `0.4018925444` (`-0.0000633500`). Q4 improves both components and its paired-issuance bootstrap positive fraction is `0.782` over 2,000 fixed-seed replicates.
- Q3 contained two incomplete blocks and 43 rows per group; all such rows remained exact parent as predeclared. Q2 and Q4 consisted entirely of complete blocks. Exact rerun reproduced OOF SHA-256 `1d1ba64dd2e3544ecbf87239612f380b8558f2362f91c8b808dec299959173bb` and receipt SHA-256 `452512369655b95f1204b8f215774ef5cf75cc63ed685c0fe4d79cca92b738f5`.
- M268 is frozen before implementation: byte-pinned M266 parent, official cached test issuance mapping, exactly 365 complete 24-hour blocks, identical M267 recipes/capacities, and no model fit or score. It must validate and reproduce twice; online score remains user-only.

### 2026-08-03 M268 full-history deployment result

- M268 resolved exactly 26,280 official test group rows into 8,760 forecast keys and 365 complete 24-hour issuance blocks. It changed 7,339/4,440/3,805 serialized rows in groups 1/2/3 under the exact promoted M267 recipes; no label, fit, score, or selection was used.
- Two complete builds reproduced CSV SHA-256 `696324f316891d03cd89b61d5c64812faf14ba37e62124c0241ac13ad0e21b10`, receipt SHA-256 `ae4a9d0c4a9986acb3ae7d41773a802a6e9f49b14ecbc3ca64e8a9ae87277265`, and builder SHA-256 `d32284b7f09352987db211c437868ac4663a4b265f5ee7ff8e9ec66e2e6893b1`.
- Submission validation passed for UTF-8 BOM, exact key/order/schema, 8,760 rows, finite nonnegative capacity-bounded values, and stable policy/source hashes. Sixteen targeted submission tests passed; artifacts total 799,964 KiB, below the 6 GiB cap.
- Receipt fields keep local/online scores null, score calls zero, model fit false, 2024 metric/slice/comparison false, and external actions empty. M268 is the highest-priority unverified candidate, not evidence of DACON `Total > 0.66000`.

### 2026-08-03 post-M268 covariate-shift audit closure

- M163 is the exact existing implementation: calendar fields are excluded, a groupwise binary NWP domain classifier estimates history-to-target density ratios, ratios are clipped to `[0.2,5.0]`, square-root tempered, and used as label-free sample weights for the adapted multiclass model.
- On unseen Q3, M163's raw best policy scored Total/`1-NMAE`/FICR `0.601051/0.858371/0.343731`, below M265 Total `0.602278` on the same fold. Its `0.608282` blended result selects group policies and parent masses on the scored fold and is not transfer evidence.
- The prior novelty audits already close generic covariate-shift, CORAL, and MMD-style transductive variants. There is no independent next-fold positive premise for another cosmetic implementation, so the M269 lane closes before code, scoring, or test construction.


## 2026-08-08 recursive workflow intake and bounded local audit

- The new request changes the workflow architecture, not merely a candidate model. Intake state is `SKELETON_APPROVAL_PENDING(SK@v4)`; no new web/deep research or implementation is authorized until exact approval.
- `reports/m271_framework_gate_receipt.json` directly supports that the existing foundation has deterministic graph routing, dynamic fanout, checkpoint revival, and bounded parallel semantics.
- `reports/m271_close_cycles_receipt.json` directly supports a material ledger failure: 88 cycles were unrecorded and the stall counter remained stale at 97 before later contract work.
- `research/nodes/S12-N21_workflow_audit.json` directly supports uneven stage descent: dedicated deep-research lanes existed only for S3/S5/S6/S9; S11 remained open.
- `research/nodes/S16-N11_delivery.json` places the current strict local champion at `0.6361842493883538`, leaving `0.0238157506116462` to the requested strict `0.66` local exit threshold.
- `research/nodes/S16-N2_verify_oracle.json` shows an oracle member-selection Total of `0.7233330924`, but S16-N3/N4/N6/N7/N8/N9 learned or rule-based selectors failed to exceed the `0.6361842494` champion. The reachable oracle and deployable selector must remain separate evidence nodes.
- Created `SK@v4` at SHA-256 `474dfdd3912ab255b26b6988442d8107a0c1c0c9c2e7d3db8ceb43ad1ebb4530`. Its bounded research manifest covers local workflow audit, evaluation, data/features, models/decisions, loop engineering, and external-input eligibility.
- The premise that “SOTA modules guarantee 0.66 or first place” remains provisional: exact end-to-end transfer must be demonstrated under this competition's chronology, availability, action-valued metric, adaptive search history, and module interactions.


## 2026-08-08 routing correction and bounded lane dispatch

- The initial request fits two explicit preprocessing bypass classes: bounded inspection/diagnosis of an existing workflow and execution of an already approved active plan. Creating `SK@v4` as a prerequisite was an over-routing error; it is retained unapproved and withdrawn, not silently approved.
- Six independent research lanes now cover local workflow semantics, adaptive validation, data/features, model/decision, loop engineering, and eligible external inputs. Their outputs remain raw evidence until root source validation and synthesis.


## 2026-08-08 root frontier audit while lanes run

- `research/nodes/registry.json` reports current strict fold-outside champion Total `0.636184`, 1-NMAE `0.861866`, FICR `0.410503`; at fixed NMAE, `0.66` requires FICR `0.458134` (`+0.047631`).
- The registry carries a stale `lockbox_2024: untouched` field even though current `AGENTS.md` records two uses. This is a governance contradiction; registry metadata cannot be trusted as the lockbox guard until reconciled against canonical receipts. No lockbox artifact was opened in this audit.
- S16 concludes the remaining gap is an “information gap,” but S11 explicitly overturns the only kNN conditional-MAD information-floor argument as radius-dominated and marks it inconclusive. Thus “no information in existing NWP” is not yet directly supported as a global impossibility claim; only the tested representations/channels are closed.
- S14 Model Confidence Set placed the champion and several alternatives in the same 10% MCS; S15 later measured the champion as a +1.90-SD seed draw and estimated an honest expected Total near `0.634573`. The visible `0.636184` should remain a candidate observation, not an expectation guarantee.
- S15's composed point pipeline raises point accuracy (`1-NMAE ~0.8671`) but yields low FICR; transferring it into the D/DEPAVG decision architecture repeatedly fails. The binding open problem is not generic point accuracy or generic diversity, but learning a sharp conditional density/action from information that improves point location without destroying settlement-band mass.
- The full saved 15-member pool and its 1-/2-DOF blends have already been exhaustively rechecked; current best remains DEPAVG+D. Further saved-pool recombination is duplicate work.

- The durable pipeline tests pass, but the staged recursive loop package is not lint-clean: 19 Ruff findings across `__init__.py`, `graph.py`, `graph_staged.py`, `ontology.py`, `registry.py`, `research.py`, `router.py`, and `stages.py`. This directly supports that workflow machinery was added without completing the same quality gate as the forecasting pipeline.
- No test file name under `tests/` contains `loop`; therefore `156 passed` does not establish the staged recursive engine contract. A dedicated contract suite is required before calling that workflow operational.


## 2026-08-08 root source audit of `src/baram/loop`

The staged loop package is currently a prototype, not an operational implementation of the requested S1-S11 workflow:

1. **Evidence contract violation:** `research.ACTIONABLE_TAGS` includes `near_match_only` and `contradicts_premise`. The project workflow permits only `directly_supported` to confirm/drive an executable decision; a contradiction must invalidate/revise, and a near match is navigation-only.
2. **Stage coverage mismatch:** `stages.py` implements only D1 preprocessing, D2 features, D3 modelling, D4 validation, D5 improvement. It omits explicit purpose, data characteristics, precision analysis, metric understanding, problem-solving workflow, and appropriateness audit, and therefore cannot enforce S1-S11 descent.
3. **Blocked-precondition inversion:** `router.route()` says an unmet precondition outranks the candidate but returns `dispatch`; `select_spec()` then returns the same blocked spec. The prerequisite is not scheduled.
4. **No executable research handoff in the base graph:** `graph.py::research()` only increments a counter. With no candidates the router repeatedly chooses research until budget exhaustion; no request is emitted and no halt occurs.
5. **False stage completion is possible:** an empty emitted set satisfies `len(retired)==len(emitted)`; a `MISS_NO_RUNNER` candidate is retired as `NO_SCORE`, so a stage can close without an experiment or supported unknown record.
6. **Wrong termination oracle:** `graph_staged.py` delivers when all five prototype stages are marked done, regardless of whether strict local Total reaches `0.66`. It has no target-score gate.
7. **Evaluation is not project-valid:** acceptance is only `score > best + 0.001013`; no strict fold-outside assertion, row-key/policy lineage, seed floor, block uncertainty, multiplicity, group/component guardrails, or official-score receipt is enforced.
8. **No re-entry/contradiction transition:** stage `done` flags are never invalidated when a closure premise changes or a contradiction arrives.
9. **Registry durability gaps:** writes are non-atomic and unsorted, generic `register()` can overwrite prior specs, and no schema/version migration or lockbox-state assertion exists.
10. **Declared recursion limit is unused** and no dedicated tests import any loop module.

These defects explain why the prose/JSON research lineage can be extensive while the durable engine itself does not guarantee the claimed workflow. They should be converted into contract tests before any behavioural source fix.


## S17-N1 durable loop-contract repair

- The repaired package now enforces direct-support-only execution, explicit S1-S11 order, blocked-precondition refusal, explicit research/execution handoff, contradiction revision, target-based success, score-evidence receipts, deterministic re-entry, and atomic deterministic registry saves.
- Functional tests verify that missing runners do not fabricate failures, contradictions emit no candidate, and completed stages below `0.66` halt as target-not-reached rather than success.
- Narrow PASS: 26 dedicated loop tests; full suite 182 passed; Ruff clean. This is workflow-contract evidence only and creates no model score.
- The local/loop audits show the next binding workflow node is an append-only typed CycleEnvelope/event ledger. The current repair does not yet reconcile the 88 historical cycles or make graph/coverage/comparison/halt projections derive from one event stream.


## RWA-EVAL — strict retrospective evaluation audit

- Report `research/rwa_eval_research.md` SHA-256 `2b418c28ecd4f9e3a27e30176d9d76298961b37cffc613447b7c36e5eb34b84c`; 10 primary sources and explicit evidence tags.
- Direct local contradiction: current other-two-fold (`~sel`) policy/blend selection is not strict chronology, hourly midnight normalization splits one 01:00–00:00 issuance, the fixed chunk sampler is not a moving/stationary bootstrap, `p_better` is not a posterior, and comparison counts commonly reset.
- No reuse of 2023 can make it a fresh holdout. The strongest future label is retrospective, chronology-repaired, multiplicity-aware Q3–Q4 support.
- Binding repair before any new promotion claim: one 72-cell issuance-day atom, past-only nested/frozen actions, exact daily Total losses, one complete comparison ledger, and joint dependent resampling. SPA/MCS guarantees remain near-match unless stationary/mixing and frozen-family assumptions are justified.

## RWA-EXT — eligible external information audit

- Report `research/rwa_external_eligibility.md` SHA-256 `958ea7a0b96371b6d773920038ec8d6e07e7903b4b26f5dd8443dd6d354bf4e5`; 12 official source packages and metadata-only collection.
- New high-VOI candidate: ECMWF deterministic IFS `D-2 18Z` short cycle, fixed +22…+45 h leads, common 2023 fields `10u/10v/u925/v925`, CC-BY-4.0, and historical public-mirror completion comfortably before 05:00 UTC. Archive starts 2023-01-18; only six Q2–Q4 dates are absent.
- Root must independently reproduce the archive/index/Last-Modified/licence receipt before acquisition. The proposed first collection is 2023 Q2–Q4 only (~6 GB byte ranges), not 2024/2025.
- GEFS remains eligible but its already-run spread VOI probe was negative; ECMWF ENS is much more costly. FourCastNet v1 and Aurora pre-cutoff weights are eligible only for local inference from issued GFS forecast states, not analysis/reanalysis inputs.


## RWA-MODEL — decision-learning audit

- Report `research/rwa_model_decision_research.md` SHA-256 `0626868e9fbb6818021be659d719f7a95d53f829bb48f310aaac917de9734c7a`; 14 primary sources plus targeted S15/S16 local artifacts.
- Outcome-aware closest-member oracle `0.723333` is reachability, not learnability. FFORMA, hit gates, pool expansion, IDR, shifts, band-feature selection, and equal-mass cross-group transfer already failed or tied; generic stacking is not a supported new operation.
- Hit-event indicator correlation/Q/double-fault, at both 6% and 8%, is the correct diversity diagnostic; it is not itself a basis-time selector.
- The sole non-duplicate saved-action proposal is `COST5-SPO+`, a fixed five-action full-cost regret learner. It is lower priority than a genuinely new external forecast source, but is a valid one-shot representation test.
- Root correction: the memo's proposed existing 7-day arbiter and +0.001 gate cannot be used after RWA-EVAL. Any execution must use issuance-day strict past-only selection, complete comparison indexing, repaired dependent resampling, and the frozen local margin 0.001635; label remains retrospective.


## S17-N2 authoritative cycle event ledger

- Added a canonical-hashed, previous-hash-chained SQLite event store with `BEGIN IMMEDIATE`, `synchronous=FULL`, exactly-once event IDs, conflict rejection, restart replay, tamper detection, receipt-hash checks, and score-comparison monotonicity.
- Dispatch now requires a matching immutable `PREDECLARED` event. Valid result closure commits before registry retirement, and stage/halt decisions consume event-derived projections. The older LangGraph `ledger` list is telemetry only.
- Exact legacy manifest `reports/s17_legacy_cycle_manifest.json` (sha `c6c3f4b5bba9ef4742d4b2225cb4987b36e4bc774b398330e59084e46893ba74`) computes the 88 gap as `CYCLES - direct m271_ledger_history node IDs`. Those cycles are `LEGACY_UNRECONSTRUCTABLE`, not evidence-complete and not score-bearing.
- Live event store now has 88 legacy events plus a closed S17-N2 event; S11 alone is covered and frontier `S17-N3_STRICT_PREQUENTIAL_EVALUATION_REPAIR` remains. Snapshot digest `3b289845723ab8099832bd18530242c6142f73bc424e781139ac4edf4cc85093`.
- Narrow PASS only: 36 loop tests, full suite 194 passed, Ruff clean. No model score was created.


## RWA-DATA — issuance-safe feature audit

- Report `research/rwa_data_feature_research.md` SHA-256 `16351bb387dfebf0dc7ccdc6ef0e8ec77406964bf3d0b40c52fcdc3496284504`; 12 primary/official sources and 22 tagged claims.
- No literature effect can be translated to the `0.0238157506` Total gap. Mandatory contract: an issuance-keyed cube retaining source, run/reference, actual availability, valid time, lead, vertical level, grid and interval support; missingness is source/run state, not a generic imputation problem.
- Curtailment restoration, label shifts, circular/physical/grid summary/Jensen/trajectory operations remain closed.
- Conditional survivors are (R1) issue/valid/interval audit, (R2) fold-local vertical profile/PCA, (R3) height-only terrain correction with roughness control, and only-if-unadjudicated (R4) source-specific preprocessing versus concat.
- Synthesis with RWA-EXT: ECMWF `D-2 18Z` common `10u/v + u/v925` naturally couples the mandatory R1 cube to the non-duplicate R2 vertical-profile representation. This is a stronger new-information node than reopening feature-only transforms on supplied NWP.


## S17-N3 strict retrospective comparison protocol

- Added `baram.evaluation.prequential`: 72-cell operating-day validation using `(valid-1h).normalize()`, optional feature/prediction availability gates, strict fold selection, exact daily loss, joint fold-stratified stationary bootstrap, centered max-t simultaneous lower bounds, exact-Total resampling, and 90% `T_R` MCS.
- Critical chronology correction: the immediately preceding operating day is **not** fully labelled at the next D-1 14:00 basis. Training now requires `label_available_time = operating_day + 1 day < test_basis`, creating the necessary one-day boundary embargo.
- Every evaluated model/fold must carry immutable action provenance (fit/selection cutoff, policy, predeclaration hash, prediction hash, and past-only weight surface); the run also requires the live EventStore comparison count and a frozen-family manifest hash.
- Burn-in is excluded from assessment. Blocks 7 (primary) and 3/14 (sensitivity) use one joint day draw matrix. Additive and exact-resampled signs must agree. The only allowed validation label is retrospective chronology-repaired multiplicity-aware support.
- The old `research/engine/arbiter.py` fixed-chunk entry points now raise a typed retirement error. Synthetic `T_R` results matched `arch 8.0.0` on included/excluded models.
- Narrow PASS: 11 targeted and 205 full tests; no project candidate score. Receipt `6f25b5bdff627de2a1bcebe537204d91e0be0bf5be63cc8018193c1754c0a080`.


## S17-N4 ECMWF D-2 18Z fixed-lead eligibility — root reproduced

- Official 2023 S3 keys and JSON-line indices preserve `18z/0p4-beta/scda`, fixed steps 21–45 h, and exactly one each of `10u`, `10v`, `u@925`, `v@925`; using leads 21/24…45 to interpolate target leads 22…45 never stitches a later run.
- Prespecified 2023-02-01, 2023-06-01 and 2023-10-01 45h objects all had `Last-Modified=00:58:31 UTC`, 4.0247 h before the 05:00 UTC cutoff. Current official schedule independently gives 18 UTC delivery through 00:27.
- Official open-data page states CC-BY-4.0 and commercial redistribution/use with attribution.
- Top-level 2023 archive has 342 dates. Q2–Q4 has 269/275 = 97.818182% coverage; the exact six-date gap is 2023-04-27 through 2023-05-02. Deterministic whole-source fallback is mandatory.
- Four fields x nine steps are 21,925,656 representative bytes/day, about 5.898 GB for the 269 present days. Root fetched only 437,968 metadata bytes and zero GRIB bytes.
- This is eligibility/availability evidence, not value or score. Receipt `29d4be21be4815e6ebc4b7d149d72f6095e65fac4d32b9f9fa5ddbdf3954c1fb`; frontier is a 2023-only innovation probe.


## S17-N5 ECMWF extraction falsified by full operational availability

- Correct operational mapping is operating D in 2023-04-01..12-31 → init D-2 18Z in 2023-03-30..12-29. Extraction code verified canonical inputs, exact GRIB ranges, four-nearest IDW, component interpolation, and source availability.
- The first late object stopped the run before any target/value/score use. A complete 275-date 45h HEAD audit then found **13 unusable init dates**, not six: 7 HTTP-missing dates (2023-04-26..05-02) and 6 late dates (2023-04-25, 05-03..05-05, 06-28, 11-16). Their mapped operating days are listed in `artifacts/external/ecmwf_18z_audit/head_operational_init_20230330_20231229.json`.
- Coverage is 262/275 = **95.2727%**, below the frozen >=97% gate. N5 is REFUTED; the threshold was not relaxed post hoc. Twenty-six completed day checkpoints and all smoke artifacts were moved to `artifacts/external/ecmwf_18z_2023_rejected_s17n5` with `REJECTED_DO_NOT_USE`; no final cube exists.
- Root accounted at least 715.659 MB of external responses; failed concurrent/retry bodies are conservatively marked unmetered. No raw GRIB message remains, and no power target/model fit/official score/2024/test/Dacon action occurred.
- This corrects N4's spot/listing-only inference: archive date presence and three punctual spots were insufficient to prove operational coverage. Future forecast archives require a **full per-cycle Last-Modified audit before bulk acquisition**.
- Receipt `9f82caf9cc85dce3aad3703b844bd62a4b24f2f89d3139b91946f77303e71674`. The next surviving non-duplicate axis is the prequential COST5-SPO+ decision learner.

## 2026-08-08 — S17-N6 saved-action chronology adjudication

- Existing `S7-N8_D` fitting uses valid-time cutoff `idx<a`, not label-availability cutoff. For Q3/Q4 its max implied label time is respectively `2023-07-01 00:00` / `2023-10-01 00:00`, versus first basis `2023-06-30 14:00` / `2023-09-30 14:00`; the 10-hour violation is structural.
- M102/M113/M115 receipts provide neither `fit_max_time`, `selection_max_time`, nor original predeclaration proof. M102 additionally used selected iterations `60/40/60`, so the declared same saved action was not a Q2-frozen procedure.
- Common action keys retain 90 complete Q3 days plus 92 complete Q4 days; missingness itself is repairable by full-day exclusion, but it cannot repair fit/selection chronology.
- A mechanically shifted outer origin (drop the first operating day) would place the basis at 14:00 after the batch fit's 00:00 max label time. Together with saved `M102_TOP100_I60` on Q3 and Q2-frozen iterations/policies, this is a distinct follow-up reconstruction, not a relaxation of N6. It must be separately predeclared before reading candidate scores.

## 2026-08-08T22:18:30.058234+09:00 — S17-N7..N17 authoritative checkpoint

- The current unchanged strict champion remains **Total 0.6314827308346854** on the repeatedly exposed Q3+Q4 retrospective surface. Comparison count is 3; none promoted.
- COST5-SPO+ was strictly reconstructed and refuted (`-0.0118992864` vs CHAMPION). R2 vertical PCA lost to both CHAMPION (`-0.0681023053`) and its zero-column control (`-0.0012093920`). Source separation lost to CHAMPION and concat D under both equal and Q2-only weights; every 3/7/14-day MCS retained CHAMPION only.
- Issuance remapping did not reach its frozen `0.01` Fisher-z gate. The exact published KMAPP height correction cannot be instantiated: the printed formula lacks an identifiable inverse-length factor and `u(h_HC)` is absent from the supplied 5/10 m mean-vector fields. No DEM body was downloaded.
- Copernicus GLO-30 N37/E128 metadata/licence/static-elevation prerequisites passed, so a different explicitly sourced direction-dependent static-terrain representation remains researchable; it must not silently reuse the unidentified KMAPP equation.
- Source-separated DL/DG artifacts were provenance-reconstructible and safe for N17, but their negative result gives zero gap-closing credit. They remain outage accounting only.
- N18 is a no-fit/no-score evidence re-entry. A candidate may advance only if primary/official support, exact local applicability, strict chronology, commercial licence, deployment symmetry and nonduplication are all proven; at most one next prerequisite audit can be emitted.


## S17-N22 terrain result (2026-08-08T23:50:11.570475+09:00)

- The bounded prefix loader exposed exactly 52,560 chronological 2022–2023 surface rows and 52,560 values from each 78,912-element M64B cache member; the 26,352-element 2024 tail was not exposed. Terrain vectors were finite/nonzero for all 840,960 grid-row pairs.
- `terrain__sx300_h8_mean16` ranges `-2.7752635479..15.8036489487°` on the development surface (mean `4.7097277641°`).
- The unchanged 100-feature M115 refit reproduced N7 M115 actions exactly on Q2/Q3/Q4 (`max_abs_kwh=0`), resolving the N21 historical-receipt ambiguity. The refit-zero Champion also reproduced exactly.
- The single appended terrain feature changed the strict Total only `+0.00036331040239234724` to `0.6318460412370778`; `1-NMAE +0.0000912406891292`, FICR `+0.0006353801156555`. It was negative on Q3 and positive on Q4, remained in the MCS with Champion, but failed every preregistered promotion gate and closes with zero gap-closing credit.
- Duplicate zero controls must be proven value/hash-identical and excluded from `mcs_tr`; S17-N3 intentionally fail-closes rather than silently accepting identical loss columns. This is evaluator hygiene, not a candidate choice.
- Terrain Sx is now a weak positive retrospective signal, not a promotable family. Any follow-on must add a genuinely different representation/information source; retuning the same mean16 feature is prohibited.

## S17-N23 frontier findings (2026-08-09T00:14:01.058165+09:00)

- **No zero-tuning executable candidate survived the complete directly-supported chain.** Candidate count is zero; this is a diagnostic/INCONCLUSIVE result and consumes no comparison.
- Winstral-style exposure belongs at the target cell, so N22's all-16 mean is not a site exposure. That does not authorize a rescue: turbine-local values are absent from the frozen lookup, `h=8 m` is only near-match for 117 m hubs, and no source fixes one multi-turbine group reducer.
- WorldCover 2021 v200 passes pre-cutoff/static/CC-BY-4.0 distribution gates. A static `z0` log-law is already represented and closest Korea/LDAPS evidence is unfavorable; the distinct directional roughness treatment depends on missing `z0/d` tables, unfrozen boundary-layer choices, and proprietary/non-macOS PyWAsP.
- Exact layout metadata is not the power/wake blocker. The missing evidence is configuration-matched V126-3.6 and U136-4.2 power plus `C_T` definitions with commercial reuse/chronology, together with one coefficient-complete farm assembly. Generic NREL/FLORIS/PyWake reference turbines would substitute identities and reopen a closed variant axis.
- The only repository-native literal gap found is a past-only circular SCADA-direction teacher, but target aggregation, loss, normalization, missingness and downstream treatment are all unfrozen; existing M168/M178 already cover issued-NWP direction/wake and local wind-speed teacher mechanisms.
- Operational NWP audit found no ready archive. GEFS is non-novel under prior frozen negative VOI evidence. CFSv2 is the strongest novel near match because official NCEI/NCO pages distinguish operational forecasts and preserve init/member/wind inventory, but missing latency/continuity/access/decode/formula proof prevents even a prerequisite from being selected under N23's exact-treatment rule.
- Scope hygiene is substantive evidence: narrative aggregate exposure invalidates a delegated lane even without raw labels. Fail-closed containment (discard conclusions, independent root audit, zero candidate/score) preserves the ledger but must remain disclosed in the receipt.

## S17-N24 secondary-frontier findings (2026-08-09T00:35:29.212906+09:00)

- Exact manifest arithmetic closes the “unused supplied payload” hypothesis: 35 GFS and 30 LDAPS documented value variables each appear with all `mean/std/min/max/q10/q50/q90` features. The only literal omissions are metadata identities/strings, whose group-level reductions are functions of existing group/fleet identity or require arbitrary encodings.
- Published whole-boundary-layer wind profiles are not computable from the supplied state. Target height and low-level winds exist, but surface friction velocity, aerodynamic roughness and Obukhov length/virtual-temperature flux do not; mixing GFS wind with LDAPS BLH would not be a documented same-column treatment.
- A fully written CFSv2 feature name/formula is still only an analyst freeze. NOAA docs do not prescribe its D-2 cycle, four-member mean, bilinear spatial map or linear interpolation, and exact availability/continuity/schema/access remain unverified. Frozen choices remove tuning degrees but do not create direct provider support; existing external-NWP closure independently blocks value credit.
- Prospective electricity-system data exist, but the useful plant-specific day-ahead plan is notified at 17:00 KST or later and is non-public. Earlier weekly/monthly outage plans lack an electrical incidence matrix and impact formula for the three groups. A maintenance list cannot be converted into curtailment without inventing topology/weights.
- Open terrain-flow code availability is not enough. The reviewed engines require absent compiled/geospatial/ML stacks and expose material domain, resolution, direction-category, initialization, vegetation, stability, height and aggregation choices. Installing them would not cure formula or target-group identifiability.
- N24 emits zero candidates and zero prerequisites. The cumulative frontier now lacks a directly supported representation with plausible target-scale leverage; any tertiary intake must be a genuinely different information class, not another operational-NWP, terrain-flow, metadata-encoding or physical-proxy variant.


## S17 terminal evidence-exhausted closure — 2026-08-09

- **N25 terminal intake:** no item completed the conjunction of fixed-issue 2022–2023 history, D-1 14:00 chronology, anonymous commercial deployment, exact three-group mapping, source-prescribed treatment, runtime, and nonduplication. Near matches are navigation evidence only.
- **Farm forecast:** KPX rules directly support submissions at D-1 10:00 and 17:00, but not a public immutable archive. Only the earlier issue is basis-safe, and member/account access plus missing group mapping is fatal.
- **Regional forecast:** direct wind-generation services do not document preserved issued runs or an exact G1/G2/G3 map; clearly reusable regional data are ex-post, while exact 14:00 generation products found are solar.
- **Grid outages:** official plant/phase identities do not expose group-to-substation/line/bay topology. Planned outages are conditional and mutable, so no coefficient-free hourly derate follows from the rules.
- **Microscale/icing:** FuXi-CFD is closest technically but CC BY-NC, lacks a literal 117 m/preprocessing contract, and has no local ONNX runtime. Published icing chains require missing liquid-water/drop inputs and fitted/site-specific or ex-post operating information.
- **S4 repair:** the official scorer uses the inclusive 10% actual filter, capacities 21,600/21,600/21,000 kWh, inclusive 6%/8% tiers, actual-weighted within-group FICR, equal-group means, and 0.5/0.5 Total. The strict prequential evaluator recomputes the exact denominators on shared day draws.
- **Formal conclusion:** EventStore, not narrative, now proves all stages covered and no pending/frontier node. `HALT_ALLOWED_EXHAUSTED` is an unsuccessful evidence halt; it does not satisfy the strict `0.66` objective and is not a universal impossibility theorem.

## S17-N27/N28 scalar-gradient findings — 2026-08-09

- **Exact formula is no longer the blocker.** Couto & Estanqueiro (2022) directly fixes first-order centred `MSLPGrad` on a projected 12 km WRF grid, with a 00 UTC 48-hour run and last-24-hour market evaluation.
- **The reported effect is not the same treatment.** `MSLPGrad` appears among top-ranked PCs at several parks, but the 13–37% gain is for 11–20 site-specific supervised PCA/SFFS/ANN features. The paper reports worse transfer of one park's selected set to other parks and no `MSLPGrad`-only ablation.
- **The ramp algorithm is exact but unevaluated.** TradeRES D4.9 fixes spatial gradient, gradient-norm tendency, 2/98 percentiles, 150,000 km² area, 720 km tracking, and 2 h/120 km h⁻¹ filters, then explicitly says forecast accuracy will be assessed in WP5. Its binary alert cannot be converted into a BARAM action without a new mapping.
- **Spatial PCA is a near match, not new information.** The paper's benefit uses a much larger WRF domain against a nearest-point wind-vector baseline. Applying PCA to the supplied local GFS/LDAPS grids would be a new compression choice over already exposed raw grids, with no demonstrated official-Total effect.
- **Conjunctive gate result:** formula/mechanism, arithmetic runtime and scalar-gradient core novelty pass; exact supplied-source/domain/output mapping and isolated target-scale action effect fail. Candidate count remains zero.
- Sequence 158 restores the formal evidence halt with no comparison consumed. The conclusion is bounded evidence exhaustion, not a proof that `0.66` is mathematically impossible.

## SK@v5 bounded workflow-audit findings — 2026-08-09

- **Stage taxonomy PASS, recursive discovery FAIL.** The code has the user's eleven semantic stages, but reports stage position as depth and selects only the first incomplete linear stage.
- **Evidence and hypothesis are incorrectly conflated.** Only `directly_supported` findings emit nodes. This protects claims but prevents a separately labelled, bounded migration hypothesis from being locally tested.
- **Shallow closure is possible.** A batch of only `near_match_only`/`insufficient` findings becomes `explicit_unknown`, which can satisfy stage completion with zero experiment.
- **Experimental feedback is not structural.** Accepted/rejected results update score and stagnation but do not generate a residual-deficit child or causal upstream re-entry.
- **Ontology breadth is not depth.** The 55-subcapability ontology records all six preprocessing areas as research `없음` and all six feature areas as `취소(L3)`, while EventStore binary S1–S11 coverage still permits exhaustion.
- **Repair principle:** retain exact evidence tags, predeclaration, chronology and append-only events; add a distinct provisional migration-hypothesis layer, recursive child depth, bounded sibling families, local experimental adjudication, residual children, result-driven re-entry, and pause-vs-success semantics.
- Proposed authority is `SK@v5` SHA `c2c813475ab5f0a741e6adbceab196d5b38447f2aeb02f9716a8f5be8973c820`; no research or implementation begins before exact approval.
