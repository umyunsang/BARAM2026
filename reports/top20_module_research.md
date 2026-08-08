# BARAM 2026 Top-20 Module Research Report

## Document state

- Date: 2026-08-01 (Asia/Seoul)
- Research authority: `SK@v3`
- Design output: `.planning/2026-08-01-leaderboard-top-4-loop/DS_v3.md`
- Evidence status: bounded read-only research complete; design pending `DS@v3` approval
- Modeling boundary: competition-supplied data only
- Excluded: implementation, training, dependency installation, external datasets or forecasts, pretrained weights, remote inference/compute, Dacon upload/account mutation, remote Git, and reuse of the consumed 2024 lockbox
- Execution topology: one root session; no subagents, worktrees, or background sessions

## Executive conclusion

The current pipeline should not be replaced by an unconstrained deep-model search. The strongest evidence-supported route is a **spatially aware classical-probabilistic pipeline**:

1. Preserve the supplied LDAPS/GFS grid geometry and turbine/group geometry instead of collapsing every weather field to one global timestamp mean.
2. Train strong capacity-normalized LightGBM point and quantile models under issuance-time chronological folds.
3. Estimate conditional predictive distributions using cross-fitted quantiles and a Quantile Regression Forest benchmark; activate CatBoost only when it adds residual diversity.
4. Select each submitted point prediction by expected official composite utility, using only predictions and residual distributions produced by earlier folds.
5. Treat a Temporal Fusion Transformer as a conditional challenger, not the default path, because its known-future-covariate structure is relevant but its compute/dependency cost and incremental value are not yet established.

The saved public leaderboard implies a hard rank-20 reference of `0.65971`, a rank-15 intermediate reference of `0.66033`, and a movement-safe operating guide of total `>=0.66200`, `1-NMAE>=0.87500`, and `FICR>=0.44900`. These are external guideposts, not honest local-validation thresholds: the local folds and the hidden 2025 test period have different distributions, and only a separately authorized Dacon result could measure actual placement.

## 1. Context survey and evidence quality

### 1.1 Authoritative inputs

| Source | Role | Evidence class | Freshness / limitation |
| --- | --- | --- | --- |
| Official Dacon evaluation page | Metric, public/private split, stage structure, presentation rubric | `directly_supported` | Current page inspected 2026-08-01; image-embedded rubric was visually verified |
| Official Dacon rules page | Availability, code-reproduction, external-data/model restrictions | `directly_supported` | Current page inspected 2026-08-01 |
| Official Dacon data page and immutable `open.zip` | Forecast horizon, columns, NWP issue/availability contract, turbine metadata | `directly_supported` | Archive hash reverified; no mutation |
| Official metric code-share | Reference implementation of `1-NMAE`, FICR, and composite | `directly_supported` | Must remain the executable scoring oracle |
| User-supplied saved leaderboard HTML | Rank/component snapshot and authenticated summary | `directly_supported` for that snapshot only | Not a live/final/private leaderboard; per-submission history is absent |
| Current repository and receipts | Existing pipeline, local scores, lockbox state | `directly_supported` for local execution only | Does not prove Dacon score or rank |
| Twelve primary papers/proceedings | Method choices and known tradeoffs | `directly_supported` for methods; `near_match_only` for this exact dataset | No external data, forecasts, or weights imported |

### 1.2 Immutable source identity

| Input | SHA-256 |
| --- | --- |
| `/Users/um-yunsang/Downloads/open.zip` | `920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b` |
| `/Users/um-yunsang/Downloads/baseline.ipynb` | `712b26f4d2748860c94cff1e0100c23810468c983173f8e9ef8d009fe82df48c` |
| User-supplied leaderboard HTML | `f1328f9b68772da6a91a2cd715d649193e6f76252b47c5dafd36ece04567257c` |

## 2. Official objective and qualification context

The official online score is:

`Total = 0.5 × (1 - NMAE) + 0.5 × FICR`

Only observations whose actual output is at least 10% of facility capacity enter evaluation. The official settlement schedule awards 4 won/kWh when capacity-relative absolute error is at most 6%, 3 won/kWh when it is above 6% and at most 8%, and zero beyond 8%. This discontinuity makes conditional uncertainty near the 6% and 8% boundaries operationally important.

The public leaderboard uses a sampled 40% and the private leaderboard the remaining 60%; first-stage ranking uses the private set. The private top 30, including ten reserves, submit reproducibility materials; after code validation, the top 20 advance to the offline presentation. Final ranking combines private score and presentation evaluation at 50% each. Thus a public rank snapshot is a useful target guide, never qualification proof.

Official offline rubric:

| Evaluation item | Points | Pipeline evidence to retain |
| --- | ---: | --- |
| Task understanding | 20 | Availability contract, metric decomposition, group/fleet analysis |
| Technical excellence | 30 | Leakage-safe features, models, calibration, ablations, exact metric |
| Problem-solving | 15 | Hypothesis ledger, failed-path evidence, bounded promotion logic |
| Applicability | 20 | Settlement-aware decisions, uncertainty, reproducible daily inference |
| Presentation completeness | 15 | Clear receipts, diagrams, limitations, reproducible result tables |
| **Total** | **100** | End-to-end evidence package |

Official sources: [evaluation](https://dacon.io/competitions/official/236727/overview/evaluation), [rules](https://dacon.io/competitions/official/236727/overview/rules), [data](https://dacon.io/competitions/official/236727/data), and [metric code-share](https://dacon.io/competitions/official/236727/codeshare/14035).

## 3. Leaderboard target ladder

### 3.1 Chart contract

- Question: how do the current authenticated best, the rank-20 cutoff, the rank-15 buffer, and the operating guide compare on total, `1-NMAE`, and FICR?
- Decision: identify the binding component and avoid treating a barely clearing total as safe.
- Data grain: four discrete snapshot/reference rows; no time axis and no trend inference.
- Chart family: grouped comparison bars with exact-value companion table.
- Key takeaway: the current `1-NMAE` is close to the target envelope while FICR is the dominant gap; the rank-20 and rank-15 total buffers are too small for a movement-safe target.
- Non-color encoding: fixed row order, direct labels, and exact numbers in the table.
- Limitation: the three metrics share a 0-1 range but different operational meanings; the chart is a target comparison, not a causal or temporal view.

### 3.2 Exact snapshot values

| Reference | Rank | Total | 1-NMAE | FICR | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Current authenticated summary | 689 | 0.62366 | 0.87305 | 0.37426 | Best summary visible in supplied snapshot; two submissions |
| Hard finals cutoff | 20 | 0.65971 | 0.87991 | 0.43952 | Reject as an operating target because buffer is zero |
| Intermediate buffer | 15 | 0.66033 | 0.87740 | 0.44326 | Only `0.00062` above rank 20 |
| Recommended operating guide | — | **0.66200** | **0.87500** | **0.44900** | Balanced movement-safe guide above the visible rank-10 total |

At unchanged `1-NMAE=0.87305`, rank 20 requires `FICR=0.44637`, and total `0.66200` requires `FICR=0.45095`. Relative to the saved authenticated summary, the operating guide therefore requires total `+0.03834`; if NMAE is unchanged, FICR must rise by approximately `+0.07669`. This is why the decision layer is the primary bottleneck while deterministic error reduction remains a necessary guardrail.

## 4. Competition-data audit

### 4.1 Time and availability contract

- Training covers 2022-2024; test covers all 8,760 hourly timestamps of 2025.
- Each daily NWP batch is initialized at 09:00 KST, becomes available at 13:00, and predicts the next operating day from 01:00 through 00:00.
- LDAPS supplies 16 grids at about 1.5 km resolution; GFS supplies nine grids at about 0.25 degree resolution.
- KPX groups 1 and 2 have labels from 2022-2024; group 3 begins in 2023. Any shared model must explicitly handle group-age and label-availability differences.
- Test data has no SCADA, so target lags and SCADA-only features are prohibited even if they improve training folds.

### 4.2 Turbine and group topology

The immutable `info.xlsx` describes 17 turbines:

| KPX group | Fleet | Count | Unit capacity | Group capacity | Hub / rotor |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | VESTAS V126 | 6 | 3.6 MW | 21.6 MW | 117 m / 126 m |
| 2 | VESTAS V126 | 6 | 3.6 MW | 21.6 MW | 117 m / 126 m |
| 3 | UNISON U136 | 5 | 4.2 MW | 21.0 MW | 117 m / 136 m |

All turbine coordinates are supplied. This geometry may be used to aggregate supplied NWP grids by group and turbine distance. It does not permit adding external terrain, roughness, wake maps, power curves, or weather observations.

### 4.3 Quality findings

| Finding | Consequence |
| --- | --- |
| Prepared train rows: 78,912; group-expanded test rows: 26,280; submission rows: 8,760 | Preserve exact timestamp/group keys and `3 × 8,760 -> 8,760` submission aggregation contract |
| GFS blank cells: 0 | GFS can act as a supplied-data fallback/context source |
| LDAPS test: 752 blank cells across 48 grid rows at three timestamps (`2025-04-08 17:00`, `2025-06-18 18:00`, `2025-07-18 06:00`) | Add source/variable missing flags; imputation must be deterministic and fold-fitted |
| Group-3 label nulls: 8,766; groups 1/2: about 100 each | Never interpret missing labels as zero; use availability masks |
| Group-3 above-capacity observations: 38 | Treat strict nominal-capacity clipping as a model choice to validate, not a data truth |
| SCADA train series complete but test SCADA absent | Keep SCADA quarantined from model features |

## 5. Baseline and live-pipeline gap analysis

### 5.1 Tutorial baseline

The supplied notebook globally averages weather variables by timestamp, adds basic calendar fields, median-imputes, fits independent Random Forest regressors, clips outputs to capacity, and uses unrestricted `n_jobs=-1`. It lacks chronological validation, the exact official metric, forecast-availability guards, spatial/group physics, predictive uncertainty, FICR-aware decisions, and reproducibility receipts.

### 5.2 Current repository

The repository already provides canonicalization/data-quality checks, 24-hour issuance batches, expanding chronological folds, leakage guards, broad GFS/LDAPS aggregates, physics proxies, LightGBM/XGBoost/CatBoost adapters, exact scoring, calibration, two-parent blending, decision policies, inference/submission code, and receipts.

Current limitations that matter for DS@v3:

- Weather grids are still summarized primarily by global statistics; turbine/group distance and grid direction are unused.
- Hub-height conversion uses a fixed proxy, while spatially varying shear and fleet rotor information are not exploited.
- The chosen calibration search is coarse. Its prior 2024 lockbox choice reached positive grid edges for several groups, which is diagnostic evidence only; the lockbox cannot be reopened.
- The utility layer uses coarse median-residual state shifts rather than a calibrated predictive distribution.
- Blending accepts exactly two parents and produced only a marginal prior improvement; residual diversity is not yet a promotion requirement.
- The deep tier is deferred and has no approved dependency/GPU evidence.

### 5.3 Existing local evidence

| Artifact | Total | 1-NMAE | FICR | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 2023 Q3-Q4 pooled raw cross-fit | 0.569738 | 0.849583 | 0.289892 | Honest development reference |
| 2023 Q3-Q4 cross-fit calibration | 0.590169 | 0.853588 | 0.326750 | `+0.020431` total; decision calibration is high priority |
| Prior two-parent blend | 0.570645 | — | — | Marginal gain; broad stacking is not justified |
| Consumed 2024 calibrated-tree lockbox | 0.627605 | 0.874558 | 0.380652 | One-time evidence only; never a tuning oracle |

The local values are not directly interchangeable with the saved public score. The correct promotion signal is paired improvement on untouched chronological folds with component and group guardrails; the Dacon target ladder remains an external design guide.

## 6. Module-by-module research synthesis

| Module | Role | Competition-data inputs | Evidence-backed design | Benchmark / ablation | Promotion evidence |
| --- | --- | --- | --- | --- | --- |
| M0 Contract & reproducibility | Data engineer | Immutable archive, schema, issue/valid timestamps | Hashes, canonical keys, availability assertions, deterministic seeds/workers | Current prepare/split receipts | Exact row/schema/hash parity and no future availability |
| M1 Data quality & EDA | Data-quality engineer | Labels, supplied NWP, turbine metadata | Missingness map, capacity/season/group regimes, source coverage, target support | Existing manifest plus group/time diagnostics | No silent null-to-zero; deterministic imputation plan |
| M2 Spatial/physical features | Feature engineer | LDAPS/GFS grids, turbine coordinates, hub/rotor/fleet metadata | Distance-weighted group summaries, nearest-grid values, vector wind, grid dispersion/gradients, source disagreement, shear/density/power proxies, missing flags | Global-only vs spatial; source-only vs combined; physics on/off | Paired OOF improvement without worst-fold/group failure |
| M3 Deterministic models | ML engineer | M2 features and group-capacity-normalized target | LightGBM L1/Huber workhorse; CatBoost only as diversity challenger; RF retained as reference | RF, LightGBM, CatBoost; shared vs group-specific heads | Stable total and NMAE improvement across folds |
| M4 Distributional models | Probabilistic-ML engineer | Same features, cross-fitted residuals | LightGBM quantiles as workhorse; QRF benchmark; NGBoost conditional if distribution diagnostics justify it | Median point model, quantile grid, QRF | Pinball/coverage diagnostics plus downstream composite gain |
| M5 Ensemble | Ensemble engineer | Strictly preceding-fold OOF predictions | Constrained convex blend only across residual-diverse parents; group/lead weights if supported | Champion alone vs 2-3 diverse parents | Gain survives every later fold; complexity receipt |
| M6 FICR decision & calibration | Decision engineer | Cross-fitted conditional distribution, capacity, official scorer | Choose point on bounded action grid to maximize expected official composite; hierarchy by group/lead/season/wind/threshold proximity with shrinkage | Median, coarse scale/offset, expected-utility policy | Total/FICR lift with NMAE and group guardrails |
| M7 Temporal challenger | Temporal-ML engineer | 24-hour known-future NWP sequence and static group data | TFT only after classical residual-value and compute/dependency gate | Best classical vs TFT residual correction | Predeclared material gain; otherwise retire |
| M8 Self-evaluation | Evaluation engineer | OOF predictions, exact metric, run receipts | Fold/group/month/lead/wind/error-band slices; paired deltas; uncertainty and failure audit | Champion vs every ablation | Reproducible, conservative promotion decision |
| M9 Orchestrator | Pipeline engineer | Configs, manifests, candidates | Stage budgets, cache keys, artifact hashes, stop conditions, no-lockbox guard | Dry-run/reproduction | Same input/config produces same candidate hash |

## 7. Feature design in detail

### 7.1 Spatial aggregation from supplied geometry

For each group and supplied NWP source/variable, construct a compact fixed family:

- nearest-grid value to each turbine and group centroid;
- inverse-distance-weighted mean across grids/turbines, with deterministic distance floor;
- weighted dispersion and min/max range to retain local heterogeneity;
- vector-average wind components before deriving speed/direction, avoiding scalar-angle averaging;
- local directional gradients or upwind/downwind contrasts where the supplied grid geometry supports them;
- LDAPS-minus-GFS level and vector disagreement as a model-uncertainty/context signal;
- grid/source missing counts and imputation indicators.

This is a controlled replacement for indiscriminate global quantile expansion. Every feature must be computable identically from the test-period supplied forecasts.

### 7.2 Physical and fleet features

- Convert 80 m/100 m winds toward the supplied 117 m hub using a shear estimate computed only from supplied levels, with bounded handling of invalid ratios.
- Derive air-density proxies from supplied pressure/temperature/humidity variables where the data contract supports them.
- Form normalized kinetic-power proxies such as density × hub-wind-speed cubed, then aggregate by group/fleet.
- Include rotor diameter/swept-area and nominal capacity only as static supplied metadata; test whether capacity normalization or group-specific heads are more stable.
- Preserve cut-in/rated/cut-out-like regimes as learned wind-speed bins or monotonic interactions, not external manufacturer power curves.

### 7.3 Availability-safe time features

- valid hour, day-of-year/month, cyclic encodings, forecast lead, issue batch, and lead × diurnal/season interactions;
- group-age and training-availability indicators for group 3;
- no target lag, rolling target, or realized SCADA feature because those inputs do not exist at 2025 inference time.

## 8. Model and decision architecture

### 8.1 Deterministic workhorse

Use group-capacity-normalized targets and LightGBM with L1/robust loss as the first workhorse. Compare shared-with-group-indicators, shared trunk plus group-specific calibration, and fully separate group models under identical chronological folds. CatBoost is retained only if its OOF residual correlation and group failures show complementary information. XGBoost/RF remain controlled references rather than automatic ensemble members.

### 8.2 Predictive distribution

Fit quantile models on a predeclared grid (for example, central and tail quantiles) using exactly the same fold boundaries. Repair quantile crossing deterministically, check conditional coverage by group/lead/season/wind regime, and benchmark against QRF. NGBoost is conditional because its distribution family and calibration cost must improve decision quality, not merely likelihood.

### 8.3 Expected official-utility action

For each validation or test row, generate a bounded candidate action grid around the conditional quantiles/median. Approximate

`a*(x) = argmax_a E[0.5 × local 1-NMAE contribution + 0.5 × normalized settlement contribution | X=x]`

using only the predictive distribution learned from earlier data. Score candidate policies with the unmodified official implementation over full held-out folds. The policy hierarchy may condition on group, lead, season, wind regime, and distance to the 6%/8% error thresholds, but every extra cell must shrink toward a global policy when support is weak.

This is decision-focused forecasting, not post-hoc label access: calibration parameters and residual distributions for a fold must be learned from preceding folds only.

### 8.4 Ensemble policy

Ensemble only models that:

1. were trained on identical allowed data,
2. have complete preceding-fold OOF predictions,
3. show meaningfully different residuals or complementary group/regime strengths, and
4. improve every later evaluation fold under constrained weights.

Unbounded stacking, leaderboard-weight fitting, and same-fold calibration are excluded.

## 9. Chronology-safe validation and experiment loop

### 9.1 Validation contract

- Preserve daily 24-hour issuance batches; never split individual future hours from their forecast batch context.
- Use expanding/prequential development folds inside the already approved 2023 development horizon.
- Learn preprocessing, model parameters, quantile repair, calibration, utility policies, and ensemble weights from training or preceding OOF data only.
- Keep the consumed 2024 lockbox closed. Its prior receipt can diagnose weaknesses but cannot select features, hyperparameters, or policies.
- Evaluate exact total, `1-NMAE`, FICR, group components, month/lead/wind regimes, and settlement error bands.

### 9.2 Proposed bounded later implementation budget

This is a design budget, not current compute authority:

| Stage | Maximum full chronology candidates | Stop condition |
| --- | ---: | --- |
| Contract/parity | 2 | Exact schema/metric/current-baseline parity or block |
| Spatial/physical feature families | 8 | Retain only families with paired stable benefit |
| Deterministic model/objective | 10 | Freeze best stable workhorse and one diverse challenger |
| Quantile/distribution | 8 | Retain only calibrated distributions useful to decisions |
| Decision policies | 10 | Stop when added policy complexity no longer improves later folds |
| Ensemble | 6 | At most three residual-diverse parents |
| Final reproduction | 2 | Two identical approved runs and candidate hash parity |

All later model workers remain capped at six. Deep/TFT work is outside this budget until separately activated.

### 9.3 Promotion gates

| Gate | Required evidence |
| --- | --- |
| G0 Contract | Hash/schema/key parity, exact scorer parity, no availability violation |
| G1 Feature | Positive paired total delta on every evaluation fold or an explicitly justified conservative aggregate; no material worst-group collapse |
| G2 Deterministic | Stable total/NMAE improvement and bounded output/support checks |
| G3 Distribution | Better held-out pinball/coverage diagnostics and positive downstream official-composite effect |
| G4 Decision | FICR and total lift with predeclared NMAE floor and no unsupported small-cell policy |
| G5 Ensemble | Gain over champion on every later fold, proven residual diversity, limited parents |
| G6 Candidate | Reproduced twice from the same immutable inputs/config; 2024 remains unopened |
| External readiness | Compare expected component envelope with `0.662/0.875/0.449`, but label it a proxy until a separately authorized Dacon result exists |

A statistically or operationally material delta threshold must be fixed in `IP@v2` after estimating run-to-run variance; this report does not invent one from a single historical run.

## 10. Architecture alternatives

| Alternative | Description | Strength | Main risk | Decision |
| --- | --- | --- | --- | --- |
| A. Deterministic incremental | Add spatial features to the current LightGBM and retain coarse calibration | Lowest implementation risk | Does not model threshold uncertainty; likely leaves FICR value unused | Keep as mandatory benchmark |
| B. Spatial classical + distribution + decision | Spatial/physics features, LightGBM quantiles/QRF, cross-fitted expected-utility action, residual-diverse blend | Directly addresses both score components with manageable data/compute | Calibration leakage and sparse regime cells require strict controls | **Recommended primary architecture** |
| C. TFT-first temporal stack | 24-hour sequence model with static group/fleet context | Matches known-future-covariate structure | Higher complexity, dependency/compute cost, small effective dataset, uncertain lift | Conditional challenger only |

## 11. Primary-source research matrix

| Source | Direct method support | Use in this project | Guardrail |
| --- | --- | --- | --- |
| Landry et al., *Probabilistic Gradient Boosting Machines for GEFCom2014 wind forecasting* | Gradient boosting and probabilistic wind forecasting | Tree-based quantile/distribution workhorse | Competition data only; no GEFCom data transfer |
| Ke et al., *LightGBM* | Efficient gradient-boosted trees | Deterministic and quantile models | Chronology folds and bounded workers |
| Prokhorenkova et al., *CatBoost* | Ordered boosting and categorical handling | Group/fleet diversity challenger | Activate only on honest residual diversity |
| Meinshausen, *Quantile Regression Forests* | Conditional response distributions | Distributional benchmark | Compare calibration and downstream utility |
| Duan et al., *NGBoost* | Natural-gradient probabilistic prediction | Conditional distribution challenger | Justify distribution family and extra complexity |
| Gneiting, *Making and Evaluating Point Forecasts* | Point forecasts depend on the target functional/loss | Align final point with official composite utility | Preserve exact official scorer as oracle |
| Elmachtoub & Grigas, *Smart Predict-then-Optimize* | Decision-focused learning | Motivate expected-utility action selection | No same-fold decision fitting |
| Fang & Chiang, *Extended NWP variables for wind forecasting* | Rich NWP-variable construction | Source/level/physics feature family | Only supplied NWP variables |
| Bergmeir, Hyndman & Koo, *Cross-validation for autoregressive time series* | Conditions and cautions for time-series CV | Expanding/prequential validation | Issuance batches remain intact |
| Bates & Granger, *The Combination of Forecasts* | Forecast combination | Constrained residual-diverse blend | No unlimited stacking or public-score weighting |
| Lim et al., *Temporal Fusion Transformers* | Known-future/static temporal covariates | Conditional 24-hour temporal challenger | Separate value/compute/dependency gate |
| Romano, Patterson & Candès, *Conformalized Quantile Regression* | Distribution-free interval calibration under assumptions | Coverage diagnostic / optional calibration | Time dependence and shift assumptions must be stated |

Primary links: [Landry et al.](https://doi.org/10.1016/j.ijforecast.2016.02.002), [LightGBM](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [CatBoost](https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), [QRF](https://www.jmlr.org/papers/v7/meinshausen06a.html), [NGBoost](https://proceedings.mlr.press/v119/duan20a.html), [Gneiting](https://doi.org/10.1198/jasa.2011.r10138), [Smart Predict-then-Optimize](https://doi.org/10.1287/mnsc.2020.3922), [Fang & Chiang](https://doi.org/10.1049/iet-rpg.2016.0339), [time-series CV](https://doi.org/10.1016/j.csda.2017.11.003), [forecast combination](https://doi.org/10.1057/jors.1969.103), [TFT](https://doi.org/10.1016/j.ijforecast.2021.03.012), and [CQR](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html).

## 12. Limitations and confidence

| Claim | Confidence | Limitation |
| --- | --- | --- |
| Official score/stage/rubric interpretation | High | Official pages can be revised; preserve dated receipt |
| Saved rank-20 and component envelope | High for snapshot | Not live, final, or private ranking |
| FICR is the current visible bottleneck | High for saved summary | Only best summary, not per-submission history |
| Spatial-probabilistic-decision path is the best next architecture | Medium-high | Requires honest implementation ablations |
| Any local score maps to public `0.662` | Low / unsupported | Distribution shift and only two public summaries prevent calibration |
| TFT will improve score | Unknown | Must pass a later classical-residual and compute gate |

## 13. Decision requested

Approve `DS@v3` only if the recommended architecture, target ladder, module boundaries, validation contract, and exclusions match the intended competition strategy. Exact approval phrase: `DS@v3 승인`.

Approval would authorize preparation of `IP@v2`; it would still not authorize implementation, training, dependency changes, Dacon upload/account mutation, external data/weights, remote compute, remote Git, or reopening the 2024 lockbox.
