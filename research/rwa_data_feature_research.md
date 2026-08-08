# RWA-DATA — issuance-safe day-ahead wind data and feature operations

**Lane:** read-only RWA-DATA  
**Research cut-off / retrieval date:** 2026-08-08  
**Primary/official sources used:** 12 (limit: 14)  
**Local output:** this file only; no data download, preprocessing run, fit, score, lockbox, account, or source-code change was performed.

## 1. Decision summary

The literature does **not** justify crediting any data/feature operation with BARAM's remaining
`0.0238157506` official-`Total` gap. Because
`Total = 0.5*(1-NMAE) + 0.5*FICR`, closing that gap requires a combined component movement of
`0.0476315012`; effects reported as surface-wind RMSE, annual NMAE, pinball loss, CRPS, or turbine
conversion MAE cannot be mapped to it.

What the primary evidence does establish is narrower:

1. The forecast table must be an **issuance-keyed cube**, not a valid-time-only table: retain NWP
   source, forecast reference/run, actual availability time, valid time, lead, vertical level, grid
   point, and temporal support. Historical training inputs must reproduce the run/lead available at
   BARAM's D-1 14:00 KST basis time. This is a contract/audit, not an expected score gain.
2. Missing NWP is an **operational state by source and run**. A deterministic source-specific
   fallback is better supported than one generic imputer, but BARAM's source-specific missing rates
   are not yet established and no score gain should be assigned before that audit.
3. Curtailment/outage treatment depends on the forecast target. BARAM scores **actual production**
   and supplies no future setpoint, Power Available, or outage schedule. Literature that removes
   curtailed rows or adds curtailed energy targets a different estimand and cannot be copied.
4. The only information-adding feature family still worth a bounded re-entry is **same-run vertical
   profile / terrain-height reconstruction**. Even this is not new: S6/S13 already own the relevant
   nodes. Korean evidence is close in terrain and horizon, but its metrics and populations differ.
5. Circular direction, Jensen-aware conversion, generic spatial summaries, generic physical
   corrections, and valid-time-neighbour trajectory features are already represented or explicitly
   closed locally. Generic multi-NWP stacking is also not established as better than concatenation.

Accordingly, §6 retains only **four conditional/re-entry experiments**. None is represented as a
stand-alone route to `0.66`.

## 2. Scope and evidence rules

### 2.1 BARAM migration surface

All migration statements below use this fixed target surface:

- forecast made by **D-1 14:00 KST** for the next operating day;
- competition-supplied LDAPS/GFS forecasts and public, pre-cutoff static information only;
- no reanalysis, test-period observation, or post-basis forecast revision;
- actual hourly group production as target;
- official `Total = 0.5*(1-NMAE) + 0.5*FICR`;
- strict issuance-batch chronological OOF, with fold-local fitting for every learned transform;
- comparison on identical row keys and one fixed decision policy.

The existing magnitude reference is `0.15 * 0.0238157506 = 0.00357236259 Total`. It is used only to
avoid chasing microscopic operations; it is **not** a new promotion rule and does not replace the
project's frozen gates.

### 2.2 Allowed evidence tags

Every material external claim appears once in §4 and has exactly one of the four allowed tags:
`directly_supported`, `contradicts_premise`, `near_match_only`, or `insufficient`.
`directly_supported` means that the cited source supports the stated operation or contract in its
stated scope; it does not mean that its numerical effect transfers to BARAM.

## 3. Targeted de-duplication against S5/S6

Only the four requested local lane documents were used for this check:

- `research/lanes/S5_preprocessing_research.md`
- `research/lanes/S6_feature_research.md`
- `research/lanes/S13_S5_preprocessing_deep.md`
- `research/lanes/S13_S6_features_deep.md`

| Requested operation | Existing local owner / closure | Disposition here |
|---|---|---|
| Label/SCADA clock alignment | S5 §1.2: zero-shift hour-ending alignment wins the ±2 h scan; closed. | CF metadata does not reopen label alignment. |
| NWP reference/valid/interval alignment | S5 P10 and S13/S5 N8 explicitly leave the NWP-side convention unresolved. | Retain only as re-entry **R1**, not a new idea. |
| Missing NWP | S5 P11 defines training-vs-operational missingness; S13/S5 N10 already proposes source-specific drop/fallback and records that local NWP missing rates were not counted. | Mandatory audit/robustness contract; score candidate only if real source/run gaps exist. |
| Curtailment, outage, anomalies | S5 P1/P5/P6 and S13/S5 N3/N10 already cover removal, censoring/restoration, and rule-based anomaly handling. S5 also shows availability deficits are weakly predictable. | No new node without forecast-time setpoint/Power-Available/outage metadata. |
| Spatial aggregation/downscaling | S6 §§1.6–1.7 estimates only about 2–3% incremental pc-MAE information in the 16-cell field and owns C01–C05/C12/C13; S13/S5 N4/N7 owns terrain-height correction and all-cell/order-statistic alternatives. | Generic IDW, all-cell, PCA, direction-aware grid, and terrain labels are duplicates. R3 is only a one-shot re-entry of N4. |
| Hub-height / vertical reconstruction | S6 C08/C12 and S13/S6 F6 already specify rotor/profile and actual-elevation sampling; the follow-up explicitly notes the risk that coarse-model height can lose to resolution. | R2 is a controlled re-entry, not a new family. |
| Physical and circular features | S6 closes measured direction, NWP direction realignment, simple speed bias/shrinkage, static terrain indices, geostrophic wind, veer/advection, intra-hour point correction, and generic terrain speed-up. S5 confirms current curve-then-average Jensen handling. | Density, direction sine/cosine, and Jensen operations are evidence for the existing design, not next nodes. |
| Multi-NWP fusion | S6 C07 and S13/S6 F8 already own LDAPS–GFS disagreement/phase features; the project also closes acquisition of additional NWP on correlation/availability evidence. | R4 is conditional on an untested **existing-source, source-separated vs concat** contrast; no new source is proposed. |
| Forecast trajectory | S13/S6 F12 explicitly closes ramp/trajectory shape because the deployed representation already has ±1/2/3/6 h valid-time windows. | Literature lead/current/lag features are a duplicate; no trajectory node. |

## 4. Claim-level evidence ledger

| ID | One material claim | Tag | Primary/official source and date (all retrieved 2026-08-08) | Exact population / horizon | Inputs, target, metric | Exact BARAM migration or non-transfer reason |
|---|---|---|---|---|---|---|
| C01 | A forecast field must distinguish forecast reference/run time from forecast valid `time`. | `directly_supported` | CF Conventions 1.13 (2025), §§5.7 and Example 5.14, [S1] | Normative metadata standard; no empirical population or horizon. | Forecast coordinates; no target/model/metric. | Store both axes and derive lead as valid minus reference. Add a separate `available_at` receipt because CF reference time alone does not prove delivery before D-1 14:00. A valid-time-only join is inadmissible. |
| C02 | A timestamp alone does not identify whether a value represents an instant or an interval; bounds and `cell_methods` carry that support. | `directly_supported` | CF Conventions 1.13 (2025), §§1.4 and 7.3, [S1] | Normative standard; all gridded forecast horizons. | Coordinate bounds and cell aggregation method; no score. | Preserve instantaneous/mean/maximum/accumulation semantics. Match an hourly NWP mean to its declared bounds; do not infer a half-hour shift from the coordinate or average adjacent steps without metadata. |
| C03 | Day-ahead training weather should use day-ahead forecast runs, not realized weather/analysis; multiple run revisions for one valid time make the issue key operationally necessary. | `directly_supported` | Olauson, Viotti & Huss (2026) [S3]; Pu et al. (2025) [S4] | HEFTCom: 1200 MW Hornsea 1; daily next-day 48 half-hours. Pu fixes 00 UTC reference and lead 23–47 h; Olauson trains 2020-09-21–2022-12-31 and tests 2023. | Historical GFS/DWD/MEPS forecasts; wind/combined energy; pinball loss. Pu notes forecasts revised every six hours. | Build the identical BARAM run/lead slice in every fold and in inference. Never choose “latest for valid time” using a revision that arrived after the historical basis. The score effects do not transfer because the source grids, target, and pinball metric differ. |
| C04 | A production pipeline needs an explicit pre-deadline fallback when the preferred NWP run has not arrived. | `near_match_only` | Olauson, Viotti & Huss (2026), operational schedule in §4, [S3] | HEFTCom next-day forecasts submitted 09:20 UTC; MEPS 06 run was used if ready, otherwise the already downloaded 00 run. | Run availability, weather-only component models; pinball loss. | A BARAM fallback may select only a run whose real availability is no later than D-1 14:00. Provider schedules differ, so Olauson's 00/06 rule cannot be copied literally; predeclare the BARAM source/run priority. |
| C05 | Setpoints, operational capacity, status/quality flags, and Power Available distinguish meteorological production from curtailment/outage. | `directly_supported` | IEA Wind Task 36/51 Recommended Practice Part 4 (2022), Ch. 3, [S2] | Recommended practice for real-time renewable forecasting; emphasis from seconds/minutes to hours, not one empirical farm. | Active power, controller setpoint, capacity in operation, breaker/status, Power Available; actual or unconstrained production; no single metric. | These would be admissible only if known by the BARAM basis time. They are absent for the target year, so inferred future curtailment is unavailable; retain flags only for training diagnosis and document the target as actual production. |
| C06 | Curtailment rows must be handled according to the estimand: remove them when evaluating available power, or retain and report them separately when evaluating actual production. | `directly_supported` | Messner et al., *Wind Energy* 23 (2020), pp. 1461–1481, [S5] | Forecast-evaluation guidance; examples include 24 h wind forecasts, but the curtailment recommendation is application-defined, not an effect estimate. | Deterministic/probabilistic forecasts; MSE, MAE, quantile score and event metrics. | BARAM evaluates actual production, so available-power evaluation masks cannot be used for official comparison. Candidate and control must be scored on identical rows; unexpected curtailment may be reported as a diagnostic stratum. |
| C07 | HEFTCom added curtailed energy to the wind target; that made it a potential-generation target rather than actual metered production. | `contradicts_premise` | Olauson, Viotti & Huss (2026), §1.1, [S3] | Hornsea 1, daily next-day half-hourly quantiles; 2023 test and 2024 competition. | Actual wind plus provided curtailed-energy series; pinball loss. | Do **not** copy target restoration into BARAM: no 2025 curtailed-energy/Power-Available label exists and actual production is scored. This directly defeats an argument that a winning benchmark universally endorses restoring BARAM labels. |
| C08 | Turbine anomaly masks in SDWPF are tied to turbine SCADA/status semantics, not universal wind-farm thresholds. | `near_match_only` | Zhou et al., SDWPF/KDD Cup (2022), [S8] | One farm, 134 turbines, 245 days, 10-min data; 48 h (288-step) forecasts without future weather. | Turbine wind, direction, pitch and active power; unknown/missing rows excluded; mean of RMSE and MAE. | Rules such as power≤0 with wind>2.5 m/s or pitch>89° cannot migrate to aggregate BARAM labels without turbine pitch/status and test-time observability. S5/S13 already own anomaly diagnosis. |
| C09 | Source-specific models can make NWP outages survivable by dropping unreliable rows within a source and filling the meta-input from available component forecasts. | `near_match_only` | Olauson, Viotti & Huss (2026), §§2–4, [S3] | Same HEFTCom scope; any-variable missing rates over 2020-09-21–2023-12-31 were GFS 0.3%, DWD 1.5%, MEPS 1.8%. | Separate weather-source CatBoost quantiles; incomplete component rows dropped; meta-level filled across available outputs; pinball loss. | First count BARAM missingness by source/run/lead/variable/grid. Use issue-block source dropout, not random scalar masking. Its value may be submission robustness only; the paper does not establish BARAM `Total` gain. This duplicates S13/S5 N10. |
| C10 | For operationally missing recent SCADA inputs, retraining on remaining inputs outperformed imputing the missing inputs in a very-short-term VAR case study. | `near_match_only` | Tawn, Browell & Dinwoodie, PSCC (2020), [S6] | Missingness characterized on 30 European farms; experiment on 10 complete sites, half-hourly, 2.5 h horizon. | Recent power/wind lags, VAR/LASSO; NMAE. Training missingness and operational missingness were studied separately. | This does not validate day-ahead NWP imputation: horizon, input process, topology, and missing mechanism differ. It supports only a fallback ablation, not a blanket “drop beats impute” rule. |
| C11 | Joint generative treatment of missing inputs and targets beat “impute then predict” on CRPS under simulated missingness. | `near_match_only` | Wen, Pinson, Gu & Jin (2024), arXiv:2403.03631, [S7] | U.S. WIND Toolkit, hourly 2007–2013; random MAR missingness up to 25%; lead 1–3 h; local and neighbouring-farm cases. | Historical power, generative probabilistic model; CRPS. | BARAM missing NWP may be block/source/run dependent rather than random MAR, the horizon is 24 h class, and the official target is a deterministic settlement action. Do not add this modeling-heavy imputer as an RWA-DATA node. |
| C12 | Retaining the complete local NWP grid beat a central point or spatial mean in a day-ahead offshore benchmark. | `near_match_only` | Olauson, Viotti & Huss (2026), §3.2, [S3] | Hornsea 1 MEPS component model; daily next-day 30-min wind quantiles. | All grid cells vs one central cell vs spatial mean; pinball loss. One point and mean increased wind pinball loss by about 2% and 1%, respectively. | BARAM already retains/derives 16-cell information and locally measures only about 2–3% incremental pc-MAE information over IDW; S6 C02/C03 and S13 N7 own this. Offshore pinball percentages cannot be credited to `Total`. |
| C13 | Higher-resolution physical downscaling is not automatically better: the original KMAPP failed to reduce LDAPS errors and produced about 0.1 m/s in deeply overcorrected valleys. | `contradicts_premise` | Keum et al., *Atmosphere* 31(1) (2021), [S9] | 16 ICE-POP AWS sites, 390–1416 m elevation, February 2018; daily 00 UTC forecasts to +36 h, hourly verification. | LDAPS→100 m KMAPP surface wind; AWS surface wind; MBE/RMSE/IOA. | This directly rejects generic “add finer terrain/roughness downscaling” as a BARAM candidate. Any re-entry needs an explicit height-only control and valley/extreme-value checks. |
| C14 | With revised terrain parameters, interpolation plus height correction was the best KMAPP-Wind sensitivity cell; roughness adjustment did not improve it. | `near_match_only` | Keum et al. (2021), [S9] | Same 16-site, one-winter, +36 h surface-wind population. | Height-only cell RMSE 1.82 m/s; height+roughness 1.91; interpolation-only 2.39. The revised system reduced forecast error 21.2% versus original KMAPP. | This is close in Korean complex terrain and large enough to falsify once, but it is surface wind rather than hub-height power, a one-month winter sample, and RMSE rather than `Total`. It is exactly S13/S5 N4, not a novel node. |
| C15 | Same-run model-level heights can be converted to fixed-height winds by fitting wind speed against log height through the lowest model levels. | `insufficient` | Olauson, Viotti & Huss (2026), §2, [S3] | HEFTCom/MEPS; daily next-day half-hourly wind forecast. | Time/space-varying heights of eight lowest model levels; cubic fit in log height; derived 50/100/150 m wind; pinball model. | The operation is issue-safe if all levels/heights come from the selected run, but no isolated ablation is reported. It cannot support a gain claim; it only informs R2's reconstruction cell. |
| C16 | Site-specific vertical-layer selection and PCA across layers improved a Korean complex-terrain day-ahead forecast relative to using only lower/turbine-height layers. | `near_match_only` | Lee, Park, Kim & Hong, *Energy* 288 (2024), 129713, [S10] | Yeongyang farm, 25 turbines; 2019–2021 train/validation and 2022 test; WRF lead 21–45 h, 10-min output; seven layers from ground to about 588 m. | WRF vertical wind speed/direction and meteorology, LGBM, turbine/farm power; annual NMAE. Authors report reduction “up to 1.2%.” | This is the closest population/horizon match, but it is one WRF hindcast/farm, uses a different learner and NMAE only, and the reported percentage cannot be translated to FICR or `Total`. PCA/layer choice must be fit inside each chronological fold. Duplicates S6 C08/C12 and S13/S6 F6. |
| C17 | Wind-direction and calendar sine/cosine encodings were used in the HEFTCom winner, but their standalone effect was not isolated. | `insufficient` | Olauson, Viotti & Huss (2026), §3.2, [S3] | Hornsea 1, daily next-day half-hourly quantiles. | NWP direction and calendar; CatBoost; pinball feature importance/overall score only. | BARAM already has circular direction/calendar features and S6 closes direction realignment. Usage without an ablation is not evidence to reopen it. |
| C18 | Density-normalized wind speed reduced power-curve estimation error in measured turbine data, especially in extreme temperatures. | `near_match_only` | Dupré et al., Annales Geophysicae Discussions (2019), [S12] | Bonneval, France: six Vestas V80-2 MW turbines, 2015–2017, 10-min observations. | Measured wind/temperature plus nearest ERA5 pressure; measured power; normalized MAE/NRMSE. Overall normalized MAE was 0.96%→0.77%. | The study is power-curve estimation, not day-ahead forecast; it uses test-time measurements/reanalysis that are forbidden here. Only same-run forecast T/p would be legal, and S6 already owns density/physical features. No new node. |
| C19 | Turbine/farm wind-speed dispersion matters when a nonlinear power curve is applied to an average wind. | `directly_supported` | McCandless & Haupt, *Wind Energy Science* 4 (2019), [S11] | Simulated 100-turbine farm and measured five-turbine, 10 MW Shagaya farm. | Turbine wind dispersion and power curves; conversion MAE in kW per 2 MW turbine. Mean+SD RF reduced 68.83 to 51.15 kW. | This supports Jensen-aware aggregation, but S5 verifies that BARAM already uses curve-then-average. It is not evidence for a second Jensen node or for a direct day-ahead `Total` effect. |
| C20 | Sister-model stacking of DWD and GFS improved HEFTCom wind pinball loss relative to either single source. | `near_match_only` | Pu et al. (2025), [S4] | Hornsea 1; fixed 00 UTC reference, lead 23–47 h; tests 2023-02-01–08-01 and 2024-02-20–05-19. | DWD 36 and GFS 9 grid points; 100/10 m wind; LightGBM quantiles; MPL/MCRPS/MWS. Stacking improved MPL 6.32–10.87% vs single-source models. | The comparator was **single-source**, not a same-feature concatenated two-source model; source errors, probabilistic metric, offshore population, and topology differ. This cannot be treated as a 6–11% BARAM gain. |
| C21 | Multi-NWP stacking is not intrinsically superior to concatenating the same weather sources. | `contradicts_premise` | Olauson, Viotti & Huss (2026), Table 1 and §5, [S3] | Full-year 2023 HEFTCom test; common 93.2% timestamps. | Three-source stacked CatBoost vs all-weather-in-one CatBoost without lagged target; pinball 28.5 vs 28.6. Authors call performance “very similar.” | R4 must compare source separation against the current concat topology on common BARAM rows. Literature supports robustness/diversity, not a presumed stacking gain. |
| C22 | Valid-time neighbours within the **same issued forecast** are a standard trajectory representation, but the cited evidence is feature use/importance rather than a causal ablation. | `near_match_only` | Olauson et al. (2026) [S3]; Pu et al. (2025) [S4] | HEFTCom daily next-day 30-min targets; Pu uses lead 23–47 h. | Olauson uses wind-speed ±1/2 samples (lagged importance 64% vs raw 30%); Pu uses spatial min/mean/max at lead/current/lag; pinball-based models. | Neighbour values are legal only from the same pre-basis run. BARAM already has ±1/2/3/6 h windows and S13/S6 closes F12. Importance is not incremental score evidence, so no new trajectory/ramp node. |

## 5. Operational contracts that should exist even if they score zero

These are pipeline invariants, not candidates for closing the gap.

### 5.1 Canonical issuance cube

Minimum key/metadata per value:

```text
source, forecast_reference_time, available_at, cycle,
valid_time, lead_time, temporal_support, bounds,
variable, level_type, level_value, grid_id, value, quality_flag
```

Required assertions:

1. `available_at <= D-1 14:00 KST` for every value used by a prediction.
2. `lead_time = valid_time - forecast_reference_time`; no duplicate full keys.
3. One predeclared cycle-selection rule is used in train/OOF/inference.
4. Valid-time neighbours come from the same `source × forecast_reference_time` cube.
5. Accumulations/means have bounds and operation type; resets are handled by run, never by
   cross-run differencing.
6. A forecast archive is rejected if it is only a latest-revision series and cannot reconstruct the
   historical basis-time snapshot.

### 5.2 Missing-NWP state machine

Classify missingness at issue granularity: entire source/run absent, variable/level absent, grid-cell
absent, or isolated corrupt value. Record a source/run/lead coverage matrix. The inference policy
must terminate with finite predictions under each state, with a predeclared priority such as:
preferred source-run → older still-valid run → other existing source component → frozen climatology
or benchmark fallback. Never select a fallback after observing the target. Evaluate the normal-data
score and source-block-drop robustness separately.

### 5.3 Operational-state/target contract

Store, when available, timestamp, quality flag, installed/operational capacity, controller setpoint,
Power Available, outage/maintenance status, and the settlement-meter target definition. Since these
are unavailable for BARAM 2025, no derived future curtailment feature is admissible. Training-only
anomaly flags may define diagnostics or weights, but official evaluation remains on actual output.

### 5.4 Comparison contract

Every experiment must state: input policy, forecast cycle, fitting surface (fold-local or fixed),
row-alignment keys, and output decision policy. Compare on the intersection of rows and also report
coverage. Report `Total`, `1-NMAE`, FICR, group/month blocks, and missing-source stress separately;
a gain caused by silently dropping difficult rows is invalid.

## 6. At most five next-node candidates (four survive)

The four below are **conditional falsification/re-entry nodes**, ordered by information value and
cost. A prior local receipt showing the exact contrast closes that row immediately.

### R1 — NWP issue/valid/interval semantics audit (re-entry of S5 P10 / S13 N8)

- **Operation:** Materialize §5.1 and read the supplied NWP metadata. Freeze the basis-time cycle and
  temporal support for every source/variable. If metadata leave the hourly support ambiguous, compare
  only the four already predeclared mappings: current `t`, `t-1`, mean(`t-1`,`t`), and
  mean(`t`,`t+1`), where both members must belong to the same issued run.
- **Availability/leakage test:** Prove each selected run was available by D-1 14:00; prohibit
  valid-time “latest revision” joins; assert same-run neighbours; do not use target error to choose a
  cycle. Metadata resolve the node before any fit when possible.
- **Discriminating experiment:** First compare NWP wind against training-only SCADA wind under the
  four fixed mappings. Only a clear winner proceeds to one root-run strict chronological OOF with
  every other feature/model/policy fixed. The row key is `(issuance_batch, valid_time, group)`.
- **Gap sizing / stop:** A gross one-hour or interval error could be macroscopic and is the only
  reason this audit survives. If metadata validate current mapping or OOF gain is below the existing
  materiality reference, close with zero expected gain. This is not a new feature search.

### R2 — Fold-local vertical-layer selection/PCA (re-entry of S6 C08/C12 / S13 F6)

- **Operation:** From the chosen same-run LDAPS/GFS vertical profile, create (a) a fixed physical
  interpolation at the documented hub/effective elevation and (b) a small fold-local PCA of u/v or
  speed/direction across native levels. Keep component count fixed before scoring; retain source
  identity.
- **Availability/leakage test:** Every level and model-level height must come from the same pre-basis
  run. Fit scaling, PCA loadings, and any site/group layer selection on the training part of each
  chronological fold only. No 2024 lockbox, 2025 observation, reanalysis, or whole-period
  correlation-based layer choice.
- **Discriminating experiment:** Three cells with identical learner/policy: current vertical block;
  fixed-height reconstruction; fixed-height plus fold-local PCA. Match feature count with inert/noise
  controls if the learner is sensitive to dimension. Report official components plus error by NWP
  wind regime. Do not combine with R3 until one cell independently passes.
- **Gap sizing / stop:** Lee et al.'s closest match reports annual NMAE reduction up to 1.2%, which is
  potentially above the local materiality reference but cannot imply FICR or `Total`. Reject if gain
  is only in point accuracy while FICR offsets it, or if fold loadings/layer choices are unstable.

### R3 — Static terrain-height correction, height-only control (re-entry of S13/S5 N4)

- **Operation:** Apply the published KMAPP-style correction using only the already selected forecast
  wind and pre-cutoff static site/model terrain metadata. Test interpolation-only, height-only, and
  height-plus-roughness as separate fixed cells; do not silently bundle generic downscaling.
- **Availability/leakage test:** DEM/terrain parameters must be public, licensed, static, and fixed
  before OOF; no coefficient may be calibrated on validation/test SCADA or reanalysis. Check sign,
  finite range, valley/ridge behaviour, and extrapolation at every grid/group before fitting power.
- **Discriminating experiment:** Same baseline and rows for all three cells. First test training-only
  hub/SCADA wind RMSE and bias by elevation/sector, then one strict OOF official-score comparison.
  Height-only must beat both current and height+roughness to validate the claimed mechanism.
- **Gap sizing / stop:** The Korean source's 21.2% surface-wind RMSE reduction is large enough for a
  one-shot test but is a one-winter surface-station result. Stop after this factorial; any valley
  overcorrection, no wind-skill gain, or official gain below the existing materiality reference
  closes the family.

### R4 — Existing-source, source-separated pipeline versus concat (conditional)

- **Operation:** Use only independently issued LDAPS and GFS already present. Train one otherwise
  identical component per source, then combine outputs with (a) fixed equal weights and (b) at most
  one fold-outside linear weight. Predeclare an inference fallback for a whole-source outage.
- **Availability/leakage test:** Each component uses its own pre-basis run and source-specific
  preprocessing; meta-predictions for training rows must be out-of-fold; the combiner is never fit on
  the row it predicts. State each input policy, fitting surface, and alignment key. No new NWP, latest
  archive, test observation, or per-group/per-lead weight search.
- **Discriminating experiment:** On the same common rows compare current all-source concat,
  source-separated equal blend, and source-separated one-degree-of-freedom fold-outside stack.
  Separately simulate entire-source/run outages by issuance block. This distinguishes source-specific
  preprocessing from ensemble averaging and robustness from normal-data score.
- **Gap sizing / stop:** Pu et al.'s 6.32–10.87% pinball gains are only versus single-source models;
  Olauson et al. find stack 28.5 versus concat 28.6. Therefore this is low-confidence and runs only if
  the exact BARAM contrast has not already been adjudicated. If concat ties/wins, retain at most the
  deterministic fallback and assign zero gap-closing credit.

### Explicit non-candidates

No fifth node is warranted. Do not reopen: label/SCADA shifts; generic missing-value imputation;
curtailment-label restoration; turbine anomaly thresholds; central-point/mean/IDW variants; generic
terrain or speed bias correction; air density; direction/calendar circular encoding; Jensen
conversion; or trajectory/ramp summaries. They are unavailable, duplicated, locally closed, or too
weakly matched to the BARAM target/metric.

## 7. Source inventory (12/14)

All links were retrieved on **2026-08-08**. The Lee article's publisher/institutional abstract and
indexed author-manuscript text were available; its transfer is therefore deliberately tagged only as
a near match. No secondary article supplies a substantive claim.

1. **[S1]** CF Community (2025), *NetCDF Climate and Forecast (CF) Metadata Conventions*, v1.13.
   https://cfconventions.org/Data/cf-conventions/cf-conventions-1.13/cf-conventions.pdf
2. **[S2]** IEA Wind Task 36/51 (2022), *Recommended Practice for the Implementation of Renewable
   Energy Forecasting Solutions, Part 4: Meteorological and Power Data Requirements for Real-time
   Forecasting Applications*, 1st ed.
   https://iea-wind.org/wp-content/uploads/2022/06/IEAWind_Task36_Recommended_Practice_Part4_1st_Edition_public.pdf
3. **[S3]** Olauson, J., Viotti, O. & Huss, J. (2026), “The HEFTCom2024 winning model: A stacked
   CatBoost approach for probabilistic wind and solar power forecasting,” *International Journal of
   Forecasting*. https://doi.org/10.1016/j.ijforecast.2026.02.005
4. **[S4]** Pu, C., Fan, F., Tai, N., Liu, S. & Yu, J. (2025), “A Hybrid Strategy for Probabilistic
   Forecasting and Trading of Aggregated Wind-Solar Power: Design and Analysis in HEFTCom2024.”
   https://arxiv.org/html/2505.10367v2
5. **[S5]** Messner, J. W., Pinson, P., Browell, J., Bjerregård, M. B. & Schicker, I. (2020),
   “Evaluation of wind power forecasts—An up-to-date view,” *Wind Energy* 23, 1461–1481.
   https://doi.org/10.1002/we.2497
6. **[S6]** Tawn, R., Browell, J. & Dinwoodie, I. (2020), “Missing Data in Wind Farm Time Series:
   Properties and Effect on Forecasts,” PSCC 2020.
   https://pscc-central.epfl.ch/repo/papers/2020/720.pdf
7. **[S7]** Wen, H., Pinson, P., Gu, J. & Jin, Z. (2024), “Tackling Missing Values in Probabilistic
   Wind Power Forecasting: A Generative Approach.” https://arxiv.org/html/2403.03631v1
8. **[S8]** Zhou, J. et al. (2022), “SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting
   Challenge at KDD Cup 2022.” https://arxiv.org/html/2208.04360v2
9. **[S9]** Keum, W.-H., Lee, S.-H., Lee, D.-I., Lee, S.-S. & Kim, Y.-H. (2021), “Evaluation and
   Improvement of the KMAPP Surface Wind Speed Prediction over Complex Terrain Areas,” *Atmosphere*
   31(1), 85–100. https://doi.org/10.14191/Atmos.2021.31.1.085
10. **[S10]** Lee, K., Park, B., Kim, J. & Hong, J. (2024), “Day-ahead wind power forecasting based
    on feature extraction integrating vertical layer wind characteristics in complex terrain,”
    *Energy* 288, 129713. https://doi.org/10.1016/j.energy.2023.129713
11. **[S11]** McCandless, T. C. & Haupt, S. E. (2019), “The super-turbine wind power conversion
    paradox: using machine learning to reduce errors caused by Jensen's inequality,” *Wind Energy
    Science* 4, 343–353. https://doi.org/10.5194/wes-4-343-2019
12. **[S12]** Dupré, A., Drobinski, P., Badosa, J., Briard, C. & Plougonven, R. (2019), “Air Density
    Induced Error on Wind Energy Estimation,” *Annales Geophysicae Discussions*.
    https://doi.org/10.5194/angeo-2019-88

## 8. Final handoff

The operational priority is: **R1 metadata audit → R2 vertical representation → R3 height-only
terrain correction → conditional R4 existing-source separation**. Missingness and operational-state
handling are mandatory reliability contracts, but receive zero gap-closing credit until BARAM-specific
availability and score evidence exists. Root owns any fit, adjudication, closure, and repository work
outside this report.
