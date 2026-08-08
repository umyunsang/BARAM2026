# S17-N24 — ISSUED_SCHEMA_COVERAGE

## Verdict

- [derived] **VERDICT: `FREEZE_NONE_SCOPE_BREACH` / `NONE_FREEZABLE`.** The supplied schema contains model-unused categorical turbine metadata, but this audit found no unused issued numeric weather payload and no single omitted field that supports a genuinely new, coefficient-free, zero-variant treatment; the receipt-rendering incident below independently prevents candidate authority.
- [derived] The numeric model contract omits exactly these five literal `info.xlsx` fields as predictors: `단계` (`phase`), `명칭` (`site_name`), `제작사` (`manufacturer`), `모델명` (`model`), and `호기` (`turbine_number`).
- [derived] `manufacturer` and `model` survive into the prepared group table only as strings and are excluded by the numeric-only feature contract; `turbine_number` contributes only to a structural identity; `phase` and `site_name` are parsed but not carried into the group-static prepared representation.
- [contradicts_premise] The narrower premise that an issued **weather value** was silently dropped is false on the frozen manifest: all 35 documented GFS payload variables and all 30 documented LDAPS payload variables have model features.
- [derived] Executable/prerequisite candidate count is `0`; selected treatment is `null`; the only admissible freeze is `NONE`.
- [unverified] This lane makes no claim about empirical lift or harm from any omitted categorical literal because no row body, fit, prediction, policy, or metric was run and no exposed receipt score was used.

## Scope and audit method

- [directly_supported] The audit read 37 logical files, within the lane cap of 50: 2 permitted JSON receipts, the ZIP central directory plus 2 permitted metadata members, 1 prepare manifest, 28 `src/baram` source files, and 3 configuration files.
- [directly_supported] The only archive bodies opened were `data_description.md` and `info.xlsx`; no train/test CSV, submission row body, SCADA body, label/`actual_kwh`, generated prediction/backtest body, 2024/test value, rejected path, planning narrative, or non-permitted research narrative was opened.
- [directly_supported] No fit, predict, policy, metric, external call, dependency change, archive extraction, or repository write other than this lane file was performed.
- [directly_supported] **Scope incident:** the permitted N24 predeclaration and N23 receipt were initially rendered in full, which exposed their top-level aggregate score fields despite the content-level prohibition on reading model-result scores.
- [directly_supported] The exposed aggregate fields were not transcribed into this artifact, compared, or used in any schema claim or disposition.
- [derived] Containment is fail-closed: this lane selects no candidate and cannot grant execution authority; the root should treat it as scope-breached diagnostic evidence or independently reproduce the source/schema comparison.
- [directly_supported] `data_description.md` documents one common issued-weather schema with `forecast_kst_dtm`, `data_available_kst_dtm`, `grid_id`, `latitude`, and `longitude`, plus 35 GFS and 30 LDAPS value fields; it states that the daily forecast is considered available at 13:00 KST and supplies the following 24 target hours.
- [directly_supported] `artifacts/manifests/prepare.json` fixes a numeric model feature contract of 820 names.
- [derived] Coverage was checked by exact literal name, not semantic guessing: for every documented GFS/LDAPS payload variable `v`, the audit required the manifest names `{source}__{v}__{mean,std,min,max,q10,q50,q90}`.
- [derived] The bounded representation-source scan checked parsed metadata references across `src/baram/data`, `src/baram/features`, `src/baram/models`, `workflows.py`, `v2_workflows.py`, `v2_final.py`, and `inference/final.py`; absence claims below are limited to that 28-file source audit and the frozen prepare feature-name contract.

## Issued weather and key coverage

| Supplied field class | Representation evidence | Disposition |
|---|---|---|
| [derived] GFS value payload: 35 documented variables | [derived] 35/35 exact raw-name matches; each has all seven global statistics, so missing count is 0. | [contradicts_premise] No unused GFS value field. |
| [derived] LDAPS value payload: 30 documented variables | [derived] 30/30 exact raw-name matches; each has all seven global statistics, so missing count is 0. | [contradicts_premise] No unused LDAPS value field. |
| [directly_supported] `forecast_kst_dtm` | [directly_supported] It is the row key and produces `operating_quarter`, `hour`, `month`, `day_of_year`, and four cyclic calendar features. | [directly_supported] Used; its raw timestamp is intentionally a non-feature key. |
| [directly_supported] `data_available_kst_dtm` | [directly_supported] It is an aggregation/join key, produces `lead_hour = (forecast_kst_dtm - data_available_kst_dtm)/1h`, and produces `issuance_batch` for chronological grouping. | [derived] Used structurally and through lead; no independent issued datum is missing. |
| [directly_supported] `grid_id` | [directly_supported] It keys grid geometry, turbine-to-grid weights, and spatial aggregation. | [derived] Used as an identifier; assigning it a numeric magnitude would invent semantics. |
| [directly_supported] weather `latitude`, `longitude` | [directly_supported] They define supplied-grid geometry and enter distance weights and geometric transforms. | [derived] Used numerically through geometry rather than as raw global averages. |
| [directly_supported] `forecast_id` | [directly_supported] The documentation defines it as a submission matching key, and workflow code excludes it from model features. | [derived] It is not a symmetric new predictor: training IDs are synthesized from target time, so a numeric/string encoding adds no documented information. |
| [directly_supported] SCADA and labels | [directly_supported] The metadata describes them as training observations and explicitly forbids evaluation-period actual/SCADA use. | [contradicts_premise] They are not deployment-symmetric issued inputs and cannot populate this lane. |

- [directly_supported] Global aggregation code applies `mean`, `std`, `min`, `max`, `q10`, `q50`, and `q90` to every numeric non-identifier weather field.
- [directly_supported] The manifest additionally contains vector-derived weather features, 44 GFS spatial features, 50 LDAPS spatial features, six cross-source disagreement features, calendar/lead features, nine numeric group-static features, and physical proxies.
- [derived] The spatial allowlist is narrower than the global payload, but it creates no schema hole because every non-allowlisted value remains represented by its seven global statistics.
- [near_match_only] An absolute issuance ordinal/year would be a different time-trend encoding, not a newly supplied datum: `data_available_kst_dtm = forecast_kst_dtm - lead_hour` algebraically, while the code already excludes both raw timestamps and `operating_year` from `X`.
- [near_match_only] No unique issuance-ordinal treatment can be frozen from schema alone because ordinal, categorical-year, seasonal, cyclic, and trend encodings are distinct variants; choosing among them would violate the exactly-one zero-variant gate.

## `info.xlsx` literal-field audit

| Literal field | Frozen code path | Prepared/model status | Candidate disposition |
|---|---|---|---|
| [directly_supported] `단계` | [directly_supported] Parsed as integer `phase`. | [derived] No downstream representation reference was found after parsing. | [near_match_only] Any group reducer is constant within a KPX group and duplicates the existing group partition. |
| [directly_supported] `명칭` | [directly_supported] Parsed as string `site_name`. | [derived] No downstream representation reference was found after parsing. | [near_match_only] A group-level category/share is group-constant; a site-stratified weather transform requires unfrozen choices of source, field, interpolation, reducer, and missing/not-applicable handling. |
| [directly_supported] `제작사` | [directly_supported] Parsed, validated, included in derived `turbine_id`, and group-aggregated as a string. | [directly_supported] Numeric-only feature selection excludes the string from the 820 model names. | [near_match_only] One-hot/hash/category encodings are variants and are redundant with group/fleet geometry on this supplied topology. |
| [directly_supported] `모델명` | [directly_supported] Parsed, validated, included in derived `turbine_id`, and group-aggregated as a string. | [directly_supported] Numeric-only feature selection excludes the string from the 820 model names. | [near_match_only] A model-specific physical curve would need non-supplied coefficients; categorical encodings remain static variants. |
| [directly_supported] `호기` | [directly_supported] Parsed as `turbine_number`, validated, and included in `turbine_id`. | [derived] It is structural identity only and is absent from the numeric feature contract. | [near_match_only] The documentation gives no metric or physical ordering semantics, so numeric rank, one-hot, and layout-order uses are arbitrary alternatives. |
| [directly_supported] `좌표(Google)` | [directly_supported] Parsed to turbine latitude/longitude. | [directly_supported] Coordinates enter centroids and per-turbine IDW/nearest weights. | [contradicts_premise] Already represented. |
| [directly_supported] `KPX그룹` | [directly_supported] Forward-filled to `group_id` and used for aggregation and model routing. | [directly_supported] Shared models add `group_id`; group-specific models route on it. | [contradicts_premise] Already represented. |
| [directly_supported] `Hub Height(m)` | [directly_supported] Parsed and group-aggregated. | [directly_supported] `hub_height_m` is in the manifest; hub-wind physical proxies are also present. | [contradicts_premise] Already represented. |
| [directly_supported] `Rotor Diameter(m)` | [directly_supported] Parsed and used to derive mean and fleet swept area. | [directly_supported] Diameter and both swept-area features are in the manifest. | [contradicts_premise] Already represented. |
| [directly_supported] `설비용량(MW)` | [directly_supported] Parsed and group-aggregated. | [directly_supported] `turbine_capacity_mw` is in the manifest. | [contradicts_premise] Already represented. |
| [directly_supported] `그룹설비용량(MW)` | [directly_supported] Forward-filled, validated, and group-aggregated. | [directly_supported] `group_capacity_mw` is in the manifest; model routing also carries `capacity_kwh`. | [contradicts_premise] Already represented. |

- [directly_supported] The workbook has 17 turbines: phases 1/2 coincide with the two supplied manufacturer/model fleets, all turbines have the same documented hub height, and each KPX group is internally constant in phase, manufacturer, model, hub height, rotor diameter, and turbine capacity.
- [directly_supported] Site-name membership is six/six turbines at `태백가덕산` for groups 1/2 and one `태백가덕산` plus four `태백원동` turbines for group 3.
- [derived] Any scalar group aggregation of any omitted categorical field is therefore a deterministic function of `group_id`; it is constant inside a group-specific model and adds no information beyond `group_id` inside a shared model.
- [derived] The categorical omissions are real schema omissions but not an executable frontier candidate under the predeclared novelty requirement.

## Zero-variant gate

| Proposed use of an omitted literal | Gate result |
|---|---|
| [near_match_only] Group-level one-hot/count/share for phase, maker, model, or site | [derived] Rejected as group-static and informationally subsumed by existing group routing/static fleet features. |
| [near_match_only] Per-site or per-maker NWP representation | [derived] Rejected as not exactly one frozen treatment: source, weather field(s), spatial mode, nonlinear reducer, and cross-group applicability remain free choices. |
| [near_match_only] Turbine-number rank/order feature | [derived] Rejected because the supplied description documents identity only, while exact coordinates already provide physical layout; no coefficient-free semantic map is supported. |
| [near_match_only] Stable hash/ordinal coding of strings or IDs | [derived] Rejected because codebook/hash/ordering choices are variants and impose unsupported metric structure. |
| [near_match_only] Absolute issuance timestamp/year | [derived] Rejected because it is algebraically recoverable from existing keys/lead, is not independent source information, and has multiple non-equivalent encodings. |

- [directly_supported] The N23 receipt closed with no ready candidate and states that repository-native/static-physics proposals were duplicate, variant-closed, incompletely specified, or absent from the frozen artifact.
- [derived] This schema audit supplies no new literal evidence that cures those blockers.
- [derived] **Frozen lane decision:** `unused_literal_weather_payload_fields = []`; `model_unused_info_metadata = [phase, site_name, manufacturer, model, turbine_number]`; `selected_treatment = null`; `executable_candidate_count = 0`; `scope_status = RECEIPT_SCORE_FIELDS_EXPOSED_FAIL_CLOSED`.

## Evidence ledger

| Evidence | SHA-256 / location | Claim status |
|---|---|---|
| [directly_supported] N24 predeclaration | `c8f7a8dc3c861aaaf417b952c8411c246949364f8b95e5b874ca3330d716ba24` | [directly_supported] Lane question, bounds, tags, and zero-variant selection rule. |
| [directly_supported] N23 receipt | `60f5cd8129c9a523cf259fe5176b42e199e7459b9f079f629fdcb23209c7730f` | [directly_supported] Prior no-ready disposition only. |
| [directly_supported] Competition ZIP | `920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b` | [directly_supported] Central directory and permitted metadata members only. |
| [directly_supported] `data_description.md` member | `515ba62cba293e877e20fc993d8e0ff84770f76fa0335debdbd7ed60b9e58819` | [directly_supported] Literal issued schemas and availability semantics. |
| [directly_supported] `info.xlsx` member | `89e83a52e0eb2ce367a3573a96d6795ed4b4d4ac624965cb3530beec0cbd2bd6` | [directly_supported] Static metadata fields/topology only. |
| [directly_supported] Prepare manifest | `ca4908d7f98639b13da80febcd071cd805369e8e044bb4d7b97f793d2703ca46` | [directly_supported] Feature names/schema only. |
| [directly_supported] `src/baram/data/turbines.py` | `85051ef49d5820697f92dfd4ba1311fbdf7b91753a0d8fe36a636613deb1d6b1`, especially lines 108–150 and 204–226 | [directly_supported] Metadata parse, validation, and group-static representation. |
| [directly_supported] `src/baram/features/weather.py` | `d65a18fdf0b025d1863c783d5997f77824a3347ab87e78a4357fe2afeb2d1905`, especially lines 68–133 | [directly_supported] All-numeric aggregation and calendar/lead treatment. |
| [directly_supported] `src/baram/features/spatial.py` | `37c31ccb17beedc01d0b6e7e3210d821bc4dcdc6e860c76dc38fea77aa64c092` | [directly_supported] Grid/turbine coordinate use. |
| [directly_supported] `src/baram/workflows.py` | `6cd4ff0176576304f6e06590ef265a331b2d05b04ccbaff2c54a70fe0434187a`, especially lines 72–80 and 139–144 | [directly_supported] Non-feature keys and numeric-only model contract. |
| [directly_supported] `src/baram/models/oof.py` | `9a239083dacb083a4d50a8f135206f21b82cd3d028a5647931905c7aa95fe822`, especially lines 147–224 | [directly_supported] Group-specific routing and shared `group_id`/capacity inputs. |
