# S17-N24 — CFSv2 exact-treatment documentation audit

## 0. Contract and verdict

**Lane:** `CFSV2_EXACT_TREATMENT`  
**Research window:** 2026-08-09 KST, bounded to less than 25 minutes.  
**Evidence set:** seven unique primary NOAA documentation/legal URLs; five are controlling and two
legacy/special-purpose documents are recorded only as exclusions. No catalogue, THREDDS, data/API
or object endpoint was opened. No `HEAD`, range/list request, data body, download, inference,
account action, local generated-data read, label, fit, prediction, metric, score, or prohibited-period
value was used. [directly_supported]

**Verdict: `NOT_READY_PREREQUISITE_ONLY`; selection result `NONE`.** [derived]

The official documentation establishes that the named collection is a real-time **operational
CFSv2 forecast** record, not operational analysis, reanalysis, or retrospective reforecast; it also
establishes nominal global archive coverage from 2011 onward, GRIB2, four issue cycles, four member
IDs, six-hourly wind families, and NOAA public-domain status. [directly_supported] One and only one
zero-choice candidate representation is written below. It is **not executable or `READY`**, because
the allowed pages do not establish a worst-case publication time for the required `D-2 00Z` run,
day-by-day/member-by-member archive continuity, an explicit anonymous historical-access contract,
or the exact coordinate/vertical/time metadata needed to apply the frozen mapping without an object
audit. [unverified] The nominal match is not promoted. [near_match_only]

Evidence tags have their predeclared meanings: `[directly_supported]` is literal controlling official
text; `[derived]` is arithmetic or a deterministic rule built from it; `[near_match_only]` cannot
support `READY`; `[contradicts_premise]` is literal contrary evidence; and `[unverified]` is a gate
not resolved in this documentation-only lane.

## 1. Operational forecast, not analysis/reforecast

The current NCEI product page places four different collections in separate categories: **CFSv2
Operational Forecasts** and **CFSv2 Operational Analysis** begin in April 2011, while **CFS
Reforecasts** and **CFS Reanalysis (CFSR)** are historical products ending by March 2011. It further
says that the operational analysis and forecasts continue the earlier CFSRR record. [S1]
[directly_supported]

The forecast-specific NCEI metadata record is titled *Climate Forecast System Version 2 (CFSv2)
Operational Forecasts*, identifies itself as `NCDC-CFSV2_FORECAST`, describes four-times-daily
real-time operational forecasts, gives global coverage and GRIB2 format, and marks the record ongoing.
Its lineage says NCEP produced the outputs and NCEI archived them. [S2] [directly_supported]

The official CFSv2 paper independently says CFSv2 became operational at NCEP in March 2011 and
explicitly distinguishes the coupled reanalysis, retrospective forecasts, and subsequent real-time
operational forecasts. Appendix C separately describes the operational run configuration. [S4]
[directly_supported]

Therefore a file selected from the forecast record by a preserved initialization is operational
forecast information. It is not made admissible merely by the word “CFSv2”: the analysis and
reforecast collections remain excluded. [derived] The current NCO naming convention preserves an
initial date and member identifier, so this is not documentation for a latest-run stitched valid-time
series. [S3] [directly_supported] Actual GRIB initialization/step metadata was not inspected and
remains an object-audit prerequisite. [unverified]

## 2. Issue, member, cadence, and `D-2 00Z` audit

### 2.1 What the official pages support

- The NCO product page states that `YYYYMMDDHH` is the **initial date** and `xx` is member `01`,
  `02`, `03`, or `04`. It names the six-hourly time-series family
  `wnd10m.xx.YYYYMMDDHH.daily.grb2`. [S3] [directly_supported]
- NCEI lists operational forecast cycles at `00`, `06`, `12`, and `18 UTC`, with six-hourly forecast
  products `+00, +06, +12, ...` extending for months. [S1] [directly_supported]
- Appendix C of the CFSv2 paper says each day has four control runs, one from each of those cycles,
  extending nine months. At `00 UTC`, three additional perturbed runs extend one season; at each of
  `06/12/18 UTC`, three additional perturbed runs extend 45 days. Thus the four runs at a `00Z`
  initialization all exceed the short lead range needed here. [S4] [directly_supported]
- The allowed documentation does **not** map `xx=01` to “control” and `xx=02..04` to specific
  perturbations. Selecting “member 01 because it is control” would therefore be unsupported.
  [unverified] The frozen candidate instead uses all four IDs symmetrically, so it needs no such
  mapping. [derived]

### 2.2 Basis-time arithmetic

For operating-block date `D`, the established basis is `D-1 14:00 KST = D-1 05:00 UTC`, and the
hourly valid block is `D-1 16:00 UTC` through `D 15:00 UTC`. Relative to initialization
`D-2 00:00 UTC`, those valid times are leads `+40` through `+63 h`. Component-wise linear
interpolation from a six-hour grid requires only the bounding leads
`{+36,+42,+48,+54,+60,+66 h}`. [derived]

At the basis, that initialization is already 29 hours old, and all required leads are far shorter than
the documented 00Z member horizons. [derived] This explains why `D-2 00Z` is the single
conservative cycle frozen here; it is not a claim that NOAA prescribes this cycle. [derived]

Crucially, “near real-time,” four cycles per day, and a long forecast horizon do **not** state when the
`+66 h` wind content becomes publicly available. None of the allowed pages gives a maximum NCEP
production time, NCEI ingest lag, or historical publication SLA. [unverified] Consequently
`D-2 00Z` is chronology-plausible but **not directly proved available by `D-1 05:00 UTC` on every
issue date**. [near_match_only] No later cycle, fallback cycle, or latest-available substitution is
authorized. [derived]

## 3. Wind-field and height audit

The current NCO page groups the relevant files as “CFS Zonal and Meridional Wind” and lists these
six-hourly choices: surface wind stress, `wnd10m`, and winds at 1000, 925, 850, 700, 250, and
200 hPa. [S3] [directly_supported]

| Documented family | Literal vertical meaning | Exact-treatment disposition |
|---|---|---|
| `wndstrs` | surface stress | Not a wind-velocity field; excluded. [directly_supported] |
| **`wnd10m`** | “10 Meters,” zonal and meridional wind | The only frozen field. [directly_supported] |
| `wnd1000`, `wnd925`, `wnd850`, `wnd700`, `wnd250`, `wnd200` | pressure surfaces | Pressure is not a fixed height above local ground or turbine hub; excluded from this one-field treatment. [derived] |

No 100 m or turbine-hub wind family appears in the allowed operational product landing page.
[directly_supported] The page literally supports a 10 m label, but does not spell out the vertical
datum in its landing text; proving the exact GRIB level descriptor would require the forbidden
inventory/object audit. [unverified] The candidate therefore uses the documented 10 m field **as a
raw 10 m predictor only**. It applies no power/log-law shear, roughness, stability correction, or hub-
height extrapolation. [derived] Calling it a documented hub-wind estimate would contradict the
available field inventory. [contradicts_premise]

The NCO page places the time-series zonal/meridional wind families in a nominal 0.5-degree product
section, while the NCEI overview separately distinguishes 0.5-degree latitude/longitude pressure
products from T126 Gaussian surface/flux products. [S1, S3] [directly_supported] Those pages do not
resolve the exact coordinate array, longitude convention, scan order, or whether the extracted
`wnd10m` time series inherits a regular regrid rather than the native surface grid. [unverified]
Therefore “0.5 degree” alone does not directly support a particular farm-to-grid cell or interpolation
operator. [near_match_only]

There is also a documentation-level cadence warning: the broad NCEI overview calls the generic
nine-month time-series output hourly, while the current variable-specific NCO page labels
`wnd10m` as `6hrly`; the forecast metadata record also describes all six-hourly forecasts. [S1, S2,
S3] [directly_supported] The zero-choice treatment follows the more specific `wnd10m` six-hourly
label, but only a later object decode could settle the exact time coordinate and instantaneous-versus-
averaged semantics. [derived]

## 4. The sole attempted zero-choice representation

**Frozen name:** `CFSV2_D2_00Z_M01TO04_VECTOR_MEAN_WND10M_BILINEAR_LINEAR6H_SPEED`.
[derived]

This is the only treatment considered; its specification has no member, field, cycle, spatial,
temporal, height, or fallback variant:

1. For every row in block `D`, select initialization `i = D-2 00:00 UTC` from the operational
   forecast collection, never analysis/reforecast and never a stitched/latest run. [derived]
2. Require exactly members `M={01,02,03,04}`. Missing any member invalidates the row; do not
   renormalize, substitute, or fall back. [derived]
3. Read only the zonal and meridional components represented by
   `wnd10m.xx.YYYYMMDDHH.daily.grb2`; use no pressure-level wind and no other CFS field.
   [derived]
4. For each component and member, use the six-hour valid times bracketing target UTC hour `t`.
   Let `h=(t-i)` in hours, `q0=6 floor(h/6)`, `q1=q0+6`, and
   `a=(h-q0)/6`. [derived]
5. At each bounding time, bilinearly interpolate the component in decoded native latitude/longitude
   coordinate space to the already-frozen BARAM site coordinate associated with the row. There is
   no nearest-neighbour fallback, extrapolation, or site-coordinate change. [derived]
6. Interpolate each component linearly in time, average the four members component-wise with
   fixed weight `1/4`, and take vector magnitude **after** those linear operations. With `B_s` the
   frozen spatial bilinear operator:

   `u(s,t) = (1/4) sum_m [(1-a) B_s(u[m,q0]) + a B_s(u[m,q1])]`,  
   `v(s,t) = (1/4) sum_m [(1-a) B_s(v[m,q0]) + a B_s(v[m,q1])]`,  
   `cfsv2_wnd10m_ms(s,t) = sqrt(u(s,t)^2 + v(s,t)^2)`. [derived]
7. Output that one scalar at 10 m. Apply no bias correction, calibration, member spread, direction,
   hub-height transform, spatial averaging across sites, clipping, tuning, or alternate feature.
   [derived]

The member mean, bilinear mapping, and component-wise linear time interpolation are deterministic
engineering choices; NOAA does not prescribe them on the cited pages. [unverified] Freezing them
removes degrees of freedom but does not convert them into provider-supported semantics. [derived]
The representation cannot be executed until the exact component names, level, grid coordinates,
time-step semantics, and availability are directly verified. [unverified]

## 5. Archive, access, licence, and deployment symmetry

| Gate | Official documentation result | Decision |
|---|---|---|
| Nominal archive | NCEI gives global coverage, an ongoing April-2011-to-present operational-forecast record, GRIB2, and lineage saying NCEI archives NCEP output. [S1, S2] | **PASS only at collection level.** [directly_supported] |
| Exact continuity | No page promises every issue/member/lead is present or undamaged. No catalogue or object was queried. | **UNVERIFIED.** [unverified] |
| Historical access | NCEI advertises direct-download and TDS access and says electronic downloads are generally free; the product page says NCEI serves historical near-real-time data. [S1, S2] | Access routes exist, but an explicit no-account/anonymous contract was not found. **UNVERIFIED.** [unverified] |
| Basis-time latency | Real-time output is described as near real time and separately served from a seven-day rotating NCEP archive. [S1, S4] | No maximum completion or ingest time. **UNVERIFIED.** [unverified] |
| Commercial licence | The dataset lineage identifies NOAA/NCEP production. NCEI policy says NOAA/federal environmental data are fully and openly available unless exempt and are in the U.S. public domain; the dataset lists citation/liability constraints, not a commercial prohibition. [S2, S5] | **PASS for commercial-use compatibility; cite NOAA/NCEI.** [derived] |
| Pre-cutoff publication | The forecast record, paper, metadata, and policy all predate 2026-07-05; the current NCO page is dated 2026-06-04. [S2–S5] | **PASS at documentation/product level.** [directly_supported] |
| Train/deployment symmetry | NCEI documents historical service while NCEP supplies a rotating real-time service. | Identity of schema, completeness, and availability timing across those routes is not attested. **UNVERIFIED.** [unverified] |
| Unchanged local transform | The candidate formula is fully specified without a fit or new dependency. | Actual GRIB compatibility and coordinate decode were not tested and therefore do not pass. **UNVERIFIED.** [unverified] |

Collection-level archive and licence evidence cannot substitute for issue-level chronology or
schema evidence. [derived]

## 6. Binding blockers and disposition

`READY` is denied by five independent blockers:

1. no official worst-case proof that the `D-2 00Z` run through the `+66 h` wind bound was public by
   the basis time; [unverified]
2. no all-required-date receipt for all four members and bounding times; [unverified]
3. no explicit anonymous historical-access statement and no proof that historical and real-time
   routes expose the same artifact contract; [unverified]
4. no allowed-page proof of the exact 10 m vertical descriptor and full U/V GRIB message/time
   semantics; [unverified]
5. nominal 0.5-degree prose does not define the actual coordinate array or endorse the frozen
   bilinear/linear interpolation. [unverified]

Thus the exact string in Section 4 is a **single frozen prerequisite specification**, not an executable
candidate. [near_match_only] No acquisition, fit, score-bearing comparison, or deployment action is
authorized. [derived]

A future reconsideration would require separate root authorization for one bounded metadata/object
receipt proving publication time, continuity, anonymous access, component/level/step metadata,
grid coordinates, and unchanged-environment decoding. Until every item passes, the verdict remains
`NOT_READY_PREREQUISITE_ONLY`. [derived]

## 7. Primary official URL ledger (7 unique; cap 10)

All were consulted as documentation/legal pages only on 2026-08-09 KST. No link from them to a
catalogue, THREDDS, data/API, inventory, or object endpoint was followed.

- **S1 — NCEI, Climate Forecast System product page:** operational/analysis/reforecast/reanalysis
  separation, cycles, cadence, grids, archive/access prose.  
  https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system
- **S2 — NCEI, CFSv2 Operational Forecasts metadata landing page:** forecast identity, period,
  format, coverage, lineage, access/use constraints.  
  https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00877
- **S3 — NCEP Central Operations, CFS product documentation:** initial-date/member convention,
  six-hourly `wnd10m`, candidate wind families, nominal time-series grid class.  
  https://www.nco.ncep.noaa.gov/pmb/products/cfs/
- **S4 — NOAA/NCEP, *The NCEP Climate Forecast System Version 2*:** independent operational versus
  retrospective status, Appendix C member/horizon configuration, near-real-time wording.  
  https://cfs.ncep.noaa.gov/cfsv2.info/CFSv2_paper.pdf
- **S5 — NCEI Open Data Policy:** federal/NOAA open-data and U.S.-public-domain rule.  
  https://www.ncei.noaa.gov/sites/default/files/2023-12/NCEI%20PD-10-2-02%20-%20Open%20Data%20Policy%20Signed.pdf
- **S6 — CPC wgrib2 `-fix_CFSv2_fcst` documentation:** read only to distinguish special monthly-mean
  metadata repair; not used to promote the six-hourly operational candidate. [near_match_only]  
  https://www.cpc.ncep.noaa.gov/products/tools/wgrib2/fix_CFSv2_fcst.html
- **S7 — NOAA, 2008 *Documentation of Operational NCEP CFS Data Files*:** pre-CFSv2 legacy CFS
  documentation; excluded from all positive CFSv2 claims. [near_match_only]  
  https://cfs.ncep.noaa.gov/cfs_data.pdf
