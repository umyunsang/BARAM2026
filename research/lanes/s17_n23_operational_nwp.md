# S17-N23 — operational NWP archive frontier intake

## 0. Contract, evidence standard, and verdict

**Lane:** `S17-N23_POST_TERRAIN_FRONTIER_RESEARCH_INTAKE / OPERATIONAL_NWP`  
**Retrieval window:** 2026-08-08/09 KST; evidence ledger capped at 14 primary official documentation/metadata URLs.  
**Question:** does a new operational forecast archive covering South Korea pass, simultaneously, the
full archive, fixed-run chronology, D-1 14:00 KST availability, variables, commercial licence,
anonymous reproducibility, and unchanged Apple-M1 transform gates?

- `[directly_supported]` means literal provider documentation/metadata or a nonmutating local package
  observation.
- `[derived]` means arithmetic or a gate decision using only directly supported premises.
- `[near_match_only]` means that some structural gates pass but at least one required gate does not.
- `[contradicts_premise]` means the official source expressly defeats a required premise.
- `[unverified]` means this documentation-only lane could not establish the fact. An unverified gate
  can never support `READY`.

**Verdict: `NO_READY_OPERATIONAL_NWP_ARCHIVE`.** [derived]

The two closest official candidates are operational NOAA GEFS and CFSv2. Neither is `READY` under
this lane's strict evidence rule. GEFS has nominal 2017-present NODD coverage, a fixed-cycle forecast
product, a commercially usable no-sign bucket, and global coverage; however NOAA expressly says the
NODD copy is **not officially archived**, and this no-endpoint lane did not verify the required
2022-2023 00Z objects, literal wind records, or completion before the basis time. [near_match_only]
CFSv2 has a true NCEI operational-forecast record nominally from 2011-present and fixed init/member
filenames, but no cited official publication-latency bound, all-date continuity receipt, explicit
anonymous historical-access contract, or actual-file decode receipt. [near_match_only]

No acquisition, feature formula, fit, or executable candidate is authorized from this lane. [derived]

## 1. Chronology gate

The fixed basis is `D-1 14:00 KST = D-1 05:00 UTC`. The operating block spans valid times after that
basis, so an admissible product must preserve an **issue cycle and forecast lead**, rather than expose
a continuous series made from whichever run was latest at each valid hour. [derived]

For a nominal `D-1 00Z` run, the relevant GEFS block needs approximately `+16...+39 h`; a three-hour
product therefore needs bounding forecasts through `f039` to have landed by 05:00 UTC. An official
statement that a model is produced four times daily proves its cycles, but does **not** prove that
`f039` was publicly complete by 05:00 on every required issue date. [derived]

For CFSv2, a deliberately older `D-2 00Z` issue would need approximately `+40...+63 h`, bounded by
six-hour outputs. That removes same-morning pressure but still does not turn the phrase “near-real
time” into a guaranteed maximum archive-ingest delay. [derived]

This lane queried no forecast bucket, catalogue, THREDDS object, `.idx`, API/data endpoint, or
`Last-Modified` header. Consequently exact object existence, lead metadata, and public completion
latency are marked unverified wherever the human-readable official documentation does not state
them. [directly_supported]

## 2. Gate matrix

| Candidate | Official archive/coverage evidence | Fixed issue and lead | Wind / Korea | Commercial + anonymous access | Unchanged M1 transform | Decision |
|---|---|---|---|---|---|---|
| **NOAA operational GEFS NODD** | Nominal 2017-present and global, but NOAA says NODD is “not officially archived” | Four cycles/day; forecast table is 3-hourly `+000...+384`; exact 2022-2023 00Z keys and completion latency not checked | `pgrb2a/pgrb2b` named; literal required wind records not documented here | NODD “can be used as desired”; bucket supports `--no-sign-request` | Regular-grid GRIB2 path is locally plausible; no body decode | **NEAR MATCH** |
| **NOAA CFSv2 operational forecasts** | NCEI says 2011-present, global 0.5°, GRIB2 | Four init cycles/day; six-hour forecasts; filenames preserve init and member | Official inventory names 10 m and pressure-level winds | Federal NOAA data are public domain; historical HTTPS/TDS is linked, but an explicit no-account historical contract was not found | Existing `cfgrib/eccodes`; actual file compatibility untested | **NEAR MATCH** |
| **ECCC GDPS** | Global 15 km operational product, but free Datamart retains only 30 days and ECCC says archived NWP retrieval is not online | Twice daily / ten-day forecast; exact public latency not established | Global and roughly 30 vertical levels | Commercial licence passes; historical retrieval is manual cost recovery | GRIB2 is locally plausible; no body decode | **REJECT** |
| **DWD ICON Global** | Operational since 2015, global 13 km; Pamore exposes only roughly the last 1.5 years of forecasts | 00/06/12/18Z with documented 120/180 h horizons | Global; offered as weather-element GRIB2 packages, but required literal wind inventory not frozen | Current Open Data is no-registration CC BY 4.0; archive registration is restricted to named institutional classes | DWD prescribes CDO for triangular-to-lat/lon interpolation; `cdo` is absent locally | **REJECT** |
| **KMA GDAPS/KIM** | Portal claims global holdings from May 2011-present, with resolution varying by period | 00/06/12/18Z and documented forecast horizons; public delivery latency not stated | Global/Korea; pressure- and single-level elements | Every relevant download/query route is labelled “API utilization application”; anonymous and commercial terms not established | GRIB/NC is structurally plausible; exact historical schema/decode untested | **REJECT** |
| **Met Office Global deterministic 10 km / ASDI** | Official sheet says only two years of historical data; that does not establish full required 2022-2023 retention at the cutoff | Full 00/12Z and short 06/18Z runs; stated four-hour delay; fixed lead cadence | Global; 10 m speed/direction and pressure-level winds documented | The cited sheet does not state commercial licence or anonymous acquisition | NetCDF encoding is unspecified and the project lacks `netCDF4`, `h5netcdf`, and `h5py` | **NEAR MATCH / UNVERIFIED** |

## 3. Candidate audits

### 3.1 NOAA NODD operational GEFS — operational, not reforecast/reanalysis, but not archive-safe

The NCEI product page explicitly separates **GEFS Forecasts** from **GEFS Analysis**. Its forecast
row gives a period of record `1/1/2017–Present`, cycles `00, 06, 12, 18UTC`, and three-hourly leads
`+000 to +384`; the AWS registry separately describes global coverage and production four times per
day. These are operational issued forecasts, not a retrospective reforecast/reanalysis collection
and not a stitched latest-valid-time service. [directly_supported]

The same NCEI page also states, literally, “Data on the NODD is not officially archived.” Nominal
period-of-record prose therefore cannot prove complete preservation of every required issue. This is
a direct contradiction of the strict all-row archive guarantee, not a minor documentation omission.
[contradicts_premise]

The AWS registry directly establishes the bucket identity `noaa-gefs-pds`, no-account
`--no-sign-request` access, and that NODD data “can be used as desired,” with attribution requested.
Commercial and reproducible anonymous transport pass at the service level. [directly_supported]

What remains unverified because endpoint/object queries were forbidden:

1. every required **2022-2023** date has the fixed `00Z` object family; [unverified]
2. each required object preserves the literal initialization and forecast step rather than only the
   high-level product convention; [unverified]
3. the chosen historical parameter stream contains the exact required wind components/heights and a
   stable member/schema definition; [unverified]
4. the last bounding lead was public by `D-1 05:00 UTC` on every issue date. “Four times a day” is not
   a delivery SLA. [unverified]

Thus GEFS is not `READY` in this intake. This finding does not relabel the operational bucket as a
reforecast; it says only that official prose is insufficient for the strict object-completeness and
basis-latency contract. [derived]

### 3.2 NOAA CFSv2 operational forecasts — the strongest long-record near match

NCEI explicitly distinguishes `CFSv2 Operational Forecasts` from operational analysis, CFS
reforecasts, and CFS reanalysis. The operational forecast period is `April 1, 2011—Present`, format
is GRIB2, coverage is global at roughly 0.5°, cycles are `00/06/12/18UTC`, and six-hour forecast
products extend for months. [directly_supported]

NCEP's official product inventory says `YYYYMMDDHH` is the **initial date** and `xx` is member
`01...04`, and lists 10 m wind plus 1000, 925, 850, 700, 250, and 200 hPa wind products. Those
filenames are fixed issue/member products, not a latest-run stitched time series. [directly_supported]

NOAA's NCEI Open Data Policy says federal environmental data are fully and openly available unless
explicitly exempt and are in the US public domain. Commercial-use compatibility therefore passes for
this NOAA-produced product. [directly_supported]

Four strict gaps prevent `READY`: the pages give no worst-case public completion/ingest time for a
chosen old issue, do not attest day-by-day completeness over all required rows, do not expressly say
that the historical TDS/HTTPS route is no-account, and do not prove that the literal `wnd10m` files
have a stable component/coordinate contract decodable by the unchanged project environment.
[unverified]

### 3.3 ECCC GDPS — legal and global, no online history

ECCC documents GDPS as global at 15 km, with forecasts twice daily to ten days and fields on roughly
30 vertical levels. Its end-use licence is worldwide, royalty-free, perpetual, non-exclusive, and
expressly includes commercial use. [directly_supported]

The free HTTPS Datamart and alternative HPFX server each retain only 30 days. More decisively, the
official FAQ answers “Are historical data forecasts ... available?” with “does not have an online
service”; retrieval is a manual cost-recovery service. This fails reproducible anonymous historical
acquisition and full archive coverage. [contradicts_premise]

### 3.4 DWD ICON Global — operational mechanics pass, historical access and local transform fail

DWD documents ICON Global as operational since 25 January 2015, global at 13 km, output in GRIB2,
run at 00/06/12/18 UTC, hourly through +78 h, and available in free parameter packages. DWD's Open
Data FAQ says registration is unnecessary and reuse is CC BY 4.0. [directly_supported]

Archived forecasts are instead served by Pamore: only the most recent roughly 1.5 years are online,
and registration is offered to research/education institutions, German authorities, and disaster
prevention institutions. This cannot reproduce all required older assessment issues anonymously.
[contradicts_premise]

DWD further says the triangular ICON grid can be converted to regular latitude/longitude with CDO.
The unchanged project host has no `cdo` executable, so even a hypothetical archive receipt would
require a forbidden dependency/tool mutation. [contradicts_premise]

### 3.5 KMA GDAPS/KIM — highly relevant coverage, authenticated acquisition only

KMA's API Hub documents global GDAPS/KIM holdings from May 2011-present, production at
00/06/12/18 UTC, long forecast horizons, and pressure/single-level elements. It is geographically the
closest new operational system in this matrix. [directly_supported]

However, every relevant route is presented as an **API utilization application** (`활용신청`), not
an anonymous immutable archive. The same page warns that queryable periods may differ under the
retention policy and that resolution varies over time. Exact fixed historical objects, public
completion latency, commercial-use terms for this delivery, and anonymous reproduction are not
established. [unverified]

### 3.6 Met Office Global deterministic 10 km — timing near match, retention/licence/runtime gaps

The official datasheet (PDF metadata dated 2024-07-16, before the competition cutoff) describes an
operational global 10 km deterministic model, full 00/12Z and short 06/18Z runs, fixed hourly/3-hour/
6-hour lead cadence, a four-hour update delay, NetCDF output, 10 m wind speed/direction, and
pressure-level winds. A nominal 00Z run at +4 h would precede the 05Z basis. [directly_supported]

But the same sheet states only “2 years' worth of historical data”; it does not establish retention
of every required 2022-2023 issue at the cutoff or today. It also contains no commercial licence or
anonymous acquisition contract. Finally, exact NetCDF encoding is unspecified, while the unchanged
project lacks the common NetCDF4/HDF5 Python backends. This is not `READY`. [near_match_only]

## 4. Apple-M1 transform audit

A nonmutating check of the project interpreter observed `Darwin arm64`, `xarray 2026.7.0`,
`cfgrib 0.9.15.1`, `eccodes 2.47.0`, `numpy 2.5.1`, `pandas 2.3.3`, and `scipy 1.18.0`.
`netCDF4`, `h5netcdf`, `h5py`, and the `cdo` executable are absent. [directly_supported]

- A regular latitude/longitude GRIB2 source such as nominal CFSv2/GEFS could, in principle, be opened
  with the already installed Python ecCodes/cfgrib stack, cropped to Korea, converted to u/v or speed,
  and interpolated in time without a new dependency. Because no data body was opened, actual message
  compatibility and memory footprint remain unverified. [near_match_only]
- ICON's official triangular-grid conversion explicitly calls for CDO, which is absent. [contradicts_premise]
- The Met Office file encoding/backend is not specified by the sheet; `xarray+scipy` would cover only
  compatible NetCDF variants, so local decoding cannot be certified without a body or stronger format
  metadata. [unverified]

Local plausibility is not a transform PASS. A `READY` decision requires an exact frozen field list,
coordinate/crop rule, issue/lead mapping, temporal interpolation rule, and one no-install decode
receipt; none was established here. [derived]

## 5. Exclusions and decision

- Reanalysis, operational analysis, observations, retrospectively generated reforecasts, and any
  stitched latest-run service were excluded rather than treated as forecast archives. [directly_supported]
- The already partial/rejected ECMWF N5 lineage was not reopened. [directly_supported]
- Unsupported pretrained weather-model inference and weight acquisition were not reopened. [directly_supported]
- No forecast/data/weight body, API/data endpoint, remote inference, fit, prediction, scorer, label,
  actual, 2024/test row/value body, dependency, account, or external action was touched. [directly_supported]

**Selection result: `NONE`; zero executable candidates.** [derived]

A future GEFS reconsideration would first require a separately authorized, frozen metadata audit of
fixed 00Z 2022-2023 object existence, literal parameter inventory, and publication timestamps. A
future CFSv2 reconsideration would require an official latency bound or equivalent frozen object
receipt plus one unchanged-environment decode. Those are prerequisites, not approvals to query or
acquire anything in this lane. [derived]

## 6. Primary official source ledger (14 URLs)

All links below were read as provider documentation/metadata; no linked data endpoint was queried.

1. NOAA NCEI, **Global Ensemble Forecast System** — archive status, cycles, lead cadence, parameter sets:  
   https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast [directly_supported]
2. NOAA/AWS Registry, **NOAA GEFS** — global service, licence, bucket, no-sign access:  
   https://registry.opendata.aws/noaa-gefs/ [directly_supported]
3. NOAA NCEI, **Climate Forecast System** — operational-forecast period, coverage, cycles, formats:  
   https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system [directly_supported]
4. NOAA NCEP, **CFS products** — fixed init/member naming and wind inventory:  
   https://www.nco.ncep.noaa.gov/pmb/products/cfs/ [directly_supported]
5. NOAA NCEI, **Open Data Policy** — federal-data public-domain/open policy:  
   https://www.ncei.noaa.gov/sites/default/files/2023-12/NCEI%20PD-10-2-02%20-%20Open%20Data%20Policy%20Signed.pdf [directly_supported]
6. ECCC, **GDPS data and products** — global domain, resolution, cadence, fields, access routes:  
   https://eccc-msc.github.io/open-data/msc-data/nwp_gdps/readme_gdps_en/ [directly_supported]
7. ECCC, **MSC Open Data FAQ** — no online historical NWP service / cost recovery:  
   https://eccc-msc.github.io/open-data/faq/readme_en/ [directly_supported]
8. ECCC, **MSC Datamart** — anonymous HTTPS mechanics and 30-day retention:  
   https://eccc-msc.github.io/open-data/msc-datamart/readme_en/ [directly_supported]
9. ECCC, **Data Servers End-use Licence** — explicit commercial permission:  
   https://eccc-msc.github.io/open-data/licence/readme_en/ [directly_supported]
10. DWD, **NWP forecast data** — ICON global operation, grid, cycles/leads, GRIB2/CDO:  
    https://www.dwd.de/EN/ourservices/nwp_forecast_data/nwp_forecast_data.html [directly_supported]
11. DWD, **Open Data FAQ** — no-registration current access and CC BY 4.0:  
    https://www.dwd.de/DE/leistungen/opendata/faqs_opendata.html [directly_supported]
12. DWD, **Pamore archived forecast model data** — archive span and registration classes:  
    https://www.dwd.de/EN/ourservices/pamore/pamore.html [directly_supported]
13. KMA API Hub, **Numerical models** — GDAPS/KIM period, domain, cycles, leads, application routes:  
    https://apihub.kma.go.kr/apiList.do?seqApi=9 [directly_supported]
14. Met Office, **Global NWP ASDI datasheet** — global model, two-year history, latency, cycles, NetCDF and variables:  
    https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/data/global-nwp-asdi-datasheet.pdf [directly_supported]
