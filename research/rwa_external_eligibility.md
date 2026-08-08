# RWA-EXT — external forecast / open-weight eligibility audit

**Lane:** read-only research; the only repository write is this file  
**Retrieved:** 2026-08-08  
**Question:** which public forecast products or open pretrained weather weights can add information beyond the supplied deterministic GFS/LDAPS while respecting `D-1 14:00 KST`, no reanalysis input, no test-period observations, no remote API inference, and the 2026-07-05 weight-release/licence cutoff?  
**Collection performed:** only HTML/JSON/XML metadata, GRIB index files, object `HEAD`s, and one 24-hour Open-Meteo response; no GRIB/NetCDF/model weight was downloaded and no inference or fit was run.

## 0. Evidence convention and conclusion

Only three tags are used, with at most one on each claim:

- **[directly_supported]** — literal official full text or a machine response from the provider's archive/model repository.
- **[derived]** — arithmetic or eligibility/value judgement from directly supported facts.
- **[near_match_only]** — a name/snippet without the required archive/version/time/licence chain; no positive eligibility decision relies on this tag.

**Bottom line:** the highest-value newly cleared input is **ECMWF IFS from the `D-2 18Z` short cycle**, not the previously assumed `D-2 12Z` cycle: its required `+22…+45 h` fields were already in the official AWS mirror by 00:58 UTC in 2023 and 01:27 UTC in 2025, before the 05:00 UTC cutoff. [derived]

**Other legally eligible candidates are GEFS and two locally executed weights, NVIDIA FourCastNet v1 and Microsoft Aurora pinned to pre-cutoff files.** [derived]

**GEFS is not a fresh value probe: the existing root-owned 60-day receipt `reports/m270_gefs_probe.md` found spread-error Spearman correlations near zero and no monotone band-hit separation, so this lane does not recommend paying to repeat it.** [directly_supported]

**Open-Meteo's ordinary Historical Forecast series is not admissible because it explicitly stitches each run's first hours; NOAA's 2023 AIWP files are fixed-lead but retrospective, with a 2025 object timestamp.** [directly_supported]

## 1. Exact information set and lead arithmetic

Let `D` be the first KST date in an operating block, whose targets are `D 01:00 KST` through `D+1 00:00 KST`. The hard cutoff is `D-1 14:00 KST = D-1 05:00 UTC`, while target valid times are `D-1 16:00 UTC` through `D 15:00 UTC`. [derived]

| Candidate cycle | Required lead range | Machine-observed completion | Cutoff result | Tag |
|---|---:|---|---|---|
| GEFS `D-1 00Z`, `pgrb2b.0p50` | `+16…+39 h` (interpolate `f015…f039`) | `gep30 ... f039` was `04:03:00 UTC` on 2025-06-01; six spot dates in 2023–2025 were `04:00–04:05` | pass, about 55 min margin | [directly_supported] |
| ECMWF IFS `D-2 18Z`, `scda` | `+22…+45 h` (interpolate `21…45 h`) | deterministic `45h` was `00:58:31` in 2023 and `01:27:14` in 2025 on the following UTC day | pass, at least 3 h 32 min margin | [directly_supported] |
| ECMWF ENS `D-2 18Z`, `enfo` | `+22…+45 h` | `45h` was `01:47:31` in 2023 and `02:05:09` in 2025 | pass, about 2 h 55 min margin | [directly_supported] |
| ECMWF IFS `D-1 00Z` | `+16…+39 h` | `39h` was `08:34:07 UTC` on 2025-06-01 | **fail** | [directly_supported] |
| GFS `D-1 00Z f006/f012` as local-ML initial states | generated forecast states, not observations at valid time | `f006=03:37:20`, `f012=03:38:50 UTC` on 2025-06-01 | pass; use forecast states, not `f000:anl` | [directly_supported] |

The 2026 ECMWF dissemination page independently says 18Z short-cycle atmospheric delivery is `23:45→00:27 UTC`, while the historical S3 `Last-Modified` values above are the stricter evidence for what the public mirror actually held. [directly_supported]

## 2. Fixed-lead versus latest-run stitching audit

| Archive/path | What was tested | Fixed issue/run and lead preserved? | Eligibility consequence | Tag |
|---|---|---|---|---|
| NOAA GEFS S3 | key contains `gefs.YYYYMMDD/00/...f039`; `.idx` contains `d=YYYYMMDD00` and `39 hour fcst` | **yes** | no latest-run stitching | [directly_supported] |
| ECMWF AWS mirror | key contains `YYYYMMDD180000-45h-scda-fc`; `.index` contains `time=1800`, `step=45` | **yes** | no latest-run stitching | [directly_supported] |
| Open-Meteo Historical Forecast API | official page says “Each run's first few hours are stitched into a continuous hourly timeseries” and “seamless time series” | **no** | day-ahead use leaks later runs | [directly_supported] |
| Open-Meteo Previous Runs API | official page says fixed lead-time offsets of 1–7 days; a direct `jma_msm` query returned both day-1/day-2 series at 37.27 N, 128.95 E on 2023-06-01 | fixed offset **yes**, original run/version receipt **no** | day-2 is time-safe, but provenance/service gate remains | [directly_supported] |
| Open-Meteo Single Runs API | official page says original run structure is preserved, but non-HRES models start only 2026-04-02 and HRES starts 2024-03 | yes, but no 2023 development archive | coverage fail | [directly_supported] |
| NOAA AIWP `FOUR_v200_GFS` | file name encodes init and `f000_f240_06`; 2023-06-01 file has `Last-Modified=2025-01-29` | lead yes, historical availability **no** | retrospective training input fails literal basis-time rule | [directly_supported] |

For the 24-hour operating block, Open-Meteo `_previous_day1` is safe only through the 14:00 KST target; the final ten target hours were predicted less than 24 hours before a common 14:00 KST cutoff, whereas `_previous_day2` is safe for all 24 hours. [derived]

## 3. Forecast-data candidates

### 3.1 NOAA GEFS — legally eligible, but existing spread VOI is negative

- The public source is `s3://noaa-gefs-pds`; the first listed prefix is `gefs.20170101/`, and year-prefix enumeration returned 365/365/366/365 days for 2022/2023/2024/2025. [directly_supported]
- The fixed product is `atmos/pgrb2bp5/{gec00|gep01…gep30}.t00z.pgrb2b.0p50.fFFF`; `gep30` exists and `gep31` returns 404, giving one control plus 30 perturbed members. [directly_supported]
- The registry prose still describes an older 21-member configuration, so the current machine inventory—not that stale count—is the controlling version receipt. [directly_supported]
- The 2022, 2023, 2024, and 2025 `.idx` spot checks each contain `UGRD` and `VGRD:100 m above ground` at `f039`, with records about 0.272 MB each. [directly_supported]
- NOAA's NODD licence says the data are open to the public and “can be used as desired”, with attribution requested and no implied endorsement. [directly_supported]
- GEFS's ensemble spread is a new conditional-uncertainty channel, while its mean is likely highly redundant with the supplied GFS lineage. [derived]
- All 31 members × two 100 m components × nine three-hour steps cost about **152 MB/day**, **42 GB for 2023 Q2–Q4**, or **97 GB for development plus 2025 test** before local cropping. [derived]
- The existing 60-day internal receipt measured spread-error Spearman near zero and no monotone hit-rate ordering, so remaining value is below order `10^-3` absent a new mechanism; legal eligibility does not justify another collection. [derived]

### 3.2 ECMWF IFS 18Z — eligible and the best first probe

- The official cloud mirror is `s3://ecmwf-forecasts`; its first date prefix is `20230118/`, with 342 days in 2023, 366 in 2024, and 365 in 2025. [directly_supported]
- The 23 absent 2023 dates are 1–17 January plus 27 April–2 May; only the six-day April/May hole intersects the repository's Q2–Q4 development surface. [directly_supported]
- The usable historical paths are `.../18z/0p4-beta/scda/...-Nh-scda-fc` in 2023 and `.../18z/ifs/0p25/scda/...-Nh-scda-fc` in 2024–2025. [directly_supported]
- The 2023 0.4° index has 10 m and 925/1000 hPa `u/v` but no `100u/100v`; the 2025 0.25° index has all of them, so a leakage-safe validation must freeze the common 2023 definition rather than silently add 100 m only in test-era files. [directly_supported]
- ECMWF states that the open IFS/AIFS subset is CC-BY-4.0 and may be redistributed and used commercially with attribution. [directly_supported]
- Pulling only `10u,10v,u925,v925` for nine steps costs about **6.0 GB on 2023 Q2–Q4** and **16.6 GB for development plus 2025**, using byte ranges from `.index`. [derived]
- The value prior is `10^-3–10^-2` Total because IFS contributes a genuinely different forecast system, but it is six hours older and changes 0.4°→0.25° between development and test. [derived]

### 3.3 ECMWF ENS 18Z — legally eligible, economically second-line

- The `enfo` 2023 `45h` index contains 51 `10u` and 51 `10v` records; its 2025 counterpart contains 51 members for both 10 m and 100 m components. [directly_supported]
- Surface-wind records alone are about 62 MB/step in 2023 and 89 MB/step for 10 m in 2025, implying roughly **0.45 TB** for nine steps over 2023 Q2–Q4 plus 2025. [derived]
- Its independent spread could be more useful than a deterministic third mean, but the transfer/engineering cost is one to two orders above deterministic IFS. [derived]

## 4. Eligible open pretrained weights, local inference only

### 4.1 NVIDIA FourCastNet v1

- Pin **`nvidia/fourcastnet1@f63c56bd37c3fad04836422fd5a3a10329f95141`**; the official NVIDIA card gives release date 2023-10-25, model version v1, “ready for commercial/non-commercial use”, and Apache-2.0 terms. [directly_supported]
- The `fcn.mdlus` checkpoint is 301,168,640 bytes with LFS SHA-256 `995cdfdc3b64330caade5518aff09e0ce8f941b4262f3f8eb792b6fea8b6423a`, uploaded on 2026-03-02/05 and therefore before 2026-07-05. [directly_supported]
- The card lists 26 inputs and outputs, including native `u100m/v100m`, on a 0.25° global grid; runtime support is local PyTorch on Linux/NVIDIA Ampere, Hopper, or Turing. [directly_supported]
- NOAA GFS v16.3 `f012` has every required field in its `.idx`, was public at 03:38 UTC, and the 26 byte ranges total about 22.5 MB per daily initial state. [directly_supported]
- To obey “no test-period observations”, initialise from the already-issued **GFS forecast state `f012`**, not `f000:anl`; roll forward locally at six-hour cadence and interpolate only the resulting forecast fields. [derived]
- Order cost is 0.30 GB weights, about 15 GB of initial-state ranges for development plus test, and order 1–10 GPU-hours; expected incremental value is around `10^-3` Total because the propagator is new but the initial state is still GFS and output is six-hourly. [derived]

### 4.2 Microsoft Aurora

- Pin weights repository **`microsoft/aurora@d8753da77552eab0f27150c0efb91cb3acb425d5`** and code tag **`v1.7.0`**; both predate 2026-07-05. [directly_supported]
- The weight repository card is MIT, and the code's MIT text expressly permits use, modification, distribution, sublicensing, and sale. [directly_supported]
- The README's invitation to email about commercial applications is informational and does not add a condition to the MIT grant. [derived]
- The 0.25° fine-tuned checkpoint is 5,038,339,714 bytes, SHA-256 `25e429a3d615b4e2ef449fbb433eddaffbcc9c3678306585ef9567721bde6146`, first committed 2024-08-21; the small pretrained checkpoint is 451,339,106 bytes, SHA-256 `f80f78de1524a9faba8c9053e4a8ce6a2114ec01cff7f7b4efe9377200d50621`. [directly_supported]
- Do **not** use the Aurora 1.5 files added in HF commit `a96afd7...` on 2026-07-23, after the cutoff. [directly_supported]
- Aurora's documented batch uses two times of `2t,10u,10v,msl` plus pressure-level `z,u,v,t,q`; GFS `f006/f012` can provide forecast-state inputs before the cutoff without ERA5 or remote inference. [derived]
- Aurora adds model-form diversity but no documented native 100 m output in the pinned example; order cost is 0.45–5 GB of weights, tens of GB of inputs, and a GPU-class local rollout, so it ranks below FourCastNet for this wind task. [derived]

## 5. Source ledger — 12 official source packages maximum

All were retrieved 2026-08-08; machine URLs below are representative members of an endpoint family, not bulk downloads.

| ID | Official source package | Proof used |
|---|---|---|
| S01 | [NOAA GEFS, Registry of Open Data on AWS](https://registry.opendata.aws/noaa-gefs/) | NODD licence, update frequency, bucket identity |
| S02 | [NOAA GEFS S3 endpoint](https://noaa-gefs-pds.s3.amazonaws.com/?list-type=2&delimiter=%2F&max-keys=5) | archive prefixes, year counts, `pgrb2b.0p50` `.idx`, `Last-Modified`, member existence |
| S03 | [NOAA GFS official registry](https://registry.opendata.aws/noaa-gfs-bdp-pds/) and its linked `noaa-gfs-bdp-pds` endpoint | GFS v16.3, NODD licence, f006/f012 issue time and 26-field inventory |
| S04 | [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data) | CC-BY-4.0/commercial wording, IFS variables/steps, official cloud replication |
| S05 | [ECMWF dissemination schedule](https://confluence.ecmwf.int/display/DAC/Dissemination+schedule) | 00/06/12/18 delivery windows and stream names |
| S06 | [ECMWF official AWS mirror endpoint](https://ecmwf-forecasts.s3.amazonaws.com/?list-type=2&delimiter=%2F&max-keys=5) | archive dates, path/version transitions, `.index`, object times/sizes |
| S07 | [Open-Meteo Historical Forecast documentation](https://open-meteo.com/en/docs/historical-forecast-api) plus its linked Previous Runs endpoint | stitched-vs-fixed-vs-single-run definitions, start dates, one bounded JMA-MSM response |
| S08 | [Open-Meteo Terms](https://open-meteo.com/en/terms) | free API non-commercial; paid API commercial and returned data CC-BY-4.0 |
| S09 | [NVIDIA FourCastNet v1 model repository](https://huggingface.co/nvidia/fourcastnet1) | model card, release/version/licence, inputs/outputs/hardware, checkpoint commit/hash/size |
| S10 | [Microsoft Aurora official release package](https://huggingface.co/microsoft/aurora), including code tag `github.com/microsoft/aurora/tree/v1.7.0` | weight commit/hash/size/card licence; input contract; MIT text and release date |
| S11 | [Google DeepMind pre-cutoff GraphCast README at `c037656...`](https://github.com/google-deepmind/graphcast/blob/c0376564b94581c973cd085df7c515237b77cc6b/README.md) | model-weight CC-BY-NC-SA status as of the cutoff snapshot |
| S12 | [NOAA AIWP Registry](https://registry.opendata.aws/aiwp/) and linked `noaa-oar-mlwp-data` objects | reforecast status/licence/layout; 2023-vs-2025 object timestamps |

## 6. Final disposition tables

### Eligible

| Candidate | Exact eligible artifact/cycle | Why it clears legal/time/archive gates | Main independent channel | Tag |
|---|---|---|---|---|
| **ECMWF IFS deterministic** | `D-2 18Z`, 2023 `0p4-beta/scda`, 2024–25 `ifs/0p25/scda`, common `10u/v + u/v925` | fixed leads, public by cutoff, CC-BY-4.0 commercial, dev archive except six days | independent forecast-system error | [derived] |
| **NOAA GEFS** | `D-1 00Z pgrb2b.0p50`, all 31 members, 100 m `u/v` | fixed run/lead, complete 2022–25, public by cutoff, NODD open | ensemble spread rather than mean | [derived] |
| **ECMWF ENS** | `D-2 18Z enfo`, 51-member 10 m wind on common years | fixed run/lead, public by cutoff, CC-BY-4.0 commercial | independent ensemble spread | [derived] |
| **FourCastNet v1 weights** | `nvidia/fourcastnet1@f63c56b...`, local only, GFS `f012` forecast state | Apache-2.0, public before cutoff, no reanalysis input or remote inference | learned propagator + native 100 m wind | [derived] |
| **Aurora weights** | `microsoft/aurora@d8753da...` + code `v1.7.0`, local only, GFS `f006/f012` forecast states | MIT, public before cutoff, local inference | different learned propagator | [derived] |

### Ineligible

| Candidate/path | Binding failure | Tag |
|---|---|---|
| Open-Meteo **Historical Forecast** series / ordinary bulk stitched series | later runs' first hours are stitched by construction, so it is not a fixed day-ahead information set | [directly_supported] |
| Open-Meteo **Single Runs** for global non-HRES models | original runs are preserved, but archive begins 2026-04-02; HRES begins 2024-03, leaving no 2023 development surface | [directly_supported] |
| NOAA AIWP 2023 `FOUR_v200_GFS` archive as a ready-made training feature | 2023-06-01 reforecast object was created 2025-01-29, not by its historical cutoff | [directly_supported] |
| ECMWF `D-1 00Z` IFS/AIFS | public delivery is after 05:00 UTC; use `D-2 18Z` instead | [directly_supported] |
| GraphCast/GenCast weights as of 2026-07-05 | latest pre-cutoff official README licences model weights CC-BY-NC-SA-4.0; the commercial relicensing landed 2026-08-06, after cutoff | [directly_supported] |
| Reanalysis (ERA5/MERRA-2/etc.) as an inference input | expressly prohibited by the task even when data licences are permissive | [directly_supported] |

### Insufficient — do not promote without curing the named gap

| Candidate | What passes | Missing proof/blocker | Tag |
|---|---|---|---|
| Open-Meteo Previous Runs `jma_msm` / `wind_speed_10m_previous_day2` | fixed 48-hour offset and site/date coverage; paid API data is CC-BY-4.0 | response does not identify original init/run/model version; free endpoint is non-commercial, and paid-service authority is absent | [directly_supported] |
| GEFS ensemble **mean** as a third deterministic source | all legal/time gates | likely near-duplicate of supplied GFS, while the existing spread probe was negative; no remaining mechanism is established | [derived] |
| ECMWF 100 m wind as a uniform train/test feature | exists in 2024–25 | absent from the 2023 0.4° archive; using it only after the model-version transition breaks feature symmetry | [directly_supported] |
| Aurora 0.25° fine-tuned versus small-pretrained choice | both files and licences pass | no task-specific evidence that either retains useful Korean ridge-wind skill after GFS forecast-state substitution | [derived] |

## 7. One Grade-A value-of-information probe for the root

### Grade-A probe 1 — ECMWF 18Z deterministic innovation

Acquire **only 2023 Q2–Q4**, `D-2 18Z`, steps `21,24,…,45`, variables `10u,10v,u925,v925` by `.index` byte range (about 6 GB); do not touch 2024 and do not fetch 2025 until this gate passes. Align only on `forecast_kst_dtm` after the fixed `D-2 18Z` mapping, freeze the common 0.4° feature definition, and compare IFS hub-wind error with the supplied GFS/LDAPS errors before any score fit; any later score test must use fold-outside weights and the already frozen champion policy. Continue only if coverage is at least 97%, residual-error correlation is below 0.75, and a root-owned fold-outside source model improves fixed-policy Total by at least 0.002. [derived]
