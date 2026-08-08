# S17-N23 site-mapped terrain evidence intake

- **Node:** `S17-N23_POST_TERRAIN_FRONTIER_RESEARCH_INTAKE / SITE_MAPPED_TERRAIN`
- **Completed:** 2026-08-09 KST
- **Lane mode:** read-only research; this designated file is the only repository write
- **Decision:** **`0` executable candidates selected**

## Verdict

**No exactly-one, zero-variant, group-specific/site-mapped Sx scalar reaches `[directly_supported]`.**

The narrower literature audit is nevertheless decisive:

1. `[directly_supported]` Winstral's Sx is evaluated **at the target/site cell** (a station's DEM cell, or every fine output raster cell), using terrain radiating from that cell. It is not an average of Sx evaluated at unrelated coarse-grid locations.
2. `[contradicts_premise]` N21/N22's refuted feature is `mean16`: it evaluates Sx at all 16 LDAPS coordinates with each coordinate's own wind direction, then takes an unweighted mean and gives the same scalar to all three KPX groups. No reviewed primary source supports that operation as the exposure of a target site.
3. `[derived]` The official archive supplies 17 turbine coordinates, KPX membership, rated capacities, and 117 m hub height. It does **not** supply a single coordinate per KPX group or a rule for reducing turbine-level Sx to one group scalar.
4. `[near_match_only]` Nearest-LDAPS lookup, group-centroid lookup, inverse-distance interpolation, or capacity-weighted averaging would each add an unprescribed spatial/reduction rule. The 2017 paper's nearest-neighbor choice maps the **coarse forecast field** to a station; the paper still computes Sx at the station itself.
5. `[unverified]` Exact-turbine Sx could be a distinct terrain primitive, but the frozen N20 lookup contains only `16 x 72` values at LDAPS coordinates, not the 17 turbine coordinates. More importantly, no direct primary evidence fixes one group reduction and its weights. Therefore no exact candidate formula or weights are emitted and no execution handoff is supported.

This means N22 refutes **`terrain__sx300_h8_mean16` only**. It does not establish that target-local terrain exposure is physically false; it also does not supply the missing evidence needed to authorize a one-feature group representation.

## What the primary papers actually do

### Source-supported point primitive (not an authorized BARAM group feature)

For a target cell/site `q`, direction `A`, target height `h`, and search distance `dmax`, the reviewed papers support the point-local construction

\[
S_x(q,A;h,d_{max})=
\frac{1}{7}\sum_{\delta\in\{-15,-10,-5,0,5,10,15\}}
\max_{p\in R(q,A+\delta,d_{max})}
\operatorname{atan2}\!\left(z(p)-[z(q)+h],\ d(q,p)\right),
\]

reported in degrees. The seven-ray form is the N19-frozen realization of the published 30-degree window at 5-degree increments. Positive values denote shelter and negative values exposure.

Evidence classification:

- `[directly_supported]` Winstral, Elder & Davis (2002), section 3a, defines maximum upwind slope from **"the cell of interest"**, states the objective is shelter/exposure **"upwind of each pixel"**, and averages directional searches only within the upwind angular window.
- `[directly_supported]` Winstral, Jonas & Helbig (2017), section 3 and Eq. (1), derives high-resolution topography **at each station**, explicitly gives the example **"grid cell containing each station"**, and section 4d derives Sx **at each 25 m grid cell** for a spatial product.
- `[directly_supported]` The 2017 paper assigns the coarse COSMO forecast speed/direction to each target cell/station and demonstrates use of forecast wind direction.
- `[near_match_only]` The paper's `h=8 m, dmax=300 m` experiment concerns 6.5/10 m station winds and snow-oriented downscaling. All official BARAM turbines have 117 m hub height. Keeping `h=8` is the zero-variant N19 lineage, but its turbine-site applicability is not direct. Replacing it with `h=117` would be a new, unvalidated parameter choice.
- `[near_match_only]` The papers do not address wind-farm power aggregation, KPX settlement groups, or a scalar reduction across multiple turbines.

### Why the 2017 nearest-neighbor passage does not fix our weights

The paper compares nearest-gridcell COSMO forecasts with inverse-squared interpolation to station coordinates, finds no universal advantage, and adopts nearest gridcell for its snow/high-wind objective. That passage selects a **forecast-field transfer**. In the same method, local Sx remains computed from the high-resolution DEM at the station coordinate. It therefore does not support substituting an LDAPS-grid Sx for turbine-site Sx, nor averaging several such substitutes into a KPX-group scalar.

## Frozen repository geometry

### The 16 N19/N20 Sx query coordinates

These are the exact LDAPS coordinates in frozen `research/scratch/grid_coords.json` and the N19 aggregate audit. They are the only query locations in the N20 `16 x 72` lookup.

| LDAPS grid | latitude | longitude |
|---:|---:|---:|
| 01 | 37.3032 | 128.9443 |
| 02 | 37.3027 | 128.9617 |
| 03 | 37.3022 | 128.9790 |
| 04 | 37.2899 | 128.9263 |
| 05 | 37.2894 | 128.9437 |
| 06 | 37.2888 | 128.9610 |
| 07 | 37.2883 | 128.9784 |
| 08 | 37.2878 | 128.9958 |
| 09 | 37.2760 | 128.9257 |
| 10 | 37.2755 | 128.9430 |
| 11 | 37.2750 | 128.9604 |
| 12 | 37.2745 | 128.9778 |
| 13 | 37.2740 | 128.9951 |
| 14 | 37.2617 | 128.9424 |
| 15 | 37.2612 | 128.9598 |
| 16 | 37.2607 | 128.9771 |

### Official turbine/site coordinates

Source: the immutable competition archive `info.xlsx`, sheet title **태백가덕산풍력발전 정보**. Archive SHA-256 is `920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b`; the metadata member SHA-256 is `89e83a52e0eb2ce367a3573a96d6795ed4b4d4ac624965cb3530beec0cbd2bd6`. XLSX core metadata identifies creator `dacon`, created `2026-06-30T11:13:20Z` and modified `2026-06-30T11:14:02Z`, before the 2026-07-05 cutoff. KPX membership is fixed by merged ranges `H5:H10`, `H11:H16`, and `H17:H21`.

Decimal coordinates below are exact DMS-to-decimal conversions; the DMS strings, capacity, and hub height are direct archive values.

| KPX group | stage-turbine | official coordinate | latitude | longitude | MW | hub m |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1-1 | 37°16'55.61"N 128°57'02.10"E | 37.28211389 | 128.95058333 | 3.6 | 117 |
| 1 | 1-2 | 37°17'04.05"N 128°56'58.35"E | 37.28445833 | 128.94954167 | 3.6 | 117 |
| 1 | 1-3 | 37°17'11.49"N 128°56'58.99"E | 37.28652500 | 128.94971944 | 3.6 | 117 |
| 1 | 1-4 | 37°17'23.11"N 128°57'03.68"E | 37.28975278 | 128.95102222 | 3.6 | 117 |
| 1 | 1-5 | 37°17'28.20"N 128°57'15.58"E | 37.29116667 | 128.95432778 | 3.6 | 117 |
| 1 | 1-6 | 37°17'19.48"N 128°57'24.96"E | 37.28874444 | 128.95693333 | 3.6 | 117 |
| 2 | 1-7 | 37°17'16.20"N 128°57'34.67"E | 37.28783333 | 128.95963056 | 3.6 | 117 |
| 2 | 1-8 | 37°17'11.29"N 128°57'47.24"E | 37.28646944 | 128.96312222 | 3.6 | 117 |
| 2 | 1-9 | 37°17'00.97"N 128°57'57.44"E | 37.28360278 | 128.96595556 | 3.6 | 117 |
| 2 | 1-10 | 37°16'52.77"N 128°58'04.18"E | 37.28132500 | 128.96782778 | 3.6 | 117 |
| 2 | 1-11 | 37°16'44.89"N 128°58'01.12"E | 37.27913611 | 128.96697778 | 3.6 | 117 |
| 2 | 1-12 | 37°16'30.58"N 128°58'02.54"E | 37.27516111 | 128.96737222 | 3.6 | 117 |
| 3 | 2-1 | 37°16'59.73"N 128°57'44.97"E | 37.28325833 | 128.96249167 | 4.2 | 117 |
| 3 | 2-2 | 37°16'40.41"N 128°58'13.80"E | 37.27789167 | 128.97050000 | 4.2 | 117 |
| 3 | 2-3 | 37°16'28.03"N 128°58'22.54"E | 37.27445278 | 128.97292778 | 4.2 | 117 |
| 3 | 2-4 | 37°16'18.58"N 128°58'29.01"E | 37.27182778 | 128.97472500 | 4.2 | 117 |
| 3 | 2-5 | 37°16'06.83"N 128°58'35.68"E | 37.26856389 | 128.97657778 | 4.2 | 117 |

`[directly_supported]` The metadata establishes 6/6/5 turbine members and group capacities 21.6/21.6/21.0 MW. **It does not establish Sx aggregation weights.** Although capacity normalization would be an obvious derivation because capacities are equal within each group, no reviewed Sx source prescribes that reduction; it is intentionally not frozen here.

`[derived]` Diagnostic-only nearest-point geometry (no feature values read) maps the turbines to LDAPS IDs `[10,5,5,5,6,6]`, `[6,6,6,11,11,11]`, and `[6,12,12,12,12]` for groups 1–3, respectively; nearest distances span about 162–996 m. No turbine coordinate equals an N19/N20 lookup coordinate. This demonstrates why “choose one grid,” “average mapped grids,” and “use a centroid” are substantive mapping choices, not clerical row alignment. These IDs are **not** a candidate or authorized weights.

## N19–N22 artifact audit

| Stage | Aggregate evidence | Classification |
|---|---|---|
| N19 | Frozen primitive uses 16 LDAPS query locations, `dmax=300 m`, `h=8 m`, 72 five-degree bins, seven rays, and no variants. Its own boundary says BARAM gain is near-match/unverified. | `[directly_supported]` implementation/formula lineage; `[near_match_only]` target applicability |
| N20/N20A | Frozen Copernicus GLO-30 lookup has shape `16 x 72`, grid IDs 1–16, all 1,152 values finite, exact NPY/Parquet parity, and no turbine-site rows. Raster body was not opened in this lane. | `[directly_supported]` lookup identity; `[contradicts_premise]` that it is already site-mapped |
| N21 | Freezes exactly one new feature: for each timestamp, look up Sx separately at all 16 LDAPS grids from each grid's `u10/v10`, then take the arithmetic mean of 16 values. | `[directly_supported]` description of what was run; `[derived]` mean16 representation |
| N22/N22A | Exact strict control reproduction succeeded. Saved aggregate evaluation gives mean16 delta `+0.0003633104`, below the predeclared `+0.001635` margin; one outer-fold delta is negative, promotion is false, and gap-closing credit is zero. | `[directly_supported]` refutation of mean16 only |

The N22 feature is exactly

\[
F_{mean16}(t)=\frac{1}{16}\sum_{j=1}^{16}
S_x(q^{LDAPS}_j,A_j(t);8\text{ m},300\text{ m}),
\]

and contains no `group_id`. A target-local primitive would instead change `q` to a turbine/site coordinate and would not be algebraically the same representation. However, turning the resulting 6/6/5 point values into **one** group scalar still needs an unsupported reducer.

## Candidate gate

| Required gate | Result | Reason |
|---|---|---|
| Literal primary support for local/site Sx | PASS | 2002 per-pixel and 2017 per-station/per-output-cell text |
| Exact one-feature KPX-group formula | **FAIL** | no source specifies group coordinate or multi-turbine reducer |
| Exact weights | **FAIL** | nearest, centroid, inverse-distance, equal, or capacity weights are all derived choices |
| Zero parameter variants | **FAIL** | N19 `h=8` is not direct for 117 m hub targets; changing height creates a new choice |
| Executable from frozen lookup alone | **FAIL** | lookup covers 16 LDAPS coordinates, not 17 turbine coordinates |
| Novel versus refuted mean16 | PASS in principle | target-local `q` is structurally different, but no admissible scalar is fully specified |
| Basis chronology/static availability | PASS for primitive | DEM is static/pre-cutoff and issued LDAPS direction is the intended basis-safe source |
| Commercial-compatible DEM asset | PASS | official GLO-30-F licence grants general-public, worldwide, no-time-limit reproduction/distribution/communication/adaptation rights, subject to notices |

**Selection result: `0` candidates.** A pointwise target mechanism is publication-aligned, but publication alignment does not justify inventing the missing KPX-group operator. No fit, comparison, or data acquisition is proposed.

## Primary-source register (6 unique URLs)

All were accessed 2026-08-09 KST. “Licence” describes reuse status; papers were consulted only as citations and no paper text/code/data is incorporated into a model asset.

| URL | Title / date | Licence/status | Applicability |
|---|---|---|---|
| https://www.dora.lib4ri.ch/wsl/dload/wsl%3A12719/PDF/Winstral-2017-Statistical_downscaling_of_gridded_wind-(published_version).pdf | *Statistical Downscaling of Gridded Wind Speed Data Using Local Topography*, Winstral, Jonas & Helbig, J. Hydrometeorology 18, 2017, DOI 10.1175/JHM-D-16-0054.1 | © 2017 American Meteorological Society; citation/evidence use only | `[directly_supported]` Eq. 1, station-cell Sx, each-25 m-cell spatial use, forecast direction; `[near_match_only]` BARAM power/group gain |
| https://journals.ametsoc.org/view/journals/hydr/3/5/1525-7541_2002_003_0524_ssmowr_2_0_co_2.xml | *Spatial Snow Modeling of Wind-Redistributed Snow Using Terrain-Based Parameters*, Winstral, Elder & Davis, 2002 | AMS publisher page/copyright; citation/evidence use only | `[directly_supported]` cell-of-interest/per-pixel Sx and within-window ray mean; no cross-site/group mean |
| https://centaur.reading.ac.uk/20812/ | *An efficient method for distributing wind speeds over heterogeneous terrain*, Winstral, Marks & Gurney, 2009, DOI 10.1002/hyp.7141 | University bibliographic record; states full text is not archived | `[unverified]` metadata confirms publication only; cannot support an exact group mapping or weights |
| https://registry.opendata.aws/copernicus-dem/ | *Copernicus Digital Elevation Model (DEM)*, Copernicus DEM 2021 release | Registry points GLO-30 Public to the official free general-public licence | `[directly_supported]` identifies `s3://copernicus-dem-30m` as GLO-30 Public COG, matching frozen N20 object lineage |
| https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf | *Copernicus Contributing Mission Data Access / Licence for COP-DEM-GLO-30-F*, issue 2025-02-21 | Full, Free & Open; worldwide/no time limit; reproduction, distribution, public communication, adaptation/modification/combination; attribution/notices required | `[directly_supported]` external DEM eligibility and commercial-compatible field of use; no support for a terrain feature formula |
| https://documentation.dataspace.copernicus.eu/FAQ.html | *Copernicus Data Space Ecosystem FAQ*, live official documentation | Refers users to official terms/licences | `[near_match_only]` corroborative navigation only; not used for the licence decision |

## Access and prohibition accounting

- **Repository input file bodies inspected:** 21 (cap 40). Directory names were enumerated only for discovery; no additional bodies were opened.
- **Repository writes:** exactly 1, this designated file. No temp/cache/receipt/script/config write.
- **External source URLs fetched/read:** 6 unique (cap 8). Search-result snippets were discovery only and are not evidence.
- **External requests for data, raster, NWP, weights, inference, or APIs:** 0. Publication/licence/document pages only.
- **External data/weight bodies acquired:** 0. The existing local Copernicus TIFF body was not opened; frozen lookup values were not opened, only N20/N21 aggregate metadata.
- **Competition archive access:** SHA-256 plus only `info.xlsx`; within that member only worksheet/shared-string/core metadata XML. No train, label, SCADA, forecast, test, or 2024 member body.
- **`actual_kwh` or label values:** 0.
- **2024/test values:** 0.
- **Model fits / predict calls / policy calls / metric or score calls:** 0 / 0 / 0 / 0.
- **Dependencies installed or changed:** 0.
- **Dacon/account/browser/submission actions:** 0.

Local input inventory:

```text
reports/s17_n23_post_terrain_frontier_intake_predeclaration.json
research/scratch/grid_coords.json
src/baram/constants.py
artifacts/audits/s17_n19_terrain_sx300_no_body.json
reports/s17_n19_terrain_sx300_formula_manifest.json
reports/s17_n19_terrain_sx300_lineage_input_manifest.json
reports/s17_n19_terrain_sx300_no_body_receipt.json
reports/s17_n20_terrain_sx300_static_extraction_receipt.json
reports/s17_n20a_terrain_sx300_no_nodata_recovery_receipt.json
artifacts/external/copdem_s17_n20/manifest.json
artifacts/external/copdem_s17_n20/recovery_manifest.json
artifacts/external/copdem_s17_n20/acquisition_receipt.json
artifacts/audits/s17_n21_terrain_model_family_prerequisite.json
reports/s17_n21_terrain_model_family_prerequisite_receipt.json
reports/s17_n22_m115_terrain_family_spec.json
reports/s17_n22_m115_terrain_strict_receipt.json
reports/s17_n22a_m115_terrain_no_refit_receipt.json
artifacts/backtests/s17_n22a_m115_terrain_recovery/evaluation.json
artifacts/backtests/s17_n22a_m115_terrain_recovery/family_manifest.json
artifacts/backtests/s17_n22a_m115_terrain_recovery/typed_recovery.json
inputs/competition/open_wind_236727.zip (hash + info.xlsx metadata member only)
```

Key evidence hashes:

```text
research/scratch/grid_coords.json                                      7091127c22032a55c61da572b46dc4900a0333d9ab6b69675a8307b886a7a997
reports/s17_n19_terrain_sx300_formula_manifest.json                     ca1022d8236acdefa2b886c20945029a060f3dddecb4d879cb4c6e5a99cf6b80
artifacts/external/copdem_s17_n20/recovery_manifest.json                5b2f6918b22e0b4f8168367b15355d6eec6603d3d08538d5989a97dae076ab0e
artifacts/audits/s17_n21_terrain_model_family_prerequisite.json         73dc743b24a14927336bbb7c5298a7f089dd9fd6658f5ee168f8d5d873ed3157
reports/s17_n22_m115_terrain_family_spec.json                           d0458e634d975f931868899c574daa961a343d52b369bc8143e5b7f6b6bb3c00
artifacts/backtests/s17_n22a_m115_terrain_recovery/evaluation.json      c1cac9d812eeeb1a2b341b99f9a8d7ca83ab243f8c2bb15cc7cc5caec9387552
```
