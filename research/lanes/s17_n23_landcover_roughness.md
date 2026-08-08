# S17-N23 land-cover / aerodynamic-roughness intake

## Verdict

- [derived] **VERDICT: `NOT_READY_NO_DIRECTLY_SUPPORTED_NEW_REPRESENTATION`.** ESA WorldCover 2021 v200 passes the static-source, pre-cutoff, Korea-coverage, commercial-use, and metadata-level deployment gates, but the evidence does not freeze one *new* BARAM wind representation with zero tuning degrees.
- [directly_supported] The closest exact Korea/LDAPS mechanism is a neutral log-law roughness adjustment, but that representation is already present in the repository as a log-law/effective-`z0` family and the closest Korean sensitivity experiment found roughness adjustment worse than the matched height-only treatment.
- [directly_supported] The genuinely distinct alternative—a direction-conditioned roughness rose/internal-boundary-layer speed-up—is supported as a wind mechanism, but the published implementation is proprietary; the current PyWAsP distribution requires a licence and network validation and explicitly does not support macOS.
- [unverified] Therefore this lane authorizes **no raster acquisition, feature materialization, fit, or comparison**. A static-source receipt alone must not be promoted as an executable candidate.

## 1. Audit accounting and constraints

- [directly_supported] Research was performed on 2026-08-09 KST in under 25 minutes and cites **10 unique primary official/paper URLs**, below the limit of 12.
- [directly_supported] External access was limited to search metadata, official HTML/landing pages, rendered documentation/source documentation, and primary-paper HTML/EPUB text; there was no raster/data/weight object GET, object HEAD, bucket listing, data/inference API query, remote inference, dependency installation, or external mutation.
- [directly_supported] No model was fit, no prediction/policy/metric/score was called, and no label, `actual_kwh`, 2024, competition-test, or lockbox value was read.
- [directly_supported] The only repository write from this lane is this file.

## 2. Primary/official source register

1. [directly_supported] **P1 — ESA WorldCover official data access:** <https://esa-worldcover.org/en/data-access>.
2. [directly_supported] **P2 — WorldCover 2021 v200 versioned Zenodo record:** <https://zenodo.org/records/7254221>.
3. [directly_supported] **P3 — AWS Open Data Registry entry managed by VITO:** <https://registry.opendata.aws/esa-worldcover-vito/>.
4. [directly_supported] **P4 — CC BY 4.0 legal code:** <https://creativecommons.org/licenses/by/4.0/legalcode.en>.
5. [near_match_only] **P5 — DTU Data direct GWA4 land-cover/roughness landing:** <https://data.dtu.dk/articles/dataset/Global_Wind_Atlas_v4_Roughness_length_/28955279>.
6. [near_match_only] **P6 — Global Wind Atlas official dataset description:** <https://globalwindatlas.info/about/dataset>.
7. [directly_supported] **P7 — Floors et al. (2021), primary roughness-map/WAsP paper:** <https://wes.copernicus.org/articles/6/1379/2021/>.
8. [directly_supported] **P8 — Keum et al. (2021), primary Korea LDAPS/KMAPP-Wind paper:** <https://j-komes.or.kr/_EP/view/index.php?aidx=28740&bidx=2522>.
9. [directly_supported] **P9 — official PyWAsP 2.0 installation/platform documentation:** <https://docs.wasp.dk/pywasp/latest/getting_started/installation.html>.
10. [directly_supported] **P10 — official WindKit 1.0.2 rendered land-cover-table source documentation:** <https://docs.wasp.dk/windkit/v1.0.2/_modules/windkit/topography/landcover.html>.

## 3. Static data eligibility

### 3.1 ESA WorldCover 2021 v200 — source gate passes

- [directly_supported] P1 and P2 identify **WorldCover 2021 v200**, a global 2021 land-cover map based on Sentinel-1 and Sentinel-2, with **10 m resolution and 11 classes**; P2 records publication on **2022-10-28**, before the 2026-07-05 cutoff.
- [directly_supported] Global coverage includes the Korean project sites, and the 2021 vintage is static and predates every future deployment row; unlike a forecast archive, it has no issue-time stitching or test-observation dependency.
- [directly_supported] P1 documents **2,651 3°×3° Cloud-Optimized GeoTIFF tiles in EPSG:4326**, and gives the version-pinned anonymous path `s3://esa-worldcover/v200/2021/map` plus a Zenodo route whose 60°×60° macrotiles contain the 3°×3° tiles.
- [directly_supported] P2 independently records the dataset as “Open”, version `v200`, with a 124.0 GB file collection; P3 identifies public bucket `arn:aws:s3:::esa-worldcover`, region `eu-central-1`, and anonymous AWS CLI access (“No AWS account required”).
- [directly_supported] These two independent, versioned distribution channels are archive-stable enough for a later root-verifiable, bounded tile acquisition prerequisite.
- [unverified] Per this lane's prohibition, no Korean tile object was listed, HEADed, range-read, or downloaded, so its eventual object key, byte length, ETag/Last-Modified, and SHA-256 remain a mandatory acquisition receipt rather than evidence from this lane.

### 3.2 Licence gate passes for the static source

- [directly_supported] P1 states verbatim that the ESA WorldCover product is “provided free of charge, without restriction of use” and applies **Creative Commons Attribution 4.0 International**.
- [directly_supported] P4 §2 grants worldwide, royalty-free, irrevocable rights to reproduce/share the material and produce/reproduce/share adapted material; it contains no non-commercial restriction.
- [derived] WorldCover is therefore commercially compatible, subject to attribution and modification notices; this is a licence-text reading, not legal advice.
- [directly_supported] P1 supplies the requested attribution form, and P4 §3 requires retaining creator/credit, copyright/licence/warranty notices and a source URI where supplied, indicating modifications, and identifying CC BY 4.0.

### 3.3 Direct `z0` alternative is not needed to establish source eligibility, and is not independently deployment-ready here

- [near_match_only] P5/P6 describe a GWA4 land-cover/roughness product derived from WorldCover v200, resampled to `1/2400°` (about 50 m), with a class-specific roughness conversion; official indexed metadata identifies CC BY 4.0 and a pre-cutoff 2025 posting.
- [unverified] The P5 landing request from this environment returned an AWS-WAF `202` challenge rather than the file manifest, while P6 rendered only the application shell; this lane could not verify exact filenames, geographic packaging, checksums, or anonymous object access without crossing the no-data-body/no-data-API boundary.
- [derived] WorldCover's directly documented COG/Zenodo routes are the stronger acquisition prerequisite; using P5 would remove a class-to-`z0` conversion step but would not solve the missing wind-treatment formula or runtime blockers below.

## 4. What the primary wind literature actually supports

### 4.1 Directional roughness physics (new in concept, not executable here)

- [directly_supported] P7 Eq. (1) gives the Monin–Obukhov wind profile in terms of friction velocity, aerodynamic roughness length `z0`, displacement height `d`, and a stability correction; it explicitly says `d` is important at forested sites.
- [directly_supported] P7 §4.1 defines a polar “spider-grid” analysis with default first radial segment `r0=25 m`, 5% radial-spacing growth, and 12 direction sectors of 30° centred on north.
- [directly_supported] P7 Eq. (10) freezes within-cell heterogeneous roughness as the area-fraction geometric mean, `ln(z0_cell)=Σ_i f_i ln(z0_i)`, while Eq. (11) uses the arithmetic area-fraction mean for `d`.
- [directly_supported] P7 then states that WAsP distance-weights roughness changes, retains changes explaining most weighted variance up to `n_max`, and applies internal-boundary-layer equations to obtain **sector-wise speed-up factors**; nearby roughness has greater impact.
- [near_match_only] P7 validated climatological horizontal cross-predictions at eight instrumented sites, not day-ahead LDAPS-to-117 m power prediction or the BARAM step score; WorldCover itself was not the evaluated high-detail forest product.
- [contradicts_premise] P7 warns that global land-cover-to-roughness tables can be unrepresentative, especially for heterogeneous forest, and its better result used collocated tree height/LAI to derive both `z0` and `d`, not a land-cover class alone.
- [directly_supported] P7's Code Availability says its numerical results were generated with proprietary software.
- [directly_supported] P9 says PyWAsP “requires a license to run”, requires outbound network access to DTU's licence server, and lists **macOS: Not supported**; it also downloads additional global files on first run.
- [derived] Thus a row feature such as `landcover__roughness_speedup12(t)`—selecting a published sector-wise speed-up by supplied forecast direction—would be genuinely direction-varying and distinct from DEM `Sx`, but cannot be frozen or executed self-contained on the Apple-M1 project environment from P7/P9.

### 4.2 Exact Korea/LDAPS log-law treatment (executable algebra, but neither new nor supported as beneficial)

- [directly_supported] P8 §2.1.1 defines the neutral roughness correction
  `u_RA(z) = u(h_RA) * ln(z/z0) / ln(h_RA/z0)`, where `h_RA` is the roughness-reference height and `z0` is local vegetation roughness.
- [directly_supported] P8 used 30 m Korean Ministry land-cover classes mapped through a lookup table and coupled this correction to terrain-dependent reference heights; it did not define “set `h_RA=10 m` and extrapolate to the BARAM 117 m hub”.
- [contradicts_premise] P8 Table 3 reports that interpolation + height correction **without** roughness was best (`RMSE 1.82 m/s`, case C5), whereas adding roughness gave `1.91 m/s` (case C2); the authors conclude roughness adjustment had limited value in Korean complex terrain.
- [near_match_only] P8's population was 16 ICE-POP AWS sites, one winter month, +36 h near-surface wind, so it cannot directly prove a BARAM power/settlement loss; it does directly reject promoting a generic Korea/LDAPS roughness correction on mechanism alone.
- [directly_supported] P8 also says the neutral correction should eventually account for atmospheric stability, leaving a neutral-only 117 m transfer physically incomplete.

## 5. Repository novelty audit

- [directly_supported] `research/nodes/s9_n6_rews_geom.py:11-15,40-63,93-101` already solves a two-level log law for time-varying `z0`, clips it, and constructs hub/rotor-equivalent wind; a static-`z0` log-law treatment is therefore a variant of an existing representation, not a new representation.
- [directly_supported] `research/lanes/S6_ext_B_terrain.md:348-365,469-474` already records this dynamic effective-`z0` family and explicitly closes a static roughness-map feature as constant/inaccurate; `research/lanes/s17_n18_terrain_sx.md:70` and `research/lanes/s17_n21_terrain_model_integration.md:22` already own direction-conditioned DEM exposure.
- [derived] A site/group `z0` scalar by itself is constant over time and adds no row-varying wind information beyond fixed group/grid identity.
- [directly_supported] The bounded text search found no implemented land-cover roughness rose or internal-boundary-layer speed-up; that is the only conceptually new representation identified by this lane.
- [unverified] P10 confirms that WindKit knows `WorldCover` and `GWA4` land-cover tables with `z0` and `d` fields, but the rendered page does not expose the actual `WorldCover.json`/`GWA4_micro.json` values; no authoritative table file was acquired in this lane.

## 6. Full-chain gate and explicit blockers

| Gate | Finding | Status |
|---|---|---|
| Pre-2026-07-05 publication | [directly_supported] WorldCover v200 published 2022-10-28. | **PASS** |
| Korea / all-row availability | [directly_supported] Global static 2021 map; no issue-time or observation dependency. | **PASS** |
| Commercial-use licence | [directly_supported] WorldCover says unrestricted use under CC BY 4.0; P4 grants reuse/adaptation. | **PASS** |
| Archive-stable metadata route | [directly_supported] Versioned anonymous S3 COG route plus open Zenodo DOI macrotiles. | **PASS_PREREQUISITE** |
| Exact object identity | [unverified] No Korea object HEAD/body/listing was allowed. | **LATER_RECEIPT** |
| Exactly one new representation | [derived] Only the 12-sector roughness-speed-up concept is new; static log-law `z0` is already represented. | **CONCEPT_ONLY** |
| Zero-tuning published formula | [unverified] Full speed-up needs class `z0/d`, domain extent, `n_max`, distance weighting, and internal-boundary-layer implementation not frozen by accessible open evidence. | **FAIL** |
| Self-contained Apple-M1 execution | [directly_supported] PyWAsP requires licence/network and does not support macOS. | **FAIL** |
| Closest Korea evidence | [contradicts_premise] Roughness worsened 1.82 to 1.91 m/s versus height-only in P8. | **FAIL_FOR_PROMOTION** |

- [unverified] **Blocker B1:** obtain an authoritative, commercially compatible WorldCover-v200 class-to-`z0` **and `d`** table with version/hash; a roughness-only table is insufficient for forested terrain under P7.
- [unverified] **Blocker B2:** obtain an open, locally runnable, fully specified roughness-change/internal-boundary-layer implementation, including fixed outer radius, `n_max`, weighting, and sector tie rules; PyWAsP is not an admissible Apple-M1 prerequisite as documented.
- [unverified] **Blocker B3:** independently receipt the minimal Korean tile object(s), hashes, and exact site/group coverage after root authorizes data acquisition.
- [contradicts_premise] **Blocker B4:** even if B1–B3 were solved, the closest LDAPS/Korean sensitivity evidence gives roughness no positive credit; any later experiment would be exploratory/near-match, not `[directly_supported]` promotion evidence.
- [derived] Since no candidate satisfies all gates simultaneously, the lane returns **NOT READY** rather than inventing a radius, class lookup, neutral 117 m extrapolation, or proprietary runtime.
