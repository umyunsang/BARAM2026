# S17-N18 terrain lane — published direction-dependent DEM exposure (`Sx`)

- [derived] **VERDICT: READY — prerequisite only.** Promote at most the single no-fit/no-body audit `TERRAIN_SX300_H8_LDAPS16`; do **not** promote a model, a score comparison, or DEM extraction from this lane.
- [near_match_only] The published evidence establishes that `Sx` is computable from a DEM and direction and that it was useful inside a fitted near-surface wind downscaler; it does **not** establish a BARAM power/settlement gain, a 117 m hub-height correction, or the proposed use as sixteen grid-conditioned features.
- [directly_supported] This lane performed no model/optimizer fit, metric call, DEM tile GET/range/body request, competition-test or operating-2024 access, rejected/quarantined access, dependency change, or external mutation.
- [unverified] `READY` is not execution authority: root must pass the no-body prerequisite below and separately authorize any later DEM body acquisition; no `[derived]` statement here may be treated as executable authority.

## 1. Evidence accounting

- [directly_supported] **External source count: 4 primary/official source packages (limit 10).** Multiple URLs under P3 are one official Copernicus product/licence package; the two P2 URLs identify the same paper.
- [directly_supported] Retrieval/audit time was 2026-08-08 KST; all quoted external material below is from the exact URLs listed.

### P1 — original primary definition

- [directly_supported] Winstral, Elder & Davis (2002), *Journal of Hydrometeorology* 3, 524–538, “Spatial Snow Modeling of Wind-Redistributed Snow Using Terrain-Based Parameters”: <https://journals.ametsoc.org/view/journals/hydr/3/5/1525-7541_2002_003_0524_ssmowr_2_0_co_2.xml>, DOI <https://doi.org/10.1175/1525-7541(2002)003%3C0524:SSMOWR%3E2.0.CO;2>.
- [directly_supported] In §3a, immediately before Eq. (1), the authors state: “**The objective of the maximum upwind slope (Sx) parameter was to quantify the extent of shelter or exposure provided by the terrain upwind of each pixel.**”
- [directly_supported] In §3a around Eq. (2), they state: “**Values of Sx were determined along vectors at 5° increments within the upwind window and averaged**,” and then list `50, 100, 300, 500, 1000, and 2000 m` search distances.
- [directly_supported] Figure 3 defines the algorithm as the greatest upward slope on each directional vector followed by averaging across vectors; its caption says, “**A 5° increment between search vectors was applied in this study.**”
- [contradicts_premise] The 2002 sector is not a universal frozen constant: §3a calls its 60° width “**arbitrarily chosen**,” and §5 reports that `Sx100`, not `Sx300`, was the strongest predictor for that snow-depth population.

### P2 — closest primary wind-forecast application and parameter freeze

- [directly_supported] Winstral, Jonas & Helbig (2017), *Journal of Hydrometeorology* 18, 335–348, DOI <https://doi.org/10.1175/JHM-D-16-0054.1>; institutional full PDF: <https://www.dora.lib4ri.ch/wsl/dload/wsl%3A12719/PDF/Winstral-2017-Statistical_downscaling_of_gridded_wind-(published_version).pdf> (1,283,321 bytes; SHA-256 `4b10c35d426d902dc97959a0eca7a73a4db6256409280c29459b78580b5f0cc5`).
- [directly_supported] Page 339, Eq. (1), defines `Sx` from elevation, horizontal distance, an explicit `height`, azimuth, and `dmax`; the following text says the height was introduced “**to account for instrument heights**” and that positive values indicate shelter while negative values indicate exposure.
- [directly_supported] Page 339 states: “**Similar to prior applications, Sx values were derived at 5° increments within 30° upwind windows centered on each increment.**” It freezes `height = 8 m` and evaluates `dmax = 300 m`, `500 m`, and `2 km`; only the latter two omit the nearest 100 m.
- [directly_supported] Page 343, immediately above Table 3, states: “**Sx-300 outperformed the other Sx derivations and was a significant predictor in all classes ... except for ... COSMO-2 ... valley sites.**”
- [directly_supported] Page 345 states that substituting COSMO forecast directions for observed directions left results “**virtually unchanged**,” with the greatest degradations reported as 0.02 m/s RMSE, 0.02 m/s MBE, and 0.01 KSD; §4d then describes a spatial application using forecast wind directions and a precomputable `Sx` library.
- [near_match_only] P2 used 25 m DEM terrain, 6.5/10 m station winds, COSMO 2/7 km forecasts, fitted bias coefficients, and wind metrics; BARAM uses GLO-30, supplied LDAPS grids, a 117 m turbine hub height, a power target, and a discontinuous settlement metric.

### P3 — official DEM/product/licence package

- [directly_supported] Official product page: <https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM> (locally receipted SHA-256 `f1e566e63f7e256d7ae4f5c78b13fc0c253252f219d7dfc617b8736540fb8a1c`). It says the Copernicus DEM is a DSM representing “**buildings, infrastructure and vegetation**,” was acquired by TanDEM-X in `2011–2015`, was made available in `2019`, and that GLO-30 has global 30 m coverage.
- [directly_supported] The same page’s “Horizontal resolution and vertical accuracy” / “Processing” sections specify GLO-30 `1.0"` spacing, WGS84-G1150 / EPSG:4326 horizontal CRS, EGM2008 / EPSG:3855 vertical reference, metres vertically, and `RasterPixelIsPoint` for DGED.
- [directly_supported] Official AWS packaging note: <https://copernicus-dem-30m.s3.amazonaws.com/readme.html> (SHA-256 `922461d55070e3c7e0cf2903ec54190d69e3773da2dc9e776a3f03566cfd56d2`) says the data are Cloud Optimized GeoTIFFs, documents removal of shared east/south edge rows, and links the licence.
- [directly_supported] Official licence: <https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf> (3 pages; SHA-256 `9cd37d37ea654bbcaf0a2e059e6a3a5b5f76072824d8dd860ccf274ada8951bd`). Articles 3–5 grant worldwide, unlimited-time, free-of-charge rights to reproduce, distribute, communicate, adapt, modify, and combine the data.
- [directly_supported] Licence Article 6(b) requires this notice for adapted/modified data: “**produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved**.”

### P4 — official software implementation cross-check, not an exact oracle

- [directly_supported] SAGA 9.8.0 official tool documentation: <https://saga-gis.sourceforge.io/saga_tool_doc/9.8.0/ta_morphometry_29.html>; pinned source: <https://raw.githubusercontent.com/saga-gis/saga-gis/d0dd586dac6e8bf3644b3012d6fc9466353b2af8/saga-gis/src/tools/terrain_analysis/ta_morphometry/wind_shelter.cpp> (SHA-256 `297d7fb76568a1a2502e32bb8f558bdc4973cc2c7fa521b6a8fae7931ebd797d`).
- [contradicts_premise] SAGA is **not** a drop-in reproduction oracle for P2: its documented direction is the direction “**into which the wind blows**,” its default excludes negative slopes, it has no P2 `height` term, and the source returns a transformed shelter angle rather than P2 Eq. (1) directly.
- [directly_supported] The pinned source header licenses SAGA under GPL v2 or any later version.

## 2. Exact candidate formula and one frozen configuration

- [directly_supported] P2 Eq. (1), written with `atan2` only to make the quadrant/unit handling explicit, is

  \[
  s_{j,\phi}(D,H)=\max_{p\in L(j,\phi,D)}
  \frac{180}{\pi}\operatorname{atan2}
  \left(z(p)-[z_j+H],\ d(p,j)\right),
  \]

  where `j` is the cell/query location, `z` is DEM elevation, `phi` is the upwind azimuth, `D=dmax`, `d(p,j)` is horizontal metric distance, and `L` is the set of DEM cells along the fixed search segment.
- [directly_supported] P1 Eq. (2) and P2’s method average the per-ray maximum angles within the upwind window.
- [derived] The **only** frozen BARAM lookup admitted by this lane is therefore

  \[
  Sx_j(q)=\frac{1}{7}\sum_{k=-3}^{3}s_{j,\,q+5k^\circ}(300\text{ m},8\text{ m}),
  \quad q\in\{0,5,\ldots,355\}^\circ.
  \]

- [derived] For finite supplied LDAPS 10 m eastward/northward components `(u_j(t),v_j(t))`, the meteorological **from** direction and deterministic bin are

  \[
  \theta_j(t)=\operatorname{mod}\!\left[\frac{180}{\pi}\operatorname{atan2}(-u_j(t),-v_j(t)),360\right],\qquad
  q_j(t)=5\left\lfloor\frac{\theta_j(t)+2.5}{5}\right\rfloor\bmod360,
  \]

  and the runtime feature is `terrain__sx300_h8_ldaps_jXX(t) = Sx_j(q_j(t))` in degrees.
- [derived] Exact zero vectors or nonfinite components produce `NaN`; this lane adds no learned calm threshold, interpolation, imputation rule, or wind-speed multiplier.
- [derived] The static table is exactly `16 × 72 = 1,152` values and yields exactly sixteen runtime columns; it is static in storage but time-varying after basis-safe forecast-direction lookup.

### Frozen choices — no local performance tuning

| Item | Frozen value | Evidence status |
|---|---|---|
| DEM | Copernicus GLO-30 DSM, one receipted version only | [directly_supported] eligible/static/licensed in P3 and local N14; [near_match_only] 30 m DSM vs P2 25 m DEM |
| Query locations | all 16 fixed LDAPS grid coordinates, no turbine/group-only `Sx` | [derived] representation choice; [directly_supported] 16 grids in local N14 |
| Horizontal geometry | native GLO-30 cells; metric distances/azimuths in WGS84 / UTM 52N (EPSG:32652), no vertical-datum mixing | [derived] deterministic raster adaptation, not prescribed by P2 |
| Cell traversal | supercover every native DEM cell footprint intersected by the 300 m ray; central `z_j` is the native cell containing the fixed coordinate | [derived] exact local freeze of P2’s “all cells along a fixed search line” wording |
| Radius | `dmax = 300 m` only | [directly_supported] closest wind application P2 says Sx-300 outperformed its alternatives |
| Height | `H = 8 m` only | [directly_supported] exact P2 setting; [near_match_only] it is not BARAM’s 117 m hub height |
| Window | centered `q ± 15°`, endpoints included | [directly_supported] P2’s centered 30° window; [derived] explicit endpoint convention |
| Rays | seven rays at 5° increments | [directly_supported] P2 5° increments; [derived] inclusive enumeration |
| Direction bins | 72 bins, nearest 5° with half-ties clockwise; no interpolation | [directly_supported] P2 rounded to nearest 5°; [derived] explicit tie convention |
| Sign/unit | degrees; retain negative exposure and positive shelter | [directly_supported] P2 definition |
| Proximal exclusion | none | [directly_supported] P2 applies the 100 m exclusion to `Sx-500*` and `Sx-2k*`, not `Sx-300` |
| Extra variants | none: no 100/500/2000/5000 m, 60° window, `H=0/10/117`, TPI, Sb, RIX, or SAGA output | [derived] multiplicity freeze |

- [contradicts_premise] The earlier local `{500, 2000, 5000 m}` proposal is not a primary-paper freeze: `5000 m` does not occur in P1/P2’s tested `Sx` sets, and keeping three radii creates three selectable representations.
- [near_match_only] `H=8 m` deliberately reproduces the closest published wind application; changing it to the known 117 m hub height would remain computable without `u(h_HC)` but would be an unsupported new variant and is not authorized here.
- [derived] No reviewed alternative is better for this narrow gate: static direction-averaged indices collapse to constants, while the documented SAGA Wind Shelter Index changes sign/direction/height semantics and therefore cannot replace P2 Eq. (1) without another algorithmic choice.

## 3. Why this avoids the KMAPP identifiability failure

- [directly_supported] `reports/s17_n15a_r3_corrected_formula_gate_receipt.json` adjudicates the exact KMAPP height correction as refuted/unidentified because its printed equation is dimensionally inconsistent and supplied same-source mean LDAPS vectors do not identify `u(h_HC)`.
- [directly_supported] P2 `Sx` requires only DEM elevations, fixed horizontal geometry, a fixed scalar `height`, `dmax`, and an azimuth; neither its Eq. (1) nor the sector average contains wind speed at any terrain-derived height.
- [derived] BARAM runtime direction can be calculated from the already identified LDAPS 10 m `(u,v)` vector; no interpolation/extrapolation to `h_HC`, no relabeling of 50 m extrema, and no cross-source wind substitution is needed.
- [directly_supported] `reports/s17_n12_vertical_profile_prerequisite_receipt.json` records comparable LDAPS mean vectors at 5 and 10 m and a known hub height of 117 m; it is the unavailable **magnitude at `h_HC`**, not the 10 m direction, that blocked KMAPP.
- [near_match_only] P2 supports using forecast rather than observed direction in its own COSMO population; it does not directly validate LDAPS direction or BARAM deployment.

## 4. Deployment symmetry and chronology

- [directly_supported] P3’s DEM was acquired in 2011–2015 and available from 2019, before the competition cutoff; local `reports/s17_n14_r3_terrain_prerequisite_receipt.json` already passed licence, time, one-tile HEAD availability, and 16-grid static-orography gates without reading DEM values.
- [derived] Development and deployment must use the same hashed GLO-30 bytes, the same 16 coordinates, the same 1,152-value lookup, the same forecast-`u/v` direction formula, and the same missing rule.
- [derived] Only the D-1 basis-safe issued LDAPS run may select `q_j(t)`; observed wind direction, SCADA, labels, reanalysis, stitched latest-run archives, or test-period observations may never select or alter the lookup.
- [derived] The lookup must be generated once before any chronological fold comparison; radius, height, sector, traversal, and column set may not be selected from fold results.
- [near_match_only] A downstream learner could use the sixteen columns only under root-owned chronological fitting; the present terrain computation itself has zero fitted parameters, but that does not make the eventual power treatment “zero-fit.”

## 5. Bounded nonduplication proof

- [directly_supported] The idea is **not novel to the research record**: `research/lanes/S6_ext_B_terrain.md` §B4 candidate A already sketches `Sx_grid`, and freezes a different 36-direction / three-radius representation.
- [directly_supported] It is nevertheless **not materialized in the current pipeline record**: `research/engine/pipeline_spec.json` at `stages.B4.current` says exactly, “`geometric block and G2 encoding; no DEM, no exposure index`.”
- [directly_supported] The implemented adjacent artifact `research/nodes/s9_n4_upwind_projection.py` explicitly says it runs “BEFORE candidate A (Sx_grid)” and “needs no DEM”; its code uses group/grid coordinates and wind direction to weight supplied grid wind speeds, not terrain elevation or an exposure lookup.
- [directly_supported] `research/lanes/S13_S6_features_deep.md` closes **group-constant** static `Sx/TPI/...` as group-dummy-collinear; the proposed output is sixteen direction-indexed time series, not three static group constants.
- [directly_supported] A bounded text scan across `src/`, `scripts/`, `configs/`, `research/engine/`, `research/nodes/`, and `reports/` (explicitly excluding forbidden path classes) found zero `Winstral` implementation hits; the only candidate-style `Sx_grid` code hit was the S9-N4 docstring saying it had not been run.
- [near_match_only] `research/rwa_data_feature_research.md` closes generic terrain labels and owns the failed KMAPP R3 re-entry, but P2 `Sx` is a direction-conditioned representation rather than the refuted height-correction formula.
- [derived] Thus the candidate is a conceptual continuation of S6, not a duplicate of a materialized DEM feature or KMAPP treatment on the named local evidence.
- [unverified] Binary/generated feature bodies were intentionally not opened; root must perform the schema/lineage-only gate below before claiming exhaustive artifact nonduplication.

### Named local evidence and hashes

| Local path | SHA-256 | Relevant location/fact |
|---|---|---|
| `reports/s17_n18_post_rwa_frontier_intake_predeclaration.json` | `2f00184f84f990d06125f5b89cb174a6b1f15a3afd1094d51568203deb33312f` | [directly_supported] lane question, source/time/action bounds |
| `reports/s17_n14_r3_terrain_prerequisite_receipt.json` | `c576917f22522cc3a08bb014cb9809d9113295e42a01a6fa7b248a71121922e7` | [directly_supported] licence/time/one-tile HEAD/16-grid prerequisite results; zero DEM body |
| `reports/s17_n15a_r3_corrected_formula_gate_receipt.json` | `66fa03a50ac0a3858bc244442a2a3a6e83cc5a2429fd0acce44f94901e3dfc9f` | [directly_supported] exact KMAPP treatment refuted/unidentified |
| `reports/s17_n12_vertical_profile_prerequisite_receipt.json` | `5f6d35ccc7193e683180b55b76ceca8c702be1ecc1257d682d93808a7d76bebe` | [directly_supported] 117 m hub; LDAPS 5/10 m mean-vector support |
| `research/lanes/S6_ext_B_terrain.md` | `aed46e006b7f5a9c52672b099b7e40d32e6634dbe99a385411d3d3f83e4248d2` | [directly_supported] §B4 candidate A conceptual predecessor and multiradius proposal |
| `research/lanes/S13_S6_features_deep.md` | `adff7bfc4f4d2c28bc9c859af16437f95d981aa5deaff0692564560b9bd65217` | [directly_supported] §1/§7 group-static closure |
| `research/rwa_data_feature_research.md` | `16351bb387dfebf0dc7ccdc6ef0e8ec77406964bf3d0b40c52fcdc3496284504` | [directly_supported] generic terrain/R3 ownership and closure context |
| `research/engine/pipeline_spec.json` | `e08ffc881937bdaac1d04cb95938e16835862679a8693b30e0b7298cba21f69d` | [directly_supported] `stages.B4.current`: no DEM/exposure index |
| `research/nodes/s9_n4_upwind_projection.py` | `bc74ac01cbec4c0f1632377f845378f9b966f7d344e3d89120d9555a46b1f5e8` | [directly_supported] adjacent geometry-only implementation, explicitly before `Sx_grid` |

## 6. Licence and implementation considerations

- [directly_supported] Copernicus permits commercial-field use through broad worldwide/free rights without a noncommercial restriction, but redistribution/communication of a derived `Sx` table must carry the Article 6(b) adapted-data notice and the Article 6(c) no-liability notice.
- [directly_supported] GLO-30 is a DSM, so canopy/building heights are part of `z`; P2’s maximum operator can make a 300 m `Sx` sensitive to a single elevated or erroneous cell.
- [derived] The extraction receipt must therefore record DEM object URL, byte hash, Last-Modified/ETag, CRS/vertical datum, PixelIsPoint handling, nodata/fill-mask policy, tile-edge handling, and every frozen formula parameter.
- [directly_supported] The project’s declared core dependencies contain no `rasterio`, `pyproj`, GDAL, or `tifffile`; the current project-local interpreter has Pillow 12.3.0 with libtiff/zlib but Pillow is not declared in `pyproject.toml`.
- [unverified] Pillow has not been exercised on the forbidden DEM body in this lane, so reproducible COG float/tag decoding and EPSG:32652 geometry remain prerequisite gates; a hidden system command or new dependency is not authorized.
- [contradicts_premise] Copying or linking SAGA source would introduce GPL obligations and still would not implement the frozen P2 equation; the proposed route is an independent implementation of the paper’s mathematics, not copied SAGA code.
- [unverified] The papers provide scholarly definitions, not a software licence or patent clearance for a BARAM implementation; no patent/legal opinion was performed, so root should retain citations and treat legal clearance beyond the explicit DEM/GPL facts as unresolved.

## 7. Smallest no-fit prerequisite audit (the only promotion candidate)

- [derived] **Candidate ID:** `TERRAIN_SX300_H8_LDAPS16_NO_BODY_AUDIT`.
- [derived] **Scope:** metadata/header/lineage and analytic-contract checks only; zero DEM value bytes, zero feature values, zero label/SCADA access, zero fit, zero metric, and zero policy comparison.

1. [derived] Hash the existing 16-coordinate source and verify exactly 16 unique, finite LDAPS locations plus the already receipted 10 m `(u,v)` schema; do not read forecast rows.
2. [derived] Compute the union of 300 m WGS84 buffers around those coordinates, enumerate every required 1° GLO-30 tile key, and issue `HEAD` only; require HTTP 200, positive length, stable ETag/Last-Modified, and pre-cutoff availability. N14 checked only N37/E128, so neighbouring-tile coverage is not yet proven.
3. [derived] Freeze a machine-readable manifest for exactly the formula/configuration in §2, including FROM-direction cardinal checks: `(u=0,v=-1)→0°` and `(u=1,v=0)→270°`.
4. [derived] Prove two independent implementations agree on in-memory analytic elevation arrays (flat, constant planar slope, one upwind obstacle, all-negative horizon) to tight numerical tolerance; do not use SAGA as the oracle.
5. [derived] Prove the unchanged project interpreter can decode an in-memory synthetic float GeoTIFF and expose required geotags with no dependency change; if it cannot, return `BLOCKED_DEPENDENCY` rather than downloading a DEM.
6. [derived] Scan only feature schemas/manifests and code lineage for `sx`, `winstral`, `shelter`, `dem_exposure`, and equivalent anonymous lineage; if a matching 16-grid direction lookup already exists, return `CLOSED_DUPLICATE`.
7. [derived] Emit only a prerequisite receipt with source count, URLs, hashes, HEAD metadata, formula manifest hash, environment facts, and PASS/FAIL gates; any DEM GET/range/body request invalidates the audit.

## 8. Falsification and closure gates

- [derived] **Formula gate:** fail if any implementation references `u(h_HC)`, any wind magnitude in the `Sx` value, a fitted coefficient, or a non-DEM dynamic input other than issued forecast direction.
- [derived] **Parameter gate:** fail if more than `dmax=300 m`, `H=8 m`, one 30°/5° sectorization, or more than sixteen columns is admitted; local score-based choice among variants closes this predeclaration.
- [derived] **Direction gate:** fail on FROM/TO reversal, observed-direction use, interpolation not frozen here, or SAGA-default semantics.
- [derived] **Coverage gate:** block if any required tile is absent, mutable/unreceipted, post-cutoff, or cannot cover the full 300 m rays including seams.
- [derived] **Decoder gate:** block if body decoding would require an undeclared/new dependency, system-package mutation, or an unaudited TIFF/CRS path.
- [derived] **DEM-quality gate for later extraction:** close if nodata/fill/seam artifacts dominate any lookup or if the 16×72 table is nonfinite; no repair may be selected after seeing outcome metrics.
- [derived] **Nonduplication gate:** close if schema/lineage audit finds the same DEM×direction lookup already materialized; a differently named equivalent counts as duplicate.
- [derived] **Symmetry gate:** close if development and deployment cannot use the byte-identical table and same basis-safe issued-run direction logic.
- [derived] **Information gate for any later root-owned feature-only audit:** close without fit if every direction lookup is constant/degenerate or byte-identical to an existing column; do not rescue it with extra radii or heights.
- [near_match_only] Even if all prerequisite gates pass, the next state is only `READY_STATIC_LOOKUP_EXTRACTION`; expected power/settlement benefit remains unverified until a separately predeclared, root-owned chronological comparison.

## 9. Final adjudication

- [derived] `Sx` is mathematically identifiable without KMAPP’s missing `u(h_HC)` and has a defensible single published near-surface freeze (`300 m`, `30°`, `5°`, `8 m`).
- [derived] The exact representation is conceptually prefigured in S6 but is nonduplicate of the named materialized pipeline/code evidence because no DEM exposure index is present and the adjacent implementation is geometry-only.
- [near_match_only] The BARAM use remains an inductive-bias feature, not a published physical correction at 117 m and not evidence of score gain.
- [derived] **Final lane verdict: READY for exactly `TERRAIN_SX300_H8_LDAPS16_NO_BODY_AUDIT`; BLOCKED for DEM extraction or modeling until that audit passes and root explicitly promotes it.**
