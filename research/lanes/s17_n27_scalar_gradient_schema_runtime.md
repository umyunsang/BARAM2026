# S17-N27 — SCALAR_GRADIENT_SCHEMA_FORMULA_RUNTIME

## Verdict

**FAIL_CLOSED / NO ZERO-CHOICE FORMULA.** `[directly_supported]` The local issued schema and NumPy runtime can support deterministic scalar-plane arithmetic. `[unverified]` They do not identify one provider-prescribed field/level/source/domain/estimator/output contract, and exact LDAPS unit/provider semantics were not established from the bounded official pages.

## Literal local availability

The fixed `prepare.json` feature-name list contains all seven existing spatial summaries (`mean/std/min/max/q10/q50/q90`) for:

- GFS: `meanSea_0_prmsl`, `surface_0_sp`, `heightAboveGround_2_2t`, `isobaricInhPa_{850,700,500}_t`, and `isobaricInhPa_500_gh`;
- LDAPS: `meanSea_0_prmsl`, `surface_0_sp`, `heightAboveGround_2_t`, and `surface_0_h`.

`[directly_supported]` `weather.py` builds those summaries from numeric raw grid columns keyed by `forecast_kst_dtm` and `data_available_kst_dtm`. `geometric.py` additionally requires literal `grid_id/latitude/longitude`, converts them to local east/north kilometres, and uses NumPy pseudoinverse plane fits for wind-vector fields. Thus the unchanged pandas/NumPy Darwin environment is sufficient for arithmetic; no dependency is needed.

## Official semantics reached

- NOAA/NCEP GRIB2 Table 4.2-0-3 identifies `PRES` and pressure reduced to mean sea level `PRMSL` in Pa, and geopotential/geopotential height in their stated units. `[directly_supported]`
- NOAA/NCEP GRIB2 Code Table 4.5 identifies mean sea level and specified height above ground as distinct fixed-surface types. `[directly_supported]`
- `[unverified]` Those GFS definitions do not by themselves prove that the competition's LDAPS column aliases have identical units, vertical processing, or grid projection. No bounded official LDAPS page prescribed a horizontal-gradient feature.

Official documentation:
- https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-3.shtml
- https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-5.shtml

## Unresolved treatment choices

At least the following materially different choices remain; freezing one as an analyst convention would not make it provider-supported:

1. GFS, LDAPS, or both; and if both, concatenate, average, or difference.
2. `PRMSL`, surface pressure, converted 850 hPa geopotential, 500 hPa geopotential height, or temperature/baroclinic gradients.
3. Direct Pa-distance gradient versus Zhu's hydrostatic conversion to a reference-pressure geopotential surface.
4. Reference pressure, temperature average, and whether/how to remove grid/time/month bias.
5. All source grid points versus a farm-centred subset or group-specific weights.
6. Finite differences, local neighbours, weighted regression, or the existing unweighted global planar pseudoinverse.
7. Spherical/UTM/geodesic coordinates versus the existing local equirectangular kilometres.
8. Output components, magnitude, direction, geostrophic components, or projections relative to issued wind/layout.
9. Missing-grid treatment and source-specific grid domains.

`[directly_supported]` Zhu et al. fixes one observation-network treatment (`pref=850 hPa`, 12 stations, monthly centering, plane fit), but applying it to already forecast MSLP grids—or substituting the repository's existing plane geometry—is a new bridge. Couto & Estanqueiro names a 48 h WRF `MSLPGrad` feature but the inspected source does not disclose a coefficient-complete estimator. These do not collapse the list above to one treatment.

## Gate result

| Gate | Result |
|---|---|
| Same-issuance computability | PASS |
| Literal GFS fields/units | PASS for NCEP aliases |
| Literal LDAPS units/provider semantics | **UNVERIFIED** |
| One source-prescribed estimator/domain/output | **FAIL** |
| No scale/level/source/projection choices | **FAIL** |
| Unchanged runtime | PASS |

## Provenance

The delegated schema lane left no durable artifact; root reconstructed the result from the fixed local metadata/source and the two official tables. No weather values, labels, result scores, 2024/test values, model fit/prediction/metric, external data object, attachment, dependency, account, or Dacon action was used.
