# S17-N27 — SCALAR_GRADIENT_REPOSITORY_NOVELTY

## Verdict

**CORE OPERATION NOVELTY SUPPORTED; ADMISSION NOT SUPPORTED.** `[directly_supported]` The fixed production feature source contains no oriented horizontal gradient of pressure, temperature, or geopotential. This establishes a representation gap, not usefulness or permission to execute it.

## Root-independent source audit

Only operation definitions were inspected; no source was executed and no model result artifact was read.

- `src/baram/features/weather.py::aggregate_weather` groups each scalar grid field by issuance/valid time and emits `mean/std/min/max/q10/q50/q90`. `[directly_supported]` These statistics are invariant to permutations of grid locations and therefore discard east/north orientation.
- `src/baram/features/spatial.py::aggregate_group_weather` emits turbine-group IDW and nearest-grid scalar values. `[directly_supported]` It preserves a site-weighted value, not a horizontal derivative or oriented field.
- `src/baram/features/geometric.py::_source_geometric_features` fits planar coefficients only to configured **u/v vector pairs** and derives divergence, vorticity, stretch, shear and gradient norm of those vector fields. The configured GFS/LDAPS maps contain no pressure, temperature or geopotential scalar. `[directly_supported]`
- `src/baram/features/physics.py` consumes aggregate/IDW surface pressure only inside dry-air-density and `rho*v^3` proxies. It contains no latitude/longitude derivative. `[directly_supported]`
- `.planning/2026-08-01-leaderboard-top-4-loop/run_classifier_iteration_sweep.py` names aggregate `meansea_0_prmsl` among all-weather variables, but the fixed feature-construction path supplies no pressure-plane or geostrophic operation. `[directly_supported]`
- A root operation-name scan over `src/` and `scripts/` found no `pressure_gradient`, `geostrophic`, `frontogenesis`, `baroclinic`, `temperature_gradient`, or `scalar_gradient` implementation. `[directly_supported]`

## Distinction from closed axes

| Existing operation | Why it is not the proposed operation |
|---|---|
| global scalar summaries | no coordinate orientation |
| group IDW/nearest | location-weighted scalar, no derivative |
| wind-vector planar geometry | derivatives of u/v, not scalar pressure/temperature/geopotential |
| GFS–LDAPS source disagreement | inter-source delta at a site, not within-source horizontal gradient |
| density/power proxies | pointwise scalar transform, no spatial forcing direction |

`[derived]` A scalar plane-gradient family would therefore add a new mathematical operation at the core-source level. Novelty alone supplies no exact field/level/domain choice and no target-scale evidence.

## Scope incident and containment

The delegated novelty lane left no file. Before root stopped it, an allowed planning-Python search printed a source line containing the **identifier** `actual_kwh`; no data value, row, aggregate, score or result was exposed. It also searched more planning Python files than needed. The delegated conclusion is not used. The verdict above was independently rebuilt from the five exact source files and bounded operation-name search, with no project import or execution.

## Disposition

Retain `PASS_NOVELTY_CORE` only as one conjunct. Because the physics and schema/formula lanes fail the full chain, do not emit a candidate, fit, score, or prerequisite.
