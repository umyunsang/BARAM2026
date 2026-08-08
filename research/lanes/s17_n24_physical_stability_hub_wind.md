# S17-N24 — physical stability / hub-height wind intake

## Verdict

- [directly_supported] **FAIL_CLOSED — `NO_EXECUTABLE_CANDIDATE`.** The supplied schema can support the target height, multiple forecast wind levels, and one boundary-layer-height field, but it does not supply the surface stress/friction velocity, aerodynamic roughness length, or Obukhov length/surface virtual-temperature flux required by the closest coefficient-complete published stability profile.
- [contradicts_premise] A constant-exponent hub transfer and a two-level 80–100 m shear-exponent transfer are already encoded locally, so either is duplication rather than a new representation.
- [near_match_only] The closest distinct representation is the published Gryning whole-boundary-layer profile. Its equations are frozen, but its required state and its homogeneous/flat or marine applicability are not established for the supplied site. No substitution, inverse calibration, roughness default, cross-model diagnostic, or fallback can be selected with zero choices.
- [derived] **Recommendation: admit no executable treatment and no acquisition prerequisite from this lane.** Closing the blockers would require several independent physical inputs plus an applicability decision, not one bounded prerequisite.

## Scope and evidence discipline

- [directly_supported] The audit used only the N24-whitelisted local archive metadata/header, feature sources, turbine parser, and prepared feature-name manifest; no weather row body or target value was read for this report.
- [directly_supported] The external evidence set contains four URLs, all official institutional/publisher pages or a primary paper, and all publications used predate the 2026-07-05 cutoff.
- [directly_supported] No external data/catalogue/object endpoint, external inference, dependency change, model execution, or repository write other than this file was used.
- [directly_supported] Evidence tags have their N24 meanings: `[directly_supported]`, `[derived]`, `[near_match_only]`, `[contradicts_premise]`, and `[unverified]`.

## 1. Exact supplied-variable audit — performed before literature selection

### 1.1 Literal schema support

| Evidence | Required physical quantity | Exact supplied support | Gate |
|---|---|---|---|
| [directly_supported] | Hub target height | `info.xlsx` declares `Hub Height(m)`; the whitelisted parser rejects any turbine value other than **117 m** (`src/baram/data/turbines.py:27,174-175`). | PASS |
| [directly_supported] | GFS low-level wind | Paired components exist at 10, 80, and 100 m: `heightAboveGround_10_10u/v`, `heightAboveGround_80_u/v`, and `heightAboveGround_100_100u/v` (archive `data_description.md` §8 and the GFS header). | PASS |
| [directly_supported] | LDAPS low-level wind | Paired 10 m components and paired 5 m boundary-layer components exist. The four 50 m fields are explicitly component maxima/minima (`50MUmax/min`, `50MVmax/min`), not one documented contemporaneous mean 50 m vector (archive `data_description.md` §7 and the LDAPS header). | PARTIAL |
| [directly_supported] | Thermodynamic fields | Both sources include 2 m temperature, dew point, humidity fields, and surface pressure. GFS also includes pressure-level temperature at 850/700/500 hPa; the supplied GFS schema does not include a matching 850/700 hPa geopotential height, while LDAPS supplies no second-height air temperature (same archive sections/headers). | PARTIAL |
| [directly_supported] | Boundary-layer depth | LDAPS supplies `etc_0_blh`; GFS does not expose a boundary-layer-height column in its exhaustive supplied header. | ONE SOURCE ONLY |
| [directly_supported] | Static height/land indicators | LDAPS supplies `surface_0_h` and `surface_0_lsm`. Neither is an aerodynamic roughness length or displacement height by its documented meaning. | NEAR MATCH ONLY |
| [derived] | Surface stress / friction velocity `u_*` | Neither exhaustive weather header contains surface stress, momentum flux, or friction velocity. | MISSING |
| [derived] | Obukhov length `L` or its flux input | Neither exhaustive weather header contains `L`, sensible/virtual-temperature flux, or a surface kinematic heat flux. | MISSING |
| [derived] | Aerodynamic roughness `z0` / displacement `d` | Neither weather header nor turbine metadata supplies `z0` or `d`. | MISSING |
| [derived] | Bulk-stability vertical pair | A literal surface-layer virtual-potential-temperature difference at two known heights is absent: each source has only a 2 m near-surface temperature, and the GFS pressure-level temperatures lack their corresponding supplied height except at 500 hPa. | MISSING |

### 1.2 Basis/deployment symmetry

- [directly_supported] The official archive metadata says both weather sources use the daily 09:00 KST initialization, expose the following 24 target hours as one issuance, and mark that issuance available at 13:00 KST (`data_description.md:100-127`). These supplied fields therefore clear the D-1 14:00 KST basis-time gate.
- [derived] A deterministic transform using only literal supplied fields would have basis/deployment symmetry because the same weather schemas and availability key are provided on both sides of deployment.
- [contradicts_premise] The missing `u_*`, `z0`, and `L` cannot receive symmetry credit: importing reanalysis values, observations, or after-the-fact flux diagnostics is outside this lane and outside the official boundary.

## 2. Local duplication audit

- [directly_supported] `src/baram/features/weather.py:18-33,151-152` already forms speed and direction features for GFS 10/80/100 m winds and LDAPS 5/10/50-max/50-min winds; its aggregation also retains every numeric supplied field through fixed summary operators (`:68-104`).
- [directly_supported] `src/baram/features/physics.py:51-54` already encodes the fixed power-law transfer `V117 = V100 (117/100)^0.2`, 100–80 m speed shear, dry density, and density-times-hub-speed-cubed.
- [directly_supported] `src/baram/features/physics.py:72-94` already derives `alpha = ln(V100/V80)/ln(100/80)`, applies its documented validity guard/fallback and clip, and emits a second `V117 = V100 (117/100)^alpha` plus the associated physical proxy.
- [directly_supported] `src/baram/features/geometric.py:18-36,332-352` already retains the multi-height vector fields and GFS 80–100 m vector alignment; its spatial shear is horizontal velocity-gradient shear, not an atmospheric-stability correction.
- [directly_supported] `artifacts/manifests/prepare.json` literally lists both existing hub-speed families and the shear-alpha, density, cubic-speed, and fleet-area proxies; this is feature-name metadata, not a generated row body.
- [contradicts_premise] Consequently, another fixed-alpha, observed-two-level-alpha, neutral log-law, dry-density, or cubic-wind construction is not a new N24 representation. The N24 predeclaration also forbids roughness rescue without curing every prior blocker.

## 3. One published candidate audited

### Candidate C1 — Gryning whole-boundary-layer stability profile

- [directly_supported] Gryning et al. (2007) formulate an extension of the surface-layer wind profile to the whole boundary layer using a surface-layer length, a middle-boundary-layer length `L_MBL`, and an upper-boundary-layer length; the DTU primary-paper landing states that the middle-layer parameterization depends on stability and was derived from measurements at two sites [E1].
- [directly_supported] The companion open primary paper describes the target population as **flat terrain**, reports that ordinary Monin–Obukhov surface-layer scaling progressively departs above about 50–80 m, and identifies boundary-layer height as influential in the lowest few hundred metres [E2].
- [directly_supported] Valkonen et al. (2022) publish the stable branch as
  `u(z)=u*0/kappa [ln(z/z0) - psi_m(z/L)(1-z/(2 zi)) + z/L_MBL - (z/zi) z/(2 L_MBL)]`,
  with corresponding published neutral and unstable branches and a fixed empirical `L_MBL` parameterization [E3].
- [directly_supported] In that equation `u*0` is surface friction velocity, `z0` aerodynamic roughness length, `L` Obukhov length, `zi` boundary-layer depth, and `psi_m` the stability correction [E3].
- [directly_supported] The AMS definition independently identifies `L` as a function of friction velocity, surface kinematic virtual-temperature flux, reference virtual temperature, von Kármán's constant, and gravity; its sign selects stable versus unstable stratification [E4].
- [directly_supported] The profile is coefficient-complete as published when those state variables are supplied; setting `z=117 m` and computing Coriolis frequency from supplied latitude introduces no fitted coefficient.

### Candidate-to-supplied mapping

| Evidence | Published input | Literal mapping | Decision |
|---|---|---|---|
| [directly_supported] | `z=117 m` | Exact turbine hub height. | PASS |
| [directly_supported] | `zi` | LDAPS `etc_0_blh` is documented as boundary-layer height. | PASS, LDAPS ONLY |
| [derived] | Coriolis frequency | Deterministic from supplied latitude and a physical constant. | PASS |
| [derived] | Reference wind | GFS has paired 80/100 m vectors, but C1 does not replace `u_*`, `z0`, and `L` with one reference wind. | INSUFFICIENT |
| [derived] | `u*0` | No supplied stress, momentum flux, or friction-velocity field. | FAIL |
| [derived] | `z0` | No supplied aerodynamic roughness or displacement field. `surface_0_h` and `surface_0_lsm` are not `z0`. | FAIL |
| [derived] | `L` / flux | No supplied Obukhov length or surface virtual-temperature flux. | FAIL |
| [derived] | Bulk-Richardson alternative | The paper's observational route requires a virtual-potential-temperature difference at a known second height. That exact vertical pair is absent. | FAIL |
| [near_match_only] | One internally consistent atmospheric column | The only literal `zi` is LDAPS, while the only 80/100 m reference pair is GFS. Combining them is chronology-safe but is not the published same-column treatment and has no supplied physical-consistency guarantee. | FAIL CLOSED |

### Zero-choice and applicability audit

- [contradicts_premise] Solving `u_*`, `z0`, and `L` inversely from the three GFS wind levels is not the published forward representation and would require admissible domains, root selection, calm handling, and failure rules; those are tunable implementation choices.
- [contradicts_premise] Substituting the marine Charnock roughness used in the 2022 evaluation would impose a marine surface model on a site whose supplied metadata does not certify marine applicability [E3].
- [near_match_only] The original profile was developed for homogeneous terrain, with its applied expression based on two measurement sites and only 10 m winds above 3 m/s [E1]; the companion paper is explicitly for flat terrain [E2]. The whitelisted supplied metadata contains coordinates and model-grid surface height but no statement that the turbine inflow sectors are homogeneous or flat.
- [near_match_only] The 2022 study evaluates marine/offshore reduction from observation heights to 10 m, not land-site upward transfer to 117 m [E3]. It supports the mechanism, not this deployment population.
- [derived] No fixed default for `z0`, neutralizing `L`, borrowing prohibited reanalysis state, or selecting a cross-source diagnostic can cure all missing-input and applicability gates without adding at least one unsupported choice.

## 4. Gate decision

| Evidence | N24 requirement | Result |
|---|---|---|
| [directly_supported] | Pre-cutoff published equation | PASS — 2007/2022. |
| [directly_supported] | Commercial-use-compatible cited implementation basis | PASS for the 2022 paper text (`CC BY 4.0` on its landing); no third-party code or weights are required merely to state the equation [E3]. |
| [directly_supported] | Exact target height and basis-time fields | PASS. |
| [derived] | Every required physical state supplied | **FAIL** — `u_*`, `z0`, and `L`/surface flux are absent. |
| [contradicts_premise] | Genuinely new rather than local duplicate | Fixed/shear hub transfers FAIL as duplicates; C1 would be new only if executable. |
| [derived] | Zero tunable choices | **FAIL** — any route from the supplied fields to the missing state requires unfrozen diagnostics and failure rules. |
| [near_match_only] | Published applicability to supplied deployment | **FAIL** — homogeneous/flat or marine evidence does not establish the supplied turbine inflow population. |
| [derived] | Train/deployment symmetry | PASS for literal supplied fields; **FAIL** for any observational/reanalysis completion. |
| [derived] | One self-contained executable treatment or one bounded prerequisite | **FAIL** — multiple missing states plus applicability cannot be reduced to one prerequisite. |

- [directly_supported] **Final lane verdict: `FAIL_CLOSED / NONE_READY`.** No candidate is handed to root for execution.

## Evidence register

### Whitelisted local evidence

- [directly_supported] **L1:** `reports/s17_n24_secondary_frontier_intake_predeclaration.json`, SHA-256 `c8f7a8dc3c861aaaf417b952c8411c246949364f8b95e5b874ca3330d716ba24`.
- [directly_supported] **L2:** `inputs/competition/open_wind_236727.zip`, SHA-256 `920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b`; only `data_description.md` and the two training weather CSV header lines were inspected.
- [directly_supported] **L3:** `src/baram/data/turbines.py`, SHA-256 `85051ef49d5820697f92dfd4ba1311fbdf7b91753a0d8fe36a636613deb1d6b1`.
- [directly_supported] **L4:** `src/baram/features/weather.py`, SHA-256 `d65a18fdf0b025d1863c783d5997f77824a3347ab87e78a4357fe2afeb2d1905`.
- [directly_supported] **L5:** `src/baram/features/physics.py`, SHA-256 `881443f644acf802c8a222fedc7a0f7062fc2345d44580b01124bbfd9a2358bf`.
- [directly_supported] **L6:** `src/baram/features/geometric.py`, SHA-256 `86e977ed4167c3f9229d0ef017da678cf709fdf39fe774f908a2edc1182406d4`.
- [directly_supported] **L7:** `artifacts/manifests/prepare.json`, SHA-256 `ca4908d7f98639b13da80febcd071cd805369e8e044bb4d7b97f793d2703ca46`; feature-name metadata only.

### External primary/official evidence — 4 URLs

1. [directly_supported] **E1 — DTU Research Database, Gryning et al. (2007), primary paper landing and abstract:** <https://orbit.dtu.dk/en/publications/on-the-extension-of-the-wind-profile-over-homogeneous-terrain-bey/>
2. [directly_supported] **E2 — IOPscience, Gryning et al. (2007), open primary companion paper:** <https://iopscience.iop.org/article/10.1088/1742-6596/75/1/012066>
3. [directly_supported] **E3 — Tellus, Valkonen et al. (2022), peer-reviewed original research with explicit Gryning equations and input diagnostics:** <https://tellusjournal.org/articles/10.16993/tellusa.43>
4. [directly_supported] **E4 — American Meteorological Society Glossary, official Obukhov-length definition:** <https://glossary.ametsoc.org/wiki/obukhov-length/>
