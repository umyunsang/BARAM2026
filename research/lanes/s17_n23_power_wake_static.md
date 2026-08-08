# S17-N23 power-curve / wake / static-physics frontier intake

## 0. Verdict

- **Verdict: `BLOCKED_NO_EXACT_LICENSED_ALL_FLEET_POWER_CT_WAKE_REPRESENTATION`.** No candidate is `READY`, and this lane authorizes no prerequisite acquisition or executable treatment. [derived]
- [directly_supported] The competition metadata identifies all 17 turbine locations, manufacturers, base model names, hub heights, rotor diameters, individual ratings, and group membership. Geometry is not the blocker.
- [contradicts_premise] Exact static turbine geometry is already represented locally; a generic kinetic-power proxy is already present; an empirical power-curve axis and a parameterized wake-sector axis have already been encoded. Repackaging any of those is not a new representation.
- [directly_supported] The only exact-OEM public curve found is the U136 page's displayed power table, but it covers only group 3, has an internal operating-range contradiction, provides no thrust curve, has no affirmative commercial-reuse licence, and has no pinned pre-cutoff publication receipt for the exact table. Vestas publishes the existence of a 3.6 MW Power Optimised Mode, but not a configuration-specific public power or thrust table at the reviewed official sources.
- [directly_supported] Permissively licensed NREL/FLORIS/PyWake assets were available before `2026-07-05`, but none contains VESTAS V126-3.6-POM or UNISON U136-4.2 power **and** thrust definitions. Their documentation instead exposes the missing choices: turbine-specific power/`C_T` curves, wake expansion, superposition, rotor averaging, and inflow/turbulence inputs.
- [derived] Consequently, there is no exact zero-variant formula whose inputs are all identified, chronology-safe, commercially eligible, and symmetric at deployment. Supplying a reference-turbine curve, `C_T=0.8`, `k=0.075`, a hub-wind interpolation, or an OEM sub-variant would invent a manufacturer/configuration, coefficient, or transform and fails the intake contract.

Audit time: `2026-08-09 00:04 KST`. Local bound: 20 logical files/metadata entries (limit 40). External evidence: 6 primary/official packages and exactly 10 canonical source URLs (limit 10). No secondary page is used as evidence. [directly_supported]

## 1. Competition metadata: identity, specification, and layout

### 1.1 Frozen fleet contract

The immutable archive (`open_wind_236727.zip`, SHA-256 `920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b`) was opened only for `data_description.md`, the ZIP directory, and the two worksheet XML members needed to read `info.xlsx`; no CSV row was opened. [directly_supported]

| KPX group | Exact supplied turbine rows | Manufacturer / model text | Count | Hub | Rotor | Per-turbine rating | Group rating |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | VESTAS WTG01–06 | `VESTAS` / `V126` | 6 | 117 m | 126 m | 3.6 MW | 21.6 MW |
| 2 | VESTAS WTG07–12 | `VESTAS` / `V126` | 6 | 117 m | 126 m | 3.6 MW | 21.6 MW |
| 3 | UNISON WTG01–05 | `UNISON` / `U136` | 5 | 117 m | 136 m | 4.2 MW | 21.0 MW |

`info.xlsx` writes the group ID only at turbine-row boundaries; the validated topology forward-fills boundaries `(0,6,12)`. This is the repository's explicit contract in `src/baram/data/turbines.py` (SHA-256 `85051ef49d5820697f92dfd4ba1311fbdf7b91753a0d8fe36a636613deb1d6b1`), not an inferred grouping. [directly_supported]

### 1.2 Exact supplied coordinates

| Group | Turbine | `좌표(Google)` exactly as supplied |
|---:|---|---|
| 1 | VESTAS-01 | `37°16'55.61"N 128°57'02.10"E` |
| 1 | VESTAS-02 | `37°17'04.05"N 128°56'58.35"E` |
| 1 | VESTAS-03 | `37°17'11.49"N 128°56'58.99"E` |
| 1 | VESTAS-04 | `37°17'23.11"N 128°57'03.68"E` |
| 1 | VESTAS-05 | `37°17'28.20"N 128°57'15.58"E` |
| 1 | VESTAS-06 | `37°17'19.48"N 128°57'24.96"E` |
| 2 | VESTAS-07 | `37°17'16.20"N 128°57'34.67"E` |
| 2 | VESTAS-08 | `37°17'11.29"N 128°57'47.24"E` |
| 2 | VESTAS-09 | `37°17'00.97"N 128°57'57.44"E` |
| 2 | VESTAS-10 | `37°16'52.77"N 128°58'04.18"E` |
| 2 | VESTAS-11 | `37°16'44.89"N 128°58'01.12"E` |
| 2 | VESTAS-12 | `37°16'30.58"N 128°58'02.54"E` |
| 3 | UNISON-01 | `37°16'59.73"N 128°57'44.97"E` |
| 3 | UNISON-02 | `37°16'40.41"N 128°58'13.80"E` |
| 3 | UNISON-03 | `37°16'28.03"N 128°58'22.54"E` |
| 3 | UNISON-04 | `37°16'18.58"N 128°58'29.01"E` |
| 3 | UNISON-05 | `37°16'06.83"N 128°58'35.68"E` |

A metadata-only haversine audit using the repository's fixed mean-Earth radius `6,371,008.8 m` gives: [derived]

| Geometry slice | Minimum pair distance | Maximum pair distance | Minimum in mean rotor diameters |
|---|---:|---:|---:|
| within group 1 | 230.341 m | 1,059.731 m | 1.828 D |
| within group 2 | 254.747 m | 1,566.740 m | 2.022 D |
| within group 3 | 332.396 m | 2,055.040 m | 2.444 D |
| group 1 ↔ 2 | 259.231 m | 2,173.719 m | 2.057 D |
| group 1 ↔ 3 | 783.546 m | 3,265.534 m | 5.981 mean-D |
| group 2 ↔ 3 | 308.842 m | 2,615.227 m | 2.358 mean-D |

The closest cross-group pairs are VESTAS-06↔VESTAS-07 (259.231 m) and VESTAS-09↔UNISON-01 (308.842 m). This proves only that an all-17 layout is geometrically identifiable and that group boundaries are not physical isolation boundaries; it does **not** prove a wake or its magnitude. [derived]

## 2. Current repository feature coverage and nonduplication

| Surface | Directly verified coverage | Intake consequence |
|---|---|---|
| Static turbine block | `group_static_metadata()` already emits `turbine_count`, centroids, hub height, rotor diameter, turbine/group rating, per-turbine swept area, and fleet swept area. | [contradicts_premise] Basic turbine static physics is not missing. |
| Current prepare manifest | `artifacts/manifests/prepare.json` (SHA-256 `ca4908d...`) has 820 feature names: the nine numeric turbine/static fields above, 12 `phys*` fields, one `phys_v2__fleet_power_proxy_w`, and zero literal `wake`, `curve`, `layout`, or `terrain` names. | [directly_supported] The manifest itself has no OEM curve/wake column, but exact input constants are already exposed. |
| Static physics | `src/baram/features/physics.py` computes dry density, 80–100 m shear, a guarded 117 m speed, `rho*v^3`, and `0.5*rho*v^3*fleet_swept_area`. | [contradicts_premise] Betz/kinetic-flux or swept-area repackaging is duplicate information. |
| Spatial geometry | `spatial.py` uses every turbine coordinate for per-turbine nearest/IDW weights and then group-averages them. `geometric.py` adds group principal-axis projections, vector spread/coherence, and grid gradients. | [directly_supported] Coordinates are used, although pairwise turbine wakes are not part of the tracked prepare manifest. |
| Wake-sector experiment | `run_wake_sector_classifier.py` (SHA-256 `11c65c...`) already forms every ordered **within-group** pair, rotates separations into forecast wind, and uses `R=D/2+kx`, a distance decay, a Gaussian crosswind factor, exposure summaries, and six alternative wind vectors. `k` is a CLI degree of freedom in `[0.02,0.20]` (default `0.075`), and an M168 wake artifact exists by filename. | [contradicts_premise] Another cone/sector/expansion feature—including an all-17 or cross-group repair—is a variant of an encoded axis, not a clean new information source. The historical formula is also not an OEM wake model: it has no turbine `C_T` curve and retains selectable coefficients/levels. |
| Empirical curve experiment | `research/scratch/powercurve.py` (SHA-256 `fcd2c9...`) defines and optimizes a four-parameter per-group curve from historical SCADA/target data. Only source text was inspected; no input or parameter-value file was opened. | [contradicts_premise] A learned curve axis already exists, while it is ineligible for this zero-fit/public-static intake. |
| Exact static ratios | From supplied values, swept areas are 12,468.981 m² (V126) and 14,526.724 m² (U136); specific powers are 288.716 and 289.122 W/m². Both numerator and denominator are existing columns and are group constants. | [derived] Their ratio adds no new row information and is group-dummy-equivalent. |
| Terrain interaction | N21/N22's fixed mean-16 `Sx` treatment is already adjudicated non-promotable; N23 explicitly forbids retuning it. | [contradicts_premise] `wake × Sx`, a new radius, or group-specific rescue cannot enter through this lane. |

Thus the genuine local gap is narrower than “wake is absent”: an **exact, configuration-matched, commercially eligible power+thrust definition and coefficient-complete farm-wake transform** is absent. [derived]

## 3. Primary / official source audit (10 canonical URLs)

### P1 — UNISON U136 official product page

1. <https://www.unison.co.kr/product/4MW_Platform_U136>

- [directly_supported] The page matches group 3 exactly on `U136`, 4,200 kW, 136 m rotor, and the 117 m hub option; it states rated wind 11.3 m/s and operating wind 3–22 m/s.
- [directly_supported] Its embedded `chart_1` contains a one-metre-per-second power table with non-null entries from 3 through 25 m/s and nulls at 26–27 m/s. That conflicts with the same page's stated 3–22 m/s operating range, and the page specifies neither an interpolation rule nor reference air-density/normalisation semantics.
- [directly_supported] Raw HTML retrieved `2026-08-09 KST`, 25,326 bytes, SHA-256 `c985737738d4ac6c7a727a19746d0efeef3ae31743c316d863dbf31c4580439e`; footer: `©UNISON Co., Ltd. 2020 All rights reserved`.
- [unverified] The footer year proves neither the exact curve table's publication date nor pre-cutoff byte identity. No affirmative commercial reuse/data licence was found. Under the project fail-closed licence rule, public visibility is insufficient.
- [near_match_only] Even if the table were eligible, it is power-only, group-3-only, and lacks `C_T`; it cannot complete an all-fleet power or wake representation.

### P2 — Vestas official V126 product and Power Optimised Mode chronology

2. <https://www.vestas.com/en/energy-solutions/onshore-wind-turbines/4-mw-platform/V126-3-45-MW>  
3. <https://www.vestas.com/en/media/company-news/2018/new-72-mw-order-introduces-the-v126-3-45-mw-in-china-c2963363>

- [directly_supported] The product page describes `V126-3.45 MW`, a 126 m rotor, a 117 m hub option, 3 m/s cut-in, 22.5 m/s cut-out, and optional Power Optimised Mode. It publishes no tabular power curve or thrust curve.
- [directly_supported] The dated Vestas release (`2018-07-06`) proves that V126-3.45 turbines can be delivered in a **3.6 MW Power Optimised Mode**.
- [near_match_only] The supplied workbook says only `V126` and 3.6 MW. It does not name `V126-3.45`, the exact Power Optimised Mode revision, noise/load mode, high-wind option, control software, or site-specific curve. The 2018 release proves such a mode exists, not that these 12 machines use the identical curve/configuration.
- [unverified] The current product HTML is all-rights-reserved and the retrieved response was not a pre-cutoff pinned asset. Neither official page grants commercial reuse of a curve dataset; no curve dataset is present in any event.

### P3 — NREL/National Laboratory turbine-model archive

4. <https://natlabrockies.github.io/turbine-models/>  
5. <https://github.com/NatLabRockies/turbine-models/tree/e9e4ccc6a87f50c0c65d236154a2897e752c47c9>

- [directly_supported] Commit `e9e4ccc6a87f50c0c65d236154a2897e752c47c9` is dated `2026-04-03`, before the cutoff, and the repository is BSD-3-Clause (commercial-use compatible with notice conditions).
- [directly_supported] Its purpose is public tabular power and, when available, thrust curves. The pinned tree contains Vestas V27/V29/V47/V82 entries, but zero `V126`, `U136`, or `UNISON` path.
- [contradicts_premise] `NREL_5MW_126_RWT` shares a 126 m diameter but is a 5 MW reference turbine, not a Vestas V126 in 3.6 MW mode. Substituting it would invent turbine identity and control physics. The archive itself warns that ideal/reference curves can deviate in practice.

### P4 — FLORIS turbine/wake framework

6. <https://natlabrockies.github.io/floris/turbine_models.html>  
7. <https://github.com/NatLabRockies/floris/releases/tag/v4.6.6>

- [directly_supported] Release `v4.6.6`, commit `6b27f3acadd6d1e37748e639b23091b027a8478a`, was published `2026-06-25`, before the cutoff; FLORIS is BSD-3-Clause.
- [directly_supported] The pinned documentation blob (`docs/turbine_models.ipynb`, blob `53c6e3b1...`) says FLORIS represents an actuator disk with a wind-speed-indexed **power curve and thrust-coefficient curve**. Its prepackaged models are reference turbines, and the pinned tree contains no V126/U136/Vestas/Unison definition.
- [near_match_only] FLORIS supplies a permissive engine, not the missing OEM inputs. Running its default NREL 5 MW turbine, default turbulence, shear, wake model, or superposition would be a model substitution, not direct support.

### P5 — primary Gaussian wake paper

8. <https://doi.org/10.1016/j.renene.2014.01.002>

- [directly_supported] Bastankhah & Porté-Agel, *Renewable Energy* 70 (2014), derives a Gaussian deficit from mass/momentum conservation and validates it on wind-tunnel/LES cases. It predates the cutoff.
- [near_match_only] It is a deficit-model paper, not a Taebaek farm turbine definition. A paper equation does not supply V126/U136 `C_T`, power curves, inflow at each rotor, or a unique multi-wake farm assembly.
- [unverified] The publisher page supplies scholarly access/citation, not an affirmative software/data licence. A permissively licensed independent implementation exists (P6), so software licensing is not the decisive blocker; missing turbine/site inputs are.

### P6 — DTU PyWake literature implementation and its explicit ambiguity audit

9. <https://topfarm.pages.windenergy.dtu.dk/pywake/notebooks/literature_verification/Gaussian.html>  
10. <https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake/-/tags/v2.6.11>

- [directly_supported] PyWake `v2.6.11` is dated `2025-03-27`, before the cutoff, and is MIT licensed. The pinned tree contains Vestas V80/V112 examples but no V126 or U136/Unison definition.
- [directly_supported] The pinned Gaussian notebook (blob `cb4472aa...`) states that the 2014 model requires the user to provide wake expansion `k*`; its validation values were fitted to LES. The example also supplies a dummy `C_T=0.8`, not an OEM constant.
- [directly_supported] For the later wind-farm reproduction, the DTU authors explicitly report that the exact Vestas V80 power/thrust curves were unclear, that alternatives had to be tried, and that rotor-reference/averaging choices affect results. This directly refutes treating a published deficit equation or library default as a unique farm-power representation.
- [near_match_only] MIT code is commercially usable, but code availability cannot identify absent `C_T`, power tables, ambient turbulence, wake expansion, rotor averaging, superposition, yaw/control state, or the correct forecast wind vector for this site.

## 4. Exact-representation gate

The following is a **dependency schematic, not an admitted formula**:

`issued NWP → turbine-level free-stream rotor wind → OEM power/C_T lookup → pairwise wake deficits from all 17 coordinates → fixed rotor averaging and wake superposition → group aggregation`.

Every arrow must be unique before `READY`. The audit result is: [derived]

| Required input/choice | Supplied or directly licensed? | Gate |
|---|---|---|
| 17 identities, coordinates, `D`, hub, rating, group | Yes, exact competition metadata | PASS |
| Vestas exact 3.6 MW mode/sub-variant | Base model/rating only; official 3.6-POM existence is a near match | BLOCK |
| Configuration-specific V126 power curve | No official table found | BLOCK |
| Configuration-specific V126 `C_T` curve | No | BLOCK |
| U136 power curve | Public table exists, but licence/chronology/interpolation/range are unresolved | BLOCK |
| U136 `C_T` curve | No | BLOCK |
| Turbine-level hub/rotor-equivalent free-stream wind | NWP levels exist, but no OEM-prescribed unique mapping; the local 117 m proxy has guards/clips and is already encoded | BLOCK |
| Ambient turbulence / unique `k*` | No directly supplied TI and no site-fixed published value | BLOCK |
| Multi-wake superposition, rotor averaging, yaw/control, cross-group handling | Multiple legitimate implementations; none frozen by the OEM/source | BLOCK |
| Commercial-use-compatible external asset licence | Permissive frameworks yes; exact OEM curve assets no | BLOCK |
| Exact pre-cutoff availability | Frameworks/paper yes; exact U136 table bytes and Vestas curve no | BLOCK |

There is therefore **no exact zero-variant formula to print or authorize**. Printing one would conceal at least one invented item. [derived]

## 5. Closed-axis contradictions and falsification

1. **“The manufacturer is unknown.”** False: base manufacturer/model are supplied. **But** extending `V126`/3.6 MW to a precise V126-3.45 POM revision and curve is not supplied. [directly_supported] [near_match_only]
2. **“A same-diameter open reference turbine is close enough.”** False: NREL 5 MW/126 m is not Vestas 3.6 MW/126 m; rotor diameter does not identify power or thrust control. [contradicts_premise]
3. **“The wake paper fixes the coefficients.”** False for deployment: PyWake's pinned reproduction requires user `k*`, uses dummy `C_T`, and documents fitted/ambiguous choices. [contradicts_premise]
4. **“Use onshore `k=0.075`.”** This would both invent a site coefficient and reproduce the existing M168 default/variant axis. [contradicts_premise]
5. **“Use only within each KPX group.”** The supplied layout has cross-group separations as small as 2.06–2.36 mean rotor diameters. Ignoring or including them is a consequential assembly choice not resolved by group labels. [derived]
6. **“Use the U136 table for group 3 and leave groups 1/2 unchanged.”** That is a subgroup treatment, lacks a qualifying licence/chronology/interpolation rule, and fails all-row symmetry. [contradicts_premise]
7. **“Add swept area, specific power, or `rho v^3`.”** Already encoded or algebraically recoverable from existing numeric fields; all exact static values are group constants. [contradicts_premise]
8. **“Fit the missing curve/coefficient from SCADA.”** A fitted empirical curve already exists, but fit/tuning is forbidden here and does not provide future operating state or OEM `C_T`. [contradicts_premise]
9. **“Repair the terrain result with wake×terrain/group/radius variants.”** N23 forbids mean-16 retuning and demands genuinely new information. [contradicts_premise]

### Minimal evidence that could unlock a later lane (not available here)

A later intake would need, before any formula is admitted: (a) OEM/site documentation proving the exact Vestas power mode; (b) pre-cutoff, affirmatively commercial-use-compatible V126-3.6 and U136-4.2 power **and** `C_T` tables with reference-density, interpolation, cut-in/out/hysteresis semantics; and (c) one published, coefficient-complete farm assembly whose TI/wake-expansion, wind-vector height, rotor average, superposition, yaw, and cross-group rules are all computable from issued inputs. [derived]

Absent that bundle, the axis remains closed as `BLOCKED`, not `READY_PREREQUISITE`. [derived]

## 6. Bound and forbidden-access accounting

- [directly_supported] Local logical files/metadata entries read: 20/40. They were: archive `data_description.md` and `info.xlsx` worksheet XML; three prior S17 lanes; N23 predeclaration/event; `prepare.json`; six `src/baram` metadata/feature modules; two feature-research lanes; two planning source scripts; and `research/scratch/powercurve.py` source. No generated row body was opened.
- [directly_supported] External evidence: 6 official/primary packages, 10 canonical URLs. Repository tree/tag/licence/documentation metadata were inspected within those packages; no turbine CSV, raster, model weight, forecast, or dataset body was acquired or written.
- [directly_supported] Zero NWP/weather values, SCADA values, labels/targets, `actual_kwh`, operating-2024 values, test values, generated predictions, rejected/quarantined artifacts, or lockbox values were read.
- [directly_supported] Zero fit, predict, policy, optimizer, score, or metric calls; only static coordinate/area/distance arithmetic was performed.
- [directly_supported] Zero dependency changes, model/data downloads, API inference, Dacon/account action, or repository writes other than this designated file.

**Final verdict: `BLOCKED_NO_EXACT_LICENSED_ALL_FLEET_POWER_CT_WAKE_REPRESENTATION`.**
