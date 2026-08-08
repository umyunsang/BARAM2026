# S17-N25 — ICING_AVAILABILITY_PHYSICS

**Verdict: FAIL_CLOSED / NO CANDIDATE.** No coefficient-complete published turbine-icing or availability correction was found that is computable from the literal supplied issued fields, applies directly to **both VESTAS V126 and UNISON U136**, and avoids fitting, empirical thresholds, cloud-liquid/drop-size assumptions, and turbine operating-state inputs. `[derived]`

## Bounded audit

- Read only the predeclaration and the whitelisted local metadata/code: `artifacts/manifests/prepare.json` (`feature_names` only), `src/baram/features/weather.py`, `src/baram/features/physics.py`, and `src/baram/data/turbines.py`. No weather/data rows, labels, results, scores, test/2024 values, or receipts were read. No fit, threshold search, dependency, or download was performed. `[directly_supported]`
- The frozen turbine contract is a 117 m fleet whose exact model set is `{VESTAS V126, UNISON U136}` (rotor diameters 126 m and 136 m). `[directly_supported]`
- Four primary-paper/official URLs were inspected (listed below), within the lane cap. `[directly_supported]`

## Closest published correction and why it is not executable

The closest end-to-end published candidate is the Davis et al. production-loss chain: NWP hydrometeors feed the physical **iceBlade** ice-mass model; a decision tree plus two generalized additive models (GAMs) then map icing state/mass and wind speed to power loss. `[directly_supported]`

1. **It is fitted, not a fixed coefficient transfer.** The peer-reviewed paper describes a *statistical modeling approach*, develops observed icing power loss from turbine power observations, and explicitly describes the model as being fit to individual wind parks or to six wind parks. The thesis states that the transfer is a decision tree plus two GAMs, with the icing branch using wind speed, total ice mass, and accumulated ice mass. This violates the zero-fit requirement; a literal published, frozen coefficient vector transferable to this fleet was not directly established. `[contradicts_premise]`
2. **The physical prerequisite needs unavailable hydrometeors.** iceBlade uses the Makkonen accretion structure `dM/dt = alpha1*alpha2*alpha3*omega*A*v`. Its implementation obtains `omega` from cloud-water and rainwater mixing ratios and requires cloud-droplet median volumetric diameter (MVD), or droplet concentration plus cloud liquid-water content (LWC), to determine collision efficiency. The paper prescribed MVD/droplet-concentration sensitivity cases when those quantities were unavailable. `[directly_supported]`
3. **Literal issued-field mismatch.** Relevant supplied names include temperature, relative/specific humidity or dew point, wind, pressure, cloud fraction, surface precipitation/rate, and snow-related fields (for example `gfs_spatial__idw__heightAboveGround_2_2t`, `gfs_spatial__idw__heightAboveGround_2_2r`, `gfs_spatial__idw__surface_0_prate`, `ldaps__heightAboveGround_2_q__*`, and `ldaps__surface_0_avg_lsprate__*`). No literal feature is cloud LWC, cloud/rainwater mixing ratio, MVD, or droplet number concentration. Surface precipitation, humidity, and cloud fraction cannot be silently substituted for those distinct quantities. `physics.py` derives only wind/shear, dry-air density, and `rho*v^3`-type proxies; it does not create icing hydrometeors or ice state. `[directly_supported]` `[derived]`
4. **It contains empirical/site choices and unknown operating physics.** The iceBlade paper removes all ice after an above-freezing temperature-duration rule that it says was tested at that location and may need modification elsewhere; it also converts ice mass to a binary event with a fixed mass threshold. Blade-relative velocity uses a generic RPM curve because the studied turbine RPM data were unavailable, and shedding assumes operation on an idealized power curve. These are expressly disallowed threshold/operating-state assumptions. `[directly_supported]` `[contradicts_premise]`
5. **Exact fleet applicability fails.** The physical paper studies Vestas **V90** turbines and, because their airfoil information was unavailable, substitutes cylinder/leading-edge geometry from the NREL 5 MW reference turbine. That is not a V126 correction and provides no U136 applicability. The six Scandinavian wind-park paper's official record does not establish that either V126 or U136—much less both—was represented. `[directly_supported]` `[unverified]`
6. **The standardized availability alternative is ex-post SCADA, not issued weather.** IEA Wind Task 19's official method requires heated nacelle wind speed, nacelle temperature, turbine output power, yaw angle, operational mode, a learned non-iced reference power curve, and consecutive 10-minute observations. Those signals are absent from the issued schema and would be test-period operating observations, so this method is inadmissible for D-1 inference. `[directly_supported]` `[contradicts_premise]`

## Fail-closed gate

| Requirement | Evidence | Gate |
|---|---|---|
| One frozen published coefficient/formula | Closest power-loss transfer is a fitted decision-tree/GAM system; frozen transferable coefficients not established | **FAIL** |
| Computable from literal issued fields | LWC/cloud-water, MVD/drop concentration, ice state, and operating inputs are absent | **FAIL** |
| No empirical thresholds or analyst choices | Published iceBlade uses site-tested shedding/event rules and prescribed microphysics cases | **FAIL** |
| Exact V126 applicability | Audited physical model is V90 plus NREL-reference geometry | **FAIL** |
| Exact U136 applicability | No direct paper evidence | **FAIL_CLOSED** |
| D-1 deployment symmetry | Standard availability method needs ex-post SCADA/operational mode | **FAIL** |
| Commercial anonymous reproducibility | No admissible executable correction survives the prior gates; licence/runtime therefore not promoted | **UNVERIFIED / FAIL_CLOSED** |

**Handoff: none.** A humidity/temperature/precipitation icing flag, an assumed MVD/LWC, a constant derate, or a generic shutdown multiplier would invent precisely the thresholds or coefficients prohibited by this lane. `[derived]`

## Primary / official sources (4 URLs; accessed 2026-08-09 KST)

1. Davis, Hahmann, Clausen & Žagar, *Forecast of Icing Events at a Wind Farm in Sweden*, JAMC 53 (2014), full primary article: https://journals.ametsoc.org/view/journals/apme/53/2/jamc-d-13-09.1.xml
2. Davis et al., *Identifying and characterizing the impact of turbine icing on wind farm power generation*, Wind Energy 19 (2016), official DTU record and DOI metadata: https://orbit.dtu.dk/en/publications/identifying-and-characterizing-the-impact-of-turbine-icing-on-win/
3. Davis, *Icing Impacts on Wind Energy Production* (PhD thesis), official DTU record: https://orbit.dtu.dk/en/publications/icing-impacts-on-wind-energy-production/
4. IEA Wind TCP Task 19, official `T19IceLossMethod` specification and required SCADA signals: https://iea-wind.org/task19/t19icelossmethod/
