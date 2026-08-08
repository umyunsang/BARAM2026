# S17-N27 — SCALAR_GRADIENT_PHYSICS

## Verdict

**FAIL_CLOSED / NO ADMISSIBLE CANDIDATE.** A pressure-gradient/geostrophic signal has direct physical and short-horizon forecasting support, and a 48-hour WRF wind-power study includes an `MSLPGrad` feature. The audited sources do **not** jointly prescribe the same coefficient-complete treatment for issued 12–36 h NWP, isolate its target-scale effect, and remove site/grid/level/estimation choices. The bridge would therefore be `[derived]`, not executable `[directly_supported]` evidence.

## Direct evidence and boundary

1. **Zhu, Bowman & Genton (2014), Annals of Applied Statistics.** `[directly_supported]`
   - The paper states that raw pressure and temperature did not improve its TDD model, while geostrophic wind did.
   - It gives a literal construction: convert station pressure to a reference-pressure geopotential height
     `Z = Zi + (R*Tbar/g0) * log(pi/pref)`; fit `Z(x,y)=a0+a1*x+a2*y`; then `ug=-(g0/f)*a2`, `vg=(g0/f)*a1`, `wg=hypot(ug,vg)` with `f=2*Omega*sin(latitude)`.
   - Its instantiated treatment fixes `pref=850 hPa`, averages temperature over 12 stations, removes monthly average geopotential height, and fits one plane per hour. The downstream model chooses geostrophic-wind lags by modified BIC and uses a 45-day moving fit.
   - It reports 5.3–8.2% MAE reduction against the best prior space–time methods for a two-hour forecast, but the experiment predicts 10 m winds 1–6 h ahead from contemporaneous surface-station observations in West Texas. `[contradicts_premise]` This is not an issued 12–36 h NWP treatment, not a 117 m complex-terrain power forecast, and not deployment-symmetric under the project's test-observation ban.

2. **Couto & Estanqueiro (2022), Renewable Energy.** `[directly_supported]`
   - The official repository abstract reports 13–37% RMSE reduction for wind-power forecasts from a WRF-derived meteorological feature suite plus an artificial neural network, over seven Portuguese wind parks.
   - Indexed primary-publisher text states that every WRF run begins at 00 UTC and covers 48 h, and explicitly names `MSLPGrad` as “mean sea level pressure gradient”. This establishes a day-ahead issued-NWP near match and target-scale relevance.
   - `[contradicts_premise]` The gain belongs to a site-specific sequential-forward-selected feature/PC system, not an isolated `MSLPGrad` ablation. The abstract says every park required specific meteorological inputs. The inspected official text does not prescribe the exact finite-grid `MSLPGrad` estimator, grid subset, vector/magnitude output, or transfer rule. Therefore its 13–37% cannot be assigned to the proposed single diagnostic.

3. **Browell, Drew & Philippopoulos (2018), Wind Energy.** `[directly_supported]`
   - Surface wind and sea-level-pressure fields plus 500 hPa geopotential from MERRA-2 were clustered into atmospheric modes; 1–6 h accuracy improved at all sites, averaging 3.1% at six hours.
   - `[contradicts_premise]` The operation is a tuned self-organizing-map/regime model using reanalysis and recent observations, not a fixed scalar-gradient feature, and the effect is short-horizon and below the required scale.

4. **MFWPN, Nature Communications (2025).** `[directly_supported]`
   - The model fuses gridded wind, temperature, geopotential, and elevation; the paper explicitly connects temperature differences to pressure-gradient forcing and reports better vector-wind performance primarily within six hours.
   - `[contradicts_premise]` It learns spatial/temporal fusion with CNN/Transformer gates from the preceding 24 h of ERA5 reanalysis. The paper says performance decays with horizon and requires retraining/fine-tuning elsewhere. It neither isolates a pressure-gradient diagnostic nor supplies an issued-NWP zero-choice formula.

## Gate table

| Mandatory gate | Result |
|---|---|
| Physical pressure-gradient identity | PASS — Zhu equations are literal `[directly_supported]` |
| Issued 12–36 h wind-power relevance | NEAR MATCH — Couto has 48 h WRF and wind power |
| Same treatment across the two evidentiary links | **FAIL** — geostrophic station construction is not the unreported WRF `MSLPGrad` estimator |
| Isolated plausible target-scale effect | **FAIL** — Couto's gain is the whole selected suite; Zhu is 1–6 h/10 m observations |
| No fitted/site/scale/level choices | **FAIL** — reference level, station/grid domain, monthly centering, lag/feature selection and downstream fit differ |
| Deployment symmetry | **FAIL** for Zhu/Browell/MFWPN observation/reanalysis instantiations |

## Disposition

`[derived]` A fixed pressure-gradient feature is meteorologically sensible and appears novel locally. That is insufficient under N27: direct sources support different treatments and different horizons. Do not create a fit, score, or N28 prerequisite from this lane.

## Primary sources

- Zhu, Bowman & Genton, “Incorporating geostrophic wind information for improved space–time short-term wind speed forecasting,” DOI 10.1214/14-AOAS756; author manuscript: https://arxiv.org/pdf/1412.1915
- Couto & Estanqueiro, “Enhancing wind power forecast accuracy using the weather research and forecasting numerical model-based features and artificial neuronal networks,” DOI 10.1016/j.renene.2022.11.022; official repository record: https://repositorio.lneg.pt/entities/publication/fe74061c-6baa-477d-b1bd-c21a5ea3b4fc
- Browell, Drew & Philippopoulos, “Improved very-short-term wind forecasting using atmospheric regimes,” DOI 10.1002/we.2207; official repository: https://strathprints.strath.ac.uk/63855/
- “A machine learning model for hub-height short-term wind speed prediction,” Nature Communications (2025): https://www.nature.com/articles/s41467-025-58456-4

## Provenance

The delegated lane exceeded its time bound and left no durable artifact. Root independently reconstructed only the bounded primary-source facts above. No competition data, label, score, model result, 2024/test value, fit, prediction, metric, external data body, account, dependency, or Dacon action was used.
