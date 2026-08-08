# S17-N28 — TradeRES full-text scalar-gradient reopen audit

## Verdict

**FAIL-CLOSED / NO PREREQUISITE CANDIDATE.** The newly retrieved full texts materially strengthen the documentary record: Couto & Estanqueiro (2022) gives a literal first-order centred finite-difference `MSLPGrad` equation on a projected WRF grid and evaluates the last 24 hours of a 48-hour 00 UTC run; TradeRES D4.9 edition 2 gives literal pressure-gradient tendency and ramp-tracking equations. They still do not provide one locally mapped, zero-choice treatment with an isolated validated day-ahead wind-power effect. N27's reopen condition is therefore not cured.

No supplied weather value, label value, 2024/test value or result aggregate was read. No fit, prediction, policy, project metric, dependency, external inference, Dacon or account action occurred.

## Fixed source receipts

1. **Couto & Estanqueiro (2022), Renewable Energy 201, 1076–1085**, DOI `10.1016/j.renene.2022.11.022`.
   - Official LNEG repository object: <https://repositorio.lneg.pt/bitstream/10400.9/4016/2/RenewableEnergy_Vol.201_1076-1085.pdf>
   - Public text retrieval used for this audit: <https://r.jina.ai/https://repositorio.lneg.pt/bitstream/10400.9/4016/2/RenewableEnergy_Vol.201_1076-1085.pdf>
   - Retrieved text bytes: `56,493`; SHA-256 `ed3d342c63eb55751f9e69bd55e048b2b24a97b178e49964d99540838a40f3dd`.
   - Publication date: 12 November 2022; repository metadata marks the paper open access under CC BY-NC-ND 4.0. The paper is evidence only; no attachment, code, data or weights are redistributed or used at deployment.

2. **TradeRES Deliverable D4.9 edition 2, _New forecast tools to enhance the value of VRE on the electricity markets_**.
   - Official European Commission document endpoint: <https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e51750507a&appId=PPGMS>
   - Edition date 26 January 2024; `Ares(2025)354641 - 16/01/2025`; public dissemination level.
   - Retrieved PDF bytes: `3,948,209`; SHA-256 `f58e65e25870ec4a1f3833b0ec3a0897bde36c8f9cb3d7c52f90f1efc220c5f6`.
   - The PDF body was fixed before this audit and was not written into the repository.

Both sources predate the competition cutoff of 2026-07-05.

## Treatment A — 2022 `MSLPGrad` feature

### What is directly fixed

Couto & Estanqueiro section 2.1.1 directly states:

- MeteoGalicia WRF, 12 km projected grid, initialized at 00 UTC, 48-hour horizon;
- the last 24 hours are evaluated because they support Iberian-market bids;
- `MSLPGrad` is computed from mean-sea-level pressure at the same grid points using first-order centred finite differences:
  `dMSLP/dx ≈ (MSLP[i+1,j]-MSLP[i-1,j])/(2*dx)` and
  `dMSLP/dy ≈ (MSLP[i,j+1]-MSLP[i,j-1])/(2*dy)`;
- PCA is applied separately to each meteorological variable after grid-point z-score normalization, retaining the minimum number of components above 90% explained variance;
- the complete method ranks PCs against observed park power, runs site-specific sequential-forward selection, and feeds an ANN.

The full text supplies real target-horizon relevance. `MSLPGrad PC#1` appears in the top-five Table 2 ranking for wind parks 1, 2 and 4. The full selected multifeature system improves NRMSE by 13–37% over its nearest-point wind-vector benchmark.

### Why this still cannot be transferred as one candidate

1. **The effect is not isolated.** No `MSLPGrad`-only ablation is reported. The 13–37% belongs to a wrapper-selected PCA/ANN suite of 11–20 site-specific components. The paper explicitly reports that each park needed different inputs and that transferring one park's final set to another performed worse.
2. **The local source is not fixed.** The published treatment is a single 12 km WRF domain. The supplied archive offers GFS and LDAPS fields with different grids and semantics. Choosing GFS, LDAPS, both, or a cross-source construction remains an analyst bridge.
3. **The local domain/output is not fixed.** The equation defines an oriented vector field at centred interior cells. It does not prescribe boundary handling or how a small supplied grid becomes one downstream feature: east/north components, magnitude, PCA scores, site interpolation, group aggregation or another summary are materially different outputs.
4. **The validated pipeline is not strict prequential.** Its calibration uses randomly withheld months and supervised distance-correlation/SFFS; replacing that with strictly preceding issuance folds is necessary here but is a new treatment whose cited effect is not established.
5. **The objective differs.** The paper optimizes NRMSE. It explicitly leaves market-remuneration objectives to future work, while BARAM's official action score combines capacity-relative MAE with discontinuous settlement bands.

The direct formula therefore cures formula existence but not the one-treatment, mapping and isolated-effect gates.

## Treatment B — 2024 TradeRES ramp detector

D4.9 section 4.2.1 directly prescribes:

- a centred spatial gradient of forecast mean-sea-level pressure;
- `d||grad(P)||/dt ≈ (||grad(P)||[t]-||grad(P)||[t-1])/dt` to add memory;
- negative/positive events below the 2nd or above the 98th cosine-latitude-weighted spatial percentile;
- contiguous-region/convex-hull tracking, minimum area 150,000 km², maximum consecutive-centre distance 720 km, and lifetime/speed filters of 2 h / 120 km/h.

This is coefficient-complete for its broad-domain **alert** purpose, but it is not validated evidence for a BARAM deterministic action feature:

- the same section says ramp forecast accuracy **“will be assessed and analysed”** in WP5; it reports no accuracy or isolated day-ahead power-score effect;
- the report describes ramp output as binary information that complements, and cannot substitute for, deterministic forecasts;
- the spatial domain and mapping to a local park/group are not prescribed for the supplied GFS/LDAPS grids, and the 150,000 km² event-area contract is not a local-grid reduction;
- converting the alert into a prediction correction, class feature, policy switch or settlement action would introduce a new learned/analyst mapping.

Thus the exact alert formula fails the validated-effect and target-action gates.

## Spatial-PCA alternative exposed by the full paper

The paper also isolates a `Spatial–PCA` scenario: only the benchmark wind speed and `u/v` variables are expanded from a nearest WRF point to spatial PCA scores, retaining 90% variance and using the same ANN. It reduces NRMSE, although three of seven parks improve by less than 5%; the TradeRES report summarizes the average PCA-only improvement as about 8%.

This does not authorize a new local PCA candidate:

- its information gain comes from the paper's much larger 12 km WRF domain, whereas the fixed supplied representation contains only its provided local GFS/LDAPS grid supports;
- current project inputs already expose raw per-grid wind fields and fixed spatial representations; PCA would be a new compression/regularization choice, not a new issued information source;
- the paper fixes neither a transfer to the supplied grids nor an official-Total action model, and its direct effect remains against a limited nearest-point wind-vector benchmark rather than the current Champion.

It is therefore `[near_match_only]`, not a cure for N27.

## Admission-gate adjudication

| N27 gate | Full-text result |
|---|---|
| Direct pre-cutoff formula/mechanism | **PASS** — both centred-gradient formulas are literal |
| Literal supplied field availability | **PASS for GFS PRMSL availability; incomplete transfer semantics for LDAPS** |
| Same-issuance computability | **PASS in principle** — no future observation is required |
| One deterministic source/grid/projection/output with no analyst choice | **FAIL** |
| Operation-level novelty | **PASS only for the scalar-gradient core** |
| Plausible target-scale effect for the same treatment | **FAIL** — suite-selected or explicitly unevaluated |
| Unchanged runtime / no external deployment dependency | **PASS for arithmetic** |

Because the gates are conjunctive, ready candidate count is **0**. No N29 prerequisite, implementation, fit or comparison index is emitted. Comparison count remains 4 and index 5 remains reserved.

## Scope incident and containment

During a bounded fixed-source inspection after predeclaration, one printed source header contained the identifier `actual_kwh`. No value, row, aggregate, score, prediction, 2024/test datum or artifact body was exposed. The output was unnecessary and is excluded from the adjudication. The decisive failure above follows entirely from the two fixed public documents and the already-closed N27 source/schema record. This audit consequently closes fail-closed and does not promote a candidate.
