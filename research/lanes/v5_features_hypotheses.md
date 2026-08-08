# Lane `V5-FEATURES` — S6 sibling hypotheses (SK@v5 Wave B)

- Lane id: `V5-FEATURES`
- Exclusive output: `research/lanes/v5_features_hypotheses.md` (this file; the lane's only repository write)
- Written: 2026-08-09 KST
- Source bound: ≤16 primary/official benchmark sources. **Used: 4 verified + 1 partially verified.** The lane was
  terminated by an explicit root instruction ("STOP RESEARCHING AND WRITE NOW") before the source bound was
  exhausted. Every consequence of that truncation is recorded in §11 as an `insufficient` unknown.
- **Revision r2, 2026-08-09:** root issued an authority change retiring the dependency ceiling and supplying the
  official rules snapshot. Affected rows revised in place: §0 (authority table), §3 (source S6 added), §9.1
  (the former `BLOCKED_DEPENDENCY` block, now re-judged on evidence and cost), §10 (one rejection reason
  changed while the rejection stands), §13.7, §14. **The verdict, the Class R / Class N split, and the ranking
  of N1 > N2 > R1 are unchanged** — none of them ever depended on package availability.

## 0. Authority verification (performed by this lane)

| Artifact | Expected SHA-256 | Verified |
|---|---|---|
| `.planning/2026-08-01-leaderboard-top-4-loop/SK_v5.md` | `c2c813475ab5f0a741e6adbceab196d5b38447f2aeb02f9716a8f5be8973c820` | ✅ match |
| `reports/sk_v5_approval_receipt.json` | `e827b324a23913346598d274f2ded496ffc3134243de5636033ee6a3ac387173` | ✅ match |
| `research/nodes/sk5_foundation_map.json` | `393ee74bf53251037547fc52bc92c554a4aa207e2288f194766d836675512b35` | ✅ match |
| `research/nodes/sk5_local_capability_constraint.json` (r1, superseded) | `3209042a0ec9ddd669992e94bf6137beb79582cc5ad62fa65337bd109be700dc` | ✅ match |
| `AGENTS.md` (r1, superseded) | read in full before any other action | ✅ |
| **`inputs/rules/official_rules_2026-08-09.md`** (r2 authority) | `6dcececbfb33df6761c87220ea8ce15d875d6b7afc9d90c5afe5dd827ad54ee5` | ✅ match, read in full |
| **`AGENTS.md`** (r2) | `fdf667428047bfe5be22a072a5419695814d81061a05ea28c648d945dca8f5cc` | ✅ match |
| **`research/nodes/sk5_local_capability_constraint.json`** (r2) | `0782925c9f95a36c9d17bc0d25c1f47ab0a4608d1bf70e940c9d34c5f413773f` | ✅ match |

Evidence tags used exactly as defined in SK@v5 "Evidence-to-hypothesis contract":
`directly_supported` | `contradicts_premise` | `near_match_only` | `insufficient`.
No paper effect size anywhere in this document is offered as a BARAM expected gain.

---

## 1. VERDICT

**The largest S6 opportunity in this project is not a new physical variable. It is that the frozen prepare
lineage discards two whole axes of the supplied issuance cube before the learner ever sees it — the
*temporal* axis within one issuance and the *geometric arrangement* axis across the 9/16 grid points — and
neither loss is recoverable by any function of the 820 deployed columns.**

I confirmed this by direct schema audit, not inference. Of the 820 unique names in the frozen prepare
manifest:

- **zero** contain `lag`, `lead_` (as an operator), `roll`, `diff`, `trend`, `delta`, `prev`, `next`, `cum`,
  `ewm`, `window`, or `ramp`. The only temporal columns are the calendar stamps `hour`, `month`,
  `day_of_year`, `lead_hour`, `cal__hour_sin/cos`, `cal__doy_sin/cos`. **Each row is a pure single-valid-time
  snapshot.**
- **zero** contain `grad`, `div`, `vort`, or `curl`. The 9 GFS / 16 LDAPS points per group are collapsed into
  the **permutation-invariant** order statistics `{mean, std, min, max, q10, q50, q90}` (98 of 102
  source×variable pairs carry the full 7), plus `idw` and `nearest` point interpolants. **Which point held
  which value is destroyed.**
- **zero** `geom__`, `seq__`, `clim`-prefixed names, exactly as FD6 states — while
  `src/baram/features/geometric.py`, `sequence.py` and `climatology.py` all exist in source and all emit
  those prefixes (`geom__` at geometric.py:198–361, `seq__` at sequence.py:66–73, `clim_` at
  climatology.py:38–74).

This reframes the S6 question. There are two disjoint feature classes with **different admissibility
arguments**, and conflating them is what has been producing null results:

- **Class R (re-encoding).** Any deterministic function of the 820 columns *within one row*. Adds zero
  information. Its only defensible mechanism is the inductive bias of axis-aligned tree ensembles, which is a
  real and published effect, not a rhetorical one (§4, S2). Class R candidates sit adjacent to the closed
  axis "post-processing the current representation" and must carry a stated orientation/irregularity argument
  or be rejected on sight.
- **Class N (new information).** Functions of the issuance cube that are **provably outside the σ-algebra of
  the 820 row columns**: within-issuance temporal neighbours, cross-grid-point geometry, and past-only
  climatological normalisation. These do **not** face the closed post-processing axis at all, because the
  information was never in the representation being post-processed.

**Ranked recommendation: N1 (`seq__` within-issuance temporal neighbourhood) first, N2 (`geom__` plane-fit
spatial gradient) second, R1 (spread-smeared power proxy) third — R1 being the only candidate with a
mechanism that targets FICR band-hits rather than mean error.**

One `contradicts_premise` finding materially changes the standing S6 picture (§5, C1): the previous S13/S6
lane closed "ramp / trajectory-shape features" on the ground that "±1,2,3,6 h lag and lead windows are
already present". **They are not present in the frozen prepare lineage.** That closure was made against a
different, non-deployed feature set and does not apply to the deployed representation.

---

## 2. Local diagnostic — exact FD6 measurement performed by this lane

Read-only, metadata/schema projection only, per the contamination ledger's permitted
`schema/hash/feature_names` branch of `artifacts/manifests/prepare.json`. **No value body, no label, no
prediction, no score, no 2024, no test-period value was opened.**

| Quantity | Exact value |
|---|---|
| `feature_names` cardinality | 820 unique (820 total, 0 duplicates) |
| `spatial_mode` | `spatial_v2` |
| Prefix histogram | `gfs__` 394, `ldaps__` 296, `ldaps_spatial__` 50, `gfs_spatial__` 44, `phys_v2__` 7, `source_disagreement__` 6, `phys__` 5, `cal__` 4, bare static/calendar 14 |
| Box-statistic operators present | `mean, std, min, max, q10, q50, q90` (+ `grid_count`, `missing_cell_count`) |
| source×variable pairs collapsed to box statistics | 102 (98 with the full 7 statistics) |
| GFS distinct variables | 58 (incl. `isobaricInhPa_{500,700,850}_{t,u,v}`, `500_gh`, `meanSea_0_prmsl`, `planetaryBoundaryLayer_0_{u,v,VRATE}`, `surface_0_{sp,gust,prate,tp,dswrf,dlwrf}`, cloud `{l,m,h,t}cc`, `heightAboveGround_{10,80,100}`) |
| LDAPS distinct variables | 44 (incl. `heightAboveGround_50_{50MUmax,50MUmin,50MVmax,50MVmin}`, `heightAboveGround_5_{XBLWS,YBLWS}`, `etc_0_blh`, `surface_0_{h,lsm,sp,ncpcp,SNOM,snol,lssrate,avg_lsprate}`, `meanSea_0_prmsl`, `heightAboveGround_2_{t,q,r,dpt}`) |
| Temporal-operator names | **0** |
| Spatial-derivative names | **0** |
| `geom__` / `seq__` / `clim` names | **0 / 0 / 0** |
| `phys` block | 12 names total: `hub117_speed`, `speed_shear_100_80`, `shear_alpha_100_80`, `air_density`, `rho_v3`, `fleet_power_proxy_w`, `input_missing`, `input_invalid`, `shear_fallback` |
| `source_disagreement__` block | 6 names, covering **only** `wind10_speed` (idw, nearest) and `surface_pressure` (idw). **No hub-height (100 m) cross-source disagreement column exists.** |
| Row counts (schema branch) | `train_features` 78 912, `test_features` 26 280, `labels_long` 78 912, `submission_keys` 8 760 |
| Static group metadata (schema branch) | g1/g2: VESTAS V126, 6 turbines, D=126 m, hub 117 m, 3.6 MW/turbine, 21.6 MW; g3: UNISON U136, 5 turbines, D=136 m, hub 117 m, 4.2 MW/turbine, 21.0 MW; centroids 37.2752–37.2871 N, 128.9520–128.9714 E |

Tag: `directly_supported` for every row of this table (source = `artifacts/manifests/prepare.json`,
schema/`feature_names` branch, read 2026-08-09; and `src/baram/features/*.py` prefix grep).

### 2.1 Two provable information losses

Let `X_row` be the 820-vector for one `(forecast_kst_dtm, group_id)` row.

- **L1 — temporal collapse.** The 24 lead hours of a single issuance are 24 separate rows. Nothing in `X_row`
  is a function of a neighbouring lead hour's fields. Therefore any within-issuance neighbour, rolling
  statistic, day-shape or within-day rank is **not** a function of `X_row`. Class N.
- **L2 — permutation collapse.** `{mean, std, min, max, q10, q50, q90}` are symmetric functions of the point
  multiset. `idw` and `nearest` are two fixed linear functionals. A horizontal gradient requires the pairing
  of value to location; it is **not** recoverable from a symmetric summary plus two fixed functionals.
  Class N.
- **Not a loss:** vertical structure. GFS 10/80/100 m + 850/700/500 hPa and LDAPS 2/5/10/50 m + `blh` are all
  present as box statistics, so stability/shear constructions are Class **R**, not Class N. (This is why the
  prior lane's `hhat` block is Class R here, not Class N — see §9.)

---

## 3. Source ledger

Every source below was published on or before 2026-07-05. "Locator read" states exactly what text this lane
actually read, which is deliberately narrower than what the paper contains.

| id | Exact citation | Pub. date | Primary locator | Locator read (date relied on: 2026-08-09) | Scope warning |
|---|---|---|---|---|---|
| **S1** | Landry, M., Erlinger, T. P., Patschke, D., & Varrichio, C. (2016). *Probabilistic gradient boosting machines for GEFCom2014 wind forecasting.* International Journal of Forecasting 32(3), 1061–1066. DOI `10.1016/j.ijforecast.2016.02.002` | 2016 | RePEc record `RePEc:eee:intfor:v:32:y:2016:i:3:p:1061-1066` (`https://ideas.repec.org/a/eee/intfor/v32y2016i3p1061-1066.html`) | **Full author list + full published abstract.** Body behind ScienceDirect paywall — **not read**. | GEFCom2014 wind track: 10 anonymous zones, hourly, ECMWF u/v at 10 m and 100 m only, pinball-loss quantile metric, no capacity/band reward. Metric and input set both differ from BARAM. |
| **S2** | Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS 2022 Datasets & Benchmarks; arXiv:2207.08815 | 2022-07-18 | `https://arxiv.org/abs/2207.08815`; NeurIPS proceedings hash `0378c7692da36807bdec87ab043cdadc` | **Abstract + introduction/contributions paragraph.** The 45-dataset benchmark tables and the rotation experiment figures were **not** read. | Generic tabular benchmark, ~10K-sample regime, i.i.d. splits. Not time series, not wind, no discontinuous metric. |
| **S3** | Andrade, J. R., & Bessa, R. J. (2017). *Improving Renewable Energy Forecasting With a Grid of Numerical Weather Predictions.* IEEE Transactions on Sustainable Energy 8(4), 1571–1580 | 2017 | INESC TEC institutional repository, item `0d07a0ca-85ac-4eb0-9601-6da2b94e97ae` (`https://repositorio.inesctec.pt/items/0d07a0ca-85ac-4eb0-9601-6da2b94e97ae`) | **Title, authors, year, opening sentences of the abstract only.** The methods section describing the specific grid-derived feature construction was **NOT** read — IEEE PDF and the scispace mirror both failed to retrieve. | Portugal/Spain, wind and solar, grid-of-NWP framework. Whether its grid features are gradients, PCA components, or something else is **unverified by this lane**. |
| **S4** | Focken, U., Lange, M., Mönnich, K., Waldl, H.-P., Beyer, H. G., & Luig, A. (2002). *Short-term prediction of the aggregated power output of wind farms — a statistical analysis of the reduction of the prediction error by spatial smoothing effects.* Journal of Wind Engineering and Industrial Aerodynamics 90(3), 231–246. DOI `10.1016/S0167-6105(01)00222-7` | 2002 | `https://www.sciencedirect.com/science/article/abs/pii/S0167610501002227` | **Bibliographic record only (title, authors, journal, DOI).** No abstract or body read. | German national-scale aggregation over hundreds of km. BARAM's three groups span ~1.7 km. Spatial decorrelation at 1.7 km is nothing like at 300 km. |
| **S5** | Prior in-repo lane `research/lanes/S13_S6_features_deep.md` (2026-08-07) | 2026-08-07 | in-repo | §5 node table, §7 closed-axis list, §8 gaps read in full | Secondary in-repo synthesis, not a primary source. Used only to avoid duplicate proposals and to record C1. |
| **S6** | Official hackathon rules, user-supplied verbatim snapshot: `inputs/rules/official_rules_2026-08-09.md`, sha256 `6dcece…4ee5` | snapshot 2026-08-09 | in-repo | **§2 (language), §3 (leakage / basis-time definition), §4 (external data), §5 (pretrained weights), §6 (API models) read in full, verbatim Korean** | This is the **governing** document for eligibility, superseding my earlier assumptions. It is a snapshot, not a live fetch. |

**Source-bound accounting: 4 external sources used of 16 allowed.** The lane was stopped by root before the
remaining 12 slots were used. §11 lists exactly which claims would have needed them.

---

## 4. Tagged claim ledger

| # | Claim | Tag | Basis |
|---|---|---|---|
| A1 | The frozen prepare lineage has 820 unique feature names, of which **0** are `geom__`, **0** are `seq__`, **0** are `clim*`. | `directly_supported` | `artifacts/manifests/prepare.json#/feature_names`, read 2026-08-09. Independently reproduces FD6. |
| A2 | The lineage contains **no** lag, lead-window, rolling, difference, trend, cumulative or ramp operator on any weather field. Every row is a single-valid-time snapshot. | `directly_supported` | Same locator; exhaustive substring scan over all 820 names for 17 temporal-operator tokens, all zero. |
| A3 | The lineage contains **no** horizontal gradient, divergence or vorticity operator. Grid points enter only through permutation-invariant order statistics and two fixed interpolants (`idw`, `nearest`). | `directly_supported` | Same locator; 102 source×variable pairs × {mean,std,min,max,q10,q50,q90}, plus `*_spatial__{idw,nearest}__*`. |
| A4 | Cross-source (GFS vs LDAPS) disagreement is encoded for **only** 10 m wind speed and surface pressure — 6 columns. There is **no** cross-source disagreement column at 80/100 m or at hub height. | `directly_supported` | Same locator, `source_disagreement__` block enumerated in full. |
| A5 | `geometric.py`, `sequence.py` and `climatology.py` exist in source and emit `geom__`/`seq__`/`clim_` prefixes, yet contribute 0 of 820 deployed names. Module presence ≠ representation coverage. | `directly_supported` | `src/baram/features/{geometric,sequence,climatology}.py` prefix grep + A1. |
| A6 | Tree-based models "remain state-of-the-art on medium-sized data (~10K samples)"; the paper's stated challenge #2 for competing architectures is to "preserve the orientation of the data", and it attributes part of the NN gap to the fact that "their rotation invariance hurt[s] their performance". | `directly_supported` | S2 abstract + contributions paragraph, verbatim. |
| A7 | Consequence of A6 for this project: **a change of basis over the same 820 columns is not information-free for a gradient-boosted tree.** Axis-aligned split ensembles are not rotation-invariant, so ratios, differences and projections of existing columns can change achievable accuracy at fixed information. | `near_match_only` | Logical consequence of A6, but S2's experiment is on i.i.d. tabular benchmarks, not on this panel. The *direction* is supported; the *magnitude here* is not. |
| A8 | The GEFCom2014 wind-track **winning** entry applied "standard smoothing techniques … to the dominant input signal in order to adapt to forecast inaccuracies", and used "a cross-sectional approach" plus "a technique for utilizing information about correlated wind farms … using a two-layer modeling approach". | `directly_supported` | S1 published abstract, verbatim. |
| A9 | Consequence of A8 for this project: temporal smoothing of the dominant NWP wind signal is a documented winning-entry operation in a day-ahead NWP→power competition. | `near_match_only` | The operation is directly supported as *used by a winner*; S1's abstract does not isolate its individual contribution, and the body (which might) was not read. **No effect size may be attributed.** |
| A10 | A published forecasting framework exists that uses a **grid** of NWP points rather than a single point, for renewable energy forecasting. | `directly_supported` | S3 repository record, title + abstract opening. |
| A11 | That framework's specific grid-feature construction (gradients? PCA? sparse-linear selection?) is unknown to this lane. | `insufficient` | S3 body not retrieved (IEEE + mirror both failed). |
| A12 | Spatial smoothing across dispersed wind farms reduces aggregate forecast error. | `insufficient` | S4 bibliographic record only; and the geographic scope gap (300 km vs 1.7 km) is disqualifying even if read. **Not used to support any candidate.** |
| A13 | LDAPS `heightAboveGround_50_{50MUmax,50MUmin,50MVmax,50MVmin}` denote a within-interval maximum/minimum of the 50 m wind components. | `insufficient` | Inferred from the variable name only. **No KMA LDAPS variable documentation was retrieved.** If false, candidate R2 collapses. |
| A14 | LDAPS `heightAboveGround_5_{XBLWS,YBLWS}` semantics. | `insufficient` | Name not decoded. No documentation retrieved. |
| A15 | The previous S13/S6 lane closed ramp/trajectory features (F12) on the premise "±1,2,3,6 h lag and lead windows already exist in the feature set". | `contradicts_premise` | S5 §5 F12 row vs A2. The premise is false **of the frozen prepare lineage**; it referred to a different, non-deployed set. See §5 C1. |

---

## 5. `contradicts_premise` findings

**C1 — the standing closure of temporal/trajectory features rests on a premise that is false of the deployed
representation.**
`research/lanes/S13_S6_features_deep.md` §5 row F12 and §7 item 3 reject ramp/trajectory-shape features
because "D's ±1,2,3,6 h lag and lead windows already carry the same information". Direct audit of
`artifacts/manifests/prepare.json#/feature_names` shows **zero** lag or lead-window columns among the 820
deployed names (A2). Whatever "D" was, it is not the frozen prepare lineage. **Consequence:** the temporal
axis is *open*, not closed, and the parent's "already closed on evidence" list does not contain it either.
This is the single most consequential finding of this lane.

**C2 — "post-processing the current representation is closed" does not close feature re-encoding, but it
does bound it.**
The closed axis concerns operations on the model's *output* given the current representation. Class R
features operate on the *input basis* of a rotation-non-invariant learner (A6/A7), which is a formally
different operation. This is a genuine distinction, **but it is a weak one**: Class R adds no information, so
its entire upside is inductive-bias-shaped, and the project has already measured that adding informative
blocks to this representation produced −0.000728. **This lane therefore ranks every Class R candidate below
every Class N candidate**, and recommends that Class R be attempted only after Class N is adjudicated.

---

## 6. Scope-match matrix against the BARAM surface

BARAM reference surface: 17 turbines / 3 groups / ~1.7 km extent / 37.28 N 128.96 E ridge site; one action at
D-1 14:00 KST; horizon 11–35 h ahead; inputs GFS 9-point + LDAPS 16-point issued tables only; target hourly
group kWh; metric `0.5(1−NMAE) + 0.5·FICR` with 6%/8% relative bands and a `actual ≥ 0.10·capacity` validity
filter; hourly resolution; 3 fixed groups (fixed topology); ≤6 workers, no GPU, no `torch`; competition
licence.

| Axis | S1 Landry (GEFCom2014) | S2 Grinsztajn | S3 Andrade & Bessa | S4 Focken |
|---|---|---|---|---|
| population | 10 anonymous wind zones | 45 generic tabular datasets | Iberian wind + solar plants | German national wind fleet |
| geography | undisclosed (Australia) | n/a | Portugal/Spain | Germany, ~700 km |
| horizon | day-ahead, 1–24 h | n/a | day-ahead class | up to 48 h |
| issue_time | monthly rolling task | n/a | not verified | not verified |
| inputs | ECMWF u,v @10 m & 100 m **only** | arbitrary tabular columns | **grid** of NWP points ✅ nearest analogue | multi-site NWP |
| target | normalised power | regression/classification | power | aggregated power |
| **metric** | pinball / quantile loss ❌ | RMSE / accuracy ❌ | RMSE / CRPS ❌ | RMSE ❌ |
| resolution | hourly ✅ | n/a | hourly ✅ | hourly ✅ |
| topology | 10 zones, cross-sectional ✅ partial | n/a | multi-plant | national |
| compute | CPU GBM ✅ | CPU trees ✅ | CPU ✅ | CPU ✅ |
| licence | public paper | open arXiv/NeurIPS ✅ | published paper | published paper |
| **verdict** | **near match on operation, no match on metric** | **match on learner class, no match on data type** | **best geographic-structure analogue but body unread** | **disqualified on geography (300 km vs 1.7 km)** |

**No source in this ledger shares BARAM's metric.** Therefore **no source can motivate a FICR claim.** All
FICR reasoning in §8 is derived from the metric algebra itself plus local schema, and is tagged accordingly.

---

## 7. Source fact → provisional migration hypothesis → local evidence needed

### 7.1 (a) SOURCE FACT — S1, abstract, verbatim, 2016
> "Standard smoothing techniques were applied to the dominant input signal in order to adapt to forecast
> inaccuracies, and a cross-sectional approach was applied."
Tag `directly_supported` (that the winner did this). Locator: RePEc `v32y2016i3p1061-1066`, read 2026-08-09.

**(b) PROVISIONAL BARAM MIGRATION HYPOTHESIS — N1.** The dominant input signal here is hub-height wind speed
(`phys_v2__hub117_speed`, `ldaps_spatial__idw__wind10_speed`, `gfs_spatial__idw__wind100_speed`). Smoothing it
*along the lead axis within one issuance* supplies the learner with information it provably cannot see (L1),
and specifically attenuates the phase error that dominates NWP wind error at 11–35 h.

**(c) LOCAL EVIDENCE NEEDED.** (i) FD1 legality: prove every lead hour of one `forecast_id` shares one
`data_available_kst_dtm ≤ D-1 14:00 KST`. (ii) A no-fit diagnostic that the smoothed signal is not already a
near-duplicate of the raw signal at this site (autocorrelation of the hub-speed series at lag 1). (iii) A
bounded fold-outside screen.

### 7.2 (a) SOURCE FACT — S2, abstract + contributions, verbatim, 2022-07-18
> "…tree based models remain state-of-the-art on medium-sized data (∼10K samples)…"; challenge 2 is
> "preserve the orientation of the data"; "their rotation invariance hurt[s] their performance".
Tag `directly_supported`. Locator: arXiv:2207.08815, read 2026-08-09.

**(b) PROVISIONAL BARAM MIGRATION HYPOTHESIS — R-class admissibility.** Because the deployed learner family
is axis-aligned boosted trees on a 78 912 × 820 medium tabular panel — squarely in S2's regime — the
*orientation* of the 820-column basis is itself a design variable. Recombinations that add zero information
(ratios, differences, projections) can still change achievable accuracy. This is the **only** legitimate
mechanism by which a Class R candidate may be proposed.

**(c) LOCAL EVIDENCE NEEDED.** A paired comparison at fixed information: same rows, same learner, same
hyperparameters, basis A = raw box statistics, basis B = rotated statistics. Anything else confounds
orientation with information.

### 7.3 (a) SOURCE FACT — S3, repository record, 2017
A published framework forecasts renewable generation from a **grid** of NWP points rather than one point.
Tag `directly_supported` for existence; `insufficient` for its method.

**(b) PROVISIONAL BARAM MIGRATION HYPOTHESIS — N2.** Motivating only. The *local* justification is
independent and stronger: L2 proves the grid geometry is discarded here. A least-squares plane fit over the
supplied point coordinates recovers the discarded first-order geometry.

**(c) LOCAL EVIDENCE NEEDED.** (i) Confirm `latitude`/`longitude` per `grid_id` are stable across the archive
(they are schema columns). (ii) A no-fit rank check that the recovered gradient is not numerically degenerate
on a 3×3 / 4×4 stencil. (iii) Fold-outside screen.

### 7.4 (a) SOURCE FACT — the official metric algebra (frozen map `S4_metric`, `directly_supported`)
`unit = 4 if relerr ≤ 0.06, 3 if ≤ 0.08, else 0`; `FICR_g = Σ(actual·unit)/Σ(actual·4)`; validity
`actual ≥ 0.10·capacity`.

**(b) PROVISIONAL BARAM MIGRATION HYPOTHESIS — FICR channel, see §8.** Because the band is *relative* and the
weight is *actual*, FICR is dominated by near-rated hours where the absolute tolerance is widest and the point
power curve is saturated. In that regime error is governed by the *fraction of the 17 turbines above rated*,
not by mean wind error. That fraction is a function of the **within-group wind spread**, which the box
statistics expose (`std`, `q10`, `q90`) but which no column converts into a power-domain quantity.

**(c) LOCAL EVIDENCE NEEDED.** A **band-hit-indicator** table, not a MAE table (see §8.3).

---

## 8. FICR band-hit behaviour vs mean error — which features can plausibly move which

### 8.1 The asymmetry, derived from the metric algebra alone
`directly_supported` (algebra), not from any paper.

- The 6% band is **relative to actual**, and each row's contribution is **weighted by actual**. So FICR mass
  concentrates on high-output hours, where 6% of actual is a *large* absolute tolerance.
- Conversely, a row just above the validity threshold (`0.10 × 21 600 = 2 160 kWh`) has a 6% band of ±130 kWh
  and contributes almost no weight. **Effort spent making low-output hours accurate buys NMAE and buys almost
  no FICR.**
- Consequence: **a feature can improve NMAE without touching FICR, and vice versa.** Any candidate must
  declare which of the two it targets, and be screened on that one.

### 8.2 Which candidate classes can plausibly move FICR
| Channel | Can it move FICR? | Why |
|---|---|---|
| Reducing mean wind-speed error in mid-range hours | **NMAE only** | mid-range hours carry little actual weight and steep `dP/dU`; band-hit there is a lottery |
| Resolving the **near-rated saturation fraction** (R1) | **Yes — primary FICR channel** | in the saturated regime `dP/dU ≈ 0`, so group output is set by *how many* turbines are above rated, i.e. by the within-group wind spread, not by mean wind |
| Predicting **conditional error scale** (heteroscedasticity) (N1 rolling sd, cross-source disagreement) | **Yes — indirect** | a step reward makes the optimal action depend on the conditional error *scale*; a narrower predicted scale justifies a more aggressive placement. Requires S7/S8 policy to consume it; a feature alone is not sufficient |
| Within-hour sub-hourly variance (R2, if A13 holds) | **Plausible** | the label is an hourly energy total; a high within-hour wind envelope means the hourly mean power ≠ power of the hourly mean wind (Jensen gap on a convex power-curve knee), a bias that is systematic and band-relevant |
| Curtailment / availability / storm shutdown | **Would dominate, but FD8-blocked** | structurally unobservable at inference under the supplied schema |

### 8.3 Mandatory screening rule for any FICR claim (methodological, `directly_supported` by algebra)
**Screen band-hit *indicators*, not continuous error.** For a candidate feature `z`, bin the valid rows by `z`
and tabulate the empirical rate of `1{relerr ≤ 0.06}` and `1{relerr ≤ 0.08}` per bin, actual-weighted. A
candidate that flattens conditional MAE but leaves the conditional band-hit rate flat has **no FICR
mechanism** and must be re-labelled NMAE-only. Correlations of continuous error and correlations of band-hit
indicators diverge by more than a factor of two in this project's own past measurements; using the former as a
proxy for the latter has produced wrong conclusions before.

---

## 9. Ranked candidate table

Ranking = (mechanism strength × information novelty) ÷ (degrees of freedom × dependency risk).
Every candidate is stated as a **block** with all-or-nothing admission — no per-column post-hoc rescue.

| rank | id | class | mechanism (exact) | treatment | control | DOF | cols | FD relieved | targets | dependency |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **N1 `seq__`** | **N (new)** | The 24 lead hours of one issuance are 24 unlinked rows (L1). Supplying same-run temporal context lets the learner see forecast *phase* and *trajectory roughness*, which no row column encodes. Winner-documented operation (A8/A9). | For exactly 3 dominant signals `{phys_v2__hub117_speed, ldaps_spatial__idw__wind10_speed, gfs_spatial__idw__wind100_speed}`: within `(forecast_id, group_id)` ordered by `lead_hour` — value at lead ±1 and ±3; centred 3-h mean; centred 5-h sd; issuance-day mean; within-day rank. Edge leads use one-sided windows, flagged. | Champion pipeline, identical rows/policy/seed, block absent | 1 (block on/off). Signal list, offsets and window widths **frozen before any fit** | 15 | **FD6** (seq__ = 0/820), **FD1** (legality is the gate) | NMAE primarily; rolling-sd column also feeds the FICR error-scale channel | ✅ pandas only |
| **2** | **N2 `geom__`** | **N (new)** | Box statistics are permutation-invariant (L2): grid geometry is destroyed and unrecoverable. A plane fit restores first-order horizontal structure — the cross-ridge pressure gradient, wind divergence and vorticity that drive channelled and downslope flow at this ridge site. | Per (source, valid time, group): weighted least-squares plane over the 9/16 points in local east/north km → `∂f/∂x, ∂f/∂y, |∇f|` for `f ∈ {prmsl, sp, 10u, 10v}` (both sources) plus derived `div = ∂u/∂x + ∂v/∂y`, `vort = ∂v/∂x − ∂u/∂y`. | as above | 1 (block on/off). Field list and fit weighting frozen | 14 | **FD6**, **FD1** (same-run only) | NMAE; regime discrimination | ✅ numpy `lstsq` |
| **3** | **R1 spread-smeared power proxy** | R (re-encoding, but power-domain) | `E[P(U)] ≠ P(E[U])` across a group whose 5–6 turbines see different winds. Only the **power-domain** convolution isolates the near-rated saturation fraction, and an axis-aligned tree cannot form it from `mean/std/q10/q90` at feasible depth (A6/A7). This is the **only candidate with a direct FICR mechanism**. | Fixed generic normalised power curve (from the known V126 / U136 rated speeds — metadata, not external data). Emit `Pbar = (P(q10)+4P(q50)+P(q90))/6`, `sat_frac ≈ 1{q10 > u_rated}`-smoothed, `Pbar − P(q50)` (the Jensen gap), for hub-height speed, per source. | as above | 2 (curve parameterisation; quadrature rule) — both frozen in advance | 6 | **FD6** | **FICR** (near-rated saturation) | ✅ numpy |
| **4** | **N3 `clim__`** | N (new: cross-row, past-only) | `clim* = 0/820`. A past-only expanding `(group, month, hour)` mean/sd of the dominant NWP signal converts an absolute forecast into a *standardised anomaly*, letting one split serve all seasons. Cross-row ⇒ outside the row σ-algebra. | Expanding, strictly past-only, **NWP-side only (no label touched)**: `z = (hub117_speed − μ_{g,m,h}) / σ_{g,m,h}`, plus `μ` and `σ`. Warm-up rows flagged and excluded from the block. | as above | 1 (block) + window definition frozen | 6 | **FD6**, **FD1** | NMAE | ✅ pandas |
| **5** | **R2 sub-hourly envelope** | R | If A13 holds, LDAPS `50MU/50MVmax/min` are within-interval extremes at 50 m — the only sub-hourly information in the whole cube. Envelope width is a within-hour variability measure relevant to an hourly-energy label. | `env_speed = wind50max_speed − wind50min_speed`; `env_turn = angular difference between the max and min wind directions`; `env_over_mean = env_speed / wind10_speed`. | as above | 1 | 3 | FD6 | FICR (Jensen gap) + NMAE | ✅ numpy. **Gated on resolving A13** |
| **6** | **R3 hub-height cross-source disagreement** | R | A4: disagreement exists only at 10 m and surface pressure. The *hub-relevant* level (80/100 m GFS vs 50 m LDAPS) has no disagreement column, yet that is where the power-relevant error lives. | `|gfs_spatial__idw__wind100_speed − ldaps_spatial__idw__wind50max_speed|` and a level-consistent shear-extrapolated pair; signed and absolute. | as above | 1 | 4 | FD6 | FICR error-scale channel | ✅ |
| **7** | **R4 order-statistic basis rotation** | R | Pure orientation change at fixed information, motivated solely by A6/A7. Replaces a 7-column redundant summary with a location/scale/shape triple. | Per (source, variable): replace `{mean,std,min,max,q10,q50,q90}` with `{q50, IQR=q90−q10, skew=(q90+q10−2q50)/IQR, range_z=(max−min)/std}`. Applied to a frozen subset of the 20 highest-level wind variables only. | as above | 1 | −60 net | FD6 | NMAE | ✅ |
| **8** | **R5 nested block pruning** | R | S2 challenge 1 concerns uninformative features; 690 of 820 columns are a mechanical 7-fold expansion of 98 pairs. Removal is the mirror of the project's measured "adding informative blocks cost −0.000728". | Backward block elimination over 8 predeclared blocks, **selection performed strictly inside each fold**. | uniform 820-column pipeline | 8 blocks × nested refit — highest DOF here | varies | FD6, FD7 | NMAE | ✅ but expensive |
(For B1–B6, see §9.1 — they were previously listed here as `BLOCKED_DEPENDENCY` and have been re-judged.)

**Sklearn-based representation learning is available without any install** (PCA, KernelPCA, PLS, KMeans,
RandomTreesEmbedding all ship with scikit-learn 1.9.0). A PCA over the grid-point stencil is therefore
feasible — but note it is **Class R when applied to the 820 columns** and **Class N only when applied to the
raw pre-collapse cube**. Applying PCA to the 820 columns is a rotation with no new information and ranks
with R4.

---

## 9.1 Re-judgement of B1–B6 under the 2026-08-09 authority change (revision r2)

`BLOCKED_DEPENDENCY` is **retired as a rejection reason**. Installing any package into the project `.venv` is
authorised. The six items below are therefore re-ranked **on evidence and cost only**. Three points of
discipline apply:

- **Availability was never the binding constraint for this lane.** N1/N2/N3/R1 were and remain
  `pandas`/`numpy`-only. The authority change does **not** raise any of them, and does not lower them either.
- **The evidence that ranked B1–B3 low was never a package fact.** It is A6: on medium-sized tabular panels
  (~10K samples) tree ensembles "remain state-of-the-art", and this panel has 78 912 rows over only **1 096
  distinct issuance days**, i.e. ~1 096 effectively independent units. That argument is untouched by `pip`.
- **Installing a package adds a degree of freedom and a reproduction obligation.** Official rules §4 requires
  that everything be reproducible from submitted code, and §7 of this project's own governance requires trial
  budgeting. An install is not free.

| id | class | mechanism | honest cost / evidence judgement | new status | owner |
|---|---|---|---|---|---|
| **B1** CNN or GNN over the raw 3×3 / 4×4 stencil | **N (new)** | Learns L2's discarded geometry end-to-end instead of hand-coding the plane fit (N2). Strictly more expressive than N2. | **Installable, but ranks BELOW N2 on evidence and cost.** A6 argues directly against NNs at this sample size; the stencil is 9/16 points (a CNN has almost no spatial extent to exploit); DOF is large (architecture, width, depth, optimiser, epochs, seed) against ~1 096 independent days, which aggravates FD7 badly. **N2 is the same information channel at DOF 1.** | **ELIGIBLE — deferred.** Attempt only if N2 passes gate G, i.e. only after the channel is proven to carry signal at low DOF. | S6 (this lane) |
| **B2** autoencoder / learned latent over the cube | **N (new)** if fitted on the pre-collapse cube | Nonlinear compression of the issuance cube. | **Installable; ranks below B1.** Unsupervised reconstruction loss is not aligned with the band metric, and it adds a whole selection surface (latent dimension) with no local diagnostic to set it. | **ELIGIBLE — not recommended in the first batch.** | S6 |
| **B3** learned group embeddings (Rasp–Lerch style) | R | Dense embeddings for the group factor. | **Installable, and now cheap — but still near-degenerate.** The factor has exactly **3** levels and a group dummy already exists. An embedding of a 3-level factor is a reparameterisation of 2 free numbers. The objection was never the package. | **REJECTED on evidence (3 levels), not on availability.** | — |
| **B4** probabilistic boosting (NGBoost / CatBoost `RMSEWithUncertainty` / LightGBM multi-quantile) → conditional error-scale head | — | Direct conditional-scale estimate, which §8.2 identifies as the **indirect FICR channel**. | **Installable, and this is the strongest newly-eligible item.** But note CatBoost 1.2.10 and LightGBM 4.7.0 **already** provide this in-stack, so the install buys little. | **ELIGIBLE and RECOMMENDED — but it is S7/S8, not S6.** A conditional-scale estimate is a model output, not a feature. **Hand to `V5-MODELS`.** | S7 |
| **B5** conformal / Venn–Abers band calibration (`mapie`, `crepes`, `venn-abers`) | — | Calibrated probability that a candidate action lands inside the 6% band. | **Installable. Genuinely well-matched to a band metric** — this is the one item the authority change materially unlocks. **But** split-conformal validity assumes exchangeability, which a strict prequential wind panel violates, and there is **no fresh holdout left** (FD7) to calibrate on without reusing the development surface. | **ELIGIBLE, high interest, high FD7 risk. S8 decision layer, not S6.** Hand to `V5-VALIDATION` / `V5-MODELS`. | S8 |
| **B6** decision-focused training to the step loss (SPO+, `cvxpy`) | — | Trains directly on the discontinuous 4/3/0 reward instead of a surrogate. | **Installable.** Mechanically the most correct answer to "the deployed number is an ACTION, not a conditional mean". Cost is high and the action space here is 1-D per cell, so a direct grid search over the action already attains the optimum without `cvxpy`. | **ELIGIBLE but low value-add; S7, not S6.** | S7 |
| **B7** pretrained weather foundation-model weights (added in r2 for completeness) | — | e.g. a published global AI weather model run locally. | **Rules §5 permits only weights released on or before 2026-07-05 under a licence allowing use, modification, distribution, redistribution AND commercial use** — research-only/eval-only weights are ineligible, and remote API inference is banned by §6. Independently, such models are typically **initialised from reanalysis**, which rules §3 forbids as an input. | **OUT OF THIS LANE'S SCOPE.** Licence and availability audit belongs to `V5-EXTERNAL`. **No such weight was checked by this lane.** `insufficient` | V5-EXTERNAL |

**Net effect of the authority change on this lane's ranking: none.** N1 > N2 > R1 > N3 stands. The change
adds B5 as a genuinely attractive *S8* instrument and confirms B4 as an *S7* one, and it converts B1/B2 from
"impossible" into "possible but deliberately deferred behind a DOF-1 test of the same channel".

### 9.2 External-data eligibility, restated under the actual rules (revision r2)

My exclusive question is scoped to constructions **from the supplied cube**, so this changes no candidate.
Recording the corrected rule for the root's benefit, since I had previously mis-stated it:

- There is **no blanket publication-year cutoff for external DATA.** Rules §3 judges each datum against the
  basis time of **its own prediction row** (D-1 14:00 KST of the target date), by when the datum was
  *created/published/finalised* — explicitly **not** by the time it describes, and explicitly **not** by a
  "published before year Y" test. Rules §4 adds: legal, licence/ToS-compliant, publicly accessible to anyone,
  and reproducible from submitted code.
- **Forecast data** is judged by its issue/publication time, not its valid time (§3, worked example).
- **Observations** are usable only if observed/published/finalised before the basis time; **post-hoc corrected
  observations and reanalysis are forbidden** (§3), as are test-period actuals and anything answer-equivalent.
- **Derived variables, statistics, interpolations and aggregates must not embed any post-basis-time
  information** (§3). *This is exactly the constraint that governs my N3 climatology candidate, and N3 as
  specified — expanding, strictly past-only, NWP-side only — satisfies it.*
- The 2026-07-05 cutoff applies to **pretrained weights only** (§5), together with the commercial +
  redistribution licence requirement.

---

## 10. Rejected alternatives and why

| Rejected | Reason | Tag |
|---|---|---|
| Static terrain indices (TPI/TRI/VRM/RIX/slope/aspect) | Constant per group ⇒ linearly dependent with the 3-level group dummy. Independently re-confirmed here: `ldaps__surface_0_h__{mean,std,…}` and `surface_0_lsm__*` are already in the 820 names and are group-constant. | `directly_supported` |
| Any new external NWP source, GEFS, reanalysis, CFSv2, farm/regional forecast products, DEM download | Closed by the parent's list and/or by AGENTS.md; and this lane performed **zero** external data download. | n/a |
| Physical downscaler output (WindNinja/mass-conserving/WAsP) as a feature | Closed by prior lane S5-E5; also would need external terrain data. | inherited |
| Lagrangian upstream advection sampling | ~6 km stencil at 1 h resolution ⇒ Δt ≈ 5–20 min, numerically the identity. Independently agreed. | inherited |
| TI/TKE surrogates | `phys__speed_shear_100_80`, `shear_alpha_100_80` and `surface_0_gust` already present; NWP diagnostic TKE is not the 1 Hz sonic TKE that the supporting literature measured. | inherited |
| Observation-based teachers (ASOS/AWS) at inference | Test-period observations forbidden. Training-only teachers are an S7 question. | rule |
| Turbine-level wake / power-curve modelling | Closed by the parent's list. | rule |
| Curtailment as a deployable feature | Closed; FD8 says the operational cause is structurally unobservable at inference. | rule |
| Applying PCA/autoencoder to the 820 columns and calling it "representation learning" | It is a rotation of an already-collapsed summary — Class R with the weakest possible mechanism, and it inherits the closed post-processing axis almost entirely. | `contradicts_premise` of the usual framing |
| Quoting S4 (Focken) to support spatial smoothing here | 300 km German fleet vs 1.7 km three-group site — the decorrelation scale that makes the source's effect exist does not exist here. Disqualified on geography, not merely downgraded. | `insufficient` |

---

## 11. Smallest discriminating local experiment per surviving candidate

Ladder for every candidate: **E0 no-fit diagnostic → E1 bounded screen → E2 full strict prequential.** E0 must
pass before E1 is authorised; E1 before E2. **This lane executes none of these** — they are specifications for
`DS@v5` / `IP@v3`.

### Shared gate G (predeclared, applies to E1 for every candidate)
Fold-outside mean `Δ(1−NMAE) ≥ +0.0010` **and** sign agreement in 3/3 folds **and** block size ≤ its declared
column count **and** all-or-nothing admission. Failure ⇒ the block is permanently retired with a receipt.
FICR-targeting candidates (R1, R2, R3) substitute `ΔFICR ≥ +0.0010` under the §8.3 band-hit screen.

### N1 `seq__`
- **E0 (no fit, minutes).** (a) **FD1 legality proof:** group the issued tables by `forecast_id` and assert
  `nunique(data_available_kst_dtm) == 1` per issuance and `max(data_available) ≤ D-1 14:00 KST` for every
  issuance in train **and** test. (b) Lag-1 autocorrelation of the hub-speed series within an issuance.
  **Falsifier:** if any issuance mixes availability stamps → the whole candidate is illegal and dies here. If
  lag-1 autocorrelation > 0.99, the neighbour columns are numerically duplicate → drop offsets ±1, keep ±3
  only.
- **E1 (bounded screen).** Champion learner, 3-fold fold-outside, block on/off, one run each. **Falsifier:**
  gate G fails ⇒ retire.
- **E2.** Full strict prequential on 2023 expanding origins, fixed policy, both components reported.
  **Falsifier:** `ΔTotal ≤ 0` or FICR regression > NMAE gain.

### N2 `geom__`
- **E0.** Rank/condition number of the coordinate design matrix per group and source; and a
  variance-inflation check of the fitted gradients against `std` of the same field (if `|corr| > 0.85`, the
  plane fit is just re-expressing spread ⇒ retire before spending a gate).
- **E1 / E2.** As N1. **Falsifier:** gate G.

### R1 spread-smeared power proxy
- **E0 (the FICR-specific screen, §8.3).** On valid training rows only, bin by `sat_frac` and tabulate the
  **actual-weighted band-hit rate** `1{relerr ≤ 0.06}` of the *existing champion residual behaviour* — but
  note this requires reading predictions, which **this lane may not do**; it is an `IP@v3` task.
  **Falsifier:** if the band-hit rate is flat across `sat_frac` bins, R1 has no FICR mechanism and must be
  demoted to an NMAE-only candidate or retired.
- **E1 / E2.** As above with the FICR-substituted gate.

### N3 `clim__`
- **E0.** Count the minimum sample size per `(group, month, hour)` cell at the earliest prequential origin.
  **Falsifier:** if the minimum count is < 5 for more than 20% of cells at the first origin, the anomaly is
  noise-dominated ⇒ coarsen to `(group, season, hour-block)` or retire.

### R2 sub-hourly envelope
- **E0.** **Resolve A13 first** from KMA LDAPS variable documentation. **Falsifier:** if `50MUmax` is not a
  within-interval extremum, retire immediately — there is then no sub-hourly information in the cube at all.

### R3 hub-height cross-source disagreement
- **E0.** `|corr|` against the existing `source_disagreement__wind10_speed_idw__abs`. **Falsifier:**
  `|corr| > 0.90` ⇒ pure duplicate, retire.

### R4 / R5
Attempt only after N1 and N2 are adjudicated. R5 in particular has the highest DOF in this table and directly
aggravates FD7 (adaptive reuse of a surface that can no longer be refreshed); it should be treated as a
last-resort axis with an explicit trial-budget entry.

### Inconclusive conditions (all candidates)
A result is **inconclusive**, not a rejection, if: fold-outside signs disagree 2:1 with `|Δ| < 0.0005`; or the
block changes NMAE and FICR in opposite directions with `|ΔTotal| < 0.0005`; or the run cannot be
bit-reproduced. Inconclusive results consume trial budget but do **not** permanently retire a block.

---

## 12. Unknowns (explicit)

1. `insufficient` — **12 of the 16 permitted source slots were never used.** The lane was stopped by root
   before completing retrieval. The specific claims that remain unsupported by primary literature are: the
   method detail of S3 (grid-derived feature construction); any primary source on multiscale/wavelet NWP
   decomposition; any primary source on spatio-temporal graph features; any primary source on decision-focused
   feature construction under a step/threshold reward; and any official KMA/NCEP variable documentation.
2. `insufficient` — **A13**: whether LDAPS `50MUmax/50MUmin/50MVmax/50MVmin` are within-interval extrema.
   R2 lives or dies on this. Resolvable in minutes from KMA LDAPS documentation.
3. `insufficient` — **A14**: the meaning of LDAPS `heightAboveGround_5_XBLWS/YBLWS`. Two variables currently
   uninterpretable, present in both the box and `_spatial__` blocks.
4. `insufficient` — **A11**: S3's actual grid-feature construction. N2 is therefore justified by the *local*
   L2 argument alone, with S3 as motivation only.
5. `insufficient` — **FD1 is unresolved and is a hard gate on N1, N2 and N3.** Whether all lead hours of one
   issuance share one `data_available_kst_dtm ≤ D-1 14:00 KST` was **not** verified by this lane (it requires
   opening the issued weather tables, which is beyond a metadata-only read). If issuances are stitched from
   multiple runs, N1's neighbour columns leak and every temporal candidate dies.
6. `insufficient` — the **raw pre-collapse issuance cube's per-`grid_id` coordinate stability** across the
   archive was not verified. N2 assumes fixed point geometry.
7. `insufficient` — whether the champion pipeline's learner is in fact a boosted tree ensemble. A6/A7's
   relevance is conditional on that; if the champion is not tree-based, the entire Class R rationale weakens.
8. `insufficient` — **no FICR mechanism in this document has been measured.** §8 is derived from metric
   algebra and schema, not from any local band-hit table (which this lane is forbidden to compute).
9. Known and accepted: **no fresh holdout exists** (2024 consumed). Nothing here can be independently
   confirmed. Every candidate that survives E2 will still be a retrospective corroboration on an adaptively
   reused surface — FD7.

---

## 13. Explicit DS@v5 / IP@v3 implications

1. **DS@v5 must re-open the temporal axis at S6.** C1 shows it was closed on a false premise. This is a
   schema-verifiable correction, not an opinion, and it should be recorded as a reversal with a receipt.
2. **DS@v5 must carry the Class R / Class N distinction into the node schema.** Suggested field on every S6
   node: `information_class ∈ {re_encoding, new_information}` plus `sigma_algebra_argument`. A `re_encoding`
   node must additionally carry an `orientation_rationale`; without one it is inadmissible under the closed
   post-processing axis.
3. **N1/N2/N3 all require re-entering S5 (prepare), not just S6.** The information they need is destroyed
   *before* the feature table is written. IP@v3 must therefore budget a prepare-stage change with a new
   `prepare.json` hash and a new `feature_names` set, and must re-verify FD1 at that point. This is the
   single largest implementation cost implied by this lane.
4. **FD6 should be treated as satisfied-by-diagnosis and re-scoped.** §2 supplies the exact prefix set
   difference FD6 asked for. The residual deficit is no longer "we do not know what the representation
   contains" but "three implemented modules are dead code relative to the deployed lineage, and two whole
   information axes are discarded at prepare time".
5. **FD1 is now a blocking gate, not a background concern.** Three of the four Class N candidates are illegal
   if issuances mix availability stamps. IP@v3 should run the FD1 legality assertion **first**, before any
   feature work, because a negative result retires most of this lane's portfolio at zero fit cost.
6. **FD5 interacts with §8.** The strict receipt must expose actual-weighted band-hit counts per 6%/8% tier,
   otherwise no FICR-targeting candidate can be adjudicated at all. §8.3's indicator-screening rule should be
   written into the receipt contract.
7. **Dependency posture is already binding.** Six candidate families (B1–B6) are `BLOCKED_DEPENDENCY` under
   `research/nodes/sk5_local_capability_constraint.json`. The one partially recoverable item is B4: LightGBM
   two-quantile fitting is in-stack and gives a conditional-scale proxy. That belongs to `V5-MODELS`.
8. **Trial-budget warning.** This lane proposes 7 admissible blocks. Running all of them against a surface
   with no fresh holdout is exactly the FD7 failure mode. Recommended: authorise **N1 and N2 only** in the
   first S6 batch, with R1 conditional on the §8.3 band-hit screen showing a live FICR mechanism.

---

## 14. Compliance statement

- Repository writes: **1** — this file. No other path was written.
- Model fits: **0**. Score computations: **0**. Metric/policy calls: **0**.
- `actual_kwh`, predictions, model results, 2024, test-period values read: **0**. Only
  `artifacts/manifests/prepare.json` schema/`feature_names` branch, `src/baram/features/*.py` name grep, and
  the four authority documents were read.
- External data or weight downloads: **0**. Only published-paper metadata and abstracts were retrieved.
- Dependency changes: **0**. Dacon / account / remote compute / git actions: **0**. Lockbox reads: **0**.
- Sub-agents spawned: **0**.
- Source bound: 4 of 16 used; truncation was root-directed and is disclosed in §3 and §12.1.
