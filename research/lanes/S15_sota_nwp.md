# S15 — SOTA / BENCHMARK LANE: NWP INGESTION AND SITE TRANSFER (B1–B5)

- **Lane**: `S15_sota_nwp` (read-only research lane; writes confined to `research/`)
- **Date**: 2026-08-09 (local session time)
- **Queries**: 106 (full log: `research/lanes/S15_sota_nwp.searchlog.json`)
- **Repository writes**: this file + that searchlog only. No model fit, no lockbox, no git mutation,
  no install, no clone, no upload.
- **Scope claim**: this lane covers stages B1–B5 only. Nothing here is a PASS of anything.

---

## §0 EVIDENCE-GRADE CONVENTION (read this before any number below)

Every quantitative claim carries exactly one grade tag.

| Tag | Meaning |
|---|---|
| **[A-FULLTEXT]** | I downloaded and read the **primary PDF/HTML** and the number is quoted from the paper's own table or sentence. Highest grade available to this lane. |
| **[B-SNIPPET]** | Number appears verbatim in a search snippet but I did **not** open the primary text. The surrounding table/context is unverified. |
| **[C-SECONDARY]** | Number is reported by a *citing* work, not by the original. Transcription risk. |
| **[D-QUALITATIVE]** | Direction of effect only; no number I trust. |
| **[OURS-DESC]** | **Descriptive statistic I computed on our own repo data in this lane.** No model was fitted; these are means, medians, ratios and Pearson correlations only. Reproduction recipe is given inline. |

Two further tags qualify **transfer**, not truth:

| Tag | Meaning |
|---|---|
| **[BASELINE-MISMATCH]** | The paper's gain is measured against *raw nearest-grid NWP* or *a single NWP source*. Our baseline is already a GBM fed 25 raw grid cells from two sources. **Most of the paper's gain is already inside our baseline.** Do not transfer the percentage. |
| **[NEAR-MATCH]** | The *method* is supported but the *use we propose* is not the use the paper measured. |

**Standing rule inherited from AGENTS.md** and enforced here: every recommendation states (a) which
policy/target produced each input, (b) whether anything was fitted in-sample, (c) the row-alignment
key. All `[OURS-DESC]` numbers below are fitted on nothing and aligned on
`forecast_kst_dtm` ↔ `kst_dtm` (hourly, right-labelled, right-closed).

---

## §1 THE FIVE MEASUREMENTS THAT REORGANISE THIS WHOLE CLUSTER

I computed these on our own cached parquet before writing a single recommendation. They change the
ranking. **Read §1 before §3.**

### 1.1 We already own a measured hub-height wind time series for the entire training period

`research/scratch/scada_vestas.parquet` carries `vestas_wtg01..12_ws` and `_wd` at 10-minute
resolution from **2022-01-01 01:00 to 2025-01-01 00:00 KST**, and
`research/scratch/scada_unison.parquet` carries `unison_wtg01..05_ws/_wd` from **2023-01-01**.
All 17 turbines are at **hub height 117 m**. The power columns are time-scrambled and unusable, but
**`ws` and `wd` are valid** (stated by the brief, and the ranges below are physically coherent).

This is the single most important asset in the cluster, and it is the asset that the entire
literature of §3 assumes you have. Winstral 2017 trained on 200+ stations; Wind-Topo on 261;
DEVINE validated on 61; Optis 2021 on 2 floating lidars. **We have one station, but it is
*the* station — the exact target site, at the exact hub height, for the exact training window.**

Consequence: **B2, B3 and B4 stop being physics guesses and become a supervised regression
`NWP fields → measured hub wind` with 26,304 hourly labels.** That is the framing every SOTA
method in this lane actually uses. Our current pipeline does not use it.

Legality: this is a **training-period observation of our own site**. Labels end 2024-12-31 and the
graded period is all of 2025 submitted at once, so there is no leakage path and no
"recent observed generation at prediction time" violation — the SCADA is used only to *fit the
mapping*, and at inference only NWP columns are read. This is exactly the structure of
Winstral 2017 and Wind-Topo. It is **not** on the closed-axis list.

### 1.2 [OURS-DESC] The vertical layer we are extrapolating *from* is already above the target

Hourly means over the 26,304 overlapping hours (Vestas nacelle `ws`, per-hour **median across the
12 turbines**, then hourly mean; NWP columns are the 16-cell / 9-cell box means from
`train_grid_pivot.parquet`):

| series | mean (m/s) | ratio hub/NWP | implied α to 117 m |
|---|---:|---:|---:|
| **measured hub 117 m** | **7.086** | — | — |
| LDAPS `wind10_speed` box-mean | 4.836 | **1.465** | **α = 0.155** |
| LDAPS `wind50max_speed` box-mean | 7.497 | **0.945** | **α = −0.066** |
| GFS `wind100_speed` box-mean | 3.834 | **1.848** | — |

Three things fall out and each of them contradicts a current design choice.

1. **LDAPS 10 m → 117 m needs α ≈ 0.155**, which is almost exactly the textbook `1/7 = 0.143`.
   The fixed power law is, on average, *nearly right* from 10 m. There is little mean bias to win here.
2. **LDAPS 50 m box-mean already exceeds the measured 117 m wind** (7.497 > 7.086). The implied
   exponent is **negative**. Any positive-exponent power law or log law applied upward from the
   LDAPS 50 m field is **guaranteed to over-predict on average**. (`wind50max_speed` is built from
   `50MUmax/50MVmax`, i.e. a max-type component field, so this is partly a definitional artefact —
   but that is precisely the point: the current B2 two-point fit treats it as a level wind.)
3. **GFS 100 m is 54% of the measured 117 m wind.** A 0.25° cell (~25 km) smooths the Gadeoksan
   ridge out of existence. This is not noise, it is a **multiplicative scale error of ~1.85**.

### 1.3 [OURS-DESC] The extrapolation error is a clean, fully forecastable stability cycle

Median ratio `measured_hub / NWP`, by hour of day (all 26,304 hours, `ld10 > 1 m/s`):

| hour | 00 | 04 | 07 | 10 | 13 | **14** | 16 | 19 | 22 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hub / LDAPS10 | 1.606 | 1.636 | **1.719** | 1.390 | 1.192 | **1.179** | 1.195 | 1.442 | 1.579 |
| implied α(10→117) | 0.193 | 0.200 | **0.220** | 0.134 | 0.071 | **0.067** | 0.072 | 0.149 | 0.186 |
| hub / GFS100 | 2.441 | 2.484 | **2.499** | 1.941 | 1.339 | **1.279** | 1.324 | 1.828 | 2.316 |

By LDAPS boundary-layer-height quartile (`ldaps__etc_0_blh`, a *forecast* variable, available at
D-1 14:00 with everything else):

| BLH quartile | median BLH (m) | hub/LDAPS10 | hub/LDAPS50 | hub/GFS100 |
|---|---:|---:|---:|---:|
| Q1 (most stable) | 142.7 | **1.695** | 1.200 | 2.464 |
| Q2 | 171.8 | 1.668 | 1.089 | 2.535 |
| Q3 | 316.2 | 1.346 | 0.870 | 1.781 |
| Q4 (most convective) | 634.8 | **1.248** | 0.797 | **1.357** |

**The implied shear exponent swings 0.067 → 0.220, a factor of 3.3, on a perfectly diurnal cycle,
and is monotone in a variable we already hold.** This is the classic stability signature
(Touma 1977 reports α 0.1–0.3 by day and 0.2–0.8 at night, `[B-SNIPPET]`
<https://www.tandfonline.com/doi/pdf/10.1080/00022470.1977.10470503>; the ridge site compresses the
range as expected). A **fixed** exponent is therefore wrong by ±0.077 in a way that is
**deterministically predictable from `blh`, hour, and the radiation fields**.

Note the trap this creates, and note that hour-of-day is *not* a new feature — the learner already
has it, and lead time is collinear with it. So the win is **not** "add hour". The win is that
`blh` and the radiation fields let the correction be **conditioned on the actual stability of the
forecast day**, not on the climatological hour. Q1-vs-Q4 spans 1.695→1.248 **within** the same
hours.

### 1.4 [OURS-DESC] The two columns built *specifically* for site transfer are the two worst columns

I checked what is actually in the 830-column feature matrix and correlated every existing reduction
against the measured hub wind, 26,304 hours. **This is the most surprising result in the lane.**

| existing / candidate column | corr vs measured hub wind |
|---|---:|
| `ldaps__grid13__wind50max_speed` (best single cell — **not currently a feature**) | **0.8486** |
| `ldaps__wind50max_speed__max` (exists) | **0.8475** |
| `ldaps__wind50max_speed__q90` (exists) | 0.8432 |
| `ldaps__grid12__wind50max_speed` (**not currently a feature**) | 0.8427 |
| `ldaps__wind50max_speed__mean` (exists) | 0.8141 |
| **`ldaps_spatial__idw__wind50max_speed`** (exists — *the B1 CURRENT method*) | **0.7933** |
| **`ldaps_spatial__nearest__wind50max_speed`** (exists) | **0.7705** |

**The IDW column ranks 6th of 7 and the nearest-cell column ranks 7th of 7.** The two features
engineered specifically to answer "what is the wind at the site" are beaten by a plain
`max()` over the box by **+0.054 / +0.077 correlation**.

Note this *complicates* rather than confirms Winstral: he found nearest **better** than IDW at ridge
sites; **we find nearest worse than IDW**. The common ground — and the part that replicates — is
that **both point-transfer operators lose to a high-order statistic of the box**. The physical
reading: the 1.5 km cell containing the site is a *smoothed* ridge (its orography is 80–140 m below
the true summit), so no single cell and no distance-weighted blend of cells represents the summit;
what tracks the summit is **the strongest cell in the neighbourhood**, wherever it happens to be.

The best cell (`grid13`, 0.8486) beats box-max (0.8475) by only **+0.0011** — i.e. **box-max already
captures essentially all of the available single-cell signal, and box-max is already a feature.**
And the best cell is neither the nearest nor the highest: `grid13` orography is 896.2 m, while
`grid06`/`grid11` are at 1001.2 / 999.1 m.

### 1.4b [OURS-DESC] The correct reduction operator is **different for each source** — decisive for B5

Same test, GFS side (9 cells at 0.25° ≈ 25 km):

| column | corr vs measured hub wind |
|---|---:|
| `gfs_spatial__idw__wind100_speed` | **0.7200** |
| `gfs_spatial__nearest__wind100_speed` | 0.7146 |
| `gfs__wind100_speed__max` | **0.5371** |

**The ordering inverts.** For LDAPS, `max` (0.8475) crushes `idw` (0.7933). For GFS, `idw` (0.7200)
crushes `max` (0.5371). A single shared reduction rule is therefore **provably wrong for one of the
two sources**, whichever rule is chosen.

The physics is clean: LDAPS at 1.5 km partially resolves the ridge, so the *extremum* over the box
is the ridge signal and the *mean* is the valley-contaminated smear. GFS at 25 km resolves nothing
local at all — its box is pure synoptic forcing, where the *mean* is the signal and the *max* is
sampling noise. **This is the single strongest piece of evidence in the lane for per-source
treatment, and it is measured on our own data rather than transferred from a paper.**

This is exactly what Winstral 2017 measured **[A-FULLTEXT]**:

> "At the underpredicted, **ridge-type sites, the interpolations further decreased forecasted wind
> speeds and degraded performance**. The spatial interpolation, however, improved results at the
> valley locations. Without a clear-cut advantage to either method … **the nearest-gridcell method
> was adopted** for the foregoing analysis. Prior downscaling and debiasing wind studies have
> similarly used a nearest-neighbor approach."
> — Winstral, Jonas & Helbig 2017, *J. Hydrometeor.* 18:335–348, §4a
> <https://www.dora.lib4ri.ch/wsl/dload/wsl%3A12719/PDF/Winstral-2017-Statistical_downscaling_of_gridded_wind-(published_version).pdf>

and what Jiménez & Dudhia 2012 state in their abstract **[A-FULLTEXT]**:

> "The importance of selecting appropriate grid points to compare with observations is also
> examined. **The wind speed from the nearest grid point is not always the most appropriate one for
> this comparison, nearby ones being more representative.**"
> — Jiménez & Dudhia 2012, *JAMC* 51:300–316
> <https://www2.mmm.ucar.edu/wrf/users/physics/phys_refs/SURFACE_LAYER/topo_wind.pdf>

**Our B1 CURRENT is inverse-distance weighting on horizontal distance. Three independent sources —
Winstral's ridge measurement, JD2012's statement, and our own correlation table — say that is the
wrong operator for a ridge site.** IDW is a *smoother*; the ridge is the *maximum*; smoothing a
ridge destroys the signal you want.

### 1.5 [OURS-DESC] Direction-conditional cell choice is real but small outside the dominant sector

Same correlation test, split by LDAPS 10 m box-mean wind direction (meteorological, 8 sectors):

| sector | n | box-mean r | best cell | best r | gain |
|---|---:|---:|---|---:|---:|
| N | 560 | 0.5726 | grid11 | 0.5801 | +0.0075 |
| NE | 2406 | 0.7277 | grid10 | 0.7381 | +0.0104 |
| E | 3027 | 0.6902 | grid05 | 0.7086 | +0.0183 |
| SE | 1618 | 0.5147 | grid01 | 0.5229 | +0.0082 |
| S | 887 | 0.7637 | grid11 | 0.7727 | +0.0090 |
| SW | 2446 | 0.7905 | grid12 | 0.7956 | +0.0050 |
| **W** | **14569 (55.4%)** | 0.7642 | **grid13** | **0.8229** | **+0.0588** |
| NW | 791 | 0.5960 | grid06 | 0.6077 | +0.0118 |

**The flow is 55.4% westerly** — consistent with the brief's "U dominates V (0.68 vs 0.28),
prevailing flow close to east–west across a north–south ridge". The best cell *does* rotate with
sector, but **outside W the gain is +0.005 to +0.018**, i.e. within noise for our purposes.
The whole direction-conditional story collapses into "in the westerly regime, read `grid13`".

**This is why the previously-tested learned displacement lost (−0.000329).** It spent 24 regimes of
freedom to express a signal that lives almost entirely in one regime and can be captured by a
single column. That null is **not** evidence that the box reduction is fine; it is evidence that
*regime-partitioned* selection is the wrong parameterisation of a *one-regime* effect.

---

## §2 THE ONE PIECE OF THE BRIEF I HAVE TO CORRECT

> "the HEFTCom2024 winner stacked per source rather than concatenating"

**This is half true and the half that is false is the operative half.** I read the winner's accepted
manuscript in full.

Olauson, Viotti & Huss (2026), *Int. J. Forecasting*, "The HEFTCom2024 winning model: A stacked
CatBoost approach", **Table 1** **[A-FULLTEXT]**
<https://www.diva-portal.org/smash/get/diva2:2046959/FULLTEXT01.pdf>:

| Model set | Pinball loss | Coverage | Comment (their words) |
|---|---:|---:|---|
| 1. **Stacked CatBoost** | **28.5** | 100% | Main model |
| 2. CatBoost | **28.4** | 93.2% | "Same as 1, except with lagged wind power as a feature and **all weather data in the same model (no stacking)**" |
| 3. Reference (organiser QR) | 38.0 | 99.6% | benchmark |
| CatBoost no target | **28.6** | 93.2% | "Same as 2, except without lagged wind power … thus making a **direct comparison with 'Stacked CatBoost' more fair**" |

And their own explanation of the choice:

> "An observant reader may note that **the second priority model has a slightly lower pinball loss
> than the first priority model. The reason for putting the 'Stacked CatBoost' model first was that
> this model does not use lagged wind power observations as a feature.**"

So on the fair comparison, stacking is **28.5 vs 28.6 = 0.35% better**, and on the unfair one
concatenation actually *wins*. **The winner chose stacking for robustness and coverage (100% vs
93.2% when a source drops out), not for accuracy.** Pu et al. (4th place) say the same thing
explicitly **[A-FULLTEXT]** <https://arxiv.org/html/2505.10367v2>:

> "stacking sister models offers an additional advantage—it **mitigates the risk of missing data
> from one NWP source** during online testing. … with a stacked ensemble, one sister model can
> still generate forecasts even when the other fails."

**We submit all of 2025 at once from a frozen archive. We have zero operational-dropout risk.
The entire measured advantage of stacking in the HEFTCom2024 literature is an advantage we cannot
collect.** See §3.5 for what remains.

---

## §3 THE FIVE STAGES

Ranked at the end by EXPECT/COST. Every stage is filled, as requested, so reverse ablation has a
complete surface.

---

### B1 — GRID-TO-SITE REDUCTION

**SOTA.** **Do not reduce.** Feed every grid cell as its own column and let the learner weight them
— Olauson's HEFTCom2024-winning practice ("All grid points were included in X; i.e. no spatial
aggregation was employed"). Where a reduction *is* used, it must be **chosen per source**: a
high-order statistic for the terrain-resolving source, a distance-weighted mean for the coarse
synoptic source. **Retire IDW and nearest-cell as the site-transfer anchors.**

**EVIDENCE.**
1. **[A-FULLTEXT]** Olauson et al. 2026 IJF, post-competition ablation on the MEPS individual model
   <https://www.diva-portal.org/smash/get/diva2:2046959/FULLTEXT01.pdf>:
   > "All grid points were included in X; i.e. **no spatial aggregation was employed**. In a
   > post-competition analysis, the impact from using only one central grid point or a spatial mean
   > of all grid points was investigated … The pinball loss for solar power was increased by around
   > 10% (one point) and 2% (spatial mean) as compared to the model with all points. **For wind
   > power, the increase was around 2% and 1%, respectively.**"
   Effect size: **all-cells beats spatial-mean by ~1% pinball and beats single-cell by ~2% pinball**,
   1200 MW offshore farm, full-year-2023 test, 31 grid points per sub-group.
   Benchmark: **HEFTCom2024**, winning entry. Same paper also finds spatial detail matters *more*
   than height detail is usually assumed to: "the most important sub-group is **50 m winds lagged
   1 h forward (16%)** … All heights contribute significantly, the most important being **50 m (42%)**".
2. **[A-FULLTEXT]** Winstral et al. 2017 JHM 18:335 §4a — at ridge sites, inverse-**squared**-distance
   interpolation from cell centres **degraded** performance relative to nearest-cell:
   > "At the underpredicted, **ridge-type sites, the interpolations further decreased forecasted wind
   > speeds and degraded performance**. The spatial interpolation, however, improved results at the
   > valley locations. … **the nearest-gridcell method was adopted**."
   Their Table 2 shows the magnitude of what smoothing destroys: at ridges observed mean
   **4.89 m/s** vs COSMO-2 **2.97** vs COSMO-7 **2.61**.
3. **[A-FULLTEXT]** Jiménez & Dudhia 2012 JAMC 51:300, abstract:
   > "**The wind speed from the nearest grid point is not always the most appropriate one for this
   > comparison, nearby ones being more representative.**"
4. **[OURS-DESC]** §1.4 — in our data **IDW ranks 6th of 7 and nearest ranks 7th of 7** against
   measured hub wind; box-`max` beats IDW by +0.054 and nearest by +0.077 correlation.
   **Note the partial disagreement with Winstral**: he found nearest > IDW at ridges; we find
   nearest < IDW. What replicates is that **both lose to a high-order statistic of the box**.
5. **[OURS-DESC]** §1.4b — **the correct operator inverts between sources**: LDAPS `max` 0.8475 >
   `idw` 0.7933; GFS `idw` 0.7200 > `max` 0.5371.
6. **[BASELINE-MISMATCH]** Bengtsson 2025 (KTH / rebase.energy) tunes **Gaussian-blur** box
   reductions per NWP and finds the σ=0.24 blurred 100 m wind to be the **top Shapley feature**
   `[A-FULLTEXT]` <https://www.diva-portal.org/smash/get/diva2:2022553/FULLTEXT01.pdf>. That supports
   *smoothing* — but offshore over the Belgian North Sea, a smooth field and a 2.2 GW distributed
   portfolio: the opposite of a single ridge. Cited so the build does not over-generalise. It is also
   independent support for finding 5: **the right kernel width is a per-source, per-terrain choice.**

**BENCHMARK.** HEFTCom2024 (IEEE DataPort / rebase.energy; overview Browell et al. 2025
<https://arxiv.org/html/2507.01579v1>). Secondary: GEFCom2014 wind track; Kaggle *Hill of Towie*
(Scotland, complex terrain, turbine-level).

**MIGRATION.** *(the field that matters)*
- **PRECONDITION ALREADY CHECKED IN THIS LANE — read this before scoping.** I enumerated the 830
  columns of `train_features.parquet`. Present already: `ldaps__wind50max_speed__{mean,std,min,max,
  q10,q50,q90}`, `ldaps_spatial__{idw,nearest}__wind50max_speed`, the GFS equivalents, and
  `source_disagreement__wind10_speed_{idw,nearest}`. **Absent: the per-cell columns — only 2 of the
  830 names contain `grid` at all.**
  Therefore **"add box-max" is NOT a live action (it exists), and it already captures all but
  +0.0011 of the best single cell.** The live actions are the two below.
- **Live action 1 — add the raw per-cell columns.** `artifacts/cache/<sha>/train_grid_pivot.parquet`
  (914 cols, key `forecast_kst_dtm`) already holds **LDAPS 34 vars × 16 cells** and
  **GFS 41 vars × 9 cells**. Olauson's "all grid points" representation **exists on disk and is not
  being used**. Join a wind-relevant subset — `ldaps__grid{01..16}__{wind10_speed, wind50max_speed,
  wind50min_speed}` and `gfs__grid{01..09}__{wind100_speed, wind80_speed}` = **66 columns** — onto the
  feature frame. **Zero derivation.**
- **Live action 2 — demote IDW and nearest.** Move `ldaps_spatial__idw__*` and
  `ldaps_spatial__nearest__*` out of the anchor set into a control arm. They are the 6th- and
  7th-ranked columns and they are what the current pipeline leans on for site transfer.
  **Do not delete them** — keep them so the reverse ablation can price the change.
  Keep the **GFS** idw column (it is the best GFS reduction) — this asymmetry *is* B5.
- **What is missing / substitute**: nothing. **B1 needs no DEM and no external data at all.**
- **What breaks because our setting differs**: (a) Olauson's 1–2% is *pinball* on a **probabilistic**
  forecast of a **distributed offshore** portfolio; ours is a **point** forecast at **one ridge**
  under `0.5(1−NMAE)+0.5·FICR`. Only the ordering transfers, not the percentage.
  (b) +66 columns on 830 is an ~8% widening. This is a **feature-space** expansion, not the
  **weight-space** expansion the AGENTS.md fold-outside gate rejected (7×3 = 21 dof blends), so it is
  not the same failure mode — but it must run the same fold-outside discipline.
  (c) **Overlap warning**: the repo has already tested "grid pivot + divergence/vorticity/coherence
  geometric" and a "learned spatio-temporal displacement". Diff the proposed column names against
  those receipts before building. Raw per-cell *levels* appear new; per-regime *selection* is not.

**RISK (how this fails *here*, with our numbers).** The strongest reduction is **already a feature**
(`__max`, 0.8475, vs the best cell's 0.8486). So a tree that is already splitting on `__max` has most
of the signal, and the incremental value of 66 raw cell columns is whatever the *combination* of
cells adds beyond their maximum — which is exactly what the previously-tested displacement block
tried to extract and failed at (**−0.000329** against a **−0.000411** noise arm). Second risk: 66
new columns give the GBM 66 new places to overfit at a site where three folds already oscillate.
Third: demoting IDW/nearest may do nothing at all, because a tree can simply stop splitting on a
weak column — the demotion is only a real change if those columns are currently being used as a
*base* the rest of the model corrects against.

**COST.** **2–4 h.** Nothing to install. No download. Pure column plumbing on parquet already cached.
The 20-minute precondition check that would normally open this stage **has already been performed in
this lane** (see MIGRATION), so the stage starts from a known state.

**EXPECT.** **+0.0006 on Total** (range −0.0004 … +0.0020). Revised **down** from my first estimate
after finding that `__max` already exists: the headline "retire IDW" turns out to be mostly
*already mitigated* by an order statistic sitting next to it. Our paired measurement sd is 0.00075,
so this is **under 1σ**. Its real value is that it is the **cheapest** stage, it is the **carrier of
B5's per-source asymmetry**, and it is a **precondition** for B4 (a terrain index is pointless while
the box is still being smeared).

### B2 — VERTICAL EXTRAPOLATION TO HUB HEIGHT (117 m)

**SOTA.** **Replace the fixed power law and the two-point log law with a learned,
stability-conditioned extrapolator supervised on the measured nacelle wind.** Named method:
**Optis et al. 2021 random-forest vertical extrapolation with an explicit stability input**, in the
Bodini & Optis round-robin validation protocol. The physics fallback if a learner is refused is a
**stability-corrected Monin–Obukhov / Businger–Dyer profile** with ψ_m driven by a bulk stability
proxy — but the literature is clear the learned version wins.

**EVIDENCE.**
1. **[A-FULLTEXT]** Optis, Bodini, Debnath & Doubrawa 2021, *Wind Energ. Sci.* 6:935–948, **Table 8**
   <https://wes.copernicus.org/articles/6/935/2021/wes-6-935-2021.pdf>. Random-forest extrapolator at
   NYSERDA buoy E05, **with vs without one stability feature (ΔT_air–sea)**:

   | metric | all data w/o | all data **with** | high-shear w/o | high-shear **with** |
   |---|---:|---:|---:|---:|
   | Bias (m/s) | 0.03 | 0.04 | −1.05 | **−0.58** |
   | **cRMSE (m/s)** | 1.07 | **0.84** | 1.46 | **1.29** |
   | EMD (m/s) | 0.19 | 0.12 | 1.05 | 0.58 |
   | R² | 0.95 | 0.97 | 0.89 | 0.91 |

   **Adding a single stability variable cut unbiased RMSE by 21.5% overall and cut the high-shear
   bias by 45%.** Also: "the random forest machine-learning model **significantly outperformed** the
   other models" (log profile, single-column model, DTU method, and WRF itself).
2. **[A-FULLTEXT]** Bodini & Optis 2020, WES 5:489, round-robin discipline
   <https://wes.copernicus.org/preprints/wes-2020-2/wes-2020-2.pdf>:
   > "Using the round-robin approach proposed here, this improvement **drops to 20% and 14%**,
   > respectively. … round-robin validation should be the standard for machine-learning-based,
   > wind-speed extrapolation methods."
   **This is the honest number: ~14–20%, not the same-site number.** Note that *we are the same
   site* — there is only one site — so the same-site number is arguably the right one for us, but
   the temporal fold-outside discipline must replace the spatial one.
3. **[B-SNIPPET]** Touma 1977, *JAPCA* <https://www.tandfonline.com/doi/pdf/10.1080/00022470.1977.10470503>:
   > "The power law exponents varied from **0.1 to 0.3 during the day** when superadiabatic and
   > neutral lapse rates prevail and from **0.2 to 0.8 during nighttime**."
4. **[OURS-DESC]** §1.2 and §1.3. Our own implied α runs **0.067 → 0.220** diurnally and
   **1.695 → 1.248** (ratio) across BLH quartiles. Our LDAPS 50 m field already **overshoots** the
   117 m truth (ratio 0.945).
5. **[B-SNIPPET]** REWS, IEC 61400-12-1 Ed.3 (2022): "the REWS method exhibited a **2.86%
   improvement** over the TPC method"
   <https://www.sciencedirect.com/org/science/article/pii/S2352097326000519>. Included for
   completeness — see MIGRATION for why REWS is only half-available to us.

**BENCHMARK.** No single public leaderboard for vertical extrapolation. The de-facto benchmark is
the **NYSERDA floating-lidar round-robin** (Optis/Bodini, NREL) and, for the power-conversion end,
HEFTCom2024. Olauson's MEPS feature analysis is the closest competition-grade evidence that height
matters: **[A-FULLTEXT]** "All heights contribute significantly to the model, the most important
being **50 m (42%)** and the least important being **10 m (12%)**."

**MIGRATION.** *(the field that matters)*
- **Target column to construct** — this is the new object:
  `research/scratch/` → build `hub_ws_obs` = per-hour **median across the 12 Vestas `_ws` columns**
  (and a second series from the 5 Unison `_ws` for g3, 2023+), resampled `1h`, `label='right'`,
  `closed='right'` on `kst_dtm`. Aligns 1:1 with `forecast_kst_dtm`. **26,304 rows, no gaps** in the
  Vestas join (I verified `n=26304`). Median, not mean, because individual nacelle anemometers ice
  and stick; the brief warns the power columns are corrupted, so treat `ws` as trusted but noisy.
- **Inputs already on disk, no derivation needed**: `ldaps__grid*__wind10_speed`,
  `..__wind50max_speed`, `..__wind50min_speed`, `..__wind5_speed`, `gfs__grid*__wind80_speed`,
  `..__wind100_speed`, `..__wind10_speed`.
- **Stability inputs already on disk — this is the part that is currently unused**:
  - `ldaps__grid{01..16}__etc_0_blh` — **boundary-layer height, per cell, forecast**. This is the
    Optis ΔT_air–sea analogue and §1.3 shows it separates the ratio 1.695 vs 1.248.
  - `ldaps__grid*__heightAboveGround_2_t` and `gfs__grid*__isobaricInhPa_850_t` → **bulk lapse rate**
    over the layer, the standard bulk-Richardson numerator.
  - `ldaps__grid*__surface_0_NDNSW`, `..__surface_0_NDNLW` (net down SW/LW), `gfs__..__surface_0_dswrf`,
    `..__dlwrf` → **surface radiative forcing**, i.e. the sign of the sensible heat flux, i.e.
    the Pasquill class. Free.
  - `gfs__grid*__surface_0_gust` ÷ `gfs__grid*__wind10_speed` → **gust factor**, a standard turbulence
    -intensity / stability proxy `[B-SNIPPET]` Jeans 2023 WES 9:2001: "A simple generalised form of
    gust factor relationship is adopted … a **pragmatic proxy for stability classification**".
  - `gfs__grid*__planetaryBoundaryLayer_0_u/v` and `..__VRATE` — PBL-mean transport wind.
  - `ldaps__grid*__heightAboveGround_5_XBLWS/YBLWS` — 5 m boundary-layer wind components, giving a
    **third low level** so the profile is over-determined rather than exactly-determined.
- **Concrete construct** — do **not** re-derive α as a fixed constant. Build:
  `alpha_hat(t) = ln(w50(t)/w10(t)) / ln(50/10)` per cell (a *data* column, not a parameter), then
  hand the learner `alpha_hat`, `blh`, `hub/blh = 117/blh`, lapse rate, gust factor, and let it
  learn `hub = f(w50, alpha_hat, stability)`. This is the Optis structure exactly.
- **What is missing and what the substitute is**: we have **no measurement above 100 m other than the
  nacelle itself**, so a physically-closed profile is impossible. Substitute: the nacelle *is* the
  117 m measurement, so we never need to extrapolate at training time — we **regress** to it.
  For **REWS**: V126 sweeps 54–180 m, U136 sweeps 49–185 m; our highest NWP level is 100 m, covering
  only the **lower half** of the rotor. So REWS is computable only as a 3-level (10/50/100 m)
  area-weighted cubic mean with the upper half extrapolated. **It is still worth building**, because
  V126 and U136 have *different* rotor discs — this is the one place where a genuine
  `profile-shape(t) × rotor-geometry(g)` interaction exists that the 3-level group dummy cannot
  absorb (see the standing "static constants are absorbed by group dummies" lesson).
- **What breaks**: (a) The nacelle anemometer sits **behind the rotor** and reads a wake-distorted
  speed; industrially this is corrected by a **Nacelle Transfer Function**
  `[D-QUALITATIVE]` <https://powerveritas.com/insights/nacelle-transfer-function-accuracy>. We have
  no NTF. **But we do not need one**: the NTF is a monotone distortion and our downstream target is
  power at the same turbines, so a wake-distorted hub wind is arguably a *better* regression target
  than free-stream. Do not "correct" it toward free-stream — that is precisely the
  **teacher-target-realism** trap already replicated twice in this repo (making the teacher closer to
  metered truth LOSES, because its value is being noise-free).
  (b) Unison SCADA starts 2023-01, and **g3 has no 2022 labels anyway** — the gap aligns, so g3 loses
  nothing it had. (c) Trees cannot extrapolate; if 2025 contains hub winds outside the 2022–24 range
  the learned extrapolator saturates. Mitigation: keep the physical power/log law as a **fallback
  column**, not as the anchor.

**RISK (how this fails *here*).** The dominant risk is **that it works on wind and does not move
Total.** Our decomposition says NWP-to-hub-wind carries 0.13022 of 0.13858 MAE, and perfect measured
wind scores 0.869922 vs our 0.636184 — so the headroom is 0.2337. But the mapping wind→Total is
compressed by the power curve: below cut-in and above rated, large wind errors produce zero power
error, and the metric only scores rows with actual ≥ 0.1·capacity. A 21.5% cRMSE cut on wind will
**not** produce a 21.5% cut on scored MAE. Second risk: the SCADA `ws` is 10-min and the brief flags
the *power* columns as time-scrambled — if the scrambling touched the **index** rather than the power
column alone, `ws` is misaligned too and this entire stage is built on sand. **Verify the `ws`
alignment independently before building**: e.g. check that `hub_ws_obs` reproduces the known diurnal
and seasonal cycle and correlates ≥0.8 with LDAPS (it does — §1.4 — which is itself the alignment
receipt, and is the strongest available evidence that `ws` is *not* scrambled).

**COST.** **8–14 h.** Nothing to install (lightgbm/sklearn/scipy present; no torch needed — Optis
used a random forest). Two sub-steps: (1) build and validate `hub_ws_obs` (3 h, and it is reusable by
B3/B4/B5); (2) stability feature block + learned extrapolator (5–11 h).

**EXPECT.** **+0.0045 on Total** (range +0.0010 … +0.0110). This is the **largest single number in
the cluster** and I want to justify it rather than assert it. Reasoning: Optis measures −21.5%
cRMSE from adding stability to a learned extrapolator; our stability signal is at least as strong
(3.3× swing in α, monotone in an available forecast variable); the wind→Total headroom is 0.2337;
even capturing 3% of that headroom is +0.0070, and I am discounting to +0.0045 for power-curve
compression and for the fact that a GBM with `blh` in the matrix already recovers some of it.

---

### B3 — SYSTEMATIC BIAS CORRECTION / MOS ON THE NWP FIELDS

**SOTA.** **Per-source, stability- and direction-conditional multiplicative MOS applied to the wind
fields before the power learner**, supervised on `hub_ws_obs`. In postprocessing language this is
**EMOS/DRN with the NWP field as covariate**; in our deterministic setting the right reduction is
**Distributional Regression / quantile-forest MOS collapsed to its conditional median**, or simply a
learned multiplicative correction `w_corrected = w_raw · g(stability, direction, source)`.

**EVIDENCE.**
1. **[B-SNIPPET]** Bias correction of NWP wind with explainable ML, *Earth and Space Science* 2025
   <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025EA004801>:
   > "ML model substantially improves the original NWP forecasts, **reducing the root mean square
   > error of approximately 20%–40% at most stations**"
   `[BASELINE-MISMATCH]` — baseline is raw NWP at a station, not a GBM with 830 columns.
2. **[B-SNIPPET]** Schulz & Lerch 2022, *Mon. Wea. Rev.* 150:1, the systematic ML-postprocessing
   comparison (DRN, BQN, HEN, QRF, EMOS, …) for wind gusts
   <https://journals.ametsoc.org/view/journals/mwre/150/1/MWR-D-21-0150.1.xml>; DRN = Rasp & Lerch
   2018 distributional regression network. This is the **method taxonomy** to name, not a number to
   transfer.
3. **[B-SNIPPET]** Taillardat et al. 2016 QRF postprocessing: "**QRF_M is the best technique for
   CRPS and CRPSS** … sharper, more reliable" <https://meteofrance.hal.science/meteo-03544106/document>.
4. **[OURS-DESC]** §1.2/§1.3 — **the biases are enormous and structured**: GFS 100 m carries a
   multiplicative deficit of **1.848 overall**, swinging **1.279 (14:00) → 2.499 (07:00)** and
   **1.357 (convective) → 2.464 (stable)**. LDAPS 50 m carries **0.945**, swinging 0.768 → 1.127.
   These are not small residual biases; **GFS is wrong by a factor of two at night.**

**BENCHMARK.** EUPPBench / the Schulz–Lerch wind-gust benchmark
(<https://github.com/benediktschulz/paper_pp_wind_gusts>) for method ranking. There is no
competition benchmark for "MOS as a preprocessing stage feeding a power learner", which is itself
informative — see RISK.

**MIGRATION.**
- **Fit surface**: `hub_ws_obs` (from B2) as target; per-source raw wind columns as input.
  Two separate correction functions, one per source — **`g_ldaps`** and **`g_gfs`** — because §1.2
  shows their biases differ by a factor of 2 and have **opposite sign** (LDAPS 50 m over-predicts at
  0.945; GFS 100 m under-predicts at 1.848).
- **Conditioning variables** (all already on disk, listed in B2 MIGRATION): `blh`, hour, lapse rate,
  gust factor, and **wind direction sector** from `ldaps__grid*__heightAboveGround_10_10u/10v`.
- **Output columns**: `ld_w50_mos`, `ld_w10_mos`, `gfs_w100_mos`, `gfs_w80_mos` — 4 corrected
  columns, added **alongside** (never replacing) the raw ones, so the reverse ablation can attribute.
- **Fold discipline**: the correction must be fitted **fold-outside**. Fitting `g` on all of
  2022–2024 and then scoring 2022–2024 folds is in-sample and will read as a phantom gain. Use the
  same fold structure as the existing 3-fold surface; note AGENTS.md's warning that three folds
  cannot estimate per-group weights — `g` here has far fewer effective dof than that, but the
  warning applies to any per-group variant of `g`.
- **What is missing**: nothing external.
- **What breaks**: nothing structural. This stage is pure re-expression of columns we hold.

**RISK (how this fails *here*).** **The redundancy risk is severe and specific.** A GBM given
`w_raw`, `blh`, and `hour` can already learn `w_raw · g(blh, hour)` internally — a tree splits on
`blh`, then splits on `w_raw`, and reproduces a piecewise-constant `g`. **So the honest expected
gain from B3 is not the 20–40% of the literature; it is the residual that trees cannot express
efficiently**, which is the *smooth multiplicative* part. The reason to expect a non-zero residual is
that trees approximate `w·g` with axis-aligned steps and need many splits to do it, whereas the MOS
column hands it over in one. This is a **regularisation / sample-efficiency** gain, not an
information gain — the same honest framing the earlier terrain lane applied to `Sx(g,θ)`.
Second risk: correcting the field toward the *nacelle* wind while the downstream target is *power*
introduces a second teacher; the repo has twice measured that making a teacher more "realistic"
loses. Mitigate by adding `*_mos` columns rather than substituting.

**COST.** **5–8 h**, and **~2 h of it is shared with B2** (the `hub_ws_obs` target). Nothing to
install. Strictly: build B2's target first, then B3 is cheap.

**EXPECT.** **+0.0015 on Total** (range −0.0005 … +0.0040). Discounted hard for redundancy with a
tree learner that already holds the conditioning variables.

---

### B4 — TERRAIN AND MICRO-SITING ADJUSTMENT

**SOTA.** Two tiers.

**Tier 1 (no DEM, do this first):** the **model-orography-deficit correction** — Jiménez & Dudhia's
insight that the *sign* of the terrain-induced wind bias is set by the **Laplacian of the model
orography**, combined with Pogumirskis et al.'s finding that the **model-minus-real elevation
difference** predicts wind bias better than elevation itself. **We already hold per-cell model
orography** (`ldaps__grid{01..16}__surface_0_h`, values **868.8 – 1001.2 m**), so the nondimensional
Laplacian and the subgrid σ are computable **from the columns we have, with no DEM at all**.

**Tier 2 (needs a DEM):** **Winstral Sx** — direction-dependent maximum-upwind-slope exposure,
computed **per LDAPS cell** (not per group), indexed at runtime by that cell's own forecast wind
direction; plus **TPI** terrain classification, plus **RIX** as a flow-separation flag.

**EVIDENCE.**
1. **[A-FULLTEXT]** Jiménez & Dudhia 2012 JAMC 51:300, Eq. (1) and §4:
   > "D²h_{i,j} = 0.25(h_{i+1,j} + h_{i,j+1} + h_{i−1,j} + h_{i,j−1} − 4h_{i,j}) … **Positive values
   > indicate the presence of a minimum (a valley), and negative values denote a maximum (hills or
   > mountains)** … a threshold value of −20 m has been selected to classify the individual grid
   > cells."
   And the operational recipe: "WRFnew uses the wind speed at the **elevation of the maximum height
   of the subgrid-scale orography** to represent the wind speeds at those grid cells with
   **D²h < −50 m and σ_sso > 100 m**."
   Their core lesson, which the earlier repo lane also flagged: **variance alone does not give you a
   sign; you need the Laplacian.**
2. **[A-FULLTEXT]** Pogumirskis et al. 2026, WES preprint wes-2026-24, **508 observation campaigns
   across Europe at wind-turbine rotor heights**, 7 model datasets
   <https://wes.copernicus.org/preprints/wes-2026-24/wes-2026-24.pdf>:
   > "The magnitude of the wind speed bias ranges from **0.1 to 0.9 m s⁻¹ per 100 m of elevation
   > difference, depending on the season and time of day**." … "**the difference between model and
   > real-world elevation provides significantly stronger statistical explanatory power for the model
   > bias than the real-world elevation alone**" … "Models generally underestimate wind speeds in
   > mountainous regions".
   **Our LDAPS orography deficit is 80–140 m below the true ridge**, so this predicts a low bias of
   **~0.1–1.3 m/s**, seasonally and diurnally modulated. §1.2/§1.3 confirm exactly that modulation.
3. **[A-FULLTEXT]** Winstral et al. 2017 JHM, **Table 4**, downscaling with Sx + TPI. **Ridge class
   only** (our terrain class), COSMO-2:

   | | calibration | validation |
   |---|---|---|
   | RMSE ridges (m/s) | (3.55) → 3.49 | (3.59) → **3.38** |
   | **MBE ridges (m/s)** | (**−1.92**) → **−0.04** | (**−2.01**) → **−0.24** |
   | KSD ridges | (0.29) → … | … |

   parentheses = uncorrected. **Read this carefully: on ridges the method removes essentially all of
   a −1.92 m/s bias but improves RMSE by only 1.7% (calibration) / 5.8% (validation).**
   Sx-based terrain downscaling is a **bias** instrument, not a **variance** instrument.
4. **[C-SECONDARY]** Wind-Topo (Dujardin & Lehning 2022, QJRMS 148:1368): alpine validation
   COSMO-1 bias 0.72 → −0.07, MAE **1.77 → 1.21 m/s (−31.6%)**
   <https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4265>. `[BASELINE-MISMATCH]`, and it is a
   CNN trained on 261 stations — we have 1.
5. **[B-SNIPPET]** Wagenbrenner/Forthofer 2016, ACP 16:5229: "The NWP models tended to **under
   predict wind speeds on the windward slopes, ridgetops**, and surrounding flat terrain, and over
   predict on the lee side of the butte."
   <https://research.fs.usda.gov/download/treesearch/61477.pdf>
6. Convergence check: **five independent sources** — Winstral (ridges MBE −1.92), JD2012 (WRF low
   bias over mountains/hills), Pogumirskis (models underestimate in mountainous regions),
   Forthofer (under-predict on ridgetops), and KMAPP/Kum & Ho 2021 (LDAPS **under**-estimates at high
   elevation) — **all say the same thing about ridge tops.** Our own §1.2 shows LDAPS 10 m at 1.465×
   deficit. This is the best-corroborated fact in the entire cluster.

**BENCHMARK.** None competition-grade. The nearest thing is the Jülich **benchmark dataset for
meteorological downscaling** (Langguth et al., includes a 100 m-wind downscaling task)
<https://juser.fz-juelich.de/record/1034617/files/paper.pdf> `[B-SNIPPET]`.

**MIGRATION.**
- **Tier 1, zero external data.** `train_grid_pivot.parquet` gives `ldaps__grid{01..16}__surface_0_h`
  on a **4×4 lattice** (I read the values: 992.6, 936.6, 868.8, 926.5 / 997.6, 1001.2, 934.4, 869.4 /
  889.2, 966.7, 999.1, 959.8 / 896.2, 956.6, 967.9, 933.8). Reshape to 4×4 and compute JD2012's
  `D²h` at the 4 interior cells (grid06, grid07, grid10, grid11) directly. Also compute
  `sigma_sso = std(16 cells) = ~44 m` and `dz = 1078 − h_cell` per cell.
  **These are constants — and constants are absorbed by the 3-level group dummy.** They only become
  informative as **`dz × f(t)`**: e.g. `dz_x_shear = dz · alpha_hat(t)`,
  `dz_over_blh = dz / blh(t)`, `dz_x_stability = dz · (1/blh)`. This is the repo's own established
  "static site constant must be multiplied by a time function" pattern, and Pogumirskis independently
  supplies the physics for it ("depending on the season and time of day").
  Columns: ~6. Derivation: pure numpy on 16 numbers.
- **Tier 2, needs a DEM.** **Which DEM, and the smallest way to get it for one 4×4 box:**

  | DEM | res | licence | commercial? | verdict |
  |---|---|---|---|---|
  | **ALOS AW3D30 (JAXA)** | 30 m | JAXA ToU | **"Any of the commercial and non-commercial purposes can be used free of charge under the conditions of the '5. Terms of Use'"** `[A-FULLTEXT]` <https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm> | **CHOOSE THIS.** The licence text is unambiguous on commercial use. Attribution © JAXA. |
  | **Copernicus GLO-30** | 30 m | Copernicus | "available worldwide with a **free licence**" `[B-SNIPPET]` <https://dataspace.copernicus.eu/...> ; "GLO-30 **Public** provides **limited** worldwide coverage" `[B-SNIPPET]` <https://registry.opendata.aws/copernicus-dem/> | Good backup / cross-check. Slight tile-availability uncertainty; verify the Korea tile exists. |
  | FABDEM | 30 m DTM | **CC BY-NC-SA** | **NO** | **Excluded on licence.** |
  | NGII 5 m (국토지리정보원) | 5 m | 공공누리, application required | uncertain | Excluded on schedule risk (written application/approval). |
  | Global Wind Atlas | 250 m | CC BY 4.0 | yes, but | **Excluded**: DTU states GWA is built by "**dynamically downscaling ERA5 reanalysis**". Reanalysis-derived → forbidden by AGENTS.md. Its clean components (RIX, roughness, elevation) are DEM-only functions we can compute ourselves. |

  **Smallest acquisition**: the site is **37.16–37.30 N, 128.92–129.00 E**. A ±20 km domain fits
  inside **one 1°×1° tile: N37/E128**. AW3D30 tile `N037E128` is a single ~50–70 MB GeoTIFF; or
  OpenTopography's bounding-box API returns just the clip
  (<https://portal.opentopography.org/apidocs/>, free API key). **One file, one download, done.**
  Note both AW3D30 and GLO-30 are **DSMs** (canopy included) — fine for Sx/TPI at 500–5000 m scales,
  **not** fine for interpreting local TRI/VRM as bare-earth roughness.
- **Sx construction** (per LDAPS cell, so it survives the group-dummy rank argument): precompute a
  lookup `Sx_j(θ, d)` for j=1..16, θ ∈ {0°,10°,…,350°}, d ∈ {500, 2000, 5000} m; at runtime index by
  that cell's own forecast direction `θ_j(t) = atan2(−u_j, −v_j)`. **Predeclare the radii and the 30°
  sector width and never tune them** — otherwise the selection is the result. Implementation
  reference: `topocalc` (USDA-ARS-NWRC, <https://github.com/USDA-ARS-NWRC/topocalc>) or SAGA
  `ta_morphometry_29` — but a direct numpy ray-scan is ~40 lines and avoids an install.
- **What breaks**: (a) the group-dummy rank argument kills any per-group static index — enforce
  per-**cell** or ×f(t). (b) Winstral's own numbers say the ridge gain is **bias**, and a GBM with a
  group dummy **already removes a constant per-group bias**. So the transferable part of Winstral is
  only the *direction-dependent* part of the bias, which our §1.5 measures as small outside the
  westerly sector. (c) A DSM over forest contaminates short-scale indices.

**RISK (how this fails *here*, with our numbers).** The repo has **already** run a
dimensionless-mountain-height block (`Hhat = h0·N/U`) which passed a |corr|<0.85 novelty screen and
scored **−0.000640** against a noise arm that itself scored **−0.000411**. That is the closest prior
to Tier 1 and it lost. **The honest reading**: it lost *alone*, at a granularity where paired sd is
0.00075, so it is a ~0.3σ null, not a refutation — but it is real evidence that the effect size here
is **small**, not large. Tier 2 adds a download, a licence trail, ~48 new columns, and a
selection-bias surface (radii, sector width), for a mechanism whose own source paper says the ridge
RMSE gain is **1.7–5.8%** and whose bias gain is **already absorbed by our group dummy**. I rate
Tier 2 as **the weakest expected-value stage in the cluster** and I would not build it before B1/B2/B5.

**COST.** **Tier 1: 3–5 h**, nothing to install, no download. **Tier 2: 12–20 h** plus a DEM
download, plus a licence receipt (source, licence text, retrieval date — AGENTS.md requires this),
plus optional `rasterio`/`rioxarray` (**check first**: neither is listed as installed, and anything
that would downgrade sklearn/scipy/numpy is disqualified — `rasterio` normally does not, but verify;
a raw GeoTIFF can also be read with numpy + the tile's affine header if it comes as a plain array).

**EXPECT.** **Tier 1: +0.0006** (range −0.0005 … +0.0018). **Tier 2: +0.0004** (range −0.0010 …
+0.0020). Combined **+0.0010**. Low on purpose: a directly-adjacent prior already measured −0.000640.

---

### B5 — MULTI-SOURCE COMBINATION OF LDAPS AND GFS

**SOTA.** The literature's headline — "stack per source" — **does not transfer to us for the reason
it is usually given** (§2). What *does* transfer is narrower and, I think, more valuable:
**per-source preprocessing before a shared learner.** Give each source its **own** bias/stability
correction (B3) and its **own** grid reduction (B1), then concatenate. Keep a per-source sister
model **only** as a diversity probe, not as the deliverable.

**EVIDENCE.**
1. **[A-FULLTEXT]** Olauson et al. 2026 IJF, Table 1 (§2 above): stacked **28.5** vs concatenated
   **28.6** on the fair comparison — **0.35%** — and the stated reason for preferring stacking is
   **coverage 100% vs 93.2%**, i.e. source-dropout robustness. HEFTCom2024 **winning entry**.
2. **[A-FULLTEXT]** Pu et al. 2025 (HEFTCom2024, team GEB, 4th forecasting), Table 2 and §5.2
   <https://arxiv.org/html/2505.10367v2>:

   | approach | Case I wind MPL | MCRPS | MWS | Case II wind MPL | MCRPS | MWS |
   |---|---:|---:|---:|---:|---:|---:|
   | DWD only | 28.96 | 53.18 | 357.72 | 18.90 | 34.66 | 230.98 |
   | GFS only | 30.44 | 55.87 | 381.51 | 19.19 | 35.27 | 243.49 |
   | **Stacking** | **27.13** | **49.74** | **334.65** | **17.69** | **32.46** | **220.20** |

   > "the stacked models achieve improvements of **10.87% and 6.32%** compared to the GFS and DWD
   > models in Case I. In Case II … **7.82% and 6.40%**."
   **[BASELINE-MISMATCH — critical]** their baseline is a **single-source** model. Ours is already
   two-source concatenated. **The 6–11% is the gain from adding a second source at all, which we
   already have.** Their own explanation is spatial: "we observed **discrepancies in grid point
   density and coverage area**".
   Also note: for **solar** the same stacking gave **0.15% / 0.29%** — i.e. the gain appears only
   where the sources genuinely differ.
3. **[A-FULLTEXT]** Bengtsson 2025 (KTH / rebase.energy), Belgian offshore, **MAE and RMSE in % of
   capacity — the same unit as our NMAE**:

   | model | MAE (%cap) | RMSE (%cap) |
   |---|---:|---:|
   | Elia TSO baseline | 6.238 | 10.119 |
   | ECMWF_HRES alone | 6.305 | 10.057 |
   | NCEP_GFS alone | 7.457 | 11.297 |
   | MeteoFrance ARPEGE-EU alone | 7.579 | 11.344 |
   | MetOffice GlobalHiRes alone | 6.778 | 10.197 |
   | Naive mean of 4 | 6.118 | 9.035 |
   | **Cov.-weighted mean** | **5.960** | **8.990** |

   Combining 4 sources bought **MAE 6.305 → 5.960 = −0.345 pp of capacity**. Their residual
   correlation matrix (their Table 3) is **0.575–0.645** between sources.
   **[BASELINE-MISMATCH]** — and decisively so, because **our minimum pairwise error correlation is
   0.934** (repo-measured, closed axis). Bengtsson's gain is bought with diversity we do not own.
4. **[OURS-DESC]** LDAPS and GFS are **not** interchangeable for us, on four independent counts:
   - **Skill**: correlation with measured hub wind **0.8475 (LDAPS box-max) vs 0.7200 (GFS idw)**.
   - **Bias sign and size**: multiplicative bias **0.945 (LDAPS 50 m, over-predicts)** vs
     **1.848 (GFS 100 m, under-predicts by a factor of ~2)** — *opposite sign* (§1.2).
   - **Stability sensitivity**: GFS's bias swings **1.279 → 2.499** across the diurnal cycle;
     LDAPS 50 m swings only **0.768 → 1.127** (§1.3). GFS is twice as unstable a source.
   - **Optimal reduction operator, and this is the decisive one**: for LDAPS `max` (0.8475) beats
     `idw` (0.7933); for GFS `idw` (0.7200) beats `max` (0.5371). **The ordering inverts** (§1.4b).

   **Per-source *treatment* is therefore not a stylistic preference — a single shared rule is
   provably wrong for one of the two sources. Per-source *stacking*, by contrast, is unsupported.**

**BENCHMARK.** **HEFTCom2024** (the only benchmark with a published per-source ablation by the actual
winner). Secondary: Predico / Elia collaborative forecasting (Bengtsson's setting).

**MIGRATION.**
- **Do**: fit `g_ldaps` and `g_gfs` separately in B3; reduce LDAPS's 16 cells and GFS's 9 cells
  separately in B1 (different geometry: 1.5 km vs 0.25°, so a shared reduction rule is wrong on its
  face); then **concatenate** into the one design matrix, as now.
- **Do not**: build two sister models and blend their outputs. AGENTS.md already records that the
  fold-outside gate rejects multi-dof blends (7×3 = 21 dof: in-sample 0.646821 → fold-outside
  0.643888; per-group 3 dof: 0.640253 → 0.635453 vs uniform 0.639170), and that action-level
  ensembling is a **closed axis** (min pairwise error correlation 0.934, MCS returns 8 tied models).
  A 2-member sister stack is 1 dof and would survive the gate — but its *ceiling* is Olauson's
  measured **0.35%**, and on our correlations even that is optimistic.
- **One diagnostic worth running** (cheap, non-committal): compute the **error correlation between an
  LDAPS-only and a GFS-only** hub-wind regression. If it comes back below ~0.8, revisit; if it comes
  back at 0.93 like every other pair in this repo, close B5 permanently with a receipt.
- **What is missing**: nothing. **What breaks**: GFS 9 cells at 0.25° span ~75 km — GFS is not
  resolving our site at all, it is resolving the synoptic forcing. Treat GFS columns as
  **synoptic/stability context** (850 hPa, PBL, gust, radiation), **not** as a competing site wind.
  That reframing is, I think, the actual content of B5 for us.

**RISK (how this fails *here*).** B5 as "stack the two sources" returns **0.000 or negative** — I am
fairly confident of this, on Olauson's own table plus our 0.934 correlation floor. B5 as "treat the
sources differently in B1/B3" is not really a separate stage at all; it is a **constraint on how
B1 and B3 are built**, which is why it must be built *with* them and why reverse ablation will
struggle to attribute it independently. **Flag that to the ablation design now**: if B1 and B3 are
implemented per-source from the start, removing "B5" leaves nothing to remove.

**COST.** **1–2 h** if implemented as a constraint on B1/B3 (essentially free — it is a design rule).
**6–10 h** if implemented as an actual sister-model stack, which I recommend against.

**EXPECT.** **+0.0010 on Total** as a design constraint (range 0.0000 … +0.0028) — revised **up**
after §1.4b: the reduction-operator inversion is a real, measured, currently-uncorrected defect
(GFS is presently reduced the same way LDAPS is), not merely a stylistic point.
**−0.0002** as a sister stack (range −0.0015 … +0.0008). **Do not build the stack.**

---

## §4 RANKING BY EXPECT / COST

| rank | stage | EXPECT (Total) | COST (h) | EXPECT/COST (per 10 h) | needs anything not installed? |
|---:|---|---:|---:|---:|---|
| **1** | **B5** as a *design constraint* (per-source reduction + per-source MOS) | **+0.0010** | **1.5** | **+0.0067** | no |
| **2** | **B2** stability-conditioned learned extrapolation on `hub_ws_obs` | **+0.0045** | **11** | **+0.0041** | no |
| **3** | **B3** per-source multiplicative MOS on the fields | **+0.0015** | **6** (2 shared w/ B2) | **+0.0025** | no |
| **4** | **B1** grid reduction (raw per-cell cols; demote IDW/nearest) | **+0.0006** | **3** | **+0.0020** | no |
| **5** | **B4 Tier 1** orography-deficit × time-function (no DEM) | **+0.0006** | **4** | **+0.0015** | no |
| **6** | **B4 Tier 2** Sx / TPI / RIX from AW3D30 | **+0.0004** | **16** | **+0.0003** | DEM download + possibly `rasterio` |
| — | B5 as a sister-model **stack** | **−0.0002** | 8 | negative | no — **do not build** |

Two notes on reading this table honestly.

**(a) The ranking moved once I checked the repo instead of only the literature.** B1 fell from 1st to
4th when I found `ldaps__wind50max_speed__max` already in the 830 columns (it captures all but
+0.0011 of the best single cell). B5 rose from 2nd to 1st when I measured the reduction-operator
inversion between LDAPS and GFS (§1.4b). **Neither move came from a paper.** If you only implement
one thing from this lane, implement the per-source split — it is 1.5 hours and it fixes a defect
that is currently costing GFS about 0.18 of correlation.

**(b) The additive total is not the joint total.** Naive sum over stages 1–5: **+0.0082**. I do not
believe it. These stages share one mechanism — the NWP-to-hub-wind transfer — and the redundancies
are structural: B3 is largely inside B2 (a learner given `blh` can build its own MOS), and B4 Tier 1
is largely inside B3 (an orography-deficit term conditioned on stability *is* a stability-conditioned
bias correction). **My honest joint expectation for the whole cluster is +0.0035 to +0.0050 on
Total**, i.e. roughly **0.6375 → 0.641–0.642**. That is **5–7× our paired measurement sd**, so it is
genuinely detectable — and it does **not** on its own reach 0.66. This cluster is a real gain, not a
rescue.

---

## §5 BUILD ORDER

Ordered for **evidence per hour**, with cheap kill-switches first. Each step names the thing that
would stop the next one.

**1. B5-as-constraint + B1 — per-source grid reduction (4 h combined).**
   These are one edit, not two. Split the reduction rule by source: keep `max`/`q90` as the LDAPS
   anchor and **demote `ldaps_spatial__idw__*` and `ldaps_spatial__nearest__*` to a control arm**;
   keep `gfs_spatial__idw__*` as the GFS anchor and **demote `gfs__wind100_speed__max`**. Then add
   the 66 raw per-cell wind columns from `train_grid_pivot.parquet`. Score fold-outside.
   *Record explicitly that B5 is entangled here — see step 6.*
   *No kill-switch needed: the precondition check was already done in this lane (§B1 MIGRATION).*

**2. B2a — build and VALIDATE `hub_ws_obs` (3 h). THE PIVOT OF THE WHOLE CLUSTER.**
   Hourly median of the 12 Vestas `_ws` (2022-01→2024-12) and the 5 Unison `_ws` (2023-01→2024-12),
   resampled `1h`, `label='right'`, `closed='right'` on `kst_dtm`.
   **Validation gate before proceeding**: reproduce §1.2 (mean **7.086 m/s**, n = **26,304**) and
   §1.4 (corr **0.8141** against `ldaps__wind50max_speed__mean`, **0.8475** against `__max`).
   *Kill-switch: if those correlations do not reproduce, the `ws` columns share the power columns'
   time-scrambling and **B2, B3, and half of B4 are all dead**. Stop and report. Note that the fact
   they DO reproduce is itself the strongest available receipt that `ws` is not scrambled.*

**3. B2b — stability-conditioned learned extrapolation (8 h).**
   Feature block: `alpha_hat(t) = ln(w50/w10)/ln(5)` per cell, `blh` per cell, `117/blh`, bulk lapse
   rate (`ldaps 2m t` vs `gfs 850 t`), gust factor (`gfs gust / gfs wind10`), net-radiation sign
   (`NDNSW`/`NDNLW`), `XBLWS/YBLWS` as a third low level. Target `hub_ws_obs`.
   Keep the existing fixed power law and two-point log law as **fallback columns**, not anchors.
   Add the 3-level REWS separately for the V126 and U136 rotor discs — the one genuine
   `profile-shape(t) × rotor-geometry(g)` interaction the group dummy cannot absorb.
   *Highest expected value in the cluster. Fold-outside per Bodini & Optis's round-robin argument.*

**4. B3 — per-source multiplicative MOS (4 h incremental; 2 h already spent in step 2).**
   `g_ldaps` and `g_gfs` fitted **fold-outside** on `hub_ws_obs`, conditioned on stability and
   direction sector. Emit `*_mos` columns **alongside** the raw, never replacing.
   *Kill-switch: if B2b already captured it, B3 reads ~0.000 — a legitimate and informative ablation
   result, not a failure. Expect this outcome; do not spend more than the 4 h chasing it.*

**5. B4 Tier 1 — orography deficit × time function (4 h).**
   `dz_j = 1078 − ldaps__grid{j}__surface_0_h` per cell (the 16 values run 868.8–1001.2 m, so
   `dz` runs 77–209 m); JD2012's `D²h = 0.25(h_{i+1,j}+h_{i,j+1}+h_{i−1,j}+h_{i,j−1}) − h_{i,j}` on
   the 4×4 lattice (interior cells grid06, grid07, grid10, grid11); `sigma_sso ≈ 44 m`.
   Enter them **only** as `dz·alpha_hat(t)`, `dz/blh(t)`, `D²h·wind_speed(t)`.
   *Never as a static per-group constant — the 3-level group dummy spans it exactly, so a static
   terrain index adds precisely zero degrees of freedom.*

**6. REVERSE-ABLATION PASS (2 h).**
   Remove one stage at a time. **Design note for the ablation**: B5 was folded into step 1 by
   construction, so it has **no independent removal**. Either (i) report B1 and B5 as one joint
   attribution, or (ii) run one extra arm with a *shared* reduction rule to isolate B5. Do not
   report a spurious zero for B5.

**7. B4 Tier 2 — Sx / TPI / RIX from AW3D30 (16 h) — ONLY IF steps 1–6 land above +0.003.**
   One tile `N037E128` covers the ±20 km domain (site 37.16–37.30 N, 128.92–129.00 E). JAXA licence
   receipt (source / licence text / retrieval date) filed per AGENTS.md. **Predeclare** radii
   {500, 2000, 5000} m and 30° sector width **before** looking at any score. Per-**cell** Sx indexed
   by that cell's own forecast direction, giving 16 time series rather than 3 constants.
   *If steps 1–6 land below +0.003 the mechanism is not there and Tier 2 will not rescue it: its own
   source paper reports only 1.7–5.8% ridge RMSE gain, its bias gain is already absorbed by our group
   dummy, and the closest prior the repo has already run (`Hhat = h0·N/U`) scored **−0.000640**.*

---

## §6 WHAT I WOULD TELL YOU IF YOU ONLY READ ONE PARAGRAPH

The cluster's value is **not** in the terrain literature. It is in two things sitting in the repo
unused. First, **17 nacelle anemometers at exactly 117 m, at exactly the target site, covering
exactly the training window** — every SOTA method in §3 (Winstral, Optis, Wind-Topo, DEVINE) is a
*supervised* NWP→observed-wind map, and we have been attempting this cluster unsupervised. Build
`hub_ws_obs` first and gate on its validation; B2/B3/B4 then become ordinary regressions instead of
physics guesses. Second, **LDAPS and GFS need opposite spatial reductions** — `max` over the box for
the 1.5 km source that partially resolves the ridge, `idw`/mean for the 25 km source that resolves
only synoptics — and we currently apply one rule to both, which measurably costs GFS about 0.18 of
correlation. Two things to stop believing: that IDW is a reasonable site-transfer operator here (it
ranks 6th of 7 against measured hub wind, behind a plain `max()`), and that the HEFTCom2024
"stacking" headline will pay — the winner's own Table 1 shows stacked **28.5** vs plain concatenated
**28.6**, and he chose stacking for **source-dropout robustness we do not need**, because we submit
all of 2025 at once from a frozen archive.
