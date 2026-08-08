# S16 — SOTA / BENCHMARK LANE: complementarity by construction under a step-reward metric

**Lane type:** read-only external research (AGENTS.md bounded allowance).
**Repository writes:** this file + `research/lanes/S16_sota_decide.searchlog.json`. **Nothing else.**
Model fits **0**. Lockbox reads **0**. Git mutations **0**. Installs **0**. Uploads **0**. External data downloads **0**
(every PDF/HTML was parsed in memory and discarded). Scratch scripts were written to `/tmp` only and are not in the repo.
**Date:** 2026-08-09 (agent clock). **Queries:** 129. **Primary documents fetched and read in raw:** 14.

**Prior lanes read before starting:** `S15_sota_model.md`, `S14_oss_ecosystem.md`, `S14_foundation_insight.md`,
`S13_S7_modelling_deep.md`. Grep confirms this lane opens genuinely new ground: across every prior lane file,
`negative correlation learning` 0 hits, `DivBO` 0, `distributional boosting` 0, `Vincentization` 0,
`ambiguity decomposition` 2 (mentioned, never used), `centroid combiner` 0, `FFORMA` 0, `modal interval` 0,
`elicitab*` 0, `dynamic ensemble selection` 0.

---

## 0. Evidence grades

| Tag | Meaning |
|---|---|
| **A** | Primary document fetched into this session; the quoted string is a verbatim slice of extracted text. |
| **B** | Search snippet or abstract carried the number literally; full table unverified. |
| **M** | **Measured by me in this session**, zero fits, arithmetic on already-saved artifacts (`S7-N8_D_prob.npy`, `artifacts/backtests/metric-aligned-probe/*-policies.parquet`) through the project's own `.venv` and the project's own `official_total`. Reproducible; the champion number reproduces to 16 digits. |
| **I** | My derivation on numbers you supplied. |
| **X** | Closed by you or by a prior lane; listed only so you can see I checked. |

---

## 1. THE BRIEF'S PREMISE IS HALF WRONG, AND THE HALF THAT IS WRONG IS THE EXPENSIVE HALF  **[M]**

You asked how to build a member that is complementary by construction, because "they are all
0.92–0.99 error-correlated". I reproduced your champion exactly (`0.6361842493883538`, 19,785 rows,
11,486 scored) and then measured correlation **in the space the metric actually pays in**.

### 1.1 Continuous error correlation vs band-hit correlation

Members: `D` (your DART member under its fold-outside policy) and the three deployed stems
`M102_TOP100 @T0.5_G1.5`, `M113_LGBM_DART @T0.5_G0.5`, `M115_XGBOOST @T0.6_G0.35` (whose mean is `DEPAVG`).

**Continuous error correlation** (`corr(a_k − y)`, scored rows) — the number your project has been quoting:

|  | D | M102 | M113 | M115 |
|---|---:|---:|---:|---:|
| **D** | 1.0000 | 0.9031 | 0.9115 | 0.9137 |
| **M102** | 0.9031 | 1.0000 | 0.9856 | 0.9816 |
| **M113** | 0.9115 | 0.9856 | 1.0000 | 0.9870 |
| **M115** | 0.9137 | 0.9816 | 0.9870 | 1.0000 |

`corr(err_CHAMP, err_D) = 0.9562`.

**Band-hit indicator correlation** (`corr(1{|a_k − y| ≤ 0.06 cap})`, i.e. correlation of the event the
FICR half of the metric actually rewards) — **the number nobody in this project has measured**:

|  | D | M102 | M113 | M115 |
|---|---:|---:|---:|---:|
| **D** | 1.0000 | **0.4261** | **0.4245** | **0.4105** |
| **M102** | 0.4261 | 1.0000 | 0.8555 | 0.7974 |
| **M113** | 0.4245 | 0.8555 | 1.0000 | 0.8332 |
| **M115** | 0.4105 | 0.7974 | 0.8332 | 1.0000 |

**D and the deployed members are 0.91 correlated in error and 0.42 correlated in band-hit.**
The complementarity you have been trying to manufacture **already exists**. It is invisible to the
statistic you have been measuring because MAE-space correlation is dominated by the shared NWP signal,
while the band-hit event is a thresholded functional whose agreement is far weaker.

### 1.2 What that unmeasured diversity is worth  **[M]**

| quantity | value |
|---|---:|
| u=4 band-hit rate, `D` | 0.3424 |
| u=4 band-hit rate, `M102` / `M113` / `M115` | 0.3378 / 0.3351 / 0.3338 |
| u=4 band-hit rate, **CHAMPION (0.30 D + 0.70 DEPAVG)** | **0.3503** |
| u=4 band-hit rate, **at least one of the 4 members** | **0.4985** |
| u=4 band-hit rate, at least one of the 5 (4 members + champion) | 0.5064 |
| rows where some member hits and the champion misses | 0.1561 |
| rows where the champion hits and **no** member hits | **0.0079** |

Averaging **almost never manufactures a hit that no member had** (0.79% of scored rows) and
**throws away 15.6 pp of hits that members already had**.

| combiner | Total | Δ vs champion |
|---|---:|---:|
| CHAMPION `0.30·D + 0.70·DEPAVG` | 0.636184 | — |
| **ORACLE: per-row pick the closest of the 4 member actions** | **0.723333** | **+0.087149** |
| ORACLE: per-row pick the closest of 5 (incl. champion) | 0.726476 | +0.090292 |

The oracle over an operator you already own is **3.4× your entire remaining gap to 0.66**.
No new member is required to expose it. Every member you have ever built is already in the pool.

### 1.3 Where the headroom lives: action dispersion  **[M]**

`spread` := (max − min) of the four member actions, in capacity-factor units.

| `spread` band | scored n | champion u=4 hit | any-member u=4 hit | gap |
|---|---:|---:|---:|---:|
| [0.00, 0.03) | 2,589 | 0.471 | 0.512 | 0.041 |
| [0.03, 0.06) | 2,669 | 0.410 | 0.507 | 0.097 |
| [0.06, 0.10) | 2,720 | 0.290 | 0.468 | 0.178 |
| [0.10, 0.16) | 2,381 | 0.273 | 0.522 | **0.249** |
| [0.16, ∞)    | 1,127 | 0.240 | 0.469 | **0.229** |

Distribution of `|D − DEPAVG| / cap` on scored rows: mean 0.0632, median 0.0483,
`P(>0.06) = 0.4191`, `P(>0.12) = 0.1494`, `P(>0.16) = 0.0636`.
On 14.9% of scored rows the two blend inputs are more than **two full band-widths** apart, so their
arithmetic mean is guaranteed to be outside **both** members' u=4 bands.

This is a textbook **spread–skill relationship** (well established in NWP ensemble verification, e.g.
Whitaker & Loughe / Hopson lineage) reproduced on your own artifacts, and it is the signal a gate would use.

---

## 2. THE LOSS-MATCHED COMBINER: THEORY SAYS REPLACE THE MEAN, MEASUREMENT SAYS DO NOT  **[A] + [M]**

This is the single cleanest theoretical result I found, and it produced an honest negative.

### 2.1 The theory

- **SOTA** — Wood, Mu, Webb, Reeve, Luján & Brown, *A Unified Theory of Diversity in Ensemble Learning*,
  **JMLR 24 (2023)**. The generalised ambiguity decomposition defines the ensemble combiner **as a
  property of the loss**, not as a free choice.
- **EVIDENCE [A]** — <https://jmlr.org/papers/volume24/23-0041/23-0041.pdf>
  > "**Proposition 3 (Generalised Ambiguity Decomposition)** … ℓ(y,q) [ensemble loss] = (1/m)Σ ℓ(y,qᵢ)
  > [average loss] − (1/m)Σ ℓ(q,qᵢ) [ambiguity], where q := arg min_{z∈Y} (1/m)Σ ℓ(z,qᵢ) is the ensemble
  > combination. **We highlight that the ensemble combiner is defined as a property of the loss.** For
  > squared loss, this results in q = (1/m)Σqᵢ, the commonly used arithmetic mean combiner."
  > "**Definition 4 (Centroid Combiner rule)** … q := arg min_{z∈Y} (1/m) Σ ℓ(z, qᵢ)."
  > "For the 0/1 loss, the centroid is the mode of the random variable (Domingos, 2000) … this means q is a
  > **majority vote** of the individuals in the ensemble. Or, with the absolute loss, q is the **median**
  > prediction of the ensemble members."
  > "we should not be 'maximising diversity' as so many works aim to do—instead, we have a
  > **bias/variance/diversity trade-off to manage**."
- **Why it should bite here [I]:** your metric is `0.5·(1−NMAE) + 0.5·FICR`. The NMAE half is an
  **absolute** loss (centroid = median), the FICR half is a **3-level 0/1** loss (centroid = band vote).
  **Neither half has the arithmetic mean as its centroid.** The mean is the correct combiner for squared
  loss and for nothing else in your objective.
- **Elicitability check [A]** — Brehmer & Gneiting, *Scoring interval forecasts: equal-tailed, shortest and
  modal interval*, **Bernoulli 27(3), 2021**, <https://arxiv.org/pdf/2007.05709>:
  > "The modal interval also is elicitable, with a **sole consistent scoring function**, up to equivalence.
  > However, the shortest interval fails to be elicitable…"
  > "Under this convention **S(x,y) := −1(x−c ≤ y ≤ x+c)** … is a **strictly consistent scoring function**
  > for m_c on the class of distributions with Lebesgue densities, **whence m_c and MI_c are elicitable**."
  > "**Theorem 3.9.** … any scoring function that is strictly consistent for the l_k functional relative to
  > the class F is equivalent to k-zero-one-loss."
  Contrast **Heinrich (Biometrika 101(1), 2014)**, *The mode functional is not elicitable*
  (<https://academic.oup.com/biomet/article/101/1/245/2365689>):
  > "it is shown that the mode is not elicitable, or, in other words, that it is impossible to find a loss
  > or scoring function under which the mode is the Bayes predictor."
  **Consequence for you [I]:** your FICR reward `4·1(|e|≤0.06 cap) + 3·1(0.06 < |e| ≤ 0.08 cap)` is a positive
  combination of exactly two modal-interval scores (c = 0.06 and c = 0.08). Your target functional is
  therefore **elicitable** — you may train a member directly against it — whereas the naked mode is not.
  Every "learn the mode" idea is dead; every "learn the fixed-width modal interval midpoint" idea is alive.

### 2.2 The measurement — HONEST NEGATIVE  **[M]**

I implemented the centroid combiner exactly as Definition 4 requires, using the project's own per-row
utility parameterisation (`z* = argmax_z Σ_k w_k [ −|z−a_k| + γ·a_k·u(|z−a_k|)/(4·mean_gen_g) ]`,
weights `w = (0.30, 0.233, 0.233, 0.233)` matching the deployed champion, action grid = the project's `ACT`,
γ selected fold-outside from `{0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8}`).

| combiner | Total | 1−NMAE | FICR | u=4 hit | Δ vs champion |
|---|---:|---:|---:|---:|---:|
| **arithmetic mean (champion)** | **0.636184** | 0.861866 | 0.410503 | 0.3503 | — |
| centroid combiner, fold-outside γ (all folds picked γ=0) | 0.634440 | 0.858479 | 0.410401 | 0.3450 | **−0.001744** |
| median of the 4 actions | 0.633710 | 0.857507 | 0.409913 | 0.3427 | −0.002474 |
| snap the mean to the nearest member action | 0.634201 | 0.858412 | 0.409989 | 0.3446 | −0.001983 |
| mean of the two closest members | 0.632516 | — | — | — | −0.003668 |
| mean of the tightest triple | 0.633269 | — | — | — | −0.002915 |
| most-central member (min Σ|aᵢ−aⱼ|) | 0.634020 | — | — | — | −0.002164 |

**Verdict:** the loss-matched centroid combiner **loses**, and every unconditional non-linear combiner
loses. Reason, readable in the columns: the band vote reproduces the champion's FICR almost exactly
(0.410401 vs 0.410503, a wash) while forfeiting the mean's variance reduction on the NMAE half
(0.858479 vs 0.861866, −0.0034). The mean is not right because of decision theory; it is right because
member errors are 0.91–0.99 correlated in MAE-space, and averaging correlated errors is the one thing
that still pays there. **Do not build a loss-matched static combiner. This axis is closed by measurement.**

### 2.3 The corollary that stays open

The centroid combiner is *unconditional*. The oracle in §1.2 is *conditional*. The entire 0.087 lives in
conditioning, not in the shape of the combination rule.

---

## 3. THE AXIS THAT IS OPEN: INSTANCE-CONDITIONAL WEIGHTING (FFORMA), WITH A MEASURED BREAK-EVEN

### 3.1 SOTA

- **SOTA** — Montero-Manso, Athanasopoulos, Hyndman & Talagala, **FFORMA: Feature-based FORecast Model
  Averaging**, *IJF* 36(1) 2020 (M4 runner-up). A meta-learner maps per-instance features to a **softmax
  weight vector over the member pool**, trained with a **custom objective whose loss is the actual
  forecasting loss of each member on that instance**, not a cross-entropy on "which member won".
- **EVIDENCE [A]** — <https://robjhyndman.com/papers/fforma.pdf>
  > "FFORMA resulted in the **second most accurate** point forecasts and prediction intervals amongst all
  > competitors in the M4 competition."
  > "w(fₙ)ₘ = exp(p(fₙ)ₘ)/Σ … Lₙₘ is the contribution to the OWA error measure of method m for the series n
  > … **Gₙₘ = ∂L̄ₙ/∂p(fₙ)ₘ = wₙₘ(Lₙₘ − L̄ₙ)** … The functions G and Ĥ were passed to xgboost."
  > "The average OWA error of the **model selection** approach was **10% larger** than FFORMA … On the other
  > hand, FFORMA deviates significantly from **simple averaging**. The latter produces a **14% increase in
  > error** for the same pool of methods."
  > "roughly **40% of the time series receive a weight profile similar to equal weights** (a simple average)
  > while for the remaining **60% … one of the methods in the pool is clearly dominant**."
  > "**One advantage of our approach is that its form is independent of the forecasting loss measure.**
  > Forecast errors enter the model as additional pre-calculated values. **This allows FFORMA to adapt to
  > arbitrary loss functions**."
- **BENCHMARK** — M4 (100,000 series), 9-method pool, xgboost meta-learner. FFORMA beats *both* endpoints:
  simple averaging by 14% and hard selection by 10%. Related: **FFORMS** (Talagala, Hyndman & Athanasopoulos,
  <https://robjhyndman.com/papers/fforms.pdf>) is the hard-classification variant, and it is the weaker one.
  **DivBO** (Shen et al., NeurIPS 2022, <https://proceedings.neurips.cc/paper_files/paper/2022/file/13b2f88be223cd2b4d6be67b56e02fa8-Paper-Conference.pdf>)
  is the AutoML analogue on the *search* side: "Empirical results on 15 public datasets show that DivBO
  achieves the **best average ranks (1.82 and 1.73)** on both validation and test errors among 10 compared
  methods"; against the strongest baseline RB-ES it is better on 8, tied on 4, worse on 3 of 15 datasets.

### 3.2 The break-even, measured on YOUR data  **[M]**

I simulated a gate of known quality on the high-disagreement subset (trigger `spread ≥ 0.08 cap`,
5,213 of 19,785 rows = 26%; below trigger the champion action is kept unchanged). A gate of "effective
top-1 accuracy p" picks the oracle-best member with probability p and a uniformly random member otherwise
(so p = 0.250 is a blind gate). 20 replicates, sd ≈ 0.0010.

| effective gate top-1 | Total | Δ vs champion 0.636184 |
|---:|---:|---:|
| 0.250 (blind) | 0.631497 | −0.004687 |
| **0.325** | 0.637569 | **+0.001385  ← break-even ≈ 0.32** |
| 0.400 | 0.644893 | +0.008709 |
| 0.475 | 0.650995 | +0.014811 |
| 0.550 | 0.657744 | +0.021560 |
| 0.625 | 0.664590 | +0.028406 |
| 0.700 | 0.670628 | +0.034444 |
| 0.850 | 0.684069 | +0.047885 |
| 1.000 (oracle) | 0.697026 | +0.060842 |

Slope ≈ **+0.00088 Total per +1 pp of gate top-1 accuracy.** Your seed floor is 0.001635, so the gate must
reach ≈ **0.34** to be *measurable at all*, and ≈ **0.58** to reach Total 0.66 from the champion alone.

Trigger sensitivity (oracle upper bound at each trigger): `spread≥0.00 → +0.0871`, `≥0.06 → +0.0717`,
`≥0.08 → +0.0608`, `≥0.10 → +0.0465`, `≥0.12 → +0.0350`, `≥0.16 → +0.0163`.
A **soft** gate `p = champ + w·(gated − champ)` on all rows scales roughly linearly in `w`
(oracle: `w=0.1 → +0.0092`, `0.3 → +0.0287`, `0.5 → +0.0476`, `1.0 → +0.0871`), so `w` is a usable risk dial:
at effective top-1 0.400, `w=0.3 → +0.0045`, `w=0.5 → +0.0070`, `w=1.0 → +0.0122`.

### 3.3 THE TRAP IN THAT TABLE — read this before you build  **[M]**

The simulation assumes **gate errors are conditionally random**. They will not be. Measured label
distribution of the oracle-best member on `spread ≥ 0.08` scored rows: **D 0.510, M102 0.302, M113 0.100,
M115 0.088**. So the constant rule "always take D" already achieves **51% top-1** — above every break-even in
the table — and yet:

| rule | Total | Δ |
|---|---:|---:|
| always D on the trigger rows | 0.630117 | **−0.006068** |
| best hard fold-outside selection over (group × champion-action bins), 1/3/5/10 bins | 0.625651 / 0.625651 / 0.625651 / 0.625670 | **−0.0105** |

A 51%-accurate constant gate **loses 0.006**. The reason is in the error magnitudes: on trigger rows the
median |error| of the best member is 0.0612 cap and of the second-best is 0.1169 cap — nearly **2×**. When
the gate is wrong it is expensively wrong, and a constant rule is wrong on a systematically adverse subset.
**Top-1 accuracy is necessary and not sufficient; the gate must also be right about the rows where being
wrong is expensive.** This is exactly why FFORMA's authors reported that *averaging* beat *selection* by 10%
even with the same features and the same learner — and why the FFORMA construction (soft weights, loss-valued
objective, shrinkage toward equal weights) is the one to copy, not the classifier.

### 3.4 MIGRATION onto your data

- **Inputs, all already on disk, no new fit needed for the meta-data:**
  `research/nodes/S7-N8_D_prob.npy` + `S7-N8_D_keys.parquet` → D's action via the existing
  `loop_lib.utility_frames` / `fo_policy`; and
  `artifacts/backtests/metric-aligned-probe/{M102_TOP100,M113_LGBM_DART,M115_XGBOOST}-{fold}-policies.parquet`
  columns `T0.5_G1.5`, `T0.5_G0.5`, `T0.6_G0.35` respectively (this is exactly `loop_lib.DEP`, so the
  AGENTS.md `prediction_kwh` trap does not apply — you are reading named policy columns, not `prediction_kwh`).
- **Meta-label per row `i`, member `k`:** `L_ik = −[ −|a_ik − y_i|/cap + γ·(y_i/(4·mean_gen_g))·u(|a_ik − y_i|) ]`,
  i.e. the negative of that member's own contribution to `official_total`. This is the FFORMA `L_nm`,
  and it makes the meta-learner metric-matched rather than accuracy-matched. `γ` is the project's existing
  FICR/NMAE trade parameter; take γ = 1 (metric-matched) and do not tune it (the gamma frontier is flat — X).
- **Meta-features (all already derivable, zero new data):** the *disagreement geometry* is free and is the
  literature's own signal — `spread`, pairwise `|a_j − a_k|`, rank of each member's action, distance of each
  action from the member mean and from the member median, and `mean_gen_g`; plus the existing surface's
  NWP features via `harness.surface(('G2','DROP:grid__'))` and `pc_hat` from the physics teacher. **Do not add
  lead-time features** (X, closed) and **nothing requiring observed generation at prediction time** (X, closed).
- **Learner:** `lightgbm.LGBMRegressor` / `xgboost` with a **custom objective** implementing FFORMA's
  `G_nm = w_nm (L_nm − L̄_n)` and their clamped-Hessian upper bound. `xgboost 3.3` and `lightgbm 4.7` both
  support this natively. **No install. No downgrade.**
- **Shrinkage (mandatory, this is where your fold-outside history says it will die):** final weights
  `w = (1−λ)·w_champion + λ·w_meta`, with `λ ∈ {0, 0.15, 0.3, 0.5, 0.7, 1.0}` chosen **fold-outside**.
  One extra dof on top of the meta-learner. Your own record — 21-dof blend `0.646821 → 0.643888`
  fold-outside, 3-dof per-group weights `0.640253 → 0.635453` — says an unshrunk per-row weighter is the
  most likely thing in this document to collapse.
- **Trigger:** apply the meta-weights only where `spread ≥ τ`, `τ` picked fold-outside from
  `{0.06, 0.08, 0.10, 0.12}`. This caps the blast radius to 26–35% of rows.
- **What is missing:** nothing. Every input exists. This is why it ranks first.

### 3.5 RISK, citing your numbers

1. Your fold-outside gate over `(group × champion-action bins)` **already failed at −0.0105** [M]. That gate
   had **zero features**; the whole bet is that NWP + disagreement geometry carries information that
   coarse binning does not. If it does not, this node returns −0.005 to −0.010, not 0.
2. Seed floor 0.001635 ⇒ break-even ≈ gate top-1 0.34, only 9 pp above blind.
3. dev-2023 **contains no winter** and the graded period is all of 2025. A gate is a *selector*, and
   selectors extrapolate worse than averages. The soft-`w` dial and the shrinkage `λ` are the mitigations.
4. `g3` has no 2022 labels and is your worst group (`NMAE 0.1514`, `FICR 0.3528`); expect the gate to be
   thinnest exactly there.

### 3.6 EXPECT and COST

**EXPECT +0.004 (central), 10th–90th percentile −0.006 to +0.015.** **COST ≈ 10 h**, of which ~2 h is
meta-data assembly (fit-free), ~3 h the custom objective, ~3 h the fold-outside λ/τ protocol, ~2 h arbitration.
**Install: none.**

---

## 4. STAGE D2 — A MEMBER THAT IS SHARP AND DECORRELATED BY CONSTRUCTION

Only **one** family in the literature is measured to be *simultaneously* sharper than its competitors and
built on an estimation principle unrelated to GBDT-on-features. That family is order-restricted
nonparametric distributional regression: **IDR / EasyUQ**.

### D2-a ★ RANK 1 — IDR / EasyUQ as a **second density arm**, not as a replacement

- **SOTA** — Henzi, Ziegel & Gneiting, *Isotonic Distributional Regression*, **JRSS-B 83(5) 2021**;
  Walz, Henzi, Ziegel & Gneiting, *Easy Uncertainty Quantification (EasyUQ)*, **SIAM Review 66(1) 2024**.
  IDR is the unique CRPS-optimal conditional distribution subject to stochastic monotonicity in a
  **partial order** on the covariates. Zero tuning parameters. EasyUQ = IDR applied to a single-valued model
  output; Smooth EasyUQ adds kernel smoothing with one-fit grid search.
- **EVIDENCE [A]** — <https://arxiv.org/pdf/1909.03725>
  > "IDR … learns conditional distributions that are calibrated, and simultaneously optimal relative to
  > comprehensive classes of relevant loss functions, subject to isotonicity constraints in terms of a
  > **partial order on the covariate space**."
  > "it does not involve **any parameter tuning nor implementation choices**, except for the selection of a
  > partial order on the covariate space."
  > "**Componentwise order** … x ≼ x′ ⟺ xᵢ ≤ x′ᵢ for i = 1,…,d."
- **EVIDENCE, and this is the sharpness number [A]** — Schulz & Lerch, *Machine learning methods for
  postprocessing ensemble forecasts of wind gusts*, **MWR 150(1) 2022**, <https://arxiv.org/pdf/2106.09512>,
  Table 3 (COSMO-DE-EPS, 175 stations, 6 years, all lead times):

  | Method | CRPS | MAE | **PI length** | Coverage (nominal 90.48%) |
  |---|---:|---:|---:|---:|
  | EMOS | 0.95 | 1.32 | 5.94 | 92.51% |
  | MBM | 0.97 | 1.34 | 6.10 | 90.81% |
  | **IDR** | 0.98 | 1.36 | **4.72** | **84.04%** |
  | EMOS-GB | 0.88 | 1.23 | 5.24 | 91.04% |
  | QRF | 0.87 | 1.22 | 5.41 | 91.38% |
  | DRN | 0.84 | 1.18 | 5.05 | 91.49% |
  | BQN | 0.84 | 1.18 | 4.94 | 90.42% |
  | HEN | 0.86 | 1.21 | 5.07 | 90.23% |

  > "Among the well-calibrated postprocessing methods, the NN-based methods yield the **sharpest** forecast
  > distributions … we conclude that the **gain in predictive performance is mainly based on an increase in
  > sharpness**."

  **IDR produces the narrowest interval in the whole table (4.72, 21% narrower than EMOS, 7% narrower than
  BQN) while having the *worst* CRPS of the postprocessing methods (0.98).** That is precisely the profile
  the brief asks for: **sharp but not the most accurate**, i.e. a member that will *not* be a near-duplicate
  of your accuracy-maximising D.
- **BENCHMARK** — Schulz & Lerch MWR 2022 (wind gusts, ensemble postprocessing); Henzi et al. JRSS-B 2021
  (ECMWF 52-member → 24 h precipitation, 4 airports, 6–10 years); Walz et al. SIAM Review 2024 (WeatherBench
  upper-air temperature, 2,990,080 forecast cases, plus UCI-style ML benchmarks; "find EasyUQ to be
  **competitive with conformal prediction, as well as more elaborate input-based approaches**").
- **MIGRATION.** Two variants, both zero-install:
  - **(i) EasyUQ on your own point pipeline.** Covariate = the S15 composed point prediction (`pc_hat` or the
    composed member's action, in `research/nodes/S15-N7_preds.npy` / `S15-N7_keys.parquet`), response = `cf`.
    One-dimensional IDR = **for each of your 26 class thresholds `z`, run `sklearn.isotonic.IsotonicRegression`
    of `1{cf ≤ z}` on the covariate**, then enforce monotone-in-`z` by a row-wise cumulative max. ~40 lines,
    `sklearn 1.9.0` only, no new dependency, no downgrade. Output slots straight into `loop_lib.utility_frames`
    as a 26-column probability matrix, so the entire existing policy/blend machinery applies unchanged.
  - **(ii) Two-covariate IDR as a *nonlinear combiner*.** Componentwise partial order on `(a_D, a_DEPAVG)`.
    This is the sharpness-preserving alternative to `0.30·D + 0.70·DEPAVG` and it is *conditional* (§2.3),
    unlike the centroid combiner that failed. Cost: the 2-D PAVA is the expensive part; the paper's own
    remedy is **subsample aggregation** ("can be combined with subsample aggregation, with the benefits of
    smoother regression functions and gains in computational efficiency").
  - Fit per group (3 fits) — this is the natural place for **Mondrian/group-conditional** treatment (§5).
- **RISK.** IDR's 84% coverage at nominal 90% (Table 3) says it is *underdispersed*. Under your metric
  underdispersion is a feature for FICR and a liability for NMAE. Your law — "any operation that raises point
  accuracy by smoothing the predictive distribution loses more FICR than it gains NMAE" — runs the other way
  here for the first time: IDR sharpens rather than smooths. That is why it is worth a probe. Counter-risk:
  IDR conditions on a **1-D** covariate, so it discards all the feature information D uses; expect its solo
  Total to be poor (D solo is 0.6257) and its value to be entirely in the blend.
- **EXPECT.** As a blend arm: **+0.003 (central), −0.001 to +0.008.** As a solo member: negative.
- **COST.** Variant (i): **4 h**, no install. Variant (ii): **+4 h**. Total 8 h.

### D2-b RANK 2 — keep the 26-class softmax, do not swap the head

- **EVIDENCE [A]** — Schulz & Lerch Table 3 above: HEN (the histogram head, i.e. **your** architecture)
  scores CRPS 0.86 vs DRN/BQN 0.84 and PI length 5.07 vs 4.94/5.05. A 2% CRPS gap. Your D2 head is
  **not** the bottleneck, and the two heads that beat it are neural (torch — **disqualified**).
- Corroborating [A/B]: Imani & White, *Improving Regression Performance with Distributional Losses*, ICML 2018
  (<http://proceedings.mlr.press/v80/imani18a/imani18a.pdf>) and the follow-up JMLR 27 (2026)
  *Investigating the Histogram Loss in Regression* (<https://www.jmlr.org/papers/volume27/24-0260/24-0260.pdf>)
  both find the benefit of a histogram target is an **optimisation/representation** effect, not a density
  effect — the JMLR paper explicitly characterises the *bias of the mean* of the HL-Gauss minimiser.
  Translation for you: HL-family heads are engineered to make the **mean** better. You do not want the mean.
- **Do not build.** **EXPECT 0.000.**

### D2-c RANK 3 (do NOT build) — distributional boosting arms

NGBoost (Duan et al., ICML 2020) and XGBoostLSS/LightGBMLSS (März) are parametric-density boosters. They fit
a *smooth parametric* conditional density, which is the exact operation your nine replications say costs more
FICR than it gains NMAE. NGBoost's own headline is "**competitive performance in terms of NLL, especially on
smaller datasets**" — NLL, not sharpness, not a step reward. **EXPECT −0.002 to +0.001. Do not build.**

---

## 5. STAGE D3 — CALIBRATION THAT DOES NOT COST SHARPNESS

The brief asks for calibration that *improves* sharpness. The literature's answer is narrow and specific.

### D3-a ★ RANK 1 — Mondrian (group-conditional) conformal predictive distributions

- **SOTA** — Boström, Johansson & Löfström, *Mondrian Conformal Predictive Distributions*, **COPA 2021 (PMLR 152)**.
  Stratify the calibration residuals by category (here: `group_id`, and optionally an action-level bin), then
  build the conformal predictive distribution within each stratum.
- **EVIDENCE [A]** — <https://proceedings.mlr.press/v152/bostrom21a/bostrom21a.pdf>
  > "the two Mondrian approaches **clearly outperform** both standard and normalized conformal predictive
  > systems. This is confirmed by a Friedman test … followed by a Nemenyi post-hoc test, showing the
  > differences to be **significant at α = 0.05** … This is of course a **very strong result**."
  > "the use of Mondrian conformal predictive distributions results in **as tight prediction intervals** as
  > produced by normalized conformal regressors, while **improving upon the point predictions** of the
  > underlying regression forest."
  Table 2 (CRPS, 33 regression datasets, 10-fold CV, random-forest base): `bank8fm 0.030 → 0.020` (−33%),
  `puma8nm 0.052 → 0.038` (−27%), `friedm 0.042 → 0.031` (−26%), `kin8nm 0.060 → 0.050` (−17%),
  `bank8nm 0.025 → 0.019` (−24%). CPS→MCPSn improves on 30 of 33.
- **BENCHMARK** — 33 UCI-style regression datasets, ~0.2k–20k rows — **the same scale as your 19,795 rows**.
- **MIGRATION.** `crepes 0.9.1` depends on `numpy`, `pandas`, `scipy` **only** — verified clean against your
  no-downgrade rule in S14/S15. But you do not need it: a Mondrian CPS is `np.searchsorted` over
  group-stratified sorted residuals, ~30 lines. Strata: `group_id` (3) × action decile, calibration residuals
  taken **fold-outside** from the two non-held folds. Output is a CDF per row → resample onto your 26 classes
  → `utility_frames` unchanged.
- **RISK.** This is a *calibration* operation on the tails, and the champion's advantage is concentrated in the
  band. 30.6% of your rows are unscored (`actual < 0.1 cap`), and the residual pool must be restricted to
  scored rows or the strata are contaminated. Your S15 lane already ranked Mondrian as D4-b RANK 2 and it has
  not been built — this lane raises it because the measured evidence (30/33 datasets, α=0.05) is stronger than
  anything else in the D3 stage.
- **EXPECT +0.002, range 0.000 to +0.005. COST 5 h. Install: none required.**

### D3-b RANK 2 — Venn–Abers on the band-hit probability, with an explicit caution

- **SOTA** — Vovk, Petej & Fedorova, Venn–Abers predictors (NeurIPS 2015); the 2026 large-scale replication.
- **EVIDENCE [A]** — *Classifier Calibration at Scale: An Empirical Study of Model-Agnostic Post-Hoc
  Calibration*, <https://arxiv.org/html/2601.19944v1> (TabArena-v0.1 binary tasks, 5-fold CV, five calibrators):
  > "Across tasks and architectures, **Venn–Abers predictors achieve the largest average reductions in
  > log-loss**, followed closely by Beta calibration, while Platt scaling exhibits weaker and less consistent
  > effects. … Venn–Abers displays **fewer instances of extreme degradation** and slightly more instances of
  > extreme improvement."
  > "Importantly, we find that commonly used calibration procedures, **most notably Platt scaling and
  > isotonic regression, can systematically degrade proper scoring performance for strong modern tabular
  > models**."
  > "Overall classification performance is often preserved, but calibration effects vary substantially …
  > **no method dominates uniformly**."
- **BENCHMARK** — TabArena-v0.1, a modern tabular suite, exactly your data type.
- **MIGRATION.** Only useful if you first have a *scalar* band-hit probability to calibrate — i.e. it is
  downstream of the §3 gate, not of the 26-class head. Calibrate `P(member k hits the u=4 band | x)` per member
  with `venn-abers 1.5.3` (deps: `numpy`, `scikit-learn`, `pandas` — clean) and feed the calibrated
  probabilities into the FFORMA softmax as a prior. Cross-conformal / inductive split required, fold-outside.
- **RISK.** The quoted finding that **isotonic regression can systematically degrade proper scoring for strong
  tabular models** applies directly to any naive isotonic post-hoc step you might be tempted to add. Venn–Abers
  is the safe member of that family, and it is safe because it is *less* aggressive, i.e. it will move less.
- **EXPECT +0.001, range −0.001 to +0.003. COST 3 h + 1 clean install. Build only after §3.**

### D3-c RANK 3 — CORP as a **measurement**, not a treatment (zero fits, 2 h)

- **EVIDENCE [B/A]** — Dimitriadis, Gneiting & Jordan, *Stable reliability diagrams for probabilistic
  classifiers*, **PNAS 118(8) 2021**, <https://www.pnas.org/doi/10.1073/pnas.2016191118>: the CORP approach
  uses PAVA to produce "provably statistically **consistent**, **optimally binned**, and **reproducible**"
  reliability diagrams with the additive decomposition `S = MCB − DSC + UNC`.
- **WHY YOU WANT IT.** Applied to the event `1{|a − y| ≤ 0.06 cap}` for the champion and for each member, the
  **DSC** term is the *discrimination* the current representation has about band-hits — i.e. it upper-bounds
  what §3's gate can possibly recover, and it does so **before you spend 10 h on the gate**. `MCB` tells you
  whether the residual loss is calibration (cheap to fix) or discrimination (needs new information).
  Pure `sklearn.isotonic` — no install.
- **EXPECT 0.000 directly.** Its value is that it can **delete §3** for 2 h instead of 10.

### D3-d (do NOT build) — temperature/Dirichlet rescaling of the 26-class softmax

Your `utility_frames` already sweeps `T ∈ {0.6 … 4.0}` fold-outside; a learned global temperature is the same
one-parameter family you already optimise. **EXPECT 0.000.** X.

---

## 6. STAGE E1 — ACTION SELECTION

Your action selector (`argmax_a  E[−|a−Y|] + γ·E[a·u]/(4·mean_gen)`) is already the Bayes act for the metric,
the gamma frontier is flat (X), and §2.1 now supplies the *theoretical* endorsement you were missing: the
target functional is the **modal-interval midpoint**, which is **elicitable with a unique consistent score**
(Brehmer & Gneiting Theorem 3.9), while the naked mode is **not elicitable** (Heinrich 2014).

The one thing this changes: because the functional is elicitable, you may **train against it directly**.

### E1-a RANK 1 (cheap, novel, and it is the only genuinely new *member* construction in this lane)

- Train a LightGBM member whose objective is the **smoothed modal-interval score**
  `−σ((0.06 − |a − y|)/h)` with `h ≈ 0.01`, i.e. a differentiable surrogate for `−1(|a−y| ≤ 0.06 cap)`,
  supplied as a `lightgbm` custom objective (gradient and Hessian both closed form). Add the metric's
  `a·y/(4·Σy)` weight so the surrogate is metric-matched.
- **Why it should decorrelate [I]:** it is the *first* member in this project whose loss has a **bounded
  influence function**. Every existing member minimises L1 or cross-entropy, both of which have unbounded or
  near-unbounded influence and so are driven by the same large-error rows — which is exactly why they are
  0.90–0.99 error-correlated. A bounded, band-local loss ignores rows it cannot save and concentrates capacity
  on rows near the band edge, a genuinely different allocation of capacity.
- **Evidence that the construction is legitimate [A]:** Brehmer & Gneiting (2021) Theorem 3.9 + eq. (11);
  Taggart, *Point forecasting and forecast evaluation with generalized Huber loss*, <https://arxiv.org/abs/2108.12426>:
  > "Each Huber functional is **elicitable**, generating the precise set of minimizers of an expected score …
  > Each elementary score can be interpreted as the relative economic loss of using a particular forecast for a
  > class of investment decisions where **profits and losses are capped**."
  A capped-loss decision problem is exactly a step reward. Taggart's generalised Huber loss is the smooth,
  elicitable bridge between your L1 half and your capped half, and it is a *strictly better-founded* surrogate
  than an ad-hoc sigmoid: use asymmetric Huber with the transition at `0.06 cap`.
- **RISK.** Non-convex surrogate; LightGBM will need `min_child_samples` raised and multiple seeds. Your
  S16-N1 `FOCUS` arm (reweighting toward champion misses) is the crude version of this idea; the Huber/modal
  formulation is the principled one and does not reweight the data, so it does not inherit the
  hard-example-mining pathology.
- **EXPECT +0.002 in blend (central), −0.002 to +0.006. COST 6 h. Install: none.**
  Judge it on **band-hit correlation with D**, not on solo Total. Target < 0.35.

### E1-b (do NOT build) — anything that moves the action toward the conditional mean. X, nine replications.

---

## 7. STAGE E2 — COMBINATION. Ranked verdicts

| construction | status | measured / cited |
|---|---|---|
| static linear weights over member actions | **X closed** | your own record; per-group 3-dof `0.640253 → 0.635453` fold-outside |
| **loss-matched centroid combiner (band vote / median)** | **CLOSED BY THIS LANE** | **−0.001744 [M]**, §2.2 |
| hard per-row selection among members | **CLOSED BY THIS LANE** | **−0.0105 [M]** (fold-outside, group×bin), and always-D **−0.0061 [M]** |
| **FFORMA-style instance-conditional soft weights** | **OPEN, RANK 1** | break-even at gate top-1 0.32, +0.00088/pp [M]; FFORMA beats averaging by 14% and selection by 10% on M4 [A] |
| **two-covariate IDR as a nonlinear combiner** | **OPEN, RANK 2** | §4 D2-a(ii) |
| probability-averaging the member densities | **avoid** | see below |
| quantile averaging (Vincentization) of member densities | **neutral-to-good** | see below |

### 7.1 Never linearly pool the densities; Vincentize them  **[A]**

- **EVIDENCE [A]** — Schulz & Lerch (2022), §3.3, on aggregating ten HEN model runs:
  > "Instead of simply averaging the bin probabilities across the ensemble of model runs, which is **known to
  > yield underconfident forecasts that lack sharpness** (Ranjan and Gneiting, 2010; Gneiting and Ranjan, 2013),
  > we take a **Vincentization** approach."
- **EVIDENCE [A]** — Tibshirani et al., *Flexible Model Aggregation for Quantile Regression*,
  <https://www.stat.cmu.edu/~ryantibs/papers/quantagg.pdf>, Proposition 4 (Lichtendahl, Grushka-Cockayne &
  Winkler, *Management Science* 59(7) 2013):
  > "(i) A probability and quantile average always have **equal means** … (ii) A **quantile average is always
  > sharper than a probability average**: mₖ(F̄) ≤ mₖ(F) for any even k ≥ 2."
  With the honest caveat the same authors attach:
  > "Note that sharpness is only a desirable property if it does not come at the expense of calibration."
  And the mechanism: "the probability density f̄ at the level-u quantile is a (weighted) **harmonic mean** of
  the densities fⱼ at their respective level-u quantiles."
- **DIRECT RELEVANCE TO YOUR BINDING BARRIER [I].** Your composed member's failure signature — band mass 0.567
  vs D's 0.759, entropy 1.941 vs 1.404 — is the *exact* signature the linear-pool literature describes.
  If anywhere in the composed pipeline you average class-probability vectors (across seeds, across sub-models,
  across sources, or across the 26-class outputs of a bagged set), **that step is the flattener**, and replacing
  it with Vincentization (average the quantile function, i.e. average the inverse CDFs, then re-discretise) is
  a **1–2 h, zero-risk, zero-install** change that recovers sharpness *without touching accuracy*.
  **This is the highest EXPECT/COST item in the entire document and it should be checked first.**
  Check: `grep` the composed pipeline for any `P.mean(axis=0)` / `np.mean([...prob...])` over probability
  matrices. If none exists, this item costs 20 minutes and returns 0.
- **EXPECT.** If a probability average exists in the composed pipeline: **+0.002 to +0.004**, entirely from
  restoring band mass. If not: 0.000. **COST 0.3 h to check, 2 h to fix.**

### 7.2 Why "average the actions" survived here and "average the densities" should not

Averaging *actions* is Vincentization of Dirac measures (a quantile average), which the proposition above says
is the **sharp** operation. Averaging *probabilities* is the flat one. Your deployed blend accidentally does the
right one; your composed member may be doing the wrong one internally. That asymmetry explains both §2.2's
negative and §1.2's oracle in one sentence.

---

## 8. STAGE E3 — CONSTRAINTS AND POST-PROCESSING

Small, cheap, and mostly already done. Honest negatives included.

- **Capacity clip / non-negativity.** Already in `loop_lib` via `SC = {1: 0.985, 2: 0.989, 3: 1.005}` and the
  `0.10` floor mask. Literature support for the general principle [B]: *Non-Negative Forecast Reconciliation*,
  **Forecasting 7(4) 2025**, <https://www.mdpi.com/2571-9394/7/4/64> — "non-negativity constraints
  **consistently improve accuracy** compared to base forecasts. Overall, **set-negative-to-zero achieves
  near-optimal**" results. Note the honest reading: the simplest clip is as good as the optimised projection.
  **EXPECT 0.000 (already done). Do not re-derive.**
- **Hierarchical reconciliation across g1/g2/g3.** [X] Do **not** build. Your three groups have **no coherent
  aggregate** in the metric — NMAE and FICR are computed per group and then averaged, so there is no total-level
  series to reconcile to, and MinT's variance argument does not apply. Reconciliation would smooth toward a
  cross-group mean, which is the operation your nine replications forbid.
- **Monotone constraints (`monotone_constraints` in LightGBM) on hub wind speed → power.** Physically motivated,
  free, and it is the one constraint that could *sharpen* by removing non-monotone wiggle. But your S15 SCADA-only
  power-curve stage (+0.001912) already imposes monotonicity structurally. **EXPECT 0.000–0.001, COST 1 h.**
  Low priority; include only as a rider on another fit.
- **Honest negative to record.** Every unconditional post-processing operator I measured in §2.2 lost. There is
  no free lunch left in E3 on this metric; the remaining money is all in E2/§3.

---

## 9. WHAT THE DIVERSITY-BY-CONSTRUCTION LITERATURE ACTUALLY BUYS — AND WHY I RANK IT LAST

The brief asked specifically for measured numbers on NCL, diversity-regularised ensembles, ambiguity as a
training objective, DivBO, and residual-of-ensemble boosting. Here is the honest answer.

| method | what it buys, measured | what it costs | verdict for you |
|---|---|---|---|
| **Negative Correlation Learning** (Liu & Yao 1999; Brown et al.; Buschjäger et al., *Generalized NCL*, arXiv 2011.02952) | diversity via an explicit `λ·Σ(qᵢ−q̄)²` penalty, deep ensembles | penalty λ trades *individual accuracy* for diversity; the decomposition is **exact only for squared loss** — Wood et al.: "Buschjäger et al. (2020) used a Taylor approximation on twice-differentiable losses, showing an **exact decomposition only when higher derivatives are zero, e.g., squared loss, but not cross-entropy**" **[A]** | **Do not build.** Your loss is neither squared nor twice-differentiable. NCL's guarantee evaporates exactly where you need it. |
| **Diversity-regularised ensembles generally** | Wood et al. JMLR 2023 is the authoritative synthesis and it is a **warning**: "we should **not** be 'maximising diversity' as so many works aim to do—instead, we have a **bias/variance/diversity trade-off to manage**" **[A]**; and "Several measures (including also non-pairwise measures) were explored, **with no single measure proving more successful than any other**" **[A]** | — | **Do not build.** 20 years of heuristic diversity measures with no measure beating any other. |
| **DivBO** (NeurIPS 2022) | "best average ranks (**1.82 and 1.73**) on both validation and test errors among 10 compared methods"; vs the strongest baseline RB-ES: better on 8, same on 4, worse on 3 of 15 **[A]** | a diversity surrogate + a temporary pool + a weighted acquisition; it is a **search-side** method requiring hundreds of CASH iterations | **Do not build.** The gain is a rank improvement over AutoML baselines, not a Δ-metric; it needs a large configuration budget; and your MCS over 23 members already returns **eight tied models including plain DEPAVG**, i.e. your pool is not short of configurations, it is short of *conditioning*. |
| **Boosting/stacking on the residual of the current ensemble** | this is exactly your S16-N1 `STACK` arm | leakage discipline, and the residual target inherits the ensemble's own noise | **Neutral.** Keep it as built; it is not the axis. |
| **Hard-example reweighting toward champion misses** (your `FOCUS` arm) | AdaBoost.R2-style reweighting is the classical form; the classical finding is that it is **noise-amplifying** | | **Expect it to underperform `STACK`.** Rows where the champion misses are disproportionately rows where the *label* is unpredictable (see §1.3: at spread ≥ 0.16 even the union of all four members only hits 0.469). Reweighting toward them buys variance. |

**Bottom line for question 1 of the brief:** the diversity-by-construction literature buys, at best, a
rank improvement on AutoML benchmarks, and its central theoretical result (Wood et al.) says the whole
programme is misconceived — diversity is a *dimension of model fit to be traded*, not a quantity to maximise,
and the only clean guarantees hold for squared loss with an arithmetic-mean combiner, i.e. neither half of
your metric. **I recommend building none of it.** The measured evidence in §1 says your ensemble is not
short of diversity; it is short of *knowing when to use it*.

---

## 10. RANKING BY EXPECT / COST

| # | node | stage | EXPECT (ΔTotal) | COST | EXPECT/COST | install |
|---|---|---|---:|---:|---:|---|
| **1** | **Vincentization audit**: find any probability-vector average inside the composed pipeline and replace with quantile averaging | E2 | 0.000 to +0.004 (0.002 central *if* such an average exists) | **0.3 h audit + 2 h fix** | **~0.9e-3/h** | none |
| **2** | **CORP DSC on the band-hit event** — measures the gate ceiling before you build the gate | D3 | 0.000 direct; **deletes or licenses node 3** | **2 h** | gating value | none |
| **3** | **FFORMA instance-conditional soft weights**, trigger `spread ≥ τ`, shrinkage `λ`, both fold-outside | E2 | **+0.004** (−0.006 … +0.015) | **10 h** | 0.4e-3/h | none |
| 4 | **EasyUQ / 1-D IDR density arm** on the composed point pipeline | D2 | +0.003 (−0.001 … +0.008) | 4 h | 0.75e-3/h | none |
| 5 | **Mondrian group-conditional conformal predictive distribution** | D3 | +0.002 (0.000 … +0.005) | 5 h | 0.4e-3/h | none (or `crepes`, clean) |
| 6 | **Modal-interval / asymmetric-Huber member** (bounded-influence loss) | E1 | +0.002 (−0.002 … +0.006) | 6 h | 0.33e-3/h | none |
| 7 | Two-covariate IDR as a nonlinear combiner of `(a_D, a_DEPAVG)` | E2 | +0.002 (−0.002 … +0.005) | 4 h | 0.5e-3/h | none |
| 8 | Venn–Abers on the calibrated band-hit probability (downstream of #3 only) | D3 | +0.001 | 3 h | 0.33e-3/h | `venn-abers` (clean) |
| 9 | `monotone_constraints` rider on the wind→power learner | E3 | 0.000–0.001 | 1 h | rider only | none |
| — | NCL / diversity regularisation / DivBO / hard-example FOCUS | D2/E2 | ≤ 0 | 8–20 h | **negative** | — |
| — | loss-matched centroid combiner, median combiner, hard per-row selection | E2 | **−0.0017 to −0.0105 MEASURED** | — | **closed** | — |
| — | DRN/BQN head swap, NGBoost, LightGBMLSS, treeffuser | D2 | ≤ 0 or disqualified | — | closed | torch/pins |

---

## 11. BUILD ORDER — composable in one go

**Phase 0 (4 h, zero fits, may delete half of this document).**
1. **Vincentization audit.** `grep -n "mean(axis=0)\|np.mean(\[" ` over the composed-member pipeline
   (`research/nodes/s15_n7_compose.py`, `s15_n9_member_compose.py`, `s15_n10_member_nodensity.py`) for any
   averaging of 26-class probability vectors. If found → replace with quantile-function averaging and re-run
   the fit-free `evaluate_prob`. If not found → 20 minutes spent, item closed.
2. **CORP DSC** on `1{|a − y| ≤ 0.06 cap}` for the champion and each of the four members, per group,
   fold-outside. Report `MCB / DSC / UNC`. **Decision rule:** if the *incremental* DSC of a model that sees the
   disagreement geometry over one that does not is < 0.005, **do not build node 3** — the gate cannot reach
   0.34 top-1 and the whole §3 axis is dead for 2 h instead of 10.

**Phase 1 (10 h, the one big bet).** FFORMA gate, exactly as §3.4 specifies: metric-valued meta-labels,
softmax weights, custom `G = w(L − L̄)` objective in LightGBM, trigger `τ ∈ {0.06, 0.08, 0.10, 0.12}` and
shrinkage `λ ∈ {0, 0.15, 0.3, 0.5, 0.7, 1}` **both** chosen fold-outside, 3 seeds per contract R10, arbitrated
against the **honest** champion `0.634573 ± 0.000849`, not the deployed `0.636184`. Report the gate's
**top-1 accuracy on the trigger subset** alongside the Total — that number, not the Total, tells you whether
the axis is alive, and §3.2 converts it directly.

**Phase 2 (9 h, the two density arms, composable with each other and independent of Phase 1).**
EasyUQ/IDR arm (#4) and Mondrian CPS (#5). Both emit a 26-class matrix, so both drop into `loop_lib` unchanged
and can be scored fit-free. Judge each on **band-hit correlation with D** (target < 0.60) *before* looking at
Total; a member that lands there is the first genuinely complementary density this project has produced.

**Phase 3 (6 h, only if Phase 2's correlation target was met).** Modal-interval / asymmetric-Huber member (#6),
then two-covariate IDR combiner (#7) over whatever survives.

**Never build:** NCL, diversity-regularised objectives, DivBO, hard-example FOCUS reweighting, loss-matched
static combiners, hard per-row selection, hierarchical reconciliation across groups, DRN/BQN/NGBoost/LSS heads.

---

## 12. THE DIRECT ANSWER: CAN A MEMBER BE BOTH SHARP AND DECORRELATED FROM DEPAVG?

**Not ruled out — but you are asking the wrong question, and the measurement says so.**

1. **Decorrelation from DEPAVG already exists at the level the metric pays.** D's band-hit correlation with the
   deployed members is **0.41–0.43** while its continuous error correlation is 0.90–0.91 [M]. The union of the
   four members hits the u=4 band on **49.85%** of scored rows against the champion's **35.03%**, and the
   per-row oracle over those same four actions scores **0.723333**, i.e. **+0.087149** [M]. There is no
   complementarity shortage. There is a **conditioning** shortage.
2. **Sharp + decorrelated is achievable by exactly one construction in the literature I searched.** IDR/EasyUQ:
   in a like-for-like wind-gust postprocessing benchmark it produces the **narrowest prediction interval of
   any method tested** (4.72 vs EMOS 5.94, BQN 4.94, DRN 5.05) while carrying the **worst CRPS of the
   postprocessing methods** (0.98 vs 0.84) and **under-covering** (84.0% at nominal 90.5%) [A]. Sharp,
   deliberately not the most accurate, and estimated by an order-restricted nonparametric principle that shares
   no machinery with your GBDT family. That is the profile that will *not* reproduce the 0.9643 duplication.
   Second candidate: a bounded-influence, band-local loss (Taggart's asymmetric Huber at the 0.06 transition),
   which is the only loss in your pool whose influence function differs qualitatively from L1/cross-entropy.
3. **Every accuracy-oriented route to sharpness is ruled out.** Wood et al. prove the combiner is a property of
   the loss and warn against maximising diversity; NCL's exact decomposition holds only for squared loss;
   DRN/BQN buy sharpness with torch; NGBoost/LSS buy smooth parametric densities, which your own nine
   replications say cost more FICR than they gain NMAE. The set of things that make a *smooth* density better
   is disjoint from the set of things that make a *step-reward action* better, and that is now both measured
   (§2.2) and derived (§2.1).
4. **The expected value is lopsided.** A sharp-and-decorrelated *member* is worth, on my estimates, +0.002 to
   +0.003. Learning *which member to trust on a given row* is worth up to **+0.087** with an oracle and
   **+0.0087 at only 40% gate accuracy** [M]. Build the gate first.

---

## 13. HONEST GAPS

- I did **not** measure whether a feature-driven gate can actually exceed 0.34 top-1. That is Phase 0 item 2's
  job and it is the single number that decides this whole document.
- My gate-accuracy curve assumes conditionally-random gate errors. §3.3 shows a *constant* 51%-accurate rule
  loses 0.006, so the curve is an **upper** envelope for any given top-1 accuracy. Treat +0.0087 at 40% as
  optimistic by an unknown factor.
- My oracle uses `y`. It is a ceiling, not a forecast.
- I could not verify the Lichtendahl et al. (2013) *Management Science* article in raw (paywalled); the
  Proposition 4 statement is quoted verbatim from Tibshirani et al.'s transcription, which is grade A for the
  transcription and grade B for the original.
- I did not verify the runtime of 2-D PAVA at 19,795 rows; the IDR paper offers subsample aggregation as the
  remedy but I have not measured it.
- Schulz & Lerch's PI-length column is nominal-90.48% interval width in m/s on wind gusts, not band mass on
  capacity factor. The *ranking* transfers; the magnitude does not.
- `crepes`, `venn-abers` and `isodisreg` dependency sets were verified clean by the S14/S15 lanes, not
  re-verified here.

---

## 14. COMPLIANCE

Repository writes: this file and its search log, both under `research/lanes/`. Model fits **0**
(the numbers marked [M] are arithmetic over already-saved `.npy`/`.parquet` artifacts executed through the
project's own `.venv/bin/python` and the project's own `research/scratch/lib.py::official_total`; the champion
reproduces to `0.6361842493883538`). Lockbox reads **0**. Git staging **0**. Installs **0**. Uploads **0**.
Scratch scripts were written to `/tmp` and are not in the repository. All external documents were parsed in
memory and discarded.
