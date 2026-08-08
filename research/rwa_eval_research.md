# RWA-EVAL — adaptive time-series forecasting evaluation after repeated experiments

- Lane: read-only methodological research; retrieval date **2026-08-08**.
- Bound respected: **10 primary papers/methodological documents**, no subdelegation, no model fit, no score/lockbox/2024 inspection, and no repository write other than this file. [directly_supported]
- Local audit was limited to evaluation code, receipts, and key-only ordering metadata in M270/S15/S16; no target value or competition score was recomputed. [directly_supported]

## 0. Bottom line

1. A defensible BARAM comparison is an **issuance-level, strictly forward, nested prequential comparison of frozen action-producing procedures**, scored by the exact official `Total`; ordinary “other-fold” cross-fitting is not a chronological substitute. [directly_supported]
2. The repeatedly inspected 2023 development surface is no longer a fresh test set. Nested rerunning can remove mechanical look-ahead, and joint bootstrap methods can account for a frozen, logged candidate family, but neither can make past human-adaptive reuse disappear. [contradicts_premise]
3. The current paired-bootstrap idea is worth retaining, but the current implementation is not a moving-block bootstrap, splits BARAM's 01:00–00:00 24-hour issuance vector at midnight, calls a bootstrap sign fraction a “posterior,” and usually resets the comparison count to one. [contradicts_premise]
4. White's Reality Check / Hansen's SPA and Hansen–Lunde–Nason's Model Confidence Set (MCS) are the benchmark-supported joint-comparison tools. Their formal guarantees require a declared candidate set and stationary/mixing loss differentials; BARAM's adaptive candidate construction and visible regime variation make unqualified coverage claims inappropriate. [near_match_only]
5. Under the official step reward, the deployed output is an **action**, not necessarily a mean or median forecast. Candidate construction and evaluation must therefore use the exact combined `Total` loss, while raw band-hit correlation remains only a diagnostic. [directly_supported]

## 1. Evidence tags

Each material claim carries exactly one of these tags:

- `directly_supported`: the cited primary source or frozen official code directly establishes the claim in the stated scope.
- `contradicts_premise`: primary methodology or a key-only local audit directly conflicts with an existing evaluation premise/label.
- `near_match_only`: the method is relevant, but its assumptions or forecast unit differ materially from BARAM.
- `insufficient`: the retrieved evidence does not justify the claimed threshold, coverage, or migration.

## 2. Primary-source ledger (exactly 10 documents)

All full texts below were retrieved and read on **2026-08-08**. [directly_supported]

| # | Primary source and URL | Exact locator used | BARAM scope difference / migration constraint |
|---:|---|---|---|
| 1 | Cerqueira, Torgo & Mozetič, *Evaluating time series forecasting models: an empirical study on performance estimation methods* (Machine Learning, 2020), [primary preprint](https://arxiv.org/pdf/1905.11744) [directly_supported] | §§2.1–2.2, Fig. 3, pp. 3–5; §3.4.2, pp. 13–14; §5.1, p. 22 | Univariate next-value forecasts with immediate feedback, not one daily 24-hour vector. Migrate only the forward block/growing-window design; the BARAM test atom must be one complete issuance day. |
| 2 | Varma & Simon, *Bias in error estimation when using cross-validation for model selection* (BMC Bioinformatics 7:91, 2006), [official Europe PMC full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1397873/fullTextXML) [directly_supported] | “Background,” “Nested CV with shrunken centroids and SVM,” Eq. (8), Table 1, “Conclusion” | IID classification and random CV. Migrate the **inner selection / outer assessment separation**, but make both loops chronological and issuance-blocked. |
| 3 | Giacomini & White, *Tests of Conditional Predictive Ability* (Econometrica, 2006), [authors' primary working-paper version](https://econwpa.ub.uni-muenchen.de/econ-wp/em/papers/0308/0308001.pdf) [directly_supported] | §§3.1–3.2, pp. 9–12; Eqs. (3)–(4), Theorem 1 | Formal result uses a finite rolling estimation window; the one-step statistic differs from BARAM's expanding-history, 24-hour issuance. Use the “whole forecasting method + chosen loss” principle, not the theorem unmodified. |
| 4 | Politis & Romano, *The Stationary Bootstrap* (JASA 89, 1994), [primary article copy](https://users.ssc.wisc.edu/~behansen/718/Politis%20Romano.pdf) [directly_supported] | pp. 1303–1307; §2 algorithm; Theorems 1–2 | Strictly stationary, weakly dependent series. BARAM must first form a chronological series of complete issuance-day loss vectors; raw group-stacked hourly rows do not meet that migration. |
| 5 | White, *A Reality Check for Data Snooping* (Econometrica 68, 2000), [primary article copy](https://users.ssc.wisc.edu/~behansen/718/White2000.pdf) [directly_supported] | pp. 1097–1105; null and statistic at pp. 1101–1102; bootstrap at pp. 1103–1105 | A finite searched family with its relative-performance matrix available. It does not license omitted failed variants or unrestricted human-adaptive creation from score feedback. |
| 6 | Hansen, *A Test for Superior Predictive Ability* (JBES 23, 2005), [institutional primary copy](http://cdr.lib.unc.edu/downloads/zp38wf793) [directly_supported] | pp. 365–371; Eq. (1), Table 1, SPA statistic and null recentering at p. 368; bootstrap §3 | Assumes stationary, mixing relative losses and excludes recursively estimated parameters in its stated framework. Apply only to frozen daily OOF action losses, with assumptions reported. |
| 7 | Hansen, Lunde & Nason, *The Model Confidence Set* (Econometrica 79, 2011), [primary article copy](https://www.kevinsheppard.com/files/teaching/mfe/advanced-econometrics/Hansen_Lunde_Nason.pdf) [directly_supported] | Defs. 1–2 and Theorem 1, pp. 458–459; Assumption 2, p. 463; `T_R`, `T_max`, and elimination rules, pp. 465–466 | MCS coverage is relative only to the supplied `M0` and assumes stationary/mixing loss differentials. A 23-model MCS cannot certify all earlier adaptive trials. |
| 8 | Dwork et al., *Preserving Statistical Validity in Adaptive Data Analysis* (STOC 2015), [primary preprint](https://arxiv.org/pdf/1411.2664) [directly_supported] | Abstract and §1, pp. 1–2; informal Theorem 1, p. 4 | IID statistical-query theory, not dependent forecasts. It directly diagnoses adaptive holdout reuse, but its privacy-based reusable-holdout mechanism is not a drop-in BARAM time-series test. |
| 9 | Gneiting, *Making and Evaluating Point Forecasts* (JASA 106, 2011), [primary preprint](https://arxiv.org/pdf/0912.0902) [directly_supported] | Eq. (5) and decision setting, pp. 9–10; Eq. (6), p. 10; Theorem 2.2, p. 12 | General decision theory directly applies; BARAM's Bayes action must use the official combined loss rather than MAE alone. |
| 10 | Brehmer & Gneiting, *Scoring Interval Forecasts: Equal-Tailed, Shortest, and Modal Interval* (Bernoulli 27, 2021), [primary preprint](https://arxiv.org/pdf/2007.05709) [directly_supported] | §3.4, Eqs. (9)–(11), Theorems 3.9–3.10, pp. 13–14 | A single fixed-width, unweighted hit reward. BARAM adds two bands, generation weighting, eligibility filtering, group ratios, and MAE; modal-interval theory is therefore only a component analogue. |

## 3. What benchmark practice actually supports

### 3.1 Rolling-origin / prequential design

- Blocked prequential evaluation first tests on a later sequential block and only then makes that block available to subsequent training; growing, sliding, and gap variants are explicit benchmark designs. [directly_supported]  
  **Locator/date:** Cerqueira et al. §2.2, Fig. 3, pp. 4–5; retrieved 2026-08-08.  
  **Migration:** BARAM block `d` is the whole action vector issued together at D−1 14:00 KST, not an hour and not a randomly shuffled row.

- The empirical benchmark found temporally ordered out-of-sample methods more accurate under real-world nonstationarity, while stationary synthetic cases could favor cross-validation; it did **not** establish one universally best estimator. [directly_supported]  
  **Locator/date:** Cerqueira et al. abstract and §5.1, pp. 1 and 22; retrieved 2026-08-08.  
  **Migration:** BARAM should preserve order because regime change is plausible, but must report that the benchmark was univariate next-step, not 24-hour multigroup wind actions.

- Giacomini–White evaluate the complete “forecasting method”—model, estimator, estimation window, and weights—through a simulated rolling out-of-sample sequence of length `n = T − τ − m + 1`, using the user's loss. [directly_supported]  
  **Locator/date:** Giacomini–White §3.1, p. 9; retrieved 2026-08-08.  
  **Migration:** policy temperature, reward weight, blend, threshold, training-window rule, and seed aggregation are all part of the BARAM method and must be frozen or selected inside the chronological inner loop.

- Their conditional equal-predictive-ability null and one-step Wald statistic are
  \[
  H_0:E[\Delta L_{m,t+\tau}\mid\mathcal F_t]=0,\qquad
  T^h_{n,m}=n\bar Z_{m,n}'\widehat\Omega_n^{-1}\bar Z_{m,n}\Rightarrow\chi_q^2,
  \quad Z_{m,t+1}=h_t\Delta L_{m,t+1}.
  \] [directly_supported]  
  **Locator/date:** Giacomini–White Eqs. (3)–(4), pp. 10–11; retrieved 2026-08-08.  
  **Scope:** its stated theorem fixes finite `m` and treats one-step losses; applying this chi-square reference directly to overlapping hourly BARAM outcomes would be unsupported.

### 3.2 Nested selection

- Reusing the same CV estimate both to choose parameters and to report performance is optimistically biased; every choice, including feature and parameter selection, must be repeated inside the inner loop, while the outer loop assesses the resulting algorithm. [directly_supported]  
  **Locator/date:** Varma–Simon “Background,” “Nested CV…,” and “Conclusion”; retrieved 2026-08-08.  
  **Migration:** every BARAM feature set, learner, policy `(T,G)`, blend weight, threshold, and seed-selection rule must be chosen only from issuance dates earlier than the held outer origin.

- In the paper's illustrative equal-error, median-unbiased setup, selecting the minimum among `K` noisy estimates gives
  \[
  \Pr\{\min_k e_k<E\}=1-(1/2)^K,
  \]
  illustrating how the minimum becomes optimistic as the search grows. [directly_supported]  
  **Locator/date:** Varma–Simon Eq. (8); retrieved 2026-08-08.  
  **Scope:** the displayed calculation is illustrative and does not model correlated time-series trials; it is not a BARAM bias correction formula.

- Fixed-hypothesis multiplicity theory does not automatically cover hypotheses and analysis choices created adaptively from earlier results; conventional holdout/CV reuse can overfit the reused surface. [directly_supported]  
  **Locator/date:** Dwork et al. abstract and §1, pp. 1–2; retrieved 2026-08-08.  
  **Migration:** a retrospective nested rerun can test a now-frozen algorithm, but Q3/Q4 cannot be described as “unseen” for a lineage designed after their scores were repeatedly observed.

### 3.3 Dependent/block bootstrap

- The stationary bootstrap draws start indices uniformly on a circular series and draws block lengths `L` geometrically,
  \[
  \Pr(L=m)=(1-p)^{m-1}p,\qquad E[L]=1/p,
  \]
  continuing the next original observation with probability `1-p` and restarting uniformly with probability `p`. [directly_supported]  
  **Locator/date:** Politis–Romano §2, pp. 1303–1304; retrieved 2026-08-08.  
  **Migration:** resample a chronological sequence of complete BARAM issuance-day vectors so contemporaneous 24 hours and all three groups remain together.

- The first-order consistency results require stationary/weak dependence and a sequence `p_N→0` with `N p_N→∞`; the paper notes practical `p` choice is consequential beyond first order. [directly_supported]  
  **Locator/date:** Politis–Romano Theorems 1–2 and discussion, pp. 1306–1307; retrieved 2026-08-08.  
  **Migration:** seven days is not validated merely because it was used previously; it must be preregistered and accompanied by block-length sensitivity and regime diagnostics.

- `research/engine/arbiter.py:28–53` first normalizes calendar timestamps, partitions each fold into **non-overlapping** seven-day chunks, and resamples those chunks; this is not the moving-block algorithm named in its module docstring. [contradicts_premise]  
  **Locator/date:** local code audit 2026-08-08; Politis–Romano §2 and White pp. 1104–1105 define random-start fixed or geometric blocks.  
  **Migration:** use overlapping/circular or stationary starts within each chronological outer block, never fixed chunk boundaries chosen by the calendar.

- Because BARAM delivery days run 01:00 through the following 00:00, `.dt.normalize()` places the final hour in the next calendar day and breaks the issuance vector. [contradicts_premise]  
  **Locator/date:** `arbiter.py:28–32`, `research/scratch/lib.py:5–9`, and a key-only audit of `S7-N8_D_keys.parquet` on 2026-08-08.  
  **Migration:** define `issuance_day = (forecast_kst_dtm - 1 hour).normalize()` for the current 01:00–00:00 convention, then assert exactly 24 target hours × 3 groups before resampling.

### 3.4 Multiplicity and selection bias

Let `d_{k,t}=L_{0,t}-L_{k,t}` be candidate `k`'s realized loss advantage over the benchmark. [directly_supported]

- White's data-snooping null and Reality Check statistic are
  \[
  H_0:\max_{k\le K}E[d_{k,t}]\le0,\qquad
  V_K=\max_{k\le K}\sqrt n\,\bar d_k,
  \]
  with the **joint** maximum's null law obtained by dependent bootstrap rather than by testing the selected winner alone. [directly_supported]  
  **Locator/date:** White pp. 1101–1105; retrieved 2026-08-08.  
  **Migration:** the matrix must include every saved candidate/policy/threshold whose 2023 result influenced selection, with the same issuance days and exact official loss orientation.

- Hansen's more powerful SPA uses studentization and a sample-dependent null:
  \[
  T_{SPA}=\max\!\left\{0,\max_k\frac{\sqrt n\,\bar d_k}{\hat\omega_k}\right\},\qquad
  \hat\mu_k^c=\bar d_k\mathbf1\!\left\{\frac{\sqrt n\bar d_k}{\hat\omega_k}\le-\sqrt{2\log\log n}\right\}.
  \] [directly_supported]  
  **Locator/date:** Hansen pp. 367–371, especially p. 368; retrieved 2026-08-08.  
  **Scope:** its formal assumptions require stationary/mixing relative losses and do not accommodate recursive parameter estimation as stated, so BARAM should report SPA as assumption-qualified sensitivity evidence.

- A scalar threshold based only on the number of trials is not equivalent to resampling the correlated maximum, and omitted adaptive variants cannot be repaired by setting `n_comparisons` to a larger integer after the fact. [contradicts_premise]  
  **Locator/date:** White pp. 1097–1105; Dwork et al. pp. 1–2; local `arbiter.py:69–79`; audited 2026-08-08.  
  **Migration:** replace per-candidate arbitration by a single joint family analysis after freezing and reconstructing the experiment ledger.

- In inspected S15/S16 nodes, most calls pass `n_comparisons=1` (for example `s15_n6_draw_verdict.py:81`, `s16_n3_fforma_gate.py:160`, `s16_n4_hit_gate.py:150`, and `s16_n11_delivery_arbitrate.py:60`), so contract R6 is not globally accumulated in those decisions. [contradicts_premise]  
  **Locator/date:** local code audit 2026-08-08.  
  **Migration:** comparison identity belongs to the shared surface/experiment ledger, not to an individual script invocation.

- `p_better = mean(bootstrap_delta > 0)` in `arbiter.py:61` is a bootstrap sign fraction, not a Bayesian posterior probability; the papers retrieved here provide no basis for the label “posterior.” [contradicts_premise]  
  **Locator/date:** local code audit 2026-08-08; White/Hansen use centered bootstrap null distributions and p-values.  
  **Migration:** report a bootstrap p-value or simultaneous confidence bound with its null-centering rule, never `P(better)` or “posterior” without a probabilistic model and prior.

### 3.5 Model Confidence Sets

For a frozen set `M`, define
\[
d_{ij,t}=L_{i,t}-L_{j,t},\quad
t_{ij}=\frac{\bar d_{ij}}{\sqrt{\widehat{\mathrm{var}}(\bar d_{ij})}},\quad
\bar d_{i\cdot}=|M|^{-1}\sum_{j\in M}\bar d_{ij}.
\]
The two published omnibus statistics and coherent eliminations are
\[
T_{R,M}=\max_{i,j\in M}|t_{ij}|,\quad e_{R,M}=\arg\max_i\sup_j t_{ij};
\qquad
T_{\max,M}=\max_{i\in M}t_{i\cdot},\quad e_{\max,M}=\arg\max_i t_{i\cdot}.
\]
[directly_supported]  
**Locator/date:** Hansen–Lunde–Nason pp. 465–466; retrieved 2026-08-08.

- Repeatedly test equal predictive ability, remove the coherently identified worst model after rejection, and stop at the surviving set; under the paper's assumptions the set contains the best member(s) of `M0` with asymptotic probability at least `1−α`. [directly_supported]  
  **Locator/date:** Definition 2 and Theorem 1, p. 459; retrieved 2026-08-08.  
  **Migration:** report the complete `M0`, `α`, bootstrap method, block length, exact loss, and included/excluded set; a large MCS means the data do not identify one champion.

- MCS is explicitly silent about models outside `M0`, and its bootstrap implementation assumes strictly stationary, mixing loss differentials with finite moments. [directly_supported]  
  **Locator/date:** Hansen–Lunde–Nason p. 458 and Assumption 2, p. 463; retrieved 2026-08-08.  
  **Migration:** S16's 10% MCS can only describe its enumerated actions, not all M270/S15/S16 adaptive choices or unseen model families.

- `run_mcs.py:25–37` builds only solos and fixed `0.3/0.7` blends from a limited member list, while `mcs.py:48` bootstraps `168` raw rows. Key-only ordering is fold → all hours of group 1 → group 2 → group 3, so raw-row stationary blocks do not preserve the three-group issuance vector and can cross a time reset at group boundaries. [contradicts_premise]  
  **Locator/date:** local code and key-only metadata audit 2026-08-08.  
  **Migration:** aggregate exact additive loss contributions to one row per issuance day before MCS, and state the MCS family narrowly unless every influential candidate can be reconstructed.

- The current full-sample linearization in `mcs.py:1–45` reproduces the observed official `Total` algebraically, but the retrieved MCS paper does not establish finite-sample coverage for bootstrapping that nonlinear ratio metric with denominators held fixed. [insufficient]  
  **Locator/date:** frozen official metric notebook cell 1, `mcs.py:1–45`, and Hansen–Lunde–Nason Assumption 2; audited/retrieved 2026-08-08.  
  **Migration:** run both (a) MCS on the exact additive daily decomposition with fixed observed denominators and (b) a joint daily bootstrap that recomputes official `Total` inside every resample; disagreement blocks an inferential claim.

### 3.6 Actions under the discontinuous official reward

For group `g`, capacity `c_g`, eligible rows `V_g={i:y_{ig}\ge0.1c_g}`, `n_g=|V_g|`, error `e_{ig}=|a_{ig}-y_{ig}|/c_g`, and
\[
u(e)=4\mathbf1(e\le0.06)+3\mathbf1(0.06<e\le0.08),
\]
the frozen official notebook implements
\[
\mathrm{Total}=
\frac12\left[1-\frac13\sum_g\frac1{n_g}\sum_{i\in V_g}e_{ig}\right]
+\frac12\left[\frac13\sum_g\frac{\sum_{i\in V_g}y_{ig}u(e_{ig})}{4\sum_{i\in V_g}y_{ig}}\right].
\]
[directly_supported]  
**Locator/date:** `inputs/notebooks/metric_official.ipynb`, cell 1; audited 2026-08-08.

With `S_g=\sum_{i\in V_g}y_{ig}`, an exact observed-sample additive decomposition is
\[
0.5-\mathrm{Total}
=\sum_g\sum_{i\in V_g}\left[\frac{e_{ig}}{6n_g}-\frac{y_{ig}u(e_{ig})}{24S_g}\right].
\]
[directly_supported]  
**Locator/date:** direct algebra from the frozen official cell 1, identical to `research/engine/mcs.py:30–44`; audited 2026-08-08.  
**Migration:** sum the bracketed terms over all 24 hours and three groups to form one daily loss observation; keep `Total` itself as the primary reported statistic.

- A Bayes action minimizes expected task loss, `a*(x)=argmin_a E[L(a,Y)|X=x]`; evaluating a point forecast with a mismatched score can select the wrong functional. [directly_supported]  
  **Locator/date:** Gneiting Eqs. (5)–(6), pp. 9–10, and Theorem 2.2, p. 12; retrieved 2026-08-08.  
  **Migration:** do not force BARAM actions toward the conditional mean merely because MAE improves.

- The population analogue of the groupwise BARAM action, with `μ_g=E[Y|Y≥0.1c_g,g]`, is
  \[
  a_g^*(x)\in\arg\min_a E\!\left[
  \mathbf1\{Y\ge0.1c_g\}\left
  \{\frac{|a-Y|}{c_g}-\frac{Y\,u(|a-Y|/c_g)}{4\mu_g}\right\}\mid X=x,g\right].
  \] [directly_supported]  
  **Locator/date:** algebraic population counterpart of official cell 1 plus Gneiting's Bayes-act equation; audited/retrieved 2026-08-08.  
  **Migration:** estimate `μ_g` only inside the past-only inner training data; using the outer period's realized `S_g/n_g` to choose actions leaks outcomes.

- For a single unweighted fixed half-width `c`, the hit-maximizing action is the midpoint of a modal interval,
  \[
  MI_c(F)=\arg\max_{[a,b]:b-a\le2c}\{F(b)-F(a^-)\},\quad
  S(x,y)=-\mathbf1(x-c\le y\le x+c).
  \] [directly_supported]  
  **Locator/date:** Brehmer–Gneiting Eqs. (9) and (11), pp. 13–14; retrieved 2026-08-08.  
  **Scope:** this does not by itself solve BARAM's MAE + two-tier + `y`-weighted objective.

- S16's unweighted `u=4` band-hit indicator correlation is a useful diversity diagnostic, but it omits the `u=3` shell, generation weights, group normalization, and the MAE half of official `Total`; it cannot be the promotion loss. [near_match_only]  
  **Locator/date:** `research/lanes/S16_sota_decide.md:51–72` versus official metric cell 1; audited 2026-08-08.  
  **Migration:** additionally record correlation/covariance of the realized official utility contribution `y·u(e)` and always arbitrate on exact `Total`.

## 4. Benchmark-supported method versus proposed BARAM convention

### 4.1 Benchmark-supported

| Practice | Status |
|---|---|
| Forward blocked/prequential evaluation that preserves temporal order [directly_supported] | Supported by Cerqueira et al.; choose growing or sliding training as part of the method. |
| All tuning inside an inner loop, assessment in an outer loop [directly_supported] | Supported by Varma–Simon; chronology is the BARAM-specific adaptation. |
| Paired loss differences on the same outcomes [directly_supported] | Built into Giacomini–White, White RC, SPA, and MCS. |
| Dependent resampling with random-start consecutive blocks [directly_supported] | Supported by Politis–Romano; stationarity/mixing assumptions must be stated. |
| Joint max test for a searched family, preferably studentized SPA over unstudentized RC [directly_supported] | Supported by White and Hansen, within their assumptions. |
| MCS instead of declaring a unique winner when data are uninformative [directly_supported] | Supported by Hansen–Lunde–Nason, relative to a frozen `M0`. |
| Evaluate the issued action with the decision loss [directly_supported] | Supported by Gneiting; modal-interval theory supports only the single-band component. |

### 4.2 Proposed local conventions (not literature mandates)

| Proposed convention | Evidence boundary |
|---|---|
| One resampling atom is `issuance_day=(timestamp−1h).normalize()` and must contain 72 group-hours. [near_match_only] | Exact mapping is dictated by BARAM's current 01:00–00:00 layout, not by any paper. |
| Stratify bootstrap resampling by frozen outer quarter, using stationary blocks within each quarter. [near_match_only] | Preserves BARAM's designed fold composition; formal MCS/SPA results do not specifically prove this stratified nonlinear-Total implementation. |
| Predeclare expected block length 7 days and show sensitivity at 3 and 14 days. [insufficient] | Seven days has local precedent and roughly the right dependence scale, but no retrieved source identifies it as optimal for BARAM. Never pick the most favorable length. |
| Use at least 10,000 joint bootstrap replicates with a fixed seed. [near_match_only] | A Monte Carlo precision convention; the papers require sufficiently many replicates but do not mandate 10,000. |
| Use `α=0.10` to match the existing 90% MCS, and practical superiority margin `δ=max(0.001, 0.001635)=0.001635 Total`. [insufficient] | `α`, the old minimum effect, and the S15 seed floor are project choices, not literature-derived constants. |
| Retain “Q3 and Q4 point deltas both positive” as a preregistered regime guardrail, not a p-value. [near_match_only] | It is conservative temporal stress testing with only two quarters; it has no claimed nominal error rate. |

## 5. Which existing M270 / S15 / S16 gates survive

| Existing gate or practice | Verdict | Exact reason / required repair |
|---|---|---|
| M270 basis-time availability, licence/provenance, and no-2024 rules | **KEEP unchanged** [directly_supported] | They are deployment/legal integrity gates and remain prerequisites to any statistical comparison. |
| Strict chronological base OOF fitting (`train time < held origin`) | **KEEP, assert per issuance** [directly_supported] | This is the core prequential condition; assert maximum label/feature availability time against D−1 14:00 KST. |
| `fo_policy` / `fo_blend_1dof` selecting on “the other two folds” | **FAIL / replace** [contradicts_premise] | `loop_lib.py:94–143` uses `~sel`, so Q2 can use Q3/Q4 and Q3 can use Q4. Replace with past-only inner selection. |
| M270 `Q3/Q4 both positive` | **KEEP only as guardrail** [near_match_only] | It detects sign reversal across two future blocks but is not an inferential test and does not repair adaptive reuse. |
| M270 `Q4 bootstrap > 0.50` | **RETIRE** [insufficient] | A sign fraction above one-half provides neither a confidence level nor family-wise control. Use a centered joint bootstrap/SPA bound. |
| M270 monthly positive-share/median/worst-month table | **KEEP descriptive; do not call a test** [near_match_only] | Nine adjacent months reveal regimes but are dependent and too few to justify post-hoc thresholds. Predeclare only a collapse guardrail. |
| S15/S16 R1 paired candidate–champion comparison on identical rows | **KEEP principle; replace implementation** [directly_supported] | Pairing is correct, but resample complete issuance days jointly and use a valid centered statistic. |
| R4 provenance: policy, weight-fitting status, row key | **KEEP unchanged** [directly_supported] | These fields define the forecasting method and make nested chronology auditable. Add basis-time and outer/inner origin IDs. |
| R5 predeclaration | **KEEP and strengthen** [directly_supported] | Freeze candidate family, primary official `Total`, margin, loss orientation, origins, block lengths, and multiplicity method before seeing results. |
| R6 multiplicity tracking | **KEEP principle; current counter fails** [contradicts_premise] | Use the full loss matrix and joint SPA/MCS. Per-node `n_comparisons=1` and scalar threshold bumps do not track the search. |
| R10 at least three seed refits and seed averaging | **KEEP as local stability gate** [near_match_only] | It protects against lucky initialization but is not a substitute for temporal sampling uncertainty or multiplicity control. |
| `MIN_EFFECT=0.001` / seed floor `0.001635` | **KEEP only as declared decision margin** [insufficient] | Require a simultaneous lower bound above the chosen margin; do not present either constant as a literature significance threshold. |
| S16 band-hit indicator correlation | **KEEP diagnostic, expand** [near_match_only] | Add the weighted official utility `y·u(e)` and the `u=3` shell; promotion remains exact `Total`. |
| S16 10% MCS | **KEEP method idea; current result is narrow/descriptive** [near_match_only] | Rebuild on issuance-day losses, explicit stationary diagnostics, and a complete declared `M0`; do not generalize beyond that set. |
| Threshold/policy variants chosen after inspecting the same surface, then arbitrated as one comparison | **FAIL** [contradicts_premise] | Every inspected variant belongs to the selection family; selection must be inner-loop or included in the joint multiplicity audit. |

## 6. Minimal validation protocol that never inspects 2024

### Step 0 — freeze and label the evidential ceiling

- Before any calculation, hash the candidate code/action artifact, incumbent, exact policy, seed-aggregation rule, blend/threshold rule, row key, primary `Total`, margin `δ`, outer origins, candidate family `M0`, bootstrap method, block lengths `{3,7,14}`, `α`, and replicate count. [directly_supported]
- State in the receipt: **“2023 has been adaptively reused; this is retrospective, selection-adjusted evidence, not a fresh unbiased holdout.”** [contradicts_premise]
- Do not read, aggregate, or use any 2024 target or score. [directly_supported]

### Step 1 — reconstruct strict chronological nested OOF actions

- Define one outer case as the vector issued at D−1 14:00 KST for target hours 01:00–00:00 and groups 1–3; assert 72 cells, uniqueness, and all feature availability times `≤ basis_time`. [directly_supported]
- For each outer origin `o`, enforce
  \[
  \max\{\text{label/selection time used for fit, features, policy, blend, threshold}\}<o.
  \]
  All selection is run by code frozen before the outer loop; no later fold can enter. [directly_supported]
- With the existing 2023 surface, use Q2 only as burn-in/inner selection and freeze the selected procedure before producing Q3 and Q4 outer actions; alternatively, create earlier inner rolling origins entirely before each outer quarter. [near_match_only]
- Existing meta-actions produced by `fo_policy`/`fo_blend_1dof` are inadmissible for this strict check until re-created past-only, even if the underlying model predictions themselves were trained chronologically. [contradicts_premise]

### Step 2 — exact loss tensor and diagnostics

- For every frozen model/action and issuance day, calculate the daily sum of
  \[
  \ell_{ig}=\frac{e_{ig}}{6n_g}-\frac{y_{ig}u(e_{ig})}{24S_g};
  \]
  assert over all days that `sum(loss)=0.5−official_Total` to machine tolerance. [directly_supported]
- Primary output is paired `ΔTotal`; report `Δ(1−NMAE)`, `ΔFICR`, Q3, Q4, monthly deltas, `u=4`, `u=3` shell, and `y·u(e)` only as prespecified diagnostics. [directly_supported]
- Plot daily loss differentials, ACF, and quarter/month means; if the mean visibly changes by regime, do not claim stationary-bootstrap/MCS nominal coverage. [near_match_only]

### Step 3 — one joint dependent-resampling analysis

- Form a time-ordered daily loss matrix for the incumbent and every reconstructable candidate/variant that influenced selection; deduplicate identical action/loss columns but retain an alias ledger. [directly_supported]
- Within each outer quarter, apply a stationary bootstrap to complete days, concatenate the resampled quarters, and recompute **exact official `Total`** for every model in every replicate. Run the preregistered primary expected length 7 and sensitivity lengths 3 and 14 with the same joint draws across models. [near_match_only]
- Compute (i) the studentized SPA maximum against the incumbent and (ii) the 90% `T_R` MCS. Separately repeat MCS on the exact additive daily losses; if the exact-Total and additive-loss conclusions differ, mark inference `insufficient`. [near_match_only]
- Replace `p_better` with the centered joint-bootstrap p-value, simultaneous lower confidence bound, MCS membership, and Monte Carlo standard error. [directly_supported]

### Step 4 — promotion rule

A candidate may replace the incumbent only if all of the following were frozen in Step 0. [near_match_only]

1. all availability, alignment, provenance, strict chronology, and exact-metric reproduction assertions pass; [directly_supported]
2. its seed-averaged point improvement is at least the local margin `δ=0.001635 Total`; [near_match_only]
3. its multiplicity-adjusted one-sided lower bound exceeds `δ`; a margin-shifted joint test with `d'_{k,t}=d_{k,t}-δ` rejects “no candidate improves by `δ`,” the candidate remains in the 90% MCS, and the incumbent is excluded; [near_match_only]
4. Q3 and Q4 point deltas are both positive under the single frozen rule, and the result does not reverse at block lengths 3, 7, or 14 days; [near_match_only]
5. no decision is made from component metrics, a favorable month, a favorable block length, a lucky seed, or a selected threshold alone. [directly_supported]

### Step 5 — honest conclusion without 2024

- If the gate fails, retain the incumbent; failure proves only lack of robust evidence on this reused 2023 surface, not universal inferiority. [directly_supported]
- If the gate clears, the strongest permissible label is **“retrospective, chronology-repaired, multiplicity-aware support on 2023 Q3–Q4.”** It is not fresh validation and cannot establish full-year/winter or 2025 superiority. [contradicts_premise]
- Without an untouched legal evaluation period or a purpose-built reusable-holdout mechanism for dependent time series, no method in the ten retrieved sources can recover a genuinely independent post-selection estimate from the already reused data. [insufficient]

## 7. Recommended immediate disposition

- Preserve M270's availability/provenance/no-2024 gates, R1 pairing, R4 provenance, R5 preregistration, R10 seed averaging, official `Total`, and S16's hit-space diagnostics in their narrow roles. [directly_supported]
- Stop using `fo_policy`'s other-two-fold selection, raw-row seven-day “moving” bootstrap, `p_better` posterior language, per-node multiplicity resets, and `Q4 bootstrap > 0.50` as championship evidence. [contradicts_premise]
- Rebuild only the evaluation layer first: issuance-day keys, past-only nested selection, complete trial ledger, and a joint exact-Total SPA/MCS receipt. No 2024 inspection is needed for that repair, but its conclusion must remain retrospective. [near_match_only]
