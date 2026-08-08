# RWA-MODEL — probabilistic wind forecasting and decision learning for a 6%/8% step reward

**Retrieval date:** 2026-08-08  
**Lane:** read-only research; brief was already routed.  
**External evidence cap:** 14 primary papers / official competition reports, all read from primary PDF/HTML.  
**Local inspection:** only the targeted S15/S16 reports, scripts, and saved JSON diagnostics listed below.  
**Actions in this lane:** model fits 0; score runs 0; lockbox reads 0; installs/data actions 0; subdelegations 0; repository writes 1 (this file only). **[C01]**

## Evidence-tag rule

Every quantitative, causal, comparative, or recommendation claim below carries exactly one evidence tag (trailing in prose; the tag column in tables). `E..` means one external primary source, `L..` means one saved local S15/S16 artifact, `D..` means algebra performed in this memo, `S..` means an explicitly marked synthesis of separately cited evidence, `R..` means a proposed test/decision rule rather than observed evidence, and `C..` means a lane-compliance observation. **[C02]**

## 1. Executive verdict

1. The strongest public wind benchmarks support non-parametric quantile/tree models, source-aware preprocessing, and ensembles under **proper distributional scores**; they do not provide a measured gain for BARAM's generation-weighted 6%/8% step utility. **[E01]**
2. FFORMA is strong evidence for *series-level* soft weighting on M4, where it beat both hard selection and equal averaging. **[E04]**
3. The already-built BARAM row-level FFORMA gate achieved top-1 `0.535894` and still reduced Total from `0.636184` to `0.633169`; another row-level hard/soft gate is therefore a duplicate, not an open axis. **[L05]**
4. The four-member local closest-action oracle Total `0.723333` is an **outcome-aware reachability ceiling** because it selects after seeing `y`; the saved local development champion is `0.636184`. **[L04]**
5. None of the tested out-of-sample selectors converted that oracle gap into a reliable gain. **[S02]**
6. IDR, hit-probability gates, pool expansion, location shifts, and band-hit feature selection have already produced honest negatives; renaming any of them “distributional,” “modal,” or “diverse” does not create a new operation. **[S01]**
7. Only one non-duplicate one-shot experiment survives this audit: a regret-aware SPO+ selector trained on the **complete cost vector** of the four member actions plus the incumbent champion as a safe reference action. **[R01]**

## 2. What the reward actually asks the learner to estimate

Let `c_g` be group capacity and define the step reward below. **[D01]**

`u(e) = 4·1(e <= 0.06 c_g) + 3·1(0.06 c_g < e <= 0.08 c_g)`. **[D01]**

Because `u(e) = 1(e <= 0.06 c_g) + 3·1(e <= 0.08 c_g)`, the conditional FICR part of an action `a` is a positive combination of two fixed-width window coverages under the **generation-size-biased** law (`Y` multiplies the reward), not the raw conditional mode. With positive group constants `k_M,g` and `k_F,g`, the Bayes action has the form below. **[D01]**

`a*(x) in argmax_a {-k_M,g E[|a-Y|/c_g | x] + k_F,g E[Y·u(|a-Y|) | x]}`. **[D01]**

Thus the target combines an L1 functional with 6% and 8% modal-window functionals; moving toward the conditional mean is not generally score-improving. **[D01]**

Brehmer and Gneiting prove elicitability for a **single** fixed-width modal interval with score `-1{|a-Y| <= h}`. Their theorem does not by itself cover BARAM's sum of two widths, its `Y` weight, or its simultaneous NMAE term. **[E08]**

For any frozen finite action set `A={a_1,...,a_K}`, observing `Y=y` reveals the entire cost vector `c_j(y)=-U(a_j,y)` for all `K` actions. Selecting one action is therefore a one-hot linear optimization problem `min_j E[c_j(Y)|X=x]`, even though the original utility is discontinuous in error. **[D02]**

## 3. Oracle reachability is not selector learnability

| Observation | Exact scope | Evidence |
|---|---|---|
| Four saved actions with a per-row closest-action oracle: champion u=4 hit `0.350340`, any member hit `0.498520`, oracle Total `0.723333`, champion Total `0.636184`; the average discards `0.156103` of scored-row hits and manufactures only `0.007923`. | Same aligned 19,785-row local development surface; the oracle uses realised `y`. | **[L04]** |
| The selected D seed's champion is optimistic relative to six sibling seeds: mean blend `0.634573`, seed SD `0.000849`, selected champion `0.636184` (`z=1.897`). | Seed variation on the saved S15 surface; it is not an online estimate. | **[L01]** |
| FFORMA seed-average Total `0.633169`, point delta `-0.003016`, paired 95% interval `[-0.005430,-0.000331]`, despite top-1 `0.535894`. | Four-member, fold-outside, per-row exact member-loss gate. | **[L05]** |
| Four binary hit models' best margin rule scored `0.636247`, only `+0.000063` point delta; `p_better=0.555` and its paired interval crossed zero. | Four-member hit-attainability gate; this is a statistical tie, not a bankable gain. | **[L06]** |
| Expanding from four to fourteen members raised any-hit `0.498520 -> 0.642173` and oracle `0.723333 -> 0.807182`. | Outcome-aware pool ceiling only. | **[L07]** |
| The fourteen-member learned gate's best saved rule was `0.635821`, below `0.636184`, although its realised u=4 hit rate was `0.350949`. | Same expanded pool, out-of-sample gate. | **[L08]** |

The closest-action `0.723333` number proves that useful actions coexist in the pool; it does not prove that the basis-time feature set identifies which action will score. The local experiments show that top-1 accuracy, union hit coverage, and oracle Total can all rise without feasible Total rising. **[D03]**

Accordingly, this memo treats `~0.723` as **oracle reachability** and `~0.636` (with a seed-mean warning at `~0.6346`) as the current **learnable development evidence**. No expected-value calculation below transfers the oracle gap into an out-of-sample forecast gain. **[D04]**

## 4. Primary-source register (14 materials)

| Tag | Primary material and URL | Exact measured/theoretical scope | Migration gap to BARAM |
|---|---|---|---|
| **[E01]** | Hong et al. (2016), *Probabilistic energy forecasting: GEFCom2014 and beyond*, [PDF](http://pierrepinson.com/docs/Hongetal2016.pdf) | Official competition report: 581 participants from 61 countries; wind track issued daily 24-h forecasts for 10 Australian zones as 99 quantiles. All top five used non-parametric methods; the winner used two-layer GBMs and had a `56.6%` benchmark-relative rating. | Proper pinball score, 99 quantiles, different basis time/data, and no 6%/8% action reward; it validates a forecasting baseline, not BARAM Total. |
| **[E02]** | Olauson, Viotti & Huss (2026), *The HEFTCom2024 winning model*, [accepted PDF](https://www.diva-portal.org/smash/get/diva2:2046959/FULLTEXT01.pdf) | Full-year 2023 concurrent test (`93%` timestamps): source-separated stacked CatBoost pinball `28.5`; same weather inputs concatenated in one no-target CatBoost `28.6`; organiser reference `38.0`. Three NWP sources were GFS, DWD and MEPS. | Hybrid wind+solar pinball task; stacking and concatenation are effectively tied (`0.1`, about `0.35%`), and separate source preprocessing is confounded with architecture. |
| **[E03]** | Browell et al. (2025), *The Hybrid Renewable Energy Forecasting and Trading Competition 2024*, [PDF](https://arxiv.org/pdf/2507.01579) | Live three-month, 3.6-GW hybrid portfolio. Forecast pinball significantly predicted revenue (`p<0.001`), yet strategic bids added over `£500k` versus bidding `q50`; perfect deterministic forecasting yielded `£92.0m` versus `£105.2m` for perfect decisions. | Quadratic market revenue with price-spread information, not a fixed band reward; it establishes that decision quality matters but gives no BARAM effect size. |
| **[E04]** | Montero-Manso et al. (2020), *FFORMA*, [PDF](https://robjhyndman.com/papers/fforma.pdf) | M4's 100,000 series and a nine-method pool: hard model selection had `10%` larger OWA error than FFORMA and equal averaging `14%` larger; about `40%` of series received near-equal weights and `60%` had a dominant method. | One meta-instance was an entire time series with series-level features/loss, not an autocorrelated hourly row; OWA is not BARAM utility. |
| **[E05]** | Gutiérrez et al. (2013), *Ordinal and nominal classification of wind speed from synoptic pressure patterns*, [PDF](https://helvia.uco.es/xmlui/bitstream/handle/10396/14327/EAAI.pdf?sequence=1&isAllowed=y) | Four ordered daily-mean wind-speed classes at five farms: ordinal EBC(SVM) accuracy `69.26%` versus nominal SVM `69.14%`; authors conclude ordering did not clearly outperform nominal methods despite rank-test differences. | Wind-speed reconstruction from pressure patterns, not day-ahead power density or settlement action; the absolute accuracy difference is `0.12 pp`. |
| **[E06]** | Henzi, Ziegel & Gneiting (2021), *Isotonic Distributional Regression*, [PDF](https://arxiv.org/pdf/1909.03725) | IDR is CRPS-optimal subject to a chosen partial order and has no tuning choice beyond that order. In 24-h ECMWF precipitation postprocessing it was competitive and calibrated, while HCLR generally had the best CRPS. | Precipitation/CRPS with 1,500–3,000 training days and NWP ensembles; no wind-power step-reward result and the partial order is a substantive modelling assumption. |
| **[E07]** | Schulz & Lerch (2022), *Machine learning methods for postprocessing ensemble forecasts of wind gusts*, [PDF](https://arxiv.org/pdf/2106.09512) | Six years, 175 stations, eight methods: CRPS `0.84` DRN/BQN, `0.86` HEN, `0.87` QRF, `0.98` IDR; nominal-90.48% interval coverage was `90.23%` HEN and `84.04%` IDR. | Wind gust, not power; ensemble postprocessing and CRPS reward calibration, whereas BARAM rewards a narrow action window. HEN is only representation-analogous to the 26-bin head. |
| **[E08]** | Brehmer & Gneiting (2021), *Scoring interval forecasts: equal-tailed, shortest, and modal interval*, [PDF](https://arxiv.org/pdf/2007.05709) | The midpoint of a fixed-width modal interval is elicitable with `-1{|a-y|<=c}` as a strictly consistent score; shortest intervals fail elicitability on relevant classes. | The result is theoretical and single-width; it supplies no estimator benchmark for BARAM's two bands, generation weighting, L1 term, or small chronological sample. |
| **[E09]** | Donti, Amos & Kolter (2017), *Task-based End-to-end Model Learning in Stochastic Optimization*, [PDF](https://proceedings.neurips.cc/paper/2017/file/3fc2c60b5782f641f76bcefc39fb2392-Paper.pdf) | On a real grid-scheduling task, task training reduced task loss `38.6%` versus an RMSE-trained stochastic model and `8.6%` versus cost-weighted RMSE; training used 7 years and testing the next 1.75 years. | Differentiation through a smooth stochastic program with a Gaussian neural predictor; their battery gains were variable/not statistically significant, and no discontinuous band action was tested. |
| **[E10]** | Elmachtoub & Grigas (2022), *Smart “Predict, then Optimize”*, [PDF](https://par.nsf.gov/servlets/purl/10339524) | SPO+ is a convex surrogate for decision regret, with consistency results under stated conditions; it handles polyhedral, convex, and mixed-integer feasible sets with a **linear objective**, tested on shortest path and portfolio optimisation. | No time-series wind benchmark; BARAM must first be represented as a finite one-hot action/cost-vector problem, and finite-sample performance is unknown. |
| **[E11]** | Cai et al. (2019), *Instance-based transfer learning embedded GBDT*, [HTML](https://www.mdpi.com/1996-1073/12/1/159) | On GEFCom2014 zones, Table 9 reports `0.54%–2.40%` quantile-score gains over DL-GBDT (mean `1.46%`); Table 11 reports `4.74%` at 5% target data. However, Table 10's all-task columns (`DL 0.0368`, `IBT 0.0374`) are reversed in Table 11 at 100%, an internal inconsistency in the primary paper. | Cross-zone proper quantile score, tuned target/source weights, and internally inconsistent aggregate tables; evidence licenses only a cheap falsification, not an expected gain. |
| **[E12]** | Muñoz, Pinson & Kazempour (2023), *Online Decision Making for Trading Wind Energy*, [PDF](https://backend.orbit.dtu.dk/ws/files/328300831/s10287_023_00462_2.pdf) | On 43,200 hourly cases from 2016–2021, online newsvendor learning cut deviation cost `38.6%` versus forecast-only and added `7.6%` versus rolling LP, with 179-s execution time. | It is an approximately 1-h task using the last three realised-power lags, recent penalties, online label updates, and a 33-combination tuning grid; these inputs/actions are unavailable or forbidden at BARAM D-1 basis time. |
| **[E13]** | Kuncheva & Whitaker (2003), *Measures of Diversity in Classifier Ensembles*, [author PDF](https://www.lucykuncheva.co.uk/papers/lkml.pdf) | Defines Q, binary-output correlation, disagreement and double-fault from the 2x2 **correct/incorrect** table. Ten measures were highly related, but no definitive connection to ensemble accuracy was found on real problems. | Majority-vote classification, not weighted band utility; correctness-event diversity is the right measurement space but is not itself a selection algorithm. |
| **[E14]** | Wood et al. (2023), *A Unified Theory of Diversity in Ensemble Learning*, [PDF](https://jmlr.org/papers/volume24/23-0041/23-0041.pdf) | The loss defines the centroid combiner; diversity must be traded against bias/variance rather than maximised. For 0/1 loss, the diversity effect is necessarily label-dependent. | Theory does not identify a learnable gate, and BARAM combines L1 with two weighted indicator rewards rather than a single standard loss. |

## 5. Targeted local S15/S16 evidence register

| Tag | Saved artifact | Material result and exact scope |
|---|---|---|
| **[L01]** | `research/nodes/S15-N6_draw_verdict.json` | Six D seeds: mean blend `0.634573`, SD `0.000849`; selected/deployed development champion `0.636184`. |
| **[L02]** | `research/nodes/S15-N12_density_diag.json` | D versus composed density: point `1-NMAE 0.862828 -> 0.865320`, but point FICR `0.366175 -> 0.348776`, band mass `0.759075 -> 0.566984`, entropy `1.404 -> 1.941`. |
| **[L03]** | `research/nodes/S15-N14_three_member.json` | Adding composed member to D+DEPAVG gave Total `0.634008` versus `0.636184`; fold-outside weights were unstable and the paired point delta was `-0.002176`. |
| **[L04]** | `research/nodes/S16-N2_verify_oracle.json` | Base-four hit-event correlations are much lower than continuous-error correlations; any-hit/oracle/champion figures in §3 reproduce from this file. |
| **[L05]** | `research/nodes/S16-N3_fforma.json` | Exact-loss FFORMA gate negative despite high top-1; all recorded hard trigger rules remained below champion. |
| **[L06]** | `research/nodes/S16-N4_hit_gate.json` | Four binary band-hit models produced only a statistical tie at their best saved margin rule. |
| **[L07]** | `research/nodes/S16-N5_pool_expand.json` | Pool expansion inflates union coverage and outcome-aware oracle, reaching `0.807182` at 14 members. |
| **[L08]** | `research/nodes/S16-N6_pool14_gate.json` | The corresponding 14-member learned gate did not improve Total. |
| **[L09]** | `research/nodes/S16-N7_idr.json` | IDR-composed blend `0.635089`; IDR-D blend `0.635002`; both below champion, with band-hit correlation to D `0.500` and `0.689` respectively. |
| **[L10]** | `research/nodes/S16-N8_shift.json` | Moving D's sharp density toward the more accurate composed location monotonically damaged the blend after the smallest shift: `0.636184 -> 0.635506` at shift `0.25`, while band mass stayed near `0.759`. |
| **[L11]** | `research/nodes/S16-N9_bandsel.json` | A band-hit **feature-selector** (not a direct modal-action learner) scored `0.632239`, delta `-0.003945`; selected-feature overlap with the MAE selector was `0.6163`. |
| **[L12]** | `research/nodes/S15-N8_ablate.json`, `s15_n7_compose.py`, `s15_n4_seed_ensemble_D.py` | The targeted lineages already fit pooled cross-group models with group indicators; the composed pipeline's explicit equal-group-mass `D4` weighting reduced mean Total by `0.000563` (`FULL 0.605359`, `no_D4 0.605922`). |

## 6. Findings by requested axis

### 6.1 Mixture of experts, member selection, FFORMA and gating

FFORMA's decisive ablation is soft averaging versus two endpoints on M4: hard selection costs 10% and equal averaging costs 14% relative error, with the same features, pool, and learner. This is strong evidence that a gate should optimise member losses and retain soft weights rather than classify the winning member. **[E04]**

FFORMA learns one weight vector per whole series across 100,000 M4 series. **[E04]**

The completed S16 implementation instead learned per-hour weights from a few temporally dependent quarters, so the meta-instance is not the paper's unit. **[L05]**

Nevertheless, S16 already implemented the stronger local interpretation—exact per-row official loss, soft weights, fold-outside training—and lost `0.003016`; confidence filtering merely approached the champion while firing on easy/agreement rows. **[L05]**

**Verdict:** close further per-row FFORMA, hard-selection, confidence-trigger, and binary-hit-gate variants. A different learner, threshold, or pool size changes cosmetics rather than the failed statistical unit/identification problem. **[R02]**

### 6.2 Stacking versus feature concatenation

HEFTCom's clean comparator is stacked source-specific CatBoost `28.5` versus one concatenated no-target CatBoost `28.6` on concurrent 2023 data; the large gain is against the organiser reference `38.0`, not against concatenation. **[E02]**

The winner's architecture trained one model per NWP source and then a linear quantile meta-model, so its evidence supports source-specific handling and robustness/fallbacks more strongly than it supports stacking as an intrinsically superior combiner. **[E02]**

**Verdict:** do not open a generic “stack instead of concatenate” node. Only a source-specific preprocessing hypothesis with a named physical difference would be new, and this lane found no primary benchmark effect large enough to override the local external-NWP closure. **[R03]**

### 6.3 Distributional, ordinal, IDR and modal-interval learning

GEFCom2014 establishes quantile GBMs/trees as hard-to-beat probabilistic baselines, but it optimises pinball loss rather than the action reward. **[E01]**

On the closest systematic distribution-head benchmark found, HEN's CRPS `0.86` was only `0.02` behind DRN/BQN and better than IDR `0.98`; this does not identify the 26-bin representation as BARAM's bottleneck. **[E07]**

The wind ordinal study found only `0.12 pp` nominal accuracy difference and explicitly declined to claim clear superiority for ordered classifiers. **[E05]**

IDR supplies a tuning-free CRPS benchmark under an explicitly chosen order restriction. **[E06]**

The local EasyUQ/IDR arms were already diverse in band-hit space and still blended below champion. **[L09]**

Modal-interval theory legitimises direct fixed-band scoring, but it is not evidence that a smoothed modal loss will generalise on this dataset. **[E08]**

The local negative in `S16-N9` rejects only band-based feature selection, not the still-unbuilt exact full-cost decision learner. **[L11]**

The composed density improved point `1-NMAE` while losing FICR and band mass, showing why another accuracy-oriented calibration head is risky. **[L02]**

A separate sharpness-preserving location-shift experiment also hurt the blend. **[L10]**

**Verdict:** close ordinal-head, IDR/EasyUQ, generic calibration, and mean-shift variants. Preserve one genuinely different possibility: learn the complete action-cost vector with a regret loss, not another approximation to `p(Y|X)`. **[R04]**

### 6.4 Decision-focused and prescriptive learning

Donti et al. show that a deliberately worse predictive model can be better for an energy-system task, with a 38.6% scheduling-loss reduction under their setting. **[E09]**

HEFTCom independently shows that decision logic has comparable economic leverage to forecast skill, but its prices and quadratic revenue make the effect non-portable to BARAM. **[E03]**

SPO+ is the closest method whose scope can be mapped exactly: once each candidate action is a one-hot decision and its realised BARAM cost is one component of `c(y)`, the downstream objective is linear in that one-hot decision. **[D02]**

S16-N3 selected/weighted four member actions through exact member loss, and `53.6%` top-1 still lost Total. **[L05]**

S16-N4 instead predicted per-member hit probability and produced only a tie. **[L06]**

SPO+ is materially different because it trains against **decision regret and mistake magnitude**, which is the unresolved failure mode of those gates. **[D02]**

### 6.5 Domain and temporal adaptation

The best directly relevant transfer paper reports small cross-zone quantile gains and a larger low-data gain, but its own Tables 10 and 11 reverse the 100%-data DL/IBT aggregate values. **[E11]**

The strongest online wind-trading result is inadmissible here because it uses recent realised generation, penalties, and continuous online updates at an approximately 1-h horizon. **[E12]**

The targeted S15 lineage already pools groups with group indicators, and its explicit equal-group-mass weighting ablation was negative by `0.000563`. **[L12]**

**Verdict:** no online/test-label adaptation, deep meta-learning programme, or equal-mass donor-pooling rerun. Adaptive source weights would add tuning degrees of freedom on evidence whose primary aggregate tables are internally inconsistent. **[R05]**

### 6.6 Ensemble diversity as hit-event complementarity

For a step reward, define each member's binary correctness output as `H_k^6=1{|a_k-y|<=0.06c}` and separately `H_k^8=1{|a_k-y|<=0.08c}`; Q, correlation, disagreement and double-fault should be computed on those indicators, not continuous residuals. **[E13]**

Locally, the base-four continuous-error correlations span roughly `0.903–0.987`, while band-hit correlations span roughly `0.411–0.856`; continuous residual correlation therefore materially understates the action pool's event complementarity. **[L04]**

The classifier-ensemble literature finds no definitive diversity-to-accuracy mapping, so low event correlation is not sufficient for gain. **[E13]**

Locally, higher 14-member union/oracle coverage also failed to improve the learned gate. **[L08]**

Theoretical diversity under indicator loss is label-dependent, so measuring it requires realised historical outcomes and cannot become a basis-time selector by itself. **[E14]**

**Standing measurement rule:** every future member receipt should report solo Total, u6/u8 hit rates, pairwise Q/correlation/double-fault on `H^6` and `H^8`, and strict-held-out marginal union hits; continuous-error correlation may remain a secondary NMAE diagnostic only. **[R06]**

## 7. Operations explicitly closed by this audit

- Another per-row FFORMA, hard selector, confidence trigger, or hit-probability gate over the same member pool is closed. **[L05]**
- Enlarging the pool merely because it increases the outcome-aware oracle is closed; 14 members already demonstrated oracle inflation without learned gain. **[L08]**
- IDR/EasyUQ as another blend arm is closed on the tested representations. **[L09]**
- Mean/location correction, even with sharpness approximately preserved, is closed by the shift experiment. **[L10]**
- Band-hit feature selection is closed; this does **not** close an exact full-cost regret learner. **[L11]**
- Generic stacking in place of concatenation is not supported by the direct HEFTCom comparison. **[E02]**
- Ordinal relabelling of the 26 bins is too weakly supported to justify a node. **[E05]**
- Online adaptation using test-period outcomes or market observations is outside the basis-time/data boundary. **[E12]**
- Equal-mass cross-group donor pooling is already represented in the targeted S15 lineage and its explicit weighting ablation was negative. **[L12]**
- Diversity regularisation or member admission based only on low correlation is closed; diversity is a trade-off and a diagnostic, not an objective by itself. **[E14]**

## 8. Surviving non-duplicate candidate and smallest strict-chronology test

### Candidate 1 — `COST5-SPO+`: regret-aware selection over four members plus the champion

**Operation.** On each historical row, compute the exact negative official per-row utility for the same four aligned member actions used by `S16-N2` and for the incumbent champion; this yields a fully observed five-component cost vector. Train one deterministic linear multi-output cost predictor with SPO+ on the already-frozen S16 gate feature map, and deploy the minimum-predicted-cost action. **[D02]**

**Why it is not a duplicate.** The completed FFORMA arm optimised soft member weights and lost despite `0.535894` top-1. **[L05]**

The completed hit gate optimised binary correctness labels rather than action regret. **[L06]**

`COST5-SPO+` instead penalises the regret magnitude of the action actually chosen. **[D02]**

**Smallest discriminating experiment.** Freeze the five actions, row keys, feature map, group normalisers, SPO+ formulation, and one regularisation value before scoring. Use expanding issuance-day chronology only: train on dates strictly earlier than Q2 and test Q2; extend through Q2 and test Q3; extend through Q3 and test Q4. Do not search triggers, shrinkage, action subsets, or hyperparameters. **[R07]**

**Decision.** PASS only if aligned aggregate delta is at least `+0.001` and the existing 7-day block arbiter gives `P(delta>0)>=0.90`, with no quarter below `-0.001`; otherwise close regret-aware selection on the existing member-plus-champion representation. Report conditional regret and u6/u8 hits in addition to Total. **[R08]**

**Only after a PASS.** A second, separately predeclared experiment may replace the four member actions with the existing full action grid while retaining the champion reference; doing both at once would confound learnability with oracle expansion. **[R09]**

## 9. Final priority

Run `COST5-SPO+` because it directly attacks the measured learnability failure without inventing new data or members. No parameter sweeps or other variants are recommended from this literature set. **[R10]**

## 10. Compliance close

The only repository write made by this lane is `research/rwa_model_decision_research.md`; all web material was read in memory, and no project model, score, lockbox, account, dependency, or dataset was touched. **[C03]**
