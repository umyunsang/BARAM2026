# S17-N18 step-reward lane — exact prescriptive partitioning is novel, but not learnable on the remaining chronology

**Node:** `S17-N18_POST_RWA_FRONTIER_RESEARCH_INTAKE / step_reward`  
**Retrieval date:** 2026-08-08 (bounded session)  
**Verdict:** **`CLOSE_STEP_REWARD_ESTIMATOR_AXIS_NO_STRICT_TEST`**. The only structurally nonduplicate method found is an exact task-cost *prescriptive tree/forest* that learns its partitions under the complete BARAM loss and solves a weighted sample-average action problem in each local neighbourhood; it fails the present learnability/chronology gate, and every low-complexity reduction of it collapses to a locally closed policy switch, member selector, or conditional offset. **[derived]**

## 0. Contract and evidence convention

This lane used **7 external primary papers/official software papers**, below the cap of 12, plus the named local records in §1; it performed 0 model fits, 0 score/component calls, 0 target/test/2024 reads, 0 rejected/quarantined-artifact reads, 0 installs, and 0 mutations other than this file. **[directly_supported]**

Tags mean: **[directly_supported]** = stated in a fetched primary source or named local record; **[derived]** = algebra or applicability inference shown here; **[near_match_only]** = real result under a materially different loss/sample/chronology; **[contradicts_premise]** = evidence directly defeats the proposed applicability; **[unverified]** = not established by the bounded source set. **[directly_supported]**

## 1. Named local evidence boundary

| ID | Local source | What is admissible from it | Tag |
|---|---|---|---|
| L1 | `research/rwa_eval_research.md`, §3.6 | Exact official formula, exact additive loss, population Bayes-action analogue, and issuance-day evaluation unit. | [directly_supported] |
| L2 | `research/rwa_model_decision_research.md`, §§2–8 | Direct hit gates, FFORMA/member selection, IDR, shifts, pool expansion, and band-feature selection were compared; the then-survivor was COST5-SPO+. | [directly_supported] |
| L3 | `reports/s17_n7_strict_action_reconstruction_receipt.json` | Strictly reconstructible development surface has 90/89/91 retained issuance days in Q2/Q3/Q4 and two chronological outer transitions. | [directly_supported] |
| L4 | `reports/s17_n9_cost5_recovery_receipt.json` | COST5-SPO+ Total `0.619583` versus strict champion `0.631483`, delta `-0.011899`; `1-NMAE` rose `0.003875` while FICR fell `0.027674`. | [directly_supported] |
| L5 | `reports/m271_cycle4_bandpolicy.md`; `reports/m271_cycle7_policygate.md` | Same-surface conditional-policy oracle was only `+0.001720` over the global optimum; 62 frozen policies yielded 0 gate passes. | [directly_supported] |
| L6 | `reports/m271_p4_consolidate.md`; `reports/m271_n5_metric_aligned_loss.md`; `reports/m271_n7_cdf_rejudge.md` | Smoothed metric-aligned fit, band target, quantile-CDF representation, and conditional decision-policy families are documentary negatives. | [directly_supported] |
| L7 | `reports/m277_conditional_calibration_receipt.json`; `reports/n401_cqr_decision_receipt.json`; `reports/m285_within_bin_gate_receipt.json` | Conditional calibration, CQR action construction, and within-bin postprocessing did not pass their saved gates. | [directly_supported] |
| L8 | `reports/s17_legacy_cycle_manifest.json` | The M271 cycles are explicitly `LEGACY_UNRECONSTRUCTABLE`; L5–L7 are mechanism/documentary context, not fresh atomic comparisons. | [directly_supported] |

Accordingly, the strict negative that carries the most weight here is L4; the older M271 records are used only to reject renamed duplicates, not to claim a new unbiased performance estimate. **[derived]**

## 2. External primary-source register (7 sources)

| ID | Primary source and URL | Verbatim locator used here | Applicability tag |
|---|---|---|---|
| E1 | Gneiting (2011), *Making and Evaluating Point Forecasts*, [primary preprint](https://arxiv.org/pdf/0912.0902) | “the optimal point forecast, namely the Bayes rule, `x̂ = arg min_x E_F S(x,Y)`”; the abstract warns that score and task must be matched. | [directly_supported] |
| E2 | Brehmer & Gneiting (2021), *Scoring interval forecasts*, [primary preprint](https://arxiv.org/pdf/2007.05709) | Eq. (9): `MI_c(F)=argmax_[a,b]{F(b)-F(a−):b-a≤2c}`; Eq. (11): `S(x,y)=-1{x-c≤y≤x+c}`; Theorem 3.10 says this is essentially the sole strictly consistent score for that midpoint functional. | [directly_supported] |
| E3 | Bertsimas & Kallus (2020), *From Predictive to Prescriptive Analytics*, [MIT author manuscript](https://dspace.mit.edu/bitstream/handle/1721.1/133675/1402.5481.pdf?sequence=2&isAllowed=y) | Eq. (3): `ẑ_N(x) ∈ argmin_z Σ_i w_{N,i}(x)c(z;y_i)`; its dependence extension is for stationary mixing processes, not arbitrary chronological drift. | [directly_supported] |
| E4 | Kallus & Mao (2023), *Stochastic Optimization Forests*, [primary full text](https://par.nsf.gov/servlets/purl/10406742) | The abstract says splits “directly optimize the downstream decision quality”; Eq. (3) is a forest-weighted SAA. Theorem 1 requires twice continuously differentiable population objectives, a unique minimizer, and a positive-definite Hessian for its fast split approximation. | [directly_supported] |
| E5 | Stratigakos et al. (2022), *Prescriptive Trees … Trading of Renewable Energy*, [HAL author manuscript](https://hal.science/hal-03330017v3/document) | Eq. (7) chooses a split by `min_{j,s}[min_{z_l}Σ_{R_l}c(z_l;y_i)+min_{z_r}Σ_{R_r}c(z_r;y_i)]`; Eq. (5) deploys weighted SAA. Models used one year of 2019 and tested the first four months of 2020, 20 features, while “implicitly assum[ing] stationarity.” | [directly_supported] |
| E6 | Kitagawa & Tetenov (2018), *Who Should Be Treated?*, [author manuscript](https://tetenov.com/Who_should_be_treated_Kitagawa_Tetenov.pdf) | Theorem 2.1 bounds EWM regret by a universal constant times `sqrt(v/n)` for VC dimension `v`; the authors state the bound rises with class complexity and hence fixed-`n` overfitting/regret. | [directly_supported] |
| E7 | Sverdrup et al. (2020), *policytree*, [JOSS software paper](https://joss.theoj.org/papers/10.21105/joss.02232.pdf) | Given an `N×D` reward matrix, the software performs globally optimal weighted search over depth-`k` trees; exact search costs `O(P^k N^k(log N + D))`, with depth 1 `O(NPD+NP log N)`. | [directly_supported] |

E5 is the closest empirical source, but its trading loss is convex-market-cost/prediction-error based rather than BARAM's two discontinuous bands, and its offline one-year/forward-four-month design does not measure a 90-day expanding fit under regime change. **[near_match_only]**

## 3. Exact population action

For group `g`, let capacity be `C_g`, eligibility be `V=1{Y≥0.1C_g}`, and `μ_g=E[Y|V=1,g]`. Define **exactly**

\[
u(e)=4\mathbf 1(e\le .06)+3\mathbf 1(.06<e\le .08)
     =\mathbf 1(e\le .06)+3\mathbf 1(e\le .08).
\]

This is the official two-tier utility decomposition. **[directly_supported]**

Up to the positive group constant `1/6`, maximizing population Total is equivalent to minimizing

\[
R_g(a;x)=E\!\left[V\left\{
 { |a-Y|\over C_g}-{Y\over4\mu_g}
 \left(\mathbf 1\{|a-Y|\le .06C_g\}
 +3\mathbf 1\{|a-Y|\le .08C_g\}\right)
\right\}\mid X=x,g\right].
\]

Therefore

\[
a_g^*(x)\in\arg\min_{a\in[0,C_g]}R_g(a;x).
\]

The indicator form is equivalent to conditioning on eligibility because `P(V=1|X=x,g)` is positive and does not depend on the action; `μ_g` must be estimated from past-only eligible training observations. **[derived]**

A useful exact representation sets

\[
A_x(a)=E[V|a-Y|\mid X=x,g],\qquad
H_x(t)=E[VY\mathbf1\{Y\le t\}\mid X=x,g].
\]

Then, with `W_h(a)=H_x(a+hC_g)-H_x((a-hC_g)^-)`,

\[
R_g(a;x)={A_x(a)\over C_g}
 -{W_{.06}(a)+3W_{.08}(a)\over4\mu_g}.
\]

Thus the sufficient conditional objects are one absolute-moment curve and two **generation-size-weighted sliding-window** curves; a raw conditional mode, one unweighted hit probability, or one modal interval is not the population action. **[derived]**

For any randomized action law `π(da|x)`, conditional risk is `∫R_g(a;x)π(da|x)≥inf_a R_g(a;x)`; randomization/dithering cannot improve the population optimum. **[derived]**

The official objective is additive over rows once its population group constants are fixed and has no cross-hour feasibility constraint, so a joint 24-hour decision rule cannot improve on the componentwise Bayes actions merely by coupling actions; sequence features can help only by estimating the same marginal conditional risks better. **[derived]**

For a finite frozen action set `A={a_1,…,a_D}`, observing `Y_i` reveals the entire reward/cost row for all actions, so BARAM is a **full-information cost-sensitive policy problem**, not a bandit or missing-counterfactual problem. **[derived]**

## 4. The only structurally new estimator found

An exact task-cost prescriptive partition would learn tree splits using

\[
\widehat C(R_L,R_R)=
 \min_{a\in A}\sum_{i:X_i\in R_L}\ell_g(a,Y_i;\widehat\mu_g)
 +\min_{a\in A}\sum_{i:X_i\in R_R}\ell_g(a,Y_i;\widehat\mu_g),
\]

where `ℓ_g` is the bracketed exact loss in §3 and all denominators use past-only data; deployment uses forest/local-neighbour weights in the E3/E5 form and re-enumerates the same frozen action set. **[derived]**

This is genuinely different **only** when raw basis-time features determine the partition and the leaf/neighbour empirical law determines a fresh action: the split itself is chosen for complete BARAM decision cost rather than prediction error, band probability, a smooth surrogate, or which saved member won. **[derived]**

Kallus–Mao's fast perturbative split cannot be promoted from this evidence set because this lane has no evidence for its required unique minimizer and positive-definite Hessian on the BARAM population risk. **[unverified]**

The two-window risk can be nonconvex or set-valued, while smoothing its jumps would return to the already-negative smoothed metric-loss family. **[derived]**

The exact E5/E7 enumeration avoids derivatives and can optimize a discontinuous reward matrix, so discontinuity is computationally manageable on a finite grid. **[directly_supported]**

It nevertheless supplies no new basis-time information: it is another estimator of the exact conditional risk in §3, and its possible advantage would have to come solely from a better low-sample partition of `X`. **[derived]**

## 5. Nonduplication audit against repository attempts

| Proposed operation | Relation to the exact action | Repository comparison | Decision | Tag |
|---|---|---|---|---|
| Predict a conditional CDF/density, then enumerate `R_g(a;x)` | Plug-in estimate of all three curves in §3. | Current 26/46-bin Bayes decision, quantile CDF, IDR/QRF/CQR, calibration, and pooled-density Bayes action already cover this operation class. | Duplicate; estimator-family changes alone lack a surviving local premise. | [directly_supported] |
| Predict `P(|a-Y|≤hC|X,a)` directly | Estimates the window terms, with extra work still needed for `Y` weighting and MAE. | Direct hit estimation/hit gates and band-feature selection were closed or tied in L2. | Duplicate estimand, not a new action. | [directly_supported] |
| Modal-interval regression/postprocessing | Maximizes one `W_h` only. | Modal-window/postprocessing and band-target families were tried; E2 itself does not cover the combined objective. | Incomplete population target and locally closed. | [derived] |
| COST5-SPO+ | Learns a five-action complete cost vector with a linear regret surrogate. | Strict L4 result is `-0.011899`; both evaluated quarters were negative. | Refuted on that representation. | [directly_supported] |
| Exact shallow reward tree over the same saved actions | Replaces SPO+ by exact EWM but still selects members. | Member selection, hard/soft gates, and confidence fallback are closed in L2. | Renaming member selection. | [derived] |
| Tree leaves choose a `(T,G)` policy or incumbent offset | Conditional postprocessing. | L5's band-conditional oracle/frozen policy family and L7's shifts/calibration cover the smallest such tree. | Renaming conditional policy/offset selection. | [derived] |
| Deep exact prescriptive tree/forest from raw `X` to a fresh action | Learns task-aware partitions and weighted empirical conditional laws. | No exact duplicate found in the named local records. | Structurally novel, but fails §6 learnability. | [derived] |

The apparent escape—use the full action grid rather than COST5's five actions—changes the action set and learner simultaneously after the smaller action-selection problem failed, while also increasing `D` and selection complexity; it is not a smallest discriminating follow-up. **[derived]**

## 6. Learnability and chronology audit

The strict outer procedure has only two learnable transitions: Q3 can use 90 retained Q2 issuance days, and Q4 can use 90+89=179 prior issuance days; the 91 Q4 days are assessment only. **[derived]**

The 6,480/12,888 hourly-group rows implied by those day counts are not 6,480/12,888 independent policy examples: all 72 group-hour cells are issued together and the local evaluation contract treats the full issuance day as the dependence block. **[derived]**

E5 instead trained on one full calendar year and evaluated four later months, used 20 features, and explicitly said its offline analysis implicitly assumes stationarity; its separate hyperparameter study randomly sampled `n=1000` observations, repeated ten times, and settled on `B=50`, `K=3d_x/4`, `n_min=10`. **[directly_supported]**

Those defaults are not a frozen small-`n` recipe for BARAM: `B`, feature-subsampling `K`, minimum leaf size, tree depth, action grid, honesty split, and day-block handling would all be new degrees of freedom, while no untouched forward surface remains for selecting them. **[derived]**

E6's `O(sqrt(v/n))` regret result confirms the direction of the problem—richer tree classes cost more samples—but assumes IID bounded welfare data and supplies neither a BARAM-scale constant nor evidence that `+0.001` Total is detectable with 90/179 dependent, regime-changing days. **[near_match_only]**

E3's non-IID asymptotics do not rescue the chronology: its stated extension requires a stationary mixing sequence, whereas the repository's strict protocol exists precisely because quarter-to-quarter regime transfer cannot be assumed stationary. **[contradicts_premise]**

A depth-0 prescription is the already-searched global policy; a depth-1 rule with saved members, `(T,G)`, or offsets is one of the closed operations in §5; a raw-feature tree deep enough to map the full output range is the first nonduplicate form, but it is also the form whose partition search cannot be frozen or selected honestly from 90/179 days without another validation surface. **[derived]**

No fetched primary source reports the BARAM two-band, generation-weighted objective under comparable strict 90-day expanding chronology, and no source gives a defensible fixed prescriptive-tree specification for this sample. **[unverified]**

## 7. Decision, falsifiers, and stopping rule

**Decision: close the discontinuous-reward estimator/decision axis; propose zero executable tests.** The population decision is already fully characterized, the finite-action problem already exposes all costs, and the sole new algorithmic operation—task-aware raw-feature partitioning—cannot pass the present small-`n`/chronology prerequisite without tuning a new high-complexity family on reused development quarters. **[derived]**

A “one stump” experiment is deliberately **not** proposed: making it small enough to learn forces its actions to be saved members, fixed policies, or offsets and therefore renames a closed method; making it map raw features to fresh absolute actions requires more leaves and ceases to be the smallest strict test. **[derived]**

The verdict is falsified only by one of the following conditions. **[derived]**

1. A materially longer basis-safe history plus an untouched strictly later assessment period becomes available, allowing the complete action set, raw feature set, day-level honesty rule, depth, leaf-day minimum, and seed to be frozen without Q3/Q4 selection. **[derived]**
2. A primary result demonstrates exact task-cost prescriptive partitioning under a comparable two-band weighted reward and comparable dependent expanding chronology with a predeclared small policy class. **[unverified]**
3. A repository-native audit finds a genuinely untested low-VC raw-feature-to-fresh-action class that is not algebraically a member selector, `(T,G)` router, conditional shift, CDF plug-in, or saved-member remix, together with a fresh evaluation surface. **[derived]**
4. The strict local record for COST5 or the asserted direct-hit/postprocessing closures is invalidated by a provenance error; this would reopen the prior method first, not justify a prescriptive forest automatically. **[derived]**

Until a falsifier occurs, further “decision-focused,” “policy-tree,” “modal,” “distributionally robust,” or “safe-switch” labels do not identify a new estimand or new information source and should not consume another comparison. **[derived]**
