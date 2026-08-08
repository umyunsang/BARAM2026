# V5-VALIDATION — honest evaluation after adaptive reuse of dependent time-series

**Lane:** `V5-VALIDATION`  
**Output contract:** research only; no model fit, score, lockbox, Dacon action, dependency change, raw/external data acquisition, or source edit.  
**Research bound:** 10 primary methodological papers/statistical sources; all published before 2026-07-05; web verification performed 2026-08-09 KST.  
**Authority verified before research:** `SK_v5.md` SHA-256 `c2c813475ab5f0a741e6adbceab196d5b38447f2aeb02f9716a8f5be8973c820`; approval receipt SHA-256 `e827b324a23913346598d274f2ded496ffc3134243de5636033ee6a3ac387173`; Wave-A manifest SHA-256 `4690826f4f2f629220c085ffc87883c1ec2fa8990f7f92430483ff4f975ae312`.

## Verdict

**`NO_FRESH_CONFIRMATION_AVAILABLE / RETROSPECTIVE_SELECTION_ADJUSTED_AUDIT_ONLY`.** A nested, strictly chronological replay is still necessary because it removes *within-protocol* tuning leakage and evaluates the selection procedure rather than its hindsight winner. It cannot make the repeatedly exposed 2023 quarters fresh again. On the current surface, the strongest honest result is therefore a reproducible, chronology-correct **retrospective corroboration conditional on the recorded candidate family and dependence assumptions**. A level called “confirmatory” must remain `CONFIRMATION_PENDING` until outcomes not previously exposed to the analyst or pipeline arrive under a predeclared protocol.

Two especially important distinctions follow.

1. Re-running the same deterministic strict evaluation twice proves computational reproduction, not two statistically independent confirmations.
2. White's Reality Check or a Model Confidence Set can adjust a *specified, recorded family* of contemporaneous loss series. Neither method reconstructs unknown historical trials, erases adaptive human feedback, proves the best model outside that family, or retroactively restores an untouched holdout.

Throughout this report:

- **[SF] source fact** means a claim stated or proved by one of S01–S10.
- **[MH] migration hypothesis** means a proposed BARAM/DS@v5 design assembled from those facts; it is not a paper-prescribed local recipe.
- **[LE] local evidence needed** means a check DS@v5 must run later. No such check or score was run in this lane.

## Local foundation and claim target

These are local authority facts, not additions to the ten-source research ledger.

- **L01:** `SK@v5` requires 72-cell issuance-day atomicity, strictly preceding fit/selection/label availability, exact Total plus components and stability views, an append-only comparison index, fixed-policy reproduction, and no claim of an independent holdout where none remains.
- **L02:** the task states that the same 2023 quarters have been repeatedly exposed during adaptive development.
- **L03:** `AGENTS.md` states that the 2024 lockbox has already been consumed twice and no independent validation surface remains.
- **L04:** the skeleton's operational `SUCCESS_EXIT` is two reproduced strict local runs at the target. That is an operational/reproducibility rule; it must not be translated into “two independent statistical replications.”
- **L05:** no fresh lockbox is available in this lane, and no new data or online feedback may be acquired.

The target of inference must be named before any comparison:

1. **Historical-surface estimand:** performance on the exact exposed outer-origin rows. This can be computed/reproduced, but its inferential scope is retrospective.
2. **Selection-procedure estimand:** performance of a fully specified rule that chooses preprocessing, features, model, hyperparameters, calibration, blend, and action policy using only information available before each origin. Nested prequential evaluation can target this rule.
3. **Fixed-final-pipeline estimand:** performance of one final pipeline selected after all development. Nested outer results for a varying per-origin selector do not automatically estimate this different object.
4. **Future-deployment estimand:** performance on new issuance days under future regimes. No current historical rearrangement supplies an independent estimate of this target after repeated exposure.

## Source ledger — exactly 10 primary methodological sources

| ID | Primary source (publication date) | Direct source fact used | Assumptions and scope that matter here | Exact locator |
|---|---|---|---|---|
| **S01** | Dawid, A. P., “Present Position and Potential Developments: Some Personal Views: Statistical Theory: The Prequential Approach,” *JRSS A* 147(2), 278–290 (1984) | [SF] The prequential position evaluates a sequence of forecasts through the observations that subsequently materialize, rather than just an in-sample fitted object. | Foundational principle, not a ready-made nested time-series resampling theorem and not a repair for already exposed outcomes. | DOI [10.2307/2981683](https://doi.org/10.2307/2981683); [publisher record](https://academic.oup.com/jrsssa/article/147/2/278/7106293). |
| **S02** | Tashman, L. J., “Out-of-sample tests of forecasting accuracy: an analysis and review,” *International Journal of Forecasting* 16(4), 437–450 (2000-10) | [SF] Out-of-sample design choices must be explicit. For an individual series, rolling origins, coefficient recalibration, and multiple test periods can improve efficiency/reliability and reduce sensitivity to one origin or business-cycle phase. | Multiple origins improve coverage; they do not make dependent origins independent. The paper does not address years of adaptive analyst reuse. | DOI [10.1016/S0169-2070(00)00065-0](https://doi.org/10.1016/S0169-2070(00)00065-0); [publisher abstract and section record](https://www.sciencedirect.com/science/article/abs/pii/S0169207000000650). |
| **S03** | Cawley, G. C. & Talbot, N. L. C., “On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation,” *JMLR* 11(70), 2079–2107 (2010) | [SF] Finite-sample noise in a selection criterion can itself be overfit; observed degradation can be comparable to algorithm differences. Model selection must be rerun inside evaluation, and the object compared should be the learning-algorithm/model-selection-procedure combination. Nested/double cross-validation is proposed as a rigorous route. | The empirical work is not a dependent nonstationary time-series theorem. Ordered nested origins are therefore a migration, not a directly prescribed local recipe. | [Official JMLR article and PDF](https://jmlr.org/papers/v11/cawley10a.html). |
| **S04** | Diebold, F. X. & Mariano, R. S., “Comparing Predictive Accuracy,” *Journal of Business & Economic Statistics* 13(3), 253–263 (1995-07) | [SF] Pairwise equal predictive accuracy can be tested through a loss-differential series under general loss functions and correlated/non-Gaussian errors. The large-sample argument assumes the loss differential is covariance-stationary and short-memory and uses its long-run variance. | A generic DM-style p-value is not licensed when the local differential is nonstationary, long-memory, too short, or selected after seeing the same outcomes. The paper's small-sample simulations are not local guarantees. | DOI [10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599); [publisher abstract](https://www.tandfonline.com/doi/abs/10.1080/07350015.1995.10524599). |
| **S05** | Giacomini, R. & White, H., “Tests of Conditional Predictive Ability,” *Econometrica* 74(6), 1545–1578 (2006-11) | [SF] The forecasting **method** includes model and estimation procedure/window. Their framework preserves estimation uncertainty, accommodates nested/non-nested and possibly misspecified methods, and permits conditional forecast selection. Its main asymptotics use finite-memory rolling/fixed estimation, mixing and moment/positive-definiteness conditions, with the number of evaluation observations growing. | This is the closest nonstationary/heterogeneous source, but it is not a license for arbitrary expanding-window local tests, data-dependent regime instruments, very small samples, or adaptive reuse of the evaluation outcomes. | DOI [10.1111/j.1468-0262.2006.00718.x](https://doi.org/10.1111/j.1468-0262.2006.00718.x); [Econometric Society record](https://www.econometricsociety.org/publications/econometrica/2006/11/01/tests-conditional-predictive-ability). |
| **S06** | Politis, D. N. & Romano, J. P., “The Stationary Bootstrap,” *JASA* 89(428), 1303–1313 (1994-12) | [SF] The stationary bootstrap resamples geometrically distributed random-length blocks and develops consistency/weak convergence for statistics from weakly dependent **stationary** observations. | Blocking preserves local dependence only under its conditions. The word “stationary” is substantive: this procedure alone does not solve regime drift or analyst adaptivity. | DOI [10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870); [publisher abstract](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870). |
| **S07** | White, H., “A Reality Check for Data Snooping,” *Econometrica* 68(5), 1097–1126 (2000-09) | [SF] Reusing one time-series history for inference/model selection creates data-snooping risk. The Reality Check tests whether the best model encountered in a specification search has predictive superiority over a benchmark and implements dependence-aware resampling, including the stationary bootstrap. | It is conditional on the candidate/performance matrix supplied and asymptotic regularity. It cannot account for omitted or unknown searches. Recomputing a current-family p-value after each addition is not, by itself, proof of optional-stopping validity over an adaptively inspected research program. | DOI [10.1111/1468-0262.00152](https://doi.org/10.1111/1468-0262.00152); [Econometric Society record](https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping). |
| **S08** | Hansen, P. R., Lunde, A. & Nason, J. M., “The Model Confidence Set,” *Econometrica* 79(2), 453–497 (2011-03) | [SF] MCS sequentially removes significantly inferior objects until equal predictive ability is not rejected; uninformative data honestly leave a large set. It is specific to the finite initial candidate set and silent about omitted objects. For its forecast-loss implementation, relative loss differentials require moments, strict stationarity, nonzero variance and alpha-mixing; individual loss levels may be nonstationary if their differences satisfy the conditions. | The coverage result is asymptotic and candidate-set-relative. Membership is not proof that every survivor is good; it may reflect low power. | DOI [10.3982/ECTA5771](https://doi.org/10.3982/ECTA5771); [Econometric Society record and supplement](https://www.econometricsociety.org/publications/econometrica/2011/03/01/model-confidence-set). |
| **S09** | Foster, D. P. & Stine, R. A., “α-Investing: a Procedure for Sequential Control of Expected False Discoveries,” *JRSS B* 70(2), 429–444 (2008) | [SF] Alpha-investing controls **mFDR**, a weaker ratio-of-expectations criterion than FDR. Tests need not be independent, but each incoming true-null test must retain its assigned conditional level given prior rejection outcomes. | This is not retroactive FWER control. A p-value derived from the same fully exposed quarters after candidate generation from their exact scores has not been shown conditionally valid. The local goal must decide whether mFDR is even the desired error criterion. | DOI [10.1111/j.1467-9868.2007.00643.x](https://doi.org/10.1111/j.1467-9868.2007.00643.x); [author/institutional copy](https://repository.upenn.edu/entities/publication/61b1d001-d6d9-4775-8145-d1a7e9a03ad9). |
| **S10** | Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O. & Roth, A., “The reusable holdout: Preserving validity in adaptive data analysis,” *Science* 349(6248), 636–638 (2015-08-07) | [SF] Ordinary fixed-procedure validity does not cover intrinsically adaptive analyses; the paper demonstrates a specially mediated, privacy-inspired mechanism for safely reusing a holdout while revealing controlled information. | This is not evidence that ordinary repeated release of exact local scores is safe. The source does not supply a retrospective undo operation or a directly validated dependent, nonstationary BARAM implementation. | DOI [10.1126/science.aaa9375](https://doi.org/10.1126/science.aaa9375); [official PubMed record/abstract](https://pubmed.ncbi.nlm.nih.gov/26250683/). |

## Tagged claim ledger

The tags have the exact SK@v5 meanings: `directly_supported`, `contradicts_premise`, `near_match_only`, and `insufficient`.

| claim_id | skeleton_field_or_local_gate | finding | evidence_tag | source_locator_and_date | scope_match | implication | proposed_delta |
|---|---|---|---|---|---|---|---|
| **V5V-C01** | evaluation hierarchy / chronology | Forecasts must be generated and scored in observation order, using only information available at each origin. | `directly_supported` | S01 (1984), S02 (2000) | Direct principle; local issuance mechanics still need implementation proof. | Random row CV is inadmissible as the primary strict evaluator. | Make `outer_origin_id`, `basis_time`, and availability cutoff mandatory. |
| **V5V-C02** | multiple origins / regime coverage | Rolling origins, recalibration and multiple test periods are preferable to a single terminal split for an individual series. | `directly_supported` | S02 (2000) | Direct design fact; independence is not claimed. | Use multiple chronological origins, but treat their losses as dependent. | Store origin-by-origin predictions and losses, not only a pooled score. |
| **V5V-C03** | hyperparameter and pipeline selection | Every material selection step must be repeated inside the training side of each outer origin; the evaluated object is the entire selection procedure. | `directly_supported` | S03 (2010) | Direct for selection bias; ordered time-series nesting is a migration. | Preprocessing, features, hyperparameters, blend and policy cannot be chosen on outer outcomes. | Add `selection_algorithm_hash` and a full inner-origin receipt. |
| **V5V-C04** | nested/prequential selection | “Nested rolling origin” is the appropriate local combination of S02 and S03. | `near_match_only` | S02 (2000) + S03 (2010) | No cited paper proves this exact BARAM design. | It is experiment-admissible only with chronology, overlap, availability and unit tests. | Label it `provisional_migration_hypothesis`, not provider-prescribed. |
| **V5V-C05** | no fresh holdout | A clean nested replay now cannot restore the independence lost when the same 2023 outcomes guided earlier development. | `contradicts_premise` | S07 (2000), S10 (2015), plus L02 | Direct adaptivity/data-snooping mechanism; local exposure history is asserted by the brief. | Call current outer results retrospective corroboration, never fresh confirmation. | Add `surface_exposure_status=EXPOSED` and claim linting. |
| **V5V-C06** | success/reproduction | Two deterministic reproductions on the same rows establish reproducibility, not two independent replications or a squared evidence gain. | `contradicts_premise` | S07 (2000), S10 (2015), plus L04 | Exact statistical interpretation; does not alter the skeleton's operational exit rule. | Separate operational success from statistical confirmation. | Add `reproduction_count` and distinct `independent_confirmation_count` (currently zero). |
| **V5V-C07** | pairwise forecast comparison | A paired loss-differential comparison may use long-run-variance logic only if the differential is sufficiently stationary/short-memory and the comparison was not selected on that same series. | `directly_supported` | S04 (1995) | Direct assumptions; local satisfaction unknown. | Naive row-level standard errors and ordinary paired t-tests are inadmissible. | Require assumption/status fields before inferential language. |
| **V5V-C08** | nonstationarity | Giacomini–White can compare forecasting methods under heterogeneous/misspecified settings with finite-memory estimation and mixing/moment conditions. | `directly_supported` | S05 (2006) | Close but not exact: local windows, sample size and instruments are unknown. | It is an optional conditional comparison, not a blanket nonstationarity cure. | Freeze estimation window and instruments as treatment DOF; otherwise stay descriptive. |
| **V5V-C09** | dependent bootstrap | Stationary/bootstrap blocks must operate on the dependent time unit and require a defensible stationary/weak-dependence approximation. | `directly_supported` | S06 (1994), S07 (2000), S08 (2011) | Method facts direct; BARAM day-block mapping is local. | Resample issuance-day bundles, never individual 72 cells. | Set `resampling_unit=issuance_day`; require block-length sensitivity. |
| **V5V-C10** | block/bootstrap under drift | Applying a stationary bootstrap to a visibly regime-varying differential does not make the result robust to nonstationarity. | `contradicts_premise` | S06 (1994), S08 (2011) | Direct mismatch of assumptions; actual local differential status unknown. | If assumptions fail, use regime-wise descriptive uncertainty and defer confirmatory p-values. | Add an `assumption_failure -> descriptive_only` transition. |
| **V5V-C11** | fixed-batch multiplicity | Reality Check addresses “best recorded candidate vs benchmark”; MCS returns a confidence set relative to a finite supplied family. | `directly_supported` | S07 (2000), S08 (2011) | Strong match for a frozen complete family with common rows/loss. | Include control and all screened candidates in one aligned loss matrix. | Add `comparison_family_id`, family manifest hash and candidate count. |
| **V5V-C12** | historical multiplicity | RC/MCS cannot adjust for unlogged failures, analyst ideas abandoned after seeing results, or models omitted from the loss matrix. | `contradicts_premise` | S07 (2000), S08 (2011) | Exact candidate-set limitation. | A historical family with unknown size cannot support a globally adjusted superiority claim. | Add `historical_unknown_trials`; if true, restrict claim to recorded family. |
| **V5V-C13** | sequential multiplicity | Alpha-investing is only eligible if each incoming p-value remains conditionally valid at its assigned level; it controls mFDR, not FWER. | `directly_supported` | S09 (2008) | Exact statistical requirement; local eligibility not established. | Same-quarter adaptive p-values must not be placed into an alpha-investing ledger as if fresh. | Add an eligibility gate and explicitly choose FWER/FDR/mFDR target. |
| **V5V-C14** | adaptive holdout | A reusable holdout is a protocol installed before exposure and mediated to limit information release; ordinary exact-score reuse is not that protocol. | `contradicts_premise` | S10 (2015) | Direct distinction; dependent local transfer is absent. | Do not rename exposed 2023 quarters a “reusable holdout.” | Reserve `holdout` for access-controlled surfaces with a pre-exposure receipt. |
| **V5V-C15** | small-sample inference | MCS may honestly retain many/all candidates when information is weak; failure to eliminate is not evidence of equivalence or quality. | `directly_supported` | S08 (2011) | Direct. | Report the survivor set and low-power interpretation; do not force a winner. | Add `inconclusive` as a valid adjudication outcome. |
| **V5V-C16** | local assumptions | The effective number of issuance-day blocks, dependence length, stability of loss differentials, and complete historical trial count are presently unknown. | `insufficient` | L01–L05; source requirements S04–S09 | No local diagnostic was allowed in this lane. | No valid p-value, CI coverage claim or block length can be promised now. | DS@v5 must require diagnostics and retain `UNKNOWN` rather than default PASS. |
| **V5V-C17** | exact official metric | Paired block resampling should keep each day intact and recompute exact Total and components for each resample; treating 72 cells as independent would be pseudoreplication. | `near_match_only` | S04/S06/S07 plus L01 | Dependence principle direct; exact official-metric bootstrap mapping needs local metric audit. | Use a common resample index for candidate/control and exact metric code. | Add a metric-resampling contract test before inference. |
| **V5V-C18** | current maximum claim | With no fresh surface, a passing node can at most claim chronology-correct, reproducible retrospective performance and recorded-family conditional corroboration. | `directly_supported` | S03, S07, S10 plus L02–L05 | Direct logic from selection/adaptivity facts. | `CONFIRMED_SUPERIOR` is unavailable now. | Add statuses `RETROSPECTIVE_CORROBORATION` and `CONFIRMATION_PENDING`. |

## Assumptions and scope audit

### Assumptions that no evaluation label can silently inherit

| Method | Required/limiting assumption from source | DS@v5 consequence |
|---|---|---|
| Rolling/prequential origins | Forecasts and all updates are based only on information available at the origin; origins/test periods are explicitly specified [S01, S02]. | Prove timestamp, label availability, horizon overlap and recalibration behavior for every origin. |
| Nested selection | Every tuning/selection operation is rerun without the outer outcomes; compare selection procedures, not a model chosen with outer feedback [S03]. | Treat preprocessing through settlement-action policy as one pipeline. Human choices informed by outer results are outside a nominal inner loop and must be counted as meta-selection. |
| DM-style pairwise inference | Covariance-stationary, short-memory loss differential and a consistent long-run variance [S04]. | If not defensible, report paired effects descriptively. “HAC” is not a magic label for arbitrary drift or tiny samples. |
| Giacomini–White | Forecasting method/window is part of the object; finite-memory rolling/fixed estimation, mixing, moments, positive-definite covariance, growing evaluation sample [S05]. | A local expanding-window procedure or data-mined regime instrument is only a near match unless separately justified. |
| Stationary bootstrap | Weakly dependent stationary observations; random geometric blocks [S06]. | Resample whole days and audit differential stability. Regime breaks are not cured by choosing a longer block. |
| Reality Check | Complete supplied candidate performance matrix, common benchmark/loss/rows and asymptotic dependence conditions [S07]. | Claim is “best among these recorded candidates vs this benchmark,” not “best pipeline.” Unknown historical attempts remain unadjusted. |
| MCS | Finite initial family; valid equivalence test/elimination; forecast-loss differentials with moments, strict stationarity and alpha-mixing [S08]. | A large survivor set is a legitimate low-information outcome. Coverage is candidate-set-relative and asymptotic. |
| Alpha-investing | Each null test has conditional size no larger than assigned alpha given prior rejection outcomes; target is mFDR [S09]. | Full-score-driven repeated-quarter tests are ineligible absent a proof of conditional validity. It cannot repair past looks. |
| Reusable holdout | A special access/release mechanism is installed before adaptive queries [S10]. | Exact historical score exposure cannot be “unseen.” No direct dependent-time-series migration is established here. |

### Scope language

Use these exact qualifiers:

- **“outer-origin”** means temporally later than that fold's training/selection data; it does **not** mean globally unseen to the research program.
- **“out-of-sample within the replay”** is acceptable with the fold receipt.
- **“independent holdout,” “fresh confirmation,” “unbiased generalization estimate,”** and **“future superiority”** are prohibited for the exposed 2023 surface.
- **“selection-adjusted”** is permitted only with the exact adjustment method and complete candidate family named.
- **“stable”** must name the checked origins/quarters/regimes and cannot be generalized to unobserved regimes.
- **“best”** must be replaced by “best observed in family F” or “member of the MCS for family F.”

## Alternatives for a three-level diagnostic → screen → confirm hierarchy

No relabeling can manufacture Level 3 today. The alternatives differ in what they do while confirmation is unavailable.

| Architecture | Level 1 — diagnostic | Level 2 — bounded screen | Level 3 — terminal comparison | Honest claim now | Recommended use |
|---|---|---|---|---|---|
| **A. Future-sequestered confirmation** (**preferred scientific hierarchy**) | Use exposed history freely for mechanism diagnostics; mark every result exploratory. Freeze hypothesis, transform, family, falsifier and practical effect before the screen. | Inner chronological origins only; bounded candidate family; no confirmatory p-values; choose a single selection rule or small frozen shortlist. | One predeclared comparison on chronologically new outcomes not released during development; use one-look FWER or a valid forward alpha ledger. | Currently `CONFIRMATION_PENDING`; no Level-3 claim. | Canonical DS@v5 architecture. It is the only alternative that can recover genuinely new evidence without external data today. |
| **B. Frozen-family retrospective audit** | Same diagnostic tier, but all prior exposure is recorded in an exposure ledger. | Full nested/prequential replay. All candidates and the control generate aligned outer-origin predictions; the selection algorithm is frozen. | One-shot RC (“any recorded candidate beats control?”) or MCS (“which candidates cannot be shown inferior?”), with day-block uncertainty and assumption sensitivity. | `RETROSPECTIVE_SELECTION_ADJUSTED_CORROBORATION`, conditional on recorded family, rows, metric and stationarity/mixing. | Best available current adjudication when a complete loss matrix exists. It cannot be renamed confirmation. |
| **C. Conservative audit/stress hierarchy** | Diagnostics on all history. | One bounded champion/control screen using nested origins. | No p-value. Require exact reproduction plus sign/effect concentration views across origins, quarters, groups, lead and declared regimes; return pass-for-engineering, fail, or inconclusive. | `REPRODUCIBLE_HISTORICAL_STRESS_PASS` only. | Use when candidate history is incomplete or bootstrap/test assumptions are implausible. More honest than a misspecified inferential test. |

A leave-one-quarter-out rotation is a useful **stress test**, not Architecture A's confirmation, because each 2023 quarter has already influenced development. Likewise, hiding quarter labels from the current run does not erase what the analyst learned previously.

## Nested/prequential pseudoprotocol

The following is an **[MH] proposed DS@v5 protocol**, assembled from S01–S05. It is not claimed as an exact recipe from any one source.

```text
INPUT
  atomic unit U_d := one issuance day containing all 72 cells
  exposed development surface H
  deployment basis-time rule B (including D-1 14:00 KST availability)
  frozen parent/control C
  frozen candidate-generating/selection procedure A
  frozen outer origins O_1 < ... < O_J and outer evaluation blocks E_j
  frozen inner-origin generator I(.), purge/embargo rule, primary metric Total,
  component diagnostics, practical effect delta, and comparison budget

PREDECLARATION / ACCOUNTING
  hash A, C, candidate-family generator, data schema, row keys, policy,
  metric code, origin list, label-availability rule, and analysis plan
  set surface_exposure_status = EXPOSED for the repeatedly used 2023 surface
  record all prior known trials; set historical_unknown_trials honestly

FOR each outer origin O_j in chronological order:
  P_j := all rows whose features AND labels were available before O_j under B
  assert no target/feature/selection value from E_j enters P_j
  assert an embargo/purge covers any overlapping target horizon

  INNER SELECTION (only within P_j):
    construct ordered inner origins I_j = I(P_j)
    for every material branch of A:
      fit preprocessing/imputation/feature construction only on inner past
      fit model only on inner past
      choose hyperparameters, window, calibration, ensemble/blend,
        threshold/action policy and stopping rule only from inner validation
    apply the frozen selection rule to choose configuration a_j

  OUTER FORECAST:
    refit the chosen procedure a_j on all eligible P_j only
    generate predictions for every cell in E_j exactly once
    freeze prediction artifact and provenance BEFORE exposing E_j labels
    when labels become available, append day-level paired losses/components
      for a_j and C; never use E_j to revise its own prediction

AFTER all outer origins:
  concatenate only the once-frozen outer predictions on identical row keys
  evaluate the stream produced by procedure A versus C
  do not substitute a hindsight-fixed winner for the per-origin selections
  recompute exact Total/NMAE/FICR and prespecified stability diagnostics
  run either:
    (a) descriptive paired day-block uncertainty; or
    (b) frozen-family RC/MCS if and only if assumptions/family completeness pass
  tag the result RETROSPECTIVE_CORROBORATION when any outer outcomes were
    previously exposed at the research-program level

FUTURE CONFIRMATION:
  do not inspect Level-3 outcomes until prediction/protocol hashes are frozen
  release only the predeclared comparison; update a forward error ledger
  if no such outcomes exist, retain CONFIRMATION_PENDING
```

### What nesting fixes, and what it does not

- **[SF]** S03 supports rerunning model selection inside evaluation and comparing the combined procedure.
- **[MH]** chronological inner origins nested inside chronological outer origins apply that rule to the BARAM dependence structure.
- **[LE]** DS@v5 must prove all preprocessing, feature choice, calibration, blend and action-policy branches are inside the inner loop.
- **Not fixed:** prior analyst knowledge of outer-quarter scores, unlogged experiments, selection of this protocol after seeing outcomes, future regime shift, and lack of a new confirmation surface.

## Experiment and comparison accounting

### Count the decision, not merely the fitted model

A **material comparison** is any branch whose value could change because a diagnostic, score, component, fold, group, lead, regime, online response, or previous comparison was observed. Count at least:

- data inclusion/exclusion/window and missingness treatment;
- transformations and imputation;
- feature families and feature-selection thresholds;
- model family, architecture and loss;
- every hyperparameter search family, including range expansions after results;
- seeds when the favorable seed is retained or seeds affect promotion;
- calibration, clipping, blend members/weights and post-processing;
- action/settlement policy, threshold and group-specific policy;
- metric/subgroup/quarter chosen for the claim;
- reruns motivated by a failed or favorable result;
- a human-designed child whose design uses the parent's exposed result.

Routine deterministic reproduction under an already frozen hash is not a new candidate, but it is a new **look** if its score can change a decision.

### Required append-only records

For each `comparison_id` (continuing the skeleton's append-only index), record:

```text
comparison_id, node_id, parent_ids, stage_id
surface_id, surface_exposure_status, outer_origin_ids, exact row keys
candidate_family_id, candidate_id, control_id
planned_candidates, realized_candidates, known_prior_candidates
historical_unknown_trials (bool + reason)
all material degrees_of_freedom and trial-generation rule
selection_algorithm_hash, pipeline_hash, prediction_hash, policy_id
which outcomes/statistics were revealed to whom and when
primary estimand/loss/direction/practical delta
screen-only vs retrospective-audit vs future-confirmatory tier
multiplicity family, method, error criterion, alpha allocation
block unit/rule/sensitivity grid and assumption status
result, component/stability diagnostics, claim scope, child-generation reason
```

### Family boundaries

1. **Inner screening family:** ranking only. Do not attach confirmatory p-values.
2. **One retrospective family per frozen node:** control plus *all* candidates whose choice used that node's screen. RC/MCS must see the whole aligned family, not only the winner.
3. **Sequential research-program family:** every future confirmatory hypothesis. Choose FWER or mFDR before starting; do not mix them rhetorically.
4. **Secondary endpoints:** Total is primary. NMAE, FICR, group/fold/lead/regime and failure concentration are mechanistic/stability diagnostics unless their individual claims and multiplicity plan were predeclared.
5. **Unknown past search:** never encode `0`. Use `UNKNOWN`; any adjusted claim then says “among recorded family F,” not “after all development.”

## Dependent uncertainty and multiplicity options

### Atomic loss construction

**[MH]** Let `d` index issuance days. Keep the 72 cells and all candidate/control rows for day `d` together. For a pairwise comparison, use the same resampled day/block indices for both forecasts. If exact Total is not safely expressible as an additive daily loss under the official implementation, resample whole day bundles and recompute exact Total, NMAE and FICR inside every replicate. Do not create 72 nominally independent observations from one issuance decision.

This gives uncertainty **conditional on the frozen predictions and observed historical sequence**. It does not automatically include uncertainty from model fitting, stochastic training, candidate generation or analyst search.

### Options and gates

| Option | Proper question | Strength | Mandatory caveat/current disposition |
|---|---|---|---|
| **Paired descriptive origin/day distribution** | Where, when and in which direction did candidate minus control differ? | Minimal assumptions; exposes concentration and regime reversal. | No coverage/p-value claim. Always report before inferential summaries. |
| **DM/HAC-style paired loss differential** [S04] | Is mean loss differential zero for two fixed forecasts? | Supports general loss and serial correlation under covariance-stationary short-memory differential. | Ineligible after winner selection on the same differential unless selection is nested/adjusted; ineligible if stationarity/effective sample is not defensible. Exact Total may require a defined additive loss or delta argument. |
| **Giacomini–White conditional predictive ability** [S05] | Can current, predeclared information predict which fixed forecasting method performs better? | Can preserve estimation uncertainty and accommodate heterogeneous/misspecified, nested methods. | Freeze finite-memory window and a very small instrument set. Asymptotic, mixing and moment requirements plus local sample size make this optional, not default. |
| **Moving-block bootstrap of issuance days** | What is prediction-conditional uncertainty under a weakly stationary dependent-day approximation? | Fixed contiguous blocks preserve local order/dependence. | Block length is a material choice; show a predeclared short/central/long sensitivity grid. Few long blocks imply very low effective information. |
| **Stationary bootstrap of issuance days** [S06] | Same, with geometric random block lengths and stationary resamples. | Natural input to RC and some MCS implementations. | Assumes stationarity/weak dependence; random wrap/restarts may be scientifically implausible across regimes. It does not cure drift. |
| **White Reality Check** [S07] | Does the best member of frozen recorded family F beat benchmark C? | Directly targets data snooping over a family and can use dependent bootstrap. | Conservative and asymptotic; requires every candidate loss vector. It says nothing about omitted historical attempts or future regimes. |
| **Model Confidence Set** [S08] | Which members of frozen family F cannot be shown inferior under the chosen loss? | Avoids forcing one winner; accounts for multiple candidates through sequential elimination. | Requires stationary/mixing loss differentials. Survivors can include poor models in small samples; output is family-relative. |
| **Forward Bonferroni/alpha-spending** | Can a bounded sequence of future primary claims control FWER? | Simple and auditable when incoming p-values are valid. | Spend only on genuinely eligible future comparisons; no earning/recycling. Does not repair past exposure. |
| **Alpha-investing** [S09] | Can a stream of eligible hypotheses control mFDR while earning wealth after rejection? | Adaptive ordering; tests need not be independent if conditional validity holds. | Not the default: mFDR is weaker than FDR/FWER, and same-2023 score-driven hypotheses do not yet meet the conditional-size gate. |
| **Special reusable-holdout mechanism** [S10] | Can controlled information release support many adaptive queries? | Directly addresses adaptivity when installed prospectively. | No retrospective application; no direct dependent/nonstationary BARAM migration in this source set. A new access-controlled service would require separate authority and research. |

### Nonstationarity decision rule

1. Plot/report the paired day differential by ordered origin, quarter and predeclared operational regime; inspect sign reversal, mean shift, variance change and failure concentration. This is diagnostic, not a formal stationarity proof.
2. Audit autocorrelation/overlap at the issuance-day level and record candidate block lengths before looking at inferential conclusions.
3. If a stationary loss-differential approximation is scientifically defensible, run the prespecified central method and block-length sensitivity; label the result assumption-conditional.
4. If results change materially with block length/regime definition, or the differential has obvious shifts, demote pooled p-values to `UNSTABLE_ASSUMPTIONS`; report regime-wise descriptive effects and `INCONCLUSIVE`.
5. Never choose the block length, regime partition or test because it gives the smallest p-value. Such a choice is another comparison and must be nested or multiplicity-adjusted.

## Screening versus confirmation rules

### Diagnostic tier

- May use exposed 2023 history and all component/failure views.
- Produces a mechanism, falsifier, migration gaps and bounded candidate generator.
- Does not say “significant,” “validated,” or “improves future performance.”
- Any diagnostic that influences a child is entered in the exposure/comparison ledger.

### Screen tier

- Runs the whole candidate generator only on inner prequential origins.
- Freezes the candidate cap and selection rule before viewing the screen result.
- Can reject families cheaply, prioritize one rule, or retain a shortlist.
- Scores are optimization information, not confirmatory evidence.
- A screen winner's raw best score is never the reported effect estimate; use its outer selection-procedure stream.

### Confirmation tier

A node may enter this tier only if:

1. the prediction/protocol was frozen before any target outcome was exposed;
2. no analyst/pipeline query obtained exact or proxy information about those outcomes;
3. the primary estimand, direction, practical effect, test, dependence unit, block/window rule and multiplicity allocation were frozen;
4. the comparison is made once under the planned release rule; and
5. all prior confirmatory looks are represented in the error ledger.

No current 2023 replay satisfies condition 1 at the research-program level. Thus “strict full comparison” and “confirmatory” must be separate fields: a strict retrospective run can pass while confirmation remains pending.

## What may and may not be claimed

### Permitted now, if the corresponding receipts exist

- “Pipeline/procedure A reproduced exact historical metrics on rows R under policy P and chronology contract C.”
- “In a nested rolling-origin replay, the *selection procedure* A had observed paired effect X versus frozen control C on exposed historical surface H.”
- “The effect had the following concentration/sign pattern across the named origins, quarters, groups, leads and regimes.”
- “Under explicitly stated stationary/mixing and block assumptions, the prediction-conditional interval/test for the frozen recorded family was ….”
- “Within recorded family F, the Reality Check rejected/did not reject no superiority over C,” or “MCS retained set M,” provided family completeness for F and alignment are proved.
- “This is retrospective corroboration and a candidate for future confirmation.”

### Prohibited now

- “Independent holdout,” “untouched test,” or “fresh out-of-sample confirmation” for any repeatedly exposed 2023 quarter.
- “Unbiased estimate of future deployment/leaderboard performance.”
- “Best pipeline/model” without the finite candidate family and metric qualifier.
- “Multiplicity corrected for the whole project” when historical trials are unknown or omitted.
- “No difference” or “equivalent” because a low-powered test/MCS failed to eliminate.
- “Robust to nonstationarity” merely because a block bootstrap or HAC covariance was used.
- “Two independent validations” because a deterministic run reproduced twice.
- “Alpha-investing-valid” when the next p-value was generated after viewing full previous statistics on the same outcomes.
- Any causal claim that a preprocessing/feature mechanism caused the observed gain without a separately identified design.

## Small-sample and limited-surface limits

1. **Effective sample size is days/blocks, not cells.** The 72 within-day cells share one issuance decision and must not inflate `n`.
2. **Only a few quarters cannot identify rich regime effects.** A quarter-by-treatment interaction or leave-one-quarter-out sign pattern has very few independent regime units; it is a stress view, not a population estimate.
3. **Nested evaluation spends data twice structurally.** Inner selection leaves fewer origins for outer estimation. Overlapping training windows and temporally adjacent evaluation days further reduce effective information.
4. **Long blocks leave few resampling units.** Short blocks understate dependence; long blocks produce unstable tails. Block-length sensitivity may expose that no useful inferential resolution exists.
5. **Asymptotic methods may be weakly calibrated.** DM, GW, stationary bootstrap, RC and MCS all rely on asymptotics/regularity. Their source simulations do not certify this local sample.
6. **Step/band components are discrete and tied.** FICR changes can concentrate near thresholds. Smooth normal approximations can be poor, and exact Total should be recomputed inside whole-day resamples.
7. **Nonstationarity and dependence are different problems.** Blocks address local dependence under an approximation; regime drift changes the target distribution and cannot be averaged away without a scope decision.
8. **MCS may be large.** That is the honest consequence of limited information, not a procedure failure.
9. **Nested replay estimates a procedure, not necessarily the final hindsight fit.** If hyperparameters/policies vary across outer origins, the concatenated stream represents the selection rule. A final pipeline picked using all outer results is a new adaptively selected object.
10. **Historical adaptivity is partly unidentifiable.** If discarded experiments or exact feedback paths are missing, no numeric adjustment can reconstruct the true search multiplicity. The correct state is `UNKNOWN`, with narrower claims.

## Concrete DS@v5 recommendation

### 1. Separate operational status from evidence status

Retain SK@v5's operational success rule, but add independent evidence fields:

```text
run_reproducibility_status:
  NOT_RUN | SINGLE_REPRODUCTION | DOUBLE_REPRODUCTION

evaluation_evidence_status:
  EXPLORATORY_DIAGNOSTIC
  BOUNDED_SCREEN
  RETROSPECTIVE_CORROBORATION
  RETROSPECTIVE_SELECTION_ADJUSTED_CORROBORATION
  CONFIRMATION_PENDING
  FRESH_CONFIRMATION
  INCONCLUSIVE
  ASSUMPTION_FAILED
```

A node may be operationally reproducible and still be `CONFIRMATION_PENDING`. `FRESH_CONFIRMATION` requires a nonzero `independent_confirmation_count` and a pre-exposure receipt.

### 2. Add mandatory DS@v5 fields

In addition to the node schema already required by SK@v5, add:

```text
evaluation_tier
primary_estimand                       # historical surface / selector / fixed pipeline / future
surface_id, surface_rows_hash
surface_exposure_status                # FRESH / EXPOSED / CONSUMED / UNKNOWN_NOT_FRESH
exposure_ledger_ids
basis_time_rule, label_availability_rule
atomic_unit, forecast_horizon, purge_embargo_rule
outer_origin_ids, outer_block_ids, inner_origin_generator
selection_algorithm_hash
pipeline/prediction/policy/metric hashes
comparison_family_id
planned/realized/known_prior candidate counts
historical_unknown_trials
primary loss, direction, practical_delta
secondary endpoints and their claim status
multiplicity method, error criterion, family, alpha ledger
resampling unit, block method/length rule/sensitivity grid
assumption checklist and failure action
claim_scope, independent_confirmation_count
```

Default `surface_exposure_status` to `UNKNOWN_NOT_FRESH`, never `FRESH`, when provenance is incomplete.

### 3. Required contract tests before any promotion

These are **[LE] tests to implement later**, not results from this lane.

| Test ID | Required test | Passing evidence | Failure action |
|---|---|---|---|
| **V5VT-01 Authority/freshness** | Hash skeleton/manifest/input contracts; resolve exposure status for every evaluation block. | Pre-exposure access receipt for `FRESH`, or explicit `EXPOSED/CONSUMED`. | No confirmatory label. |
| **V5VT-02 Chronology** | For every predicted cell, assert every fit/selection feature and label timestamp is strictly available by its basis time. | Machine-readable row-level availability audit. | Fail closed; re-enter upstream leakage stage. |
| **V5VT-03 Day atomicity/overlap** | Assert all 72 cells of an issuance day remain in one fold/block and purge target overlap across inner/outer boundaries. | Fold-membership and horizon-overlap contract. | Evaluation invalid. |
| **V5VT-04 Outer-label isolation** | Prove candidate generation, stopping, preprocessing, feature, model, calibration, blend and policy selection cannot read outer labels/metrics. | Dependency graph plus label-blind artifact hash before label join. | Demote to diagnostic. |
| **V5VT-05 Full-pipeline nesting** | Re-run the entire selection procedure in every outer past; do not inject global tuned state. | Per-origin inner receipt and selection hash. | Selection-bias gate fails. |
| **V5VT-06 Alignment/policy reproduction** | Candidate/control have identical row keys, origin set, target/policy definition; policy column exactly reproduces predictions. | Exact equality/uniqueness assertions and hashes. | Comparison inadmissible. |
| **V5VT-07 Comparison completeness** | Reconcile planned, realized and known-prior candidates, including failures and human result-conditioned branches. | Append-only ledger; `historical_unknown_trials` explicit. | Restrict to recorded-family claim or audit-only Architecture C. |
| **V5VT-08 Selector-vs-control estimand** | Verify concatenated outer predictions are from the per-origin selected procedure, not a hindsight winner substituted after scoring. | Origin-to-selected-config mapping frozen before each label join. | Report only fixed artifact reproduction; no selector estimate. |
| **V5VT-09 Exact metric resampling** | On a deterministic toy/index test, whole-day resampling and exact recomputation preserve pairing and official Total/component semantics. | Metric contract tests across repeated/missing day indices. | No bootstrap inference. |
| **V5VT-10 Dependence/regime diagnostic** | Ordered paired differential, ACF/overlap, quarter/regime sign and concentration audit; freeze block grid before inference. | Diagnostic artifact with `SUPPORTED/DOUBTFUL/FAILED/UNKNOWN`. | If not supported, descriptive-only. |
| **V5VT-11 Block sensitivity** | Recompute the paired interval/test over prespecified short/central/long day-block rules and no-block diagnostic. | Same scientific conclusion or explicit instability. | `INCONCLUSIVE`/`ASSUMPTION_FAILED`; never select favorable block. |
| **V5VT-12 Fixed-family multiplicity** | If eligible, RC includes control and all recorded candidates; or MCS begins with the same complete finite family. | Family manifest hash equals loss-matrix columns; common rows/loss. | No selection-adjusted wording. |
| **V5VT-13 Sequential-alpha eligibility** | For a proposed future test, document why its p-value is valid conditional on prior outcomes and why FWER/FDR/mFDR is the chosen target. | New outcome block, frozen protocol, alpha-ledger entry before release. | Do not spend/invest alpha; classify as retrospective. |
| **V5VT-14 Claim lint** | Compare result text to exposure, family, assumptions and estimand. | Automated prohibition of “fresh/independent/best/future” when gates fail. | Block promotion/report. |
| **V5VT-15 Reproduction interpretation** | Run the required fixed-policy reproduction and verify bit-identical lineage. | Two reproduced runs plus `independent_confirmation_count=0` unless a separate fresh receipt exists. | Operational reproduction fails; never infer independence from reruns. |

### 4. Promotion/adjudication rule

A concrete node decision should be:

```text
if chronology/alignment/policy/nesting fails:
    FAIL_CLOSED_NODE and re-enter the implicated upstream stage
elif only diagnostic evidence exists:
    BOUNDED_SCREEN or RESEARCH_PAUSE
elif exposed outer replay passes practical/stability gates but family inference is ineligible:
    RETROSPECTIVE_CORROBORATION + CONFIRMATION_PENDING
elif exposed replay plus complete-family RC/MCS passes under supported assumptions:
    RETROSPECTIVE_SELECTION_ADJUSTED_CORROBORATION + CONFIRMATION_PENDING
elif a predeclared genuinely fresh chronological block passes its one-look/ledgered gate:
    FRESH_CONFIRMATION
else:
    INCONCLUSIVE, retain benchmark and generate a residual-deficit child
```

The practical effect `delta`, candidate budget, maximum future looks, block rule and component safety gates belong in DS@v5/IP@v3 and must be frozen **before** the relevant screen/confirmation. Failure to reject zero is not equivalence; if practical equivalence is a goal, define and test that separate estimand prospectively.

### 5. Recommended current hierarchy

- Use **Architecture A** as the canonical scientific hierarchy.
- For immediate historical adjudication, use **Architecture B** only when the complete aligned candidate loss matrix and assumptions exist; otherwise use **Architecture C**.
- Mark every present 2023-based result `EXPOSED` and at most retrospective.
- Treat 2024 as `CONSUMED` and unavailable, consistent with `AGENTS.md`.
- Do not assume any other block is fresh without an access/exposure receipt.
- Make the append-only comparison and exposure ledger a parent-level resource so result-conditioned children inherit all prior looks.

## Direct answers to the research question

1. **Nested/prequential selection:** nest the complete pipeline-selection algorithm inside each past-only outer origin; keep forecast horizon/label availability strict; evaluate the algorithm-plus-selector stream. This removes within-run tuning leakage but not historical meta-overfit.
2. **Screening versus confirmation:** diagnostics and screens may use exposed history but cannot make inferential claims. Confirmation requires a pre-exposure frozen protocol and new outcomes. Current confirmation is pending.
3. **Sequential multiplicity:** fixed batches may use RC/MCS under their assumptions. A future stream may use conservative alpha-spending or, only with conditional-validity proof and acceptance of mFDR, alpha-investing. Same-quarter adaptive p-values are ineligible.
4. **Dependent forecast comparison:** pair candidate/control on issuance days; preserve 72-cell days in contiguous/random-length blocks; recompute the exact official metric. DM/GW/RC/MCS each have explicit stationarity/mixing/window/sample limitations.
5. **Hyperparameter and pipeline selection bias:** count and nest preprocessing, feature, model, tuning, calibration, blend and action policy; report the selector, not the hindsight winner.
6. **Repeated 2023 exposure:** claim only exact historical reproduction, chronology-correct retrospective effects, and—if complete/eligible—recorded-family selection-adjusted corroboration. Do not claim an independent holdout, future generalization or project-wide multiplicity control.

## Contradictions and tensions to preserve in root synthesis

- **C-A:** `two reproduced strict runs` is an operational success criterion, not two independent statistical confirmations. DS@v5 must carry both statuses.
- **C-B:** `outer fold` is not synonymous with `fresh holdout` after project-level exposure.
- **C-C:** blocking/HAC handles dependence only under assumptions; it does not itself handle arbitrary nonstationarity.
- **C-D:** MCS does not identify a universally true/best model and can retain weak candidates; RC only addresses the supplied family.
- **C-E:** alpha-investing's allowance for dependent tests does not waive its conditional-size requirement and does not retroactively repair same-data adaptive hypotheses.
- **C-F:** Dwork et al.'s reusable holdout is a prospective controlled-release mechanism, not a name for ordinary repeated exact-score access.

## Unknowns / required root follow-up

1. Complete historical comparison universe: how many candidate families, policies, subgroup choices, reruns and online/local feedback events influenced the current lineage, including discarded failures?
2. Exact exposure granularity: totals only, components, quarter/fold/group/lead values, row-level predictions/residuals, or all of these?
3. Whether an aligned day-by-candidate outer loss matrix exists for the entire recorded family, with one fixed policy per input and exact row keys.
4. Number of issuance days and effective blocks on each permitted surface; forecast-horizon overlap and plausible dependence/seasonal lengths.
5. Whether candidate-control loss differentials—not just raw losses—are plausibly stable/mixing, and whether this conclusion survives quarter/regime views.
6. Whether the current estimator is expanding, fixed, rolling or adaptive-window; GW's direct scope depends on this.
7. Exact day-level algebra of Total/NMAE/FICR under duplicate/block-resampled days and whether the official implementation supports exact resample recomputation.
8. Whether training/model stochasticity is material; prediction-loss resampling alone conditions on fitted artifacts.
9. Which error criterion the project wants for future primary claims: strong FWER, FDR, or mFDR. They are not interchangeable.
10. The practical effect threshold and component safety constraints that must be frozen before a future confirmation.
11. Whether any genuinely unexposed chronological outcomes can arrive under the competition deadline and rules. If none can, confirmation must remain pending rather than being simulated by resplitting.
12. How DS@v5 will prevent full-score leakage from a future confirmation service; a reusable-holdout mechanism for dependent time series is not established by this source set.

## DS@v5 bottom line

Implement an **exposure-aware nested prequential evaluator**, an **append-only trial/feedback ledger**, **issuance-day paired uncertainty**, and **candidate-family-relative multiplicity**. Make `CONFIRMATION_PENDING` a first-class valid state. The present surface can support rigorous engineering decisions and conditional retrospective corroboration, but not a new independent scientific confirmation. That limitation should narrow claim scope, not be hidden by another split, another bootstrap, or another deterministic rerun.
