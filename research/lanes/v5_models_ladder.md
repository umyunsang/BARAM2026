# V5-MODELS — S7 model ladder under a discontinuous two-band settlement reward

- **Lane:** `V5-MODELS` (SK@v5 Wave B, read-only research)
- **Exclusive output:** `research/lanes/v5_models_ladder.md`
- **Written:** 2026-08-09 (bounded session; terminated early by root instruction to write)
- **Authority verified by SHA-256 in-lane:**
  - `AGENTS.md` = `91a11f95f94e8de86f5c84b204e1893830b3ed5693e3d7ede7f2e351e9891e9c`
  - `.planning/2026-08-01-leaderboard-top-4-loop/SK_v5.md` = `c2c813475ab5f0a741e6adbceab196d5b38447f2aeb02f9716a8f5be8973c820` (matches brief)
  - `reports/sk_v5_approval_receipt.json` = `e827b324a23913346598d274f2ded496ffc3134243de5636033ee6a3ac387173` (matches brief)
  - `research/nodes/sk5_foundation_map.json` = `393ee74bf53251037547fc52bc92c554a4aa207e2288f194766d836675512b35` (matches brief)
  - `research/nodes/sk5_local_capability_constraint.json` = `3209042a0ec9ddd669992e94bf6137beb79582cc5ad62fa65337bd109be700dc` (root-supplied; independently re-hashed here)

**Tag legend (exactly one per material claim):** `directly_supported` = stated in a fetched primary
source or a named local record; `contradicts_premise` = the evidence defeats a proposed
applicability; `near_match_only` = a real result under a materially different loss/sample/chronology,
which may motivate a hypothesis but never claims transfer or expected gain; `insufficient` = not
established by this bounded source set. A separate `[derived]` marker denotes algebra performed in
this lane on top of tagged inputs; `[derived]` is never evidence of transfer.

---

## 0. VERDICT

**`LADDER_IS_FEASIBILITY_BOUND_NOT_METHOD_BOUND` — the state of the art that BARAM can actually
execute is a two-line ladder, and the model axis is not where the remaining 0.0285 Total lives.**

Three independent findings converge on this. `directly_supported`

1. **The decision layer is already solved analytically, so no model family can add a new action
   mechanism.** The exact population action was derived and closed in the repository's own
   `S17-N18` lane: the sufficient conditional objects are one absolute-moment curve `A_x(a)` and two
   *generation-weighted* sliding-window masses `W_.06(a)`, `W_.08(a)`, and the objective is additive
   over rows once the population group constants are fixed, so a joint 24-hour or cross-group
   decision rule cannot beat componentwise Bayes actions. Every model family in (a)–(e) is therefore
   reduced to a single question: *which estimator of those three curves is best under n≈90–179
   dependent issuance days?* `directly_supported` (local record `research/lanes/s17_n18_step_reward.md` §3, §5)
2. **The local capability constraint removes essentially the whole modern (c) and most of (b) and
   (d) literature from feasibility.** No `torch`, no `ngboost`, no `crepes`, no `mapie`, no
   `venn-abers`, no `quantile-forest`, no `cvxpy`, and dependency changes are forbidden by IP@v3.
   Every neural sequence/graph architecture, every pretrained tabular foundation model, and every
   distributional-boosting library is `BLOCKED_DEPENDENCY`. `directly_supported`
   (`research/nodes/sk5_local_capability_constraint.json` sha `3209042a…`; independently confirmed by
   `.venv/bin/python -c "import importlib.metadata"` probe in-lane)
3. **FD7 makes low-degrees-of-freedom the dominant ranking criterion, not accuracy.** With no fresh
   holdout, no unconsumed lockbox, and only two learnable outer transitions, a model family's cost is
   its *tuning surface*, not its compute. This inverts the usual SOTA ordering: a zero-tuning
   estimator with mediocre reported accuracy outranks a strong estimator with a large frozen-choice
   budget. `directly_supported` (foundation map `claim_limits`, FD7; `s17_n18_step_reward.md` §6)

**Consequence.** Only **two** rungs survive as genuinely new *mechanisms* under the present
constraints, and both are FD5-gated diagnostics rather than model swaps. Everything else in the
requested taxonomy is either a re-run of the closed "post-processing the current representation"
axis, a re-run of the closed step-reward estimator axis, or `BLOCKED_DEPENDENCY`. I do **not** claim
either surviving rung will reach `0.660000`; no such claim is admissible with zero fresh
confirmation surfaces.

---

## 1. SOURCE LEDGER

Bound: ≤16 primary or official benchmark sources. **Used: 5 external primary sources fetched and
verbatim-quoted in this lane, plus 6 named local records.** The lane was stopped by root instruction
before the remaining budget was spent; §9 lists exactly which questions that left open.

### 1.1 External primary sources actually fetched and quoted

| ID | Exact citation | Date relied on | Primary locator | What was used | Scope warning |
|---|---|---|---|---|---|
| E1 | Grinsztajn, L., Oyallon, E., Varoquaux, G. — *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS 2022 Datasets & Benchmarks. | preprint 2022-07-18; retrieved 2026-08-09 | `https://arxiv.org/abs/2207.08815`; also `https://papers.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html` | Verbatim abstract: "Results show that tree based models remain state-of-the-art on medium-sized data (∼ 10K samples) even without accounting for their superior speed."; and the three stated NN deficits: "1. be robust to uninformative features, 2. preserve the orientation of the data, and 3. be able to easily learn irregular functions." Benchmark = 45 datasets, "20 000 compute hours hyperparameter search for each learner". | IID tabular benchmark, **not** time series, **not** a discontinuous banded reward, and the ∼10K claim is about *medium*-sized data. BARAM's per-group fit surface is smaller still. Transfer of the *ranking* is `near_match_only`. |
| E2 | Ćevid, D., Michel, L., Näf, J., Bühlmann, P., Meinshausen, N. — *Distributional Random Forests: Heterogeneity Adjustment and Multivariate Distributional Regression.* JMLR 23(333):1−79, 2022. | 2022; retrieved 2026-08-09 | `https://jmlr.org/papers/v23/21-0585.html` | Verbatim abstract: "Part of its appeal and reason for its versatility is its (implicit) construction of a kernel-type weighting function on training data, which can also be used for targets other than the original mean estimation… The induced weights define an estimate of the full conditional distribution, which in turn can be used for **arbitrary and potentially complicated targets of interest**." | The *paper's* method (MMD splitting criterion, `drf` package) is `BLOCKED_DEPENDENCY`. Only the **generic mechanism** — a forest induces a kernel weighting usable for arbitrary targets — is used here, and that mechanism is available from any sklearn forest's leaf co-occurrence with **zero new dependencies**. No claimed effect size transfers. |
| E3 | Elmachtoub, A. N., Grigas, P. — *Smart "Predict, then Optimize".* Management Science 68(1):9–26, 2022. DOI `10.1287/mnsc.2020.3922`. | 2022 (©2021 INFORMS); retrieved 2026-08-09 | `https://par.nsf.gov/servlets/purl/10339524` (publisher-deposited full text) | Verbatim: "Our SPO+ loss function can tractably handle any polyhedral, convex, or even mixed-integer optimization problem **with a linear objective**." And the framework definition: the prediction model predicts "key unknown parameters of the optimization model", with quality "measured by the decision error". | This is the **scope boundary** of SPO+, and it is the finding that matters most for BARAM — see §3 claim C4. |
| E4 | Gibbs, I., Candès, E. J. — *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021. | preprint 2021-06-01; retrieved 2026-08-09 | `https://arxiv.org/abs/2106.00170`; proceedings `https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html` | Verbatim abstract: "a general wrapper that can be combined with **any black box method** that produces point predictions of the unseen label or estimated quantiles of its distribution… our adaptive approach **provably achieves the desired coverage frequency over long-time intervals irrespective of the true data generating process**. We accomplish this by modelling the distribution shift as a learning problem in **a single parameter** whose optimal value is varying over time and must be continuously re-estimated." | The guarantee is **long-run marginal coverage of a prediction set**, not conditional coverage and not band-hit revenue. It is a coverage statement, **not** a Total statement. Any migration to FICR is `near_match_only` at best. The "single parameter" property is the part that matters for FD7. |
| E5 | Angelopoulos, A. N., Candès, E. J., Tibshirani, R. J. — *Conformal PID Control for Time Series Prediction.* NeurIPS 2023 / arXiv 2307.16895. | 2023; retrieved 2026-08-09 | `https://arxiv.org/abs/2307.16895`; author copy `https://www.stat.berkeley.edu/~ryantibs/papers/scorecast.pdf` | Located and identified as the online, past-only, distribution-free successor to E4 for time series. **Abstract not verbatim-extracted before the lane was stopped.** | Because the verbatim locator was not captured, every claim resting on E5 alone is downgraded to `insufficient` in §3. It is retained in the ladder only as a named alternative, not as evidence. |

### 1.2 Named local records used (not external sources; do not count against the bound)

| ID | Local record | What is admissible from it | Tag |
|---|---|---|---|
| L1 | `research/nodes/sk5_foundation_map.json` sha `393ee74b…` | Frozen S1/S4 facts, alignment keys, `claim_limits`, and FD1–FD9. | `directly_supported` |
| L2 | `research/lanes/s17_n18_step_reward.md` | §3 exact population action and its `A_x`/`W_.06`/`W_.08` decomposition; §3 additivity and no-randomization results; §5 nonduplication table; §6 learnability audit; verdict `CLOSE_STEP_REWARD_ESTIMATOR_AXIS_NO_STRICT_TEST`. Its own E1–E7 register (Gneiting 2011; Brehmer–Gneiting 2021; Bertsimas–Kallus 2020; Kallus–Mao 2023; Stratigakos 2022; Kitagawa–Tetenov 2018; `policytree`). | `directly_supported` |
| L3 | `research/lanes/S15_sota_model.md` §2.3 | Dependency audit disqualifying `treeffuser`, `lightgbmlss`, `xgboostlss` under the no-downgrade rule; `crepes`/`venn-abers`/`mapie`/`ngboost`/`gpboost` classified clean-but-absent. | `directly_supported` |
| L4 | `research/lanes/S16_sota_decide.md` §§1–2, 6 | Band-hit vs continuous-error correlation distinction; the loss-matched combiner measured negative; "do NOT build — anything that moves the action toward the conditional mean", nine replications. | `directly_supported` |
| L5 | `research/nodes/sk5_local_capability_constraint.json` sha `3209042a…` | Present/absent package set; `BLOCKED_DEPENDENCY` rule. | `directly_supported` |
| L6 | `AGENTS.md` "Established measurements" | Score algebra; the local→online offset does not transfer across method classes; the closed axes list; "the deployed prediction is an ACTION". | `directly_supported` |

**Sources deliberately NOT used as evidence:** no blogs, no listicles, no summary sites, no vendor
pages, no leaderboard aggregators. Search-result snippets were used only for locator discovery and
never quoted as evidence.

---

## 2. SCOPE-MATCH MATRIX AGAINST THE BARAM SURFACE

Columns are the eleven required scope dimensions. `=` match, `~` partial, `✗` mismatch, `?` unknown.

| Dim → | population | geography | horizon | issue_time | inputs | target | metric | resolution | topology | compute | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **BARAM** | 3 KPX groups, 17 turbines, ~90–179 dependent issuance days per learnable transition | Korea (KPX) | D+1, 24 leads | fixed D-1 14:00 KST | GFS 9 pts + LDAPS 16 pts, issued tables; static turbine meta; train-only SCADA | hourly group kWh | `0.5(1−NMAE)+0.5·FICR`, FICR = 4/3/0 in inclusive 6%/8% capacity-relative bands, actual-weighted, equal-group | hourly, 72 cells/day | 3 groups × 25 grid points, no explicit graph supplied | ≤6 workers, no GPU, no torch | must be commercial-use, pre-2026-07-05 |
| **E1** Grinsztajn | ✗ IID tabular, 45 datasets ~10K rows | ✗ n/a | ✗ no horizon | ✗ none | ~ generic numeric/categorical | ✗ generic regression/classification | ✗ RMSE/accuracy | ✗ n/a | ✗ n/a | ✗ 20 000 h HPO per learner | = open |
| **E2** DRF | ✗ IID | ✗ n/a | ✗ | ✗ | ~ generic | ~ multivariate response | ✗ MMD/CRPS-type | ✗ | ✗ | ~ CPU-feasible | ✗ `drf` pkg absent |
| **E3** SPO+ | ✗ synthetic shortest-path / portfolio | ✗ | ✗ | ✗ | ~ generic features | ✗ **cost vector `c`, linear in objective** | ✗ SPO regret | ✗ | ✗ | ~ | = open |
| **E4** ACI | ~ online time-indexed real data | ✗ US equities / election | ~ 1-step online | ✗ | ~ any black box | ~ any scalar label | ✗ **set coverage**, not revenue | ~ | ✗ | = trivial, numpy-only | = open |
| **E5** Conformal PID | ~ time series | ✗ | ~ | ✗ | ~ | ~ | ✗ coverage | ~ | ✗ | = numpy-only | = open |

**Reading of the matrix.** Not one source matches BARAM on `metric`. That single column is the whole
problem: the requested taxonomy (a)–(e) is organised by *estimator family*, but BARAM's binding
constraint is a `metric` × `population-size` × `degrees-of-freedom` interaction that no source in
this set measures. Therefore **no expected gain is quoted anywhere in this document**, and the
ladder is ordered by *evidence strength and feasibility*, exactly as the brief requires, not by
reported accuracy. `[derived]`

---

## 3. TAGGED CLAIM LEDGER

| # | Claim | Tag |
|---|---|---|
| C1 | Tree-based models remain state of the art on medium-sized (∼10K sample) tabular data even ignoring their speed advantage, per a 45-dataset benchmark with 20 000 compute hours of HPO per learner. | `directly_supported` (E1, verbatim) |
| C2 | The three stated inductive-bias deficits of NNs on tabular data are non-robustness to uninformative features, non-preservation of data orientation, and difficulty learning irregular functions. | `directly_supported` (E1, verbatim) |
| C3 | A random forest implicitly constructs a kernel-type weighting function on training data that "can be used for targets other than the original mean estimation", and the induced weights estimate the full conditional distribution usable for "arbitrary and potentially complicated targets of interest". | `directly_supported` (E2, verbatim) |
| C4 | **SPO+ is proved and tractable for optimization problems with a *linear* objective in the predicted parameters.** BARAM's per-cell cost `|a−y|/C − (y/4μ)(1{|a−y|≤.06C}+3·1{|a−y|≤.08C})` is *not* linear in the uncertain quantity `y`; `y` enters through a discontinuous kernel of `\|a−y\|` and again as a multiplicative weight. The BARAM decision problem is therefore **outside the class for which SPO+ consistency is claimed**. | `contradicts_premise` (E3 verbatim scope statement, against the local COST5-SPO+ construction's theoretical warrant) |
| C5 | The local strict result for COST5-SPO+ was Total `0.619583` vs strict champion `0.631483` (`−0.011899`), with `1−NMAE` up `0.003875` and FICR down `0.027674`. | `directly_supported` (L2 §1 record L4) |
| C6 | C4 supplies a *mechanism* for C5 that was previously unexplained: the surrogate's consistency guarantee never covered this loss, so the local negative is the predicted outcome of a scope violation, not evidence that decision-focused learning as a field fails here. | `[derived]` on `directly_supported` inputs |
| C7 | Adaptive conformal inference wraps **any** black-box point or quantile predictor, needs only **one** continuously re-estimated parameter, and provably attains the desired coverage frequency over long time intervals irrespective of the data-generating process. | `directly_supported` (E4, verbatim) |
| C8 | E4's guarantee is *long-run marginal coverage of a prediction set*. It is **not** a guarantee about FICR, about band-hit revenue, about conditional coverage, or about Total. | `directly_supported` (E4 abstract scope, read strictly) |
| C9 | The exact BARAM population action depends on three conditional curves — `A_x(a)=E[V|a−Y| | x]`, and generation-weighted window masses `W_h(a)=H_x(a+hC)−H_x((a−hC)^-)` for `h∈{.06,.08}` with `H_x(t)=E[V·Y·1{Y≤t} | x]` — and a raw conditional mode, one unweighted hit probability, or one modal interval is **not** the population action. | `directly_supported` (L2 §3) |
| C10 | The official objective is additive over rows once population group constants are fixed and has no cross-hour feasibility constraint, so a joint 24-hour decision rule cannot beat componentwise Bayes actions; sequence/graph structure can help **only** by better estimating the same marginal conditional risks. | `directly_supported` (L2 §3) |
| C11 | Randomization/dithering of the action cannot improve the population optimum, since `∫R(a;x)π(da|x) ≥ inf_a R(a;x)`. | `directly_supported` (L2 §3) |
| C12 | BARAM is a **full-information cost-sensitive policy problem**, not a bandit: observing `Y_i` reveals the entire reward row for every action in a frozen grid. | `directly_supported` (L2 §3) |
| C13 | `torch`, `ngboost`, `mapie`, `crepes`, `venn-abers`, `quantile-forest`, `cvxpy` are absent and dependency changes are forbidden; `treeffuser`, `lightgbmlss`, `xgboostlss` were separately disqualified by the no-downgrade rule. | `directly_supported` (L5, L3; re-verified in-lane) |
| C14 | Present and usable: `lightgbm 4.7.0`, `xgboost 3.3.0`, `catboost 1.2.10`, `scikit-learn 1.9.0`, `statsmodels 0.14.6`, `numpy 2.5.1`, `scipy 1.18.0`, `pandas 2.3.3`, `pyarrow 23.0.1`. | `directly_supported` (L5; re-verified in-lane) |
| C15 | Therefore the *mechanism* of C3 is executable with zero new dependencies: `sklearn` forests expose `.apply()` leaf indices, from which the kernel weights `w_i(x)` are pure numpy, and the exact `A_x`/`W_.06`/`W_.08` curves of C9 are then finite weighted sums over past-only `Y_i`. Multi-quantile arms are likewise native (`lightgbm` quantile objective, `xgboost reg:quantileerror`, `catboost MultiQuantile`). Adaptive conformal (C7) is ~20 lines of numpy and needs no absent library. | `[derived]` on `directly_supported` inputs |
| C16 | Whether any of C15's arms improves Total on BARAM is **unmeasured** in this lane; no fit, score, or metric call was performed. | `insufficient` |
| C17 | Whether conformal PID (E5) outperforms plain ACI on a banded settlement target. | `insufficient` — verbatim locator not captured before the lane was stopped |
| C18 | Sequence-architecture SOTA (PatchTST/DLinear/TFT/N-BEATS) and the Monash/M5 small-data evidence base. | `insufficient` — identified as locators only; **moot in practice** because all are `BLOCKED_DEPENDENCY` (C13) |
| C19 | Mixture-of-experts / regime-switching evidence base (Markov-switching AR for wind, FFORMA-style instance weighting). | `insufficient` in this lane's external set. Note `statsmodels 0.14.6` *is* present and does provide `MarkovAutoregression`/`MarkovRegression`, so this family is **not** dependency-blocked — it is evidence-blocked. |
| C20 | Whether the KPX/Jeju 6%/8% settlement band has a primary optimal-bidding literature that already characterises this exact action problem. | `insufficient` — one candidate locator surfaced (`new.kpx.or.kr/boardDownload.es?bid=0209&list_no=68418&seq=2`) but was **not** fetched or verified. This is the single highest-value unclosed question in the lane; see §9. |
| C21 | The local→online offset does not transfer across method classes (3.2× difference), so any ladder rung's local delta cannot be converted into an online delta. | `directly_supported` (L6) |
| C22 | No fresh confirmation surface exists: `fresh_confirmation_available=false`, `independent_confirmation_count=0`, 2024 lockbox `CONSUMED_UNAVAILABLE`. | `directly_supported` (L1 `claim_limits`) |

---

## 4. SOURCE FACT → PROVISIONAL MIGRATION HYPOTHESIS → LOCAL EVIDENCE NEEDED

Each block separates (a) source fact, (b) provisional BARAM migration hypothesis, (c) local evidence
needed. No block claims transfer.

### H1 — Forest kernel weights as the exact plug-in for the C9 curves  *(CONDITIONAL REOPEN, FD5-gated)*

- **(a) SOURCE FACT.** A forest's implicit kernel-type weighting function over training data "can be
  used for targets other than the original mean estimation", and the induced weights estimate the
  full conditional distribution for "arbitrary and potentially complicated targets of interest".
  `directly_supported` (E2, JMLR 23(333), 2022, `https://jmlr.org/papers/v23/21-0585.html`, retrieved 2026-08-09)
- **(b) PROVISIONAL MIGRATION HYPOTHESIS.** Take a forest fitted with an **ordinary** squared-error
  splitting rule on basis-time features (no task-aware splits — see §6 R2). Use `.apply()` leaf
  co-occurrence to form weights `w_i(x)`, and evaluate the C9 curves *exactly* as
  `Â_x(a)=Σ w_i V_i|a−Y_i|`, `Ŵ_h(a)=Σ w_i V_i Y_i 1{|a−Y_i|≤hC}`, then enumerate `a` on the frozen
  action grid. **Mechanism claim:** the incumbent 26/46-bin softmax Bayes decision reaches the same
  *operation* through a binned, unweighted-in-`Y` density, whereas the weighted-SAA form carries the
  `V_i·Y_i` weighting of FICR natively and never discretises `Y`. If the incumbent's window mass is
  in fact unweighted in `Y`, that is a *specification* gap, not an estimator preference.
- **(c) LOCAL EVIDENCE NEEDED.** A **zero-fit** source-reading audit of the incumbent action code:
  does the deployed Bayes action weight the in-band mass by `Y` (equivalently by `actual`), and does
  it include the `A_x/C` NMAE term with the correct relative scaling `1/(4μ_g)`? This is answerable
  by reading the action-selection function alone, with no data, no labels, and no score call.
- **Honest status.** `s17_n18_step_reward.md` §5 row 1 classifies "predict a conditional CDF/density,
  then enumerate `R_g(a;x)`" as a **duplicate** operation and states that "estimator-family changes
  alone lack a surviving local premise". H1 is therefore admissible **only** if the (c) audit finds
  the specification gap. Absent that finding, H1 is a re-run and must not be built. `[derived]`

### H2 — One-parameter adaptive conformal recentring as the *only* legal past-only nonstationarity correction

- **(a) SOURCE FACT.** ACI wraps any black box, uses a single continuously re-estimated parameter,
  and provably attains the target coverage frequency over long time intervals irrespective of the
  data-generating process. `directly_supported` (E4, `https://arxiv.org/abs/2106.00170`, retrieved 2026-08-09)
- **(b) PROVISIONAL MIGRATION HYPOTHESIS.** Under FD7, the scarce resource is *frozen choices*, not
  accuracy. A one-parameter online update that is provably robust to arbitrary drift is the
  cheapest possible adaptivity per degree of freedom. Migrate it not as a prediction-set method but
  as a **single scalar online state** driving the width used in the C9 window evaluation.
- **(c) LOCAL EVIDENCE NEEDED.** (i) A schema-only check that the update is computable strictly from
  labels available before each D-1 14:00 basis time (label-latency audit, FD1); (ii) whether the
  scalar's update trajectory is stable across the two learnable outer transitions, or oscillates in
  the way per-group blend weights already did (`AGENTS.md` records `g3: 1.00/1.00/0.15` oscillation
  over three folds).
- **Honest status.** **This is largely a re-run of the closed "post-processing the current
  representation" axis** and overlaps `m277_conditional_calibration_receipt.json` and
  `n401_cqr_decision_receipt.json`, which did not pass their gates. Its *only* non-duplicate feature
  is the 1-dof online form. C8 forbids claiming any FICR consequence from E4's coverage guarantee.
  `[derived]`

### H3 — Reclassify the COST5-SPO+ negative as a scope violation, not a family refutation

- **(a) SOURCE FACT.** "Our SPO+ loss function can tractably handle any polyhedral, convex, or even
  mixed-integer optimization problem with a **linear objective**."
  `directly_supported` (E3, Management Science 68(1):9–26, `https://par.nsf.gov/servlets/purl/10339524`, retrieved 2026-08-09)
- **(b) PROVISIONAL MIGRATION HYPOTHESIS.** BARAM's uncertainty enters non-linearly and
  discontinuously, so SPO+ never had a consistency guarantee here. The correct reading of the local
  `−0.011899` is "the surrogate was applied outside its proved class", which means **the local record
  does not license the broader conclusion that task-loss learning is dead on BARAM** — it licenses
  only that *this* surrogate on *that* five-action representation is dead.
- **(c) LOCAL EVIDENCE NEEDED.** None to establish the scope fact; it is already established. What
  would be needed to *act* on it is a task-loss method whose guarantees survive discontinuity — and
  `s17_n18_step_reward.md` §6 already shows the exact-enumeration alternatives fail the
  learnability/chronology gate on 90/179 dependent days.
- **Honest status.** **This is a bookkeeping correction with high evidential value and no
  build.** It repairs the reasoning record (FD3/FD5) without reopening the axis. It is the cleanest
  `contradicts_premise` finding in the lane.

### H4 — Regime/mixture structure is dependency-*available* but evidence-*absent*

- **(a) SOURCE FACT.** `statsmodels 0.14.6` is present. `directly_supported` (C14)
- **(b) PROVISIONAL MIGRATION HYPOTHESIS.** Regime-conditional conditional-law estimation (as opposed
  to the closed *regime-conditional policy switching*) is the one branch of (d) that is neither
  dependency-blocked nor obviously a renamed member selector, because it changes the *estimator* of
  `W_h`, not the action rule.
- **(c) LOCAL EVIDENCE NEEDED.** External: a primary wind-power result for regime-switching under
  comparable sample size. Local: whether `s17_n18` §5's "tree leaves choose a `(T,G)` policy or
  incumbent offset ⇒ renaming conditional policy/offset selection" also captures a regime-conditional
  *density*. My reading is that it does **not**, but I did not verify this against `m271_cycle4_bandpolicy.md`.
- **Honest status.** `insufficient` on both legs. Listed so the root does not mistake silence for closure.

---

## 5. RANKED CANDIDATE TABLE

Ranked by **evidence strength × local feasibility under ≤6 workers and a frozen dependency set**, per
the brief — explicitly *not* by reported literature accuracy. "Class" marks the brief's required
distinction.

| Rank | Candidate | Taxonomy | Class | Feasible now? | DOF | Relieves | Evidence |
|---|---|---|---|---|---|---|---|
| **1** | **H3 — record the SPO+ scope violation (C4) in the diagnostic lineage; no build** | (e) | **NEW FINDING, zero build** | Yes — documentation only | 0 | **FD5** (component/boundary reasoning integrated), **FD3** (lineage registry) | `contradicts_premise`, verbatim E3 |
| **2** | **H1(c) — zero-fit source audit of the incumbent action: is in-band mass `Y`-weighted?** | (b)+(e) | **NEW diagnostic** (the *build* it might license is a re-run) | Yes — read one function, no data | 0 | **FD5** primarily; FD3 | C9 `directly_supported`; the gap itself is `insufficient` until read |
| **3** | H1(b) — forest-kernel weighted-SAA arm, **only if rank 2 finds the gap** | (b) | CONDITIONAL REOPEN of a closed axis | Yes, zero new deps (C15) | Low–medium: forest size, leaf minimum, action grid, honesty | FD5 | mechanism `directly_supported` (E2); transfer `near_match_only`; effect `insufficient` |
| **4** | H2 — 1-dof adaptive conformal scalar | (b)+(e) calibration | **MOSTLY RE-RUN** of closed post-processing | Yes, numpy-only | **1** | FD7 (minimal DOF), FD5 | method `directly_supported` (E4); FICR consequence explicitly **not** supported (C8) |
| **5** | Keep tree ensembles as the point-regression backbone; do not swap to NN/foundation models | (a) | RE-CONFIRMATION | Yes (status quo) | 0 | FD7 | `directly_supported` (E1) + `directly_supported` (C13); *ranking* transfer `near_match_only` |
| **6** | H4 — regime-conditional **density** (statsmodels Markov switching) | (d) | POSSIBLY NEW, unverified | Dependency-OK | Medium–high | FD5 | `insufficient` both legs |
| — | Sequence / graph architectures (TFT, DLinear, PatchTST, N-BEATS, ST-GNN) | (c) | **BLOCKED_DEPENDENCY** | **No** — needs `torch` | n/a | — | Also **structurally demoted** by C10: cannot add a decision mechanism, only a marginal-risk estimate |
| — | Tabular foundation models (TabPFN family) | (a)+(b) | **BLOCKED_DEPENDENCY** | **No** — `torch` + weight download (download forbidden in this lane) | n/a | — | Would have been attractive under FD7 for its near-zero tuning surface; unavailable |
| — | Distributional boosting (NGBoost, treeffuser, LightGBMLSS, XGBoostLSS) | (b) | **BLOCKED_DEPENDENCY** | **No** | n/a | — | C13, L3 |
| — | Conformal libraries (MAPIE, crepes, venn-abers) | (b) | **BLOCKED_DEPENDENCY as libraries** | Algorithms re-implementable in numpy (C15) | — | — | C13 |
| — | Task-aware prescriptive trees/forests (Kallus–Mao, Stratigakos) | (e) | **CLOSED** | — | High | — | L2 §6 learnability gate |
| — | Any rung moving the action toward the conditional mean | (a) | **CLOSED**, 9 replications | — | — | — | L4, L6 |

---

## 6. REJECTED ALTERNATIVES AND WHY

- **R1 — Neural sequence/graph models.** Rejected twice over: `BLOCKED_DEPENDENCY` (C13) *and*
  structurally demoted by C10, which proves cross-hour coupling cannot help the decision. Even with
  `torch`, the argument for them would have to be "better marginal `W_h` estimate on ≤179 dependent
  days", which runs directly against E1's finding for ≪10K-sample tabular problems. `directly_supported` + `[derived]`
- **R2 — Task-aware split criteria (prescriptive trees/forests).** Already closed by `s17_n18` §6 on
  learnability: the split search cannot be frozen or selected honestly from 90/179 dependent days
  without another validation surface, and no fresh surface exists (C22). I explicitly do **not**
  re-propose it; H1 deliberately keeps **ordinary** splits precisely to stay outside this closure. `directly_supported`
- **R3 — Re-running SPO+ with a larger action grid.** `s17_n18` §5 already names this as "not a
  smallest discriminating follow-up" (it changes the action set and the learner simultaneously after
  the smaller problem failed), and C4 now shows the surrogate lacked a guarantee for this loss at any
  grid size. Rejecting it on both counts. `directly_supported`
- **R4 — Randomised / dithered actions to spread mass across bands.** Mathematically excluded by C11.
  This is worth recording because it is the intuitive first idea under a step reward and it is
  provably worthless at the population level. `directly_supported`
- **R5 — Treating BARAM as a bandit / RL problem.** Excluded by C12: full-information cost rows are
  observable, so exploration machinery buys nothing and adds large DOF. `directly_supported`
- **R6 — Quoting E1's or E2's benchmark margins as a BARAM expected gain.** Forbidden by the evidence
  contract and by C21 (offsets do not transfer across method classes). No number from any external
  source appears as a BARAM expectation anywhere in this file.
- **R7 — Declaring (c) and (d) "covered".** I did not establish the sequence/graph or MoE evidence
  base (C18, C19). I mark them `insufficient` rather than closing them.

---

## 7. SMALLEST DISCRIMINATING LOCAL EXPERIMENT PER SURVIVING CANDIDATE

Ladder per candidate: **cheap no-fit diagnostic → bounded screen → full strict comparison.** All of
these are *proposals for the root under IP@v3*; this lane executed none of them.

### Rank 2 / H1 — `Y`-weighting specification audit
- **Cheap no-fit diagnostic.** Read the incumbent action-selection function. Record whether in-band
  mass is weighted by `Y`, whether the NMAE term uses `1/C_g`, and whether the FICR term uses
  `1/(4μ_g)` with `μ_g` estimated past-only. **Cost:** minutes. **No data, no labels, no score.**
- **Falsifier.** If the incumbent already implements the exact C9 objective with `Y`-weighting and
  past-only `μ_g`, then **H1 and H3's build implications are dead** and rank 3 must not be built.
- **Inconclusive if.** The action is assembled across modules so that no single function determines
  the weighting; then escalate to a symbolic trace, not to a fit.

### Rank 3 / H1(b) — forest-kernel weighted-SAA arm (only if the audit finds the gap)
- **Cheap no-fit diagnostic.** On past-only training rows already loaded for an existing fit,
  compare the incumbent binned window mass against the leaf-weighted `Y`-weighted window mass for a
  handful of frozen `x`, reporting only their *disagreement magnitude* — no score call.
- **Bounded screen.** Single frozen configuration (no HPO): fixed `n_estimators`, fixed
  `min_samples_leaf`, fixed action grid identical to the incumbent's. One outer transition only.
- **Full strict comparison.** Strict prequential, expanding past-only origins, both learnable
  transitions, fixed policy, alignment keys `(fold_id, group_id, forecast_kst_dtm)`, with the
  component/tier-migration receipt FD5 demands.
- **Treatment / control.** Treatment = leaf-weighted exact-SAA action. Control = incumbent action,
  **identical features, identical action grid, identical fit surface** — the estimator of `W_h` is
  the only thing that changes.
- **Degrees of freedom.** 4 (`n_estimators`, `min_samples_leaf`, `max_features`, honesty split), all
  predeclared and frozen before the strict run. **Trial budget: 1.**
- **Falsifier.** Total delta ≤ 0 on the first learnable transition, **or** FICR falls while `1−NMAE`
  rises (the exact C5 signature of trading settlement for point accuracy).
- **Inconclusive if.** The delta is within the fit-seed noise band, or the two transitions disagree
  in sign — in which case, under C22, it must be recorded as inconclusive, **not** rescued.

### Rank 4 / H2 — 1-dof adaptive conformal scalar
- **Cheap no-fit diagnostic.** Label-latency audit only (FD1): confirm every label entering the
  update is available before the D-1 14:00 basis time. Schema and timestamps only.
- **Bounded screen.** Trajectory-stability check of the single scalar across the two transitions,
  reported as the parameter path, not as a score.
- **Full strict comparison.** Only if the path is monotone/stable; otherwise stop.
- **Falsifier.** Any label used by the update post-dates the basis time (fatal, leakage), **or** the
  scalar oscillates across folds in the manner already observed for per-group blend weights.
- **Inconclusive if.** Stable path but Total delta inside noise — record and stop; do **not** convert
  a coverage improvement into a Total claim (C8).

### Rank 5 — backbone re-confirmation
- No experiment. It is a decision to *not* spend the budget, justified by E1 + C13.

---

## 8. WHAT IS NEW MECHANISM VS RE-RUN OF THE CLOSED AXIS (explicit, as required)

| Rung | Verdict |
|---|---|
| H3 (SPO+ scope violation) | **NEW** — a source-level scope fact never recorded locally; changes the interpretation of an existing negative without proposing a build. |
| H1(c) audit | **NEW diagnostic** — a specification question about the incumbent, not a new model. |
| H1(b) forest-kernel SAA | **RE-RUN** of "predict a conditional density then enumerate", *unless* the H1(c) audit surfaces the `Y`-weighting gap, in which case it becomes a specification fix. Conditional, not asserted. |
| H2 adaptive conformal | **MOSTLY RE-RUN** of "post-processing the current representation". Only its 1-dof online form is new, and its guarantee is about coverage, not Total. |
| Rank 5 backbone | **RE-CONFIRMATION**, no new mechanism. |
| H4 regime density | **UNDETERMINED** — cannot classify without the evidence in C19. |

I want to be blunt about the shape of this result: **four of six rungs are re-runs or
re-confirmations.** Given C9–C12 (the decision layer is analytically closed) and C13 (the modern
model zoo is unavailable), that is the honest state of the model axis, and I do not think the
remaining `0.660000 − 0.6314827 = 0.0285` is reachable from S7 alone.

---

## 9. UNKNOWNS

Explicit `insufficient` items. The lane was stopped by root instruction after 5 of ≤16 permitted
external sources; these are the questions that budget would have addressed.

1. **KPX / Jeju 6%–8% settlement-band optimal-bidding literature (C20).** Highest value unclosed
   item. If a primary source characterises the optimal action under *this exact* two-band scheme, it
   would be the only true scope match on the `metric` column in §2. One unverified locator surfaced:
   `https://new.kpx.or.kr/boardDownload.es?bid=0209&list_no=68418&seq=2`. **Not fetched, not verified,
   not evidence.**
2. **Conformal PID (E5) verbatim content (C17)** — located but not extracted; all claims downgraded.
3. **Sequence/graph small-data evidence base (C18)** — moot for feasibility, not for the record.
4. **MoE / regime-switching wind evidence (C19)** — the one dependency-available unexplored family.
5. **Whether the incumbent action already `Y`-weights its in-band mass** — decidable locally in
   minutes; blocks the rank-2/3 fork.
6. **Finite-sample validity of the C10 additivity argument.** C10 holds "once the population group
   constants are fixed". FICR's denominator `Σ 4·actual` and NMAE's row count are realised over a
   ~90-day fold, so they are random, not constant. I believe the induced coupling is second-order and
   action-independent (neither denominator depends on `a`), but I did **not** verify the magnitude.
   `insufficient`.
7. **Detectability floor.** Whether a `+0.001`-scale Total change is even distinguishable on 90/179
   dependent issuance days. `s17_n18` §6 raises this; no source in my set answers it. This is the
   binding question for the entire ladder and it is a **validation** question, not a model question.

---

## 10. DS / IP IMPLICATIONS

**For `DS@v5`:**

1. **Do not open S7 as a model-family search.** C9–C12 reduce (a)–(e) to a single estimator question,
   and C13 reduces the feasible estimator set to tree ensembles plus numpy. A "try more models" batch
   would consume the multiplicity budget FD7 says is nearly exhausted, for a mechanism that C10 shows
   cannot add decision value.
2. **Promote the two zero-cost items first.** H3 (record the SPO+ scope violation) and H1(c) (the
   `Y`-weighting audit) cost approximately nothing, require no fit, and one of them can *delete* the
   rest of the ladder. Schedule them before any S7 compute is allocated.
3. **Adopt DOF-per-rung as a first-class ranking field**, not a footnote. Under FD7 with
   `independent_confirmation_count = 0`, a rung's frozen-choice count is its true price. This is why
   H2 ranks above H4 despite weaker mechanism evidence.
4. **Make FD5's component receipt a precondition, not a report.** C5's signature (`1−NMAE` up, FICR
   down) is the failure mode that pooled Total deltas hide. Every S7 comparison should be *required*
   to emit component/group/fold/lead/tier-migration fields before it is adjudicated, which is exactly
   FD5's `smallest_diagnostic`.
5. **Record the closure asymmetry.** `s17_n18` closed the step-reward *estimator* axis. C4 now shows
   one of its inputs (the SPO+ negative) was a scope violation. This does **not** reopen the axis —
   `s17_n18` §6's learnability gate is independent and still binds — but the lineage should not carry
   "task-loss learning fails on BARAM" as a general finding.

**For `IP@v3`:**

6. **The dependency freeze is load-bearing and should be stated as a modelling constraint, not an
   ops detail.** It is the single largest determinant of this ladder. If the user ever authorises
   `torch`, the (b)/(c)/(d) branches change materially — but note that C10 still caps what (c) could
   contribute, so a `torch` authorisation should be justified on distributional grounds, not
   sequence-architecture grounds.
7. **No rung in this ladder is authorised to claim independent confirmation** (C22), and no local
   delta may be converted to an online delta (C21). Any receipt phrasing must reflect both.
8. **Prohibitions honoured in this lane:** 0 model fits, 0 score/metric/policy calls, 0 reads of
   `actual_kwh`/predictions/model results/2024/test values, 0 external data or weight downloads,
   0 dependency changes, 0 repository writes outside this file, 0 Dacon/account/remote/git actions,
   0 lockbox reads, 0 delegations to further subagents. The `.venv` probe read package **versions
   only** via `importlib.metadata`, imported no project module, and touched no data.

---

## 11. WHAT I DID NOT DO

- I did not spend the full 16-source budget: **5 external sources** were fetched and quoted. The lane
  was stopped by explicit root instruction to write immediately.
- I did not fetch or verify the KPX locator in §9.1, the conformal-PID text, the sequence-model
  literature, or the regime-switching literature.
- I did not open any data body, run any fit, or compute any score.
- I did not re-derive the frozen foundation facts; I used them as given after hash verification.
