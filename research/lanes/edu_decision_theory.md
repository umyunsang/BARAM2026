# Lane `EDU-DECISION-THEORY` — the named object is the **modal interval / weighted modal-interval functional**, and it is a RE-ALLOCATION device, not an accuracy device

**Lane:** `EDU-DECISION-THEORY`
**Exclusive output:** `research/lanes/edu_decision_theory.md`
**Retrieval window:** 2026-08-09, bounded session, terminated early by root instruction ("STOP RESEARCHING AND WRITE NOW").
**Sources used:** 6 fetched primary teaching/primary-literature documents + 4 bibliographically identified but **not fetched** (tagged `insufficient`). Cap was 14; the lane stopped at 6 fetched by root order, so §7 lists the unclosed retrieval explicitly.
**Actions performed:** 0 model fits, 0 score/metric calls, 0 target/test/2024-lockbox reads, 0 downloads of data or weights, 0 Dacon interaction, 0 remote model inference, 0 delegation, 0 repository writes outside this file.

---

## 0. VERDICT (read this first)

`NAME_FOUND_BUT_AXIS_IS_REALLOCATION_ONLY`

Three findings, in decreasing confidence:

1. **The estimation problem "choose `a` to maximise `P(|a-Y| <= h | X)`" has an exact textbook name: the midpoint of the *modal interval of length 2h* of the conditional law.** It is the Bayes rule for the loss `L(theta,a) = 1{|theta-a| > c}`, which the same teaching literature also names the **"large deviation loss"**. It is *not* mean regression, *not* quantile regression, and *not* classification-with-sharpening. Its regression form is the named family **modal regression / modal-interval (conditional-shorth) regression**. `[directly_supported]`

2. **The actual-weighting in FICR has an exact textbook treatment too, and it is not a new estimator: weighting the loss by `w(y)` is *identical* to leaving the loss alone and tilting the distribution by `w`.** Therefore the FICR-optimal action is the midpoint of the modal interval of the **energy-tilted conditional density** `q(y|x) ∝ y · 1{y >= 0.1C} · f(y|x)`, not of `f(y|x)`. `[directly_supported]` for the tilting theorem, `[derived]` for the BARAM instantiation.

3. **DECISIVE FOR THE ROOT'S SHARPENED QUESTION: a mode-type estimator CANNOT improve MAE. It can only re-allocate.** The MAE-minimising action *is by definition* the conditional median (textbook derivation, §2). Any functional that is not the conditional median has weakly larger expected absolute error under the same conditional law. Modal/HDR actions deliberately move *away* from the median toward the density peak, so on the accuracy axis they are weakly harmful by construction. Given the root's new measurement that the binding constraint is raw point accuracy and that re-allocation is exhausted, **this lane's entire estimator family cannot close the 0.0285 gap.** `[directly_supported]` for the median result, `[derived]` for the consequence.

**One genuinely new, accuracy-side item survives** (§4.2): the NMAE numerator is summed **only over rows with actual >= 10% of capacity**, and that eligibility indicator is a `w(y)` weight in exactly the sense of the tilting theorem. So the MAE-optimal action for the official NMAE is the median of the **truncated** law `P(Y | X, Y >= 0.1C)`, **not** the median of `P(Y|X)`. If the pipeline currently targets the untruncated median/mean, this is free accuracy with no new information and no new model. This is the single cheapest local test (§6.0).

---

## 1. Evidence contract and tag definitions

Every material claim below carries exactly one tag:

- `directly_supported` — stated in a fetched primary source, with an exact locator (document, section/equation/exercise number).
- `contradicts_premise` — the fetched source directly defeats a premise of the task or of a proposal here.
- `near_match_only` — a real result, but under a materially different loss, sampling design, or asymptotic regime than BARAM.
- `insufficient` — not established by the fetched source set in this bounded window; identified bibliographically only, or not retrieved at all.
- `derived` — algebra performed in this file on top of a tagged source fact. Not an empirical claim.

**Separation rule enforced throughout:** every numbered item states (a) the *source fact*, (b) the *provisional BARAM migration hypothesis*, (c) the *local evidence needed*. **No source effect size is ever quoted as a BARAM expected gain.** No number in any cited source is transferred to BARAM.

---

## 2. Source register (6 fetched)

| ID | Source (course / venue, year, exact title) | Stable URL | Exact locator used | Tag |
|---|---|---|---|---|
| **E1** | **Thomas S. Ferguson, *Mathematical Statistics: A Decision Theoretic Approach*, UCLA course materials (Dept. of Mathematics), "Solutions to the Exercises of Section 1.8"** | https://www.math.ucla.edu/~tom/MathematicalStatistics/Sec18.pdf | Ex. **1.8.1** (squared error -> mean); Ex. **1.8.2** (absolute error -> median); Ex. **1.8.3** (asymmetric linear loss `k1/k2` -> the `p = k1/(k1+k2)` posterior quantile); Ex. **1.8.5** (finite-width 0-1 loss -> **midpoint of the modal interval**); Ex. **1.8.6** (**weighted loss == tilted prior**) | `directly_supported` |
| **E2** | **Larry Wasserman, CMU 36-705 Intermediate Statistics, "Lecture Notes 14" (decision theory)** | https://www.stat.cmu.edu/~larry/=stat705/Lecture14.pdf | §1.1 "The Risk Function", loss catalogue: names `L(theta, theta-hat) = I(|theta-hat - theta| > c)` **"large deviation loss"**, listed alongside squared error, absolute error, Lp, zero-one, KL | `directly_supported` |
| **E3** | **Y.-C. Chen, C. R. Genovese, R. J. Tibshirani, L. Wasserman (2016), "Nonparametric Modal Regression", *Annals of Statistics* 44(2), 489–514 (Carnegie Mellon University)** | https://www.stat.berkeley.edu/~ryantibs/papers/modal-aos.pdf | Abstract; §1 contributions 1–6; §2 review (linear modal regression `Mode(Y|X=x)=beta0+beta'x`, Lee 1989 / Sager–Thisted 1982 / Yao–Li 2014); Eq. (4) joint KDE; Eq. (5) modal set `{y : p_{y,n}=0, p_{yy,n}<0}`; **Algorithm 1 partial mean-shift**; Lemma 1 `grad m_j = -p_{yx}/p_{yy}`; Theorem 2; **assumption (A3) `|p_yy| > lambda_2 > 0`**; §1 item 4 "prediction sets ... can be smaller than those based on the regression function" | `directly_supported` |
| **E4** | **Stefan Wager, *Causal Inference: A Statistical Learning Approach*, Stanford University (course book for STATS 361 Causal Inference), draft 26 Nov 2025** | https://web.stanford.edu/~swager/causal_inf_book.pdf | **Chapter 5 "Policy Learning"**, §5.1 "Policy evaluation", §5.2 "Empirical-welfare maximization" (contents, pp. 57–68); Ch. 3 "Doubly Robust Methods" | `directly_supported` (structure/existence of the named family); `insufficient` (chapter body not read in window) |
| **E5** | **Richard J. Samworth (2018), "Recent Progress in Log-Concave Density Estimation", *Statistical Science* 33(4), 493–509 (Univ. of Cambridge, Statistical Laboratory; used in Cambridge Part III teaching)** | http://www.statslab.cam.ac.uk/~rjs57/STS666.pdf | Abstract: log-concave MLE is "a fascinating alternative to traditional nonparametric smoothing techniques, such as kernel density estimation, **which require the choice of one or more bandwidths**"; §1 Introduction: shape-constrained MLE "does not require the choice of any tuning parameter" | `directly_supported` |
| **E6** | **Ryan Tibshirani (with Larry Wasserman), CMU 36-708 Statistical Machine Learning, Spring 2017, "Nonparametric Regression (and Classification)"** | https://www.stat.cmu.edu/~ryantibs/statml/lectures/nonpar.pdf | §1.1 "Basic setup": `f0(x) = E(Y|X=x)` "is called the regression function"; the additive-noise sampling model and the explicit warning that independence of `epsilon_i` and `x_i` "is a pretty strong assumption, and you should think about it skeptically" | `directly_supported` |

Additional course-index anchors located but not used as claim sources: CMU 36-708/statml course index https://www.stat.cmu.edu/~ryantibs/statml/ ; Wasserman `=sml` notes index https://www.stat.cmu.edu/~larry/=sml/ ; Wasserman nonparametric regression notes https://www.stat.cmu.edu/~larry/=sml/nonpar-regression.pdf ; Wager teaching index https://web.stanford.edu/~swager/teaching.html.

---

## 3. The exact chain of textbook results (the answer to the central question)

### 3.1 The four Bayes actions, from one course text (E1)

All four come from the **same** UCLA Ferguson exercise set, which makes the contrast airtight:

| Loss | Bayes action | Locator |
|---|---|---|
| `(theta - a)^2` | **posterior mean** | E1 Ex. 1.8.1 — "`E(Z-b)^2 = Var(Z) + (EZ-b)^2` obviously takes on its minimum value of `Var(Z)` when `b = EZ`" |
| `|theta - a|` | **posterior median** | E1 Ex. 1.8.2 — proves `f(b)=E|Z-b|` is minimised at any median `b0` |
| `k1|theta-a|` if `a<=theta`, `k2|theta-a|` if `a>theta` | **posterior `p`-quantile, `p = k1/(k1+k2)`** | E1 Ex. 1.8.3 ("Rule: ... estimate `theta` as the `p`th quantile of the posterior distribution") |
| **`0` if `|theta-a| <= c`, `1` if `|theta-a| > c`** | **midpoint of the MODAL INTERVAL of length `2c`** | **E1 Ex. 1.8.5** |

**Verbatim, E1 Ex. 1.8.5:** *"An interval of length 2c, say (b − c, b + c), is said to be a modal interval of length 2c for the distribution of a random variable θ, if P(b − c ≤ θ ≤ b + c) takes on its maximum value out of all such intervals. For the loss function L(θ, a) = 0 if |θ − a| ≤ c, 1 if |θ − a| > c, EL(θ, a) = P(|θ − a| > c) = 1 − P(a − c ≤ θ ≤ a + c) is minimized if a is chosen to be the midpoint of the modal interval of length 2c. **Rule:** In the problem of estimating a real parameter θ with the above loss function, a Bayes decision rule with respect to a given prior is to estimate θ as the midpoint of the modal interval of length 2c of the posterior distribution of θ given the observations."* `[directly_supported]`

**Independent naming confirmation:** CMU 36-705 Lecture 14 (E2) lists `L(theta, theta-hat) = I(|theta-hat - theta| > c)` in its standard loss catalogue and calls it **"large deviation loss"**. `[directly_supported]`

**(a) Source fact.** The functional "argmax over `a` of `P(|a - Y| <= h)`" is the *modal interval midpoint*; it is a **fourth, distinct** Bayes functional, at the same level of the taxonomy as mean, median and quantile. It is **not** the mode (the mode is the `c -> 0` degenerate limit), and it is **not** any quantile.
**(b) BARAM migration hypothesis.** BARAM's `FICR` unit function `u(e) = 1{e <= 0.06C} + 3·1{e <= 0.08C}` is a **two-tier** finite-width hit loss, i.e. a positive combination of two large-deviation losses at `c = 0.06C` and `c = 0.08C`. Its Bayes action is therefore the maximiser of a *weighted sum of two modal-interval probabilities*, an object of the same class, and provably **not** any mean, median, quantile, or mode.
**(c) Local evidence needed.** None — this is exact algebra on the official formula; it needs no fit. The only thing to verify locally is that the deployed policy's argmax is being taken over the same weighted-two-band objective (§6.2).

> **Note on prior local work.** `research/lanes/s17_n18_step_reward.md` §2 already records Brehmer & Gneiting (2021) Eq. (9) `MI_c(F) = argmax_{[a,b]} {F(b) - F(a-) : b - a <= 2c}` and Eq. (11). **E1 Ex. 1.8.5 is the same functional, 40+ years earlier, in an undergraduate/graduate course text, with the name "modal interval".** This lane therefore *confirms* rather than extends the naming — the object was already correctly identified locally. `[directly_supported]`, and this is the honest scope limit of finding (1).

### 3.2 The weighted variant is NOT a new estimator — it is a TILT (E1 Ex. 1.8.6)

**Verbatim, E1 Ex. 1.8.6:** *"If τ is a prior distribution for θ with density g(θ), and if c = Ew(θ) = ∫w(θ)g(θ)dθ < ∞, then g*(x) = (1/c)w(θ)g(θ) is a density of a prior distribution, call it τ*, for θ. If d is Bayes with respect to τ for loss L(θ,a) = (θ−a)^2 w(θ), then d is Bayes with respect to τ* for loss L*(θ,a) = (θ−a)^2, **because in either case d minimizes ∫(θ−d)^2 f(x|θ) w(θ) g(θ) dθ**."* `[directly_supported]`

The reason given in the source — *"because in either case d minimizes ∫ L(θ,d) f(x|θ) w(θ) g(θ) dθ"* — is **loss-agnostic**: it uses only that the integrand factorises as `L · w · g`. It applies verbatim with `(theta-a)^2` replaced by the large-deviation loss. `[derived]` from the source's own stated reason.

**(a) Source fact.** A `w`-weighted loss and a `w`-tilted prior give the *same* Bayes rule.
**(b) BARAM migration hypothesis (the answer to task item 3).** FICR weights each valid row by the actual production `y`. Therefore:

> **The FICR-optimal action is the midpoint of the (two-tier) modal interval of the ENERGY-TILTED conditional density**
> `q(y | x) = y · 1{y >= 0.1C} · f(y|x) / E[ Y · 1{Y >= 0.1C} | x ]`
> **— not of `f(y|x)`.**

The tilt `q ∝ y·f` is a *rightward* shift: it upweights high-production outcomes proportionally to their energy. Because a right-tilt of a right-skewed wind-power conditional law moves probability mass toward capacity, the energy-tilted modal interval sits **above** the untilted one. `[derived]`
**(c) Local evidence needed.** On saved champion artifacts, with **zero fit**: for each row take the existing binned conditional probability vector `p_k` over bin centres `y_k`, form `p_k^{tilt} ∝ y_k · p_k · 1{y_k >= 0.1C}`, re-run the *existing* settlement argmax on `p^{tilt}` instead of `p`, and score. Cheap, no model, no lockbox. Falsifier: if the resulting action is within a rounding tolerance of the current action on >95% of rows, the pipeline already implements the tilt implicitly and there is nothing here.

### 3.3 The sharpening exponent `T` and the tilt are DIFFERENT operations

Sharpening is `p_k -> p_k^{1/T} / Z` (a power/temperature tilt on the *probability*). Energy tilting is `p_k -> y_k p_k / Z` (a *linear-in-outcome* tilt). They do not commute and they are not reparameterisations of each other. `T = 0.75` being the argmax of a 25-point and a 12-point grid establishes the optimum **only within the power family**; it says nothing about the linear-in-`y` family. `[derived]`. This is a live, unexplored one-dimensional axis (see §6.1), though §4.1 bounds how much it can be worth.

---

## 4. THE ROOT'S SHARPENED QUESTION: accuracy improvement or re-allocation only?

### 4.1 Answer: RE-ALLOCATION ONLY. Stated plainly.

**(a) Source fact.** E1 Ex. 1.8.2 proves that `b -> E|Z - b|` is minimised at a median of `Z`. E2 §1.1 confirms the same taxonomy. `[directly_supported]`

**(b) Derivation, exact and unavoidable.** NMAE is capacity-relative mean absolute error. For a fixed conditional law and a fixed row, the action minimising the expected contribution to the MAE numerator **is the conditional median**, by E1 Ex. 1.8.2. Every other functional — mean, mode, modal-interval midpoint, HDR centre, quantile at `p != 0.5`, sharpened argmax — has **weakly larger** expected absolute error under that same law. Hence:

> **No change of action functional can push 1-NMAE above the value attained by the (correctly conditioned) conditional median of the conditional law we already estimate.** The conditional-median MAE is a hard ceiling on the entire "choose a better action" axis. `[derived]`

And in the specific direction of this lane: modal / HDR / modal-interval actions move *deliberately away* from the median toward the density peak. On a right-skewed conditional law the mode is **below** the median, so a mode-type action makes MAE **worse**, not better. `[derived]`

**(c) Consistency with the local measurements in the brief.** The three independent routes reported by the root all show the same signature — 1-NMAE **rises** by ~`+0.0029` while FICR **falls**. That `+0.0029` is exactly the size of the accuracy re-allocation available from moving the action toward a proper/L1-type point, and it is the *entire* functional-axis accuracy budget observed so far. The root now needs `1-NMAE` from `0.85848` to about `0.8797`, i.e. `+0.0212`. **`+0.0029` is 13.7% of `+0.0212`.** The functional axis is ~7x too small. `[derived]` — this is arithmetic on numbers the root supplied, not a source effect size.

**Conclusion the root asked for, unhedged:** *modal/HDR regression is the correct name for the object, and it is only a re-allocation device. It cannot supply the accuracy the score now requires. Under the root's own elasticity measurement this lane's family is not on the critical path.* `[derived]`

**What would falsify this conclusion.** Exactly one thing: a demonstration that the current conditional law `f(y|x)` is *not* the one whose median we are comparing against — i.e. that a mode-type criterion, used as a **training objective** rather than as a **decision rule**, produces a *different and better* conditional law. That is the one loophole, and it is addressed in §4.3.

### 4.2 THE ONE SURVIVING ACCURACY ITEM — the eligibility filter is a tilt, so the MAE-optimal action is a TRUNCATED median

**(a) Source fact.** E1 Ex. 1.8.6: `w`-weighted loss == `w`-tilted distribution, same Bayes rule; reason given is loss-agnostic. `[directly_supported]`

**(b) BARAM migration hypothesis.** The official NMAE numerator sums `|a - y|/C` **only over rows with `y >= 0.1C`**, and the denominator (count of eligible rows) does **not** depend on `a`. Therefore the per-row objective is

`min_a E[ 1{Y >= 0.1C} · |a - Y| | X = x ]`

which by E1 Ex. 1.8.6 with `w(y) = 1{y >= 0.1C}` equals the unweighted `L1` problem under the **truncated** law `P(Y | X, Y >= 0.1C)`. By E1 Ex. 1.8.2 its solution is

> **`a*_NMAE(x) = median of P(Y | X = x, Y >= 0.1C)`, NOT `median of P(Y | X = x)`.**

These differ by a strictly positive amount whenever `P(Y < 0.1C | x) > 0`, and the difference grows with that probability. Concretely, if `P(Y < 0.1C | x) = m`, the truncated median is the `(1+m)/2`-quantile of the untruncated law — i.e. **an upward quantile shift of `m/2` in probability units.** For low-wind hours where `m` can be 0.2–0.4, that is a shift from the 50th to the 60th–70th conditional percentile. `[derived]`

This is the only mechanism found in this lane that raises 1-NMAE **without new information and without a refit**, because the model already emits the full binned conditional distribution.

**(c) Local evidence needed.** §6.0 — the single cheapest test.

**Sharp caveat, stated because the evidence contract requires it.** Eligibility is judged on the **actual**, so this is a *selection on the outcome*. Ineligible rows are dropped from scoring regardless of what we predict; therefore predicting high on a row that turns out ineligible costs nothing in NMAE. That asymmetry is real and is exactly what the truncated median exploits. **But** the same rows do enter FICR's validity set, and the brief states FICR is over "valid" rows; if "valid" and "eligible" are the same set, the tilt is consistent across both components; if they differ, the two components pull the action apart and the unified action of §4.4 must be used. **The identity of the two row sets is `insufficient` in this lane** — it must be read off `inputs/notebooks/metric_official.ipynb` before acting. This is the first thing to check and it is a 2-minute read.

### 4.3 The one loophole — mode-type criteria as a TRAINING objective (mixture / regime structure)

**(a) Source fact.** E3 abstract: *"Modal regression estimates the local modes of the distribution of Y given X = x, instead of the mean, as in the usual regression sense, and **can hence reveal important structure missed by usual regression methods**."* E3 §1: *"the conditional mean both fails to capture the major trends present in the response, and produces unnecessarily wide prediction bands."* E3 §1 contribution 4: modal-regression prediction sets *"can be smaller than those based on the regression function."* E3 §9/§8 relates modal regression to **mixture regression** and **density ridge estimation**. `[directly_supported]`

**(b) BARAM migration hypothesis.** The root's ensemble facts — oracle per-row member selection worth `+0.094`, member spread exceeding the full band on 40.2% of rows — are the empirical signature of a **multi-modal / mixture-structured** conditional law: several members sit on different modal manifolds. E3's Lemma 1 (`grad m_j = -p_{yx}/p_{yy}`) and Theorem 2 give the modal manifolds a *smooth, low-dimensional* structure in `x`, and E3 documents that manifolds **bifurcate and merge** as `x` varies (Fig. 3: "after `X = x2` the conditional density becomes unimodal and the first (left) mode disappears"). If the 40.2% high-spread rows are bifurcation regions, then a *manifold-identity* covariate — which modal branch is active — is genuine **new information for the mean/median model**, not a re-allocation. `[near_match_only]`: E3's setting is i.i.d. `(X,Y)` with a smooth joint density and no chronological drift; BARAM is a drifting day-ahead operational series.
**(c) Local evidence needed.** No fit required for step 1: on saved artifacts, restrict to the 40.2% high-spread rows and ask whether the *realised* actual falls near one member consistently (bimodal, manifold structure) or scatters uniformly across the member spread (pure realization noise). **The root has already effectively answered this**: "two independent learned gates failed to capture any of it, so that headroom is realization noise." **That measurement is `contradicts_premise` for this loophole.** Recording it as such: the mixture/manifold reading of the ensemble spread is locally falsified, and this lane does not reopen it.

### 4.4 The unified BARAM Bayes action (exact, for completeness)

Combining §3.1, §3.2 and §4.2. Let `V = 1{Y >= 0.1C}`, `e = |a - Y|/C`, `lambda_A` and `lambda_F` the positive constants converting each component to `Total` units. The per-row objective is

`max_a  E[ V · ( lambda_F · Y · (1{e <= 0.06} + 3·1{e <= 0.08}) - lambda_A · e ) | X = x ]`

By E1 Ex. 1.8.6 this is the Bayes action under the **single tilted measure** `q(y|x) ∝ 1{y >= 0.1C} · f(y|x)`, with an objective that is (a positive multiple of) `lambda_F · E_q[Y·u] - lambda_A · E_q[|a-Y|]/C`. Its solution is a **regularised modal-interval midpoint**: the modal-interval term pulls toward the energy-weighted density peak, the `L1` term pulls toward the truncated median, and `lambda_A/lambda_F` sets the trade. `[derived]`

**This is the object the deployed `T`/`G` policy is approximating heuristically.** The `G` parameter plays the role of `lambda_A/lambda_F`; `T` plays the role of a variance-reduction regulariser on the plug-in density (§5.2). Nothing in the taught theory says the heuristic is wrong; §5.2 says it is the *expected* thing to do.

---

## 5. Named estimator families, ranked, with assumptions

### Rank 1 — **Modal-interval regression (conditional shorth) / conditional-mode regression**

- **Definition & primary teaching source:** E1 Ex. 1.8.5 (functional); E2 §1.1 (loss name "large deviation loss"); E3 (regression form, estimators, theory).
- **Named estimators, all from E3:** (i) **plug-in KDE modal regression** — joint KDE Eq. (4), modal set Eq. (5) `{y : p_{y,n}(x,y)=0, p_{yy,n}(x,y)<0}`, attributed to Scott (1992); (ii) **partial mean-shift** (E3 Algorithm 1, Einbeck & Tutz 2006) — the iteration `y <- sum_i Y_i K((x-X_i)/h)K((y-Y_i)/h) / sum_i K(...)K(...)`, which E3 notes "is indeed a gradient ascent update on the function `f(y) = p_n(x,y)` (for fixed `x`), with an implicit choice of step size"; (iii) **modal linear regression** `Mode(Y|X=x) = beta_0 + beta'x` (E3 §2, attributed to Lee 1989, Sager & Thisted 1982, Yao & Li 2014); (iv) **local polynomial mode smoothing** (Yao, Lindsay & Li 2012, via E3 §2); (v) **EM-based mode hunting** (Yao 2013, via E3 §2).
- **Assumptions (E3 §4, verbatim locators):** (A1) joint density `p` in `BC^4`; (A2) modal manifolds factorise into connected curves `{(x, m_j(x))}`; **(A3) there exists `lambda_2 > 0` such that `|p_yy(x,y)| > lambda_2` at every `(x,y)` with `p_y(x,y)=0`** — i.e. **the density must be sharply curved at its mode**; (K1) smooth kernel with finite second moments; (K2) VC-type kernel class.
- **Why it is harder than mean/quantile regression, from the source:** E3 §2 states that because the objective `f(y) = p_n(x,y)` "is generically nonconcave, we are not guaranteed that gradient ascent will actually attain a (global) maximum, but it will converge to critical points under small enough step sizes"; the target `M(x)` is **multi-valued**, so E3 must measure error in **Hausdorff distance** (Theorem 2) rather than a norm; and E3 needs a full **fourth-derivative** smoothness assumption (A1) where mean regression needs two.
- **BARAM applicability:** `contradicts_premise` on assumption (A3). The root reports predicted IQR `0.067`–`0.119` in capacity fraction against a paying band of half-width `0.06` (full width `0.12`). A conditional law whose IQR is at or wider than the entire band is **nearly flat over the band**, i.e. `|p_yy|` at the mode is near zero, i.e. `lambda_2 ≈ 0`. E3's rate constants and its uniqueness of manifold factorisation both degrade as `lambda_2 -> 0` (E3 explicitly: "this factorization is unique if the second derivative `p_yy(x,y)` is uniformly bounded away from zero"). **The modal-interval functional is weakly identified on BARAM's conditional law.** This is the cleanest theoretical explanation available for why exact plug-in Bayes actions underperform a tuned heuristic locally.
- **Smallest local diagnostic (no fit):** §6.2 flatness/curvature probe.
- **Accuracy verdict:** re-allocation only (§4.1).

### Rank 2 — **Weighted / tilted Bayes actions (energy-tilted and eligibility-truncated)**

- **Definition & source:** E1 Ex. 1.8.6.
- **Assumptions:** only `E[w(Y)] < infinity`, which holds trivially for `w(y) = y·1{y >= 0.1C}` on a bounded support `[0, C]`.
- **Two instantiations:** (i) **eligibility truncation** -> truncated median for NMAE (§4.2) — the only accuracy-relevant item in this lane; (ii) **energy tilt** `q ∝ y·f` for FICR (§3.2).
- **BARAM applicability:** `directly_supported` for the theorem, `insufficient` for whether the pipeline already does it.
- **Smallest local diagnostic:** §6.0 and §6.1. **Both are pure arithmetic on stored bin probabilities. Zero fit.**
- **Accuracy verdict:** (i) genuinely accuracy-improving with no new information, magnitude unknown; (ii) re-allocation.

### Rank 3 — **Shape-constrained (log-concave / unimodal) conditional density estimation**

- **Definition & source:** E5 (Samworth 2018, Cambridge). Abstract: log-concave MLE is "a fascinating alternative to traditional nonparametric smoothing techniques, such as kernel density estimation, **which require the choice of one or more bandwidths**"; §1: shape-constrained MLE "does not require the choice of any tuning parameter". `[directly_supported]`
- **Why it belongs in a decision-theory lane.** A log-concave density is **unimodal**, so the modal interval of any width is **unique** and its midpoint is a well-defined, stable functional. Log-concavity is the exact structural hypothesis that removes the (A3) pathology of Rank 1: it converts a non-concave argmax into a well-posed one. `[derived]`
- **BARAM applicability:** `near_match_only`. E5 is about *unconditional* density estimation; BARAM needs the *conditional* law, and the fetched window did not reach E5's conditional/regression sections. Also, BARAM's conditional wind-power law is boundary-piled at `0` and at capacity `C` and is plausibly **not** log-concave; imposing log-concavity would then be a bias-inducing misspecification (which, per §5.2, may nonetheless *help*).
- **Smallest local diagnostic (no fit):** on saved per-row binned distributions, test whether `log p_k` is concave in `k` (finite second differences `<= 0`) after excluding the two boundary bins. Report the fraction of rows passing. If a large majority pass, sharpening with `T` is already doing approximately what a unimodality constraint would do and Rank 3 is redundant; if a large majority fail, the distributions are genuinely multi-modal and a unimodality projection is a live, cheap regulariser to test **against** `T=0.75`.
- **Accuracy verdict:** re-allocation for the action; but a unimodality projection changes the *distribution*, so it could in principle also move the median. Magnitude unknown, `insufficient`.

### Rank 4 — **Policy learning / empirical-welfare maximisation (weighted-classification reduction)**

- **Definition & source:** E4 (Stanford STATS 361 course book), **Ch. 5 "Policy Learning"**, §5.1 "Policy evaluation", §5.2 "Empirical-welfare maximization"; and Ch. 3 "Doubly Robust Methods". `[directly_supported]` for the existence and naming of the family and its two-stage structure (evaluate a policy's value, then maximise the empirical value over a policy class). `insufficient` for chapter body — not read in this bounded window.
- **Why relevant.** FICR is literally an **empirical welfare**: `sum_i y_i · u(|a_i - y_i|) / sum_i 4 y_i`. Maximising it over a policy class `a(·)` is empirical-welfare maximisation with outcome weights `y_i`. This is the correct home for the *y*-weighting question (task item 3) on the *learning* side, complementing E1 Ex. 1.8.6 on the *decision* side.
- **BARAM applicability:** `near_match_only` and, more importantly, **already locally closed**. `research/lanes/s17_n18_step_reward.md` §2 records Kitagawa & Tetenov (2018) EWM with regret `O(sqrt(v/n))` in VC dimension `v`, and the local receipt `reports/s17_n9_cost5_recovery_receipt.json` records COST5-SPO+ at Total `0.619583` vs champion `0.631483`. The AGENTS.md fold-outside gate independently rejects every multi-degree-of-freedom policy fit. **This lane recommends NOT reopening Rank 4.**
- **Accuracy verdict:** re-allocation only, and locally falsified.

### Rank 5 — **Smoothed discontinuous-criterion M-estimation (maximum-score family) and cube-root asymptotics**

- **Status: `insufficient` — identified bibliographically, NOT fetched, no locator verified.** The lane was stopped before retrieval. Recorded here as the highest-value unclosed item because it is the family whose *entire subject matter* is "the argmax of an indicator-valued criterion", which is exactly BARAM's objective.
- Bibliographic anchors located by search but not opened: J. L. Horowitz (1992), "A Smoothed Maximum Score Estimator for the Binary Response Model", *Econometrica*, https://www.econometricsociety.org/publications/econometrica/1992/05/01/smoothed-maximum-score-estimator-binary-response-model ; Kim & Pollard (1990), "Cube Root Asymptotics", *Annals of Statistics*, https://www.jstor.org/stable/2241541 ; Cattaneo, Jansson & Nagasawa (2020), "Bootstrap-Based Inference for Cube Root Asymptotics", *Econometrica*, https://mdcattaneo.github.io/papers/Cattaneo-Jansson-Nagasawa_2020_ECMA.pdf ; D. Pollard, "Empirical Processes: Theory and Applications" (NSF-CBMS lecture notes), http://www.stat.yale.edu/~pollard/Books/Iowa/Iowa-notes.pdf ; Seijo & Sen, https://sites.stat.columbia.edu/bodhi/research/maxscsto.pdf .
- **Why it matters and what it would decide (§5.2).** `insufficient`, but the hypothesis is precise and falsifiable, so it is stated in full below.

---

## 5.2 Task item 4 — does the teaching literature warn that plug-in Bayes under an over-dispersed density is worse than a sharpened surrogate?

**Honest answer: `insufficient` in this bounded window. No fetched source states this warning verbatim.** What the fetched sources do supply is the two halves of the mechanism, and the missing half is named:

- **Half one, `directly_supported` (E3).** Modal-type estimation requires the density to be *curved* at the mode: E3 assumption (A3) `|p_yy| > lambda_2 > 0`, and E3's own remark that the modal-manifold factorisation "is unique if the second derivative `p_yy(x,y)` is uniformly bounded away from zero." An over-dispersed / flat conditional density is precisely the `lambda_2 -> 0` failure case, where the argmax location is ill-conditioned and its sampling variance explodes. Sharpening `p -> p^{1/T}` with `T < 1` **manufactures curvature**: it multiplies the log-density by `1/T > 1`, so `(log q)'' = (1/T)(log p)''`, i.e. it inflates the curvature at the mode by exactly `1/T = 1.333` at `T = 0.75` while leaving the argmax of the *density* unchanged. Under a **band** objective (not a point mode) that curvature inflation strictly changes the argmax and strictly reduces its variance. `[derived]` from E3's (A3) plus elementary algebra.
- **Half two, `directly_supported` (E5).** E5's stated motivation for shape constraints is that kernel methods "require the choice of one or more bandwidths" while shape-constrained MLE "does not require the choice of any tuning parameter." This is the same bias-variance economy: a *structural* restriction substitutes for a *smoothing* parameter. `T` is a smoothing parameter in that sense.
- **Half three, `insufficient`.** The explicit theorem that a *smoothed/biased* criterion beats the *exact* discontinuous criterion — improving the rate from `n^{-1/3}` cube-root asymptotics toward `n^{-2/5}` or better — is the content of the Horowitz smoothed-maximum-score result (Rank 5). **Not fetched. Not verified. Do not act on it as if it were confirmed.**

**(b) BARAM migration hypothesis, stated so it can be killed.** `T` is not a hack; it is the bandwidth of a smoothed discontinuous-criterion estimator, and `T = 0.75` being an interior argmax of two independent grids is the empirical signature of an interior bias-variance optimum, which is what that theory predicts and what a "use the exact Bayes action" prescription does not.
**What would falsify it.** If `T*` is stable across sample size — i.e. re-running the `T` grid on 50% subsamples of the development surface returns the same `T* ≈ 0.75` — then `T` is **not** behaving like a bandwidth (a bandwidth must shrink toward 1, i.e. toward no smoothing, as `n` grows) and the smoothed-criterion reading is wrong. **If instead `T*` drifts toward 1 as `n` grows, the reading is supported and the implied prescription is that more data, not a better action, is the lever — which agrees with the root's elasticity finding.** `[derived]`; the diagnostic is §6.3.
**Local evidence needed.** §6.3. Note this is a *grid re-evaluation on saved actions*, not a fit — but it does require re-scoring, so it is the root's call whether it counts as a fit under AGENTS.md.

---

## 6. Ranked local diagnostics — cheapest first

### 6.0 THE SINGLE CHEAPEST TEST (this is the one to run)

**`TRUNCATED-MEDIAN CEILING PROBE`.** Zero fit, zero new model, pure arithmetic on already-saved per-row binned conditional distributions and already-saved actual values.

Compute three NMAE numbers on the existing outer folds:
1. `NMAE(champion action)` — the anchor, known.
2. `NMAE(a_med)` where `a_med(x) = ` the median of the row's stored binned distribution `p_k` — the naive proper point.
3. `NMAE(a_trunc)` where `a_trunc(x) = ` the median of `p_k` **restricted and renormalised to bins with `y_k >= 0.1C`** — the E1 Ex. 1.8.6 / Ex. 1.8.2 optimum for the official NMAE.

**What it decides, in one shot:**
- Number 3 minus number 1 is the **entire accuracy headroom available from any change of action functional whatsoever**, because by E1 Ex. 1.8.2 no action can beat the truncated median on this objective. If `(3) - (1) << +0.0212`, then **the action axis is closed for accuracy, definitively, with a certificate rather than an anecdote**, and the root can stop testing action-side ideas.
- Number 3 minus number 2 isolates whether the eligibility truncation of §4.2 is worth anything on its own.
- If `(3) - (2)` is material, the tilt is free accuracy and should be integrated immediately.

**Falsifier for §4.2:** `(3) - (2) ≈ 0`. **Falsifier for the whole lane's usefulness:** `(3) - (1)` small, which is the outcome this lane *predicts* (§4.1).
**Prerequisite (2-minute read, do first):** confirm in `inputs/notebooks/metric_official.ipynb` whether FICR's "valid" row set is identical to NMAE's `y >= 0.1C` eligible set. Currently `insufficient`.

### 6.1 `ENERGY-TILT PROBE` (no fit)

Replace the stored `p_k` by `p_k^{tilt} ∝ y_k · p_k · 1{y_k >= 0.1C}`, re-run the **existing unchanged** `T`/`G` settlement argmax, score. Then repeat with a one-parameter family `p_k^{tilt}(gamma) ∝ y_k^gamma · p_k`, `gamma in {0, 0.25, 0.5, 0.75, 1}`; `gamma = 0` reproduces today's pipeline exactly, which is the built-in sanity check.
**Decides:** whether the pipeline already implicitly implements the energy tilt of §3.2. **Falsifier:** the `gamma` curve is flat, or its argmax is at `gamma = 0`. **Degrees of freedom: one.** This respects the AGENTS.md fold-outside standing rule; report the fold-outside `gamma`, not the in-sample one.

### 6.2 `BAND-FLATNESS / CURVATURE PROBE` (no fit)

For each row, evaluate the stored conditional distribution's band-hit function `H(a) = sum_k p_k · u(|a - y_k|/C)` on a fine grid of `a`, and record (i) `max_a H(a)`, (ii) the **width of the near-argmax plateau** `{a : H(a) >= 0.99 · max H}`, (iii) a finite-difference curvature at the argmax.
**Decides:** E3 assumption (A3). If the plateau width is comparable to or larger than the band half-width `0.06C` on most rows, the modal-interval functional is **weakly identified** and every estimator in Rank 1 is structurally blocked — no amount of better modal machinery helps, and the `+0.094` oracle per-row selection headroom is confirmed as realization noise from a second, independent direction.
**Falsifier for the "flatness" reading:** narrow, sharply peaked plateaus on the majority of rows.

### 6.3 `T-BANDWIDTH SCALING PROBE` (requires re-scoring; root's call)

Re-run the existing `T` grid on random 25% / 50% / 100% subsamples of the development surface (fixed model, fixed actions, only the temperature grid re-evaluated). Plot `T*(n)`.
**Decides:** §5.2. **Supported** if `T*` rises toward 1 with `n`. **Falsified** if `T*` is flat in `n`.

### 6.4 `LOG-CONCAVITY AUDIT` (no fit)

Fraction of rows whose stored `log p_k` is concave in `k` after excluding boundary bins. Decides whether Rank 3 is redundant with `T`.

---

## 7. Explicit `insufficient` gaps (lane stopped early by root order)

| Gap | Status | What was needed |
|---|---|---|
| Horowitz smoothed maximum score — exact theorem and rate | `insufficient` | Fetch Econometrica 1992 or an econometrics course treatment (MIT 14.382, Harvard Ec 2144). This is the **single highest-value unclosed source**: it is the only literature that would *directly* support "a smoothed surrogate beats the exact discontinuous criterion", i.e. task item 4. |
| Kim & Pollard cube-root asymptotics, `n^{-1/3}` rate for shorth/mode | `insufficient` | Fetch Pollard's Iowa notes (http://www.stat.yale.edu/~pollard/Books/Iowa/Iowa-notes.pdf) — free lecture notes, would give the rate statement. |
| E3's explicit convergence-rate theorems (§4 beyond assumptions) | `insufficient` | Fetch pages 12–20 of the AoS PDF. Assumptions were captured; the rate expressions were not. |
| E4 Wager Ch. 5 body — the EWM regret statement and the weighted-classification reduction | `insufficient` | Fetch pp. 57–68. Partly redundant with the Kitagawa–Tetenov material already in `s17_n18_step_reward.md`. |
| MIT OCW 6.7920 / 6.435 / 15.093 material | `insufficient` | Not retrieved at all. The task explicitly requested MIT sources and this lane produced none. |
| CMU 36-708 classification notes on plug-in vs direct rules ("it is easier to estimate the decision boundary than the regression function") | `insufficient` | Would be the second-best support for task item 4. |
| Hyndman (1996) HDR — "Computing and Graphing Highest Density Regions" | `insufficient` | HDR as a named family is asserted in the task brief; this lane did **not** verify a definition source. Treat "HDR regression" as unconfirmed naming here; the confirmed name is **modal interval** (E1 Ex. 1.8.5). |
| Whether FICR's "valid" row set == NMAE's `y >= 0.1C` eligible set | `insufficient` | Local read of `inputs/notebooks/metric_official.ipynb`. **Blocks §4.2 and §6.0 from being acted on, though not from being run.** |
| Whether the champion pipeline already applies an energy tilt or an eligibility truncation | `insufficient` | Local code read. |

---

## 8. What this lane does NOT claim

- It does **not** claim any expected BARAM gain. No source effect size appears anywhere above as a BARAM number. The only numbers used are the root's own (`+0.0029`, `+0.0212`, `0.85848`, `0.8797`, `+0.094`, `40.2%`, IQR `0.067`–`0.119`, `T = 0.75`).
- It does **not** claim modal regression will help. §4.1 says the opposite, plainly.
- It does **not** claim the `T = 0.75` heuristic is theoretically justified. §5.2 marks that `insufficient` and gives the falsifier.
- It does **not** re-open the step-reward / prescriptive / EWM axis closed by `research/lanes/s17_n18_step_reward.md` and `reports/s17_n9_cost5_recovery_receipt.json`.
- It confirms rather than extends the Brehmer–Gneiting modal-interval identification already held locally; the contribution there is a 1967-vintage course-text name and derivation, plus the weighting theorem (E1 Ex. 1.8.6), which was **not** previously held.
