# EDU-DENSITY-SHARPNESS — teaching-literature attack on predictive over-dispersion

Lane: `EDU-DENSITY-SHARPNESS`. Exclusive output. Read-only research lane under
`/Users/um-yunsang/BARAM2026/AGENTS.md` (2026-08-06 bounded-lane allowance): no repository write
outside `research/`, no fit, no score, no data or weight download, no Dacon, no 2024 lockbox,
no remote model inference, no delegation.

Written: 2026-08-09. Bound: 90 minutes / 14 sources. **Terminated early by explicit root order**
("STOP RESEARCHING AND WRITE NOW"). 8 sources fetched and read in full-text extract; 6 further
sources were located by search but **not fetched** and are therefore tagged `insufficient` below.
The lane is delivered in that reduced state, on purpose.

Evidence tags used exactly as specified: `directly_supported` | `contradicts_premise` |
`near_match_only` | `insufficient`. Every material claim carries exactly one tag and a locator.
Claims are split into **(a) SOURCE FACT**, **(b) BARAM MIGRATION HYPOTHESIS (provisional)**,
**(c) LOCAL EVIDENCE NEEDED**. No source effect size is ever quoted as a BARAM expected gain.

---

## 0. HEADLINE VERDICT (read this if you read nothing else)

**The teaching literature names our object, explains our over-dispersion, and — on the root's own
new elasticity measurement — says that almost every device in the brief is the wrong axis.**

1. Our predictive object is a **linear pool** (champion = `0.30*D + 0.2333*(three members)`).
   Gneiting & Ranjan (2013) prove that **dispersion increases under linear pooling** and that the
   traditional linear pool is **not flexibly dispersive** — i.e. it cannot be made sharper by
   reweighting. Our `T=0.75` sharpening is an unnamed ad-hoc undo of a *known, proved, structural*
   defect of the aggregator we chose. There are two named, principled replacements with theory:
   the **spread-adjusted linear pool (SLP)** and the **beta-transformed linear pool (BLP)**, and
   the BLP is proved *flexibly dispersive*. Both are 1–2 degrees of freedom, which is the only
   thing that survives the AGENTS.md fold-outside gate. `directly_supported`.
2. **BUT**: SLP/BLP/temperature/focal/isotonic/mode-seeking are all **RE-ALLOCATION devices**.
   They change *which functional of a fixed conditional law* you report, or *how sharp* the
   reported law is. None of them shrinks the conditional law itself. Root's elasticity file says
   we need `s ≈ 0.85`, a **15% cut in raw capacity-relative MAE**. **No choice of functional of a
   fixed `p(y|x)` can deliver that.** Say it plainly: my lane's entire device catalogue is
   re-allocation, and root has measured re-allocation as exhausted.
3. Therefore the **only question my lane can usefully answer** is root's re-prioritised one:
   **how much of the 0.14152 capacity-relative MAE is irreducible?** I give a **rigorous, two-sided,
   ZERO-FIT bracket** for the irreducible MAE floor built from matched pairs (§5.2). It costs one
   pass over an artifact we already have (the M252 analog retrieval index) and it is the single
   cheapest local test in this document. It can **close the accuracy axis outright** if it comes
   back high.
4. One `contradicts_premise` finding against the brief's own reading: **member disagreement is the
   textbook signature of EPISTEMIC (reducible) uncertainty, not aleatoric**. The brief says the
   40.2% over-band member spread "suggests a large irreducible component". The taught definition
   says the opposite. See §5.3.

---

## 1. SOURCES

Read in full-text extract (8):

| # | Institution / course | Document | Locator |
|---|---|---|---|
| S1 | Carnegie Mellon (authors CMU) | Chen, Genovese, Tibshirani & Wasserman, *Nonparametric Modal Regression*, Annals of Statistics 2016, Vol. 44 No. 2, 489–514, DOI 10.1214/15-AOS1373 | https://www.stat.berkeley.edu/~ryantibs/papers/modal-aos.pdf |
| S2 | CMU **36-708 Statistical Methods for Machine Learning**, Spring 2018 (Larry Wasserman) | course index / syllabus | https://www.stat.cmu.edu/~larry/=sml/ |
| S3 | CMU **36-708**, lecture notes | *Density Estimation* (36-708 notes) — histograms §3, minimax lower bound Thm 2, Stone's theorem Thm 12 | https://www.stat.cmu.edu/~larry/=sml/densityestimation.pdf |
| S4 | CMU **36-708 / Statistical Machine Learning Spring 2015**, Ryan Tibshirani with Larry Wasserman | *Nonparametric Regression* lecture notes — minimax rate over Hölder/Sobolev class | https://www.stat.cmu.edu/~larry/=sml/nonpar.pdf |
| S5 | Stanford **EE364a Convex Optimization I** (Boyd & Vandenberghe; slides revised by Boyd, Vandenberghe, Nobel) | Lecture 6 slides *Approximation and fitting*, pp. 6.4–6.5 | https://web.stanford.edu/class/ee364a/lectures/approx.pdf |
| S6 | Heidelberg / Univ. of Washington | Gneiting & Ranjan, *Combining predictive distributions*, Electronic Journal of Statistics 7 (2013) 1747–1782 | https://arxiv.org/pdf/1106.1638 |
| S7 | Michigan **EECS 598 Statistical Learning Theory, Winter 2014** (Clayton Scott) | Topic 14 *Calibrated Surrogate Losses*, scribe Efrén Cruz Cortés | https://web.eecs.umich.edu/~cscott/past_courses/eecs598w14/notes/14_calibrated.pdf |
| S8 | UPenn **CIS 7000 Trustworthy Machine Learning, Spring 2024** (Osbert Bastani) | Lecture 14 *Aleatoric vs. Epistemic Uncertainty* | https://www.seas.upenn.edu/~obastani/cis7000/spring2024/docs/lecture14.pdf |

Located by search, **NOT fetched, tagged `insufficient` wherever cited** (6):

| # | Target | Why wanted | Locator found |
|---|---|---|---|
| N1 | Gneiting, Balabdaoui & Raftery 2007 JRSS-B, *Probabilistic forecasts, calibration and sharpness* | the verbatim "maximise sharpness subject to calibration" sentence | not fetched; secondary pointer https://www.jstor.org/stable/30130759 |
| N2 | Guo et al. 2017 ICML, *On Calibration of Modern Neural Networks*; UPenn CIS 7000 Lecture 11 | direction of temperature scaling (T>1 vs our T<1) | https://proceedings.mlr.press/v70/guo17a/guo17a.pdf ; https://www.engineering.upenn.edu/~obastani/cis7000/spring2024/docs/lecture11.pdf |
| N3 | Univ. South Carolina **STAT 824** slides, Lec 04; gregorkb nonparametric notes | difference-based / model-free residual-variance estimation | https://people.stat.sc.edu/gregorkb/STAT_824_sp_2025/STAT_824_Lec_04_slides.pdf ; https://gregorkb.github.io/nonparm/nonparmregvarest.html |
| N4 | Cover & Hart 1967; Cornell **CS4780** Lecture 2 kNN notes | `R* <= R_1NN <= 2R*` as the taught irreducible-error bracket | https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf ; https://www.cs.cornell.edu/courses/cs4780/2018sp/lectures/lecturenote02_kNN.html |
| N5 | Ehm, Gneiting, Jordan & Krüger 2016 JRSS-B, *Of quantiles and expectiles* (Schervish mixture representation, Murphy diagrams) | the exact "a proper score is a mixture over elementary decision problems" statement | https://arxiv.org/pdf/1503.08195 |
| N6 | Manski maximum score / Horowitz 1992 Econometrica smoothed maximum score; Kim & Pollard 1990 cube-root asymptotics | rates for maximising a *discontinuous* band objective | https://www.econometricsociety.org/publications/econometrica/1992/05/01/smoothed-maximum-score-estimator-binary-response-model ; https://www.jstor.org/stable/2241541 |

---

## 2. THE NAMED OBJECT WE MISSED — and its honest limit

### 2.1 The reward has a textbook name: **deadzone-linear penalty**

**(a) SOURCE FACT** — `directly_supported`.
Stanford EE364a, *Approximation and fitting*, slide 6.4 (penalty function approximation) lists
verbatim: "**deadzone-linear with width a: φ(u) = max{0, |u| − a}**", alongside quadratic,
log-barrier and (slide 6.6) Huber. Slide 6.5, "Example: histograms of residuals", states verbatim:
"**shape of penalty function affects distribution of residuals**", and displays the residual
histograms for `|u|`, `u^2`, `deadzone (u) = max{0,|u|−0.5}` and log-barrier on the same
`A ∈ R^{100×30}` problem, showing the deadzone penalty piling residual mass *inside* the deadzone.
Locator: https://web.stanford.edu/class/ee364a/lectures/approx.pdf , pp. 6.4–6.5.

**(b) BARAM MIGRATION HYPOTHESIS (provisional)** — our FICR reward is *exactly* a two-level
deadzone with widths `a1 = 0.06C` (pays 4) and `a2 = 0.08C` (pays 3), actual-weighted. The taught
object for training against it is the **convex** deadzone-linear penalty at `a = 0.06C`, whose
subgradient is `sign(u)·1{|u| > a}` — i.e. **L1 restricted to band violators, which is NOT the
median**. This is materially different from the smoothed step reward that we already tried and
whose gradient "degenerated to L1, i.e. the conditional median": the deadzone penalty is convex,
non-degenerate, and its stationarity condition is a *censored* median condition, not a median
condition.

**FALSIFIER, and it is a strong one.** Root's own energy-weighted tiers say **56.9% of delivered
energy is already beyond 8%**, i.e. far outside the deadzone. On any row with `|u| >> a`, the
deadzone penalty equals `|u| − a`, which differs from L1 by a **constant**, so its gradient is
identical to L1's. Prediction: if the row-weighted fraction outside the band is of the same order
(~2/3), the deadzone-linear fit will be **numerically almost indistinguishable from an L1 fit**,
and the device will fail for the same reason the smoothed-step LightGBM failed.
**(c) LOCAL EVIDENCE NEEDED** — one no-fit number: the **row-weighted** (not energy-weighted)
fraction of valid rows with `|a−y|/C ≤ 0.06` under champion actions. If that fraction is below
about 0.5, do not spend a fit on the deadzone penalty. This is a `wc`-level computation on the
existing outer-fold artifact.

### 2.2 The action has a textbook name: **modal regression** (and the rate is bad)

**(a) SOURCE FACT** — `directly_supported`.
Chen, Genovese, Tibshirani & Wasserman, AoS 2016, abstract, verbatim: "**Modal regression estimates
the local modes of the distribution of Y given X = x, instead of the mean, as in the usual
regression sense, and can hence reveal important structure missed by usual regression methods.**"
The paper's contribution list (Section 1) includes verbatim: "We prove consistency of the
nonparametric modal regression estimator, and furthermore derive explicit **convergence rates**,
with respect to various error metrics"; "We propose a method for constructing **prediction sets**,
based on plug-in methods, and prove that the population prediction sets from this method **can be
smaller than those based on the regression function**"; and "We propose a rule for selecting the
smoothing bandwidth of the KDE based on **minimizing the size of prediction sets**".
The estimator is defined (their eq. 4–5) as the *local maxima in y* of a **joint KDE** `p̂(x,y)`:
`M_n(x) = {y : p̂_y(x,y) = 0, p̂_yy(x,y) < 0}`.
Locator: https://www.stat.berkeley.edu/~ryantibs/papers/modal-aos.pdf , abstract and §1.

**(b) BARAM MIGRATION HYPOTHESIS (provisional)** — `arg max_a P(|a − Y| ≤ h | x)` for small `h` is
the **local-mode functional**, so our "binned distribution, power `1/T`, argmax of settlement-weighted
utility" **is a hand-rolled modal regression estimator with a hand-rolled sharpening bandwidth**.
`T = 0.75` plays the role of the KDE bandwidth `h` in S1, and our 25-point/12-point grid search over
`T` is a hand-rolled version of S1's bandwidth-selection rule. **Two immediate consequences:**
(i) the S1 bandwidth rule — pick `h` to *minimise the size of the prediction set* — is a
**principled, calibration-free** substitute for our grid search over `T`, and it does not consume a
validation surface the way a score-based grid search does;
(ii) mode functionals inherit **slower-than-`n^{-1/2}` rates** and non-Gaussian limits (the
cube-root / Chernoff regime, S1 §"convergence rates"; the general theory is Kim & Pollard 1990,
tagged `insufficient` here because N6 was not fetched). This is the *statistical explanation* for
the brief's most striking measurement: **the exact plug-in Bayes action scored 0.6192 versus a tuned
heuristic 0.6371 on the same distribution.** Plug-in mode estimation is the slow, unstable
functional; the tuned heuristic is an implicitly regularised (over-smoothed) version of it, and
over-smoothing a mode estimator is *variance reduction at the price of bias*, which is the standard
bias–variance trade for mode functionals. `near_match_only` for the specific rate constant, because
I did not extract the rate exponents from S1 before the stop order.

**(c) HONEST ANSWER TO ROOT'S DIRECT QUESTION** — "can a mode-type estimator also IMPROVE ACCURACY,
or is it only a re-allocation device?" **It is a re-allocation device.** The modal functional and
the mean functional are two different summaries of the *same* `p(y|x)`. Moving from mean to mode
strictly *increases* `E|Y − a|` whenever the conditional law is asymmetric (the median minimises
MAE, the mode does not), so a mode-type estimator **worsens NMAE by construction** and can only pay
through FICR. Root's elasticity table says FICR needs `+0.0295` at fixed 1-NMAE, and the brief's
own measurements say the action grid is already at its argmax. **Modal regression is the correct
NAME for what we are already doing; it is not a new source of gain.** `directly_supported` for the
functional-vs-functional argument (median minimises `E|Y−a|` is standard decision theory; the
teaching statement of the corresponding surrogate/target correspondence is S7 §1).

---

## 3. WHY THE FITTED CONDITIONAL DENSITY IS OVER-DISPERSED — three taught mechanisms

### M1. Predictive dispersion = aleatoric ⊕ epistemic (law of total variance)

**(a) SOURCE FACT** — `directly_supported`.
UPenn CIS 7000 (Spring 2024) Lecture 14, agenda verbatim: "Aleatoric vs. epistemic uncertainty ·
Linear regression example · **Bootstrapping ensembles for estimating epistemic uncertainty** ·
Application to active learning". Definitions verbatim: "**Epistemic uncertainty** · Uncertainty due
to limitations in our knowledge about the world · **Can be eliminated by obtaining additional
labels/information**"; "**Aleatoric uncertainty** · 'Intrinsic' uncertainty that can't be avoided ·
Not helpful to obtain additional labels/information". The slide also states the framing point:
predictive uncertainty is "**Useful for decision-making**" but "**aggregates multiple sources of
uncertainty**".
Locator: https://www.seas.upenn.edu/~obastani/cis7000/spring2024/docs/lecture14.pdf , "Aleatoric vs.
Epistemic Uncertainty" slide and "Predictive Uncertainty" slide.

**(b) BARAM MIGRATION HYPOTHESIS** — an honest plug-in predictive is the *convolution* of the true
conditional law with the estimation-error law, hence **always at least as wide as the aleatoric
law**. The step-band reward wants the mode of the **aleatoric** law. Sharpening with `T < 1` is a
crude approximate **deconvolution** of the epistemic smear. That is why `T = 0.75` helps and why it
has an interior optimum: too little sharpening leaves the epistemic smear in, too much amplifies
the noise in the density estimate.
**FALSIFIER**: if the optimal `T` is **invariant to training-set size**, the width being removed is
not epistemic (epistemic shrinks with `n`) and this whole story is wrong.
**(c) LOCAL EVIDENCE NEEDED** — refit the *same* classifier on 50% and 100% of training rows and
re-run the existing 12-point `T` grid on each. If `T*(50%) < T*(100%)` (more sharpening needed with
less data), M1 is confirmed. This is 2 fits, so it is **not** the cheapest test; it is listed as the
second-cheapest.

### M2. **Linear pooling provably increases dispersion** — the strongest finding in this lane

**(a) SOURCE FACT** — `directly_supported`, and this is the single most load-bearing quotation I
obtained.
Gneiting & Ranjan (2013), *Combining predictive distributions*, EJS 7, verbatim from the paper's own
summary of its main result: "**A major result is, roughly, that dispersion tends to increase under
linear pooling. This helps explain the success of linear combination formulas in aggregating
underdispersed component distributions, and allows us to show that the traditional linear pool
fails to be flexibly dispersive. Parsimonious nonlinear alternatives include generalized linear
pools, the spread-adjusted linear pool, which has been used successfully in meteorological
applications, and the beta-transformed linear pool proposed by Ranjan and Gneiting (2010), which we
demonstrate to be flexibly dispersive.**"
Locator: https://arxiv.org/pdf/1106.1638 , introduction / abstract paragraph.

**TRAP — the sign convention in S6 is the opposite of the colloquial one, and mis-reading it will
invert every conclusion below.** S6 Definition (their §2), verbatim: "(c) The forecast F is
**overdispersed if var(Z_F) < 1/12**, **neutrally dispersed if var(Z_F) = 1/12**, and
**underdispersed if var(Z_F) > 1/12**", where `Z_F` is the probability integral transform (PIT).
And: "**U-shaped histograms correspond to underdispersed predictive distributions with prediction
intervals that are too narrow on average, while hump or inverse U-shaped histograms indicate
overdispersed predictive distributions.**" So in S6's language, **over-dispersed = intervals too
WIDE = hump-shaped PIT**, which is our situation. Anyone re-reading S6 must check this convention
before quoting a `var(Z_F)` inequality.

**(b) BARAM MIGRATION HYPOTHESIS** — our champion is *literally* a linear pool
(`0.30*D + 0.2333*(three members)`), and the brief reports **member spread exceeding the full band
on 40.2% of rows**. S6 says a linear pool of that construction is over-dispersed by *theorem*, not
by accident, and that **no reweighting can fix it** ("fails to be flexibly dispersive") — which is
independently consistent with the brief's own measurement that **oracle re-weighting of the four
fixed actions is worth only +0.0016**. Two facts that looked unrelated are the same fact.
The named, theory-backed replacements are:
  * **SLP — spread-adjusted linear pool**: one extra scalar parameter `c` that rescales each
    component's spread before pooling; "used successfully in meteorological applications" (S6).
    **CONCENTRATES.** 1 dof.
  * **BLP — beta-transformed linear pool** (Ranjan & Gneiting 2010): apply a Beta CDF to the pooled
    CDF; S6 "demonstrate[s]" it is **flexibly dispersive**, i.e. it *can* be made sharper. 2 dof.
  * **generalized linear pools** (incl. logarithmic / geometric pooling): products of experts,
    sharper than the arithmetic pool by construction. `near_match_only` — S6 names "generalized
    linear pools" but I did not extract its formal statement about the log pool specifically.
**Crucially, these pool the DISTRIBUTIONS, not the actions.** Our champion pools **four already-
computed ACTIONS** (`0.30*D + 0.2333*3`). Pooling actions is a *third* thing, and it is the worst
of the three for a band reward: the arithmetic mean of four actions that are more than a band apart
on 40.2% of rows lands **between** the members' bands and can therefore be outside *every* member's
band. That is a mechanism for losing FICR that neither SLP, BLP nor `T` addresses.

**FALSIFIER** — if, on rows where member spread exceeds the band, the champion's blended action hits
the paying band **at least as often as the best single member does**, then action-averaging is not
hurting and this hypothesis is dead.
**(c) LOCAL EVIDENCE NEEDED — this is the second cheapest test in the document and it is ZERO FIT.**
On the 7453 valid outer-fold rows already in `reports/s17_n47_ficr_elasticity.json`'s source
artifact, restrict to the 40.2% high-spread rows and compare, energy-weighted: `FICR(blend)` versus
`FICR(each single member)` versus `FICR(median of the four member actions)`. The **median of member
actions** is the one-line, zero-parameter, no-fit alternative to the arithmetic blend, and it is the
natural robust aggregator when members are more than a band apart. If the member-median beats the
arithmetic blend on the high-spread subset, that is direct evidence for the mechanism and it costs
one `numpy.median` call.
**Honest expected magnitude: small.** Root's elasticity says we need `s ≈ 0.85`; nothing in this
paragraph is an `s = 0.85` device. It is worth doing only because it is free.

### M3. Fine binning is a variance problem with a taught rate

**(a) SOURCE FACT** — `directly_supported`.
CMU 36-708 *Density Estimation* notes, §3 Histograms, verbatim setup: "Divide X into bins, or
sub-cubes, of size h. ... There are **N ≈ (1/h)^d such bins and each has volume h^d**", with
`θ̂_j = (1/n) Σ_i I(X_i ∈ B_j)` "the fraction of data points in bin `B_j`". The notes then bound
bias and variance over the Lipschitz class `P(L)`, and give a matching **minimax lower bound**,
Theorem 2 verbatim: "There exists a constant `C > 0` such that
`inf_p̂ sup_{P ∈ P(L)} E ∫ (p̂(x) − p(x))^2 dx ≥ C (1/n)^{2/(d+2)}`."
The concentration analysis in the same section bounds
`P(‖p̂_h − p_h‖_∞ > ε) ≤ Σ_j P(|θ̂_j − θ_j| > ε h^d)` by Bernstein's inequality using
`θ_j(1−θ_j) ≤ θ_j ≤ C h^d` — i.e. the sup-norm error is driven by the **per-bin count `n·θ_j ≈ n h^d`**.
Locator: https://www.stat.cmu.edu/~larry/=sml/densityestimation.pdf , §3 and Theorem 2.
Companion: CMU 36-708 / SML Spring 2015 *Nonparametric Regression* notes give the regression-side
minimax rate verbatim: "the minimax risk is
`min_f̂ max_{f0 ∈ Σ(k,L)} E‖f̂ − f0‖²_{L2} = Ω(n^{−2k/(2k+1)})`", citing Tsybakov (2009) §2.6.2.
Locator: https://www.stat.cmu.edu/~larry/=sml/nonpar.pdf , §"minimax".

**(b) BARAM MIGRATION HYPOTHESIS** — "our finest binning collapsed at 181 classes for roughly 20000
rows" is exactly the `n h^d` regime of S3: 181 classes over ~20000 rows is ~110 rows per class
**marginally**, and the *conditional* density that the action actually uses is supported by far
fewer effective rows than that (the effective local sample size, not `n`). The sup-norm control in
S3 degrades as `1/(n h^d)`, and the **argmax** functional we then apply is exactly the functional
that is most sensitive to sup-norm error in the density. That is a second, independent explanation
of the plug-in Bayes-action failure (0.6192 vs 0.6371), complementary to §2.2.
**A specific, taught remedy we have NOT used: unequal-width binning concentrated where the reward
lives.** The FICR reward only ever distinguishes three sets — `|a−y| ≤ 0.06C`, `0.06C < |a−y| ≤ 0.08C`,
and beyond. So the **only functionals of `p(y|x)` the action needs** are two window probabilities.
A 181-class equal-width density estimates *far more than the decision requires*, paying full
`1/(n h)` variance on bins the reward cannot distinguish. `near_match_only` — S3 teaches the rate and
the bin-count/variance mechanism, but I did not fetch a course treatment of adaptive/unequal-width
binning specifically (`insufficient` for that sub-claim).
**FALSIFIER** — if re-binning to an unequal grid whose *cell widths equal the reward resolution*
(e.g. bin edges at multiples of `0.02C` near the action, coarse in the tails) leaves the tuned
score unchanged to 3 decimals, the density resolution was never the binding constraint.
**(c) LOCAL EVIDENCE NEEDED** — no fit required if the existing 181-class predicted probabilities are
stored: simply **re-aggregate** the 181 class probabilities into a reward-aligned coarse grid and
re-run the existing action rule. Pure post-processing of a saved artifact.

---

## 4. DEVICE TABLE — concentrates vs merely re-calibrates, and step-band compatibility

`C` = concentrates (reduces dispersion), `R` = re-calibrates only (monotone CDF map, dispersion
follows only incidentally), `A` = acts on the action, not on the density.

| Device | C / R / A | Compatible with step-band reward? | Named theory | Tag | Cheapest local diagnostic |
|---|---|---|---|---|---|
| **Temperature `T<1` on binned probs** (current) | **C** | yes (it is our current mechanism) | none in the sources I read; it is an unnamed heuristic in our code | `insufficient` (N2 not fetched) | already at grid argmax — do not re-search `T` |
| **SLP — spread-adjusted linear pool** | **C** | yes, and it is 1 dof so it survives the fold-outside gate | Gneiting & Ranjan 2013 (S6) | `directly_supported` | requires member *distributions*, not member actions — check whether we still store them |
| **BLP — beta-transformed linear pool** | **C** (proved *flexibly dispersive*) | yes, 2 dof | Ranjan & Gneiting 2010 via S6 | `directly_supported` | same prerequisite as SLP |
| **Generalized / logarithmic (geometric) pool** | **C** by construction | yes | named in S6 | `near_match_only` | geometric mean of member prob vectors, then existing action rule — no fit |
| **Median-of-member-actions instead of mean** | **A** | yes; robust when members are >1 band apart | not in my sources; folk robust statistics | `insufficient` | **one `numpy.median` on saved actions — free** |
| **Platt scaling** | **R** | yes but pointless: a monotone map of a *sigmoid* form cannot concentrate a multi-bin density | standard | `insufficient` (not fetched) | skip |
| **Isotonic regression** | **R** (can incidentally sharpen only where data supports it) | yes | standard | `insufficient` (not fetched) | skip — and note it consumes a validation surface we no longer have |
| **Focal loss / entropy penalty (negative-entropy regulariser)** | **C** | yes | not covered in my 8 sources | `insufficient` | — |
| **"Reverse label smoothing" (anti-smoothing)** | **C** | yes; mathematically the same family as `T<1` | not covered | `insufficient` | already covered by `T` grid |
| **Deadzone-linear training penalty at `a = 0.06C`** | acts on the **fit**, so it is the only device here that could touch raw MAE | yes, exactly matched to the reward | Boyd & Vandenberghe / EE364a §6.4–6.5 (S5) | `directly_supported` for the object; `near_match_only` for the migration | §2.1(c): row-weighted in-band fraction; if < 0.5, expect L1 degeneracy |
| **Modal-regression bandwidth rule (minimise prediction-set size)** | **A** | yes | Chen et al. 2016 (S1) contribution 5 | `directly_supported` | replaces the `T` grid search without consuming score |

**Cross-cutting warning about ALL "just optimise a proper score" advice** — `directly_supported`.
Michigan EECS 598 Topic 14 states the surrogate problem exactly: the target loss "is neither convex
nor differentiable ... which poses computational challenges", a surrogate is used, and "**Using a
surrogate loss raises the question of whether minimizing `R_L(f)` is still meaningful**". The notes
then define the transfer bound via `ψ_L`: `L` is **classification-calibrated** iff `H_L(η) > 0`
for all `η ≠ 1/2` (Definition 1), `L` is CC iff `ψ_L` is invertible (Theorem 2), and the excess-risk
transfer is `ψ_L(R(f) − R*) ≤ R_L(f) − R*_L`, with the hinge special case
`R(f) − R* ≤ R_L(f) − R*_L` (Corollary 1).
Locator: https://web.eecs.umich.edu/~cscott/past_courses/eecs598w14/notes/14_calibrated.pdf , §1–§3.
**Migration**: our smoothed-step LightGBM is a surrogate whose `ψ_L` is *flat near zero* — a large
improvement in the surrogate risk buys an arbitrarily small improvement in the target risk, and the
minimiser drifted to the median. The taught prescription is to check calibration of the surrogate
**for the actual target loss** before fitting, which is a pen-and-paper check, not an experiment.
`directly_supported` for the source statement; `near_match_only` for the migration, because S7 states
the theory for binary 0/1 loss and our loss is a three-level band on a continuous response.

---

## 5. IRREDUCIBLE VS REDUCIBLE — the ceiling argument (ROOT'S RE-PRIORITISED QUESTION)

### 5.1 What the target quantity actually is

For **MAE**, the irreducible floor is not `Var(Y|X)`; it is
`MAE_floor = E_X[ E( |Y − m(X)| | X ) ]` where `m(X) = median(Y|X)`.
The median, not the mean, attains it. Our current capacity-relative MAE is **0.14152**; root needs
`≈ 0.1203` (`s = 0.85`). **The accuracy axis is open iff `MAE_floor` is comfortably below 0.1203.**
`directly_supported` for the decision-theoretic fact (standard; the corresponding teaching framing
is S8's aleatoric/epistemic split and S4's minimax setup). The number 0.1203 is root's, restated.

### 5.2 ★ THE CHEAPEST LOCAL TEST IN THIS DOCUMENT: a zero-fit two-sided bracket on `MAE_floor`

**(a) SOURCE FACT** — `near_match_only`. The taught device is **difference-based / matched-pair
estimation of the irreducible noise level in nonparametric regression**: estimate `σ²` from
differences of responses at neighbouring covariate values **without fitting any regression function**
(classical Rice-type estimators). I located, but did **not fetch**, USC STAT 824 Lec 04 slides and
the companion notes page (N3), so the *teaching locator* is unverified and this is `near_match_only`,
not `directly_supported`. The related taught bracket for classification is Cover–Hart
`R* ≤ R_1NN ≤ 2R*` (N4, `insufficient` — not fetched).

**(b) BARAM MIGRATION HYPOTHESIS — with a complete derivation that needs no source at all.**
Let `Y_1, Y_2` be two responses drawn at (nearly) the same covariate value `x`, conditionally i.i.d.
Let `D = E|Y_1 − Y_2|` (capacity-relative). Then:
  * **Upper bound.** `Y_2` is a *feasible predictor* of `Y_1`, so
    `D = E|Y_1 − Y_2| ≥ min_a E|Y_1 − a| = MAE_floor(x)`. Hence **`MAE_floor ≤ D`**.
  * **Lower bound.** Triangle inequality through the conditional median `m`:
    `E|Y_1 − Y_2| ≤ E|Y_1 − m| + E|Y_2 − m| = 2·MAE_floor(x)`. Hence **`MAE_floor ≥ D/2`**.
  * So **`D/2 ≤ MAE_floor ≤ D`**, exactly, for any conditional law, with no distributional
    assumption and **no model fit**. (Sanity check: Gaussian gives `D = 2σ/√π ≈ 1.128σ` and
    `MAE_floor = σ√(2/π) ≈ 0.798σ`, and indeed `0.564σ ≤ 0.798σ ≤ 1.128σ`.)

**HOW TO GET MATCHED PAIRS WITH ZERO NEW WORK.** We already own an analog/retrieval model, **M252**.
A retrieval model's whole job is to find, for each target row, historical rows with near-identical
NWP covariates. **Those retrieved neighbours ARE the matched pairs.** Compute
`D̂ = mean over rows of |y_target − y_neighbour| / C`, restricted to the *k=1* nearest analog and to
the same validity window / lead time, on training rows only (never the lockbox, never the test
period). One pass over an existing index. No fit, no score, no new data.

**WHAT IT DECIDES — and this is the point.**
  * If **`D̂/2 > 0.1203`** (equivalently `D̂ > 0.2406`): the irreducible floor **provably exceeds**
    the accuracy root needs. The 15% MAE reduction is **impossible from the current conditioning
    information**, the entire accuracy axis is closed, and the only live axis left is *new
    information* (better NWP, more informative covariates) — which is a different lane's problem.
    This single number can close a whole axis.
  * If **`D̂ < 0.1203`**: the floor is *not* the binding constraint; the gap is estimation error and
    model form, i.e. **reducible**, and the accuracy axis is open.
  * If `0.1203` falls **inside** `[D̂/2, D̂]`: the bracket is inconclusive and must be tightened by
    matching more tightly (smaller neighbour radius) or by using `k` neighbours to average.

**BIAS DIRECTION, stated honestly.** Analog neighbours are not exactly at the same `x`, so `D̂`
contains a *signal* component in addition to the noise component and is therefore **biased upward**.
`D̂` is thus a **conservative (over-)estimate of `MAE_floor`'s upper bound** and a *possibly*
over-estimated lower bound. The clean consequence: **the `D̂ < 0.1203` branch is trustworthy** (if
even a biased-up estimate is below the target, the floor certainly is), while the
`D̂/2 > 0.1203` branch needs the neighbour radius reported alongside it before it is allowed to close
an axis. Tighten by plotting `D̂(radius)` and extrapolating to radius → 0 — still zero fit.

**FALSIFIER of the whole construction**: if analog neighbour distance is not small in NWP space
(i.e. the retrieval index is returning far neighbours), `D̂` is measuring signal variation, not noise,
and the bracket says nothing. Report the neighbour-distance distribution with the number or the
number is inadmissible.

**(c) LOCAL EVIDENCE NEEDED** — exactly two columns from the existing M252 artifact: neighbour
identity (or neighbour actual) and neighbour covariate distance. Runtime: seconds.

### 5.3 ★ `contradicts_premise` — member disagreement is EPISTEMIC, i.e. REDUCIBLE

**The brief states**: "Our members disagree by more than the full band on 40 percent of rows, which
suggests a large irreducible component."

**(a) SOURCE FACT** — `contradicts_premise`, and this is the one place where the teaching literature
directly contradicts a premise handed to me.
UPenn CIS 7000 Lecture 14 is organised around exactly the opposite identification. Its agenda item is
verbatim "**Bootstrapping ensembles for estimating epistemic uncertainty**", and its definitions are
verbatim "**Epistemic uncertainty** · Uncertainty due to limitations in our knowledge about the world
· **Can be eliminated by obtaining additional labels/information**" versus "**Aleatoric uncertainty** ·
'Intrinsic' uncertainty that can't be avoided · **Not helpful to obtain additional
labels/information**". The lecture's motivating example makes the operational distinction explicit:
"Robot is not sure if an object is a fork or a spoon · Is it worth moving closer to get a better
look? · **epistemic uncertainty → yes!** · **aleatoric uncertainty → no!**"
Locator: https://www.seas.upenn.edu/~obastani/cis7000/spring2024/docs/lecture14.pdf , slides
"Aleatoric vs. Epistemic Uncertainty" and "Motivation: Active Learning".
**In the taught vocabulary, ensemble disagreement is the canonical ESTIMATOR OF THE EPISTEMIC TERM.**
Disagreement is therefore evidence of the **reducible** component, not the irreducible one.

**(b) BARAM MIGRATION HYPOTHESIS** — the 40.2% over-band spread is, in the taught reading, a signal
that **there is real headroom** and that the members are individually under-determined. That is
independently consistent with the brief's own "**oracle per-row member selection is worth +0.094**".
**Two important caveats that keep this honest:**
  * S8's identification assumes a *bootstrap ensemble* — same hypothesis class, same loss, resampled
    data. Our members are **different model classes** (analog retrieval / classifier / DART), so their
    spread mixes epistemic uncertainty with **model-form (structural) uncertainty**. It is still not
    aleatoric under S8's definition, but S8's "eliminated by more labels" promise does not transfer
    cleanly to structural disagreement. `near_match_only` for the migration.
  * The brief also reports that **two independent learned gates failed to capture any of the +0.094**,
    and concludes it is realization noise. Those two readings are reconcilable in exactly one way:
    the oracle is computed **post hoc over realizations**, so it is contaminated by selection on the
    noise. **The taught way to settle it is a null calibration**, below.

**(c) LOCAL EVIDENCE NEEDED — a ZERO-FIT null-oracle calibration (third cheapest test).**
Recompute the *identical* per-row oracle-selection statistic under a null in which the members carry
**no** row-level information: e.g. permute member identities independently per row, or replace the
actual `y` by a draw from the members' own pooled predictive at that row. If the **null oracle also
returns roughly +0.09**, then the +0.094 is pure selection-on-realization and there is *no* headroom
— the gates did not fail, there was nothing to capture. If the null oracle returns, say, +0.03, then
roughly two thirds of the +0.094 is genuine, conditionally-identifiable headroom and the failure was
in the gate's feature set, not in the premise. **This is the single measurement that decides whether
the ensemble axis is dead or merely badly exploited, and it costs one permutation and one rescore.**

### 5.4 What a ceiling argument looks like, and what my sources do NOT license

`insufficient`. The formally correct ceiling argument for "how much accuracy is extractable from
13:00 KST NWP at 12–35 h lead" is an **information-theoretic lower bound** (Fano-type: bound the
achievable error by the mutual information between the covariates and the target). I located
teaching notes for exactly this (TTIC info-theory L8, Yale 598 Lec 13, CMU 10-704 Lec 21, Wisconsin
ORIE 7790 Lec 20, UMass CS 690M Lec 21) but **fetched none of them** before the stop order, so I make
**no** Fano claim here. §5.2's matched-pair bracket is the *non-information-theoretic* substitute and
is strictly weaker but requires no assumptions and no fit.

---

## 6. SMALL-SAMPLE DENSITY LIMITS — what the taught rules actually say

`directly_supported` for the following, all from S3 (CMU 36-708 *Density Estimation*):
  * Histogram construction and bin count: "There are **N ≈ (1/h)^d such bins and each has volume
    `h^d`**"; per-bin estimate `θ̂_j` is a **binomial proportion on `n` draws with success
    probability `θ_j ≈ C h^d`**, so its relative error scales as `1/sqrt(n h^d)`.
  * Minimax lower bound over the Lipschitz class (their Theorem 2):
    `inf_p̂ sup_{P∈P(L)} E ∫ (p̂ − p)^2 ≥ C (1/n)^{2/(d+2)}` — **no estimator beats this**, so refining
    the binning past the point where `n h^d` is small buys nothing and costs variance.
  * Bandwidth/bin-width selection: leave-one-out CV for histograms is available in closed form,
    `R̂(h) = 2/((n−1)h) − ((n+1)/((n−1)h)) Σ_j θ̂_j²`, and **Stone's theorem** (their Theorem 12)
    licenses CV-chosen bandwidth as asymptotically optimal for the *density*.
  Locator: https://www.stat.cmu.edu/~larry/=sml/densityestimation.pdf , §3, §3.1, Theorem 2, Theorem 12.

**The migration point that matters most here** — `near_match_only`: Stone's theorem licenses CV for
**density accuracy**, and S1 (modal regression) explicitly proposes a **different** bandwidth rule
for the **mode** — "selecting the smoothing bandwidth of the KDE based on **minimizing the size of
prediction sets**". **The bandwidth that is right for the density is NOT the bandwidth that is right
for the mode.** Our 181-class binning was presumably chosen for density fidelity and our `T` was
tuned for score; the taught structure says these are two different smoothing parameters serving two
different functionals, and we have been conflating them. Concretely: `181 classes` + `T=0.75` is one
over-resolved density plus one global re-smoothing, where the theory wants **one bandwidth chosen
for the mode functional**. This predicts that a **coarser binning with `T` closer to 1** should reach
a *similar* score, and that the `(bins, T)` surface has a ridge rather than an isolated peak.
**FALSIFIER**: if the `(bins, T)` surface has a sharp isolated optimum rather than a ridge, the
two-parameter conflation story is wrong.
**Cheapest test**: re-aggregate saved 181-class probabilities to 60 and 30 classes (pure
post-processing, no refit) and re-run the existing `T` grid on each. If the ridge exists, we gain a
free variance reduction; if not, we learn the current setting is genuinely isolated.

---

## 7. RANKED RECOMMENDATIONS (cheapest and most decisive first)

1. **★ Matched-pair MAE-floor bracket `D̂/2 ≤ MAE_floor ≤ D̂` from the M252 analog index** (§5.2).
   Zero fit, seconds, uses only training rows. **Can close the entire accuracy axis in one number.**
   Report `D̂`, the neighbour-distance distribution, and `D̂` as a function of neighbour radius.
2. **★ Null-oracle calibration of the +0.094 per-row member-selection headroom** (§5.3c). Zero fit,
   one permutation. Decides whether the ensemble axis is dead or merely badly exploited, and
   retroactively explains the two failed gates either way.
3. **Median-of-member-actions vs arithmetic blend on the 40.2% high-spread rows** (§M2c). Zero fit,
   one `numpy.median`. Expected magnitude small; do it because it is free.
4. **Re-aggregate 181 classes to a reward-aligned coarse grid and re-run the existing `T` grid**
   (§M3c, §6). Pure post-processing of saved probabilities. Tests the bins/`T` conflation.
5. **Row-weighted in-band fraction** (§2.1c). One number; gates whether a deadzone-linear training
   loss is worth a fit at all. My prediction from root's energy tiers: **it is not**.
6. **SLP / BLP instead of the arithmetic pool** — only if member *distributions* (not just actions)
   are still stored. 1–2 dof, so it survives the AGENTS.md fold-outside gate where the 21-dof and
   3-dof blends did not. This is the one device with a *theorem* behind it (S6).
7. **Modal-regression bandwidth rule** (minimise prediction-set size, S1) as a score-free replacement
   for the `T` grid search. Worth it mainly because it does not consume a validation surface, and we
   have none left.

**What I will NOT recommend, and why**: Platt, isotonic, focal, reverse-label-smoothing, and further
`T` search. The first two only re-calibrate and consume a validation surface we no longer possess;
the second two are in the same family as the `T` we have already grid-searched to its argmax; and
**all four are re-allocation devices in a situation that root has measured as accuracy-limited.**

---

## 8. EXPLICIT `insufficient` GAPS (what a resumed lane should fetch first)

1. **N1** — Gneiting, Balabdaoui & Raftery 2007 JRSS-B: the verbatim "**maximise the sharpness of the
   predictive distributions subject to calibration**" sentence. I never obtained it. Everything I
   wrote about the sharpness principle rests on S6 instead, which cites it but does not restate it.
2. **N5** — Ehm, Gneiting, Jordan & Krüger 2016: the Schervish mixture representation, which is the
   exact theorem behind "maximising a proper score does not maximise a downstream decision payoff"
   (a proper score is a *mixture over elementary cost-loss decision problems*, so it optimises an
   average over decision problems, not ours). **This is the single most important unfetched item**;
   it would have supplied the brief's requested "exact statement" and would license the construction
   of a **reward-tailored weighted scoring rule** concentrated on our band.
3. **N2** — Guo et al. 2017 + CIS 7000 Lecture 11. Needed to state the direction argument rigorously:
   modern NNs are **over-confident** and need `T > 1`; **we need `T < 1`**, i.e. our object is
   under-confident. That asymmetry is diagnostic — it points at the *pooling* mechanism (M2), not at
   the network-calibration mechanism — but I could not verify the Guo direction from a fetched source.
4. **N3** — a fetched teaching locator for difference-based residual-variance estimation. §5.2's
   bracket is self-contained algebra and does not depend on it, but the *taught* framing does.
5. **N4 / Fano** — no information-theoretic ceiling claim is made anywhere in this document. §5.4.
6. **N6** — Kim & Pollard cube-root asymptotics and Horowitz's smoothed maximum score. This is the
   exact literature for "maximising a **discontinuous** band objective", including the result that the
   *smoothed* version attains a faster rate than the raw one when the smoothing bandwidth is chosen
   correctly — which is very likely the correct diagnosis of the smoothed-step LightGBM failure
   (**bandwidth too wide, so the objective collapsed to its L1 limit**). I state this as a hypothesis
   only; it is unverified.
7. **Adaptive / unequal-width binning** — I found no fetched teaching source. §M3's reward-aligned
   re-binning proposal is therefore an engineering suggestion, not a cited one.
8. **Rate exponents from S1** — I read S1's abstract, contribution list and estimator definition, but
   did **not** extract its explicit convergence-rate theorem. The claim "mode functionals have
   slower-than-`n^{-1/2}` rates" is `near_match_only` in this document.

---

## 9. STANDING-RULE COMPLIANCE

Per AGENTS.md §"Standing rule", none of the proposals in §7 are blends or comparisons yet, so no
policy/weight-provenance/row-key triple is asserted. When proposal 3 or 6 is executed by the root, it
must state (a) which policy produced each input, (b) whether weights were fitted in-sample or
fold-outside, and (c) the row-alignment key set. Proposals 1, 2, 4 and 5 are diagnostics that
produce no submittable artifact and involve no fit, no score against the lockbox, and no external
data acquisition.
