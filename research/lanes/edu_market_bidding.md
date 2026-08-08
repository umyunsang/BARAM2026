# Lane EDU-MARKET-BIDDING — the settlement rule is a *size-biased modal-interval* problem, not a newsvendor

**Lane:** `EDU-MARKET-BIDDING`
**Retrieval window:** 2026-08-09, truncated early by an explicit root STOP instruction ("write now, a short file that exists beats a long one that does not").
**Sources touched:** 3 fetched to text + 8 located-by-URL-only (cap was 14; the lane was stopped, not exhausted).
**Fits run:** 0. Scores computed: 0. Lockbox reads: 0. Downloads: 0. Files written: this one only.

**VERDICT (one line):** the taught object is **not** the newsvendor / critical fractile and **not** quantile
regression; it is the **maximiser of a boxcar-kernel-smoothed *size-biased* conditional density** — in the
teaching literature a **conditional-mode / modal-regression** (equivalently **modal-interval / HDR-centre**,
equivalently **maximum-score / P-model target-hitting**) functional applied to the **energy-weighted
(length-biased) law**. This naming is real and it produces a **closed-form four-point first-order condition**
we have not implemented. **But it is an ACTION-SIDE re-allocation device only**: it cannot move the
accuracy frontier, and the root's `s17_n47` elasticity measurement says the action side is already
near-efficient. **The only accuracy-side consequence of the market framing is a TRAINING-WEIGHT
consequence (size-biased empirical risk), not an action consequence.**

---

## 0. Evidence contract

Every material claim carries exactly one of:

| tag | meaning |
|---|---|
| `directly_supported` | a locator I actually fetched and read, or a named local BARAM record |
| `contradicts_premise` | evidence that defeats a stated premise of the lane brief |
| `near_match_only` | a real taught result under a materially different loss/market/weighting |
| `insufficient` | URL located but text **not** verified in this bounded window, or not reached at all |

Three things are kept separate throughout: **(a) source fact**, **(b) provisional BARAM migration
hypothesis**, **(c) local evidence needed**. **No source effect size is ever quoted as a BARAM expected
gain.** `directly_supported`

---

## 1. Source register

### 1.1 Fetched and read (text verified)

| ID | Source | Exact locator | Tag |
|---|---|---|---|
| **S1** | **MIT 15.772J / EC.733 *D-Lab: Supply Chains*, Fall 2014**, handout *"Newsvendor Inventory Problem"*, MIT OpenCourseWare, `MIT15_772JF14_Newsboy.pdf`. <https://ocw.mit.edu/courses/15-772j-d-lab-supply-chains-fall-2014/6952be57b43aa185119c6f114908bcc5_MIT15_772JF14_Newsboy.pdf> | Verbatim: *"We choose the first Q such that `Pr[do not sell Q+1st unit] = F(Q) >= underage_cost/(underage_cost + overage_cost)`"*; and the equivalent form *"`Pr[sell Q+1st unit] < overage_cost/(underage_cost + overage_cost)`"*; worked example *"the underage cost is $3.00 ... the overage cost is $0.50. Therefore, the critical fractile is 0.14"*. | `directly_supported` |
| **S2** | **Chen, Genovese, Tibshirani, Wasserman, *Nonparametric Modal Regression*, Annals of Statistics 2016, 44(2), 489–514, DOI 10.1214/15-AOS1373** (author-hosted PDF at Berkeley; this is the standard teaching reference for the mode functional in CMU 36-708 / 10-702 lineage). <https://www.stat.berkeley.edu/~ryantibs/papers/modal-aos.pdf> | Abstract verbatim: *"Modal regression estimates the local modes of the distribution of Y given X = x, instead of the mean, as in the usual regression sense, and can hence reveal important structure missed by usual regression methods."* §on estimation: the estimator is a KDE of the joint law plus a **partial mean-shift** iteration (Einbeck & Tutz 2006); verbatim caveat: *"Because this function f is generically nonconcave, we are not guaranteed that gradient ascent will actually attain a [global mode]"*; the paper *"propose[s] techniques for constructing confidence sets and prediction sets. The latter is used to select the smoothing bandwidth."* | `directly_supported` |
| **S3** | **KPX 전력시장운영규칙 (market rule) portal**, full-text index. <https://marketrule.kpx.or.kr/lmxsrv/law/lawFullLawTitle.do?SEQ=2&SEQ_HISTORY=41&showAttach=1> | Only the chapter index resolved: 제1장 총칙 / 제2장 가격결정 / 제3장 전력의 거래 / **제4장 계량과 정산 및 결제** / 제5장 전력계통 운영. The 재생에너지 발측/예측제도 settlement article itself was **not** reached before the stop. | `insufficient` |

### 1.2 Located by URL only — text NOT verified (all `insufficient`)

| ID | Source | Why it was on the list | Tag |
|---|---|---|---|
| S4 | **DTU course 31761 / 46755 *Renewables in Electricity Markets* (P. Pinson)**, exercise session 5. <http://pierrepinson.com/31761/Exercises/31761-exsession5.pdf> | This is the canonical *course* derivation of the optimal wind offer under imbalance penalties. | `insufficient` |
| S5 | Kazempour et al., *Optimal Offering Strategies for Wind Power in Energy and Primary Reserve Markets*, DTU Orbit. <https://backend.orbit.dtu.dk/ws/files/121887341/07399764.pdf> | Companion research to S4. | `insufficient` |
| S6 | Morales, Conejo, Madsen, Pinson, Zugno, *Integrating Renewables in Electricity Markets*, Springer 2014 (course text for S4 and for OSU Conejo's Electricity Markets course <https://u.osu.edu/conejo.1/courses/electricity-markets/>). | Textbook chapter deriving offer-as-quantile. | `insufficient` |
| S7 | Stanford **EE364a** lecture *Stochastic programming* <https://web.stanford.edu/class/ee364a/lectures/stoch_prog.pdf> and *Chance constrained optimization* <https://stanford.edu/class/ee364a/lectures/chance_constr.pdf> | Needed for the taught statement that probability-maximisation objectives are generically non-convex. | `insufficient` |
| S8 | CERC (India) **Deviation Settlement Mechanism** regulations. <https://www.cbip.org/regulationsdata/CERC/CERC_3_Deviation/Consolidated%20Regulation%20Deviation%20Settlement.pdf>, <https://cercind.gov.in/Regulations/168_reg.pdf> | The only *other* real market with an explicit **absolute-error slab** (band) deviation charge for wind/solar. | `insufficient` |
| S9 | KPX board item on the 재생에너지 발전량 예측제도 <https://new.kpx.or.kr/boardDownload.es?bid=0209&list_no=68418OOO202212150934139612&seq=2>; KIEE paper <http://www.tkiee.org/kiee/XmlViewer/f415504>; IKEEE 26(3):355 <https://dx.doi.org/10.7471/ikeee.2022.26.3.355>; DBpia NODE11143185 | Korean-language candidates for a **metric-matched** derivation of the optimal action under the 6%/8% rule. **This would have been the single most valuable source in the whole project and it was not reached.** | `insufficient` |
| S10 | MIT **IDS.505J / ESD.934 / 15.032J *Engineering, Economics and Regulation of the Electric Power Sector*, Spring 2010** (Pérez-Arriaga), OCW lecture notes. <https://ocw.mit.edu/courses/ids-505j-engineering-economics-and-regulation-of-the-electric-power-sector-spring-2010/> | Institutional framing of imbalance settlement. | `insufficient` |
| S11 | Brehmer & Gneiting (2021), *Scoring interval forecasts*, Eq. (9) `MI_c(F)=argmax_{[a,b]}{F(b)-F(a-) : b-a<=2c}`, Eq. (11) `S(x,y) = -1{x-c<=y<=x+c}`, Thm 3.10. | **Carried over from the local record `research/lanes/s17_n18_step_reward.md` §2 (E2)**, where it is already recorded as fetched-and-verified. Not re-fetched here. | `directly_supported` (via named local record) |

---

## 2. Source fact: what the courses actually derive, and why it is NOT our rule

### 2.1 The newsvendor / critical-fractile recipe is a **CDF** condition, and it is a **convexity** artefact

`directly_supported` **(S1)**. MIT 15.772J derives, by marginal analysis, that the optimum is the first `Q` with

```
F(Q) >= Cu / (Cu + Co)          [critical fractile / critical ratio]
```

The reason it is a *quantile* is structural and the handout makes it visible: the payoff is **piecewise
LINEAR** in the deviation `Q - D`, so the derivative of the expected payoff contains `Pr[D <= Q] = F(Q)`,
a **CDF**. Every taught generalisation in that family (multi-period, salvage, backorder, service-level
target) keeps that property: *linear-in-deviation payoff ⇒ CDF ⇒ quantile solution*.

**Consequence for BARAM — `contradicts_premise` for any "find the right fractile" plan.** Our FICR term is
**piecewise CONSTANT** in the deviation, not piecewise linear. Differentiating a piecewise-constant reward
against a smooth predictive law produces **DENSITY** evaluations, not CDF evaluations. So no critical
fractile exists for the FICR half. The lane brief's question — "does a band reward yield a quantile rule,
a mode rule, or neither?" — has the answer **mode rule**, and specifically a *weighted, two-scale,
size-biased* mode rule. `derived`

### 2.2 The taught power-market result *is* the critical fractile, and it applies to the wrong market

`near_match_only` **(S4–S6, `insufficient` on text)**. The standard course derivation in DTU 31761/46755
and Morales–Conejo–Madsen–Pinson–Zugno is: under a **two-price imbalance settlement** with linear
up/down penalty rates, the revenue-maximising day-ahead offer is a **quantile of the predictive
distribution at level `psi_down / (psi_up + psi_down)`** — literally the newsvendor critical fractile with
imbalance penalties substituted for underage/overage. **I did not verify the exact equation text in this
window; treat the level formula as `insufficient` and the *class* claim as `near_match_only`.**

**This is the crucial negative result of the lane.** The taught optimal-offering literature that everyone
would reach for is built for **linear imbalance pricing**. Korea's 예측제도 is a **tolerance-band incentive
payment**, which is a different object in the same taxonomy. Importing the quantile recipe would be a
category error, and it is exactly the error our own "39-level calibrated quantile function" attempt made
(brief: 1-NMAE +0.0029, FICR −0.0075, Total −0.0023). `derived`

### 2.3 The correct taught name: conditional mode / modal interval / HDR centre

`directly_supported` **(S2, S11)**. The functional `argmax_x E[-1{x-c <= Y <= x+c}]`, i.e. the centre of the
highest-probability window of fixed half-width `c`, is named the **modal interval `MI_c(F)`** in
Brehmer & Gneiting (S11 Eq. 9/11), and its regression version — estimate `argmax_y f(y|x)` rather than
`E[Y|x]` — is **modal regression** (S2, verbatim in §1.1). S2 supplies three things a grid search does not:

1. an **estimator with known asymptotics** (KDE of the joint law + partial mean-shift, Einbeck–Tutz);
2. a **bandwidth-selection rule** derived from prediction-set coverage rather than from density fit;
3. an explicit warning that the criterion is **generically non-concave**, so local ascent may miss the
   global maximiser (verbatim in §1.1).

**Named estimator classes for our object, in the order a course would name them:**

| Name | Where it is taught | What it buys us |
|---|---|---|
| **Modal regression / conditional-mode estimation** | S2 (AoS 2016; CMU 36-708 / 10-702 lineage) | the estimator, the bandwidth rule, the non-concavity warning |
| **Modal interval `MI_c(F)` / HDR centre** | S11 (via local record) | the exact functional and its *only* strictly consistent score |
| **Maximum score / empirical welfare maximisation** | econometrics + policy-learning courses | direct criterion maximisation instead of plug-in; cube-root-type rates |
| **Chance-constrained "P-model" / target-oriented satisficing** | stochastic-programming courses (S7 `insufficient`) | "maximise `P(hit target)`" as a first-class objective; non-convexity |
| **Size-biased (length-biased) distribution** | probability courses (inspection paradox) | **the name for our actual-weighting** — see §4 |

`insufficient`: I did not verify a course page that *names* the band-payment power-market problem in these
terms. The mapping in §3 is `derived` algebra, not a quoted taught result.

---

## 3. Derived: the exact BARAM objective and its four-point first-order condition

All of §3 is **`derived`** — algebra on the official formula as recorded in `AGENTS.md` and in
`research/lanes/s17_n18_step_reward.md` §3. No fit, no data read.

Work in capacity-relative units: `z = Y/C`, `alpha = a/C`, half-widths `h1 = 0.06`, `h2 = 0.08`.
Utility decomposition (already in the local record, S11 lane): `u = 4*1{e<=.06} + 3*1{.06<e<=.08} = 1{e<=h1} + 3*1{e<=h2}`.

For one group with `N` eligible rows and mean eligible output `m = mean(z)`:

```
Total_g  =  0.5*(1 - (1/N) sum_i |alpha_i - z_i|)  +  0.5 * sum_i z_i*u_i / (4 * sum_i z_i)
```

Maximising row-wise (the FICR denominator does not depend on the action), the per-row objective is,
up to a positive constant,

```
J(alpha) =  E[ z * ( 1{|alpha-z|<=h1} + 3*1{|alpha-z|<=h2} ) ]  -  4m * E|alpha - z|
```

**Reading 1 — the FICR term is a boxcar-smoothed SIZE-BIASED density.**
Let `mu(x) = E[z|x]` and let `f~(z|x) = z f(z|x)/mu(x)` be the **size-biased (length-biased) conditional
density**. Then

```
E[z*(...)] = mu(x) * [ P~(|alpha-z| <= h1) + 3 * P~(|alpha-z| <= h2) ]
           = mu(x) * (K * f~)(alpha),   K(u) = 1{|u|<=h1} + 3*1{|u|<=h2}
```

So the FICR-optimal action is **the argmax of the size-biased conditional density convolved with a
two-step boxcar kernel of half-widths 0.06 and 0.08 and weights 1 and 3.** That is a modal-regression
target with a *known, fixed, non-tunable kernel supplied by the market rule itself.*

**Reading 2 — the exact FOC (this is the structure we have not implemented).**
Differentiating `G(alpha) = int_{alpha-h1}^{alpha+h1} z f dz + 3 int_{alpha-h2}^{alpha+h2} z f dz`:

```
G'(alpha) = [(alpha+h1) f(alpha+h1) - (alpha-h1) f(alpha-h1)]
        + 3*[(alpha+h2) f(alpha+h2) - (alpha-h2) f(alpha-h2)]
```

and `d/d alpha E|alpha - z| = 2F(alpha) - 1`. Hence the **exact stationarity condition** for the full
BARAM objective is

```
(alpha-h1) f(alpha-h1) + 3 (alpha-h2) f(alpha-h2)
   -  (alpha+h1) f(alpha+h1) - 3 (alpha+h2) f(alpha+h2)
   +  4m * (2F(alpha) - 1)   =   0                                   (*)
```

Three structural facts follow immediately:

1. **The action depends on the conditional law at exactly FIVE numbers**: the density at the four band
   edges `alpha +/- 0.06`, `alpha +/- 0.08`, and the CDF at `alpha`. It does **not** depend on the other 34
   levels of a 39-level quantile function. This is a *precise diagnosis of why proper predictive objects
   degrade*: CRPS/pinball/log-score calibration spends its statistical budget on global shape, while the
   action is a **local four-point density contrast**. Global calibration does not imply local
   density-contrast calibration. `derived`, and it is consistent with all three measured failures in the brief.
2. **Setting `m = 0` recovers the classical modal-interval FOC** "equal (weighted) density at the interval
   endpoints", the band analogue of the shortest-interval / HDR condition. The NMAE half enters only as a
   **restoring force of strength `4m` pulling toward the conditional median** (`2F(alpha)-1` vanishes at the
   median). So the deployed action is a *mode-median compromise with an explicitly computable exchange rate*
   `4m`. We have never written that exchange rate down. `derived`
3. **`T = 0.75` sharpening is a crude surrogate for the size-bias plus mode-seeking.** Raising a binned
   conditional mass to `1/T` and renormalising is a *tempering* that monotonically concentrates mass on the
   modal bin; as `T -> 0` the argmax of the tempered utility converges to the mode. That our grid search
   picked `T < 1` is `directly_supported` evidence *from our own measurements* that the target functional is
   mode-like, not mean-like or quantile-like. But tempering is a **global** operation and the FOC (*) is a
   **local four-point** condition; they coincide only by accident. `derived`

---

## 4. Actual-weighting: the size-bias, and where it does and does not help

This is the root's priority question. Answer in two parts.

### 4.1 Source-side naming

`directly_supported` (definition-level): a density proportional to `y f(y)` is the **size-biased** (or
**length-biased**) version of `f`. It is standard teaching material (renewal theory / inspection paradox).
`insufficient`: I did **not** reach a course page that applies size-biasing to a forecast-accuracy incentive
payment; the application in §3 is `derived`.

### 4.2 Two distinct places the size-bias can act — only one of them is an accuracy device

| Where | What it changes | Is it a re-allocation device or an accuracy device? |
|---|---|---|
| **Inside the ACTION** (`E[z*u]` instead of `E[u]`) | shifts each `alpha` upward relative to the plain modal interval, because the size-biased law is stochastically larger | **Re-allocation only.** It moves us along the fixed (1-NMAE, FICR) frontier of the *same* predictive law. |
| **Inside the FIT** (row weights `w_i` proportional to the metric's own weighting) | changes which rows the model is *accurate on* | **Accuracy device, in the sense the root cares about** — it does not add information, but it re-allocates *estimation capacity* to the rows the metric actually pays for. |

**Explicit answer to the root's question:** *the mode/HDR/market-bidding framing is a re-allocation device
for the ACTION.* Any action rule `alpha(x)` is a measurable function of the same conditional law; it cannot
change the information content of the forecast and therefore cannot move the accuracy frontier. Given
`reports/s17_n47_ficr_elasticity.json` (action policy already converts accuracy into settlement efficiently;
reaching FICR 0.461520 needs `s ~= 0.85`, a 15% uniform MAE reduction), **I do not propose any action-side
change as a path to the 0.0285 gap.** `derived` from the root's own measurement — this lane
*confirms* rather than contests it.

The one non-exhausted consequence of the market framing is the **training weight**, and it is genuinely
different from everything in the "closed on evidence" list because it changes the fit, not the action.

### 4.3 The exact metric-implied row weight (derivable with no fit)

From §3, the total objective is `0.5*(1/(NC)) * sum |a_i - y_i|` plus `0.5/(4 sum y) * sum y_i u_i`. So the
**metric's own per-row weight on getting row `i` right is**

```
w_i  proportional to   (1/N)  +  (y_i / sum_j y_j) * (band-crossing sensitivity at row i)
```

The first term is uniform over eligible rows; the second is **proportional to delivered energy**. A model
trained with uniform weights is therefore matched to the NMAE half and **mismatched to the FICR half**,
which carries 50% of the score. `derived`

### 4.4 What accuracy the high-energy hours require — arithmetic on the root's numbers only

Using only `s17_n47`: energy shares 34.0% pay 4, 9.1% pay 3, 56.9% pay 0, giving
`FICR = (4*0.340 + 3*0.091)/4 = 1.633/4 = 0.40825`. Target `FICR = 0.461520` needs
`4*p4 + 3*p3 = 1.846`. Holding `p3 = 0.091` fixed, **`p4` must rise from 0.340 to 0.393** — i.e.
**+5.3 percentage points of DELIVERED ENERGY must move inside the 6% band.** `derived` (arithmetic only;
this is a restatement of the root's own target, not an independent estimate, and it is **not** a claimed gain).

Since 56.9% of energy is currently beyond 8%, the required 5.3 points can in principle come from anywhere
in that mass; the root already established that the 6–10% shell holds only 17.0% of energy, so boundary
shaving cannot supply it. **The lane's contribution here is only the restatement in energy-share units,
which is the natural unit for the size-biased objective.**

---

## 5. Formulation table — each taught rule mapped to our exact metric

| # | Taught rule (source) | Its exact optimality condition | Maps to BARAM as | Reproduces / contradicts / new structure | Tag |
|---|---|---|---|---|---|
| F1 | **Newsvendor critical fractile** (S1, MIT 15.772J F14) | `F(Q) >= Cu/(Cu+Co)` — a **CDF** condition | Applies to **neither** half cleanly. The NMAE half alone would give `F(a)=1/2` (median). The FICR half gives **no** fractile at all. | **Contradicts** any "find the right quantile level" plan; **confirms** why the 39-level quantile route lost. | `directly_supported` (S1) + `derived` (mapping) |
| F2 | **Power-market optimal offering as a quantile of the predictive density at level `psi_down/(psi_up+psi_down)`** (S4–S6) | CDF condition, inherited from linear imbalance pricing | Correct for a **two-price imbalance** market; **wrong market** for a tolerance-band incentive payment | **Contradicts** direct import. Important because this is the result a power-systems course would hand us. | `near_match_only`; equation text `insufficient` |
| F3 | **Modal interval `MI_c(F)`** (S11 via local record) | `argmax_{[a,b]} {F(b)-F(a-) : b-a <= 2c}` | With `c = 0.06C`, this is the **single-tier, unweighted** version of our FICR term | **Reproduces** our heuristic's *direction* (mode-seeking) and explains why `T<1` won the grid | `directly_supported` |
| F4 | **Modal regression** (S2, AoS 2016) | `argmax_y f(y|x)`, estimated by KDE + partial mean-shift; non-concave criterion; bandwidth chosen by prediction-set coverage | Our target is the **kernel-smoothed** mode with the market-supplied boxcar kernel `K = 1{|u|<=.06} + 3*1{|u|<=.08}` applied to the **size-biased** law | **New structure**: a named estimator, a principled bandwidth rule, and a non-concavity warning that our grid search silently assumes away | `directly_supported` (S2) + `derived` (mapping) |
| F5 | **Chance-constrained P-model / target-oriented satisficing** (S7) | maximise `P(outcome in target set)`; generically non-convex | Exactly the FICR half if one drops the energy weight | **Reproduces** the objective's *shape*; supplies the non-convexity vocabulary | `insufficient` (course text unverified) |
| F6 | **Size-biased / length-biased distribution** | `f~(y) = y f(y)/E[Y]` | The **actual-weighting** in FICR, exactly | **New structure** if — and only if — our action currently maximises `E[u]` rather than `E[z*u]`; and **new structure** for the fit weights regardless | `derived` |
| F7 | **Four-point density-balance FOC (*)** (§3, no external source) | `(a-h1)f(a-h1) + 3(a-h2)f(a-h2) - (a+h1)f(a+h1) - 3(a+h2)f(a+h2) + 4m(2F(a)-1) = 0` | Exact stationarity for the **full** BARAM objective, both halves | **New structure.** Turns the action into a **Z-estimator on 5 numbers** instead of a plug-in over a whole predictive law | `derived` |
| F8 | **CERC (India) DSM absolute-error slabs**; **Spain RD 661/2007 deviation tolerance band**; **KPX 재생에너지 발전량 예측제도 6%/8%** | tolerance-band deviation charge / incentive | The only real-world families **metric-matched** to ours | Unknown — **not reached** | `insufficient` (S3, S8, S9) |

---

## 6. Proposals, falsifiers, and the cheapest local diagnostic per candidate

Each proposal states (a) source fact, (b) BARAM migration hypothesis, (c) local evidence needed, plus an
explicit falsifier. **No expected gain is quoted anywhere.**

### P1 — Size-bias audit of the deployed action  ← **CHEAPEST TEST IN THE LANE, ZERO FIT**
- **(a) Source fact:** the FICR numerator is `sum(y_i * u_i)`, so the per-row expected settlement is
  `E[z * u]`, i.e. an expectation under the **size-biased** conditional law, not `E[u]`. `derived` from the
  official formula.
- **(b) Migration hypothesis:** if the deployed argmax maximises a *probability-weighted* utility
  `sum_bins p_b * u(bin_b)` rather than an *energy-weighted* utility `sum_bins p_b * z_b * u(bin_b)`, then
  every action is systematically biased **low** relative to the exact FICR optimum, uniformly across the
  dataset.
- **(c) Local evidence needed (10 minutes, no fit, no new data):** read the action-construction code and
  check whether a `z_b` (bin centre in capacity units) factor multiplies the utility inside the argmax.
  If absent, recompute the argmax on the **already-saved** binned conditional distributions with the `z_b`
  factor included and re-score.
- **FALSIFIER:** if the `z_b` factor is already present, P1 is dead on the spot, at zero cost. If it is
  absent but adding it does not change the argmax on >1% of rows (plausible, since `z_b` varies slowly over
  a ±0.06C window), P1 is also dead. **Either outcome closes the axis in one sitting.**

### P2 — Four-point FOC residual audit
- **(a) Source fact:** stationarity condition (*) in §3. `derived`
- **(b) Migration hypothesis:** if the deployed action already satisfies (*) to within grid resolution, then
  the entire decision-theoretic axis is *provably* closed and the root's `s17_n47` conclusion is confirmed
  analytically rather than empirically. If the residual is **systematically one-signed**, there is a free
  constant shift.
- **(c) Local evidence needed:** on saved champion actions and saved binned distributions, evaluate the four
  edge densities and `F(alpha)`, form the residual of (*), and histogram it by group. No fit, arithmetic only.
- **FALSIFIER:** a residual histogram centred at zero with no group structure kills every remaining
  action-side idea, permanently and cheaply. **This is the correct way to *close* the axis rather than
  merely observe that grid search has converged.**

### P3 — Energy-targeted vs uniform accuracy counterfactual  ← **the root's lane-4 question, answered by arithmetic**
- **(a) Source fact:** the settlement is proportional to delivered energy (§4), so an MAE reduction on a
  high-`y` hour is worth strictly more than the same reduction on a low-`y` eligible hour, for the FICR half
  and equally for the NMAE half.
- **(b) Migration hypothesis:** the required `s = 0.85` **uniform** shrink may be achievable as a much
  milder shrink applied **only to the top energy decile/quartile of rows**.
- **(c) Local evidence needed:** re-run the exact `s17_n47` elasticity computation, but with `s` applied to
  a **restricted row set** (top-`q` by actual energy) and `s = 1` elsewhere; sweep `q` in {0.1, 0.25, 0.5}
  and find the `s'` reaching FICR 0.461520. Pure arithmetic on the arrays already loaded for `s17_n47`;
  **no fit, no new data, reuses an existing script.**
- **FALSIFIER:** if the required `s'` on the top quartile is not materially milder than 0.85, then energy
  concentration buys nothing and the accuracy requirement really is uniform — which would close the entire
  "concentrate effort on high-energy hours" idea with one number. **Note the honest asymmetry: this
  diagnostic uses actual energy to define the row set, so it is an ORACLE upper bound, not a deployable
  policy. Its value is as a ceiling: if the ceiling is not attractive, stop.**

### P4 — Metric-matched training weights (fit required, root-only)
- **(a) Source fact:** §4.3 exact weight decomposition. `derived`
- **(b) Migration hypothesis:** training with `w_i ∝ (1/N) + c * y_i / sum(y)` instead of uniform weights
  matches the estimator's loss to the metric's own weighting.
- **(c) Local evidence needed:** a **no-fit precursor first** — decompose the current `sum|a-y|` by actual-
  energy decile and compare each decile's share of total error to its share of settlement weight. If error
  share and weight share already coincide, uniform training is already matched and P4 is dead **without any
  fit**. Only if there is a material mismatch is a re-weighted fit warranted.
- **FALSIFIER:** coincidence of the two shares. Also note `contradicts_premise` risk: because NMAE is
  uniform over eligible rows and carries 50% of the score, over-weighting high-energy rows will *raise*
  NMAE; this is a frontier move, not a free lunch, and must be scored on both halves.

### P5 — Modal-regression estimator proper (fit required, LOW priority)
- **(a) Source fact:** S2 supplies KDE + partial mean-shift, coverage-based bandwidth selection, and a
  non-concavity warning. `directly_supported`
- **(b) Migration hypothesis:** replacing the binned-distribution + tempering pipeline with a direct
  conditional-mode estimator with a coverage-selected bandwidth would target the right functional directly.
- **(c) Local evidence needed:** none yet — **P2 gates this.** If the FOC residual is already ~0, P5 is
  strictly redundant.
- **FALSIFIER:** P2 returning a zero-centred residual. **Also: per the root update, P5 is an action-side
  re-allocation device and therefore cannot on its own supply the 15% MAE reduction the gap requires.
  I do not recommend it as a gap-closing candidate.**

---

## 7. Explicit `insufficient` gaps left by the early stop

| Gap | Why it matters | Tag |
|---|---|---|
| The exact KPX 전력시장운영규칙 article text for the 재생에너지 발전량 예측제도 (6%/8%, 4원/3원) | Would be the **only metric-matched official source in the project**; would also settle whether the real scheme pays on *actual* or on *contracted* energy | `insufficient` (S3, S9) |
| Any Korean university lecture note / thesis deriving the optimal bid under that exact rule | Would either confirm or refute the modal-interval framing against a domestic teaching source | `insufficient` (S9) |
| DTU 31761/46755 exercise text and the Morales et al. chapter equation | Would upgrade F2 from `near_match_only` to `directly_supported` and pin the quantile-level formula | `insufficient` (S4–S6) |
| Stanford EE364a chance-constraint lecture text | Would supply the taught non-convexity statement for F5 | `insufficient` (S7) |
| CERC DSM slab regulation text | The only other band-slab market; any published optimal-bidding analysis there would be near-metric-matched | `insufficient` (S8) |
| A taught treatment of **actual-weighted** (size-biased) forecast incentives specifically | §4 is `derived`, not sourced | `insufficient` |

---

## 8. Bottom line for the root

1. **Named object:** *size-biased, two-scale, boxcar-smoothed conditional mode* — taught as **modal
   regression** (S2) / **modal interval** (S11), **not** newsvendor, **not** quantile regression. The
   critical-fractile recipe is a **CDF** condition that exists only because newsvendor payoffs are
   piecewise-linear (S1, verbatim); ours are piecewise-constant, so it does not exist for us.
2. **New structure we have not implemented:** the four-point FOC (*), which shows the action depends on the
   conditional law at **five numbers only** and explains all three measured "proper predictive object"
   failures as a mismatch between global calibration and a local density contrast.
3. **Honest answer to the root's sharpening question:** the market/bidding/mode framing is an **action-side
   re-allocation device** and, given `s17_n47`, I do **not** propose it as a route to the 0.0285 gap. The
   only accuracy-relevant consequence is the **training weight** (§4.3), and even that is a frontier move
   because NMAE is uniform while FICR is energy-weighted.
4. **Single cheapest local test:** **P1** — grep the action-construction code for the energy factor `z_b`
   inside the argmax utility (`E[z*u]` vs `E[u]`). Zero fit, zero data, minutes. It either exposes a
   uniform low-bias in every deployed action or closes the size-bias axis outright.
5. **Most decision-relevant cheap test:** **P3** — re-run the `s17_n47` elasticity with the shrink applied
   only to the top energy quantile, to price "concentrate effort on high-energy hours" as an oracle ceiling
   before any modelling effort is spent on it.
