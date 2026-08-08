# S14 — FOUNDATION-INSIGHT LANE: what university AI/ML teaching material says about *this* problem

**Lane type:** read-only research (AGENTS.md bounded allowance). No repository write outside `research/`.
No model fit, no lockbox touch, no git mutation, no upload.
**Source class (deliberately different from S5/S6/S7/S12/S13):** university course material — lecture notes,
section notes, problem sets, exams, slide decks — and the textbooks those courses set. **Not** papers,
**not** benchmarks, **not** competition write-ups.
**Date:** 2026-08-08 (agent clock). **Queries:** 145. **Documents fetched and read in raw:** 49.

---

## 1. Evidence-grade convention

| Tag | Meaning |
|---|---|
| **[원문]** | I fetched the primary course document (PDF/HTML) into this session and the quoted string is a verbatim slice of the extracted text. URL given. Reproducible with the fetch script in §2.3. |
| **[원문-요약]** | Primary document fetched and read; the statement is my paraphrase of a passage I read in raw, not a verbatim quote. |
| **[전문미확인]** | I have only a search snippet / abstract / table-of-contents line. The claim may be right but I did not read the source text. **Treat as a lead, not evidence.** |
| **[유도]** | My own derivation from a quoted principle plus the MEASURED FACTS given in the brief. Mathematics is mine; errors are mine. Not a course claim. |

PDF text extraction is `pypdf`; ligatures and math glyphs are sometimes mangled (e.g. `ﬁ`, `𝓁`, `/g2/g3`).
I have preserved the mangling in quotes rather than silently "fixing" it, so that a reviewer can grep
the source. Where a formula is unreadable after extraction I restate it in `[유도]` form.

**Standing warning.** Everything in §4 and §5 is *reasoning about* the situation, not measurement of it.
The only measurements in this document are the eight facts supplied in the brief. Any number I compute
in §4 is arithmetic on those eight facts and is tagged **[유도]**.

---

## 2. Search log

### 2.1 What I searched for and why

Five prior lanes searched the literature of *wind power forecasting*. This lane searched the literature of
*teaching people what a supervised learning problem is*. The hypothesis behind the lane: after 30 failed
treatments, the binding constraint is unlikely to be a missing technique; it is more likely to be a
mis-stated problem, and mis-stated problems are exactly what introductory and theory courses are built to
diagnose. So I hunted for the places where courses say "here is how you tell that you are working on the
wrong thing".

Search ordering: (A) Bayes error / irreducible risk → (B) decision theory for non-standard losses →
(C) truncation & selection → (D) statistical learning theory & why noise features hurt →
(E) model selection / winner's curse / stopping → (F) ensemble algebra → (G) transduction & covariate
shift → (H) hierarchical/Stein → (I) problem reformulation. Plus opportunistic follow-ups.

### 2.2 Full query list (145, in execution order)

```
  1. Stanford CS229 lecture notes bias-variance decomposition noise term sigma^2
  2. STATS214 CS229M machine learning theory lecture notes Stanford Tengyu Ma
  3. CS229 lecture notes bias variance tradeoff irreducible error Bayes error
  4. MIT 6.036 lecture notes bias variance noise decomposition
  5. CS229 section notes decision theory Bayes risk optimal predictor absolute loss median
  6. Boyd EE364a convex optimization lecture slides approximation and fitting robust
  7. CS229 main notes pdf regularization model selection cross validation
  8. Stanford CS228 probabilistic graphical models notes decision theory utility
  9. MIT 6.7900 machine learning lecture notes
 10. Shalev-Shwartz Ben-David understanding machine learning Bayes optimal predictor irreducible error approximation estimation
 11. MIT 6.867 machine learning lecture notes ocw
 12. Stanford STATS305 applied statistics lecture notes truncation censoring
 13. MIT 18.657 mathematics of machine learning lecture notes Rigollet
 14. Hastie ESL elements of statistical learning chapter 7 model assessment selection optimism
 15. Murphy probabilistic machine learning book decision theory optimal action loss function
 16. MIT 6.041 6.431 probabilistic systems analysis least mean squares estimation notes
 17. CS109 Stanford probability for computer scientists notes
 18. self-training pseudo-labeling theory lecture notes Stanford CS229M unlabeled data
 19. MIT 6.437 inference and information lecture notes Bayesian estimation cost function
 20. Stanford CS229 problem set solutions locally weighted regression asymmetric loss
 21. covariate shift importance weighting lecture notes Stanford CS229 domain adaptation
 22. transductive learning lecture notes course unlabeled test inputs
 23. MIT 6.390 intro machine learning lecture notes feature representation
 24. Stanford CS229 notes generalization bounds Tengyu Ma
 25. James-Stein estimator shrinkage lecture notes Stanford STATS
 26. Cornell CS4780 lecture notes bagging boosting variance reduction correlated
 27. Wooldridge truncated regression versus censored regression lecture slides
 28. ambiguity decomposition ensemble error correlation lecture notes Krogh Vedelsby
 29. quantile regression pinball loss lecture notes course optimal predictor quantile
 30. Stanford CS229 notes 'newton's method' 'GLM' exponential family choose the right loss
 31. MIT 14.382 econometrics lecture notes sample selection
 32. ESL chapter 8.7 bagging chapter 16 ensemble learning correlation rho variance
 33. truncated regression Tobit lecture notes course selection on the dependent variable
 34. Heckman selection model lecture notes econometrics course MIT 14.382
 35. bias variance covariance decomposition ensemble lecture notes Brown
 36. CS230 lecture slides bias variance avoidable bias error analysis ceiling analysis
 37. Stanford CS229 lecture notes error analysis ablative analysis debugging learning algorithms
 38. why adding irrelevant features hurts random forest theory course notes noise features
 39. STATS214 lecture notes uniform convergence Rademacher complexity dimension d over n
 40. cs229 notes advice for applying machine learning error analysis ceiling
 41. Andrew Ng machine learning yearning avoidable bias human level performance optimal error rate
 42. curse of dimensionality nearest neighbor lecture notes ESL chapter 2 local methods high dimensions
 43. Stanford CS230 deep learning lecture notes error analysis human level performance bayes error
 44. MIT 6.036 6.390 lecture notes 'the machine learning problem' choosing loss function decision
 45. MIT 18.657 lecture notes empirical risk minimization excess risk approximation error estimation error
 46. MIT 6.S191 lecture uncertainty evidential deep learning aleatoric epistemic
 47. adaptive data analysis holdout reuse lecture notes Dwork ladder leaderboard overfitting
 48. aleatoric versus epistemic uncertainty lecture notes course irreducible noise
 49. Kaggle leaderboard overfitting theory ladder mechanism Blum Hardt
 50. Stanford CS234 reinforcement learning notes maximizing expected reward action versus prediction
 51. Stanford STATS course notes multiple testing selective inference winner's curse post-selection
 52. hierarchical bayes partial pooling lecture notes multilevel model shrinkage unequal group sizes
 53. MIT 6.7900 lecture notes model selection generalization
 54. MIT 6.842 lecture notes randomness in computation
 55. multiple comparisons winner's curse validation set selection lecture notes overfitting the validation set
 56. domain adaptation theory H-divergence bound Ben-David lecture notes course
 57. cs229 notes regularization bayesian statistics MAP prior shrinkage
 58. Murphy pml book chapter 5 decision theory pdf probabilistic machine learning an introduction
 59. MIT 6.437 recitation notes MMSE MAP Bayes least squares cost criterion
 60. Gelman multilevel model partial pooling lecture notes shrinkage factor n_j sigma
 61. test-time adaptation batch norm statistics lecture course notes distribution shift
 62. feature selection lecture notes wrapper filter forward search cs229 notes5
 63. CS229 lecture notes k-means EM unsupervised learning mixture model test data clustering
 64. multi-task learning bound shared representation sample complexity lecture notes
 65. MIT 6.435 Bayesian modeling and inference hierarchical models lecture notes
 66. Stein paradox empirical Bayes lecture notes shrinkage dominates MLE three or more means
 67. Efron large-scale inference empirical Bayes lecture notes Stanford stats 305
 68. Vapnik transductive inference lecture overall risk minimization estimate values at given points
 69. zero-one loss posterior mode MAP optimal estimator lecture notes derivation delta function loss
 70. Berger statistical decision theory lecture notes optimal action minimize posterior expected loss all-or-nothing loss
 71. Stanford lecture notes semi-supervised learning cluster assumption low density separation
 72. learning curves noisy features generalization error increases with dimension lecture notes
 73. signal to noise ratio effective sample size regression lecture notes number of features n>>d
 74. gradient boosting overfitting noise variables lecture notes greedy split spurious
 75. CS228 notes decision making expected utility maximum expected utility MEU influence diagram
 76. MIT 6.390 lecture notes 'model selection' 'validation' cross validation notes
 77. estimating Bayes error rate from repeated measurements duplicate inputs lecture notes
 78. lack of fit pure error regression lecture notes replicates F test
 79. Stanford CS229 lecture notes 'learning theory' bias variance tradeoff hypothesis class size log|H|
 80. irreducible error estimation with replicate observations pure error lack of fit ANOVA lecture notes
 81. ESL 7.10 cross-validation the wrong way and the right way feature selection bias
 82. conditional variance of y given x noise floor estimation nearest neighbor estimator Bayes error lecture
 83. 6.390 notes 'feature representation' 'encoding' one hot standardization
 84. MIT 6.390 notes regression loss function asymmetric cost matters more than model
 85. ESL chapter 7 'the wrong way to do cross-validation' quote
 86. CS234 lecture notes reward hacking optimizing the wrong objective lecture
 87. probability calibration lecture notes reliability sharpness proper scoring rule course
 88. surrogate loss consistency Bartlett Jordan McAuliffe lecture notes classification calibrated
 89. Goodhart's law machine learning lecture notes optimizing a metric
 90. Stanford EE364b subgradient nonconvex step function surrogate loss lecture
 91. cost sensitive classification threshold optimal lecture notes expected cost
 92. structured prediction argmax inference decision lecture notes course
 93. MIT 6.S191 lecture notes uncertainty deep evidential regression
 94. convex surrogate loss for 0-1 loss lecture notes calibration classification consistency
 95. Stanford STATS 315 modern applied statistics learning notes
 96. MIT 9.520 statistical learning theory lecture notes regularization ill-posed inverse problem
 97. Gneiting sharpness subject to calibration probabilistic forecasting lecture notes
 98. ESL 7.9 optimism of the training error rate effective number of parameters expected optimism
 99. Stanford CS229 notes SVM hinge loss why we use surrogate not 0-1 loss NP-hard
100. one standard error rule cross validation model selection lecture notes parsimony
101. MIT 6.390 notes 'reward' 'policy' 'MDP' decision making under uncertainty chapter
102. interval forecasting scoring rule pinball winkler lecture notes course
103. Stanford CS229 lecture notes reinforcement learning policy versus value estimation reward function
104. conformal prediction lecture notes course exchangeability distribution free
105. lecture notes ill-posed inverse problem regularization Tikhonov machine learning 9.520
106. Ng Jordan discriminative vs generative naive bayes logistic regression sample complexity
107. bandit best arm identification lecture notes sample complexity stopping rule
108. multi-armed bandit lecture notes CS234 exploration regret best arm
109. MIT 6.390 chapter 'model selection' lecture notes overfitting to the validation set repeated evaluation
110. regression on a truncated sample inconsistent OLS attenuation lecture notes proof
111. Greene censoring truncation survey truncated regression moments conditional mean inverse Mills ratio
112. MIT 14.387 applied econometrics lecture notes
113. paired t-test model comparison correlated cross validation folds Dietterich lecture notes
114. Stanford STATS 300 lecture notes conditional inference selection
115. 'incidental truncation' 'explicit truncation' econometrics lecture notes difference
116. Stanford CS229 notes 'generative' vs 'discriminative' when to model the joint
117. Bayesian optimization lecture notes acquisition function expected improvement course
118. MIT 6.7920 dynamic programming decision under uncertainty lecture notes
119. sample selection lecture notes 'we only observe y when y > c' truncated normal mean
120. Heckman 1979 sample selection bias as a specification error lecture notes omitted variable
121. selection on the dependent variable bias lecture notes course political science
122. 'simple regret' best arm identification course lecture notes gap-dependent
123. cross validation variance of the estimate paired comparison lecture notes standard error
124. sequential testing early stopping decision rule lecture notes SPRT
125. Nadeau Bengio inference for the generalization error variance cross validation
126. 'representation' matters more than model lecture notes deep learning feature learning Stanford CS230
127. Simpson's paradox averaging over groups lecture notes machine learning evaluation
128. MIT 6.390 notes chapter 'regression' 'ridge regression' 'validation' spring 2025 pdf
129. training with a weighted loss to match evaluation weighting lecture notes course sample weights
130. macro average versus micro average metric unequal group sizes lecture notes evaluation
131. length-biased sampling size-biased distribution lecture notes weighted distribution renewal
132. size biased sampling y-weighted expectation lecture notes E[Yg(Y)]/E[Y]
133. CS231n lecture notes data preprocessing feature representation why representation matters
134. importance weighting for a weighted loss training reweighting lecture notes cost sensitive
135. Stanford CS229 notes 'the k-means' 'EM' 'factor analysis' 'PCA' representation learning lecture
136. Boyd convex optimization estimation maximum likelihood chapter 7 statistical estimation
137. 'evaluation criterion' 'model type' 'model class' MIT 6.390 introml notes taxonomy
138. Stanford CS231n notes 'nearest neighbor' 'validation' 'never touch the test set'
139. CS229 lecture notes 'the reject option' abstain classifier
140. deadzone-linear penalty function approximation Boyd convex optimization
141. MIT 6.390 notes chapter 'what is the right question' problem formulation machine learning
142. 'six components' machine learning problem formulation MIT 6.390 notes problem class assumptions evaluation criterion
143. Boyd EE364a lecture 6 approximation and fitting penalty function approximation deadzone-linear
144. Boyd convex optimization book chapter 6 approximation and fitting penalty function robust approximation
145. learning with abstention reject option lecture notes selective prediction course
```

### 2.3 Primary documents fetched and read in raw

Fetched with `httpx` + `pypdf`/`BeautifulSoup`, cached at `/tmp/s14cache`. Sizes are extracted-text
characters, i.e. how much I could actually read.

- `http://papers.neurips.cc/paper/1699-transductive-inference-for-estimating-values-of-functions.pdf` (16,173 chars extracted)
- `http://www2.stat.duke.edu/~rcs46/modern_bayes17/lecturesModernBayes17/lecture-2/02-intro-to-Bayes.pdf` (9,015 chars extracted)
- `https://ai.stanford.edu/blog/understanding-self-training/` (17,640 chars extracted)
- `https://arxiv.org/abs/1506.02629` (5,705 chars extracted)
- `https://cs229.stanford.edu/extra-notes/loss-functions.pdf` (7,726 chars extracted)
- `https://cs229.stanford.edu/main_notes.pdf` (373,219 chars extracted)
- `https://cs229.stanford.edu/materials/ML-advice.pdf` (12,458 chars extracted)
- `https://cs229.stanford.edu/notes2021fall/cs229-notes12.pdf` (30,290 chars extracted)
- `https://cs229.stanford.edu/notes2021fall/cs229-notes5.pdf` (15,296 chars extracted)
- `https://cs229.stanford.edu/summer2019/BiasVarianceAnalysis.pdf` (14,324 chars extracted)
- `https://cs229.stanford.edu/summer2019/cs229-notes4.pdf` (20,949 chars extracted)
- `https://cs230.stanford.edu/files/cs230exam_fall18.pdf` (28,313 chars extracted)
- `https://esl.hohoweiya.xyz/book/The%20Elements%20of%20Statistical%20Learning.pdf` (1,869,924 chars extracted)
- `https://gradml.mit.edu/main/Lectures/` (773 chars extracted)
- `https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf` (157,488 chars extracted)
- `https://introml.mit.edu/_static/spring24/LectureNotes/6_390_lecture_notes_spring24.pdf` (326,585 chars extracted)
- `https://introml.mit.edu/_static/spring25/notes.pdf` (328,057 chars extracted)
- `https://introml.mit.edu/notes/` (21,839 chars extracted)
- `https://introml.mit.edu/notes/feature_representation.html` (16,442 chars extracted)
- `https://introml.mit.edu/notes/mdp.html` (23,582 chars extracted)
- `https://jmlr.org/papers/volume6/brown05a/brown05a.pdf` (67,962 chars extracted)
- `https://nignatiadis.github.io/assets/lecture_notes/Empirical-Bayes.pdf` (314,618 chars extracted)
- `https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/pages/lecture-notes/` (3,823 chars extracted)
- `https://ocw.mit.edu/courses/18-657-mathematics-of-machine-learning-fall-2015/81406c87dccb9e873cfafa876a4d69c3_MIT18_657F15_LecNote.pdf` (221,360 chars extracted)
- `https://ocw.mit.edu/courses/18-657-mathematics-of-machine-learning-fall-2015/86f311c7073869c5e0c199008787d5c9_MIT18_657F15_L2.pdf` (16,035 chars extracted)
- `https://pages.stern.nyu.edu/~wgreene/Censoring-Truncation-Survey.pdf` (107,201 chars extracted)
- `https://pages.stern.nyu.edu/~wgreene/DiscreteChoice/Readings/Greene-Chapter-19.pdf` (193,441 chars extracted)
- `https://probml.github.io/pml-book/book1.html` (8,966 chars extracted)
- `https://proceedings.mlr.press/v37/blum15.pdf` (37,547 chars extracted)
- `https://raw.githubusercontent.com/kerasking/book-1/master/ML%20Machine%20Learning-A%20Probabilistic%20Perspective.pdf` (2,370,453 chars extracted)
- `https://sites.stat.columbia.edu/gelman/research/published/rsquared.pdf` (53,622 chars extracted)
- `https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf` (113,584 chars extracted)
- `https://spia.uga.edu/faculty_pages/tyler.scott/teaching/PADP8130_Spring2017/readings/gelman.hill.2007.ch12.pdf` (78,496 chars extracted)
- `https://tselilschramm.org/mltheory/ma.pdf` (360,289 chars extracted)
- `https://web.mit.edu/6.437/www/info17.pdf` (12,124 chars extracted)
- `https://web.stanford.edu/class/ee364a/lectures/approx.pdf` (9,271 chars extracted)
- `https://web.stanford.edu/class/stats214/` (1,441 chars extracted)
- `https://www.bauer.uh.edu/rsusmel/phd/imbens%20-%20selection%20models.pdf` (8,644 chars extracted)
- `https://www.columbia.edu/~yt2661/STL/slides/Lecture-2.pdf` (45,164 chars extracted)
- `https://www.cs.cmu.edu/~psarkar/sds383c_16/lecture9_scribe.pdf` (8,621 chars extracted)
- `https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html` (12,260 chars extracted)
- `https://www.cs.cornell.edu/courses/cs4780/2022sp/notes/LectureNotes17.html` (10,080 chars extracted)
- `https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf` (881,734 chars extracted)
- `https://www.cs.purdue.edu/homes/amakur/docs/6.437%20Recitation%20Notes%20Anuran%20Makur.pdf` (80,009 chars extracted)
- `https://www.lancaster.ac.uk/users/esqn/windsor04/handouts/vapnik.pdf` (57,465 chars extracted)
- `https://www.reed.edu/economics/parker/312/online_slides/312_4-27.pdf` (8,301 chars extracted)
- `https://www.stat.berkeley.edu/~jsteinhardt/stat260/notes/lect20.pdf` (7,879 chars extracted)
- `https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/calibration.pdf` (46,417 chars extracted)

Non-course primary sources fetched for a specific formal statement that a course sets or cites
(Greene ch.19 is the standard limited-dependent-variable chapter; Gneiting–Raftery is the scoring-rule
source that Tibshirani's Berkeley statlearn lecture teaches from; Blum–Hardt is the ICML paper whose
mechanism is taught as the answer to leaderboard overfitting; Brown is the JMLR source of the
ambiguity/covariance decomposition; Vapnik's Windsor handout is a lecture handout, not a course).

---

## 3. PRINCIPLES TABLE

The table is long because the brief asked for coverage of nine areas. Read the "IMPLIES HERE" column
first; it is where the work is.

### A. Bayes error / irreducible risk

| # | Principle | Course + URL | Quoted statement | Formal form | What it implies HERE | Facts |
|---|---|---|---|---|---|---|
| A1 | **The noise term is not a target.** | Stanford **CS229**, main lecture notes §8.1 (Ng, upd. Ma/Avati), `https://cs229.stanford.edu/main_notes.pdf` | **[원문]** "MSE(x) = σ2 \|{z} unavoidable + (h⋆(x)−havg(x))2 \| {z } ≜ bias2 + var(hS(x))\| {z } ≜ variance … **There is nothing we can do about the first term σ2 as we can not predict the noise ξ by definition.**" | `E[(y−ĥ(x))²] = σ² + bias² + var`, σ² independent of the learner | Fact 1 says 85.1% of our residual **variance** sits in the "NWP-to-hub-wind" channel. From *the learner's* seat that channel is σ². **But σ² is defined relative to the input σ-algebra.** σ² is not a property of the phenomenon; it is a property of *what x contains*. So the CS229 statement is not "give up" — it is "the only operation that reduces σ² is changing x." | 1, 2 |
| A2 | **Irreducible error only decomposes cleanly under squared loss.** | Stanford **CS229** summer-2019 section notes, Avati, *Bias-Variance Analysis: Theory and Practice*, `https://cs229.stanford.edu/summer2019/BiasVarianceAnalysis.pdf` | **[원문]** "= τ 2 \|{z} Irreducible error + E [ ˆfn(x∗)−f(x∗) ]2 \| { z } Bias2 + V [ ˆfn(x∗) ] … **Such a clean decomposition into Bias and Variance terms exists only for the squared error loss. Proposals have been made for more general losses, though none are widely accepted.**" | decomposition valid for `L=(y−ŷ)²` only | **Our loss is neither squared nor even a metric.** It is `0.5(1−ℓ1/C) + 0.5·(y-weighted deadzone indicator)`. So the "26.4% / 85.1% variance share" decomposition in fact 1 is a decomposition *of squared error*, which is **not the objective**. It bounds nothing about Total. The 0.869922-with-perfect-wind number is the trustworthy statement; the variance shares are decoration. | 1 |
| A3 | **Optimal / unavoidable error rate and *avoidable* bias.** | **CS230**-adjacent set text, Ng, *Machine Learning Yearning* ch.22, `https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf` (CS230 exam defines the same, `https://cs230.stanford.edu/files/cs230exam_fall18.pdf`) | **[원문]** "Optimal error rate ("unavoidable bias"): 14%. Suppose we decide that, even with the best possible speech system in the world, we would still suffer 14% error. … **Avoidable bias: 1%. This is calculated as the difference between the training error and the optimal error rate.**" | `Err = Err* + avoidable_bias + variance` | Fact 1 hands us an *empirical* optimal-error-rate probe of exactly Ng's type: substituting the ground-truth intermediate (measured wind) gives Total 0.869922 against our 0.636184. The **avoidable** gap is 0.2337 Total, and it is *entirely* attributable to one pipeline component. We have been spending our effort on components that Ng's own arithmetic prices at ≤0.01. | 1 |
| A4 | **kNN is not a Bayes-error estimator in 15 dimensions.** | Cornell **CS4780** lecture 2 (Weinberger), `https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html`; ESL §2.5 | **[원문-요약]** The lecture derives that to capture a fraction `p` of the volume around a query in `d` dimensions you need an edge length `p^(1/d)`, so neighbourhoods in high `d` are not local; the 1-NN ≤ 2×Bayes result holds only "as n→∞", which the notes call out as unattainable in practice. | `ℓ ≈ p^{1/d}`; for d=15, p=0.01 ⇒ ℓ=0.74 of the range | **Fact 2 does not establish that we are near the Bayes error.** A mean neighbour radius of **1.22 standard deviations** in a 15-d standardised space is not a local neighbourhood — it is roughly a *global* neighbourhood. The measured conditional MAD 0.110 is therefore an **upper bound on σ(x) polluted by the bias of a non-local average**, exactly the quantity ESL warns about. The project may have concluded "irreducible" from an estimator that cannot support the conclusion. | 2 |
| A5 | **How to estimate irreducible error correctly: replicates → pure error.** | Penn State **STAT 462** notes §3.7–3.8, `https://online.stat.psu.edu/stat462/node/111/`, `https://online.stat.psu.edu/stat462/node/113/` | **[전문미확인]** (snippet) "That is, if each x value in the data set is unique, then the lack of fit test can't be conducted on the data set." The lack-of-fit F is `MS(LOF)/MS(PE)`. | `SSE = SS(PE) + SS(LOF)`; PE needs replicated x | **We have replicates and have never used them.** Three turbine groups share the *same* NWP box: for each hour, three responses at (almost) the same x. That is a designed replicate structure. Decomposing residual variance into *shared-across-groups* (= common NWP error, **reducible** in principle by anything that senses NWP error) and *group-idiosyncratic* (= closer to true σ²) is a two-hour computation that would tell us the actual noise floor instead of the kNN upper bound. | 1, 2 |

### B. Decision theory: the Bayes action for a non-standard loss

| # | Principle | Course + URL | Quoted statement | Formal form | What it implies HERE | Facts |
|---|---|---|---|---|---|---|
| B1 | **The Bayes decision rule minimises *posterior expected loss*, and the answer depends on the loss, not on the model.** | Murphy, *Machine Learning: A Probabilistic Perspective* §5.7 (the set text of CS228 / 6.867 / 6.7900 lineages), `https://raw.githubusercontent.com/kerasking/book-1/master/ML%20Machine%20Learning-A%20Probabilistic%20Perspective.pdf` | **[원문]** "In the Bayesian approach to decision theory, **the optimal action, having observed x, is defined as the action a that minimizes the posterior expected loss**: ρ(a\|x) ≜ E p(y\|x) [L(y,a)] … Hence the Bayes estimator, also called the Bayes decision rule, is given by δ(x)=argmin_a ρ(a\|x)". | `δ(x)=argmin_a ∫L(y,a)p(y\\|x)dy` | This is the frame the project already half-uses ("the deployed prediction is an ACTION"). What the project has **not** done is actually *compute* the argmin, because computing it requires `p(y\|x)`, and we only ever produce a point. The current "policy `T…_G…`" is a two-scalar reparametrisation of a functional argmax. | 5, 6 |
| B2 | **Which functional you get is dictated by the loss: mean↔ℓ2, median↔ℓ1, mode↔0-1.** | Murphy §5.7.1.1–5.7.1.4 (same URL) | **[원문]** "5.7.1.3 Posterior mean minimizes 𝓁2 (quadratic) loss … ŷ = E[y\|x] … **5.7.1.4 Posterior median minimizes 𝓁1 (absolute) loss** … The optimal estimate is the posterior median, i.e., a value a such that P(y<a\|x)=P(y≥a\|x)=0.5." and "5.7.1.1 MAP estimate minimizes 0-1 loss … the action that minimizes the expected loss is the posterior mode". | as quoted | Our loss is a **0.5/0.5 mixture of ℓ1 and a widened 0-1**. Therefore the Bayes action is a **mixture-of-functionals**: neither the median nor the mode, but the maximiser of a *boxcar-smoothed density* traded off against absolute risk. See **[유도] §4.2** for the exact object. The project's mechanism note ("moving toward the conditional mean improves accuracy and damages settlement") is a *qualitative shadow* of this; the quantitative object has never been written down. | 5, 6 |
| B3 | **The dead-zone penalty is a named, standard object, and the shape of the penalty determines the shape of the residual distribution.** | Stanford **EE364a** lecture 6, *Approximation and fitting*, Boyd & Vandenberghe, `https://web.stanford.edu/class/ee364a/lectures/approx.pdf` | **[원문]** "penalty function approximation minimize 𝜙(r1)+···+𝜙(rm) … examples ▶ quadratic: 𝜙(u)=u2 ▶ **deadzone-linear with width a: 𝜙(u)=max{0,\|u\|−a}** … Example: histograms of residuals … **shape of penalty function affects distribution of residuals**" | `φ_dz(u)=max{0,\\|u\\|−a}` | **FICR *is* a dead-zone penalty with a=0.06C (and a soft shoulder to 0.08C).** Boyd's slide 6.5 shows the exact empirical consequence: fitting under a dead-zone penalty produces a residual histogram with a large **spike inside the dead zone and heavy tails**, whereas ℓ1/ℓ2 produce unimodal residuals. That spiky-with-tails residual shape is *precisely* the shape that maximises FICR, and it is *precisely* what a model trained on ℓ1/ℓ2/Huber will never produce. **We have never trained under the dead-zone penalty.** We have only post-processed an ℓ-trained model. | 5, 6 |
| B4 | **A surrogate loss is only safe if it is *calibrated* for the target loss.** | Michigan **EECS598** notes *Calibrated Surrogate Losses* (fetch blocked, **[전문미확인]**) `https://web.eecs.umich.edu/~cscott/past_courses/eecs598w14/notes/14_calibrated.pdf`; primary Bartlett–Jordan–McAuliffe, `https://sites.stat.washington.edu/courses/stat527/s14/readings/Bartlett_etal_JASA_2006.pdf` **[전문미확인]** | **[전문미확인]** (snippet) "Consistency results provide reassurance that optimizing a surrogate does not ultimately hinder the search for a function that achieves the Bayes risk". Stanford **CS229** supplemental notes (Duchi) give the motivation for surrogates in raw: **[원문]** "the loss ϕzo is discontinuous, non-convex … and perhaps even more vexingly, NP-hard to minimize. So we prefer to choose losses that have the shape given in Figure 1." (`https://cs229.stanford.edu/extra-notes/loss-functions.pdf`) | φ is classification-calibrated iff minimising φ-risk drives 0-1 excess risk to 0 | We use ℓ1/ℓ2 as a surrogate for a mixture containing a step. **Nobody has checked calibration.** For a step reward, ℓ2 is *not* calibrated: the ℓ2-optimal action (conditional mean) can be strictly worse in FICR than a deliberately biased action. This is the formal statement of the project's "mechanism to remember", and it also explains **fact 6**: a *more accurate* surrogate teacher gives a *worse* action precisely because the surrogate is uncalibrated — accuracy in an uncalibrated surrogate carries no guarantee whatsoever. | 5, 6 |
| B5 | **Optimisation Verification test: separate "wrong objective" from "bad search".** | Ng, *Machine Learning Yearning* ch.44–45 (CS230 material), same URL as A3 | **[원문]** "Let S_out be the output transcription … Let S* be the correct transcription … compute Score_A(S*) and Score_A(S_out). Then check whether Score_A(S*) > Score_A(S_out). … **Case 2: Score_A(S*) ≤ Score_A(S_out) … you know that the way you're computing Score_A(.) is at fault** … The Optimization Verification test tells you that you have an objective (scoring) function problem." | if `Score(y*) ≤ Score(ŷ)` blame the objective, else blame the search | **This is a two-line diagnostic we have never run and it is decisive.** For each scored row, compute our model's *training objective* value at (i) our deployed action and (ii) the action that would have maximised the realised Total on that row. If the training objective ranks our action **above** the metric-optimal action on most rows, then all 30 treatments were search improvements on a wrong objective — and no amount of estimator swapping can help. Given fact 5 and fact 6, this is the *expected* outcome. | 5, 6 |
| B6 | **Rational agency = pick the action minimising expected loss; the loss is a modelling choice, not a given.** | MIT **6.390** lecture notes §1.3 and §2.1, `https://introml.mit.edu/_static/spring25/notes.pdf`, `https://introml.mit.edu/_static/spring24/LectureNotes/6_390_lecture_notes_spring24.pdf` | **[원문]** "**The choice of loss function is part of modeling your domain.** In the absence of additional information about a regression problem, we typically use squared loss". And **[원문]** "There is a theory of rational agency that argues that you should always select the action that minimizes the expected loss." | — | 6.390 puts loss choice *before* model choice in its six-component decomposition (§I1). We inverted the order: we fixed squared/absolute loss "in the absence of additional information" even though the competition *gives us* the information, in closed form. | 5, 6 |

### C. Truncation / selection on the outcome

| # | Principle | Course + URL | Quoted statement | Formal form | What it implies HERE | Facts |
|---|---|---|---|---|---|---|
| C1 | **Truncated regression: conditioning on `y > a` changes the conditional mean by an inverse-Mills term and *attenuates every marginal effect*.** | Greene, *Econometric Analysis* ch.19 (set text of MIT 14.382 / NYU limited-dependent-variable course), `https://pages.stern.nyu.edu/~wgreene/DiscreteChoice/Readings/Greene-Chapter-19.pdf` | **[원문]** "E [yi \| yi > a] = x′i β + σ φ[(a − x′i β)/σ] / (1 − Φ[(a − x′i β)/σ]). **The conditional mean is therefore a nonlinear function of a, σ, x, and β.**" and "= β(1 − δi) … **Because (1−δi) is between zero and one, we conclude that for every element of xi, the marginal effect is less than** [the coefficient]". | `E[y\\|y>a] = x'β + σλ(α)`, `∂E[y\\|y>a]/∂x = β(1−δ)` | **This is the single most under-exploited fact in the brief.** NMAE is computed **only on rows where `y ≥ 0.1·capacity`** — selection *on the outcome*, i.e. exactly Greene's truncation, not censoring. Consequences, all testable: (i) a model that targets `E[y\\|x]` or `median(y\\|x)` is **systematically biased low on the scored subpopulation**, by `σ·λ(α)`, which is largest exactly where `p(y\|x)` straddles the 0.1C threshold — the low-wind hours; (ii) any coefficient/response fitted on the truncated sample is **attenuated by (1−δ)**; (iii) the correct target for the NMAE half is `median(y \\| x, y ≥ 0.1C)`, which is a *different function* from `median(y\\|x)`. | 5, 6, 8 |
| C2 | **Truncation ≠ censoring, and the two demand different estimators.** | Reed **Econ 312** slides, `https://www.reed.edu/economics/parker/312/online_slides/312_4-27.pdf`; Greene as above | **[원문]** (Reed) "Censored regression: we observe the limit value for extreme observations • **Truncated regression: we observe neither x nor y for observations beyond the limit** • Incidental truncation where the criterion for truncation depends on the …" | — | Our situation is a **third** case the courses do teach but rarely emphasise: we *observe* the sub-threshold rows in training (they are in the data), but the *evaluation* discards them. That is truncation of the **risk functional**, not of the sample. The right response is therefore **not** Heckman/Tobit estimation but a **re-weighted / re-targeted risk**: minimise `E[ℓ(y,a) · 1{y≥0.1C}]`, whose Bayes action is the ℓ-functional of the *truncated* posterior. | 8 |
| C3 | **Sample selection bias is an omitted-variable problem.** | Imbens, ARE213 Lecture Notes 12 (Berkeley/Harvard applied econometrics), `https://www.bauer.uh.edu/rsusmel/phd/imbens%20-%20selection%20models.pdf`; Heckman 1979 **[전문미확인]** | **[원문-요약]** Imbens' notes set up the selection model as an outcome equation plus a selection equation and derive that OLS on selected observations omits `E[ε\\|selected]`, which is a function of x. | `y = x'β + ε`, observed iff `S=1`; `E[y\\|x,S=1] = x'β + E[ε\\|x,S=1]` | The omitted term `E[ε\\|x, y≥0.1C]` is **estimable from our own training data** (we have the sub-threshold rows!) without any econometric machinery: fit `E[y − ŷ \| x, y ≥ 0.1C]` as a function of `ŷ` and predicted spread. That is an *additive correction surface on the scored subpopulation*. It is not "target recalibration" in the sense already tried (which recalibrated on all rows). | 5, 8 |
| C4 | **The FICR half re-weights the population by `y` — a size-biased (length-biased) distribution.** | Standard weighted-distribution result, taught in survey-sampling/renewal sections. Course-grade source not located; **[전문미확인]** for the pedagogy, **[유도]** for the application. | — | `p̃(y\\|x) = y·p(y\\|x)/E[y\\|x]` | `FICR = Σ y_i u_i / Σ 4 y_i` means the FICR half is a **hit probability under the size-biased posterior `p̃`, not under `p`**. The FICR-optimal action is therefore pulled *upward* relative to the NMAE-optimal action by the size-biasing, **on top of** the truncation pull of C1. Both pulls point the same way (up), both are computable, and the project's grid-searched `T`/`G` policy is a crude one-parameter proxy for their sum. | 5 |

### D. Statistical learning theory

| # | Principle | Course + URL | Quoted statement | Formal form | What it implies HERE | Facts |
|---|---|---|---|---|---|---|
| D1 | **Error decomposes into approximation + estimation; only estimation depends on sample size and class complexity.** | Shalev-Shwartz & Ben-David, *Understanding Machine Learning* §5.2 (set text of STATS214/CS229M, 18.657, 6.7900), `https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf` | **[원문]** "L_D(h_S) = ϵapp + ϵest where: ϵapp = min_{h∈H} L_D(h), ϵest = L_D(h_S) − ϵapp. • **The Approximation Error** – the minimum risk achievable by a predictor in the hypothesis class … **does not depend on the sample size** … • **The Estimation Error** … **increases (logarithmically) with \|H\| and decreases with m**." Also **[원문]** "we cannot hope that the learning algorithm will find a hypothesis whose error is smaller than the minimal possible error, that of the Bayes predictor." | `L(h_S) = ϵ_app + ϵ_est` | 21 extra noise columns move `log\|H\|` by `log(2^21)/log(2^872) ≈ 2.4%` of an already-tiny term at m≈19k. **Uniform-convergence theory therefore does *not* explain fact 3.** Do not reach for VC/Rademacher here; it is the wrong instrument, and reaching for it would repeat the "danger of over-theorizing" that CS229's ML-advice deck warns about. | 3 |
| D2 | **Excess risk is weighted by how far the conditional law is from the decision boundary.** | MIT **18.657** *Mathematics of Machine Learning* lecture 2 (Rigollet), `https://ocw.mit.edu/courses/18-657-mathematics-of-machine-learning-fall-2015/86f311c7073869c5e0c199008787d5c9_MIT18_657F15_L2.pdf` | **[원문]** "the quantity R(h)−R(h∗) … is called the **excess risk** … Equation (1.1) makes clear that **the excess risk weighs the discrepancy between h and h∗ according to how far η is from 1/2. When η is close to 1/2, no classifier can perform well and the excess risk is low.**" | `E(h) = E[\\|2η(X)−1\\| 1{h≠h*}]` | Transplanted to our step reward **[유도]**: rows where `p(y\\|x)` is diffuse relative to the ±0.06C window contribute *almost nothing* to achievable FICR no matter what we predict, and rows where `p(y\\|x)` is concentrated contribute nearly all of it. **The achievable FICR is concentrated on a minority of "sharp" hours.** Optimising a global loss over all hours therefore spends capacity on hours that cannot pay. A per-row *sharpness* estimate is the routing variable we have never built. | 1, 2 |
| D3 | **Why noise features hurt tree ensembles: split-candidate dilution.** | ESL §15.3.4 (set text of STATS315A / CS229 supplementary), `https://esl.hohoweiya.xyz/book/The%20Elements%20of%20Statistical%20Learning.pdf` | **[원문]** "**When the number of variables is large, but the fraction of relevant variables small, random forests are likely to perform poorly with small m. At each split the chance can be small that the relevant variables will be selected.** … At the top of each pair we see the hyper-geometric probability that a relevant variable will be selected at any split." | `P(relevant chosen) = 1−C(p−r,m)/C(p,m)` | This is the *textbook* explanation for fact 3 — **and fact 3 explicitly rules it out**: "Raising colsample_bytree does not repair it." Increasing `m` is exactly the ESL remedy. **The remedy failing is itself evidence**, and it points to D4. | 3 |
| D4 | **Optimism of a maximum: every greedy split is a selection over candidates, so adding candidates biases the split criterion upward even when the candidates are pure noise.** | ESL §7.4 *Optimism of the Training Error Rate*, same URL | **[원문]** "the training error err … **will be less than the true error Err_T, because the same data is being used to fit the method and assess its error**. A fitting method typically adapts to the training data, and hence the apparent or training error err will be an overly optimistic estimate … We define the optimism as … op ≡ Err_in − err." | `ω = E[op] = (2/N)Σ Cov(ŷ_i, y_i)` | **[유도]** A GBDT split is `argmax` over `(#candidate columns × #thresholds)`; the optimism of that argmax grows like `σ√(2 log k)` in the number of candidates `k`. Adding 21 pure-noise columns adds 21 chances for a *spuriously* good split at every node, and the tree then *commits* to it, spending depth budget on a split with zero population gain. **Raising `colsample_bytree` makes this worse, not better** (more candidates per node) — which is exactly what fact 3 reports. This explanation is consistent with *all three* numbers in fact 3, including the otherwise-baffling one: the **physically motivated** block costs *more* (−0.000640) than pure noise (−0.000411), because physically motivated columns are **correlated with the true carriers**, so they win the argmax more often while carrying no *independent* information. Pure noise loses the argmax; near-duplicates win it and then underperform. | 3, 5 |
| D5 | **Corollary of D4: the admission criterion for a feature is orthogonality, not plausibility.** | **[유도]** from D4 + ESL 15.3.4 + fact 3 | — | admit block `B` iff `partial R²(B \\| existing 872) > θ` | Every feature-engineering lane so far has proposed blocks that are *physically motivated*, i.e. functions of the same NWP fields already present, i.e. **highly collinear with the existing 872**. D4 predicts those must be net-negative, and fact 3 measured exactly that (−0.000640). "Dropping 252 columns GAINS +0.000245" is the same law with the sign flipped. **The feature axis is not closed; the *plausible-feature* axis is closed.** Only a block with low projection onto the existing span can help, and only two things have that property: (i) NWP *error-state* variables, (ii) *observation-history* variables. | 3 |

### E. Model selection and the winner's curse

| # | Principle | Course + URL | Quoted statement | Formal form | What it implies HERE | Facts |
|---|---|---|---|---|---|---|
| E1 | **Selection over many candidates on one holdout makes the maximum optimistic; the holdout stops being valid.** | Blum & Hardt, *The Ladder* (ICML 2015 — the standard teaching reference for competition leaderboards), `https://proceedings.mlr.press/v37/blum15.pdf`; Dwork et al. NeurIPS 2015 **[전문미확인]** `https://arxiv.org/abs/1506.02629` | **[원문]** "As participants are allowed to **repeatedly evaluate their submissions on the leaderboard, they may begin to overfit to the holdout data that supports the leaderboard.** … Existing approaches therefore often resort to poorly understood heuristics such as limiting the bit precision of answers and the rate of resubmission." | Ladder: publish `R_t` only if `R_t < R_{t−1} − η`, else republish `R_{t−1}` | **We are the leaderboard organiser of our own 3-fold CV, and we have queried it ~30 times.** The Ladder gives the exact fix and it is one line of policy: **accept a new champion only if its score beats the incumbent by more than η**, and otherwise *do not even record* the new number, so that later decisions cannot condition on it. AGENTS.md already forbids further lockbox reads; the Ladder says the CV surface needs the same discipline. | 4, 5 |
| E2 | **Paired/blocked comparison is the right estimator, and the right variance.** | CMU **SDS 383C** lecture 9, `https://www.cs.cmu.edu/~psarkar/sds383c_16/lecture9_scribe.pdf`; ESL §7.10 | **[원문]** "**One Standard Error Rule**: The One Standard Error Rule can be used to compare models with different numbers of parameters in order to select the most parsimonious model with low error. To use, find model with minimum error, then **select the simplest model whose mean falls within 1 standard deviation of the minimum**." | pick simplest `h` with `CV(h) ≤ CV(min) + se` | Fact 4 gives `sd_paired ∈ [0.00055, 0.00093]`. The one-SE rule, applied *paired*, says: **prefer the simpler model unless the paired gain exceeds ~0.0009 Total.** Applied to the project's record, this retro-justifies keeping M266 and rejecting every 21-dof blend — and it is the same conclusion the fold-outside gate reached empirically. Good: two independent routes to the same rule. | 4, 5 |
| E3 | **The optimism of the maximum over k candidates.** | **[유도]** from E1 + E2 + standard extreme-value fact taught in every multiple-testing lecture | — | `E[max_k N(0,σ)] ≈ σ·2.04` at k=30; bound `σ√(2 ln k) = 2.61σ` | With `σ_paired ≈ 0.0007` and `k = 30` treatments (measured this session: `E[max of 30 std normals] = 2.041`), **a "winner" needs a paired gain > ≈0.0014 (point estimate) or >≈0.0018 (bound) to be distinguishable from selection noise.** Fact 5 says all 30 landed *at or below* the champion — so not one has cleared even a **zero** bar. **Read that again: the problem is not that we keep picking noise winners; it is that no treatment has produced an effect large enough to be seen at all.** That is a statement about the *family* of treatments, not about the selection procedure. | 4, 5 |
| E4 | **Stopping rule.** | **[유도]** synthesising E1–E3 with CS229 ML-advice | — | — | The stopping rule implied by the courses is: **(i)** fix `η = 0.0015` Total; **(ii)** any candidate is evaluated *paired* against the incumbent on the same folds/rows; **(iii)** if `Δ ≤ η`, the result is *not recorded as a score* — only as "≤η", so no later decision can condition on the exact value (Ladder); **(iv)** if a whole *family* of ≥5 candidates all return `Δ ≤ η`, declare the **family** closed and require the next candidate to change a *different* one of 6.390's six components (§I1) before it is admitted. Under (iv), the ensembling family, the decision-layer family, the estimator-swap family and the plausible-feature family are all already closed. | 4, 5 |
| E5 | **Cross-validation must wrap the *entire* modelling sequence.** | ESL §7.10.2 "The Wrong and Right Way to Do Cross-validation", same URL | **[원문]** "**In general, with a multistep modeling procedure, cross-validation must be applied to the entire sequence of modeling steps. In particular, samples must be "left out" before any selection or filtering steps are applied.** There is one qualification: initial unsupervised screening steps can be done before samples are left out." | — | Two live consequences. (a) The project's own TRAP note — `prediction_kwh` mixing policies per fold/group — is exactly ESL's "wrong way": the policy was selected *using* the fold it is scored on. (b) **The qualification is a licence**: "initial **unsupervised** screening steps can be done before samples are left out." That is ESL explicitly authorising the use of **test-set inputs** for unsupervised steps (§G). | 4, 5, 8 |

### F. Ensemble theory

| # | Principle | Course + URL | Quoted statement | Formal form | What it implies HERE | Facts |
|---|---|---|---|---|---|---|
| F1 | **The averaging identity: correlation is a hard floor.** | ESL §15.2 eq. (15.1) (STATS315A / CS229 supplementary), same URL | **[원문]** "If the variables are simply i.d. (identically distributed, but not necessarily independent) with positive pairwise correlation ρ, the variance of the average is (Exercise 15.1) **ρσ2 + (1−ρ)/B σ2**. As B increases, the second term disappears, but the first remains, and hence **the size of the correlation of pairs of bagged trees limits the benefits of averaging.**" | `Var(avg) = ρσ² + (1−ρ)σ²/B` | **[유도]** With the project's measured `ρ = 0.93–0.99`, the `B→∞` variance floor is `ρσ²`, so the **maximum** achievable error-sd reduction is `1−√ρ`: **3.56% at ρ=0.93, 1.01% at ρ=0.98, 0.50% at ρ=0.99**. Applied to MAE 0.13858 that is an absolute MAE gain of at most `0.00494 / 0.00139 / 0.00069`, i.e. a **Total** gain of at most **0.00247 / 0.00070 / 0.00035** — and that is the ceiling *if the entire MAE were variance*, which it is not (fact 1: most of it is a fixed input-channel error, i.e. common bias, which averaging cannot touch at all). Against `sd_paired ≈ 0.00055–0.00093` (fact 4), **the ensemble ceiling at the measured ρ is inside the measurement noise.** The ensembling axis is closed *analytically*, not just empirically. | 4, 5 |
| F2 | **Ambiguity / bias-variance-covariance decomposition: ensemble error = average error − average ambiguity.** | Brown et al., JMLR 2005, *Managing Diversity in Regression Ensembles* (the standard course reference for Krogh–Vedelsby), `https://jmlr.org/papers/volume6/brown05a/brown05a.pdf` | **[전문미확인]** (abstract/snippet read; full derivation not extracted) "The bias-variance-covariance decomposition from Ueda and Nakano (1996) breaks the mean squared error (MSE) into three [components]". Krogh–Vedelsby ambiguity: `E_ens = Ē − Ā`. | `E_ens = Ē − Ā`, `Ā = (1/M)Σ(f_i − f̄)²` | Same conclusion by a second route: `Ā` *is* the disagreement, and with `ρ≈0.98` the disagreement is ~2% of the individual error variance. **There is no diversity to harvest.** The project already measured this ("no diversity to exploit beyond the single analog member"); F1/F2 give the *reason* and the *ceiling*, which converts an empirical dead end into a closed axis. | 5 |
| F3 | **Diversity must be *created*, not selected — and it is created by decorrelating the inputs, not the estimators.** | ESL §15.2 (same URL): random forests decorrelate by restricting the *variables* available at each split | **[원문]** "The idea in random forests … is **to improve the variance reduction of bagging by reducing the correlation between the trees**, without increasing the variance too much." | — | Every member we have ensembled sees the **same 872 columns from the same two NWP grids**. `ρ≈0.98` is not a property of the estimators; it is a property of the shared input. **Therefore the only way to move `ρ` is to give a member a genuinely different input.** This is the ensemble axis and the information axis turning out to be the *same axis*. | 5 |

### G. Transductive / semi-supervised / domain adaptation — using the test inputs (fact 8)

| # | Principle | Course + URL | Quoted statement | Formal form | What it implies HERE | Facts |
|---|---|---|---|---|---|---|
| G1 | **Vapnik's imperative: don't solve a harder problem than the one you have; predict at the given points.** | Vapnik, *Problems of Empirical Inference Science*, Windsor 2004 lecture handout, `https://www.lancaster.ac.uk/users/esqn/windsor04/handouts/vapnik.pdf` | **[원문]** "THE VC IMPERATIVE FOR HIGH DIMENSIONAL EMPIRICAL INFERENCE. Solving a problem of interest, do not solve a more general problem as an intermediate step. … • Do not estimate a density if you need to estimate a function. • **Do not estimate a function if you need to estimate it values at given points. (Try to perform transduction not induction.)** • **Do not estimate predictive values if your goal to act well. (Good strategy of action not necessary rely on good prediction.)**" | — | Two of Vapnik's three bullets are, verbatim, the two things this project is doing wrong. We fit a *general function* (induction) when the test inputs are **handed to us** (fact 8), and we optimise *predictive accuracy* when the score pays for *actions* (§B). This single slide is the thesis of this lane. | 5, 8 |
| G2 | **Covariate shift: `p(y\\|x)` fixed, `p(x)` changes; unlabelled test `x` gives importance weights.** | Berkeley **STAT 260** lecture 20 (Steinhardt), `https://www.stat.berkeley.edu/~jsteinhardt/stat260/notes/lect20.pdf` | **[원문]** "**Assumption 1.1 (Covariate Shift).** For a train distribution p̃ and test distribution p∗, we assume that p̃(y\|x) = p∗(y\|x) for all x. Thus the only thing that changes between train and test is the distribution of the covariates x … we furthermore assume that we observe labeled samples (x1,y1),…,(xn,yn) ∼ p̃, **together with unlabeled samples x̄1,…,x̄m ∼ p∗**." | `w(x)=p*(x)/p̃(x)`; minimise `E_p̃[w(x)ℓ]` | **Fully legal, zero leakage**: importance weights are a function of `x` only, and the test `x` are supplied. If the test period's NWP-state distribution differs from training (different season / different regime mix), our training risk is weighting the wrong hours. This is *cheap* to measure (a train-vs-test classifier's AUC on `x` alone) and has never been measured. | 8 |
| G3 | **The unsupervised-screening licence.** | ESL §7.10.2, same URL | **[원문]** "There is one qualification: **initial unsupervised screening steps can be done before samples are left out.**" | — | An explicit textbook authorisation for the class of test-input uses in G2/G4/G5: standardisation, PCA/whitening, clustering, regime discovery, density estimation — all on `train ∪ test` **x** — are *not* leakage and do *not* invalidate CV, provided they never touch `y`. | 8 |
| G4 | **Self-training on pseudo-labels provably improves on the pseudo-labeller.** | Stanford AI Lab / **CS229M** authors (Wei, HaoChen, Ma), `https://ai.stanford.edu/blog/understanding-self-training/` | **[원문]** "The core idea is to use some pre-existing classifier F_pl (referred to as the "pseudo-labeler") to make predictions … on a large unlabeled dataset, and then retrain a new model with the pseudo-labels. … **In practice, F will often be more accurate than the original pseudo-labeler F_pl** … showing that retraining in self-training **provably** improves accuracy compared to the original pseudo-labeler." | expansion assumption ⇒ error contraction | Legal here (the pseudo-labels are *our own* predictions on supplied test `x`; no test `y` is touched). What it buys is **smoothing along the test manifold**, which is a *sharpness* operation — and by D2 sharpness is where the FICR mass is. Note the honest caveat: the theory's expansion assumption is about neighbourhood connectivity, unverified for our data. | 8 |
| G5 | **Domain adaptation bounds: source risk + posterior drift + covariate-shift distance.** | Columbia **STL** (Statistical Transfer Learning) Lecture 2, `https://www.columbia.edu/~yt2661/STL/slides/Lecture-2.pdf` | **[원문]** "R(0)(ĥ) ≤ min_h R(0)(h) [oracle] + 2C√(VC(H)/n1) [cost of learning from samples] + 2E_{X∼P(1)}\|f(1)(X) − f(0)(X)\| [**posterior drift**] + 2d_TV(P(1),P(0)) [**covariate shift**]." | as quoted | Gives the *budget*: if measured `d_TV` between train and test NWP-state distributions is small, covariate-shift correction cannot pay and G2 should be dropped fast. **Measuring `d_TV` is therefore a cheap gate, not a project.** | 8 |
| G6 | **Transduction as risk minimisation over the *given* test points.** | Chapelle, Vapnik & Weston, NeurIPS 1999, `http://papers.neurips.cc/paper/1699-transductive-inference-for-estimating-values-of-functions.pdf` | **[원문-요약]** The paper formalises estimating the values of a function *only at the given points*, and shows the resulting overall-risk objective differs from the inductive one. | — | **[유도]** Our metric is not separable in the way we have been treating it. `FICR = Σ y_i u_i / Σ 4 y_i` and `NMAE = mean over scored rows`. Both **normalisers are sums over the whole test group**, and both are unknown *because they depend on the unknown `y`*. But with the test `x` supplied, `E[Σ y]` and `E[N_scored]` can be estimated **transductively and very precisely** (concentration over ~20k rows), and those two numbers *fix the relative weight* `λ1 : λ2` between the NMAE half and the FICR half. **That relative weight is currently a hand-tuned grid parameter (`T…_G…`).** Replacing a grid-searched scalar with a transductively estimated one is a legal, principled, never-tried move. | 5, 8 |

### H. Hierarchical / multi-task / partial pooling

| # | Principle | Course + URL | Quoted statement | Formal form | What it implies HERE | Facts |
|---|---|---|---|---|---|---|
| H1 | **Partial pooling: the group estimate is a precision-weighted average of the group's own estimate and the pooled one; small groups shrink more.** | Gelman & Hill ch.12 (set text of many applied-Bayes courses incl. MIT 6.435 lineage), `https://spia.uga.edu/faculty_pages/tyler.scott/teaching/PADP8130_Spring2017/readings/gelman.hill.2007.ch12.pdf` | **[원문]** "The multilevel estimate of αj is a **weighted average of the no-pooling estimate for its group** (ȳj − X̄jβ) **and the regression prediction α̂j**: estimate of αj ≈ [ (nj/σ²y) / (nj/σ²y + 1/σ²α) ]·(estimate from group j) + [ (1/σ²α) / (nj/σ²y + 1/σ²α) ]·(estimate from regression). … **there is more pooling when the group-level standard deviation σα is small, and more smoothing for groups with fewer observations.**" | `w_j = (n_j/σ²_y)/(n_j/σ²_y + 1/σ²_α)` | Direct prescription for the group-3 deficit: **do not fit group 3 separately and do not pool it away — fit the shrinkage weight.** The formula also says *how much*: the weight is a known function of `n_j` and two variance components, both estimable from the 3-group residual structure (see A5). This is a **1-dof** device, unlike the 3-dof per-group weights that the fold-outside gate already rejected, so it survives fact 4's precision budget. | 4, 5 |
| H2 | **James–Stein: for `p ≥ 3` simultaneous means, shrinkage *dominates* the MLE.** | Ignatiadis, *Empirical Bayes* lecture notes ch.4, `https://nignatiadis.github.io/assets/lecture_notes/Empirical-Bayes.pdf` | **[원문]** "**The James-Stein phenomenon—that one can uniformly improve upon the maximum likelihood estimator when estimating three or more normal means**—represents one of the [central results]" | `θ̂_JS = (1 − (p−2)σ²/‖y‖²)y`, dominates for `p≥3` | We have **exactly 3 groups** — the boundary case where the theorem first bites. Any per-group parameter (bias offset, policy width, calibration slope) should be estimated by a James–Stein/empirical-Bayes shrunk estimator toward the common value rather than fitted freely. This is precisely the fix for the observed pathology "per-group fold-outside weights oscillate (`g3: 1.00 / 1.00 / 0.15`)" — that oscillation is the MLE's variance, and JS is the textbook cure. | 4, 5 |
| H3 | **Macro-averaging over unequal groups changes the optimal training weights.** | **[유도]** from H1 + the metric definition; course statement of macro-vs-micro found only in non-course sources **[전문미확인]** | — | `L = (1/3)Σ_g L_g` ⇒ row weight `∝ 1/(3 n_g)` | The score macro-averages over 3 groups of very unequal size, so **a row in the smallest group is worth several times a row in the largest**. If training minimises a pooled (micro) loss — which is the default in every GBDT call — we are optimising the wrong functional by a known, fixed, computable set of row weights. This is a **zero-risk, one-line** check. | 5 |

### I. Problem reformulation

| # | Principle | Course + URL | Quoted statement | Formal form | What it implies HERE | Facts |
|---|---|---|---|---|---|---|
| I1 | **A learning problem has six components; "model class" and "algorithm" are only two of them.** | MIT **6.390 / 6.036** lecture notes ch.1, `https://introml.mit.edu/_static/spring25/notes.pdf` and `https://introml.mit.edu/notes/` | **[원문]** (contents + §1.3) "1.1 **Problem class** … 1.2 **Assumptions** … 1.3 **Evaluation criteria** … 1.4 **Model type** … 1.5 **Model class and parameter fitting** … 1.6 **Algorithm**" and "**Once we have specified a problem class, we need to say what makes an output or the answer to a query good** … We specify evaluation criteria at two levels: **how an individual prediction is scored, and how the overall behavior of the prediction or estimation system is scored.**" | — | **This is the cleanest audit instrument in the lane.** Sort the ~30 failed treatments into the six bins. Ensembling, estimator swaps, preprocessing, feature blocks, decision layers, target recalibration all land in **1.5 / 1.6**, with a couple of edge cases in 1.4. **Zero treatments have changed 1.1 (problem class), 1.2 (assumptions) or 1.3 (evaluation criterion used in training).** Fact 5 is then not surprising at all; it is what you get from 30 draws from two of six bins. | 5 |
| I2 | **Spotting a flawed pipeline: if every component is at its ceiling and the system is not, the *inputs* are the problem.** | Ng, *Machine Learning Yearning* ch.57 (CS230 material), same URL as A3 | **[원문]** "**What if each individual component of your ML pipeline is performing at human-level performance … but the overall pipeline falls far short of human-level? This usually means that the pipeline is flawed and needs to be redesigned.** … **The only possible conclusion is that the ML pipeline is flawed.** In this case, the Plan path component is doing as well as it can **given its inputs**, but the inputs do not contain enough information. **You should ask yourself what other information, other than the outputs from the two earlier pipeline components, is needed** … In other words, **what other information does a skilled human driver need?**" | — | **This is fact 1, verbatim, in Ng's language.** Substituting a perfect intermediate (measured wind) lifts Total 0.6362→0.8699. The downstream component is near its ceiling given its inputs; the inputs do not contain enough information. Ng's prescribed next action is not "try another model" — it is **"ask what other information a skilled forecaster needs"**. The answer for a mountain-ridge site with one NWP issuance is: (a) the *error state* of that issuance, (b) the *stability/wave regime*, (c) an *independent* wind estimate. We have never asked the question in this form. | 1, 5 |
| I3 | **Error analysis by parts, and attributing error to one part by substituting ground truth.** | Ng, *MLY* ch.53–55; Stanford **CS229** ML-advice deck (Ng), `https://cs229.stanford.edu/materials/ML-advice.pdf` | **[원문]** (CS229) "**How much error is attributable to each of the components? Plug in ground-truth for each component, and see how accuracy changes.** … Conclusion: Most room for improvement in face detection and eyes segmentation." and "**Ablative analysis** tries to explain the difference between some baseline (much poorer) performance and current performance … Remove components from your system one at a time, to see how it breaks." | — | Fact 1 is an error analysis; **fact 3 (drop 252 columns → +0.000245) is an ablative analysis**. The CS229 deck says these two analyses answer *different* questions and both should be run to completion. We have a partial error analysis (2 components) and a partial ablative analysis (1 block). Running the ablative analysis **to completion over feature blocks** would produce the orthogonality gate of D5 for free. | 1, 3 |
| I4 | **Beware premature statistical optimisation and over-theorising.** | Stanford **CS229** ML-advice deck (Ng), same URL | **[원문]** "**Premature (statistical) optimization.** … Very often, it's not clear what parts of a system are easy or difficult to build … **The only way to find out what needs work is to implement something quickly, and find out what parts break.**" and the slide "**The danger of over-theorizing**" (mail-delivery robot → … → VC dimension). | — | Cuts both ways and should be quoted honestly. It warns *against* a long theory detour — but it also warns against the 30-treatment grind, because that grind was **optimisation of a subsystem whose price we had already measured at ≤0.01** while the 0.23 component sat untouched. Ng's actual instruction is: *let the diagnostic choose the target*. Our own diagnostic (fact 1) chose a target 20 months ago and we went elsewhere. | 1, 5 |
| I5 | **Identification: perfectly collinear factors cannot be separated by any estimator.** | Stanford **EE364a** lecture 6 (least-norm problems), `https://web.stanford.edu/class/ee364a/lectures/approx.pdf`; and every regression course | **[원문]** "least-norm problem: minimize ∥x∥ subject to Ax = b … x★ is smallest point in solution set {x \| Ax = b}" | `lead = hour + 11` ⇒ design matrix rank-deficient in that direction | **Fact 7 is not a nuisance; it is a statement about the data-generating design.** No feature, no estimator, no reparametrisation can separate lead-time skill decay from diurnal effects, because the two are the same column. Any proposal that claims to model "forecast lead-time error growth" is, in this dataset, a proposal to model hour-of-day, and vice versa. **Corollary:** the only way to break the aliasing is a second issuance — the same lever as I2. | 7 |
| I6 | **Sharpness subject to calibration; interval scores reward width *and* coverage.** | Berkeley **statlearn s23** lecture, Tibshirani, *Forecast Scoring and Calibration*, `https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/calibration.pdf` | **[원문]** "interval score is defined by IS_α([ℓα,uα],y) = (uα − ℓα) + (2/α)·dist(y,[ℓα,uα]) … We can see that this **combines a reward for sharpness (first term) and a penalty for miscoverage (second term)**." and Gneiting–Raftery **[원문]** "propose the intuitively appealing **interval score as a utility function in interval estimation that addresses width as well as coverage**." | `IS_α = (u−ℓ) + (2/α)dist(y,[ℓ,u])` | **FICR is an interval score with the width *fixed* at ±0.06C and only the *coverage* term paid.** That is an unusual and exploitable structure: since we cannot buy anything by widening, the *entire* FICR problem is "place a fixed-width window to maximise `y`-weighted coverage". That is a **1-d placement problem per row**, whose solution is the mode of a boxcar-smoothed size-biased truncated posterior (§4.2) — and it is a *density* problem, not a regression problem. | 5, 6 |

---

## 4. WHAT THIS SAYS WE HAVE BEEN GETTING WRONG

Blunt, as requested. Six items, in descending order of how much I think they matter.

### 4.1 We have been doing regression when the score pays for a *density functional*. (B1–B3, B6, I6)

The competition score is
`Total = 0.5·(1 − NMAE) + 0.5·FICR`, and Boyd's EE364a names the second half exactly: a **dead-zone
penalty** `φ(u)=max{0,|u|−a}` with `a = 0.06·C` and a shoulder to `0.08·C`. Boyd's slide 6.5 is a picture
of the consequence — *"shape of penalty function affects distribution of residuals"* — and the dead-zone
histogram in that picture is a **spike inside the dead zone with heavy tails**, structurally different from
the unimodal residuals that ℓ1 and ℓ2 produce.

Every model we have fitted has been fitted under ℓ1, ℓ2, Huber or a quantile loss. **We have never once
fitted under the penalty the score actually uses.** What we have done instead is fit under an ℓ-penalty and
then apply a two-scalar post-hoc "policy `T…_G…`". Murphy §5.7 says the optimal action is
`argmin_a E_{p(y|x)}[L(y,a)]` — a *functional of the whole conditional distribution*. A two-scalar
transform of a point estimate is a rank-1 approximation to a functional argmax. It is not surprising that
30 refinements of that approximation all landed within noise of each other: they are all the same
approximation.

**The class of work that is misconceived:** "post-processing the current representation", "decision
layers", "target recalibration", and the entire "policy grid". Not because post-processing is bad, but
because all of them post-process a *point*, and the object the score prices is a *distribution*.

### 4.2 The Bayes action, written out. [유도]

Let `C` be capacity, `c = 0.1C`, and for a scored row let `p(y|x)` be the conditional density. Then, up to
positive constants,

```
J(a | x)  =  −λ₁ · E[ |a − y| · 1{y ≥ c} | x ]
             + λ₂ · E[ y · ( 4·1{|a−y| ≤ 0.06C} + 3·1{0.06C < |a−y| ≤ 0.08C} ) | x ]
```

with `λ₁ = 0.5/(N_scored·C)` and `λ₂ = 0.5/(4·Σ y)`. Define the **truncated, size-biased** density

```
p̃(y | x)  =  y · p(y | x) · 1{y ≥ c}  /  E[ y·1{y≥c} | x ]
```

and the **trapezoid kernel** `K(u) = 1 for |u|≤0.06C, 0.75 for 0.06C<|u|≤0.08C, 0 otherwise`. Then

```
a*(x)  =  argmax_a  [  w₂ · (p̃ ⋆ K)(a)   −   w₁ · E[ |a − y| · 1{y≥c} | x ]  ]
```

i.e. **the maximiser of a boxcar-smoothed, `y`-tilted, truncation-conditioned predictive density, penalised
by truncated absolute risk.** Three separate distortions push `a*` away from the conditional median, and
*all three point upward*:

1. **truncation** (Greene 19-6, C1): conditioning on `y ≥ 0.1C` adds `+σλ(α)`, largest where `p(y|x)`
   straddles the threshold — i.e. exactly the low-wind hours we currently predict most confidently;
2. **size-biasing** (C4): the `y`-weight in FICR tilts the density upward;
3. **mode-vs-median** (B2, I6): the FICR half wants a smoothed mode; for a right-skewed power distribution
   the smoothed mode and the median differ systematically.

The current pipeline collapses all three into one grid-searched scalar. **That is the single largest
identified modelling error in this document.**

### 4.3 We have been treating the evaluation truncation as a scoring detail. It is a *different estimand*. (C1–C3)

`NMAE` is computed **only on rows with `y ≥ 0.1·capacity`**. That is Greene's truncation — selection on the
*outcome*. It has a textbook consequence stated in raw above: `E[y | y > a] = x'β + σλ(α)`, and
`∂E[y|y>a]/∂x = β(1−δ)`, so **"the marginal effect is less than"** the underlying coefficient. Concretely:

- a model targeting `E[y\|x]` or `median(y\|x)` is **biased low on the scored subpopulation**, and the bias is
  a *known nonlinear function of the predicted level and the predictive spread*;
- the bias is **estimable from our own training data without any econometrics**, because unlike Greene's
  setting we *observe* the discarded rows;
- nobody has run the simplest possible check: **mean signed residual on scored rows, binned by predicted
  level.** If C1 is operating, that curve is monotone and non-zero, and it is a free correction.

We have written "the evaluation set is truncated" in the metric definition for months and never once
treated it as changing the target function.

### 4.4 Fact 3 is not about capacity or dilution. It is the winner's curse *inside every split*. (D3–D5, E1–E3)

ESL's textbook explanation for noise features hurting trees is split-candidate dilution, whose remedy is
raising `m` (`colsample_bytree`). **Fact 3 says that remedy fails.** So the mechanism is the other one: a
greedy split is an `argmax` over candidates, the optimism of a maximum grows like `σ√(2 log k)`, and adding
candidates raises that optimism *even when the candidates are pure noise*. Raising `colsample_bytree`
**increases** `k` per node and therefore makes it worse — exactly as measured.

This explanation is the only one consistent with all three numbers in fact 3, including the one that
otherwise makes no sense: **the physically motivated block (−0.000640) hurts more than pure Gaussian noise
(−0.000411)**. Pure noise usually loses the argmax; near-duplicates of the true carriers usually **win** it
and then contribute nothing new. **Physical plausibility is not merely a weak admission criterion for a
feature — under this mechanism it is an actively *harmful* one.** The correct criterion is orthogonality to
the existing 872 columns.

And note the scale invariance: the same law that makes 21 noise columns cost −0.0004 makes 30 selected
treatments look like progress when they are not. **Fact 3 and fact 5 are the same phenomenon at two
different granularities.**

### 4.5 The ensemble axis is closed analytically, and it was closed before we started. (F1–F3)

ESL (15.1): `Var(avg) = ρσ² + (1−ρ)σ²/B`. At the project's measured `ρ = 0.93–0.99` the maximum achievable
sd reduction is `1−√ρ` = **3.56% / 1.01% / 0.50%**, i.e. a Total gain of at most **0.0025 / 0.0007 /
0.0003** even at `B=∞` and even if the entire MAE were variance — which fact 1 says it is not, since most of
it is a *common* input-channel error that averaging cannot touch. Against `sd_paired = 0.00055–0.00093`
(fact 4), the ceiling at the measured `ρ` is **inside the measurement noise**.

Two lessons. (i) This ceiling was computable from a one-line textbook identity plus one correlation number,
*before* running any of the blend experiments. (ii) `ρ≈0.98` is not a property of the estimators — every
member sees the *same 872 columns from the same two grids*. ESL's own remedy is to decorrelate by changing
what the members can *see*. **So "get more ensemble diversity" and "get more information" are not two axes.
They are one axis, and it is the axis in §4.6.**

### 4.6 We have been optimising the cheap 14% and ignoring the expensive 86%. (A1, A3, I2, I4)

Ng's *Machine Learning Yearning* ch.57 describes our situation with no adaptation required:

> "each individual component … is performing at … near-human-level performance, but the overall pipeline
> falls far short … **The only possible conclusion is that the ML pipeline is flawed.** … the Plan path
> component is doing as well as it can **given its inputs**, but the inputs do not contain enough
> information. **You should ask yourself what other information … is needed.**"

Fact 1 is that diagnostic with the numbers filled in: perfect wind → 0.8699, actual → 0.6362. The 0.234 gap
lives in one place. **Ng's next instruction is a question, not a model:** what does a skilled forecaster
know that our 872 columns do not?

There is one more thing to say here, and it is uncomfortable. CS229's own ML-advice deck warns against
"**premature (statistical) optimization**" and "**the danger of over-theorizing**". Both warnings apply to
*us*, and they point the same way: 30 treatments in bins 1.5/1.6 of 6.390's six-component decomposition,
zero treatments in bins 1.1/1.2/1.3, while a diagnostic we ran ourselves priced the whole 1.5/1.6 region at
≤0.01. That is not bad luck across 30 draws. **It is 30 draws from the wrong two bins.**

### 4.7 One thing we may be getting wrong in the *pessimistic* direction (A4, A5)

The claim "we are near the irreducible error" rests on fact 2 — a kNN probe in 15 standardised dimensions
with **mean neighbour radius 1.22 sd**. Cornell CS4780's lecture 2 and ESL §2.5 are explicit that
neighbourhoods at that radius in that dimension are **not local**, so the 0.110 conditional MAD is an
*upper bound contaminated by non-locality bias*, not an estimate of `σ(x)`. Meanwhile the textbook way to
estimate pure error — **replicates** (STAT 462 §3.7) — is sitting unused in plain sight: three turbine
groups share the same NWP box, so every hour is a triplicate. Decomposing residual covariance into
*shared-across-groups* (common NWP error, reducible in principle) versus *group-idiosyncratic* (closer to
true `σ²`) would replace a contaminated bound with a real number. **We may have talked ourselves into a
noise floor we have never actually measured.**

---

## 5. NEXT-NODE HYPOTHESES (16, diverse; mechanism + originating principle)

Selection is the parent's job. These are deliberately heterogeneous in cost, risk and axis. **No SOTA
research has been done on any of them, per instruction.** Each line is: *mechanism* — `principle`
— *cheapness/risk note*.

**Decision-theoretic / loss-shape family**

1. **Conditional-density head + explicit Bayes action.** Replace the point regressor's output with a
   discretised conditional density over `y∈[0,1]` (e.g. 50-bin multinomial or quantile grid), then deploy
   `a*(x) = argmax_a [w₂(p̃⋆K)(a) − w₁E|a−y|1{y≥c}]` computed exactly per row. — `B1, B2, I6, §4.2` —
   *medium cost; the highest-value single item in this document; replaces a grid-searched scalar with the
   actual Bayes rule.*

2. **Train under the dead-zone penalty itself.** Fit a booster with a custom objective
   `0.5·|r|/C − 0.5·w_y·K(r)` (smoothed for gradients, e.g. logistic-bump approximation to `K`), rather
   than fitting ℓ1/ℓ2 and post-processing. — `B3 (Boyd EE364a), B4` — *low cost, immediately testable;
   Boyd's slide predicts the residual histogram will change shape, which is directly observable even if
   Total does not move.*

3. **Optimisation Verification test as a go/no-go gate.** For each scored row compute the *training*
   objective at (i) our deployed action and (ii) the realised Total-maximising action; count how often the
   training objective prefers ours. If >50%, formally declare the objective (not the search) the fault and
   close bins 1.5/1.6. — `B5 (Ng MLY ch.44–45)` — *near-zero cost; produces a decisive, citable verdict on
   whether the last 30 treatments were ever capable of working.*

4. **Sharpness routing.** Estimate per-row predictive spread; partition rows into "sharp" (FICR-winnable)
   and "diffuse" (FICR-hopeless); apply the Bayes action only where it can pay and the pure ℓ1 action
   elsewhere. — `D2 (18.657 excess-risk weighting), I6` — *cheap; also a diagnostic — if achievable FICR is
   concentrated on <30% of rows, that reframes the whole optimisation.*

**Truncation / selection family**

5. **Truncation-corrected target.** Retarget the model at `median(y \| x, y ≥ 0.1C)` rather than
   `median(y\|x)`, either by explicit two-stage estimation or by simply reweighting the training risk with
   `1{y ≥ 0.1C}`. — `C1, C2 (Greene 19-6)` — *cheap; and note the training-risk reweighting version is a
   two-line change.*

6. **Scored-subpopulation residual surface.** Fit `E[y − ŷ | ŷ, spread, group]` restricted to scored rows
   and add it as an additive correction; this is the empirically estimated inverse-Mills term. — `C3
   (Imbens ARE213), C1` — *very cheap; first check is just the binned signed-residual curve, which if flat
   kills the whole hypothesis in ten minutes.*

7. **Metric-matched row weights.** Train with row weights `w_i ∝ λ₁·1{y_i≥c}/(3n_g) + λ₂·y_i/(3n_g)`,
   i.e. exactly the weights the score implies, including the macro-average over 3 unequal groups. — `H3,
   C4, B6` — *trivially cheap and, per H3, we are currently provably optimising a different functional than
   the one we are scored on.*

**Information / pipeline-redesign family (the §4.6 axis)**

8. **NWP error-state features.** Add features describing the *error state* of the current issuance —
   recent verification residuals of LDAPS/GFS against observed generation at this site over the preceding
   D-2..D-8 window, lagged so they are available at the D-1 14:00 basis time. These are the only features I
   can name that are **orthogonal to the 872** (they are functions of past `y`, not of the forecast fields).
   — `I2 (Ng MLY ch.57), D5, F3` — *medium cost; the single strongest candidate on the information axis;
   must be audited hard against the basis-time rule, but past-period observations are not test-period
   observations.*

9. **Spatio-temporal displacement of the NWP field.** Instead of more scalar summaries of the 4×4/3×3 box,
   estimate a *field displacement* (which grid cell's forecast best matches the realised ridge wind, per
   regime) and read the forecast from the displaced location. This changes what the member *sees*, so it is
   the only kind of change that can move `ρ`. — `F3 (ESL 15.2), I2` — *medium-high cost; also generates
   genuine ensemble diversity as a by-product.*

10. **Second-issuance / independent-source lever, stated as an identification problem.** Because
    `lead = hour + 11` is exactly collinear, *no* amount of modelling can separate lead-time skill decay
    from diurnal structure; only a second issuance can. Frame the question as "what is the cheapest legal
    second look at the atmosphere?" rather than "what feature encodes lead time?" — `I5, I2` — *research
    question, not an experiment; but it correctly retires an entire family of proposed features.*

**Transductive / test-input family (fact 8, currently at zero utilisation)**

11. **Transductive estimation of the metric normalisers.** Use the supplied test `x` to estimate `E[Σ y]`
    and `E[N_scored]` over the test set, and use them to *derive* the `λ₁:λ₂` weight between the NMAE and
    FICR halves instead of grid-searching `T`/`G`. — `G6, G1 (Vapnik), G3 (ESL unsupervised-screening
    licence)` — *cheap, legal, never tried; converts a tuned hyperparameter into a computed constant.*

12. **Covariate-shift audit, then importance weighting.** Train a train-vs-test classifier on `x` only;
    read its AUC as a direct estimate of `d_TV`. If large, reweight the training risk by
    `w(x)=p_test(x)/p_train(x)`; if small, close the axis immediately on the G5 bound. — `G2 (Berkeley
    STAT260), G5 (Columbia STL)` — *cheap and self-terminating: the audit alone is worth running because it
    either opens a lane or closes one.*

13. **Transductive regime discovery.** Cluster `train ∪ test` inputs (unsupervised — explicitly permitted by
    ESL 7.10.2) into atmospheric regimes; fit per-regime decision policies with shrinkage across regimes.
    — `G3, G1, H2` — *medium cost; note the shrinkage requirement, or this becomes another rejected
    multi-dof blend.*

14. **Self-training on test inputs.** Pseudo-label the test rows with the champion, retrain with the
    pseudo-labels plus regularisation, and check whether the retrained model is *sharper* (not more
    accurate) on the test manifold. — `G4 (SAIL/CS229M), D2` — *medium cost, honest caveat: the expansion
    assumption underlying the theory is unverified here; treat a null as informative.*

**Diagnostic / measurement family (cheap, and they gate everything else)**

15. **Cross-group replicate decomposition of the noise floor.** Use the 3 groups as designed replicates at
    (nearly) the same `x` to split residual variance into shared-NWP-error and group-idiosyncratic
    components, replacing the kNN upper bound with a pure-error estimate. — `A5 (STAT462 §3.7–3.8), A4` —
    *cheap; if the shared component is large, §4.6 is confirmed quantitatively; if the idiosyncratic
    component dominates, the project really is near the floor and should stop.*

16. **Ladder-disciplined stopping rule + six-component audit.** Adopt `η = 0.0015` Total (from
    `sd_paired·E[max of 30 std normals] = 0.0007×2.04`), record sub-`η` results only as "≤η", and require
    every new candidate to declare which of 6.390's six components it changes; auto-reject a candidate
    whose component-bin already contains ≥5 sub-`η` results. — `E1 (Blum–Hardt), E2 (one-SE rule), E3, E4,
    I1` — *free; this is a process change, and per §4.4 it is the *same* correction as the feature-admission
    gate, one level up.*

**Hierarchical family**

*(folded into #7 and #13, but stated separately for selection convenience)*

17. **James–Stein shrinkage of all per-group parameters.** Any quantity estimated per group (bias offset,
    policy width, calibration slope) should be a shrunk estimator toward the common value, `p=3` being
    exactly where the theorem starts to bite. This is the textbook cure for the observed
    `g3: 1.00/1.00/0.15` oscillation. — `H1 (Gelman–Hill 12.16), H2 (James–Stein)` — *cheap; and unlike the
    already-rejected 3-dof per-group weights, this is 1 dof, so it fits inside fact 4's precision budget.*

---

## 6. What I would tell the root in one paragraph

The courses are unanimous on three points that our record violates. **(1)** The Bayes action for our loss
is a functional of a conditional *density* — specifically the maximiser of a boxcar-smoothed, `y`-tilted,
truncation-conditioned posterior — and we have been approximating it with a two-scalar transform of a point
estimate, which is why 30 refinements of that transform all landed within noise. **(2)** Ng's own pipeline
diagnostic, applied to our own fact 1, says the downstream component is at its ceiling given its inputs and
the *inputs* are the flaw; ESL's `ρσ² + (1−ρ)σ²/B` then says the ensembling ceiling at our measured `ρ` is
smaller than our measurement noise, and that `ρ` is high *because every member sees the same 872 columns* —
so "diversity" and "information" are one axis, not two. **(3)** The winner's-curse arithmetic
(`σ_paired·E[max of 30] ≈ 0.0014`) says nothing we have tried has produced an effect large enough to be
seen — which is a verdict on the *family* of treatments, not on our selection procedure; and the same
arithmetic, one level down, explains why noise columns and near-duplicate physical columns both hurt, with
the near-duplicates hurting *more*. The cheapest decisive next moves are the ones that cost almost nothing
and settle a whole axis: the Optimisation Verification test (#3), the binned signed-residual curve on the
truncated scored set (#6), the metric-matched row weights (#7), the covariate-shift audit (#12), and the
cross-group replicate decomposition of the noise floor (#15).

---

*End of S14. All repository writes confined to `research/lanes/`. No fits, no lockbox, no git, no upload.*
