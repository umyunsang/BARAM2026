# SK@v5 Workflow — Adversarial Review

2026-08-09, root session, after 11 nodes (N29–N50) and comparison index 5.
Champion 0.63148273, target 0.660000, gap 0.0285173.

This review is written as if by a hostile reviewer who wants the workflow to fail.
Every claim below is a genuine risk, not a strawman.

---

## 1. The approval cascade is ceremonial latency, not safety

SK@v5 → DS@v5 → IP@v3 required three separate user approval phrases before any
code could be written. For a single-developer hackathon with no downstream
consumers, this turns every decision into a blocking synchronous round-trip.

The EventStore predeclare/close discipline is the same pattern at the node level:
every diagnostic, no matter how trivial, requires a predeclaration event and a
closure event appended to SQLite before the next node can start. In the last
session this consumed ~30% of root turns on governance rather than on experiment
design.

**A reviewer would ask:** what specific safety failure did the three-gate approval
prevent that a `git tag` and a frozen SHA-256 receipt would not have prevented?

---

## 2. The session was scientifically productive and score-unproductive

Eleven nodes were predeclared and closed (N29–N50). Ten were diagnostics that
consumed zero comparison index. One (N37) was a score-bearing comparison at
index 5 that failed its gate (+0.000914 against a predeclared minimum effect of
0.001). The remaining ten closed axes by proving things *don't* work.

This is good science: ruling out dead ends is cheaper than pursuing them.
But from a score perspective the session produced **zero net gain** in 11 nodes
and roughly 9 hours of compute.

**A reviewer would ask:** was there a cheaper way to reach the same conclusions?
N38 (action-rule landscape) took ~2 hours of XGBoost fits to prove what N40
(ensemble ceiling) later proved in 10 seconds: the action rule is not the
bottleneck. The ordering was backwards.

---

## 3. Perturbing M115 was provably a dead end from N40 onward

N40 measured the ensemble dilution factor precisely: every raw-M115-arm gain is
multiplied by 0.70/3 = 0.2333 before reaching the champion. The oracle
convex-reweighting ceiling on the four fixed member actions is +0.001556. This
is 5.5% of the required gain.

Yet N41 (conditional gate), N42 (metric-aligned gate), N43 (quantile density),
N44 (band-mass objective), and N45 (bin resolution) all continued to perturb
the M115 arm inside this capped space. All five failed. The cap was known from
N40; the experiments confirmed it five more times.

**A reviewer would ask:** why was N40 not run *before* N41–N45? The ceiling
would have killed the entire axis in one cheap diagnostic.

---

## 4. The champion is a black box we never opened

We proved CHAMPION = 0.30·D + 0.2333·(M102 + M113 + M115) to 1.1e-11 kWh
residual, but we never answered:

- What is D's architecture? Why does D have the best 1-NMAE (0.8616) of any
  member, and could it be improved?
- Why is the weight 0.30 on D? Was it grid-searched, fitted, or chosen ad hoc?
- M102, M113, and M115 are all GBDT variants on the same 100 selected features.
  Are they genuinely diverse, or are they three draws from the same family?
- Could D be replaced by a stronger accuracy-focused member?

All 11 nodes perturbed the model members (M115 primarily). None touched D.
The member with the best accuracy—the exact thing the elasticity measurement
says we need—was never studied.

**A reviewer would ask:** if D is an analog/retrieval model with no trainable
parameters, and it has the best 1-NMAE, what does that say about the GBDT
members? That they are adding complexity without adding accuracy.

---

## 5. Feature selection is frozen from a prior run

The per-fold 100 selected features were inherited from a previous probe. The
selection criterion, the selector algorithm, and the fold on which selection
was performed are not documented in any node receipt in the current session.

We proved that the frozen prepare lineage has 820 deployed numeric names with
zero temporal operators and zero spatial-derivative operators, but we never
re-ran feature selection on the union of the 820 names plus the 54 new geom__
columns.

**A reviewer would ask:** if the prepare representation is missing two whole
information axes (temporal and spatial), why would feature selection on the
remaining 820 names produce an optimal set? The selector never saw what isn't
there.

---

## 6. No pretrained weights deployed despite authorization

The dependency freeze was lifted mid-session. Five Apache-2.0 licensed pretrained
models were verified eligible:
- `amazon/chronos-bolt-base` (821 MB)
- `google/timesfm-2.0-500m-pytorch` (2.0 GB)
- `ibm-granite/granite-timeseries-ttm-r2` (3.24 MB — trivially small)
- `Datadog/Toto-Open-Base-1.0` (605 MB)
- `amazon/chronos-2` (license UNRESOLVED, blocked)

Granite TTM at 3.24 MB is the cheapest possible probe of the pretrained-weight
axis and was never downloaded or evaluated. The session spent ~9 hours on
GBDT hyperparameter sweeps that capped at +0.0013 while a potentially
transformative model family sat untested.

**A reviewer would ask:** did you spend 9 hours proving that a 2016-vintage
GBDT with 100 features cannot reach 0.66, when a 3 MB pretrained model might
have answered the same question in 30 minutes?

---

## 7. The matched-pair floor bracket is too wide to be informative

The irreducible-noise measurement (N50, edu-density lane) gives:
- Lower bound: D/2 = 0.0703 cf
- Upper bound: D = 0.1405 cf
- Target: 0.1203 cf
- Current MAE: 0.1410 cf

The bracket straddles the target. The null hypothesis "MAE cannot fall below
0.1203" is neither rejected nor accepted. The measurement says "maybe"—
the same answer we had before running it.

The bracket could be tightened by conditioning on richer features, but
conditioning on richer features requires... building better features, which
is the thing we need to decide whether to do.

**A reviewer would ask:** is this a measurement or a tautology? You measured
that a model with 820 collapsed features has an MAE floor that might or might
not be above target. The measurement is correct but the conditioning set is the
same one you already know is deficient.

---

## 8. The discovery graph is a checklist with one re-entry

SK@v5 promised recursive node-depth discovery: result → residual deficit →
deepen same stage, advance, branch, or re-enter upstream. In practice:

- One genuine upstream re-entry: N38 (action landscape) showed the density was
  the issue → N43 re-entered S7 for a density-family swap.
- One intra-stage descent: N44 → N45 refined bin resolution.
- Everything else was linear: measure axis A, close it, move to axis B.

The "graph" has depth 1 in most places. The causal edges (REENTERS_UPSTREAM,
DEEPEN) exist in the EventStore but the actual scientific path was mostly a
flat list.

**A reviewer would ask:** does the graph structure add value beyond a
spreadsheet of hypotheses sorted by cost? If the answer is "auditability,"
the reviewer would point out that auditability requires someone to actually
audit, and no external auditor exists.

---

## 9. No fresh holdout means every conclusion is retrospective

The session evaluated everything on the same 2023 surface that guided prior
development. The claim "two deterministic reproductions at Total >= 0.66
prove success" is an operational rule, not a statistical claim. If the 2023
surface has been overfit through repeated adaptive development, two
reproductions prove only that the pipeline is deterministic, not that it
generalizes.

The 2024 lockbox is consumed. The leaderboard is not a validation surface
(the local-to-online offset does not transfer across method classes, per
AGENTS.md). So the only out-of-sample test is the final submission upload,
which is a single number with no components and no error bars.

**A reviewer would ask:** what prevents the workflow from selecting a pipeline
that happens to work on 2023 and fails on the leaderboard? The answer is
"nothing we can measure locally."

---

## 10. The single biggest missed opportunity

The champion's MAE (0.1410 cf) equals the matched-pair D (0.1405 cf) to within
0.3%. This means the model is performing at the level of "draw another sample
from the conditional distribution induced by a coarse champion-action binning."

The model has learned essentially nothing beyond the information in the
conditioning set. The gap is not an optimization problem; it is a
**representation problem**.

We spent 11 nodes optimizing the model on the existing representation when we
should have spent them rebuilding the representation. The prepare stage
discards grid structure and temporal structure; no model can recover
information that isn't in its input.

**A reviewer would summarize the session as:** you measured that your model is
at the irreducible-noise floor of a deficient conditioning set, then spent
nine nodes trying to improve the model, all of which failed because the
bottleneck is the conditioning set.

---

## 11. What a reviewer would attack

1. **"You proved the ensemble architecture cannot reach 0.66, then kept
   experimenting inside it."** N40 closed the ensemble axis quantitatively
   (+0.0016 oracle ceiling). N41–N45 should not have been run.

2. **"You have five authorized pretrained models and used zero."** The
   dependency freeze cost was paid (three approval rounds), the benefit was
   never collected.

3. **"D has the best accuracy and you never studied it."** The member with
   the property you need was ignored in favor of the member you happened to
   have a script for.

4. **"The prepare-stage information loss was measured (FD10, FD11) and the
   prepare stage was never rebuilt."** The diagnosis was correct; the treatment
   was never applied.

5. **"The EventStore contains 178 events, of which approximately 160 are
   governance and 18 are results."** The governance-to-science ratio is ~9:1.

6. **"The workflow produces receipts but not models."** A reviewer reading
   the reports/ directory would find 50+ JSON receipts documenting failures
   and zero promoted candidate pipelines.

---

## 12. Concrete recommendations

### Stop doing

- Single-member perturbations. If the member changes, all four members must
  change together, or the ensemble dilution caps the gain at ≤ +0.002.
- Density-family swaps on the existing conditioning set. The density is
  wide because the conditioning set is weak; no functional can fix that.
- Diagnostics whose answer is already known from previous diagnostics.
  The session's last five nodes all confirmed what N40 and N47 already
  implied.

### Start doing

1. **Rebuild the prepare stage.** The grid pivot exists (`train_grid_pivot.parquet`,
   914 columns with per-grid values). The temporal structure exists (exactly
   one issuance per operating day, N29-verified). The prepare stage should
   produce features that preserve this structure, not collapse it. FD10 and
   FD11 are the highest-leverage open deficits, and they cost zero new data.

2. **Test the cheapest pretrained model first.** Granite TTM at 3.24 MB.
   If it beats the champion's 1-NMAE on the Q2 burn-in fold, the entire
   modeling strategy changes. If it doesn't, the pretrained axis closes at
   the cost of one inference pass.

3. **Study D.** If D is an analog model, its accuracy comes from direct
   retrieval of similar historical days. Understanding why retrieval beats
   GBDT on this problem may reveal what features the GBDT is missing.

4. **Tighten the matched-pair bracket.** Condition on the full feature set
   rather than the champion-action bin. If the bracket still straddles the
   target, the uncertainty is genuine; if it narrows, the ceiling becomes
   informative.

5. **Collapse the approval gates.** For a single-developer workflow, replace
   SK→DS→IP with a single `DESIGN_FROZEN` receipt that freezes the plan,
   authority hashes, and the next action. The three-gate structure adds
   latency without adding safety.

### Keep doing

- The predeclare/close discipline for score-bearing comparisons. That one
  gate prevented N37 from being misreported as a success.
- The elasticity measurement (N47). Converting "need 0.0285 Total" into
  "need 15% MAE reduction" is the single most valuable result of the session.
- The FICR band-hit screening rule. Separating NMAE and FICR mechanisms
  prevented several false positives (every density-family swap raised
  1-NMAE and lost FICR—the screen would have caught this at T1).

---

## Bottom line for the next session

The workflow is architecturally sound but operationally mis-weighted.
It spends ~70% of its turns on governance and ~20% on perturbing a known-capped
member, and ~10% on the representation problem that actually gates the target.
Flip that ratio: rebuild the representation, test pretrained weights, and only
then optimize models on the new conditioning set. And collapse the three-gate
approval into one.
