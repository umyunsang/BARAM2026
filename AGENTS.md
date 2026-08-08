# BARAM2026 Project Rules

- Root session owns all repository writes, model fits, lockbox actions, and submissions. Worktrees and background agent sessions remain forbidden.
  - Updated 2026-08-06 on the user's explicit instruction ("AGENTS.md 파일을 수정하거나 초기화해 나로 진행해"). The prior blanket ban on subagents is replaced by a bounded allowance: **read-only research lanes are permitted** when they (a) perform no repository write outside `research/`, (b) run no model fit and touch no lockbox, (c) carry an explicit source/time bound and evidence tags, and (d) are stopped by the root once their deliverable is integrated. Any lane needing a repository write, a fit, or an external action must hand the work back to the root.
- Treat the frozen competition archive and baseline notebook as immutable inputs and verify their SHA-256 before every full run. **Canonical in-repo copies (2026-08-06 consolidation):**
  - `inputs/competition/open_wind_236727.zip` — sha256 `920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b`
  - `inputs/notebooks/baseline.ipynb` — sha256 `712b26f4d2748860c94cff1e0100c23810468c983173f8e9ef8d009fe82df48c`
  - `inputs/notebooks/metric_official.ipynb` — the official scoring formula notebook
  - `inputs/pages/competition_page_2026-08-01.html` — saved official page snapshot (leaderboard tab; **stale as of 2026-08-04**, see anchors below)
  - **TRAP (fixed 2026-08-06):** `/Users/um-yunsang/Downloads/open.zip` is a DIFFERENT competition (baseball/trackman) that overwrote the name on 2026-08-05. The wind archive survives as `open (1).zip`. `configs/default.yaml` already points at the correct path; this file previously did not. Prefer the in-repo `inputs/` copies.
- Follow the official competition rules verbatim, not a stricter local paraphrase. The authoritative snapshot is `inputs/rules/official_rules_2026-08-09.md` sha256 `6dcececbfb33df6761c87220ea8ce15d875d6b7afc9d90c5afe5dd827ad54ee5` (user-supplied 2026-08-09). Binding consequences:
  - **Language is Python.**
  - **Leakage gate is per prediction row, by availability time.** Every target day uses basis time D-1 14:00 KST for all 24 hours. Only information *created, published, or finalised and actually usable before that basis time* may be used. Eligibility is judged by when the datum became available, never by the time it describes. Derived variables, statistics, interpolations and aggregates must also contain no post-basis-time information.
  - **There is no blanket year cutoff for external data.** The rules state explicitly that availability is *not* judged by "was it public before year X" but per row against that row's basis time. The former local clause "external data published before 2026-07-05" was **stricter than the rules and is withdrawn** (corrected 2026-08-09).
  - **External data** must be legally unencumbered, licence/privacy/ToS compliant, publicly accessible to anyone, and leakage-compliant. Private, institution-restricted or internal operational data is forbidden, as is any data containing or implying the evaluation answers. Second-stage review requires submitting every external file plus source, collection method, collection time, usage period, variables, licence and preprocessing code, and the pipeline must be reproducible from the submission alone.
  - **Pretrained weights** are permitted only if officially released on or before **2026-07-05** under a licence permitting use, modification, distribution, **redistribution and commercial use**. Non-commercial, research-only or evaluation-only weights are forbidden. Record model, weights, licence and download path.
  - **Still forbidden:** reanalysis products, post-hoc corrected observations, test-period observations, evaluation-period actuals, and remote API model inference (OpenAI/Gemini/HF Inference/Together/OpenRouter and equivalents). Weights must be loaded and executed locally.
  - Updated 2026-08-05 on the user's explicit instruction. The prior clause ("use only competition-supplied data") was stricter than `~/Desktop/풍력발전량 예측 규칙.html`; M271 cycles 46-48 showed that extra strictness had been closing a live axis on a false premise.
- Use the project-local Python 3.12 environment and at most six model workers. Do not modify system packages.
  - **Updated 2026-08-09 on the user's explicit instruction** ("의존성 변경해도돼 ... 공식해커톤 규칙만 준수한다면 직접 설치해도 돼, 승인할게"). Installing additional Python packages and frameworks into the project `.venv` is now **permitted and pre-approved**, provided the official rules above are satisfied. The prior blanket dependency freeze is withdrawn. Constraints that remain: install into the project venv (not system packages), pin the exact version in the receipt, keep every install reproducible from the submitted code, and never use a package to perform remote model inference. `BLOCKED_DEPENDENCY` is no longer a valid rejection reason on its own.
- Keep operating year 2024 as a one-use lockbox. Freeze development decisions before reading its scores. **It has already been consumed twice** — 2026-08-01 (freeze `f25afd46`, run `lockbox-2024`) and 2026-08-04 (`C1N32_LOCKBOX_OPERATOR`, user-authorised; record `artifacts/locks/lockbox-2024.second-use.json`). No independent validation surface remains; treat any further 2024 read as a knowingly biased estimate.
- Do not upload to Dacon, mutate a browser/account/team, push a remote repository, or deploy anything without separate explicit authority.
- Stage paths explicitly; never use `git add .`.
- Label PASS narrowly: a unit test, contract suite, lockbox decision, or reproduction receipt proves only its stated scope.
- Generated competition data, models, predictions, and CSV files stay untracked; reviewer-facing manifests and reports may be committed.

## Established measurements (do not re-derive; 2026-08-06 consolidation)

- **Score algebra:** `Total = 0.5*(1-NMAE) + 0.5*FICR`, so reaching Total `0.66` requires `(1-NMAE) + FICR = 1.32`. The required FICR is therefore `1.32 - (our 1-NMAE)`, **not** rank-20's FICR.
- **Online anchors (user-performed uploads; no agent ever uploads):**

  | Lineage | Local Total | Online Total | Online 1-NMAE | Online FICR | Offset (Total) |
  |---|---:|---:|---:|---:|---:|
  | M252 (analog/retrieval) | 0.605760 | 0.6268784 | 0.8659033 | 0.3878535 | +0.021119 |
  | M261 (classifier) | 0.629973 | 0.6365274 | 0.8578854 | 0.4151695 | +0.006554 |
  | M266 (M263 + g3 DART aug) | — | **0.6374709** | 0.8587750 | 0.4161667 | — |

  M266 was uploaded 2026-08-06 21:05 KST and is the **current best**. It is `M263 * 0.6 + g3 DART * 0.4`
  under policy `T0.6_G0.5`, and `M263` is itself a convex ensemble of `M261` (classifier) and `M252`
  (analog). `M263` is built but **never uploaded**, so the `+0.0009434` gain over M261 is not yet
  decomposed into ensemble effect versus group-3 augmentation effect. A predeclaration is frozen at
  `reports/n513_m263_predeclaration.json`.

- **The local->online offset does NOT transfer across method classes** (3.2x difference). Never apply an offset measured on one class to a ceiling or candidate of another. Local ranking does not preserve online ranking across method classes.
- **Saved leaderboard HTML is stale** (snapshot 2026-08-01 22:48, shows the pre-repository submission `0.62366`). The current best online result is M261 `0.6365274` (2026-08-04). Rank-20 snapshot reference: Total `0.65971` / 1-NMAE `0.87991` / FICR `0.43952`.
- **Closed on evidence:** external NWP sources (LDAPS-GFS error correlation ~0.78 caps averaging gain at 4.6% against ~11.0% required); post-processing the current representation (oracle FICR ceiling below requirement); direct band-hit estimation; sequence/issuance representation (an oracle daily-mean correction moves 1-NMAE `+0.020365` but FICR `-0.014849`).
- **Mechanism to remember:** the deployed prediction is an ACTION maximising expected settlement under a step reward, not a conditional mean. Moving it toward the conditional mean improves point accuracy and damages settlement by construction.

## Blend axis and artifact traps (2026-08-06 session)

### Validated offset model

`predicted_online(blend) = local(blend) + [w * 0.006554 + (1 - w) * 0.021119]`
where `w` is the classifier weight and `1-w` the analog weight. Validated against both online
anchors on a **single fixed policy** surface (`T0.5_G1.5`): `w=0` error `+0.0000000`,
`w=1` error `-0.000077`. Best blend `w = 0.70`, predicted online `0.639170`
(`blends/BLEND_M261w070_M252.csv`, sha `3d3e7041a1d9`). One degree of freedom, no fold-outside collapse.

### TRAP — `prediction_kwh` in `metric-aligned-probe` artifacts is not a single policy

`M102_TOP100-*.parquet` carries `T0.5_G1.5` on Q2, `T0.4_G2` on Q3, `T0.6_G1` on Q4.
`M115_XGBOOST-*.parquet` matches no single policy at all and mixes policies **per group**.
Its widely quoted local `0.638410` is therefore a post-hoc fold/group selection, not an
achievable single-policy score; the best single policy for M115 is `0.630662`.
**Before using any `prediction_kwh` column, assert that one policy column reproduces it on every
fold.** Six separate misreadings in this session traced back to skipping that check.

### Fold-outside gate rejects every multi-degree-of-freedom blend

- 7 members x 3 groups (21 dof): in-sample `0.646821` -> fold-outside `0.643888`, below the baseline
- per-group weights (3 dof): in-sample `0.640253` -> fold-outside `0.635453`, below uniform `0.639170`
- per-group fold-outside weights oscillate (`g3: 1.00 / 1.00 / 0.15`) — three folds cannot estimate them

Member error correlation against M115 is `0.984-0.994` for every classifier-family member and
`0.944` for the analog M244. There is no diversity to exploit beyond the single analog member.

### Standing rule

Any blend or comparison must state (a) which policy produced each input, (b) whether weights were
fitted in-sample or fold-outside, and (c) the row-alignment key set. Receipts without those three
fields are not admissible evidence.
