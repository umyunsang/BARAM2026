# V5-EXTERNAL — external data / pretrained-weight eligibility ledger (SK@v5 Wave B)

**Lane:** `V5-EXTERNAL` (read-only research). The only repository write from this lane is this file.
**Executed:** 2026-08-09 KST, session start 05:20 KST. Terminated early on an explicit root STOP-AND-WRITE
instruction; remaining gaps are recorded below as `insufficient` unknowns rather than researched further.
**Collection performed:** anonymous HTTPS GET of provider API metadata JSON, repository card front-matter,
and licence text files only. **No data body, no model weight, no checkpoint, no GRIB, no NetCDF was
downloaded.** No fit, no score, no metric/policy call, no label/`actual_kwh`/2024/test read, no dependency
change, no install, no git, no Dacon, no remote inference, no delegation.

## 0. Authority verification (performed first, self-checked)

| Artifact | Expected SHA-256 | Observed | Result |
|---|---|---|---|
| `.planning/2026-08-01-leaderboard-top-4-loop/SK_v5.md` | `c2c813475ab5f0a741e6adbceab196d5b38447f2aeb02f9716a8f5be8973c820` | identical | MATCH |
| `reports/sk_v5_approval_receipt.json` | `e827b324a23913346598d274f2ded496ffc3134243de5636033ee6a3ac387173` | identical | MATCH |
| `research/nodes/sk5_foundation_map.json` | `393ee74bf53251037547fc52bc92c554a4aa207e2288f194766d836675512b35` | identical | MATCH |
| `AGENTS.md` | `91a11f95f94e8de86f5c84b204e1893830b3ed5693e3d7ede7f2e351e9891e9c` (recorded inside the frozen map) | identical | MATCH |
| `research/nodes/sk5_local_capability_constraint.json` (root message, mid-lane) | `3209042a0ec9ddd669992e94bf6137beb79582cc5ad62fa65337bd109be700dc` | identical | MATCH |

`[directly_supported]` All five hashes were recomputed locally with `shasum -a 256` before the referenced
content was used.

---

## 1. VERDICT

**`ELIGIBLE_SET_IS_EMPTY_UNDER_CURRENT_AUTHORITY`.**

`[directly_supported]` Five time-series / tabular pretrained-weight repositories clear **all five stated
eligibility gates** (pre-cutoff release, commercial licence, D-1 14:00 availability, anonymous reproducible
retrieval, deployment symmetry). They are the only candidates this lane found that clear all five.

`[directly_supported]` **Every one of them is `BLOCKED_DEPENDENCY`.** The project environment has no
deep-learning runtime and `uv.lock` contains zero entries for `torch`, `transformers`, `huggingface-hub`,
`safetensors`, or `gluonts`; dependency changes are forbidden by `AGENTS.md`, by this lane's hard
prohibitions, and by IP@v3 per the root-frozen capability constraint. There is therefore **no external
candidate that is simultaneously eligible and actionable** at this moment.

`[derived]` The correct reading is a two-column disposition, not one: *eligibility* (a licence/archive/time
property of the source) and *actionability* (a local runtime/authority property). Collapsing them would
either wrongly promote unusable weights or wrongly discard a class that a future explicit dependency
authorisation could unlock. This lane keeps them separate and reports both.

`[derived]` **An empty actionable set is the operative answer.** Nothing in this lane should be read as
recommending an external acquisition. The one structural insight worth carrying forward is that the *only*
external class which can ever satisfy gates 3 and 5 without argument is the class that consumes **no
external values at inference at all** — weights applied to the competition's own supplied tables. Every
external *data* class must instead win a live archive/latency argument, and every such argument in scope
has already been closed.

---

## 2. Source ledger

All retrieved 2026-08-09 KST, anonymously (no `Authorization` header, no account, no API key). Evidence type
is stated per row. Budget: **9 official source packages of a permitted 10.**

| ID | Official source package | Primary locator | Date relied on | What was used | Evidence type | Scope warning |
|---|---|---|---|---|---|---|
| X01 | Prior Labs TabPFN v2 weight repository | `https://huggingface.co/api/models/Prior-Labs/TabPFN-v2-reg`, `.../TabPFN-v2-clf`, raw `LICENSE.txt` at `main`, `/commits/main`, `/revision/4972a65a1b30?blobs=true` | created 2025-01-04 (reg) / 2025-01-02 (clf); last pre-cutoff commit `4972a65a1b30` 2025-11-04 | full licence text, commit list, file sizes | provider API metadata + verbatim licence text | licence tag on the card is `other`; only the full text is decisive |
| X02 | Prior Labs TabPFN-2.5 weight repository | `https://huggingface.co/api/models/Prior-Labs/tabpfn_2_5`, raw `LICENSE` at `main` | created 2025-11-03; licence "Last Revised: December 9, 2025" | full licence text | verbatim licence text | supersedes an earlier snippet-level reading in `research/lanes/S6_ext_C_repr.md` |
| X03 | Amazon Chronos-Bolt (base) | `https://huggingface.co/api/models/amazon/chronos-bolt-base`, raw `README.md`, `/commits/main`, `/revision/5d9f166d69f4?blobs=true` | created 2024-11-25; last pre-cutoff commit `5d9f166d69f4` 2025-11-21 | card front-matter licence, commits, weight size | provider API metadata + repository card front-matter | no standalone `LICENSE` file in the repo (HTTP 404); front-matter is the repository's licence declaration |
| X04 | Amazon Chronos-2 | `https://huggingface.co/api/models/amazon/chronos-2`, `/commits/main` | created 2025-10-30; all 11 commits pre-cutoff, latest 2026-06-05 | release dates, licence tag | provider API metadata | successor flagged by X03's `new_version` field; card licence tag only, full text not read |
| X05 | Google TimesFM 2.0 500m (PyTorch) | `https://huggingface.co/api/models/google/timesfm-2.0-500m-pytorch`, raw `README.md`, `/commits/main`, `/revision/dc2443792ce5?blobs=true` | created 2024-12-24; last pre-cutoff commit `dc2443792ce5` 2025-04-16 | card front-matter licence, commits, weight size | provider API metadata + card front-matter | no standalone `LICENSE` file (404) |
| X06 | Salesforce Moirai 1.1-R-large | `https://huggingface.co/api/models/Salesforce/moirai-1.1-R-large`, raw `README.md` | created 2024-06-14 | card front-matter licence | provider API metadata + card front-matter | audited to close the class, not to promote it |
| X07 | IBM Granite TinyTimeMixers r2 | `https://huggingface.co/api/models/ibm-granite/granite-timeseries-ttm-r2`, raw `README.md`, `/commits/main`, `/revision/d6a79570cac0?blobs=true` | created 2024-10-08; last pre-cutoff commit `d6a79570cac0` 2025-02-26 | card front-matter licence, commits, weight size | provider API metadata + card front-matter | no standalone `LICENSE` file (404) |
| X08 | Datadog Toto-Open-Base-1.0 | `https://huggingface.co/api/models/Datadog/Toto-Open-Base-1.0`, raw `README.md`, `/commits/main`, `/revision/0411ceb27bdf?blobs=true` | created 2025-04-30; last pre-cutoff commit `0411ceb27bdf` 2026-05-14 | card front-matter licence, commits, weight size | provider API metadata + card front-matter | repository has post-cutoff commits; pinning is mandatory |
| X09 | Korea `data.go.kr` special-day (public holiday) open API | `https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo?solYear=2023&solMonth=06` and `https://www.data.go.kr/data/15012690/openapi.do` | probed 2026-08-09 | live authentication behaviour of the endpoint | live endpoint response (HTTP status + error body) | probe returned metadata/error XML only; no dataset content was retrieved |

**Local (non-external) evidence used, not counted against the source bound:**

| ID | Locator | What was used | Evidence type |
|---|---|---|---|
| L01 | `/Users/um-yunsang/BARAM2026/pyproject.toml` | declared dependency tiers (`core`, `challenger`, `dev`, `experiment`, `graph`, `nwp`) | repository source text |
| L02 | `/Users/um-yunsang/BARAM2026/uv.lock` | 273 locked packages; zero entries named `torch`, `transformers`, `huggingface-hub`, `safetensors`, `gluonts` | repository lockfile text |
| L03 | `.venv/bin/python -c "importlib.util.find_spec(...)"` run through the **project's own interpreter** | `torch/transformers/huggingface_hub/safetensors/jax/onnxruntime` ABSENT; `xarray/cfgrib/lightgbm/sklearn` PRESENT | project-environment import-spec probe (no import executed, no install) |
| L04 | `research/nodes/sk5_local_capability_constraint.json` (root-frozen, SHA verified) | absent-package list and the IP@v3 dependency prohibition | root-frozen node |
| L05 | `research/rwa_external_eligibility.md`, `research/lanes/s17_n18_pretrained_nwp.md`, `research/lanes/s17_n23_landcover_roughness.md`, `research/lanes/S6_ext_C_repr.md` | prior-lane closures respected, not re-derived | prior in-repo lane artifacts |

---

## 3. Tagged claim ledger

Exactly one tag per claim.

### 3.1 Licence facts (verbatim primary text or repository card declaration)

| # | Claim | Tag |
|---|---|---|
| C01 | The TabPFN v2 weight repository ships `LICENSE.txt` headed "Prior Labs License / Version 1.1, May 2025", which states it "is a derivative of the Apache 2.0 license ... with a single modification: The added Paragraph 10 introduces an enhanced attribution requirement inspired by the Llama 3 license." | `directly_supported` |
| C02 | That licence's Section 2 grants "a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work", with **no field-of-use or non-commercial restriction anywhere in Sections 1–9**. | `directly_supported` |
| C03 | Paragraph 10 requires only attribution: a copy of the licence with distributed materials, prominent display of "Built with PriorLabs-TabPFN", and the "TabPFN" name prefix on distributed derived models; it also states "internal benchmarking and testing without external communication shall not qualify as distribution ... and no attribution under this Section 10 shall be required." | `directly_supported` |
| C04 | **TabPFN-2.5 is a different and non-commercial licence.** Its `LICENSE` ("TABPFN-2.5 License v1.1, Last Revised: December 9, 2025") makes the model available "freely available for your non-commercial and non-production use", and grants a licence "solely for your Non-Commercial Purposes". | `directly_supported` |
| C05 | The TabPFN-2.5 licence's definition of "Non-Commercial Purpose" **expressly includes** "Data Science Competitions", defined as "a publicly accessible contest hosted on established platforms (such as Kaggle, DrivenData, or ChallengeData) or by academic/non-profit institutions where participants compete to develop predictive models for specified datasets", "provided the results are not used in commercial decision-making, client deliverables, or paid products/services". | `directly_supported` |
| C06 | The repository card front-matter declares `license: apache-2.0` for `amazon/chronos-bolt-base`, `amazon/chronos-2`, `google/timesfm-2.0-500m-pytorch`, `ibm-granite/granite-timeseries-ttm-r2`, and `Datadog/Toto-Open-Base-1.0`. | `directly_supported` |
| C07 | The repository card front-matter declares `license: cc-by-nc-4.0` for `Salesforce/moirai-1.1-R-large`. | `directly_supported` |
| C08 | None of X03, X05, X06, X07, X08 serves a standalone `LICENSE` blob at `main` — each returned HTTP 404 "Entry not found" — so the licence declaration relied upon is the repository card front-matter, not a verbatim licence file in the weight repository. | `directly_supported` |

### 3.2 Release-date and pinnability facts (gate 1 and archive reproducibility)

| # | Claim | Tag |
|---|---|---|
| C09 | Repository creation dates, all on or before 2026-07-05: TabPFN-v2-reg 2025-01-04, TabPFN-v2-clf 2025-01-02, tabpfn_2_5 2025-11-03, chronos-bolt-base 2024-11-25, chronos-2 2025-10-30, timesfm-2.0-500m-pytorch 2024-12-24, moirai-1.1-R-large 2024-06-14, granite-timeseries-ttm-r2 2024-10-08, Toto-Open-Base-1.0 2025-04-30. | `directly_supported` |
| C10 | Each candidate has an identifiable **latest commit at or before 2026-07-05T23:59:59Z**, so a pre-cutoff revision pin exists: TabPFN-v2-reg `4972a65a1b30` (2025-11-04), chronos-bolt-base `5d9f166d69f4` (2025-11-21), timesfm-2.0-500m-pytorch `dc2443792ce5` (2025-04-16), granite-timeseries-ttm-r2 `d6a79570cac0` (2025-02-26), Toto-Open-Base-1.0 `0411ceb27bdf` (2026-05-14). | `directly_supported` |
| C11 | `Datadog/Toto-Open-Base-1.0` carries 50 commits with `lastModified` 2026-05-14 and `Prior-Labs/TabPFN-v2-clf` carries `lastModified` 2026-04-26; both are pre-cutoff, but repositories are mutable, so **`main` is not an admissible reference** and a commit pin is mandatory for reproducibility. | `directly_supported` |
| C12 | Weight-object sizes at the pinned revisions (metadata only, no body fetched): TabPFN-v2 regressor default 44,390,977 B; chronos-bolt-base `model.safetensors` 821,203,576 B; timesfm-2.0 `model.safetensors` 1,995,406,976 B; granite TTM-r2 `model.safetensors` 3,240,592 B; Toto-Open-Base-1.0 `model.safetensors` 605,239,264 B. | `directly_supported` |
| C13 | The HF revision-blobs endpoint returned **no LFS `oid` (sha256) field** for these weight objects in the anonymous metadata response, so a content hash for each pinned weight was **not** established by this lane. | `directly_supported` |

### 3.3 Retrieval and availability facts (gates 3, 4, 5)

| # | Claim | Tag |
|---|---|---|
| C14 | Every provider request in this lane succeeded with **HTTP 200 and no `Authorization` header, no token, no account and no API key**, across `api/models/{repo}`, `api/models/{repo}/commits/main`, `api/models/{repo}/revision/{sha}?blobs=true`, and `{repo}/raw/{rev}/{file}`. Anonymous reproducible metadata retrieval is therefore demonstrated for X01–X08. | `directly_supported` |
| C15 | A pretrained weight applied to the **competition's own supplied tables** consumes **no external values at inference**, so gate 3 (values available by D-1 14:00 KST from a fixed archive) has no external object to fail on, and gate 5 (train/inference symmetry under the same latency) holds by construction because the same frozen local checkpoint is used at both times. | `derived` |
| C16 | The Korean government special-day/holiday open API returns **HTTP 401 with `<errMsg>SERVICE_KEY_IS_NULL</errMsg>` and `<returnAuthMsg>서비스 접근거부</returnAuthMsg>`** to an unauthenticated request, so it **fails gate 4 (anonymous retrieval)** as an operational input path. | `directly_supported` |
| C17 | The `data.go.kr` dataset landing page for that service is anonymously reachable (HTTP 200) and serves `<meta name="robots" content="noindex, nofollow">`; landing-page reachability is **not** dataset reachability and does not cure C16. | `directly_supported` |

### 3.4 Local capability facts (actionability, not eligibility)

| # | Claim | Tag |
|---|---|---|
| C18 | `pyproject.toml` declares six dependency tiers (core, `challenger`, `dev`, `experiment`, `graph`, `nwp`) and **not one of them contains `torch`, `transformers`, `huggingface-hub`, or `safetensors`**. | `directly_supported` |
| C19 | `uv.lock` contains 273 `name = ` entries and **zero** matches for `torch`, `transformers`, `huggingface-hub`, `safetensors`, `gluonts`. | `directly_supported` |
| C20 | Probed through the project's own interpreter `.venv/bin/python`, `find_spec` returns `None` for `torch`, `transformers`, `huggingface_hub`, `safetensors`, `jax`, `onnxruntime`, and non-`None` for `xarray`, `cfgrib`, `lightgbm`, `sklearn`. | `directly_supported` |
| C21 | **Therefore every candidate in §3.1–§3.3 that clears all five eligibility gates is nonetheless non-actionable**: each requires a PyTorch-class runtime that does not exist locally and cannot be added, because dependency change is prohibited by `AGENTS.md`, by this lane's hard prohibitions, and by IP@v3 per the root-frozen `sk5_local_capability_constraint.json`. | `derived` |
| C22 | The prior in-repo lane `research/lanes/s17_n18_pretrained_nwp.md` independently reached `NO_EXECUTABLE_LOCAL_PRETRAINED_NWP_IN_UNCHANGED_ENVIRONMENT` for the *NWP*-weight class on a 16 GiB Apple-M1 macOS host with none of the runtime packages installed; the present lane finds the identical blocker for the *time-series/tabular* weight class. Two independent weight classes, one shared blocker. | `directly_supported` |

### 3.5 `contradicts_premise` findings

| # | Finding | Tag |
|---|---|---|
| P01 | **The premise that "eligible" and "usable" can be judged on one axis is contradicted locally.** Five candidates pass all five stated eligibility gates and zero of them are usable. Gate design that stops at licence/date/archive/anonymity/symmetry cannot detect the binding constraint, which is the local runtime and the dependency-change prohibition. Any future external-eligibility gate set should carry an explicit sixth *runtime-and-authority* gate. | `contradicts_premise` |
| P02 | **The in-repo reading that TabPFN-2.5 is simply barred is incomplete.** `research/lanes/S6_ext_C_repr.md` records TabPFN-2.5 as "비상업 라이선스 ... 대회 규칙 위반" from a snippet. The verbatim licence in fact carves out "Data Science Competitions" *inside* its definition of Non-Commercial Purpose (C05). The BARAM disposition is unchanged — `AGENTS.md` requires a licence that **permits commercial use**, which TabPFN-2.5 does not — but the *reason* recorded in that prior lane is not what the licence says, and a snippet-derived licence claim was wrong on its face. | `contradicts_premise` |
| P03 | **"The licence file is the licence" does not hold on this platform.** Five of the audited weight repositories serve HTTP 404 for `LICENSE` and declare terms only in README front-matter (C08). An audit method that demands a verbatim licence blob would have returned five spurious `UNRESOLVED` rows. | `contradicts_premise` |

---

## 4. Scope-match matrix against the BARAM surface

Applied to the surviving class (pretrained time-series/tabular weights consuming only supplied inputs).
`MATCH` = no gap; `GAP` = a real difference that must be paid for locally; `N/A` = the axis does not bind.

| Axis | BARAM surface | Candidate class | Disposition |
|---|---|---|---|
| population | 17 turbines, 3 KPX groups, Korean sites | generic pretrained corpora, not wind-plant specific | GAP — no wind prior is claimed |
| geography | Korea, complex ridge terrain | global/agnostic; no geography in the model | N/A for weights, GAP for any claimed skill |
| horizon | 24 hourly targets 01:00–00:00 at fixed D-1 14:00 issuance | Chronos/TimesFM/Toto/TTM are fixed-context autoregressive horizon forecasters; TabPFN is a tabular regressor with no horizon notion | GAP — the BARAM task is covariate-driven regression at a fixed issuance, not free-running extrapolation |
| issue_time | one action per day at D-1 14:00 KST | weights are static; no issue-time semantics | MATCH |
| inputs | supplied GFS (9 pts) + LDAPS (16 pts) issued tables + static turbine metadata | **no external inputs required** | MATCH — this is the class's single decisive advantage |
| target | hourly group production, kWh | continuous series / tabular target | MATCH in form |
| metric | `Total = 0.5*(1-NMAE) + 0.5*FICR`, discontinuous 6%/8% settlement bands, equal-group | all candidates are trained on continuous point/probabilistic losses | GAP — none optimises a step reward; the deployed number is an ACTION, not a conditional mean |
| resolution | hourly, 72 cells per issuance | hourly-capable | MATCH |
| topology | 3 groups, equal-weight aggregation | no group structure | GAP — group equality must be imposed externally |
| compute | project `.venv`, no GPU, no torch, no dependency change | all require a PyTorch-class runtime | **BLOCKING GAP** (C18–C21) |
| licence | must permit commercial use | X01/X03/X04/X05/X07/X08 pass; X02/X06 fail | split — see §6 |

`[derived]` The metric-axis gap deserves emphasis independent of the runtime block. Even with a runtime,
a foundation model imported for its point-forecast quality would push the deployed number **toward the
conditional mean**, which the frozen foundation records as the mechanism that improves accuracy while
damaging settlement by construction. The only defensible use of this class would be as a **feature/embedding
producer feeding the existing action policy**, never as a direct predictor — and that use is exactly the one
with no published transfer evidence.

---

## 5. Source fact → provisional migration hypothesis → local evidence needed

### H1 — Tabular in-context prior as an alternative estimator family (TabPFN v2)

- **(a) SOURCE FACT** `[directly_supported]` — `Prior-Labs/TabPFN-v2-reg` `LICENSE.txt`, "Prior Labs License
  v1.1, May 2025", Section 2 unrestricted commercial grant, Paragraph 10 attribution only; pinnable at
  `4972a65a1b30` (2025-11-04); 44,390,977 B default regressor; retrieved anonymously 2026-08-09.
- **(b) PROVISIONAL BARAM MIGRATION HYPOTHESIS** `[near_match_only]` — a prior-fitted in-context tabular
  regressor could supply an error signal decorrelated from the GBDT family, relieving the diversity floor
  that `AGENTS.md` records (member error correlation 0.984–0.994 across the classifier family).
- **(c) LOCAL EVIDENCE NEEDED** — (i) an authorised dependency exception for a PyTorch-class runtime;
  (ii) a schema-only count of development rows and prepared feature width against the model's documented
  operating envelope; (iii) *only then* a fold-outside band-hit-indicator correlation against the champion.
- **Status: `BLOCKED_DEPENDENCY`.** Note that a prior in-repo lane already records a scale objection
  (~44k rows × ~830 columns against a documented 10,000-row / 500-feature envelope). This lane did **not**
  re-derive that and does not rely on it; it is flagged so the root does not treat H1 as merely runtime-blocked.

### H2 — Pretrained sequence encoder as a feature producer (Chronos-Bolt / TimesFM / Toto / TTM)

- **(a) SOURCE FACT** `[directly_supported]` — four Apache-2.0-declared repositories, all created before the
  cutoff, all pinnable pre-cutoff (C09, C10), all anonymously retrievable (C14), weight sizes 3.2 MB
  (TTM-r2) to 2.0 GB (TimesFM 2.0).
- **(b) PROVISIONAL BARAM MIGRATION HYPOTHESIS** `[near_match_only]` — frozen encoder embeddings of the
  supplied NWP series, consumed as additional GBDT features, could add representation that the current
  820-name prepared lineage lacks (FD6 records zero `geom__`/`seq__`/`clim` prefixes).
- **(c) LOCAL EVIDENCE NEEDED** — (i) dependency authority; (ii) proof that the embedding is computed
  strictly from past-only, pre-issuance inputs under the prequential origin rule; (iii) a no-fit diagnostic
  that the embedding is not a rank-deficient re-encoding of existing columns.
- **Status: `BLOCKED_DEPENDENCY`.** `[derived]` TTM-r2 at 3,240,592 B is by far the cheapest member of this
  class and would be the correct first probe **if and only if** a runtime exception is ever granted.

### H3 — Public calendar/holiday state as a lawful proxy for operational regime

- **(a) SOURCE FACT** `[directly_supported]` — the Korean government special-day open API refuses
  unauthenticated requests with `SERVICE_KEY_IS_NULL` / `서비스 접근거부` (C16).
- **(b) PROVISIONAL BARAM MIGRATION HYPOTHESIS** `[insufficient]` — a deterministic day-type state known at
  D-1 14:00 might carry residual information about the operational regime behind FD8.
- **(c) LOCAL EVIDENCE NEEDED** — none pursued. The retrieval path fails gate 4, the mechanism route runs
  through curtailment (a closed axis), and no primary evidence of a wind-production effect was sought.
- **Status: `INELIGIBLE (gate 4)` as an API path; the underlying hypothesis is `insufficient` and unpursued.**

---

## 6. Eligibility ledger — every row ends ELIGIBLE / INELIGIBLE(gate) / UNRESOLVED(missing evidence)

Gate 1 = released ≤ 2026-07-05 · Gate 2 = commercial-use licence · Gate 3 = values available by
D-1 14:00 KST from a fixed archive, not a later revision · Gate 4 = anonymous reproducible retrieval ·
Gate 5 = deployment symmetry. **Actionability is reported separately and is not a gate.**

| # | Candidate | Pin | G1 | G2 | G3 | G4 | G5 | **Disposition** | Actionability |
|---|---|---|:--:|:--:|:--:|:--:|:--:|---|---|
| E1 | TabPFN v2 (`Prior-Labs/TabPFN-v2-reg`) | `4972a65a1b30` | PASS | PASS (Prior Labs v1.1 = Apache-2.0 + attribution) | PASS (no external values) | PASS | PASS | **ELIGIBLE** | `BLOCKED_DEPENDENCY` (torch absent) + prior-lane scale objection |
| E2 | Chronos-Bolt base (`amazon/chronos-bolt-base`) | `5d9f166d69f4` | PASS | PASS (apache-2.0) | PASS | PASS | PASS | **ELIGIBLE** | `BLOCKED_DEPENDENCY` |
| E3 | TimesFM 2.0 500m PyTorch (`google/timesfm-2.0-500m-pytorch`) | `dc2443792ce5` | PASS | PASS (apache-2.0) | PASS | PASS | PASS | **ELIGIBLE** | `BLOCKED_DEPENDENCY` |
| E4 | Granite TinyTimeMixers r2 (`ibm-granite/granite-timeseries-ttm-r2`) | `d6a79570cac0` | PASS | PASS (apache-2.0) | PASS | PASS | PASS | **ELIGIBLE** | `BLOCKED_DEPENDENCY` (smallest at 3.24 MB) |
| E5 | Toto-Open-Base-1.0 (`Datadog/Toto-Open-Base-1.0`) | `0411ceb27bdf` | PASS | PASS (apache-2.0) | PASS | PASS | PASS | **ELIGIBLE** | `BLOCKED_DEPENDENCY` |
| E6 | Chronos-2 (`amazon/chronos-2`) | last pre-cutoff commit 2026-06-05 | PASS | PASS (card tag `apache-2.0`; **full text not read**) | PASS | PASS | PASS | **UNRESOLVED** — missing: verbatim licence text and an explicit pinned commit id | `BLOCKED_DEPENDENCY` regardless |
| E7 | TabPFN-2.5 (`Prior-Labs/tabpfn_2_5`) | n/a | PASS | **FAIL** — "non-commercial and non-production use" | PASS | PASS | PASS | **INELIGIBLE (gate 2)** | n/a |
| E8 | Moirai 1.1-R-large (`Salesforce/moirai-1.1-R-large`) | n/a | PASS | **FAIL** — `cc-by-nc-4.0` | PASS | PASS | PASS | **INELIGIBLE (gate 2)** | n/a |
| E9 | Korea `data.go.kr` holiday/special-day API | n/a | PASS | not reached | not reached | **FAIL** — HTTP 401 `SERVICE_KEY_IS_NULL` | not reached | **INELIGIBLE (gate 4)** | n/a |
| E10 | Self-computed deterministic ephemeris (solar zenith/azimuth, daylight fraction) from the supplied lat/lon and timestamps | n/a | N/A — no external release | N/A — not a licensed work | PASS — deterministic, known arbitrarily far ahead | N/A — nothing is retrieved | PASS | **ELIGIBLE (not an external source at all)** | implementable on the present stack; **but no mechanism evidence was gathered by this lane** — see U4 |

**Closed axes respected, not re-argued** (no budget spent, no new primary evidence found or sought):
external NWP averaging including GEFS, ECMWF IFS/ENS acquisition, reanalysis of any kind, remote API
inference, test-period observations, CFSv2, direct farm/regional forecast products, grid outage feeds,
icing products, DEM/terrain shading, land-cover roughness (prior lane verdict `NOT_READY`), and the
NWP-weight class FourCastNet / Aurora / AIFS / GraphCast / Pangu (prior lane verdict
`NO_EXECUTABLE_LOCAL_PRETRAINED_NWP_IN_UNCHANGED_ENVIRONMENT`).

---

## 7. Ranked candidates mapped to FD deficits

Ranking is by *expected usefulness conditional on a dependency exception being granted*. Absent that
exception the correct rank order is: **do nothing external.**

| Rank | Candidate | FD deficit it would relieve | Why ranked here | Blocker |
|---|---|---|---|---|
| 1 | **Granite TTM-r2** (E4) | **FD6** — implemented representation is not active prepared lineage (zero `seq__` prefixes) | smallest payload in the class by two orders (3.24 MB), Apache-2.0, pinned pre-cutoff; cheapest possible test of whether any pretrained sequence encoder adds a `seq__`-class feature | `BLOCKED_DEPENDENCY` |
| 2 | **Chronos-Bolt base** (E2) | **FD6** | most-used member of the class by a wide margin, encoder-only and non-autoregressive, 821 MB | `BLOCKED_DEPENDENCY` |
| 3 | **Toto-Open-Base-1.0** (E5) | **FD6** | different pretraining corpus from E2/E4, 605 MB, Apache-2.0 | `BLOCKED_DEPENDENCY` |
| 4 | **TabPFN v2** (E1) | **FD6**, and indirectly the diversity floor behind the blend axis | only non-GBDT tabular estimator family that clears licensing; but carries a documented scale objection from a prior lane | `BLOCKED_DEPENDENCY` + scale |
| 5 | **TimesFM 2.0** (E3) | **FD6** | 2.0 GB for no documented advantage over E2 on covariate-driven tasks | `BLOCKED_DEPENDENCY` |
| — | **Chronos-2** (E6) | **FD6** | cannot be ranked until its verbatim licence is read | `UNRESOLVED` + `BLOCKED_DEPENDENCY` |

`[derived]` **No candidate in this lane addresses FD1, FD2, FD3, FD4, FD5, FD7, FD8 or FD9.** Those eight
deficits are properties of the *supplied* data lineage, the evaluation population, the diagnostic registry,
the metric-inference surface, the adaptive-reuse ledger, the unobservable operational state, and the engine.
**No external dataset or pretrained weight can relieve any of them.** This is the single most important
routing conclusion of the lane: seven of nine active deficits are structurally outside the reach of the
external axis, and the eighth (FD8) is unobservable by construction under the supplied schema.

---

## 8. Rejected alternatives and why

| Alternative | Why rejected | Tag |
|---|---|---|
| TabPFN-2.5 / 2.6 / 3 | licence permits only non-commercial and non-production use; `AGENTS.md` requires a commercial-use licence. The licence's Data-Science-Competition carve-out does not cure this, because the BARAM gate is commercial-use, not competition-use | `directly_supported` |
| Moirai / Moirai-MoE | `cc-by-nc-4.0` fails the commercial-use gate | `directly_supported` |
| Korean government holiday API | fails anonymous retrieval (HTTP 401 without a service key); the mechanism route also runs through curtailment, a closed axis | `directly_supported` |
| Python `holidays`-style packaged calendars | would be a dependency change, forbidden | `derived` |
| Any external NWP source (GEFS, ECMWF IFS/ENS, CFSv2, JMA, KMA products) | closed axis; prior in-repo lanes already hold the primary archive/latency evidence and this lane found no new mechanism | `directly_supported` |
| Reanalysis (ERA5, MERRA-2, and all others) | forbidden outright by `AGENTS.md` | `directly_supported` |
| Remote API inference of any hosted model | forbidden outright by `AGENTS.md`; also breaks deployment symmetry | `directly_supported` |
| NWP-weight class (FourCastNet, Aurora, AIFS, GraphCast, Pangu) | GraphCast/Pangu fail the commercial gate at the cutoff snapshot; the rest fail local executability. Prior lane holds the evidence; not re-derived | `directly_supported` |
| Land-cover / roughness rasters | prior lane verdict `NOT_READY_NO_DIRECTLY_SUPPORTED_NEW_REPRESENTATION` | `directly_supported` |
| KPX / EPSIS published generation series | test-period observations are forbidden, and publication latency cannot meet a D-1 14:00 cutoff for the operative day | `derived` |

---

## 9. Smallest discriminating local experiment per surviving candidate

Every ladder below is **gated on a prior, explicit dependency authorisation that does not currently exist.**
None may start under this lane or under IP@v3 as frozen. Steps are ordered cheap-diagnostic → bounded
screen → full strict comparison, and each carries its falsifier.

### Ladder A — does the pretrained sequence-encoder class add anything at all? (E4 first, then E2, E3, E5)

1. **No-fit diagnostic (no weights, no runtime, no download).** On the *existing* prepared lineage, compute
   the rank and condition number of the current feature block restricted to per-series temporal columns, and
   the number of distinct `seq__`-class quantities it already spans. **Falsifier:** if the existing block
   already spans a temporal subspace of dimension comparable to the encoder's output width, an embedding
   cannot be more than a rotation and the whole ladder is abandoned. **Cost:** minutes, present stack only.
2. **Bounded screen (requires dependency authority).** Pin `d6a79570cac0`, compute frozen embeddings for one
   group and one fold under strict past-only origins, and measure the **band-hit indicator correlation**
   (`1{|a−y| ≤ h}`, not continuous error correlation) between an embedding-augmented member and the champion.
   **Falsifier:** hit-indicator correlation ≥ 0.95 ⇒ no exploitable diversity ⇒ stop.
3. **Full strict comparison.** Fold-outside weights only, champion policy frozen, all three groups, strict
   prequential expanding origins on 2023. **Falsifier:** fold-outside Total improvement < 0.002, or any
   group's FICR degrading while NMAE improves ⇒ reject.

`[derived]` Step 1 is the one step that is **executable today on the present stack** and is the only part of
this lane's output that could be acted on without any new authority. It is also the step most likely to kill
the ladder cheaply.

### Ladder B — TabPFN v2 as a diversity member (E1)

1. **No-fit diagnostic.** Schema-only count of development rows and prepared feature width against the
   model's documented operating envelope. **Falsifier:** if rows or features exceed the documented envelope
   by a factor the provider's own publication warns against extrapolating past, stop before any runtime work.
2. **Bounded screen.** Only if step 1 passes: single group, single fold, hit-indicator correlation against
   the champion. **Falsifier:** ≥ 0.95 ⇒ stop.
3. **Full strict comparison.** As Ladder A step 3.

### Ladder C — resolve E6 (Chronos-2)

1. **One metadata read** of the verbatim licence and one commit pin. **Falsifier:** any non-commercial or
   field-of-use term ⇒ `INELIGIBLE (gate 2)`. **Cost:** one HTTP GET. Not performed here for budget reasons.

---

## 10. Unknowns (explicit)

| # | Unknown | Tag | Exact missing evidence |
|---|---|---|---|
| U1 | Verbatim licence text for `amazon/chronos-2` | `insufficient` | one anonymous GET of the repository's licence text plus a pinned pre-cutoff commit id |
| U2 | Content hash (LFS sha256) of each pinned weight object | `insufficient` | the anonymous blobs endpoint returned no `oid`; an LFS pointer read or a `HEAD` on the resolve URL would supply it |
| U3 | Whether any of E1–E5 runs on CPU within this host's memory at usable latency | `insufficient` | not investigated — moot while the dependency prohibition stands |
| U4 | Whether self-computed ephemeris (E10) carries any signal for this target | `insufficient` | no mechanism evidence gathered; E10 is recorded as *eligible*, not as *recommended* |
| U5 | Whether the Dacon rule text itself would accept TabPFN-2.5's competition carve-out | `insufficient` | the official rule page was not re-read in this lane; `AGENTS.md`'s commercial-use requirement was applied fail-closed instead |
| U6 | Any external source class not enumerated in §6 | `insufficient` | the lane terminated early on a root STOP instruction with 1 of 10 audit slots unspent |
| U7 | Provider terms-of-service constraints on bulk anonymous weight retrieval | `insufficient` | platform ToS was not read; only per-repository licences were |

---

## 11. DS/IP implications

**For `DS@v5`:**

1. `[derived]` **Record the external axis as closed-with-a-conditional-reopen, not open.** The eligible set is
   non-empty on the five stated gates but empty in actionable terms. `DS@v5` should carry exactly one
   external-axis node: *"pretrained time-series/tabular weights, FD6, `BLOCKED_DEPENDENCY`, revival premise =
   an explicit dependency authorisation."* No acquisition node, no data-collection node.
2. `[derived]` **Add a sixth gate to the external-eligibility contract: runtime-and-authority.** P01 shows the
   five-gate design cannot see the binding constraint. Any future lane using the five gates alone will keep
   producing eligible-but-useless rows.
3. `[derived]` **Route effort to FD1–FD5 and FD7–FD9.** Eight of nine active deficits are unreachable from the
   external axis. The external lane's most valuable output is the negative result that narrows the search.
4. `[derived]` **Ladder A step 1 is a legitimate present-stack DS node.** A rank/condition diagnostic on the
   existing prepared lineage tests FD6 directly, needs no external anything, and is cheap.

**For `IP@v3`:**

5. `[derived]` **Do not request a dependency exception on this lane's evidence.** Nothing here establishes an
   expected gain; every transfer claim in §5 is `near_match_only` by construction, and per the evidence
   contract a paper's or benchmark's reported improvement can never be quoted as a BARAM expected gain.
   The honest ask, if ever made, is for Ladder A step 1 *first* — which requires no exception at all.
6. `[directly_supported]` **If an exception is ever granted, pin by commit, never by `main`.** C11 shows two
   audited repositories mutated after their initial release and one has 50 commits; `main` is not reproducible.
7. `[directly_supported]` **Attribution obligations are live for E1.** Prior Labs Paragraph 10 requires the
   "Built with PriorLabs-TabPFN" notice and a "TabPFN" name prefix *on distribution*, though explicitly not
   for internal benchmarking without external communication. Apache-2.0 candidates require NOTICE retention.
8. `[derived]` **Claim limits are unchanged.** The 2024 lockbox is consumed and no fresh holdout exists, so
   nothing proposed here could be independently confirmed even if it were executed.

---

## 12. What this lane did NOT do

No data body, weight, checkpoint, GRIB, NetCDF or raster was downloaded. No model was fit. No score, metric
or policy was computed or called. No `actual_kwh`, label, prediction, model result, 2024 value, lockbox value
or test-period value was read. No dependency was added, removed or resolved. No package was installed. No
repository write was made outside this file. No git, Dacon, account, browser, remote compute or remote
inference action was taken. No subagent was spawned. One of ten permitted audit slots was left unspent
because the lane was stopped early by the root; U1 and U6 are the direct consequence.
