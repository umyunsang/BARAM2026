# S17-N18 repository mechanism audit

## 0. Contract and verdict

- [directly_supported] This lane answers only `S17-N18_POST_RWA_FRONTIER_RESEARCH_INTAKE.repo_mechanism`; the frozen contract says repository-only/no-web, 30 minutes, no fit or score, and at most one next prerequisite (`reports/s17_n18_post_rwa_frontier_intake_predeclaration.json:39-44,53-62,68-84`, SHA-256 `2f00184f84f990d06125f5b89cb174a6b1f15a3afd1094d51568203deb33312f`).
- [contradicts_premise] The previously advertised `atm__`-to-site-wind-teacher gap is no longer an untested mechanism: later S15 code puts `atm__` into the common feature matrix and fits the supervised hub-wind teacher on that matrix.
- [contradicts_premise] A circular observed-wind-direction teacher is not a surviving distinct axis: an earlier repository lane directly bounded the oracle substitution of measured direction for NWP direction at only about a 1% residual-SD change and explicitly closed both measured-direction information and direction-remapping.
- [derived] **Verdict: `NONE_FOUND__NO_PREREQUISITE_AUDIT_PROPOSED`.** Within the admissible repository evidence, no chronology-reconstructible, genuinely untested information/representation axis survives the required exclusions and the two reconciliations above.
- [unverified] This is not a universal claim that no conceivable feature exists; it is a bounded repository-intake verdict, and untracked files without typed provenance cannot establish fresh-holdout novelty.

## 1. Evidence and provenance boundary

- [directly_supported] Git `HEAD` was `383265af810eee24129de2d7ce99457bb8a757ef`; the tracked HEAD version of `src/baram/features/weather.py` has SHA-256 `51feecba35173f53a875d98cd2cb2d462f40085005d23ba5f7d2e198c93545a4` (Git blob `41562b27f83e2af5a67f703c62f9f9475098b39b`) and constructs speed plus direction sine/cosine from supplied `u,v` (`:37-58`) while excluding raw angular columns from scalar aggregation (`:62-75`).
- [directly_supported] The tracked HEAD `artifacts/manifests/prepare.json` (line 1, JSON pointer `/feature_names`; SHA-256 `222ba303de7e963901cc0071b7769a1df3608eee85302bc697b76bc85833ca30`, Git blob `60ae899a0f64be8b56840567a243a2ada07753fa`) declares 704 features, including 77 `*_dir_sin*` and 77 `*_dir_cos*` names and no `scada` or raw `_wd` name.
- [directly_supported] The candidate-relevant later scripts and reports cited below are **not tracked by Git**: `git status --short --untracked-files=all` reports `??` for `research/lanes/repo_wind_axis.md`, `research/scratch/featbuild.py`, `research/nodes/harness.py`, `research/nodes/s15_n7_compose.py`, and `research/nodes/s15_n8_ablate.py`; `.planning/...` mechanism scripts are also untracked, while `artifacts/backtests/**` sidecars are ignored by `.gitignore:14`.
- [derived] Consequently, a current file plus a generated result is evidence of repository existence, not a cryptographically bound execution provenance, unless a sidecar pins its inputs/code; self-reported dates and current source text alone do not prove which bytes produced an artifact.
- [directly_supported] The repository's own legacy accounting reaches the same distinction: 88 declared cycles are classified `LEGACY_UNRECONSTRUCTABLE` because no atomic predeclared/closed transaction existed (`reports/s17_legacy_cycle_manifest.json:21-24,115-120`, SHA-256 `c6c3f4b5bba9ef4742d4b2225cb4987b36e4bc774b398330e59084e46893ba74`).

## 2. Required reconciliation: `atm__` regime features

### 2.1 What the older hypothesis actually established

- [directly_supported] `research/lanes/repo_wind_axis.md:383-417` (SHA-256 `11feef7627740c1c7f4bf4f218004274ca6db5b7b5ba7bd2a3f9e4631b56e2e4`) said `atm__` was created for an output PLS ranker but absent from the older teacher surface; the document itself described its negative claims as grep-based (`:327-330,586-589`).
- [directly_supported] The older surface supports that historical reading: `.planning/2026-08-01-leaderboard-top-4-loop/run_sequence_classifier.py:95-109` (SHA-256 `eb4ebf8bcfaaf2f7b46931cfa3ad81ccd204a5e7f89a12d5756f849c17f84d31`) selects support prefixes `gfs_spatial__`, `ldaps_spatial__`, `source_disagreement__`, `phys__`, and `phys_v2__`, not `atm__`; `run_site_wind_teacher.py:109-137` (SHA-256 `8749791de2ed005396f238131acf8f53033c61c95a5178c8b1020db2e6faa43b`) then takes numeric columns already present on that surface.

### 2.2 M205, M258, and M261 do not by themselves move `atm__` into that older teacher

- [directly_supported] M205 constructs an atmospheric block and concatenates it into an output classification/source-rank matrix (`.planning/.../run_atmospheric_regime_pls_rank.py:193-232,252-303`; SHA-256 `c75fbdbd8cbfa585e9af081c15642f95bf6887d4242c7d8233208a3e44614b19`), and its sidecar identifies architecture `strict_atmospheric_regime_m195_m196_source_rank_half_m197` with 200 atmospheric candidates (`artifacts/backtests/metric-aligned-probe/M205_STRICT_ATMOSPHERIC_PLS_RANK_Q3-dev-2023-Q3.json:1`, JSON pointers `/architecture`, `/atmospheric_feature_count`; SHA-256 `ab3d539d99aa8a7e71376f562a47e3b7d26f9f832a66653bfe05de1a8dc88065`).
- [derived] M205 therefore tests `atm__` in a downstream output ranker, not in the site-wind teacher; it neither proves nor disproves the older teacher-reach claim by itself.
- [directly_supported] M258 pins three fold-specific M64B site-wind caches and their hashes (`.planning/.../run_tweedie_trust_blend.py:34-45`) and declares `sitewind_source = fold_specific_M64B_strict_preceding_crossfit_cache` (`:420-445`; script SHA-256 `683b0658f209d1e6218232e779e9b0a94fea45f331cc9340ac7498a4c3d4c503`; sidecar SHA-256 `d9dcdf6b084eea9c197d3ec82240b558e1c4c904c25b15efad212be65a9f7c58`).
- [directly_supported] M261's static builder pins M102/M107/M64B receipts (`.planning/.../build_full_history_strict_temporal_champion.py:52-65,189-236`; SHA-256 `f6b95481a3551256dec9458a3bfb270ca9056efbbc45d29d6d2929f5a9737959`) and contains zero literal `atm__` occurrences.
- [derived] M258 and M261 thus reuse the older M64B speed-teacher lineage; neither is evidence that `atm__` reached that teacher. No generated M261 prediction, deployment-period row, or receipt body was used in this audit.

### 2.3 S15 is the decisive later contradiction

- [directly_supported] `research/scratch/featbuild.py` defines the atmospheric features (`:86-136`) and `build()` concatenates `add_atm(df)` into every returned feature frame (`:165-176`); `build2()` retains that frame (`:248-256`; file SHA-256 `bfbb8e2973df593ddd7d769b37053940168a3e4b12db2b88044273b609c4d1ab`).
- [directly_supported] `research/nodes/harness.py:40-64` calls `featbuild.build2()` and defines `COLS` as all columns except its three target/id fields (SHA-256 `406da8abf51ccbec3a6d8c7493e7f05ea7ed6a3e8752317a5d0e1211fa471ae5`), so `COLS` includes `atm__`.
- [contradicts_premise] S15-N7 obtains that `COLS`, constructs a historical hub-wind observation target, and fits/predicts the supervised `hub__ws_pred` teacher with `frame[COLS]` (`research/nodes/s15_n7_compose.py:112-171`; SHA-256 `28886e110b9449b9b2facd850a7472b910ed26f58da9bb9e812049a5e5ee8ac1`).
- [contradicts_premise] S15-N8 repeats the same chain and explicitly has a `no_B2` arm while otherwise fitting the hub-wind model on `COLS` (`research/nodes/s15_n8_ablate.py:29-83`; SHA-256 `2817337d223153985f09ea35bc59c1855617dab325c467f493f3759c2ed46338`).
- [unverified] N8 has no `no_atm__` arm, so the isolated incremental value of the atmospheric block is not attributable from that ablation; moreover, the S15 scripts/results are untracked and their small JSON outputs do not pin a code hash.
- [derived] That attribution gap is not a new information/representation mechanism: `atm__` has already entered the teacher, and an isolated re-fit ablation would be a follow-up attribution experiment requiring fits/scores, not a no-fit prerequisite.

## 3. Required reconciliation: circular observed-wind-direction teacher

- [directly_supported] The repository contains a near-match implementation that converts observed turbine `_wd` to sine/cosine before hourly aggregation (`.planning/.../run_turbine_wind_power_stack.py:85-114`; SHA-256 `f25dd05514a9bb8f495fe89bf6cba8560d986c742625d74ccd354210d562a5a0`), but it does not establish a fold-strict NWP-to-direction auxiliary teacher.
- [directly_supported] A binary artifact named `M14_SCADA_WIND_VECTOR_AUX-q4.parquet` exists (SHA-256 `096675f825f2cd768ab0ac29f47aed9e368d6357f97e6cb95bc2581b561755e6`, 127,393 bytes); footer-only inspection found 6,624 rows, one row group, and seven columns, but there is no companion M14 JSON or repository text that binds a producer/source hash to it.
- [unverified] The M14 filename cannot prove whether its direction was a valid-time observation, a predicted auxiliary representation, or something else; artifact existence is not recipe provenance.
- [directly_supported] The stronger nonduplication gate is already negative: `research/lanes/S6_feature_research.md:155-184` (SHA-256 `acd9a4b00cb585b13abab5a01823226afa7011cad5d40b03eb0b24b9133194aa`) fixes the direction convention, reports that LDAPS is within 30 degrees on 92.2% of generation-relevant rows, and reports only about a 1% residual-SD change when measured direction replaces NWP direction.
- [directly_supported] The same report explicitly closes both “measured `_wd` as information” and “NWP direction correction/remapping” (`research/lanes/S6_feature_research.md:964-970`).
- [derived] A fold-strict NWP-to-observed-direction teacher has no extra inference-time source beyond NWP and is bounded above by the measured-direction substitution; it is therefore not materially distinct from the closed axis, even though its exact code path was not found.
- [contradicts_premise] The tentative circular-direction candidate is closed and is not promoted to a prerequisite audit.

## 4. Other repository holes checked

- [directly_supported] Turbine-spread/intra-hour auxiliary teachers are not genuinely untested: `research/nodes/s7_enrich.py:2-4,19-48` (SHA-256 `b3b4c8ca312afa448ecb193a8c820c58136f5d1c3d725762e20c11c59fa7dff3`) explicitly predicts `v_spread`, `v_intra`, and `v_mean` and feeds them to a downstream member.
- [directly_supported] That script selects a policy on the other folds (`research/nodes/s7_enrich.py:75-81`), so its evaluation is excluded by this lane's future-fold-selection ban; independently, S6 bounds the NWP-explainable intra-hour/Jensen contribution and closes intra-hour point improvement (`research/lanes/S6_feature_research.md:435-465,975`).
- [directly_supported] The existing teacher construction already applies a curve at turbine/10-minute resolution before aggregation and records spread/intra summaries (`research/scratch/powercurve.py:59-78`; SHA-256 `fcd2c90a8b0007c52a4f7ee3bb15a933ddf8a64e281908bb270485a9e85ff852`), so “retain turbine/10-minute structure” is not a clean untouched representation premise.
- [derived] A static source scan over 553 admissible `.py` files under `src/`, the planning loop, `research/nodes/`, `research/scratch/`, and `research/engine/` produced index SHA-256 `e5243416d95e1bf412a3f30514b440efbb3800591480e36118ac38258e95d44e`: zero `scada_wd`, zero direction-target pattern, and zero fit-to-direction pattern; the only raw `_wd` hits were the two SCADA parsing scripts plus tracked angular-column exclusion.
- [derived] An exact-token scan over 249 development JSON sidecars produced index SHA-256 `3f0fba414f7a4cdeeb6354d45a6c79a26bedbbb87c9bb55761e9b91826b8dd79` and found no `scada_wd`, `direction_target`, `target_direction`, `_wd`, `direction_sin`, or `direction_cos`; this is bounded negative evidence, not proof of universal absence.
- [unverified] No repository-native transform establishes interval/accumulation semantics for a deaccumulation candidate, so under the no-web/no-raw-body bound it is not established as an applicable, chronology-reconstructible axis and is not elevated merely because a name search is empty.

## 5. Exclusion and selection decision

| Axis | Decision | Evidence tag |
|---|---|---|
| Policy remixing / postprocessing / direct band-hit / COST5 | Excluded by brief; not searched for promotion. | [directly_supported] |
| Future-fold selection | Excluded; `s7_enrich.py:75-81` is an example, not admissible candidate evidence. | [directly_supported] |
| Source separation, R2 vertical PCA, exact KMAPP correction, issuance remapping | Excluded by brief; no reopening claim is made. | [directly_supported] |
| `atm__` into wind teacher | Closed as already reached in S15; isolated attribution is not a new axis. | [contradicts_premise] |
| Circular observed-direction teacher | Closed by the measured-direction upper-bound/closure in S6. | [contradicts_premise] |
| Spread/intra-hour auxiliary representation | Existing implementation plus independent small upper bound; its recorded evaluation also violates the allowed selection rule. | [derived] |

- [derived] **No no-fit prerequisite audit is proposed.** An M14 provenance audit would not revive a direction axis already upper-bounded and closed; an `atm__` ablation would require refits/scores and would only attribute an already-used block.

## 6. Forbidden-access accounting

| Item | Accounting | Evidence tag |
|---|---|---|
| Web/network research | 0 calls; repository only. | [directly_supported] |
| Repository writes | Exactly this file; no other repository path written. | [directly_supported] |
| Model/optimizer fits | 0. | [directly_supported] |
| Score/metric calls or performance computation | 0; existing performance fields were not recomputed or used for candidate ranking. | [directly_supported] |
| Target/label values | 0 values/arrays/columns loaded; source-code literals and sidecar contract names are not data reads. | [directly_supported] |
| Parquet data rows | 0; only the M14 footer/schema metadata was inspected, and its bytes were hashed without row decoding. | [directly_supported] |
| Test-period/operating-period rows or generated predictions | 0; M261 was reconciled from static source contracts only, with no generated candidate/prediction body opened. | [directly_supported] |
| Rejected ECMWF / quarantined N10 artifacts | 0 artifact bodies opened; neither contributes evidence here. | [directly_supported] |
| Dependency changes, external actions, account/submission mutations | 0. | [directly_supported] |

- [derived] Final verdict remains `NONE_FOUND__NO_PREREQUISITE_AUDIT_PROPOSED`.
