# S17-N21 terrain → M115 model integration audit

## Scope and verdict

- Node: `S17-N21_TERRAIN_SX300_H8_MODEL_FAMILY_PREREQUISITE_AUDIT`
- Frozen input bundle: `4f212b859c78e18c57d62551b0c74fd6540f9a8410ead3a66ef2d79385eb49dd` (31 local source packages)
- External requests: 0; model fits/predictions/policy calls/scores: 0.
- **Verdict: `READY_ONE_M115_TERRAIN_STRICT_FAMILY`.** This authorizes only a separately predeclared, fail-closed materialization. It does not claim that the historical M115 runtime will reproduce or that terrain improves score. [directly_supported] [unverified]

## Exact source contract

- `.planning/2026-08-01-leaderboard-top-4-loop/run_alternative_booster_classifier.py` `94b0fd79746fbbda8d1a32542d1d74e8e4b94f72cbe7e104df8e2badcdf06e22`: `_feature_names` line 34; XGBoost arm lines 74–87; 100-iteration-capable parameters include learning rate 0.03, depth 5, min-child-weight 20, subsample 0.9, colsample 0.8, seed 20260802, and six workers. Training is `forecast_kst_dtm < fold start` at line 166. [directly_supported]
- `.planning/2026-08-01-leaderboard-top-4-loop/run_sequence_classifier.py` `eb4ebf8bcfaaf2f7b46931cfa3ad81ccd204a5e7f89a12d5756f849c17f84d31`: raw grid join line 145. [directly_supported]
- `.planning/2026-08-01-leaderboard-top-4-loop/run_site_wind_classifier.py` `a97193a84970b07c7d55c795688ab27a92fc219247e8e9cc08724cbcc91ce5ef`: the 14 site-wind additions start at line 175 and the action grid is 0.075..1.075 by 0.0025 at line 207. [directly_supported]
- Every fold's M115 receipt has exactly 100 unique features, the exact M102 feature list, iteration 100, sweep {60,100,140}, 14 site-wind features, and no `terrain__sx300_h8_mean16`. [directly_supported]
- The historical raw policy differs by fold (`Q2 T0.75_G2`, `Q3 T0.4_G0`, `Q4 T0.75_G0.5`), so those post-hoc per-fold choices are forbidden. N7's Q2-frozen `T0.75_G2` is the only admissible policy. [contradicts_premise]

## Static/dynamic feature contract

- `train_grid_pivot.parquet` schema has exactly `ldaps__grid01..16__heightAboveGround_10_10u/10v`; no row values were read in this audit. [directly_supported]
- The frozen lookup is exactly 16×72, 1,152 finite values, and NPY equals Parquet. [directly_supported]
- Frozen feature: per timestamp and exact grid `j`, compute `atan2(-u_j,-v_j) mod 360`, nearest 5° with clockwise half ties, look up `Sx_j`, require all 16 finite/nonzero-direction inputs, and take the unweighted mean. Append only `terrain__sx300_h8_mean16`; no magnitude, coefficient, interaction, group variant, or parameter sweep. [directly_supported for executable formula; unverified for gain]

## Saved control identities

- On 19,440 N7 rows, `CHAMPION = 0.30*D + 0.70*mean(M102,M113,M115)` has max absolute error `0`. [directly_supported]
- N7 M115 equals saved `T0.75_G2` at max absolute error `0` on Q2/Q3/Q4 (6,480/6,408/6,552 rows). Q3/Q4 recorded vector hashes are `936902ba7de6a294c87088aa695db5591666a325416146` and `cbaab5480e8d344d1d810d4e966cac5223d8ff6fef3f8afcd3638a2ec0170635`. [directly_supported]
- Therefore a terrain member can replace only the M115 term as `CHAMPION + (0.70/3)*(M115_TERRAIN-M115_CONTROL)`, while Q2 remains unchanged burn-in. The weight `0.70/3` is algebraic, not fitted. [derived]

## Fail-closed blockers carried into the executable node

1. Historical M115 JSON receipts omit several fields emitted by the current source (training floor, class width, sequence and seasonal fields). Numerical refit identity is **unverified**. The next node must fit the unchanged control first and stop before score unless its fixed-policy action matches N7 at `max_abs <= 1e-6` and reconstructed Champion at `<=1e-9`. [unverified]
2. M64B `legacy.npy`/`allweather.npy` members have 78,912 elements (1,096 days×24×3), while only the first 52,560 are 2022–2023. Current `_surface()` plus `np.load` is unbounded and would materialize the 26,352-element 2024 tail. It is forbidden. [contradicts_premise]
3. The next runner must be self-contained: filtered 2022–2023 surface, bounded NPY-member prefix decoder exposing exactly 52,560 values, no `_surface()` invocation, and no assessment `actual_kwh` until all materialization gates pass. Prefix alignment remains [unverified] until the independent N7 action guard reproduces.

## Frozen later experiment

- Six fits only: three unchanged 100-feature M115 controls and three 101-feature terrain arms; at most six workers.
- Fixed iteration 100, fixed `T0.75_G2`, no policy or hyperparameter search.
- Q2 candidate action is unchanged Champion burn-in. Q3/Q4 use the algebraic replacement, clipped only to the frozen 0..1.075 capacity-factor action range.
- If every pre-score gate passes, S17-N3 evaluates Champion, matched zero control, and terrain candidate jointly at comparison index 4. Otherwise the node closes diagnostic with comparison null.

## Forbidden-access accounting

No weather feature value, label/actual value, 2024/test value, rejected ECMWF, quarantined N10, model fit, prediction, policy evaluation, metric, dependency change, Dacon action, or external request occurred. Static lookup values and saved action/control columns only were read. The delegated child ended without a file/reply; its bounded transcript findings were independently reproduced above, and root wrote this lane.
