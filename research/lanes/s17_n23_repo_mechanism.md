# S17-N23 repository-mechanism lane — fail-closed scope audit

## Verdict

**`REJECTED_SCOPE_BREACH_NO_EXECUTABLE_CANDIDATE`.** The delegated reader was stopped and its substantive candidate conclusions are inadmissible. It opened `research/lanes/repo_wind_axis.md`, a pre-existing narrative that contains 2024/test-surface aggregate statements, despite the lane's explicit prohibition. It did not read raw labels/`actual_kwh`, run a fit/prediction/policy/metric, or write an artifact, but the text exposure alone prevents using that delegated lane as fresh research evidence. [contradicts_premise]

## Root-bounded independent facts

- `.planning/2026-08-01-leaderboard-top-4-loop/run_turbine_wind_power_stack.py` (SHA `f25dd05514a9bb8f495fe89bf6cba8560d986c742625d74ccd354210d562a5a0`) lines 38–56 filters SCADA strictly before the outer cutoff; lines 93–124 construct hourly direction sine/cosine; lines 220–271 define past-only cross-fit/apply masks. [directly_supported]
- The same runner drops direction from the teacher target at lines 560–571: it selects only `turbine_kwh` and `wind_speed`, then calls `_crossfit_teacher(..., wind_target, ...)`. Therefore a circular SCADA-direction auxiliary target is not implemented in that lineage. [directly_supported]
- Contemporaneous validation SCADA direction is unavailable at action time and is forbidden. A past-only direction teacher could in principle predict from issued NWP/static/calendar inputs, but its circular target aggregation, two-output loss, normalization, seeds, missing-direction treatment, and downstream single treatment are not frozen. It is constructible only `[derived]`, not an executable zero-variant family. [unverified]
- `.planning/2026-08-01-leaderboard-top-4-loop/run_wake_sector_classifier.py` (SHA `11c65cdd3d4d6d2a2fcfc71d22b0a1edc83bcedbd45e34cc3742e1b205f43c04`) already transforms issued group-level NWP u/v and supplied turbine layout into 60 wake/sector features. `M168_WAKE_SECTOR_Q3-dev-2023-Q3.json` (SHA `eb688c6a150b74a03676c960276b7539c279001e5379a002774f4676d756a729`) confirms `observed_scada_feature_count=0`; this is not a SCADA-direction teacher. [directly_supported]
- `M178_STRICT_TURBINE_WIND_STACK_Q3-dev-2023-Q3.json` (SHA `e6c6792ff3395bd9858717d60a2f4a084667c35eee6bbde323a96321bc76bdbd`) confirms the existing strict turbine teacher uses predicted wind-speed profiles and issued NWP direction features, but no predicted local-direction target. That historical Q3 screen does not authorize a new strict comparison. [directly_supported] [near_match_only]
- The frozen N22 action-only diagnostic (no actual values) shows the mean16 terrain arm changed Q3 actions on 154/142/100 rows for groups 1/2/3 and Q4 on 300/267/233. This describes mechanism coverage only and supplies no label-based choice. [directly_supported]

## Closed-axis/novelty conclusion

COST5/direct utility, issuance remapping, vertical PCA, source separation, direct hit estimation, existing site-wind speed teachers, NWP-direction wake sectors, and mean16 terrain are already failed or refuted in the authoritative ledger. The only literal repository gap found here—an auxiliary local-direction target—still has multiple unfrozen choices and cannot satisfy N23's exact one-treatment/zero-tuning execution rule. **Selected candidate: NONE.** [directly_supported]

## Access accounting

- Delegated files opened before stop: 21 local text/schema files; delegated repository writes: 0.
- Scope breach: one pre-existing narrative containing 2024/test aggregate statements; its performance claims were not used.
- Root raw label/`actual_kwh` reads: 0; raw 2024/test dataset reads: 0. A bounded child-transcript preview exposed pre-existing narrative aggregate statements from the prohibited lane; they are explicitly inadmissible and were not used for candidate selection. Fits/predict/policy/metric calls: 0; external requests: 0; dependency/Dacon/account actions: 0.
