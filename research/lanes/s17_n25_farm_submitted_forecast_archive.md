# S17-N25 — FARM_SUBMITTED_FORECAST_ARCHIVE

**Verdict: FAIL-CLOSED / DO NOT INTAKE.** [derived] The eligible official HTML documentation does not establish a public, anonymous, commercially reusable, historically fixed-issue archive of the exact Taebaek Gadeoksan day-ahead submitted forecast, or an exact crosswalk to the three target groups.

## Gate determination

| Requirement | Determination | Evidence |
|---|---|---|
| Exact plant submitted-forecast archive | **Not established** | [unverified] No eligible official page identifies a Taebaek Gadeoksan submitted-forecast dataset or proves that the farm was an active prediction-program resource throughout 2022–2023. |
| Complete, historically fixed issue for 2022–2023 | **Fails** | [directly_supported] The historical rule fixes **two** submissions—D-1 10:00 and D-1 17:00—for 24 hourly values, accepts the value nearest each deadline, and may carry forward an earlier submission. It proves the submission protocol, not archive publication, completeness, or retained issue identity. [S1] |
| Available by D-1 14:00 KST | **Conditional internal near-match only** | [derived] Of the two rule-defined issues, only the 10:00 issue precedes 14:00; the 17:00 issue does not. No public archive with an issue/receipt-time field was verified, so basis-safe selection cannot be reproduced. [S1] |
| Anonymous access | **Fails documented route** | [directly_supported] The rules describe member disclosure through an information system with managed access accounts; KPX describes market/system information as available to market participants through the trading system. [S2][S3] |
| Commercial reuse | **Not established** | [directly_supported] KPX grants commercial and non-commercial use when data is actually opened as public data, while non-listed data requires an application. The market-information rules separately restrict sale or onward provision of supplied information and derived statistics. No open-data designation or licence for this exact archive was found. [S2][S4] |
| Exact three-group mapping | **Fails** | [near_match_only] The prediction rule defines submission for the participant's relevant **resource**; the eligible HTML text supplies no plant-resource-to-three-group crosswalk or group-level forecast fields. [S1][S5] |

## Decisive evidence

- [directly_supported] A “renewable prediction quantity” is a participant's day-ahead hourly submission to KPX; the chapter applies to prediction-program applicants/participants and their submissions/settlement. This is an operational submission, not a stated public product. [S5]
- [directly_supported] Individual-member or individual-facility trading, metering, settlement, and operating information may be withheld where business interests could be harmed. Ad-hoc member disclosure uses account-managed access. [S2]
- [contradicts_premise] The only documented operational access path is participant/account-oriented, while the only documented route for public data absent from the published list is an application; neither establishes anonymous deployment symmetry. [S2][S3][S4]
- [unverified] Publication of the exact historical rows, preservation of both issue vintages, full 2022–2023 coverage, plant participation continuity, an exact three-group crosswalk, and an asset-specific commercial licence all remain unproved. Under the predeclared fail-closed rule, any one is fatal.

## Official sources

- **[S1]** KPX, historical Electricity Market Operation Rule §14.4.1, selected range 2022-01-01 through 2023-12-30: <https://marketrule.kpx.or.kr/lmxsrv/law/joHistoryContent.do?SEQ=2&SEQ_CONTENTS=21504&DATE_START=20220101&DATE_END=20231230>
- **[S2]** KPX, Electricity Market Operation Rule historical full text (including §§8.2.3.1, 8.2.3.7, 8.3.2 and Chapter 14): <https://marketrule.kpx.or.kr/lmxsrv/law/lawFullContent.do?SEQ=2&SEQ_HISTORY=33>
- **[S3]** KPX, “Electricity Trading System” documentation: <https://www.kpx.or.kr/menu.es?mid=a10401030000>
- **[S4]** KPX, “Public Data Application and Guidance”: <https://www.kpx.or.kr/menu.es?mid=a10107010000>
- **[S5]** KPX, historical Electricity Market Operation Rule §14.1.1: <https://marketrule.kpx.or.kr/lmxsrv/law/joHistoryContent.do?SEQ=2&SEQ_CONTENTS=5389&DATE_START=20220630&DATE_END=20221228>
