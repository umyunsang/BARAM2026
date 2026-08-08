# S17-N25 — REGIONAL_RENEWABLE_FORECAST

- Audit window: 2026-08-09 00:38–00:47 KST (bounded desk audit).
- Source budget: 11/12 official provider documentation or catalogue-metadata landing pages.
- Boundary: metadata/documentation HTML only. No operational API call, catalogue search call, file/attachment, preview, row body, account, or application was used.
- Admission test: a direct regional wind/renewable **generation forecast** must preserve fixed issue runs for all of 2022–2023, be available no later than D-1 14:00 KST, permit anonymous commercial deployment, and provide an authoritative choice-free mapping to G1, G2, and G3.

## Evidence audit

| Product / official evidence | Forecast class and issue chronology | 2022–2023 fixed-run archive | Licence / anonymous deployment | Exact three-group map | Gate |
|---|---|---|---|---|---|
| KMA `에너지기상 요약정보` [1] | [directly_supported] Began 2026-06-26; displayed around 06:00 daily; covers today through the day after tomorrow; publishes regional irradiance and wind-speed summaries from KIM-지역. [derived] A D target is in the D-1 morning horizon, but this is meteorological-model output, not wind/renewable generation. | [contradicts_premise] The documented service began after the required archive interval; no preserved issued runs for that interval are documented. | [unverified] Public browser reading is anonymous. KMA policy permits reuse only when the item has the applicable KOGL mark; otherwise prior consultation is required [4]. The inspected service pages do not document a product-specific mark or anonymous machine interface. | [contradicts_premise] Nine administrative regions are documented, but no competition-group crosswalk or farm output is supplied. | Reject |
| KMA weekly sunlight/wind climate outlook [2] and platform description [3] | [directly_supported] The outlook began in 2025 and is a Thursday-issued next-week mean/probability map. The platform describes wind-speed, direction, and turbulence observation/prediction/analysis. [contradicts_premise] Neither is a fixed-issue wind-generation forecast. | [contradicts_premise] Start dates and descriptions do not support a 2022–2023 issue archive. | [unverified] Same product-specific licence and anonymous deployment gap under [4]. | [unverified] No G1/G2/G3 identifiers or authoritative crosswalk. | Reject |
| KMITI catalogue: ML Jeju wind-generation forecast [5] | [directly_supported] Direct Jeju wind-generation forecast; four releases per day and a +48-hour horizon are stated. [unverified] Exact issue times and publication latency are absent, so a unique run known by D-1 14:00 cannot be selected. | [contradicts_premise] The landing metadata was created only at the end of 2023 and explicitly calls the offered object a sample; it does not document a complete 2022–2023 issued archive. | [contradicts_premise] “Free” is a price, not a reuse licence. Additional supply requires contacting the provider, and the landing workflow requests a use purpose; anonymous repeatable deployment is not documented. | [contradicts_premise] Jeju is one aggregate region, with no farm/group identifiers or allocation rule. | Reject |
| KMITI catalogue: farm wind-speed/generation forecast [6] | [directly_supported] Direct forecast concept, four releases per day, +48-hour horizon. [contradicts_premise] The only described object is an unspecified random sample; detailed output requires plant equipment and generation records and a plant-specific model. | [contradicts_premise] Created after the required interval; no historical issued archive is documented. | [contradicts_premise] Provider contact and plant inputs are required; no commercial reuse licence or anonymous deployment contract is stated. | [contradicts_premise] The sample deliberately identifies no plant, so it cannot map any competition group. | Reject |
| 60Hertz `햇빛바람지도` provider pages [7][8] | [directly_supported] Provider documentation calls it an open public service forecasting future solar and wind generation; official history places the map in service by 2021. [unverified] No immutable issue timestamp, lead-time contract, forecast schema, or archive interface is documented. | [unverified] Service existence during the interval does not establish retained 2022–2023 issued runs. | [unverified] The inspected provider pages state public viewing but no data-reuse licence, anonymous programmatic interface, or stable deployment terms. | [unverified] The documentation does not expose stable facility identifiers or an authoritative G1/G2/G3 crosswalk. | Reject |
| KPX Jeju annual renewable-output forecast catalogue [9] | [directly_supported] One-time Jeju annual summary for 2021, containing forecast and ex-post summary fields under an operating-plan basis. [contradicts_premise] It is not an hourly/day-ahead fixed-issue archive. | [contradicts_premise] Only the one-time 2021 summary is documented, not 2022–2023 runs. | [near_match_only] The file landing says download can be anonymous, but its licence field is blank; generated API use requires membership/application. | [contradicts_premise] Jeju aggregate only; no three-group crosswalk or disaggregation rule. | Reject |
| KPX regional hourly solar/wind generation catalogue [10] | [contradicts_premise] Metadata defines settlement-adjustable **electricity transactions**, not forecasts; it is ex-post operational information. Wind is only land/Jeju aggregated from late 2023 onward. | [contradicts_premise] Even if dates were present, ex-post records cannot satisfy fixed-issue forecast chronology. | [directly_supported] Landing metadata says unrestricted reuse and anonymous file download; this does not cure the information-class failure. | [contradicts_premise] Land/Jeju bins cannot directly produce three farm-group forecasts. | Reject |
| EWP day-ahead nationwide provincial solar forecast [11] | [directly_supported] Direct regional generation forecast and explicitly based on KMA's 14:00 short-range release. [contradicts_premise] It forecasts solar, not wind. [unverified] The page gives no proof that the derived output itself exists by exactly 14:00; computation necessarily follows its named 14:00 input release. | [unverified] The landing page does not attest complete preserved 2022–2023 issue coverage. | [unverified] It is exposed as an OpenAPI catalogue item, but the inspected metadata does not establish account-free calls or a commercial-use licence. | [contradicts_premise] Province-level solar output has no authoritative mapping or physical identity to the three wind groups. | Reject |

## Choice-free mapping audit

A permissible rule would have to be fully specified as `x_g(t) = F(R(g), issue<=D-1 14:00, t)` with an official, stable `R(g)` and no nearest-region, capacity-share, unit-conversion, or model choice. [derived] None of [1]–[11] supplies `R` for the competition groups.

| Farm group | Admissible official region/product mapping |
|---|---|
| G1 | ∅ — [unverified] no authoritative crosswalk in the admissible documentation |
| G2 | ∅ — [unverified] no authoritative crosswalk in the admissible documentation |
| G3 | ∅ — [unverified] no authoritative crosswalk in the admissible documentation |

[derived] Assigning the same Jeju/land/administrative aggregate to a group, selecting a map point, or distributing a regional total would introduce an analyst choice and therefore is not a zero-choice mapping.

## Verdict

**NO EXECUTABLE OR PREREQUISITE CANDIDATE.** [derived] The direct wind-generation near-matches fail the complete fixed-issue 2022–2023 archive, strict basis-time, licence/anonymous-deployment, and/or exact three-group mapping gates. The only documented strict regional issue product is generic meteorological wind output; the clearly reusable KPX regional series is ex-post; the exact 14:00 regional generation product is solar.

## Official sources (11)

1. KMA, “에너지기상 요약정보 서비스 개시” (2026-06-25): https://www.weather.go.kr/kma/news/press_01.jsp?mode=view&num=1194673
2. KMA, “햇빛과 바람 기후예측정보…” (2025-09-23): https://www.weather.go.kr/kma/news/press_01.jsp?mode=view&num=1194548
3. KMA Renewable Energy Weather Information Platform, platform description: https://energy.kma.go.kr/kmaem/portal/common/platformInfo.do
4. KMA copyright policy: https://www.kma.go.kr/kma/guide/copyright.jsp
5. KMITI / Environmental Big Data Platform, “머신러닝 기반 제주도 풍력 발전량 예측 자료” metadata: https://www.bigdata-environment.kr/user/data_market/detail.do?id=092186f0-a064-11ee-a443-a7e161ec5b2c
6. KMITI / Environmental Big Data Platform, “풍력 발전단지 풍속 및 발전량 예측 자료” metadata: https://www.bigdata-environment.kr/user/data_market/detail.do?id=b1ee36a0-e10c-11ee-a9a8-3be98374ecf3
7. 60Hertz, renewable distributed-resource service documentation: https://60hz.io/business/tvpp
8. 60Hertz, official history: https://60hz.io/history
9. KPX / Public Data Portal, “제주 연간 신재생에너지 출력 예측치” metadata: https://www.data.go.kr/data/15103255/fileData.do
10. KPX / Public Data Portal, “지역별 시간별 태양광 및 풍력 발전량” metadata: https://www.data.go.kr/data/15065269/fileData.do
11. EWP / Public Data Portal, “하루 전 전국 태양광 발전량 예측값 조회” metadata: https://www.data.go.kr/data/15144541/openapi.do
