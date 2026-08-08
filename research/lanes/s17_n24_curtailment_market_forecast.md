# S17-N24 — curtailment / market-forecast intake

## 0. Contract and verdict

**Lane:** `S17-N24_SECONDARY_FRONTIER_RESEARCH_INTAKE / CURTAILMENT_MARKET_FORECAST`  
**Retrieval window:** 2026-08-09 00:17–00:28 KST (under 12 minutes).  
**Evidence bound:** 10 unique primary official Korean provider/legal-documentation URLs.

Only documentation landing pages, board-list metadata, catalogue metadata, and an official historical
market-rule clause were read. No attachment, file, API, Swagger page, catalogue endpoint, post row
body, or data row was requested; no account/access application was made. No observation, label,
`actual_kwh`, test-period value, fit, prediction, score, or remote inference was accessed.

Evidence tags follow the predeclaration:

- `[directly_supported]` — literal provider/legal documentation or list metadata.
- `[derived]` — a gate decision from directly supported premises.
- `[near_match_only]` — some structural gates pass, but at least one mandatory gate fails.
- `[contradicts_premise]` — the official documentation expressly defeats a required premise.
- `[unverified]` — the documentation-only lane cannot establish the fact; it receives no positive credit.

## Verdict: **`NO_READY_CURTAILMENT_MARKET_FORECAST`** [derived]

Two genuinely planned sources do have public 2022–2023 archive metadata and basis-safe nominal
publication schedules: weekly generator preventive-maintenance plans and monthly transmission/
substation outage plans. [directly_supported] Neither source supplies an exact electrical mapping
from its listed units/facilities to the competition wind farm's three KPX groups, a source-prescribed
impact formula, or a populated commercial-use licence in the reviewed metadata. [unverified]
Therefore neither can be converted into an executable feature without inventing topology, weights,
aggregation, and usage rights. [derived]

The closest literal curtailment source, KPX's non-central curtailment board, is a conditional D-1
forecast channel, but its public archive begins only after 2023 and its metadata is regional rather
than farm/group keyed. [contradicts_premise] Plant-specific day-ahead plans and transmission-
constraint reasons exist under the market rules, but the 2022–2023 rule delivers them to relevant
market participants at 17:00 KST (or later in exceptional cases), after the fixed 14:00 KST basis,
and permits withholding grid-vulnerability information. [contradicts_premise] The public
SMP/load-forecast API refreshes around 23:00 KST, is application based, and likewise misses the
basis/anonymity gates. [contradicts_premise]

**Selected executable or acquisition-prerequisite candidates: `0`.** [derived] No formula, download,
or root execution handoff is authorized. [derived]

## 1. Gate matrix

| Candidate | Genuinely forecast/planned? | 2022–2023 archive | Available by D-1 14:00 KST | Anonymous + commercial deployment | Exact farm/group map + zero-choice formula | Decision |
|---|---|---|---|---|---|---|
| KPX non-central curtailment notices | Yes when KPX expects curtailment; otherwise no forecast notice; unexpected operation may appear only as an actual notice | Public list starts after 2023 | Page explicitly describes a D-1 forecast convention, but no fixed posting-time guarantee was found | Landing/list page is anonymously visible; attachment/post transport and licence were not established | Titles/metadata distinguish Jeju vs mainland, not this farm or KPX groups 1–3 | **REJECT** |
| KPX weekly generator preventive-maintenance plan | Yes, next-week plan | Weekly list metadata exists in both years | Official disclosure schedule says every Friday | List is public; attachment acquisition was forbidden and commercial licence was not established | No documented causal/topological allocation from an offline generator to this wind farm; no source-prescribed aggregation | **NEAR MATCH / REJECT** |
| KPX monthly transmission/substation outage plan | Yes, planned outage | Monthly list metadata exists in both years | Official schedule says the 25th of the preceding month | List is public; specific public-data metadata leaves the licence field unpopulated | Metadata has facility/work/timing fields but no network topology, contingency effect, curtailment quantity, farm identifier, or group mapping | **NEAR MATCH / REJECT** |
| Participant day-ahead generation/constraint plan | Yes; rule includes plant plans, exclusion reasons, and a transmission-constraint review | Rule operated during the target years | **Fails:** relevant result is notified at 17:00 KST, with exceptional delay possible | Delivered to relevant market parties, not an anonymous public archive; sensitive grid information may be withheld | Potentially plant-specific, but both time and access gates already fail; no competition-group crosswalk is documented | **REJECT** |
| Public hourly SMP + day-ahead load forecast API | Yes, system market/load information | 2022–2023 fixed-issue archive was not established | **Fails:** metadata says once-daily refresh around 23:00 KST | OpenAPI utilization is application/approval based, not anonymous | Mainland/Jeju system series only; no farm/group mapping or curtailment transform | **REJECT** |
| KPX weekly demand outlook / annual fault statistics | Weekly outlook is prospective; fault statistics are retrospective | Disclosure schedule exists; row archive/schema not audited here | Weekly outlook is nominally prior-Friday; annual statistics are published the following April | Licence and attachment reproducibility not established | No source-supported site/group transform | **REJECT / EX-POST SEPARATED** |

All decisions in the last column are `[derived]`; literal cells are supported by the evidence ledger
below unless explicitly marked unverified in the detailed audit.

## 2. Candidate audits

### 2.1 KPX non-central curtailment board — literal D-1 forecast, wrong archive and granularity

The board note says that when D-1 forecasting indicates no curtailment, KPX posts no separate
forecast notice; if weather changes cause unexpected curtailment, only an operation-result notice
may be posted. [directly_supported] Thus this is a genuinely prospective but **positive-only,
conditional publication channel**, not a complete daily binary series. [derived] Treating “no post”
as zero would additionally require a complete issue-time snapshot of the board for every day, which
this lane did not establish. [unverified]

The last archive-list page begins entirely after the end of 2023; its first titles concern Jeju
forecast/result notices. [directly_supported] Consequently there is no public 2022–2023 forecast
archive on this board from which a train-symmetric feature can be constructed. [contradicts_premise]
The reviewed list metadata identifies only `제주` or `육지`, not 태백가덕산풍력 and not KPX groups
1, 2, and 3. [directly_supported] A mainland-wide flag broadcast to all groups would be an analyst-
chosen spatial rule rather than an exact provider mapping. [derived]

“실적” posts are explicitly operation results. [directly_supported] They are ex-post observations,
not forecasts, and were neither opened nor credited. [contradicts_premise]

### 2.2 Weekly generator preventive maintenance — archive and chronology pass, mechanism does not

KPX labels this source a generator-by-generator **preventive-maintenance plan (weekly)**, and its
statutory-disclosure table gives a daily-granularity publication every Friday. [directly_supported]
List-only archive pages show weekly plans during both 2022 and 2023, with the issue date preceding
the named following week in the ordinary entries. [directly_supported] This clears plan-versus-
actual and nominal basis-time coverage at documentation level. [near_match_only]

No attachment was opened, so exact historical row schema, revisions, missing weeks, and stable
machine parsing remain unverified. [unverified] More importantly, reviewed documentation supplies
no rule mapping the maintenance of a conventional or other listed generator to curtailment at this
wind farm, and no map to KPX groups 1–3. [unverified] Even a plausible system scalar such as “planned
offline capacity active in hour” would require attachment fields plus an analyst-selected sum,
normalization, overlap rule, and broadcast across groups. [derived] Those degrees of freedom violate
the exact-formula gate. [derived]

### 2.3 Monthly transmission/substation outage plan — strongest archive near-match, but topology absent

KPX's disclosure table labels this source a **transmission and substation equipment outage plan**,
monthly, published on the 25th of the preceding month. [directly_supported] A single list page shows
contiguous metadata entries covering part of 2022 and part of 2023, including a January plan posted
in the preceding December. [directly_supported] This is genuine planned grid information and is
comfortably older than the D-1 basis. [near_match_only]

The official Public Data Portal metadata describes monthly 765/345/154 kV transmission/substation
plans and names fields for first-level office, office, facility, construction name/summary,
requested time, adjusted time, and work type. [directly_supported] It does **not** document bus/line
connectivity, contingency transfer limits, affected generators, expected curtailment magnitude,
probability, or a crosswalk to 태백가덕산풍력 and its three KPX groups. [unverified] The same metadata
renders the `이용허락범위` heading without a populated value, so commercial-use compatibility is not
established. [unverified]

A candidate would require a provider-supported coefficient or incidence map in

\[
x_{g,t}=\sum_j M_{g,j}\,\mathbf 1\{t\in[\text{adjusted-start}_j,\text{adjusted-end}_j)\},
\]

where `j` is a listed outage and `g` is a competition KPX group. The reviewed source supplies neither
`M_{g,j}` nor a rule for deriving it. [unverified] Setting it by facility-name matching, geographic
proximity, assumed substation ownership, equal broadcast, or hand-built topology would create an
unsupported formula. [derived] The lane therefore emits **no** `x_{g,t}` implementation. [derived]

### 2.4 Day-ahead plans and constraint reasons — real signal, legally too late and non-public

The historical market-rule page contains the operative 2022–2023 versions. It defines the day-ahead
plan as cost-minimizing and constraint-aware, including a transmission-constraint review. It says
the relevant participant notification includes the generator's day-ahead plan, reserve plan,
price-exclusion reason (including transmission constraint), and that review. [directly_supported]
This is the only reviewed source that is potentially plant-specific and coefficient-complete enough
to describe an operational action. [near_match_only]

The same rule says the result is ordinarily notified to the relevant generator/transmission party
by **17:00 KST on D-1**, not by 14:00, with later exceptional deadlines possible. [directly_supported]
It also allows non-disclosure where business interests or grid-vulnerability/national-safety
information is implicated. [directly_supported] Therefore this channel fails strict basis time,
anonymous access, and reproducible public-archive symmetry simultaneously. [contradicts_premise]
No participant login, market system, or private notice was accessed. [directly_supported]

### 2.5 Public SMP/load forecast API — system-level and after cutoff

The official metadata says the API provides hourly mainland/Jeju SMP and demand forecasts, updates
once per day around **23:00 KST**, and is an OpenAPI utilization/application product. [directly_supported]
That refresh is nine hours after the fixed D-1 14:00 basis. [derived] The reviewed metadata does not
establish a fixed-issue 2022–2023 archive available at each historical basis, and it has no wind-farm
or KPX-group key. [unverified] Application/approval access is not anonymous deployment. [contradicts_premise]
No API or specification endpoint was called. [directly_supported]

## 3. Exact mapping/formula fail-closed check

An admissible source must provide all of the following without an analyst choice:

1. an issue/publication timestamp no later than D-1 14:00 KST for every operating day;  
2. an archived 2022–2023 record preserving the information available at that issue;  
3. an identifier crosswalk to 태백가덕산풍력 **and** a deterministic allocation to KPX groups 1–3;  
4. a literal quantity or event definition and one exact hourly transform;  
5. anonymous reproducible transport and explicit commercial-use permission.

No reviewed source satisfies all five. [derived] In particular, a shared mainland flag is not an
exact group mapping, a facility name is not an electrical-impact coefficient, a maintenance list is
not a curtailment forecast, and an operation-result notice is not basis-time information. [derived]
Because missing mapping and formula terms cannot be promoted from `[unverified]` or
`[near_match_only]`, the mandated fail-closed result is zero candidates. [derived]

## 4. Primary evidence ledger (10 unique URLs)

1. **KPX non-central curtailment list and publication convention** —
   https://www.kpx.or.kr/menu.es?mid=a10109050000  
   Conditional D-1/no-notice and actual-only exception language. `[directly_supported]`
2. **KPX oldest curtailment-board list page** —
   https://www.kpx.or.kr/board.es?mid=a10109050000&bid=0216&nPage=35  
   First public-list entries are post-2023 and Jeju forecast/result titled. `[directly_supported]`
3. **KPX statutory/advance disclosure schedule** —
   https://www.kpx.or.kr/menu.es?mid=a10109030000  
   Weekly demand, weekly maintenance, monthly outage, and annual fault-statistic publication timing. `[directly_supported]`
4. **KPX generator preventive-maintenance archive landing page** —
   https://www.kpx.or.kr/menu.es?mid=a10109030500  
   Identifies the continuing weekly plan archive. `[directly_supported]`
5. **KPX 2023 preventive-maintenance list metadata** —
   https://www.kpx.or.kr/board.es?mid=a10109030500&bid=0018&nPage=18  
   Weekly 2023 plan titles and registration dates only. `[directly_supported]`
6. **KPX 2022 preventive-maintenance list metadata** —
   https://www.kpx.or.kr/board.es?mid=a10109030500&bid=0018&nPage=22  
   Weekly 2022 plan titles and registration dates only. `[directly_supported]`
7. **KPX 2022–2023 transmission/substation outage-plan list metadata** —
   https://www.kpx.or.kr/board.es?mid=a10109030600&bid=0019&nPage=5  
   Monthly plan titles and issue dates spanning both years. `[directly_supported]`
8. **Public Data Portal metadata for the KPX transmission/substation outage plan** —
   https://www.data.go.kr/data/15046164/fileData.do  
   Voltage scope and metadata field inventory; no populated licence value in the retrieved page. `[directly_supported]` / `[unverified]`
9. **Public Data Portal metadata for KPX hourly SMP/day-ahead demand forecast** —
   https://www.data.go.kr/data/15131225/openapi.do  
   System-level scope, approximately 23:00 daily refresh, and application/approval service metadata. `[directly_supported]`
10. **KPX historical market-rule clause covering 2022–2023 day-ahead plans** —
    https://marketrule.kpx.or.kr/lmxsrv/law/joHistoryContent.do?SEQ=2&SEQ_CONTENTS=4983&DATE_START=20210501&DATE_END=20231230  
    Constraint-aware plan contents, participant notification deadline, and disclosure exceptions. `[directly_supported]`

No evidence credit is assigned to search snippets, third-party articles, or uncited pages. [derived]
