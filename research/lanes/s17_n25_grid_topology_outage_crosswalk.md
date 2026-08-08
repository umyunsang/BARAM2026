# S17-N25 — GRID_TOPOLOGY_OUTAGE_CROSSWALK

## Verdict

**FAIL-CLOSED — no executable candidate or bounded prerequisite.** Within the permitted official, public, metadata/documentation-only surface, there is no literal electrical join from each of the three KPX groups to the N24 outage-facility keys, and no source-prescribed coefficient-free rule converting a listed facility outage into an hourly group impact. A place-name, proximity, shared operator, phase, substation-name, voltage, or project-name match would be **[derived]** and is not promoted.

## Evidence

1. **Licensed/project identities do not supply an electrical crosswalk.**
   - The Ministry's Gadeoksan transfer notice identifies the `태백 가덕산풍력 발전사업`, but its public HTML contains no connection substation, line, bay/breaker, one-line topology, or KPX-group identifier. **[directly_supported]**  
     https://www.motir.go.kr/kor/article/ATCLc01b2801b/65081/view
   - The separate Ministry notice identifies the `태백 원동풍력 발전사업`; it likewise contains no electrical connection or group mapping. Equating this licence name with a particular anonymised group is not prescribed by the notice. **[directly_supported]** / **[unverified]**  
     https://www.motie.go.kr/kor/article/ATCLc01b2801b/66556/view
   - Taebaek City's phase-one and phase-two HTML releases establish the two project phases and their turbine/farm descriptions, but neither contains the three KPX group literals nor a turbine/phase-to-substation/line/bay mapping. Phase membership therefore cannot stand in for electrical topology. **[directly_supported]** / **[near_match_only]**  
     https://www.taebaek.go.kr/www/selectBbsNttView.do?key=359&bbsNo=31&nttNo=116647  
     https://www.taebaek.go.kr/www/selectBbsNttView.do?key=359&bbsNo=31&nttNo=139532

2. **The public outage and planning surfaces do not close the join.**
   - KPX's monthly page is only metadata for a transmission/substation outage-plan attachment; the inline page gives no farm/group connectivity. The attachment body was not opened under this lane's bound. **[near_match_only]**  
     https://new.kpx.or.kr/board.es?mid=a10109030600&bid=0019&act=view&list_no=45976
   - The KPX archive index establishes that monthly transmission/substation plans are published, but provides titles/dates rather than a generator-to-facility topology. **[directly_supported]**  
     https://new.kpx.or.kr/menu.es?mid=a10109030600
   - KPX's public grid-acceptance map explicitly describes itself as regional/substation planning reference, masks substation names, warns that later plans and actual reinforcement dates may differ, and says actual connectability depends on licence review and lower-voltage network conditions. It is not an as-built one-line map or plant/group crosswalk. **[contradicts_premise]**  
     https://www.kpx.or.kr/menu.es?mid=a10403090000
   - The permitted intake does not enumerate the N24 facility keys. Under the instruction to read no other local input, an asset-by-asset literal join cannot be independently reconstructed here. **[unverified]**

3. **Official rules do not prescribe a deterministic hourly coefficient.**
   - The historical KPX market-rule HTML (through the cited history) says in §§5.9.2–5.9.5 that outage plans are reviewed and adjusted to minimise expected grid constraints, may be changed before work, may be rescheduled when operating conditions change, and may be postponed or cancelled. Thus a planned facility outage is not, by rule, a prescribed `all groups off`, `one group off`, or fixed-fraction event. **[contradicts_premise]**
   - The same rule's §§5.1.1–5.1.2 places transmission constraints inside the KPX day-ahead generation-plan process; the resulting plan/constraint notice is sent to the relevant market participants by D-1 17:00 in the ordinary case, not made anonymously public by the required D-1 14:00 basis time, and disclosure may exclude commercially sensitive or grid-vulnerability information. **[contradicts_premise]**  
     https://marketrule.kpx.or.kr/lmxsrv/law/lawFullContent.do?SEQ=2&SEQ_HISTORY=33
   - KPX's public page for the outage-management standard states that it is a detailed standard delegated by the market rules, but exposes no inline farm-specific impact formula; its document attachment was not opened. **[directly_supported]**  
     https://www.kpx.or.kr/board.es?mid=a10502000000&bid=0045&act=view&list_no=51654

## Requirement gate

| Required literal item | Status |
|---|---|
| Exact Gadeoksan/phase identity | Partial, official **[directly_supported]** |
| Each of three KPX groups → physical generating subset | Missing **[unverified]** |
| Each group → every N24 outage facility by electrical topology | Missing; geography/name substitution forbidden **[derived]** |
| Source-prescribed coefficient-free hourly impact | Missing; official process is conditional and adjustable **[contradicts_premise]** |
| Anonymous availability by D-1 14:00 KST | Not established; operative participant notice is later **[contradicts_premise]** |
| Commercially reproducible implementation | Not established **[unverified]** |

**Final disposition:** do not promote an outage feature, zeroing rule, fractional derate, or topology prerequisite from this lane.
