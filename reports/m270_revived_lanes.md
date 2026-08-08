# M270 P0 — 실험 그래프 부트스트랩 및 소급 재개방 판정

- 적재 노드: **133개** (결정표 + 진행로그에 문서화된 M-노드)
- 재개방(`revived`) 후보: **1개**
- 확대 역량: 신규 의존성 / 공개 외부데이터(조건부) / OSS 가중치(조건부)
- 유지 금지: 원격 API 추론, 비공개 데이터, 기준시점 이후 관측, 재분석·사후보정, 평가구간 정답

## 1. 상태 분포

| status | 개수 |
|---|---:|
| documented | 98 |
| rejected | 20 |
| promoted | 12 |
| built | 2 |
| predeclared | 1 |

## 2. 종결 전제 분포

| closure_premise | 개수 | 재개방 가능 |
|---|---:|---|
| `NO_SUBJECT_ROW` | 112 | 아니오 |
| `EVIDENCE_NEGATIVE` | 15 | 아니오 |
| `UNCLASSIFIED` | 4 | 아니오 |
| `NO_SELECTABLE_NWP` | 1 | 예 |
| `SAME_FOLD_ORACLE_ONLY` | 1 | 아니오 |

## 3. 재개방 후보 노드

| 노드 | 전제 | 매칭 문구 | 종결 기록(요약) |
|---|---|---|---|
| `M251` | `NO_SELECTABLE_NWP` | `multiple nwp configurations` | Close the ramp/weather-pattern lane without M251 |

## 4. 수동 검증 레인 판정

키워드 분류기는 1차 필터일 뿐이며, 아래는 원본 행을 직접 읽어 확인한 결과다.

| 레인 | 판정 | 기록된 종결 사유 | 근거 |
|---|---|---|---|
| ramp / weather-pattern classification | **REVIVED** | the published method selects among multiple NWP configurations, but the competition exposes only the supplied GFS/LDAPS forecasts | the blocker was the absence of alternative NWP sources, not negative evidence. Public forecast archives are now permitted, so the premise no longer holds. |
| TFT / NGBoost / deep probabilistic models | **UNOPENED** | execution forbidden; would require unauthorized dependency expansion | never executed even once, so there is no negative evidence to overturn. New dependencies are now permitted. This is an unexplored lane, not a revived one. |
| external public weather / multi-source NWP features | **UNOPENED** | external data excluded by project rule | the project rule was stricter than the competition rule. Public data is admissible subject to the availability-time and reproducibility capability gate (P2). |
| pretrained time-series foundation models | **UNOPENED** | pretrained weights excluded by project rule | admissible only for weights publicly released on or before 2026-07-05 under a commercial-use-permitting OSS license, loaded locally. Gated at P2. |
| group-3 missing 2022 labels | **STAYS_CLOSED** | no supplied source overlaps the missing 2022 group-3 labels | the gap is missing GENERATION LABELS, which no public source can supply and which private operational data may not supply either. External data can improve group-3 FEATURES, but that is a different lane and does not reopen this one. |
| contemporaneous SCADA as an inference feature | **STAYS_CLOSED** | SCADA is unavailable at inference time | a structural property of the task, unchanged by the widened scope. Private operational data remains forbidden and post-reference-time observation is forbidden. |

## 5. 판정 규율 및 한계

- 분류기는 **보수적**이다. 명시적 역량 차단 문구가 있을 때만 재개방으로 판정하고 나머지는 증거 기반 종결로 남긴다. 과소 재개방은 검토로 복구되지만 과대 재개방은 실제로 실패한 레인을 되살린다.
- 이 표는 **후보 목록이지 승격 목록이 아니다.** 각 노드는 P2 역량 게이트를 통과한 역량에 한해, 그리고 원래 종결 사유가 정말 그 역량에만 의존했는지 개별 확인한 뒤에만 실행된다.
- `UNCLASSIFIED`와 `documented` 상태 노드는 기록이 진행로그에만 있거나 판정 문구가 없는 경우입니다. 이후 라운드에서 개별 확인 대상입니다.
- 원본 무결성: `task_plan.md` `7bd4dedd8f6b2056...`, `progress.md` `281f4836198ab61c...`
