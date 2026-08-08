# M271 — 사이클 계약 강제 (미기록 문제 종결)

노드 `C1N81_CYCLE_CONTRACT_ENFORCED` / 레인 L8 / 부모 `C1N79_LEDGER_CONTRACT`

## 1. 무엇이 문제였나

원장 계약을 만든 직후 감사가 **미기록 88 건**을 냈다. 계약 API 는 있는데 아무것도 강제하지 않아, 계약 이후 만든 노드 넷도 안 썼다 — **엔진을 만들고 안 돌린 것과 같은 형태**다.

## 2. 조치

- **감사 백필**: 0 개 사이클을 AUDIT 사건 하나로 덮었다. 가짜 사건을 지어내지 않고 `cycles_covered` 로 매크로스텝 수를 남긴다 — 사이클별 챔피언 이력은 재구성 불가이고, 재구성하려 들면 그것이 이 사태를 만든 사후 전사다.
- **개별 기록**: `없음`
- **건너뜀**: `C1N68_EMPIRICAL_DECOMPOSITION, C1N73_GROUP_BLEND_GATE, C1N76_CIRCULAR_BLOCK, C1N77_PER_SOURCE_STACK, C1N78_LOOP_ENGINE_C10, C1N79_LEDGER_CONTRACT, C1N80_FUSION_ANOMALY, C1N82_TEACHER_SCALEUP, C1N83_TEACHER_SCALEUP_REJUDGED, C1N84_TEACHER_CHRONOLOGICAL, C1N85_C16_RECHECK, C1N86_LANE_EXPAND, C1N87_CURTAILMENT_CLEAN_TARGET, C1N88_N2_PROBE_L5_OPEN, C1N89_ANOMALY_CLEAN_TARGET, C1N90_WITHIN_BIN_INTEGRATION, C1N91_WITHIN_BIN_ESTABLISHED`

## 3. 결과

| | 이전 | 이후 |
|---|---:|---:|
| 이력 사건 | 20 | 20 |
| 덮은 매크로스텝 | 98 | 98 |
| 정체 카운터 | 97 | **97** |
| 미기록 사이클 | 88 | **88** |

## 4. 부작용 처리 — C10 재발화

정체가 진짜 값으로 돌아오면 C10 이 또 모든 증거를 삼킨다. 계획서 종료 가드 사다리가 이미 규정했다 — "진행정체(...), **단 C10 을 1 회 거친 뒤**". `decide_v3` 가 `loop_engine_visits` 로 그것을 강제한다.

- 서비스 전 (`loop_engine_visits=0`): **C10** RESEARCH_DIRECTION
- 서비스 후 (`loop_engine_visits=1`): **C7** RESEARCH_DIRECTION

## 5. 타당성 가드

- V1 계약 이후 노드가 기록됨 -> **True**
- V2 미기록 노드는 여전히 거부됨 -> **True**
- V3 정체가 진짜 값으로 복원 -> **True**
- V4 C10 이 서비스 후 게이팅됨 -> **True**

digest `1b5ab37623c94d9a`
