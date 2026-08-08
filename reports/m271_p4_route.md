# M271 P4 — 엔진 가동: 라우터가 정한 다음 노드

라우터 `M271_ROUTER_v3_frozen_2026-08-06` / 증거 7 건 / **실제 `m271_app.py` 실행**

## 1. 라우터 상태 (그래프·원장에서 계산)

- 레인별 LIVE: `{'L1': 4, 'L2': 8, 'L3': 6, 'L4': 4, 'L5': 2, 'L6': 8, 'L7': 6, 'L8': 31}`
- 결손 질량: `{'L8': 0.04875}`
- 미설명 잔여질량 **0.02509** (임계 0.010)
- **정체 카운터 97** (한계 3) — 원장 이력 20 건에서 **계산**된 값이다
- 원장 갱신 필요: **False** / 미기록 사이클 **81** 건
- 뒤집힌 전제: `없음`

## 2. 판정

| 증거 | 레인 | 조건 | 행동 | 리서치 종류 | 범위 |
|---|---|---|---|---|---|
| C1N68 | L6 | **C7** | RESEARCH_DIRECTION | scale_up | 확인된 방향의 고도화 |
| C1N73 | L7 | **TERMINATION_GUARD** | HALT | - | - |
| C1N76 | L4 | **C16** | REJECT_DIRECTION | - | - |
| C1N87 | L1 | **C16** | REJECT_DIRECTION | - | - |
| C1N84 | L6 | **C16** | REJECT_DIRECTION | - | - |
| C1N80 | L3 | **C16** | REJECT_DIRECTION | - | - |
| C1N77 | L3 | **C16** | REJECT_DIRECTION | - | - |

## 3. 이유

- **C1N68** (C7 / RESEARCH_DIRECTION) — 확인됨 & 잔여 질량 있음 — 같은 방향 고도화
  - 함께 발화한 조건: `C10_SUPPRESSED_ALREADY_SERVICED`
- **C1N73** (TERMINATION_GUARD / HALT) — 정체가 지속되고 C10 을 이미 1 회 거쳤는데 다른 조건이 하나도 발화하지 않는다. 계획서 종료 가드 사다리의 종료 조건이다.
  - 함께 발화한 조건: `C10_ALREADY_SERVICED, NO_OTHER_CONDITION`
- **C1N76** (C16 / REJECT_DIRECTION) — C16 크기 게이트 (이득 0.00000 < 0.15 x 격차 0.02969)
  - 함께 발화한 조건: `C10_SUPPRESSED_ALREADY_SERVICED, C16`
- **C1N87** (C16 / REJECT_DIRECTION) — C16 크기 게이트 (이득 0.00119 < 0.15 x 격차 0.02969)
  - 함께 발화한 조건: `C10_SUPPRESSED_ALREADY_SERVICED, C16`
- **C1N84** (C16 / REJECT_DIRECTION) — C16 크기 게이트 (이득 0.00016 < 0.15 x 격차 0.02969)
  - 함께 발화한 조건: `C10_SUPPRESSED_ALREADY_SERVICED, C16`
- **C1N80** (C16 / REJECT_DIRECTION) — C16 크기 게이트 (이득 0.00248 < 0.15 x 격차 0.02969)
  - 함께 발화한 조건: `C10_SUPPRESSED_ALREADY_SERVICED, C16`
- **C1N77** (C16 / REJECT_DIRECTION) — C16 크기 게이트 (이득 0.00092 < 0.15 x 격차 0.02969)
  - 함께 발화한 조건: `C10_SUPPRESSED_ALREADY_SERVICED, C16`

## 4. 엔진이 연 리서치 노드

- C1N68: `dir::C1N68_EMPIRICAL_DECOMPOSITION::loop_engine::L8`
- C1N73: `dir::C1N73_GROUP_BLEND_GATE::loop_engine::L8`
- C1N76: `dir::C1N76_CIRCULAR_BLOCK::loop_engine::L8`
- C1N87: `dir::C1N87_CURTAILMENT_CLEAN_TARGET::loop_engine::L8`
- C1N84: `dir::C1N84_TEACHER_CHRONOLOGICAL::loop_engine::L8`
- C1N80: `dir::C1N80_FUSION_ANOMALY::loop_engine::L8`
- C1N77: `dir::C1N77_PER_SOURCE_STACK::loop_engine::L8`

이 표는 **내가 고른 것이 아니다.** 동결된 라우터 표가 receipt 에서 뽑은 증거 서명을 읽고 낸 판정이며, 상태그래프를 실제로 통과시킨 결과다.

