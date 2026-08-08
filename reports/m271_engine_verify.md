# M271 P2 — 엔진 검증

- 판정일: 2026-08-05 (UTC)
- 라우터 버전: `M271_ROUTER_v2_frozen_2026-08-04`
- 판정: **PASS**

승인된 계획 §7 항목을 실행으로 판정한다. 말로 통과시키지 않는다.

| 항목 | 판정 |
|---|---|
| `1_router_deterministic` | PASS |
| `2_reducers_order_insensitive_and_guarded` | PASS |
| `3_gate_signature_branches` | PASS |
| `4_dynamic_fanout_follows_state` | PASS |
| `5_graph_digest_reproducible` | PASS |
| `6_langgraph_app_routes_the_decided_target` | PASS |
| `7_premise_flip_revives_subgraph` | PASS |
| `8_ledger_is_generative` | PASS |
| `9_two_stage_research_separation` | PASS |
| `11_prune_absorbing_and_multiparent` | PASS |
| `13_frontier_refills_from_deficit` | PASS |
| `14_gate_signature_normalised` | PASS |
| `15_research_guards_enforced` | PASS |
| `16_materialize_creates_lineage` | PASS |
| `17_candidate_space_is_open` | PASS |

## 핵심 판정

### 조건 분기 (계획의 요점)

같은 게이트 실패라도 서명이 다르면 다른 리서치로 가야 한다. 일괄 리서치는 이 구분을
원리적으로 할 수 없다.

| 증거 서명 | 발화 조건 | 리서치 종류 |
|---|---|---|
| G1 실패 & G3 통과 | `C3` | `regime_conditional` |
| G3 실패 & G1 통과 | `C4` | `amplify` |

### 폐기의 흡수성

- 폐기된 노드: `['child']`
- `expand()` 예외 거부: **True**
- 다른 LIVE 조상으로 살아남은 후손: **True**

우선순위를 낮추는 게 아니라 금지다. 다만 폐기 사유가 기계검증 술어로 남으므로
전제가 뒤집히면 C9 이 되살린다.

### 프론티어가 비지 않음 (직전 세션 실패 모드의 역)

- 전 노드 폐기 후 프론티어: `[]`
- 남은 미설명 결손 질량: `0.371400`
- 발화한 조건: `C6` -> `SEND_FANOUT`

직전 세션은 하드코딩 4-튜플이 소진되면 멈췄다. 여기서는 결손 질량이 남는 한 조건이
발화해 새 노드를 낳는다.

### 원장의 생성성

- 셀 108 -> 109 (축 추가)
- 손실 질량 보존: **True**
- 미설명 셀: 109

축을 추가하면 새 셀이 생기고 전부 `UNEXPLAINED` 로 태어나므로 C1 발화 대상이 늘어난다.
손실 질량은 보존되므로 A7 이 확인한 가법 항등식이 유지된다.

## 동결 라우터 표

| 코드 | 우선순위 | 동작 | 리서치 종류 | 조건 |
|---|---:|---|---|---|
| `C1` | 1 | `RESEARCH_DIRECTION` | `explain` | 기전이 없는 결손 셀 |
| `C2` | 2 | `RESEARCH_DIRECTION` | `replace_mechanism` | 증거가 기전을 반증 |
| `C3` | 3 | `RESEARCH_DIRECTION` | `regime_conditional` | G1 실패 & G3 통과 — 효과 있으나 비일관 |
| `C4` | 4 | `RESEARCH_DIRECTION` | `amplify` | G3 실패 & G1 통과 — 일관하나 미미 |
| `C5` | 5 | `RESEARCH_DIRECTION` | `anomaly` | 사전확약과 부호가 반대 |
| `C6` | 6 | `SEND_FANOUT` | `lane_expand` | 레인 기아 — 생존가설 0 이고 결손 질량이 임계 초과 |
| `C7` | 7 | `RESEARCH_DIRECTION` | `scale_up` | 확인됨 & 잔여 질량 있음 — 같은 방향 고도화 |
| `C8` | 8 | `PRUNE` | - | 정보량 미달 — 굴착지점 종결. 리서치 없음 |
| `C9` | 9 | `REVIVE` | - | 전제가 뒤집힘 — 체크포인트 포크로 하위그래프 부활 |
| `C10` | 10 | `RESEARCH_DIRECTION` | `loop_engine` | 루프 정체 — 라우터 표 자체의 SOTA 를 조사해 엔진을 제자리 개선 |
| `C12` | 12 | `REFINE_AXIS` | - | 새 기전이 **손실의 분할 축** — 결손 축 재분해 |
| `C13` | 13 | `RESEARCH_DIRECTION` | `structural_consequence` | 새 기전이 구조적 사실 — 분할이 아니라 그 결과를 조사 |
| `C14` | 14 | `CLOSE_AXIS` | - | 확인된 부정 — 그 축을 닫는다. C8(정보 못 냄)과 다르다 |
| `C15` | 15 | `SEND_FANOUT` | `explain` | 원장이 세워짐 — 상위 미설명 셀에 기전 리서치를 뿌린다 |

## 남는 한계

1. 임계값 `TAU_DEFICIT_MASS`·`EPSILON_INFORMATION`·`STALL_LIMIT` 은 **선언 관례**이지
   보정된 값이 아니다. 이 문제에서 held-out 으로 적합한 바 없다(계획 R3).
2. C11 의 가치 함수도 선언 관례다. 기대이득·정보가치·탐험항의 상대 비중에 근거가 없다.
3. 병렬 리듀서 적용 순서의 결정성은 P0 이 측정한 행동이지 API 보증이 아니다. 그래서
   리듀서를 순서 무관하게 쓰고 셀 쓰기 소유권을 강제한다.
4. 이 검증은 엔진이 **계획대로 배선되었는지**만 판정한다. 라우팅이 좋은 리서치를
   고르는지는 실제 사이클을 돌려야 알 수 있다.
