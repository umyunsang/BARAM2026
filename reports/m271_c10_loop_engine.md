# M271 C10 — 루프엔진 제자리 개선 (라우터가 지시한 노드)

노드 `C1N78_LOOP_ENGINE_C10` / 레인 L8 / 부모 `C1N76_CIRCULAR_BLOCK`

**내가 고른 노드가 아니다.** `m271_p4_route.py` 가 상태그래프를 통과시켰고 증거 세 건이 전부 C10 으로 갔다 (`stall_counter` 77 >= 3).

## 1. 방향 리서치 (실제 수행)

- ShinkaEvolve — novelty rejection-sampling, exploration-exploitation 균형 부모 선택, bandit 기반 선택. 150 표본으로 circle packing SOTA — <https://arxiv.org/abs/2509.19349> (`directly_supported`)
- 임베딩 코사인 유사도가 임계 초과면 제안 기각 — '사소한 변형' 낭비 차단 — <https://sakana.ai/shinka-evolve/> (`directly_supported`)
- AI4Research 서베이 — 획득함수가 기대개선을 불확실성 대비로 정규화 — <https://arxiv.org/pdf/2507.01903> (`near_match_only`)

## 2. 라우터 v3 개정

- **C16 크기 게이트** — `expected_gain < 0.15 x 남은격차` 면 방향을 추격하지 않는다. 획득함수 정규화의 최소형이다.
- **C17 신규성 기각** — 같은 레인 x 같은 kind 가 최근 6 개 중 3 회 이상이면 기각. ShinkaEvolve 의 유사도 기각을 노드 메타데이터로 옮긴 것.
- **원장 갱신 계약** — `stall_counter` 77 의 근본 원인. 연료계가 죽어 있었다.

남은 격차 **0.029690** (목표 0.66 - 챔피언 0.630310)

## 3. 이번 실패 재생 — v3 가 잡는가

| 증거 | 레인 | 기대이득 | 격차대비 | v3 판정 | v2 였다면 |
|---|---|---:|---:|---|---|
| C1N60 | L7 | 0.008990 | 0.303 | **C17** REJECT_DIRECTION | C10 RESEARCH_DIRECTION |
| C1N61 | L7 | 0.001169 | 0.039 | **C16** REJECT_DIRECTION | C10 RESEARCH_DIRECTION |
| C1N71 | L2 | 0.001238 | 0.042 | **C16** REJECT_DIRECTION | C10 RESEARCH_DIRECTION |
| C1N73 | L7 | 0.004931 | 0.166 | **C17** REJECT_DIRECTION | C10 RESEARCH_DIRECTION |

## 4. 타당성 가드

- V1 발화하지 않는 증거에서 v2 판정 보존 -> **True** (C1N68: C10 / RESEARCH_DIRECTION)
- V2 이번 실패를 잡는다 (4/4 차단) -> **True**
- V3 결정성 -> **True**

## 5. 사전확약

- H1 세 가드 전부 -> **True**
- H2 자릿수 맞는 방향(C1N68, 격차대비 0.57)은 통과 -> **True**
- H3 C1N73 이득비 0.166 vs 문턱 0.15 — 차단. 경계값을 결과를 보고 조정하지 않았다.

## 6. 판정

**ROUTER_V3_ADOPTED**

임계값은 **선언 관례**이며 보정된 바 없다(계획 R3 와 같은 지위). 후보가 통과하도록 조정하지 않았고, v2 는 지우지 않아 롤백 경로가 남는다.

digest `c4b3f610690325df`
