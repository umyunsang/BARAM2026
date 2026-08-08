# M271 P0 — LangGraph 채택 게이트

- 판정일: 2026-08-04 (UTC)
- 런타임: Python 3.12.0 / macOS-26.5-arm64-arm-64bit
- 판정: **PASS**

승인된 계획 §3 의 네 조건과, M271 엔진이 실제로 의존하는 프리미티브를 함께 실증한다.
적격은 유용을 뜻하지 않는다. 이 문서는 채택 가능 여부만 판정한다.

## 판정 요약

| 항목 | 판정 |
|---|---|
| `1_licenses_permissive` | PASS |
| `2_versions_pinned` | PASS |
| `3_no_llm_state_and_conditional_edge` | PASS |
| `3b_no_outbound_network` | PASS |
| `4a_parallel_reducer_order_stable` | PASS |
| `4b_parallel_eq_sequential` | PASS |
| `4c_true_concurrency` | PASS |
| `4d_send_dynamic_fanout` | PASS |
| `4e_checkpoint_fork_revival` | PASS |

## 조건 1 — 라이선스

전이 폐포 전체가 허용적 라이선스다. 상업이용을 제한하는 항목은 없다.

| 패키지 | 라이선스 | 설치 버전 |
|---|---|---|
| `langgraph` | MIT | `1.2.10` |
| `langgraph-checkpoint` | MIT | `4.1.1` |
| `langgraph-checkpoint-sqlite` | MIT | `3.1.1` |
| `langgraph-prebuilt` | MIT | `1.1.0` |
| `langgraph-sdk` | MIT | `0.4.2` |
| `langchain-core` | MIT | `1.5.3` |
| `langsmith` | MIT | `0.10.15` |
| `pydantic` | MIT | `2.13.4` |
| `xxhash` | BSD-2-Clause | `3.8.1` |

## 조건 2 — 버전 고정

`pyproject.toml` 의 `graph` extra 에 **정확 핀**으로 고정했다. 다른 extra 는 범위를 쓰나
여기만 정확 핀인 이유는 조건 4 의 결정성 논거가 이 버전에서 *측정된* 행동에 의존하고,
그 행동이 문서화된 API 보증이 아니기 때문이다. 마이너 범프가 테스트를 깨뜨리지 않은 채
재현성만 조용히 무너뜨릴 수 있다.

- `langgraph==1.2.10`
- `langgraph-checkpoint-sqlite==3.1.1`

## 조건 3 — LLM 없이 동작

`StateGraph` + 채널 리듀서 + `add_conditional_edges` 만으로 동작한다. LLM 호출 없음.

`langchain-core` 가 `langsmith`(상용 트레이싱 클라이언트)를 끌어오므로 외부 전송 여부를
말이 아니라 실행으로 판정했다. `socket.socket.connect` 를 예외로 막고 그래프를 돌렸고,
연결 시도는 **0건**이었다.
트레이싱은 opt-in 이며 켜지 않았다.

## 조건 4 — 병렬 실행의 결정성

### 4a 리듀서 적용 순서

분기 sleep 을 등록 순서와 **역순**으로 주어 완료 순서를 인위로 뒤집었다.
그런데도 관측된 순서는 `['root', 'b1', 'b2', 'b3', 'b4', 'b5', 'join']` 로 **등록 순서**였다.
5회 모두 동일했다.

→ 병렬 분기의 쓰기는 완료 순서가 아니라 **노드 등록 순서**로 리듀스된다.
다만 이는 문서화된 보증이 아니라 이 버전에서 *측정된* 행동이다. 계획의 완화책
(순서 무관 리듀서 / 정규 정렬 / 결손 셀 쓰기 소유권)은 그대로 유지한다.

### 4b 병렬 ≡ 순차

- 병렬: `['root', 'b1', 'b2', 'b3', 'b4', 'b5', 'join']`
- 순차: `['root', 'b1', 'b2', 'b3', 'b4', 'b5', 'join']`
- 원본 동일: `True`
- 정규 다이제스트 동일: `True` (`c46fcfab880f12fb`)

### 4c 실제 동시성

순서만 안정적이고 실제로는 순차 실행이라면 원칙 3 이 성립하지 않으므로 벽시계로 판정했다.

- 순차 예산 `0.5s`
- sync `0.1128s` / async `0.1096s`
- 서로 다른 워커 스레드 **5개**: `['ThreadPoolExecutor-11_0', 'ThreadPoolExecutor-11_1', 'ThreadPoolExecutor-11_2', 'ThreadPoolExecutor-11_3', 'ThreadPoolExecutor-11_4']`

**운영 제약**: sync path is a ThreadPoolExecutor, so GIL-bound Python CPU work will not parallelise. Model workers must be budgeted as (per-node num_threads x concurrent nodes) <= 6 per AGENTS.md.

### 4d `Send` 동적 fan-out

라우터가 반환한 `Send` 개수만큼 분기가 생긴다(1개 → 1분기, 3개 → 3분기).
C6 이 몇 개 레인에서 발화할지는 실행 전에 알 수 없으므로 이 경로가 필수다.

### 4e 체크포인트 포크 = C9 부활

`get_state_history` 로 과거 체크포인트를 찾고 `update_state` 로 전제를 뒤집어 재실행하면
하위 경로가 새 전제로 다시 돈다. 계획의 C9(전제가 뒤집히면 폐기 하위그래프 부활)를
런타임이 네이티브로 제공한다는 뜻이다.

- 원래: `['a', 'b(NO_EXTERNAL_DATA)']`
- 전제 뒤집은 뒤: `['a', 'b(EXTERNAL_DATA_ALLOWED)']`

## 남는 제약

1. `StateGraph` 노드 집합은 컴파일 시점에 고정된다. 새 노드는 NetworkX 발굴 그래프에서
   생성되고 런타임에서는 `Send` 파라미터 인스턴스로 실행된다(계획 R4).
2. 리듀서 적용 순서의 결정성은 **측정된 행동이지 API 보증이 아니다**. 핀을 바꾸기 전에
   이 게이트를 다시 돌려야 한다.
3. 의존성이 늘었다. `uv.lock` 항목 **+37**, 프로젝트 venv 신규 설치 **35개**
   (venv 에 이미 있던 2개 차이). 기존 패키지 **버전 변경 0건, 제거 0건**이므로
   기존 산출물의 재현성은 유지된다. 예측 파이프라인 자체는 이 중 어느 것도 쓰지
   않으며 `graph` extra 로 격리했다.
