# M271 — 결손 원장 갱신 계약

노드 `C1N79_LEDGER_CONTRACT` / 레인 L8 / 부모 `C1N78_LOOP_ENGINE_C10` / **C10 이 지목한 근본 결함의 수정**

## 1. 무엇이 고장나 있었나

- A7 이 정책 `T0.5_G1.5` **0.6286045208903154** 로 원장을 세웠고 현 챔피언은 **0.63031** 이다. 챔피언이 바뀌어도 원장을 다시 계산하지 않으므로 회수가능질량이 **원리적으로 움직일 수 없었다**.
- 셀의 `status`/`mechanism`/`owner` 가 전부 초기값이라 C1(미설명 셀)이 영원히 발화하고 설명된 질량이 보이지 않았다.
- 그래서 `stall_counter` 가 **1 -> 이제 1** 로 **이력에서 계산**된다. 상수 주장이 아니다.

## 2. 계약

- 이력 `artifacts/registry/m271_ledger_history.json` 에 append-only 기록
- 셀 귀속에 **소유자 필수** — 없으면 `ContractViolation`
- `stall_counter` 는 질량 무감소 연속 기록 수로 **계산**. 한 번이라도 줄면 0
- `refresh_due` 가 원장 기준 점수와 현 챔피언 불일치를 노출
- `unrecorded_cycles` 가 `CYCLES` 대비 미기록 노드를 감사

## 3. 백필 (receipt 가 셀 단위로 지지하는 것만)

- 기준선: 이미 존재
- C1N63 천장 귀속: **0 셀**
- 천장 도달(회수 불가) 셀: **0**
- 원시 회수가능질량 0.04875 -> 천장 제외 후 **0.04875**

**77 개 사이클을 사후에 셀에 배정하지 않았다.** 그것이 이 사태를 만든 사후 전사 습관이다. C1N63 만이 108 셀 각각에 실현 단위와 물리 천장을 계산했으므로 유일하게 백필 가능하다.

## 4. 감사

- 이력 사건 **2** 건
- 귀속된 셀 **19/108**
- **미기록 사이클 76** 건 (예: `C1N10_BLEND_LOCALITY, C1N11_V2_REPRESENTATION, C1N12_G3_HISTORY, C1N13_ENSEMBLE_VARIANCE, C1N14_SHRINKBLEND, C1N15_WALKFORWARD, C1N16_ROBUSTNESS_AUDIT, C1N17_METRIC_ALIGNED_COMBINER`)
- 갱신 필요 **True**

## 5. 타당성 가드

- V1 이력 적용이 멱등 -> **True**
- V2 천장 셀이 질량에서 제외됨 -> **True**
- V3 `stall_counter` 가 이력에서 계산됨 -> **True**

## 6. 남는 것

refresh_due 를 자동으로 해소하지 않는다. 원장을 현 챔피언으로 다시 계산하려면 A7 분해를 챔피언 예측 위에서 재실행해야 하고 그것은 별도 노드의 일이다. 여기서는 '원장이 낡았다' 를 라우터가 볼 수 있게 만드는 데까지 한다.

digest `fa18f0549c838591`
