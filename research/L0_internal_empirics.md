# Lane L0 (self) — 내부 데이터 실증 Findings
근거: 제공된 `train_labels.csv` 2024 홀드아웃(8,778h) + 공식 metric() 이식. 외부자료 0건.

## Executive summary
1. 오차분포의 **뾰족함(peakedness)** 이 FICR을 지배한다. 평균절대오차를 완전히 고정해도
   형상 β=3(가벼운꼬리)→β=1(라플라스)만으로 FICR 0.376→0.476, Score 0.629→0.681 (+0.052).
2. 우리 LB 프로파일(1-NMAE .87305 / FICR .37426)은 β≈2.5~3, **1위(.87964/.46767)는 β≈1.0**에 대응.
   → 1위와의 격차는 '더 정확해서'가 아니라 **오차분포 형상**이 다르기 때문.
3. 베이스라인 RandomForest는 120트리 평균 → 중심극한정리로 오차가 가우시안화 → FICR 구조적 손해.
4. 결정층(10%컷오프 조건부 + FICR인지 점예측 탐색)만으로 σ=16%에서 +0.0255 (0.6365→0.6619).
   분해: 컷오프 조건부 +0.0180, FICR임계 인지 +0.0075.
5. 결정층의 가정σ는 **실제보다 작게 잡는 쪽이 안전**(과대추정 시 급격히 손해).

## Evidence ledger
| ID | Finding | Tag | Source | Scope |
|---|---|---|---|---|
| L0-1 | 동일 E\|err\| 하 β=2→1로 FICR +0.076, Score +0.040 | directly_supported | 2024 홀드아웃 시뮬 | 정확히 본 대회 지표·용량·컷오프 |
| L0-2 | 1위 프로파일은 β≈1(라플라스) 오차와 정합 | directly_supported | LB 실측 + 시뮬 대조 | 동일 지표 |
| L0-3 | 결정층 순이득 +0.0255 @σ16% | directly_supported | 시뮬(캘리브레이션 정규 가정) | 분포가정 있음 |
| L0-4 | 10%컷오프 조건부 최적화 단독 +0.0180 | directly_supported | 시뮬 | 규칙상 합법(산식 공개) |
| L0-5 | 목표 0.67 도달에 필요한 σ ≈ 15%(결정층 有) vs 13.5%(無) | directly_supported | σ 스윕 | |
| L0-6 | 동분산 하 두꺼운꼬리(t4)가 FICR 유리 → 분산이 아닌 0근방 집중도가 핵심 | directly_supported | 시뮬 | |
| L0-7 | 평가대상(≥10%용량) 시간 비율: g1 60.7% g2 60.7% g3 53.7% | directly_supported | 라벨 실측 | |
| L0-8 | g3는 2022년 라벨 전무(2023~2024만 8,759/8,778행) | directly_supported | 라벨 실측 | |

## 적용 후보 ExperimentSpec (우선순위)
- **P1 `loss_l1_quantile`** (model): RF/MSE → LightGBM `objective=mae`·`quantile` 로 전환.
  기대 Score +0.03~0.05. 난이도 S. M1 실현가능 O. 규칙 pass.
- **P1 `decision_layer_ficr`** (decision): 분위수 예측 → 10%컷오프 조건부 + 기대정산금 최대화 1D 그리드 탐색.
  기대 +0.02~0.03. 난이도 M. 규칙 pass(산식 공개).
- **P2 `band_hit_objective`** (model): P(err<=6%) 직접 최적화(밴드 적중 분류/커스텀 목적).
  기대 미상, 상방 큼. 난이도 M~L.
- **P2 `sigma_tuning`** (decision): 결정층 가정 분산을 하이퍼파라미터로 튜닝(과소 쪽으로).
- **P3 `no_averaging_ensemble`** (ensemble): 앙상블 평균이 오차를 가우시안화 → 중앙값/최빈 결합 등 대안 결합 규칙 검토.

## 하지 말아야 할 것
- **MSE/RMSE 목적함수 사용** — 오차를 가우시안화해 FICR을 구조적으로 깎는다 (L0-1, L0-3).
- **단순 산술평균 앙상블 남용** — 같은 이유로 오차를 정규화시켜 FICR 손해 가능 (검증 필요).
- 1-NMAE 개선에 자원 집중 — 격차 기여 6.6%에 불과.
