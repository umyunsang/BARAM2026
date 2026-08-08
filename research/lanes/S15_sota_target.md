# Lane · S15 — TARGET AND LABEL CONSTRUCTION (A1–A4) SOTA/벤치마크 발굴

조사일 2026-08-08 · 도구 `websearch`(Serper/Google) **113 쿼리** + 공개 원문 **HTTP 읽기전용 열람 16건**(PDF 12 / 소스코드 3 / HTML 1)
레인 성격: **읽기 전용**. 저장소 쓰기는 `research/lanes/` 아래 이 문서와 검색로그뿐. **모델 fit 0회, 설치 0건, clone 0건, 락박스 미열람, git 조작 없음.**
저장소 읽기: `research/scratch/{powercurve.py, labels.parquet, scada_unison.parquet, scada_vestas.parquet, teacher_targets.parquet}`,
`research/nodes/{harness.py, s13_n9_turbine_curve.py, s9_n15_bin_width.py, s12_n8_ordinal_smoothing.py, S12-N8_*.json, S12-N10_*.json, S13-N2_*.json, S13-N3_*.json}`,
`src/baram/{models,decisions,evaluation}`. **수정 0건.**
내부 측정은 전부 `.venv/bin/python`으로 실행한 **기술통계(descriptive)** 이며 학습·적합은 없다.

---

## 0. 증거 등급 규약

| 등급 | 정의 |
|---|---|
| **A** | 원문(PDF/소스코드)을 이 레인이 직접 내려받아 읽고 표·본문·코드에서 수치/문장을 그대로 옮김. 인용은 원문 그대로. |
| **B** | 학술지 공식 초록 페이지 또는 기관 공식 페이지 본문만 읽음. |
| **C** | 검색 스니펫만 봄 → **`[전문미확인]`** 태그 필수. 이 태그를 지우고 인용하지 마라. |
| **I** | 이 레인의 저장소 내부 판독/기술통계. 새 fit 없음. |

**두 세계 구분 규약(부모 지시 이행).** 모든 권고에 `[WORLD: AVAILABLE]` 또는 `[WORLD: METER]` 를 붙였다.
- `[WORLD: AVAILABLE]` = 그 증거가 **가용전력(available power)** 을 채점하는 세계에서 측정된 것. 풍력 운영해석(OA)·IEC 61400-12·PCWG 문헌의 **거의 전부**가 여기다.
- `[WORLD: METER]` = 그 증거가 **계량기 실적**을 채점하는 세계에서 측정된 것. GEFCom/SDWPF/HEFTCom 같은 예측 경진대회와 KPX 정산제도가 여기다.
- 우리 채점기는 **METER** 다. AVAILABLE 세계의 권고를 그대로 이식하면 부모가 두 번 재현한 역전(teacher를 실측에 가깝게 만들수록 downstream이 나빠짐)이 반복된다.

---

## 1. 이 레인이 새로 측정한 내부 수치 (등급 I) — MIGRATION의 근거

기존 문서에 없던 것만 적는다. 전부 재현 가능한 기술통계다.

### I-1. **g3(UNISON)의 계량값은 SCADA로 사실상 복원된다** — A1의 상한을 확정

`scada_unison.parquet`의 `unison_wtg0i_power_kw10m`는 **10분 구간 kWh**다(시간합 = 6 × 10분평균).
5기 시간합 vs `labels.kpx_group_3`(2023-01-01 ~ 2025-01-01, 6슬롯 완비 시간 17,538개):

| 지표 | 값 |
|---|---|
| 총량비 meter/SCADA | **1.00354** (계량기가 SCADA 합보다 +0.35 %) |
| corr | **0.9965** |
| 전체 시간 MAE / cap | **0.01403** |
| **채점행(cf ≥ 0.1)만** MAE / cap | **0.02261** |
| (meter−SCADA)/cap 중앙값 | **0.0000** |
| 동 분위 (1 %, 25 %, 75 %, 99 %) | −0.0671 / −0.0063 / +0.0072 / +0.0788 |

> **해석.** "완벽한 터빈단 발전량 지식"을 가져도 시간 계량값과의 잔차는 채점행에서 **0.0226 cap**이다.
> 우리 라벨/가용성 채널 총량이 **0.04804** 이므로, A1(계량 복원)이 원리적으로 걷어낼 수 있는 최대치는 그 절반 미만이고,
> 그나마도 g3에만 존재한다(g1/g2는 power 열이 손상). **A1은 구조적으로 작은 스테이지다.**

### I-2. **타임스탬프 규약은 이미 정확하다** — A1의 가장 흔한 대박 가설을 죽임

`corr(meter_t , teacher_{t+k})` 및 MAE 스캔:

| lag k | g1 corr / MAE | g2 corr / MAE | g3 corr / MAE |
|---:|---|---|---|
| −1 | 0.9191 / 0.07825 | 0.9308 / 0.07720 | 0.9092 / 0.07799 |
| **0** | **0.9626 / 0.03757** | **0.9693 / 0.03649** | **0.9515 / 0.04201** |
| +1 | 0.9257 / 0.07474 | 0.9365 / 0.07343 | 0.9094 / 0.07673 |

> lag 0이 세 그룹 모두에서 압도적. **hour-ending/hour-beginning 오정렬, 30분 이동, DST 잔재는 전부 없다.** 재검토 금지.

### I-3. **가용성 결손은 채점행에서 빠져나가지 않는다** — "0.1 cap 필터가 고장을 걸러준다"는 희망을 죽임

g3, 6슬롯 완비 시간 17,534개:
- 채점행 비율 53.69 %
- `teacher − cf ≥ 0.05` 비율: 전체 **13.84 %**, **채점행에서 20.93 %**
- 비채점행(cf < 0.1) 8,120개 중 teacher > 0.2 인 행은 **230개(2.8 %)** 뿐

> 즉 `actual ≥ 0.1·cap` 필터가 걷어내는 것은 **거의 전부 진짜 저풍속 시간**이고, 고장/출력제한 시간은 **채점행 안에 그대로 남는다.**
> 결손은 채점행에서 **더 흔하다**(20.9 % > 13.8 %). A2를 무시할 수 없는 이유.

### I-4. **가용성 상태는 블록이고, 그 블록의 발생률은 연도 간 비정상(non-stationary)이다** — A2의 상한을 확정

`teacher ≥ 0.30`인 행에서 "저하(degraded) = teacher − cf ≥ 0.05" 비율:

| 그룹 | 전체 | 2022 | 2023 | 2024 |
|---|---|---|---|---|
| g1 | 0.2672 (n=11,349) | 0.356 | 0.186 | 0.257 |
| g2 | 0.2760 (n=11,763) | 0.281 | 0.198 | 0.350 |
| g3 | 0.2940 (n=6,452) | — | 0.299 | 0.289 |

- **요일별**: g1 0.233–0.316, g2 0.243–0.320, g3 0.264–0.326 → **평탄. 정비요일 신호 없음.**
- **시간대별**: g1/g2는 정오~14시 0.34–0.36 vs 심야 0.24 (약 1.4배), g3는 05–08시 0.34–0.37로 **모양이 다름.**
- **월별**: 그룹 간 **부호가 안 맞음**(8월 g1 0.514 / g3 0.039, 10월 g2 0.097 / g3 0.559) → 계절 물리가 아니라 **개별 정비 캠페인**.
- g3 SCADA 직접 관측: 시간별 "가동 중 터빈 수" 평균 3.042/5, **lag-1 자기상관 0.925, lag-24 0.322**(부모 측정 0.90/0.26–0.34와 일치).
- 터빈별 10분 슬롯 중 `ws > 4 m/s`인데 출력 0: **9.6 %–15.4 %**.

> **핵심 함의.** 결손률은 (a) 요일에서 예측 불가, (b) 월에서 그룹 간 비일관, (c) **연도 간 절대 ±0.08 로 흔들린다**(g2: 0.198 → 0.350).
> 2025년 결손률은 학습기간 평균으로 이식했을 때 **상대오차 30 %대**를 각오해야 한다. 이것이 A2 EXPECT를 크게 못 잡는 이유다.

### I-5. **손상된 VESTAS power 열은 "주변분포"가 살아 있고, 그로부터 파워커브를 복원할 수 있다** — A3의 새 재료

`vestas_wtg01_power_kw10m` 상태: min −42,512,957 / max +42,512,957, **음수 15.7 %**, `|w| > 1e5` 0.05 %,
`corr(ws, power) = 0.02` (완전 무상관 → 페어링 파괴 확정). 그러나 `|w|`의 분위수는 **정상**이다:
q50 = 102, q90 = 568, q99 = 601, **q99.9 = 602** → V126 1기 정격 3,600 kW = **600 kWh/10min** 과 정확히 일치.
`|w| ≤ 600` 비율 98.51 %.

> 즉 power 열은 **행 순열 + 부호/스케일 파손**이지 값 자체의 파괴가 아니다.
> `P = f(V)`가 단조라면 페어링 없이도 **f(v) = Q_P(F_V(v))** (분위수 매핑 / rank matching)로 커브를 복원할 수 있다.

**이 복원을 정답이 있는 UNISON에서 검증했다**(등급 I):

| 터빈 | 10분 페어 MAE | ws 4–6 | ws 8–10 | ws 12–14 | ws 16–18 |
|---|---|---|---|---|---|
| unison wtg01 | 66.2 kWh10m (0.0945 터빈cap) | −31 | **+16** | +176 | +302 |
| unison wtg03 | 54.5 kWh10m (0.0779 터빈cap) | −19 | **−11** | +84 | +185 |

(값은 `mean(f̂) − mean(P_true)`, 단위 kWh/10min, 터빈 cap 700)

> **분위수 매핑은 정격 이하에서 편향 ±31 kWh10m(≤ 0.044 터빈cap) 이내로 정확하고, 정격 이상에서 크게 과대추정한다.**
> 과대추정의 원인은 정확히 **고풍속 구간의 가용성/디레이트가 단조성을 깨기 때문**이다.
> 다시 말해 **분위수 매핑이 만드는 커브는 "가용성이 제거된 커브"** 이며, 그것이 바로 우리 teacher가 원하는 성질이다.

### I-6. **26개 균등 bin은 이미 분위 균등에 가깝다** — A4의 "비균등 bin" 아이디어를 죽임

채점행(cf ≥ 0.1) 41,220개의 폭 0.04 bin별 점유율:

| cf 구간 | 행 비중 | y-가중(FICR 분자) 비중 |
|---|---|---|
| 0.12–0.16 | 0.0767 | 0.0214 |
| 0.28–0.32 | 0.0444 | 0.0267 |
| 0.52–0.56 | 0.0396 | 0.0428 |
| 0.76–0.80 | 0.0432 | 0.0675 |
| 0.88–0.92 | 0.0394 | 0.0710 |
| 0.96–1.00 | 0.0232 | 0.0453 |

> 0.16 이상 전 구간에서 행 비중이 **0.033–0.062**로 거의 평탄. 분위 간격 bin ≈ 균등 bin.
> **비균등/분위 bin 재설계는 얻을 게 없다.** 다만 **FICR가 y로 가중하므로 고 cf 영역의 상대 중요도는 행 수의 1.6–2.0배**다
> (cf ≥ 0.9: 행 7.7 % / y-가중 14.5 %). bin 폭을 **고 cf에서만** 좁히는 것은 별개의(살아있는) 축이다 → A4-2 참조.

---

## 2. STAGE A1 — 라벨 QC 및 계량 복원

### A1-1 (권장) — SCADA 정합 기반 **행 신뢰도 라벨링**(값 교체가 아니라 가중/게이트)

- **SOTA**
  NREL WP3/OpenOA 운영해석 표준절차: 계량기 에너지와 터빈 SCADA 에너지를 **동일 기간 합**으로 대조해 전기손실을 정의하고
  (`Electrical Losses = 1 − Plant Revenue Meter Energy / Turbine SCADA Energy`), **"모든 터빈이 모든 타임스텝을 보고한 날"만** 대조에 쓴다.
  계량기 불확실성은 IEC 60688:2012 / ANSI C12.1-2014 기준 **σ = 0.5 %** 로 둔다.
  여기에 SDWPF(KDD Cup 2022) 참가팀의 이상치 규칙표를 결합해, 값을 고치는 대신 **행에 신뢰도 등급을 붙인다.**
- **EVIDENCE**
  - Todd, Optis, Bodini, Fields, Perr-Sauer, Lee, Simley, Hammond, *Wind Energy* 25(11):1775-1790, 2022 — NREL fy22osti/81032, https://docs.nlr.gov/docs/fy22osti/81032.pdf **(A)** `[WORLD: AVAILABLE]`
    원문: "Electrical Losses = 1 − (Plant Revenue Meter Energy / Turbine SCADA Energy)";
    "we calculate daily sums of energy production from the turbine SCADA and from the revenue meter data, **only for days when all turbines within the wind plant are reporting at all time steps**";
    "the choice of a **revenue meter uncertainty of 0.5 %** is consistent with what is typically assumed by wind energy consultants ... based on **IEC 60688:2012** and **ANSI C12.1-2014**".
    효과크기: 10개 단지 × 68건 EYA 제출에서 P50 bias **−1.2 %**, TIE 과대예측 **+3.9 %**(표준편차 4.9 %), 전기손실 **+0.5 %**, 가용성손실 **+2.1 %**, 기타/미설명 **+2.4 %**; 하류 손실항의 표준편차는 최대 1.6 %.
  - Craig et al. 2018 (Optis et al., WES preprint 2019-12, https://wes.copernicus.org/preprints/wes-2019-12/wes-2019-12.pdf 에서 인용) **(A)** `[WORLD: AVAILABLE]`
    "different combinations of data flagging algorithms applied to nacelle wind speed and turbine power data, along with different choices for power curve models, resulted in **3.0 % total spread and 0.7 % interquartile range** when estimating gross energy production for a wind plant over a year-long period".
    → **QC 선택 자체가 총생산 추정에 3 % 폭을 만든다**는 정량 근거.
  - Liu et al., *Solution to Spatial Dynamic Wind Power Forecasting*, Baidu KDD Cup 2022 Workshop paper 1286, https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_1286.pdf **(A)** `[WORLD: METER]`
    규칙표 원문 그대로: "The abnormal conditions include: • Patv<0. • Wspd<1 and Patv>10 • Wspd<2 and Patv>100 • Wspd<3 and Patv>200 • **Wspd>2.5 and Patv==0** • Wspd==0 and Wdir==0 and Etmp==0 • Etmp<−21 • Itmp<−21 • Etmp>60 • ITmp>70 • Wdir>180 or Wdir<−180 • Ndir>720 or Ndir<−720 • Pab1>89 or Pab2>89 or Pab3>89".
    효과크기: "the total number of Patv less than 0 or equals to Nan is **1,312,580**, which is **582,716** after the above three steps of data processing"(결측/이상 55.6 % 감축).
  - Zhou, Lu, Xie et al., *SDWPF*, arXiv 2208.04360v2, https://arxiv.org/html/2208.04360v2 **(A)** `[WORLD: METER]`
    "When Patv ≤ 0, and Wspd > 2.5 at time t, the actual active power Patv of this wind turbine at time t is **unknown**"; "These unknown values will **also not be used for evaluating the model**."
- **BENCHMARK** Baidu KDD Cup 2022 (SDWPF, 134기, 10분, 24개월/1,140만 레코드) — 이상·미지값을 **평가에서 제외**하는 규칙이 공개된 유일한 벤치마크. 보조로 IEA Wind Task 43 Open Data / ENGIE La Haute Borne(OpenOA 회귀테스트 데이터).
- **MIGRATION**
  - 입력: `research/scratch/scada_unison.parquet`(g3, power 유효), `research/scratch/labels.parquet`.
  - 파생: 시간종료 키 `he = ceil(kst_dtm − 1s)`로 그룹핑 → ① `n_slot`(6이어야 함), ② `n_report`(비-NaN 터빈 수), ③ `scada_kwh = Σ_turbine Σ_slot power_kw10m`, ④ `resid_cf = (meter − scada_kwh)/cap`.
  - 출력 컬럼(그룹3 전용): `g3_qc_full`(= n_slot==6 & n_report==5), `g3_meter_scada_resid_cf`.
  - **g1/g2에는 SCADA power가 없다** → 대체물은 `teacher − cf` 결손과 `ws` 기반 규칙(`ws>4 & cf≈0`, KDD Cup 규칙의 이식)뿐. 이 비대칭을 반드시 receipt에 명시하고 그룹별로 다른 QC를 쓴다.
  - 값 교체는 하지 않는다. **`sample_weight` 감쇠 또는 게이트로만 소비**한다 — `harness.py`의 `cfg['teacher_weight']` / `cfg['teacher_rows']` / `cfg['calib_rows']` 훅에 그대로 꽂힌다.
  - 깨지는 것: NREL 절차는 **월/일 합계**로 정의됐고 우리는 **시간 단위**로 쓴다. 시간 단위에서는 0.5 % 계량 불확실성이 상쇄되지 않아 I-1의 0.0226 cap 잔차가 그대로 남는다.
- **RISK**
  I-1이 이미 한계를 못박았다: 채점행에서 meter−SCADA MAE가 **0.0226 cap**이므로, 이 채널을 완벽히 다뤄도 라벨채널 0.04804의 절반 이하만 움직인다.
  더 나쁘게, g3에만 적용하면 NMAE가 그룹 평균이므로 총점에 **1/3만 반영**된다.
  가장 큰 실패 모드: `n_report < 5` 행을 학습에서 빼면 고장 시간이 학습분포에서 사라져 teacher가 아니라 **잔차 분포(ztab)** 가 낙관적으로 바뀐다 → I-3에 따라 그 행들은 채점행에 그대로 남으므로 **순손실**이 될 수 있다.
- **COST** 3–5 h. 추가 설치 없음(pandas/numpy만). OpenOA 설치 **불필요**.
- **EXPECT** **+0.0003** (범위 0.0000 ~ +0.0010). g3 단독 개선이라 그룹평균 희석을 반영한 값.

### A1-2 (부결 권고) — 계량기-SCADA 오프셋 보정 / 타임스탬프 재정렬

- **SOTA** 계량기 총량비(우리는 1.00354)로 라벨을 재스케일하거나 시간축을 ±1 h 이동.
- **EVIDENCE** Todd et al. 2022의 계량기 불확실성 **σ = 0.5 %** (A) `[WORLD: AVAILABLE]`; 내부 I-1, I-2 (등급 I).
- **BENCHMARK** 없음.
- **MIGRATION** 해당 없음 — **하지 마라.**
- **RISK** I-2에서 lag 0이 세 그룹 모두 최적임이 확정됐다(lag ±1은 MAE가 2배). 총량비 편차 0.35 %는 계량기 불확실성 0.5 %보다 작아 **통계적으로 0과 구별 불가**.
- **COST** 2 h.
- **EXPECT** **0.0000**. 이 축은 이 레인이 닫는다.

---

## 3. STAGE A2 — 가용성 / 고장 / 출력제한 식별

### A2-1 (권장, 이 스테이지의 최선) — **블록 단위 게이팅**: 행이 아니라 구간을 자른다

- **SOTA**
  ① OpenOA `filters.unresponsive_flag(data, threshold=3, col=None)` — "Flag time stamps for which the reported data **does not change for `threshold` repeated intervals**". 구현 원문: `flag = subset.diff(axis=0).ne(0).rolling(threshold - 1).sum()` → `flag = flag == 0` → 선행 threshold개까지 shift로 확장.
  ② OpenOA `filters.bin_filter(bin_col, value_col, bin_width, threshold=2, center_type∈{mean,median}, bin_min, bin_max, threshold_type∈{std,scalar,mad}, direction∈{all,above,below})` — 풍속 bin 안에서 중심 대비 임계 밖을 플래그. **`direction='below'`** 로 아래쪽만 자르면 고장/디레이트만 선택적으로 걸린다.
  ③ 그 위에 **변화점 탐지(이진분할/PELT) 또는 2-상태 HMM**을 결손 시계열에 걸어 **블록 경계**를 잡는다. 풍력에서의 표준 결합형은 Bayesian change-point + 사분위 결합이다.
- **EVIDENCE**
  - NREL/OpenOA `openoa/utils/filters.py`, https://raw.githubusercontent.com/NREL/OpenOA/main/openoa/utils/filters.py **(A, 소스코드 원문)** `[WORLD: AVAILABLE]` — 위 시그니처·구현은 원문 그대로.
  - Optis, Perr-Sauer, Bodini et al., WES preprint 2019-12 **(A)** `[WORLD: AVAILABLE]`: 모듈 성숙도 "Filtering (P): **85 % unit test coverage**", "Power Curves (P): **95 %**", "Plant Analysis (P): Monte Carlo Method AEP **99 % integration test coverage**".
  - Bisgaard, Ritschel et al., *Hidden Markov Models for Bounded, Inflated Time Series: Forecasting Icing on Wind Turbine Blades*, **Wind Energy, 2026** (10.1002/we.70110; DTU Orbit 438665298) **(B)** `[WORLD: AVAILABLE]`
    공식 초록 원문: "Time series analysis of icing-induced power loss in wind turbines pose several challenges: the response is **bounded, serially dependent, intermittently missing**"; "This study proposes a **discrete-time HMM with state-dependent zero-inflated beta distributions** for forecasting power loss in wind [turbines]"; "traditional forecasting methods **overestimate power production**".
    → **우리 결손 시계열의 성질(유계·계열상관 0.925·간헐결측)과 정확히 같은 문제 정의**이고 처방이 HMM + zero-inflated beta 다.
  - Messner, Pinson, Browell, Bjerregård, Schicker, *Evaluation of Wind Power Forecasts — An up-to-date view*, **Int. J. Forecasting 36(3), 2020**, http://pierrepinson.com/wp-content/uploads/2020/02/Messneretal2020.pdf **(A, [추출왜곡])** `[WORLD: 두 세계를 명시적으로 양분]`
    원문(대문자는 PDF 폰트 매핑 왜곡): "ANOTHER IMPORTANT DECISION TO BE MADE IS WHETHER **CURTAILMENT DATA SHOULD BE KEPT OR REMOVED** FROM THE DATA BEFORE EVALUATION. AGAIN THIS DECISION SHOULD BE MADE BASED ON THE INTENDED APPLICATION. **IF THE FORECAST USER IS INTERESTED IN THE AVAILABLE POWER AND NOT IN THE REAL POWER PRODUCTION**, DATA WITH CURTAILMENT SHOULD BE REMOVED FROM THE EVALUATION DATASET SINCE ERRORS WHEN NOT PREDICTING THESE CASES ARE NOT MEANINGFUL. **IF PERIODS OF CURTAILMENT ARE RETAINED, IT MAY BE INSTRUCTIVE TO SEPARATE ERRORS THAT RESULTED FROM UNFORESEEN CURTAILMENT FROM THOSE THAT RESULTED FROM OTHERS, AS AVERAGE SCORES WILL CONFLATE THESE EFFECTS.**"
    → **이 문헌이 직접 두 세계를 갈라놓았다.** 우리는 METER 세계이므로 처방은 "제거"가 아니라 **"분리해서 다르게 처리"** 다. 효과크기 없음(정성).
  - Z. Wang et al., *Abnormal data cleaning of wind turbine power curve using Bayesian change point-quartile combined algorithm*, Proc. IMechE Part A, 2023 **(C)** `[전문미확인]` `[WORLD: AVAILABLE]`.
  - X. Shang et al., *A Comprehensive Cleaning Method for Outliers in Wind [Turbine Data]*, **Energies 19(5):1161, 2026** **(C)** `[전문미확인]` `[WORLD: AVAILABLE]`
    스니펫 원문 "In terms of power-curve fitting accuracy, the average **NMAE decreases by 8.65 %, 5.07 %, 7.57 %, and 4.06 %**, while the average NRMSE decreases by 10.78 %, 7.99 %, ...". **대조군이 스니펫에 없다. 인용만 하고 근거로 쓰지 마라.**
- **BENCHMARK** GEFCom2014 wind track(출력제한 처리 논의의 출처), SDWPF/KDD Cup 2022(미지값 규칙), ENGIE La Haute Borne(OpenOA 필터 회귀테스트).
- **MIGRATION**
  - 입력: 현재 게이트가 쓰는 결손 `d_t = pc_true_t − cf_t` (harness/s9_n15에서 `gapv`로 이미 계산됨). 그룹별 시간 시계열 3개.
  - 파생: ① `d_t ≥ 0.05` 이진열의 **run-length encoding** → 블록 길이 분포, ② 블록 경계를 **이진분할(binary segmentation)** 또는 2-상태 Gaussian HMM(수동 EM, ~60줄)로 정련.
    **`ruptures` 설치 금지**(추가 설치 불가). scipy 1.18 + numpy 2.5로 이진분할이면 충분하다.
  - 출력: `block_id`, `block_len`, `block_mean_deficit`, `in_outage_block`(bool).
  - **소비 지점은 `cfg['calib_rows']` 하나뿐이다.** 현행 "개별 행 `d_t ≥ 0.05` 제외"를 "`in_outage_block`(길이 ≥ L) 제외"로 바꾼다. L은 fold-outside로 선택하고, **L=1이 현행을 정확히 재현하므로 중첩 모형**이라 게이트가 baseline으로 되돌릴 수 있다.
  - g3에 한해 SCADA 직접 신호 `n_operating_turbines`(I-4)로 블록 라벨을 **검증**할 수 있다(피처로 쓰지 말 것 — 2025에 없다).
  - 깨지는 것: 원 문헌들은 전부 **10분 터빈단**에서 블록을 잡는다. 우리는 **시간 그룹단**이다. 터빈 1기 정지 = g1/g2에서 0.167 cap, g3에서 0.200 cap 이므로, **현행 임계 0.05 cap은 "1기 정지의 25–30 %"** 에 불과해 **지나치게 민감**하다 — 아래 RISK 참조.
- **RISK**
  I-4가 정면으로 경고한다: 결손률이 연도 간 **0.198 ↔ 0.350** 로 흔들리므로 2022–2024에서 고른 L이 2025에 최적이 아닐 수 있다.
  구체적 실패 경로: 블록 게이팅은 `ztab`(잔차 분위표)을 더 깨끗하게 만들어 **예측분포를 좁힌다** → ±0.06 밴드 적중률은 오르지만 실제 고장 시간(채점행의 20.9 %, I-3)에서 완전히 빗나간다 → **1−NMAE와 FICR의 부호가 갈릴 수 있다.**
  **가장 중요한 위험**: 임계 0.05 cap이 부모가 측정한 정격이상 커브편향(+0.054/+0.066/+0.064)과 **같은 크기**다. 즉 **현재 게이트는 고장이 아니라 파워커브 오차를 자르고 있을 가능성이 크다.** A3-1이 커브를 고치면 이 게이트의 의미 자체가 바뀐다 → **A3-1 → A2-1 순서 의존성이 있다.**
- **COST** 4–6 h. 설치 없음(scipy/numpy만; `ruptures` 금지).
- **EXPECT** **+0.0010** (범위 −0.0005 ~ +0.0025).

### A2-2 (조건부) — **가용성 혼합(mixture)을 예측분포에 넣기** (teacher는 건드리지 않음)

- **SOTA**
  `y = A · P` 분해. `P`는 잡음 없는 물리 파워(현행 teacher 유지), `A`는 가용성 승수.
  회귀기 타깃은 **그대로 `pc_true`**, 대신 결정층에 들어가는 표본을 `(1−q)`-정상 성분 + `q`-저하 성분의 **2성분 혼합**으로 만든다.
  `q`와 `a|저하`는 학습기간의 **주변분포**에서 추정하고, 조건화는 그룹 × 시간대까지만 허용. 분포족은 **zero-inflated beta**(Bisgaard)가 정확히 이 모양이다.
- **EVIDENCE**
  - Bisgaard et al., Wind Energy 2026 **(B)** `[WORLD: AVAILABLE]` (위 인용).
  - IEA Wind Task 36/51, *Recommended Practice for the Implementation of Renewable Energy Forecasting Solutions, Part 4*, 1st ed., 2022, https://iea-wind.org/wp-content/uploads/2022/06/IEAWind_Task36_Recommended_Practice_Part4_1st_Edition_public.pdf **(A, [추출왜곡])** `[WORLD: 양분 명시]`
    "operational data such as **plant availability** (e.g. proportion of turbines/panels in service) and **control actions** (e.g. curtailments) **are also required as they change the nature of the power measurement** ... must be calibrated to predict the variable of interest to the user: **what the actual power production is expected to be** in the future, **or** the power production **would be** expected ... **if no control actions were** [taken]".
  - Pinson, *Very-short-term probabilistic forecasting of wind power with generalized logit-Normal distributions*, **JRSS-C 61(4):555-576, 2012**, http://pierrepinson.com/docs/pinson11_wpfore_rev.pdf **(A)** `[WORLD: METER]`
    "The improvements in terms of the **CRPS** criterion are significant when going from the Normal predictive densities to **GL-Normal** predictive densities, **in the order of 7.5 %**, and consistent over the evaluation period." (Horns Rev 160 MW, 10분 리드타임, 약 8개월)
    → **예측분포의 형태를 목표변수의 실제 지지집합/모양에 맞추면 CRPS 7.5 % 개선**이라는 유일한 A급 수치. 우리 경우의 "모양"은 유계성이 아니라 **혼합성**이다.
- **BENCHMARK** Horns Rev(Pinson 2012), GEFCom2014 wind.
- **MIGRATION**
  - `harness.py`의 `samples = np.clip(pc[:,None] + off + infl*sd[:,None]*ztab[g][bi], 0.0, cap_hi)` **한 줄이 유일한 접점**이다.
    `samples_deg = pc[:,None] * a_draws[None,:]` 를 concat하고, `sharpen_weights(samples, tp)` 로 가는 가중을 `(1−q)`/`q` 로 분할한다.
  - `q`는 그룹 × 시간대 3×24 표(= I-4의 표)로 충분. `a|저하`는 `cf/pc`(pc ≥ 0.3)의 경험분위 20개.
  - `ACTIONS` 격자와 `err/units` 계산은 **손대지 않는다** → 정책/밴드 축을 건드리지 않는다.
  - 깨지는 것: 부모가 닫은 "decision-layer/policy/band tricks"는 **주어진 분포 위에서의 γ·온도 스윕**이다. 여기는 **분포 자체를 구조적으로 바꾸는 것**이라 형식상 다른 축이지만 argmax를 움직이는 경로는 같다 → **부모가 그 구분을 인정하지 않으면 착수 전에 폐기.**
- **RISK**
  계단 보상에서 2점 혼합에 질량 `q`를 추가해도 `q < 0.5` 면 argmax는 **고모드에 그대로 머문다**(고모드 항이 모든 행동에 대해 `(1−q)`로 균일 축소되어 순위 불변). 실제로 움직이는 것은 `a`가 연속적으로 퍼져 ±0.06 밴드가 **두 모드를 동시에 걸치는 좁은 영역**뿐이다.
  → **구조적으로 효과가 0에 가까울 가능성이 높다.** 게다가 I-4의 연도 간 `q` 비정상성(±0.08)이 그 좁은 영역의 위치를 흔든다.
- **COST** 6–8 h. 설치 없음.
- **EXPECT** **+0.0000** (범위 −0.0015 ~ +0.0015). **동전 던지기. 리버스 어블레이션 슬롯 채우기 용도로만 만들고 단독 채택 근거로 쓰지 마라.**

### A2-3 (부결) — 가용전력 재구성 / 검열회귀(Tobit) / 곡선 상향 복원

- **SOTA** Messner & Pinson (2014) 역파워커브 변환 + 검열회귀; censored SCADA available-power 추정.
- **EVIDENCE** Messner, Zeileis, Broecker, Mayr, *Probabilistic wind power forecasts with an inverse power curve transformation and censored regression*, **Wind Energy 17(11), 2014** (https://centaur.reading.ac.uk/47581/) **(B)** `[WORLD: METER]` — "The results show that with our **inverse (power-to-wind) transformation**, simpler linear regression models with **censoring** perform equally or better". *Censored Available-Power Data Bias Dispatch Decisions in Curtailed Wind Farms*, SSRN 2026 **(C)** `[전문미확인]` `[WORLD: AVAILABLE]`.
- **BENCHMARK** —
- **MIGRATION** 없음.
- **RISK** 이것이 정확히 **부모가 두 번 재현해 닫은 축**이다(등온회귀 재보정 −0.022 ~ −0.027 of 1−NMAE; 터빈 전달계수 + 스톰커브 −0.0003 ~ −0.0014). 가용전력 복원은 AVAILABLE 세계의 처방이고 우리 채점기는 METER다. **제안 금지.**
- **COST** —
- **EXPECT** **≤ −0.010** (과거 재현 기준).

---

## 4. STAGE A3 — 파워커브 추정과 teacher 타깃  ← **이 클러스터의 본체**

### A3-1 (강력 권장, 클러스터 1순위) — 커브를 **계량기가 아니라 SCADA 출력**에 적합시킨다

> **이것이 "teacher realism 금지" 제약을 위반하지 않고 teacher를 개선하는 유일한 경로다.**
> 현행 `research/scratch/powercurve.py`의 목적함수는 원문 그대로
> `y = lab[f'kpx_group_{g}']/cap` … `return float(np.abs(j.p-j.y).mean())`
> 즉 **커브 파라미터를 계량기 실적에 적합**시킨다. teacher는 "잡음 없는 풍속의 함수"이지만
> **그 함수 자체가 가용성 잡음이 섞인 목적함수에서 나왔다.**
> SCADA 출력에 적합하면 teacher는 여전히 풍속의 결정론적 함수이고, **적합 목표에서 가용성이 빠진다.** 이동 방향이 정반대다.

- **SOTA**
  ① **IEC 61400-12-1 binned power curve**: 폭 0.5 m/s bin 평균, 결측 bin 선형보간, 컷오프 밖 0. OpenOA `power_curve.IEC(bin_width=0.5, windspeed_start=0, windspeed_end=30, interpolate=)`.
  ② **5-파라미터 로지스틱(5PL)**: `P(v) = d + (a − d) / (1 + (v/c)^b)^g` (a 하한점근, b 기울기, c 변곡점, d 상한점근, g 비대칭). OpenOA `power_curve.logistic_5_parametric`, `differential_evolution` + least-squares, 경계 `((1200,1800), (−10,−1e−3), (1e−3,30), (1e−3,1), (1e−3,10))`.
  ③ **단조 스플라인 / 등온회귀**: 단조성을 보존하는 비모수 커브.
  ④ **분위수 매핑 복원**(I-5): 페어링이 파괴된 VESTAS power 열에서 `f(v) = Q_P(F_V(v))`.
- **EVIDENCE**
  - NREL/OpenOA `openoa/utils/power_curve/functions.py`, `parametric_forms.py` **(A, 소스코드 원문)** `[WORLD: AVAILABLE]`
    "Use **IEC 61400-12-1-2 method** for creating a binned wind-speed power curve. Power is set to zero for values outside the cutoff range: [windspeed_start, windspeed_end]"; 5PL 구현 원문 `return d + (a - d) / (1 + (x / c) ** b) ** g`.
    **라이선스/의존성 직접 확인**(`pyproject.toml` 원문): `license = {file = "LICENSE.txt"}`, classifier `"License :: OSI Approved :: BSD License"`, 그리고 **`"scikit-learn>=1.0,<1.7"`** — 우리 sklearn 1.9.0을 강등시키므로 **설치 불가**(부모 판단 확인됨). 그 밖에 `pygam>=0.11.0`, `bokeh>=3.3`, `attrs>=22.2` 등 무거운 의존성 다수.
    **그러나 `filters.py`는 numpy/scipy/pandas + `sklearn.cluster.KMeans`(오직 `cluster_mahalanobis_2d` 한 함수에서만)에만 의존하고, `power_curve/parametric_forms.py`는 numpy/pandas만, GAM 경로만 `pygam`을 쓴다.**
    → `range_flag / unresponsive_flag / std_range_flag / window_range_flag / bin_filter` 5함수와 `logistic5param`, `logistic5param_capped` 는 **의존성 0으로 vendoring 가능**하다(BSD-3, 저작권·라이선스 고지 유지 필요).
  - Lee, Fields, Perr-Sauer, Williams, Simley, Bodini, Optis 외, *The Power Curve Working Group's assessment of wind turbine power performance prediction methods*, **Wind Energy Science 5:199-223, 2020**, https://wes.copernicus.org/articles/5/199/2020/wes-5-199-2020.pdf **(A)** `[WORLD: AVAILABLE]`
    데이터: **55건의 파워커브 성능시험, 9개 기관 제출**(Share-3 exercise). 지표 정의 원문:
    `NME = Σ(P_method(t) − P_actual(t)) / ΣP_actual(t)`, `NMAE = Σ|P_method(t) − P_actual(t)| / Σ|P_actual(t)|`, 10분 샘플.
    보정법 5종(baseline / Den-Turb / Den-2DPDM / Den-Augturb / Den-3DPDM), 전부 IEC 61400-12-1(2005) 밀도보정 포함.
    결과 원문: "The trial methods **reduce power-production prediction errors compared to the baseline method at high wind speeds**, which contribute heavily to power production; however, the trial methods **fail to significantly reduce prediction uncertainty** in most meteorological conditions."
    "**More than 60 % of the submissions report prediction error reduction** by switching to a trial method from the baseline for LWS cases"; HWS에서는 "only **Den-2DPDM, Den-Augturb, and Den-3DPDM** perform significantly better than the baseline in the HWS-LTI condition".
    "for **more than half of the submissions, the data set has a large influence on the effectiveness of a trial method**."
    "For the meteorological conditions when a wind turbine produces **less than the power its reference power curve suggests**, using **power deviation matrices** leads to more accurate power prediction."
  - St. Martin, Lundquist, Clifton, Poulos, Schreck, *Atmospheric turbulence affects wind turbine nacelle transfer functions*, **WES 2:295-306, 2017**, https://wes.copernicus.org/articles/2/295/2017/wes-2-295-2017.pdf **(A)** `[WORLD: AVAILABLE]`
    "Corrections to the nacelle anemometer wind speed measurements can be made with **NTFs** and used to calculate an **AEP that comes within 1 % of an AEP calculated with upwind measurements**." 데이터: GE 1.5sle, NREL NWTC, 2012-11-29 ~ 2013-02-14(2.5개월), 10분 평균.
    "During periods of **low stability** as defined by the Bulk Richardson number, the nacelle-mounted anemometer **underestimates the upwind wind speed more** than during periods of high stability at some wind speed bins below rated [speed]."
  - Jing, Qian, Wang 외, *Wind Turbine Power Curve Modelling with Logistic Functions Based on Quantile Regression*, **Applied Sciences 11(7):3048, 2021** **(C)** `[전문미확인]` `[WORLD: AVAILABLE]`
    스니펫 원문: "**5-parameter logistic function (5PL) generally has the best fitting effect.** cubic spline interpolation (CSI) can fit smooth and accurate power [curves]"; "In Table 5, **5PL is selected as the benchmark** for deterministic WTPC".
  - Villanueva & Feijóo, *Comparison of logistic functions for modeling wind turbine power curves*, **Electric Power Systems Research 155, 2018** **(C)** `[전문미확인]` `[WORLD: AVAILABLE]`
    "The **6PLE** function is the best option to model a wind turbine power curve due to its performance. • The **3PLE and 5PLE** functions are strongly recommended". 후속(2021) **(C)**: "generally provides better accuracy with **mean absolute percentage error values below 0.02** for the five-parameter [function]".
  - Mehrjoo, Jozani, Pawlak, *Wind turbine power curve modeling for reliable power prediction using monotonic regression*, **Renewable Energy 147(P1):214-222, 2020** **(B)** `[WORLD: AVAILABLE]`
    "we present two nonparametric techniques based on **tilting method and monotonic spline regression** methodology to construct wind turbine power curves"; "Results show that **monotone spline regression performs the best** while the tilting approach performs similar to the [other] methods".
  - McCandless & Haupt, **WES 4:343-353, 2019** **(A, 선행레인 S13/S5 확보)** `[WORLD: AVAILABLE]` — Jensen 부등식 항, 초터빈 변환 MAE 68.83 → 51.15 / 50.41 kW (−25.7 % / −26.8 %).
    **우리 코드는 이미 터빈별·10분별로 커브를 먹인 뒤 평균하므로 이 항은 처리 완료**(`s13_n9_turbine_curve.py` 주석이 그 사실을 명시적으로 기록).
- **BENCHMARK** **PCWG Share-3** (55 turbines / 9 orgs, NME·NMAE) — 파워커브 보정법의 유일한 다기관 공개 벤치마크. 보조: ENGIE La Haute Borne(OpenOA 회귀테스트), IEA Wind Task 43 Open Data.
- **MIGRATION** — **여기가 핵심이다.**
  1. **g3 (UNISON, power 유효, 2023-01-01 ~ 2025-01-01, 105,264행 × 5기).**
     터빈별 `(unison_wtg0i_ws, unison_wtg0i_power_kw10m)` 페어를 직접 얻는다.
     정제: (a) `power < 0` 제거(존재하지 않음), (b) `power > 700` 제거 — I-5에서 정격 = 21000/5/6 = **700 kWh/10min** 확인, 관측 max 709 + 희소 800/900/1000 이상치,
     (c) vendored `unresponsive_flag(threshold=3)` 로 고착 신호 제거,
     (d) vendored `bin_filter(bin_col=ws, value_col=power, bin_width=0.5, threshold=2, center_type='median', threshold_type='mad', direction='below')` 로 **아래쪽 이상치(고장/디레이트)만** 제거.
     그 위에 **IEC 0.5 m/s bin 평균** 또는 **5PL**을 적합 → 터빈별 `f_{g3,i}(v)` 5개. **계량기를 전혀 보지 않는다.**
  2. **g1/g2 (VESTAS, power 손상, 157,819행 × 12기).**
     `|power|` 를 취하고 `> 700` 제거(I-5: q99.9 = 602, 정격 600 kWh/10min) → 주변분포 확보.
     터빈별 `f_i(v) = Q_{|P|}(F_{V_i}(v))` 분위수 매핑.
     **정격 이상 구간은 신뢰하지 마라** — I-5의 unison 검증에서 12–14 m/s +176, 16–18 m/s +302 kWh10m 과대.
     **처방**: 분위수 매핑은 `v ≤ v_rated` 에서만 채택하고, 정격 이상은 (i) g3에서 SCADA로 실측한 디레이트 형상을 `v/v_rated` 무차원화해 이식하거나 (ii) 그냥 평탄 1.0으로 둔다.
     **없는 것**: g1/g2의 진짜 고풍속 디레이트 형상. 이 데이터에서 식별 불가능하다. receipt에 명시하라.
  3. **소비 지점**: `powercurve.py`의 `curve()` 와 `params[g]` 만 교체하면 `teacher_targets.parquet`의 `g{1,2,3}_pc / _pc_intra / _pc_spread` 가 자동 갱신되고 harness의 `pc_true` 가 그대로 바뀐다. **하류 코드 변경 0.**
  4. 부수효과: 터빈별 커브를 쓰면 `pc_spread`(터빈 간 산포)의 의미가 바뀐다 — 지금은 풍속 산포만, 이후에는 **기기 이질성**까지 포함. 피처 해석 문서를 갱신하라.
  5. **깨지는 것**: PCWG/IEC 문헌은 **met mast 기준 자유류 풍속**을 전제한다. 우리는 **나셀 풍속만** 있다. St. Martin 2017의 "AEP 1 % 이내"는 met mast가 있을 때 얘기다.
     NTF를 추정할 기준이 없으므로 우리 커브는 "나셀풍속 → 출력" 커브다. **그것이 오히려 우리 목적에 정확하다** — teacher의 입력도 나셀풍속이므로 나셀 편향이 상쇄된다.
     **이 상쇄가 성립하려면 커브 적합과 teacher 생성이 같은 풍속열을 써야 한다. 현재 코드는 그렇다. 절대 깨지 마라.**
- **RISK**
  1. **가장 큰 위험은 A3-1이 S13-N9의 재탕으로 보인다는 것이다.** 차이를 명확히 하라: N9는 `a_i`와 스톰램프를 **계량 cf에 대한 MAE 최소화**로 적합했다(원문 주석: "All parameters are fitted by minimising MAE against the metered hourly capacity factor on the TRAINING window of each fold only"). A3-1은 **계량값을 보지 않는다.** 목적함수가 다르다.
     그럼에도 결과가 같은 방향(정격 이상 하향)이고 downstream이 다시 지면 — 부모의 제약은 "적합 목표"가 아니라 **"커브 형상 자체"** 에 관한 것이고, 그때는 **A3 전체를 닫아야 한다.** 이것이 이 스테이지의 결정적 판별 실험이다.
  2. g3만 SCADA 직접 적합이 가능하므로 **그룹 비대칭**이 생긴다. g3는 라벨이 2023부터라 이미 학습량이 2/3다.
  3. `bin_filter(direction='below')` 는 **정상적인 저출력 산포**(고풍속 난류)까지 자를 수 있다. `threshold`를 fold-outside로 고르지 않으면 과적합.
  4. 분위수 매핑은 `V`와 `P`가 **같은 기간·같은 결측 패턴**을 가져야 성립한다. vestas power의 음수 15.7 %를 버리면 `F_V`와 `Q_P`의 표본이 어긋난다 → **음수는 `|·|`로 살리고 `>700`만 버려라**(I-5 근거).
- **COST** 6–10 h. **설치 없음.** OpenOA는 **vendoring**만(filters 5함수 ≈ 200줄 + `logistic5param` ≈ 40줄, BSD-3 고지 포함). `pygam` 불필요 — GAM 대신 `scipy.interpolate.UnivariateSpline`(scipy 1.18) 또는 `sklearn.isotonic.IsotonicRegression`(sklearn 1.9, 설치됨) 사용.
- **EXPECT** **+0.0015** (범위 −0.0010 ~ +0.0035). 판별 실험 성격이라 분산이 크다.

### A3-2 (권장, A3-1 이후) — 정격 이상 형상: **하드 컷아웃을 SCADA 실측 디레이트로 교체**

- **SOTA** 고풍속 라이드스루(HWRT)/스톰컨트롤: 컷아웃에서 계단 정지 대신 **고풍속대에서 출력 기준을 점진 감쇠(soft cut-out)하고 재기동에 히스테리시스**를 둔다. 5PL/6PL은 정격 이상 평탄부를 표현하지 못하므로 **구간별(piecewise) 처리**가 표준.
- **EVIDENCE**
  - Petrović & Bottasso, *Wind turbine optimal control during storms*, **J. Phys. Conf. Ser. 524:012052, 2014** **(C)** `[전문미확인]` `[WORLD: AVAILABLE]`
    "**Soft cut-out approaches reduce the wind turbine power reference in strong winds**, based on offline-shaped strategies. If the shaping is adequate, loads are [bounded]".
  - Castellani 외, **ASME J. Solar Energy Engineering 141(1):014501** **(C)** `[전문미확인]` `[WORLD: AVAILABLE]`
    "This is commonly done by **raising the cut-out velocity and the high wind speed cut-in**, regulating the **hysteresis logic**."
  - **내부(I)**: 부모 측정 `median(teacher − metered)` = **+0.054 / +0.066 / +0.064** (12–14 / 14–16 / 16–18 m/s), g3 최대 +0.311, 8–10 m/s에서 −0.021.
    `research/nodes/S13-N2_label_channel.json` 원문: 12–14 m/s `mean_pc 0.927159 / mean_cf 0.831237 / mae_label 0.10076 / n 1034`; 14–16 m/s `mean_pc 0.97032 / n 455`.
    이 레인의 I-5가 **독립 경로로 같은 부호를 재확인**: unison 분위수매핑 편향 +176 / +302 kWh10m(터빈cap의 0.25 / 0.43) at 12–14 / 16–18 m/s.
- **BENCHMARK** 없음(제어전략 문헌은 벤치마크가 아니다). PCWG Share-3의 HWS(high wind shear) 범주가 가장 가깝다.
- **MIGRATION**
  - **g3에서만 직접 관측 가능하다.** `scada_unison.parquet`에서 `ws ∈ [12, 25]` 구간의 터빈별 `power` 분포에서 **가동 중인 터빈만**(vendored `bin_filter(direction='below')` 통과분) 남긴 조건부 평균 → **가용성이 제거된 디레이트 곡선**.
  - `curve()` 시그니처를 `(v, vin, vr, vout, k, v_storm, storm_floor)` 로 확장 — **`s13_n9_turbine_curve.py::curve()` 가 이미 `vstorm/storm_floor` 인자를 가지고 있어 그대로 재사용 가능.**
  - **결정적 차이**: 파라미터를 `meter_cf`가 아니라 **g3 SCADA 조건부 평균**에 적합한다. g1/g2에는 g3 형상을 `v/v_rated` 로 무차원화해 이식한다(BARAM 표준 무차원화 패턴).
  - 없는 것: g1/g2(V126)의 고유 스톰전략. 대체물은 g3(U136) 형상. **기종이 다르므로 이식은 근사다.**
- **RISK** S13-N9가 스톰램프를 이미 시도해 **−0.0003 ~ −0.0014** 로 졌다. 유일한 차이는 적합 목표(계량 → SCADA). 그 차이가 부호를 뒤집지 못하면 진다.
  또 하나: 12 m/s 이상 행은 채점행의 소수다(S13-N2: 12–14 m/s n=1,034, 14–16 m/s n=455 vs 채점행 총 41,220 → **3.6 %**). **레버가 짧다.**
- **COST** 4–6 h(A3-1 완료 후 +2 h). 설치 없음.
- **EXPECT** **+0.0006** (범위 −0.0014 ~ +0.0020).

### A3-3 (선택) — 커브족 교체 벤치마크: 4-파라미터 `clip^k` → **IEC binned / 5PL / 단조스플라인**

- **SOTA** 4종을 동일 데이터·동일 fold에서 비교하되, **PCWG 정의(NME/NMAE)로 커브 자체를 먼저 채점**하고 그 다음 downstream Total로 채점한다. **두 지표가 갈리는지가 이 레인 전체 가설의 최종 검증이다.**
- **EVIDENCE** OpenOA 소스 **(A)**, PCWG Lee 2020 **(A)**, Jing 2021 **(C)**, Villanueva 2018 **(C)**, Mehrjoo 2020 **(B)** — 위 인용 전부.
- **BENCHMARK** PCWG Share-3.
- **MIGRATION** `powercurve.py::curve()` 를 전략 패턴으로 바꾸고 4개 구현을 꽂는다.
  `sklearn.isotonic.IsotonicRegression`(설치됨)은 단조 커브를 **파라미터 없이** 준다. `scipy.interpolate.UnivariateSpline`(설치됨)은 단조 제약이 없으므로 **등온회귀 → 스플라인 평활** 순서로 조합.
  주의: **IEC binned 커브는 계단 함수**라 teacher가 미분 불연속이 된다. LightGBM 자체는 무관하지만 `pc_intra / pc_spread` 통계의 의미가 왜곡된다.
- **RISK** 현행 `clip((v−vin)/(vr−vin),0,1)^k` 는 파라미터 4개로 이미 매우 유연하다. 5PL과의 차이는 정격 부근 어깨 곡률뿐이고 그 구간은 이미 `k`가 흡수한다. **모형족 교체만으로는 거의 아무것도 안 바뀐다는 것이 가장 유력한 결과다.**
- **COST** 3–4 h. 설치 없음.
- **EXPECT** **+0.0002** (범위 −0.0005 ~ +0.0008).

---

## 5. STAGE A4 — 타깃 변환과 지지집합(support)

### A4-1 (권장, EXPECT/COST 최상위) — **HL-Gauss: 학습 시점의 타깃 스무딩** (예측 스무딩이 아니다)

> `research/nodes/S12-N8_ordinal_smoothing.json` 은 **예측분포 q를 커널로 컨볼브**해서 졌다:
> `box_0`(현행) blend **0.6361842**, `box_1` blend **0.6342889**, `delta_vs_current_best = −0.0018954`.
> HL-Gauss는 **정반대 지점**에 작용한다 — 학습 라벨을 원-핫 대신 **잘린 가우시안의 bin 적분**으로 준다.
> 전자는 결정층 적분을 흐리게 만들고, 후자는 **적합 자체의 정칙화**다. 이 저장소에서 아직 시도된 적이 없다.

- **SOTA**
  `HL-Gauss(y, h_x) = − Σ_i c_i log h_i(x)` , `c_i = (1/2Z)[erf((l_i + w_i − y)/(√2 σ)) − erf((l_i − y)/(√2 σ))]`, `Z` 는 지지집합 `[a,b]` 절단 정규화.
  실질 하이퍼파라미터는 **σ/ς 하나**(ς = bin 폭). 권장값 **σ/ς = 0.75**.
- **EVIDENCE**
  - Farebrother, Orbay, Vuong, Ali Taïga, Chebotar, Xiao, Irpan, Levine, Castro, Faust, Kumar, Agarwal, *Stop Regressing: Training Value Functions via Classification for Scalable Deep RL*, **ICML 2024, PMLR v235**, https://raw.githubusercontent.com/mlresearch/v235/main/assets/farebrother24a/farebrother24a.pdf **(A)** `[WORLD: 무관 · 일반 회귀]`
    하이퍼파라미터 원문: "**99.7 % of the samples** obtained by sampling from a standard Normal distribution should lie within three standard deviations of the mean with high confidence, which corresponds to approximately **6·σ/ς bins**. Thus, a more interpretable hyper-parameter that we recommend tuning is **σ/ς**: setting it to K/6 distributes most of the probability mass to ⌈K⌉+1 neighbouring locations for a mean value centered at one of the bins. **Unless specified otherwise, we set σ/ς = 0.75** for our experiments, which distributes mass to approximately 6 locations."
    스윕 원문: "we fix the value range [v_min, v_max] while varying the **number of bins in {21, 51, 101, 201}** and the ratio of standard deviation σ to bin width ς in **{0.25, 0.5, 0.75, 1.0, 2.0}**. Figure 11 shows that **HL-Gauss outperform Two-Hot across a wide range of σ values**, suggesting reduced overfitting due to the spread of probability mass to neighboring locations. … **the optimal value of σ seems to be independent of the number of bins**."
    효과크기: "**HL-Gauss cross-entropy loss consistently and significantly outperforms mean squared error (MSE) regression across all high-capacity Transformer**" (Atari 13종 online RL IQM, Wordle, 체스, 로보틱스 다중과제).
    bin 폭 정의 원문: "bins of width **ς = (v_max − v_min)/m**".
  - Imani, Luedemann, Scholnick-Hughes, Elelimy, White, *Investigating the Histogram Loss in Regression*, **JMLR 27, 2026**, https://www.jmlr.org/papers/volume27/24-0260/24-0260.pdf **(A)** `[WORLD: 무관 · 일반 회귀]`
    "For a **larger number of output bins, the bias is negligible**, and it increases with a small number of bins. We **bound the bias based on bin width** and further show that, **even with a small number of bins, bias can be made small by appropriately selecting a variance parameter**."
    "Imani and White (2018) proved an upper bound on the **local Lipschitz constant** for HL, showing it is **smaller than for ℓ2**; it is known that a smaller Lipschitz constant improves generalization performance with **stochastic gradient descent**."
    검증 가설 목록 원문(그 중 5번이 우리에게 핵심): "5. if **HL is more robust to corrupted targets** in the data set, and 6. if HL finds a model whose output is **less sensitive to input perturbations**."
  - Imani & White, *Improving Regression Performance with Distributional Losses*, **ICML 2018, PMLR v80** **(B)**: "We introduce a novel distributional regression loss, and similarly find it **significantly improves prediction accuracy**."
- **BENCHMARK** Atari-13 online RL(IQM Normalized Score), Wordle, 체스, 로보틱스(Farebrother 2024); JMLR 논문의 4개 회귀 데이터셋 + 대규모 시계열/가치예측.
- **MIGRATION**
  - 대상: `research/nodes/s7_more.py` 계열의 **DART 26-class multiclass**
    (`objective='multiclass', boosting_type='dart', n_estimators=400, learning_rate=0.08, num_leaves=31, min_child_samples=60, subsample=0.85, colsample_bytree=0.4, reg_lambda=3.0, n_jobs=6`),
    그리고 그 정확한 사본인 `s9_n12`, `s12_n3`, `s12_n15`, `s12_n9`, `s13_n5`.
  - **LightGBM 4.7에는 soft-label multiclass 손실이 없다.** 세 가지 이식 경로:
    (a) **샘플 복제 + 가중** — 각 행을 인접 K개 bin으로 복제하고 `sample_weight = c_i`. σ/ς = 0.75 → 실질 **6개**이므로 행 6배(78,912 × 6 ≈ 473k). LightGBM 4.7 / 6 workers로 충분.
    (b) **Frank & Hall 누적링크 분해** — 25개 `binary` 모델. 학습비용 25배.
    (c) `xgboost 3.3`(설치됨)의 커스텀 objective로 soft-label CE 직접 구현.
    → **(a)를 권장한다.** 코드 변경이 데이터 복제 함수 + `fit(X, y_int, sample_weight=w)` 한 줄뿐이다.
  - 지지집합: 현행 `NC = int(np.ceil(1.08 / W)) + 1` (`s9_n15_bin_width.py` 원문), `[v_min, v_max] = [0, 1.08]`, ς = 0.04 → **σ = 0.75 × 0.04 = 0.030**.
    **우리 보상 밴드가 ±0.06 = 1.5 bin** 이므로 σ = 0.030은 밴드 반폭의 정확히 1/2 — 우연히 잘 맞는다.
  - σ/ς ∈ {0, 0.25, 0.5, 0.75, 1.0} 5점만 fold-outside 스윕. **σ/ς = 0 이 현행을 정확히 재현하므로 중첩 모형이고 게이트가 baseline으로 되돌릴 수 있다.**
  - **깨지는 것**: Farebrother/Imani의 증거는 전부 **신경망 + SGD**에서 나왔고 Lipschitz 논거도 SGD 전제다. **GBDT에서는 그 메커니즘이 성립하지 않는다.**
    남는 논거는 오직 "**타깃 손상(corrupted targets)에 대한 강건성**"이고 — **우리 타깃은 정확히 손상되어 있다**(가용성 잡음, 채점행의 20.9 %, I-3). 그 경로로만 기대를 걸어라. receipt에 이 한정을 명시하라.
- **RISK**
  ① GBDT 증거 부재(위). ② `S12-N8`이 이미 "스무딩 계열"에서 졌으므로 리뷰어가 같은 축으로 볼 위험 — **작용 지점이 다름을 코드 diff로 증명**해야 한다(전자는 `q @ K`, 후자는 `sample_weight`).
  ③ 샘플 6배 복제는 `min_child_samples=60` 의 의미를 바꾼다(실질 10). **반드시 `min_child_samples`를 복제배수만큼 상향**하라. 안 그러면 과적합으로 진다.
  ④ DART는 drop-rate 때문에 가중 샘플에 민감할 수 있다. `boosting_type='gbdt'` 대조군을 같이 돌려라.
- **COST** 2–4 h. 설치 없음.
- **EXPECT** **+0.0012** (범위 −0.0010 ~ +0.0030).

### A4-2 (권장, 최저 비용) — **bin 폭 스윕 + 고 cf 구간만 세분화**

- **SOTA** 이산화 해상도는 이산화 편향 vs 클래스별 추정 분산의 트레이드오프. Imani et al.이 **bin 폭으로 편향 상한을 명시적으로 bound** 한다.
  우리 지표는 **±0.06 / ±0.08 계단**이므로 bin 폭 ς가 밴드 반폭에 근접하면 밴드 판정이 bin 내부에서 결정 불가가 된다.
- **EVIDENCE**
  - Imani et al., JMLR 27, 2026 **(A)**: "We **bound the bias based on bin width**"; "For a larger number of output bins, the bias is negligible, and it increases with a small number of bins."
  - Farebrother et al., ICML 2024 **(A)**: bins ∈ {21, 51, 101, 201} 스윕에서 "**the optimal value of σ seems to be independent of the number of bins**" → **bin 수와 σ는 분리해서 튜닝 가능**(A4-1과 순차 실행해도 상호작용이 작다).
  - **내부(I)**: `research/nodes/s9_n15_bin_width.py` 는 W ∈ {0.03, 0.06} 스윕을 위해 **작성되어 있으나 결과 JSON이 존재하지 않는다**(`research/nodes/` 에 `S9-N15_*.json` 없음; `ls research/nodes/*.json` 로 확인). **코드가 이미 있고 실행만 안 됐다.**
  - **내부(I-6)**: 채점행 cf 분포는 0.16 이상에서 bin당 3.3–6.2 %로 평탄 → **분위 간격 bin ≈ 균등 bin**. 다만 FICR의 y-가중은 고 cf에 **1.6–2.0배** 실린다(cf ≥ 0.9: 행 7.7 % / y-가중 14.5 %).
- **BENCHMARK** Atari-13(Farebrother), JMLR 4개 회귀셋.
- **MIGRATION**
  - `research/nodes/s9_n15_bin_width.py` 를 **그대로 실행**한다(W ∈ {0.03, 0.05, 0.06}). 이미 `NC = int(np.ceil(1.08 / W)) + 1` 로 일반화되어 있다.
  - **비균등 격자 변형(신규)**: `edges = concat(arange(0, 0.72, 0.04), arange(0.72, 1.09, 0.02))` — 저 cf는 유지, 고 cf만 절반 폭(총 클래스 18 + 18 = 36).
    근거는 I-6의 y-가중 편중과 `FICR = Σ(y·u)/(4·Σy)` 정의.
  - `harness.py`의 `ACTIONS = np.arange(0.05, 1.0801, 0.0025)` 는 손대지 않는다 — 행동 격자와 분포 격자는 독립이다.
- **RISK** W를 줄이면 클래스당 표본이 줄어 DART 400트리로는 꼬리 클래스가 비어버린다(cf ≥ 1.00 구간은 이미 행 0.09 %, I-6).
  비균등 격자는 `src/baram/decisions/expected_utility.py` 의 bin 중심 quadrature를 다시 써야 하는데, **S12-N1a에서 sub-bin quadrature 정밀화가 오히려 졌다**(coarse quadrature가 우연히 평활자로 작동했다는 기록)는 선례가 있다. 같은 함정 주의.
- **COST** **1–2 h**(코드 이미 존재) + 비균등 변형 2 h. 설치 없음.
- **EXPECT** **+0.0005** (범위 −0.0005 ~ +0.0015). **EXPECT는 작지만 COST가 최소라 비율은 최상위.**

### A4-3 (선택) — 회귀기 타깃의 **일반화 로짓 변환**(generalized logit)

- **SOTA** `γ(y; ν) = log( y^ν / (1 − y^ν) )`, ν > 0. 유계 [0,1] 변수를 실선으로 보내고 ν로 비대칭 조절. 역변환 시 Jensen 항 보정 필요.
- **EVIDENCE**
  - Pinson, *Very-short-term probabilistic forecasting of wind power with generalized logit-Normal distributions*, **JRSS-C 61(4):555-576, 2012** **(A, 원문 확보)** `[WORLD: METER]`
    "The improvements in terms of the **CRPS** criterion are significant when going from the Normal predictive densities to **GL-Normal** predictive densities, **in the order of 7.5 %**, and consistent over the evaluation period."
    점예측 대비 원문: "the more advanced approaches only propose overall improvements **up to 5 %**" (persistence 대비) → **점예측 이득은 확률예측 이득의 2/3 이하**.
    데이터: Horns Rev(덴마크), 10분 리드타임, 약 8개월.
  - Pierrot & Pinson, *Adaptive Generalized Logit-Normal Distributions for Wind Power Short-Term Forecasting*, **IEEE PowerTech 2021 / arXiv 2012.08910** **(A, 원문 확보)** `[WORLD: METER]` — ν를 적응 추정.
  - Messner et al., **Wind Energy 17(11), 2014** **(B)** `[WORLD: METER]`: "with our inverse (power-to-wind) transformation, **simpler linear regression models with censoring perform equally or better**".
- **BENCHMARK** Horns Rev(Pinson 2012), GEFCom2014 wind.
- **MIGRATION**
  - 대상은 `harness.py::_fit_pooled(A, COLS, tr, cfg['mu_params'], 'pc_true', ...)` 의 μ-모델(LightGBM `objective='l1'`).
  - 변환: `z = logit(clip(pc_true, ε, 1−ε)^ν)` 로 학습, 예측 후 `pc_hat = sigmoid(z)^(1/ν)`.
  - **문제**: 후단 `resid = cf − pc_hat` 과 `ztab` 는 **원공간 경험분위표**라 변환의 이득이 여기서 대부분 소멸한다. GL의 이득은 "예측분포 형태"에서 나오는데 우리는 **비모수 경험분위표**로 형태를 이미 데이터에서 배운다.
  - 제대로 이식하려면 잔차 분위표도 변환공간에서 만들어야 하고, 그러면 `np.clip(..., 0.0, cap_hi)`, `soft_cap`, `group_offset` 로직 전체가 바뀐다. **하류 파급이 크다.**
- **RISK** 위 이유로 **구조적 중복**이다. Pinson의 7.5 %는 **Normal(모수) 대비**이지 경험분위표 대비가 아니다. 우리 baseline이 이미 GL이 고치려는 문제를 다른 방법으로 고쳐놨다.
- **COST** 3–5 h(잔차표까지 바꾸면 8 h+). 설치 없음.
- **EXPECT** **+0.0002** (범위 −0.0010 ~ +0.0010).

### A4-4 (부결, A4-1 승리 전까지) — 순서형 분류(Frank & Hall 누적링크)로 multiclass 교체

- **SOTA** Frank & Hall (2001) 이진 분해: K−1개 `P(y > k)` 이진 분류기 → 차분으로 클래스 확률. 순서 정보를 **구조적으로** 주입.
- **EVIDENCE** Frank & Hall, *A Simple Approach to Ordinal Classification*, **ECML 2001** **(C)** `[전문미확인]`: "a simple method that enables standard classification algorithms to **make use of ordering information** in class attributes". Vargas, Gutiérrez, Hervás-Martínez, *Cumulative link models for deep ordinal classification*, **Neurocomputing 401, 2020** **(C)** `[전문미확인]`.
- **BENCHMARK** —
- **MIGRATION** 26 → 25개 LightGBM `binary` 모델. 학습비용 25배. 단조성 보장을 위해 `P(y>k)` 를 k에 대해 단조화(등온회귀)해야 한다.
- **RISK** `S12-N8`이 이미 "순서구조를 사후 주입"해서 졌다. Frank & Hall은 사전 주입이라 다르지만, **25배 비용에 EXPECT의 근원이 A4-1과 같다(순서정보)**. **A4-1이 훨씬 싸게 같은 정보를 넣는다.** A4-1이 이기면 그때 고려하라.
- **COST** 8–12 h.
- **EXPECT** **+0.0003** (범위 −0.0010 ~ +0.0012). **EXPECT/COST 최하위.**

---

## 6. EXPECT / COST 랭킹 (클러스터 내부)

| 순위 | 항목 | EXPECT (Total) | COST (h) | EXPECT/h | 설치 | 중첩(baseline 복귀) |
|---:|---|---:|---:|---:|---|---|
| **1** | **A4-2** bin 폭 스윕 (코드 이미 존재) | +0.0005 | 1–2 | **3.3e-4** | 없음 | ○ (W=0.04) |
| **2** | **A4-1** HL-Gauss 타깃 스무딩 | +0.0012 | 2–4 | **4.0e-4** | 없음 | ○ (σ/ς=0) |
| **3** | **A3-1** SCADA 적합 파워커브(계량기 아님) | +0.0015 | 6–10 | 1.9e-4 | 없음(vendoring) | ○ |
| 4 | **A2-1** 블록 단위 게이팅 | +0.0010 | 4–6 | 2.0e-4 | 없음 | ○ (L=1) |
| 5 | **A3-2** SCADA 실측 스톰 디레이트 | +0.0006 | 4–6 | 1.2e-4 | 없음 | ○ |
| 6 | **A1-1** SCADA 정합 행 신뢰도 | +0.0003 | 3–5 | 0.8e-4 | 없음 | ○ |
| 7 | **A3-3** 커브족 벤치마크(IEC/5PL/등온) | +0.0002 | 3–4 | 0.6e-4 | 없음 | ○ |
| 8 | **A4-3** 일반화 로짓 타깃 | +0.0002 | 3–5 | 0.5e-4 | 없음 | △ |
| 9 | **A2-2** 가용성 혼합 예측분포 | +0.0000 | 6–8 | 0.0 | 없음 | ○ (q=0) |
| 10 | **A4-4** 순서형 분해(Frank & Hall) | +0.0003 | 8–12 | 0.3e-4 | 없음 | ○ |
| — | A1-2 계량 오프셋 / 타임스탬프 재정렬 | 0.0000 | 2 | 0 | — | **이 레인이 닫음** |
| — | A2-3 가용전력 복원 / Tobit | ≤ −0.010 | — | — | — | **금지(2회 재현된 역전)** |

**클러스터 전체 합산 기대치(중복 제거 후): +0.0025 ~ +0.0035.**
근거: A3-1과 A2-1은 **같은 결손 신호를 다르게 소비**하므로 가산적이지 않다(A3-1이 커브오차를 걷어내면 A2-1의 0.05 임계가 재정의된다).
A4-1과 A4-2도 같은 이산화 축이라 부분 중복. 단순 합 +0.0053에서 약 40 % 상쇄를 가정했다.

**정직한 상한 진술.** 이 클러스터가 소유한 채널은 0.13858 중 0.04804 지만, I-1(완벽한 터빈단 지식으로도 채점행 잔차 0.0226 cap)과 I-4(결손률 연도 간 ±0.08 비정상)가 천장을 낮게 못박는다.
**+0.0035 를 넘는 기대치를 이 클러스터에 부여하지 마라.**

---

## 7. 이 레인이 닫는 축 (재검토 금지)

| 축 | 닫는 근거 |
|---|---|
| 라벨 타임스탬프 재정렬(±1 h, 30분, DST) | **I-2**: lag 0이 g1/g2/g3 전부 최적, lag ±1은 MAE 2배 |
| 계량기 총량 오프셋 보정 | **I-1**: meter/SCADA = 1.00354, 계량기 불확실성 0.5 %(IEC 60688:2012) 이내 → 0과 구별 불가 |
| "`actual ≥ 0.1·cap` 필터가 고장을 걸러준다"는 가정 | **I-3**: 결손 ≥ 0.05 비율 전체 13.8 % vs **채점행 20.9 %** — 오히려 채점행에 집중, 비채점행 중 고 teacher는 2.8 %뿐 |
| 정비 요일 / 캘린더로 가용성 예측 | **I-4**: 요일별 결손률 0.233–0.326 평탄, 그룹 간 월별 부호 불일치(8월 g1 0.514 vs g3 0.039) |
| 학습기간 결손률을 2025로 그대로 이식 | **I-4**: 연도 간 0.198 ↔ 0.350(절대 ±0.08) |
| 비균등 / 분위 간격 bin 격자(전 구간) | **I-6**: cf ≥ 0.16 전 구간에서 bin 점유율 0.033–0.062로 이미 평탄 (단, **고 cf 국소 세분화는 살아 있음** → A4-2) |
| 가용전력 재구성 / 등온 재보정 / Tobit | 부모가 2회 재현한 역전 + Messner et al. 2020의 명시적 세계 구분 |
| 커브의 Jensen 항(터빈별 평가 후 평균) | `powercurve.py`가 이미 터빈별·10분별 평가 후 평균 — McCandless & Haupt 2019가 요구하는 형태를 충족 |

---

## 8. BUILD ORDER

**전제.** 각 단계는 **중첩(nested)** 이어야 한다 — 파라미터 0이 현행을 정확히 재현해야 fold-outside 게이트가 baseline으로 되돌릴 수 있다.
부모가 리버스 어블레이션으로 귀속하므로 **각 단계는 독립 아티팩트를 남기고 이전 단계를 덮어쓰지 않는다.**

```
STEP 0  (0.5 h)  현행 재현 확인 — 이 단계 없이 진행하면 어떤 델타도 귀속 불가
        powercurve.py 재실행 → teacher_targets.parquet SHA-256 대조.
        pc_true 불변 확인 + baseline Total 재측정.

STEP 1  (1–2 h)  A4-2 · bin 폭 스윕                  [최저 비용 · 코드 이미 존재]
        research/nodes/s9_n15_bin_width.py 를 W in {0.03, 0.05, 0.06} 로 실행.
        산출: research/nodes/S9-N15_bin_width.json
        게이트: fold-outside 로 W 선택. W=0.04 가 이기면 그대로 두고 STEP 2 로.
        ※ W가 바뀌면 STEP 2 의 sigma(=0.75*W)도 따라 바뀐다. 순서 고정.

STEP 2  (2–4 h)  A4-1 · HL-Gauss 타깃 스무딩          [EXPECT/COST 최상]
        DART 26-class 에 샘플복제+sample_weight 방식으로 soft label 주입.
        sigma/W in {0, 0.25, 0.5, 0.75, 1.0}; min_child_samples 를 복제배수만큼 상향;
        boosting_type='gbdt' 대조군 병행.
        산출: research/nodes/S15-A4_hlgauss.json
              (sigma/W=0 행이 STEP 1 결과와 정확히 일치해야 한다 — 불일치 시 구현 버그)
        게이트: fold-outside. 지면 즉시 폐기하고 STEP 3 로.

STEP 3  (6–10 h) A3-1 · SCADA 적합 파워커브            [클러스터 본체 · 판별 실험]
        3a. OpenOA filters 5함수 + logistic5param 을
            research/vendor/openoa_min.py 로 vendoring (BSD-3 고지 포함).
            ※ pip install openoa 금지 — scikit-learn<1.7 핀이 sklearn 1.9.0 을 강등시킨다.
        3b. g3: unison SCADA 로 터빈별 커브 직접 적합
            (|P|>700 제거, unresponsive_flag(3),
             bin_filter(ws, P, 0.5, thr, 'median', 'mad', direction='below')).
        3c. g1/g2: vestas |power| 주변분위 -> f_i(v)=Q_P(F_V(v)), v <= v_rated 구간만 채택.
            정격 이상은 g3 형상을 v/v_rated 무차원화로 이식하거나 평탄 1.0.
        3d. powercurve.py 의 curve()/params 만 교체 -> teacher_targets 재생성(원본 보존).
        산출: research/nodes/S15-A3_scadacurve.json + teacher_targets_scada.parquet
        ★ 판별 규칙: 새 커브가 정격 이상에서 현행보다 낮아지고 downstream 이 지면,
          부모의 제약은 "적합 목표"가 아니라 "커브 형상" 자체에 관한 것이다.
          그 경우 STEP 5(A3-2)를 실행하지 말고 A3 전체를 닫아라.

STEP 4  (4–6 h)  A2-1 · 블록 단위 게이팅                [STEP 3 이후에만 의미 있음]
        deficit = pc_true - cf 에 이진분할/2상태 HMM (scipy+numpy; ruptures 설치 금지).
        calib_rows 를 in_outage_block(len >= L) 로 교체. L in {1,2,3,4,6}; L=1 이 현행.
        ※ STEP 3 이 커브오차를 걷어낸 뒤여야 임계 0.05 가 "진짜 고장"을 뜻한다.
        산출: research/nodes/S15-A2_blockgate.json

STEP 5  (4–6 h)  A3-2 · SCADA 실측 스톰 디레이트         [STEP 3 이 이겼을 때만]
        g3 SCADA 의 ws in [12,25] 조건부 평균(가동 터빈만)으로 디레이트 형상 측정,
        v/v_rated 무차원화로 g1/g2 이식. s13_n9 의 curve(vstorm, storm_floor) 재사용.
        산출: research/nodes/S15-A3_storm.json

STEP 6  (3–5 h)  A1-1 · SCADA 정합 행 신뢰도
        g3 전용 QC 컬럼(n_slot, n_report, resid_cf) 생성 -> sample_weight 감쇠로만 소비.
        값 교체 금지. g1/g2 는 KDD Cup 규칙(ws>4 & cf~0)만 이식.
        산출: research/nodes/S15-A1_qc.json

STEP 7  (6–8 h)  A2-2 · 가용성 혼합 예측분포            [동전 던지기 · 어블레이션 슬롯용]
        harness.py 의 samples 구성 한 줄에 2성분 혼합 추가. q=0 이 현행.
        ※ 부모가 "decision-layer 축"으로 판정하면 착수 전에 폐기.
        산출: research/nodes/S15-A2_mixture.json

STEP 8  (3–4 h)  A3-3 · 커브족 벤치마크 (IEC / 5PL / 등온+스플라인)
        PCWG 정의(NME/NMAE)로 커브를 먼저 채점하고, downstream Total 로 다시 채점.
        두 지표의 부호가 갈리는지가 이 레인 전체 가설의 최종 검증이다.
        산출: research/nodes/S15-A3_family.json

[실행 금지]  A1-2(오프셋/재정렬), A2-3(가용전력 복원/Tobit),
            A4-4(순서형 분해 — A4-1 승리 전),
            pip install openoa / ruptures / pygam.
```

**중단 규칙.** STEP 1–3 을 마쳤을 때 누적 fold-outside 델타가 **+0.0005 미만**이면 이 클러스터 전체를 닫아라.
근거: I-1(계량 상한 0.0226 cap)과 I-4(결손률 연도 간 ±0.08 비정상성)가 이 클러스터의 천장을 이미 낮게 못박았다.

---

## 9. 이 레인이 직접 열람한 원문 (등급 A / B)

| # | 출처 | URL | 형식 | 등급 |
|---|---|---|---|---|
| 1 | NREL/OpenOA `openoa/utils/filters.py` | raw.githubusercontent.com/NREL/OpenOA/main/openoa/utils/filters.py | 소스 | A |
| 2 | NREL/OpenOA `openoa/utils/power_curve/functions.py` | 〃 /power_curve/functions.py | 소스 | A |
| 3 | NREL/OpenOA `openoa/utils/power_curve/parametric_forms.py` | 〃 /power_curve/parametric_forms.py | 소스 | A |
| 4 | NREL/OpenOA `pyproject.toml` (라이선스·의존성 핀) | 〃 /pyproject.toml | 소스 | A |
| 5 | Lee et al., PCWG Share-3, WES 5:199-223, 2020 | wes.copernicus.org/articles/5/199/2020/wes-5-199-2020.pdf | PDF 25p | A |
| 6 | Optis et al., OpenOA, WES preprint 2019-12 | wes.copernicus.org/preprints/wes-2019-12/wes-2019-12.pdf | PDF 14p | A |
| 7 | Todd et al., Wind Energy 25(11), 2022 (NREL 81032) | docs.nlr.gov/docs/fy22osti/81032.pdf | PDF 16p | A |
| 8 | St. Martin et al., WES 2:295-306, 2017 (NTF) | wes.copernicus.org/articles/2/295/2017/wes-2-295-2017.pdf | PDF 12p | A |
| 9 | Pinson, JRSS-C 61(4), 2012 (GL-Normal) | pierrepinson.com/docs/pinson11_wpfore_rev.pdf | PDF 23p | A |
| 10 | Pierrot & Pinson, PowerTech 2021 / arXiv 2012.08910 | pierrepinson.com/wp-content/uploads/2021/10/2012.08910.pdf | PDF 6p | A |
| 11 | Messner et al., Int. J. Forecasting 36(3), 2020 (평가 실무) | pierrepinson.com/wp-content/uploads/2020/02/Messneretal2020.pdf | PDF 26p | A `[추출왜곡]` |
| 12 | Farebrother et al., ICML 2024 (HL-Gauss) | raw.githubusercontent.com/mlresearch/v235/.../farebrother24a.pdf | PDF 23p | A |
| 13 | Imani et al., JMLR 27, 2026 (Histogram Loss) | jmlr.org/papers/volume27/24-0260/24-0260.pdf | PDF 54p | A |
| 14 | Zhou et al., SDWPF, arXiv 2208.04360v2 | arxiv.org/html/2208.04360v2 | HTML | A |
| 15 | Liu et al., KDD Cup 2022 Workshop paper 1286 | baidukddcup2022.github.io/papers/...1286.pdf | PDF 6p | A |
| 16 | IEA Wind Task 36/51 RP Part 4, 1st ed., 2022 | iea-wind.org/.../IEAWind_Task36_Recommended_Practice_Part4_1st_Edition_public.pdf | PDF 123p | A `[추출왜곡]` |
| 17 | DNV GL, Definitions of Availability Terms, EAA-WP-15, 2017 | ourenergypolicy.org/.../Definitions-of-availability-terms-...pdf | PDF 17p | A |
| 18 | Tawn et al., Missing Data in Wind Farm Time Series, PSCC 2020 | pscc-central.epfl.ch/repo/papers/2020/720.pdf | PDF 8p | A |
| 19 | Würth et al., Energies 12:712 (IEA Wind minute-scale) | iea-wind.org/wp-content/uploads/2021/04/energies-12-00712.pdf | PDF 30p | A |
| 20 | Bel-Hadj et al., WES preprint 2025-255 (운전상태 추론) | wes.copernicus.org/preprints/wes-2025-255/...version2.pdf | PDF 31p | A |
| 21 | Bisgaard et al., Wind Energy 2026 (HMM + zero-inflated beta) | onlinelibrary.wiley.com/doi/full/10.1002/we.70110 | 초록 | B |
| 22 | Mehrjoo et al., Renewable Energy 147:214-222, 2020 | ideas.repec.org/a/eee/renene/v147y2020ip1p214-222.html | 초록 | B |
| 23 | Messner et al., Wind Energy 17(11), 2014 (inverse PC + censoring) | centaur.reading.ac.uk/47581/ | 초록 | B |

**`[전문미확인]`(등급 C) 로만 남은 것**: Jing 2021(5PL), Villanueva 2018(6PLE/3PLE/5PLE), Z. Wang 2023(Bayesian CP-quartile), Shang 2026(Energies 19:1161), Petrović 2014(soft cut-out), Castellani(ASME 141:014501), Frank & Hall 2001, Vargas 2020.
**이 태그를 지우고 인용하지 마라.**

---

## 10. 부모가 준 CRITICAL CONSTRAINT에 대한 이 레인의 최종 입장

부모의 제약은 "teacher를 실측에 가깝게 만들면 진다"였고 근거는 두 번의 재현이다.
이 레인의 판독은 그 제약을 **더 좁게** 다시 쓴다:

> **진짜 제약은 "teacher가 계량 실적을 목적함수로 보면 진다"이지, "teacher의 커브가 물리적으로 정확하면 진다"가 아닐 수 있다.**

근거:
- 실패한 두 시도(등온 재보정, 터빈 전달계수 + 스톰커브)는 **둘 다 계량 cf를 목적함수로 최소화**했다(S13-N3, s13_n9 코드 원문 확인).
- 성공적으로 유지되고 있는 현행 teacher **역시 계량 cf를 목적함수로 적합**되었다(`powercurve.py` 원문). 즉 현행도 이미 오염되어 있고, 실패한 시도들은 **같은 오염을 더 많이 주입한 것**이다.
- I-5는 계량기를 전혀 보지 않는 제3의 적합 경로가 **실제로 존재하고 검증 가능**함을 보였다(unison에서 정격 이하 편향 ≤ 0.044 터빈cap).

**따라서 STEP 3(A3-1)은 이 제약에 대한 정면 판별 실험이다.**
이기면 제약은 "목적함수"에 관한 것이고 A3-2가 열린다.
지면 제약은 "커브 형상"에 관한 것이고, **A3 전체와 이 클러스터의 절반이 닫힌다** — 그 경우에도 A4-1/A4-2는 독립적으로 살아 있다.
어느 쪽이든 **단 한 번의 실험으로 클러스터의 절반이 결정된다.** 그것이 이 레인이 STEP 3를 3순위 EXPECT/COST에도 불구하고 앞으로 당긴 이유다.
