# Lane · S13 / S5 — 데이터 전처리(preprocessing) 심층 발굴 2차

조사일: 2026-08-08 · 도구: `websearch`(Serper/Google) **110 쿼리**(영어 93 / 한국어 16 / 중국어 1) + 공개 원문 **HTTP 읽기 전용 열람 18건**
원시 로그: `S13_S5_preprocessing_deep.searchlog.json`
레인 성격: **읽기 전용**. 저장소 쓰기는 `research/lanes/` 아래 이 2개 파일뿐. 모델 fit 0회. 락박스(2024) 미열람. 업로드·git·계정 조작 없음.
저장소 읽기: `src/baram/features/spatial.py`, `src/baram/evaluation/official.py`, `src/baram/features/*`, `research/scratch/grid_coords.json`, `research/scratch/labels.parquet` 스키마, 선행 레인 문서 8건. **수정 0건.**

---

## 0. 증거 등급 규약 (이 문서 전체에 적용)

| 등급 | 정의 |
|---|---|
| **A** | 논문/보고서 **원문(PDF·HTML)을 이 레인이 직접 내려받아 읽고**, 표·본문에서 수치를 그대로 옮긴 것. 인용문은 원문 그대로(영/한/중). |
| **B** | 학술지 공식 초록 페이지 또는 기관 공식 페이지 원문을 읽은 것. 본문 표는 못 봄. |
| **C** | 검색 엔진 스니펫만 본 것 → 반드시 `[전문미확인]` 태그를 붙였다. **이 태그를 지우고 인용하지 마라.** |
| **I** | 이 레인의 저장소 내부 판독(코드·스키마). 새 계산·fit 없음. |

**A급 원문 18건**(§7에 전체 목록). PDF 텍스트 추출 시 폰트 매핑이 깨진 문서(IEA Wind Task 36 RP Part 4, Messner et al. 2020)는 인용문에 `[추출왜곡]`을 병기했다 — 글자 치환이 있으나 문장 구조와 어휘는 복원 가능했다.

**교차 확인 규칙.** 효과크기는 (a) 무엇을 무엇과 비교했는지, (b) 어떤 지표인지, (c) 어떤 데이터셋인지를 함께 적지 않으면 인용하지 않았다. "72.1 % MAE 감소" 같은 대조군 미정의 수치는 §6.3에 격리했다.

---

## 1. 선행 레인과의 관계 — 무엇을 반복하지 않았는가

이 저장소에는 이미 S5 전처리 레인(`S5_preprocessing_research.md`, 후보 P1–P16, 62쿼리)이 있다.
이 레인은 **그 문서를 먼저 전부 읽고**, 다음 세 가지만 새로 했다.

1. **문헌 등급을 C→A로 올렸다.** 선행 레인은 "PDF 전문은 내려받지 않았다"고 자기 제한했다. 이 레인은 공개 원문 18건을 실제로 읽어 **표 안의 숫자**를 확보했다. 예: KMAPP 논문의 LDAPS MBE `+1.08±1.53 m/s`, RMSE `2.32→1.82 m/s`는 선행 레인에 없던 수치다.
2. **선행 레인이 비워둔 3개 계열을 채웠다.** 선행 P1–P16에는 (5) **입력측 NWP 편의보정**, (6) **격자 축약 규칙**, (7)의 절반인 **NWP 결측 규약**이 사실상 없다(P10 valid-time 하나뿐). 이 레인의 N4·N5·N7·N8·N10이 그 자리다.
3. **부모가 새로 준 전제(“지표 채점행만으로 학습 → 1−NMAE +0.006161”)를 이론적으로 정박시켰다.** 이것은 우연한 트릭이 아니라 **절단회귀(truncated regression)의 정확한 최적성 조건**이며, 그 사실이 다음 실험(N1·N2)의 설계를 결정한다. §3.1 참조.

**선행 레인/타 레인이 이미 소유한 것(여기서 재론 안 함):** P1 물리잔차 게이팅, P2 g3 비대칭, P3/P4 지표정합 가중의 1차 형태, P5 파워커브 클리닝 기법 카탈로그, P7 소프트캡, P9 Jensen, P11 결측(라벨측), P12–P16.
`S6_feature_research.md`의 C01–C18(피처), `S6_ext_B_terrain.md`의 지형지수, `windskill_lit.md`의 C1–C15(MOS/다운스케일링 모델 계열)도 S6/S7 소관이므로 여기서는 **전처리로서 성립하는 부분만** 다룬다.

---

## 2. 이 레인이 새로 확보한 A급 증거 (핵심 10개)

| # | 출처(등급) | 원문 인용(발췌) | 효과크기 | 데이터셋/지표 |
|---|---|---|---|---|
| E1 | 금왕호·이상현 외, *Atmosphere*(한국기상학회) 31(1):85–100, 2021 — KMAPP 복잡지형 지상풍속 **(A)** | "LDAPS 예측 풍속은 … **지형 고도가 높은 지점에서 과소 모의 경향을 나타내고 지형 고도가 낮은 지점에서 과대 모의하는 경향**을 보였다"; "reducing the forecast error by **21.2%**" | LDAPS MBE **+1.08±1.53 m/s**, RMSE **2.32±0.66**; KMAPP 0.49 / 2.31; **KMAPP-Wind 0.05±1.15 / 1.82±0.36** | ICE-POP 2018 평창 산악 AWS 16지점(고도 390–1416 m), 2018-02, +36h 예보, 1시간 |
| E2 | 같은 논문 민감도 실험 **(A)** | Table 3: `C1 INTP+RA+HC(ΔH>0)` … `C6 INTP` | MBE/RMSE = C1 1.34/2.47, C2 0.24/1.91, C3 1.24/2.48, C4 1.21/2.42, **C5 0.05/1.82**, C6 1.02/2.39 | 동일 |
| E3 | Olauson·Viotti·Huss, *Int. J. Forecasting* 42(3):724–735 (HEFTCom2024 **우승**) **(A)** | "All grid points were included in X; i.e. **no spatial aggregation** was employed. … The pinball loss for solar power was increased by around 10 % (one point) and 2 % (spatial mean) … **For wind power, the increase was around 2 % and 1 %**, respectively." | 격자 축약 손실: 단일 중심셀 **+2 %**, 공간평균 **+1 %** (풍력 pinball) | Hornsea 1(1200 MW), MEPS 31격자, 2023 테스트 |
| E4 | 같은 논문 **(A)** | "it is important that **the uncertainty of the input features is represented in the training data. We therefore also used day-ahead weather forecasts for training.**" / "Rows in X for the individual models with any missing data were **dropped**" | 결측률 GFS 0.3 % · DWD 1.5 % · MEPS 1.8 %; MEPS 추가 시 풍력 pinball **−8 %** | 동일 |
| E5 | McCandless & Haupt, *Wind Energy Science* 4:343–353, 2019 **(A)** | "using a random forest … reduces the error in the wind speed to power conversion when given the predictors that quantify the differences due to **Jensen's inequality**" | 초터빈 변환 MAE **68.83 kW/2 MW기** → RF(평균+표준편차) **51.15** → RF(개별터빈 풍속) **50.41** (**−25.7 %/−26.8 %**). 모의: 11 m/s·σ=2 m/s에서 초터빈 편차 **8 %** | Shagaya 풍력(쿠웨이트, **5기**) + 100기 가상단지 |
| E6 | Messner, Pinson, Browell, Bjerregård, Schicker, *Int. J. Forecasting* 36(3), 2020 **(A, [추출왜곡])** | "ANOTHER IMPORTANT DECISION … IS WHETHER **CURTAILMENT DATA SHOULD BE KEPT OR REMOVED** FROM THE DATA BEFORE EVALUATION. … IF THE FORECAST USER IS INTERESTED IN THE AVAILABLE POWER AND NOT IN THE REAL POWER PRODUCTION, **DATA WITH CURTAILMENT SHOULD BE REMOVED FROM THE EVALUATION DATASET** SINCE ERRORS WHEN NOT PREDICTING THESE CASES ARE NOT MEANINGFUL" | 정성(효과크기 없음) | GEFCom2014 wind, 1개 단지 |
| E7 | IEA Wind Task 36/51, *Recommended Practice … Part 4*, 1st ed., 2022 **(A, [추출왜곡])** | "operational data such as **plant availability** (e.g. proportion of turbines/panels in service) and **control actions** (e.g. curtailments) **are also required as they change the nature of the power measurement** … must be calibrated to predict the variable of interest to the user: **what the actual power production is expected to be** in the future, **or** the power production **would be** expected … **if no control actions were** [taken]" / 표: "capacity in operation — WFS SCADA — **scaling forecast based on available capacity**" | 정성(권고) | 국제 권고안 |
| E8 | Zhao, Milani Fard, Narasimhan, Gupta, ICML 2019 — Metric-Optimized Example Weights **(A)** | "the weights on the training examples are **learned to optimize the test metric on a validation set**" | 회귀 과제: 테스트 지표 **52.00±0.31 → 46.33±0.16 (−10.9 %)**; 분류 과제: precision@95 %recall **20.8 %(균등) → 21.8 %(중요도가중) → 23.2 %(MOEW)**, Bayes 25 % | 공개 벤치마크 + 사내 과제 |
| E9 | 양성병 외 / KIEE(대한전기학회) — 재생에너지 발전량 예측제도 **(A)** | "예측제도 등록 기준은 직전 3개월 평균 예측오차율 10 % 이하 … **시간별 설비 이용률이 10 % 미만인 발전기는 오차율 산정에서 제외된다**"; `ε_t = |P_t − x_t| / C × 100 %` | 제도 규정(6 %/8 % 계단 정산) | 한국 전력거래소 제도 |
| E10 | Zhou & Zhai(SDWPF/Baidu KDD Cup 2022) **(A)** + 우승권 해법 **(A)** | "In some time, the wind turbines are stopped to generate power by external reasons such as **wind turbine renovation and/or actively scheduling the powering to avoid overloading the grid**. In these cases, the actual generated power … **is unknown. These unknown values will also not be used for evaluating the model.**" | 규칙표(§N10에 이식) | 134기 풍력단지 10분 SCADA |

보조 A급: Jonas et al. 2024 (arXiv 2402.13916) NRMSE **35 %→22 %**, 보정모델 MAE는 무보정 대비 **CNN 64.2 % / LSTM 65.9 % / NN 67.2 % / GB 74.5 %**;
Dupré et al. 2019 (angeo-2019-88) 공기밀도 정규화로 MAE **0.96 %→0.77 %(−20 %)**, T≥25 °C 구간만 보면 **−33 %**;
Pu et al.(팀 GEB, HEFTCom 4위) sister-model 스태킹으로 풍력 MPL **−10.84 %/−6.32 %**, 그리고 `ŷ' = min(ŷ, c_τ·Q_available)`의 절단계수 **c_τ ∈ [0.9, 1.0]**;
von Zuben & Schell(WES 2025-29) "Power curve predictions … were improved significantly by **averaging available wind speeds**";
Tawn et al.(EPSR 2020) "**Retraining a model without missing forecast inputs is preferable to imputation**";
Astolfi et al.(*Energies* 15:5225) 노후화 **−0.63 %/yr** **(C)**, D. Kim et al.(*Energy*, 2025) "**Annual energy production decreased by 0.72 % in complex** [terrain]" **(C)**.

---

## 3. 이론 정박 — 왜 "채점행만 학습"이 이겼는가, 그리고 어디까지 더 갈 수 있는가

### 3.1 절단(truncation)은 버그가 아니라 최적성 조건이다 **(I·유도)**

채점기 원문(`src/baram/evaluation/official.py::_score_group`, 40행 `valid = part.loc[part["actual_kwh"].ge(0.1 * capacity)]`)에서
NMAE 반쪽의 기여는 다음과 같다.

```
NMAE ∝ E_x[ 1{y ≥ c} · |ŷ(x) − y| ] ,   c = 0.1·cap
     = E_x[ P(y ≥ c | x) · E( |ŷ(x) − y| | x, y ≥ c ) ]
```

`P(y ≥ c | x) ≥ 0` 는 `ŷ` 에 의존하지 않으므로, **각 x 에서의 최적 행동은 절단분포 `F(y | x, y ≥ c)` 의 중앙값**이다.
전체 행으로 학습하면 모델은 `median(y | x)` 로 수렴하고, 절단분포는 그것보다 **확률적으로 크므로** 예측이 체계적으로 낮다.

> **따라서 부모가 측정한 “채점행만 학습 → 1−NMAE +0.006161”은 트릭이 아니라 추정량을 목적함수에 맞춘 결과다.**
> 그리고 이 유도는 **다음 세 가지를 즉시 함의한다** — 이것이 이 레인이 제안하는 N1·N2의 근거다.

1. **손실은 L1(또는 τ=0.5 pinball)이어야 한다.** 절단표본 위에서 L2를 쓰면 절단평균으로 가는데, 최적은 절단중앙값이다. (제곱손실로 학습 중이면 이 한 줄이 공짜 이득이다.)
2. **이득은 `P(y≥c|x)` 가 중간인 x 영역에 전부 몰려 있다.** `P≈0` 이면 채점되지 않아 무관하고, `P≈1` 이면 절단이 분포를 바꾸지 않는다. g1/g2 는 미채점행이 37 %, g3 는 45 %(선행 레인 §1.1)이므로 **경계영역이 두껍다** → 아직 남은 이득도 거기 있다.
3. **FICR 반쪽의 가중은 다르다.** `FICR = Σ_valid y·u / Σ_valid 4y` 이므로 행 가중이 `y` 에 비례한다. 두 반쪽을 합친 행 가중은 **`w_i = a + b·y_i` (채점행), `0`(미채점행)** 꼴이며, `a`(균등, NMAE) 와 `b`(생산비례, FICR) 의 비는 **점수 단위에서 0.5:0.5**로 고정된다. 저장소가 이미 쓰는 "production-proportional sample weights"는 `a=0` 극단이고, 균등가중은 `b=0` 극단이다 — **둘 다 지표가 말하는 지점이 아니다.**

### 3.2 이 절단은 한국 제도의 복사본이다 **(A: E9)**

`이용률 10 % 미만 행 제외 + 오차율 |P−x|/C + 6 %/8 % 계단`은 대회가 만든 규칙이 아니라
**전력거래소 재생에너지 발전량 예측제도의 정산 규칙 그대로**다(E9 원문 인용 §2 표).
함의 두 가지:
- 게이트는 **계측치(실측)** 에 걸린다. 예보 시점에 알 수 없다 → `P(scored|x)` 를 추정하는 것 외에 방법이 없다(N2).
- 문헌에서 "가용출력(available power)을 예측하라"는 권고(E6·E7)는 **우리에게 그대로 적용되지 않는다.** 우리 지표는 **실계측 발전량**을 채점한다. 아래 N3의 핵심 위험이 여기서 나온다.

---

## 4. 노드표 (순위: 기대이득 × 근거강도 ÷ 비용)

`Δ` 는 이 레인의 **기대치 추정**이며 측정치가 아니다. 단위는 대회 Total(=0.5(1−NMAE)+0.5·FICR).
목표는 부모가 준 `+0.014839`.

| # | 노드 | 메커니즘 요약 | 증거 + 효과크기 | 왜 이전되나 | 실패 양식 | 비용 | 게이트(측정할 것) | 기대 Δ |
|---|---|---|---|---|---|---|---|---|
| **N1** | **두 게이트의 결합 재탐색** (지표게이트 × 가용성게이트) | 학습표본을 `{y ≥ 0.1cap} ∩ {gap < g*}` 로 두고 **`g*` 를 지표게이트 적용 후 다시 스윕**. 두 게이트는 같은 행을 상당부분 공유하므로 기존 `g*` 는 더 이상 최적이 아니다 | 지표게이트 단독 **+0.006161**(부모 측정, I). 가용성 게이팅의 이득방향은 선행 레인 §1.4·§1.8(I). 결합 상호작용은 **아무도 측정한 적이 없다** | 두 게이트 모두 **라벨 기준 선택**이고, `y<0.1cap` 행과 완전정지 행은 대부분 같은 행 → 겹침이 크면 두 번째 게이트의 최적 강도가 이동한다 | 겹침이 100 %면 이득 0. `g*` 를 3 fold에서 다시 고르면 자유도 소모 | **낮음**(표본 마스크만, 재학습 6~10회) | 3 fold 전부에서 Total 상승 + `g*` 가 fold 간 인접격자만 이동 | **+0.002 ~ +0.005** |
| **N2** | **하드컷 → 절단정합 추정량** (soft-truncation + 절단중앙값 + `P(scored|x)` 분해) | ① 손실을 L1/pinball(0.5)로, ② 가중 `w_i = σ((y_i−c)/τ)` 로 완화, ③ 이득을 `P̂(scored\|x)` 십분위로 분해해 보고 | 유도 §3.1 (I). MOEW(E8): 지표정합 가중으로 회귀지표 **−10.9 %**; 균등 20.8 %→ 중요도가중 21.8 %→ 학습가중 23.2 % | 부모의 +0.006161이 절단효과임이 확정되면, **같은 축의 남은 절반**(손실형·soft weight)이 곧바로 열린다 | τ 를 fold에서 고르면 자유도 1 추가. L1 전환이 FICR 반쪽을 해칠 수 있음(밴드는 최빈값을 원한다) | **낮음**(손실·가중 교체) | 십분위 분해표에서 `P̂∈(0.2,0.8)` 구간 개선이 전체 개선의 과반일 것. 아니면 메커니즘 오판 | **+0.002 ~ +0.006** |
| **N3** | **결손행 라벨 복원 후 재중심화**(제거의 대안) | `gap ≥ g*` 행의 라벨을 `y_avail = pc_physics` 로 **치환**해 표본을 유지하고, 그룹별 상수 1개로 재중심화 | E6(제거는 **평가**에서), E7("available capacity로 예보를 스케일"), Bruninx 2026(제거), GEB(E: `min(ŷ, c_τ·Q)`), HEFTCom 우승팀은 **curtailed energy를 타깃에 되더함** | 표본을 6~9 % 잃지 않으면서 하향편의를 없앨 수 있다. g3는 표본이 절반이라 손실이 특히 아프다 | **우리 지표는 실계측을 채점한다**(§3.2). 복원 타깃으로 학습하면 2025 결손시간에서 예측이 위로 뜨고 **NMAE 반쪽이 직접 손해**. 결손행 FICR 기여는 이미 0이라 얻을 것도 그쪽엔 없다 | 중간 | `pc_physics` MAE(0.032~0.045)가 치환값에 그대로 섞인다 → 치환행 비율 × 그 MAE 가 총 NMAE에 더해지는 양을 사전 계산하고 그보다 이득이 커야 함 | **−0.002 ~ +0.003** (부호 불확실) |
| **N4** | **KMAPP 폐형 고도보정을 LDAPS 입력에 적용** | `u_HC(z) = u(h_HC)·exp(σ·ΔH)`, `σ = (A/S)/(H/2)`, `ΔH = z_site − z_model`. `H/2`·`A/S` 는 공개 DEM으로 LDAPS 셀당 15×15(100 m) 창에서 산출(KMAPP-Wind 방식). **학습 자유도 0** | **E1·E2 (A)**: 복잡지형 RMSE **2.32→1.82 m/s(−21.2 %)**, MBE **+1.08→+0.05**. 민감도: 고도보정만(C5) 최적, 거칠기보정 추가는 악화(C2 1.91) | 우리 부지는 **능선 1078 m**, LDAPS 1.5 km 지형은 평활화로 산악을 낮게 본다 → E1이 관측한 "고지대 과소모의"가 정확히 우리 조건 | ① 저장소의 셀별 2모수 보정이 **평균 수준은 이미 먹었다**(S6 §1.4) → 남는 것은 상호작용뿐 ② KMAPP는 협곡에서 **과대보정**했다(C1 MBE −1.34) ③ DEM은 외부자료 → 라이선스·취득일 영수증 필요 | 중간(DEM 취득+셀당 지형모수 산출) | 보정 전후로 (a) 허브고도 풍속 대리지표의 편의, (b) `pc_prior` MAE, (c) Total. **셀별 2모수 보정을 끈 대조군에서도 이득이 남는가**를 반드시 함께 | **+0.000 ~ +0.004** |
| **N5** | **시간내 스미어링**: `f(E[v])` → `E[f(v)]` 를 **추론시에도** | 시간평균 풍속 하나가 아니라 시간내 분포 `N(v̄, σ_intra²)` 를 가정해 파워커브를 컨볼루션. `σ_intra` 는 (풍속, 안정도, 계절)의 학습기간 SCADA 조건부 회귀 또는 경험적 난류 PSD로 산출 | **E5 (A)**: 초터빈 변환 MAE **68.83→51.15 kW/2 MW기(−25.7 %)**, 평균+표준편차만으로 대부분 회수. 선행 S5 §1.3(I): `f(mean v)` 는 곡선적분 대비 MAE **+18~46 %**, 개별시간 최대 **0.68 cf**. 손은국 외, *풍력에너지저널* 16(2):48–57, 2025 **(B)**: LDAPS 1시간 해상도가 단기 변동성 모의를 제약, 경험적 PSD로 보완 시 MAE·RMSE 유의 감소 | 우리 라벨은 **시간평균 발전량**, 입력은 **시간대표 풍속 1개** → Jensen 항이 구조적으로 존재하고 이미 내부 측정됐다 | ① `σ_intra` 자체가 예측 대상이 되어 새 오차원을 들여옴 ② 밴드폭 0.06 대비 스미어 보정량이 작으면 순효과 미미 ③ 저장소 S6가 `pc_smear` 를 **피처로** 이미 검토 중 → 중복 위험(피처 아니라 **입력 변환**으로 구현해야 구분됨) | 중간 | 스미어 적용 전후 `pc_prior` 의 조건부 편의(풍속 구간별), 그리고 cf 0.4–0.8 구간 unit-4 비율 | **+0.001 ~ +0.004** |
| **N6** | **노후화 추세 정규화**(타깃 디트렌딩) | 학습 라벨을 `y / (1 − δ·(t − t_2025))` 로 되돌려 **2025 등가**로 정렬. `δ` 는 문헌값 고정(0.0063~0.0072 /yr) 또는 그룹별 1 dof 추정 | Astolfi et al. *Energies* 15:5225 **−0.63 %/yr (C)**; D. Kim et al. *Energy* 2025 "**AEP decreased by 0.72 % in complex** [terrain]" **(C)**. 우리 실측 상한 {0.985, 0.989, 1.005}(선행 §1.7, I)와 정합 | 학습 2022–24, 평가 2025 → g1/g2는 **최대 3년치 열화**가 외삽 구간에 있다. `δ=0.0065` 면 약 **−2 %** 수준 편의 | ① 열화와 정비·개보수 회복이 뒤섞여 단조롭지 않음 ② g3는 2023 신설이라 초기 running-in 과 방향이 반대 ③ 부모가 이미 **global affine rescaling 을 닫았다** → 이 노드는 *전역 스케일*이 아니라 **시간의존 스케일**이어야 구분된다 | **매우 낮음** | 연도별 잔차평균의 단조성(2022→2023→2024 중 열람 가능한 것만), 그리고 δ 부호가 세 fold에서 일치 | **+0.000 ~ +0.003** |
| **N7** | **격자 축약 규칙 교체**: 순수 1/d² IDW → (고도인지 가중 \| 전셀 유지 + 순서통계) | `src/baram/features/spatial.py`는 **수평거리만**으로 IDW(`raw = 1/max(d,0.1)²`)를 만든다(I). 능선에서 정보축은 **셀 지형고도**다. ① `w ∝ exp(−|z_cell − z_hub|/h)` 로 교체, 또는 ② 축약하지 말고 전셀 + 순서통계 | **E3 (A)**: 풍력에서 **단일셀 +2 %**, **공간평균 +1 %** pinball 손실(전셀 대비). von Zuben 2025 **(A)**: "improved significantly by averaging available wind speeds" | 4×4 LDAPS 상자(격자 좌표 실측: 위도 37.2607–37.3032, 경도 128.9257–128.9958, `grid_coords.json`, I)는 11 km 능선을 가로지른다. 거리가 같아도 **셀 지형고도는 수백 m 다르다** | ① E3가 말하는 이득 폭이 **1~2 %(pinball)**로 작다 → 우리 Total 환산 시 미미할 수 있음 ② S6 레인이 `grid_frame` 4×4 재배열 **결함**을 이미 보고했다(C01) → 그 버그 수정이 선행되지 않으면 이 노드의 측정이 오염된다 | 낮음(①) / 중간(②는 DEM 필요) | 축약 규칙만 바꾼 A/B에서 3 fold Total. **버그 수정 후에** 측정 | **+0.000 ~ +0.002** |
| **N8** | **NWP valid-time 규약 확정** (선행 P10 승계, 아직 미해결) | 라벨측은 hour-ending 확정(선행 §1.2). **NWP 측 시각 라벨이 순간값인지 시간평균인지, 시간-시작인지 시간-종료인지** 미확인. ±1h 시프트 격자 + 인접시각 평균(0.5·t + 0.5·t−1)까지 4셀 스캔 | HEFTCom 우승팀(E3, A): "lagged features are **more important than raw (64 % vs 30 %)**" — 시간 이웃이 정보의 대부분을 지닌다. 이는 정렬 오차 또는 시간평균 불일치의 전형적 징후 | 잘못된 1시간은 라벨측 스캔에서 MAE를 **2.1~2.5배** 악화시켰다(선행 §1.2, I) → NWP측에 같은 오정렬이 있으면 즉시 대형 손실 | 이미 정렬이 맞다면 이득 0(그러나 **닫는 것 자체가 가치**) | **매우 낮음** | 4셀 스캔에서 0-시프트가 단봉 최적인지. 최적이 t−0.5h(=인접평균)면 시간평균 불일치가 확정 | **+0.000 ~ +0.006**(이분적) |
| **N9** | **지표대수 그대로의 혼합가중** `w_i = a + b·y_i` | §3.1-3의 유도 그대로. `λ∈{0, 0.25, 0.5, 0.75, 1}` 로 두 극단 사이를 스윕하되 **λ=0.5 를 사전선언 앵커**로 | E8 (A): 지표정합 가중으로 −10.9 %. 저장소는 이미 생산비례(=λ=1) 채택 → **반대 극단만 채택돼 있다** | 지표가 문자 그대로 두 반쪽의 가중을 지정한다. 추측이 아니다 | 고출력 편중으로 저·중출력 밴드 적중 붕괴. 소프트캡(P7)과 상호작용 | **매우 낮음** | 1−NMAE 와 FICR 을 **분리 보고**. Total만 보면 원인을 잃는다 | **+0.000 ~ +0.002** |
| **N10** | **결측·이상 규약의 명문화 및 이식** | ① NWP 결측행은 **대치 말고 소스별 모델에서 드롭 + 스택이 메움**(E4) ② 라벨/SCADA 이상행은 KDD Cup 규칙표를 이식(§6.2) | **E4 (A)**: "Rows … with any missing data were **dropped**"; 결측률 0.3/1.5/1.8 %. Tawn et al. 2020 **(B)**: "**Retraining a model without missing forecast inputs is preferable to imputation**". Xie/Wang 2024(arXiv 2403.03631) **(A)**: "Deletion is suitable for **short periods** … inadequate for **sporadically distributed** missing values" | 우리도 소스가 2개(LDAPS/GFS)라 스택이 결측을 메울 수 있는 구조다 | 우리 데이터 결측률이 이미 0에 가깝다면 무의미 → **먼저 결측률을 재라** | 낮음 | 소스별·변수별 결측률 표 1장. 0.5 % 미만이면 노드 폐기 | **+0.000 ~ +0.001** |
| **N11** | **g3 비대칭을 절단이 아니라 미세조정으로**(선행 P2 재설계) | g1/g2 포함 전체로 학습 → **g3 표본만으로 소학습률 미세조정**. 절단(2023-10 제거)보다 표본 보존 | Jonas et al. 2024 **(A)**: 전략1(구 2년 모델) < 전략2(신 6개월만 학습) < **전략3(미세조정)**; "Despite being trained on **significantly less data**, the models trained with this strategy show a **superior performance to the original models**" | g3는 학습연도 1년, 그것도 초기운전연도 → 절단하면 표본이 더 준다. 미세조정은 표본을 버리지 않는다 | GBDT에는 층 동결이 없다 → 잔차부스팅(warm start) 또는 그룹별 오프셋으로 대체 구현해야 함. 저장소의 **DART 증강**이 이미 부분적으로 이 역할일 가능성 | 중간 | g3 FICR 상승 + g1/g2 |Δ| < 0.001 | **+0.001 ~ +0.003** |
| **N12** | **공기밀도 등가풍속으로 입력 정규화** | `v_ρ = v·(ρ/ρ₀)^{1/3}`, `ρ = p/(R·T)` 를 **파워커브 입력에** 적용(열 추가 0) | Dupré et al. 2019 **(A)**: MAE **0.96 %→0.77 %(−20 %)**, **T≥25 °C 구간 −33 %**, 저온(≤5 °C) 발생빈도 10.7 % | 부지 1078 m·강원 산악 → `ρ` 가 겨울 ≈1.16, 여름 ≈1.04 kg/m³ 로 **±5 %** 흔들린다(고도 1078 m, p≈890 hPa 가정, 이 레인 산술) | S6 레인 B-10과 **중복**. 전처리로 성립하려면 *피처 추가*가 아니라 *입력 치환*이어야 함 | 매우 낮음 | 치환 전후 `pc_prior` 계절별 편의 | **+0.000 ~ +0.002** |
| **N13** | *(닫음)* 입력 풍속의 **분위수매핑(QM)** | 분포정합으로 NWP 풍속을 관측분포에 사상 | Maraun, *J. Climate* 26:2137–2143, 2013 **(B)**: QM이 규모 불일치를 메우려 할 때 **인플레이션 문제**; `windskill_lit` C10이 이미 음의 결과로 판정. Maciel-Tiburcio 2025 **(C)**의 "QM 최저오차"는 **자원평가(연간 CF)** 결과이지 시간별 예측이 아니다 | — | 시간별 MAE를 **악화**시킬 수 있음(분산 팽창) | — | **제안하지 마라.** 필요시 근거는 위 두 줄 | — |
| **N14** | *(저순위)* 터빈수준 파워커브 클리닝(DBSCAN/RANSAC/LOF…) | 산점도 기반 이상제거 | 선행 P5가 이미 카탈로그화. 이 레인 추가: Italiano et al. arXiv 2607.13544 **(A)** "cluster-based methods … achieving **higher accuracy than manual filtering**" (탐지 정확도이지 예측 정확도 아님) | — | VESTAS 12기 `power` 축 파손 → g1/g2 적용 불가. 목적도 다르다(그룹 시간라벨 정제) | — | P1의 `g*` 자동선택 보조로만 | — |
| **N15** | *(권고: 하지 마라)* 리드타임별 표본 분리 학습 | 11–35 h 를 구간별 별도 모델 | Wessel et al., *QJRMS* 150, 2024 **(C)**: lead-time-**continuous** 모델이 "similar and, **in small data situations, even improved** performance compared with the classical lead-time-**separated** models" `[전문미확인]` | — | 표본을 24등분하면 g3는 한 리드타임당 ~365행 | — | 리드타임은 **피처로** 주고 분리하지 마라 | — |

---

## 5. 상위 5개와 단일 최강 권고

### 5.1 Top-5

| 순위 | 노드 | 한 줄 이유 | 선행조건 |
|---|---|---|---|
| **1** | **N1 두 게이트 결합 재탐색** | 부모가 이미 **+0.006161**을 얻은 바로 그 축이고, 두 게이트의 **상호작용은 아직 한 번도 측정된 적이 없다**. 표본 마스크만 바꾸므로 하루 안에 끝난다 | 없음 |
| **2** | **N2 절단정합 추정량(L1 + soft-truncation + `P̂` 분해)** | §3.1 유도가 "왜 이겼는지"를 확정하므로 **같은 축의 남은 절반**이 이론적으로 보장된다. 특히 **손실이 L2라면 L1 전환은 공짜** | 현재 손실 형태 확인 |
| **3** | **N8 NWP valid-time 규약 확정** | 비용이 사실상 0인데 결과가 **이분적**이다(오정렬이면 +0.006 급, 아니면 0). 아직 열려 있는 유일한 "큰 구멍" | 없음 |
| **4** | **N4 KMAPP 폐형 고도보정** | 우리 부지 조건(능선 1078 m, LDAPS 1.5 km)과 **정확히 같은 조건에서 측정된 −21.2 % RMSE**. 한국 기상청 모델 자체에 대한 연구라는 점에서 이전성이 최상 | 공개 DEM 취득 + 영수증(라이선스·공개일·취득일) |
| **5** | **N5 시간내 스미어링(추론시)** | 내부 측정(최대 0.68 cf)과 외부 측정(−25.7 % MAE)이 **같은 방향으로 큰 값**을 가리키는 유일한 항목 | `σ_intra` 조건부 모형 |

### 5.2 단일 최강 권고 — **N1 + N2 를 하나의 실험격자로 묶어 먼저 실행**

**왜 이것인가.**
부모가 준 7개 전제 중 유일하게 “**아직 소진되지 않았다**”고 명시된 축이 지표정합 행선택이고,
이 레인이 §3.1에서 그 축의 **정확한 최적성 조건**을 유도했다. 유도가 말하는 것은 세 가지다 —
(a) 최적 예측은 **절단분포의 중앙값**이므로 **L1 계열 손실**이어야 하고,
(b) 남은 이득은 `P(scored|x)` 가 중간인 **경계영역에 국한**되며,
(c) 두 반쪽의 행 가중은 `a + b·y` 로 **지표가 문자 그대로 지정**한다.
이 셋은 전부 **표본 마스크·가중·손실만 바꾸면 되는 조작**이라 모델·피처·정책을 건드리지 않는다.
즉 **저장소의 다른 모든 결정을 고정한 채** 측정할 수 있는, 이 단계에서 가장 싼 실험이다.

**구체적 격자(예시).**

```
행선택   S ∈ { all , y≥c , (y≥c) ∧ (gap<0.10) , (y≥c) ∧ (gap<0.15) , soft: w=σ((y−c)/τ) }
손실     L ∈ { 현행 , L1/pinball0.5 }
행가중   λ ∈ { 0(균등) , 0.5(지표대수 앵커) , 1(현행 생산비례) }
```
5×2×3 = 30셀이지만, **1단계는 S×L 10셀만** 돌리고 λ는 승자 위에서만 스윕한다(자유도 절약).

**반드시 함께 보고할 것.**
1. `1−NMAE` 와 `FICR` **분리 수치**(Total만 보면 원인을 잃는다 — 선행 레인 §1.8이 두 반쪽이 **반대 방향**으로 움직이는 사례를 이미 측정했다).
2. **`P̂(scored|x)` 십분위별 이득 분해.** 유도가 맞다면 개선의 과반이 `P̂∈(0.2,0.8)` 에서 나와야 한다. 그렇지 않으면 메커니즘 오판이므로 즉시 중단.
3. `0.10 ≤ cf < 0.20` 구간의 unit-4 비율(경계 붕괴 감시지표).
4. fold-outside 선택으로 고른 `g*`·`τ` 가 fold 간 **인접격자만** 이동하는지.

**주된 위험(정직하게).**
- **자유도 인플레이션.** 락박스가 소진돼 독립 검증면이 없다(AGENTS.md). 30셀 격자를 dev-2023 단일면에서 고르면 그 자체가 과적합이다. → **1단계 10셀 + 사전선언 앵커(λ=0.5, τ=0.02cap)** 로 자유도를 묶고, 승자 셀을 `reports/` 에 **예측선언(predeclaration)** 으로 고정한 뒤 확인하라. 저장소는 `reports/n513_m263_predeclaration.json` 선례가 있다.
- **L1 전환이 FICR을 해칠 수 있다.** 밴드 보상은 중앙값이 아니라 **최빈값/밴드확률 최대점**을 원한다(S6 O3). NMAE 반쪽이 절단중앙값을, FICR 반쪽이 밴드-argmax를 요구하므로 **두 요구가 갈린다**. 이 갈림은 선행 §1.8에서 g3가 `평균편의 +0.0227` 인데 `밴드최적 오프셋 +0.020` 이라는 **부호 역전**으로 이미 관측됐다. → 그래서 §5.2의 게이트 1번(분리 보고)이 선택이 아니라 **필수**다.
- **이득이 이미 회수됐을 가능성.** 부모의 +0.006161이 절단효과의 **전부**였다면 N2의 잔여는 0이다. 그 경우에도 (b)의 십분위 분해가 “왜 없는지”를 확정해 주므로 축을 **닫는 값**은 남는다.

---

## 6. 부록

### 6.1 왜 "가용출력으로 타깃을 복원하라"는 문헌 권고를 그대로 따르면 안 되는가 (중요)

문헌은 두 갈래를 말한다.
- **E6(Messner et al. 2020, A)**: 사용자가 **available power** 에 관심 있으면 curtailment 행을 **평가에서 제거**하라.
- **E7(IEA Wind Task 36/51 RP Part 4, A)**: 예측 시스템은 "actual power production" **또는** "power production … if no control actions were [taken]" 중 **사용자가 원하는 쪽으로 보정**되어야 한다.
- **HEFTCom2024 우승팀(E3, A)**: 주최자 지시에 따라 **curtailed energy 를 타깃에 되더했다**. 즉 그 대회의 정답은 *potential generation* 이었다 — "Note that y includes curtailed energy as per instructions from the transmission system operator. **This made the forecasting problem less difficult**".

**우리 대회는 셋 다 아니다.** 채점기는 계측 발전량 `actual_kwh` 를 그대로 쓰고(I), 결손행을 평가에서 빼주지 않는다.
따라서 **가용출력 복원은 "정답 정의"를 바꾸는 조작이 아니라, 우리 정답에 대해 편의를 주입하는 조작**이다.
선행 레인이 측정한 사실 — 결손시간의 FICR 기여가 **정확히 0.000000** — 때문에 FICR 반쪽에서 잃을 것은 없지만,
**NMAE 반쪽에서는 결손시간의 오차가 그대로 카운트된다.** 부모의 전제 1(“병목은 1−NMAE”)과 결합하면
**N3의 기대부호는 음(−)일 가능성이 실질적으로 있다.** 그래서 N3는 순위 3위가 아니라 표에서 부호 불확실로 남겼다.

### 6.2 이식용 이상치 규칙표 (Baidu KDD Cup 2022 우승권 해법 원문, A)

> "Filter the abnormal data by setting the attributes of the abnormal data records other than TurbId and Day to Nan. The abnormal conditions include:
> `Patv<0` · `Wspd<1 and Patv>10` · `Wspd<2 and Patv>100` · `Wspd<3 and Patv>200` · `Wspd>2.5 and Patv==0` · `Wspd==0 and Wdir==0 and Etmp==0` · `Etmp<-21` · `Itmp<-21` · `Etmp>60` · `ITmp>70` · `Wdir>180 or Wdir<-180` · `Ndir>720 or Ndir<-720` · `Pab1>89 or Pab2>89 or Pab3>89`"
> — 그리고 "the total number of Patv less than 0 or equals to Nan is **1,312,580**, which is **582,716** after the above three steps"(=**−55.6 %**).

우리 데이터로의 번역: `Wspd>2.5 and Patv==0` 이 곧 **선행 레인의 “SCADA ws ≥ 5 m/s인데 cf = 0”**(g1 93 h / g2 153 h / g3 41 h)이다. 즉 **우리는 이미 이 규칙의 가장 중요한 한 줄을 독립적으로 재발견해 두었다.** 남은 것은 `Pab`(피치각) 계열인데, 유니슨 SCADA에 피치각이 있으면 **정지·제한 판정의 독립 증거**가 되어 `g*` 스윕의 자유도를 없앨 수 있다 → **N1의 부속 점검항목으로 추가하라.**

### 6.3 격리 구역 — 인용하면 안 되는 수치

- "RANSAC 기반 클리닝으로 **MAE −72.1 %**"(Yang et al., *Sci. Rep.* 15:5105) `[전문미확인]` — 대조군이 "오염 데이터로 학습·오염 데이터로 평가"일 가능성이 매우 높다. **이득 추정에 쓰지 마라.**
- "다운스케일링 보정으로 **RMSE 개선율 83.97 %**"(Liu et al., *Atmosphere* 15:1090) `[전문미확인]` — 개선"율" 정의가 표준 RMSE 감소율과 다를 개연성.
- "클리닝으로 **15.4 % 더 정확**"(Subuh 2025) — **탐지 성능**이지 예측 성능이 아님.
- Maciel-Tiburcio 2025(*Renewable Energy* 247) "quantile mapping shows lower errors" `[전문미확인]` — **연간 용량계수 추정(자원평가)** 이지 시간별 예측이 아니다. N13의 근거로 오용 금지.

### 6.4 이 레인이 명시적으로 **닫는** 축

| 닫는 것 | 근거 |
|---|---|
| 입력 풍속 분위수매핑(QM)/분산 인플레이션 | Maraun 2013 (B) + `windskill_lit` C10 (선행) + 자원평가↔예측 혼동 정리(§6.3) |
| 터빈수준 파워커브 클리닝의 g1/g2 확장 | VESTAS `power` 시간축 파손(선행 §S 제약) — 데이터가 없다 |
| 리드타임 분리 학습 | Wessel et al. 2024 (C) + g3 표본수 산술(리드타임당 ~365행) |
| "가용출력을 정답으로" 재정의 | §6.1 — 우리 채점기는 계측치를 채점한다(I, 원문 확인) |
| 월별 가용성 사전분포 피처 | 선행 레인 §1.5가 이미 스퓨리어스로 판정 |

### 6.5 이 레인이 확인하지 못한 것 (정직한 공백)

1. **저장소의 현재 손실 형태**(L2인지 L1인지 pinball인지)를 코드에서 끝까지 확인하지 못했다 — N2의 (a)항 기대이득은 이 확인에 조건부다.
2. **NWP 결측률**을 실제로 세지 않았다(모델 fit 금지 범위는 아니나, 시간 배분상 문헌 우선). N10은 그 수치 없이는 순위를 확정할 수 없다.
3. **LDAPS 셀별 모델 지형고도**는 대회 데이터에 없다(`grid_coords.json` 은 위경도만, I). N4는 공개 DEM으로 **근사**해야 하며, 그 근사오차는 측정되지 않았다.
4. Maciel-Tiburcio 2025, Astolfi 2022, D. Kim 2025, Wessel 2024, Tawn 2020 은 **전문 미열람**(출판사 차단). 각각 `[전문미확인]`/(B)/(C)로 표기했다.
5. 국내 대회 상위 해법의 전처리 기록은 **찾지 못했다** — 본 대회(BARAM 2026)는 진행 중이고, 과거 회차의 코드/후기가 공개 검색으로 잡히지 않았다(관련 쿼리 6건, 전부 공지·보도자료만 반환).

---

## 7. 원문 열람 목록 (A급 증거의 출처)

| 문헌 | URL | 열람 |
|---|---|---|
| 금왕호 외 2021, KMAPP 복잡지형 지상풍속 (*Atmosphere* 31-1) | https://j-komes.or.kr/xml/28740/28740.pdf | 전문 PDF |
| Olauson et al. 2026, HEFTCom2024 우승 (*IJF* 42-3) | https://www.diva-portal.org/smash/get/diva2:2046959/FULLTEXT01.pdf | 전문 PDF |
| Pu et al. 2025, 팀 GEB (*IJF*) | https://arxiv.org/html/2505.10367v1 | 전문 HTML |
| McCandless & Haupt 2019 (*WES* 4:343) | https://wes.copernicus.org/articles/4/343/2019/ | 전문 HTML |
| Messner et al. 2020, 풍력예보 평가 (*IJF*) | http://pierrepinson.com/wp-content/uploads/2020/02/Messneretal2020.pdf | 전문 PDF |
| IEA Wind Task 36/51 RP Part 4 (2022) | https://iea-wind.org/wp-content/uploads/2022/06/IEAWind_Task36_Recommended_Practice_Part4_1st_Edition_public.pdf | 전문 PDF |
| Zhao et al. 2019, MOEW (ICML) | https://proceedings.mlr.press/v97/zhao19b/zhao19b.pdf | 전문 PDF |
| Jonas et al. 2024, SCADA 편의보정+연속학습 | https://arxiv.org/html/2402.13916v1 | 전문 HTML |
| Xie et al. 2024, 확률예측의 결측값 | https://arxiv.org/html/2403.03631v1 | 전문 HTML |
| Zhou et al. 2022, SDWPF 데이터셋 | https://arxiv.org/html/2208.04360v2 | 전문 HTML |
| Liu 2022, KDD Cup 2022 해법(trymore) | https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_1286.pdf | 전문 PDF |
| Italiano et al. 2026, SCADA 군집 필터링 | https://arxiv.org/html/2607.13544v1 | 전문 HTML |
| von Zuben & Schell 2025, 최소 공개데이터 부분집합 (WES 프리프린트) | https://wes.copernicus.org/preprints/wes-2025-29/wes-2025-29.pdf | 전문 PDF |
| Dupré et al. 2019, 공기밀도 정규화 (ANGEO 프리프린트) | https://angeo.copernicus.org/preprints/angeo-2019-88/angeo-2019-88.pdf | 전문 PDF |
| Zhou & Esau 2026, 풍속계열 최소 길이 (WES 프리프린트) | https://wes.copernicus.org/preprints/wes-2025-25/wes-2025-25.pdf | 전문 PDF |
| 양성병 외, 예측제도 기반 ESS 연계 태양광 (KIEE) | http://www.tkiee.org/kiee/XmlViewer/f415504 | 전문 HTML |
| 손은국 외 2025, LDAPS 풍속 시간해상도 (*풍력에너지저널* 16-2) | https://koreascience.kr/article/JAKO202523236003791.page | 초록 HTML |
| Maraun 2013 (*J. Climate* 26:2137) | https://journals.ametsoc.org/abstract/journals/clim/26/6/jcli-d-12-00821.1.xml | 서지 HTML(초록 미추출) |

기타 스니펫 근거 URL은 §2·§4 본문과 검색 로그에 있다.

---

## 8. 검색 로그 요약

- 총 **110 쿼리**. 언어: **영어 93 · 한국어 16 · 중국어 1**(로그에서 기계 집계).
- 계열별 배분(수작업 분류, 근사): 타깃/라벨 조건화 13 · SCADA 잡음제거 8 · 지표정합 표본선택 17 · 분포이동/시간조건화 13 · NWP 편의보정 12 · 공간·시간 리샘플링 11 · 이상치/결측 10 · 대회 write-up 14 · 한국 문헌·제도 12.
- 전체 원시 응답은 `S13_S5_preprocessing_deep.searchlog.json` (쿼리·시각·반환 텍스트 전문).

## 9. 준수 확인

- 저장소 쓰기: `research/lanes/S13_S5_preprocessing_deep.md`, `research/lanes/S13_S5_preprocessing_deep.searchlog.json` **2건뿐**.
- 모델 fit: **0회**. 락박스(2024): **미열람**. 예측 생성·제출: **없음**. git 조작: **없음**. 계정/브라우저 조작: **없음**.
- 외부 접근: `websearch`(Serper) + 공개 문서에 대한 **읽기 전용 HTTP GET**. 어떤 원격 추론 API도 호출하지 않았다(규칙 R 무관).
- 이 문서의 모든 수치는 (A) 원문 표/본문, (B) 공식 초록, (C) 스니펫 `[전문미확인]`, (I) 저장소 판독 중 하나로 등급이 붙어 있다.
