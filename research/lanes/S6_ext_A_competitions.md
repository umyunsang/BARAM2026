# S6 외부문헌 레인 A — 풍력 예측 경진대회 상위해법의 피처구성

- 레인: `S6_ext_A_competitions` (읽기전용 외부 문헌 조사)
- 수행일: 2026-08-08
- 도구: `websearch` (Serper/Google) 단독. **165개 쿼리**. 검색 로그: `research/lanes/S6_ext_A_competitions.searchlog.json`
- 저장소 쓰기: 이 파일 + 위 로그 JSON 2개뿐. 모델 학습·데이터 읽기·git 조작 없음.
- **증거 등급 규약**
  - 인용부호(`"..."`)로 감싼 문장은 Google/Serper가 반환한 **원문 스니펫 그대로**다.
  - PDF/HTML 전문은 내려받지 않았다. 스니펫으로만 확인된 수치·주장에는 `[전문미확인]`을 붙였다.
  - 스니펫이 문장 중간에서 잘린 경우 `…`로 표시하고 추정 보간을 하지 않았다.

---

## §0 이 레인이 실제로 찾아낸 것 (요약)

1. **HEFTCom2024 상위팀의 풍력 피처는 "격자 원시장"이 아니라 "격자 위 저차원 순서통계(order statistics) × 소폭 시간창"이다.**
   팀 GEB(예측 4위·거래 3위·학생 1위)의 풍력 피처는 정확히 9개이며, 구성은
   `{max, mean, min}(100m 풍속, 대상영역 격자) × {lag, current, lead}` 이다.
   우리가 이미 실패로 판정한 "grid pivot 914열 + divergence/vorticity"와는 **차원과 불변성이 다른 축**이다.
2. **HEFTCom2024 우승모델(Olauson, stacked CatBoost)의 핵심은 "NWP 소스별로 따로 학습한 뒤 스태킹"이다.** 소스를 열로 합치지 않는다.
3. **파워커브를 "물리 사전(physical prior)"으로 모델에 주입하는 구성**은 2026년 문헌(UniWind)에서 명시적 모듈로 등장한다.
   경진대회 벤치마크 자체도 풍속→파워커브 변환형이었다.
4. **격자 선택은 "최근접 1점"이 아니라 "풍향 투영(tangential/normal) + 영역 통계"가 실증된 형태**다(Bengtsson 2025, KTH).
5. **한국 복잡지형(=우리 세팅과 가장 가까운 도메인)에는 "연직층 선택"이 별도 축으로 존재한다**(Lee et al. 2024, Energy 288:129713).
   LDAPS가 로터면을 덮는 복수 연직층(예: 60 m, 93 m)을 제공한다는 국내 논문 근거가 있다.

---

## §A 대회별 정리

### A-1. GEFCom2012 — Wind Power Forecasting Track

**개최/데이터/평가**

| 항목 | 값 | 출처 |
|---|---|---|
| 개최 | 2012 (Kaggle 호스팅) | https://www.kaggle.com/c/GEF2012-wind-forecasting |
| 과제 | 7개 풍력단지, 시간별 발전량, **48시간 선행** | `"A wind power forecasting problem: predicting hourly power generation up to 48 hours ahead at 7 wind farms."` |
| 학습기간 | 3년 시간별 이력 | `"7 wind farms • 3 years of hourly history • Wind forecasts issued twice a day • 48 hours ahead forecasting"` (Hong ISF2014 슬라이드) |
| NWP | 풍속/풍향 + u/v 성분, **하루 2회 발행** | 동상 |
| 지표 | **RMSE** | `"The error score for the wind power forecasting track is the Root Mean Square Error (RMSE)."` (Hong, Pinson & Fan 2014) |
| 리뷰논문 | Hong, Pinson, Fan (2014), *IJF* 30(2):357-363 | http://pierrepinson.com/31761/Literature/hong2014.pdf |

**중요한 세팅 차이(우리와 다른 점, 먼저 명시):** GEFCom2012는 **롤링 예측**이고 과거 발전량(previous known values)을 쓸 수 있었다.
아래 Silva의 베이스 모델이 바로 그 지속성 기반이다. 우리 세팅에는 그것이 없다.

**우승/상위 해법**

- 트랙 1위: `"Wind Power Forecasting Track: #1. Top 2 entries combined forecasts. Multiple linear regression, gradient boosting"`
  (Hong, ISF2014 발표자료, https://forecasters.org/wp-content/uploads/gravity_forms/7-2a51b93047891f1ec3608bdbd77ca58d/2014/07/HONG_TAO_ISF2014.pdf) `[전문미확인]`
- **Team Leustagos — L. Silva (2014), "A feature engineering approach to wind power forecasting: GEFCom 2012", IJF 30(2):395-401**
  https://www.sciencedirect.com/science/article/abs/pii/S0169207013000836
  - 피처 생성 원칙(원문 스니펫):
    `"The feature creation step had two main guiding principles: 1. Model the wind power generation equation, based on constants, the wind strength and direction, and …"`
    (원칙 2번은 스니펫이 잘려 확인 불가 `[전문미확인]`)
  - **정량 사다리(원문 스니펫, 이 레인에서 확보한 유일한 대회 ablation 수치):**
    `"the first model, containing only forecasts and previous known values, scored 0.1685 RMSE. Then, by adding some seasonal features (hour, month and year), it decreased to 0.16393. Next were included in the set the intervals of 36 h that were available in the test."`
    → **(hour, month, year) 계절 피처만으로 RMSE 0.1685 → 0.16393 = −2.71 %**
    → 그 다음 이득은 "테스트에서 사용 가능한 36 h 구간" 피처(=발행 구조 피처)
  - 풍속 멱승 피처: `"Silva [91] believes that the square and cubic of wind speed are important features to input into WPF models. However, not all features have sufficient relevance …"`
    (ResearchGate가 인용한 2차 문헌 서술, https://www.researchgate.net/publication/259119632) `[전문미확인]`
  - 결론 문장: `"We have shown that with some clever combining of well-known …"`
- **E. Mangalova & E. Agafonov (2014), "Wind power forecasting using the k-nearest neighbors algorithm", IJF 30(2):402-406**
  https://www.sciencedirect.com/science/article/abs/pii/S0169207013000848
  `"The following modeling steps are proposed: factor selection, raw data pretreatment, model evaluation and optimization. Both heuristic and formal methods …"`
  → 상위권이 **kNN(=아날로그/검색형)** 으로도 진입했다는 사실은 우리 M252 계열(analog/retrieval)에 대한 외부 지지 근거다.
- 논문 모음: http://blog.drhongtao.com/2014/03/gefcom2012-papers.html

**주의(오독 방지):** `"According to Table 3 of our GEFCom2012 paper, the winning teams improved the benchmark by about 30%"` 라는 스니펫은
**부하(load) 트랙 페이지**(`GEFCom2012 Load Forecasting Data`)에서 나온 문장이다. 풍력 트랙 수치로 전용하면 안 된다. `[전문미확인]`

**우리 질문(u/v vs 속도/방향, sin/cos, 시차보간, 인접팜 교차피처, kernel/GBM/RF 조합)에 대한 답:**
스니펫 수준에서 확인된 것은 (a) 풍속의 제곱·세제곱, (b) 계절 피처(hour/month/year), (c) 발행구간(36 h) 피처,
(d) 상위 2개 엔트리의 **선형회귀 + GBM 결합** 뿐이다.
**sin/cos 방향 인코딩, 인접 팜 교차피처, 시차보간의 개별 이득은 165개 쿼리로도 스니펫에서 확인되지 않았다.**
근거 없는 일반론을 쓰지 않기 위해 **미확인으로 남긴다.**

---

### A-2. GEFCom2014 — Probabilistic Wind Power Forecasting Track

**개최/데이터/평가**

| 항목 | 값 | 출처 |
|---|---|---|
| 개최 | 2014 | Hong et al. (2016) IJF, http://pierrepinson.com/docs/Hongetal2016.pdf |
| 과제 | 10개 zone, **24 h 선행 확률예측**(비모수 분위수) | `"The aim of the probabilistic wind power forecasting track of GEFCom2014 was to predict the wind power generation 24 h ahead in nonparametric …"` |
| NWP | **ECMWF u, v @ 10 m 및 100 m — 이것이 트랙의 핵심 설계** | `"The GEFCom2014 dataset contains hourly wind farm data from 10 locations in Australia, wind components forecasted by ECMWF at 10 and 100 meters height"` (UT Dallas / NREL, https://personal.utdallas.edu/~jiezhang/Conference/JIE_2017_BDCAT_DataDiversity.pdf) |
| 지표 | **Pinball loss** | `"The GEFCom2014 has now brought the pinball loss function to the attention of the wider energy forecasting community."` |

**상위 해법**

| 팀/논문 | 방법 | URL |
|---|---|---|
| **Landry, Erlinger, Patschke, Varrichio (2016), IJF 32(3):1061-1066** — 트랙 **우승** | 다중분위 **GBM** | https://www.sciencedirect.com/science/article/abs/pii/S0169207016000145 |
| Mangalova & Agafonov (2016), IJF 32(3):1067-1073 | **kNN** | https://www.sciencedirect.com/science/article/abs/pii/S0169207015001429 |
| Zhang & Wang (2016), IJF 32(3):1074-1080 | **kNN + KDE** (로그변환·경계커널) | https://www.sciencedirect.com/science/article/abs/pii/S0169207015001417 |
| Nagy et al. (2016), IJF 32(3):1087-1093 | **generalized additive tree ensemble** | https://www.sciencedirect.com/science/article/abs/pii/S0169207015001521 |
| Jing Huang & Matthew Perry (CSIRO) | 트랙 수상자 공지 | http://blog.drhongtao.com/2015/10/gefcom2014-winners.html |
| Hong et al. (2016) 리뷰 | `"Summary of the methods used by the top five teams in the wind track of GEFCom2014."` | http://pierrepinson.com/docs/Hongetal2016.pdf |

**두 층(10 m/100 m)을 어떻게 결합했나 — 이 레인의 핵심 질문**

- 직접 증거 (프랑스 HDR 학위논문이 GEFCom2014 계열 피처집합을 열거한 대목):
  `"— The wind shear between 10m and 100m (1 variable), — The two …"`
  (https://theses.hal.science/tel-04017641v1/file/HDR.pdf, 같은 문서가 `"K-nearest neighbors for GEFCom2014 probabilistic wind power …"`를 인용)
  → **10 m–100 m 사이의 wind shear를 "1개 변수"로 명시적으로 넣는 구성이 실재했다**는 것은 확인됨.
- **그러나 그 결합(shear/veer/층간 각도차)의 개별 정량 이득은 스니펫에서 확인되지 않았다.** `[전문미확인]`
- Landry 논문 본문 구조만 확인: `"… features in each model. Section 4 analyzes the performances of our predictions …"` — 피처 목록 원문 미확인 `[전문미확인]`

**우리 세팅에 대한 시사(중요):**
GEFCom2014가 "두 층 u/v"를 준 이유는 **허브고도가 두 층 사이에 있기 때문**이다.
우리 세팅의 허브고도는 **117 m**이고, 국내 문헌은 LDAPS가 **로터면을 덮는 60 m·93 m 층**을 제공한다고 적는다
(`"LDAPS 데이터는 예측 정확도를 높이기 위해 로터 회전 면적을 커버하는 93 m, 60 m 층(layer)에서의 풍속과 풍향 변수를 사용했다"`,
한국신재생에너지학회지, https://journalksnre.com/xml/46563/46563.pdf).
즉 **GEFCom2014의 "두 층" 구조는 우리 입력에도 존재할 가능성이 높다.** 다만 우리 팀은 이미 `alpha shear exponent`(−0.9 %)를 시험했으므로,
**shear를 별도 열로 추가하는 축은 재발굴 대상이 아니다.** 살아있는 것은 §B-2의 **REWS(로터등가풍속)** 처럼 *풍속 자체를 재정의*하는 형태다.

---

### A-3. HEFTCom2024 — Hybrid (Renewable) Energy Forecasting and Trading Competition

**개최/데이터/평가**

| 항목 | 값 | 출처 |
|---|---|---|
| 개최 | 2024년 2~4월, IEEE PES **Working Group on Energy Forecasting**, Ørsted·rebase.energy 후원 | https://arxiv.org/html/2507.01579v1 , https://www.rebase.energy/challenges/heftcom2024 |
| 대상 | **Hornsea 1 해상풍력 1200 MW + 태양광 2400 MW 합산** | Olauson et al. (2026) |
| NWP(공식제공) | **DWD ICON-EU + NCEP GFS**, 시간해상도 1 h, **1일 4회 갱신** | `"DWD's ICON-EU and NCEP's GFS, Both are hourly resolution with four updates per day. Hornsea 1 wind farm"` |
| 풍력 제공 변수 | `"For wind power, the following forecasted variables were provided by the competition … Wind speed and direction at 10 m and 100 m above ground. • Temperature at 2 m …"` | https://www.sciencedirect.com/science/article/pii/S0169207026000269 |
| 지표 | **Pinball Score** (예측), 실거래 수익(거래) | https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition |
| 데이터 | Zenodo 공개 | https://zenodo.org/records/13950764 |
| 대회 논문 | Browell et al. (2025), *IJF*; arXiv:2507.01579 | https://arxiv.org/abs/2507.01579 |
| 스타터 코드 | https://github.com/jbrowell/HEFTcom24 , 분석 재현: https://github.com/jbrowell/HEFTcom24-Analysis |

**대회 총평(원문):** `"The forecasting track reaffirms the competitiveness of popular gradient boosted tree algorithms for day-ahead wind and solar power forecasting, …"`

#### A-3-1. 우승모델 — Olauson, Viotti, et al. (2026), "The HEFTCom2024 winning model: A stacked CatBoost approach", *IJF* 42(3):724-735

- URL: https://www.sciencedirect.com/science/article/pii/S0169207026000269 (오픈 전문: https://www.diva-portal.org/smash/get/diva2:2046959/FULLTEXT01.pdf)
- **구조(원문):** `"CatBoost models (gradient boosting decision trees) were fit for each source of NWP (DWD, GFS, and MEPS) separately, and independently for wind …"`
  → **소스별 독립 모델 → 스태킹**. 소스를 하나의 피처테이블에 concat 하지 않는다.
  → MEPS(MET Norway MetCoOp EPS)는 참가팀이 **외부에서 추가로 가져온 소스**다.
- **피처 중요도(원문):** `"As can be seen in Table 2, wind speed is by far the most important feature for all wind power models. For MEPS, air pressure was included and has some non-…"`
  → 풍속이 압도적. 부가 피처는 주변부. **"복잡한 파생을 많이 만드는" 방향이 아니다.**
- **지속성 변형과의 대비(원문):** `"The 'CatBoost' model is the same as 'Stacked CatBoost' except that lagged wind power was used as a feature and weather data from GFS, DWD, and MEPS were …"`
  → 즉 우승 구성은 **lagged power를 쓰지 않는 쪽**이었다(대회 실시간 제약 때문). **우리 세팅과 정확히 같은 제약.**
- **가용성/케이블 사고 처리의 크기(원문):** `"Due to the Hornsea 1 cable problem, which was not handled in the benchmark model, the pinball loss was 142% higher for the benchmark than our model during …"` `[전문미확인 — 기간 한정 조건 불명]`
- 향후 제언: `"For a potential follow-up HEFTCom, we think it would be interesting to …"`

#### A-3-2. 팀 GEB — Pu et al. (2025/2026), *IJF* 42(3):736-751; arXiv:2505.10367

- 성적: `"ranked 3rd in trading, 4th in forecasting, and 1st among student teams"`
- URL: https://arxiv.org/abs/2505.10367 , HTML: https://arxiv.org/html/2505.10367v2
- **구성요소(원문):** `"Key components include: (1) a stacking-based approach combining sister forecasts from various Numerical Weather Predictions (NWPs) to provide …"`
- **★ 풍력 피처 전체 명세(원문, 이 레인의 최대 수확):**
  `"For wind power forecasting, 9 features are selected by combining the maximum, mean, and minimum wind speeds at 100 meters in the target region with lagged, leading, and current time steps."`
  → 즉 **{max, mean, min} × {t−1, t, t+1} = 9열이 전부**다.
  → 대상영역 격자를 **순서통계로 축약**(permutation-invariant)하고, 시간창은 ±1로 최소화.
- **다중소스 결합 효과(원문):** `"For the first issue, results of case studies show a reduction in pinball loss when combining multi-source NWP data in wind power forecasting. The stacking …"`
  및 `"We compare the performance of single NWP models and stacked sister models that combine multiple NWP sources in the two cases. The results are shown in Table 2, …"`
  → **Table 2의 수치는 스니펫으로 확보하지 못했다** `[전문미확인]`
- 학습 구성: `"Specifically, two different series of models using DWD and GFS NWP data are trained, respectively."`

#### A-3-3. 거래 트랙 1위 SVK — 소스 추가의 정량 이득

- 원문: `"SVK reported an 8% improvement in Pinball Score after combining forecasts from the MET Norway's MetCoOp Ensemble Prediction System …"`
  (https://arxiv.org/html/2507.01579v1 ; 저널판 동일 문장: https://www.sciencedirect.com/science/article/pii/S0169207025001013)
- → **NWP 소스 1개 추가 = Pinball Score 8 % 개선.** 단, 추가된 것이 **앙상블 예측 시스템(EPS)** 이라는 점이 중요하다(§C 참조).

---

### A-4. EEM 시리즈 · Kaggle · Dacon

**EEM (European Energy Market) wind forecasting competitions**

| 대회 | 우승/상위 해법 | 핵심 | URL |
|---|---|---|---|
| **EEM 2017** | Browell & Gilbert, "Cluster-based Regime-switching AR for the EEM 2017 Wind Power Forecasting Competition" | **regime-switching AR** — `"This paper describes the regime-switching autoregressive models used to win the EEM 2017 Wind Power Forecasting Competition."` | https://eprints.gla.ac.uk/248843/ |
| **EEM 2020 (EEM20)** | Browell & Gilbert, "Quantile Combination for the EEM20 Wind Power Forecasting Competition" | `"We combine quantile forecasts from two models with different characteristics: a 'discrete' tree-based model and 'smooth' generalised additive model."` | https://strathprints.strath.ac.uk/74066/1/Browell_etal_EEM_2020_Quantile_combination_for_the_EEM20_wind_power_forecasting_competition.pdf |
| EEM20 총평 | `"Tree-based ensembles and physics-inspired input features seem to be the most popular and successful method to generate a regional forecast with an uncertainty …"` | 트리 앙상블 + **물리기반 입력피처**가 지배적 | https://www.researchgate.net/publication/347266905 |
| EEM20 (다른 상위팀) | `"The forecasting model contains 3 aspects: an approach that predicts the level of expected wind production based on an analysis of similar …"` (유사사례/아날로그) | | https://www.linkedin.com/pulse/congrats-our-winners-eem2020-wind-power-forecasting-kariniotakis |

→ **EEM2017 우승법(AR)은 우리 제약에서 완전히 닫힌다**(§C). EEM20의 "tree ensemble + physics-inspired features + 두 모델 분위수 결합"은 우리 M263 계열 구조와 같은 계열이다.

**Kaggle**

- GEFCom2012 wind track 자체가 Kaggle 호스팅(https://www.kaggle.com/c/GEF2012-wind-forecasting). 규정: `"Winning solutions must be posted or linked to in the forums."` — 그러나 **포럼 게시물 본문은 165개 쿼리로 스니펫에서 열리지 않았다.** `[전문미확인]`
- "Hill of Towie Wind Turbine Power Prediction" (최근 Kaggle): `"Predict the Active Power of a wind turbine, given data from nearby turbines. … IMO data with wake impacts should benefit from …"`
  (https://www.kaggle.com/competitions/hill-of-towie-wind-turbine-power-prediction/discussion/605519)
  → **인접 터빈 실황을 쓰는 문제로 우리 세팅에 이식 불가**(§C).

**Dacon**

- 우리 대회: https://dacon.io/competitions/official/236727/overview/description (`"기상청 기상예보 데이터와 풍력발전량 데이터를 활용하여, 특정 풍력단지(3개 그룹)의 향후 발전량을 예측"`)
- **부정적 결과(명시):** Dacon **풍력** 대회의 상위 해법 공개 코드/피처 리스트는 이 레인의 검색 범위에서 **발견되지 않았다.**
  공개 코드가 확인된 것은 **태양광** 계열뿐이다(예: https://github.com/hyeonho1028/Solar-power-generation-forecast ,
  https://github.com/thinpig99/dacon-solarPanel , https://github.com/mg4432/Dacon-korea-east-west-power-competition ,
  https://github.com/HiddenBeginner/2022_oibc_competition).
  → **Dacon 풍력 상위해법 이식은 근거 없음. 이 축에 시간을 더 쓰지 말 것.**

---

### A-5. NWP 격자 선택 / 공간 가중 — 대회·문헌 정량 비교

이 절이 우리 질문 5번의 답이다. **"최근접 1점"을 쓴 상위해법 근거는 하나도 찾지 못했다.** 실증된 형태는 세 가지다.

**(a) 대상영역 격자의 저차원 순서통계 — HEFTCom2024 팀 GEB**
`"9 features … combining the maximum, mean, and minimum wind speeds at 100 meters in the target region with lagged, leading, and current time steps."`

**(b) 풍향 투영(tangential) 성분 — Bengtsson (2025), KTH 석사논문,
"Grid-Based Feature Engineering and Model Combination for Belgian Offshore Wind Power Forecasting"**
- URL: https://www.diva-portal.org/smash/get/diva2:2022553/FULLTEXT01.pdf (레코드: http://www.diva-portal.org/smash/record.jsf?pid=diva2:2022553)
- 설정: `"Weather data from 4 different numerical weather predictors …"`
- **피처 정의(원문):** `"TANGENTIAL WIND SPEED: Wind speeds along the mean wind direction were calculated using an inner product between the wind speed at each point …"`
- **정량 결과(원문):** `"around a 26 basis point improvement in MAE and a 80 basis point improvement in RMSE compared to the baseline. Each of the individual single …"`
  → basis point의 정규화 기준(설비용량 대비 %p로 추정)은 스니펫에서 확정 불가 `[전문미확인]`

**(c) 다층/다격자 대량 피처 후 트리 — Xu et al. (2020), *Atmosphere* 11(7):738**
- URL: https://www.mdpi.com/2073-4433/11/7/738
- `"The results show that, after using about 300 features at different height and pressure layers, the GBDT algorithm can output more accurate wind speed forecasts …"`
  → 개선폭 수치는 스니펫 미확인 `[전문미확인]`

**연직층 선택(복잡지형, 한국) — Lee, Park, et al. (2024), *Energy* 288:129713,
"Day-ahead wind power forecasting based on feature extraction integrating vertical layer wind characteristics in complex terrain"**
- URL: https://www.sciencedirect.com/science/article/pii/S0360544223031080 (SSRN 프리프린트: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4509803)
- `"Wind speed (ws) varies significantly within the wind farm in complex mountainous …"`
- `"In complex terrains with significant terrain variability, it is crucial to meticulously analyze and select the optimal vertical layer for each site or turbine …"`
- 모델: `"LGBM utilizes two key techniques, Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) …"`
- **개선폭 수치는 스니펫 미확인** `[전문미확인]`. **다만 도메인이 우리와 가장 가깝다(한국·산악복잡지형·day-ahead·LGBM).**

**다중 NWP 소스 결합의 정량 상한(참고)**

| 근거 | 수치 | URL |
|---|---|---|
| von Bremen et al. (2007) | 두 NWP 결합이 더 좋은 단일 모델을 명확히 상회, `"This is a relative improvement of about 6 %"` | https://uol.de/f/5/inst/physik/ag/enmet/publications/wind/journal/2007/combination_of_deterministic_probabilistic_models_to_enhance_wind_farm_power_forecasts.pdf |
| Yakoub et al. (2023), *Heliyon* | `"The combined use of multiple NWP sources reduced forecasting errors by 8 %–30 % for direct and indirect WPFs, respectively."` / `"the aggregated turbine-level improved WPF accuracy by 10 % and 15 % for RMSE and …"` | https://www.sciencedirect.com/science/article/pii/S2405844023086875 |
| HEFTCom2024 SVK | Pinball Score **8 %** 개선(EPS 추가) | arXiv:2507.01579 |

**터빈 단위 데이터의 가치(학습기간에만 가능)**
- Browell et al. (2017), PowerTech: `"performance of the best-performing benchmark, the gradient boosting machine, is improved by 12% at Clyde South wind farm and by 6% at Gordonbush."`
  https://strathprints.strath.ac.uk/60591/1/Browell_etal_Powertech2017_Use_of_turbine_level_data_for_improved_wind_power.pdf
- Gilbert et al. (2020), "Leveraging Turbine-Level Data for Improved Probabilistic Wind Power Forecasting": http://eprints.gla.ac.uk/248826/1/248826.pdf
- 같은 논문의 배경 서술: `"The two winning teams from GEFCom (2012 and 2014) utilised Gradient Boosting regression Trees (GBT), the latter for quantile regression to produce den- sity …"`

---

### A-6. 파워커브를 피처/타깃 변환으로 쓰는 방법

**(a) 물리 사전(physical prior)을 모델 안에 넣는 최신 구성 — UniWind (2026)**
- URL: https://arxiv.org/html/2607.01670v1 (abs: https://arxiv.org/abs/2607.01670)
- 원문: `"First, the Physical Prior Estimator combines density-aware wind-speed normalization, site-conditioned physical calibration, and a shared learnable power curve …"`
  및 `"UniWind first employs a Physical Prior Estimator to …"`
- → **(i) 밀도 보정 풍속 정규화, (ii) 사이트 조건부 물리 캘리브레이션, (iii) 공유 학습가능 파워커브** 3요소.
  우리의 "3그룹 풀링(−1.8 %)"이 통했던 이유를 정확히 설명하는 구조이며, 그 위에 (i)(iii)이 추가되는 형태다.
  **정량 이득은 스니펫 미확인** `[전문미확인]`

**(b) direct(직접 발전량) vs indirect(풍속→파워커브) 비교**
- Yakoub et al. (2023): `"Direct and indirect forecasting methods present similar performance."`
  → **둘 중 하나가 우월하다는 근거는 없다.** 따라서 "타깃을 pc로 두기"는 그 자체로 이득이 아니라 **표현(representation) 선택**이다.
- Bouché et al.: `"we compare the direct and indirect (wind speed predictions passed through a power curve) approaches to prediction."`
  https://bouchedimitri.github.io/files/pdf/nowcast_to_short_term.pdf
- 반대 방향 근거(ARIMA 계열): `"The comparison shows that the direct approach produce significantly more accurate forecasts compared with the indirect approach"`
  https://www.researchgate.net/publication/233647530 `[전문미확인]`

**(c) 로터등가풍속(REWS) — 파워커브의 "가로축"을 바꾸는 방법**
- Wagner (2010), DTU Risø PhD, *Accounting for the speed shear in wind turbine power performance measurement*:
  `"Using the equivalent wind speed accounting for the wind shear in the power performance measurement was shown to result in a more repeatable power curve than the …"`
  https://orbit.dtu.dk/files/6433255/ris-phd-58.pdf
  → **"더 재현성 있는 파워커브"**. 우리 목표(pc-MAE 감소)와 직결되는 진술이다.
- Murphy, Lundquist, Fleming (2020), *WES* 5:1169: `"We measure shear using metrics such as α (the log-law wind shear exponent), β_bulk (a measure of bulk rotor-disk-layer veer), β_total (a measure …"`
  https://wes.copernicus.org/articles/5/1169/2020/
- Tumenbayar et al. (2023): 베어(veer) 유형별 발전량 편차
  `"the PDC values ranged from −3.90 to 4.21% depending on the four types"`, `"The PDCs for the types VV and BV were 3.0 % and 4.2 %, respectively, meaning a power gain while those for the types VB and BB were −2.9 % and −3.9 %"`
  https://ijred.cbiore.id/index.php/ijred/article/view/47905 , https://pmc.ncbi.nlm.nih.gov/articles/PMC10319733/
  → **베어만으로 ±4 % 수준의 발전량 편차가 실측된다.** 이는 파워커브 잔차의 체계적 성분이다.

**(d) 공기밀도 보정 — 파워커브 정규화의 산업표준**
- `"This standard describes how to correct power curves measured at one site specific air density to the standard air density of 1.225 kg/m3."` (WindPRO/IEC)
  https://help.emd.dk/knowledgebase/content/ReferenceManual/PowerCurveOptions.pdf
- UniWind의 `"density-aware wind-speed normalization"` 과 동일 개념.

**(e) 시간 평활 — Hallgren et al. (2021), *WES*, "The smoother the better?"**
- URL: https://wes.copernicus.org/preprints/wes-2021-31/wes-2021-31.pdf , https://uu.diva-portal.org/smash/get/diva2:1596379/FULLTEXT01.pdf
- 원문: `"Smoothing the forecast improved the performance at all wind speeds, all stratifications and for all synoptic …"`
- → **NWP 풍속을 파워커브에 넣기 전에 평활**하는 것이 6가지 후처리 중 가장 견고했다는 결과. **개선폭 수치 미확인** `[전문미확인]`

---

## §B 우리 제약에 이식 가능한 후보 (§A에서 추린 것만)

기준: (1) 발행시각 D-1 14:00 KST에 계산 가능, (2) 평가기간 관측 불요, (3) 재분석·원격추론 불요,
(4) **이미 실패로 판정된 축(theta/Ri/alpha/gust factor, lag-lead ±1/2/3/6 + anomaly/rank, grid pivot 914열 + divergence/vorticity/coherence, 시드배깅, 대형 HP, L1)과 구별될 것.**

각 후보의 `이미 시도한 축과의 차이` 필드가 그 4번 조건에 대한 방어다.

---

### B-1. 격자 순서통계 × 최소 시간창 (9~21열) `[directly_supported]`

- **출처:** HEFTCom2024 팀 GEB(예측 4위) 풍력 피처 전체 명세. `"9 features … maximum, mean, and minimum wind speeds at 100 meters in the target region with lagged, leading, and current time steps."`
  보강: Bengtsson (2025) 격자 피처 패키지 `"26 basis point improvement in MAE / 80 basis point improvement in RMSE"`.
- **이미 시도한 축과의 차이:**
  - `grid pivot 914열`은 **격자 인덱스에 의존적(permutation-variant)** 이다. 격자 순서통계는 **불변(invariant)** 이며 차원이 9~21이다.
  - `divergence/vorticity/coherence`는 **미분 연산자**로 고주파 노이즈를 증폭한다. 순서통계는 **분위수 연산자**로 강건하다.
  - `lag/lead ±1/2/3/6 창`은 시간축만 확장했다. 여기서는 **공간축을 축약한 뒤 시간창을 ±1로 최소화**한다. 조합이 다르다.
- **의사코드**

```text
for each group g:
    S_g = { 격자셀 c : dist(c, centroid_g) <= R }        # R은 2~3셀 (LDAPS 1.5 km 격자 기준 3~5 km)
    for src in {LDAPS, GFS}:
        for t in {valid-1h, valid, valid+1h}:
            v = ws_hub(src, c, t) for c in S_g            # ws_hub = 허브고도 보간 풍속
            f[src,g,t,"max"]  = max(v)
            f[src,g,t,"mean"] = mean(v)
            f[src,g,t,"min"]  = min(v)
            f[src,g,t,"p75"]  = quantile(v, .75)          # 선택 확장
            f[src,g,t,"p25"]  = quantile(v, .25)
            f[src,g,t,"rng"]  = max(v) - min(v)           # ★ 공간 스프레드 = 앙상블 스프레드 대체재
```

- **추가 열 수:** 코어 = 2소스 × 3시각 × 3통계 = **18열**. 확장 포함 최대 36열.
- **기대이득 근거:** GEB가 이 9열만으로 HEFTCom2024 예측 4위. Bengtsson 격자패키지 MAE −26 bp. `[전문미확인 — 두 수치 모두 우리 지표로 환산 불가]`
- **구현비용:** 낮음(격자→그룹 반경 집계 한 번).
- **추가 자유도/과적합 위험:** 낮음. 열 수가 작고 통계량이 강건. **단 `rng`(공간 스프레드)는 지형 고정효과와 교란될 수 있어 그룹별 z-score 필요.**
- **왜 우리 병목에 맞는가:** 우리 병목은 그룹평균 풍속 RMSE 1.49~1.78 m/s이다. `mean`은 이미 쓰고 있을 것이고, `max/min/rng`가 주는 것은
  **"이 시각 이 그룹 안에서 풍속장이 얼마나 갈라져 있는가"** 이며, 이는 앙상블이 없는 우리에게 **유일하게 남은 불확실성 프록시**다.

---

### B-2. 로터등가풍속(REWS) + 베어 보정 — 풍속 자체의 재정의 `[near_match_only]`

- **출처:** Wagner (2010) `"more repeatable power curve"`; Murphy et al. (2020) WES 5:1169; Tumenbayar et al. (2023) 베어 유형별 발전량 편차 **−3.90 % ~ +4.21 %**;
  GEFCom2014의 10 m/100 m 두 층 설계; 국내 근거로 LDAPS 60 m·93 m 층 사용 사례(한국신재생에너지학회지).
- **이미 시도한 축과의 차이:** `alpha shear exponent`(−0.9 %)는 **shear를 별도 설명변수로 추가**한 것이다.
  REWS는 **파워커브에 들어가는 풍속 스칼라 자체를 교체**한다. 트리 입장에서 전자는 "새 분기축 1개", 후자는 "기존 주축의 좌표 변환"이다.
  후자는 파워커브 비선형성을 통과할 때 **오차 전파 자체**를 바꾼다.
- **의사코드**

```text
# 허브 117 m, 로터 반경 Rr (D=2*Rr). 로터면을 n개 수평 슬라이스로 분할.
# z_i = 슬라이스 i 중심고도, A_i = 슬라이스 i 면적, sum(A_i) = A

u(z)  = 두(이상) NWP 연직층(예: 60 m, 93 m, 혹은 사용 가능한 층들)으로부터
        멱법칙/로그법칙 보간:  u(z) = u(z_ref) * (z/z_ref)^alpha_hat,
        alpha_hat = ln(u(z2)/u(z1)) / ln(z2/z1)          # 이미 계산되어 있음
phi(z)= 같은 층들의 풍향 선형(각도 unwrap) 보간

REWS      = ( sum_i (A_i/A) * u(z_i)^3 ) ^ (1/3)
REWS_veer = ( sum_i (A_i/A) * ( u(z_i) * cos(phi(z_i) - phi(z_hub)) )^3 ) ^ (1/3)
veer_bulk = phi(z_top) - phi(z_bot)                        # 도(°), unwrap
```

- **추가 열 수:** `REWS`, `REWS_veer`, `veer_bulk`, 그리고 **`REWS_veer`를 넣은 pc 사전**(B-3과 결합) = **3~4열**.
  단, **`ws_hub`를 `REWS_veer`로 "대체"하는 변형도 함께 시험할 것**(열 추가 0).
- **기대이득 근거:** 베어 유형별 발전량 편차 실측 ±4 % 수준(Tumenbayar). 이는 pc 잔차의 **체계적** 성분이므로 pc-MAE 0.099 대비 무시할 크기가 아니다.
  단 그 편차는 **10분 SCADA 수준**의 측정이고, **NWP가 베어를 맞추는지는 별개 문제**다 → `[near_match_only]`, `[전문미확인]`
- **구현비용:** 중간(연직층 가용성 확인 필요; 우리 입력에 60/93 m 상당 층이 없다면 이 후보는 즉시 닫힌다).
- **추가 자유도:** 매우 낮음(3~4열). 슬라이스 수 n은 하이퍼파라미터가 아니라 물리적 이산화(n=3 또는 5로 고정).

---

### B-3. 파워커브 물리 사전(pc_prior)과 그 기울기 `[directly_supported]`

- **출처:** UniWind (2026) `"a shared learnable power curve"` / `"site-conditioned physical calibration"`;
  HEFTCom2024 벤치마크가 풍속→발전량 변환형이라는 대회 설계; Yakoub (2023) `"Direct and indirect forecasting methods present similar performance."`
- **이미 시도한 축과의 차이:** 지금까지 시도된 것은 모두 **입력 피처의 추가**였다. 여기서는 **타깃 기하의 사전 주입**이다.
  트리는 S자 곡선을 분기로 근사해야 하는데, `pc_prior`를 주면 남는 학습 대상이 **거의 평평한 보정항**이 된다.
- **의사코드**

```text
# 1) 학습기간 SCADA로 그룹별 파워커브를 한 번 적합 (단조 스플라인 / 등온회귀)
pc_hat_g : ws -> [0,1]          # 그룹 g의 용량계수 곡선, 단조 비감소로 제약

# 2) 피처
pc_prior      = pc_hat_g( ws_in )                       # ws_in = ws_hub 또는 REWS_veer (B-2)
pc_slope      = d pc_hat_g / d ws  |_{ws_in}            # ★ "풍속 1 m/s 오차가 몇 %p 발전량 오차인가"
pc_prior_lo   = pc_hat_g( ws_in - s )                   # s = 그룹평균 풍속 RMSE (1.49~1.78)
pc_prior_hi   = pc_hat_g( ws_in + s )
pc_band       = pc_prior_hi - pc_prior_lo               # ★ 예측 민감도 폭
pc_prior_smear= E_{eps~N(0,s^2)}[ pc_hat_g( ws_in + eps ) ]   # 수치적분 9점이면 충분
                                                        # ★ "번짐 보정된" 사전 = 조건부 평균에 더 가까움

# 3) 타깃(선택): 잔차학습
y_resid = y - pc_prior            # 모델은 y_resid를 학습, 최종 예측 = pc_prior + f(x)
```

- **추가 열 수:** 4~5열 (+ 잔차 타깃 옵션).
- **왜 우리 지표에 특별히 맞는가 (FICR 관점):**
  `pc_slope`와 `pc_band`는 **"이 시각의 풍속 오차가 발전량 오차로 얼마나 증폭되는가"** 를 직접 인코딩한다.
  우리 지표는 |오차| ≤ 6 %cap → 4점, ≤ 8 % → 3점의 **계단 보상**이므로, 최적 행동은 조건부 평균이 아니라
  **"6 %cap 밴드 안에 들어갈 확률을 최대화하는 값"** 이다. 그 확률은 `pc_band`의 함수다.
  즉 `pc_band`는 정책 결정(T/G 파라미터)이 **행마다 조건부로** 달라져야 하는 이유를 모델에 넘겨준다.
  현재 정책은 `T0.6_G0.5`처럼 **전역 상수**다. `pc_band`는 이를 **조건부**로 만드는 최소 재료다.
- **기대이득 근거:** UniWind가 이 3요소를 명시적 모듈로 채택 `[전문미확인 — 수치 미확보]`.
  direct vs indirect가 "비슷하다"(Yakoub)는 것은 **pc를 타깃으로 두는 것 자체는 무이득**임을 뜻하므로,
  **이득은 `pc_slope`/`pc_band`/`pc_prior_smear` 쪽에 있다고 봐야 한다.** 이 판단은 우리 것이며 문헌 직접 근거는 없다 → 해당 3열은 `[speculative]`.
- **구현비용:** 낮음(파워커브는 이미 프로젝트에 존재).
- **추가 자유도:** 낮음. 단 `s`(풍속 오차 스케일)를 그룹·시각별로 학습하기 시작하면 자유도가 폭발한다 → **`s`는 상수 3개(그룹당 1개)로 고정할 것.**

---

### B-4. 풍향 조건부 상류격자 + 접선/법선 분해 `[near_match_only]`

- **출처:** Bengtsson (2025) `"TANGENTIAL WIND SPEED: Wind speeds along the mean wind direction were calculated using an inner product between the wind speed at each point …"`,
  결과 `"26 basis point improvement in MAE and a 80 basis point improvement in RMSE"`.
  보강: Lee et al. (2024) 복잡지형 `"Wind speed (ws) varies significantly within the wind farm in complex mountainous …"`
- **이미 시도한 축과의 차이:** `divergence/vorticity/coherence`는 **격자 위의 국소 미분**이다.
  여기서는 **바람 방향을 따라 유한거리 떨어진 지점의 값을 가져온다**(advective sampling). 국소 미분이 아니라 **비국소 이류 지연**이다.
  산악 복잡지형에서 능선 도달 기류의 기원이 상류라는 물리에 대응한다.
- **의사코드**

```text
wd_bar = 그룹 평균 풍향 (허브고도, 해당 시각)
h_hat  = ( cos(wd_bar), sin(wd_bar) )          # 바람이 "불어오는" 방향의 단위벡터 (부호 규약 주의)
h_perp = ( -sin(wd_bar), cos(wd_bar) )

# (a) 접선/법선 분해 — 격자 인덱스 대신 풍향 좌표계
for c in S_g:
    ws_tan[c] = dot( (u[c], v[c]), h_hat  )
    ws_nor[c] = dot( (u[c], v[c]), h_perp )
f_tan_mean, f_tan_max, f_nor_std = mean(ws_tan), max(ws_tan), std(ws_nor)

# (b) 상류 샘플링 — 유한거리 d
for d in {3 km, 6 km, 12 km}:
    x_up = centroid_g - d * h_hat
    ws_up[d]  = bilinear( ws_hub_field, x_up )
    dws_up[d] = ws_up[d] - ws_site
    dT_up[d]  = T(x_up) - T(site)              # 상류-현장 온위차 (열적 대비, theta와는 다른 양)
```

- **추가 열 수:** (a) 3열 + (b) 3 거리 × 2 = 6열 → **9열** (소스 1개 기준; 2소스면 18열).
- **기대이득 근거:** Bengtsson MAE −26 bp / RMSE −80 bp (해상풍력, 4 NWP) `[전문미확인 — 정규화 기준 불명, 해상↔산악 도메인 차 큼]`
- **구현비용:** 중간(격자 좌표계·이류거리 보간).
- **추가 자유도:** 중간. 거리 d의 격자수가 자유도다 → **d를 {6 km} 하나로 고정하고 시작할 것.** 3개 다 넣으면 이전 실패(914열)의 축소판이 된다.
- **위험:** 우리 팀은 이미 격자 원시장에서 −0.4 %밖에 못 얻었다. 상류 샘플링이 다른 결과를 낼 근거는 **물리적 논증**이며 실증은 해상 도메인이다 → `[near_match_only]`.

---

### B-5. 소스별 독립 모델 + 스태킹, 그리고 소스 불일치를 피처로 `[directly_supported]`

- **출처:** Olauson et al. (2026) 우승모델 `"CatBoost models … were fit for each source of NWP (DWD, GFS, and MEPS) separately, and independently for wind …"`;
  GEB `"a stacking-based approach combining sister forecasts from various Numerical Weather Predictions (NWPs)"`;
  SVK **Pinball Score 8 % 개선**; von Bremen (2007) **상대 6 %**; Yakoub (2023) **8–30 %**.
- **이미 시도한 축과의 차이:** 우리 저장소는 "외부 NWP 소스 추가"를 **오차상관 0.78 → 평균화 이득 상한 4.6 %** 논거로 닫았다(AGENTS.md).
  **이 후보는 그 논거의 사정거리 밖이다.** 이유:
  1. 이미 손에 있는 **LDAPS와 GFS 두 소스**를 쓰는 것이지, 새 소스를 들여오는 것이 아니다.
  2. 얻으려는 것이 **평균화 이득이 아니다.** (i) 소스별 모델의 편향구조 차이를 스태킹으로 흡수, (ii) **소스 불일치 `|ws_L − ws_G|` 를 불확실성 피처로 사용**.
     (ii)는 평균화 이득 상한과 무관한 양이다.
- **의사코드**

```text
# 학습
M_L = fit( X_LDAPS_only, y )          # 소스별 독립 피처테이블
M_G = fit( X_GFS_only,   y )
Z   = [ M_L.predict(oof), M_G.predict(oof), disagree_features ]
M_S = fit( Z, y )                     # 스태킹 메타 (열 4~6개, 선형 또는 얕은 트리)

# 불일치 피처
disagree_ws   = ws_hub_LDAPS - ws_hub_GFS
disagree_absw = abs(disagree_ws)
disagree_wd   = angdiff( wd_LDAPS, wd_GFS )              # [-180,180]
disagree_pc   = pc_hat_g(ws_hub_LDAPS) - pc_hat_g(ws_hub_GFS)   # ★ 발전량 단위 불일치
```

- **추가 열 수:** 불일치 4열 + 스태킹 메타 입력 2열.
- **기대이득 근거:** SVK 8 %, von Bremen 6 %, Yakoub 8–30 %. **모두 "소스 추가" 효과이지 "불일치 피처" 효과가 아니다.**
  불일치 피처 자체의 정량 근거는 문헌에서 확인되지 않았다 → **`disagree_*` 4열은 `[speculative]`**, 스태킹 구조는 `[directly_supported]`.
- **구현비용:** 중간(소스별 학습 2회 + 메타). 6워커 제약 안에서 가능.
- **추가 자유도:** 스태킹 메타의 자유도는 2~6. **fold-outside로 반드시 검증할 것**(우리 저장소의 표준 규칙).

---

### B-6. 그룹 풀링을 "그룹 ID"가 아니라 "그룹 정적 공변량"으로 `[near_match_only]`

- **출처:** UniWind `"site-conditioned physical calibration"`; Konstantinou et al. (2024) `"Our results show a clear correlation between certain topographical features and forecast accuracy"`
  (https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2023.1328899/full);
  Lee et al. (2024) `"select the optimal vertical layer for each site or turbine"`;
  Camal et al. (2024) 조건부·정규화 접근 (https://www.sciencedirect.com/science/article/pii/S2213138824001395).
- **이미 확인된 우리 결과:** 3그룹 풀링 학습 = pc-MAE **−1.8 %** (이 레인의 발굴 대상 아님, 확장만 제안).
- **의사코드**

```text
# 공개 정적자료만 사용 (DEM 등, 2026-07-05 이전 공개, 상업이용 가능 라이선스, 출처·라이선스·취득일 영수증 기록)
per-group static covariates:
    elev_mean, elev_std                # 그룹 내 터빈 위치 고도
    slope_mean, aspect_sin, aspect_cos # 사면 경사/방위
    TRI  = terrain ruggedness index    # 국소 고도 표준편차
    Sx(theta) = 방위 theta 방향 지형차폐지수 (sheltering index), theta ∈ 8방위
    hub_height, rotor_D, n_turbine, cap_g
# 이 값들을 그룹 ID 대신 열로 준다 → 모델이 그룹 사이를 "보간"할 수 있게 됨
# 상호작용: Sx(wd_bar) = 현재 풍향 방위의 차폐지수 (★ 이것이 유일한 시변 항)
```

- **추가 열 수:** 정적 8~12열 + 시변 1열(`Sx(wd_bar)`).
- **왜 조심해야 하나:** 정적 열은 그룹당 상수이므로 **그룹 ID를 우회 인코딩할 뿐일 수 있다.** 그러면 아무 이득도 없거나 과적합만 늘어난다.
  **유일하게 진짜 정보를 주는 것은 시변 상호작용 `Sx(wd_bar)`**(풍향에 따라 능선 차폐가 바뀜)다.
  → **검증은 반드시 leave-one-group-out으로.** 그룹 3개뿐이라 통계력이 약하다는 점을 미리 인정한다.
- **기대이득 근거:** 지형-정확도 상관(Konstantinou)만 확인. **정량 이득 미확보** `[전문미확인]`.
- **구현비용:** 중간(DEM 취득·처리 + 영수증).
- **추가 자유도:** 실질 1(시변 항). 정적 열은 자유도 3(그룹 수)로 상한.

---

### B-7. 파워커브의 공간 컨볼루션 — "정격 도달 터빈 비율" `[speculative]`

- **출처(개념):** Postema et al. (2025) `"The joint PDF can be read as a probabilistic wind farm power curve, i.e., it shows the mapping from wind conditions to power production. This mapping is not …"`
  (https://whiffle.nl/wp-content/uploads/wes-10-1471-2025.pdf); Lee et al. (2024) `"Wind speed varies significantly within the wind farm in complex mountainous …"`
- **논거:** 단지 파워커브가 터빈 파워커브보다 완만한 이유는 **단지 내 풍속 분포와의 컨볼루션** 때문이다.
  B-1의 순서통계가 그 분포의 모멘트를 준다면, 이 후보는 **파워커브의 임계값 기준으로 자른 비율**을 준다.
- **의사코드**

```text
ws_cut_in, ws_rated = 그룹 g 터빈 사양 (학습기간 SCADA로 추정 가능)
v = ws_hub over S_g (또는 터빈 위치별 보간값)
frac_below_cutin = mean( v <  ws_cut_in )
frac_rated       = mean( v >= ws_rated  )
frac_mid         = 1 - frac_below_cutin - frac_rated        # 파워커브 급경사 구간 비율 = 오차 증폭 구간
pc_spatial       = mean_i pc_hat_g( v_i )                   # ★ 공간 평균 후 pc가 아니라 pc 후 공간 평균
pc_jensen_gap    = pc_spatial - pc_hat_g( mean(v) )         # ★ Jensen 갭 = 볼록성 보정항
```

- **추가 열 수:** 5열.
- **기대이득:** 문헌 정량 근거 없음 → `[speculative]`. 다만 `pc_jensen_gap`은 **부호가 물리적으로 결정된 양**(pc가 볼록한 구간에서 양, 오목한 구간에서 음)이라
  모델이 이를 학습으로 재발견해야 하는 부담을 덜어준다.
- **구현비용:** 낮음(B-1, B-3 재사용).
- **추가 자유도:** 낮음.

---

### B-8. NWP 풍속의 시간 평활(열 추가 없음) `[near_match_only]`

- **출처:** Hallgren et al. (2021) `"Smoothing the forecast improved the performance at all wind speeds, all stratifications and for all synoptic …"`
- **이미 시도한 축과의 차이:** `lag/lead 창`은 **열을 추가**했다. 평활은 **원래 열을 교체**한다. 자유도가 늘지 않는다.
- **의사코드**

```text
ws_smooth(t) = sum_{k=-K..K} w_k * ws_hub(t+k),   w = 정규화 삼각/가우스 가중, K ∈ {1, 2}
# ws_hub 대신 ws_smooth를 pc_prior(B-3)과 모델 입력에 사용
```

- **추가 열 수:** **0** (또는 원본 병기 시 1~2열).
- **⚠ 우리 지표에 대한 경고:** 평활은 예측을 조건부 평균 쪽으로 밀고, 우리 저장소가 이미 기록한 메커니즘
  ("배포 예측은 계단 보상 하 기대 정산금을 최대화하는 **행동**이지 조건부 평균이 아니다") 때문에
  **1-NMAE는 올리고 FICR은 내릴 가능성이 높다.**
  실제로 R601 계열에서 "일평균 보정 oracle이 1-NMAE +0.020365 / FICR −0.014849"였던 것과 같은 형태의 트레이드오프다.
  → **평활은 `pc_prior` 계산 경로에만 적용하고 최종 행동 산출에는 적용하지 않는 분리 실험**으로 시작할 것.
- **기대이득:** `[전문미확인 — 개선폭 수치 미확보]`
- **구현비용:** 매우 낮음. **가장 먼저 돌려볼 것.**

---

### B-9. 계절/달력 최소집합 (Silva 사다리) `[directly_supported]`

- **출처:** Silva (2014) 원문 사다리 — `"the first model … scored 0.1685 RMSE. Then, by adding some seasonal features (hour, month and year), it decreased to 0.16393."`
  → **−2.71 % RMSE**
- **이미 시도한 축과의 차이:** 우리는 시간 lag/lead를 시험했지 **절대 달력항**을 사다리로 측정하지 않았다.
- **의사코드**

```text
hour_sin, hour_cos          # 24h 주기
doy_sin,  doy_cos           # 365일 주기 (month보다 부드러움)
t_trend = (date - date0) / 365.25    # ★ year 대신 연속 추세 (2025로 외삽 가능해야 함)
```

- **⚠ 중요한 제약:** Silva의 `year`는 **범주형**이라 우리 2025 평가기간으로 외삽할 수 없다.
  반드시 **연속 추세**로 바꾸거나 드롭해야 한다. 범주형 year를 그대로 넣으면 2025에서 마지막 학습연도 값으로 붙어버린다.
  (우리 저장소의 기존 메모리 "frozen group-offset transfer: 최적 스케일은 `r_yy * sd_apply/sd_fit`"이 이 위험의 정량판이다.)
- **추가 열 수:** 5열. **구현비용 최저. 자유도 최저.**
- **기대이득:** Silva 기준 −2.71 % RMSE. 단 그의 베이스는 지속성 모델이었으므로 우리 세팅에서 같은 크기를 기대할 근거는 약함 → `[directly_supported]`이나 이득 크기는 `[전문미확인]`.

---

### B-10. 공기밀도 보정 풍속 `[near_match_only]`

- **출처:** UniWind `"density-aware wind-speed normalization"`; IEC/WindPRO 파워커브 밀도정규화 표준
  `"This standard describes how to correct power curves measured at one site specific air density to the standard air density of 1.225 kg/m3."`
- **이미 시도한 축과의 차이:** 실패한 `위치온도 theta`는 **대기 안정도 서술자**다. 밀도 보정은 **파워커브 가로축의 단위 변환**이다.
  전자는 새 분기축, 후자는 좌표 변환(B-2와 같은 계열).
- **의사코드**

```text
rho   = p / ( R_d * T_v ),   T_v = T*(1 + 0.608*q)     # NWP의 p, T, q(또는 RH)로 계산
ws_rho = ws_in * ( rho / 1.225 )^(1/3)                  # IEC Region-2 정규화
# ws_rho 를 pc_hat_g 의 입력으로 사용 (열 추가 0), 또는 rho 자체를 1열 추가
```

- **크기 감각(우리 사이트 추정):** 태백 가덕산은 고도 ~1,000 m 급 → rho가 해면 대비 약 −10 %.
  계절 기온 진폭 30 K도 rho를 약 10 % 흔든다. `(1±0.10)^(1/3) ≈ ±3.3 %` 의 **풍속 등가 이동**.
  파워커브 급경사 구간에서 3.3 % 풍속 이동은 발전량으로 수 %p다. **무시할 크기가 아니다.**
  (이 계산은 이 레인의 산술이며 문헌 인용이 아니다.)
- **추가 열 수:** 0~2열. **구현비용 낮음.**
- **⚠ 확인 필요:** 학습기간 파워커브를 SCADA로 적합할 때 **이미 현장 밀도가 반영되어 있다.**
  따라서 이득은 "밀도의 **계절·일변화**"에서만 나온다. 연평균 오프셋은 이미 흡수되어 있다. 이 점을 무시하면 이중보정이 된다.

---

### §B 우선순위 (구현비용 대비 기대이득)

| 순위 | 후보 | 태그 | 추가 열 | 비용 | 근거 강도 |
|---|---|---|---:|---|---|
| 1 | **B-1 격자 순서통계 × ±1h** | `[directly_supported]` | 18 | 낮음 | 대회 상위 4위 팀 피처 **전체 명세** |
| 2 | **B-3 pc_prior / pc_slope / pc_band** | `[directly_supported]`+`[speculative]` | 4~5 | 낮음 | UniWind 모듈 구조 + 우리 지표 구조와 직결 |
| 3 | **B-5 소스별 모델 + 스태킹(+불일치)** | `[directly_supported]` | 4~6 | 중간 | 우승모델 구조 그 자체 |
| 4 | **B-8 시간 평활(열 0)** | `[near_match_only]` | 0 | 최저 | Hallgren, FICR 위험 경고 포함 |
| 5 | **B-2 REWS + 베어 보정** | `[near_match_only]` | 0~4 | 중간 | Wagner "재현성 있는 파워커브", 베어 ±4 % |
| 6 | B-9 달력 최소집합 | `[directly_supported]` | 5 | 최저 | Silva −2.71 % |
| 7 | B-10 밀도 보정 | `[near_match_only]` | 0~2 | 낮음 | IEC 표준, 단 이중보정 주의 |
| 8 | B-7 정격도달비율/Jensen 갭 | `[speculative]` | 5 | 낮음 | 개념적 |
| 9 | B-4 상류격자/접선분해 | `[near_match_only]` | 9 | 중간 | Bengtsson, 도메인 차 큼 |
| 10 | B-6 그룹 정적 공변량 | `[near_match_only]` | 9~13 | 중간 | 그룹 3개 → 통계력 약함 |

---

## §C 우리 제약 때문에 **닫히는** 대회 기법 (명시적 목록)

| # | 기법 | 대회/문헌 근거 | 왜 닫히는가 |
|---|---|---|---|
| C-1 | **앙상블 스프레드·EPS 기반 분위수** | Bruninx et al. (2026): `"the use of an ensemble of weather forecasts can improve point forecast accuracy by up to 23%"` (arXiv:2602.13010); HEFTCom SVK의 8 %는 **EPS 추가**였음 | 우리 입력은 LDAPS·GFS **각 결정론 1런**. 멤버가 없다. → **부분 대체재는 B-1의 공간 스프레드 `rng`와 B-5의 소스 불일치뿐** |
| C-2 | **지속성 / lagged power** | Silva의 베이스 모델 `"only forecasts and previous known values, scored 0.1685 RMSE"`; Olauson의 비우승 변형 `"lagged wind power was used as a feature"` | 평가기간 2025에 관측·SCADA 전무 |
| C-3 | **regime-switching AR / 클러스터 AR** | Browell & Gilbert, EEM 2017 **우승법** (https://eprints.gla.ac.uk/248843/) | AR은 실황 필요 |
| C-4 | **온라인 후처리 / 축차보정 / 롤링 재학습** | 팀 GEB의 구성요소 (2)(3); HEFTCom이 `"real-time setting shadowing the Great Britain"` 이었다는 대회 성격 | 관측 피드백 없음 |
| C-5 | **Sister forecast(연속 런의 다중 리드타임) 스태킹** | GEB `"combining sister forecasts from various NWPs"`; HEFTCom NWP는 `"four updates per day"` | 우리는 **하루 1회 09 KST 초기화 1런**. 같은 valid time에 대한 sister가 존재하지 않는다. GEB의 sister는 "소스 간"으로만 축소 → **그 축소판이 B-5** |
| C-6 | **터빈 단위 실황 활용** | Browell (2017) `"+12% at Clyde South, +6% at Gordonbush"`; Gilbert (2020) | 평가기간 SCADA 없음. **학습기간 SCADA는 파워커브 적합(B-3)·타깃 구성에만 사용 가능** |
| C-7 | **인접 발전소/터빈 실황 교차피처** | Kaggle Hill of Towie `"given data from nearby turbines"` | 동일 |
| C-8 | **가용성/정지·출력제한 플래그** | Olauson: 케이블 사고 미처리 시 `"the pinball loss was 142% higher for the benchmark"` | 2025 정지 이력을 알 수 없음. 부분적으로는 우리 지표가 `actual < 0.1*cap` 행을 채점에서 제외하므로 **이미 일부 중화**되어 있다(기존 메모리: 가용성 결손 천장 0.0469) |
| C-9 | **재분석 기반 편의보정(ERA5/MERRA)** | Spiliotis et al. (2025) 풍속 편의보정 (75 터빈/10 단지) | 대회 규칙상 재분석 금지 |
| C-10 | **원격 API 추론 / 실시간 외부 NWP 호출** | — | 규칙상 금지. **정적 공개자료(DEM)와 2026-07-05 이전 공개 오픈소스 가중치만 허용** |
| C-11 | **Dacon 풍력 상위해법 이식** | — | 공개 코드/피처가 **존재하지 않음**(§A-4). 태양광 코드만 존재 |

---

## §D 이 레인이 **확인하지 못한** 것 (다음 레인에 넘김)

1. **Landry et al. (2016)의 GEFCom2014 우승 피처 목록 원문.** 165개 쿼리로 스니펫에서 열리지 않았다. IJF 페이월. `[전문미확인]`
2. **Silva (2014) 피처 생성 원칙 2번**과 최종 RMSE. `[전문미확인]`
3. **GEB(arXiv:2505.10367) Table 2**의 단일 NWP vs 스태킹 sister 모델 pinball 수치. `[전문미확인]`
4. **Lee et al. (2024, Energy 288:129713)** 의 연직층 선택 이득 크기. **도메인이 우리와 가장 가까우므로 최우선 후속 확인 대상.** `[전문미확인]`
5. **Bengtsson (2025)** basis point의 정규화 기준(설비용량 대비인지 평균발전량 대비인지). `[전문미확인]`
6. **UniWind(arXiv:2607.01670)** 의 physical prior 모듈 ablation 수치. `[전문미확인]`
7. GEFCom2012의 sin/cos 방향 인코딩·인접팜 교차피처·시차보간의 **개별 이득**. 어떤 스니펫에도 나오지 않았다.

---

## §E 전체 URL 목록

**대회 공식**
- GEFCom2012 wind (Kaggle): https://www.kaggle.com/c/GEF2012-wind-forecasting
- GEFCom 논문 모음(Hong 블로그): http://blog.drhongtao.com/2014/03/gefcom2012-papers.html
- GEFCom2014 수상자 공지: http://blog.drhongtao.com/2015/10/gefcom2014-winners.html
- HEFTcom2024 (rebase.energy): https://www.rebase.energy/challenges/heftcom2024
- HEFTcom2024 (IEEE DataPort): https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition
- HEFTcom2024 데이터(Zenodo): https://zenodo.org/records/13950764
- HEFTcom24 스타터 저장소: https://github.com/jbrowell/HEFTcom24
- HEFTcom24 분석 재현 저장소: https://github.com/jbrowell/HEFTcom24-Analysis
- Browell 자료 페이지: https://jethrobrowell.com/resources.html
- BARAM 2026 (우리 대회): https://dacon.io/competitions/official/236727/overview/description

**리뷰/개요 논문**
- Hong, Pinson, Fan (2014), GEFCom2012, IJF 30(2):357-363 — http://pierrepinson.com/31761/Literature/hong2014.pdf
- Hong et al. (2016), GEFCom2014 and beyond, IJF — http://pierrepinson.com/docs/Hongetal2016.pdf
- Hong (ISF2014) 슬라이드 — https://forecasters.org/wp-content/uploads/gravity_forms/7-2a51b93047891f1ec3608bdbd77ca58d/2014/07/HONG_TAO_ISF2014.pdf
- Browell et al. (2025), HEFTcom2024 대회논문 — https://www.sciencedirect.com/science/article/pii/S0169207025001013 , arXiv: https://arxiv.org/abs/2507.01579 , HTML: https://arxiv.org/html/2507.01579v1
- Bojer & Meldgaard, Kaggle forecasting competitions — https://vbn.aau.dk/ws/files/483966133/Kaggle_3_.pdf

**상위해법 논문**
- Silva (2014), IJF 30(2):395-401 — https://www.sciencedirect.com/science/article/abs/pii/S0169207013000836
- Mangalova & Agafonov (2014), IJF 30(2):402-406 — https://www.sciencedirect.com/science/article/abs/pii/S0169207013000848
- Landry et al. (2016), IJF 32(3):1061-1066 — https://www.sciencedirect.com/science/article/abs/pii/S0169207016000145
- Mangalova & Agafonov (2016), IJF 32(3):1067-1073 — https://www.sciencedirect.com/science/article/abs/pii/S0169207015001429
- Zhang & Wang (2016), IJF 32(3):1074-1080 — https://www.sciencedirect.com/science/article/abs/pii/S0169207015001417
- Nagy et al. (2016), IJF 32(3):1087-1093 — https://www.sciencedirect.com/science/article/abs/pii/S0169207015001521
- Huang & Perry (2016), IJF (semi-empirical GBM + kNN) — https://www.sciencedirect.com/science/article/abs/pii/S0169207015001375
- Olauson et al. (2026), IJF 42(3):724-735 — https://www.sciencedirect.com/science/article/pii/S0169207026000269 , 전문: https://www.diva-portal.org/smash/get/diva2:2046959/FULLTEXT01.pdf
- Pu et al. (2026), IJF 42(3):736-751 — https://arxiv.org/abs/2505.10367 , HTML: https://arxiv.org/html/2505.10367v2
- Browell & Gilbert (2017), EEM2017 우승 — https://eprints.gla.ac.uk/248843/
- Browell & Gilbert (2020), EEM20 — https://strathprints.strath.ac.uk/74066/1/Browell_etal_EEM_2020_Quantile_combination_for_the_EEM20_wind_power_forecasting_competition.pdf

**피처/방법 근거**
- Bengtsson (2025), KTH MSc — https://www.diva-portal.org/smash/get/diva2:2022553/FULLTEXT01.pdf
- Lee et al. (2024), Energy 288:129713 — https://www.sciencedirect.com/science/article/pii/S0360544223031080
- Xu et al. (2020), Atmosphere 11(7):738 — https://www.mdpi.com/2073-4433/11/7/738
- Hallgren et al. (2021), WES — https://wes.copernicus.org/preprints/wes-2021-31/wes-2021-31.pdf
- Wagner (2010), DTU Risø-PhD-58 — https://orbit.dtu.dk/files/6433255/ris-phd-58.pdf
- Murphy, Lundquist, Fleming (2020), WES 5:1169 — https://wes.copernicus.org/articles/5/1169/2020/
- Tumenbayar et al. (2023), IJRED — https://ijred.cbiore.id/index.php/ijred/article/view/47905 , https://pmc.ncbi.nlm.nih.gov/articles/PMC10319733/
- Gomez et al. (2020), NREL, 풍향 시어와 터빈 성능 — https://docs.nlr.gov/docs/fy20osti/76100.pdf
- UniWind (2026) — https://arxiv.org/abs/2607.01670 , HTML: https://arxiv.org/html/2607.01670v1
- Bruninx et al. (2026), tree-based + weather ensembles — https://arxiv.org/abs/2602.13010 , PDF: https://arxiv.org/pdf/2602.13010
- Yakoub et al. (2023), Heliyon, direct vs indirect + multi-NWP — https://www.sciencedirect.com/science/article/pii/S2405844023086875
- von Bremen et al. (2007) — https://uol.de/f/5/inst/physik/ag/enmet/publications/wind/journal/2007/combination_of_deterministic_probabilistic_models_to_enhance_wind_farm_power_forecasts.pdf
- Browell et al. (2017), PowerTech, 터빈단위 데이터 — https://strathprints.strath.ac.uk/60591/1/Browell_etal_Powertech2017_Use_of_turbine_level_data_for_improved_wind_power.pdf
- Gilbert et al. (2020) — http://eprints.gla.ac.uk/248826/1/248826.pdf
- Konstantinou et al. (2024), Frontiers in Energy Research, 복잡지형 CNN + DeepSHAP — https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2023.1328899/full
- Camal et al. (2024) — https://www.sciencedirect.com/science/article/pii/S2213138824001395
- Spiliotis et al. (2025), 풍속 편의보정 — https://www.sciencedirect.com/science/article/pii/S2213138825004308
- Postema et al. (2025), WES 10:1471, 확률적 단지 파워커브 — https://whiffle.nl/wp-content/uploads/wes-10-1471-2025.pdf
- Baquero et al. (2022), Energies 15:5518, 연직 외삽 ML vs 멱법칙 — https://www.mdpi.com/1996-1073/15/15/5518
- Yu & Vautard (2022), RSER 169 — https://www.sciencedirect.com/science/article/abs/pii/S1364032122007791
- IEC 밀도정규화(WindPRO 문서) — https://help.emd.dk/knowledgebase/content/ReferenceManual/PowerCurveOptions.pdf
- 한국신재생에너지학회지, 스태킹 앙상블 풍력예측(LDAPS 93 m/60 m 층) — https://journalksnre.com/xml/46563/46563.pdf
- 복잡지형 KMAPP 지상풍속 예측 성능평가(LDAPS 비교) — https://j-komes.or.kr/xml/28740/28740.pdf
- 기상자료개방포털 LDAPS 설명(UM 1.5 km L70, 1일 8회) — https://data.kma.go.kr/data/rmt/rmtList.do?code=340&pgmNo=65

**데이터셋 부수**
- GEFCom2012 wind data (tscompdata) — http://pkg.robjhyndman.com/tscompdata/reference/gefcom2012_wp.html
- GEFCom2014 데이터 서술(10 m/100 m ECMWF u,v) — https://personal.utdallas.edu/~jiezhang/Conference/JIE_2017_BDCAT_DataDiversity.pdf
- Kaggle Hill of Towie — https://www.kaggle.com/competitions/hill-of-towie-wind-turbine-power-prediction/discussion/605519

---

## §F 이 레인의 자기제한 진술

- 이 문서의 어떤 수치도 **우리 데이터로 검증되지 않았다.** 모든 대회 수치는 다른 지표(RMSE, pinball)·다른 도메인·다른 입력에서 측정된 것이다.
- 우리 저장소의 기존 규칙에 따라, **본 문서의 어떤 항목도 그 자체로 "PASS"가 아니다.** §B의 후보는 각각
  (a) 어떤 정책이 각 입력을 만들었는지, (b) 가중치가 in-sample인지 fold-outside인지, (c) 행 정렬 키 집합
  을 명시한 영수증과 함께 측정되어야 채택 가능하다.
- **§B의 10개 후보를 모두 한 번에 넣지 말 것.** 우리 저장소는 이미 다자유도 블렌드가 fold-outside에서 전부 기각된 기록을 갖고 있다.
  순위표 1~4번을 **하나씩** 넣고 fold-outside pc-MAE를 측정하는 방식이 유일하게 방어 가능한 경로다.
