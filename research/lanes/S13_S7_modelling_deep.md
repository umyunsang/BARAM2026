# Lane · S13 / S7 — 모델링 방법(추정기 estimator) 심층 발굴

- **레인 ID**: `S13_S7_modelling_deep` (읽기전용 외부문헌 전용)
- **작성**: 2026-08-08 (로컬 세션 시각)
- **도구**: `websearch` (Serper) **106 쿼리** (영어 96 / 한국어 10) + 공개 PDF·HTML 원문 직접 판독 **35건**
- **저장소 쓰기**: 이 파일 + `research/lanes/S13_S7_modelling_deep.searchlog.json` **2개뿐**.
  모델 적합 **0**, lockbox 접근 **0**, git 변경 **0**, 업로드 **0**, 외부 데이터 다운로드 **0**.
- **선행 레인 전수 열람**: `S6_ext_C_repr.md`(딥러닝 표현·손실함수 판정), `windskill_lit.md`(단일 결정론 NWP → 허브풍속 MOS),
  `S6_ext_A_competitions.md`, `L2_wind_sota.md`, `S13_S5_preprocessing_deep.md`, `S13_S6_features_deep.md`,
  `S12_ext_nwp_sources.md`, `S12_ext_dacon_solutions.md` — **여기 있는 내용은 반복하지 않고 §1에 승계 표로만 남긴다.**

---

## 0. 증거 등급 규약

| 등급 | 뜻 |
|---|---|
| **A** | 논문/보고서 **원문(PDF·HTML 전문)을 이 레인이 직접 내려받아 판독**하고 표·문장을 문자 그대로 인용 |
| **B** | 검색 스니펫 또는 초록에 수치가 문자 그대로 노출됐으나 전문 표·조건은 미확인 |
| **C** | 제목·초록 수준. 수치 없음 |
| **I** | 이 레인의 **내부 유도/계산** (외부 인용 아님) |
| **P5/P6/P7** | 부모가 준 **전제 5·6·7에 이미 포함**되어 있는 축 (신규성 없음) |
| **X** | 선행 레인이 이미 닫은 축 |

**환산 규약(문서 전체 적용, I).** 우리 MAE = `0.13858`(용량 정규화), `1−NMAE = 0.86142`.
따라서 **상대 MAE 감소 x% ↔ Δ(1−NMAE) = 0.0013858 · x**.

| 문헌이 보고한 상대 MAE 감소 | 우리 Δ(1−NMAE) |
|---:|---:|
| 1 % | +0.00139 |
| 3.3 % | +0.00457 |
| 4.1 % | +0.00568 |
| 5.0 % | +0.00693 |
| 5.9 % | +0.00818 |
| 6.5 % | +0.00901 |
| **10.71 %** | **+0.01484 ← 부모가 요구한 목표** |
| 17 % | +0.02356 |

**이 표가 이 레인 전체의 채점표다.** 아래 모든 후보는 "문헌이 측정한 상대 MAE 감소"를 이 축으로 환산해 평가한다.

---

## 1. 이 레인이 반복하지 않는 것 (선행 레인 승계)

| 축 | 상태 | 근거 |
|---|---|---|
| 시계열 DL 전부(N-BEATS/N-HiTS/DLinear/PatchTST/TFT/TiDE/TSMixer) | **영구 종료** — 타깃 자기회귀가 물리적으로 없음 | `S6_ext_C_repr.md` §C5 |
| TabPFN v2 / 2.5 / TS | **종료** — v2는 10k행·500열 상한 위반, 2.5+는 비상업 라이선스 | `S6_ext_C_repr.md` §C1-c, C5 |
| Entity/numeric embedding, SCARF/VIME/SubTab, CARTE, autoencoder 잠재표현 | **종료** | `S6_ext_C_repr.md` §C3 R3–R6 |
| ε-insensitive(밴드) 커스텀 목적 `O1`, soft-indicator `O2`, 다분위+상자커널 argmax `O3`, SPO/SPO+ `O5` | **선행 레인이 이미 제안·판정** — 본 레인은 우선순위만 갱신 | `S6_ext_C_repr.md` §C4 |
| 얕은 MLP/KRR을 **저상관 앙상블 멤버**로 (`R1`) | 부모 전제 5가 앙상블 축을 닫음 ⇒ **이 레인에서는 "단독 대체 추정기"로만 재검토** | `S6_ext_C_repr.md` §C3 R1 + 전제 5 |
| 아날로그(AnEn)·칼만·sister-model 소스 스태킹·QM | **종료** | `windskill_lit.md` C1/C2/C10/C11 + 전제 5 |
| 타깃을 발전량 → **허브고도 풍속**으로 내린 2단 GBDT의 *기본형* | **이미 구현됨** = 전제 7의 physics teacher | 전제 7 |
| 격자 순서통계·PCA·REWS·파워커브 사전 등 **피처** 축 | S6 소관 | `S13_S6_features_deep.md` |

**이 레인이 새로 판 곳(선행 6문서 전수 grep 결과):**
`multi-task`(0회), `monotonic`(0), `hierarchical`(0), `censored`(2, S5의 전처리 맥락), `log-cosh`(0),
`turbine-level / bottom-up`(0), `lead-time-continuous`(0), `Treeffuser/NGBoost/CQR`(0), `HL-Gauss`(0),
`Aurora / AIFS / Prithvi`(0), `FT-Transformer`(0). **이 11개 공백이 본 레인의 작업면이다.**

---

## 2. 이론 정박 — S7이 손댈 수 있는 것과 없는 것 **(I·유도)**

### 2.1 지표가 요구하는 것은 "조건부 중앙값"이다

`Total = 0.5(1−NMAE) + 0.5·FICR`. 부모 전제 1에 따르면 FICR 쪽 프런티어는 평평하다.
그러면 남은 구속은 `1−NMAE`뿐이고, **MAE를 최소화하는 행동은 조건부 중앙값 `median(y|x)`** 이다.
즉 S7의 과제는 정확히 하나의 문장으로 환원된다:

> **고정된 872열 피처면과 15,190 ~ 21,919 채점행 위에서, `median(CF | x)` 를 더 잘 추정하는 추정기가 있는가?**

이것은 "확률분포를 더 잘 추정하는가"(전제 7이 이미 26-class로 하고 있다)와도, "결정층을 더 잘 짜는가"(전제 1이 닫음)와도 다르다.

### 2.2 유효 표본은 행 수가 아니라 **발행 수**다

우리 데이터는 **하루 1발행 × 24 타깃시각**이다. 같은 날 24행은 같은 NWP 발행에서 파생되므로 오차가 강하게 자기상관한다.
fold당 15.2k ~ 21.9k 채점행 ÷ 24 ≈ **630 ~ 913 발행-일**(3개 그룹 합산)이고, 그룹당으로는 **약 210 ~ 300 발행-일**이다.
피처는 872열. **독립 단위 기준 p/n > 1 이다.**

이 숫자를 문헌의 성공사례와 나란히 두면 판정이 자동으로 나온다:

| 연구 | 독립 발행일 | 지점(그룹) 수 | 총 표본 규모 | 결론 |
|---|---:|---:|---|---|
| Schulz & Lerch 2022 (풍돌풍) | ~2,190일 (6년) | **175** | 175×6년 시간별 | NN(DRN/BQN) 승 |
| Veldkamp et al. 2021 (풍속) | ~540일 (3겨울) | **46** | 46지점 풀링 | CNN 승 |
| **우리** | **~210–300 / 그룹** | **3** | 15–22k행 | ? |

**핵심**: 두 성공사례의 이득은 모두 **지점 차원 풀링**(station embedding, 전지점 공동학습)에서 나온다.
Schulz & Lerch 원문:
> "postprocessing methods based on NNs **jointly estimating a single, locally adaptive model at all stations** provide the best forecasts and significantly outperform benchmark methods from machine learning" … "at **162 of the 175 stations** a network-based method performs best" **[A]**

우리는 지점이 **3개**다. 175 → 3 은 두 자릿수 축소이고, 이것이 §5의 여러 노드를 사전에 감점시키는 이유다.

### 2.3 리드타임 축은 우리 데이터에서 **죽어 있다** (I·유도, 중요)

발행이 D-1 14:00 KST **하나**이고 타깃이 D 01:00–24:00이므로,
`리드타임 = 타깃시각 + 11 시간` 이 **항등식**으로 성립한다.
⇒ **리드타임과 시각(hour-of-day)은 완전공선이다.** 
따라서 "리드타임별 모델"(Schulz & Lerch의 표준 관행) = "시각별 모델"이고, 
Wessel 2024의 lead-time-continuous 파라미터화도 시각의 매끄러운 함수와 구별되지 않는다.
현행 파이프라인이 시각을 피처로 갖고 있다면 **이 축은 신규 정보가 0이다.** (§6에서 닫는다)

### 2.4 전제 2가 말하는 것 — 추정기가 만질 수 있는 몫

MAE 0.13858 중 NWP→허브풍속 채널이 0.13022(분산기여 85.1%)다.
**추정기는 NWP 자체의 오차를 줄일 수 없다.** 줄일 수 있는 것은 오직
"고정된 NWP로부터 CF로 가는 사상 중 아직 학습되지 않은 구조"뿐이다.
문헌은 이 잔여 몫을 직접 측정한 적이 없지만, **추정기 클래스 교체의 측정된 효과크기**는 있다(§3 E1–E4).
그 값이 3~6%라는 것이 이 레인의 중심 발견이다.

---

## 3. 이 레인이 확보한 A급 증거 (원문 판독)

### E1. 추정기 클래스 교체의 효과크기 — 8개 방법 체계비교 (풍돌풍, 6년 × 175지점)

Schulz & Lerch (2022), *Mon. Wea. Rev.* **150**, 235–257, Table 4 (전체 리드타임·전체 지점 평균) — **원문 PDF 판독 [A]**
<https://journals.ametsoc.org/downloadpdf/journals/mwre/150/1/MWR-D-21-0150.1.pdf>

| Method | CRPS | **MAE** | RMSE |
|---|---:|---:|---:|
| EPC (기후) | 1.72 | 2.44 | 3.26 |
| EPS (원시 앙상블) | 1.33 | 1.63 | 2.16 |
| EMOS | 0.95 | 1.32 | 1.80 |
| MBM | 0.97 | 1.34 | 1.80 |
| IDR | 0.98 | 1.36 | 1.84 |
| EMOS-GB | 0.88 | 1.23 | 1.69 |
| **QRF (트리)** | 0.87 | **1.22** | 1.66 |
| **DRN (NN)** | 0.84 | **1.18** | 1.61 |
| **BQN (NN)** | 0.84 | **1.18** | 1.61 |
| HEN (NN) | 0.86 | 1.21 | 1.64 |

읽는 법(I):
- 원시 앙상블 → 최고 후처리: MAE 1.63 → 1.18 (**−27.6%**). 이 몫은 **우리가 이미 먹었다**(GBDT 후처리 파이프라인).
- **트리 최고(QRF 1.22) → NN 최고(DRN/BQN 1.18) = −3.3%** ⇒ 우리 환산 **+0.00457**.
- 저자 본문: "the basic methods already improve the ensemble by around **26%–29%**. Incorporating additional predictors via the machine learning methods further increases the skill" **[A]**

> **이것이 문헌상 가장 크고 가장 공정한 "추정기 클래스 교체" 실험이고, 그 값은 3.3%다.**

### E2. 결정론 NWP + 격자 CNN vs 트리 — 48시간 리드, 독립 시험셋

Veldkamp, Whan, Dirksen, Schmeits (2021), *Mon. Wea. Rev.* **149**, 1141–1152, Table 9
(KNMI Harmonie-Arome 2.5 km **결정론** 예보, 48 h, 네덜란드 46지점 풀링, 독립 시험셋) — **arXiv 전문 PDF 판독 [A]**
<https://arxiv.org/abs/2007.04005>

| Method | RMSE | **MAE** | CRPS | Log score |
|---|---:|---:|---:|---:|
| Climatology | 2.676 | 2.033 | 1.445 | — |
| Linear Regression | 2.399 | 1.170 | — | — |
| **QRF** | 2.217 | **1.077** | 0.776 | 4.02 |
| QRF_LR | 2.124 | 1.081 | 0.774 | 4.03 |
| CNN_LR_KMN | 1.905 | 1.034 | 0.740 | 3.96 |
| CNN_LR_N0 | 1.891 | 1.037 | 0.735 | 3.91 |
| **CNN_LR** | 1.889 | **1.033** | 0.731 | 3.93 |

읽는 법(I) — **세 가지가 동시에 읽히고, 두 개는 우리에게 불리하다**:
1. **QRF → CNN: MAE −4.1%** (1.077→1.033) ⇒ 우리 환산 **+0.00568**. 
2. 그러나 **RMSE는 −14.8%**(2.217→1.889). **공간 CNN의 이득은 압도적으로 꼬리(RMSE)에 있고 MAE에는 1/3만 나타난다.**
   우리 지표는 MAE 계열이므로 **문헌의 "CNN이 크게 이겼다"는 서사를 그대로 가져오면 3배 과대추정한다.**
3. `QRF`(관측 직접 회귀) vs `QRF_LR`(선형회귀 **잔차** 회귀): MAE 1.077 vs 1.081, CRPS 0.776 vs 0.774.
   ⇒ **트리 모델에서 "타깃을 예측하느냐, NWP 오차(잔차)를 예측하느냐"는 차이가 없다.** (부모 질문 F1-④의 직접 답)

같은 논문의 **음의 결과**(그리고 F1-④의 반쪽):
> "The convolutional networks have all been trained on the residuals of linear regression. **Convolutional neural networks trained on the observations were found to be not skillful in preliminary testing.** This was partly due to the fact that networks trained on the observations directly took longer to converge and converged to poor values more often than models trained on the residuals." **[A]**

⇒ 잔차 학습은 **NN의 최적화 문제**이지 통계적 이득이 아니다. GBDT에는 해당 사항이 없다.

### E3. 우리 지표(용량정규화 NMAE)로 측정된 day-ahead 풍력 추정기 비교 — 벨기에 해상 9개 단지, 4년

Bruninx 계열 (2026), *Probabilistic Wind Power Forecasting with Tree-Based Machine Learning and Weather Ensembles*,
arXiv:2602.13010, **TABLE I** (out-of-sample **용량정규화 MAE**, 9개 단지 평균) — **HTML 전문 판독 [A]**
<https://arxiv.org/html/2602.13010v2>

| 방법 | 평균 NMAE |
|---|---:|
| Power curve (제조사 커브) | 16.9 % |
| Wake model (보정된 후류모델) | 11.9 % |
| SVGP (가우시안과정) | 8.4 % |
| NGBoost (Gaussian) | 8.5 % |
| CQR (no tuning) | 8.2 % |
| CQR (tuned) | 8.2 % |
| **Treeffuser (조건부 확산 + LightGBM)** | **8.0 %** |

그리고 같은 논문의 **결정적 대조**:
> "The conditional diffusion model attains the best performance, with improvements of **5% in mean absolute error** and 12% in continuous rank probability score compared to the probabilistic baseline. Last, the results indicate an average improvement in point forecast accuracy of **17% by using an ensemble of weather forecasts instead of a single provider**." **[A]**
> "Compared to the average single-provider forecast, the ensemble method achieves an improvement of **17%** in MAE, and up to **23%** compared to the worst-performing provider." **[A]**

읽는 법(I) — **이 한 편이 부모의 핵심 질문에 그대로 답한다**:
- **추정기 축 전체(SVGP → NGBoost → CQR → Treeffuser)의 폭 = 8.5% → 8.0% = −5.9%** ⇒ 우리 환산 **+0.00818**.
- **입력(NWP 소스) 축 = −17%** ⇒ 우리 환산 **+0.02356**.
- 즉 **같은 논문 안에서 입력 축이 추정기 축보다 약 3배 크다.** 그런데 우리는 부모 전제 5에 의해 입력 축이 닫혀 있다.
- 부록 TABLE VII에서 ERA-5(재해석) 입력이 예보 앙상블보다도 좋았다(예: Belwind 6.7% vs 7.0%) — 재해석 금지가 왜 결정적인지의 독립 증거.

### E4. 격자 CNN vs 그래프 신경망 — **24–36 h 리드, NWP-only, 우리와 리드타임이 같음**

*Graph Neural Networks in Wind Power Forecasting* (2025), arXiv:2507.00105, Table 1 — **HTML 전문 판독 [A]**
<https://arxiv.org/html/2507.00105v1>
(3개 풍력단지 × 5년, 4년 학습 / 5년째 시험, MAE는 **설치용량 대비 %**)

| Test Year | 2022 A | 2021 A | 2022 B | 2021 B | 2023 C |
|---|---:|---:|---:|---:|---:|
| CNN (MAE) | 8.16 | 8.15 | 11.55 | 12.14 | 10.90 |
| GNN (MAE) | 8.10 | 8.23 | 11.49 | 12.12 | 10.83 |

⇒ 5개 셀 중 4개에서 GNN이 **0.2~0.7% 상대**로 앞서고 1개에서 뒤진다. **사실상 동률.**
> "We have shown that GNNs can be as accurate as CNNs in short- and medium-term wind energy production forecasting." **[A]**

부수적으로 중요한 정박점(I): **NWP-only, 24–36 h day-ahead 풍력의 실무 MAE 수준이 8~12% of capacity**이다.
우리 13.86%는 그보다 나쁘지만 **복잡산악 능선**이라는 조건을 감안하면 같은 자릿수다. 
즉 우리는 "형편없는 모델을 쓰고 있어서 11%가 남아 있는" 상태가 아니다.

### E5. 직접(power) vs 간접(wind→파워커브) — 프랑스 5개 단지, 순위 기준

Bouche, Flamary, d'Alché-Buc, Plougonven, Clausel, Roueff (2023), *Renewable Energy*, 저자판 PDF Table 3 — **원문 PDF 판독 [A]**
<https://bouchedimitri.github.io/files/pdf/nowcast_to_short_term.pdf>

풍력 **발전량** 예측의 평균 NRMSE 열화(작을수록 좋음, ×10⁻²), 괄호는 평균 순위:

| 방식 | 방법 | BM | BO | MP | RE | VE |
|---|---|---:|---:|---:|---:|---:|
| **Direct** | Nyström KRR (2.2) | 3.93 | 1.16 | 0.63 | 0.36 | 0.69 |
| Indirect | Nyström KRR (2.8) | 3.46 | 1.15 | 0.55 | 2.88 | 1.20 |
| Indirect | LASSO (3.0) | 3.96 | 0.77 | 1.08 | 2.62 | 0.86 |
| Direct | XG-Boost (4.4) | 4.54 | 2.36 | 1.50 | 1.70 | 1.80 |
| Direct | Feedforward NN (9.6) | 25.06 | 7.80 | 5.09 | 20.62 | 18.67 |

그리고 **메커니즘 문장**(부모 F1-①의 정확한 답):
> "it seems better to predict **directly** wind power … in direct prediction, we implicitly include the power curve into the learning problem. Consequently, the importance of the different wind speeds on the power curve are taken into account. For instance, **a model trained to predict wind speed first will be very eager to forecast well high values** (failing to do so would incur a high error term). **However to predict wind power, producing accurate forecasts for higher wind speeds is less critical, since in the power curve, the actual wind power as a function of the wind speed is thresholded.**" **[A]**

⇒ **풍속을 중간 타깃으로 두면 손실가중이 물리적으로 어긋난다**(정격 이상에서 풍속오차는 발전량오차 0인데 풍속손실은 계속 벌한다).
이것은 전제 7의 physics teacher 구조에 대한 **직접적 경고**이며, 동시에 §5 M4(wind-space 절단 우도)가 왜 "역파워커브"라는 형태를 취해야 하는지의 이유다.

### E6. 역파워커브 변환 + 절단(censored) 회귀 — 비선형 모델을 선형 모델로 되돌린다

Messner, Zeileis, Bröcker, Mayr (2013), Univ. Innsbruck WP 2013-01 — **원문 PDF 판독 [A]**
<https://www2.uibk.ac.at/downloads/c4041030/wpaper/2013-01.pdf>
> "The results show that with our inverse (power-to-wind) transformation, **simpler linear regression models with censoring perform equally or better than nonlinear models** with or without the frequently used wind-to-power transformation." **[A]**
> "It is shown that the **censored regression models obtained in wind space with the inverse-transformed power production are more reliable than uncensored regression models in all spaces considered** (i.e., in wind space, power space, and power-by-wind space)." **[A]**
> "the more parsimonious parametric models already perform well for **relatively small training samples** while the nonparametric models perform somewhat better in large training samples." **[A]**

⇒ 절단 우도의 측정된 이득은 **점정확도가 아니라 신뢰도(reliability)** 쪽이고, 크기는 "동등하거나 약간 우수".
그러나 마지막 문장은 우리 표본 체제(§2.2)와 정확히 맞물린다.

### E7. 터빈 단위 상향(bottom-up) 집계 — **복잡지형 사이트에서만 이득이 크다**

Gilbert, Browell, McMillan (2020), *IEEE Trans. Sustain. Energy* **11**(3), 1152–1160 — **원문 PDF 판독 [A]**
<http://eprints.gla.ac.uk/248826/1/248826.pdf>
> "The methods are tested at two utility scale wind farms and are shown to provide consistent improvements of **up to 5%**, in terms of continuous ranked probability score compared to the best performing state-of-the-art benchmark model. **The bottom-up hierarchical approach provides greater improvement at the site characterized by a complex layout and terrain**, while both approaches perform similarly at the second location." **[A]**
> Wind Farm **A**(복잡 배치·복잡 지형): "The feature engineering method reduces CRPS by **3.95%** and **5.46%** compared to direct wind farm-level forecasting … [copula 방식은] **5.01%** and **6.50%** over WF(x_t) and AnEn respectively." **[A]**
> Wind Farm **B**(단순): "the feature engineering approach provides the greatest improvement reducing CRPS by **1.24%** and **2.39%**" **[A]**
> B가 안 되는 이유: "correlation across the wind farm with only **6% of values below 0.7**, which implies that **there is little information to be gained by considering individual turbines** as forecast errors are very similar across the site." **[A]**
> A가 되는 이유: "At Wind Farm A … the covariance structure is more complex because of the wind farm's **irregular layout and terrain**. **Covariance is high within small areas of the wind farm but weak between regions.**" **[A]**
> 결정론 지표에 대해서: "The median (p50) of each predictive distribution is taken as the deterministic forecast and evaluated in terms of Mean Absolute Error … **the behaviour of the results is very similar to the probabilistic case**." **[A]**

⇒ 복잡지형에서 **CRPS −3.95 ~ −6.50%**, MAE도 "매우 유사한 거동" ⇒ 우리 환산 **+0.0055 ~ +0.0090**.
**이 레인이 찾은 허용 가능한 단일 노드 중 측정 효과크기가 가장 크다.** 그리고 그 조건(불규칙 배치 + 복잡 지형 + 구역 내 높은 공분산/구역 간 약한 공분산)은
`S13_S6_features_deep.md` §4가 실측한 우리 배치(g1 축 28.6°, g2 154.4°, g3 142.6°, 열내 1.8–3.5 D, 그룹중심 7.6–17.2 D)와 **형태가 같다**.

### E8. 집계 단위 상향의 두 번째 독립 측정

Yakoub, Mathew, Leal (2023), *Heliyon* 9(11):e21479 — **초록 문자열 [B]** (전문 403)
<https://www.sciencedirect.com/science/article/pii/S2405844023086875>
> "Direct and indirect forecasting methods present similar performance. Finally, **the aggregated turbine-level improved WPF accuracy by 10 % and 15 % for RMSE and MAE, respectively, compared to farm-level WPF.**" **[B]**

⇒ MAE **−15%** ⇒ 우리 환산 +0.0208. **단, [B]등급이고 조건 미확인**(그들은 터빈별 발전량 라벨을 가졌을 가능성이 높다).
E7과 E8이 3.95%~15%로 벌어져 있으므로, **보수적으로 E7(A급)을 채택하고 E8은 상한 힌트로만 쓴다.**

### E9. 자동 DL(격자 지도 입력) vs XGBoost — **NMAE 지표, day-ahead 계열**

Keisler et al., *WindDragon*, arXiv:2402.14385 / *Environmental Data Science* (2025), Table 1·2 — **HTML 전문 판독 [A]**
<https://arxiv.org/html/2402.14385v1>

프랑스 12개 지역 NMAE (전국 합계: WindDragon 7.7% / CNN 8.1% / ViT 8.5% / XGB-on-mean 9.2%).
지역 12개 평균 상대차(I·계산): **WindDragon vs CNN = −4.4%**, **CNN vs XGB = −16.9%**, **ViT vs CNN = +8.3%(나쁨)**.

**함정 고지 [A]**: 그들의 XGB 베이스라인은 의도적으로 약하다 —
> "**XGB on Wind Speed Mean**: … (i) Compute the **mean** wind speed for the considered region … (ii) Apply an XGBoost regressor to predict power generation based on the computed mean wind speed." **[A]**

즉 −16.9%는 "격자 지도 전체 vs 스칼라 평균 1개"의 격차이고, **우리는 이미 4×4·3×3 전장을 872열로 먹고 있으므로 그 대부분을 흡수했다.**
전이 가능한 것은 **CNN → AutoDL의 −4.4%뿐**이며, ViT가 CNN보다 8.3% 나빴다는 사실은 "복잡한 구조일수록 좋다"가 거짓임을 보인다.

### E10. GBDT는 무정보 피처에 **가장 강건한** 모델족이고, MLP 계열은 **가장 약하다**

Grinsztajn, Oyallon, Varoquaux (NeurIPS 2022 D&B) — **원문 PDF 판독 [A]**
<https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf>
> "Fig. 3 shows that the classification accuracy of a **GBT is not much affected by removing up to half of the features**." **[A]**
> "**removing uninformative features (4a) reduces the performance gap between MLPs (Resnet) and the other models** (FT Transformers and tree-based models), **while adding uninformative features widens the gap**. This shows that **MLPs are less robust to uninformative features**." **[A]**

⇒ **부모 전제 4와 정면으로 맞물린다.** 우리는 "노이즈 21열 = −0.000411"을 이미 측정했다.
같은 872열 면 위에서 MLP/DRN/CNN-헤드는 **GBDT보다 더 크게 희석당한다**.
따라서 **어떤 NN 계열 노드도 S6의 블록 프루닝(`F4 prune`)을 선행조건으로 갖는다.** 순서를 바꾸면 실패가 보장된다.

### E11. 분류(이산화) 손실의 이득은 **최적화 현상**이지 정보 현상이 아니다

Imani, Luedemann, Scholnick-Hughes, Elelimy, White (2026), *JMLR* 27, "Investigating the Histogram Loss in Regression" — **원문 PDF 판독 [A]**
<https://www.jmlr.org/papers/volume27/24-0260/24-0260.pdf>
> "Our results suggest that the **benefits of learning distributions in this setup come from improvements in optimization rather than modeling extra information**." **[A]**
> "when HL-Gauss is minimized, this difference is bounded by **half of the bin width**" **[A]**

⇒ 전제 7의 **26-class 이산화 자체는 정당**하지만(그리고 편향은 bin 폭의 절반으로 유계),
"HL-Gauss식 라벨 스무딩을 얹으면 좋아진다"는 기대는 **경사하강 최적화 병리가 있는 NN에서의 이야기**이고,
GBDT에는 그 병리가 없다. 부모 전제 5의 "ordinal smoothing 소진"과 **모순되지 않고 오히려 설명한다**.

### E12. 파워커브의 난류/시어 보정은 **불확실성을 유의하게 줄이지 못한다** (음의 결과)

Lee, Stuart, Clifton, Fields, Perr-Sauer, Williams, Cameron, Geer, Housley (2020), *Wind Energ. Sci.* **5**, 199–223
(Power Curve Working Group Share-3, 9개 기관 **55개 파워커브 시험**) — **원문 PDF 판독 [A]**
<https://docs.nlr.gov/docs/fy20osti/76102.pdf>
> "The trial methods reduce power-production prediction errors compared to the baseline method **at high wind speeds** …; however, the trial methods **fail to significantly reduce prediction uncertainty in most meteorological conditions**." **[A]**
> "**30 of the submissions report less than 0.5 %** in the statistical range of the absolute NME differences between the baseline and the trial methods." **[A]**

⇒ 시간 내 난류(TI)로 파워커브를 보정하는 축(부모 F1-③ "within-hour wind distribution")은 **업계 최대 규모의 공동실험에서 0.5% 미만**이었다.
우리 환산 **< +0.0007**. 사실상 죽어 있다.

### E13. 멀티태스크는 회귀에서 **음의 전이**를 낸다 (측정된 음의 결과)

*When Does Multi-Task Learning Fail? Quantifying Data...*, arXiv:2512.22740 — **HTML 전문 판독 [A]**
> "MTL **significantly degrades regression performance** (resistivity R² 0.897 → 0.844, p<0.01; hardness R² 0.832 → 0.694, p<0.01) while significantly improving classification performance … Analysis of the learned task relation graphs reveals **near-zero inter-task weights (~0.006)** … We attribute the regression degradation to **negative transfer caused by severe data imbalance**." **[A]**
(도메인은 소재이지만, "회귀에서의 음의 전이 + 태스크 불균형"이라는 메커니즘은 도메인 독립적이다 — 그래서 **[A/전이 C]** 로 표시한다.)

### E14. 사전학습 대기·시계열 기반모델 라이선스 실측 (부모 F4의 직접 답) — **원문 라이선스 페이지·LICENSE 파일 직접 판독**

| 모델 | 가중치 라이선스 | 상업이용 | 우리 규칙 판정 | 출처(직접 판독) |
|---|---|---|---|---|
| **Aurora** (Microsoft) | **MIT** | ✅ | **라이선스는 통과** | <https://raw.githubusercontent.com/microsoft/aurora/main/LICENSE.txt> ("MIT License … Implementation of the Aurora model. Copyright (c) Microsoft Corporation") · HF `microsoft/aurora` 카드 `License: mit` **[A]** |
| **AIFS Single 1.0** (ECMWF) | **CC BY 4.0** | ✅ | **라이선스는 통과** | HF `ecmwf/aifs-single-1.0`: "**These model weights are published under a Creative Commons Attribution 4.0 International (CC BY 4.0)**. … The notebooks and other script files are published under an Apache 2.0 licence" **[A]** |
| **FourCastNet** (NVlabs) | **BSD 3-Clause** (코드) | ✅ | 라이선스 통과, 가중치 배포처 별도 확인 필요 | <https://raw.githubusercontent.com/NVlabs/FourCastNet/master/LICENSE> ("#BSD 3-Clause License … Copyright (c) 2022, FourCastNet authors") **[A]** |
| **GraphCast** | 코드 **Apache-2.0**, 가중치 CC-BY-NC-SA | ❌ | 부모가 이미 배제 | <https://raw.githubusercontent.com/google-deepmind/graphcast/main/LICENSE> (Apache 2.0 = **코드만**) **[A]** |
| **Prithvi WxC** (NASA·IBM) | **CDLA-Permissive-2.0** | ✅ | **라이선스 통과, 그러나 입력이 MERRA-2(재해석) ⇒ 규칙 위반** | HF `Prithvi-WxC/prithvi.wxc.2300m.v1` `License: cdla-permissive-2.0`; 모델카드 "trained on 160 different variables from **MERRA-2** data" **[A]** |
| **Chronos** (Amazon) | **Apache-2.0** | ✅ | 라이선스 통과, 그러나 **타깃 히스토리 필요** ⇒ 정의역 오류 | HF `amazon/chronos-t5-large`: "This project is licensed under the Apache-2.0 License" **[A]** + `S6_ext_C_repr.md` §C5 |
| **TimesFM 2.0** (Google) | **Apache-2.0** | ✅ | 동상 | HF `google/timesfm-2.0-500m-pytorch` `License: apache-2.0` **[A]** |
| **Lag-Llama** | **Apache-2.0** | ✅ | 동상 | HF `time-series-foundation-models/Lag-Llama` `License: apache-2.0` **[A]** |
| **Moirai 1.1-R** (Salesforce) | **CC-BY-NC-4.0** | ❌ | **비상업 ⇒ 사용 불가** | HF `Salesforce/moirai-1.1-R-large` `License: cc-by-nc-4.0` **[A]** |
| TabPFN v2 / 2.5 | Prior Labs / 비상업 | ✅ / ❌ | 규모 위반 / 라이선스 위반 | `S6_ext_C_repr.md` §C1-c **[승계]** |

**판정(I)**: 라이선스가 통과하는 대기 기반모델(Aurora, AIFS, FourCastNet)은 **전부 "우리가 직접 전지구 예보를 생산한다"는 뜻**이고,
그러려면 (a) 2022–2025 전 기간의 **D-1 14:00 KST 이전 초기장**(HRES/GFS 해석장)을 확보해야 하며,
(b) 이는 부모 전제 5의 "external NWP 소진"과 `S12_ext_nwp_sources.md` §5.9가 이미 닫은 축이다.
시계열 기반모델(Chronos/TimesFM/Lag-Llama)은 라이선스는 깨끗하지만 **타깃 히스토리를 요구**하므로 정의역 자체가 없다.
⇒ **부모 질문 F4의 답: 라이선스가 통과하면서 동시에 우리 입력 제약을 만족하고, 측정된 다운스케일링 이득이 보고된 가중치는 하나도 없다.**

---

## 4. 노드표 (순위 = 측정된 효과크기 × 근거등급 ÷ (비용 + 자유도))

게이트 표기 **`G3`** = 부모가 이미 운용 중인 **3-fold fold-outside 사전등록 게이트**:
① fold-outside 평균 Δ(1−NMAE) ≥ **+0.0010**, ② 3/3 fold 부호 일치, ③ **추가 자유도 ≤ 1**,
④ all-or-nothing(사후 선택 금지), ⑤ 실패 시 영구 폐기 + receipt.

| id | 메커니즘 (정확히) | 근거 + **그들이 측정한 효과크기** | 왜 여기로 이전되나 | 실패 양식 | 비용 | 게이트 |
|---|---|---|---|---|---|---|
| **M1 `turbine17`** | **타깃 인수분해를 그룹 → 터빈으로 내린다.** 학습기 SCADA 10분 나셀풍속으로 **터빈 17기 각각의 허브풍속 teacher**를 적합 → 터빈별(또는 기종별) 파워커브 통과 → **그룹 합산**. 그룹 CF를 `pc(mean_i w_i)`가 아니라 **`mean_i pc(w_i)`** 로 만든다 | E7 [A] 복잡배치·복잡지형 사이트에서 CRPS **−3.95 ~ −6.50%**, 단순 사이트는 −1.24 ~ −2.39%; "결정론 MAE도 매우 유사한 거동" / E8 [B] MAE **−15%** | (a) Gilbert가 이득의 **조건**으로 지목한 것(불규칙 배치·복잡 지형·구역 내 높은 공분산/구역 간 약한 공분산)이 우리 실측 배치와 형태가 같다(`S13_S6_features_deep.md` §4). (b) `mean_i pc(w_i) ≥/≤ pc(mean w)` 는 **Jensen 부등식이 보장하는 계통 오차**이고, 전제 7의 현행 teacher는 이 오차를 흡수하지 못한다. (c) **10분 SCADA는 학습기간 전용이지만 teacher의 출력은 예측값이므로 시험기간 제약을 위반하지 않는다** | ① **최대 위험**: SCADA에 터빈별 **유효출력**이 없고 나셀풍속만 있으면, 터빈별 파워커브를 그룹출력만으로 식별해야 한다(약식별). ② 나셀풍속은 후류·유도계수로 편의가 있어 그대로 쓰면 안 됨(NTF 보정 필요) ③ 17개 teacher = 17배 적합 비용 ④ 가용성 결측이 터빈별로 다르면 라벨 채널(0.04804)이 오히려 커질 수 있음 | **중간~높음** (teacher 17개 재적합) | **먼저 무비용 전제확인**: SCADA 스키마에 (a) 터빈ID, (b) 유효출력, (c) 가용/운전상태 플래그가 있는가. (a)만 있으면 축소형(공통커브)으로만 진행. 이후 `G3`, 자유도 0(집계식이 물리로 고정) |
| **M2 `windspace`** | **역파워커브 변환 + 이중절단 우도.** 그룹 파워커브의 역함수로 관측 CF를 **"유효 허브풍속"** 라벨로 바꾸고, `[cut-in, rated]` 구간의 **양측 절단(censored) 우도**로 wind space에서 학습한 뒤, 다시 파워커브로 되돌린다 | E6 [A] "linear regression models with censoring perform **equally or better than nonlinear models**"; "censored … **more reliable than uncensored** … in all spaces"; "parametric models already perform well for **relatively small training samples**" / E5 [A] 직접예측이 나은 **이유**가 "정격 이상에서 풍속손실이 잘못 가중된다"인데, **역변환 라벨은 정확히 그 구간을 절단으로 처리해 그 반론을 무력화한다** | 우리 표본은 독립발행 기준 210–300/그룹(§2.2)으로 E6이 말한 "relatively small training samples" 체제 그 자체다. 그리고 CF는 **0과 1에서 실제로 절단**되어 있는데(정격 도달·저풍속), 현행 26-class는 이를 **경계 bin의 질량**으로만 표현할 뿐 절단 우도로 다루지 않는다 | ① 역파워커브는 정격 이상에서 **다대일**이라 라벨이 정의되지 않는다(→ 우측절단으로 처리해야 하며 구현 실수 시 라벨 오염) ② 그룹 커브 추정 자체가 새 자유도 ③ E6의 이득은 **신뢰도**에 있었고 점정확도가 아니었다 | 낮음~중간 | `G3`. 사전 스크린: 학습기간 채점행에서 **정격 도달(CF ≥ 0.95) 비율**과 **cut-in 근방 비율**을 세어, 합이 15% 미만이면 절단 우도의 여지가 없으므로 게이트 소모 없이 폐기 |
| **M3 `griddens`** | **872 평탄열의 격자 블록을 가중치 공유 인코더로 대체.** LDAPS 4×4 + GFS 3×3을 (변수 × 격자) 텐서로 두고, **격자 순열불변 집합 인코더 또는 소형 2D conv**로 임베딩 k(≤8)차원을 만들어 GBDT에 준다(전 과정 fold 내부) | E2 [A] QRF → CNN **MAE −4.1%** (RMSE −14.8%) / E9 [A] CNN vs ViT: **복잡한 구조가 8.3% 더 나쁨** / E10 [A] "GBT is not much affected by removing up to half of the features" | 부모 전제 4가 진단한 것은 **정보 부족이 아니라 희석**이다. 가중치 공유는 격자 블록의 파라미터 수를 O(격자수)에서 O(1)로 낮추는 **유일한 구조적 정규화**다 | ① **우리 격자는 4×4(=16점)와 3×3(=9점)이다.** E2는 60×60을, E9는 지역 규모 지도를 썼다. **공간 패턴을 배울 여지가 거의 없다** — 이 노드의 치명적 약점 ② E2의 이득의 2/3이 RMSE에만 나타났고 MAE엔 1/3만 왔다 ③ 인코더 학습 자유도(구조·차원·에폭) ≥ 3 | 중간 | **S6 `F4 prune` 통과 이후에만.** (E10: 무정보 열이 남아 있으면 NN 계열이 GBDT보다 더 크게 손해) 그리고 `G3` |
| **M4 `treeffuser`** | **조건부 밀도 추정기를 교체**: LightGBM 백본 조건부 확산(Treeffuser) 또는 NGBoost / CQR로 `f(y|x)`를 재추정하고, 기존 결정층(argmax)은 그대로 둔다 | E3 [A] 용량정규화 **NMAE 8.5%(NGBoost) → 8.2%(CQR) → 8.0%(Treeffuser)**; 논문 표현 "improvements of **5% in mean absolute error**" ⇒ 우리 환산 **+0.0069 ~ +0.0082** | 우리 지표와 **정확히 같은 정규화**(용량 대비 MAE), **같은 문제**(day-ahead 풍력), **같은 백본 계열**(LightGBM)에서 측정됐다. 전제 7의 26-class는 이산 히스토그램 밀도이고, Treeffuser는 연속 밀도라 bin 폭 편향(E11의 w/2)이 사라진다 | ① E3의 비교 상대는 **가우시안 SVGP/NGBoost**이지 **26-class 히스토그램**이 아니다 — 우리 현행이 이미 그 중간쯤이므로 증분은 5.9%보다 작다 ② 확산 샘플링으로 추론이 느려짐(E3 Table IV: 추론 0.3 s/관측, NGBoost의 3만 배) ③ 새 라이브러리 의존 | 중간 | `G3`, 자유도 ≤ 1(샘플 수 사전고정). **M2와 상호배타적으로** 실행할 것(둘 다 밀도 추정 방식 변경이라 효과가 겹친다) |
| **M5 `capacity`** | **용량 제어 3종 세트, 자유도 0**: (a) 풍속·풍속³ 계열 열에 **단조 제약**(`monotone_constraints`, 물리로 부호 확정), (b) `linear_tree=True`(잎에 선형모델 → 부드러운 파워커브 구간을 트리 계단 대신 직선으로), (c) 격자 블록에 **블록 단위 feature_fraction_bynode** | E10 [A] GBT는 절반 제거해도 거의 영향 없음 ⇒ **제거 여지가 크다** / 부모 자체 측정: 노이즈 21열 −0.000411, 랙/리드 252열 제거 **+0.000245** / E9 [A] 복잡한 구조(ViT)가 단순한 구조(CNN)보다 8.3% 나쁨 | 부모 전제 4가 이미 "빼면 좋아진다"를 한 번 측정했다. 단조 제약은 **새 정보 없이 가설공간만 줄이는** 유일한 수단이고 **자유도를 늘리지 않는다** | ① 단조 제약은 **후류·착빙·고풍속 셧다운** 구간에서 물리적으로 틀릴 수 있다(고풍속에서 출력이 실제로 감소) ⇒ **cut-out 이하로 제약을 한정**해야 함 ② `linear_tree`는 외삽에서 폭주 가능 ③ 이득이 작을 가능성이 크다 | **매우 낮음** (파라미터 3개) | `G3`. (a)/(b)/(c)를 **각각 독립 실행**(묶으면 실패 원인이 안 보인다). 자유도 0이므로 게이트 소모가 가장 싸다 |
| **M6 `drn`** | **CRPS로 학습하는 분포회귀망(DRN/BQN) + 그룹 임베딩**을 GBDT의 **대체**(앙상블 멤버가 아니라)로 | E1 [A] QRF 1.22 → DRN/BQN **1.18 = −3.3%**, 175지점 중 **162지점에서 NN 계열 우승** | 이론적으로는 §2.1의 목표(더 나은 조건부 분포 → 더 나은 중앙값)에 정확히 대응 | ① **결정적 반론**: E1의 이득 원천은 저자들이 직접 "**jointly estimating a single, locally adaptive model at all stations**"이라 적은 **175지점 풀링**이다. **우리는 3그룹이다.** 175 → 3은 이 메커니즘을 사실상 소거한다 ② E10 [A]: MLP 계열은 무정보 열에 GBDT보다 약하고 우리 면은 872열이다 ③ 학습 자유도 ≥ 4 | 중간~높음 | **권고하지 않음.** 하려면 S6 `F4 prune` 이후 + 자유도를 RealMLP식 사전고정 기본값으로 0에 가깝게 눌러야 한다 |
| **M7 `subhour`** | 시간 내 10분 풍속의 **(평균, 표준편차)** 를 2-타깃으로 예측하고 파워커브와 **합성곱**(TI 보정 파워커브) | E12 [A] PCWG Share-3, 55개 파워커브 시험: 난류/시어 보정은 고풍속에서만 오차를 줄이고 "**fail to significantly reduce prediction uncertainty in most meteorological conditions**", **30개 제출이 NME 차이 0.5% 미만** | — | 업계 최대 공동실험이 **0.5% 미만**을 측정했다. 우리 환산 **< +0.0007** | 낮음 | **권고하지 않음** (E12가 상한을 이미 못박음) |
| **M8 `mtl`** | (풍속, 풍향, 발전량) 멀티태스크 공유표현 | E13 [A/전이 C] MTL이 **회귀 성능을 유의하게 악화**(R² 0.897→0.844, 0.832→0.694), 태스크 간 가중 ≈ 0.006, 원인은 **태스크 불균형에 의한 음의 전이** / E5 [A] 풍속 타깃은 정격 이상에서 **손실가중이 어긋난다** | — | 두 개의 독립 증거가 같은 방향의 음의 결과를 준다. 게다가 GBDT는 하드 파라미터 공유 개념이 없어 구현이 곧 NN 도입이다 | 높음 | **권고하지 않음** |
| **M9 `residual`** | 타깃을 **NWP 기반 사전(파워커브 사전)의 잔차**로 바꿔 학습 | E2 [A] **QRF 1.077 vs QRF_LR 1.081** (CRPS 0.776 vs 0.774) — **트리에서는 차이 없음**; CNN에서만 필요했고 그 이유는 "converged to poor values more often"이라는 **최적화** 문제 | — | A급 직접 측정이 "트리에서는 0"이라고 말한다 | 낮음 | **권고하지 않음**(단, M3를 하게 되면 M3의 **필수 부속**으로 함께 채택) |
| **M10 `gnn`** | 터빈 배치 그래프 위 GNN | E4 [A] 24–36 h NWP-only, 3단지 5년: **CNN 8.16/8.15/11.55/12.14/10.90 vs GNN 8.10/8.23/11.49/12.12/10.83 → 사실상 동률** | — | 측정된 동률. 게다가 우리는 터빈별 라벨이 없어 그래프의 노드 타깃이 정의되지 않는다 | 높음 | **권고하지 않음** |
| **M11 `hlgauss`** | 26-class에 HL-Gauss(가우시안 라벨 스무딩) | E11 [A] "benefits … come from **improvements in optimization rather than modeling extra information**" | — | GBDT엔 그 최적화 병리가 없다. **부모 전제 5의 "ordinal smoothing 소진"과 일치** | 낮음 | **P5 — 신규성 없음** |
| **M12 `leadtime`** | 리드타임별/리드타임-연속 모델 | §2.3 [I] **리드타임 ≡ 시각 + 11** (발행이 하나뿐) ⇒ 완전공선 | — | 정보량 0 | — | **닫음(I)** |
| **M13 `foundation`** | Aurora / AIFS / FourCastNet / Prithvi / Chronos / TimesFM / Lag-Llama / Moirai | E14 [A] 라이선스 실측표. 통과하는 것은 **전지구 예보 생산**을 뜻하고(초기장 필요), TS 기반모델은 **타깃 히스토리**를 요구 | — | 라이선스·입력·정의역 삼중 구속 | — | **닫음** (전제 5 + `S12_ext_nwp_sources.md` §5.9) |
| **M14 `ftt/tabr/node/tabm`** | FT-Transformer / TabR / NODE / TabM | `S6_ext_C_repr.md` §C1-b·C5 [승계] + E10 [A] | — | 선행 레인이 닫음. E10이 그 판정을 강화 | — | **X — 승계 종료** |
| **M15 `epsilon`** | ε-insensitive 밴드 목적 | `S6_ext_C_repr.md` §C4 O1 | — | 선행 레인이 이미 1순위로 제안함 | 매우 낮음 | **P5/승계 — 본 레인은 순위만 지지**(자유도 0이라 게이트 비용이 가장 싸다) |

---

## 5. 상위 5개와 단일 최강 권고

### 5.1 Top-5 (정직한 기대치 포함)

| 순위 | 노드 | 근거등급 | 문헌 효과크기 | **정직한 우리 기대 Δ(1−NMAE)** | 왜 이 순위인가 |
|---|---|---|---|---|---|
| **1** | **M1 `turbine17`** (터빈 단위 타깃 인수분해 + 상향 집계) | **A** | CRPS −3.95 ~ −6.50% (복잡지형) | **+0.002 ~ +0.006** | 허용 가능한 노드 중 **측정 효과크기가 가장 크고**, 이득의 **조건**(불규칙 배치 × 복잡 지형)이 우리 실측 배치와 일치하는 유일한 노드. 그리고 이것은 새 데이터가 아니라 **이미 가진 10분 SCADA의 해상도를 타깃 쪽에서 쓰는 것** |
| **2** | **M4 `treeffuser`** (조건부 밀도 추정기 교체) | **A** | 용량정규화 NMAE 8.5→8.0 (−5.9%) | **+0.001 ~ +0.004** | **우리와 정확히 같은 지표·같은 문제·같은 백본**에서 측정된 유일한 숫자. 다만 우리 현행 26-class가 이미 그 중간이라 증분은 절반 이하로 봐야 함 |
| **3** | **M5 `capacity`** (단조 제약 · linear_tree · 블록 열샘플링) | B/I | — (부모 자체 측정: 랙 252열 제거 +0.000245) | **+0.000 ~ +0.002** | **자유도 0**이라 게이트 비용이 가장 싸다. 전제 4가 진단한 병(과대폭 가설공간)에 새 정보 없이 직접 작용하는 유일한 축 |
| **4** | **M2 `windspace`** (역파워커브 + 이중절단 우도) | **A** | "equally or better", 신뢰도 개선 | **+0.000 ~ +0.002** | 이론적으로 가장 깔끔하고(E5의 손실가중 반론을 절단으로 해소), 소표본 체제에 유리하다는 명시적 문장이 있음. 그러나 측정된 **점정확도** 이득이 없다 |
| **5** | **M3 `griddens`** (격자 가중치 공유 인코더) | **A** (전이 의심) | QRF→CNN MAE −4.1% | **+0.000 ~ +0.002** | 메커니즘은 전제 4에 정확히 대응하지만 **우리 격자가 16점·9점**이라 공간 패턴 학습 여지가 거의 없다. S6 프루닝 선행 필수 |

**합계의 정직한 진술 (I).**
다섯 개가 **모두 상단값으로 성공해도 +0.016**이고, 이는 노드 간 효과가 **완전히 독립**이라는 비현실적 가정 하의 값이다.
실제로 M1·M2·M4는 모두 "풍속→발전량 사상의 개선"이라는 같은 채널을 공유하므로 **상당 부분 겹친다**.
현실적 합산 기대는 **+0.004 ~ +0.009**이며, 목표 **+0.014839에는 미달한다.**

### 5.2 단일 최강 권고 — **M1 `turbine17`**

구현 사양(그대로 쓸 수 있게):

```
# 0) 무비용 전제확인 (이것 먼저, 5분)
#    SCADA 10분 테이블에 다음이 있는가?
#      (a) 터빈 식별자          -> 없으면 M1 폐기
#      (b) 터빈별 유효출력(kW)   -> 있으면 [완전형], 없으면 [축소형]
#      (c) 운전/가용 상태 플래그 -> 있으면 라벨채널(0.04804)도 함께 공략 가능

# 1) [완전형] 터빈 i 마다
#    teacher_i : X(872열, 그룹 무관 공통) -> w_i(허브풍속, 10분->시간평균)
#    curve_i   : 학습기간 (w_i, p_i) 로 단조 스플라인/등온회귀 1회 적합
#    CF_hat_g  = sum_{i in g} curve_i(w_hat_i) / cap_g
#
#    [축소형] 터빈별 출력이 없을 때
#    teacher_i : X -> w_i   (17개, 나셀풍속만으로 지도학습 가능)
#    curve_g   : 그룹 공통 커브 1개 (현행 teacher가 이미 갖고 있음)
#    CF_hat_g  = mean_{i in g} curve_g(w_hat_i)      # <- 핵심: 평균을 커브 '뒤'에서 취한다
#    (현행:      CF_hat_g = curve_g(mean_i w_i)  또는 그룹평균 풍속 기반)

# 2) 이 CF_hat_g 를 26-class 분류기의 '교사 타깃' 또는 '추가 1열'로 투입
#    (기존 결정층 argmax·정책 T/G 는 그대로 둔다 — 전제 1이 그 축을 닫았으므로)

# 3) 나셀풍속 편의 보정: NTF(nacelle transfer function)를 학습기간 데이터로 1회 적합.
#    보정하지 않으면 후류·유도계수 편의가 curve 추정으로 전가된다.
```

**왜 이것인가 (세 문장).**
(1) Gilbert et al.이 이득의 **필요조건**으로 지목한 것 — 불규칙 배치, 복잡 지형, 구역 내 공분산은 높고 구역 간은 낮음 —
은 우리 사이트의 실측 기하(`S13_S6_features_deep.md` §4: 세 그룹의 주축이 28.6° / 154.4° / 142.6°로 서로 다름)와 형태가 같고,
같은 논문의 단순 사이트에서는 이득이 1/3로 줄었다 — 즉 **이 노드는 "우리 같은 곳에서만 되는" 노드**다.
(2) `mean_i pc(w_i)` vs `pc(mean_i w_i)` 의 차이는 Jensen 부등식이 보장하는 **계통(bias) 오차**이지 잡음이 아니다.
파워커브는 cut-in 부근에서 볼록, 정격 부근에서 오목이므로 이 편차의 **부호가 풍속 구간마다 뒤집힌다** —
그래서 단일 스칼라 보정이나 affine recalibration(전제 5에서 소진됨)으로는 원리적으로 흡수되지 않는다.
(3) 새 외부 데이터가 0이고, 새 자유도가 0이며(집계식이 물리로 고정), **학습기간 전용 SCADA만 쓰므로 규칙 위반이 없다.**

**가장 큰 위험 (하나만 꼽으면): 나셀풍속의 후류 편의가 터빈별 커브 추정으로 전가되는 것.**
나셀 풍속계는 로터 후방에 있어 유도계수·후류로 계통 저평가되고, 그 편의는 **풍향과 상류 터빈 배치에 의존**한다.
이를 보정하지 않고 터빈별 커브를 적합하면, 커브가 "후류 손실을 흡수한 커브"가 되어
**예측 시점에 풍향이 학습 분포와 다를 때 그 흡수가 잘못 적용된다.**
`S13_S6_features_deep.md` §4.2가 계산한 그룹×풍향 그림자 지표는 이 위험이 실재함을 보여준다
(g1은 풍향 120°에서 차폐율 1.00, g2·g3는 300–340°에서 0.67–1.00).
⇒ **NTF 보정 없이 M1을 돌리지 말 것.** 그리고 fold-outside 게이트는 **풍향 섹터별로도** 부호 일치를 확인할 것.

---

## 6. 부모의 결정적 질문에 대한 정직한 답

> **"이 후보들 중 어느 하나라도 전제 2 하에서 1−NMAE를 +0.0148 올릴 수 있는가,
> 아니면 문헌은 복잡지형 11–35 h 리드의 단일 결정론 런에서 NWP→풍속 채널이 그만큼 줄지 않는다고 말하는가?"**

### **답: 아니오. S7(추정기) 축 단독으로는 불가능하다. 그리고 문헌은 그 이유를 두 번, 서로 독립적으로 측정했다.**

**근거 1 — 추정기 클래스 교체의 측정된 폭은 3~6%이고 우리는 10.7%가 필요하다.**

| 실험 | 조건 | 추정기 축의 폭 (상대 MAE) | 우리 환산 |
|---|---|---:|---:|
| Schulz & Lerch 2022 [A] | 6년 × 175지점, 8개 방법 체계비교 | QRF 1.22 → DRN 1.18 = **−3.3%** | +0.00457 |
| Veldkamp 2021 [A] | 결정론 NWP, 48 h, 46지점 | QRF 1.077 → CNN 1.033 = **−4.1%** | +0.00568 |
| Bruninx 2026 [A] | **day-ahead 풍력, 용량정규화 NMAE** | NGBoost 8.5% → Treeffuser 8.0% = **−5.9%** | +0.00818 |
| GNN 2025 [A] | **24–36 h, NWP-only 풍력** | CNN ↔ GNN = **±0.7%** | ±0.001 |
| WindDragon 2025 [A] | 지역 풍력, NMAE | CNN → AutoDL = **−4.4%** (XGB 대비 −16.9%는 베이스라인이 스칼라 1개) | +0.0061 |
| **필요치** | | **−10.71%** | **+0.01484** |

**다섯 개의 A급 head-to-head 중 어느 것도 절반을 넘지 못한다.**
그리고 이 다섯은 모두 **베이스라인이 우리보다 약한** 상황(단일 스칼라 입력, 지점 풀링 없음, 격자 미사용)에서 측정된 값이므로,
**이미 872열 GBDT + 26-class 밀도 + 정산최적 argmax를 돌리고 있는 우리에게서는 더 작게 나올 것으로 봐야 한다.**

**근거 2 — 같은 문헌들이 "그만큼 큰 이득은 입력 축에 있다"고 함께 측정했다.**

Bruninx 2026 [A]는 한 편의 논문 안에서 두 축을 나란히 재고, **추정기 5% vs NWP 소스 17%** 로 보고한다.
Optis & Perr-Sauer 2019 [A, `S13_S6_features_deep.md` E6 승계] 는
"**standard deviation across features (1.0%) is nearly twice that across algorithms (0.6%)**"를 측정했다 —
**피처 축이 알고리즘 축의 2배**이고, 부모는 그 피처 축이 이미 측정상 포화(전제 4)임을 확인했다.
Schulz & Lerch 2022 [A] 도 같은 형태다: 원시→후처리 26–29%, 후처리 방법 간 3.3%.

**근거 3 — 우리 데이터의 유효 표본이 문헌 성공사례의 이득 메커니즘을 소거한다 (I·§2.2).**
DRN/CNN이 이긴 두 연구의 이득은 저자들이 직접 **지점 풀링**(175지점, 46지점)이라고 적었다.
우리는 **3그룹**이고, 독립 발행일은 그룹당 210–300이다. 이 축에서 우리가 얻을 수 있는 몫은 구조적으로 작다.

### 그럼 어디에 +0.0148이 있는가 (정직한 지도)

| 축 | 문헌이 측정한 크기 | 우리 상태 |
|---|---|---|
| **입력(다중 NWP 소스)** | **−17%** [A, Bruninx] | 전제 5가 닫음 (LDAPS-GFS 오차상관 0.78) |
| **입력(앙상블/스프레드)** | 원시→후처리 −26~29% CRPS [A, Schulz] | 우리는 결정론 1런. 부모 측정: GEFS 스프레드-오차 상관 0.02–0.14 |
| **피처** | 알고리즘 대비 **2배** [A, Optis] | 전제 4가 측정상 포화 판정 |
| **추정기(S7)** | **−3.3 ~ −5.9%** [A ×3] | **본 레인의 작업면. 상한 +0.008.** |
| **결정층** | — | 전제 1이 평평하다고 측정 |
| **앙상블** | — | 전제 5가 닫음 (멤버 상관 0.934–0.994) |

**따라서 이 레인의 최종 진술은 다음과 같다.**

> **복잡지형 11–35 h 리드의 단일 결정론 런에서, 강한 GBDT 후처리가 이미 걸려 있는 상태로부터
> 추정기만 바꿔서 MAE를 10.7% 더 줄인 사례는 문헌에 없다.
> 측정된 최대치는 5.9%(+0.0082)이며, 그것도 우리보다 약한 베이스라인 대비다.
> S7은 필요조건이지 충분조건이 아니고, 부모가 요구한 +0.014839는 S7 단독으로는 도달 불가능하다.**

부수적으로, 이는 **부모 전제 2와 정합적**이다: MAE의 85.1% 분산기여가 NWP→허브풍속 채널이고,
**추정기는 NWP 자체의 오차를 만들지 못한다.** 완전 실측풍속에서 Total 0.869922라는 사실은
"정보가 부족하다"는 진술이지 "추정기가 부족하다"는 진술이 아니다.

---

## 7. 이 레인이 명시적으로 **닫는** 축

1. **멀티태스크 (풍속·풍향·발전량 공유표현)** — E13 [A] 회귀에서 유의한 음의 전이 + E5 [A] 풍속 타깃의 손실가중 불일치. 재시도 금지.
2. **시간 내 풍속 분포 → TI 보정 파워커브** — E12 [A] PCWG Share-3(55개 시험): "fail to significantly reduce prediction uncertainty in most meteorological conditions", 30개 제출이 **0.5% 미만**.
3. **NWP 오차(잔차) 예측으로의 타깃 전환** — E2 [A] 트리에서 QRF 1.077 vs QRF_LR 1.081 (차이 없음). NN에서만 필요했고 그 이유는 최적화였다.
4. **터빈 배치 그래프 GNN** — E4 [A] CNN과 동률(±0.7%), 게다가 터빈별 라벨 부재.
5. **리드타임별/리드타임-연속 모델링** — §2.3 [I] 발행이 하나이므로 리드타임 ≡ 시각 + 11, 완전공선.
6. **HL-Gauss / 분포형 회귀 손실을 GBDT에 이식** — E11 [A] 이득의 출처가 **NN의 최적화 병리 완화**. 부모 전제 5(ordinal smoothing 소진)와 일치.
7. **사전학습 대기·시계열 기반모델 전부** — E14 [A] 라이선스 실측표. 통과하는 것(Aurora MIT, AIFS CC-BY-4.0, FourCastNet BSD-3)은 초기장 확보 = external NWP 축(전제 5·S12 §5.9)이고, 통과하는 시계열 모델(Chronos/TimesFM/Lag-Llama Apache-2.0)은 타깃 히스토리를 요구하며, Moirai는 CC-BY-NC로 **사용 불가**, Prithvi WxC는 라이선스는 통과하나 **MERRA-2(재해석) 입력**으로 규칙 위반.
8. **FT-Transformer / TabR / NODE / TabM / TabPFN** — `S6_ext_C_repr.md` 승계 종료. E10 [A]가 그 판정을 강화(MLP 계열은 무정보 열에 가장 약함).

---

## 8. 이 레인이 확인하지 못한 것 (정직한 공백)

- **SCADA 10분 테이블의 실제 스키마를 열어보지 않았다.** M1(1순위)의 완전형/축소형 분기가 여기에 전적으로 달려 있다.
  터빈별 유효출력과 가용 플래그의 존재 여부는 루트가 **1분 안에** 확인해야 한다.
- **Yakoub 2023 (E8, MAE −15%) 전문을 못 봤다** (Heliyon/ScienceDirect 403). 초록 문장만 [B]. 그래서 M1의 기대치는 E7(A)로 묶었다.
- **Gilbert 2020의 Table III(결정론 MAE 수치표)를 텍스트로 추출하지 못했다.** 본문 문장("very similar to the probabilistic case")만 인용했다.
  따라서 M1의 MAE 기대치는 CRPS 수치를 통한 **간접 환산**이며, 일반적으로 CRPS 개선폭 ≥ MAE 개선폭이므로 **이 환산은 낙관 쪽으로 치우쳐 있다.**
- **Lee et al. 2024 (*Energy* 288, 한국·복잡지형·day-ahead·NMAE)의 수치를 끝내 확보하지 못했다** (SSRN·ScienceDirect 모두 403).
  이 논문은 우리와 지표·지형·국가가 모두 일치하는 유일한 문헌이며, 루트가 기관 접근으로 확보하면 본 레인의 여러 추정치를 교정할 수 있다.
- **Treeffuser를 우리 규모(15–22k행 × 872열)에서 돌린 사례를 못 찾았다.** E3의 벨기에 사례는 피처 수가 훨씬 적다.
- **AIFS/Aurora의 "가중치 공개일"을 개별 확인하지 않았다** (라이선스만 확인). 2026-07-05 기준선 검증은 별도 작업이다 — 어차피 §7-7로 닫혔으므로 추적하지 않았다.

---

## 9. 검색 로그 요약

- 총 **106 쿼리** (영어 96 / 한국어 10). 전체 로그: `research/lanes/S13_S7_modelling_deep.searchlog.json`
- **원문 직접 판독(HTTP 200 + 본문 파싱) 35건.** 그중 A급 인용의 출처:
  Schulz & Lerch 2022 (AMS PDF), Veldkamp et al. 2021 (arXiv PDF), Bouche et al. 2023 (저자 PDF),
  Messner et al. 2013 (Innsbruck WP PDF), Gilbert et al. 2020 (Glasgow eprints PDF),
  Lee et al. 2020 PCWG (NREL PDF), Bruninx et al. 2026 (arXiv HTML), GNN-in-WPF 2025 (arXiv HTML),
  WindDragon (arXiv HTML), Imani et al. 2026 JMLR (PDF), Grinsztajn et al. 2022 (NeurIPS PDF),
  Montero-Manso & Hyndman 2021 (arXiv 초록), 그리고 라이선스 8건(HF 모델카드·GitHub LICENSE 원문).
- **접근 실패**: ScienceDirect/Heliyon 403, AMS journals `/view/` 403, MDPI 403, Wiley(rmets) 403, SSRN 403, Springer 본문 미노출.
  해당 항목은 전부 [B] 이하로 강등해 표기했다.

---

## 10. 준수 확인

- 저장소 쓰기: `research/lanes/S13_S7_modelling_deep.md`, `research/lanes/S13_S7_modelling_deep.searchlog.json` — **2개뿐**
- 모델 적합 **0**, lockbox 접근 **0**, git 변경 **0**, 업로드 **0**, 외부 데이터 다운로드 **0**
  (판독한 PDF/HTML은 메모리 내에서만 파싱했고 디스크에 저장하지 않았다)
- 이 문서의 모든 수치는 §0 등급표에 따라 A/B/C/I/P5/X로 태깅되어 있다. **태그 없는 수치는 없다.**
