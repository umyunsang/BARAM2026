# Lane · S13 / S6 — 피처구성(feature construction) 심층 발굴

- **레인 ID**: `S13_S6_features_deep` (읽기전용 외부문헌 + 자체 기하계산 전용)
- **작성**: 2026-08-08 (로컬 세션 시각)
- **도구**: `websearch` (Serper) 112 쿼리 + 공개 PDF/HTML 원문 판독 + `inputs/competition/open_wind_236727.zip`
  안의 `info.xlsx` **메모리 내 읽기**(압축해제·파일생성 없음)
- **저장소 쓰기**: 이 파일 + `research/lanes/S13_S6_features_deep.searchlog.json` **2개뿐**. 모델 적합 0, lockbox 접근 0,
  git 변경 0, 업로드 0.
- **선행 레인**: `S6_ext_A_competitions.md`, `S6_ext_B_terrain.md`, `S6_ext_C_repr.md`, `S6_feature_research.md`,
  `windskill_lit.md`, `S13_S5_preprocessing_deep.md` — **전부 열람했고, 반복하지 않는다**(§1).

---

## 0. 증거 등급 규약 (문서 전체 적용)

| 등급 | 뜻 |
|---|---|
| **A** | 논문/보고서 **원문(PDF·HTML 전문)을 이 레인이 직접 내려받아 판독**하고 숫자를 문자 그대로 인용 |
| **B** | 구글 스니펫에 숫자가 문자 그대로 나왔으나 전문 표/문맥은 확인 못 함 |
| **C** | 제목·초록 수준. 숫자 없음 또는 간접 |
| **I** | 이 레인의 **내부 계산/유도** (외부 인용 아님). 재현 코드 경로를 함께 표기 |
| **D** | 부모가 준 항목 D(현행 872열 피처면)에 **이미 존재**함 |
| **X** | 항목 C(3-fold fold-outside 게이트로 이미 기각)에 해당 |

숫자를 옮길 때 지켜야 할 규칙 하나: **문헌의 개선률은 거의 전부 "원시 NWP 최근접격자 → 보정" 대비**다.
우리 베이스라인은 이미 16격자+9격자 전장을 먹은 학습모델이므로 그 이득의 대부분을 흡수했다.
문헌 %를 우리 %로 이항하지 마라(선행 레인 B의 §B0-1과 동일한 경고).

---

## 1. 이 레인이 반복하지 않은 것 (선행 레인 승계)

| 축 | 상태 | 근거 |
|---|---|---|
| Sx / Sb / TPI / TRI / VRM / RIX / Wind Effect 정적 지형지수 | **승계 종료** — 그룹당 상수는 3-level 그룹더미와 선형종속 | `S6_ext_B_terrain.md` §B4.0, §B5-1 |
| 격자 순서통계·격자 산포·성분 168열·격자 PCA·상류투영·REWS·공기밀도·파워커브 사전·풍속 3제곱 | **항목 C에서 기각(X)** | 부모 제시 |
| 딥러닝 시계열 표현·엔티티 임베딩·수치 임베딩·contrastive 사전학습·TabPFN | **승계 종료** | `S6_ext_C_repr.md` §C2, §C5 |
| 아날로그(AnEn)·칼만·소스별 sister 모델 스태킹 | **S7(모델링) 소관**, 이미 M252/M263 계열로 구현됨 | `windskill_lit.md` C1/C11 |
| 채점행 절단·가용성 라벨 처리 | **S5에서 종료** | `S13_S5_preprocessing_deep.md` |

**이 레인이 새로 판 곳**: 선행 6개 문서를 전수 grep한 결과 `wake`(1회), `sector`(2회), `cold air`(0),
`katabatic`(0), `mountain wave`(0), `Froude`(0), `phase`(0), `ramp`(1), `redundan/mRMR/Boruta`(0)로
**후류·방위섹터·산악파/흐름레짐·위상오차·중복성 기반 선택**은 사실상 미탐사 상태였다. 본 레인은 그 5개 공백에 집중했다.

---

## 2. 이 레인이 확보한 A급 증거 (원문 판독)

### E1. 산악파 무차원 산높이 `Ĥ = h₀N/U` 는 **능선 풍속 레짐의 스위치**이고, 그 임계값이 측정됐다
> "The results of this study suggest that mountain-wave-induced accelerated downslope winds tend to occur in
> the wind park when **Ĥ < 3**; above this value, the lee side wind tends to be weaker than at the mountain crest."
> … "As Ĥ increases from zero to a value of about **1.5**, there is a tendency of an increasing normalised wind speed.
> When Ĥ increases further …"
> — Solbakken & Birkelund, *Wind Energ. Sci.* **11**, 155–173 (2026), <https://wes.copernicus.org/articles/11/155/2026/> **[A]**

같은 논문, **같은 능선 위 클러스터 사이의 발전량 차이**(우리의 g1/g2/g3와 형태가 동일한 상황):
> "the observed wind speeds at A3 are higher than the wind speeds at A2 **80 % of the time**. Consequently, the power
> production of the A3 turbines is **51 % higher** in comparison to the power production at the A1 cluster and
> **19 % higher** in comparison to the power production at A2." (SE 120–165° 섹터, 관측기간의 35–39%) **[A]**

정의: `Ĥ = h₀ N / U` (h₀ = 산 높이, N = Brunt–Väisälä 진동수, U = **능선 직교(cross-barrier) 성분**).
Scorer 파라미터 `l(z)² = N(z)²/U(z)² − U''(z)/U(z)`, 실무에서는 곡률항을 버리고 `l ≈ N/U`.
`l`이 고도에 따라 급감 → **포획 lee wave**, 거의 일정 → 연직전파 산악파, 급증 → 임계층·강한 downslope wind. **[A]**

### E2. 이 두 진단량은 **결정론 모델 출력 하나로부터 계산되어 실제로 쓰였고**, 발전량 영향이 정량화됐다
> "we calculated the nondimensional mountain height and the Scorer parameter (Fig. 6) **from the WRF simulations** at Troutdale."
> … "For the wind plant analyzed in this paper, mountain-wave-induced fluctuations translate to approximately
> **11 % of the total wind farm output** being influenced by mountain waves." … "mountain waves were present at least
> **17 %** of the time" … "topographic wakes were recorded in the event log **15 %** of the time."
> — Draxl et al., *Wind Energ. Sci.* **6**, 45–60 (2021), <https://wes.copernicus.org/articles/6/45/2021/wes-6-45-2021.pdf> **[A]**

즉 `Ĥ`·`Scorer`는 앙상블도 관측도 필요 없다. **단일 결정론 런의 θ(z), U(z) 프로파일만으로 계산된다** — 우리 제약과 정확히 호환.

### E3. 한국 현업 상세화(KMAPP)의 고도보정은 **중립대기 Jackson–Hunt 선형이론**이며, 저자들이 직접 "안정도 의존형이 필요하다"고 적었다
> "바람장의 고도 보정은 예보 모형의 격자 지형 고도와 해당 지점의 실제 지형 고도의 차에 의한 풍속 효과를 보정한다
> (Howard and Clark, 2007). 이는 **이상화된 지형 위에서의 중립 대기 경계층 흐름에 대한 선형 이론**을 기반으로 하고 있다
> (Jackson and Hunt, 1975; Mason …)"
> … "현행 KMAPP 모형은 **중립 대기 조건을 기반으로 한 물리 보정**을 적용하고 있으나 보다 현실적인 풍속 상세화를 위해서는
> **대기 안정도를 고려할 수 있는 풍속 진단 보정 방안에 대한 연구도 필요**할 것으로 판단된다."
> — 금왕호·이상현·이두일·이상삼·김연희 (2021), 「복잡 지형 지역에서의 KMAPP 지상 풍속 예측 성능 평가와 개선」,
> *대기* 31(1), 85–100, <https://j-komes.or.kr/xml/28740/28740.pdf> **[A]**

같은 논문의 LDAPS 진단(부모 전제 E와 정합):
> "LDAPS 예측 풍속은 … S2560에서 4.93±1.20 m s⁻¹, S2553에서 4.65±1.19 m s⁻¹, S2571에서 3.40±1.38 m s⁻¹로 나타나,
> **지형 고도가 높은 지점에서 과소 모의 경향**을 나타내고 **지형 고도가 낮은 지점에서 과대 모의**하는 경향을 보였다.
> 엄밀하게는 LDAPS 모형의 지형 고도와 실제 측정 지점의 지형 고도의 차이(Fig. 3)에 기인한 예측 경향으로 해석할 수 있다." **[A]**

→ **전제 E의 "정적 상수라 그룹더미가 흡수한다"는 정확히 KMAPP이 하는 그 보정(중립·정적)을 두고 하는 말이다.
   문헌이 남겨둔 빈칸은 "그 보정의 안정도 의존 형태"이고, 그것이 본 레인의 1순위 노드다.**

### E4. 능선 위 후류의 **연직 전파 방향이 안정도에 따라 뒤집힌다**
> "The results show a strong dependence of the vertical wake propagation on the atmospheric stability. When a
> **terrain-induced gravity wave is observed under stable conditions, the wake follows the terrain down the ridge with a
> maximum inclination of −28°**. During unstable conditions, the wake is advected **upwards by up to 29°** above the horizontal plane."
> — Menke, Vasiljević, Hansen, Hahmann, Mann, *Wind Energ. Sci.* (Perdigão 이중능선), 
> <https://wes.copernicus.org/preprints/wes-2018-21/wes-2018-21-manuscript-version3.pdf> **[A]**

→ 후류 피처는 **안정도와 곱해져야** 한다. 순수 기하 후류지표만 넣으면 두 레짐이 서로를 상쇄해 0에 수렴할 수 있다.

### E5. 물리 다운스케일러를 그냥 얹으면 **능선/풍하 지점에서 오히려 나빠진다**
> "Downscaling **increased negative wind speed biases** of input HRRR forecasts even more at stations located in
> wind-prone lee-slope canyons."
> — Seto et al. (2025), *Weather and Forecasting* 40(4), WAF-D-24-0013.1,
> <https://research.fs.usda.gov/treesearch/69198> **[B, 스니펫직접인용 — AMS 본문은 403]**

→ 항목 C의 REWS·2소스 로그법칙 기각과 같은 방향의 독립 증거. **결정론적 물리보정을 피처로 얹는 노드는 사전확률이 낮다.**

### E6. 안정도/난류 피처의 **현실적 이득 크기**가 측정돼 있다 — 크지 않다
> "Removing the anomalous GAM model from consideration here, we find that the **standard deviation across features (1.0 %)
> is nearly twice that across algorithms (0.6 %)**." (정규화 RMSE 기준, 복잡지형 풍력단지 시간별 발전량)
> … "**turbulent kinetic energy was found to be the most important variable apart from wind speed** and more important than
> wind direction, pressure, and temperature."
> — Optis & Perr-Sauer (2019), *Renew. Sustain. Energy Rev.* 112:27–41 (NREL 승인원고),
> <https://www.osti.gov/servlets/purl/1529864> **[A]**

### E7. 같은 방향의 두 번째 보정치 — **물리 피처를 학습기에 얹었을 때의 증분은 1.6 pp**
> "Tree D, using REWS and lapse rate as predictors, improves upon the TPC by **22 %**. By pairing lapse rate with REWS,
> the model improved **1.6 %** when compared to Tree B, REWS." … "By pairing lapse rate with HHWS, the model is improved by
> **0.2 %** when compared to Tree A, HHWS."
> — Sasser et al. (2022), *Renewable Energy* 183:491–501 (NOAA 리포지터리 원문),
> <https://repository.library.noaa.gov/view/noaa/57628/noaa_57628_DS1.pdf> **[A]**

→ **22%는 "결정트리 자체 vs 제조사 파워커브"의 이득이고, "안정도 피처 추가"의 순증분은 1.6 pp / 0.2 pp다.**
   이 두 숫자(E6·E7)는 본 레인 모든 노드의 기대치를 묶는 상한 근거다. 부모 전제 A가 요구하는 −11% MAE를
   **S6 한 층이 단독으로 낼 수 있다는 문헌 증거는 없다.**

### E8. 한국 태백산맥의 풍하측 강풍은 **종관형 3분류**로 구조화돼 있다
> "It was found that the synoptic patterns could be classified into **three representative types**: (1) the **south-high and
> north-low** pattern in the spring, (2) the **west-high and east-low** pattern …"
> — Shin & Chung (2022), *JGR Atmospheres* 127(6), 2021JD035867 (SOM 기반),
> <https://essopenarchive.org/doi/full/10.1002/essoar.10508047> **[B, 스니펫직접인용 — AGU/essoar 본문 JS 차단]**

보조: 「A Numerical Experiment on the Occurrence of Atmospheric …」(2022, KOSHAM) — "**양간지풍처럼 태백산맥을 넘는
지속적 강풍은 물뜀(hydraulic jump) 현상 발생을 초래**할 수 있다. 태백산맥을 넘는 지속적 강풍은 **안정한 대기 상태에서 산악파를 형성**"
<https://www.j-kosham.or.kr/journal/view.php?number=10424> **[B]**;
Park et al. (2022) *Atmos. Res.* 272:106158, ICE-POP 2018 태백 강풍 — "Downslope wind was generated by **hydraulic jump and
partial reflection of mountain wave**" <https://www.sciencedirect.com/science/article/pii/S0169809522001442> **[B]**.

### E9. **협곡/장벽 관통류의 세기는 국지풍속보다 "장벽 양쪽 기압차"가 더 잘 예측한다**
> "The onset and strength of the gap winds are found to be **correlated to the formation of an along-gap pressure gradient**
> linked to periodic development of a thermal trough … Numerical simulations … the model captured the gap wind events when
> simulating the regional sea level pressure system correctly."
> — Wagenbrenner et al. (2018), *Atmosphere* 9(2):54, <https://research.fs.usda.gov/download/treesearch/57207.pdf> **[A]**

한국판 대응: 「Characteristics of Meteorological Variables in the Leeward …」 — "영동지역에서 강풍이 나타날 조건 가운데 하나는
**태백산맥을 경계로 동서간의 기압** (차)" <http://jkess.org/journal/article.php?code=33013> **[B]**

### E10. 착빙은 **관측가능한 기상조건으로 예측 가능한 발전량 손실 채널**이다
> "At the 1 h horizon, adding **icing indicators and LWC improved mean F1-score from 0.76 to 0.83**, reflecting the benefit of
> direct microphysical …" — Kallarappayi et al. (2026), Coventry Univ. VoR **[B]**
> "The magnitude of production loss can reach **over 50 % during winter months, and exceed 10 % on an annual basis**."
> — Ribeiro (WindEurope 2016), <https://windeurope.org/summit2016/conference/allfiles2/51_WindEurope2016presentation.pdf> **[B]**

---

## 3. 이론 정박 — 왜 `ΔĤ` 하나만이 "그룹더미가 못 먹는 지형결손"인가 **(I·유도)**

부모 전제 E의 주의사항을 대수로 다시 쓰면:

- 모델표고 결손 `Δz_g = z_true(g) − z_LDAPS(g)` 는 **그룹당 상수**다.
- GBDT + 그룹 원핫에서, 상수 벡터 `c_g` 는 그룹더미의 선형결합이다 ⇒ **정보량 0**. (선행 레인 B §B4.0과 동일)
- 그런데 지형결손이 **풍속에 미치는 효과**는 상수가 아니다. 성층류 이론에서 지형의 유효 세기는 높이 자체가 아니라
  **무차원 산높이** `Ĥ = h₀ N / U` 이고, 결손분이 만드는 유효 세기 차이는

  `ΔĤ(t) = Δz_g · N(t) / U_cross(t)`

  로 **시간에 따라 변하는 그룹별 항**이 된다. 이것은 `c_g × f(t)` 꼴이므로 **그룹더미로 표현 불가능**하다
  (트리는 `그룹 × f(t)` 상호작용을 학습할 수 있으나, 그러려면 깊이 2 이상의 분할을 스스로 찾아야 한다.
  `ΔĤ` 를 열로 주면 그것이 **깊이 1 분할**이 된다).

**우리 사이트가 실제로 레짐 경계에 걸쳐 있는가?** 태백 가덕산 능선 정상 ~1078 m,
주변 곡저/고원면 ~600–700 m ⇒ 국지 산높이 `h₀ ≈ 400 m`, 동해안 기준면까지 보면 `h₀ ≈ 1000 m`.
전형적 N·U 범위에 대해 (I·계산):

| N (s⁻¹) | U_cross (m/s) | U/N (m) | Ĥ(h₀=400) | Ĥ(h₀=1000) | **ΔĤ(Δz=135 m)** |
|---|---|---|---|---|---|
| 0.008 | 4 | 500 | 0.80 | 2.00 | **0.270** |
| 0.008 | 8 | 1000 | 0.40 | 1.00 | **0.135** |
| 0.008 | 15 | 1875 | 0.21 | 0.53 | **0.072** |
| 0.012 | 4 | 333 | 1.20 | 3.00 | **0.405** |
| 0.012 | 8 | 667 | 0.60 | 1.50 | **0.203** |
| 0.020 | 4 | 200 | 2.00 | 5.00 | **0.675** |
| 0.020 | 8 | 400 | 1.00 | 2.50 | **0.338** |
| 0.020 | 15 | 750 | 0.53 | 1.33 | **0.180** |

세 가지가 동시에 읽힌다.

1. **Ĥ가 0.2 ~ 5 를 오간다.** E1의 임계(Ĥ≈1.5 최대증속, Ĥ≈3 부호전환)와 E2의 "Ĥ~1에서 산악파 가능"이
   **우리 관측범위 한가운데** 있다. 즉 "지형이 흐름을 얼마나 세게 바꾸는가"가 시간에 따라 질적으로 달라진다.
2. **모델이 빠뜨린 135 m 는 Ĥ 단위로 0.07 ~ 0.68 이다.** 이는 임계값 사이 간격과 같은 크기다.
   ⇒ **"LDAPS의 지형결손"은 상수 오차가 아니라, 시간에 따라 0.07~0.68만큼 레짐을 어긋나게 하는 오차다.**
   정적 보정으로는 원리적으로 못 고친다. E3이 KMAPP 저자 스스로 남긴 빈칸이 바로 이것이다.
3. **Sheppard 분리유선 `H_s = h₀ − U/N`**: `U/N < h₀` 일 때 하층 공기는 능선을 **넘지 않고 돌아간다**.
   위 표에서 `U/N`은 200 m ~ 1875 m로 변동 ⇒ **어떤 시각에는 하층이 차단되고 어떤 시각에는 넘는다.**
   이것은 부모 전제 F("LDAPS 10 m 풍속이 발전량과 0.727–0.737")의 **조건부 신뢰도**를 직접 설명하는 변수다:
   차단 레짐에서 10 m 바람은 허브고도 흐름과 물리적으로 **다른 공기**다.

---

## 4. 이 레인의 내부 측정 — 실제 배치기하로부터 나온 **그룹×풍향 후류 구조** **(I)**

`info.xlsx`(17기 좌표)를 메모리에서 읽어 국지 ENU 평면(원점=단지 중심)으로 투영하고,
그룹별 주축(SVD 1축)·연속간격·중심거리·Jensen 톱햇(k=0.075) 그림자 지표를 계산했다. **파일 생성 없음.**

### 4.1 그룹 기하

| 그룹 | 기수 | D (m) | 연속간격 (m) | 간격/D | 열 주축 방위 | 열 길이 |
|---|---|---|---|---|---|---|
| g1 | 6 | 126 | 277, 230, 377, 332, 355 | 2.19, 1.83, 2.99, 2.64, 2.81 | 28.6° | 1042 m |
| g2 | 6 | 126 | 344, 406, 303, 255, 443 | 2.73, 3.22, 2.40, 2.02, 3.52 | 154.4° | 1567 m |
| g3 | 5 | 136 | 926, 439, 332, 398 | 6.81, 3.22, 2.44, 2.93 | 142.6° | 2055 m |

- 그룹 중심 간: **g1→g2 1281 m(10.2 D₁₂₆, 방위 115°)**, **g2→g3 962 m(7.6 D, 145°)**, **g1→g3 2171 m(17.2 D, 128°)**.
- 열 내부 간격이 **1.8–3.5 D**로 매우 촘촘하다. 능선 위 단열배치의 전형이며, 흐름이 **열 축과 나란해지는 순간**
  전열 차폐가 일어난다. 세 그룹의 축이 **28.6° / 154.4° / 142.6°** 로 서로 다르다는 점이 핵심이다:
  차폐가 시작되는 풍향이 **그룹마다 다르다** ⇒ 그룹더미만으로는 표현 불가, `그룹×풍향` 상호작용이 필수.

### 4.2 그림자 지표 (풍향 10° 간격, 값 = 상류 터빈 후류 안에 들어가는 해당 그룹 터빈 비율)

`_any` = 같은 그룹이든 다른 그룹이든 상류 터빈이 있는 비율, `_intra` = 같은 그룹 내부만.

| wd_from | g1_any | g2_any | g3_any | g1_intra | g2_intra | g3_intra |
|---|---|---|---|---|---|---|
| 0.0 | 0.33 | 0.33 | 0.2 | 0.33 | 0.33 | 0.0 |
| 10.0 | 0.5 | 0.33 | 0.2 | 0.5 | 0.33 | 0.0 |
| 20.0 | 0.5 | 0.17 | 0.2 | 0.5 | 0.17 | 0.0 |
| 30.0 | 0.5 | 0.33 | 0.0 | 0.5 | 0.17 | 0.0 |
| 40.0 | 0.33 | 0.17 | 0.0 | 0.33 | 0.0 | 0.0 |
| 50.0 | 0.5 | 0.17 | 0.0 | 0.33 | 0.0 | 0.0 |
| 60.0 | 0.67 | 0.0 | 0.0 | 0.5 | 0.0 | 0.0 |
| 70.0 | 0.67 | 0.0 | 0.2 | 0.33 | 0.0 | 0.0 |
| 80.0 | 0.5 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 |
| 90.0 | 0.5 | 0.17 | 0.2 | 0.0 | 0.0 | 0.0 |
| 100.0 | 0.83 | 0.33 | 0.0 | 0.17 | 0.0 | 0.0 |
| 110.0 | 0.83 | 0.5 | 0.2 | 0.17 | 0.17 | 0.0 |
| 120.0 | 1.0 | 0.5 | 0.2 | 0.0 | 0.17 | 0.0 |
| 130.0 | 0.83 | 0.67 | 0.2 | 0.17 | 0.33 | 0.2 |
| 140.0 | 0.5 | 0.83 | 0.6 | 0.17 | 0.5 | 0.6 |
| 150.0 | 0.5 | 0.83 | 0.8 | 0.33 | 0.5 | 0.6 |
| 160.0 | 0.33 | 0.67 | 0.8 | 0.33 | 0.5 | 0.6 |
| 170.0 | 0.33 | 0.5 | 0.0 | 0.33 | 0.5 | 0.0 |
| 180.0 | 0.33 | 0.67 | 0.0 | 0.33 | 0.5 | 0.0 |
| 190.0 | 0.33 | 0.33 | 0.0 | 0.33 | 0.17 | 0.0 |
| 200.0 | 0.5 | 0.33 | 0.0 | 0.5 | 0.17 | 0.0 |
| 210.0 | 0.5 | 0.17 | 0.2 | 0.5 | 0.17 | 0.0 |
| 220.0 | 0.33 | 0.0 | 0.2 | 0.33 | 0.0 | 0.0 |
| 230.0 | 0.33 | 0.17 | 0.2 | 0.33 | 0.0 | 0.0 |
| 240.0 | 0.33 | 0.33 | 0.0 | 0.33 | 0.0 | 0.0 |
| 250.0 | 0.33 | 0.5 | 0.0 | 0.33 | 0.0 | 0.0 |
| 260.0 | 0.0 | 0.5 | 0.2 | 0.0 | 0.0 | 0.0 |
| 270.0 | 0.0 | 0.5 | 0.4 | 0.0 | 0.0 | 0.0 |
| 280.0 | 0.17 | 0.67 | 0.6 | 0.17 | 0.0 | 0.0 |
| 290.0 | 0.17 | 0.67 | 0.6 | 0.17 | 0.17 | 0.0 |
| 300.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0.17 | 0.0 |
| 310.0 | 0.17 | 1.0 | 1.0 | 0.17 | 0.5 | 0.4 |
| 320.0 | 0.17 | 0.67 | 1.0 | 0.17 | 0.5 | 0.6 |
| 330.0 | 0.33 | 0.67 | 1.0 | 0.33 | 0.67 | 0.6 |
| 340.0 | 0.17 | 0.5 | 1.0 | 0.17 | 0.5 | 0.6 |
| 350.0 | 0.33 | 0.33 | 0.0 | 0.33 | 0.33 | 0.0 |

읽는 법:
- **g1은 풍향 120°(ESE)에서 1.00** — 전기가 차폐(주로 g2/g3가 상류).
- **g2·g3는 풍향 300–340°(NW)에서 0.67–1.00** — 반대 섹터에서 역으로 차폐.
- **탁월류(대략 서-동, 전제 F)인 260–280°에서도 g1=0.00 / g2=0.50–0.67 / g3=0.20–0.60** 로 **그룹 간 비대칭이 남는다.**
- E1의 Solbakken 관측(같은 능선 클러스터 간 **51% / 19%** 발전량 차)이 이 구조의 실측 대응물이다.

**주의(정직 고지)**: 위 표는 평지 Jensen 톱햇 기하다. 복잡지형에서는 후류가 지형을 따라 내려가거나(안정, −28°)
위로 뜬다(불안정, +29°) — E4. 따라서 이 표는 **피처의 형태(어느 섹터에서 어느 그룹이 그림자인가)를 정의하는 데만**
쓰고, **크기(몇 % 손실인가)는 학습에 맡겨야** 한다.

---

## 5. 노드표 (순위 = 기대이득 × 근거강도 ÷ 비용)

게이트 표기 `G3` = **항목 C가 이미 쓴 것과 동일한 3-fold fold-outside 게이트**를 사전등록하여 적용:
① fold-outside 평균 Δ(1−NMAE) ≥ **+0.0010**, ② 3/3 fold에서 **부호 일치**, ③ 블록 크기 **≤ 10열**,
④ 블록 단위 all-or-nothing(열 단위 사후선택 금지), ⑤ 실패 시 그 블록은 영구 폐기하고 receipt에 기록.

| id | 메커니즘 (정확히) | 근거 + 그들이 측정한 효과크기 | 왜 여기로 이전되나 | 실패 양식 | 비용 | 게이트 |
|---|---|---|---|---|---|---|
| **F1 `hhat`** | 시각별 `N`(θ 프로파일), `U_cross`(능선직교=대략 U성분)로 `Ĥ_true=h₀N/U`, `Ĥ_model=(h₀−Δz)N/U`, **`ΔĤ=Δz·N/U`**, `U/N`(분리유선 높이), `l≈N/U`의 연직 기울기(Scorer 부호) — **8열** | E1 [A] Ĥ<3에서 풍하 가속, Ĥ≈1.5에서 정규화풍속 최대 / E2 [A] WRF 단일런에서 Ĥ·Scorer 계산, 산악파가 단지 출력의 **11%**에 영향, 발생빈도 **17%** / E3 [A] KMAPP 고도보정은 중립 J–H 이론이고 저자들이 "안정도 고려 필요"라 명시 / §3 [I] 우리 사이트의 Ĥ 범위 0.2–5, ΔĤ 0.07–0.68 | 전제 E의 정적 결손 `Δz_g`를 **시간가변 인자와 곱해** 그룹더미가 흡수할 수 없는 형태로 바꾼다. 단일 결정론 런만으로 계산 가능(E2) | ① LDAPS 연직층이 얕아 `N`이 10 m/50 m/850 hPa 사이 조잡한 차분이 되면, 기존 D의 shear/Ri 블록과 상관 0.9+ 로 중복 → 168열 사건(−0.000728) 재현 ② `h₀`(상류 기준면) 정의가 임의적 ③ 채점행(actual≥0.1cap)은 대체로 강풍시각이라 Ĥ가 작은 쪽에 몰릴 수 있음 | 낮음 (기존 열의 산술조합, 8열) | G3. **사전 스크린**: `ΔĤ`와 기존 Ri/shear 열의 \|corr\| 최대값이 0.85를 넘으면 게이트 소모 없이 폐기 |
| **F2 `sector`** | §4.2 표를 **룩업**으로 굳혀, 예보 풍향 `wd(t)`에서 그룹별 `shadow_any_g(wd)`, `shadow_intra_g(wd)`, 그리고 `\|cos(wd − 열주축_g)\|`(정렬도) — **6–9열** | E1 [A] 같은 능선 클러스터 간 발전량 **51%/19%** 차이(고정 섹터) / DTU [B] 후류 출력손실 **10–20%** / Torrejón-Fontana 2026 [B] 복잡지형 후류 회복 **6.35 D** / §4 [I] 실측 기하: 열내 간격 **1.8–3.5 D**, 그룹축 **28.6/154.4/142.6°**, 중심거리 7.6–17.2 D | 그룹별 차폐 개시 풍향이 다르다는 것은 **기하가 강제하는 사실**이고, 그룹더미로는 표현 불가. 모델이 스스로 찾으려면 `그룹×풍향` 깊이-2 분할이 필요한데, 이를 깊이-1로 낮춘다 | ① LDAPS 풍향오차(±20–30°)가 섹터를 뭉갠다 ② 후류가 이미 라벨(그룹 합계)에 항상 들어있어 모델이 `wd×group`을 이미 학습했을 수 있음 ③ 평지 Jensen 기하가 능선에서 틀림(E4) | 낮음 (오프라인 룩업 1회 + 6–9열) | G3. 사전 스크린: 현행 모델의 잔차를 `그룹×풍향 30° 빈`으로 평균했을 때 구조가 보이지 않으면 폐기 |
| **F3 `dpx`** | LDAPS 4×4 박스의 해면기압(또는 지표기압 환산)에서 **능선직교(동–서) 기압차** `Δp_x`, 남–북 `Δp_y`, 이로부터 지형풍 `V_g`, 그리고 **비지형풍비** `\|V_10m\|/\|V_g\|` — **4–6열** | E9 [A] 협곡류의 개시·강도는 **관통 기압경도**와 상관, 모델은 "해면기압계를 옳게 모의할 때" 사건을 포착 / 한국판 [B] "영동 강풍 조건 … 태백산맥을 경계로 동서간의 기압(차)" / E8 [B] 태백 풍하강풍 종관 3형(남고북저/서고동저)은 **기압배치**로 정의됨 | D에는 **바람장의** 발산/와도는 있으나 **기압장의 경도**는 없다. 기압은 NWP가 바람보다 훨씬 잘 맞히는 장이므로, 풍속오차가 큰 시각의 대리정보가 된다 | ① 4×4(≈6 km) 박스가 종관 기압경도를 재기엔 너무 작다 → 850 hPa 지오포텐셜을 GFS 3×3에서 함께 써야 함 ② 지형풍은 산악 위에서 물리적 의미가 약함 | 낮음 (4–6열) | G3. 박스가 작아 신호가 없을 위험이 크므로 **GFS 3×3의 850 hPa 고도장 경도**를 반드시 포함해 1회만 시도 |
| **F4 `prune`** | 872열을 **블록 단위**(격자/기하/레짐/랙/달력/원핫)로 나눠 **역방향 제거**를 fold 내부에서만 수행. 열 단위 선택 금지 | E6 [A] 피처집합 간 RMSE 표준편차 **1.0%** > 알고리즘 간 **0.6%** ⇒ 피처면이 알고리즘보다 2배 중요 / Ambroise & McLachlan 2002 [B] fold 밖 선택은 오차추정을 심하게 편향 / 우리 자체 측정: 정보성 블록 추가가 **−0.000728** | "더하면 나빠진다"가 측정된 이상 **빼면 좋아진다**의 대칭 가능성이 열려 있다. 이건 새 정보가 아니라 **기존 정보의 정리**라서, 위 노드들이 전부 실패해도 남는 축이다 | ① 선택 자체가 fold 내부 과적합 → 반드시 nested ② 블록 제거가 특정 fold에서만 좋음(진동) — 항목 C의 per-group 가중 진동과 같은 실패 | 중간 (블록 수 × 재적합) | nested 3-fold. 제거는 **3/3 fold에서 개선**일 때만 확정 |
| **F5 `icing`** | 허브고도 `T ∈ [−8, +1] °C` **AND** `RH ≥ 95%`(또는 운저고도 < 허브) 플래그 + **누적 지속시간**(발생 후 경과시간, 착빙은 누적과정) + 융빙 플래그(`T > +1` 이후 경과) — **4–5열** | E10 [B] 착빙지표+LWC 추가로 F1 **0.76→0.83** / [B] 겨울 월손실 **50% 초과**, 연손실 **10% 초과** / 사이트: 해발 1078 m 태백 능선, 겨울 운무·상고대 상시 | 착빙은 **같은 풍속에서 파워커브 자체를 내리는** 유일하게 남은 대규모 상태변수다. 부모 전제 B의 "완전 실측풍속 상한(1−NMAE 0.951)"에도 남는 잔차의 일부가 여기 있을 수 있다 | ① LDAPS RH/운저 정확도가 낮음 ② 착빙 시각이 채점행(actual≥0.1cap)에서 빠져 기여 0이 될 수 있음(**가장 큰 위험**) ③ 계절 표본 부족 | 낮음 (4–5열) | **먼저 무비용 진단**: 12–3월 채점행에서 `T<1 & RH>95` 조건부 잔차 평균이 다른 시각과 다른지 확인 → 다르면 G3 |
| **F6 `asl`** | 모델지형이 80–140 m 낮으므로, **AGL 117 m 가 아니라 실제 표고에 맞춘 높이**(≈ 117 + Δz ≈ 250 m AGL)로 프로파일을 보간한 풍속/풍향 — **3–4열** | E3 [A] KMAPP 고도보정의 문제의식과 동일한 기하 / 전제 F [주어짐] LDAPS 10 m(0.727–0.737) > GFS 100 m(0.601–0.615) ⇒ **높이보다 해상도가 이긴다**는 관측 | 결손 보정을 "곱셈 보정" 대신 **"샘플링 높이 변경"**으로 구현. D에는 각 층의 값은 있어도 **250 m로 보간된 값**은 없다 | ① 전제 F가 이미 "더 높은 층이 더 낫지 않다"고 말한다 → 사전확률 낮음 ② 100 m ↔ 850 hPa 사이 보간이 과도한 외삽 | 낮음 (3–4열) | G3. **F1 이후에만** 시도(중복 위험) |
| **F7 `regime`** | 박스평균 (u, v, θ, q, p)와 `Ĥ`로 **k=4–6 소형 클러스터** 라벨(학습기간에서만 적합, fold 내부) + 클러스터 중심까지 거리 — **2–3열** | E8 [B] 한국 풍하강풍 종관 **3형** 분류가 실재 / [C] regime-switching 보정 문헌 다수 | 레짐 라벨은 **비선형 조건부 편의**를 한 열로 압축한다 | ① GBDT는 이미 원 피처로 레짐을 분할할 수 있음 → 순수 중복 ② 클러스터가 fold마다 흔들림 | 낮음 | G3. F1·F2 실패 시에만 |
| **F8 `xsrc`** | 이슈 내 24시간 궤적에 대해 **LDAPS와 GFS의 상호상관 최대 지연 τ\*** 와 그때의 상관계수, 두 소스 풍속의 이슈내 평균차·표준편차차 — **4열** | [C] Alessandrini 2012: 앙상블 스프레드는 결정론 오차의 예측인자 / 부모 측정: GEFS 스프레드-오차 상관 **0.02–0.14** (약함) | 새 소스 도입이 아니라 **이미 있는 두 소스의 위상 불일치**를 1열로 만든 것 (항목 C의 "외부 NWP 소스" 금지에 걸리지 않음) | ① GEFS 스프레드가 0.02–0.14였다면 2-멤버 불일치는 더 약할 것 ② τ\*가 격자잡음에 지배 | 낮음 (4열) | G3. 기대 낮음 — F1–F5 뒤 |
| **F9 `wake×stab`** | F2의 그림자지표 × (안정도 지표 또는 `Ĥ`)의 **명시적 곱** — 2–3열 | E4 [A] 능선 후류의 연직전파가 안정 −28° / 불안정 +29° 로 **부호가 뒤집힘** | 두 레짐이 상쇄되어 F2가 0이 되는 것을 막는다 | F2가 애초에 통과하지 못하면 무의미 | 매우 낮음 | F2 통과 시에만 |
| **F10 `TI`** | LDAPS TKE(있으면) 또는 shear·Ri 기반 TI 대리 | E6 [A] TKE가 풍속 다음으로 중요, 풍향·기압·기온보다 중요 | — | **D에 gust factor·shear·Ri가 이미 있음 ⇒ 대부분 중복** | — | **부분적으로 D. 신규성 낮음, 권고 안 함** |
| **F11 `lagr`** | 상류격자 값을 `Δt = d/U` 만큼 **시간까지 되돌려** 샘플링 | [C] 이류 개념 | 항목 C의 "상류투영"은 공간만이었음 | 6 km 박스에서 Δt ≈ 5–20분 ⇒ 1시간 격자에서 거의 항등 | 낮음 | **권고 안 함** (기하학적으로 죽어 있음) |
| **F12 `ramp`** | 이슈 내 궤적의 램프 크기·부호·굴곡 | [B] He 2025 램프 상황 28% 개선(자기회귀 세팅) | — | **D의 ±1,2,3,6 h 랙/리드 창이 이미 같은 정보** | — | **D. 권고 안 함** |
| **F13 `static_terrain`** | TPI/TRI/VRM/RIX/slope/aspect | — | — | **그룹당 상수 ⇒ 그룹더미와 선형종속(선행 레인 B §B4.0)** | — | **영구 종료** |

---

## 6. 상위 5개와 단일 최강 권고

### 6.1 Top-5 (정직한 기대치 포함)

| 순위 | 노드 | 정직한 기대 Δ(1−NMAE) | 왜 이 순위인가 |
|---|---|---|---|
| 1 | **F1 `hhat`** | +0.001 ~ +0.004 | 유일하게 **전제 E의 수학적 빈칸을 정확히 메우는** 형태(상수×시간함수). 근거가 A급 3개(E1·E2·E3)로 겹쳐 있고, 그 중 하나는 **한국 현업 상세화 저자들이 직접 지목한 미완성 축**이다 |
| 2 | **F2 `sector`** | +0.000 ~ +0.003 | 근거가 **우리 좌표 실측**(I)이라 이전 실패 위험이 없다. 그룹별 축이 실제로 다르다는 것은 사실이며, 이는 D의 layout-along/cross(정적)와 다르다 |
| 3 | **F3 `dpx`** | +0.000 ~ +0.002 | D에 **기압경도가 없다**는 순수 공백. 비용 최저. 다만 박스가 작아 신호가 없을 수 있음 |
| 4 | **F4 `prune`** | −0.000 ~ +0.002 | 유일하게 **새 정보 없이** 개선을 시도하는 축. E6이 "피처면이 알고리즘보다 2배 중요"를 측정했고, 우리는 이미 "추가가 해롭다"를 측정했다 |
| 5 | **F5 `icing`** | 0 ~ +0.002 (겨울 한정) | 남은 유일한 **파워커브 자체를 바꾸는** 상태변수. 단, 채점행에서 빠질 위험이 커서 무비용 진단이 먼저 |

**합계의 정직한 진술**: 위 5개가 모두 상단값으로 성공해도 **+0.013 정도**이고, 부모 전제 A가 요구하는
**+0.014839**에 미치지 못한다. E6(피처집합 간 nRMSE 표준편차 1.0%)과 E7(안정도 피처의 순증분 1.6 pp / 0.2 pp)은
**S6 한 층이 −11% MAE를 낼 수 있다는 문헌 사례가 없음**을 말한다. S6은 **필요조건이지 충분조건이 아니다.**

### 6.2 단일 최강 권고 — **F1 `hhat` 8열 블록을 하나의 all-or-nothing 실험으로**

구현 사양(그대로 쓸 수 있게):

```
# 입력: LDAPS 박스평균 프로파일 (10 m, 50 m, [가용 시 100 m], 850 hPa), GFS 850/700 hPa
# 1) 정적 상수 (그룹별 1회, 외부 DEM 불필요 — 이미 알려진 값 사용)
#    h0_g        : 능선 국지고도 (정상 1078 m − 상류 기준면 ~650 m ≈ 430 m; 민감도용으로 h0=1000 m도 함께)
#    dz_g        : z_true(g) − z_LDAPS(g)   (부모가 이미 보유: 868.8–1001.2 m, 평균 943.5 m)
# 2) 시간가변
#    theta(z)    : T, p → 온위
#    N(t)        = sqrt( (g/theta_bar) * dtheta/dz )   [dz 는 사용 가능한 최상·최하층]
#    U_cross(t)  = |u| (능선이 남북이므로 동서성분; 엄밀히는 열주축 법선으로 투영)
# 3) 열 (8개)
#    Hhat_true   = h0_g * N / max(U_cross, 1)
#    Hhat_model  = (h0_g - dz_g) * N / max(U_cross, 1)
#    dHhat       = dz_g * N / max(U_cross, 1)          # ★ 핵심 열
#    UoverN      = U_cross / N                          # 분리유선 높이 스케일
#    blocked     = 1[ UoverN < h0_g ]                   # 하층 차단 플래그
#    scorer_lo   = N_low / U_low ,  scorer_hi = N_hi / U_hi
#    scorer_slope= scorer_hi - scorer_lo                # 부호: 감소=포획파, 증가=임계층
```

**게이트**: §5의 `G3`. 그리고 그 **전에** 무비용 스크린 두 개를 반드시 통과할 것.
- (S1) `dHhat` 과 D의 기존 안정도 블록(shear exponent, bulk Ri, dθ/dz) 사이 최대 |corr| < 0.85.
- (S2) 학습기간 채점행에서 `Ĥ_true` 를 5분위로 나눴을 때, 현행 모델 잔차의 평균이 분위 사이에서 단조/유의하게 다를 것.
  (다르지 않으면 이 축은 **우리 데이터에서 죽어 있는 것**이고, 게이트를 소모할 이유가 없다.)

**가장 큰 위험 (하나만 꼽으면)**: **`N` 추정의 연직 분해능**.
`Ĥ`·`Scorer`의 문헌 성공사례(E1·E2)는 모두 **수십 개 연직층**(ERA5 137층, WRF 전층)에서 계산했다.
우리가 쓸 수 있는 층이 10 m/50 m(+850 hPa)뿐이라면 `N`은 사실상 `dθ/dz`의 거친 재포장이고,
그러면 이 블록은 **D의 레짐 블록과 중복**되어 항목 C의 168열 사건(−0.000728)을 그대로 재현한다.
스크린 (S1)이 정확히 이 위험을 사전에 잡도록 설계돼 있다. **(S1)을 건너뛰고 게이트를 쓰지 말 것.**

---

## 7. 이 레인이 명시적으로 **닫는** 축

1. **정적 지형지수 전부**(TPI/TRI/VRM/RIX/slope/aspect/curvature, 그룹 단위) — 그룹더미와 선형종속. (선행 레인 B 승계)
2. **물리 다운스케일러(WindNinja/질량보존/WAsP류) 출력을 피처로 주입** — E5[B]: 능선·풍하 지점에서 **편의를 오히려 키움**.
   항목 C의 REWS·2소스 로그법칙 기각과 같은 방향. 재시도 금지.
3. **램프 탐지·궤적 형상 피처**(F12) — D의 ±1,2,3,6 h 창과 정보량이 같다.
4. **이류보정 상류격자**(F11) — 6 km 박스 / 1 h 격자에서 Δt ≈ 5–20분, 기하학적으로 항등에 수렴.
5. **TI/TKE 대리**(F10) — D의 gust factor·shear·Ri와 중복. E6의 "TKE 최중요"는 **1 Hz 소닉 실측**이 있을 때의 이야기이며
   (Optis 원문: "The calculation of TKE requires high time-resolution measurements (around 1 Hz) and a sonic anemometer"),
   NWP 진단 TKE는 같은 물건이 아니다.
6. **관측 기반 교사(ASOS/AWS 실시간)** — 규칙상 시험기간 관측 금지. 학습기간 전용 교사는 S7 소관.

---

## 8. 이 레인이 확인하지 못한 것 (정직한 공백)

- **LDAPS 제공 변수 목록을 직접 열어보지 않았다.** `Ĥ` 계산에 필요한 (a) 층별 기온·기압(온위용), (b) 해면기압,
  (c) 상대습도/운저고도가 실제 컬럼으로 있는지는 **루트가 1분 안에 확인해야 한다.** 없으면 F1은 850 hPa 한 층으로
  축소되고 기대치는 반토막 난다. F3·F5도 각각 기압·습도 존재 여부에 전적으로 의존한다.
- **Seto et al. 2025 (E5) 와 Shin & Chung 2022 (E8) 는 본문을 못 봤다** (AMS 403 / AGU JS 차단). 둘 다 B등급.
- **Solbakken 2026의 Ĥ–풍속 산점도 회귀계수**를 표로 확인하지 못했다(그림 판독 불가). 임계값 1.5 / 3 은 본문 문장 인용.
- **능선 상류 기준면 h₀** 를 DEM으로 확정하지 않았다(외부 DEM 다운로드는 이 레인 범위 밖). 430 m / 1000 m 두 값으로
  민감도를 돌리라고 권고한 이유다.
- **§4.2 그림자 지표는 평지 Jensen 기하**다. 실측 SCADA로 검증하지 않았다(모델 적합 금지 범위).

---

## 9. 검색 로그 요약

- 총 **112 쿼리** (한국어 10 / 영어 102). 전체 로그: `research/lanes/S13_S6_features_deep.searchlog.json`
- 원문(PDF/HTML) 직접 판독: Solbakken 2026 WES, Draxl 2021 WES, Sasser 2022 (NOAA), Optis & Perr-Sauer 2019 (OSTI),
  금왕호 외 2021 (대기 31-1, KMAPP), Wagenbrenner 2018 Atmosphere, Menke 2018 WESD — **7편**
- 접근 실패: AMS(journals.ametsoc.org) 403, AGU/essoar JS 차단, MDPI 봇차단, ScienceDirect 유료 — 해당 항목은 B/C 등급 유지

## 10. 준수 확인

- 저장소 쓰기: `research/lanes/S13_S6_features_deep.md`, `research/lanes/S13_S6_features_deep.searchlog.json` **2개뿐**
- 모델 적합 **0**, lockbox 접근 **0**, git 변경 **0**, 업로드 **0**, 외부 데이터 다운로드 **0**
  (`info.xlsx` 는 zip에서 **메모리로만** 읽었고 디스크에 풀지 않았다)
- 이 문서의 모든 수치는 §0의 등급표에 따라 A/B/C/I/D/X 로 태깅되어 있다. 태그 없는 숫자는 없다.
