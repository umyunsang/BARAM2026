# Lane · windskill_lit — 단일 결정론 NWP → 복잡지형 허브고도 나셀풍속 다운스케일링/MOS 문헌

조사일: 2026-08-07 · 조사도구: `websearch`(Serper/Google) 41 쿼리 · 원시 로그: `windskill_lit.searchlog.json`
레인 성격: **읽기 전용**. 저장소 쓰기는 `research/lanes/` 아래 2개 파일뿐. 모델 fit·락박스·업로드 없음.

---

## 0. 증거 등급과 열람 범위 (감사용 고지)

**중요 — 본 레인이 실제로 읽은 것:** 검색 엔진이 반환한 **제목·초록 스니펫·인용수**다.
PDF 전문(full text)을 내려받아 읽지 않았다(외부 조회를 `websearch`로 한정한 제약 때문).
따라서 아래 모든 수치는 **초록/스니펫 수준 인용**이며, 각 항목의 `검증도` 칸에 그 사실을 명시했다.
전문 확인이 필요한 항목은 `[전문미확인]`으로 표시했다. 이를 지우고 인용하지 마라.

**태그 규칙**
- `[directly_supported]` — 문제 유형(단일 결정론 NWP → 지점 풍속/풍력, 리드타임 12~36 h, 복잡지형 또는 풍력단지)이
  우리와 일치하고, 우리 제약(앙상블 없음/단일 초기화/재분석 금지/실황 금지) 안에서 **그대로 구현 가능**한 것.
- `[near_match_only]` — 메커니즘은 실증됐으나 (a) 도메인이 다르거나 (b) 우리 제약 하에서 **핵심 입력이 빠져** 부분 구현만 가능한 것.
- `[speculative]` — 개별 논문 근거는 있으나 우리 세팅으로의 이전이 논리적 추론에 의존하는 것.

**우리 제약 체크리스트 (각 항목 3번에서 이 5개로 판정)**
| 코드 | 제약 |
|---|---|
| E | 앙상블 멤버 없음 (LDAPS·GFS 각 1개 결정론 런) |
| I | 하루 1회 09 KST 초기화, D+1 01시~D+2 00시 24시간 1회분만 |
| R | 재분석(ERA5/MERRA 등) 사용 금지 |
| O | 평가기간(2025) 관측 실황·SCADA 나셀풍속 **없음**, 축차 피드백 없음 |
| G | 격자 원시장 LDAPS 16점(≈1.5 km) + GFS 9점(0.25°)만, 고해상 시뮬레이션 없음 |

---

## 1. 후보 15개

### C1. Analog-based post-processing (AN / KFAN / AnEn) — 단일 결정론 런 전용으로 설계된 유일한 계열 `[directly_supported]`

**1) 메커니즘.** 과거 예보 아카이브에서, 현재 예보 벡터(풍속·풍향·기온 등 다변량, ±몇 시간 시간창 포함)와
**같은 리드타임에서** 가장 가까운 과거 예보들을 검색한 뒤, 그 시점들의 **관측값**을 모아 예측을 만든다.
평균을 쓰면 결정론 예측(AN), 분포로 쓰면 확률 예측(AnEn)이 된다. NWP 앙상블이 전혀 없어도 "과거 유사 상황의
관측 분포"가 앙상블 역할을 대신하므로, 본 계열은 **결정론 단일 런을 위해 만들어진 방법**이다. KFAN은 유사 시점들의
잔차에 칼만 필터를 겹쳐 최근 편의를 추가로 제거한다.

**2) 보고된 개선폭.**
- Delle Monache et al. (2011): WRF 10 m 풍속, 400개 지상관측소, 기준선=원시 WRF. 스니펫 원문 그대로 —
  "AN is consistently the best, with average improvements of **10%, 20%, 25%, and 35%**" (4개 지표를 나열한 것으로 보이나
  스니펫에 지표 라벨이 없다 → **지표 귀속 미확인** `[전문미확인]`).
- Plenković et al. (2018), 복잡지형(크로아티아, ALADIN): "A8 RMSE는 시험한 모든 후처리로 유의하게 감소하며,
  **KF 단독보다 analog 계열(AN, KFAN, KFAS)이 더 크게** 감소". 절대값은 스니펫에 없음 `[전문미확인]`.
- Pappa et al. (2023), Renewable Energy: AnEn이 수치모델 대비 **풍속 예측 스킬 25~43% 개선**(태양복사는 13~24%).
- Alessandrini et al. (2015), Renewable Energy, 인용 195: 단기 풍력 확률예측에 AnEn 최초 적용.

**3) 우리 데이터 실행 가능성 — 가능(E✔ I✔ R✔ O✔ G✔).**
필요한 것은 "학습기간 예보 아카이브 + 같은 기간의 타깃"뿐이고 둘 다 있다. 2025 평가기간에 관측이 없어도
검색 풀은 **학습기간에만** 있으면 되므로 O 제약을 위반하지 않는다. I 제약(하루 1회 초기화)도 무해하다 —
오히려 리드타임이 항상 동일하게 정렬돼 analog 매칭이 깨끗해진다.
**단, 본 저장소 사정으로 한계가 명확하다:** 이미 M252(analog/retrieval)가 구현돼 온라인 Total 0.6268784를 냈고
분류기 계열 M261(0.6365274)보다 **낮다**. 즉 "발전량을 직접 analog로 뽑는" 형태는 이미 소진됐다.
**미시도 영역은 타깃을 바꾸는 것**: 발전량이 아니라 **허브고도 풍속**을 analog로 예측하고, 그 결과를
기존 분류기의 추가 피처(또는 잔차 보정항)로 넣는 형태. 이는 본 레인이 찾은 가장 값싼 미개척 축이다.
풀 깊이는 학습기간 일수에 비례하므로, 다변량 매칭 차원을 3~4개로 제한해야 한다(고차원이면 이웃이 비어버림).

**4) 출처.**
- Delle Monache, Nipen, Liu, Roux, Stull (2011) "Kalman Filter and Analog Schemes to Postprocess Numerical Weather Predictions", *Mon. Wea. Rev.* 139(11) — 인용 309~312. https://opensky.ucar.edu/system/files/2024-08/articles_17411.pdf
- Odak Plenković, Delle Monache, Horvath, Hrastinski (2018) "Deterministic Wind Speed Predictions with Analog-Based Methods over Complex Topography", *J. Appl. Meteor. Climatol.* 57(9) — 인용 22. https://journals.ametsoc.org/view/journals/apme/57/9/jamc-d-17-0151.1.xml
- Delle Monache, Eckel, Rife, Nagarajan, Searight (2013) "Probabilistic Weather Prediction with an Analog Ensemble", *Mon. Wea. Rev.* — 인용 454~464.
- Pappa et al. (2023) "Analog versus multi-model ensemble forecasting", *Renewable Energy* — 인용 10. https://www.sciencedirect.com/science/article/pii/S0960148123000393
- Alessandrini et al. (2015) *Renewable Energy* 76:768-781 — 인용 195.

---

### C2. 칼만필터/축차 적응 편의보정 (state-space MOS) — **개선폭은 크지만 우리 세팅에서 메커니즘이 죽는다** `[near_match_only]`

**1) 메커니즘.** 예보-관측 잔차를 상태변수로 두고, 새 관측이 들어올 때마다 칼만 이득으로 편의 추정치를 갱신한다.
계절 변화·모델 업데이트·계기 드리프트에 자동 적응하는 것이 장점이며, 이 "적응"이 이득의 원천이다.

**2) 보고된 개선폭.**
- 허브고도 단기 풍력예보 KF 편의보정: RMSE **3.58 → 3.01 m/s (−16%)** `[전문미확인, ResearchGate 스니펫]`.
- Xu J.J. et al. (2020) 비선형 KF: 실시간 풍속 RMSE **3.26 → 2.21 m/s (−32%)**.
- Louka et al. (2008) *JWEIA*, 인용 597: 풍력용 풍속 수치예보의 후처리로 칼만필터 적용 — 계열의 표준 레퍼런스.
- Delle Monache 2011(C1)에서 KF는 analog 계열보다 **열등**하다고 보고됨.

**3) 우리 데이터 실행 가능성 — 사실상 불가 (O 위반).**
칼만 갱신은 **직전 시점의 관측-예보 쌍**을 필요로 한다. 2025 평가기간에는 나셀풍속이 없고, Dacon 오프라인
테스트셋 구조상 **발전량 실측의 축차 공개도 없다**. 학습 마지막 시점에서 상태를 얼어붙이면 KF는
**정적 상수 편의 보정**으로 퇴화하고, 위 −16%/−32% 수치의 근거인 "적응"은 사라진다.
게다가 본 프로젝트의 전역 메모리(frozen group-offset transfer)가 이미 같은 결론을 다른 경로로 측정해 놓았다:
얼린 그룹 오프셋을 미래에 적용할 때 최적 스케일은 `r_yy · sd_apply / sd_fit`이며 1이 아니다.
즉 **얼린 편의는 수축(shrink)해서 써야 하고, 개선폭은 문헌치보다 훨씬 작다.**
→ **권고: 이 축에 예산을 쓰지 마라.** 문헌의 큰 숫자는 우리에게 이전되지 않는다.

**4) 출처.**
- Louka, Galanis, Siebert, Kariniotakis, Katsafados, Pytharoulis, Kallos (2008) "Improvements in wind speed forecasts for wind power prediction purposes using Kalman filtering", *J. Wind Eng. Ind. Aerodyn.* — 인용 597. https://minesparis-psl.hal.science/hal-00505993/document
- "System bias correction of short-term hub-height wind forecasts using the Kalman filter" (2021) — https://www.researchgate.net/publication/356381248
- Xu et al. (2020) "Nonlinear Kalman filter bias correction for wind ramp event prediction" — 인용 2.
- Glahn (2014) "Determining an optimal decay factor for bias-correcting MOS…" — 인용 23. https://ams.confex.com/ams/94Annual/webprogram/Manuscript/Paper232884/

---

### C3. Grid-to-Point 딥러닝 오차보정 (G2N) — **우리 16격자 구조와 형태가 정확히 같다** `[near_match_only]`

**1) 메커니즘.** 지점 주변 NWP 격자 **패치**(2D 필드 여러 변수)를 CNN 인코더에 넣고, 여러 지점의 관측을
동시에 회귀하는 다중 출력 헤드를 붙인다. 지점별 개별 회귀와 달리, 격자 **수평 구조**(경도·곡률·이류 패턴)를
모델이 스스로 뽑아내며, 다중 지점 공유 학습이 정규화 역할을 한다.

**2) 보고된 개선폭.** Qin et al. (2023), 미세격자 NWP(PRUFS) 기준선 대비 RMSE 감소:
**10 m 풍속 −42%**, 2 m 기온 −19%, 2 m 상대습도 −24%.
관련: Tan et al. (2024) NFC-Net은 ECMWF 대비 2 m 기온 −49.71%, 10 m 풍속 −50%대 `[전문미확인]`.
Zhou S. et al. (2023) *GMD* 16:6247, WRF 10 m 풍속 하이브리드 보정 — 인용 20.

**3) 우리 데이터 실행 가능성 — 부분 가능 (E✔ I✔ R✔ O✔, **G가 병목**).**
- 구조적 일치: LDAPS 16격자 = 4×4 패치, GFS 9격자 = 3×3 패치. 이 논문들이 쓰는 패치와 **형태가 같다**.
- 그러나 4×4에서 convolution은 사실상 **완전연결(MLP)과 동치**다. 커널 3×3 한 층이면 유효 수용영역이 이미 전체를 덮는다.
  → **CNN 아키텍처 자체는 이 크기에서 의미가 없다.**
- 이전 가능한 잔여물은 아키텍처가 아니라 **표현**이다: 4×4 격자에서 명시적으로 계산 가능한
  **수평 경도(∂u/∂x, ∂v/∂y), 발산·회전, 곡률, 이류항 u·∇u, 격자 간 분산/최대-최소**를 테이블 피처로 만들어
  기존 GBDT에 넣는 것. 이는 CNN이 학습으로 찾아낼 양을 손으로 준 것과 같고, 2.6만 행 규모에서는 이쪽이 유리하다
  (본 저장소 L2 레인 C8: 중간 규모 표형 데이터에서 트리 > DL).
- 또한 다중 출력 헤드는 우리에게 **터빈 17기 / 그룹 3개 동시 회귀**로 그대로 대응된다.
- 주의: 위 논문들의 타깃은 **관측이 있는 지상관측소 10 m 풍속**이다. 우리 타깃(허브 117 m 나셀풍속)은
  학습기간에만 존재하므로 학습은 가능하되 평가기간 검증이 불가하다 → 폴드 외부(fold-outside) 검증 필수.

**4) 출처.**
- Qin, Y. et al. (2023) "Grid-to-Point Deep-Learning Error Correction for the Surface Weather Forecasts of a Fine-Scale NWP System", *Atmosphere* 14(1):145 — 인용 15. https://www.mdpi.com/2073-4433/14/1/145
- Zhou, S. et al. (2023) "A robust error correction method for numerical weather prediction wind speed", *GMD* 16:6247 — 인용 20. https://gmd.copernicus.org/articles/16/6247/2023/
- Tan, L. et al. (2024) *Acta Meteorologica Sinica* 82:539 — 인용 7.

---

### C4. 지점 중심 소격자 패치 CNN 후처리 (Veldkamp / Liu) — 공간정보가 확률예보 스킬을 올린다 `[near_match_only]`

**1) 메커니즘.** C3와 같은 계열이나 **확률 출력**에 초점. 지점 중심 소격자 영역의 순환장(circulation) 특징을
CNN으로 추출하고, 분위수 또는 분포 파라미터를 직접 출력한다. Veldkamp et al.은 CNN vs 분위수회귀숲(QRF) vs
완전연결 NN을 같은 데이터로 비교했다.

**2) 보고된 개선폭.** Veldkamp et al. (2021), *MWR* 149(4):1141 — "**Convolutional neural networks outperform
quantile regression forests and fully connected neural networks, in terms of CRPS, in all the three
cross-validation** [설정]". **개선폭 숫자는 초록 스니펫에 없다** `[전문미확인]`. 인용 86~93.
Liu, Q. et al. (2023) *Atmospheric Research* — "타깃 지점 중심의 **작은 격자 영역**의 공간 순환 특징을 추출하는
CNN 기반 모델을 설계" — 인용 44. 개선폭 수치 미확인 `[전문미확인]`.

**3) 우리 데이터 실행 가능성 — 부분 가능, 단 **핵심 미확인 사항 있음**.**
- **미확인:** Veldkamp et al.의 입력이 HARMONIE-AROME **결정론 런**인지 **앙상블**인지 본 레인은 확정하지 못했다.
  앙상블이라면 E 제약으로 그대로는 못 쓴다. **전문 확인 전에 이 항목을 근거로 결정하지 마라.**
- 확률 출력 자체는 우리에게 결정적으로 중요하다 — 본 대회 점수의 절반이 FICR(밴드 적중)이고, 저장소 기록상
  배포 예측은 조건부 평균이 아니라 **계단형 보상 하 기대 정산 최대화 행동**이다. 즉 조건부 **분포**를 얻는 것이
  정답이고, 점 추정을 잘 맞추는 것이 아니다. 이 계열은 그 분포를 주는 계열이다.
- 그러나 C3와 동일하게 4×4에서 CNN의 이점은 소멸한다 → **LightGBM 분위수 회귀 + 공간 파생 피처**가
  같은 정보를 더 싼 값에 준다는 것이 본 레인의 판단(`[speculative]` 수준의 추론임을 명시).

**4) 출처.**
- Veldkamp, Whan, Dirksen, Schmeits (2021) "Statistical Postprocessing of Wind Speed Forecasts Using Convolutional Neural Networks", *Mon. Wea. Rev.* 149(4):1141-1152 — 인용 86~93. https://journals.ametsoc.org/downloadpdf/journals/mwre/149/4/MWR-D-20-0219.1.pdf
- Liu, Q. et al. (2023) "Deep-learning post-processing of short-term station wind speed forecasts", *Atmospheric Research* — 인용 44. https://www.sciencedirect.com/science/article/pii/S0169809523004295

---

### C5. Wind-Topo — 극복잡지형 지형인지 딥러닝 다운스케일링 `[near_match_only]` (완전형은 **불가**)

**1) 메커니즘.** 조대(1.1 km) COSMO-1 대기장 + **고해상(50 m) 지형**을 함께 입력받아, 지형과 대기 상태의
상호작용(수속·박리·차폐·산악파)을 딥러닝으로 학습해 지점 풍속을 예측한다. 학습 타깃은 **다수 관측소**의 실측이며,
지형-흐름 상호작용 커널이 **관측소 간 공간 일반화**로부터 학습된다는 점이 본질이다.

**2) 보고된 개선폭.** 60개 **독립** 관측소에서 기준선(원시 COSMO-1) 대비
**bias 0.72 → −0.07 m/s**, **MAE 1.77 → 1.21 m/s (−31.6%)**. 스위스 알프스, 극복잡지형.

**3) 우리 데이터 실행 가능성 — 완전형 불가 (G·O 위반), 부분 이전만 가능.**
- 치명적 불일치: 우리는 **단일 사이트(가덕산 1곳)**다. 지형 기술자(고도·경사·곡률·TPI·TRI·풍상거리)는
  한 지점에서 **상수**가 되어 피처로서 정보가 0이다. Wind-Topo의 이득 원천(지형↔흐름 커널의 다지점 일반화)이
  구조적으로 존재하지 않는다.
- **살아남는 잔여물 하나:** 지형 기술자를 **풍향에 조건부로** 만들면 상수가 아니다. 즉 예보 풍향 θ를 따라
  상류 방향 지형 노출도(예: Winstral의 `Sx` 풍상 차폐 지수, 상류 경사, 상류 고도차)를 계산하면
  `f(θ)`가 되어 시변한다. 이는 **공개 DEM(외부 공개 데이터 — 규칙상 허용)** 만으로 계산 가능하고 fit이 아니다.
  터빈 17기 × 풍향 36섹터 노출 테이블은 단 한 번 계산하면 되고, 그룹별로 다른 값을 갖는다.
  → **이것이 C5에서 우리가 실제로 가져갈 수 있는 유일한 항목이며, 아직 시도된 적이 없다** (본 레인 판단, `[speculative]`).
- 주의: 고해상 DEM은 "2026-07-05 이전 공개 + 상업적 이용 허용" 라이선스 확인이 필요하고, 출처·라이선스·취득일을
  영수증에 기록해야 한다(AGENTS.md 외부 데이터 조항).

**4) 출처.**
- Dujardin, J. & Lehning, M. (2022) "Wind-Topo: Downscaling near-surface wind fields to high-resolution topography in highly complex terrain with deep learning", *Q. J. R. Meteorol. Soc.* 148 — 인용 94. https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4265
- 수치 출처(bias/MAE): AMS 20th Mountain Meteorology 발표초록 https://ams.confex.com/ams/20MOUNTAIN/meetingapp.cgi/Paper/402970
- 모델 공개: https://www.envidat.ch/metadata/wind-topo_model

---

### C6. CNN 풍속장 다운스케일링 아키텍처 비교 — **"구조보다 예측인자"**라는 교훈 `[near_match_only]`

**1) 메커니즘.** Höhlein et al.은 단순 선형 CNN부터 깊은 잔차 CNN까지 여러 구조를 **같은 데이터**로 비교하여
근지표 바람 단기예보의 통계적 다운스케일링에서 무엇이 실제로 성능을 좌우하는지 분해했다.

**2) 보고된 개선폭.** 초록 스니펫에는 절대 개선폭이 없다 `[전문미확인]`. 인용 168.
계열 내 후속 벤치마크(Schmidt et al., EGU24-5980; Sekiyama 2023 *AIES* 인용 22)도 같은 문제를 다룬다.
**본 레인이 이 항목을 넣는 이유는 개선폭이 아니라 설계 우선순위 교훈** 때문이다:
"아키텍처 깊이보다 **어떤 예측인자를 넣는가**(상층 지위고도·기온·습도 등 추가 필드)가 성능을 더 좌우한다"는
관측이 반복 인용된다 — 다만 이 문장 자체는 **본 레인이 초록에서 직접 확인하지 못했다** → `[speculative]` 강등 대상.

**3) 우리 데이터 실행 가능성 — 교훈만 이전 가능.**
아키텍처 탐색(E✔ I✔ R✔ O✔ G✔이지만 4×4에서 무의미)보다, **LDAPS/GFS 아카이브에 실제로 어떤 변수·연직층이
들어있는지 전수 조사**하고 미사용 필드(지위고도, 층후, 상층 풍속, 경계층고도, 지표 플럭스, 안정도 관련 변수)를
전부 후보 피처로 올리는 쪽에 예산을 쓰라는 것이 이 항목의 실행 함의다. 이는 fit 없이 **인벤토리 작업**으로 시작할 수 있다.

**4) 출처.** Höhlein, Kern, Hewson, Westermann (2020) "A comparative study of convolutional neural network models for wind field downscaling", *Meteorological Applications* 27(6):e1961 — 인용 168. https://rmets.onlinelibrary.wiley.com/doi/10.1002/met.1961

---

### C7. GBDT 기반 NWP 풍속 후처리 (피처 중요도 주도) `[directly_supported]`

**1) 메커니즘.** WRF 등 결정론 NWP의 다변량 출력(풍속·풍향·기온·기압·습도, 격자/연직층)을 그대로 테이블 피처로
만들고, GBDT로 **관측 풍속을 직접 회귀**한다. 잔차보정(additive)이 아니라 **관측 타깃 직접 회귀**라는 점이 핵심 —
NWP 풍속은 여러 예측인자 중 하나로 강등되며, 모델이 조건부 비선형 편의(풍속 구간별·풍향별·안정도별)를 스스로 학습한다.
저자들은 피처 중요도로 어떤 변수가 실제로 기여하는지 사후 분해했다.

**2) 보고된 개선폭.** Xu, W. et al. (2020): 원시 WRF 풍속 RMSE **2.7~3.5 m/s → 1.0~1.5 m/s 감소**
(= 대략 **−30% ~ −43%**), DTR(결정트리회귀)·기타 baseline보다 우수. 인용 76.
동일 계열: MDPI *Atmosphere* 17(6):549 — 24 h 예보 10 m 풍속 RMSE **1.47 → 1.10 m/s (−25.2%)**.
**이 마지막 수치가 우리 상황과 가장 가깝다** — 기준선 RMSE 1.47 m/s는 우리 현재 그룹평균 풍속 RMSE
1.49~1.78 m/s와 거의 같은 크기이고, 달성된 감소폭 −25.2%는 **우리 목표(−25~30%)의 하단과 정확히 일치**한다.
즉 **우리 목표는 문헌상 달성 사례가 있는 크기**이며, 비현실적 목표가 아니다.

**3) 우리 데이터 실행 가능성 — 완전 가능 (E✔ I✔ R✔ O✔ G✔).**
제약 위반이 하나도 없다. 이미 본 저장소가 GBDT 계열을 쓰고 있으므로 **새 모델이 아니라 새 타깃/새 피처**의 문제다.
구체적으로: (a) 타깃을 발전량에서 **허브고도 풍속**으로 바꾼 중간 모델을 하나 만들고,
(b) 그 예측 풍속을 발전량 모델의 피처로 주는 2단 구조. 학습기간에 SCADA 나셀풍속이 있으므로 (a)의 지도학습이 가능하다.
평가기간에 나셀풍속이 없다는 사실은 **중간 모델의 출력이 예측값이므로 문제되지 않는다**(O 위반 아님).

**4) 출처.**
- Xu, W. et al. (2020) "Wind Speed Forecast Based on Post-Processing of Numerical Weather Predictions Using a Gradient Boosting Decision Tree Algorithm", *Atmosphere* 11(7):738 — 인용 76. https://www.mdpi.com/2073-4433/11/7/738
- "Improving 10 m Wind Speed Forecasts over the Northwest …", *Atmosphere* 17(6):549. https://www.mdpi.com/2073-4433/17/6/549 (연도·저자 미확인 `[전문미확인]`)

---

### C8. 연직층 바람 특성 통합 피처 추출 — 복잡지형 day-ahead 풍력, **NMAE 지표, 한국** `[directly_supported]`

**1) 메커니즘.** 복잡지형 풍력단지의 day-ahead 예측에서, NWP가 모의한 **여러 연직층의 바람 특성**
(층별 풍속·풍향, 층간 시어·베어)을 통합해 피처를 구성한다. 단일 고도(10 m 또는 최근접층)만 쓰는 관행 대신
연직 프로파일 전체의 형태를 예측인자로 삼는 것이 핵심이며, 복잡지형에서 지표층 바람과 허브고도 바람의 결합이
느슨해지는(디커플링) 상황을 모델이 구분할 수 있게 한다.

**2) 보고된 개선폭.** 초록 스니펫이 **NMAE**를 평가지표로 명시한다 — 본 조사에서 발견한 문헌 중
**우리 대회 지표(0.5·(1−NMAE) + 0.5·FICR)와 지표 계열이 일치하는 유일한 논문**이다.
그러나 **개선폭 수치 자체는 스니펫에 노출되지 않았다** `[전문미확인]`. 인용 22.
→ **본 레인의 최우선 후속 조치: 이 논문 전문 확보 및 수치 추출.**

**3) 우리 데이터 실행 가능성 — 가능하되 **선행 확인 1건 필요** (E✔ I✔ R✔ O✔, G는 조사 필요).**
필요한 것은 LDAPS/GFS 아카이브가 **복수 연직층 바람**을 담고 있는지다.
- GFS 0.25°는 통상 10 m, 80 m/100 m, 그리고 다수 기압면(1000/975/950/925/900 hPa …) 바람을 포함한다.
  가덕산 정상부 고도를 감안하면 925~900 hPa 부근이 허브고도(117 m AGL)에 대응한다.
- LDAPS도 모델층/기압면 바람을 제공한다.
- **아카이브에 실제로 어느 층이 들어있는지는 본 레인이 확인하지 않았다**(읽기 전용 레인이지만 파일 조사는 가능했음에도
  범위를 문헌으로 한정). → 루트가 `inputs/competition/open_wind_236727.zip` 컬럼 인벤토리를 먼저 확인해야 한다.
- 만약 단일 고도만 제공된다면 이 후보는 **불가**로 강등된다. 이 조건부성을 반드시 명시하고 인용하라.

**4) 출처.** Lee, K., Park, B., Kim, J. et al. (2024) "Day-ahead wind power forecasting based on feature extraction integrating vertical layer wind characteristics in complex terrain", *Energy* 288 — 인용 22. https://www.sciencedirect.com/science/article/pii/S0360544223031080 · 프리프린트 https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4509803

---

### C9. 대기안정도 층화 피처 (bulk Richardson, Monin-Obukhov, 경계층고도) `[near_match_only]`

**1) 메커니즘.** 지표층 풍속 → 허브고도 풍속 외삽은 안정도에 강하게 의존한다. 안정(야간·역전) 조건에서는
시어가 커져 지표 바람으로부터 허브 바람을 과소평가하고, 불안정(주간 대류) 조건에서는 프로파일이 평평해진다.
따라서 NWP에서 bulk Richardson 수 `Ri_b`, Monin-Obukhov 길이 `L`, 경계층고도 `blh`를 계산해
(a) 피처로 넣거나 (b) 안정도 계급별로 별도 보정계수를 두는 층화 회귀를 쓴다.

**2) 보고된 개선폭.**
- Optis & Perr-Sauer (2019) *Renew. Sustain. Energy Rev.* 112:27-41, 인용 110~112 — 초록 원문:
  "We find a **considerable improvement in hourly power predictions** when some measure of turbulence or stability
  is included in the model." **정량 수치는 스니펫에 없음** `[전문미확인]`.
- Cantero et al. (2022) *Wind Energy Science* 7:221, 인용 19 — 복잡지형에서 sonic 기반 안정도가 더 정밀하지만
  **bulk Richardson이 단순·강건·저비용**이며 실용적이라고 결론.
- Lee, T.R. et al. (2020, NOAA), 인용 33 — 복잡·불균질 지형에서 **MOST(Monin-Obukhov 상사이론)의 적용 타당성 자체를
  검증**했고, 이상적 조건에서 벗어날수록 성립이 약해진다는 방향의 평가. `[전문미확인]`

**3) 우리 데이터 실행 가능성 — 부분 가능, **기대 이득은 문헌치보다 크게 작을 것**.**
- 계산 자체는 가능(E✔ I✔ R✔ O✔ G✔): NWP의 2개 이상 층 기온·풍속만 있으면 `Ri_b`는 닫힌 형태로 계산된다.
  `blh`는 LDAPS/GFS 표준 산출 필드다.
- **결정적 감쇄 요인 두 가지.**
  (i) Optis & Perr-Sauer의 이득은 **실측** 난류강도/안정도를 넣었을 때의 것이다. 우리는 **예보된** 안정도를 쓰므로
      그 자체가 오차를 갖는다. 예보 안정도의 신호대잡음이 낮으면 이득은 대부분 소멸한다.
  (ii) Lee et al. (2020) / Cantero et al. (2022)가 지적하듯 **복잡지형에서 MOST 가정이 약하다**.
      가덕산은 산악 복잡지형이므로 MOST 기반 물리 보정식을 그대로 쓰는 것은 위험하다.
- **권고 형태:** MOST 보정식을 물리적으로 강제하지 말고, `Ri_b`·`blh`·층간 시어·주야 플래그를 **GBDT 피처로만** 넣고
  모델이 층화를 스스로 학습하게 하라. 물리식 강제(hard constraint)는 근거가 약하다.
- 추가 경고: 본 저장소의 fold-outside 게이트는 자유도가 늘어난 구성을 반복적으로 기각해 왔다
  (7멤버×3그룹 21 dof, 그룹별 가중 3 dof 모두 기각). **안정도 계급별 별도 보정계수는 dof를 늘리는 구성**이므로
  in-sample 개선이 나와도 fold-outside에서 무너질 확률이 높다. 피처 추가(dof 증가 없음)로만 접근하라.

**4) 출처.**
- Optis, M. & Perr-Sauer, J. (2019) "The importance of atmospheric turbulence and stability in machine-learning models of wind farm power production", *Renew. Sustain. Energy Rev.* 112:27-41 — 인용 110~112. https://www.osti.gov/pages/biblio/1529864
- Cantero, E. et al. (2022) "On the measurement of stability parameter over complex mountainous terrain", *Wind Energy Science* 7:221 — 인용 19. https://wes.copernicus.org/articles/7/221/2022/
- Lee, T.R. et al. (2020) "Evaluation of Monin–Obukhov and Bulk Richardson…", NOAA — 인용 33. https://repository.library.noaa.gov/view/noaa/26140/noaa_26140_DS1.pdf

---

### C10. **분위수 매핑(QM)의 실패 — 인플레이션 함정** `[directly_supported]` (음의 결과)

**1) 메커니즘.** QM은 예보의 주변분포(marginal)를 관측의 주변분포에 강제로 맞춘다. 결과적으로
"분포는 맞지만 시점별로는 틀린" 예보가 만들어진다. 회귀 기반 예보는 필연적으로 분산이 축소되어 있는데(regression
attenuation), QM/인플레이션은 이 분산을 인위적으로 되돌린다. **분산을 되돌리면 결정론 오차(RMSE/MAE)는 반드시 나빠진다.**
Maraun (2013)의 핵심 논증은 QM이 **스킬을 추가하지 못하며**, 세분화(downscaling)로 오해될 때 특히 해롭다는 것이다.

**2) 보고된 결과.** 정량 개선폭이 아니라 **구조적 결론**: 인플레이션은 결정론 정확도와 분포 현실성 사이의
**정확한 트레이드오프**이며 공짜 이득이 아니다. Glahn (2016)의 코멘트와 Maraun의 반론이 이 트레이드오프의
경계 조건을 다툰다. Maraun (2013) *J. Climate* 26 — 인용 매우 높음(계열 리뷰 Maraun 2016은 인용 1122).

**3) 우리 데이터 실행 가능성 — 구현은 가능하나, **본 대회에서는 이미 이 함정에 대한 내부 측정이 존재**.**
본 저장소 AGENTS.md에 기록된 메커니즘과 **정확히 같은 구조**다:
"배포 예측은 계단형 보상 하 기대 정산을 최대화하는 **행동**이지 조건부 평균이 아니다. 조건부 평균 쪽으로 옮기면
점 정확도는 좋아지고 정산은 구조적으로 나빠진다." QM은 그 반대 방향(분포 맞추기 = 분산 확대)으로,
**FICR을 올리고 1−NMAE를 내린다**. 이미 내부적으로 오라클 일평균 보정이 1−NMAE +0.020365 / FICR −0.014849로
측정된 바 있는데, QM은 그 부호를 뒤집은 형태의 거래를 제안할 뿐 **합을 늘리지 않는다**.
→ **권고: 풍속 단계에서 QM을 쓰지 마라.** 단, 풍속 예측의 **조건부 분포**를 얻기 위한 분위수 회귀는 QM과 다르다.
QM은 무조건부 주변분포 매핑이고, 분위수 회귀는 조건부 분포 추정이다. **이 둘을 혼동하지 마라.**

**4) 출처.**
- Maraun, D. (2013) "Bias Correction, Quantile Mapping, and Downscaling: Revisiting the Inflation Issue", *J. Climate* 26(6):2137-2143.
- Glahn, B. (2016) "Comment on 'Bias correction, quantile mapping and downscaling: Revisiting the inflation issue'", *J. Climate* — 인용 2. https://repository.library.noaa.gov/view/noaa/63640/noaa_63640_DS1.pdf
- Maraun, D. (2016) "Bias Correcting Climate Change Simulations — a Critical Review", *Curr. Clim. Change Rep.* — 인용 1122. https://link.springer.com/article/10.1007/s40641-016-0050-x
- Guo, Z. et al. (2025) *Climate* 13(7):150 — QR50 / 선형회귀 / 기타 편의보정 3종 비교, 인용 2. https://www.mdpi.com/2225-1154/13/7/150

---

### C11. Sister-model 다중 NWP 소스 스태킹 (LDAPS 전용 + GFS 전용 + 메타) `[directly_supported]` — **L2 레인 중복**

**1) 메커니즘.** 두 NWP 소스를 한 모델에 섞어 넣지 않고, **소스별로 동일 구조 모델을 따로 학습**한 뒤
메타 학습기로 결합한다. 소스별 편의 구조가 다르기 때문에 개별 모델이 각자의 편의를 깨끗이 학습하고,
메타 단계가 소스 신뢰도를 상황별로 배분한다. 부수 효과로 한 소스가 결측되어도 시스템이 붕괴하지 않는다.

**2) 보고된 개선폭.** Pu et al. (2025), HEFTCom2024: 풍력 pinball loss
Case I 28.96(DWD)/30.44(GFS) → **27.13 (−6.3%)**, Case II 18.90/19.19 → **17.69 (−6.4%)**, CRPS 53.18→49.74 (−6.5%).
소스 수 확장 상한 참고: Bruninx et al. (2026) 5개 제공자 앙상블 → 점 정확도 평균 +17% (우리는 2소스이므로 훨씬 작음).

**3) 우리 데이터 실행 가능성 — 완전 가능. 단 본 저장소 L2 레인이 이미 최우선 스펙으로 등재했다.**
중복 계상하지 마라. **본 레인의 추가 기여는 하나뿐:** 스태킹 단위를 발전량이 아니라 **풍속**으로 내리면
(LDAPS→풍속 모델, GFS→풍속 모델, 메타→허브풍속) C7·C8과 자연스럽게 합성된다.

**4) 출처.** Pu, Fan, Tai, Liu, Yu (2025) arXiv:2505.10367v2 = *IJF* DOI 10.1016/j.ijforecast.2025.11.008 · Bruninx et al. (2026) arXiv:2602.13010v2. (본 저장소 `research/L2_wind_sota.md` C3·C4 참조)

---

### C12. 나셀풍속 자료동화(WRFDA) — 문헌상 유효하나 **우리에겐 완전 불가** `[directly_supported]` (불가 판정)

**1) 메커니즘.** 터빈 나셀 풍속계 관측을 WRF의 3D/4D-Var 자료동화 시스템에 직접 동화하여, 초기장 자체를
풍력단지 부근에서 수정한다. 후처리가 아니라 **모델 초기조건 수정**이므로 개선의 성격이 다르다.

**2) 보고된 개선폭.** Sun, W. et al. (2022) *Weather and Forecasting* 37(5) — 터빈 위치·허브고도 풍속 예보 개선.
정량치는 스니펫에 미노출 `[전문미확인]`. 인용 7.
관련 운영 사례: NCAR–Xcel Energy 시스템(Myers et al., 인용 46)은 Xcel이 실시간으로 공급하는
**터빈의 약 90%의 나셀 풍속계 데이터**를 받아 각 NWP 입력마다 나셀 풍속계 관측을 최적 예측하는 MOS를 생성한다.

**3) 우리 데이터 실행 가능성 — 불가. 3중 위반.**
(i) **O 위반** — 2025 평가기간에 나셀 풍속 관측이 존재하지 않는다. 동화할 관측이 없다.
(ii) **G 위반** — 우리는 격자 산출물 16+9점만 받는다. WRF/LDAPS를 다시 돌릴 초기·경계장이 없다.
(iii) 규칙 위반 위험 — 예보 기준시각(D−1 14:00 KST) 이후 관측 사용은 금지된다.
→ **이 축은 완전히 닫혀 있다.** 다만 Myers et al.의 **MOS 타깃 설계**(발전량이 아니라 나셀 풍속계 값을
직접 MOS 타깃으로 삼는 것)는 관측 없이도 이전 가능하며, 이는 C7의 2단 구조와 같은 아이디어다.

**4) 출처.**
- Sun, W. et al. (2022) "Improving Wind Speed Forecasts at Wind Turbine Locations…", *Wea. Forecasting* 37(5) — 인용 7. https://journals.ametsoc.org/view/journals/wefo/37/5/WAF-D-21-0041.1.xml
- Myers, W. et al. (NCAR) — 인용 46. https://opensky.ucar.edu/system/files/2024-09/conference_3296.pdf
- Parks, K. et al. (2011) NREL "Wind Energy Forecasting: A Collaboration of the NCAR and Xcel Energy" — 인용 52. https://docs.nlr.gov/docs/fy12osti/52233.pdf

---

### C13. CFD 없는 순수 진단형 지형 보정 (질량보존 모델 / Jackson-Hunt·WAsP류) `[speculative]`

**1) 메커니즘.** 두 갈래가 있다.
(a) **질량보존(mass-consistent) 진단 모델**(예: WindNinja): 조대 NWP 바람장을 고해상 지형격자에 내리고
    연속방정식(∇·(ρu)=0)을 만족하도록 최소 수정한다. CFD의 운동량 방정식을 풀지 않으므로 수초~수분에 끝난다.
(b) **Jackson-Hunt 선형이론 / WAsP**: 완만한 언덕 위 경계층 섭동의 준해석해로부터 **speed-up factor**를 계산한다.
    지형이 급하거나 박리가 있으면 이론 가정이 깨진다.

**2) 보고된 개선폭.** Forthofer et al. (2016, USDA FS): 4개 NWP 모델의 근지표 바람을 복잡지형에서 다운스케일할 때
질량보존 모델의 능력을 평가 — **정량 개선폭 스니펫 미노출** `[전문미확인]`.
Veronesi et al. (2016) *RSER*, 인용 62: WAsP는 Jackson-Hunt에 기반하며 **복잡지형을 다루도록 설계되지 않았다**고 명시.

**3) 우리 데이터 실행 가능성 — 사실상 불가 / 매우 낮은 기대값 (G 위반 + 검증 불가).**
- 질량보존 모델을 돌리려면 고해상 지형격자 + 외부 소프트웨어 실행이 필요하다. 이는 "fit은 아니지만" 상당한 구현이며,
  산출은 **바람 필드**이지 나셀풍속이 아니다. 필드→나셀 매핑은 결국 다시 통계 문제다.
- 더 근본적으로 **검증 관측이 단일 단지뿐**이라 필드의 공간 구조를 검증할 수 없다.
- Jackson-Hunt/WAsP 계열은 **가덕산 같은 급경사 산악에서 가정이 깨진다**고 문헌이 직접 말한다(Veronesi 2016).
→ **권고: 물리 지형모델을 돌리지 마라.** 대신 C5의 잔여물(풍향 조건부 지형 노출 지수)만 취하라.
   그것은 fit도 시뮬레이션도 아닌 **DEM 기반 결정론적 기하 계산**이다.

**4) 출처.**
- Forthofer, J.M. et al. (2016) — https://research.fs.usda.gov/treesearch/61477
- Veronesi, F. et al. (2016) "Statistical learning approach for wind resource assessment", *Renew. Sustain. Energy Rev.* — 인용 62. https://www.sciencedirect.com/science/article/abs/pii/S1364032115013830
- Troen & Petersen (WAsP 기반 이론) — http://www.wasp.dk/ (Jackson & Hunt 1975 기반)

---

### C14. KMAPP — **LDAPS 자체를 ML로 다운스케일한 한국 사례** `[near_match_only]`

**1) 메커니즘.** 기상청 KMAPP는 **LDAPS 출력**을 입력으로 100 m × 100 m 격자의 고해상 예보를 생성한다
(본질적으로 공간 다운스케일링). Shin et al.은 여기에 머신러닝(랜덤포레스트 등)을 결합해 지상 3 m 풍속 예보를
전국 규모로 생성하고, **전통적 다운스케일링 대비 ML 기반이 더 낫다**고 보고했다.

**2) 보고된 개선폭.** 초록 스니펫에 "머신러닝 기반 방법이 전통적 다운스케일링 방법보다 더 나은 성능"이라고만
기술되고 **정량 수치는 미노출** `[전문미확인]`. 인용 33.

**3) 우리 데이터 실행 가능성 — 직접 이전은 제한적, 그러나 **NWP 일치가 최대 가치**.**
- 우리와 **완전히 같은 NWP(LDAPS)** 를 다룬 유일한 발견 문헌이다. LDAPS의 알려진 편의 구조·해상도 한계·
  지형표현 오차에 대한 서술이 우리 사이트에 그대로 적용될 개연성이 가장 높다.
- 그러나 타깃이 **지상 3 m 농업용 풍속**이고 전국 관측망을 쓴다. 우리는 단일 지점 117 m다 → 모델 자체는 이전 불가.
- **실행 함의:** 이 논문(및 KMAPP 문서)에서 **LDAPS의 지형고도 오차 보정 방식**을 확인하라.
  LDAPS 1.5 km 격자의 모델 지형고도와 가덕산 실제 능선 고도의 **차이(Δz)** 가 크면, 격자 풍속은 사실상
  "실제보다 낮고 완만한 산"의 바람이다. 이 Δz를 공개 DEM으로 계산해 **격자별 고도 편차를 피처로 넣는 것**은
  값싸고 fit이 필요 없으며, 16격자 각각이 서로 다른 Δz를 가지므로 **공간 가중치를 학습시키는 데 직접 쓰인다**.
  → 본 레인이 판단하는 **가장 값싼 즉시 실행 항목**(`[speculative]` — 이 특정 형태의 문헌 근거는 없음).

**4) 출처.** Shin, J.Y., Kim, K.R. et al. (2022) "High-resolution wind speed forecast system coupling numerical weather prediction and machine learning for agricultural studies — a case study from South Korea" — 인용 33. https://pmc.ncbi.nlm.nih.gov/articles/PMC9151559/ · PubMed 35449427 (저널명 본 레인 미확인 `[전문미확인]`)

---

### C15. 풍향/기상레짐 조건부 전문가 모델 (sector-wise, K-means regime) `[speculative]`

**1) 메커니즘.** 풍향 섹터 또는 K-means로 정의한 기상 레짐별로 **별도 모델(전문가)** 을 학습하고,
예보 레짐에 따라 스위칭 또는 소프트 가중한다. 복잡지형에서 지형-흐름 상호작용이 풍향에 강하게 의존하므로
(능선 수속 / 후류 박리 / 계곡 채널링) 물리적 동기는 명확하다.

**2) 보고된 개선폭.** Cai, C. et al. (2026) *IJEPES*: 풍향 기반 사전분류 + K-means 클러스터링으로 기상 레짐을
구성하는 해상풍력 예측 프레임워크 — 인용 5. **정량 개선폭 미확인** `[전문미확인]`. 도메인은 **해상**(우리는 육상 산악).
Huang, C.L. et al. (2025) IEEE — NWP 풍속 편의보정 후 예측 정확도 개선, 인용 23, 정량치 미확인 `[전문미확인]`.

**3) 우리 데이터 실행 가능성 — 구현은 가능하나 **본 저장소 게이트가 이 형태를 반복 기각했다**.**
- E✔ I✔ R✔ O✔ G✔ — 예보 풍향은 있으므로 층화 자체는 가능하다.
- **그러나 결정적 반대 증거가 내부에 있다.** AGENTS.md 기록: 그룹별 가중(3 dof)이 in-sample 0.640253 →
  fold-outside 0.635453으로 균일가중 0.639170보다 **나빠졌고**, 폴드 간 가중이 진동했다(g3: 1.00/1.00/0.15).
  풍향 12섹터 전문가 모델은 그보다 훨씬 큰 dof다. **폴드 3개로 추정 불가.**
- **살릴 수 있는 형태 하나:** 별도 모델이 아니라 **풍향의 순환 인코딩(sin θ, cos θ)과 풍향×풍속 교호작용을
  단일 GBDT 안에 피처로** 넣는 것. 이는 dof를 늘리지 않으면서 같은 정보를 준다. 트리는 이미 이런 층화를 학습한다.
→ **권고: 명시적 레짐 스위칭 금지. 피처 인코딩으로만.**

**4) 출처.**
- Cai, C. et al. (2026) "Offshore wind power forecasting with wind-regime …", *Int. J. Electr. Power Energy Syst.* — 인용 5. https://www.sciencedirect.com/science/article/pii/S0142061525011019
- Huang, C.L. et al. (2025) "Enhancing Wind Power Forecasts via Bias Correction Technologies for NWP Model", IEEE — 인용 23. https://ieeexplore.ieee.org/document/10907933/

---

## 2. 판정 요약표

| # | 후보 | 태그 | 보고된 개선폭 | 우리 실행 가능성 | 제약 위반 |
|---|---|---|---|---|---|
| C1 | Analog post-processing (AN/KFAN/AnEn) | `directly_supported` | 풍속 스킬 25~43% (Pappa) / 10~35% (DelleMonache, 지표 불명) | **가능** — 단 발전량 타깃은 M252로 소진. **풍속 타깃 미시도** | 없음 |
| C2 | Kalman 축차 편의보정 | `near_match_only` | RMSE −16% / −32% | **불가** — 축차 관측 없음, 정적 오프셋으로 퇴화 | **O** |
| C3 | Grid-to-Point CNN (G2N) | `near_match_only` | 10 m 풍속 RMSE **−42%** | 부분 — 4×4에서 CNN 무의미. **공간 파생 피처로 치환** | G(부분) |
| C4 | 소격자 패치 CNN 확률후처리 | `near_match_only` | CRPS로 QRF·FCN 전승(수치 미확인) | 부분 — 입력이 앙상블인지 **미확인** | E(미확인) |
| C5 | Wind-Topo | `near_match_only` | MAE 1.77→1.21 (**−31.6%**), bias 0.72→−0.07 | 완전형 **불가**(단일 사이트). **풍향 조건부 지형노출만 이전** | G, 다지점 부재 |
| C6 | CNN 구조 비교 (Höhlein) | `near_match_only` | 수치 미확인 | 교훈만 — "구조보다 예측인자" | — |
| C7 | GBDT NWP 풍속 후처리 | `directly_supported` | RMSE −30~43% (Xu) / **1.47→1.10 = −25.2%** | **완전 가능** — 타깃을 풍속으로 내린 2단 구조 | 없음 |
| C8 | 연직층 바람 피처 (Lee 2024, **NMAE·한국**) | `directly_supported` | **수치 미확인(최우선 확보 대상)** | 가능 — **아카이브 연직층 존재 확인이 선행 조건** | 조건부 G |
| C9 | 안정도 층화 (Ri_b, MOST, blh) | `near_match_only` | "considerable improvement"(수치 미확인) | 부분 — 예보 안정도의 오차·복잡지형 MOST 붕괴로 감쇄. **피처로만** | 없음(이득 감쇄) |
| C10 | Quantile mapping 인플레이션 | `directly_supported` (음의 결과) | 개선 아님 — 구조적 트레이드오프 | **쓰지 마라** — NMAE↔FICR 제로섬 거래 | — |
| C11 | Sister-model 다중 NWP 스태킹 | `directly_supported` | pinball −6.3~6.4%, CRPS −6.5% | 가능 — **L2 레인 중복**. 풍속 단위로 내리는 것만 신규 | 없음 |
| C12 | 나셀풍속 자료동화 (WRFDA) | `directly_supported` (불가 판정) | 수치 미확인 | **완전 불가** | **O, G, 규칙** |
| C13 | 질량보존/Jackson-Hunt 지형보정 | `speculative` | 수치 미확인 | 낮음 — 급경사에서 이론 붕괴, 검증 불가 | G |
| C14 | KMAPP (LDAPS ML 다운스케일, 한국) | `near_match_only` | 수치 미확인 | 모델 이전 불가. **LDAPS 지형고도 편차(Δz) 피처화가 잔여물** | 없음 |
| C15 | 풍향/레짐 조건부 전문가 | `speculative` | 수치 미확인 | 구현 가능하나 **fold-outside 게이트가 기각할 형태**. 피처 인코딩으로만 | 없음(dof 위험) |

---

## 3. 본 레인의 결론 — 상위 3개와 그 이유

### 1위 · C7 (+C11 합성): **타깃을 발전량에서 허브고도 풍속으로 내린 2단 GBDT**
- 문헌이 **우리와 같은 크기의 기준선에서 −25.2% (RMSE 1.47→1.10 m/s)** 를 달성했다. 우리 목표(−25~30%)는
  **문헌상 달성 사례가 있는 크기**이며, 이것이 본 조사의 가장 중요한 발견이다 — 목표가 비현실적이지 않다.
- 제약 위반 0. 학습기간 SCADA 나셀풍속이 있으므로 중간 모델의 지도학습이 가능하고,
  평가기간에 나셀풍속이 없다는 사실은 **중간 출력이 예측값이므로 무관**하다.
- LDAPS 전용 / GFS 전용 sister model로 나눠 풍속을 각각 예측하고 메타 결합(C11)하면 계열이 합성된다.

### 2위 · C8: **연직층 바람 특성 피처** — 지표는 NMAE, 지형은 복잡지형, 국가는 한국
- 본 조사에서 찾은 **유일한 지표 계열 일치(NMAE) 문헌**이며, 복잡지형 day-ahead 풍력이라는 문제 유형도 일치한다.
- **선행 조건 1건:** `inputs/competition/open_wind_236727.zip`의 컬럼 인벤토리에서 LDAPS/GFS가 **복수 연직층 바람**
  (기압면 또는 모델층, 특히 925~900 hPa)을 담고 있는지 확인. 담고 있지 않으면 이 후보는 즉시 강등된다.
- 물리적 동기가 강하다: 허브고도 117 m는 지표층 상단이며, 복잡지형·안정 조건에서 10 m 바람과 디커플링된다.

### 3위 · C3의 치환형: **4×4/3×3 격자의 수평 구조를 명시적 파생 피처로**
- G2N이 10 m 풍속 RMSE **−42%** 를 낸 정보원은 "격자 패치의 수평 구조"다. 4×4에서는 CNN이 무의미하므로
  그 정보를 **손으로** 준다: 수평 경도, 발산·회전, 곡률, 이류항, 격자 간 분산/최대-최소, 그리고
  **LDAPS 모델지형고도와 실제 DEM 고도의 격자별 편차 Δz**(C14 잔여물).
- fit 없이 피처 생성만으로 시작 가능하고, dof를 늘리지 않으므로 fold-outside 게이트를 통과할 형태다.

### 반드시 피해야 할 3가지 (음의 발견)
1. **칼만/축차 적응 편의보정 (C2)** — 문헌의 −16~−32%는 축차 관측에서 나온다. 우리는 그것이 없다.
   얼린 오프셋은 `r_yy·sd_apply/sd_fit`로 수축해야 하며(내부 전역 메모리), 개선폭은 문헌치의 극히 일부다.
2. **분위수 매핑 (C10)** — NMAE와 FICR 사이의 제로섬 거래일 뿐 합을 늘리지 않는다.
   본 저장소가 이미 오라클 일평균 보정에서 같은 부호 구조를 측정했다(1−NMAE +0.020365 / FICR −0.014849).
3. **풍향/레짐별 별도 모델 (C15), 안정도 계급별 별도 계수 (C9의 층화형)** — dof 증가. 본 저장소 fold-outside
   게이트가 3 dof 구성조차 기각했다. 정보는 **피처 인코딩으로만** 주입하라.

---

## 4. 미해결 / 후속 조치 (본 레인이 하지 못한 것)

| 항목 | 왜 미해결인가 | 누가 처리해야 하나 |
|---|---|---|
| Lee et al. (2024) *Energy* 288 전문 및 NMAE 개선폭 | 페이월. `websearch` 스니펫에 수치 미노출 | 루트 — 전문 확보 후 수치 확정 |
| Veldkamp et al. (2021) 입력이 결정론인지 앙상블인지 | 초록 스니펫으로 확정 불가 | 루트 — 확인 전 C4를 근거로 결정 금지 |
| Delle Monache (2011)의 "10/20/25/35%"의 지표 귀속 | 스니펫에 라벨 없음 | 루트 — 인용 시 반드시 라벨 확인 |
| LDAPS/GFS 아카이브의 **연직층 인벤토리** | 본 레인은 문헌으로 범위를 한정했고 아카이브를 열지 않음 | 루트 — C8의 가부가 여기서 갈린다 |
| 고해상 DEM의 라이선스·공개일(2026-07-05 이전, 상업 이용 허용) | 미확인 | 루트 — C5/C14 잔여물 사용 전 영수증 필요 |
| 본 문서의 모든 수치는 **초록/스니펫 수준** | 전문 미열람 | 인용 시 `[전문미확인]` 태그를 지우지 말 것 |

---

## 5. 레인 준수 확인

- 저장소 쓰기: `research/lanes/windskill_lit.md`, `research/lanes/windskill_lit.searchlog.json` 2개뿐.
- 모델 fit: 없음. 락박스(2024) 접근: 없음. Dacon 업로드/브라우저: 없음. git 커밋: 없음.
- 외부 조회: `websearch` 스킬 41회. 다른 네트워크 도구 미사용.
- 시간 범위: 2026-08-07 단일 세션.
