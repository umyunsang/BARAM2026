# Lane L2 Findings — 풍력발전량 예측 SOTA · 벤치마크 · 우승 솔루션

조사일: 2026-08-06 / 조사시간 ~45분 / 확인 소스 21개 (아래 Source table)
도구 주의: 설치된 `websearch`(Serper) 키 미설정 → arXiv API · OpenAlex API · Crossref API · GitHub Search API ·
DuckDuckGo HTML(부분 차단)으로 대체 조사. **한국어 웹(DACON 코드공유 게시판) 검색은 사실상 불가**했음 → Q3 일부 insufficient.

**태그 판정 규칙(감사용 명시)**: 본 대회 지표(NMAE/FICR)와 정확히 같은 지표의 문헌은 존재하지 않는다.
따라서 (a) *지표에 의존하지 않는 주장*(문제유형=day-ahead NWP→풍력발전량, 표형 데이터, 리드타임 12~36h가 일치)에는
`directly_supported`를 부여하고 Scope 열에 지표 차이를 명시했다. (b) *개선폭이 우리 지표로 얼마나 옮겨가는가*에 대한
주장은 모두 `near_match_only` 이하로 강등했다. 반복·권위에 의한 승격은 하지 않았다.

---

## 1. Executive summary (5줄)

1. **day-ahead NWP→풍력 예측의 실전 SOTA는 여전히 GBDT다.** 2024년 실제 운영형 대회(HEFTCom2024, 3.6GW GB 포트폴리오, day-ahead)에서 상위권의 공통 요소는 *GBT + 다중모델 결합 + 피처선택 + 하이퍼파라미터 튜닝*이었고, 우승팀 SVK는 **NWP 소스별 CatBoost MultiQuantile → 선형 분위수회귀 메타모델 스태킹**이었다 `directly_supported`.
2. **가장 이식성 높은 정량 결과: "sister model 스태킹"**. 같은 구조 모델을 NWP 소스별로 따로 학습 후 스태킹 → 풍력 pinball(MPL) **-6.3%/-6.4%**, CRPS -6.5% (GEB팀, HEFTCom2024) `directly_supported`. 우리 데이터에 **LDAPS 전용 모델 + GFS 전용 모델 + 메타 스태커**로 그대로 대응된다.
3. **GEFCom2014 우승(Landry 2016) 패턴이 3그룹 구조와 정확히 맞는다**: zone별·분위수별 독립 GBM + 지배 입력신호 스무딩 + **상관 발전단지 정보를 쓰는 2-layer 모델링** `directly_supported`.
4. **분포예측 방법 선택은 이미 실증 비교가 있다**(벨기에 해상 9개 단지, 4년, day-ahead): nCRPS 평균 SVGP 6.3% / **NGBoost(Gaussian) 6.0% (최하위권)** / **CQR 5.7%** / **Treeffuser(튜닝) 5.6%**. NGBoost-정규분포는 음수 출력 구간을 만들어 물리적으로 부적합 `directly_supported`. 실험은 **Apple M3 8코어/24GB에서 수행** → M1 16GB 현실성 근거 `near_match_only`.
5. **딥러닝이 이긴 유일한 상위 사례(Rnt, 3위)는 사내 AI 기상모델 임베딩 + 관측/레이더/위성 데이터** 기반이었다 → 우리 규칙(원격 API 금지, 외부데이터 제약)·M1 환경에서 재현 불가. **표형 2.6만행에서 LSTM/Transformer가 GBDT를 이겼다는 in-scope 증거는 발견되지 않음** `insufficient`.

---

## 2. Evidence ledger

| Claim ID | Finding | Tag | Source locator | Scope match | Implication |
|---|---|---|---|---|---|
| C1 | HEFTCom2024 예측트랙: 상위팀 공통 = Gradient Boosting Trees + 다중모델 결합 + (검증기반) 피처선택 + HP 튜닝. 하위권은 EDA 기반 피처선택. 우승 SVK = NWP 소스(DWD/GFS/MEPS)별 **CatBoost MultiQuantile(9분위 동시)** + **각 분위수마다 27개 예측분위수를 입력으로 받는 선형 분위수회귀 메타모델** + 용량 클리핑. 튜닝은 **부스팅 반복수 하나만**. | directly_supported | Browell et al. (2025) "The Hybrid Renewable Energy Forecasting and Trading Competition 2024", arXiv:2507.01579v2 = IJF DOI 10.1016/j.ijforecast.2025.10.005 (§Forecasting track; Table 2) | day-ahead·NWP→풍력/태양광·운영형 대회·2024. **지표는 Pinball(NMAE/FICR 아님)** | 모델 패밀리 논쟁 종료. 아키텍처 탐색이 아니라 **스태킹·피처선택·검증설계**에 예산 투입 |
| C2 | 상위10 중 유일한 비트리 접근(Rnt, 3위)은 **사내 AI 기상모델 임베딩 → 다운스트림 NN**. 입력에 기상관측소·레이더·위성·NWP analysis 포함 | directly_supported | 동 S1 (§"Rnt is the most distinctive of the top-10…"), 참조 Andrychowicz et al. 2023 | 동일 대회·동일 리드타임 | DL 상위 진입은 **자체 AI-NWP + 관측 데이터**가 전제. 우리 규칙/HW에서 **재현 불가 → 추구 금지** |
| C3 | **Sister-model 스태킹**(동일 구조 LightGBM을 DWD 전용/GFS 전용으로 각각 학습 후 스태킹): 풍력 MPL Case I 28.96(DWD)/30.44(GFS) → **27.13 (-6.3%)**, Case II 18.90/19.19 → **17.69 (-6.4%)**; MCRPS 53.18→49.74 (-6.5%). 추가 이점: 온라인 시점에 한 NWP 소스가 결측돼도 붕괴하지 않음 | directly_supported | Pu, Fan, Tai, Liu, Yu (2025) "A Hybrid Strategy … HEFTCom2024", arXiv:2505.10367v2 = IJF DOI 10.1016/j.ijforecast.2025.11.008, Table 2 | day-ahead·2 NWP 소스·해상풍력·**MPL/CRPS 지표** | **L2 최우선 스펙**: LDAPS-only 모델 + GFS-only 모델 + 메타 스태커. 단일 모델에 두 NWP를 다 넣는 것보다 우수 |
| C4 | 5개 기상 제공자 앙상블 vs 단일 제공자 → **point 정확도 평균 17% 개선** (벨기에 해상 전체 단지, day-ahead) | near_match_only | Bruninx, van Binsbergen, Verstraeten, Nowé (2026) "Probabilistic Wind Power Forecasting with Tree-Based ML and Weather Ensembles", arXiv:2602.13010v2 (Abstract) | 해상풍력·벨기에·**5개 소스**(우리는 2개)·nMAE | 다중 NWP는 최대 레버지만 소스 수에 비례. 우리 상한은 LDAPS+GFS 2소스 → 17%보다 훨씬 작을 것 |
| C5 | GEFCom2014 wind 우승 해법: **zone별·분위수별 독립 GBM 분위수회귀**, 지배 입력신호(풍속)에 **스무딩** 적용, **cross-sectional 접근**, **상관 있는 인접 단지 정보를 쓰는 2-layer 모델링**. "최소한의 모델링 노력으로 유사한 day-ahead 풍력 과제에 재사용 가능"이라 저자들이 명시 | directly_supported | Landry, Erlinger, Patschke, Varrichio (2016) IJF 32(3):1061-1066, DOI 10.1016/j.ijforecast.2016.02.002 (초록 전문 확인) | day-ahead 풍력·10개 zone·**Pinball 지표** | 3그룹 구조에 직결: **그룹별 1층 모델 → 타그룹 1층 예측을 입력으로 받는 2층 모델**. 풍속 시계열 스무딩은 L3와 협의 |
| C6 | GEFCom2012/2014 리뷰: 대회는 방법뿐 아니라 **데이터 전처리·검증기법의 차이가 순위를 가른다**는 점을 강조; GEFCom2014는 4트랙 581명 참가 | near_match_only | Hong, Pinson, Fan, Zareipour, Troccoli, Hyndman (2016) IJF 32(3):896-913, DOI 10.1016/j.ijforecast.2016.02.001 / Hong, Pinson, Fan (2014) IJF, DOI 10.1016/j.ijforecast.2013.07.001 | 에너지 예측 대회 전반(로드/가격 포함) | 리뷰 수준 근거. **GEFCom2012 wind 트랙 개별 우승팀 기법은 본 조사에서 원문 확인 실패**(아래 C14 참조) |
| C7 | M5 대회: **GBDT(LightGBM)가 두 트랙 모두 상위 지배, 특히 딥러닝 기반 해법을 압도**. 원인 분석 논문 존재 | near_match_only | Januschowski, Wang, Torkkola, Erkkilä, Hasson, Gasthaus (2022) "Forecasting with trees", IJF 38(4):1473-1481, DOI 10.1016/j.ijforecast.2021.10.004; Makridakis et al. (2022) IJF DOI 10.1016/j.ijforecast.2021.11.013 | 소매판매 계층 시계열(풍력 아님) | 표형·외생변수 존재 시 GBDT 우선의 사전확률 강화 |
| C8 | 중간 규모 표형 데이터(~10k 샘플, 45개 데이터셋)에서 **트리 기반이 DL보다 우수**, HP 예산을 동등히 줘도 유지 | near_match_only | Grinsztajn, Oyallon, Varoquaux (2022) "Why do tree-based models still outperform deep learning on tabular data?", arXiv:2207.08815 (NeurIPS 2022 D&B) | 범용 표형 벤치마크(시계열/풍력 아님), 규모는 우리(2.6만행)와 근접 | 2.6만행·수십피처에서 DL 우선 탐색은 기대값 낮음 |
| C9 | day-ahead 풍력 **분포예측 방법 직접 비교**(벨기에 해상 9개 단지, 4년, 용량정규화 nCRPS 평균): SVGP 6.3% / **NGBoost(Gaussian) 6.0%** / **CQR 5.7%** / Treeffuser(무튜닝) 6.0% / **Treeffuser(튜닝) 5.6%**. 모든 트리 기반이 GP 베이스라인보다 확률 스킬 우수. NGBoost는 최하위권 — **정규분포 가정이 풍력 조건부분포에 부적합**하며 신뢰구간이 음수 출력을 포함 | directly_supported | Bruninx et al. (2026) arXiv:2602.13010v2, Table I/II/III + 본문 | day-ahead·풍력·NWP·트리기반·**nCRPS/nMAE 지표**, 해상(우리는 육상 복잡지형) | **NGBoost 배제**, LightGBM 분위수 + **CQR 캘리브레이션**을 기본선으로. 분포는 비모수로 |
| C10 | Treeffuser(GBT 기반 조건부 확산모델)가 최고: 확률 베이스라인 대비 **MAE -5%, CRPS -12%**. 단 **HP 튜닝 없으면 과적합으로 최하위권**. 실험 하드웨어: **Apple M3 8코어 / 24GB RAM** | near_match_only | 동 arXiv:2602.13010v2 (Abstract; §III-A "Apple machine with an 8-core M3 chip and 24 GB of RAM") | 해상풍력·4년·nMAE/nCRPS; HW는 M3/24GB(우리 M1/16GB) | 샘플 기반 분포 → 기대 FICR 최대화(L1)와 직결. **라이선스·공개일(2026-07-05 이전) 확인 필수**, 우선순위는 중 |
| C11 | 온라인(롤링) 후처리로 분포 이동 대응: 오프라인 학습 모델 대비 **MPL -9.86%, CRPS -8.31%**, 오프라인 후처리 대비 -3.52% | near_match_only | Pu et al. (2025) arXiv:2505.10367v2, Table 3 | **태양광** 설비증설로 인한 분포이동(풍력 아님), 온라인 갱신 가능한 대회 형식 | 우리 대회는 test 라벨 미공개·1회 제출 → **온라인 갱신 불가**. 대신 "최근 구간 가중/최근성 가중 학습"으로만 근사 |
| C12 | 참가팀 75%(상위10 중 9팀)가 **풍력·태양광을 따로 예측 후 결합**(추가 모델 또는 분위수 집계). 총합 직접 예측 팀은 사고(케이블 고장)에 적응이 더 어려웠음 | directly_supported | Browell et al. (2025) arXiv:2507.01579v2 §Forecasting track | 동일 대회 | **3그룹은 그룹별 개별 모델링**이 기본. 그룹합 직접 예측은 비권장 |
| C13 | 분위수 집계(aggregation) 정교화의 이득은 작음: quantile-by-quantile 합 대비 MPL **-0.15% / -0.73%**, 총합 직접 예측 대비 -1.46%; CRPS는 -2.2~2.3% | directly_supported | Pu et al. (2025) arXiv:2505.10367v2, Table 4 | 동일 대회, 풍력+태양광 결합 | 그룹 결합/집계 기법 최적화는 **ROI 낮음 → 후순위** |
| C14 | 추가 NWP(MEPS) 도입 시 우승팀 자체 보고 **Pinball -8%** (2023년을 검증연도로 사용). 그러나 상위5 중 2팀(UI BUD, GEB)은 **추가 기상데이터 없이** 상위권 | directly_supported | Browell et al. (2025) arXiv:2507.01579v2 | 동일 대회 | 외부 NWP는 강한 레버지만 **대회 규칙상 리스크**. 제공 데이터만으로도 상위권 가능함이 증명됨 |
| C15 | HEFTCom 벤치마크 모델이 케이블 고장(용량 급변)을 반영하지 못해 Pinball 53.58 vs 우승 22.18로 붕괴 | directly_supported | 동 S1, 최종 스코어보드 표 | 동일 대회 | **설비 가용용량/정지 레짐**이 모델 클래스보다 점수에 더 큰 영향. 예측 상한 클리핑·이상구간 처리 필수 |
| C16 | **DACON 제1·2회 풍력발전량 예측 대회 공개 우승 코드**: 본 조사 범위 내 확인 실패. GitHub 검색으로는 **현재 진행 중인 제3회(BARAM 2026) 참가자 저장소들만** 존재 (예: `Dacon-Organization/baram-2026-wind-power-forecasting`, `dh0728/decon-BARAM2026`, `Chankyu99/DACON_BARAM3`, `DelosIndustry/BARAM26-WindTurbine`, `SweetFriedPotato/BaramEuron`, `kohwoohyun/wind_power_forecast`, 모두 stars=0, 2026-07 업데이트) | insufficient | GitHub Search API (`/search/repositories?q=BARAM 풍력발전량 예측` 등, 2026-08-06 조회) | 대회 동일하나 **우승 해법 아님(진행 중 참가자 코드)** | 검증 불가한 경쟁자 코드에 의존 금지. 필요 시 별도 레인에서 라이선스·리크 검토 후 참고만 |
| C17 | Kaggle 예측 대회 6건 리뷰: 외생변수/계층정보가 있는 일·주 단위 시계열에서의 승리 패턴을 정리 (M 대회와 대비) | near_match_only | Bojer & Meldgaard (2021) IJF 37(2):587-603, DOI 10.1016/j.ijforecast.2020.07.007 (초록 확인, 본문 미열람) | 소매·수요 중심, 풍력 아님 | 일반 패턴(GBDT+피처공학+앙상블) 사전확률 강화. **구체 개선폭 미확보** |
| C18 | 부하예측 Kaggle 우승급 해법이 gradient boosting 기반이었음(초기 사례) | near_match_only | Ben Taieb & Hyndman (2014) "A gradient boosting approach to the Kaggle load forecasting competition", IJF 30(2), DOI 10.1016/j.ijforecast.2013.07.005 | 전력수요(풍력 아님), 2012년 | 역사적 근거일 뿐 |
| C19 | 그룹 시계열에서 **global(pooled) 모델**은 계열 간 유사성 가정 없이도 local 모델과 동등하거나 우수할 수 있음(복잡도 상향 여지 때문) | near_match_only | Montero-Manso & Hyndman (2021) "Principles and Algorithms for Forecasting Groups of Time Series: Locality and Globality", arXiv:2008.00444 = IJF 37(4) | 수천 개 계열 그룹(우리는 3그룹) | 3그룹 pooled+group-id 학습은 **시도 가치는 있으나 계열 수가 적어 기대이득 작음** |
| C20 | 시계열 CV 논쟁: (a) 순수 자기회귀·잔차 무상관이면 표준 K-fold CV가 유효하고 소표본에서 OOS 평가보다 모델선택이 정확 (Bergmeir 2018). (b) 실제(비정상) 시계열에서는 holdout/prequential 계열이 오차추정에 더 정확했다 (Cerqueira 2020) | near_match_only | Bergmeir, Hyndman, Koo (2018) CSDA 120:70-83, DOI 10.1016/j.csda.2017.11.003 / Cerqueira, Torgo, Mozetič (2020) Machine Learning, DOI 10.1007/s10994-020-05910-7 | 범용 시계열(풍력 day-ahead 아님) | 우리 설정(외생 NWP 지배·연변동 큼·test=1년) → **연단위 홀드아웃 + 월블록 CV 병행**이 안전. 단일 랜덤 K-fold 금지 |
| C21 | 실전 대회 상위팀의 실제 검증 프로토콜: SVK는 **2023년 한 해를 검증연도**로 삼아 NWP 추가 여부를 결정; GEB는 LightGBM을 **Optuna + 3-fold CV**로 튜닝 후 전체 데이터로 재학습 | directly_supported | Browell et al. (2025) arXiv:2507.01579v2; Pu et al. (2025) arXiv:2505.10367v2 §Table 1 | 동일 대회·동일 리드타임 | **연단위 홀드아웃으로 "구조적 결정"을, 3-fold CV로 "HP"를** 나눠 결정하는 2단 프로토콜 채택 |
| C22 | SCADA 등 **학습 시에만 존재하는 특권정보(LUPI)**의 일반 이론·증류 통합 프레임워크는 존재 | near_match_only | Lopez-Paz, Bottou, Schölkopf, Vapnik (2016) "Unifying distillation and privileged information", arXiv:1511.03643 (ICLR 2016) | 일반 지도학습(풍력/시계열 아님) | 방향 힌트로만. 풍력 발전량 예측에서 SCADA-특권정보 효과의 **정량 증거는 미확보(C24)** |
| C23 | 상관 있는 인접 단지 데이터를 **인스턴스 기반 전이학습 + GBDT 분위수회귀**로 활용한 확률적 풍력예측 사례 존재 | near_match_only | Zhang et al. (2019) "Probabilistic Wind Power Forecasting Approach via Instance-Based Transfer Learning Embedded GBDT", Energies 12(1):159, DOI 10.3390/en12010159 | 풍력·확률예측이나 초록만 확인, 개선폭 미확인 | group_3의 2022년 라벨 결측 보완(그룹1·2 데이터 전이)의 **방향성 근거** |
| C24 | **SCADA(train-only) → 보조타깃 멀티태스크/LUPI가 day-ahead 풍력발전량 예측을 개선한다는 정량 증거**: 본 조사 범위 내 미발견 | insufficient | (OpenAlex/arXiv 질의 4회 무수확) | — | 자체 ablation으로 검증할 것. 사전 기대치 낮게 설정 |
| C25 | 지역 단위 day-ahead 확률적 풍력예측에 **conformalized regression forest** 사용 사례(딥 CNN 결합) | near_match_only | Applied Energy (2024) DOI 10.1016/j.apenergy.2024.122900 "A novel day-ahead regional and probabilistic wind power forecasting framework using deep CNNs and conformalized regression forests" | day-ahead·지역단위·확률, 초록만 확인 | conformal 계열이 실무에서 채택되고 있다는 방증(개선폭 미확인) |

**Source table (21)**: S1 arXiv:2507.01579v2 · S2 arXiv:2505.10367v2 · S3 DOI 10.1016/j.ijforecast.2016.02.002 · S4 DOI 10.1016/j.ijforecast.2016.02.001 · S5 arXiv:2602.13010v2 · S6 DOI 10.1016/j.ijforecast.2013.07.001 · S7 DOI 10.1016/j.ijforecast.2021.10.004 · S8 DOI 10.1016/j.ijforecast.2021.11.013 · S9 arXiv:2207.08815 · S10 DOI 10.1016/j.ijforecast.2020.07.007 · S11 DOI 10.1016/j.ijforecast.2013.07.005 · S12 DOI 10.1016/j.csda.2017.11.003 · S13 DOI 10.1007/s10994-020-05910-7 · S14 arXiv:2008.00444 · S15 arXiv:1511.03643 · S16 DOI 10.3390/en12010159 · S17 DOI 10.1016/j.apenergy.2024.122900 · S18 arXiv:1910.03225 (NGBoost 원논문) · S19 DOI 10.1175/MWR-D-15-0260.1 (QRF 캘리브레이션, L3 경계) · S20 CRAN quantregForest DOI 10.32614/cran.package.quantregforest (Meinshausen QRF 구현) · S21 GitHub Search API 조회결과(BARAM 2026 참가자 저장소 목록)

---

## 3. 적용 후보 ExperimentSpec 목록

> 표기: spec_id / kind / 설명 / 기대효과 / 난이도 / M1-16GB / 규칙 / origin / 우선순위

### 우선순위 1

- **L2-S01** / `ensemble` / **Sister-model 스태킹**: 동일 구조 LightGBM(또는 CatBoost)을 ①LDAPS 피처만 ②GFS 피처만으로 각각 학습 → 각 모델의 예측(또는 다분위 예측 벡터)을 입력으로 하는 **선형 메타모델**(그룹별)로 결합. 단일 모델에 두 NWP를 합치는 방식과 **직접 비교** 필수.
  기대효과: point MAE/pinball 기준 **-5~6%** (C3의 -6.3/-6.4% 이식, 우리 지표 환산은 미검증) → 1-NMAE +0.003~0.006 추정, **FICR은 오차분산 축소로 간접 상승** / 난이도 **M** / M1: 모델 수 2×3그룹=6개, 각 2.6만행 → 수 분 / 규칙 **pass** / origin C3+C1 / **P1**
- **L2-S02** / `model` / **다분위(MultiQuantile) GBDT를 기본 예측기로 채택**: CatBoost `MultiQuantile` 또는 LightGBM `objective=quantile`을 9~19개 분위수로 학습해 **조건부 분포**를 산출. 점예측은 이 분포에서 L1의 기대 FICR 최대화 규칙으로 선택(결정층은 L1 소유).
  기대효과: 계단형 FICR 최적화의 **전제조건**. 분포 없이는 err≤6% 구간 집중 전략이 불가 / 난이도 **M** / M1: 분위수 수 × 그룹 수만큼 학습(LightGBM 분위수는 분위수당 별도 학습 → 19×3=57모델, 각 <1분 예상) / 규칙 **pass** / origin C1(SVK), C5(Landry), C9 / **P1**
- **L2-S03** / `analysis` / **검증 프로토콜 2단 분리**: (a) *구조적 결정*(피처세트·스태킹 유무·NWP 조합)은 **연단위 홀드아웃**(train 2022–2023 → valid 2024, group_3는 2023만) 로 결정, (b) *하이퍼파라미터*는 **3-fold(월 블록, embargo 24h) CV**로 결정 후 전체 데이터 재학습. 두 단계 모두 **NMAE와 FICR을 동시 리포트**하고, 대회 공식 유효구간(actual ≥ 0.10·capacity)만으로 집계.
  기대효과: 지표 오추정으로 인한 잘못된 채택 방지(직접 점수 향상은 아님, **모든 다른 스펙의 신뢰도 전제**) / 난이도 **S** / M1 문제없음 / 규칙 **pass** / origin C21+C20 / **P1**

### 우선순위 2

- **L2-S04** / `model` / **Landry식 2-layer cross-sectional 모델링**: 1층은 그룹별 독립 모델, 2층은 **타 그룹의 1층 예측 + 격자 간 공간 피처**를 입력으로 받는 잔차/보정 모델. 1층 예측은 반드시 **out-of-fold**로 생성(리크 방지).
  기대효과: GEFCom2014 우승 해법의 핵심 구성요소(개별 기여도 수치는 원논문에 미제시 → 개선폭 `insufficient`) / 난이도 **M** / M1 OK / 규칙 **pass** / origin C5 / **P2**
- **L2-S05** / `model` / **CQR(Conformalized Quantile Regression) 캘리브레이션 층**: 분위수 GBDT 위에 최근 블록(예: 2024 하반기) 기반 conformal 보정을 적용해 분위수 신뢰도를 교정.
  기대효과: 동일 비교실험에서 CQR nCRPS 5.7% vs NGBoost 6.0% vs SVGP 6.3% (**-5~10%**) / 난이도 **S** (수십 줄, 외부 의존성 불필요) / M1 OK / 규칙 **pass** / origin C9 / **P2**
- **L2-S06** / `decision`(모델측) / **가용용량/이상 레짐 가드**: 예측을 [0, capacity]로 클리핑, 학습 데이터에서 정지·출력제한(curtailment) 의심 구간을 탐지해 (a) 제외 또는 (b) 지시변수화. test에는 정지정보가 없으므로 **보수적 클리핑만** 적용.
  기대효과: HEFTCom 벤치마크는 용량 급변 미반영으로 22.18 → 53.58로 붕괴(**2.4배 악화**). 우리 쪽 상방은 알 수 없으나 **하방 리스크 제거 효과가 큼** / 난이도 **S** / 규칙 **pass** / origin C15 / **P2**
- **L2-S07** / `model` / **group_3 2022 라벨 결측 전략**: ①group_3는 2023–2024만으로 학습한 베이스 ②그룹1·2 데이터(2022–2024)로 학습한 pooled 모델(그룹 원-핫/용량 정규화 타깃)을 group_3에 전이 ③둘의 블렌드 — 를 L2-S03 프로토콜로 비교.
  기대효과: pooled/전이의 방향성 근거는 C19·C23(개선폭 미확인) → **`near_match_only` 수준 기대** / 난이도 **M** / M1 OK / 규칙 **pass** / origin C19+C23 / **P2**
- **L2-S08** / `analysis` / **HP 튜닝 예산 최소화 규율**: HEFTCom 우승팀은 **부스팅 반복수만** 튜닝하고 나머지는 기본값 사용. Optuna는 GEB 수준(핵심 7개 파라미터, 3-fold)으로 제한하고 남는 시간은 피처·스태킹에 배분.
  기대효과: 과적합·시간낭비 방지(점수 직접효과 미측정) / 난이도 **S** / 규칙 **pass** / origin C1+C21 / **P2**

### 우선순위 3

- **L2-S09** / `model` / **Treeffuser(GBT 기반 조건부 확산모델)로 조건부 분포 샘플링** → 샘플에서 기대 FICR 최대화 점 선택. 반드시 HP 튜닝(무튜닝은 과적합으로 최하위권).
  기대효과: 동 비교에서 CQR 대비 nCRPS 5.6% vs 5.7%(**약 -2%**), 확률 베이스라인 대비 MAE -5%/CRPS -12% / 난이도 **M~L** / M1: 원 실험이 **M3 8코어/24GB**에서 수행됨 → M1/16GB에서 가능성 있으나 메모리·시간 여유 확인 필요 / 규칙 **risk**(OSS 라이선스 및 2026-07-05 이전 공개 여부 **확인 필요**, 사전학습 가중치가 아닌 로컬 학습이면 문제 없음) / origin C9+C10 / **P3**
- **L2-S10** / `feature` / **SCADA를 2단계 특권정보로 활용**: 1단계 NWP→SCADA(터빈 평균 풍속/발전량) 회귀 모델을 train 구간으로 학습 → 2단계에서 그 **예측된 SCADA 값**을 피처로 사용(test에도 생성 가능). 추가로 터빈합 ≒ 그룹합 물리 제약을 보조 타깃으로 사용.
  기대효과: **`insufficient`** — in-scope 정량 증거 없음(C24). 자체 ablation 필수, 기대치 낮게 / 난이도 **M** / M1 OK / 규칙 **pass**(train 기간 데이터만 사용, 예보시점 이후 정보 없음) / origin C22+C23 / **P3**
- **L2-S11** / `ensemble` / **시드·서브샘플 다양성 평균화**: 동일 구성 GBDT를 시드 5~10개로 학습 후 평균 → 예측 분산 축소로 err≤6% 밴드 적중률 상승 기대.
  기대효과: **`insufficient`**(FICR류 계단형 지표에서의 정량 증거 없음). 다만 비용이 매우 낮음 / 난이도 **S** / M1 OK / 규칙 **pass** / origin 일반 앙상블 원리 / **P3**
- **L2-S12** / `model` / **최근성 가중 학습**(시간 감쇠 sample_weight)으로 설비/운영 레짐 변화에 대응. 온라인 갱신이 불가능한 우리 대회에서 C11의 대체안.
  기대효과: `near_match_only`(원 근거는 태양광 온라인 후처리 -9.86% MPL, 조건 상이) / 난이도 **S** / 규칙 **pass** / origin C11 / **P3**

### 우선순위 4~5 (탐색만)

- **L2-S13** / `model` / PatchTST/TFT/N-BEATS 등 시계열 DL 도입 — **비권장**. in-scope 승리 증거 없음(C2·C7·C8). 시도한다면 앙상블 다양성 목적의 소규모 실험으로 한정 / 난이도 **L** / M1: MPS 학습 가능하나 시간대비 효과 낮음 / **P5**
- **L2-S14** / `model` / NGBoost / GAMLSS류 모수적 분포회귀 — **비권장**(C9: 최하위권, 음수 구간 생성) / **P5**

---

## 4. 하지 말아야 할 것 (증거 기반 negative findings)

1. **NGBoost(정규분포) 및 정규분포 가정 모수적 분포회귀를 주력 분포모델로 쓰지 말 것.** day-ahead 풍력 직접 비교에서 nCRPS 6.0%로 CQR(5.7%)·Treeffuser(5.6%)에 밀렸고, **대칭 정규분포 때문에 음수 발전량 구간을 산출**한다 `directly_supported` (S5).
2. **Treeffuser/확산모델을 기본 하이퍼파라미터로 쓰지 말 것.** 무튜닝 버전은 train 오차 1.8% vs test 6.0%로 명백한 과적합, 최하위권 `directly_supported` (S5).
3. **3그룹 합계를 직접 예측하지 말 것.** 상위10 중 9팀이 구성요소별 예측 후 결합을 택했고, 총합 직접 예측은 CRPS는 좋아도 **pinball은 더 나빴다**(34.86 vs 34.35) `directly_supported` (S1, S2).
4. **분위수 집계 기법 정교화에 시간을 크게 쓰지 말 것.** 개선폭 0.15~0.73% MPL 수준 `directly_supported` (S2).
5. **DL 아키텍처 서베이/최신성 추종 금지.** 상위권 DL 사례는 **사내 AI 기상모델 + 관측·레이더·위성 데이터**가 본질이었고, 우리 규칙(원격 API 금지, 외부데이터 제약)·M1 환경에서 재현 불가 `directly_supported`(사례 사실) / 표형 소규모에서 DL 우위 증거는 `insufficient`.
6. **단순 랜덤 K-fold CV 금지.** 연변동·NWP 외생성이 지배하는 설정에서 오차 과소추정 위험. 연단위 홀드아웃 + 월블록/embargo 병행 `near_match_only` (S12, S13).
7. **경쟁자 GitHub 저장소를 성능 근거로 삼지 말 것.** 현재 검색되는 BARAM 2026 저장소는 모두 **진행 중 참가자 코드(stars 0, 검증되지 않음)**이며 우승 해법이 아님 `insufficient` (S21).
8. **추가 외부 NWP 도입을 "확실한 8% 개선"으로 계획하지 말 것.** 그 수치는 특정 팀의 자체 보고이며, 같은 대회 상위5 중 2팀은 추가 기상데이터 없이 상위권이었다. 우리 대회에서는 규칙 리스크도 있음 `directly_supported`(사실) (S1).

---

## 5. Open questions / insufficient 항목

- **Q2-보완**: GEFCom2012 **wind 트랙 개별 우승팀**의 기법(피처/모델/앙상블). 리뷰 논문(S6, DOI 10.1016/j.ijforecast.2013.07.001) 본문 PDF 미열람 → `insufficient`. 재조사 시 pierrepinson.com/31761/Literature/hong2014.pdf 및 IJF 2014 30(2) 특집호 개별 논문 목록부터.
- **Q3**: DACON 제1·2회 풍력발전량 예측 대회 공개 우승 코드 → 한국어 웹검색 불가로 `insufficient`. **Serper API 키를 설정하면 재조사 가치 높음**(가장 scope가 가까운 증거원).
- **Q6**: SCADA(train-only) 특권정보/멀티태스크의 **정량 효과** → `insufficient`. 우리 자체 ablation이 유일한 근거가 될 것.
- **Q5-보완**: 그룹 간 상관을 활용한 2-layer 구조의 **개별 기여도(%)** → Landry 2016 초록에 수치 없음 → `insufficient`.
- **계단형 유틸리티(FICR류) 지표에서 어떤 *모델측* 특성이 유리한가**(분산축소 vs 편향보정) → in-scope 문헌 미발견 `insufficient`. 이는 L1(결정이론)과의 접점이며, L2로서는 **분위수/샘플 기반 분포를 제공**하는 것까지가 역할.
- **라이선스 확인 필요**: Treeffuser 및 CatBoost MultiQuantile 사용 시 라이선스·공개일(2026-07-05 이전) 확인 — 본 레인에서는 미검증.
- **미확인 locator**: HEFTCom2024 우승팀 SVK의 상세 해법 논문은 S1에서 `Olauson2025`로 인용되나 본 조사에서 서지정보 확정 실패 → 재조사 대상.
