# S6 외부 문헌 조사 레인 C — 딥러닝 시계열 표현 vs GBDT 테이블 표현, 벤치마크 판정

- 레인: `S6_ext_C_repr` (읽기전용 외부 문헌 조사 전용)
- 수행일: 2026-08-08
- 조회 수단: `websearch` (Serper/Google) **단독**. PDF 전문 다운로드·저장소 쓰기·모델 학습 없음.
- 쿼리 수: **90건** (전체 로그: `research/lanes/S6_ext_C_repr.searchlog.json`)
- 증거 등급: `[스니펫확인]` = 검색 스니펫에 수치/문장이 직접 노출됨. `[전문미확인]` = 전문 본문을 열지 못해
  수치의 표/조건을 검증하지 못함. **본 문서의 모든 수치는 최소 `[전문미확인]`을 기본으로 하고,
  스니펫에 문자 그대로 나온 것만 `[스니펫확인]`으로 승격한다.**

---

## §C0 우리 세팅의 재기술 (판정의 기준선)

판정에 쓸 우리 세팅의 5개 축을 문헌 축과 정렬해 둔다.

| 축 | 우리 값 | 문헌에서 대응하는 축 |
|---|---|---|
| 표본 수 (행) | g1·g2 각 ~17,000 / g3 ~8,700 / 합 ~44,000 | tabular 벤치마크의 "medium(10k)~large(50k)" 구간 |
| **독립 표본 수 (발행)** | **1일 1발행 × (708+708+362) ≈ 1,780** | tabular 벤치마크의 "small(<3,000)" 구간 |
| 피처 수 | 830 (통계) / 914 (격자) / 304 (geometric) | TabPFN v2 한계 500, TabPFN-2.5 한계 2,000 |
| 자기회귀 | **불가** (2025 관측/SCADA 없음) | LTSF/M4/Monash/KDDCup2022 대부분이 **가능** 전제 |
| 손실 | 계단형 (0.5·(1−NMAE)+0.5·FICR, 6%/8% 밴드) | 표준 벤치마크는 전부 MSE/MAE/MASE/RMSSE |

**독립 표본 수 1,780이 이 레인의 핵심 재프레이밍이다.** 44,000행은 겉보기 행 수이고,
같은 날 24행은 하나의 NWP 발행에서 파생되므로 오차가 강하게 자기상관한다.
GBDT-vs-DL 문헌의 "표본 수"는 i.i.d. 행 수를 뜻하므로, 우리를 44k로 놓고 문헌을 읽으면
체계적으로 DL 쪽에 유리하게 오판독하게 된다. 아래 판정은 44k와 1.78k 양쪽 모두에서 수행한다.

---

## §C1 벤치마크 증거표

### C1-a. 시계열 대회 / 시계열 아키텍처 벤치마크

| # | 논문/대회 | 데이터 규모 | 자기회귀(lag target) | DL vs GBDT 승패 | 수치 | URL | 등급 |
|---|---|---|---|---|---|---|---|
| 1 | **M5 Accuracy** (Makridakis et al., IJF 2022) | 42,840 계열 / 3,049 SKU × 1,941일 (≈5,800만 행) | **사용** (판매 lag가 핵심 피처) | **GBDT 압승** | "상위 4개 방법과 상위 50개 제출의 절대다수"가 LightGBM 계열. 상위 50개 전부가 최우수 벤치마크 대비 **>14%** 개선. 반면 전체 제출의 **약 92.5%는 최우수 벤치마크조차 못 이김**. 우승(In & Jung)은 **direct + recursive LightGBM 다중 모형** | https://www.sciencedirect.com/science/article/pii/S0169207021001874 · https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf · https://pmc.ncbi.nlm.nih.gov/articles/PMC9232271/ | `[스니펫확인]` (14%, 92.5%, "top four … LightGBM", "multiple direct and recursive LightGBM") |
| 2 | **Elsayed et al. 2021**, *Do We Really Need Deep Learning Models for Time Series Forecasting?* (arXiv 2101.02118) | 다변량 TSF 표준셋 (수천~수만 스텝) | **사용** (window-based 입력 = target 자기 lag 윈도) | **GBDT 승** (단, TFT만 예외) | "window-based input transformation이 단순 GBRT를 **모든** SOTA DL 모델을 능가하는 수준으로 끌어올린다". **TFT만이 GBRT를 일관되게 이긴 유일한 DNN**이며, DeepAR·DeepState는 GBRT에 짐 | https://arxiv.org/abs/2101.02118 · https://ai-scholar.tech/en/articles/time-series/need_DL_for_TSF | `[스니펫확인]` (문장) / 표 수치는 `[전문미확인]` |
| 3 | **Zeng et al., AAAI 2023 (DLinear/LTSF-Linear)** | ETT/Electricity/Traffic/Weather/Exchange (1.7만~1,750만 포인트) | **사용** (전적으로 target 히스토리의 선형사상) | **DL(Transformer) 패배**, 승자는 *선형모델* | "LTSF-Linear가 기존 복잡한 Transformer 기반 모델을 **모든 경우에** 능가하고 종종 큰 폭(**20%~50%**)으로". 공식 repo: Exchange rate에서 FEDformer 대비 **>40%**, Traffic **~30%** | https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089 · https://github.com/vivva/DLinear | `[스니펫확인]` (20~50%, >40%, ~30%) |
| 4 | **Monash TS Forecasting Archive** (Godahewa et al., NeurIPS D&B 2021) | 20+ 데이터셋, 13 베이스라인 | **사용** (전부 단변량 자기회귀) | **혼전 / GBDT도 최하위권** | "전반적으로 **CatBoost와 FFNN이 최악의 성능**, PR(pooled regression)은 혼재. DeepAR·N-BEATS 등 DL은 [일부 셋에서 우수]". "짧고·무관하고·noisy한 계열에는 단순 단변량 모델이 더 낫다" | http://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/eddea82ad2755b24c4e168c5fc2ebd40-Paper-round2.pdf · https://forecastingdata.org/ | `[스니펫확인]` (인용문) / 표2 MASE 수치는 `[전문미확인]` |
| 5 | **NBEATSx** (Olivares et al., IJF 2023) | 전력가격(EPF) 5개 시장, 6년치 시간단위 | **사용** (lagged price 필수) | **DL 승** (exogenous 결합 시) | NBEATS 대비 **~20%**, LEAR/특화 DNN 대비 **최대 5%** 개선 | https://www.sciencedirect.com/science/article/pii/S0169207022000413 | `[스니펫확인]` (20%, 5%) |
| 6 | **TSMixer** (Chen et al., TMLR 2023) | M5 (대규모), LTSF 셋 | **사용** | DL 승(단, all-MLP) | "대규모 M5 벤치마크에서 SOTA 대비 우월". TSMixer-Ext가 covariate 결합판 | https://arxiv.org/pdf/2303.06053 | `[스니펫확인]` (문장) / 수치 `[전문미확인]` |
| 7 | **TiDE** (Das et al., TMLR 2023) | LTSF 셋 | **사용** | DL(선형 인코더) 승, 속도 5~10배 | "TiDE, PatchTST, N-HiTS, DLinear가 나머지 베이스라인보다 훨씬 낫다"; PatchTST 대비 추론 5배·학습 10배 이상 빠름 | https://arxiv.org/html/2304.08424v5 | `[스니펫확인]` |
| 8 | **Chronos-2** (Amazon, 2025-10-20) | 사전학습 foundation model | **필수** (past covariates + target history) | — | "univariate/multivariate/**covariate-informed** zero-shot 지원". **공개일 2025-10-20** | https://github.com/amazon-science/chronos-forecasting · https://arxiv.org/html/2510.15821v1 | `[스니펫확인]` (날짜) |

**C1-a 요약 판정:** 8건 중 **7건이 target lag를 쓴다**. 우리 세팅(§C0)에서 lag는 존재하지 않는다.
따라서 이 표의 "DL 승" 항목(5·6·7)은 **우리에게 이전 불가**이고, "GBDT 승" 항목(1·2)도
lag를 전제로 한 승리라는 점에서 대칭적으로 이전 불가이다.
**M4/M5/LTSF/Monash 계열 벤치마크는 통째로 우리 문제의 증거가 아니다.**
유일하게 구조적으로 이전 가능한 것은 3번의 메타 교훈이다:
*시계열 DL의 승리 대부분이 정교한 표현이 아니라 단순 선형/MLP 사상으로 재현되었다.*

### C1-b. Tabular 벤치마크 (우리 문제가 실제로 속한 부류: exogenous-only 회귀)

| # | 논문 | 데이터 규모 | 결론 | 수치 | URL | 등급 |
|---|---|---|---|---|---|---|
| 9 | **Shwartz-Ziv & Armon 2022**, *Tabular Data: DL is Not All You Need* (Inf. Fusion) | 11 데이터셋 | **GBDT 승** | "XGBoost가 이들 deep model을 **대부분의 경우** 능가"; DL 논문들이 자기 논문 데이터셋 밖에서 재현되지 않음 | https://arxiv.org/abs/2106.03253 | `[스니펫확인]` |
| 10 | **Grinsztajn et al., NeurIPS 2022 D&B** | 45 데이터셋 | **GBDT 승** | "tree-based가 **중형 데이터(~10K 표본)**에서 여전히 SOTA — 속도 우위를 계산에 넣지 않더라도". **대형 체제는 50,000 표본으로 절단해 별도 평가**했고, 격차가 좁아지지만 역전되지는 않음 | https://arxiv.org/abs/2207.08815 · https://ar5iv.labs.arxiv.org/html/2207.08815 | `[스니펫확인]` (10K, 50,000 절단) |
| 11 | **McElfresh et al., NeurIPS 2023**, *When Do Neural Nets Outperform Boosted Trees?* | **19 알고리즘 × 176 데이터셋 (역대 최대)** | **"논쟁 자체가 과대평가"** | (a) "**GBDT의 가벼운 하이퍼파라미터 튜닝이 최적 알고리즘 선택보다 성능을 더 올린다**". (b) "GBDT는 **왜곡·두꺼운 꼬리 분포 피처와 데이터셋 불규칙성**을 NN보다 훨씬 잘 다룬다". (c) "**표본 3,000 이하**에서는 TabPFN이 평균적으로 다른 모든 알고리즘을 능가" | https://arxiv.org/abs/2305.02997 · https://arxiv.org/html/2305.02997v4 · https://www.semanticscholar.org/paper/5e4125b3a2ec91e866d970498f8a138c5a5cc89b | `[스니펫확인]` (3000, 인용문) |
| 12 | **TabReD** (Rubachev et al., **ICLR 2025 Spotlight**) | **8개 산업급 데이터셋, 시간분할(temporal split), 피처 다수** | **GBDT 승 — 우리 세팅과 가장 유사한 조건** | "**TabReD 데이터셋에서 GBDT가 최고 성능**". 학계 tabular DL 벤치마크가 (i) 시간이동(temporal shift)과 (ii) 광범위 피처엔지니어링이라는 산업 현실을 빠뜨렸고, 그 두 조건을 넣으면 DL 우위가 사라짐 | https://arxiv.org/html/2406.19380v4 · https://github.com/yandex-research/tabred | `[스니펫확인]` ("GBDT show the best results on the TabReD datasets") |
| 13 | **TabArena v0.1** (Erickson et al., **NeurIPS 2025 D&B**) | 51 태스크(회귀 13 + 분류 38), 1,053개 후보에서 선별 | **혼전 — 단, 이득의 출처는 앙상블** | "**통상 튜닝 체제에서는 CatBoost가 1위**". "그러나 **post-hoc 앙상블 후에는 신경망이 평균적으로 최강 단일 모델**". 앙상블 후 순위 예: 2위 LightGBM(tree), 3위 RealMLP(NN). "GBDT는 여전히 강력한 경쟁자이나 **더 큰 데이터에서 DL이 따라잡았다**" | https://arxiv.org/html/2506.16791v1 · https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf · https://github.com/zhaoyang97/Paper-Notes-en/blob/main/docs/NeurIPS2025/self_supervised/tabarena_a_living_benchmark_for_machine_learning_on_tabular_data.md | `[스니펫확인]` (CatBoost 1위, 앙상블 후 NN, LightGBM 2위/RealMLP 3위) |
| 14 | **RealMLP / "Better by Default"** (Holzmüller, Grinsztajn, Steinwart, NeurIPS 2024) | 118+ 데이터셋 메타학습 기본값 | **동률급** | 사전튜닝된 MLP 기본값이 부스팅 트리와 **경쟁 가능** 수준까지 상승 | https://arxiv.org/abs/2407.04491 · https://hal.science/hal-04641923v2/file/paper_tabular_neurips_camera-ready.pdf | `[스니펫확인]` (문장) / 수치 `[전문미확인]` |
| 15 | **TabM** (Gorishniy et al., **ICLR 2025**) | tabular 표준셋 | **동률급** | "TabM은 **GBDT와 손쉽게 경쟁**하며 기존 tabular DL을 능가. attention/retrieval 기반의 복잡도는 값을 못 한다" | https://arxiv.org/abs/2410.24210 · https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf | `[스니펫확인]` |
| 16 | **CARTE** (Kim et al., ICML 2024) | 소규모 테이블, 컬럼명 텍스트 활용 | DL 승 (특정 조건) | 컬럼명이 **불일치하는 여러 테이블 간 공동학습**으로 작은 테이블을 큰 테이블로 보강 | https://arxiv.org/html/2402.16785v2 | `[스니펫확인]` (문장) |
| 17 | **AutoGluon-Tabular 1.0 / 1.5** | AutoML 벤치마크 / TabArena 51셋 | 앙상블·스태킹이 이득의 실체 | 1.0: 타 AutoML 대비 **82~94% 승률**, 63% 태스크 1위, **0.8 대비 평균 손실 7.4% 개선**. 1.5: **v1.4 Extreme 대비 85% 승률** | https://auto.gluon.ai/dev/whats_new/v1.0.0.html · https://auto.gluon.ai/stable/whats_new/v1.5.0.html | `[스니펫확인]` |

### C1-c. TabPFN 계열 (질문 4 전용)

| # | 항목 | 사실 | 수치/조건 | URL | 등급 |
|---|---|---|---|---|---|
| 18 | **TabPFN v2** (Hollmann et al., **Nature, 2025-01**) | 공개일 **2025년 1월** — 대회 기준일 2026-07-05 **이전, 통과** | "**최대 10,000 표본, 500 피처**의 데이터셋에서 이전 모든 방법을 능가". 분류 **2.8초**·회귀 **4.8초** 기본설정으로 4시간 튜닝 베이스라인 전부를 이김 → **AutoGluon 대비 5,140배 스피드업**. **논문이 직접 경고: "이 결과가 10,000 표본·500 피처를 넘어 잘 확장된다는 증거로 받아들여선 안 된다"** | https://www.nature.com/articles/s41586-024-08328-6 · https://pubmed.ncbi.nlm.nih.gov/39780007/ · https://huggingface.co/Prior-Labs/TabPFN-v2-clf | `[스니펫확인]` (10,000/500, 2.8s/4.8s, 5,140×, 경고문) |
| 19 | **TabPFN v2 라이선스** | **Prior Labs License = Apache 2.0 + 귀속 조항, 상업 이용 허용** | 공식 문서: "The model is licensed under Prior Labs License, **open source, commercial use with attribution**" | https://docs.priorlabs.ai/models · https://github.com/PriorLabs/TabPFN/blob/main/LICENSE | `[스니펫확인]` |
| 20 | **TabPFN-2.5** (2025-11) | **규모 문제는 풀지만 라이선스가 대회 규칙을 위반** | "최대 **50,000 데이터포인트·2,000 피처**", "튜닝된 트리 기반 모델을 실질적으로 능가". 그러나 **가중치는 비상업 라이선스**: "the model, its derivatives, and **its outputs cannot be used for any commercial or production purpose**" | https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report · https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/LICENSE · https://github.com/PriorLabs/tabpfn | `[스니펫확인]` (50k/2k, 비상업 문구) |
| 21 | **TabPFN-TS** (Hoo et al., arXiv 2501.02945, 2025-01) | TabPFN-v2를 시계열로 확장 | "**covariate-informed forecasting에서 SOTA**, 단변량에서 경쟁력". 캘린더/특징 기반 회귀로 환원해 GIFT-Eval 상위권 | https://arxiv.org/abs/2501.02945 · https://github.com/PriorLabs/tabpfn-time-series | `[스니펫확인]` |
| 22 | **TabPFN 10k 초과 확장 연구** | 우회로가 존재하나 전부 근사·비공식 | Chunked TabPFN: "10K 사전학습 한계를 넘어서도 in-context 표본 추가로 계속 이득". LocalPFN(NeurIPS 2024): 최근접이웃 컨텍스트. TabPFN Unleashed | https://arxiv.org/html/2509.00326v1 · https://neurips.cc/virtual/2024/poster/96776 · https://openreview.net/forum?id | `[전문미확인]` |

### C1-d. 재생에너지 / 풍력 특화 — **자기회귀 여부로 분리**

| # | 논문/대회 | NWP-only? | 개선치 | 우리에게 이전 가능? | URL | 등급 |
|---|---|---|---|---|---|---|
| 23 | **GEFCom2014 wind track 우승** (Landry et al., IJF 2016) | **거의 NWP-only** (10m/100m U·V 성분) | 확률예측 **GBM(gradient boosting machine)** 으로 우승. pinball loss가 zone·월별로 변동 | **가능** — 그리고 **GBDT가 이긴 사례** | https://www.sciencedirect.com/science/article/abs/pii/S0169207016000145 · https://ideas.repec.org/a/eee/intfor/v32y2016i3p1061-1066.html | `[스니펫확인]` (우승/GBM) |
| 24 | **Markovics & Mayer 2022** (Renew. Sustain. Energy Rev., 인용 562) — **우리와 구조가 가장 같은 논문** | ✅ **NWP 기반 결정론적 day-ahead, exogenous-only** | **24개 ML 방법 비교. "가장 정확한 두 모델은 kernel ridge regression과 multilayer perceptron"**. 선형회귀 베이스라인 대비 **RMSE 13.9% 감소**, 최대 **44.6% forecast skill score** | **가능. 그리고 여기서는 얕은 NN이 트리 앙상블을 이겼다** (태양광) | https://www.sciencedirect.com/science/article/pii/S136403212200274X · https://ideas.repec.org/a/eee/rensus/v161y2022ics136403212200274x.html | `[스니펫확인]` (13.9%, 44.6%, KRR·MLP 1·2위) |
| 25 | **Couto & Estanqueiro 2022** (Renewable Energy) | ✅ NWP(WRF) 기반 피처 | **WRF 기반 신규 피처로 RMSE 13%~37% 감소** (베이스라인 대비) | **가능 — 그리고 이득의 출처가 아키텍처가 아니라 NWP 피처공학** | https://ui.adsabs.harvard.edu/abs/2022REne..201.1076C/abstract · https://traderes.eu/wp-content/uploads/2023/01/TradeRES-Research-Bulletin-Enhancing-wind-power-forecast-accuracy-using-the-weather.pdf | `[스니펫확인]` (13~37%) |
| 26 | **Kibet et al. 2025**, *Minimalist Deep Learning for Solar Power Forecasting* (Energies) | ✅ NWP 베이스라인 대비 | "**선형 모델이 두 NWP 베이스라인 대비 RMSE를 최소 3.7% 개선**" — 즉 "미니멀리스트"가 결론 | **가능 — DL 아키텍처 복잡도 무용론** | https://www.mdpi.com/1996-1073/18/24/6395 | `[스니펫확인]` (3.7%) |
| 27 | **Markovics et al. 2026**, *A unified benchmark of deep learning and classical ML for weather-based solar power forecasting: Balancing complexity and skill* | ✅ NWP-based day-ahead PV | 제목 자체가 "복잡도와 skill의 균형". 세부 수치 미확보 | 가능 (후속 확인 권장) | https://www.sciencedirect.com/science/article/pii/S266654682600159X | `[전문미확인]` |
| 28 | **Baidu KDD Cup 2022 (SDWPF)** | ❌ **SCADA 이력 사용 (자기회귀)** | 48시간 앞 10분단위. 3위 해법 및 Team 88VIP 해법이 **"기본 데이터 패턴 암기용 GBDT"+NN 조합** | **이전 불가 (lag 기반)** — 단, 상위권도 GBDT를 버리지 않았다는 점만 참고 | https://baidukddcup2022.github.io/ · https://github.com/LongxingTan/KDDCup2022-WPF · https://www.semanticscholar.org/paper/b2412a9580a216b20450d667ea1d9eee3b20a081 | `[스니펫확인]` |
| 29 | **CNN/3D-CNN NWP 격자 피처추출** (Higashiyama et al. 2017/2018) | ✅ NWP 격자 압축 | "CNN 기반 피처추출로 고차원 NWP 결과를 압축" — 우리 914열 격자와 직접 대응 | **가능 (§C3에서 다룸)** | https://waseda.elsevierpure.com/en/publications/feature-extraction-of-numerical-weather-prediction-results-toward/ · https://www.researchgate.net/publication/329225365 | `[전문미확인]` |
| 30 | **Rutyna et al. 2025**, *Gated Lag and Feature Selection for Day-Ahead Wind Power* (MDPI) | ❌ SCADA-only, lag 기반 | "lag/rolling 통계 피처가 일관된 점증 개선" | **이전 불가 — 우리 세팅에서 lag 피처는 존재하지 않는다** | https://www.mdpi.com/2674-032X/5/4/28 · https://www.researchgate.net/publication/411154413 | `[스니펫확인]` |
| 31 | **다수의 CNN-LSTM / BiLSTM-CNN / Informer 하이브리드 풍력 논문** (Zhen 2020, Wang 2022, Mao 2025, Ay 2025, Harrou 2024 등) | ❌ **거의 전부 lag 사용** | 두 자릿수 % 개선 주장 다수 | **이전 불가** — 시간축 인코더의 이득이 곧 자기상관 활용분이며, 우리에겐 그 축이 물리적으로 없음 | (표 하단 URL 모음 참조) | `[전문미확인]` |

> **레인 C의 방법론적 관찰:** 조사한 풍력 DL 논문 중 **NWP-only 조건을 명시한 논문은 소수(23·24·25·26·29)** 이고,
> 그 소수 집단에서는 **개선의 출처가 아키텍처가 아니라 (a) NWP 피처공학(13~37%)** 또는 **(b) 얕은 모델(KRR/MLP/선형)** 이다.
> 반대로 두 자릿수 개선을 주장하는 시퀀스 DL 논문은 거의 전부 lag/SCADA를 쓴다.
> **"풍력 DL이 12% 개선했다"는 문장은 우리 세팅에서 기본값으로 무효이며, lag 사용 여부 확인 전까지 인용 불가하다.**

### C1-e. 목적함수 / 계단손실 (질문 6)

| # | 문헌 | 내용 | 수치 | URL | 등급 |
|---|---|---|---|---|---|
| 32 | **Elmachtoub & Grigas, *Smart "Predict, then Optimize"*, Management Science 2022** (인용 1,473) | SPO / **SPO+ 볼록 대체손실**. 예측을 하류 최적화의 결정품질로 직접 학습 | SPO loss = 예측이 유도한 결정의 초과비용 | https://par.nsf.gov/servlets/purl/10339524 · https://optimization-online.org/wp-content/uploads/2018/12/6398.pdf | `[스니펫확인]` |
| 33 | **Mandi et al., *Decision-Focused Learning: Foundations, SOTA, Benchmark and Future Opportunities* (JAIR)** | DFL 서베이 | 2단계(two-stage) 대비 DFL 이득 조건 정리 | https://arxiv.org/html/2307.13565v4 | `[전문미확인]` |
| 34 | **Alkhulaifi et al. 2025 (Expert Systems w/ Applications)** | DFL + 자동 피처공학 | "**자동 피처공학 통합이 DFL을 56% 개선**", "**배터리 저장 최적화에서 SPO+가 최저 regret**" | https://www.sciencedirect.com/science/article/pii/S0957417425041697 | `[스니펫확인]` (56%) |
| 35 | **Stratigakos et al. 2022, *Prescriptive Trees for Integrated Forecasting and Optimization* (IEEE TPWRS)** | **트리 분할 기준 자체를 하류 결정가치로 대체** — GBDT 계열의 decision-focused 판 | "표준 확률최적화 대비 prescriptive 성능 개선" | https://ieeexplore.ieee.org/document/9716858/ · https://hal.science/hal-03330017v3/document · https://github.com/akylasstrat/prescriptive_trees_power_apps | `[스니펫확인]` (문장) / 수치 `[전문미확인]` |
| 36 | **ε-insensitive loss (SVR)** | \|err\| ≤ ε 구간에서 손실 0 — **FICR의 6%cap 밴드와 정확히 같은 형태의 볼록 완화** | ε-SSVR 등 매끄러운 근사 존재 | https://www.researchgate.net/publication/297778367 · https://or.stackexchange.com/questions/10687/ | `[스니펫확인]` (형태) |
| 37 | **Grabocka et al. 2019, *Learning Surrogate Losses*** | 미분불가·비분해(non-decomposable) 손실을 신경망으로 대리학습 | — | https://arxiv.org/pdf/1905.10108 | `[전문미확인]` |
| 38 | **Piscis (2024/2025, PMC)** — F1 score의 미분가능 추정 | **Gaussian 기반 soft indicator**로 임계 판정을 매끄럽게 근사 | — | https://pmc.ncbi.nlm.nih.gov/articles/PMC10862914/ | `[스니펫확인]` (soft indicator 기법) |
| 39 | **한국 재생에너지 예측제도 (KPX)** — **우리 지표의 원형** | 예측오차율 **≤6% → 4원/kWh, 6~8% → 3원/kWh, >8% → 0원**. 즉 FICR은 실제 정산제도의 사본 | 인센티브 모델 연구(Ko 2021 등), 집합자원 구성모델(JIIS 2023) 존재. **집합화(pooling)가 오차율을 낮춰 정산금을 올리는 것이 문헌의 표준 레버** | https://www.jiisonline.org/files/DLA/20231231151800_12.pdf · https://ieeexplore.ieee.org/iel7/6287639/9312710/09429173.pdf · https://www.researchgate.net/publication/356481716 | `[스니펫확인]` (4원/3원/0원) |

---

## §C2 판정

> ### **결론 (한 문장)**
> **딥러닝 *시계열* 표현(N-BEATS/N-HiTS/PatchTST/DLinear/TiDE/TSMixer/TFT)은 우리 세팅에서 쓰지 않는다 — 자기회귀가 불가능한 순수 exogenous-only 회귀에서 이 아키텍처들의 이득 원천(타깃 자기구조 인코딩)이 물리적으로 존재하지 않고, 남는 것은 830열 테이블 위의 MLP이며, 시간분할·피처풍부 조건의 최대 규모 벤치마크는 그 조건에서 GBDT를 최고로 판정하기 때문이다.** 단, **예외 하나**를 열어 둔다: *얕은 NN(MLP/KRR)을 표현 대체가 아니라 **오차상관이 낮은 앙상블 멤버**로 쓰는 것*은 문헌상 근거가 있고 우리의 최대 병목(멤버 상관 0.984~0.994)을 정확히 겨눈다.

### 근거 수치 (3개 이상)

1. **TabReD (ICLR 2025 Spotlight), 8개 산업급 데이터셋: "GBDT show the best results on the TabReD datasets."**
   TabReD가 학계 tabular DL 벤치마크에 결여되어 있다고 지목한 두 특성이 **(i) temporal shift(시간분할)와 (ii) 광범위 피처엔지니어링**인데,
   이는 우리 세팅(2023–24 학습 → 2025 평가, 830 통계열 + 914 격자열 + 304 geometric)의 정의 그 자체다.
   `[스니펫확인]` https://arxiv.org/html/2406.19380v4

2. **Grinsztajn et al. (NeurIPS 2022): tree-based가 ~10K 표본에서 SOTA이고, 대형 체제조차 50,000 표본으로 절단해 평가했을 때 격차가 좁아질 뿐 역전되지 않는다.**
   우리 겉보기 44,000행은 이 "대형(50k)" 경계 **아래**이고, **독립 발행 기준 ~1,780**은 "중형(10k)" 경계보다 **한 자릿수 아래**다.
   두 계산 중 어느 쪽으로 읽어도 tree-favored 구간을 벗어나지 못한다.
   `[스니펫확인]` https://arxiv.org/abs/2207.08815

3. **McElfresh et al. (NeurIPS 2023, 19 알고리즘 × 176 데이터셋): "GBDT의 가벼운 하이퍼파라미터 튜닝이 최적 알고리즘 선택보다 성능을 더 많이 올린다."**
   그리고 "GBDT는 왜곡·두꺼운 꼬리 피처 분포를 NN보다 훨씬 잘 다룬다".
   풍속→발전량 변환은 3제곱 물리와 cut-in/rated/cut-out 포화가 겹쳐 **본질적으로 두꺼운 꼬리·불규칙**이다.
   *동시에 이 결과는 우리 프로젝트의 기존 관측 — 시드 배깅/대형 하이퍼파라미터/L1 목적 모두 ±0.5% 이내 — 과 정면 충돌하지 않는다:
   McElfresh의 주장은 "튜닝 이득 > 알고리즘 교체 이득"이며, 우리 측정은 **튜닝 이득이 이미 소진되었다**는 뜻이므로,
   그 부등식은 "알고리즘 교체 이득은 그보다도 작다"를 함의한다.*
   `[스니펫확인]` https://arxiv.org/abs/2305.02997

4. **Zeng et al. (AAAI 2023): LTSF-Linear가 Transformer 계열을 모든 경우에, 종종 20~50% 폭으로 능가.**
   즉 시계열 DL의 "표현 우위"라는 서사가 **선형 사상 하나로 재현**되었다.
   그런데 DLinear/NLinear의 입력은 **전적으로 타깃 히스토리**다. 우리에게 타깃 히스토리는 없다.
   따라서 이 계열은 우리 세팅에서 **입력이 빈 모델**로 퇴화한다 — 개선 여지가 아니라 정의역 오류다.
   `[스니펫확인]` https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089

5. **TabArena (NeurIPS 2025)의 반박조차 우리를 돕지 않는다.**
   "통상 튜닝 체제에서는 CatBoost가 1위"이고, NN이 최강이 되는 것은 **post-hoc 앙상블 이후**다.
   이는 본 프로젝트의 기존 확립사항(`AutoML SOTA는 자동 피처엔지니어링으로 이기지 않는다 — 스태킹/배깅이 이득의 전부`)과
   **독립적으로 같은 결론**이며, 동시에 **fold-outside 게이트가 다자유도 블렌드를 전부 기각**한 우리 측정과 충돌한다.
   즉 TabArena가 제시하는 유일한 DL 이득 경로(앙상블)는 우리 저장소에서 이미 닫혀 있다.
   `[스니펫확인]` https://arxiv.org/html/2506.16791v1

6. **NWP-only 조건을 명시한 재생에너지 문헌에서 개선의 출처는 아키텍처가 아니라 피처다.**
   Couto & Estanqueiro 2022: WRF 기반 신규 피처만으로 **RMSE 13~37% 감소**.
   Kibet 2025: "선형 모델"이 NWP 베이스라인 대비 **RMSE 최소 3.7% 개선**(미니멀리스트 결론).
   목표(pc MAE 10~12% 감소)의 규모는 **피처 축에서 관측된 적이 있고, 아키텍처 축에서는 NWP-only 조건 하에 관측된 적이 없다.**
   `[스니펫확인]` https://ui.adsabs.harvard.edu/abs/2022REne..201.1076C/abstract · https://www.mdpi.com/1996-1073/18/24/6395

### 예외 조항 (판정을 뒤집지는 않지만 기록해야 하는 반대 증거)

**Markovics & Mayer 2022** (인용 562)는 **NWP 기반 결정론적 day-ahead 예측**이라는, 우리와 구조가 가장 같은 조건에서
**24개 ML 방법을 비교해 kernel ridge regression과 MLP를 가장 정확한 두 모델로 판정**했다 (선형 베이스라인 대비 RMSE −13.9%).
이것은 "exogenous-only day-ahead 회귀에서 얕은 NN이 트리를 이길 수 있다"는 **직접적 반례**다.
다만 (a) 대상이 태양광이고, (b) 이긴 모델이 *시계열 아키텍처가 아니라 얕은 회귀기*이며, (c) 우리의 fold-outside 게이트를 통과할지는 미검증이다.
따라서 이 증거는 **"시계열 DL을 도입하라"가 아니라 "MLP/KRR을 저상관 앙상블 멤버 후보로 1회 검증하라"** 로만 번역된다 (§C3 R1).

### 질문 2에 대한 별도 판정 — 소표본 tabular에서 DL이 이기는 임계 표본수

문헌이 제시하는 경계값은 세 개이고, 서로 정합한다.

| 경계 | 출처 | 값 | 우리 위치 |
|---|---|---|---|
| **TabPFN 우세 상한** | McElfresh 2023 | **표본 ≤ 3,000** 이면 TabPFN이 평균적으로 전부 이김 | 겉보기 44k ✗ / **독립발행 1,780 ✓** |
| **tree-based SOTA 구간** | Grinsztajn 2022 | **~10,000 (medium)**, 그리고 **50,000으로 절단한 large 체제에서도 유지** | 44k, **구간 내부** |
| **TabPFN v2 적용 상한** | Hollmann 2025 (Nature) | **10,000 표본 / 500 피처**, 논문이 초과 확장을 명시적으로 경고 | 44k ✗, 830열 ✗ |
| **DL이 "따라잡는" 구간** | TabArena 2025 | "**larger** 데이터셋에서 DL이 따라잡음" (임계치 수치 미제시) | 임계 미달로 추정 |

**판정: 4.4만행은 "GBDT 쪽"이다.**
겉보기 44k는 Grinsztajn의 tree-favored 구간 내부이고, 독립 표본 ~1,780으로 읽으면 아예 소표본 체제다.
McElfresh는 나아가 **"NN vs GBDT 논쟁이 과대평가되었고 데이터셋 크기가 아니라 불규칙성 계열 메타피처가 승패를 가른다"** 고 결론하는데,
우리 데이터는 그 불규칙성 축(두꺼운 꼬리 풍속, 포화 비선형)에서 GBDT에 유리한 쪽에 있다.

### 질문 4에 대한 별도 판정 — TabPFN 적용 가능성

**적용 불가. 두 갈래 모두 막힌다 (규모 vs 라이선스의 이중구속).**

| 모델 | 공개일 (기준 2026-07-05) | 상업이용 | 규모 한계 | 우리 44k × 830열 | 판정 |
|---|---|---|---|---|---|
| **TabPFN v2** | 2025-01 ✅ 통과 | ✅ **허용** (Prior Labs License = Apache 2.0 + 귀속) | **10,000행 / 500피처** | 행 4.4배 초과, 열 1.7배 초과 | ❌ **규모 위반** |
| **TabPFN-2.5** | 2025-11 ✅ 통과 | ❌ **비상업 전용** — "outputs cannot be used for any commercial or production purpose" | 50,000행 / 2,000피처 (우리에 딱 맞음) | 규모는 맞음 | ❌ **라이선스 위반 → 대회 규칙상 사용 금지** |
| **TabPFN-2.6 / TabPFN-3** | 2025-12~ | ❌ 비상업 | — | — | ❌ |
| **TabPFN-TS** | 2025-01 ✅ | v2 기반이면 ✅ | **v2의 10k/500 상한 승계** | 동일 위반 | ❌ |

**우회로가 있는가:** 있으나 전부 자유도를 늘리는 방향이다.
(a) 그룹별로 쪼개면 g3는 8,700행으로 10k 이내 진입 — 그러나 피처 500열 제약이 여전하고, 830→500 선택 자체가 새 자유도다.
(b) 독립발행 단위(하루 1행 × 24개 타깃)로 재구성하면 ~1,780행 — TabPFN의 최적 구간이지만 타깃이 24차원 다출력이 되어 v2 회귀 API와 맞지 않는다.
(c) Chunked TabPFN / LocalPFN 류는 비공식·근사이고 `[전문미확인]`이다.
**본 레인의 권고: TabPFN 축은 열지 않는다. 열더라도 v2 + g3 단독 + 상위 500열이라는 3중 제약을 명시한 단일 사전선언 실험으로만.**

---

## §C3 이식 가능한 표현학습 후보

표기: **[태그] 구현식 / 기대이득 / 비용 / 추가 자유도 / 근거**

### R1 — `[검토권장]` 얕은 NN(MLP 또는 kernel ridge)을 **표현이 아니라 저상관 앙상블 멤버**로

- **구현식**: `p_mlp = MLP(X_830)` 를 기존 GBDT와 동일한 fold/정책 하에 학습 →
  `p = w·p_gbdt + (1−w)·p_mlp`, **w는 단일 스칼라, fold-outside로만 추정**.
  RealMLP의 기본값 레시피(사전튜닝된 기본 하이퍼파라미터)를 그대로 쓰면 튜닝 자유도를 0으로 눌러 놓을 수 있다.
- **기대이득**: 점정확도 자체는 ±0. 이득은 **오차상관 저하**에서만 나온다.
  현재 멤버 상관이 분류기 계열 0.984~0.994, analog 0.944인데, **모델 클래스가 다른 멤버는 이 표에 없다.**
  Markovics & Mayer에서 KRR/MLP가 NWP-only day-ahead에서 1·2위였다는 사실은 이 멤버가 *열등하지 않을* 근거이고,
  TabArena의 "앙상블 후 NN 최강"은 이 멤버가 *상보적일* 근거다.
- **비용**: 낮음. 학습 수 분. 별도 데이터·외부 가중치 불필요.
- **추가 자유도**: **1개 (w)**. 기존 fold-outside 게이트가 이미 이 자유도 수를 통과시킨 전례가 있음(w=0.70 블렌드).
- **근거**: #13 TabArena, #14 RealMLP, #24 Markovics & Mayer. `[스니펫확인]`
- **주의**: AGENTS.md 표준 규칙 — (a) 어느 정책이 각 입력을 만들었는지, (b) 가중치가 in-sample인지 fold-outside인지,
  (c) 행 정렬 키 집합을 반드시 명기해야 admissible.

### R2 — `[조건부]` 914열 격자장의 **PCA 잠재표현** (autoencoder 아님)

- **구현식**: 변수군별(풍속·풍향성분·기온·기압 등)로 격자열을 나눠 각 군에 PCA를 적용,
  누적분산 95%까지 상위 성분만 GBDT 피처로 주입. `X_new = [X_830, PCA_k(grid_group_j)]`.
- **기대이득**: **낮음~중간, 그리고 근거가 간접적이다.**
  이 저장소는 **원시 격자 pivot 직접 투입이 −0.4%**로 이미 실패했다.
  PCA는 그 실패의 원인이 "정보 부재"가 아니라 "차원 대비 표본 부족"일 때만 회복한다.
  독립발행 ~1,780 대비 914열이므로 표본/차원 비가 2:1 — **PCA가 도움이 될 수 있는 전형적 형태**이나,
  트리는 이미 축 정렬 분할로 무관 열을 무시하므로 회복폭은 작을 것으로 본다.
- **비용**: 낮음 (fold 내부에서 fit해야 누출 없음 — 이것이 실제 구현 비용의 대부분).
- **추가 자유도**: 성분 수 k, 변수군 분할 방식 → **최소 2개**. fold-outside 게이트를 통과시키려면 k를 사전 고정해야 한다.
- **근거**: #25 Couto(NWP 파생 피처가 13~37%), #29 Higashiyama(NWP 격자 CNN 압축). `[전문미확인]` 수준.
- **판정**: R1 이후 순위. **autoencoder 판은 권장하지 않음** — 학습 자유도(구조·잠재차원·정규화·에폭)가 최소 4개 늘고,
  소표본 tabular에서 AE 잠재표현이 GBDT를 개선했다는 **명확한 벤치마크 수치를 이 레인은 찾지 못했다**(검색 결과가 전부 분류·고차원 유전체 도메인).

### R3 — `[닫음]` Entity embedding (범주형 임베딩) → GBDT 피처 주입

- **구현식**: NN으로 범주형 임베딩 학습 후 그 벡터를 GBDT 입력에 concat (Guo & Berkhahn 2016 방식).
- **기대이득**: **≈ 0.** 원논문은 KNN/RF/GBT를 entity embedding 피처로 개선했다고 보고하나 `[전문미확인]`,
  그 이득은 **Rossmann의 1,115개 store처럼 고카디널리티 범주형**에서 나온다.
  **우리 범주형은 group(3) + hour(24) + month(12)뿐이다.** 카디널리티가 두 자릿수인 곳에서 임베딩은 one-hot/target encoding과 구별되지 않는다.
- **비용/자유도**: 중간 비용, 자유도 3+ (임베딩 차원, NN 구조, 학습 스케줄).
- **판정**: **닫음.** 근거: 이득 메커니즘(고카디널리티)이 우리 데이터에 부재.
- **근거**: #(Guo & Berkhahn 2016) https://ar5iv.labs.arxiv.org/html/1604.06737 `[전문미확인]`

### R4 — `[닫음]` 수치 피처 임베딩 (Gorishniy et al. 2022, periodic/PLE)

- Yandex의 "On embeddings for numerical features in tabular DL"은 **바닐라 MLP를 크게 끌어올린다**고 보고한다.
- 그러나 이 기법(구간화 + 학습된 조각별 선형 인코딩)은 **트리가 이미 하는 일**이다. GBDT는 단일 피처의 단조변환에 불변이므로 **이식 이득이 정의상 0**이다.
- **판정: 닫음** (R1의 MLP 멤버 *내부* 개선으로만 의미 있음).
- https://research.yandex.com/blog/embeddings-for-numerical-features-in-tabular-deep-learning `[전문미확인]`

### R5 — `[닫음]` Contrastive / self-supervised 사전학습 (SCARF, VIME, SubTab)

- SCARF(ICLR 2022)·VIME·SubTab의 보고된 이득은 **(a) DNN 하류 모델**에 대해, **(b) 라벨이 극소량인 체제**(reddit 보고: "라벨 1–5%일 때 가장 가치 있음")에 집중된다.
- **우리는 학습기간 전 구간 라벨이 있고, 하류 모델이 GBDT다.** 두 조건 모두 불일치.
- 추가로 SSL은 unlabeled 풀이 커야 하는데, 우리의 unlabeled 풀(2025 NWP)은 test 기간이라 **표현학습에 쓰면 test-period 사용 논란**을 부른다
  (NWP 자체는 D-1 14:00 KST에 가용하므로 규칙 위반은 아니나, 별도 사전선언 없이 열 축이 아니다).
- **판정: 닫음.**
- https://arxiv.org/abs/2106.15147 · https://papers.neurips.cc/paper/2021/file/9c8661befae6dbcd08304dbf4dcaf0db-Paper.pdf `[전문미확인]`

### R6 — `[닫음]` CARTE / 사전학습 tabular foundation 표현

- CARTE(ICML 2024)의 이득은 **컬럼명이 불일치하는 여러 외부 테이블을 함께 학습**하는 데서 온다.
- 우리에게 결합할 외부 테이블이 없다(반해석 자료 금지, test 기간 관측 금지).
- **판정: 닫음.** https://arxiv.org/html/2402.16785v2 `[전문미확인]`

---

## §C4 목적함수 / 계단손실 축 후보

**이 레인의 두 번째 결론: 표현 축보다 목적함수 축이 우리 지표 구조에 훨씬 가깝다.**
FICR은 문헌상 "정확도 지표"가 아니라 **한국 KPX 재생에너지 예측제도 정산식(≤6% → 4원, 6~8% → 3원, >8% → 0원)** 그 자체다(#39).
정산제도를 대상으로 한 최적화 문헌은 존재하고, 표준 TSF 문헌과는 다른 계열이다.

### O1 — `[최우선 검토]` ε-insensitive (밴드) 손실을 GBDT 커스텀 목적으로

- **구현식**: LightGBM custom objective. `L(e) = max(0, |e| − ε)` 또는 그 매끄러운 판(ε-SSVR),
  `ε = 0.06·cap`. gradient `sign(e)·1[|e|>ε]`, hessian은 상수 또는 Huber식 완화.
- **왜**: FICR의 6% 밴드는 **정확히 ε-insensitive 손실의 형태**다. L1/L2는 밴드 내부의 오차를 계속 벌하므로
  밴드 경계 근처의 예측을 밴드 밖 표본 쪽으로 끌어당긴다. ε-insensitive는 그 인력을 제거한다.
- **기대이득**: 중간. **점정확도(NMAE)는 나빠질 수 있고 FICR이 좋아지는 트레이드오프**를 만든다.
  Total = 0.5(1−NMAE) + 0.5·FICR 이므로 **순이득 여부는 반드시 측정해야 한다**.
- **비용**: 매우 낮음 (LightGBM custom objective 1개 함수).
- **자유도**: **1개 (ε, 그러나 6%cap으로 사전 고정 가능 → 0개)**. 이것이 이 축의 최대 장점이다.
- **근거**: #36 ε-insensitive/ε-SSVR, #39 KPX 정산구조. `[스니펫확인]` (형태), 수치 `[전문미확인]`
- **주의**: `actual < 0.1·cap` 행은 NMAE·FICR 양쪽 기여가 0이므로(이 저장소의 기확립 측정),
  커스텀 목적의 샘플가중치에서도 0으로 두어야 목적과 지표가 정렬된다.

### O2 — `[검토]` 계단 보상의 매끄러운 대리(soft-indicator) 직접 최적화

- **구현식**: `R̂(e) = 4·σ((0.06c − |e|)/τ) + 3·[σ((0.08c − |e|)/τ) − σ((0.06c − |e|)/τ)]`,
  발전량 가중을 곱해 최대화. τ는 온도(annealing).
- **왜**: FICR을 대리손실로 직접 최적화. Piscis(#38)가 F1에 대해 Gaussian soft indicator로 한 것과 같은 조작.
- **기대이득**: 중간~높음이나 **불안정**. 계단의 대리는 표본이 적을수록 국소최적이 많고, τ 스케줄이 결과를 좌우한다.
- **비용**: 중간. **자유도: τ + 스케줄 → 2개 이상.** fold-outside 게이트 통과가 R2/O2의 진짜 관문.
- **근거**: #37 Grabocka 2019, #38 Piscis. `[전문미확인]`

### O3 — `[이론적으로 가장 정확, 이미 부분 구현됨]` **모드 추정 = 밴드확률 최대화**

- **핵심 사실**: 계단 보상 `R(|e|)` 하에서 최적 행동은 조건부 평균도 조건부 중앙값도 아니라
  **`a* = argmax_a ∫ R(|a − y|) f(y|x) dy`**, 즉 *보상창을 커널로 쓴 평활밀도의 최빈값*이다.
  6%/8% 두 계단이면 커널은 폭 0.12c(가중 4)와 0.16c(가중 3)의 두 상자함수 합이다.
- **AGENTS.md의 "배포 예측은 조건부 평균이 아니라 계단보상 하 기대정산 최대화 ACTION" 이라는 문장이 정확히 이 사실이며,
  현재 `T*_G*` 정책이 그 근사다.** 따라서 이 축은 새 축이 아니라 **기존 정책축의 이론적 정당화**로 기록해야 한다.
- **남아있는 여지**: 현재 정책은 스칼라 파라미터(T, G) 2개로 그 argmax를 근사한다.
  **완전한 형태는 조건부 분포 f(y|x)를 추정(quantile GBDT 다분위 또는 NGBoost)한 뒤 상자커널 합성곱의 argmax를 수치적으로 취하는 것**이다.
  이는 **자유도를 늘리지 않으면서**(커널이 지표에서 유도되므로 자유 파라미터 없음) 정책을 개선할 수 있는 유일한 경로다.
- **기대이득**: **이 레인이 본 후보 중 이론적 정합성이 가장 높다.** 단, 다분위 추정 자체의 자유도(분위 격자 수)가 붙는다.
- **비용**: 중간 (분위별 LightGBM 9~19개 학습).
- **근거**: 결정이론(0-1 tolerance loss의 Bayes 추정량 = 모드) + #32 SPO의 "결정중심" 원리. `[전문미확인]` (직접 인용 논문 미확보)

### O4 — `[참고]` Prescriptive trees (Stratigakos 2022) — 분할기준을 결정가치로 대체

- 트리 분할 자체를 하류 결정 품질로 학습하는 **GBDT 계열의 decision-focused 판**이고,
  적용 도메인이 **재생에너지 거래**로 우리와 같다.
- **기대이득**: 미지. 구현비용 높음(커스텀 트리 학습기), 공개 코드 존재(GitHub).
- **판정**: 이번 사이클에서 열지 않음. O1/O3가 같은 방향을 훨씬 싸게 준다.
- https://ieeexplore.ieee.org/document/9716858/ · https://github.com/akylasstrat/prescriptive_trees_power_apps `[전문미확인]`

### O5 — `[닫음]` SPO / SPO+ (Elmachtoub & Grigas)

- SPO/SPO+는 **예측된 파라미터가 하류 최적화 문제(주로 선형계획)의 계수로 들어가는** 구조를 전제한다.
- 우리는 하류 최적화 문제가 없다. 예측값 자체가 결정이고 보상은 예측오차의 직접 함수다.
- 즉 **SPO의 대리손실 기계가 필요 없고, O3의 직접 기대보상 최대화가 이미 정확해(exact solution)다.**
- **판정: 닫음** (문헌 인용 가치만 있음). https://par.nsf.gov/servlets/purl/10339524

### O6 — `[닫음, 규칙]` 집합자원(pooling) 레버

- KPX 문헌의 표준 레버는 **여러 자원을 집합해 상대오차를 줄여 정산금을 올리는 것**(#39).
- 대회는 그룹이 고정되어 있어 이 레버가 없다. 이미 확인된 "3그룹 풀링 −1.8%"는 *학습* 풀링이지 *정산* 풀링이 아니다.
- **판정: 닫음.**

---

## §C5 닫히는 축 (증거와 함께 명시적으로 종료)

| 축 | 종료 사유 | 결정적 근거 |
|---|---|---|
| **N-BEATS / N-HiTS** | 아키텍처가 **단변량 자기회귀 basis expansion**. backcast 분기의 입력이 타깃 히스토리이며, 우리에겐 없음. N-BEATS 원 구현은 exogenous를 지원하지 않음 | https://www.sciencedirect.com/science/article/pii/S1568494625008865 ("Limited Use of Exogenous Variables: … designed for univariate time series, relying solely on historical demand data") `[스니펫확인]` |
| **DLinear / NLinear / LTSF-Linear** | **정의상 타깃 히스토리의 선형사상**. 타깃 히스토리 = ∅ 이면 모델 = 상수. 개선 여지가 아니라 정의역 오류 | https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089 `[스니펫확인]` |
| **PatchTST / Autoformer / FEDformer / Informer / TimesNet** | patching·분해·자기상관 블록이 전부 **타깃 계열의 자기구조**를 대상으로 함. 그리고 이 계열은 DLinear에 20~50% 차이로 이미 패배 | 동상 `[스니펫확인]` |
| **TFT** | Elsayed 2021에서 GBRT를 일관되게 이긴 **유일한 DNN**이지만, TFT의 이득 구조가 (i) 과거 관측 covariate 인코더 + (ii) static covariate + (iii) 다중 시계열 cross-learning인데 우리는 (i)이 없고 (iii)은 그룹 3개뿐 | https://ai-scholar.tech/en/articles/time-series/need_DL_for_TSF `[스니펫확인]` |
| **TiDE / TSMixer-Ext** | **구조적으로는 유일하게 admissible**(known-future covariate만으로 동작 가능). 그러나 그 경우 모델은 *"830열 테이블 위의 MLP + 24스텝 시간축 mixing"* 으로 환원되고, 그 질문의 답은 C1-b가 이미 준다 (TabReD: GBDT 최고 / Grinsztajn: 50k까지 tree-favored). **R1(MLP 앙상블 멤버)이 이 축의 저비용 대체재이므로 별도 축으로는 닫는다** | #10 #12 `[스니펫확인]` |
| **Chronos / Chronos-2 / TimesFM / Moirai 등 TS foundation model** | 전부 **타깃 히스토리를 컨텍스트로 요구**. Chronos-2가 covariate-informed를 지원하나 past covariate + target history 전제이며 공개일 **2025-10-20** | https://github.com/amazon-science/chronos-forecasting `[스니펫확인]` |
| **TabPFN-2.5 / 2.6 / 3** | **비상업 라이선스** — "outputs cannot be used for any commercial or production purpose". 대회 규칙("상업 이용 라이선스") 위반 | https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/LICENSE `[스니펫확인]` |
| **TabPFN v2** | 라이선스·날짜는 통과하나 **10,000행 / 500피처 상한**을 44,000행 × 830열이 위반. Nature 논문이 초과 확장을 명시적으로 경고 | https://www.nature.com/articles/s41586-024-08328-6 `[스니펫확인]` |
| **Entity embedding → GBDT** | 이득 메커니즘이 **고카디널리티 범주형**인데 우리 범주형은 group(3)/hour(24)/month(12) | §C3 R3 |
| **수치 피처 임베딩 (PLE/periodic)** | GBDT는 단일 피처 단조변환에 불변 → **이식 이득 정의상 0** | §C3 R4 |
| **SCARF / VIME / SubTab (SSL 사전학습)** | 이득이 **저라벨 체제 + DNN 하류**에 국한. 우리는 완전라벨 + GBDT 하류 | §C3 R5 |
| **CARTE** | 결합할 외부 테이블 부재 | §C3 R6 |
| **autoencoder 잠재표현 → GBDT** | 소표본 tabular에서 **AE→GBDT 개선을 보인 명확한 벤치마크 수치를 이 레인은 찾지 못함**. 자유도 4+ 추가. 원시 격자 직접 투입이 이미 −0.4% | §C3 R2 |
| **SPO / SPO+** | 하류 최적화 문제가 없어 대리손실 기계가 불필요. O3가 정확해 | §C4 O5 |
| **lag 기반 풍력 DL 논문군 전체** (CNN-LSTM, BiLSTM-CNN, Informer 하이브리드, KDD Cup 2022 계열) | 개선의 출처가 **자기상관 활용분**이며 2025 평가기간에 관측이 없어 물리적으로 이전 불가 | §C1-d #28 #30 #31 |
| **M4 / M5 / Monash / LTSF 벤치마크를 우리 판정 근거로 쓰는 것** | 8건 중 7건이 target lag 전제. **찬반 양쪽 모두 이전 불가** | §C1-a 요약 |

---

## §C6 이 레인이 권고하는 다음 행동 (우선순위)

1. **O1 (ε-insensitive 밴드 목적)** — 자유도 0(ε=0.06·cap 사전고정), 구현 수십 줄, 지표 구조와 정확히 일치. **가장 먼저.**
2. **R1 (RealMLP 기본값 MLP를 저상관 앙상블 멤버로, w 1자유도)** — 유일하게 남은 "다른 모델 클래스" 멤버.
   기존 멤버 상관 0.984~0.994라는 진단이 정확히 이 처방을 가리킨다.
3. **O3 (다분위 추정 + 상자커널 합성곱 argmax)** — 이론적으로 가장 정확하나 구현비용 중간.
4. **R2 (격자 914열 fold-내부 PCA)** — 조건부. 성분 수 k를 사전 고정할 것.
5. 나머지는 §C5에서 닫힘.

**모든 후보는 AGENTS.md 표준 규칙을 만족해야 한다: (a) 입력별 정책 명시, (b) in-sample vs fold-outside 명시, (c) 행 정렬 키 집합 명시.**

---

## 부록 A — §C1-d #31의 URL 모음 (lag 기반, 이전 불가로 분류)

- Zhen et al. 2020, BiLSTM-CNN 단기 풍력: https://www.mdpi.com/2071-1050/12/22/9490
- Wang et al. 2022, CNN+Informer 하이브리드: https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2021.788320/full
- Mao et al. 2025, day-ahead 구간예측 하이브리드 DL: https://www.mdpi.com/2071-1050/17/7/3239
- Ay et al. 2025, 단기 풍력 DL 비교: https://www.mdpi.com/2073-8994/18/1/11
- Harrou et al. 2024, VAE + self-attention 풍력: https://www.sciencedirect.com/science/article/pii/S259012302400759X
- Liu et al. 2024, 풍력 DL 리뷰: https://link.springer.com/article/10.1007/s10462-024-10728-z
- Yang et al. 2024, 풍력 ML 서베이: https://link.springer.com/article/10.1007/s00521-024-09923-4
- Lee et al. 2024, 복잡지형 day-ahead 풍력 피처공학: https://www.sciencedirect.com/science/article/pii/S0360544223031080

## 부록 B — 증거 등급 집계

- `[스니펫확인]` 로 승격된 수치: M5 14%/92.5%, DLinear 20~50%/>40%/~30%, NBEATSx 20%/5%,
  Grinsztajn 10K/50,000, McElfresh 3,000, TabPFN v2 10,000/500·2.8s/4.8s/5,140×,
  TabPFN-2.5 50,000/2,000 및 비상업 문구, TabArena CatBoost 1위·LightGBM 2위·RealMLP 3위,
  TabReD "GBDT best", Markovics 13.9%/44.6%, Couto 13~37%, Kibet 3.7%, AutoGluon 82~94%/7.4%/85%,
  DFL+AutoFE 56%, KPX 4원/3원/0원, Chronos-2 2025-10-20.
- `[전문미확인]`: Elsayed 표별 수치, Monash 표2 MASE, RealMLP/TabM 정량치, entity embedding 표,
  Higashiyama CNN 압축 정량치, Stratigakos 정량치, Grabocka/Piscis 정량치, Markovics 2026 전체,
  TabPFN 확장연구(Chunked/LocalPFN) 정량치, AE→GBDT 소표본 이득(해당 수치를 찾지 못함).

**본 레인은 저장소 쓰기를 이 파일과 `S6_ext_C_repr.searchlog.json` 2건으로 한정했고, 모델 학습·lockbox 접근·업로드를 수행하지 않았다.**
