# S6 외부문헌 레인 B — 복잡지형 풍속 다운스케일링 피처표현 + 공개 정적자료

- **레인**: `S6_ext_B_terrain` (읽기전용 외부 문헌조사 전용)
- **작성**: 2026-08-08 (로컬 세션 시각)
- **도구**: `websearch` 단독. **PDF 전문 다운로드 없음** — 제목/초록/구글 스니펫만 판독.
  본문 수치가 스니펫 문자열로 직접 인용된 경우에도 표/문맥을 못 봤으므로 **`[전문미확인]`** 을 붙인다.
  스니펫에 숫자가 문자 그대로 나온 것은 `[스니펫직접인용]` 을 추가로 붙여 구분한다.
- **쿼리 수**: 103 (전체 로그: `research/lanes/S6_ext_B_terrain.searchlog.json`)
- **저장소 쓰기**: 이 파일 + 위 searchlog 2개뿐. 모델 학습·데이터 다운로드·커밋·업로드 없음.

---

## §B0 이 레인의 결론 요약 (먼저 읽을 것)

세 줄 요약:

1. **문헌의 "지형 다운스케일링 이득"은 우리에게 거의 그대로 오지 않는다.** 인용 가능한 이득
   (Winstral 2017 RMSE −6%, Wind-Topo MAE −32%, Dupuy 2023 MAE −32%, WindNinja/HRRR +13%)은
   전부 **"원시 NWP 최근접격자 값 대비"** 이득이다. 우리 베이스라인은 이미 LDAPS 16격자 + GFS 9격자의
   원시장을 통째로 먹인 학습된 모델이라 그 이득의 대부분을 **이미 흡수했다**. 문헌 %를 우리 %로 옮기지 마라.
2. **정적 지형지수(TPI/TRI/VRM/slope/aspect/curvature/elevation)를 그룹 단위로 넣는 것은 수학적으로 죽은 축이다.**
   타깃 그룹이 3개뿐이므로 그룹당 상수 벡터는 3-level 그룹 더미와 **정확히 선형종속**이다. 새 정보가 0이다.
   이건 문헌 문제가 아니라 랭크 문제다. §B4-D, §B5-1 참조.
3. **살아있는 형태는 딱 두 갈래다.**
   (a) **격자 단위** 방향조건부 노출 `Sx(격자 j, θ_j(t))` — 16개 LDAPS 격자 각각에 대해 시간마다 값이 바뀌는
       16개 시계열이 되고, 모델이 "이 시각에 어떤 격자가 상류/노출인가"로 격자를 동적 재가중할 수 있다.
   (b) **연직 결합의 재설계** — 특히 `GFS 80/100 m → 117 m 는 내삽에 가깝고 LDAPS 10 m → 117 m 는 11.7배 외삽`
       이라는 비대칭, 그리고 `hub/PBLH` 상대고도에 의한 레짐 스위치.
   나머지는 전부 상수이거나(축 (a)/(b) 밖) 이미 시도된 축과 겹친다.

---

## §B1 지형노출 지수 정의표와 문헌 실증 이득

### B1.1 정의 (수식)

기호: `z(x)` DEM 표고, `A` 대상 셀, `θ` 바람이 **불어오는 방향(상류 방위)**, `d_max` 탐색반경.

| 지수 | 정의 | 방향 의존 | 출처 |
|---|---|---|---|
| **Sx** (maximum upwind slope / shelter index) | `Sx(A, θ, d_max) = max_{i ∈ ray(A,θ,d_max)} atan[ (z_i − z_A) / dist(i,A) ]`. **양수 = 상류에 더 높은 지형이 있음 = 차폐(sheltered)**, 음수 = 노출(exposed). 실무에서는 단일 광선이 아니라 `θ±Δ` 섹터를 5° 간격으로 훑어 평균(또는 분위수)한다. SAGA `Wind Shelter Index` 는 quantile=1 로 두면 최대기울기가 된다. | **예** | Winstral & Marks 2002; Winstral et al. 2009; Winstral et al. 2017 JHM 18:335 — <https://journals.ametsoc.org/downloadpdf/journals/hydr/18/2/jhm-d-16-0054_1.pdf> ; SAGA <https://saga-gis.sourceforge.io/saga_tool_doc/7.6.3/ta_morphometry_29.html> ; RSAGA `wind.shelter` <https://search.r-project.org/CRAN/refmans/RSAGA/help/wind.shelter.html> |
| **Sb** (slope break) | `Sb(A,θ) = Sx_local(A,θ,d_local) − Sx_outlying(A,θ, [d_1,d_2])`. 즉 **가까운 상류의 Sx** 와 **멀리 떨어진 상류창의 Sx** 의 차. 능선 바로 뒤(풍하 급변점)를 탐지. Winstral 2002 는 Sx 와 Sb 를 결합해 퇴적 강화 지점을 찍었다. | **예** | Winstral & Marks 2002; Schön et al. 2015 CRST <https://www.sciencedirect.com/science/article/abs/pii/S0165232X15000312> ; Meehan et al. 2023 TC preprint (Sb = local Sx − outlying Sx 로 명시) <https://tc.copernicus.org/preprints/tc-2023-141/tc-2023-141-manuscript-version5.pdf> |
| **TPI** (Topographic Position Index) | `TPI_R(A) = z_A − mean{ z_i : i ∈ N_R(A) }`. 양수 = 능선/볼록, 음수 = 계곡/오목. **DEV** = `TPI_R / sd(z in N_R)` (표준화형). 다중 스케일 `mTPI` 는 여러 R 을 합성. R=2000 m 급에서 "산체·주능선·주계곡"이 분리된다. | 아니오 | Weiss 2001 TPI poster <https://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf> ; ESRI <https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-analyst/how-topographic-position-index-works.html> ; SAGA mTPI <https://saga-gis.sourceforge.io/saga_tool_doc/6.0.0/ta_morphometry_28.html> |
| **TRI** (Terrain Ruggedness Index) | `TRI(A) = sqrt( Σ_{i∈8이웃} (z_i − z_A)^2 )` (Riley 1999). 단순 국지 거칠기. Trevisani 2023 "Hacking the TRI" 는 TRI 가 사실상 **국지 단파장 변동성** 측도이며 스케일·이방성 처리를 명시해야 한다고 지적. | 아니오 | Trevisani 2023 Geomorphology <https://www.sciencedirect.com/science/article/abs/pii/S0169555X23002581> |
| **VRM** (Vector Ruggedness Measure) | 각 셀의 slope/aspect 를 3D 단위 법선벡터로 분해하고, 이웃창 n 셀의 합벡터 크기 `|R|` 로부터 `VRM = 1 − |R|/n`. 0(평활)~1(극단 거침). 경사 크기와 거칠기를 **분리**하는 것이 TRI 대비 장점. | 아니오 | Sappington et al. 2007; GRASS `r.vector.ruggedness` <https://grass.osgeo.org/grass-stable/manuals/addons/r.vector.ruggedness.html> ; SAGA <https://saga-gis.sourceforge.io/saga_tool_doc/7.8.1/ta_morphometry_17.html> |
| **Ω / W** (MicroMet 지형 풍속 가중) | 구조: `W = 1 + γ_s·Ω_s + γ_c·Ω_c`. `Ω_s` = **바람 방향으로 투영된 경사**, `Ω_c` = **곡률(curvature)**, 둘 다 `[−0.5, +0.5]` 로 스케일. 보정 풍속 `U' = W·U`. 추가로 풍향 편향(diverting) 항이 있다. **`γ_s`, `γ_c` 의 구체 값은 스니펫으로 확인 못 함 — 반드시 원문 Table 확인** `[전문미확인]` | **예** (Ω_s 가 풍향 투영) | Liston & Elder 2006 JHM (MicroMet) <https://research.fs.usda.gov/treesearch/26813> |
| **Wind Effect (Windward/Leeward Index)** | 무차원. `<1` = 풍하 그늘, `>1` = 풍상 노출. 최대탐색거리와 luv/lee 지수를 파라미터로 갖는다. **Wind Exposition Index** 는 이걸 **모든 방향으로 평균** 낸 것 → **방향조건부성을 버림(우리에겐 쓸모 적음)**. | Wind Effect=예 / Exposition=아니오 | Böhner & Antonić 2009, *Developments in Soil Science* 33:195 <https://ui.adsabs.harvard.edu/abs/2009DevSS..33..195B/abstract> ; SAGA ta_morphometry_15 / _27 <https://saga-gis.sourceforge.io/saga_tool_doc/7.6.1/ta_morphometry_15.html> |
| **RIX** (Ruggedness Index, WAsP) | 사이트 주변 지형 중 **임계경사보다 가파른 지형의 면적 비율**. 임계경사는 통상 **0.3 (≈16.7°)**, 문헌에 따라 18° 또는 "30–40% 경사에서 흐름박리" 로 기술. WAsP 실무에서는 **방위섹터별**로 계산하고, 예측오차 지표는 `ΔRIX = RIX(예측지점) − RIX(마스트)`. | **예** (섹터별) | Mortensen et al., DTU/WAsP <https://backend.orbit.dtu.dk/ws/files/107110613/Field_validation.pdf> ; <http://www.wasp.dk/-/media/Sites/WASP/WAsP%20support/Literature/Improving-WAsP-predictions-paper-2006-EWEC.ashx> ; WAsP 포럼 "critical value of 0.3" <https://www.wasptechnical.dk/forum/topic/639-general-questions-about-modelling-with-wasp/> |
| **Helbig 서브그리드 파라미터** | 고해상 풍장을 회귀했을 때 **지형표고의 라플라시안 `∇²z`** 와 **경사제곱 평균 `μ² = <|∇z|²>`** 가 근지표 풍속에 가장 크게 작용하는 두 항. 중립조건 가정. | 아니오 | Helbig et al. 2017 JGR-Atmos 122:651 <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JD025593> ; 인용 확인: Le Toumelin 2023 AIES "Helbig et al. (2017) identified the local Laplacian from terrain elevations and squared slope as valuable parameters to downscale wind speeds" `[스니펫직접인용]` |
| **Jackson–Hunt speed-up** | 낮은 언덕 위 선형이론. 정상부 최대 분율증속 `ΔS ≈ 2·h/L` (h=언덕높이, L=반폭) 수준의 스케일링. 급경사(박리)에서는 무효 → RIX 가 그 한계를 재는 지표. | **예** (언덕 축 대비 입사각) | Jackson & Hunt 1975; 리뷰 <https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2022.762054/full> `[전문미확인]` |

### B1.2 문헌 실증 이득 (수치 + 출처 + 태그)

> **읽는 법**: 아래 이득은 전부 **"원시 NWP 격자값 또는 단순 내삽" 을 기준선으로 한 이득**이다.
> 우리 베이스라인(25개 격자 원시장 + GBM)은 이미 그 기준선보다 훨씬 위에 있다.

| # | 연구 | 대상/해상도 | 지형 표현 | 보고된 이득 | 태그 |
|---|---|---|---|---|---|
| 1 | **Winstral, Jonas & Helbig 2017**, *J. Hydrometeor.* 18:335–348 | 스위스 200+ 관측지점, COSMO-2(~2 km) 및 COSMO-7(~7 km) 지상풍 | **Sx**(방향의존) + **TPI** 로 지형등급 분류 후 OLS 다운스케일 | 스니펫에 표 일부가 그대로 노출: `RMSE (m s−1) Overall (2.62) 2.46 … (2.50) 2.33 2.32`. 괄호=원시, 뒤=다운스케일로 읽으면 **RMSE −6.1% (2.62→2.46), −6.8% (2.50→2.33)**. 본문 서술: "**upper slope 와 ridge 등급에서 편의(bias) 개선이 가장 컸고 KSD 가 크게 줄었다**" | `[전문미확인]` `[스니펫직접인용]` <https://www.dora.lib4ri.ch/wsl/dload/wsl%3A12719/PDF/Winstral-2017-Statistical_downscaling_of_gridded_wind-(published_version).pdf> |
| 2 | **Dujardin & Lehning 2022 "Wind-Topo"**, *QJRMS* 148:1368 | 스위스 알프스, COSMO-1(1.1 km) → 고해상 지형. **261개 관측지점으로 학습** | CNN 이 고해상 지형과 조대 NWP 의 상호작용을 학습 | **알프스 검증지점에서 COSMO-1 의 bias 0.72 / MAE 1.77 m·s⁻¹ → −0.07 / 1.21 m·s⁻¹.** MAE **−31.6%** | `[전문미확인]` `[스니펫직접인용]` <https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4265> ; <https://infoscience.epfl.ch/entities/publication/0b317150-86c1-4c74-a35c-7341655a3628> |
| 3 | **Le Toumelin et al. 2023 "DEVINE"**, *AIES* 2(1) | 프랑스 알프스, ARPS(30 m) 를 CNN 으로 에뮬레이트 후 AROME 에 적용 | CNN 이 고해상 지형맵 + 풍속/풍향을 받아 3성분 풍장 출력 | 에뮬레이션 정확도: **10-fold CV 풍속 MAE = 0.16 m/s (ARPS 대비)**. 실관측 적용: "**AROME wind speed mean bias is reduced by 27% with DEVINE**", 특히 **고도가 높고 노출된 지점**에서 | `[전문미확인]` `[스니펫직접인용]` <https://journals.ametsoc.org/view/journals/aies/2/1/AIES-D-22-0034.1.xml> ; <https://inria.hal.science/hal-03930087/> |
| 4 | **Dupuy, Durand & Hedde 2023**, *NPG* 30:553 | 프랑스 남동부, WRF 지상풍 CNN 다운스케일 | 지형 + **Dujardin & Lehning(2022) 이 도입한 "바람×지형 결합 예측자"를 추가** | **bias −0.55 → −0.01 m/s, MAE 1.02 → 0.69 m/s (−32.4%)**. 야간 안정성층 시간대에서 개선이 특히 컸다 | `[전문미확인]` `[스니펫직접인용]` <https://npg.copernicus.org/articles/30/553/2023/> |
| 5 | **Wagenbrenner et al. 2016**, *ACP* 16:5229 (WindNinja) | 복잡지형에서 4개 NWP 를 질량보존 진단모델로 다운스케일 | 물리(질량보존) 기반, 지형추종 | **부정적 결과 포함**: "WindNinja does not predict the lee side recirculation, and thus, **the downscaling does not improve directions on the lee side of the butte**" | `[전문미확인]` `[스니펫직접인용]` <https://acp.copernicus.org/articles/16/5229/2016/acp-16-5229-2016.pdf> |
| 6 | **Seto et al. 2025**, *Wea. Forecasting* 40(4) | HRRR + WindNinja, 남캘리포니아 Santa Ana | 물리 다운스케일 | "**WN improved the overall forecast accuracy by 13%, on average, relative to HRRR**" 그러나 "**Downscaling increased negative wind speed biases … at stations located in wind-prone lee-slope canyons**" — **평균 개선 / 국소 악화 동시 발생** | `[전문미확인]` `[스니펫직접인용]` <https://journals.ametsoc.org/view/journals/wefo/40/4/WAF-D-24-0013.1.xml> ; <https://www.fs.usda.gov/rm/pubs_journals/2025/rmrs_2025_seto_d001.pdf> |
| 7 | **Marsh et al. 2023 "WindMapper"**, *WRR* | 수문모델용 지형 풍속 다운스케일 | **WindNinja 로 "4개 풍속조건 × 8개 풍향" 의 풍장 라이브러리를 사전계산**해 룩업 | 정성적 결론: "**wind library approaches improve upon the terrain curvature methods for advection problems**" (곡률 기반 MicroMet류보다 낫다) | `[전문미확인]` `[스니펫직접인용]` <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022WR032683> ; 도구 <https://windmapper.readthedocs.io/> |
| 8 | **Helbig et al. 2017**, *JGR-A* 122:651 | 고해상 풍장(중립) 회귀 | `∇²z` + `<|∇z|²>` | 정량 % 미확보. 후속연구들이 "가장 가치있는 두 파라미터"로 재인용 | `[전문미확인]` <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JD025593> |
| 9 | **Jiménez & Dudhia 2012**, *JAMC* (WRF `topo_wind`) | WRF 지상풍 고편의 보정 | 미해상 지형 **표준편차 + 라플라시안 기반 sheltering** | 정성: "**A high standard deviation can produce both a high and low wind speed bias. Parametrization with just information of the variance of the unresolved topography is [불충분]**" → **분산만으로는 안 되고 부호를 주는 항(라플라시안)이 필요**하다는 것이 핵심 교훈. "WRFnew reduces the wind speed biases at most of the observational sites" | `[전문미확인]` `[스니펫직접인용]` <https://www2.mmm.ucar.edu/wrf/users/physics/phys_refs/SURFACE_LAYER/topo_wind.pdf> |
| 10 | **Fischer et al. 2015**, ISPRS Archives XL-1-W5:197 | DEM 만으로 최대풍속 예측 | GBRT + **TPI 등 지형파생** | 지형 단독 예측의 실증사례. 정량 % 미확보 | `[전문미확인]` <https://isprs-archives.copernicus.org/articles/XL-1-W5/197/2015/> |
| 11 | **Mitchell et al. 2008**, *For. Ecol. Manage.* 254:193 | 산악 풍도(windthrow) 예측 | **NWP 풍속 vs 지형노출지수 직접 비교** | 후속 인용: "Mitchell et al. (2008) confirmed the utility of mesoscale NWP data for modelling the occurrence of windthrow events" → **NWP 쪽이 지형지수 단독보다 유용**하다는 방향의 결론 | `[전문미확인]` `[스니펫직접인용]` <https://www.sciencedirect.com/science/article/abs/pii/S0378112707005683> |
| 12 | **Pawlik & Šamonil 2022**, *Sci. Total Environ.* | 산림 풍해 모델링 | 다수 지형변수 중요도 | "**Wind exposure is the most important topographic variable in half of the models**, although slope and valley depth are the [다음]" — 지형변수 **내부**에서 노출지수가 1등. 기상변수 대비 순위는 아님 | `[전문미확인]` `[스니펫직접인용]` <https://www.sciencedirect.com/science/article/pii/S0048969721070480> |

### B1.3 §B1 의 방법론적 교훈 3개

- **(i) 방향조건부성이 이득의 원천이다.** 표에서 실제 이득이 큰 항목(Sx, Wind-Topo, DEVINE, WindMapper 라이브러리)은
  전부 **풍향을 입력으로 받는다.** 방향평균 지수(Wind Exposition Index, TRI, VRM, 정적 TPI)는 정량이득 보고가 없다.
- **(ii) 분산만으론 부호가 안 나온다.** Jiménez & Dudhia 2012 의 관찰 — 지형 표준편차가 크다고 풍속편의의 부호가 정해지지
  않는다. **볼록/오목을 구분하는 항(라플라시안, 곡률, TPI 부호, Sx 부호)** 이 반드시 필요하다.
- **(iii) 물리 다운스케일러는 평균을 개선하면서 국소를 악화시킨다.** Wagenbrenner 2016(풍하 재순환 미모의) +
  Seto 2025(풍하 협곡 음편의 악화). 우리 사이트는 능선이라 풍상/정상 쪽이지만, **풍향에 따라 능선 반대편이 풍하가 된다.**
  단일 정적 보정계수는 이 부호 뒤집힘을 표현할 수 없다.

---

## §B2 다중 연직층 결합 (10 m / 50 m / 80 m / 100 m / 850 hPa / PBL → hub 117 m)

### B2.1 우리 층 구성의 비대칭 (가장 중요한 관찰)

| 소스 | 가용 층 | 117 m 까지의 관계 |
|---|---|---|
| LDAPS | 5 m, 10 m, 50 m (`wind5`, `wind10`, `wind50`, `wind50min`) | **10 m → 117 m 는 11.7배 외삽**, 50 m → 117 m 는 2.34배 외삽 |
| GFS | 10 m, 80 m, 100 m u/v | **80/100 m → 117 m 는 1.17배, 사실상 내삽에 가까운 근외삽** |
| GFS | 850 hPa | 능선 표고 ~1050 m 기준 850 hPa(≈1500 m)는 **능선 위 약 400–500 m** — 언덕 위 "주변 자유대기" 기준면에 매우 가깝다 |
| GFS | PBL height | `117 m / PBLH` 로 무차원 상대고도 구성 가능 |

**결론**: 우리가 겪는 "허브고도 나셀풍속 병목"의 상당 부분은 **고해상 LDAPS 가 10/50 m 밖에 안 주고,
117 m 에 가장 가까운 층은 저해상 GFS(80/100 m)에 있다**는 구조적 비대칭이다. 즉 **"해상도 vs 고도" 의 트레이드오프**가
층 선택 자체에 이미 박혀 있다. 이것은 문헌 인용 없이 우리 데이터 명세만으로 성립하는 사실이다.

### B2.2 문헌: 연직 결합 기법과 정량 이득

| 기법 | 정의/형태 | 보고된 정량 이득 | 출처·태그 |
|---|---|---|---|
| **REWS** (rotor-equivalent wind speed) | 로터면을 고도구간으로 나눠 면적가중 3승 평균: `REWS = [ Σ_i (A_i/A) · u_i^3 ]^{1/3}`. IEC 61400-12-1 Ed.3(2022)는 **3개 이상 고도 측정 시 REWS 산정 가능**이라 규정 | 풍력출력 예측 비교에서 "**the REWS method exhibited a 2.86% improvement over the TPC method**" | `[전문미확인]` `[스니펫직접인용]` <https://www.sciencedirect.com/org/science/article/pii/S2352097326000519> ; IEC 규정 <https://blog.ansi.org/ansi/wind-shear-turbine-performance-iec-61400-12-1/> |
| **REWS + 기온감률(lapse rate) 결합** | 결정트리가 REWS 와 lapse rate 를 결합 | "**decision tree combining rotor-equivalent wind speed and lapse rate improves prediction accuracy by 22% for the given data-set**" | `[전문미확인]` `[스니펫직접인용]` Sasser et al. 2022 NOAA <https://repository.library.noaa.gov/view/noaa/57628/noaa_57628_DS1.pdf> |
| **REWS 의 AEP 영향 규모** | 전력곡선 정규화 | "from **−2.3% to −0.5%**" (AEP 보정 크기) — **이득이 아니라 보정 크기**임에 주의 | `[전문미확인]` `[스니펫직접인용]` Jeon et al. 2017 <https://pendidikankimia.walisongo.ac.id/wp-content/uploads/2018/09/5-vol-40-oct-2017.pdf> |
| **log-law vs power-law vs 가변전단** | `u(z) = (u*/κ)·ln(z/z0)` vs `u(z)=u_r (z/z_r)^α` | "the **log-law and the variable wind shear method produce better estimates than the IEC standard**" (IEC 표준 멱법칙 대비) | `[전문미확인]` `[스니펫직접인용]` Lopez-Villalobos et al. 2022 *Energy Reports* <https://www.sciencedirect.com/science/article/pii/S2352484722011854> |
| **하이브리드 물리-데이터 전단계수** | 전단계수 α 를 물리+데이터로 재산정 후 외삽 | "reduces the wind speed extrapolation **root mean square error by 56.48%** compared to traditional power law" — **단일 100 MW 풍장, 매우 큰 값. 신뢰구간 없이 인용 금지** | `[전문미확인]` `[과대의심]` Wu et al. 2026 *Energies* 19:1302 <https://www.mdpi.com/1996-1073/19/5/1302> |
| **복잡지형 연직층 피처추출 (한국계 저자)** | "Day-ahead wind power forecasting based on **feature extraction integrating vertical layer wind characteristics** in complex terrain" — 복잡지형에서 **상류 예보 정확도에 의존**한다는 문제의식으로 연직층 바람 특성을 통합 | **정량 수치 스니펫 미확보.** 초록: "This study aims to enhance the quality of wind power forecasts in complex terrains, **dependent on the accuracy of upstream forecasts**" | `[전문미확인]` Lee & Park 2024 *Energy* 288:129713 <https://www.sciencedirect.com/science/article/pii/S0360544223031080> ; SSRN 프리프린트 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4509803> |
| **KMAPP (기상청 100 m 상세화)** — **우리 사이트에 가장 근접한 문헌** | LDAPS → 100 m 로 **거칠기 보정(중립대기 로그감소) + 고도 보정** | 정성 결론(한국어 원문 스니펫): "**LDAPS 국지 예보 모형은 산악 지역, 특히 협곡 지역에서 지상 풍속의 과대 모의 경향이 뚜렷**", "**복잡 지형 지역에서 KMAPP 의 비정상적인 풍속 과대 보정은 개선이 필요**", KMAPP-Wind 가 일부 개선 | `[전문미확인]` `[스니펫직접인용]` 금왕호 외 2021, *대기* 31(1):85–100 <https://j-komes.or.kr/xml/28740/28740.pdf> ; <https://koreascience.kr/article/JAKO202112054772243.page> |
| **LDAPS 지상풍 검증** | LDAPS 지상풍/기온 성능 | "LDAPS showed the best performance in predicting surface wind speed and temperature (**average WE = 0.42 m s⁻¹**, average TE = 0.12 °C)" (평지 포함 전국 평균) | `[전문미확인]` `[스니펫직접인용]` Kim et al. 2020 *Atmosphere* 11:1224 <https://www.mdpi.com/2073-4433/11/11/1224> |
| **PBLH 대비 상대고도 / LLJ** | `z/z_i`; LLJ 코어는 통상 **40–250 m** 사이에서 86% 관측 | LLJ 는 "hub height around 100 m 인 통상 터빈에서 **고풍속대에서는 오히려 발전을 감소**시킨다"는 반직관 보고 | `[전문미확인]` `[스니펫직접인용]` Rausch et al. 2022 *Atmosphere* 13:839 <https://www.mdpi.com/2073-4433/13/5/839> ; Haezebrouck et al. WES <https://wes.copernicus.org/articles/11/1343/2026/wes-11-1343-2026.pdf> |
| **지위풍/드래그법칙 (KAMM/WAsP)** | "The neighboring surface wind is also **scaled to the same geostrophic speed with the help of the geostrophic drag law** before the interpolation is done" — 지위풍으로 정규화한 뒤 보간하는 것이 수치풍력지도의 표준절차 | 정량 미확보 | `[전문미확인]` `[스니펫직접인용]` Frank & Landberg, DTU <https://orbit.dtu.dk/files/107075048/THE_NUMERICAL_WIND_ATLAS.pdf> |

### B2.3 §B2 의 교훈

- **LLJ "탐지"는 우리 층 구성으로 원리적으로 불가능하다.** 코어가 40–250 m 인데 우리 최고 층이 100 m(GFS)다.
  프로파일 최댓값을 잡을 수 없다. **단, `u(100m) < u(80m)` (80→100 사이 음의 전단)** 은 **"제트 노즈가 100 m 아래"**
  를 시사하는 계산 가능한 이진 신호다. 이건 "탐지"가 아니라 "부호 플래그"다. §B4-F.
- **REWS 는 우리에게 반쪽만 가능하다.** V126 은 로터직경 126 m(허브 117 m → 로터면 54–180 m),
  U136 은 136 m(49–185 m). 우리 최고 층 100 m 는 로터면 **하부 절반**만 덮는다. 상반부는 여전히 외삽이다.
  즉 REWS 를 "구현"하면 그건 사실상 **외삽 프로파일의 3승 면적가중 요약** 이고, 외삽이 틀리면 REWS 도 틀린다.
  다만 **V126 과 U136 의 로터면이 다르다** → 같은 프로파일에서도 g1/g2 와 g3 의 REWS 가 달라진다.
  **이건 그룹 상수가 아니라 "프로파일 형상(t) × 로터기하(g)" 의 진짜 상호작용이다.** §B4-E.
- **기상청 자체 연구가 "LDAPS 는 산악에서 풍속을 과대모의"라고 말한다.** 그런데 우리 사이트는 **협곡이 아니라 능선**이고,
  1.5 km 격자의 모델 표고는 실제 능선(1000–1100 m)보다 **낮게** 평활화되어 있을 가능성이 높다.
  두 효과(모델 과대모의 vs 능선 표고 과소표현)가 **반대 부호**로 겹친다. 순부호는 데이터로만 알 수 있다.
  **이 사실은 "고도차 보정" 후보(§B4-C)의 부호를 사전에 못 정한다는 뜻이므로, 반드시 부호를 학습시켜야 한다.**

---

## §B3 공개 정적자료 실물표 (URL / 해상도 / 커버리지 / 라이선스 / 공개일 / 규칙 R 판정)

**규칙 R 요약(AGENTS.md 기준)**: 외부 **공개** 데이터 + 오픈소스 사전학습 가중치는 허용.
조건 = (a) **2026-07-05 이전 공개**, (b) **상업이용 가능 라이선스**, (c) **재분석(ERA5/MERRA) 산물 금지**,
(d) **원격 API 추론 금지**, (e) **평가기간(2025) 관측 금지**, (f) 모든 입력은 **D-1 14:00 KST 기준시각에 가용**해야 함.
정적 지형/토지피복은 (f)를 자동 충족(시간불변)한다.

### B3.1 DEM

| 이름 | URL | 해상도 | 종류 | 커버리지(가덕산 37.16N/128.99E 포함?) | 라이선스 | 공개일 | **판정** |
|---|---|---|---|---|---|---|---|
| **Copernicus DEM GLO-30** | <https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM> / AWS <https://registry.opendata.aws/copernicus-dem/> / OpenTopography <https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3> | 30 m (1 arcsec) | **DSM** (TanDEM-X 2011–2015 취득) | 전지구. "GLO-30 **Public** provides **limited** worldwide coverage … a small subset of tiles covering specific countries are not yet released" — 미공개 대상으로 명시된 국가는 **Armenia/Azerbaijan/Moldova(이후 해제)** 이고 **한국은 제한국 목록에 등장하지 않음** → 포함으로 판단하되 실제 타일 존재는 다운로드 전 확인 필요 `[전문미확인]` | Copernicus DEM 라이선스(일반 공중 무료). 상업이용 조항은 라이선스 PDF 확인 필요 <https://docs.sentinel-hub.com/api/latest/static/files/data/dem/resources/license/License-COPDEM-30.pdf> `[전문미확인]` | 2019~2021 공개, 2026-07-05 이전 ✔ | **조건부 가능** — 라이선스 PDF 의 상업이용/재배포 조항을 1회 확인 후 사용. **1순위 후보** |
| **ALOS World 3D AW3D30 (v3.2 / v4.1)** | <https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm> / GEE <https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V4_1> | 30 m (1 arcsec) | **DSM** | 전지구, 한국 포함 | JAXA 이용약관: "**Any of the commercial and non-commercial purposes can be used free of charge** under the conditions of the Terms of Use" `[스니펫직접인용]` — 출처표시(© JAXA) 필요 | v3.x 2021, v4.1 이후 — 2026-07-05 이전 ✔ | **가능** — 상업이용 명시 허용. **Copernicus 와 함께 1순위** |
| **SRTM GL1 v3** | <https://www.earthdata.nasa.gov/data/catalog/lpcloud-srtmgl1-003> / <https://portal.opentopography.org/raster?opentopoID=OTSRTM.082015.4326.1> | 30 m | DSM(레이더) | **60°N–56°S** → 37°N 한국 포함 ✔ | NASA, 사실상 퍼블릭 도메인 | 2015(v3) ✔ | **가능** — 다만 GLO-30/AW3D30 보다 산악 잡음 많음. 백업용 |
| **ASTER GDEM v3** | <https://asterweb.jpl.nasa.gov/gdem.asp> | 30 m | DSM(광학 스테레오) | 83°N–83°S, 한국 포함 ✔ | NASA/METI 무료 | 2019 ✔ | **가능** — 산악 노이즈가 SRTM 보다 크다는 평가가 일반적. 3순위 |
| **FABDEM V1-2** | <https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn> | 30 m | **DTM**(수목·건물 제거) | 전지구 | **CC BY-NC-SA — 비상업 + ShareAlike**. "FABDEM may not be used for commercial [purposes]" `[스니펫직접인용]` <https://gee-community-catalog.org/projects/fabdem/> | 2022 | **불가** — 규칙 R 의 "상업이용 가능" 요건 위반. **후보에서 제외** |
| **NGII 수치표고모델 (국토지리정보원)** | data.go.kr <https://www.data.go.kr/data/15059920/fileData.do> / 국토정보플랫폼 / V-World 90 m <https://www.vworld.kr/dtmk/dtmk_ntads_s002.do?dsId=30206> | **5 m / 10 m** (2015년 이후 구축), V-World 는 90 m | DEM(진짜 지표면) | 남한 전역 ✔ **최고 해상도** | 공공누리 유형 표시 확인 필요. 제1유형(출처표시)이면 상업이용 가능 <https://www.kogl.or.kr/info/licenseType1.do>. **단 "5m 해상도 DEM 자료는 공문으로 데이터를 요청해 승인받아야" 한다는 이용자 증언 존재** `[전문미확인]` <https://swong.tistory.com/5> | 2015~ ✔ | **조건부** — 90 m 는 즉시 가능하나 우리 목적(수백 m 스케일 능선 기하)에는 부족. **5 m/10 m 는 신청·승인 절차가 필요 → 대회 일정 안에서 리스크**. 실무적으로는 GLO-30/AW3D30 으로 대체 |

**DEM 선택 권고**: **Copernicus GLO-30 을 주, AW3D30 을 검증용**으로. 근거 — (1) 둘 다 30 m 로
1.5 km LDAPS 격자를 50×50 셀로 분해 가능, (2) 라이선스가 상업이용 명시(AW3D30) 또는 공중무료(GLO-30),
(3) NGII 5 m 는 승인 절차 리스크, (4) FABDEM 은 라이선스로 배제.
**DSM vs DTM 주의**: GLO-30/AW3D30 은 **DSM**(수목 포함). 가덕산 능선은 산림이므로 표고에 수관고(수 m~20 m)가
섞인다. Sx/TPI 같은 **수백 m~km 스케일** 지수에는 영향이 미미하나, **국지 거칠기(TRI, VRM)** 는 수관 신호에
오염된다 → TRI/VRM 을 "지형 거칠기"로 해석하지 마라.

### B3.2 토지피복 / 조도

| 이름 | URL | 해상도 | 커버리지 | 라이선스 | 공개일 | **판정** |
|---|---|---|---|---|---|---|
| **ESA WorldCover v100(2020) / v200(2021)** | <https://esa-worldcover.org/en/data-access> ; Zenodo v200 <https://zenodo.org/records/7254221> ; AWS <https://registry.opendata.aws/esa-worldcover-vito/> | **10 m** | 전지구 ✔ | **CC BY 4.0** | v200 **2022-10-28 공개**, OA **76.7%** `[스니펫직접인용]` | **가능** — 토지피복 1순위 |
| **Copernicus Global Land Cover CGLS-LC100 c3** | <https://land.copernicus.eu/en/products/global-dynamic-land-cover/copernicus-global-land-service-land-cover-100m-collection-3-epoch-2015-2019-globe> ; Zenodo <https://zenodo.org/records/3939050> | 100 m | 전지구 ✔ | CC BY 4.0 | 2020, OA 80.6% | **가능** — 다만 100 m 는 1.5 km 격자 내부 구조 표현에 부족 |
| **Esri / Impact Observatory 10 m Annual LULC (2017–2025)** | <https://livingatlas.arcgis.com/landcover/> ; AWS <https://registry.opendata.aws/io-lulc/> | 10 m | 전지구 ✔ | CC BY 4.0 (AWS Open Data 등재) `[전문미확인]` | 연간 갱신 | **가능** — 연도별 레이어 존재가 장점이나 정적 피처로 쓸 거면 WorldCover 로 충분 |
| **환경부 세분류 토지피복지도** | <https://aid.mcee.go.kr/intro/land.do> (구 egis.me.go.kr) | **5 m급, 1/25,000, 22개 분류** | 남한 ✔ **최고 분해능·최고 분류세밀도** | 자료신청(로그인 후 신청서 작성) 필요. 공공누리 유형 개별 확인 필요 `[전문미확인]` | 2022/2023년판 존재 ✔ | **조건부** — 신청 절차 필요. 우리 목적(능선 1개 사이트)에는 10 m WorldCover 로 충분하므로 **우선순위 낮음** |
| **조도장 z0 룩업** | WAsP/PyWAsP CORINE 룩업 <https://docs.wasp.dk/pywasp/latest/tutorials/tutorial4_nb.html> ; WindPRO <https://help.emd.dk/mediawiki/index.php/Corine_Land_Cover> | 분류→z0 대응표 | — | 방법론(문서) | — | **가능(방법론)** — 주의: "**CORINE only has two forest roughness lengths, 0.4 and 0.5 m**, while [관측 기반] ORA data represents forest roughness lengths in **6 different bins from 0.5 to 3.0 m**" `[스니펫직접인용]` <https://wes.copernicus.org/preprints/wes-2018-10/wes-2018-10-AC2-supplement.pdf> → **분류표 기반 z0 는 산림에서 심하게 과소평가된다**. GWA 4.0 도 "11 land cover classes 를 z0 로 변환"한다고 명시 <https://globalwindatlas.info/ru/about/dataset> |

### B3.3 Global Wind Atlas — **규칙 R 하 판정: 조건부/사실상 회피 권고**

**사실관계 (확인됨)**:
- 제공 레이어: 평균풍속(10/50/100/150/200 m), 풍력밀도, **RIX(ruggedness index)**, 조도, 표고,
  **GWC(Generalized Wind Climate) 파일** — <https://globalwindatlas.info/download/gis-files>,
  <https://globalwindatlas.info/download/other>, BAMS 논문이 "we have included the ruggedness … RIX layer,
  the generalized wind climate data" 라고 명시 `[스니펫직접인용]`
  <https://orbit.dtu.dk/files/333028656/bams_BAMS_D_21_0075.1.pdf>
- **라이선스: CC BY 4.0** — "The Works are licensed under the Creative Commons Attribution 4.0 International
  license, CC BY 4.0" <https://globalwindatlas.info/about/TermsOfUse> `[스니펫직접인용]` → **상업이용 가능**
- **공개일**: **GWA 4.0 = 2025-06-18** (DTU 공지 <https://wasp.dtu.dk/newsarchive/2025/06/global-wind-atlas-4-0-released>),
  GWA 3.0 = 2019-10. 둘 다 **2026-07-05 이전** ✔
- 해상도: **250 m**, 5개 고도(10/50/100/150/200 m) — <https://globalwindatlas.info/about/method>

**⚠ 결정적 리스크 — GWA 는 ERA5 다운스케일 산물이다**:
- DTU 공식: "The data is created by **first dynamically downscaling ERA5 reanalysis data from 2008-2017 to 3km
  resolution using the WRF mesoscale model**" `[스니펫직접인용]` <https://data.dtu.dk/articles/dataset/Global_Wind_Atlas_4/28955267>
- 이어서 PyWAsP 미규모 모델로 250 m PWC 산출 <https://globalwindatlas.info/about/introduction>
- 검증: "WRF downscaling reduces the mean wind speed bias and spread relative to that of ERA5 **from −1.50±1.30
  to 0.02±0.78 m s⁻¹**" `[스니펫직접인용]`

**판정과 근거**:

| 레이어 | 실질 유래 | 판정 |
|---|---|---|
| GWA **평균풍속 / 풍력밀도 / GWC(윈드로즈·Weibull)** | **ERA5 → WRF → WAsP.** 재분석의 결정론적 함수 | **불가(회피 권고)**. 규칙은 "재분석 산물 금지"다. 기후평균이라 2025 평가기간과 시간적으로 겹치지 않는다는 방어논리는 가능하지만, **"파생물은 원본의 금지를 상속하는가"** 라는 해석 논쟁을 대회 심사에서 우리가 이길 근거가 없다. 그리고 §B4-D 에서 보듯 **어차피 그룹당 상수라 정보량이 0에 가깝다.** 리스크는 실재하고 이득은 구조적으로 0 → **닫는다.** |
| GWA **RIX 레이어** | **DEM 만의 함수** (임계경사 초과 지형 면적비). ERA5 무관 | **원리상 청정하나 조건부.** 다만 GWA 번들로 받으면 출처 기술이 "GWA(ERA5 파생)"이 되어 심사에서 설명 비용이 든다. → **우리가 Copernicus GLO-30 에서 직접 RIX 를 계산하라.** 정의가 공개돼 있고(임계경사 0.3, 섹터별) 계산이 사소하다. |
| GWA **조도/표고 레이어** | ESA WorldCover 계열 + DEM | **직접 원본(WorldCover, GLO-30)을 받아라.** GWA 경유할 이유가 없다. |

**한 줄 판정**: **GWA 는 쓰지 마라.** 얻을 수 있는 청정 성분(RIX, 조도, 표고)은 전부 원본에서 직접 계산 가능하고,
GWA 고유 성분(풍속·GWC)은 재분석 파생 + 그룹당 상수라 **리스크만 있고 이득이 없다.**

### B3.4 한국 관측망 (ASOS/AWS)

- 기상자료개방포털 <https://data.kma.go.kr/> — ASOS 105지점, AWS 약 510지점, OpenAPI 및 파일셋 제공.
- **평가기간(2025) 관측 사용 금지** — 규칙 명시.
- **가능한 유일한 합법 용법**: **학습기간 전용 교사(supervision)**. 즉 2019–2024 의 인근 산악 AWS 관측을
  이용해 "NWP 입력 → 관측풍속" 매핑(지형조건부 편의보정)을 학습하고, **추론시에는 NWP 입력만** 넣는다.
  이건 Winstral 2017(200+ 지점으로 회귀 학습), Wind-Topo(261지점 학습), DEVINE(관측검증)가 전부 쓴 구조다.
- **치명적 한계 3개**: (1) AWS 는 **지상 10 m**, 우리 타깃은 **117 m** — 전이 가정이 필요. (2) AWS 는 능선이
  아닌 곳에 있을 가능성이 높다(설치 접근성). (3) 우리는 이미 2019–2024 **발전량 실측**을 갖고 있어
  직접 감독이 가능하다 — AWS 는 **추가 감독이 아니라 대리 감독**이라 정보이득이 불명확.
- **판정**: **조건부 / 우선순위 낮음.** 이미 직접 타깃이 있는데 대리 타깃을 붙이는 것은 이득이 얇다.

---

## §B4 우리 세팅으로 이식 가능한 후보

### B4.0 모든 후보가 먼저 통과해야 하는 관문 — "정적 피처는 그룹 더미와 구별되지 않는다"

우리 예측 단위는 **그룹 3개(g1, g2, g3)** 이고 터빈 17기가 능선 위 수 km 안에 몰려 있다.
어떤 지형지수 `T` 든 그룹 중심에서 한 번 계산하면 `T(g) ∈ {t1, t2, t3}` 인 **3개 값짜리 상수**다.

> **랭크 논증**: 그룹 원-핫 `[1_{g=1}, 1_{g=2}, 1_{g=3}]` 이 span 하는 공간은 R³ 이다.
> 임의의 그룹상수 벡터 `T(g)` 는 이 공간의 원소다. 따라서 `T` 는 **그룹 더미의 선형결합**이고
> **새 자유도를 0개 추가한다.** 트리 모델에서도 마찬가지로 `T` 로 하는 어떤 분할도 그룹 id 로 재현된다.
> 지형지수를 10개 넣든 100개 넣든 결과는 같다: **정보 0, 과적합 표면만 증가.**

**따라서 모든 후보는 다음 중 하나를 만족해야 한다.**

- **(P1) 시간가변화**: 지형이 **풍향 θ(t) 또는 풍속·안정도**와 상호작용해 `T(g, θ(t))` 형태의 **시계열**이 될 것.
- **(P2) 격자단위 적용**: 그룹(3개)이 아니라 **LDAPS 16격자 / GFS 9격자**(총 25개)에 적용해
  **격자별 시계열**을 만들 것. 이때 자유도는 25개이며 그룹더미(3)로 흡수되지 않는다.
- **(P3) 로터기하 상호작용**: g1/g2(V126, D=126 m)와 g3(U136, D=136 m)의 **로터면 차이**가
  시간가변 연직 프로파일과 곱해질 것.

**(P1) 에 대한 정직한 경고**: `T(g, θ(t))` 는 그룹 3개 × 풍향의 매끄러운 함수다. 그런데 GBM 은 이미
`(group, wind_direction)` 을 갖고 있으므로 **`(group × direction)` 의 임의 함수를 학습할 수 있다.**
따라서 `Sx(g, θ)` 는 **정보를 추가하지 않고, 오직 "매끄러움 사전분포/정규화" 로만 도움이 된다.**
표본이 적은 풍향 구간에서 일반화를 돕는 효과다. 기대이득은 **작다.** 이걸 숨기지 마라.
**(P2) 만이 진짜 새 정보다** — 격자 16개는 그룹 3개보다 자유도가 크고, 격자별 노출은
"이 시각 어떤 격자를 믿을 것인가"라는 **모델이 스스로 표현하기 어려운 조합**을 준다.

---

### 후보 A — `Sx_grid`: 격자별 방향조건부 지형노출 (**최우선**)

- **태그**: `[near_match_only]` — Sx 자체는 Winstral 2017 에서 **직접 지지**되지만(관측지점 대상),
  "NWP 격자점에 Sx 를 계산해 격자 재가중 신호로 쓴다"는 용법은 문헌에서 정확히 일치하는 사례를 못 찾았다.
  가장 가까운 것은 WindMapper 의 "4풍속 × 8풍향 라이브러리 룩업"(Marsh 2023)과
  Lee & Park 2024 의 "상류 예보 정확도 의존" 문제의식이다.
- **계산식**:
  1. DEM = **Copernicus GLO-30**(30 m). 도메인 = 사이트 중심 ±20 km.
  2. 각 LDAPS 격자점 `j = 1..16` 의 위경도에서, 각 방위 `θ ∈ {0°, 10°, ..., 350°}` (36섹터)에 대해
     `Sx_j(θ, d) = mean_{φ ∈ [θ−15°, θ+15°], 5° 간격} max_{r ≤ d} atan[(z(x_j + r·e_φ) − z(x_j)) / r]`
     을 **탐색반경 `d ∈ {500 m, 2000 m, 5000 m}`** 3종으로 사전계산 → **16 × 36 × 3 = 1728개 상수 룩업표**.
     반경 3종 근거: 500 m ≈ 능선 단면, 2000 m ≈ LDAPS 격자 스케일, 5000 m ≈ 단지 전체 상류.
  3. 런타임: 각 시각 t, 격자 j 의 **그 격자 자신의 예보 풍향** `θ_j(t) = atan2(-u_j(t), -v_j(t))`
     (기상학적 "불어오는 방향")로 룩업표를 **선형보간** → 열 이름
     `sx_g{j:02d}_r500`, `sx_g{j:02d}_r2000`, `sx_g{j:02d}_r5000` (48열).
  4. 추가 파생(선택): `sx_ridge_r2000` = 3개 그룹 중심에서 같은 방식으로 계산 → 3열
     (이건 (P1) 정규화 용도로만).
- **왜 시간가변인가**: 룩업표는 상수지만 **인덱스 `θ_j(t)` 가 시간에 따라 변한다.** 따라서 출력은 시계열이다.
  격자 j 마다 지형이 다르므로 **같은 시각에도 격자마다 다른 값**을 갖는다 → 그룹더미로 흡수 불가.
- **기대이득**: 풍속 RMSE **−0.5%~−2%** 수준. 근거: Winstral 2017 이 **원시 NWP 대비** −6% 를 얻었는데
  우리 베이스라인은 이미 25격자 원시장을 다 갖고 있어 공간정보의 대부분을 흡수했다.
  pc-MAE 로는 **−0.3%~−1.5%**. **10% 목표에 단독으로 도달하지 못한다.** 정직하게 말해 보조축이다.
- **비용**: DEM 다운로드(~수십 MB 타일) + Sx 사전계산(순수 numpy, 1728개 광선 스캔 = 수 분).
  `topocalc`(USDA-ARS-NWRC, <https://github.com/USDA-ARS-NWRC/topocalc>) 또는 SAGA 로 검증 가능하나
  직접 구현이 더 간단하다. 학습 시간 증가는 열 48개 추가분(미미).
- **자유도/과적합 위험**: **중간.** 열 48개가 새로 생기지만 **모두 결정론적 상수표 × 예보풍향**이라
  적합 자유도(fitted parameter)는 0이다. 위험은 (a) 반경 3종·섹터폭을 **검증 성능 보고 고르면 선택편의**,
  (b) GBM 이 48열 중 노이즈에 붙는 것. **완화**: 반경·섹터폭을 **사전등록(predeclare)** 하고 절대 튜닝하지 마라.
  섹터폭 30°, 반경 {500, 2000, 5000} 을 **지금 고정**하는 것을 권고.

---

### 후보 B — `upstream_projection`: 풍향투영 상류거리 (**가장 저렴**)

- **태그**: `[near_match_only]` — "upstream grid point selection / flow-dependent grid selection" 의 명시적
  실증비교 논문은 못 찾았다. Lee & Park 2024 가 "complex terrain 에서 **상류 예보 정확도에 의존**"이라고
  문제를 세웠지만 수치 미확보 `[전문미확인]`.
  **주의: 저장소 이력의 "격자 pivot + divergence/vorticity/coherence geometric(pc-MAE −0.4%)" 축과 인접하다.**
  다만 그 축은 **미분연산자**(divergence/vorticity)였고 이건 **선택/가중 신호**다. 형태가 다르나
  **겹칠 위험이 실재하므로 사전에 그 축의 receipt 와 열 이름을 대조하라.**
- **계산식**: 그룹 g 중심 `x_g`, 격자 j 위치 `x_j`, 시각 t 의 그룹평균 단위풍향벡터
  `ê(t) = (u,v)/|(u,v)|` (바람이 **가는** 방향).
  - `upwind_dist_{g}_{j}(t) = −ê(t) · (x_j − x_g)` [km] — **양수면 격자 j 가 상류**
  - `cross_dist_{g}_{j}(t) = |ê(t) × (x_j − x_g)|` [km] — 흐름축에서의 측방 이탈
  - 요약열: `u_upwind_w(t) = Σ_j w_j(t)·speed_j(t) / Σ_j w_j(t)`,
    `w_j(t) = exp(upwind_dist_j(t)/L) · exp(−cross_dist_j(t)²/(2σ²))`, **L = 3 km, σ = 2 km 고정**
  - 열 이름: `upw_dist_g{g}_n{j:02d}`, `upw_wspd_g{g}`, `upw_wspd_ratio_g{g}` (= `upw_wspd_g / wind10_nearest`)
- **왜 시간가변인가**: `ê(t)` 가 시간의 함수. **DEM 이 전혀 필요 없다** — 순수 기하.
- **기대이득**: **−0.2%~−1.0%** pc-MAE. 낮게 잡는 이유: 이미 시도된 grid pivot 축과 정보가 상당히 겹칠 것.
- **비용**: **거의 0.** DEM 불필요, 좌표만 있으면 됨. 수십 줄.
- **자유도**: **낮음.** L, σ 를 고정하면 적합 자유도 0. 요약열만 쓰면 3~6열.
- **권고**: **후보 A 보다 먼저 해보라.** 비용이 압도적으로 싸고, 만약 여기서 신호가 안 나오면
  후보 A(Sx)도 안 나올 가능성이 높다(둘 다 "상류/노출" 개념을 공유). **싼 실패 먼저.**

---

### 후보 C — `dz_x_shear`: 모델표고 결손 × 시간가변 전단 (KMAPP 고도보정의 학습형)

- **태그**: `[directly_supported]` (기법) / `[near_match_only]` (형태) — KMAPP 이 실제로 **LDAPS 에
  "거칠기 보정 + 고도 보정"** 을 적용해 100 m 로 상세화하고, 기상청 자체 평가가 그 성능과 한계를 기술한다.
  금왕호 외 2021, *대기* 31(1):85–100. `[전문미확인]` `[스니펫직접인용]`
- **계산식**:
  1. `z_model(j)` = LDAPS 격자 j 의 **모델 지형고도** (제공되면 그것, 없으면 GLO-30 을 1.5 km 로 평활한 값을 대용).
  2. `z_hub(g)` = GLO-30 에서 읽은 그룹 g 터빈 위치 표고 + 117 m.
  3. `Δz(g, j) = z_hub(g) − z_model(j)` — **정적 상수** (그룹×격자 = 3×16 = 48개 상수).
  4. **시간가변 전단** `α_j(t) = ln(u50_j(t)/u10_j(t)) / ln(50/10)` (LDAPS), 또는
     GFS 로 `α^{GFS}_k(t) = ln(u100_k/u80_k)/ln(100/80)`.
  5. **상호작용 열**: `dz_shear_g{g}_n{j:02d}(t) = Δz(g,j) · α_j(t)`,
     그리고 로그형 `dz_log_g{g}_n{j:02d}(t) = α_j(t) · ln( (z_model(j)+Δz(g,j)) / z_model(j) )`.
  6. 요약: 격자평균 `dz_shear_g{g}(t)` 3열만 써도 된다.
- **왜 시간가변인가**: `Δz` 는 상수지만 `α(t)` 가 시계열. **곱이 시계열이고, Δz 가 그룹×격자마다 달라
  그룹더미로 흡수되지 않는다.**
- **저장소 이력과의 관계**: "alpha shear 대기레짐 파생 → −0.9%" 가 이미 시도됐다.
  **차이점**: 그건 `α(t)` 를 **단독 레짐지표**로 넣은 것이고, 이건 `Δz(g,j) × α(t)` 라는 **상호작용**이다.
  `Δz` 가 그룹·격자마다 부호와 크기가 다르므로 GBM 이 `α` 단독에서 뽑을 수 없는 분할을 만든다.
  **그래도 겹침 위험은 있다 → 반드시 α 단독 축의 receipt 와 성능을 대조하고 증분만 주장하라.**
- **부호 경고 (§B2.3 재수록)**: 기상청 연구는 "LDAPS 가 **산악·협곡에서 풍속 과대모의**"라 하지만
  우리는 **능선**이고 모델표고가 실제보다 **낮을** 것이다. 두 효과가 반대 부호다.
  **부호를 사람이 정하지 말고 모델이 학습하게 하라.** (즉 `Δz·α` 를 그대로 주고 계수를 안 박는다.)
- **기대이득**: **−0.5%~−2%** 풍속 RMSE. KMAPP 자체가 "복잡지형에서 과대보정한다"는 평가를 받는 만큼,
  물리식 그대로가 아니라 **학습형으로 쓰는 것이 필수**.
- **비용**: 낮음. DEM 1회 읽기 + 열 3~48개.
- **자유도**: 낮음~중간. 요약 3열 버전을 먼저 하라.

---

### 후보 D — 정적 지형지수 세트(TPI/TRI/VRM/slope/aspect/curvature/RIX) — **넣지 마라**

- **태그**: `[speculative]` 이자 **구조적으로 무효**.
- **이유**: §B4.0 랭크 논증. 그룹 3개에 대한 상수는 그룹더미와 선형종속. **정보 0.**
- **유일한 예외 2가지**:
  - **(예외 1) 격자단위 (P2)**: `TPI_2000(j)`, `slope(j)`, `curv(j)` 를 **16개 LDAPS 격자에** 붙이면
    16개 값이 되어 그룹더미(3)로 흡수되지 않는다. 하지만 **여전히 시간불변**이라 격자 id 더미와 종속이다
    (격자 id 도 상수다). → **격자단위여도 시간불변이면 무효.** 반드시 θ(t) 와 곱해야 한다 → 그건 후보 A.
  - **(예외 2) 로터기하 (P3)**: 후보 E 참조.
- **RIX 특기**: RIX 는 **섹터별로 정의**되므로 `RIX(g, θ(t))` 로 만들면 (P1) 을 만족한다.
  하지만 RIX 는 "임계경사 초과 면적비"라 **Sx 보다 정보가 거칠다**(부호가 없고 노출/차폐를 구분 못 함).
  **후보 A(Sx)의 열등한 사촌이다. Sx 를 하면 RIX 는 하지 마라.**
- **판정**: **닫는다.** (§B5-1)

---

### 후보 E — `rews_geom`: 로터기하 × 연직프로파일 상호작용

- **태그**: `[directly_supported]` (REWS 는 IEC 61400-12-1 Ed.3 표준 개념, 정량이득 보고 2건)
  / `[near_match_only]` (우리는 100 m 위를 외삽해야 함)
- **계산식**:
  1. 각 시각·격자에서 **2점 로그법칙** 적합: LDAPS `(10 m, 50 m)` → `(u*_L(t), z0_L(t))`;
     GFS `(80 m, 100 m)` → `(u*_G(t), z0_G(t))`. `u(z) = (u*/0.4)·ln(z/z0)`.
  2. 로터면 고도구간 적분(3~5구간 면적가중):
     - g1/g2 (V126, D=126 m, hub 117 m): z ∈ [54, 180] m
     - g3 (U136, D=136 m, hub 117 m): z ∈ [49, 185] m
     `REWS_g(t) = [ Σ_i (A_i/A) · u(z_i, t)^3 ]^{1/3}`
  3. 열: `rews_L_g{g}`, `rews_G_g{g}`, `rews_ratio_g{g}` (= `REWS_g / u(117, t)`),
     `hub_wspd_L_g{g}` (= 로그법칙 117 m 값), `hub_wspd_G_g{g}`.
- **왜 시간가변인가**: 프로파일 `u(z,t)` 가 시계열. **g3 의 로터면이 g1/g2 보다 넓어
  같은 프로파일에서도 `rews_ratio` 가 다르다** → (P3) 충족, 그룹더미로 흡수 안 됨.
  전단이 강한 시각일수록 g1/g2 vs g3 격차가 커지므로 **상호작용이 시간에 따라 변한다.**
- **기대이득**: 문헌 근거는 있으나 우리 조건에서 **−0.5%~−2%**. `[스니펫직접인용]` 이득(REWS +2.86%,
  REWS+lapse 22%)은 **관측 프로파일 3층 이상**을 전제로 한다. 우리는 100 m 위가 외삽이라 상반부가 부정확.
- **비용**: 낮음(닫힌형 수식). **주의: `u50 ≤ u10` 인 시각(역전단)에는 로그법칙이 발산**하므로
  `z0` 클리핑(예: `z0 ∈ [1e-3, 3.0] m`)과 결측 플래그가 필수.
- **자유도**: 낮음. 다만 **`rews_ratio` 는 g1/g2 가 동일값**이므로 실질 자유도는 2 (V126군 vs U136군).
- **부가가치**: `z0_L(t)` 자체가 **"모델이 이 시각에 이 격자를 얼마나 거칠게 보고 있는가"** 를 알려주는
  진단량이다. 이건 정적 토지피복 z0 와 달리 **시간가변**이며, **정적 조도맵을 쓸 필요를 없앤다.**

---

### 후보 F — `pbl_regime`: `hub/PBLH` 상대고도 + 100 m 역전단 플래그

- **태그**: `[near_match_only]` — LLJ 문헌은 코어 40–250 m, hub 100 m 근처에서
  "**고풍속대에서 오히려 발전 감소**"라는 반직관 효과를 보고. `[스니펫직접인용]`
  하지만 우리 층으로 LLJ 코어를 **탐지할 수 없다**(최고 100 m).
- **계산식** (GFS 격자 k):
  - `z_over_zi(t) = 117 / max(PBLH_k(t), 1)` — 그리고 `above_pbl(t) = 1{PBLH_k(t) < 117}`
  - `shear_8010(t) = u100_k(t) − u80_k(t)`; `jet_below_100(t) = 1{shear_8010(t) < 0}`
  - `veer_10_100(t) = angdiff(dir100_k(t), dir10_k(t))` [deg, ±180 wrap]
  - `decouple(t) = above_pbl(t) · (u100_k(t) − u10_k(t))`
- **왜 (P1)/(P2) 를 만족하는가**: 전부 시계열. GFS 9격자에 각각 계산하면 격자별로 다르다.
- **저장소 이력과의 관계**: "theta / bulk Richardson / alpha shear / gust factor → −0.9%" 가 이미 시도됨.
  **차이점**: 그 축은 **PBLH 를 쓰지 않았다**(theta/Ri 는 열역학 파생). `z/z_i` 와 `above_pbl` 은
  **PBL 높이라는 별도 예보량**을 쓴다. `jet_below_100` 도 새롭다.
  **겹침 위험 중간 — 반드시 이전 축의 열 목록과 대조하라.**
- **기대이득**: **−0.3%~−1.5%**. `above_pbl` 은 **드물게 발생하지만 발생하면 큰** 사건이므로
  평균 RMSE 개선보다 **꼬리(고오차 시각) 개선**으로 나타날 가능성이 높다.
  BARAM2026 지표는 `actual < 0.1·cap` 행을 아예 버리므로(기존 확립 측정), **저풍속 안정층 야간이
  얼마나 제외되는지 먼저 확인하라.** 제외되면 이 축의 가치가 크게 깎인다.
- **비용**: 거의 0. 열 5~10개.
- **자유도**: 낮음.

---

### 후보 G — `aws_teacher`: 인근 산악 AWS 를 학습기간 전용 교사로

- **태그**: `[directly_supported]` (Winstral 2017 / Wind-Topo / DEVINE 이 전부 이 구조)
  / 그러나 **우리 세팅에서는 `[speculative]`**
- **형태**: 2019–2024 태백 인근 AWS 10 m 풍속을 타깃으로 "NWP 25격자 + Sx/Δz → 관측풍속" 보정기를 학습하고,
  그 보정기의 **출력만** 발전량 모델의 입력 피처로 쓴다. 추론시 관측 불필요 → 규칙 위반 아님.
- **왜 우선순위가 낮은가**: 우리는 이미 **발전량 실측(직접 타깃)** 을 갖고 있다. AWS 는 **대리 타깃**이고
  고도(10 m vs 117 m)·노출(계곡 vs 능선)이 다르다. **대리 타깃이 직접 타깃보다 나은 신호를 줄 근거가 없다.**
  Winstral/Wind-Topo 가 관측을 쓴 이유는 **타깃이 풍속 자체**였기 때문이다.
- **기대이득**: 불명. **−0% ~ −1%.** 비용(자료취득·정합·별도 모델) 대비 위험.
- **판정**: **보류.** A/B/C/E/F 가 모두 실패한 뒤에만 고려.

---

### B4.8 후보 우선순위 (비용 대비 정보, 오름차순 비용)

| 순위 | 후보 | 비용 | 기대이득(pc-MAE) | 신규자유도 | 이전 축과의 겹침 위험 |
|---|---|---|---|---|---|
| 1 | **B** `upstream_projection` | **거의 0** (DEM 불필요) | −0.2 ~ −1.0% | 0(고정 L,σ) | **높음** (grid pivot 축) |
| 2 | **F** `pbl_regime` | 거의 0 | −0.3 ~ −1.5% | 0 | 중간 (Ri/α 축) |
| 3 | **E** `rews_geom` | 낮음 | −0.5 ~ −2.0% | 0 | 낮음 |
| 4 | **C** `dz_x_shear` | 낮음(DEM 1회) | −0.5 ~ −2.0% | 0 | 중간 (α 축) |
| 5 | **A** `Sx_grid` | 중간(DEM+사전계산) | −0.5 ~ −2.0% | 0(사전등록시) | 낮음 |
| — | **D** 정적 지형지수 | 낮음 | **0 (구조적)** | 무의미 | — |
| — | **G** `aws_teacher` | 높음 | 불명 | 큼 | — |

**총합 상한의 정직한 평가**: A~F 를 전부 성공적으로 넣어도 **효과가 독립이 아니다**(전부 "상류/노출/연직" 이라는
같은 물리를 다른 각도로 코딩). 상관을 감안한 합산 기대이득은 **pc-MAE −1% ~ −4%** 수준으로 보는 것이 타당하다.
**요구되는 −10% 에 이 축 단독으로는 도달하지 못한다.** 이 레인의 결과를 "돌파구"로 보고하지 마라.
"**병목의 일부를 깎는 보조축이며, 그중 후보 B/F 는 30분 안에 검증 가능한 싼 실험**"이 정확한 표현이다.

---

## §B5 닫는 축과 그 이유

1. **정적 지형지수를 그룹 피처로 넣는 축 (TPI/TRI/VRM/slope/aspect/curvature/elevation/RIX 정적판)**
   — **랭크 논증으로 닫는다.** 그룹 3개에 대한 상수는 그룹 원-핫과 선형종속이며 트리 모델에서도
   그룹 id 로 재현 가능한 분할만 만든다. 정보 0, 과적합 표면만 증가. **문헌을 더 뒤져도 바뀌지 않는다.**
   이건 문헌의 문제가 아니라 우리 설계행렬의 문제다.

2. **Global Wind Atlas 레이어(평균풍속/풍력밀도/GWC) 사용**
   — **이중으로 닫는다.** (i) **재분석 파생**: DTU 공식 문서가 "ERA5 를 WRF 로 동역학 다운스케일"이라
   명시(`[스니펫직접인용]`). 규칙의 "재분석 금지"를 파생물이 상속하는지에 대한 해석 논쟁에서 우리가 이길
   근거가 없다. (ii) **정보 0**: 250 m 기후평균은 그룹당 상수 → 항목 1 과 동일한 랭크 문제.
   **리스크는 실재하고 이득은 구조적으로 0.** RIX/조도/표고 성분이 필요하면 GLO-30/WorldCover 에서 직접 계산.

3. **FABDEM**
   — **라이선스로 닫는다.** CC BY-**NC**-SA. 규칙 R 의 "상업이용 가능" 요건 위반. 논쟁 불필요.

4. **WindNinja / CFD 풍장 라이브러리 (WindMapper 식 "4풍속 × 8풍향" 사전계산)**
   — **비용 대비 정보로 닫는다.** (i) 타깃이 그룹 3개뿐이라 라이브러리 출력은 결국
   **3개 그룹의 방향의존 스칼라 곡선 3개**로 축약된다 — 그건 후보 A(Sx)가 100분의 1 비용으로 주는 것과
   **같은 자유도**다. (ii) 물리적 신뢰도가 미묘하다: Wagenbrenner 2016 "does not predict the lee side
   recirculation, and thus the downscaling **does not improve directions on the lee side**",
   Seto 2025 "downscaling **increased negative wind speed biases** at wind-prone lee-slope canyons"
   (평균 +13% 개선과 **동시에** 발생) `[스니펫직접인용]`. **능선은 풍향에 따라 풍하가 되므로
   우리가 겪을 오차가 정확히 그 실패 모드다.** (iii) 대회 일정 안에서 CFD 설정·검증 비용이 비합리적.

5. **딥러닝 다운스케일러(Wind-Topo / DEVINE / GAN)의 직접 이식**
   — **감독신호 부재로 닫는다.** Wind-Topo 는 **261개 관측지점**, Winstral 2017 은 **200+ 지점**,
   DEVINE 은 **ARPS 고해상 시뮬레이션 라이브러리**로 학습했다. 우리는 **2025 평가기간 관측 0개**,
   학습기간에도 **그룹 3개 발전량**이 전부다. 공간 감독이 없는 상태에서 CNN 다운스케일러를 학습할 수 없고,
   사전학습 가중치(Wind-Topo 는 EnviDat/GitLab 에 공개)를 **스위스 알프스 지형·COSMO-1 입력**에 맞춰
   학습한 것을 **태백 능선·LDAPS 입력**에 그대로 적용하는 것은 도메인 전이 근거가 없다.
   (규칙상 사전학습 가중치 자체는 허용이나, **입력 스펙이 COSMO-1 이라 우리 LDAPS 로 급여 불가**.)

6. **LLJ 코어 탐지**
   — **관측가능성으로 닫는다.** LLJ 코어는 문헌상 40–250 m 에 분포(86% 사례)하는데
   우리 최고 층은 **GFS 100 m**다. 프로파일 최댓값을 원리적으로 못 찾는다.
   **대체로 남기는 것**은 `jet_below_100 = 1{u100 < u80}` 라는 **부호 플래그 하나뿐**이며,
   이건 후보 F 에 흡수했다. "LLJ 탐지 피처군"이라는 이름으로 별도 축을 열지 마라.

7. **방향평균 노출지수 (SAGA Wind Exposition Index, 방향평균 Sx)**
   — **정의상 닫는다.** 방향에 대해 평균을 내는 순간 시간가변성이 사라져 항목 1 로 되돌아간다.
   §B1.3(i) 참조: 문헌에서 정량이득이 보고된 지수는 전부 방향조건부다.

8. **정적 조도맵(z0) 을 피처로 넣기**
   — **정보 0 + 정확도 문제로 닫는다.** (i) 그룹/격자당 상수 → 랭크 문제. (ii) 분류표 기반 z0 는
   산림에서 심하게 틀린다: "CORINE only has two forest roughness lengths, **0.4 and 0.5 m**, while
   [관측기반] ORA data represents forest roughness lengths in **6 different bins from 0.5 to 3.0 m**"
   `[스니펫직접인용]`. (iii) **대체재가 더 좋다**: 후보 E 의 2점 로그법칙이 주는 `z0_L(t)` 는
   **시간가변 유효조도**이며 모델이 실제로 쓰고 있는 값이다. 정적 맵보다 우월하다.

9. **한국 ASOS/AWS 관측을 추론 입력으로 사용**
   — **규칙으로 닫는다.** 평가기간(2025) 관측 사용 금지. **학습기간 전용 교사**(후보 G)만
   합법이며, 그건 직접 타깃이 이미 있으므로 우선순위 최하로 보류.

10. **NGII 5 m DEM**
    — **조달 리스크로 (잠정) 닫는다.** 공문 신청·승인 절차가 필요하다는 이용자 증언이 있고 `[전문미확인]`,
    대회 일정 내 확보를 보장할 수 없다. **30 m GLO-30/AW3D30 으로 충분**한 이유: 우리가 필요한 스케일은
    500 m~5 km 상류 지형(Sx)과 1.5 km 격자 표고결손(Δz)이며, 둘 다 30 m 로 충분히 해상된다.
    5 m 가 필요한 유일한 용도는 국지 거칠기인데 그건 항목 8 에서 닫았다.

---

## §B6 이 레인이 남기는 검증 프로토콜 (근본 원칙)

이 레인의 어떤 후보든 채택 전에 다음 3개를 **receipt 에 명시**해야 한다 (AGENTS.md 의 standing rule 확장):

1. **정적/시간가변 선언**: 추가하는 각 열에 대해 "이 열은 그룹 원-핫으로 재현 가능한가?"에 **예/아니오**를 적어라.
   "예"인 열은 넣지 마라. (기계적 검사: `df.groupby('group')[col].nunique().max() == 1` 이면 정적이다.)
2. **겹침 대조**: 후보 B(grid pivot 축), C·F(α/Ri 축)는 **이미 시도된 축의 열 목록과 상관행렬**을 보고하고,
   `max |corr|` 와 **그 축을 제거했을 때의 증분**을 함께 보고하라. 증분 없는 재발굴을 금지한다.
3. **사전등록**: Sx 의 탐색반경/섹터폭, 후보 B 의 L/σ, 후보 E 의 로터구간 수는 **성능을 보기 전에 고정**하고
   receipt 에 해시로 남겨라. 이 값들을 검증점수 보고 고르면 §B4 의 "적합 자유도 0" 주장이 무효가 된다.

---

## §B7 미해결 / 후속 조사가 필요한 항목 (정직한 공백)

| # | 공백 | 왜 못 채웠나 | 다음 수단 |
|---|---|---|---|
| 1 | **MicroMet Ω 의 γ_s, γ_c 값** | 스니펫에 미노출 | Liston & Elder 2006 JHM 원문 Table 확보 |
| 2 | **Winstral 2017 표 4 의 정확한 열 배치** | PDF 표를 못 봄 (2.62→2.46 해석이 추정) | 원문 Table 3/4 확보 |
| 3 | **Lee & Park 2024 (Energy 288:129713) 의 정량 이득** | ScienceDirect 유료, 초록에 수치 없음 | SSRN 프리프린트 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4509803> 전문 |
| 4 | **KMAPP 논문의 정량 RMSE 개선치** | 한국어 스니펫에 정성 결론만 노출 | j-komes PDF 전문 <https://j-komes.or.kr/xml/28740/28740.pdf> |
| 5 | **"NWP 최근접격자 vs 풍향조건부 상류격자" 의 직접 실증비교** | **해당 비교를 정면으로 한 논문을 못 찾음** | 이 축은 사실상 문헌 공백. **후보 B 는 문헌 지지가 얇다는 점을 명시**했다 |
| 6 | **지형 피처의 SHAP/permutation 상대중요도 보고** | 풍속 다운스케일 맥락의 SHAP 보고를 못 찾음. 가장 근접한 것은 Pawlik 2022(풍해, 지형변수 내부 순위) | Bhakare 2024 *Atmosphere* 15:1085(기온 다운스케일 feature importance) 전문 |
| 7 | **Copernicus DEM 라이선스의 상업이용 조항** | 라이선스 PDF 미판독 | <https://docs.sentinel-hub.com/api/latest/static/files/data/dem/resources/license/License-COPDEM-30.pdf> 판독. **AW3D30 은 이미 상업이용 명시 허용이므로 안전한 대체재 존재** |
| 8 | **LDAPS 격자의 모델 지형고도 제공 여부** | 저장소 데이터 명세를 이 레인이 읽지 않음(읽기전용 범위 준수) | **루트 세션이 확인할 것.** 후보 C 의 `z_model(j)` 가 여기 걸려 있다. 없으면 GLO-30 을 1.5 km 평활한 대용값 사용 |
