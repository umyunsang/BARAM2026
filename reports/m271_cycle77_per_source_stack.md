# M271 P4 사이클 77 — NWP 소스별 모델 + 확률 스태킹

노드 `C1N77_PER_SOURCE_STACK` / 레인 L3 / 부모 `C1N68_EMPIRICAL_DECOMPOSITION`

**실제 딥리서치에서 나온 노드다.**

- HEFTCom2024 우승팀 SVK — CatBoost 를 NWP 소스별(DWD/GFS/MEPS) 분리 적합 — <https://arxiv.org/pdf/2505.10367> (`directly_supported`)
- 복수 NWP 소스 결합이 예측오차를 8~30% 감소 — <https://www.sciencedirect.com/science/article/pii/S0360544222027797> (`directly_supported`)
- 소스별 모델 계열 학습 후 출력 스태킹 구성 — <https://pmc.ncbi.nlm.nih.gov/articles/PMC10637996/> (`directly_supported`)

## 1. 팔

| 팔 | 피처 | Total | 1-NMAE | FICR |
|---|---:|---:|---:|---:|
| pooled | 101 | 0.604043 | 0.856870 | 0.351216 |
| gfs | 50 | 0.602321 | 0.856146 | 0.348496 |
| ldaps | 40 | 0.593511 | 0.851137 | 0.335885 |
| stack | - | 0.603122 | 0.856707 | 0.349536 |

두 소스 확률행렬 상관 **0.8550** (sitewind 상관 0.89~0.91 대비)

## 2. 타당성 가드

- V1 POOLED 0.604043 vs 대조군 0.604043 -> **True**
- V2 소스 서로소 + 각 20 개 이상 (gfs 46 / ldaps 36) -> **True**

## 3. 사전확약

- H1 STACK > POOLED -> **False** (-0.000921)
- H2 STACK > 최선단일 -> **True**
- H3 소스 상관 < 0.89 -> **True** (0.8550)
- H4 게이트 통과 -> **False** [----] (3/9 월)
- H5 FICR 우세 -> **False** (FICR -0.000840 / 1-NMAE -0.000081)

## 4. 판정

**FRONT_END_FUSION_IS_NOT_THE_BOTTLENECK**

digest `9dcdd985ed91ba02`
