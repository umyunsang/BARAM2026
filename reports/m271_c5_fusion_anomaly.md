# M271 C5 — 융합 부호 역전의 기전 (라우터가 지시한 노드)

노드 `C1N80_FUSION_ANOMALY` / 레인 L3 / 부모 `C1N77_PER_SOURCE_STACK`

**내가 고른 노드가 아니다.** 라우터가 C1N77 의 부호 역전을 읽고 C5 로 보냈다 (`voi=0.40`, 표에서 정보가치 최고).

## 1. C1N77 판정 철회

`FRONT_END_FUSION_IS_NOT_THE_BOTTLENECK` 를 철회한다. V2 가 '서로소 분할' 을 요구해 교차·불일치 피처 20 개를 양쪽에서 버렸다. POOLED 101 vs STACK 팔 합계 88 이라 융합 시점이 아니라 정보량을 쟀다.

버려졌던 피처에는 `sitewind__disagreement`, `sitewind__delta`, `geom__align__gfs10_ldaps10__cos` 처럼 **두 소스의 불일치를 재는 신호**가 들어 있었다. 소스 결합의 핵심을 빼고 결합을 시험한 셈이다.

## 2. 방향 리서치

- 모드 간 상관이 클 때 early fusion 우월. 따로 학습하면 피처 간 의존성이 소실된다 — <https://dl.acm.org/doi/pdf/10.1145/3589335.3652504> (`directly_supported`)
- 과적합 위험이 높은 상황에서는 late fusion 과 단순 피처선택이 적합 — <https://www.nature.com/articles/s41698-025-00917-6> (`near_match_only`)

## 3. 교정된 비교 — 정보량을 맞춘다

소스 배타 gfs 44 / ldaps 34 / **공유 9** (두 팔에 모두 준다)

| 팔 | 피처 | Total | 1-NMAE | FICR |
|---|---:|---:|---:|---:|
| pooled | 101 | 0.604043 | 0.856870 | 0.351216 |
| gfs_plus | 67 | 0.599800 | 0.856734 | 0.342866 |
| ldaps_plus | 57 | 0.602237 | 0.856758 | 0.347715 |
| stack_fair | - | 0.601563 | 0.857197 | 0.345929 |

확률행렬 상관 **0.9153** (C1N77 마른 분할 0.8550)

## 4. 사전확약

- V1 POOLED 재현 -> **True**
- V2 두 팔 합집합 = POOLED -> **True**
- V3 각 팔 < POOLED -> **True**
- H1 STACK_FAIR > POOLED -> **False** (-0.002480)
- H2 결합 > 최선단일 -> **False**
- H3 상관이 마른 분할보다 높다 -> **True**
- H4 공유 피처가 정보를 가졌다 (gfs_plus 0.599800 > C1N77 gfs 0.602321) -> **False**

## 5. 판정

**EARLY_FUSION_WINS_MECHANISM_UNCONFIRMED**

digest `6bf1d9de50452e3c`
