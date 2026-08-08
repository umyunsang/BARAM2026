# M271 P4 사이클 30 — 잔차가 NWP 의 시간 구조로 설명되는가

- 판정일: 2026-08-04 (UTC)
- 노드: `C1N30_TEMPORAL_SIGNAL` / 레인 L2 / 부모 `C1N27_RESIDUAL_SIGNAL`
- 사이클 27 에서 **피처의 시간 해상도만** 바꿨다 (28 이 공간을 바꿨듯)
- 게이트: `M270_MONTHLY_GATE_v1_frozen_2026-08-04` (읽기만 함) / lockbox 미개봉 / 2024 행 미사용

## 1. 유출 논거

대상일 NWP 는 09:00 KST 단일 초기화가 D+1 01:00~D+2 00:00 을 통째로 준다(A6). 같은 배치 안의 lead 는 가용하고, 배치 경계를 넘는 lag 은 더 오래된 예보이므로 더더욱 가용하다

## 2. 설정

- 변환: lag (1, 3, 6) h / lead (1, 3, 6) h / diff (1, 3) h, **전 컬럼 일률 적용**
- 동시점 65 + lag 195 + lead 195 + diff 130 = **585 피처**
- 행 19,782 (유실 0), 유효행 11,486, 적합 3 회

## 3. 시간 구조가 무언가 더하는가 (H1 · H2)

| 모형 | 피처 | fold-외 R^2 | Pearson |
|---|---:|---:|---:|
| 사이클 27 동시점 격자평균 | 65 | -0.0535 | +0.0699 |
| 사이클 28 동시점 격자별 | 795 | -0.0453 | +0.0647 |
| **사이클 30 시간문맥 격자평균** | 585 | **-0.0322** | +0.0948 |

## 4. 점수 (H3)

| | Total | 1-NMAE | FICR |
|---|---:|---:|---:|
| `M271_MEDIAN4` | 0.636597 | 0.859984 | 0.413210 |
| 잔차 보정 | 0.611736 | 0.859340 | 0.364132 |

차이 **-0.024861**, 게이트 `----` 0/9월 -> **기각**

## 5. 피처 중요도 — 조건부 판정 (H4)

H1 기각이므로 판정하지 않는다 (사이클 28 에서 교정한 설계)

참고: 상위 20 중 시간변환 17 개.

| 순위 | 피처 | 이득 | 시간변환 |
|---:|---|---:|:---:|
| 1 | `gfs__isobaricInhPa_500_v__lag6h` | 2,501 | O |
| 2 | `group_id` | 2,486 | - |
| 3 | `ldaps__heightAboveGround_5_XBLWS__lead3h` | 1,988 | O |
| 4 | `ldaps__heightAboveGround_50_50MUmax` | 1,786 | - |
| 5 | `gfs__isobaricInhPa_700_u__lag6h` | 1,729 | O |
| 6 | `gfs__isobaricInhPa_850_u__lag1h` | 1,695 | O |
| 7 | `gfs__surface_0_gust__lag6h` | 1,645 | O |
| 8 | `gfs__isobaricInhPa_700_u__lead1h` | 1,622 | O |
| 9 | `gfs__isobaricInhPa_500_t__lag6h` | 1,562 | O |
| 10 | `gfs__isobaricInhPa_850_v__diff3h` | 1,533 | O |
| 11 | `gfs__isobaricInhPa_700_v__lag6h` | 1,484 | O |
| 12 | `gfs__isobaricInhPa_700_u__lag3h` | 1,483 | O |
| 13 | `gfs__isobaricInhPa_850_v__lead6h` | 1,472 | O |
| 14 | `gfs__surface_0_gust` | 1,466 | - |
| 15 | `ldaps__heightAboveGround_5_XBLWS__lag6h` | 1,431 | O |
| 16 | `gfs__isobaricInhPa_850_u__lag6h` | 1,410 | O |
| 17 | `gfs__isobaricInhPa_500_t__lead6h` | 1,400 | O |
| 18 | `gfs__isobaricInhPa_500_t__lead3h` | 1,389 | O |
| 19 | `gfs__surface_0_dlwrf__lag6h` | 1,373 | O |
| 20 | `gfs__isobaricInhPa_700_u__lead6h` | 1,358 | O |

## 6. 사전확약 대조

- H1 `시간문맥 모형 fold-외 R^2 > 0.02` -> **False** (실측 -0.0322)
- H2 `시간문맥 R^2 > 동시점 R^2 (-0.0535)` -> **True**
- H3 `보정이 Total 개선 + 동결 게이트 통과` -> **False**
- H4 -> **판정불가**

판정: **SUPPLIED_NWP_CLOSED_IN_SPACE_AND_TIME**

승격 Total **0.636597**, 목표 0.66 까지 **+0.023403**.

## 7. 이것이 확정하는 것

공급 NWP 를 **공간(27·28)으로도 시간(30)으로도** 풀어봤고 세 해상도 모두
fold-외 `R^2` 가 음수다. 챔피언의 잔차는 공급 NWP 로 설명되지 않는다.

따라서 **같은 입력으로 새 기저모델을 만드는 경로가 닫힌다.** 새 모델이
고칠 수 있는 오차가 있었다면 잔차-NWP 구조로 나타났어야 한다.

닫히지 **않는** 것: 공급 밖 정보(외부 공개데이터 — 규칙상 허용), 그리고
라벨 자체의 시계열 구조(단, 평가기간 라벨은 없다).
