# M271 P4 사이클 31 — 공간 x 시간 동시 (2x2 마지막 칸)

- 판정일: 2026-08-04 (UTC)
- 노드: `C1N31_SPACE_TIME` / 레인 L2 / 부모 `C1N30_TEMPORAL_SIGNAL`
- 게이트: `M270_MONTHLY_GATE_v1_frozen_2026-08-04` (읽기만 함) / lockbox 미개봉 / 2024 행 미사용

## 1. 2x2 격자

| | 격자평균 | 격자별 |
|---|---:|---:|
| **동시점** | -0.0535 (C27) | -0.0453 (C28) |
| **시간문맥** | -0.0322 (C30) | **-0.0361** (이 노드) |

- 공간 정제 증분 `+0.0082` / 시간 정제 증분 `+0.0213`
- 가법 예측 `-0.0240` / 실측 `-0.0361`
- **상호작용 `-0.0121`**

## 2. 설정

- 피처 **4,770** (동시점 795 + lag 1,590 + lead 1,590 + diff 795)
- 유효행 11,486, 적합 3 회, colsample 0.15, min_child 300
- fold-외 Pearson +0.0835

## 3. 점수 (H4)

| | Total | 1-NMAE | FICR |
|---|---:|---:|---:|
| `M271_MEDIAN4` | 0.636597 | 0.859984 | 0.413210 |
| 잔차 보정 | 0.612765 | 0.858782 | 0.366749 |

차이 **-0.023832**, 게이트 `----` 1/9월 -> **기각**

## 4. 사전확약 대조

- H1 `fold-외 R^2 > 0.02` -> **False** (실측 -0.0361)
- H2 `R^2 > 앞선 최고 (-0.0322)` -> **False**
- H3 `증분이 가법 예측 (0.0295) 초과 — 공간x시간 상호작용` -> **False** (실측 증분 +0.0174)
- H4 `보정이 Total 개선 + 동결 게이트 통과` -> **False**

판정: **SUPPLIED_NWP_CLOSED_ALL_FOUR_CELLS**

승격 Total **0.636597**, 목표 0.66 까지 **+0.023403**.

## 5. 이득 상위 20

| 순위 | 피처 | 이득 | 시간변환 |
|---:|---|---:|:---:|
| 1 | `gfs__isobaricInhPa_850_u__g1__lag3h` | 1,695 | O |
| 2 | `group_id` | 1,459 | - |
| 3 | `gfs__isobaricInhPa_850_u__g5__lag1h` | 894 | O |
| 4 | `gfs__heightAboveGround_10_10v__g6__lead3h` | 720 | O |
| 5 | `gfs__heightAboveGround_80_v__g4__lead3h` | 679 | O |
| 6 | `gfs__heightAboveGround_10_10v__g8__lag3h` | 657 | O |
| 7 | `gfs__heightAboveGround_100_100u__g2` | 625 | - |
| 8 | `gfs__isobaricInhPa_700_u__g2__lag3h` | 599 | O |
| 9 | `gfs__isobaricInhPa_700_u__g1__lag3h` | 597 | O |
| 10 | `gfs__isobaricInhPa_700_u__g1` | 581 | - |
| 11 | `gfs__surface_0_tp__g1__lag1h` | 534 | O |
| 12 | `gfs__surface_0_prate__g8` | 497 | - |
| 13 | `gfs__surface_0_tp__g4__lag1h` | 489 | O |
| 14 | `ldaps__heightAboveGround_5_XBLWS__g3__lead3h` | 472 | O |
| 15 | `gfs__isobaricInhPa_850_u__g5__lag3h` | 468 | O |
| 16 | `gfs__heightAboveGround_2_2d__g3__lag3h` | 467 | O |
| 17 | `gfs__surface_0_tp__g5__lag3h` | 465 | O |
| 18 | `gfs__heightAboveGround_100_100v__g3__lead3h` | 465 | O |
| 19 | `gfs__isobaricInhPa_700_u__g3__lag3h` | 464 | O |
| 20 | `gfs__surface_0_dlwrf__g4__diff1h` | 456 | O |

## 6. 이것이 확정하는 것

공간 x 시간 2x2 **네 칸 모두** fold-외 `R^2` 가 음수다. 챔피언의 잔차는 공급
NWP 의 어떤 해상도 조합으로도 설명되지 않는다.

**같은 입력으로 새 기저모델을 만드는 경로가 닫힌다.** 새 모델이 고칠 수 있는
오차가 있었다면 이 2x2 어딘가에서 잔차 구조로 나타났어야 한다.
