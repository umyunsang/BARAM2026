# M271 P4 사이클 27 — 챔피언의 잔차에 NWP 신호가 남아 있는가

- 판정일: 2026-08-04 (UTC)
- 노드: `C1N27_RESIDUAL_SIGNAL` / 레인 L2 / 부모 `C1N20_ALPHA_ENDPOINT`
- 재검 대상: `AXIS_UNUSED_COLUMNS` (A2 의 단변량 MI 폐기)
- **이 세션의 첫 모델 적합.** 진단용 fold-외 적합이며 2024 행·lockbox 미사용
- 게이트: `M270_MONTHLY_GATE_v1_frozen_2026-08-04` (읽기만 함)

## 0. 방법 리서치 (실행 전)

- **Peng, Long & Ding (2005) — mRMR (max-relevance min-redundancy)** (`directly_supported`)
  - 단변량 관련성 스크린은 이미 선택된 변수와 **조건부로만** 정보를 주는 상보적 변수를 놓친다
  - 사용: A2 의 단변량 MI 폐기 전제를 재검할 근거
- **예보 검증의 잔차 진단 (표준 절차)** (`directly_supported`)
  - 잔차에 공변량 구조가 남아 있으면 설명 가능한 오차가 남은 것이고, 없으면 그 공변량으로는 개선이 불가능하다
  - 사용: 표적을 y 가 아니라 챔피언 잔차로 두는 근거

**한계**: 격자평균이므로 공간 세부는 지워진다. 음성 결과는 '격자평균 NWP 컬럼'을 닫는 것이지 '모든 공간 세부'가 아니다

## 1. 설정

- NWP 수치 컬럼 65 개 (그중 `spatial_v2` 선언 24 개)
- 조인 후 19,782 행 (유실 0), 유효행 11,486
- 표적: `residual_rate` / 학습기: LightGBM `regression_l1` leaves 31 / 300 rounds
- 적합 3 회 (fold 당 1 회)

## 2. 잔차에 구조가 있는가 (H1)

| 양 | 값 |
|---|---:|
| fold-외 `R^2` | **-0.0535** |
| fold-외 Pearson | +0.0699 |

## 3. 그 구조가 점수로 바뀌는가 (H2 · H3)

| | Total | 1-NMAE | FICR |
|---|---:|---:|---:|
| `M271_MEDIAN4` (조인된 행 기준) | 0.636597 | 0.859984 | 0.413210 |
| **잔차 보정** | **0.602468** | 0.856751 | 0.348186 |

차이 **-0.034128**. 동결 게이트 `----` 0/9월 p=1.0000 q05=-0.043872 -> **기각**

## 4. A2 전제 재검 — 미사용 컬럼이 결합적으로 기여하는가 (H4)

이득 상위 20 중 `spatial_v2` 미선언 **12 개**.

| 순위 | 피처 | 이득 | v2 선언 |
|---:|---|---:|:---:|
| 1 | `gfs__isobaricInhPa_500_v` | 8,987 | **X** |
| 2 | `gfs__isobaricInhPa_700_u` | 8,189 | **X** |
| 3 | `gfs__isobaricInhPa_850_u` | 8,069 | **X** |
| 4 | `gfs__isobaricInhPa_700_v` | 7,899 | **X** |
| 5 | `group_id` | 7,695 | - |
| 6 | `gfs__isobaricInhPa_500_t` | 7,540 | **X** |
| 7 | `gfs__isobaricInhPa_500_u` | 7,389 | **X** |
| 8 | `gfs__surface_0_tp` | 7,287 | **X** |
| 9 | `ldaps__heightAboveGround_5_XBLWS` | 7,137 | O |
| 10 | `gfs__isobaricInhPa_850_v` | 7,086 | **X** |
| 11 | `gfs__isobaricInhPa_500_gh` | 7,053 | **X** |
| 12 | `ldaps__surface_0_sp` | 6,784 | O |
| 13 | `gfs__heightAboveGround_2_2t` | 6,465 | O |
| 14 | `gfs__isobaricInhPa_850_r` | 6,366 | **X** |
| 15 | `ldaps__heightAboveGround_50_50MUmax` | 5,966 | O |
| 16 | `gfs__surface_0_gust` | 5,888 | O |
| 17 | `ldaps__heightAboveGround_2_r` | 5,847 | O |
| 18 | `gfs__heightAboveGround_10_10u` | 5,786 | O |
| 19 | `gfs__surface_0_dlwrf` | 5,671 | **X** |
| 20 | `gfs__isobaricInhPa_700_t` | 5,604 | **X** |

## 5. 사전확약 대조

- H1 `잔차 모형 fold-외 R^2 > 0.02` -> **False** (실측 -0.0535)
- H2 `보정이 M271_MEDIAN4 대비 Total 개선` -> **False**
- H3 `보정이 동결 게이트 통과` -> **False**
- H4 `이득 상위 20 중 미선언 컬럼 >= 3` -> **True** (실측 12)

판정: **NO_RESIDUAL_SIGNAL_GRIDMEAN_NWP_CLOSED**

승격 Total **0.636597**, 목표 0.66 까지 **+0.023403**.

