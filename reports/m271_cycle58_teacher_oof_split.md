# M271 P4 사이클 58 — teacher OOF 를 시간 블록 분할로

- 판정일: 2026-08-05 (UTC)
- 노드: `C1N58_TEACHER_OOF_SPLIT` / 레인 L3 / 부모 `C1N56_MEASURED_POWERCURVE`
- 바뀐 것: **teacher 의 OOF 분할 방식만 (shuffle -> 시간 블록)**
- 게이트: `M270_MONTHLY_GATE_v1_frozen_2026-08-04` (읽기만 함) / lockbox·외부데이터 미사용

## 1. 결함

teacher 가 학습행에 무작위 KFold OOF(누출)를, 테스트행에 최종 모형 예측(정직)을 준다. 학습 때 보는 sitewind 피처가 테스트보다 정확하다.

| 분할 | 학습행 OOF sigma (fold 평균) |
|---|---:|
| `shuffle` (현행) | **1.0923** |
| `blocked` (시간) | **1.6772** |

shuffle 쪽이 낮으면 그만큼 **학습행 피처가 과대정확**했다는 뜻이다.

## 2. 가드

V1 SHUFFLE 0.604043 vs 사이클 44 0.604043 -> 차이 **0.000000** -> **True**

## 3. 결과

| 팔 | Total | 1-NMAE | FICR |
|---|---:|---:|---:|
| 배포 | 0.628605 | 0.854745 | 0.402464 |
| `SHUFFLE` (현행) | 0.604043 | 0.856870 | 0.351216 |
| **`BLOCKED`** | **0.602016** | 0.853900 | 0.350131 |

BLOCKED - SHUFFLE = **-0.002027** (FICR -0.000543 / 1-NMAE -0.001485)

| fold | SHUFFLE | BLOCKED |
|---|---:|---:|
| Q2 | 0.607058 | 0.600319 |
| Q3 | 0.587459 | 0.589806 |
| Q4 | 0.608911 | 0.608065 |

동결 게이트 (부모 SHUFFLE): `----` 4/9월 p=0.7461 q05=-0.008598 -> **기각**

## 4. 사전확약 대조

- V1 -> **True** (0.000000)
- H1 -> **False** (-0.002027)
- H2 -> **False**
- H3 -> **False**
- H4 -> **True**

판정: **FEATURE_DISTRIBUTION_SHIFT_AXIS_CLOSED**

