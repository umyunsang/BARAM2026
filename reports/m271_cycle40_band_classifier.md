# M271 P4 사이클 40 — 밴드 인지 학습, 분류기 틀 안에서

- 판정일: 2026-08-04 (UTC)
- 노드: `C1N40_BAND_CLASSIFIER` / 레인 L3 / 부모 `C1N39_ARCHITECTURE_GAP`
- 게이트: `M270_MONTHLY_GATE_v1_frozen_2026-08-04` (읽기만 함) / lockbox·`scada_ws` 미사용

## 1. 설계

- 표현: 46 class, width 0.02
- CONTROL 목표: one-hot
- BAND 목표: q_i ∝ u(|c_i - y|), 정산 규칙이 모양을 정한 label smoothing
- 손실: `softmax CE, grad = p - q, hess = max(2p(1-p), 1e-6)`
- 결정: `Bayes: argmax_x E_p[(1/4)(c/cbar)u(|x-c|) - |x-c|]` — **두 팔 동일**
- 피처 87 개, 적합 6 회 (200 rounds x 46 class)

## 2. 타당성 가드 (V1)

CONTROL 0.584468 vs 배포 0.628605 -> **-0.044136**, 허용 `-0.03` -> **False**

## 3. 결과

| 모델 | Total | 1-NMAE | FICR |
|---|---:|---:|---:|
| 배포 `M269@T0.5_G1.5` | 0.628605 | 0.854745 | 0.402464 |
| `CONTROL` (one-hot) | 0.584468 | 0.840843 | 0.328093 |
| **`BAND`** (정산모양 목표) | **0.594240** | 0.843939 | 0.344541 |

BAND - CONTROL = **+0.009772** (FICR 기여 +0.008224 / 1-NMAE 기여 +0.001548)

| fold | CONTROL | BAND |
|---|---:|---:|
| Q2 | 0.580763 | 0.596569 |
| Q3 | 0.571736 | 0.585748 |
| Q4 | 0.591454 | 0.595447 |

BAND 대 배포 게이트: `----` 1/9월 p=0.9980 q05=-0.041202 -> **기각**

## 4. 사전확약 대조

- V1 `CONTROL 이 배포의 -0.03 이내` -> **False** (-0.044136)
- H1 `BAND > CONTROL` -> **판정안함**
- H2 `BAND 가 배포 대비 개선 + 게이트 통과` -> **판정안함**
- H3 `BAND > 0.63031` -> **판정안함**
- H4 `이득이 FICR 쪽` -> **판정안함**

판정: **CLASSIFIER_RECONSTRUCTION_INVALID_AXIS_NOT_JUDGED**

