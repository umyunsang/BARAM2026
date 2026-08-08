# M271 N4 — 구간 내부 분포 적분

노드 `C1N90_WITHIN_BIN_INTEGRATION` / 레인 L7 / 부모 `C1N68_EMPIRICAL_DECOMPOSITION` / **재학습 없음**

`bayes_decision` 이 각 구간을 중심점의 점질량으로 다룬다. 구간 폭 0.02, FICR 창 ±0.06 이므로 창 경계 근처에서 기대단위를 과대평가해 argmax 를 경계 쪽으로 미는 **체계적 편향**이 생긴다.

## 1. 팔

| 팔 | Total | 1-NMAE | FICR | point 대비 |
|---|---:|---:|---:|---:|
| point | **0.604043** | 0.856870 | 0.351216 | +0.000000 |
| uniform | **0.605933** | 0.856857 | 0.355008 | +0.001890 |
| empirical | **0.605913** | 0.857040 | 0.354787 | +0.001870 |

최선 **uniform** +0.001890 / 검출문턱 0.001013

## 2. 사전확약

- V1 point 가 C1N60 GLOBAL 0.604043 재현 -> **True**
- V3 균등 가중 합 1.0 -> **True**
- H1 uniform > point -> **True**
- H2 empirical > uniform -> **False**
- H3 검출문턱 통과 -> **True**
- H4 FICR 우세 (FICR +0.001896 / 1-NMAE -0.000006) -> **True**
- H5 게이트 통과 -> **False** [-O-O] (5/9 월)

## 3. 판정

**WITHIN_BIN_GAIN_ABOVE_DETECTION_GATE_REJECTS**

digest `263e1bf1d4aead07`
