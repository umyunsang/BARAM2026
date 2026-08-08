# M271 C7c — `deep` 의 시간 분할 재확인

노드 `C1N84_TEACHER_CHRONOLOGICAL` / 레인 L6 / 부모 `C1N83_TEACHER_SCALEUP_REJUDGED`

**평가 fold test 행**에서 잰다 — C1N69 반응곡선이 보정된 면이고 모형이 실제로 쓰는 면이다. 학습은 fold 시작 이전 행으로만 하며 **내부 KFold 가 없어 누출 경로 자체가 없다**.

적합 18 회 / test 행 19,783

## 1. sigma_v (test 행)

| 팔 | 전체 | g1 | g2 | g3 |
|---|---:|---:|---:|---:|
| base | **1.5847** | 1.4957 | 1.5947 | 1.6521 |
| deep | **1.5831** | 1.4953 | 1.5894 | 1.6532 |

C1N71 allweather 기준 g1 1.4957 / g2 1.5947 / g3 1.6521

**감소율 +0.10%** -> 환산 Total **+0.000160** / C16 문턱 2.72% (0.004453)

무작위 KFold 면(C1N83)에서는 2.83% 였다.

## 2. 사전확약

- V1 base 가 C1N71 allweather 와 ±0.05 이내 -> **True**
- H1 deep < base -> **True**
- H2 C16 문턱 통과 -> **False**
- H3 세 그룹 모두 개선 -> **False**
- H4 시간 분할 감소가 무작위 KFold 보다 작다 -> **True**

## 3. 판정

**DEEP_HELPS_BUT_BELOW_MAGNITUDE_GATE**

digest `c7bde775cec0bde7`
