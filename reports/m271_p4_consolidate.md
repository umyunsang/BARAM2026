# M271 P4 통합 — 발굴 그래프 영속화

- 판정일: 2026-08-06 (UTC)
- 로컬 Total `0.638410` / 목표 `0.66` / 격차 `0.021590`
- 그래프 해시 `7adfaccaa822055b`

## 1. 그래프 상태

- 노드 **136**, 엣지 119
- 프론티어(LIVE) **71**
- 폐기(PRUNED) **65**
- 뒤집힌 전제 없음

## 2. 닫힌 축과 기계검증 전제

폐기는 영구 삭제가 아니라 **전제가 유지되는 동안의 흡수**다. 술어가 거짓이 되면
C9 이 하위그래프를 되살린다.

| 노드 | 폐기 전제 | 되살아나는 조건 |
|---|---|---|
| `AXIS_AVAILABILITY_RECOVERY` | `AVAILABILITY_UNKNOWABLE_AT_FORECAST` | 평가기간 가용성 정보가 생기면 |
| `AXIS_UNUSED_COLUMNS` | `UNUSED_COLUMNS_CARRY_LITTLE` | 미사용 컬럼의 조건부 추가이득이 측정되면 |
| `C1N100_NOVEL_MEMBER_PROBE` | `GBM_REPARAMETERISATION_IS_NOT_DIVERSITY_CORRELATION_FLOOR_0_98` | - |
| `C1N101_PHYSICAL_MEMBER_PROBE` | `SAME_NWP_IMPLIES_CORRELATION_FLOOR_0_91` | - |
| `C1N102_RETRIEVAL_MEMBER` | `SAME_NWP_CORRELATION_FLOOR_CONFIRMED_BY_THREE_PATHS` | - |
| `C1N103_EXTERNAL_NWP_MEMBER` | `EXTERNAL_NWP_ARCHIVE_DOES_NOT_COVER_VALIDATION_SURFACE` | - |
| `C1N10_BLEND_LOCALITY` | `MULTISOURCE_BLEND_GAIN_NEGLIGIBLE` | 블렌딩 이득이 유의미해지면 (진짜 다른 소스) |
| `C1N11_V2_REPRESENTATION` | `COARSER_REPRESENTATION_IS_WORSE` | 성긴 표현이 46-bin 을 이기면 |
| `C1N12_G3_HISTORY` | `HISTORY_LENGTH_DOES_NOT_EXPLAIN_G3` | - |
| `C1N15_WALKFORWARD` | `WALKFORWARD_WEIGHTING_REJECTED` | - |
| `C1N18_SPREAD_SKILL` | `SPREAD_EXPLAINED_BY_PREDICTED_LEVEL` | - |
| `C1N1_TOPRATE_MECHANISM` | `SUPERSEDED_BY_COLLIDER_CORRECTION` | 되살아나지 않음 (방법론적 오류) |
| `C1N21_MOS_BIAS_CORRECTION` | `MOS_SUPERSEDED_BY_ELIGIBLE_POPULATION` | - |
| `C1N22_GLOBAL_SHIFT` | `MOS_SUPERSEDED_BY_ELIGIBLE_POPULATION` | - |
| `C1N23_ELIGIBLE_MOS` | `NO_STABLE_METRIC_ALIGNED_BIAS` | - |
| `C1N24_MEMBER_COUNT_REOPEN` | `MEMBER_COUNT_CLOSED_UNDER_BOTH_COMBINERS` | - |
| `C1N25_ENVELOPE_LAMBDA` | `LAMBDA_UNPREDICTABLE_FROM_CONTEXT` | - |
| `C1N26_POOL_WIDTH_SCREEN` | `POOL_WIDTH_NOT_MONTH_CONSISTENT` | - |
| `C1N27_RESIDUAL_SIGNAL` | `NO_RESIDUAL_SIGNAL_IN_SUPPLIED_NWP` | - |
| `C1N28_PERGRID_SIGNAL` | `NO_RESIDUAL_SIGNAL_IN_SUPPLIED_NWP` | - |
| `C1N30_TEMPORAL_SIGNAL` | `NO_RESIDUAL_SIGNAL_IN_SUPPLIED_NWP` | - |
| `C1N31_SPACE_TIME` | `NO_RESIDUAL_SIGNAL_IN_SUPPLIED_NWP` | - |
| `C1N32_LOCKBOX_OPERATOR` | `OPERATOR_TRANSFER_UNRESOLVED_CONFOUNDED` | - |
| `C1N34_SHRINK_TO_BEST` | `NOTHING_TO_ADD_TO_BEST_SINGLE` | - |
| `C1N36_SOURCE_SENSITIVITY` | `EXTERNAL_NWP_MARGINAL_RHO_UNMEASURED` | - |
| `C1N37_BAND_ALIGNED_LOSS` | `SUPERSEDED_BY_FIXED_IMPLEMENTATION` | - |
| `C1N38_BAND_LOSS_FIXED` | `SUPERSEDED_BY_FIXED_IMPLEMENTATION` | - |
| `C1N3_MASS_AXIS_WIND_SECTOR` | `WIND_SECTOR_LOSS_IS_PROPORTIONAL` | 어떤 섹터의 초과비율이 1 을 넘으면 |
| `C1N43_EFFECT_TREND` | `BAND_TARGET_VANISHES_AT_FRONTIER` | - |
| `C1N44_SHARPENED_DECISION` | `BAND_TARGET_AND_TEMPERATURE_ARE_SUBSTITUTES` | - |
| `C1N45_SUBMISSION_DECISION` | `NO_CANDIDATE_BEATS_SUBMITTED_INCUMBENT` | - |
| `C1N49_RHO_ACCEPTANCE` | `SUPERSEDED_BY_NONNEGATIVE_WEIGHT_FIX` | - |
| `C1N52_EXTERNAL_CLOSURE` | `EXTERNAL_NWP_MEASURED_INSUFFICIENT` | - |
| `C1N53_SUPPLIED_EXTRACTION` | `SUPERSEDED_BY_CHRONOLOGICAL_SPLIT` | - |
| `C1N55_TURBINE_CORRELATION` | `SUPERSEDED_BY_HOURLY_EMPIRICAL_CEILING` | - |
| `C1N56_MEASURED_POWERCURVE` | `BASIS_FUNCTION_IRRELEVANT_TO_GBM` | - |
| `C1N58_TEACHER_OOF_SPLIT` | `TEACHER_FEATURE_NOISE_MONOTONE_LOWER_IS_BETTER` | - |
| `C1N59_PERFECT_PROG_BOUND` | `SCADA_WS_ABSENT_IN_TEST_DIAGNOSTIC_ONLY` | - |
| `C1N5_SPATIAL_INTERPOLATION` | `GFS_INTERPOLATION_JUSTIFIED` | nearest 가 IDW 를 이기면 |
| `C1N60_LEVEL_TEMPERATURE` | `LEVEL_GAIN_CONCENTRATED_IN_TWO_MONTHS` | - |
| `C1N61_GRID_BOUNDARY` | `LEVEL_GAIN_CONCENTRATED_IN_TWO_MONTHS` | - |
| `C1N64_BAND_AVOIDANCE` | `MID_BAND_NOT_AVOIDED_ERROR_SYMMETRIC` | - |
| `C1N67_EXACT_CURVE_PROPAGATION` | `PROPAGATION_APPROXIMATION_SUPERSEDED_BY_EMPIRICAL` | - |
| `C1N6_WIND50_MIDPOINT` | `WIND50_MIDPOINT_NOT_BETTER` | 중점이 10m·max 를 이기면 |
| `C1N70_BASELINE_CORRECTION` | `VOIDED_BY_OWN_GUARD_SUPERSEDED_BY_C71` | - |
| `C1N72_CURVE_BLEND` | `VOIDED_BY_OWN_GUARD_SUPERSEDED_BY_C73` | - |
| `C1N73_GROUP_BLEND_GATE` | `DECISION_LAYER_EFFECTS_NOT_ESTABLISHED_AT_N9` | - |
| `C1N74_BLOCK_LENGTH` | `BOOTSTRAP_BIAS_UNCORRECTED_LONG_BLOCKS_UNREADABLE` | - |
| `C1N75_DAILY_GATE_ESTIMAND` | `VOIDED_BY_OWN_GUARD_SUPERSEDED_BY_C76` | - |
| `C1N77_PER_SOURCE_STACK` | `VERDICT_RETRACTED_INFORMATION_NOT_FUSION` | - |
| `C1N7_POLICY_FROZEN_GATE` | `POLICY_FAMILY_FAILS_FROZEN_GATE` | 어떤 정책이 동결 게이트를 통과하면 |
| `C1N80_FUSION_ANOMALY` | `EARLY_FUSION_CORRECT_FOR_CORRELATED_NWP_SOURCES` | - |
| `C1N82_TEACHER_SCALEUP` | `V1_REFERENCE_MISSPECIFIED_SUPERSEDED_BY_C1N83` | - |
| `C1N84_TEACHER_CHRONOLOGICAL` | `TEACHER_CAPACITY_GAIN_IS_LEAKAGE` | - |
| `C1N87_CURTAILMENT_CLEAN_TARGET` | `G3_SATURATION_IS_NOT_CURTAILMENT` | - |
| `C1N89_ANOMALY_CLEAN_TARGET` | `TARGET_ROW_REMOVAL_LOSES_MORE_THAN_IT_CLEANS` | - |
| `C1N91_WITHIN_BIN_ESTABLISHED` | `WITHIN_BIN_BIAS_EXISTS_BUT_DIRECTION_IS_ROW_DEPENDENT` | - |
| `C1N92_METRIC_ALIGNED_LOSS` | `UTILITY_IN_GRADIENT_BREAKS_PROPRIETY` | - |
| `C1N93_ORDINAL_REPRESENTATION` | `REJUDGED_BY_C1N94` | - |
| `C1N94_CDF_REJUDGE` | `SOFTMAX_BEATS_QUANTILE_CDF_ESTIMATOR_AXIS_CLOSES` | - |
| `C1N95_RECONSTRUCTION_GAP` | `REJUDGED_BY_C1N96` | - |
| `C1N96_GAP_REJUDGE` | `RECONSTRUCTION_GAP_IS_MODEL_NOT_DECISION_RULE` | - |
| `C1N97_BOOSTING_ROUNDS` | `ROUNDS_NOT_THE_GAP_DRIVER` | - |
| `C1N99_ENSEMBLE_EXHAUSTIVE` | `ENSEMBLE_OVER_STORED_ARTIFACTS_CLOSED_NO_DIVERSITY` | - |
| `C1N9_GATE_POWER` | `POLICY_FAMILY_FAILS_FROZEN_GATE` | 어떤 정책이 동결 게이트를 통과하면 |

전제 설명:

- `AVAILABILITY_UNKNOWABLE_AT_FORECAST` — 터빈 가용성은 예보 시점에 알 수 없다 (평가기간 SCADA 부재)
- `BAND_TARGET_AND_TEMPERATURE_ARE_SUBSTITUTES` — V1 가드가 통과한 유일한 실행(사이클 44, CONTROL 0.604043)에서 정산모양 soft target 의 처리효과가 **-0.001015 로 음수**였다. 두 팔이 fold-외로 고른 온도가 T>1(BAND 1.7, CONTROL 2.2) — 즉 **평활**이며, 밴드 목표가 하던 일과 같다. 대조군에 명시적 평활 자유도를 주면 밴드 목표의 암묵적 평활은 중복이 되고 약간 해롭다. 둘은 **대체재**다. 사이클 43 의 3 점 외삽이 이 점을 잔차 -0.000260 으로 예측했으므로 기전 모형도 함께 검증됐다
- `BAND_TARGET_VANISHES_AT_FRONTIER` — 정산모양 soft target 의 처리효과는 대조군 품질과 r=-0.9922 로 반비례한다(3 점: 0.5679->+0.0215, 0.5845->+0.0098, 0.5959->+0.0048). 선형 외삽의 영점이 CONTROL 0.602731 로 배포 품질 0.628605 보다 낮아, 배포 근처에서는 -0.0156 으로 외삽된다. 밴드 목표는 **잘못 지정된 모형을 구제하는 암묵적 정규화**이지 잘 지정된 모형을 개선하는 목적함수 정합이 아니다. **이 폐쇄는 3 점 외삽에 기대며 관측 범위 밖이다** — 배포 품질 대조군에서 양의 효과가 한 번이라도 관측되면 뒤집힌다
- `BASIS_FUNCTION_IRRELEVANT_TO_GBM` — `sitewind__*_powercurve` 를 손으로 쓴 램프(급경사 RMS 0.2679 오차)에서 A5 실측 커브 + IEC 밀도정규화로 교체해도 Total 이 -0.000039 움직인다. GBM 은 `sitewind__mean` 을 분할로 나눠 어떤 단조 커브든 만들 수 있으므로 기저함수가 정보를 더하지 않는다. **트리 기반 모델을 쓰는 한 기저함수 품질은 무관하다** — 선형·신경망 계열로 바꾸면 전제가 뒤집힐 수 있다
- `BOOTSTRAP_BIAS_UNCORRECTED_LONG_BLOCKS_UNREADABLE` — 사이클 74 의 이동블록 부트스트랩은 평균이 관측 델타에 앉지 않는다 — L=30 에서 -0.0025(C73) / -0.0047(C60). Total 이 비선형이라 블록이 길수록 재표집 구성의 분산이 커지고 편향이 커지는 구조적 현상이다. V2 가 그것을 잡았고 결과를 본 뒤 허용범위를 넓히지 않는다. L<=5 구간의 값은 편향 <0.00004 로 신뢰 가능하며 그 부분은 본문에 기록했다. **BCa 등 편향 보정을 넣으면 긴 블록도 읽을 수 있고 그때 뒤집힌다**
- `COARSER_REPRESENTATION_IS_WORSE` — v2 7분위 표현이 배포 46-bin 대비 동일 fold 에서 FICR -0.013928
- `DECISION_LAYER_EFFECTS_NOT_ESTABLISHED_AT_N9` — 결정층 개입 둘 다 pooled Total 에서는 커 보이나(C1N60 +0.008990, C1N73 +0.004931) 월 델타 표준편차가 0.012~0.016 이라 **0 과 구분되지 않는다** (t 0.54~0.93, p 0.38~0.60, 95% CI 가 0 을 포함). **C1N76 이 무편향 원형 블록으로 결론을 확정했다** — n=228 일, 편향 <=0.00007, 폭 평탄 상태에서 두 효과 모두 **L=1 부터 L=30 까지 전 구간에서 0 을 포함**한다(C60 L=7 [-0.001307,+0.006057], C73 L=7 [-0.003207,+0.006217]). pooled 가 커 보였던 것은 **발전량 가중이 개발면 양 끝 두 달에 쏠린 탓**이고 일별 평균은 그 1/3 이다. 세 계측기(월 게이트 n=9, 일별 이동블록, 일별 원형블록)가 일치하므로 **블록길이 선택에 결론이 달려 있지 않다**. **더 넓은 표면이 생기면 뒤집힌다**
- `EARLY_FUSION_CORRECT_FOR_CORRELATED_NWP_SOURCES` — 두 실험이 같은 방향을 낸다 — 마른 분할 -0.000921(C1N77), 정보량 맞춘 분할 **-0.002480**(C1N80). 정보량을 맞출수록 두 팔의 확률행렬 상관이 0.8550 -> 0.9153 으로 올라 앙상블 이득이 사라진다. GFS 와 LDAPS 는 **같은 대기**를 보는 두 관측이라 문헌이 말하는 early fusion 조건에 해당한다(ACM 3589335.3652504). HEFTCom2024 의 DWD/GFS/MEPS 는 서로 다른 모델군이라 상관이 낮고 late fusion 이 통했을 것이며, 문헌의 8~30% 는 우리 소스쌍에 전이되지 않는다. **상관이 낮은 새 소스가 확보되면 뒤집힌다**
- `ENSEMBLE_OVER_STORED_ARTIFACTS_CLOSED_NO_DIVERSITY` — 저장 산출물 12 개의 조합 축이 닫혔다. 크기 2~5 의 2000+ 조합을 **같은 fold 선택**(선택편향 허용 **상한**)으로 훑은 최선이 M115 단독 대비 **+0.000979** 로 검출문턱 0.001013 미달이다. 상한이 문턱 아래이므로 fold-외에서는 더 나쁠 수밖에 없다. 기전은 **후보 다양성 부재** — 12 개가 전부 같은 NWP·피처군·46 구간 틀에서 나왔다. **구조적으로 다른 후보**(다른 표적 표현, 다른 자료원, 다른 시간해상도)가 생기면 뒤집힌다 — 이 전제는 **현 12 후보 집합**에만 걸린다
- `EXTERNAL_NWP_ARCHIVE_DOES_NOT_COVER_VALIDATION_SURFACE` — 외부 NWP 독립 멤버 경로가 **데이터 가용성**으로 막혔다. Open-Meteo previous-runs 아카이브(ecmwf_ifs025·icon_global)가 2023 을 덮지 않는다 — 2023-07·2023-11 프로브가 72 시간 전부 null 을 반환하고 2024-07 은 72/72 정상이다. 검증면 2023 Q2~Q4 와 겹치는 행이 0 이므로 상관도 조합도 잴 수 없다. 규칙·정책 문제가 아니다(예보 아카이브이므로 재분석 금지 조항 무관, 사용자 수집 승인 완료). **2023 을 덮는 다른 예보 아카이브를 찾거나 2024 lockbox 를 열면 뒤집힌다**
- `EXTERNAL_NWP_MARGINAL_RHO_UNMEASURED` — **사이클 47 이 사이클 36 의 전제를 반증했다.** 36 은 요구치를 급경사 **점별** 밴드 적중 기준 0.571 m/s 로 잡았으나, FICR 은 발전량가중 평균이라 평탄 구간에서도 점수를 준다. 사이클 46 이 실측 역산한 요구치는 로컬 Total 0.66 기준 **1.871 m/s**(오차 13.3% 감소), 상위권 FICR 기준 **1.817**(15.8% 감소)로 **3.2 배 관대**하다. 정정 요구치 하에서 현실범위(q 0.8~1.0, rho 0.6~0.8) 최적가중 결합은 로컬 0.66 기준 **세 그룹 모두 실현가능**하고 상위권 FICR 기준은 2/3 이다(g2 가 0.021 m/s 차). 36 의 하한 산식 자체는 옳았다(H4 재현). **남은 미지수는 새 소스의 rho 이며 수집 전에는 알 수 없다**. **사이클 50 이 비음 가중 제약으로 합격선을 확정했다**: 구속 그룹은 g2 이고 q<=0.75 면 rho 무제약, q=0.80 -> rho<=0.680, q=0.85 -> rho<=0.505, q=1.00 -> rho<=0.245. 지배적 요구는 탈상관이 아니라 **소스 품질**이다. 알려진 모델간 오차상관이 0.7~0.85 인 점을 감안하면 q<=0.80(우리 혼합 대비 20% 이상 정확) 이 사실상 필수 조건이다
- `EXTERNAL_NWP_MEASURED_INSUFFICIENT` — **실제 소스를 재서 닫는다**(사이클 51·52). Open-Meteo Previous Runs 로 ICON global 과 ECMWF IFS025 의 100m 풍속을 2024 전년, 리드 48h(previous_day2 — 우리 리드 16~39h 보다 길어 누출 없음), truth=scada_ws 로 측정: ICON 은 q 1.20~1.28 로 우리 혼합보다 **나쁘고**(가중 0), ECMWF 는 q 0.99~1.07 / rho 0.746~0.780 으로 대등하나 결합 감소가 **3.9~6.1%** 에 그친다. 필요량은 13.3%(로컬 0.66) 이므로 **필요량의 30~45%**. 두 소스를 함께 써도(낙관적 상한) 같다. 사이클 36 은 잘못된 요구치로 닫았고 46~47 이 반증했으나, 이번엔 측정이다
- `G3_SATURATION_IS_NOT_CURTAILMENT` — 사이클 87 이 감발 정제를 시험해 **기전을 반증했다**. g3 의 theta 초과(0.775 대 0.5)와 UNISON 포화비 0.89~0.95 가 감발을 가리킨다고 봤으나, 감발 판정 행을 teacher 학습에서 빼자 **g3 가 가장 나빠졌다**(-2.16%). 그 구간은 정상 운전이었고 빼면서 고풍속 학습 영역이 얇아진 것이다. UNISON 의 낮은 포화비는 정격 4.2MW 를 실제로 내지 못하는 **기기 특성**으로 보는 것이 자료와 맞는다. **운전로그가 확보되면 뒤집힌다** — 통계적 대체가 아니라 실제 감발 기록으로 판정하면 다른 구간이 잡힐 수 있다
- `GBM_REPARAMETERISATION_IS_NOT_DIVERSITY_CORRELATION_FLOOR_0_98` — GBM 계열 내부 재모수화는 앙상블 다양성을 만들지 못한다. 표본가중·피처 스크린·결정정책을 전부 바꾼 후보의 M115 대비 오차 상관이 **0.9767** 이고, 기존 GBM 멤버들도 0.9811/0.9828 이다. 유일한 예외가 아날로그 계보 M244 의 **0.8436** 이며, C1N99 의 최선 조합이 M244 를 포함한 이유가 바로 그것이다(약체 타이브레이커가 아니라 **유일한 탈상관 멤버**). 조합 축은 M115 와의 오차 상관 **<=0.85** 후보가 생겨야 열리고, 그것은 방법군 자체가 달라야 한다 — **아날로그/최근접이웃, 물리 직접계산, 다른 표적공간**. 이 전제는 **GBM 재모수화**에만 걸린다
- `GFS_INTERPOLATION_JUSTIFIED` — GFS IDW 가 nearest 보다 낫다 (3/3 그룹)
- `HISTORY_LENGTH_DOES_NOT_EXPLAIN_G3` — g3 FICR 열세가 이력 3->9 개월에 걸쳐 단조 감소하지 않는다 (부족분 0.134 / 0.053 / 0.093)
- `LAMBDA_UNPREDICTABLE_FROM_CONTEXT` — 봉투 오라클 0.742380 은 Breiman 비음제약 스태킹의 천장이지만, 봉투 안 위치 lambda 의 층 순위가 fold 간 Spearman -0.0756 (문턱 +0.50). 층간 범위 0.78 은 커 보여도 재현되지 않는 잡음이다. lambda* 는 실현된 NWP 오차가 결정하므로 예보시점에 알 수 없다
- `LEVEL_GAIN_CONCENTRATED_IN_TWO_MONTHS` — 사이클 60 의 +0.008990 은 9 개월 중 **2 개월**(2023-12 +0.0410, 2023-04 +0.0216)이 만든다. 최대 1 개월을 빼면 월평균이 +0.004985 -> +0.000479 로 90% 사라지고 2 개월을 빼면 -0.002545 로 **부호가 뒤집힌다**. C1N62 가 사전지정 조절변수 셋(수준2 질량·위양성률·평균 rate)으로 어느 달이 발화하는지 설명하려 했으나 전부 실패했으므로, 발화 조건을 모르는 채로는 배포할 수 없다. **발화 조건을 설명하는 사전지정 조절변수가 나오면 뒤집힌다** — 사후 탐색은 n=9 에 대한 낚시이므로 부활 근거가 되지 못한다
- `MEMBER_COUNT_CLOSED_UNDER_BOTH_COMBINERS` — median 12개 0.633166 < median 4개 0.636597 이고 mean 도 같은 방향. 사이클 13 의 폐기가 결합자에 매인 판정이 아니었음이 확인됐다
- `MID_BAND_NOT_AVOIDED_ERROR_SYMMETRIC` — 사이클 64 가 중간대 비켜감을 기각했다. 주변분포 비가 1.148~1.376 으로 예측이 중간대를 **더 자주** 쓰고, 실제-중간대 행의 오차 부호가 대칭이다(g2 과소 0.412). 결정층 인공물도 방향성 편향도 아니므로 이동 보정 계열이 닫힌다. **비대칭 오차나 1 미만 주변비가 관측되면 뒤집힌다**
- `MOS_SUPERSEDED_BY_ELIGIBLE_POPULATION` — 추정 모집단(전체행)이 평가 모집단(유효행 58.1%)과 달랐다. C1N23 이 대체한다
- `MULTISOURCE_BLEND_GAIN_NEGLIGIBLE` — GFS+LDAPS 블렌딩의 실측 오차감소가 0.10~0.37% 이고 고초과 셀 국소성은 1.1 sigma. 정정된 프레임(평준화)에서도 외부 NWP 폐쇄가 유지된다
- `NOTHING_TO_ADD_TO_BEST_SINGLE` — M115 를 닻으로 3 종 상대 x 5 단 b 사다리 15 조합이 전부 기각됐다. 최선이 +0.000277 로 검출문턱 미달이고 게이트·fold 재현도 실패. Breiman 의 '결합은 최고 단일보다 낫다' 가 이 후보군에서는 성립하지 않는다
- `NO_CANDIDATE_BEATS_SUBMITTED_INCUMBENT` — 새 제출은 배포 기준선이 아니라 **이미 제출된 것**을 이겨야 한다. 현직 M261 로컬 0.629973(온라인 0.636527) 기준으로, 검출문턱 +0.001013 을 넘는 후보 2 개는 동결 게이트를 통과하지 못하고(더 쉬운 배포 기준선 상대로도 기각), 게이트를 통과한 M115@T0.6_G0.2 는 현직 대비 +0.000337 로 문턱의 33% 다. 오프셋 산포(+0.0066~+0.0211)는 그 차이의 20 배가 넘는다
- `NO_RESIDUAL_SIGNAL_IN_SUPPLIED_NWP` — 챔피언 잔차를 공급 NWP 로 설명할 수 없다. 공간x시간 2x2 **네 칸 전부** fold-외 R^2 가 음수다: 동시점x격자평균 -0.0535, 동시점x격자별 -0.0453, 시간문맥x격자평균 -0.0322, 시간x격자별 -0.0361(상호작용 -0.0121 열가법). 전부 **잔차 평균을 찍는 것보다 못하다**(문턱 +0.02). 피처 중요도가 미선언 컬럼에 몰리지만 일반화하지 못하는 모형의 중요도는 정보의 증거가 아니므로 A2 전제는 뒤집히지 않는다
- `NO_STABLE_METRIC_ALIGNED_BIAS` — 유효행 기준 지표정합 최적 이동의 fold 간 부호일치가 0.333 (문턱 0.70). 제거할 안정된 편향이 존재하지 않는다 — median 잔차 -0.04 는 왜도의 산물
- `OPERATOR_TRANSFER_UNRESOLVED_CONFOUNDED` — lockbox 3 개는 품질 스프레드 0.037 에 의도적 약체 대조군을 포함해 **결합 자체가 해로운 배치**였다(mean 도 최고단일에 -0.0116 패). median vs mean 을 가릴 검정력이 없었으므로 연산자 전이는 확인도 반증도 되지 않았다. 사전확약에 멤버 품질 동질성을 걸지 않은 것이 설계 결함이다
- `POLICY_FAMILY_FAILS_FROZEN_GATE` — (T,G) 정책 62개가 동결 게이트를 넘지 못한다. 게이트 검출력은 +0.001 로 확인됨
- `POOL_WIDTH_NOT_MONTH_CONSISTENT` — Q3 124 멤버 median 이 4 멤버 대비 pooled +0.001778 이나 월별로 2/3 만 양수 (2023-09 은 -0.001231). 봉투 커버리지는 0.21 -> 0.68 로 3 배 넓어지지만 C1N25 가 위치를 못 찾으므로 넓은 봉투가 쓸모없다
- `PROPAGATION_APPROXIMATION_SUPERSEDED_BY_EMPIRICAL` — 사이클 65~67 의 전파 모형(가우스 eps_v, 구간중심 평가, 풍속 조건화)은 관측이 **실제 출력 대역**으로 조건화된 것과 어긋났고, 무엇보다 재현하려던 풍속-only 기계 자체를 저출력대에서 과소평가했다. C1N68 이 같은 행에서 그 기계를 **경험적으로** 만들어 대체한다. 이분산 발견(C1N66)은 살아남는다
- `RECONSTRUCTION_GAP_IS_MODEL_NOT_DECISION_RULE` — 배포(0.628605)와 재구성(0.604043)의 격차 0.024562 는 **결정규칙이 아니라 모형**에 있다 — 결정규칙 기여 **-0.008475**, 모형 기여 **+0.033037**. 배포 정책을 우리 확률에 대입하면 더 나빠지고, 63 정책 **같은-fold 최적(상한)조차 0.602668** 이다. 기전: fold-외 선택 온도가 우리 쪽 **T>1(평활)** 대 배포 **T=0.5(예리화)** 로 반대이므로 우리 확률행렬은 **과확신**, 배포는 **과소확신**이다. 결정층 손잡이의 반대편에 있다. 따라서 재구성 위에서 잰 모든 결정층 효과는 배포 프레임으로 이전된다고 볼 수 없다. **배포 확률행렬을 재현해 A0 가 0.6286 근처에 오면 뒤집힌다** — 그때 결정층 축이 배포 품질에서 다시 열린다
- `REJUDGED_BY_C1N94` — C1N93 은 자기 V2 가드로 VOID 다 — 사양 `재배열 후 교차 0 건` 을 코드가 재배열 **전** 원시 행렬로 쟀다. 결과를 본 뒤 사양을 넓히지 않고 C1N94 가 교정 계측기와 새 사전확약으로 재판정했다. 산출물(팔 점수)은 재배열이 이미 적용된 값이라 유효하며 C1N94 가 재계산 없이 승계했다. **영구 대체**
- `REJUDGED_BY_C1N96` — C1N95 는 자기 V1·V3 가드로 VOID 다 — 대조군을 평온도 `bayes_decision` 으로 놓아 C1N60 의 온도 개입 0.008124 를 뺀 채 대조군이라 불렀고, 항등식도 GAP 를 상수로 고정해 성립할 수 없었다. C1N96 이 교정 대조군과 새 사전확약으로 재판정했다. 배포 결정규칙 판독(`T1_G0.5435` 대 `T0.5_G1.5`, 행동격자·그룹별 정규화)은 유효하며 C1N96 이 승계했다. **영구 대체**
- `ROUNDS_NOT_THE_GAP_DRIVER` — 부스팅 라운드는 재구성 격차 0.024562 의 최대 **9%** 만 설명한다. 과확신 기전 자체는 **확인됐다** — 라운드가 줄면 fold-외 선택 온도가 단조로 내려간다(2.2 -> 1.7 -> 1.3). 그러나 최선 이득이 +0.002263 이고 라운드·온도를 함께 fold-외로 고르면 **-0.000497** 로 떨어진다. 라운드 곡선이 비단조라 선택이 fold 를 건너 이전되지 않는다. **배포 산출기의 표본가중(`clip(rate,0.10)**1.0`)이나 피처 목록(87/100 공통)이 F1 을 넘으면 격차 설명이 갱신된다** — 이 전제는 **라운드 단독**에만 걸린다
- `SAME_NWP_CORRELATION_FLOOR_CONFIRMED_BY_THREE_PATHS` — 같은 NWP 위의 조합 축이 닫혔다. M115 대비 오차 상관 바닥 0.91 이 **세 독립 경로**로 확인됐다 — GBM 재모수화 0.9767(C1N100), 순수 물리 사상 0.9097(C1N101), 검색 기반 AnEn 0.9150(C1N102). 방법론을 회귀->물리->검색으로 완전히 갈아도 풍속 예보 오차가 모든 경로의 지배적 오차원이라 오차가 함께 움직인다. **남는 경로는 둘뿐이다** — (1) M244 의 검색 키를 복원해 0.8436 을 재현하는 것(receipt 가 비어 현재 불가), (2) **다른 NWP** 를 도입하는 것(C1N70 은 ECMWF 를 teacher 혼합으로만 쟀고 **독립 앙상블 멤버로는 미판정**이다). 둘 중 하나가 상관 <=0.85 이면서 품질 >0.6058 인 멤버를 내면 뒤집힌다
- `SAME_NWP_IMPLIES_CORRELATION_FLOOR_0_91` — 같은 NWP 를 소비하는 한 M115 대비 오차 상관의 바닥은 **0.91** 이다. GBM 을 전혀 안 거치는 순수 물리 사상(실측커브 ∘ teacher 풍속)조차 0.9097 이다. 풍속 예보 오차가 모든 경로의 지배적 오차원이라 하류 방법론을 아무리 바꿔도 오차가 같이 움직인다. 예외는 아날로그 M244 의 **0.8436** 이며, 그것은 NWP 를 회귀 입력이 아니라 **검색 키**로 쓰기 때문이다. **검색 기반 후보를 강화하거나 다른 NWP 를 도입해 상관 <=0.85 이면서 품질이 M244(0.605760)보다 높은 멤버가 생기면 뒤집힌다**
- `SCADA_WS_ABSENT_IN_TEST_DIAGNOSTIC_ONLY` — `scada_ws` 는 관측 나셀풍속으로 학습기간에만 있다(C1N39 가 확정). 이 노드는 상한 진단이지 후보가 아니다. **평가기간 SCADA 가 공급되면 뒤집힌다**
- `SOFTMAX_BEATS_QUANTILE_CDF_ESTIMATOR_AXIS_CLOSES` — 분포를 **어떻게 추정하는가** 축이 닫혔다. 결정층을 고정한 채 표현만 바꿨을 때 46-class softmax 가 분위수회귀 CDF 를 **-0.007176** 로 이기고, 표적 재형성도 **-0.001015**(C1N44) 로 진다. 세 fold 전부에서 부호가 같으므로 pooled 우연이 아니다. 기전: 분위 보간은 **구간 내부 균등밀도**를 가정하는데 FICR 은 +-6% 창 안의 **질량 집중도**로 결정되므로 그 평탄화가 곧 손실이다 — 차이가 FICR 쪽에 몰린 것(-0.003524 대 -0.000170)이 이를 확인한다. **분위 해상도를 크게 높이거나(46 빈보다 촘촘한 CDF) 구간 내부 밀도를 모수화한 표현이 F1 을 넘으면 뒤집힌다** — 이 전제는 **균등밀도 보간**에 걸린다
- `SPREAD_EXPLAINED_BY_PREDICTED_LEVEL` — 멤버 스프레드-히트율 관계는 주변적으로 23%p 단조지만 예측대역x그룹 통제 후 잔여가 4.9%p 로 사전확약 10%p 에 미달. 결정층 오라클 8.8% x 일반화 16.5% = Total 0.00045 로 게이트 검출문턱 0.001013 보다도 작다
- `SUPERSEDED_BY_CHRONOLOGICAL_SPLIT` — 사이클 53 은 teach() 의 KFold(3, shuffle=True) 를 그대로 써서 시간 인접 누출이 들어갔다(라벨 lag-1 자기상관 0.951~0.962). C1N54 가 시간분할로 교정했고 결론 자체는 생존한다
- `SUPERSEDED_BY_COLLIDER_CORRECTION` — y 를 통제한 측정이라 collider 편향에 걸렸다. C1N2 가 대체한다
- `SUPERSEDED_BY_FIXED_IMPLEMENTATION` — 사이클 37·38 은 구현 결함(학습 모집단 유효행 한정, 헤시안 상수화, 그리고 0.087 뒤처진 아키텍처 위에 손실을 얹은 오배치)으로 무효다. C1N39 가 원인을 특정했고 **손실 축 자체는 여전히 미판정**이다
- `SUPERSEDED_BY_HOURLY_EMPIRICAL_CEILING` — 사이클 55 는 SCADA 10 분 해상도로 산포를 쟀는데 라벨은 **시간** 단위다. 10 분 잔차 6 개를 합치면 평활되므로 과대추정이었다. 정규 가정도 썼는데 잔차 분포는 정지·포화 때문에 정규가 아니다. C1N57 이 시간 집계 + 경험적 분포 + 발전량 가중 전 구간으로 재계산했고 **결론이 뒤집혔다** — 천장 0.7488 로 로컬 0.66 요구 0.4459 를 크게 넘어 출력 산포는 병목이 아니다
- `SUPERSEDED_BY_NONNEGATIVE_WEIGHT_FIX` — 사이클 49 는 rho 를 0.99 까지 스캔했는데 사이클 36 의 비제약 최적가중식이 rho > q 에서 음수 가중(외삽)을 내어 수용 영역이 비단조로 오염됐다. C1N50 이 비음 제약으로 대체한다
- `TARGET_ROW_REMOVAL_LOSES_MORE_THAN_IT_CLEANS` — teacher 학습 표적에서 행을 빼는 축이 **두 경로로 닫혔다** — 손으로 쓴 물리 규칙(C1N87 감발, -0.73%)과 비지도 이상 탐지(C1N89, -4.99%) 모두 음수다. 잔차가 큰 행은 오염된 것이 아니라 **예측하기 어려운** 것이고, 빼면 teacher 가 그 영역을 못 배운 채 여전히 전 행을 예측해야 한다. C1N88 이 잰 상한(11.33%)은 '완벽히 처리했을 때' 이고 행 제거는 그 처리가 아니다. **행을 빼지 않고 다루는 방법(가중치 하향, 견고 손실, 이상 표시를 피처로)이 F1 을 넘으면 뒤집힌다** — 이 전제는 **제거**에만 걸린다
- `TEACHER_CAPACITY_GAIN_IS_LEAKAGE` — `deep`(255 잎)의 sigma_v 감소는 무작위 KFold 면에서 2.83% 였으나 시간 분할 test 행에서는 **+0.10%** 다 — **이득의 96% 가 시간 인접 누출**이었다(C1N54 가 잰 누출분 17.8~21.3%p 와 정합). 환산 Total +0.000160 으로 C16 문턱 0.004453 에 크게 미달하고 g3 는 오히려 악화한다. 잎을 늘리면 이웃 시각을 더 잘 외울 뿐 실제 미래 구간에서는 거의 아무것도 못 한다. **시간 분할에서 문턱을 넘는 용량·모델군 조합이 나오면 뒤집힌다**
- `TEACHER_FEATURE_NOISE_MONOTONE_LOWER_IS_BETTER` — 사이클 58 이 teacher OOF 분할을 두 점에서 쟀다 — shuffle sigma 1.0923 -> 0.604043, blocked-3 sigma 1.6772 -> 0.602016. 두 점이 test sigma(1.37~1.50)를 **끼우는데** 점수는 sigma 에 단조 감소한다. 따라서 '학습행 피처잡음을 평가행에 일치시키면 낫다'는 가설이 반증되고, 중간 블록수로 sigma 를 맞추는 재시험은 내삽으로 ~0.603 을 예측하므로 불필요하다. **비단조 반례가 나오면 뒤집힌다** — 예를 들어 어떤 분할이 shuffle 을 넘으면 관계가 sigma 단독이 아닌 것이다
- `UNUSED_COLUMNS_CARRY_LITTLE` — 진짜 미사용 NWP 컬럼의 평균 MI 가 선언 컬럼의 0.41배
- `UTILITY_IN_GRADIENT_BREAKS_PROPRIETY` — 결정층의 기대효용을 학습 **기울기**에 섞으면 단조로 나빠진다 — alpha 0.8 에서 -0.018203, 0.5 에서 -0.049992. 대조군 0.604043 정확 재현, 기울기 규모 비 [1.0, 4.178, 2.271] 로 C1N37 의 D2 폭주가 아니므로 **구현 결함이 아니라 실재 효과**다. 기전: 교차엔트로피는 **적정 점수규칙**이라 최적해가 참 조건부확률인데 `-E_util` 은 `p` 에 의존하지 않는 **상수 기울기**라 적정성을 깬다. 결정층은 확률의 교정도에 전적으로 의존하므로 그 입력을 망가뜨린다. **적정성을 유지하는 지표정합 대리손실**(효용을 표적 분포로 넣거나 적정 대리를 유도하는 것)이 F1 을 넘으면 뒤집힌다 — 이 전제는 **기울기 혼합**에만 걸린다
- `V1_REFERENCE_MISSPECIFIED_SUPERSEDED_BY_C1N83` — 사이클 82 는 V1 이 발화해 스스로 무효다. 기준을 C1N66 의 **평가 fold test 행** 잔차(1.5866)로 잡았는데 이 노드가 재는 것은 **학습행 무작위 KFold OOF**(1.1037)다. 다른 양이다. 결과를 본 뒤 허용범위를 넓히지 않고 C1N83 이 올바른 기준(C1N58 shuffle 1.0923)으로 같은 산출물을 재판정한다
- `VERDICT_RETRACTED_INFORMATION_NOT_FUSION` — 사이클 77 의 판정 `FRONT_END_FUSION_IS_NOT_THE_BOTTLENECK` 을 철회한다. V2 가 '두 소스 피처가 서로소' 를 요구했고, 그것을 만족시키려고 어느 접두사에도 안 걸리는 20 개를 **양쪽에서 다 버렸다** — 거기에 `sitewind__disagreement`, `sitewind__delta`, `geom__align__gfs10_ldaps10__cos` 처럼 **두 소스의 불일치를 재는 신호**가 들어 있었다. 결합의 핵심을 빼고 결합을 시험한 셈이다. C1N80 이 공유 피처를 두 팔에 모두 주어 정보량을 맞춘 비교로 다시 세운다
- `VOIDED_BY_OWN_GUARD_SUPERSEDED_BY_C71` — 사이클 70 은 사전확약 V2 가 발화해 스스로 무효다. 결과를 본 뒤 허용범위를 넓히지 않는다 — 그 가드가 없었으면 `sitewind__mean` 이 최적가중이 아니라는 것을 못 봤다. 산술 자체(교정 q 1.27~1.39, 감소 0.00~0.18%)는 참고값으로 남기되 판정 근거로 쓰지 않고, C1N71 이 별도 사전확약으로 다시 세운다
- `VOIDED_BY_OWN_GUARD_SUPERSEDED_BY_C73` — 사이클 72 는 V2 가 발화해 스스로 무효다. 참조 상수를 시뮬레이션 팔에서 가져온 것이 원인이고, 결과를 본 뒤 상수를 고치지 않는다. 팔 계산 자체는 옳았으므로 C1N73 이 올바른 참조로 다시 세운다
- `VOIDED_BY_OWN_GUARD_SUPERSEDED_BY_C76` — 사이클 75 는 V1 이 발화해 스스로 무효다. 추정량을 선형으로 바꿔도 편향이 남았으므로 C1N74 의 진단이 배제됐고, 그 배제가 C1N76 을 진짜 원인(가장자리 과소표집)으로 이끌었다. 결과를 본 뒤 허용범위를 넓히지 않는다
- `WALKFORWARD_WEIGHTING_REJECTED` — walk-forward MSE 가중이 등가중(E1 0.631934)보다 낮고 동결 게이트를 통과하지 못한다
- `WIND50_MIDPOINT_NOT_BETTER` — 50m 중점이 10m·max 보다 나셀 상관이 낮다 (3/3 그룹)
- `WIND_SECTOR_LOSS_IS_PROPORTIONAL` — 풍향 섹터별 손실이 발전량에 비례한다 (240도 초과비율 0.968)
- `WITHIN_BIN_BIAS_EXISTS_BUT_DIRECTION_IS_ROW_DEPENDENT` — 46 구간 점질량이 FICR 창 경계에서 기대단위를 과대평가하는 것은 **실재한다** — 적분 교정이 pooled +0.001866 을 낸다. 그러나 일별 평균은 +0.000011 이고 양수 일이 45.6% 로 과반 미만이라 **0 과 구분되지 않는다**. 편향의 존재는 맞고 방향이 행마다 다르다. 구간 폭을 줄이거나(재학습 필요) 손실을 연속화하면 다르게 나올 수 있으나 **적분 교정만으로는 안 된다**. **더 넓은 검증면이나 세밀한 구간 표현이 생기면 뒤집힌다**

## 3. 결손 원장

- 셀 108, 미설명 질량 0.371395
- **전 셀 회수가능질량 0.04875** (격차 0.02159 의 2.3배)

회수가능질량 = 평균 미달 셀들이 현재 평균 수준까지 올라갔을 때 사라지는 손실.
어디서도 현재 평균을 능가할 필요가 없다.

| 셀 | 손실 | 초과비율 | 회수가능 |
|---|---:|---:|---:|
| `group_id=3|month=2023-12|y_band=(0.45, 0.7]` | 0.00891 | 1.364 | **0.00238** |
| `group_id=2|month=2023-11|y_band=(0.45, 0.7]` | 0.00792 | 1.422 | **0.00235** |
| `group_id=3|month=2023-07|y_band=(0.7, 1.1]` | 0.00831 | 1.366 | **0.00223** |
| `group_id=3|month=2023-11|y_band=(0.45, 0.7]` | 0.00843 | 1.293 | **0.00191** |
| `group_id=3|month=2023-10|y_band=(0.25, 0.45]` | 0.00467 | 1.397 | **0.00133** |
| `group_id=3|month=2023-04|y_band=(0.45, 0.7]` | 0.00703 | 1.233 | **0.00133** |
| `group_id=2|month=2023-10|y_band=(0.45, 0.7]` | 0.00587 | 1.290 | **0.00132** |
| `group_id=1|month=2023-12|y_band=(0.25, 0.45]` | 0.00414 | 1.460 | **0.00131** |
| `group_id=3|month=2023-12|y_band=(0.25, 0.45]` | 0.00387 | 1.499 | **0.00129** |
| `group_id=3|month=2023-07|y_band=(0.45, 0.7]` | 0.00541 | 1.308 | **0.00127** |

## 4. 레인 인구조사

| 레인 | LIVE | PRUNED | 전체 |
|---|---:|---:|---:|
| L1 | 4 | 5 | 9 |
| L2 | 8 | 16 | 24 |
| L3 | 6 | 15 | 21 |
| L4 | 4 | 11 | 15 |
| L5 | 2 | 1 | 3 |
| L6 | 8 | 4 | 12 |
| L7 | 8 | 12 | 20 |
| L8 | 31 | 1 | 32 |

## 5. 남은 방향

세 독립 경로가 같은 결론에 수렴했다.

| 경로 | 결론 |
|---|---|
| M269 (직전 세션) | 46-bin 표현의 후처리 오라클 천장이 요구치 미달 |
| 사이클 4·7·9 | 결정정책 오라클 상한 8.8%, 게이트 0/62, 게이트 검출력은 정상 |
| 사이클 8 | 일반화 조건화 오라클 16.5%, 행 오라클 362% 는 도달 불가 |

행 오라클이 `0.742328` 이라는 것은 63 개 정책 집합 안에 이미 목표를 넘는 행동들이
들어 있다는 뜻이다. 문제는 전적으로 **어느 것을 고를지 모른다** 는 데 있고, 고르려면
조건부 분포가 더 날카로워야 한다.

남은 유일한 열린 방향은 **조건부 분포 자체를 바꾸는 것**이며, 이는 모델 적합이
들어가므로 지금까지의 읽기전용 진단과 성격이 다르다.
