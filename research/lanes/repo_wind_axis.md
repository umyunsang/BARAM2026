# 저장소 감사 — 풍속 예측(sitewind/teacher) 축에서 이미 시도된 것

- 작성: 읽기 전용 감사 레인, 2026-08-07
- 범위: `.planning/2026-08-01-leaderboard-top-4-loop/` (363개), `reports/` (363개),
  `src/baram/features/`, `artifacts/backtests/metric-aligned-probe/*site-wind*`
- 규율: 적합 0회 / 락박스 미접근 / 저장소 쓰기는 이 파일 하나
- 모든 주장에 `파일:라인` 근거를 붙였다. 근거 없는 문장은 "추론"으로 명시했다.

---

## 0. 한 줄 요약 (선행 경고)

부모 세션이 받은 브리핑 중 **"C7 teacher scale-up 이 가드로 무효화된 뒤 방치됐다"** 는
**사실이 아니다.** 저장소는 그 가드 오지정을 스스로 발견해 `C1N83`(재판정)과
`C1N84`(시간분할 재확인)로 이미 후속 처리했고, 용량 이득 2.83% 는 **누출분 96%** 로
판명됐다(정직한 면에서 +0.10%). 이 문서 3절과 5절이 그 사슬을 인용과 함께 기록한다.
반면 **`C1N53` 이 만든 "병목은 풍속이 아니다" 라는 서술은 여전히 저장소에 살아 있고,
그것이 부모의 새 증거(오라클 0.88)와 정면 충돌한다** — 5절이 그 지점을 특정한다.

가장 값싼 미시도 항목은 **`atm__*` 대기 레짐 피처군**이다 — 이미 코드로 존재하는데
접두사 필터 한 줄(`run_sequence_classifier.py:106-108`) 때문에 풍속 teacher 표면에
들어가지 않는다(4.5절). 그다음이 **관측 풍향 `_wd` 17열**로, teacher 표적으로 쓰인
적이 0건이다(4.1절).

---

## 1. `scada_ws` 타깃의 정확한 정의

단일 출처는 `run_sequence_classifier.py` 의 `_scada_wind()` 다. 다른 모든 스크립트가
이것을 import 한다(예: `run_site_wind_teacher.py:22-32`).

### 1.1 어떤 터빈

`.planning/2026-08-01-leaderboard-top-4-loop/run_sequence_classifier.py:65-68`

```
("vestas", "train/scada_vestas_train.csv", ((1, range(1, 7)), (2, range(7, 13)))),
("unison", "train/scada_unison_train.csv", ((3, range(1, 6)),)),
```

- 그룹1 = `vestas_wtg01_ws` ~ `vestas_wtg06_ws` (6기)
- 그룹2 = `vestas_wtg07_ws` ~ `vestas_wtg12_ws` (6기)
- 그룹3 = `unison_wtg01_ws` ~ `unison_wtg05_ws` (5기)
- 컬럼명 생성: `run_sequence_classifier.py:74`
  `columns = [f"{prefix}_wtg{number:02d}_ws" for number in turbine_numbers]`

### 1.2 결측/이상치 처리

`run_sequence_classifier.py:75`

```
wind = raw[columns].where(lambda values: (values >= 0.0) & (values < 50.0))
```

- `[0, 50)` m/s 밖은 NaN 으로 **마스킹만** 하고 대체(impute)하지 않는다.
- 이후 `wind.mean(axis=1)` (`:84`) 는 pandas 기본 `skipna=True` 이므로
  **가용 터빈만의 평균**이다. 터빈 수가 줄어도 보정하지 않는다.
- 최종 병합은 `how="left"` (`run_sequence_classifier.py:159`) 이므로 SCADA 가 없는
  행의 `scada_ws` 는 NaN 으로 남고, 학습 마스크가 그것을 거른다
  (`run_site_wind_teacher.py:223` `train_mask = preceding & group_mask & surface["scada_ws"].notna()`).

**실측(이 감사에서 원본 zip 을 읽어 계산, 적합 없음):**

| 그룹 | 터빈별 NaN 비율 | 전 터빈 결측 행 비율 | 행간 터빈 표준편차(평균) | 터빈 쌍 평균상관 |
|---:|---:|---:|---:|---:|
| 1 | 0.00% | 0.00% | **1.1694 m/s** | 0.8342 |
| 2 | 0.00% | 0.00% | **1.1866 m/s** | 0.9259 |
| 3 | 0.97% | 0.06% | **1.0477 m/s** | 0.9103 |

원본 커버리지: vestas `2022-01-01 01:00` ~ `2025-01-01 00:00` (157,819행),
unison `2023-01-01 00:10` ~ `2025-01-01 00:00` (105,264행), 표본간격 최빈 **10분**.
→ **그룹3 은 그룹1·2 보다 teacher 학습용 SCADA 이력이 1년 짧다.**

### 1.3 어떤 집계 (시간 스탬프 규약이 그룹별로 다르다)

`run_sequence_classifier.py:76-79`

```
if group_id < 3:
    timestamp = raw["kst_dtm"].dt.ceil("h")
else:
    timestamp = raw["kst_dtm"].dt.floor("h") + pd.Timedelta(hours=1)
```

- 그룹1·2(vestas): `ceil("h")`
- 그룹3(unison): `floor("h") + 1h`
- 두 규약은 정각(`HH:00`) 표본에서만 갈린다(vestas 는 `HH`, unison 은 `HH+1`).
  둘 다 "끝나는 시각" 라벨이지만 **동일하지 않다.**

집계는 2단이다.

1. `run_sequence_classifier.py:84` — `"scada_ws": wind.mean(axis=1)`
   (같은 10분 시점에서 **터빈 축 평균**)
2. `run_sequence_classifier.py:87-91` —
   `part.groupby(["forecast_kst_dtm", "group_id"], as_index=False)["scada_ws"].mean()`
   (같은 시각 라벨의 **10분 표본 6개 평균**)

즉 최종 타깃은 **시간당 · 그룹당 스칼라 1개** = (터빈평균 → 시간평균).
10분 해상도와 터빈별 개체성은 둘 다 이 지점에서 소멸한다.

### 1.4 누출 규율 — `scada_ws` 는 절대 피처가 아니다

- `run_site_wind_teacher.py:109-122` `_all_weather_columns()` 의 `excluded` 에 `"scada_ws"` 포함
- `m271_cycle42_teacher_restored.py:121-124` `TEACHER_EXCLUDED` 에 `"scada_ws"`, `"actual_kwh"` 포함
- `m271_cycle53_supplied_extraction.py:91` — `assert "scada_ws" not in aux_cols and "scada_ws" not in aw_cols`
- 결정적 재구성 검사: `run_site_wind_teacher.py:263-272` (재병합 결과와 불일치시 RuntimeError)
- **평가기간 부재 확정**: `reports/m271_cycle39_architecture_gap.md:11-15` —
  "scada_ws 는 관측 나셀풍속. actual_kwh 와 상관 **0.9266**, 평가기간 부재."
  같은 노드가 이것을 피처로 쓴 `0.656158` 을 **철회**했다(누출 크기 `+0.113755`,
  `m271_cycle39_architecture_gap.md:28`).
- 물리적 단서: `reports/m271_cycle66_heteroscedastic_wind.md:96` —
  "나셀 풍속계는 로터 뒤에 있어 자유유입풍속이 아니다." 따라서 `scada_ws` 를 표적으로
  삼은 잔차는 **자유유입풍속의 예보오차가 아니라 우리가 쓰는 사상의 오차**다.

---

## 2. sitewind teacher 모델의 변종

### 2.1 정식 레시피 표 (`run_site_wind_teacher.py:48-60`)

| 레시피 | 프로파일 | objective | per_group | num_leaves | sequence | exact_legacy |
|---|---|---|:---:|---:|:---:|:---:|
| `legacy_shared_l2` | legacy | l2 | X | 31 | X | **O** |
| `legacy_group_l2` | legacy | l2 | O | 31 | X | X |
| `windgeom_group_l2` | windgeom | l2 | O | 63 | X | X |
| `allweather_group_l2` | allweather | l2 | O | 63 | X | X |
| `allweather_group_l1` | allweather | l1 | O | 63 | X | X |
| `allweather_group_huber` | allweather | huber | O | 63 | X | X |
| `windgeom_seq_group_l2` | windgeom | l2 | O | 63 | **O** | X |
| `allweather_seq_group_l2` | allweather | l2 | O | 63 | **O** | X |

### 2.2 피처집합 정의 (`run_site_wind_teacher.py:125-137`)

- `legacy` → `auxiliary_columns` (**315 피처**)
  정의는 `run_sequence_classifier.py:205-229`: 토큰
  `10_10u, 10_10v, 80_u, 80_v, 100_100u, 100_100v, 50MU, 50MV, wind10_, wind50, wind80,
  wind100, gust, group_, lead_hour, month` 를 포함하는 컬럼만.
- `windgeom` → `base_columns` = `wind_columns + geometric_columns` (**629 피처**)
  `run_sequence_classifier.py:177-204, 230`
- `allweather` → `_all_weather_columns()` = **모든 수치형 컬럼**에서 식별자/라벨 제외
  (**1,347 피처**), `run_site_wind_teacher.py:109-122`
- `sequence` → `baram.features.sequence.add_issuance_sequence_context` 산출 `seq__*`
  추가 (`run_site_wind_teacher.py:140-149`). allweather+seq 는 **1,635 피처**.

### 2.3 하이퍼파라미터

**exact_legacy** (`run_site_wind_teacher.py:169-185`):
`objective=l2, n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=60,
subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=2.0,
random_state=20260801, n_jobs=6, deterministic=True, force_col_wise=True`

**그 외** (`run_site_wind_teacher.py:186-203`):
`learning_rate=0.025, min_child_samples=40, max_bin=255, subsample=0.9, subsample_freq=1,
colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=3.0, random_state=20260802, n_jobs=6`
+ `n_estimators` 는 early stopping 으로 선택 (`:228-244`, 배치시각 80% 지점 컷오프,
`early_stopping(100)`, `eval_metric="l1"`, 상한 1800)

**분류기 파이프라인이 실제로 쓰는 teacher** 는 이것과 다르다 —
`m271_cycle42_teacher_restored.py:104-120` `TEACHER_PARAMS`:
`n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=40, max_bin=255,
subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=3.0, n_jobs=1`
분할은 `m271_cycle42_teacher_restored.py:149` — **`KFold(3, shuffle=True, ...)`**
(학습행은 무작위 KFold OOF, 보류행은 전량적합 모형: `:137` docstring, `:150-163`).

### 2.4 보고된 풍속 RMSE / 상관 (실측 영수증)

**출처: `artifacts/backtests/metric-aligned-probe/*-site-wind.json`.**
지표 정의는 `run_site_wind_teacher.py:152-164` (rmse/mae/bias/correlation/p90).
R^2 는 어디에도 기록돼 있지 않다 — **상관(correlation)만** 기록된다.

`dev-2023-Q4` (`M62_SITE_WIND_LGBM_SCREEN-dev-2023-Q4-site-wind.json`, n=6,624):

| 레시피 | 피처 | pooled RMSE | MAE | corr | bias | g1 RMSE | g2 RMSE | g3 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `allweather_group_l1` | 1347 | **1.6690** | 1.2687 | **0.9081** | -0.2523 | 1.5536 | 1.7367 | 1.7107 |
| `allweather_group_l2` | 1347 | 1.6886 | 1.2791 | 0.9058 | -0.2562 | 1.5918 | 1.7376 | 1.7324 |
| `windgeom_group_l2` | 629 | 1.6965 | 1.2912 | 0.9047 | -0.2488 | 1.5871 | 1.7554 | 1.7419 |
| `legacy_group_l2` | 315 | 1.7515 | 1.3291 | 0.8983 | -0.2618 | 1.6041 | 1.8107 | 1.8308 |
| `legacy_shared_l2` | 315 | 1.7730 | 1.3459 | 0.8956 | -0.1617 | 1.6434 | 1.8501 | 1.8183 |
| `allweather_group_huber` | 1347 | 1.8217 | 1.3398 | 0.8965 | **-0.4535** | 1.6000 | 1.7778 | 2.0578 |

`allweather_seq_group_l2` (`M65B_...-dev-2023-Q4-site-wind.json`, 1,635 피처):
pooled RMSE **1.6835** / corr 0.9067 — `allweather_group_l2` 대비 **-0.0051** 뿐.

다른 부스터 계열 (같은 fold, 같은 1,347 피처):

| 후보 | 계열 | pooled RMSE | corr | 런타임 |
|---|---|---:|---:|---:|
| `M67_SITE_WIND_CAT` | CatBoost (depth=8, lr=0.03, l2_leaf_reg=5) | **1.6620** | **0.9081** | 1135s |
| `M66_SITE_WIND_XGB` | XGBoost (max_depth=7, lr=0.025) | 1.7010 | 0.9041 | 177s |

파라미터: `run_site_wind_alternatives.py:62-74` (CatBoost), `:112-120` (XGBoost).

**다른 fold** (`M62_SITE_WIND_L2-*`):
- Q2: `allweather_group_l2` RMSE **1.5435** / corr 0.8908 ; `legacy_shared_l2` 1.6006 / 0.8817
- Q3: `allweather_group_l2` RMSE **1.5192** / corr 0.8532 ; `legacy_shared_l2` 1.5567 / 0.8451

### 2.5 teacher 예측이 분류기로 들어가는 형태 (13 파생열)

`m271_cycle42_teacher_restored.py:168-179`

```
sitewind__legacy, sitewind__allweather, sitewind__mean, sitewind__delta,
sitewind__disagreement,
{legacy,allweather,mean}2, {..}3, {..}_powercurve  (normalized**3)
```
`run_sequence_classifier.py:350-352` 는 같은 계열의 축소판
(`aux_scada_ws`, `aux_scada_ws2`, `aux_scada_ws3`).

### 2.6 **면(surface)에 따라 sigma 가 3개로 갈린다 — 비교시 반드시 확인할 것**

| 면 | sigma_v | 근거 |
|---|---:|---|
| 학습행 **무작위 KFold OOF** (누출) | **1.0923** | `reports/m271_cycle58_teacher_oof_split.md:14` |
| 학습행 **시간블록** OOF | **1.6772** | `reports/m271_cycle58_teacher_oof_split.md:15` |
| 평가 fold **test 행** (allweather) | g1 **1.4957** / g2 **1.5947** / g3 **1.6521** | `reports/m271_cycle71_teacher_weight.md:19-21` |
| 평가 fold test 행 (legacy) | g1 1.5683 / g2 1.6983 / g3 1.7491 | `reports/m271_cycle71_teacher_weight.md:19-21` |
| 평가 fold test 행 (독립 계측) | **1.5866** (편향 -0.0056) | `reports/D1_F1_1_wind_bias_sitewind.json`, `reports/m271_cycle66_heteroscedastic_wind.md:9` |

**이 셋을 섞으면 판정이 뒤집힌다.** 실제로 그 사고가 저장소에서 두 번 일어났다(3.4, 3.5절).

---

## 3. 풍속 예측 정확도 자체를 개선하려 한 실험 — 전수

### 3.1 입력 표현 계열

| 노드 | 시도 | 결과 | 근거 |
|---|---|---|---|
| `C1N5_SPATIAL_INTERPOLATION` | GFS/LDAPS **IDW vs nearest** 를 나셀풍속 상관으로 비교 | IDW 우세(GFS 3/3 그룹). 판정 `GFS_INTERPOLATION_JUSTIFIED` | `reports/m271_cycle5_spatial.md:25-39` |
| `C1N6_WIND50_MIDPOINT` | LDAPS `wind50mid = ((Umax+Umin)/2,(Vmax+Vmin)/2)` 신규 피처 | **기각**. 상관 0.7859~0.7926 로 `wind10`(0.7904~0.7979)·`wind50max`(0.7926~0.8009) 에 못 미침. 잔차sd 도 열세 | `reports/m271_cycle6_wind50mid.md:20-39` |
| `C1N10_BLEND_LOCALITY` | GFS+LDAPS 블렌딩 이득의 국소성 | 그룹별 오차감소 **0.37% / 0.10% / 0.15%** (같은-fold 적합이므로 **상한**) | `reports/m271_cycle10_blendlocality.md:18-22` |
| `C1N77_PER_SOURCE_STACK` | NWP 소스별 분리 모델 + 확률 스태킹 (HEFTCom2024 우승팀 방식) | STACK 0.603122 < POOLED 0.604043 (**-0.000921**). `FRONT_END_FUSION_IS_NOT_THE_BOTTLENECK` | `reports/m271_cycle77_per_source_stack.md:13-37` |

### 3.2 사후보정 계열

| 시도 | 결과 | 근거 |
|---|---|---|
| **풍속단계 quantile mapping** (출력이 아니라 풍속에 MOS) | 편향은 이미 -0.0056 으로 없고, qmap 후 sigma **1.5866 → 1.6273 (-2.56%, 악화)** | `reports/D1_F1_1_wind_bias_sitewind.json`; 스크립트 `d1_f1_1_windbias.py:47-58` |
| teacher 두 프로파일 **fold-외 최적 가중** (`C1N71`) | legacy 가중 0.19~0.29. 잔차 감소 **0.75%** → 환산 Total **+0.001238** (검출문턱 0.001013). `OPTIMAL_WEIGHT_WORTH_PROMOTING` — **단 승격 판정은 하지 않음** | `reports/m271_cycle71_teacher_weight.md:17-21, 46-66` |

### 3.3 외부 NWP 계열 (두 번 닫힘)

- `C1N51`: Open-Meteo Previous Runs, `wind_speed_100m_previous_day2`(48h 리드),
  truth=`scada_ws`, 2024면. ICON `q=1.20~1.28`(우리보다 **나쁨**),
  ECMWF `q=0.99~1.07 / rho=0.746~0.780`. 수용영역 통과 0/6.
  `EXTERNAL_NWP_CLOSED_BY_MEASUREMENT` — `reports/m271_cycle51_external_source_probe.md:18-41`
- `C1N52`: 결합 감소 **3.9~6.1%** 대 필요 **13.3%** → 필요량의 30~45%
  — `reports/m271_p4_consolidate.md:100`
- `M281` 재검정(2026-08-06): 누출안전 변수가 **2023 검증면을 전혀 덮지 않는다**.
  2023 에 있는 것은 best-match 단기예보뿐이며 D-1 14:00 KST 이후 생성분을 포함해 **규칙 위반**.
  — `reports/m281_external_nwp_premise_retest.md:18-40`

### 3.4 teacher 용량/문맥 확장 계열 — **가장 중요한 사슬 (C7 → C7b → C7c)**

`C1N82_TEACHER_SCALEUP` (`m271_c7_teacher_scaleup.py`, ARMS `:129-134`):

| 팔 | 정의 | sigma_v (무작위 KFold 면) | 감소율 |
|---|---|---:|---:|
| `base` | 400그루 / 63잎 | 1.1037 | +0.00% |
| `deep` | **255잎** | **1.0725** | **+2.83%** |
| `long` | **1200그루** / lr 0.02 | 1.0808 | +2.07% |
| `seq` | base + 시계열문맥 **480열** | 1.1069 | **-0.29%** |
| `seq_deep` | 255잎 + 시계열문맥 | 1.0810 | +2.06% |

(`reports/m271_c7_teacher_scaleup_receipt.json` `arms`/`reductions` 블록;
`reports/m271_c7_teacher_scaleup.md:20-30`)

- **1차 판정 `GUARD_FAILED_RESULT_VOID`** — V1 이 base 1.1037 을 `C1N66` 의 **1.5866**
  과 비교해 실패. (`m271_c7_teacher_scaleup.md:34, 43`;
  receipt `checks.V1_base_matches_c66: false`)
- **원인은 하네스가 아니라 기준 오지정**이었다. 스크립트가 스스로 그것을 기록한다 —
  `m271_c7_teacher_scaleup.py:113-121`:
  "처음엔 C1N66 의 **1.5866** 을 잡았는데 그것은 **평가 fold 의 test 행** 잔차이고,
  이 노드가 재는 것은 **학습행의 무작위 KFold OOF** 다" → `C58_SHUFFLE_SIGMA = 1.0923`
- **`C1N83`(C7b) 재판정**: 올바른 기준 1.0923 대비 base 1.1037, 차 0.0114 → V1 통과.
  H2(C16 문턱 2.72%) **통과**. 판정
  `TEACHER_CAPACITY_CLEARS_GATE_PENDING_CHRONOLOGICAL_CONFIRMATION`
  — `reports/m271_c7b_rejudge.md:9-11, 23, 28, 34-36`
- **`C1N84`(C7c) 시간분할 재확인** — 내부 KFold 없이 fold 시작 이전 행으로만 적합:

  | 팔 | 전체 | g1 | g2 | g3 |
  |---|---:|---:|---:|---:|
  | base | **1.5847** | 1.4957 | 1.5947 | 1.6521 |
  | deep | **1.5831** | 1.4953 | 1.5894 | 1.6532 |

  **감소율 +0.10%** (환산 Total +0.000160, C16 문턱 0.004453).
  판정 `DEEP_HELPS_BUT_BELOW_MAGNITUDE_GATE`
  — `reports/m271_c7c_chronological.md:11-18, 32`
- 통합 서술: "`deep` 의 sigma_v 감소는 무작위 KFold 면에서 2.83% 였으나 시간 분할
  test 행에서는 **+0.10%** 다 — **이득의 96% 가 시간 인접 누출**이었다"
  — `reports/m271_p4_consolidate.md:134`

**따라서 teacher 용량 축과 시계열문맥 축은 정직한 면에서 정당하게 닫혔다.**
(단, 5.3절이 이 폐쇄의 *유효 범위*를 좁힌다.)

### 3.5 teacher 분할/누출 계열

- `C1N58_TEACHER_OOF_SPLIT`: shuffle→blocked 로 바꾸자 하류 Total **-0.002027**.
  판정 `FEATURE_DISTRIBUTION_SHIFT_AXIS_CLOSED`
  — `reports/m271_cycle58_teacher_oof_split.md:12-15, 29-31, 49`
  폐기 전제는 `TEACHER_FEATURE_NOISE_MONOTONE_LOWER_IS_BETTER`
  (`reports/m271_p4_consolidate.md:57`)
- `C1N53_SUPPLIED_EXTRACTION`: 폐기 전제 `SUPERSEDED_BY_CHRONOLOGICAL_SPLIT` —
  "사이클 53 은 `teach()` 의 `KFold(3, shuffle=True)` 를 그대로 써서 시간 인접 누출이
  들어갔다(라벨 lag-1 자기상관 0.951~0.962)" — `reports/m271_p4_consolidate.md:128`

### 3.6 풍속오차 → 점수 환산 (축을 닫은 계산의 근거)

- `C1N66`: sigma_v 가 풍속에 **선형 증가**. 기울기 g1 +0.0865 / g2 +0.1136 / g3 +0.1965,
  Spearman 0.911~0.982 — `reports/m271_cycle66_heteroscedastic_wind.md:11-15`
- `C1N69` 반응곡선 `pred(k)=커브(scada_ws + k*eps)`:
  k=0 → Total **0.847521**, k=1 → **0.612331**.
  목표 0.66 은 `k*=0.7269` → **요구 감소율 27.3%**, 가용 3.9~6.1%.
  판정 `REQUIRED_WIND_REDUCTION_EXCEEDS_MEASURED_AVAILABLE`
  — `reports/m271_cycle69_skill_response.md:11-21, 33-39`
- `C1N71` 이 쓴 환산 기울기 **0.1642 Total/단위 k** — `reports/m271_cycle71_teacher_weight.md:47`
- `C1N59` 완전예보 상한: oracle **0.845340** vs control 0.604043 (**+0.241297**),
  FICR 우세(+0.196571 대 1-NMAE +0.044726). 그룹별 +0.288 / +0.287 / **+0.149**.
  판정 `WIND_AXIS_DIRECTION_CONFIRMED_MAGNITUDE_OPEN`
  — `reports/m271_cycle59_perfect_prog_bound.md:9-11, 24, 30-33, 37`
- `C1N68` 경험적 3팔(적합 없음): 적중률 model 0.3067 / curve_teacher 0.2897 /
  **curve_oracle 0.7592** — `reports/m271_cycle68_empirical_decomposition.md:12-15`

---

## 4. 풍속 예측에서 **아직 시도되지 않은** 것

각 항목은 "저장소 전역 grep 에서 근거를 찾지 못했다" 를 근거로 한다. 부정 주장이므로
탐색 방법을 함께 적는다.

### 4.1 관측 **풍향**(`_wd`) 타깃 — 완전 미시도 (**가장 명확한 공백**)

원본 SCADA 는 17기 전 터빈의 풍향을 준다:
`unison_wtg01_wd`~`05_wd`, `vestas_wtg01_wd`~`12_wd`
(zip 컬럼 실측; `train/scada_unison_train.csv` 16열 중 5열, `train/scada_vestas_train.csv` 37열 중 12열).

전역 grep (`wtg[0-9]*_wd|_wd"|_wd'|scada_wd`, `.venv` 제외) 결과 **3곳뿐**:
- `.planning/.../m271_n0_scada.py:45` — 진단용 로드
- `.planning/.../run_turbine_wind_power_stack.py:93-105` — **피처**로만
  (`direction_sin`/`direction_cos` 시간평균)
- `src/baram/features/weather.py:76` — NWP 원시 각도열을 **집계에서 배제**하는 규칙

→ **teacher 의 표적으로 풍향을 쓴 실험은 0건.** 후류(wake)는 풍향에 강하게 의존하고
(`C1N3_MASS_AXIS_WIND_SECTOR`, `geometric.py:207` `layout_along` 이 이미 배치축 정렬을
피처로 만든다), 그룹 내 터빈 쌍 상관이 0.83~0.93 으로 완전하지 않다는 것이
방향의존 후류의 간접 증거다.

### 4.2 터빈별 개별 풍속 타깃 — 미시도

`_scada_wind()` 는 `wind.mean(axis=1)` (`run_sequence_classifier.py:84`) 로
**터빈 축을 즉시 붕괴**시킨다. 17개 타깃을 따로 학습한 흔적 없음
(grep `wtg.*_ws` 결과는 전부 컬럼 생성/진단이고 타깃 루프가 아니다).

**크기 근거(이 감사 실측):** 그룹 내 행간 터빈 표준편차가 **1.05~1.19 m/s** 인데,
정직한 면의 teacher 오차가 **1.4957~1.6521 m/s** 다(2.6절). 즉 버려지는 터빈간 분산이
teacher 총오차와 **같은 자릿수**다. 이것이 개선 가능한 신호인지 관측잡음인지는
측정되지 않았다.

### 4.3 10분 해상도 타깃 — 미시도

원본은 10분 간격(실측 최빈 dt=10분)인데 `groupby(...).mean()`
(`run_sequence_classifier.py:87-91`)이 시간평균으로 붕괴시킨다.
`grep 10min` 은 `rated_kwh_per_10min` / `power_kw10m` (출력측 정규화)에서만 걸리고
**풍속 타깃의 10분 해상도 사용은 0건**.

### 4.4 분위 회귀 / 분포형 풍속 예측 — 미시도

`run_site_wind_teacher.py:48-60` 의 objective 는 `l2`/`l1`/`huber` **점추정 3종뿐**.
CatBoost 도 `loss_function="RMSE"` (`run_site_wind_alternatives.py:63`),
XGBoost 도 `reg:squarederror` (`:113`).
`grep quantile` 에서 풍속과 걸리는 것은 `p90_absolute_error` 진단
(`run_site_wind_teacher.py:163`)과 **사후** quantile mapping(3.2절)뿐이다.

**이것이 중요한 이유:** `C1N66` 이 sigma_v(v) 의 강한 이분산을 이미 실측했고
(기울기 +0.0865/+0.1136/+0.1965, Spearman 0.911~0.982,
`reports/m271_cycle66_heteroscedastic_wind.md:11-15`), 하류 지표는 점추정이 아니라
**분포**를 요구하는 46-bin 분류기다(`m271_cycle39_architecture_gap.md:30` — 분포표현이
직접 점회귀보다 **+0.087** 우월). 그런데 teacher 는 점추정 스칼라 1개(+ 그 제곱·세제곱)만
넘긴다(`m271_cycle42_teacher_restored.py:168-179`). **teacher 의 예측 불확실성이
분류기로 전달되는 경로가 없다.**

### 4.5 안정도(stability) 층화 — **피처는 만들어졌으나 teacher 에 도달하지 않는다**

이 항목은 처음에 "미시도" 로 적었다가 grep 으로 **반증했다.** 정확한 상태는 다음과 같다.

`src/baram/features/physics.py` 가 teacher 에 주는 것:
- `phys__hub117_speed = speed100 * (117/100)**0.2` — **고정 지수 0.2** (`:51`)
- `phys_v2__shear_alpha_100_80` — 자료 추정 shear alpha, `clip(-0.2, 0.6)` (`:74, 87`)
- `phys__air_density`, `phys__rho_v3` (`:53-54`)

**그런데 본격적인 대기 레짐 피처군이 이미 존재한다** —
`.planning/.../run_atmospheric_regime_pls_rank.py`:

| 피처 | 라인 |
|---|---|
| `atm__gfs__*__bulk_richardson_proxy` = `(theta850 - t2) / (shear^2 + 0.25)` | `:114-116` |
| `atm__gfs__*__alpha_80_10`, `__alpha_100_80` (로그 전단지수) | `:106-113` |
| `atm__gfs__*__theta500_minus_theta700` | `:100` |
| `atm__gfs__*__gust_excess`, `__gust_factor` | `:101-102` |
| `atm__gfs__*__pbl_w10_ratio` | `:105` |
| `atm__ldaps__*__w50_envelope`, `__w50_midpoint`, `__w50_asymmetry` | `:124-128` |

**핵심 사실: 이 열들은 teacher 에 절대 도달하지 않는다.**
teacher 표면은 `run_sequence_classifier.py:103-109` 에서 접두사
`("gfs_spatial__", "ldaps_spatial__", "source_disagreement__", "phys__", "phys_v2__")`
로만 support 열을 고른다 — **`atm__` 은 그 목록에 없다.**
그리고 `grep -l "atm__"` 는 `.planning` 전역에서 **`run_atmospheric_regime_pls_rank.py`
단 하나**만 반환한다(site-wind/teacher/c7/cycle42 스크립트에는 0건).
즉 `atm__*` 은 그 스크립트 안에서 만들어져 **출력측 PLS source rank
(`M205_STRICT_ATMOSPHERIC_PLS_RANK_Q3`)에만 쓰이고 소멸**한다.

→ **레짐/안정도 피처는 "출력 랭커용으로 만들어졌고 풍속 teacher 에게는 준 적이 없다."**
`C1N82` 의 "입력 부족은 **아니다** — teacher 는 이미 수치형 1,347 열 전부를 먹는다"
(`reports/m271_c7_teacher_scaleup.md:16`)는 **그 1,347 열이 무엇인지**에 대한 진술이며,
`atm__*` 은 그 1,347 열 **밖**에 있다.
층화 변수(stratifier)나 그룹별 모델 게이트로 쓰는 것도 여전히 0건이다.

### 4.6 **교차 발행(cross-issuance)** 시간 문맥 — 미시도

`src/baram/features/sequence.py:12` `_KEYS = ["data_available_kst_dtm", "group_id", "forecast_kst_dtm"]`
이고 `:58-60` 이 `data_available_kst_dtm` 으로 groupby 한다 →
**문맥은 같은 발행 안에서만** 만들어진다. 기본값도 좁다:
`neighbor_offsets=(-2,-1,1,2)`, `rolling_windows=(3,5)` (`:19-20`),
계약이 `abs(offset) > 6` 을 금지(`:38-39`).

→ **직전 발행분의 같은 유효시각 예보**(예보 경향/persistence), **직전 관측 SCADA lag**,
6시간 초과 창은 전부 미시도. `seq` 팔이 실패한 것(-0.29%, 3.4절)은 **이 좁은 정의의
문맥**이 실패한 것이지 시간 문맥 일반이 아니다.

### 4.7 격자 전체를 쓰는 구조적 피처 — 부분 시도

이미 있는 것: IDW/nearest 가중평균(`spatial.py:52-111`), 벡터평균 대 스칼라평균과
그 차이(`geometric.py:185-194` `mean_speed3`, `vector_speed`, `vector_spread`),
배치축 투영(`layout_along` `:207`), 소스간 정렬(`_add_alignment` `:233-265`),
소스 불일치(`add_source_disagreement_features` `spatial.py:186`).

**정정**: 처음에 "격자 경도/발산이 없다" 고 적었다가 grep 으로 **반증했다.**
`geometric.py:135-170` 이 격자에 평면 최소자승(`design_pinv`)을 적합해 실제로 계산한다:

| 산출열 | 라인 |
|---|---|
| `geom__{source}__{level}__divergence` | `:161, 215` |
| `__vorticity` | `:162, 216` |
| `__stretch`, `__shear` (변형장) | `:163-164, 217-218` |
| `__gradient_norm` | `:165-170, 219` |
| `__coherence`, `__vector_spread` (격자간 일관성) | `:204-205` |

이 열들은 `run_sequence_classifier.py:204, 230` 을 통해 `windgeom`/`allweather` 양쪽에
**모두 들어간다**. 따라서 1차 구조적 격자 피처는 **이미 시도됐다.**

**여전히 없는 것 (grep `upstream|advect|upwind|fetch` → 유효 0건;
`fetch` 히트는 전부 HTTP 다운로더다):**
- **이류(advection)** — 풍향에 따라 **상류 격자**를 선택해 리드타임만큼 거슬러 올라간
  값을 쓰는 피처. 현재 격자 축약은 전부 **동시각·고정 가중**(IDW/nearest,
  `spatial.py:52-111`)이다.
- 전선·수렴대 같은 **비국소 패턴 식별자** (발산/회전은 국소 1차 미분일 뿐이다).

### 4.8 트리 계열 밖의 teacher — 미시도

시도된 계열: LightGBM / XGBoost / CatBoost (2.4절). 신경망·선형·물리 하이브리드 0건.
저장소가 이 방향의 **부활조건을 스스로 명시**해 두었다:
`BASIS_FUNCTION_IRRELEVANT_TO_GBM` — "**선형·신경망 계열로 바꾸면 전제가 뒤집힐 수 있다**"
(`reports/m271_p4_consolidate.md:92`).
또한 `C1N82` 의 리서치가 인용한 근거가 정확히 이것이다 —
"진단풍모델 + 신경망 후처리로 hub height 풍속 RMSE 약 30% 개선"
(`reports/m271_c7_teacher_scaleup.md:9`, `applicability: directly_supported`).
**용량 확장(255잎)은 시험됐지만 계열 교체는 시험되지 않았다.**

### 4.9 그룹3 이력 비대칭 보정 — 미시도

unison SCADA 는 2023-01-01 시작, vestas 는 2022-01-01 시작(1.3절 실측).
그룹3 teacher 는 학습 이력이 1년 짧은데, per-group 학습
(`run_site_wind_teacher.py:216-217`)이 그룹간 전이를 막는다.
`C1N12_G3_HISTORY` 는 **출력** 이력 길이를 봤지 **teacher 학습 이력**을 보지 않았다
(폐기 전제 `HISTORY_LENGTH_DOES_NOT_EXPLAIN_G3`, `reports/m271_p4_consolidate.md:104`).
그룹3 은 oracle 이득도 가장 작다(+0.149 대 +0.288/+0.287,
`reports/m271_cycle59_perfect_prog_bound.md:30-33`).

---

## 5. 오라클 0.88 이라는 새 증거로 **재개해야 하는** 축

부모 세션의 새 측정: **관측 SCADA 풍속만으로 로컬 0.8807**,
**그룹평균 풍속 스칼라 1개만으로 0.8300**.

### 5.1 `C1N53` 의 서술은 죽었다 — 그러나 저장소는 아직 그렇게 말하지 않는다

`reports/m271_cycle53_supplied_extraction.md:35`
> 판정: **SUPPLIED_DATA_ALREADY_MEETS_WIND_REQUIREMENT_BOTTLENECK_ELSEWHERE**

`:43-46`
> 외부 NWP 폐쇄는 결론이 유지되지만 **이유가 바뀐다**: 풍속이 부족해서가 아니라
> **풍속은 이미 충분한데 출력 예측에서 샌다**. 병목이 `NWP -> 풍속` 이 아니라
> `풍속 -> 출력` 이라는 뜻이며 …

이 판정의 근거인 sigma **1.002 / 1.124 / 1.052** (`:18-20`)는
`m271_cycle42_teacher_restored.py:149` 의 `KFold(3, shuffle=True)` 산출물이며
(`m271_cycle53_supplied_extraction.py:96-97` 이 `teach()` 를 그대로 호출),
정직한 값은 **1.4957~1.6521** 이다(2.6절). 즉 감소율 46~49% 는 실제로 **약 14~23%** 다.

저장소는 누출은 인정했지만 **결론은 살렸다**:
`reports/m271_p4_consolidate.md:128` — "C1N54 가 시간분할로 교정했고 **결론 자체는 생존한다**".

→ **이 "결론 생존" 문장이 부모의 새 증거와 정면 충돌한다.** 그룹평균 풍속 스칼라
1개만으로 0.8300 이 나온다면, 같은 스칼라의 teacher 추정치를 쓰는 모형이 0.60~0.63 인
것은 병목이 **`풍속 -> 출력` 이 아니라 `NWP -> 풍속`** 임을 뜻한다.
**재개 대상 1: `C1N53` 의 잔존 서술을 명시적으로 철회하고, 그 서술에 의존해 방향을
잃은 하위 노드를 되살릴 것.**

### 5.2 `C1N69` 의 "요구 27.3%" 는 재계산되어야 한다

`C1N69` 는 요구 감소율을 **커브 기계 면**에서 정의했다 —
`pred(k)=커브(scada_ws + k*eps)`, k=0 에서 Total **0.847521**
(`reports/m271_cycle69_skill_response.md:5, 11`).
부모의 새 오라클은 **0.8807** 로 그보다 높고, 사용 경로도 다르다
(커브 1회 통과가 아니라 학습된 사상).

또한 `C1N71` 이 스스로 이 환산의 한계를 못박았다:
> "Total 이득은 C69 반응곡선(커브기계 면)으로 환산한 **추정**이다. 모형 면으로의 이전을
> 가정하며 **C33·C45 에서 두 번 틀린 가정**이므로, 이 노드로 승격 판정을 하지 않는다."
> — `reports/m271_cycle71_teacher_weight.md:66`

**재개 대상 2:** 기울기 0.1642 와 요구치 27.3% 는 **모형 면에서 다시 재야 한다**.
`C1N84`(+0.10%), `C1N71`(+0.75%), `C1N52`(3.9~6.1%) 의 "문턱 미달" 판정은 전부
이 환산에 매달려 있으므로, 환산이 바뀌면 **셋 다 재판정 대상**이 된다.

### 5.3 `C1N84` 의 폐쇄는 **용량**만 닫았지 **정확도 축**을 닫지 않았다

`C1N84` 가 시험한 것은 `deep`(255잎) 하나다 —
"`long`·`seq`·`seq_deep` 은 `deep` 에 뒤졌으므로 **제외**"
(`m271_c7c_chronological.py:31-32`). 그런데 그 순위는 **누출된 무작위 KFold 면**에서
매겨졌다(2.83 / 2.07 / -0.29 / 2.06%). 누출이 이득의 96% 를 설명한다면
**그 면에서 매긴 팔 순위 자체가 신뢰할 수 없다.**

더 중요하게, `C1N82` 가 스스로 선언한 공백은 세 개였고
(`reports/m271_c7_teacher_scaleup.md:14-16`) 그중 하나는 이렇게 못박혀 있다:
> "입력 부족은 **아니다** — teacher 는 이미 수치형 1,347 열 전부를 먹는다"

이 진술은 **열의 개수**에 대한 것이지 **표적·해상도·계열**에 대한 것이 아니다.
4절의 미시도 항목(풍향 타깃, 터빈별 타깃, 10분 해상도, 분위/분포 출력, 계열 교체)은
전부 이 진술의 사정거리 **밖**에 있다.

**재개 대상 3:** "teacher 축은 닫혔다" 는 **"LightGBM 잎 수를 늘려도 안 된다"** 로
축소 해석해야 한다.

### 5.4 저장소가 스스로 적어둔 부활조건 중 지금 발화하는 것

`reports/m271_p4_consolidate.md` 의 "되살아나는 조건" 열과 전제 설명에서:

| 노드 | 근거 라인 | 부활조건 | 현재 상태 |
|---|---|---|---|
| `AXIS_UNUSED_COLUMNS` | `:22` | "미사용 컬럼의 조건부 추가이득이 측정되면" | **발화 후보 (2건)** — (가) SCADA `_wd` 17열(4.1절), (나) `atm__*` 레짐열이 teacher 표면 밖에 있음(4.5절) |
| `C1N56_MEASURED_POWERCURVE` | `:92` | "**선형·신경망 계열로 바꾸면 전제가 뒤집힐 수 있다**" | **발화 후보** (4.8절) |
| `C1N99_ENSEMBLE_EXHAUSTIVE` | `:97` | "**구조적으로 다른 후보**(다른 표적 표현, 다른 자료원, **다른 시간해상도**)가 생기면 뒤집힌다" | **발화 후보** — 10분 해상도·터빈별·풍향 타깃이 정확히 이 정의(4.1~4.3절) |
| `GBM_REPARAMETERISATION_IS_NOT_DIVERSITY...` | `:102` | "방법군 자체가 달라야 한다 — 아날로그/최근접이웃, **물리 직접계산**, **다른 표적공간**" | **발화 후보** (4.8절) |
| `C1N6_WIND50_MIDPOINT` | `:64` | "중점이 10m·max 를 이기면" | 미발화(상관 기준으로 이미 열세) |
| `C1N5_SPATIAL_INTERPOLATION` | `:59` | "nearest 가 IDW 를 이기면" | 미발화 |
| `C1N10_BLEND_LOCALITY` | `:27` | "블렌딩 이득이 유의미해지면 (진짜 다른 소스)" | 미발화(외부소스는 M281 로 별도 차단) |
| `C1N103_EXTERNAL_NWP_MEMBER` | `:98`, `m281_...:39-40` | "2023 을 덮는 다른 예보 아카이브를 찾거나 2024 lockbox 를 열면" | **미발화 — 재검정 완료, 열지 말 것** |

### 5.5 재개해서는 **안 되는** 것 (오라클 0.88 로도 안 열림)

- **외부 NWP**: 두 독립 근거로 닫혔고 2026-08-06 에 재검정까지 마쳤다.
  (가) sigma 감소 3.9~6.1% 대 필요 13.3%, (나) **누출안전 변수가 2023 검증면을 0행 덮는다**.
  — `reports/m281_external_nwp_premise_retest.md:34-40`.
  요구치가 5.2절대로 재계산돼도 **(나)는 그대로 남는다** — 데이터가 없다.
- **`scada_ws` 를 피처로 쓰는 모든 경로**: 평가기간 부재
  (`m271_cycle39_architecture_gap.md:11`), 이미 `0.656158` 철회 전례가 있다.
  오라클 0.88 은 **상한 진단**이지 후보가 아니다
  (`m271_cycle59_perfect_prog_bound.md:46` — "이 노드는 후보가 될 수 없다").
- 나셀 풍속계가 로터 뒤라 **오라클은 위쪽 편향**이다
  (`m271_cycle59_perfect_prog_bound.md:39-42`, `m271_cycle68_empirical_decomposition.md:60`).
  0.8807 도 같은 편향을 갖는다 — 도달 가능 목표가 아니라 방향 지시자로만 읽어야 한다.

---

## 6. 이 문서를 쓸 때 지킨 계량 규율

- 인용한 모든 수치는 `reports/*.md`, `reports/*_receipt.json`,
  `artifacts/backtests/metric-aligned-probe/*-site-wind.json` 원문에서 왔다.
- 1.2절과 4.2절의 터빈 통계만 이 감사가 **새로 계산**했다(원본 zip 읽기, 적합 0회).
- sigma 를 인용할 때는 **항상 면(무작위 KFold / 시간블록 / test 행)을 함께** 적었다.
  2.6절이 그 세 값을 한 표에 모아 둔 이유이며, 저장소에서 실제로 두 번(C1N82, C1N53)
  이 혼동이 판정을 뒤집었다.
- 4절의 부정 주장("미시도")은 전부 grep 으로 검증했고, **초안의 두 항목이 그 검증에서
  반증되어 정정됐다** — 4.5(안정도: 피처는 존재, teacher 표면 밖)와
  4.7(격자 경도/발산: 이미 존재하고 teacher 에 들어감). 정정 전 초안을 그대로
  두지 않았다는 사실을 남긴다.
