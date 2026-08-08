# S12 외부문헌 레인 — DACON BARAM 2026 상위해법·기전 조사

- 레인: `S12_ext_dacon_solutions` (읽기전용 외부 조사)
- 수행일: 2026-08-07 (KST)
- 도구: `websearch`(Serper/Google) **46쿼리** + GitHub Search API **14쿼리** + 공개 웹페이지 read-only fetch
- 저장소 쓰기: `research/lanes/` 아래 5개 파일뿐. 모델 학습·lockbox·git 조작·업로드 없음.
  - `S12_ext_dacon_solutions.md` (본 문서)
  - `S12_ext_dacon_solutions.searchlog.json` (웹 검색 전문 로그)
  - `S12_github_searchlog.json` (GitHub 검색 전문 로그)
  - `S12_leaderboard_snapshot_2026-08-07.csv` (공개 리더보드 top-100 원본)
  - `S12_briefing_transcript_q0zQX_9f_7Y.json` (주최측 사전설명회 영상 자동자막 891조각)
  - `research/scratch/s12/eval_img*.png` (공식 평가 탭에 base64로 박혀 있던 정산 단가표 이미지)

---

## §A 증거 등급 규약

| 등급 | 뜻 |
|---|---|
| `[원문]` | 해당 URL의 **본문 전체를 직접 받아 읽음**. 인용부호 안은 원문 그대로. |
| `[스니펫]` | Google/Serper가 돌려준 검색 스니펫만 확인. 본문 미확인. |
| `[전문미확인]` | 존재는 확인했으나 본문·수치를 검증하지 못함. |
| `[자동자막]` | YouTube 자동생성 한국어 자막. **음성인식 오류가 많음**(예: "풍력"→"폭력", "NMAE"→"M&", "정산금"→"정상금"). 뜻은 문맥으로 복원했고, 원문 그대로를 병기함. |
| `[산출]` | 본 레인이 위 원문 수치로부터 **직접 계산**한 값. 계산식을 함께 적었다. |

**중요한 한계 두 가지를 먼저 밝힌다.**
1. 본 대회는 **2026-08-14 종료 예정으로 아직 진행 중**이다. 따라서 "우승 해법 공개"는 **세상에 존재하지 않는다.**
   존재하는 것은 (a) 공식 문서, (b) 진행 중인 참가자들이 실수로/의도적으로 공개한 GitHub 저장소, (c) 공개 리더보드다.
2. 아래 §D의 GitHub 저장소들은 **현재 경쟁 중인 팀의 것**이다. 그들의 수치는 그들의 로컬 검증/공개 LB에서 잰 것이며,
   우리 파이프라인으로 재현한 적이 없다. **읽기 전용 정보로만 취급했고 코드는 clone·다운로드하지 않았다.**

---

## §B 결론 요약 — 이 레인이 실제로 바꾼 판단

> **우리는 "밴드 적중 기전(band-hitting)"을 못 찾아서 지고 있는 것이 아니다. 우리는 그냥 점예측이 부정확해서 지고 있다.**
> 우리의 밴드 배치는 이미 필드 상위 100팀 중 91팀보다 좋다. 부족한 것은 1-NMAE 하나뿐이다.

근거는 §C-3의 리더보드 실측이다. 요약 수치:

| | 1-NMAE | FICR | Total |
|---|---:|---:|---:|
| 우리 M266 (온라인 실측) | 0.858775 | 0.416167 | 0.637471 |
| 공개 LB **top-100 최솟값** | **0.86777** | **0.42665** | 0.65096 |
| 공개 LB 중앙값(top-100) | 0.87425 | 0.43842 | 0.65602 |
| 공개 LB 1위 (연식2) | 0.87964 | 0.46767 | 0.67365 |

`[산출]` **우리 1-NMAE는 top-100 전원보다 낮다.** 최하위(100위)보다도 0.00900 낮고, 중앙값보다 0.01548 낮다.
FICR도 top-100 최솟값보다 0.01048 낮다. 즉 **두 축 모두에서 top-100 전원에게 지배(dominated)당하고 있다.**

그런데 두 축은 독립이 아니다. 정확도가 오르면 FICR은 자동으로 따라온다. 전환율 κ = dFICR/d(1−NMAE) 을
경쟁팀이 실측한 값 **κ = 2.21** (§D-1, `step28_ficr_geometry.py`)로 두고 각 팀의 "모양 프리미엄"
(= 같은 정확도였다면 얼마나 더 FICR을 챙겼는가)을 **우리 작동점 기준**으로 재면:

`[산출]` `prem(team) = FICR_team − (0.416167 + 2.21 × (1−NMAE_team − 0.858775))`

| 순위 | 팀 | 1-NMAE | FICR | Total | 제출수 | 우리 대비 모양 프리미엄 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 연식2 | 0.87964 | 0.46767 | 0.67365 | 146 | **+0.0054** |
| 2 | 서울대 노원캠퍼스 | 0.88751 | 0.45718 | 0.67234 | 101 | −0.0225 |
| 4 | 늙코코 | 0.89004 | 0.45318 | 0.67161 | 67 | −0.0321 |
| 5 | J4_208 | 0.87916 | 0.46037 | 0.66977 | 125 | −0.0008 |
| 8 | GBSNU | 0.87932 | 0.45473 | 0.66703 | **23** | −0.0068 |
| 12 | AI친놈 | 0.87354 | 0.45451 | 0.66403 | 51 | **+0.0057** |
| 31 | 한다리 | 0.87991 | 0.43952 | 0.65971 | 68 | −0.0234 |
| 64 | koeun | 0.87339 | 0.43593 | 0.65466 | **1** | −0.0125 |
| 100 | BLUEE | 0.87336 | 0.42855 | 0.65096 | 85 | −0.0199 |

**top-100 중 우리 밴드배치선 위에 있는 팀은 9팀뿐이고, 최대 프리미엄이 +0.0057이다.**
(중앙값 프리미엄 −0.0140, 최소 −0.0328.)

`[산출]` κ 민감도 — 이 결론은 κ 가정에 얼마나 의존하는가:

| κ | 우리 선 위 팀수 /100 | 중앙 프리미엄 | Total 0.66에 필요한 1-NMAE |
|---:|---:|---:|---:|
| 1.0 | 78 | +0.0053 | 0.88130 |
| 1.5 | 41 | −0.0027 | 0.87680 |
| 2.0 | 18 | −0.0107 | 0.87379 |
| **2.21 (실측)** | **9** | **−0.0140** | **0.87281** |
| 2.5 | 3 | −0.0186 | 0.87165 |
| 3.0 | 0 | −0.0261 | 0.87004 |

κ가 1.0 만큼 낮아야만 "우리 밴드가 뒤처진다"는 해석이 성립하는데, κ=1.0이면 Total 0.66에 1-NMAE 0.8813이
필요해 **필드 상위 5% 수준의 정확도**가 요구된다. 어느 쪽이든 **필요한 행동은 같다 — 정확도**.

`[산출]` **필요량.** 우리 밴드 배치 성능이 유지된다는 가정(κ=2.21) 하에:

| 목표 Total | 필요 1-NMAE | (증분) | 그때 예상 FICR |
|---:|---:|---:|---:|
| 0.65971 (구 rank-20 앵커) | 0.87263 | +0.01386 | 0.44679 |
| 0.66000 | 0.87281 | +0.01404 | 0.44719 |
| 0.66703 (현 8위) | 0.87719 | +0.01842 | 0.45687 |
| 0.67365 (현 1위) | 0.88132 | +0.02254 | 0.46598 |

필요 1-NMAE 0.8728 은 **현 LB의 p25(0.87266)와 p50(0.87425) 사이**다. 즉 "필드 중위권의 평범한 GBM 정확도"만
확보하면, 우리가 이미 가진 밴드 배치로 0.66을 넘는다. `[산출]`

**보강 관측 (매우 중요).** 리더보드 64위 `koeun`은 **제출 1회**로 0.65466 (1-NMAE 0.87339)을 찍었고,
37위 `dnwnfptlvl`은 4회로 0.65738, 8위 `GBSNU`는 23회로 0.66703(전체 3위권 FICR)을 찍었다.
**이 대회는 LB 프로빙 물량으로 만드는 점수가 아니다.** 한 번에 0.8734가 나온다는 것은
그 정확도가 "특별한 기전"이 아니라 **표준적인 전처리 + 표준적인 GBM의 수렴점**이라는 뜻이다.

---

## §C 공식 자료 (원문 확인)

### C-1. 대회 정체 — 사이트가 특정되었다 `[원문]`

https://dacon.io/competitions/official/236727/overview/description

> `"[주최 / 주관] 주최/주관: 한국동서발전, GS E&R, 태백가덕산풍력발전 운영: 데이콘"`

즉 **강원 태백시 원동 가덕산 능선(해발 약 1,078 m)의 태백가덕산풍력발전단지**다. `[스니펫]`
- `"위치: 강원도 태백시 원동 산 97 일원 시설 규모: 43.2MW(3.6MW × 12기)"` — https://www.erc.re.kr/webzine/vol36/sub13.jsp
- `"태백가덕산풍력발전은 해발 1078m의 가덕산 고지에 2단계에 걸쳐 조성됐다. 2020년 12월 상업 가동을 시작한 1단계 사업에는 베스타스가 제작한 3.6메가…"` — https://v.daum.net/v/20251109120122508
- `"가덕산 능선을 따라 줄지어 선 거대한 풍력발전기가…"` / `"11km 길이로 설치돼 있다"` — https://www.electimes.com/news/articleView.html?idxno=361590 , https://dealsite.co.kr/articles/151191

→ **능선을 따라 11 km에 걸쳐 늘어선 해발 1,000 m대 산악 능선 단지.** 이것이 §F의 지형 축의 물리적 근거다.

### C-2. 평가식 원문과 정산 단가표 `[원문]`

https://dacon.io/competitions/official/236727/overview/evaluation

> `"평가 산식 : 총점(Score) = 0.5 x 평균 예측오차율(1-NMAE) + 0.5 x 정산금획득률(FICR)"`
> `"평가는 실제 발전량이 설비용량의 10% 이상인 시간대만 대상으로 합니다."`
> `"그룹별 NMAE = 평균( |예측 발전량 - 실제 발전량| / 그룹 설비용량 )"`
> `"그룹별 FICR = 획득 정산금 / 이론상 최대 정산금"`
> `"Public Score : 전체 평가 데이터 중 사전 샘플링된 40% / Private Score : 전체 평가 데이터 중 나머지 60%"`

평가 탭의 단가표는 **HTML에 base64 PNG로 인라인**되어 있어 텍스트 크롤로는 안 잡힌다. 디코딩해 판독한 결과:

| nMAE 구간 | 정산 기준 |
|---|---|
| 6% 이하 | 4원/kWh 정산 |
| 6% 초과 ~ 8% 이하 | 3원/kWh 정산 |
| 8% 초과 | 정산금 없음 |

**우리 `src/baram/evaluation/official.py` 구현(4/3/0, actual≥0.1·cap, actual 가중)과 완전히 일치한다.**
공식 산식 노트북도 코드공유 14035에 있다: https://dacon.io/competitions/official/236727/codeshare/14035 `[원문]`

2차(발표) 평가 배점표도 같은 방식으로 박혀 있었다: 과제 이해도 20 / 기술 우수성 30 / 문제 해결력 15 /
적용 가능성 20 / 발표 완성도 15 = 100점. `[원문, 이미지 판독]`

### C-3. 공개 리더보드 실측 `[원문]`

https://dacon.io/competitions/official/236727/leaderboard — 서버사이드 렌더로 **top-100 전 행**을 받았다.
원본은 `S12_leaderboard_snapshot_2026-08-07.csv`. 기준 시각: 종료까지 D-7 표기 시점(2026-08-07).

- top-100 통계: 1-NMAE mean 0.87530 / sd 0.00391 / min 0.86777 / max 0.89004
- FICR mean 0.43959 / sd 0.00852 / min 0.42665 / max 0.46767
- 제출수 mean 82.3 / min 1 / max 162
- **주의: 프로젝트 AGENTS.md의 "rank-20 = 0.65971" 앵커는 낡았다.** 0.65971은 현재 **31위**다.
  현재 rank-20 = `유채원` 0.66125, rank-30 = `하경민` 0.65977.

### C-4. 규칙 — 기준시점·외부데이터·사전학습모델 `[원문]`

https://dacon.io/competitions/official/236727/overview/rules

> `"각 예측 대상일의 발전량은 해당 일자의 전일 14:00 (KST)을 예측기준시점으로 합니다."` (8/3 17:00 FAQ로 추가 기재)
> `"데이터의 사용 가능 여부는 데이터가 가리키는 대상 시각이 아니라, 해당 데이터가 생성·공개·확정되어 실제로 활용 가능해진 시각을 기준으로 판단합니다."`
> `"파생 변수, 통계값, 보간값, 집계값 등을 생성하는 경우에도 예측기준시점 이후의 정보가 포함되어서는 안 됩니다."`
> `"누구나 접근 가능한 공개 데이터여야 합니다."`
> `"2026년 7월 6일 전, 즉 2026년 7월 5일까지 공식적으로 가중치가 공개된 오픈소스 모델만 사용할 수 있습니다."`
> `"평가 데이터셋은 제출 파일(예측 결과) 생성을 위한 추론 목적으로만 사용할 수 있습니다."`

→ **우리 AGENTS.md의 외부데이터 정책과 일치한다.** 다만 마지막 조항은 우리가 명시적으로 기록해 두지 않은 것으로 보인다:
평가기간 LDAPS/GFS 를 **판별기·피처선택·적대적 검증에 쓰면 규칙 위반**이다(경쟁팀도 이걸 뒤늦게 발견하고 축을 철회했다, §D-1).

### C-5. 주최측 사전설명회 — 주최자가 밴드 전략을 명시적으로 승인했다 `[자동자막]`

영상: https://www.youtube.com/embed/q0zQX_9f_7Y (공식 description 탭에 iframe 임베드).
자동생성 한국어 자막 891조각 전문을 받았다 (`S12_briefing_transcript_q0zQX_9f_7Y.json`).
인식 오류가 심하므로 **원문 그대로 인용하고 해석을 병기**한다.

1. **밴드 우선 전략 승인** — 이것이 조직위의 공식 입장이다.
   > `"물론이 두 산식이 코렐루레이션이 있겠죠. 이제 당연히 5차가 쪄야 정상금도 많이 받을 테니깐요. 그렇지만 이제 정상금 관점에서는 약간 어 M&를 조금 포기하는 선에서도 정상금을 극대화하는 이런 전략도 있을 거고요. 이런 것들은 모두 여러분들의 전략을 설정해 달려 있습니다."`
   (해석: "두 산식은 상관이 있다. 정확도가 좋아야 정산금도 많이 받는다. **그렇지만 정산금 관점에서는 NMAE를 조금 포기하는 선에서도 정산금을 극대화하는 전략도 있을 것이고, 이는 여러분의 전략 설정에 달려 있다**.")
   → **밴드 최적화는 합법이며 주최측이 예상하고 있는 전략이다.** 우리가 하는 정책 스캔은 규칙상 안전하다.

2. **10% 게이트를 전략에 반영하라고 명시**
   > `"실제 배정량이 네, 그룹 설비용량 즉 어 이룹에서 최대로 어 발전할 수 있는 용량의 10%가 안 되는 경우에 그러니까 그런 시간대에는 평가에서도 아예 고려를 하지 않습니다. 네. 이거는 반드시 좀 유의를 해 주시고 전략에도 좀 반영할 만한 그런 내용이죠."`

3. **min-max 클리핑이 채점기에 들어간다**
   > `"그리고 이제 민맥스 클리핑 같은 경우에도 다 적용이 될 예정이라서"`

4. **지형 데이터 사용을 명시적으로 권장**
   > `"저는 약간 지형 데이터도 좀 써 보고 싶은데요. 이런 것도 모두 다 열려 있습니다."`

5. **Public/Private 40:60이며 두 표본 모두 2025 전체 특성이 균등하게 담기도록 층화 구성**
   > `"일단은 40대 60비율로 퍼블릭 프라이빗 40대 60비율로 구성될 거고요. 이거는 뭐 대회 시작 시에 대페이지에도 공개되는 내용이고 대페이지에 공개되지 않는 내용은 이제 두 샘플 모두 이제 25년 전체 개선과 발정량 특성이 퍼블릭 프라이빗 다 좀 균등하게 담기도록 구성할 예 계획입니다."`
   → **Public↔Private 분포 이동이 작다는 조직위 발언.** Public 0.6375 는 Private 추정치로 대체로 신뢰 가능.

6. **데이터 구조 확인**: 예보 2022–2024 학습 제공, SCADA는 학습기간만, **그룹3 SCADA는 2023년부터**, 터빈 총 17기, 2025년 전체 시간단위 예측.
   > `"그룹 1 2는 2022년부터 이런 스카드 데이터가 있는데요. 어 그룹 3회 같은 경우에는 스카드 데이터가 2023년부터 있습니다."`

7. **2차 평가 환산은 순위가 아니라 점수 비율** — 즉 근소차 추격에도 의미가 있다.
   > `"나는 프라이빗 2등이니까 한 5점씩 45점인가 이건 아닙니다. 어떻게 계산이 되냐면 점수를 기준으로 비율로 계산이 됩니다."`

### C-6. **DACON 코드공유/토크 게시판에는 참가자 게시물이 0건이다** `[원문]`

- 코드 공유 탭 https://dacon.io/competitions/official/236727/codeshare — 게시물 **2건**, 둘 다 DACON 공식:
  - `[Baseline] 기상 예보 데이터 기반 RandomForest 풍력발전량 예측` (id 14031, DACONIO, 2026-07-03, 조회 4,161)
  - `평가 산식 코드` (id 14035, DACON.GM, 2026-07-06, 조회 5,171)
- 토크 탭 https://dacon.io/competitions/official/236727/talkboard — 게시물 **3건**, 전부 DACON 공식 운영 공지
  (`DAKER! 대회 관련 안내` 416829, `(6/25) 사전 워크숍 안내` 416912, `사전 워크샵 현장 중계`).
  참가자 질문·인사이트 스레드 **0건**.

> **이것이 이 레인의 두 번째 핵심 발견이다.** "상위 팀이 코드공유로 흘린 기전"은 **존재하지 않는다.**
> 0.649~0.674 구간에 100팀 이상이 몰려 있는 것은 공유 노트북 때문이 아니다.
> 경쟁팀 문서(§D-1)의 표현을 그대로 빌리면:
> `"공개 코드는 공식 RF 베이스라인뿐 — 0.649~0.659 에 30팀이 몰린 것은 공유 노트북이 아니라 \"잘 만든 GBM 이 수렴하는 지점\" 임."`

---

## §D 발견된 해법·저장소 전수 표

### D-0. 총괄표

| # | 출처 | 종류 | 점수 (출처가 주장) | 기전 요약 | URL | 등급 |
|---|---|---|---|---|---|---|
| 1 | hoonbari-S2/wind_project | 경쟁팀 공개 전략문서(85 KB) + 코드 | LB **0.641458** (v13, 당시 ~200위) | 통합학습(G1·G3) + Tweedie + 전역 단조 후처리. **7전 7패 실패 목록이 본체** | https://github.com/hoonbari-S2/wind_project | `[원문]` |
| 2 | shaun0927/BARAM-2026 | 경쟁팀 공개 실험 저장소(267파일) | 로컬 0.63149, **Public LB 0.62209** | **CatBoost quantile q=0.575/0.60 + 전역 α≈1.03 상향** 앙상블 | https://github.com/shaun0927/BARAM-2026 | `[원문]` |
| 3 | kohwoohyun/wind_power_forecast | 참가자 공개 README (10.5 KB) | LB CatBoost **0.61090** | 150피처 FE·EDA 전문. **LDAPS 10m > GFS 100m** 상관 실측 | https://github.com/kohwoohyun/wind_power_forecast | `[원문]` |
| 4 | DelosIndustry/BARAM26-WindTurbine | 참가자 공개 코드 + 리포트 | 미기재 | **그룹3 전용 캘리브레이션** 스윕(월별/광역) + LGBM | https://github.com/DelosIndustry/BARAM26-WindTurbine | `[원문 트리]` |
| 5 | Chankyu99/DACON_BARAM3 | 참가자 저장소(전처리 parquet 포함) | 미기재 | 제출 파일명이 기전을 노출: `submission_ablation_g3_bias400`, `_g3_bias1200`, `_g3_uniform60_linear40`, `_exp2_oof_postprocess` | https://github.com/Chankyu99/DACON_BARAM3 | `[원문 트리]` |
| 6 | Dacon-Organization/baram-2026-wind-power-forecasting | 참가자 저장소 (139파일, 실험로그 대시보드) | 미기재 | 실험 로그 제목이 축을 노출: turbine-spatial-pooling / wind-vector-feature / cross-year-fold / local-public-calibration / gbm-lab | https://github.com/Dacon-Organization/baram-2026-wind-power-forecasting | `[원문 트리]` |
| 7 | SweetFriedPotato/BaramEuron | 참가자 저장소 (Euron 팀) | 미기재 | RF/MLP/GRU 스모크 + spatial/wind/sequence 피처 모듈 | https://github.com/SweetFriedPotato/BaramEuron | `[원문 트리]` |
| 8 | dh0728/decon-BARAM2026 | 참가자 저장소 | 미기재 | 데이터 명세 정리 수준 | https://github.com/dh0728/decon-BARAM2026 | `[원문]` |
| 9 | Eggtakk/baram2026-wind-forecast | 참가자 저장소 (골격만) | 미기재 | 빈 스캐폴드 | https://github.com/Eggtakk/baram2026-wind-forecast | `[원문 트리]` |
| 10 | TACTICS-YJH/Wind_Power_Prediction_contest | **전년도 수상작 저장소** | 2025 BDA×동서발전 풍력 공모전 **7등/272팀 우수상** | README 31바이트, **내용 비공개** | https://github.com/TACTICS-YJH/Wind_Power_Prediction_contest | `[전문미확인]` |
| 11 | DACON 236066 (풍력 발전량 예측 AI 해커톤, 2023) | 다른 대회 코드공유 | Private 1위 / Public 1위 | **ExtraTrees + 특성 공학** (평가 MAE, 정산금 없음) | https://dacon.io/competitions/official/236066/codeshare | `[원문 목록]` |
| 12 | 〃 | 〃 | public 5위 / private 5위 | **AutoGluon** | 〃 | `[원문 목록]` |
| 13 | 〃 | 〃 | public 6위 / private 3위 | **FLAML** | 〃 | `[원문 목록]` |
| 14 | 〃 | 〃 | private 24위 | RandomForest + ExtraTrees (1:4) | 〃 | `[원문 목록]` |
| 15 | HiddenBeginner/2022_oibc_competition | POSTECH OIBC 2022 **대상 (1/64팀)** | — | **WaveNet 기반 확률모델(평균·표준편차 출력)**, 구간예측 과제 | https://github.com/HiddenBeginner/2022_oibc_competition | `[원문]` |
| 16 | lhk6565/2022_POSTECH_OIBC | OIBC 2022 장려상 (7/63팀) | — | Boosting + Moving Average 하이브리드 | https://github.com/lhk6565/2022_POSTECH_OIBC | `[원문]` |
| 17 | baek2sm 블로그 | POSTECH OIBC 2021 **본선 1위** | `"최저 오차율, 최다 일별최고입찰상 획득, 최다 인센티브 획득"` | 본문 미열람 | https://m.blog.naver.com/baek2sm/222476649147 | `[스니펫]` |
| 18 | fubabaz.tistory.com/50 | 블로그 (236066 대회) | `"점수가 좋게 나오진 않았다"` | AutoML(mljar) 시도기 | https://fubabaz.tistory.com/50 | `[스니펫]` |
| 19 | daker.ai 커뮤니티 글 4편 | AI 생성 홍보성 커뮤니티 글 | 없음 | **기전 정보 0.** 역할분담·워크숍 안내 수준 | https://daker.ai/community/baram-2026-wind-power-prediction-ai-competition-5- 외 | `[스니펫]` |

**블로그(velog/tistory/brunch/medium) 조사 결과: 본 대회(236727)에 대한 해법·점수 기술 글은 0건.**
검색 6종(velog/tistory/후기/회고/FICR/LightGBM 조합)에서 전부 대회 홍보 페이지만 반환됐다.

### D-1. hoonbari-S2/wind_project — 가장 값진 외부 증거 `[원문]`

경쟁팀이 `HANDOFF.md`(8.1 KB) + `project_strategy.md`(85 KB, 114개 절) + `_INDEX.md` 를 **공개 저장소에** 올려 두었다.
공개 LB 전 제출 이력과 실패 판정이 전부 들어 있다.

**공개된 LB 이력 (public):**

| 제출 | total | 1−NMAE | FICR | 내용 |
|---|---:|---:|---:|---|
| v11 | 0.636316 | 0.864890 | 0.407743 | 정제 타깃 |
| v12 | 0.636871 | 0.864932 | 0.408809 | |
| **v13** | **0.641458** | **0.867673** | **0.415244** | **통합학습(G1·G3). 그들의 최고점** |
| v14 | 0.629475 | 0.862562 | 0.396389 | MAE + 발전량가중 손실 |
| v13-nopost | 0.620418 | **0.872184** | 0.368653 | **후처리 제거. 정확도가 올라간다는 계측** |
| v15 | 0.640428 | 0.866593 | 0.414263 | 격자 산포 54피처 |
| v13-bag25 | 0.639500 | 0.867775 | 0.411226 | 후처리 적합 배깅 |
| v17-q50post | 0.629282 | 0.859864 | 0.398701 | q50 분위회귀 |

**그들이 실측으로 닫은 축 (우리가 재현할 필요가 없는 실패 목록):**

> `"| 후처리 함수공간 (해상도·버킷·산포변조·그룹분리) | 폐기 |"`
> `"| 손실 정렬 (MAE=v14, 분위 q50=v17) | 폐기 — LB −0.012 두 번 재현. 범인은 W 아닌 L |"`
> `"| 기대효용 배치 (B-2b placement) | 폐기 (OOF −0.0115, 정지 왼꼬리 부재) |"`
> `"| 격자 산포 피처 54개 (v15) | 폐기 (OOF 3/3 통과 후 LB −0.0010) |"`
> `"| 드리프트 대응 3종 | 폐기 — 3전 3패. 사상은 연도 안정, 데이터 양 > 최신성 |"`
> `"| 외부 관측 (ASOS, v20/v20b) | 폐기 — v20b 강제포함 실측 −0.0057 (3/3 음수, 비율 4.07, 그룹 0/3) |"`
> `"| max_depth 상향 (v22) | 폐기 — −0.0104 (3/3, 비율 6.5). 표현력은 병목이 아니고 깊이 6 이 이미 상한 |"`
> `"| A-4 격자 가중 (v21) | 폐기 — Δ −0.0002 판별 불가. gain 1위·비중 19% 인데 점수 0 => NWP 유도량 축 소진 |"`
> `"| 시드 배깅 · 설정 앙상블 E1~E5 · 폴드 혼합 | 폐기 |"`
> `"| 가용률 기후 보정 | 폐기 — 정지가 터빈 양자라 중앙값 재구성(현행)이 이미 최적 |"`

**그들의 핵심 판정 3개 (우리 프로젝트 소견과 놀랍도록 일치):**

1. `"역대 LB 상승은 전부 정보형이었음. v11(정제 타깃), v12, v13(통합 학습)만 LB 를 올렸고, 그 뒤 정규화형·정산구간형·손실정렬형·피처형 시도는 전패임."`
2. `"정산구간형(후처리 적합 변경)의 LB 전이는 감쇠가 아니라 부호 반전임. 관측 4건 … 전부 음수."`
   `"bag25 의 분해가 결정적임 — 같은 raw 예측(Δ정확도 +0.0001)에 곡선만 바꿨는데 FICR −0.0040."`
3. `"우리 OOF 가 '문턱 과적합' 이라 부른 그 날카로운 조정이 2025 public 에서는 오히려 벌고 있음. 2022–24 로 잰 '일반화' 와 2025 로의 일반화가 다른 방향임."`

**그들이 잰 FICR 기하 (§3.24) — 이 레인의 κ 출처:**

> `"전환율 dFICR/d(1−NMAE) = 2.21 (LB 작동점 보간). 정확도 1점 = 총점 1.61점."`
> `"1위(연식2)는 정확히 우리 κ-곡선 위에 있음. 정확도 우위 +0.0172 가 자동으로 끌고 오는 FICR = +0.0380, 실제 +0.0382. 모양 프리미엄 +0.0002 ≈ 0. 1위는 그냥 더 정확함."`
> `"합성데이터의 8~9 는 우리 분포가 아니었음 — 실분포는 꼬리가 두꺼워(채점행의 34.7%가 오차 15%cap 초과) 문턱 근처 질량이 얇고, 그래서 전환율이 낮음."`
> `"근접실패(6~10%)의 과대예측 비중이 전 p 구간에서 52~65% 로 거의 대칭 — 단조든 조건부든 이동으로는 근접실패를 못 고침."`

**의미:** 우리 프로젝트가 이미 실측한 "직접 밴드 적중 추정 폐기", "post-processing oracle 천장", "3.2배 오프셋 비전이"와
**독립적으로 도달한 같은 결론**이다. 밴드 축은 이 대회에서 실제로 닫혀 있다.

### D-2. shaun0927/BARAM-2026 — **살아 있는 기전 2개** `[원문]`

`experiments/ficr_postprocessing/results/conclusion.md`:

> baseline: `score 0.6117673, one_minus_nmae 0.8700287, ficr 0.3535060`
> best: `A1_global_1.0275` → `score 0.6207260 (Δ +0.0089586), one_minus_nmae 0.8711552 (Δ +0.0011266), ficr 0.3702967 (Δ +0.0167907)`
> `"verdict": "Strong accept", "reason": "passes strong score/FiCR/nMAE/worst-month criteria"`

**전역 배율 α ≈ 1.0275–1.03 (예측값을 2.75~3% 일괄 상향)이 1-NMAE와 FICR을 동시에 개선한다.**
즉 이 데이터에서 표준 회귀기는 **계통적으로 과소예측**한다. 이건 밴드 트릭이 아니라 편의(bias) 교정이다.

`experiments/ensemble_diversity_optimization/results/conclusion.md` (45개 모델 스크리닝):

> `"The dominant source of improvement is not a low-correlation deep-learning model. It is a CatBoost Quantile family shift: q=0.575 and q=0.60 variants are materially stronger single models than the previous q=0.55 candidate."`
> `"The best ensemble uses catboost_quantile_0575_depth6, catboost_quantile_060_depth6, and lgbm_l1_baseline. This is more of a strong quantile-family ensemble than a broad residual-diversity ensemble."`
> `"Low-correlation models still exist, but they did not become core candidates because their standalone score remains too weak. TabM/MLP/kernel/SVR/neighbors remain diagnostic diversity branches"`
> `"AutoGluon/PyCaret longer-budget runs were feasible, but they did not become top candidates."`
> `tabpfn_regressor | TabPFN | failed`

`experiments/error_decomposition/results/next_audit_routing.md` — **원인 진단**:

> `"weighted target-level underprediction rate: 0.3638"`
> `"mid-generation underprediction rate (10~80%): 0.4568"`
> `"high-generation underprediction rate (80%+): 0.7908"`
> `"Mid-generation bins dominate total impact, but high-generation bins have severe underprediction and q=0.60 slice wins."`

슬라이스별 손실 지도(원문 표):

| 그룹 | actual 구간 | 건수 | 정규화 MAE | FICR 실패율 | 과소예측률 | impact |
|---|---|---:|---:|---:|---:|---:|
| g2 | 30-60% | 1462 | 0.1565 | 0.699 | 0.423 | 228.8 |
| g1 | 30-60% | 1484 | 0.1401 | 0.660 | 0.436 | 207.8 |
| g3 | 30-60% | 1448 | 0.1430 | 0.689 | 0.422 | 207.0 |
| g3 | 90-100% | 438 | **0.2179** | **0.975** | **1.000** | 95.4 |

**핵심:** 정산금 손실의 대부분은 **30–60 %cap 중출력 구간**(FICR 실패율 0.66–0.70)에 있고,
**고출력 구간은 예외 없이 과소예측**된다(g3 90–100 %에서 과소예측률 **1.000**, FICR 실패율 0.975).

**검증창 판정** (`cv_protocol/results/conclusion.md`):
> `W4_2022_2023_to_2024` (전 이력) verdict `Adopt`, `"best mean B1/B2 score and passes material/stability/group criteria"`
> `W3_2023_to_2024` (최근만) 은 −0.0078 로 열등. `W5_recency_weighted` 도 W4보다 열등.
> `"full-history is supported for group 1/2, but needs public-LB calibration; group 3 remains recent-window because 2022 labels are absent."`
→ **최근성 가중은 손해. 데이터 양이 이긴다.** (hoonbari의 드리프트 3전3패와 독립적으로 일치.)

### D-3. kohwoohyun/wind_power_forecast — EDA 실측치 `[원문]`

**풍속-발전량 상관 (그룹별):**

| 소스/고도 | G1 | G2 | G3 |
|---|---:|---:|---:|
| GFS 10m | 0.585 | 0.597 | 0.588 |
| GFS 80m | 0.597 | 0.611 | 0.605 |
| GFS 100m | 0.601 | 0.615 | 0.609 |
| **LDAPS 10m** | **0.727** | **0.737** | **0.731** |

> `"세 그룹 모두 동일한 패턴: 고도가 아니라 공간해상도(LDAPS 1.5km) 가 결정적이었습니다. GFS는 25km 격자라 산악지형(태백)의 국지풍을 잘 못 잡아내는 것으로 보입니다."`

**격자 표준편차가 강한 신호:**

| 변수 | G1 | G2 | G3 |
|---|---:|---:|---:|
| `heightAboveGround_5_YBLWS_std` | 0.668 | 0.673 | 0.658 |
| `meanSea_0_prmsl_std` | 0.572 | 0.581 | 0.558 |
| `50_50MUmax_std` | 0.553 | 0.545 | 0.494 |

**LDAPS 50 m U성분이 V성분을 압도:** `50MUmax_mean` 0.681/0.671/0.664 vs `50MVmax_mean` 0.270/0.292/0.318
> `"U성분(동서방향)이 V성분(남북방향)보다 세 그룹 모두 훨씬 강함 — 이 지역 지배풍이 동서 방향에 가깝다는 신호"`

**세제곱 변환은 오히려 나빠진다** (파워커브 포화·컷아웃 구조 때문). `"세제곱 feature는 의도적으로 넣지 않았"`음.

**계절별 파워커브 = 공기밀도 효과 실측:**
> `"중간 풍속(3~10m/s대) : 세 그룹 모두 겨울이 봄가을·여름보다 뚜렷하게 높음 (예: 그룹1 4.5~5.6m/s 구간에서 겨울 9,482 vs 여름 6,919)"`
→ 같은 풍속에서 겨울 발전량이 여름의 **1.37배**. 공기밀도(=P/RT) 피처의 직접 근거.

**로컬↔LB 순위 역전 사례 (검증설계 교훈):**
> `"XGBoost가 로컬 2위에서 리더보드 최하위로, RandomForest가 로컬 최하위에서 리더보드 4위로 뒤바뀌었습니다. 기존 검증 기간(2024년 10~12월 3개월)이 계절이 한쪽으로 치우쳐 있어…"`
→ 그들이 **매년 1·4·7·10월 계절대표 fold**로 바꾸자 LB 순위와 일치했다.

**공식 베이스라인의 실제 LB 점수:** `베이스라인 0.58792 (1-NMAE 0.86371, FICR 0.31213)` `[원문]`
→ **공식 RF 베이스라인만으로도 1-NMAE 0.86371 이 나온다. 우리 M266의 0.85878보다 높다.** `[산출]`

### D-4. 그룹3 전용 보정이 필드의 공통 축이다 `[원문 트리]`

- `DelosIndustry/BARAM26-WindTurbine` — `reports/group3_calibration/`, `group3_calibration_wide/`, `group3_monthly_calibration/`,
  `scripts/tune_group3_calibration.py` (17 KB), `scripts/apply_submission_calibration.py`
- `Chankyu99/DACON_BARAM3` — `submission_ablation_g3_bias400.csv`, `submission_exp2_postprocess_v4_g3_bias1200.csv`,
  `submission_ablation_g3_uniform60_linear40.csv`, `submission_ablation_g3_exact_weighted_blend.csv`
- shaun0927 실측: g3 90–100 %cap 구간 **과소예측률 1.000 / FICR 실패율 0.975**
- hoonbari §3.20 제목: `"G3 는 왜 고풍속에서 무너지는가 — 기종이 아니라 배치다"`

→ **최소 3개 독립 팀이 그룹3(UNISON U136, 21.0 MW, 2023~ 라벨)에 별도 보정을 걸고 있다.**

---

## §E 같은 계단형 정산 지표를 쓴 선행 대회

**POSTECH OIBC CHALLENGE** (에이치에너지 × POSTECH 오픈이노베이션빅데이터센터)가
**한국 재생에너지 발전량 예측제도의 정산금을 그대로 목적함수로 쓰는 유일한 선행 대회**다.

공식 평가식 원문 (2023 제5회 규칙 PDF, https://o.solarkim.com/docs/cmpt2023/rule.pdf) `[원문 PDF]`:

> `"예측 인센티브는 시간대별 예측오차율에 따라 아래와 같이 차등 산정"`
> `"시간대별 예측오차율 6% 이하 : I_h = 4원/kWh"`
> `"시간대별 예측오차율 6% 초과 ~ 8% 이하 : I_h = 3원/kWh"`
> `"시간대별 예측오차율 8% 초과 : I_h = 0원/kWh"`
> `"예측 일자의 정산금은 (발전소의 시간대별 발전량 × 인센티브) 의 합으로 계산"`  → `Σ_h G_h × I_h`
> `"단, 설비 이용률 10% 미만 시 오차율 산정에서 제외되어 인센티브는 지급되지 않음"`

**→ BARAM 2026 의 FICR 은 이 제도의 축자 이식이다** (4/3/0원, 6%/8% 문턱, actual 가중, 10 % 이용률 게이트).
채점기 재구현이 맞다는 것을 제3자 원문으로 교차확인한 셈이다.

**그런데 OIBC 상위해법에서 "밴드 적중 트릭"은 발견되지 않았다.** `[원문/스니펫]`
- 2022 대상 Sun Capturer: `"WaveNet 기반 확률 모델 (태양광 발전량의 평균과 표준편차 출력)"` — 분포를 내되 배치 트릭은 언급 없음
  (단 2022회차는 **구간예측** 과제라 산식이 다르다).
- 2021 본선 1위(baek2sm): `"최저 오차율, 최다 일별최고입찰상 획득, 최다 인센티브 획득으로 모든 부문에서 가장 좋은 결과"` `[스니펫]`
  → **오차율 1등이 인센티브도 1등.** 밴드와 정확도가 갈라지지 않았다.
- OIBC 3·5회 심사: `"채점 기준: 정확도 (50%) + 창의성 (30%) + 발표 자료(20%)"`, `"정확도 평가: 경진대회 기간 동안 획득한 인센티브 총합"` `[스니펫]`

**제도 문헌 쪽에서 발견된 유일한 "밴드 획득 기법"은 예측 조작이 아니라 ESS 물리 제어다** `[스니펫]`:
> `"태양광 발전량이 과소 예측되어 실제 발전량이 예측값보다 과도하게 높게 나타나는 경우, ESS 충전을 통해 예측오차의 감소와 인센티브의 수혜 가능성을 높일 수 있을 것이다."`
> — 이재희 외, 「재생에너지 발전량 예측제도를 고려한 ESS 연계형 태양광 발전의 발전량 입찰 계획 기법」, 전기학회논문지 2022. http://www.tkiee.org/kiee/XmlViewer/f415504
→ **우리 세팅(사후 CSV 제출)에는 적용 불가.** 실무에서도 "예측값을 밴드에 맞춰 옮기는" 기법은 문헌에 없다.

**국내 실무 정확도 벤치마크** `[스니펫]` — 우리가 겨루는 수준의 감:
- LS일렉트릭 제주 풍력 실증 `"통상적으로 10% 수준인 예측 오차율을"` → 8 %까지 (https://www.electimes.com/news/articleView.html?idxno=365477)
- 브이피피랩 `"발전량 예측 오차율은 5.2%를 기록했다. 이는 전국 최저 수준의 풍력 발전량 예측 오차율이다."` (https://v.daum.net/v/20260729182551061)
- 우리 M266 NMAE = 14.12 %cap. **필드 top-100 은 12.0~13.2 %cap.**

---

## §F 국내 학술 — LDAPS 산악지형 풍속의 계통오차

### F-1. KMAPP / LDAPS 산악 풍속 편의 `[원문 PDF]`

금왕호·이상현·이두일·이상삼·김연희 (2021), 「복잡 지형 지역에서의 KMAPP 지상 풍속 예측 성능 평가와 개선」,
*대기* 31(1) 85-100. https://j-komes.or.kr/xml/28740/28740.pdf

> `"The one-month wintertime forecasts revealed that the operational Local Data Assimilation and Prediction System (LDAPS) has systematic errors over the complex mountainous area, especially in deep valley areas, due to the orographic smoothing effect."`
> `"LDAPS 모형의 경우, 지형 고도가 낮은 협곡 지역의 지형 고도는 실제 고도보다 높게 나타나며, 지형 고도가 높은 산악 지역의 지형 고도는 실제 고도보다 낮게 나타난다. 이는 모형 격자 해상도 영향과 산악 지형 지역에서의 수치 불안정을 피하기 위해 적용된 지형 평활화 과정의 영향으로 분석된다"`
> `"내삽된 예측 바람장은 해당 예측 지점의 지표 특성을 반영한 거칠기 보정(roughness adjustment)와 고도 보정(height correction) 과정을 순차적으로 거친 후 100m 수평 해상도의 최종 예측 풍속이 산출된다. 물리 보정 과정은 Howard and Clark (2007)에 제시된 방안을 기반으로 하며"`
> `"The KMAPP-Wind system showed better performance in predicting near-surface wind speed during the ICE-POP period than the original KMAPP version, reducing the forecast error by 21.2%. It suggests that a realistic representation of the topographic parameters is a prerequisite for the physical downscaling of near-ground wind speed over complex terrain areas."`
> (경고) `"The KMAPP reproduced the orographic height variation over the complex terrain area but failed to reduce the wind speed forecast errors of the LDAPS model. It even showed unreasonable values (~0.1 m s-1) for deep valley sites due to topographic overcorrection."`

**해석 (가덕산에 대한 함의):** 가덕산 터빈은 **해발 1,078 m 능선 정상부**에 있다. LDAPS 1.5 km 격자의 모형 지형은
평활화로 **실제 능선보다 낮게** 잡히므로, 격자값을 그대로 쓰면 로터면 고도의 풍속이 **계통적으로 과소 추정**된다.
이는 §D-2의 `고출력 구간 과소예측률 0.79`, `전역 α≈1.03 이 두 지표를 동시에 개선` 관측과 **부호가 일치**한다.
`[가설·미검증 — 우리 데이터로 확인 필요]`

### F-2. 연직층 축 `[스니펫]`

Keun-Hee Lee, Bongjoon Park 외 (2024), "Day-ahead wind power forecasting based on feature extraction
integrating vertical layer wind characteristics in complex terrain", *Energy* 288:129713.
https://www.sciencedirect.com/science/article/abs/pii/S0360544223031080 (SSRN 프리프린트 https://papers.ssrn.com/sol3/Delivery.cfm/4a1b45b2-47c7-40a9-a2af-25e35d01d69e-MECA.pdf?abstractid=4509803)
> `"This study aims to enhance the quality of wind power forecasts in complex terrains, focusing on identifying and processing appropriate wind …"`
→ 기존 레인 `S6_ext_B_terrain` 이 이미 다룬 축. 여기서는 중복 조사하지 않았다. `[전문미확인]`

### F-3. 참고 (본 레인 범위 밖) `[스니펫]`
- Shin et al. (2022), "High-resolution wind speed forecast system coupling…", *Int. J. Biometeorol.* — 한국 고해상도 풍속 예보 시스템
- WRF Workshop 2017 P43: `"The wind-energy forecasting system is a post processing system based on the KMA local data assimilation and prediction system (LDAPS)."`

---

## §G 검색 로그

웹 검색 46쿼리 전문(제목·URL·스니펫 포함)은 `S12_ext_dacon_solutions.searchlog.json`,
GitHub Search API 14쿼리 전문은 `S12_github_searchlog.json` 에 있다. 쿼리 목록:

**웹 (46):**
1. dacon 풍력 발전량 예측 AI 경진대회 코드 공유
2. 제1회 풍력발전량 예측 AI 경진대회 데이콘 수상작
3. 제2회 풍력발전량 예측 AI 경진대회 데이콘
4. dacon.io competitions official 236727 codeshare
5. 데이콘 풍력발전량 예측 정산금 획득률 FICR
6. 풍력발전량 예측 오차율 6% 8% 정산금 인센티브 예측제도
7. site:dacon.io 236727 코드공유
8. "236727" dacon 풍력 코드
9. github dacon 풍력발전량 예측 BARAM 2026
10. github "open_wind" dacon wind power korea  *(결과 0)*
11. velog 풍력발전량 예측 AI 경진대회 데이콘 후기
12. 제1회 풍력발전량 예측 AI 경진대회 가덕산 동서발전 수상 팀 발표
13. 제2회 풍력발전량 예측 AI 경진대회 BARAM 2025 수상작 발표
14. 가덕산 풍력발전단지 태백 설비용량 V126 U136
15. 한국동서발전 풍력발전량 예측 AI 경진대회 시상식 컨퍼런스 수상팀
16. BARAM 2025 풍력발전량 예측 대회 데이콘 236xxx 리더보드
17. 2025 풍력 발전량 예측 공모전 BDA 동서발전 60Hertz 수상작 방법론
18. 60Hertz 풍력 발전량 예측 공모전 2025 대상 수상 후기
19. BDA 풍력 발전량 예측 공모전 워크숍 자료
20. "풍력 발전량 예측" 공모전 velog 회고 nMAE 정산금
21. 티스토리 풍력 발전량 예측 데이콘 LightGBM 파워커브 후기
22. 데이콘 동서발전 태양광 발전량 예측 AI 경진대회 평가산식 NMAE 설비용량 10%
23. dacon 태양광 발전량 예측 경진대회 1위 솔루션 github NMAE-10
24. 데이콘 전력거래소 발전량 예측 정산금 대회 우승 코드공유
25. github 데이콘 태양광 발전량 예측 동서발전 1등 solution
26. site:github.com dacon wind power prediction korea LDAPS GFS  *(결과 0)*
27. POSTECH OIBC CHALLENGE 태양광 발전량 예측 경진대회 정산금 6% 8% 평가
28. OIBC challenge 2023 태양광 발전량 예측 대상 솔루션 github
29. 2022 OIBC 태양광 발전량 예측 대상 Sun Capturer 솔루션 발표자료
30. 인센티브 정산금 최대화 예측 전략 quantile 최적 예측값 밴드
31. step reward forecasting optimal point prediction interval band hitting decision theory
32. velog 데이콘 BARAM 2026 풍력발전량 예측 대회 후기
33. tistory BARAM 2026 풍력발전량 예측 데이콘 LightGBM FICR
34. LDAPS 국지예보모델 풍력 발전량 예측 복잡지형 연직층 논문
35. 한국풍력에너지학회 논문 LDAPS 풍력발전량 예측 태백 강원
36. 제6회 OIBC challenge 태양광 발전 입찰 인센티브 오차율 평가식
37. OIBC 태양광 발전량 예측 인센티브 최대화 전략 후기 블로그 오차율 6%
38. baek2sm 포항공대 OIBC 태양광 발전량 예측 1위 후기 방법
39. 재생에너지 발전량 예측제도 인센티브 최대화 입찰 전략 논문 최적 예측값
40. wind power forecasting Korea LDAPS bias correction complex terrain paper
41. Lee 2024 Energy LDAPS vertical level wind power forecasting complex terrain Korea
42. daker.ai BARAM 2026 nMAE 정산금 전략
43. 태백 가덕산 풍력 해발고도 능선 표고 풍력단지 위치
44. 60Hertz 풍력 발전량 예측 오차율 nMAE 실적 국내
45. dacon 코드공유 풍력 발전량 예측 AI 해커톤 236066 1등
46. github wind power forecast dacon 2026 catboost quantile FICR

**GitHub Search API (14):** `풍력발전량 예측`(10건) / `dacon 풍력`(0) / `236727`(2, 무관) / `BARAM 2026 풍력`(0) /
`wind power dacon`(0) / `LDAPS 풍력`(0) / `가덕산 풍력`(0) / `풍력 발전량 예측 경진대회`(0) /
`"kpx_group_1"`(0) / `"ldaps_train" OR "gfs_train"`(1, 무관) / `"scada_vestas_train"`(0) /
`BARAM 2026 wind`(**5건**) / `"FICR" 풍력`(0) / `가덕산 wind forecast`(0)

**직접 fetch 한 페이지 (read-only, 다운로드·clone 없음):** DACON 236727 의 description/evaluation/rules/data/
leaderboard(1~7p)/codeshare/talkboard/codeshare-14031/14035/talkboard-416829/416912, DACON 235720·236066,
YouTube q0zQX_9f_7Y 자막, GitHub raw README ×13, GitHub trees API ×9, GitHub raw 문서 ×8,
o.solarkim.com 규칙 PDF, j-komes.or.kr KMAPP PDF, 2025-bda-ewp-60hz-contest.webflow.io, bdaprogram.oopy.io,
LinkedIn 김임용 포스트.

---

## §H 우리가 놓치고 있는 것으로 보이는 기전 — 확신도 순 랭킹

> 확신도 표기: **[확정]** = 원문 실측 수치로 확인 / **[강함]** = 복수 독립 출처 일치 / **[보통]** = 단일 출처 실측 /
> **[가설]** = 물리·산식 논증만 있고 실측 없음

### M-1. 「밴드가 아니라 정확도」 — 전략 축 자체의 오배정 **[확정]**

우리 1-NMAE 0.858775 는 **공개 LB top-100 전원보다 낮다**(최하위보다 −0.00900, 중앙값보다 −0.01548).
공식 RandomForest 베이스라인의 LB 점수조차 1-NMAE **0.86371**(kohwoohyun `[원문]`)로 우리보다 높다.
반면 κ=2.21 기준 우리 밴드 배치는 top-100 중 91팀보다 우수하다.
그리고 우리 밴드 배치를 유지한 채 **1-NMAE 를 0.87281 (필드 p25~p50 사이)로 올리기만 하면 Total 0.66** 이다. `[산출]`

**행동:** 후처리·정책·블렌드 축의 잔여 탐색을 중단하고 예산 전부를 점예측 정확도로 옮긴다.
이는 우리 프로젝트가 이미 실측한 "후처리/블렌드 축 폐쇄"와 모순되지 않고, **그 폐쇄의 필연적 귀결**이다.

### M-2. 계통적 과소예측 보정 — 전역 배율 α ≈ 1.03 **[보통, 즉시 검증 가능]**

shaun0927 실측: `A1_global_1.0275` 가 1-NMAE **+0.0011266**, FICR **+0.0167907**, Total **+0.0089586** `[원문]`.
원인 진단도 함께 있다: `high-generation underprediction rate (80%+): 0.7908`, g3 90–100 %cap 과소예측률 **1.000** `[원문]`.
물리적 근거는 §F-1 (LDAPS 지형 평활화 → 능선 정상부 풍속 과소) `[가설]`.

**우리 상태:** 우리 정책은 `T*_G*` 형태의 임계/게인 조합이라 **전역 곱셈 게인을 이미 포함하고 있을 가능성이 높다.**
그러나 우리가 최적화한 것은 "policy 스캔"이지 "예측 편의 교정"이 아니다.
**검증 방법(모델 학습 0회):** 기존 raw 예측에 α ∈ [0.97, 1.05] 를 곱해 우리 채점기로 재채점하고,
1-NMAE 와 FICR 을 분해해 본다. **α*>1 이면 우리도 과소예측 중이며, 그 자체가 정확도 이득이다.**
`[주의]` hoonbari 는 "정산구간형 후처리는 LB 에서 부호 반전"이라고 판정했다. 그러나 α 는 **후처리 곡선 재적합이 아니라
스칼라 편의 교정**이며, 1-NMAE 를 **동시에** 개선한다는 점에서 그들이 폐기한 부류와 다르다. 반드시 분해해서 읽을 것.

### M-3. 비대칭 분위 타깃 (CatBoost quantile q ≈ 0.575–0.60) **[보통]**

shaun0927: `"It is a CatBoost Quantile family shift: q=0.575 and q=0.60 variants are materially stronger single models"` `[원문]`.
`catboost_quantile_0575_depth6` 단독 alpha_score 0.629741 로 **직전 앙상블 앵커(0.62609)를 단독으로 상회**.
슬라이스 진단에서 g3 80–100 %cap 구간의 승자가 일관되게 `catboost_quantile_060_depth6` (승률 0.51~0.54).

**주의 — 정면충돌하는 증거:** hoonbari 는 `q50` 분위손실로 LB **−0.0122** 를 맞고 `"중앙값/분위 손실 계열 전체 폐기. 재론 금지"` 라고 썼다.
**두 관측은 모순이 아니다.** q50 은 L1(중앙값)이고, q=0.575~0.60 은 **의도적 상향 편의**다.
우리 데이터의 타깃은 우편향(파워커브 포화 + 저출력 최빈)이라 조건부 중앙값 < 조건부 평균이고,
q50 은 과소예측을 **악화**시키는 반면 q0.6 은 **교정**한다. M-2 와 같은 기전의 학습단계 판본이다.

### M-4. 그룹3 전용 처리 **[강함]**

최소 3개 독립 팀이 그룹3 전용 보정을 걸고 있다(§D-4). 실측 근거:
- g3 90–100 %cap: 정규화 MAE **0.2179**, FICR 실패율 **0.975**, 과소예측률 **1.000** (shaun0927 `[원문]`)
- `"G3 는 왜 고풍속에서 무너지는가 — 기종이 아니라 배치다"` (hoonbari §3.20 제목 `[원문]`)
- 그룹3 = UNISON U136 5기, 21.0 MW, Hub 117 m / Rotor 136 m, **라벨 2023~ 만 존재** (kohwoohyun `[원문]`)
- 조직위 확인: `"그룹 3회 같은 경우에는 스카드 데이터가 2023년부터 있습니다"` `[자동자막]`
- 검증창 판정: `"group 3 remains recent-window because 2022 labels are absent"` `[원문]`

**행동:** 그룹3을 그룹1·2와 같은 학습·정책으로 다루고 있다면, 고풍속 구간 게인/편의를 그룹3만 분리해 잰다.

### M-5. LDAPS 격자 **표준편차**를 신호로 쓰기 **[보통]**

kohwoohyun 실측: `heightAboveGround_5_YBLWS_std` 상관 **0.658~0.673**, `meanSea_0_prmsl_std` **0.558~0.581** `[원문]`.
격자 평균만이 아니라 **격자 내 산포**가 발전량과 강하게 연결된다(기압 산포 = 종관 시스템 활동도).

**주의:** hoonbari 는 `격자 산포 피처 54개(v15)` 로 OOF 3/3 통과 후 **LB −0.0010** 을 맞았다 `[원문]`.
그러나 그들의 진단은 `"선택 예산 대체관계"` — 즉 **피처 희석**(812→200 선택)이 원인이지 신호가 없어서가 아니다.
우리가 이미 "grid pivot 914열 실패"를 겪었다면, **std 를 소수(2~3개)만 골라 넣는 형태**여야 한다.

### M-6. 계절 대표 fold 로 검증 재설계 **[강함]**

kohwoohyun: 2024 Q4 홀드아웃에서는 XGB 로컬 2위 → LB 최하위, RF 로컬 최하위 → LB 4위로 뒤집혔고,
**매년 1·4·7·10월 계절대표 fold** 로 바꾸자 LB 순위와 일치했다 `[원문]`.
평가기간이 **2025년 전체(사계절)** 이므로 계절 편향 홀드아웃은 구조적으로 오판을 낳는다.
조직위도 `"두 샘플 모두 25년 전체 기상과 발전량 특성이 균등하게 담기도록 구성"` 이라고 했다 `[자동자막]`.

**행동:** 우리 fold 구성이 시간 블록(연도/분기)이라면, **월-층화 fold** 를 병행해 모델 선택을 다시 잰다.
`[주의]` lag 피처가 없을 때만 안전하다(kohwoohyun 이 그 조건을 먼저 확인했다).

### M-7. 데이터 양 > 최신성 (전 이력 학습) **[강함]**

- shaun0927 `cv_protocol`: `W4(2022+2023→2024)` **Adopt**, `W3(2023→2024)` −0.0078, `W5(recency weighted)` −0.0027 `[원문]`
- hoonbari: 드리프트 대응 3종(연도 제외 / 표본가중 / 편차표현) **3전 3패**, `"사상은 연도 안정, 데이터 양 > 최신성"` `[원문]`
- 우리 memory `시계열 tabular 검증: alpha*는 레짐 경계 탐지기` 와 방향이 어긋나지 않는지 확인 필요.

**행동:** 우리가 최근성 가중이나 연도 필터를 쓰고 있다면 제거 A/B 를 한 번 돌린다.

### M-8. LDAPS 지형 평활화 보정 (거칠기·고도 보정) **[가설]**

§F-1. LDAPS 1.5 km 모형 지형은 **산악 정상부를 실제보다 낮게** 표현한다.
가덕산 터빈은 해발 1,078 m 능선의 Hub 117 m 지점 — 즉 모형이 보는 고도와 실제 로터면 고도의 차 `HHC` 가 크다.
KMAPP-Wind 는 이 정적 지형 파라미터를 고쳐 풍속 오차를 **21.2 %** 줄였다 `[원문]`.

**행동 가능한 최소판:** 외부 DEM 은 규칙 검토가 필요하지만, **`info.xlsx` 의 터빈 좌표·허브고도와
LDAPS 16격자의 상대 위치**만으로도 (a) 능선축 투영, (b) 격자↔터빈 고도차 대용치를 만들 수 있다.
`[주의]` KMAPP 논문 자체가 `"topographic overcorrection"` 으로 협곡에서 0.1 m/s 같은 비현실값을 냈다고 경고한다.
과보정 위험이 실재하며, 논문은 **정상부(mountain)보다 협곡(valley)에서 오차가 크다**고 했다.
우리 사이트는 정상부이므로 논문의 최악 사례는 아니다.

### M-9. 앙상블은 "다양성"이 아니라 "같은 계열 강한 모델의 소수 결합" **[보통]**

shaun0927 이 45개 모델(TabPFN·TabM·MLP·SVR·kNN·EBM·NGBoost·H2O·FLAML·AutoGluon·PyCaret 포함)을 돌린 결론 `[원문]`:
> `"Low-correlation models still exist, but they did not become core candidates because their standalone score remains too weak."`
> `"AutoGluon/PyCaret longer-budget runs were feasible, but they did not become top candidates."`
> `tabpfn_regressor | TabPFN | failed`
최종 채택은 CatBoost-quantile 2종 + LGBM-L1 의 3-멤버 `0.70/0.20/0.10`.

**이는 우리 memory `AutoML SOTA는 자동 피처엔지니어링으로 이기지 않는다`, `Exogenous-only 예보회귀에서 DL 시계열 표현은
정의역 오류 — TabPFN은 규모/라이선스 이중구속` 을 외부에서 재확인해 준다.** 새 축이 아니라 **닫힘 확인**으로 기록한다.
또한 우리가 실측한 `멤버 오차 상관 0.984~0.994` 문제와 같은 그림이다.

### M-10. 규칙 리스크 — 평가 데이터의 비추론 사용 **[확정]**

규칙 유의사항 7: `"평가 데이터셋은 제출 파일(예측 결과) 생성을 위한 추론 목적으로만 사용할 수 있습니다."` `[원문]`
hoonbari 는 "테스트 피처 적대적 검증" 축을 설계 단계에서 **규칙 위반으로 판정하고 철회**했다 `[원문]`.

**우리 점검 대상:** test LDAPS/GFS 를 이용한 (a) 분포이동 진단, (b) 피처 선택, (c) 공분산 시프트 가중,
(d) 정규화 통계 산출이 파이프라인 어디에도 없어야 한다. 2차 심사 소명 항목이다.

### M-11. 상금 산정 구조상 지금 순위를 포기할 이유가 없음 **[확정]**

`"어떻게 계산이 되냐면 점수를 기준으로 비율로 계산이 됩니다"` — 2차 평가는 **순위가 아니라 점수 비율**로 50점 환산 `[자동자막]`.
또한 Private 상위 30팀(예비 10 포함)이 산출물 제출 대상, 검증 통과 20팀이 2차 진출 `[원문]`.
현재 30위 컷은 **0.65977**. 우리 0.63747 과의 차 **0.0223** = M-1 의 1-NMAE +0.0139 하나로 넘는 거리다. `[산출]`

---

## §I 명시적 부정 결과 (찾았으나 없었던 것)

1. **본 대회 참가자 코드공유 0건, 토크 0건.** "상위팀이 흘린 노트북"은 존재하지 않는다. `[원문]`
2. **본 대회 관련 velog/tistory/brunch/medium 해법 글 0건.** `[검색 6회 실패]`
3. **제1회·제2회 BARAM 우승 해법 공개 0건.** 2025 BDA×동서발전 공모전 7등 저장소(TACTICS-YJH)는 README 31바이트로 비어 있고,
   수상자 인터뷰는 Instagram 게시물뿐(본문 미열람). `[전문미확인]`
4. **"밴드 적중 트릭(discretised action / band-hitting)"에 대한 공개 기술 문헌 0건.**
   OIBC 상위해법·제도 문헌·예측 학술 어디에도 "예측값을 밴드에 맞춰 이산 배치한다"는 기법은 없다.
   유일하게 발견된 밴드 획득 기법은 **ESS 물리 충방전**이며 우리 세팅에 적용 불가. `[스니펫]`
5. **주최측의 산식 추가 Q&A/정정 0건.** 8/3 17:00 에 "예측기준시점 정의"가 규칙 탭에 추가된 것이 전부이며,
   ±6 %/±8 % 밴드나 0.1·cap 게이트에 대한 별도 해석 공지는 없다. `[원문]`
6. GitHub 에 `open_wind`, `236727`, `kpx_group_1`, `scada_vestas_train` 로 검색되는 공개 해법 저장소는 없다
   (파일명 검색이 0건인 이유는 대부분 팀이 데이터를 gitignore 하기 때문). `[원문 API]`

---

## §J 이 레인이 남기는 한 문장

> 우리는 지난 여러 사이클 동안 "FICR 을 더 캐는 법"을 찾고 있었지만, 공개 리더보드 실측은
> **우리 FICR 이 우리 정확도 대비 이미 필드 상위 9 %** 이고 **우리 정확도는 필드 100/100 위**임을 말한다.
> 필요한 것은 새로운 기전이 아니라, 필드 중앙값 수준의 평범한 점예측 정확도다.
