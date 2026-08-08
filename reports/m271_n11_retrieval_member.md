# M271 N11 — 검색 기반 후보(AnEn)로 조합 축을 여는 시도

- 판정일: 2026-08-06T08:20:49.789155+00:00
- 노드: `C1N102_RETRIEVAL_MEMBER` / 레인 L7 / 부모 `C1N101_PHYSICAL_MEMBER_PROBE`
- 하네스 `6e2a1e3f3bb5782ec35a2454` (캐시 True) / k=200 / 가중 {'sitewind__mean': 1.0, 'hour_sin': 0.3, 'hour_cos': 0.3, 'doy_sin': 0.3, 'doy_cos': 0.3}

## 1. 신규 후보

| | 값 |
|---|---:|
| 단독 Total | **0.602807** (1-NMAE 0.854556 / FICR 0.351059) |
| M115 대비 오차 상관 | **0.9150** |
| 참조 — M244 아날로그 | 0.605760 / 상관 0.8436 |
| 참조 — M115 단독 | 0.638410 |

## 2. 최선 조합

`med` ['ANALOG_V2', 'M113_LGBM_DART', 'M115_XGBOOST', 'M129_GROUP_FINETUNE', 'M98_ORDINAL_BIN025'] = **0.638555**
(C1N99 최선 0.639389 대비 **-0.000834**)

## 3. 사전확약 대조

- V1 `누출 0` -> **True** (0)
- V2 `확률행렬 정상` -> **True**
- V3 `조인 19,782` -> **True** (19782)
- H1 `상관 <= 0.85` -> **False** (0.9150)
- H2 `상관 < 0.8436` -> **False**
- H3 `단독 > 0.60576` -> **False** (0.602807)
- H4 `조합 이득 >= 0.001013` -> **False** (-0.000834)

판정: **RETRIEVAL_FAILS_CORRELATION_TARGET_SAME_NWP_AXIS_CLOSED**
