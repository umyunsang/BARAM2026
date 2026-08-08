
"""N516 — M266(현재최고) x M252(최정확·탈상관) 결합 스윕 (적합 0회, 업로드 없음).

## 근거 (온라인 실측)
             1-NMAE     FICR      Total
M252        0.865903  0.387853  0.6268784   <- 우리 자산 중 1-NMAE 최고
M261        0.857885  0.415169  0.6365274
M266        0.858775  0.416167  0.6374709   <- 현재 최고

M252 는 점예측이 가장 정확하고(상위권 하한 0.86966 에 0.0038 차) FICR 이 최저다.
M266 은 그 반대다. 오차 상관 0.8436(M252 대 분류기 계열)이므로 예측 수준 결합은
지표에 비선형이라 두 끝점을 넘을 수 있다.

## 사전확약 (업로드 전 동결)
- H1  어떤 w 가 M266 의 0.6374709 를 초과한다
- H2  최적 w < 1.0 (M252 혼합이 도움)
- H3  1-NMAE 는 두 끝점 사이에서 단조 (선형성 확인용 통제)
- 반증: 모든 w 가 0.6374709 이하이면 아날로그 결합 축은 온라인에서도 닫힌다

## 그룹별 변형
M263 이 g2 에만 10% 를 섞은 것이 로컬 선택의 산물이므로, 전 그룹 균일 가중도 함께 낸다.

락박스 미접근 / 적합 0회 / 에이전트 업로드 없음.
"""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path("."); sys.path.insert(0, "src")
from baram.constants import CAPACITIES_KWH

SUB = ROOT / "artifacts/submissions"; OUT = SUB / "blends"; OUT.mkdir(exist_ok=True)
COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPS = {c: CAPACITIES_KWH[i] for i, c in enumerate(COLS, 1)}

base = pd.read_csv(SUB / "submission_M266.csv", encoding="utf-8-sig")
alt = pd.read_csv(SUB / "submission_M252.csv", encoding="utf-8-sig")
assert base["forecast_id"].astype(str).equals(alt["forecast_id"].astype(str))

def emit(name, frame):
    p = OUT / name
    frame.to_csv(p, index=False, encoding="utf-8-sig")
    raw = p.read_bytes()
    ok = raw.startswith(b"\xef\xbb\xbf") and len(frame) == 8760
    for c in COLS:
        ok &= bool((frame[c] >= 0).all() and (frame[c] <= CAPS[c]).all() and frame[c].notna().all())
    sha = hashlib.sha256(raw).hexdigest()
    print(f"  {name:40s} {sha[:12]}  {'PASS' if ok else 'FAIL'}")
    return dict(file=str(p), sha256=sha, valid=bool(ok))

made = []
print("M266 x M252 균일 가중")
for w in (0.90, 0.80, 0.70, 0.60):
    out = base[["forecast_id", "forecast_kst_dtm"]].copy()
    for c in COLS:
        out[c] = np.clip(w * base[c].to_numpy(float) + (1 - w) * alt[c].to_numpy(float), 0, CAPS[c])
    made.append(dict(kind="uniform", w=w, **emit(f"BLEND_M266w{int(w*100):03d}_M252.csv", out)))

Path("reports/n516_blend2_predeclaration.json").write_text(json.dumps(dict(
    node="N516_M266_M252_BLEND",
    anchors={"M252": [0.865903, 0.387853, 0.6268784],
             "M261": [0.857885, 0.415169, 0.6365274],
             "M266": [0.858775, 0.416167, 0.6374709]},
    rationale="M252 has the best 1-NMAE of all our assets (0.0038 below the top-96 minimum) and the "
              "worst FICR; M266 is the reverse. Error correlation 0.8436 makes prediction-level "
              "blending non-linear in the metric, so a blend can exceed both endpoints.",
    predeclaration={"H1": "some w exceeds M266 0.6374709", "H2": "optimal w < 1.0",
                    "H3": "1-NMAE monotone between endpoints (linearity control)",
                    "falsifier": "all w <= 0.6374709 closes the analog-combination axis online"},
    candidates=made, agent_upload=False, model_fits=0), indent=1, ensure_ascii=False))
print("\n영수증 -> reports/n516_blend2_predeclaration.json")
