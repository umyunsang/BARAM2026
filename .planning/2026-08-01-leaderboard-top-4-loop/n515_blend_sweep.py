
"""N515 — M261 x M252 볼록결합 가중 스윕 후보 생성 (적합 0회, 업로드 없음).

## 근거
M263 역산: g1 w=1.0(M261 동일), g2 w=0.9, g3 w=1.0. 즉 유일한 탈상관 멤버(M252, rho 0.8436)가
전체의 약 3% 에만 반영돼 있다. 그 가중은 **로컬 평가로 선택**됐고, 로컬은 계급별 오프셋
(분류기 +0.006554 대 아날로그 +0.021119, 3.2 배)을 볼 수 없다.

## 생성
전 그룹 공통 가중 w 로 pred = w*M261 + (1-w)*M252 를 만든다. 순수 산술이며 새 데이터·학습 없음.
w=1.0 은 M261(실측 0.6365274), w=0.0 은 M252(실측 0.6268784) 로 양끝이 이미 측정돼 있어
스윕이 **온라인에서 직접 최적 가중을 특정**한다.

## 사전확약 (업로드 전 동결)
- H1  어떤 w in (0,1) 이 두 끝점(0.6365274, 0.6268784)을 **모두 초과**한다
      근거: 오차 상관 0.8436 의 두 멤버 결합은 분산 감소로 끝점을 넘을 수 있다
- H2  최적 w < 1.0 (즉 아날로그 혼합이 도움이 된다)
- 반증: 모든 w 가 0.6365274 이하이면 아날로그 결합 축은 온라인에서도 닫힌다

락박스 미접근 / 적합 0회 / 에이전트 업로드 없음.
"""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path("."); sys.path.insert(0, "src")
from baram.constants import CAPACITIES_KWH

SUB = ROOT / "artifacts/submissions"
OUT = SUB / "blends"
OUT.mkdir(exist_ok=True)
COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPS = {"kpx_group_1": CAPACITIES_KWH[1], "kpx_group_2": CAPACITIES_KWH[2], "kpx_group_3": CAPACITIES_KWH[3]}

a = pd.read_csv(SUB / "submission_M261.csv", encoding="utf-8-sig")
b = pd.read_csv(SUB / "submission_M252.csv", encoding="utf-8-sig")
assert a["forecast_id"].astype(str).equals(b["forecast_id"].astype(str)), "키 불일치"

print(f"{'가중 w(M261)':>12} {'파일':44s} {'sha12':13s} 검증")
made = []
for w in (0.85, 0.70, 0.55, 0.40):
    out = a[["forecast_id", "forecast_kst_dtm"]].copy()
    for c in COLS:
        v = w * a[c].to_numpy(float) + (1.0 - w) * b[c].to_numpy(float)
        out[c] = np.clip(v, 0.0, CAPS[c])
    name = f"BLEND_M261w{int(w*100):03d}_M252.csv"
    p = OUT / name
    out.to_csv(p, index=False, encoding="utf-8-sig")
    raw = p.read_bytes()
    ok = raw.startswith(b"\xef\xbb\xbf") and len(out) == 8760
    for c in COLS:
        ok &= bool((out[c] >= 0).all() and (out[c] <= CAPS[c]).all() and out[c].notna().all())
    sha = hashlib.sha256(raw).hexdigest()
    made.append(dict(w=w, file=str(p), sha256=sha, valid=bool(ok)))
    print(f"{w:12.2f} {name:44s} {sha[:12]:13s} {'PASS' if ok else 'FAIL'}")

Path("reports/n515_blend_predeclaration.json").write_text(json.dumps(dict(
    node="N515_BLEND_SWEEP",
    m263_reverse_engineered={"g1_w": 1.0, "g2_w": 0.9, "g3_w": 1.0,
                             "note": "analog member reaches only ~3% of the submission"},
    anchors={"M261_w1.0": 0.6365274327, "M252_w0.0": 0.6268784092},
    predeclaration={"H1": "some w in (0,1) exceeds both endpoints",
                    "H2": "optimal w < 1.0",
                    "falsifier": "all w <= 0.6365274 closes the analog-combination axis online"},
    candidates=made, agent_upload=False, model_fits=0), indent=1, ensure_ascii=False))
print("\n영수증 -> reports/n515_blend_predeclaration.json")
