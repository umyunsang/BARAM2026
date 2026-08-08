
"""N524 — 최종 업로드 매니페스트: 전 후보 재검증 + 근거 등급 (적합 0회, 업로드 없음)."""
from __future__ import annotations
import sys, json, hashlib, glob, os
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path("."); sys.path.insert(0, "src")
from baram.constants import CAPACITIES_KWH

SUB = ROOT / "artifacts/submissions"
COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPS = {c: CAPACITIES_KWH[i] for i, c in enumerate(COLS, 1)}
sample = pd.read_parquet(ROOT / "artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/submission_keys.parquet")
UPLOADED = {"submission_M261.csv": 0.6365274327, "submission_M252.csv": 0.6268784092,
            "submission_M266.csv": 0.6374708505}
# 근거 등급: A=fold-외 검증 통과, B=구조적 논거만, C=로컬 근거(오염 가능), X=기각
GRADE = {
    "BLEND_M261w070_M252.csv": ("A", 0.639170, "N521 오염제거 곡선 최적, 자유도 1, 끝점 재현 1e-7/8e-5"),
    "BLEND_M266w080_M252.csv": ("B", None, "M266(현재최고) 기반 결합, 곡선 미측정"),
    "BLEND_M266w060_M252.csv": ("B", None, "M266 기반, 곡선 3점 확보용"),
    "BLEND_M266w090_M252.csv": ("B", None, "M266 기반"),
    "BLEND_M261w055_M252.csv": ("B", 0.637125, "N521 곡선 좌측"),
    "BLEND_M261w085_M252.csv": ("B", 0.638573, "N521 곡선 우측 보간"),
    "BLEND_M261w040_M252.csv": ("C", 0.634468, "곡선상 열위"),
    "BLEND_PERGROUP_95_75_100.csv": ("X", None, "N522 fold-외 기각"),
    "BLEND_CLEAN_PG_": ("X", None, "생성 안 됨"),
    "submission_M263.csv": ("B", 0.638756, "사전확약 N513, 앙상블/g3 분해"),
    "submission_M268.csv": ("C", None, "로컬 M261 하회"),
    "final_candidate.csv": ("X", None, "용량 초과 15행"),
}
rows = []
for p in sorted(list(SUB.glob("*.csv")) + list((SUB / "blends").glob("*.csv"))):
    n = p.name
    if n in UPLOADED:
        rows.append(dict(file=n, grade="UPLOADED", pred=None, actual=UPLOADED[n], valid=True, note="")); continue
    raw = p.read_bytes()
    try: df = pd.read_csv(p, encoding="utf-8-sig")
    except Exception as e:
        rows.append(dict(file=n, grade="X", pred=None, actual=None, valid=False, note=f"read fail {e}")); continue
    ok = raw.startswith(b"\xef\xbb\xbf") and len(df) == 8760
    ok &= df["forecast_id"].astype(str).equals(sample["forecast_id"].astype(str))
    for c in COLS:
        v = pd.to_numeric(df[c], errors="coerce")
        ok &= bool(v.notna().all() and (v >= 0).all() and (v <= CAPS[c]).all())
    g, pr, note = GRADE.get(n, ("C", None, "미분류"))
    rows.append(dict(file=n, grade=g if ok else "X", pred=pr, actual=None, valid=bool(ok),
                     note=note if ok else "형식 검증 실패", sha=hashlib.sha256(raw).hexdigest()[:12],
                     path=str(p)))

order = {"UPLOADED": 0, "A": 1, "B": 2, "C": 3, "X": 4}
rows.sort(key=lambda r: (order[r["grade"]], -(r["pred"] or 0)))
print(f"{'등급':6s} {'후보':44s} {'예측':>9} {'실측':>11} {'sha':13s} 비고")
for r in rows:
    pr = f"{r['pred']:.6f}" if r["pred"] else "-"
    ac = f"{r['actual']:.7f}" if r["actual"] else "-"
    print(f"{r['grade']:6s} {r['file'][:44]:44s} {pr:>9} {ac:>11} {str(r.get('sha','')):13s} {r['note'][:34]}")
up = [r for r in rows if r["grade"] in ("A", "B")]
print(f"\n업로드 권고 {len(up)}건 (A={sum(1 for r in up if r['grade']=='A')}, B={sum(1 for r in up if r['grade']=='B')})")
print(f"기각 {sum(1 for r in rows if r['grade']=='X')}건 / 이미 업로드 {sum(1 for r in rows if r['grade']=='UPLOADED')}건")
Path("reports/n524_upload_manifest.json").write_text(json.dumps(dict(
    node="N524_UPLOAD_MANIFEST", rows=rows, agent_upload=False,
    grades={"A": "fold-외 검증 통과", "B": "구조적 논거", "C": "열위", "X": "기각"}),
    indent=1, ensure_ascii=False))
print("영수증 -> reports/n524_upload_manifest.json")
