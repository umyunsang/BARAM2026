
"""N514 — 미업로드 제출 후보 15종 형식 검증 (적합 0회, 업로드 없음).

제출 쿼터를 낭비하지 않도록 대회 계약(8,760행 / UTF-8 BOM / 키 일치 / 용량 범위)을
프로젝트 자체 검증기 규칙으로 확인한다.
"""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path("."); sys.path.insert(0, "src")
from baram.constants import CAPACITIES_KWH

SUB = ROOT / "artifacts/submissions"
CACHE = ROOT / "artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
sample = pd.read_parquet(CACHE / "submission_keys.parquet")
key_cols = [c for c in sample.columns if c in ("forecast_id", "forecast_kst_dtm")]
UPLOADED = {"submission_M261.csv", "submission_M252.csv", "submission_M266.csv"}

rows = []
for p in sorted(SUB.glob("*.csv")):
    if p.name in UPLOADED: continue
    issues = []
    raw = p.read_bytes()
    if not raw.startswith(b"\xef\xbb\xbf"): issues.append("UTF-8 BOM 없음")
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
    except Exception as e:
        rows.append((p.name, "READ_FAIL", str(e)[:40], None)); continue
    if len(df) != 8760: issues.append(f"행수 {len(df)}")
    need = {"forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"}
    if set(df.columns) != need: issues.append(f"컬럼 {sorted(set(df.columns) ^ need)}")
    else:
        if not df["forecast_id"].astype(str).equals(sample["forecast_id"].astype(str)):
            issues.append("forecast_id 불일치")
        for g, col in ((1, "kpx_group_1"), (2, "kpx_group_2"), (3, "kpx_group_3")):
            v = pd.to_numeric(df[col], errors="coerce")
            if v.isna().any(): issues.append(f"{col} NaN {int(v.isna().sum())}")
            elif (v < 0).any(): issues.append(f"{col} 음수 {int((v < 0).sum())}")
            elif (v > CAPACITIES_KWH[g]).any(): issues.append(f"{col} 용량초과 {int((v > CAPACITIES_KWH[g]).sum())}")
    sha = hashlib.sha256(raw).hexdigest()[:12]
    rows.append((p.name, "PASS" if not issues else "FAIL", "; ".join(issues)[:60], sha))

print(f"{'후보':50s} {'판정':6s} {'해시':13s} 문제")
npass = 0
for n, st, msg, sha in rows:
    npass += st == "PASS"
    print(f"{n[:50]:50s} {st:6s} {str(sha):13s} {msg}")
print(f"\n검증 통과 {npass}/{len(rows)}")
Path("reports/n514_submission_validation.json").write_text(json.dumps(
    {"node": "N514_SUBMISSION_VALIDATION",
     "results": [{"file": n, "verdict": s, "issues": m, "sha256_prefix": h} for n, s, m, h in rows],
     "passed": npass, "total": len(rows), "agent_upload": False}, indent=1, ensure_ascii=False))
print("영수증 -> reports/n514_submission_validation.json")
