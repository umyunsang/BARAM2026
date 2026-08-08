
"""D1_F3_1 — GFS 80m/100m/PBL 풍속의 후보 피처 풀 존재 여부 감사 (적합 0회).

근거: research://IOP 2026 doi:10.1088/3049-4753/ae4c8d + ACP 23:3181 [directly_supported]
질문: 허브고도 근처 풍속이 M115 선택 100개에 0개인데, **후보 풀에는 있었는가**?
  - 있었다 -> 선택에서 탈락 (이미 평가됨, 축 닫힘)
  - 없었다 -> **애초 미탐색 축** (열림)
"""
from __future__ import annotations
import sys, json, re
from pathlib import Path
sys.path.insert(0, "src")

ROOT = Path(".")
pool = json.loads((ROOT / "artifacts/manifests/prepare.json").read_text())["feature_names"]
sel = json.loads((ROOT / "artifacts/backtests/metric-aligned-probe/M115_XGBOOST-dev-2023-Q3.json").read_text())
selected = sel.get("selected_feature_names") or []

TARGETS = {
    "GFS 80m u/v":        r"heightAboveGround_80_",
    "GFS 100m u/v":       r"heightAboveGround_100_",
    "GFS PBL u/v/VRATE":  r"planetaryBoundaryLayer_",
    "GFS 850hPa":         r"isobaricInhPa_850_",
    "GFS 700hPa":         r"isobaricInhPa_700_",
    "GFS 500hPa":         r"isobaricInhPa_500_",
    "GFS gust":           r"surface_0_gust",
    "LDAPS 50m max/min":  r"heightAboveGround_50_50M",
    "GFS 10m u/v":        r"gfs__heightAboveGround_10_",
    "LDAPS 10m u/v":      r"ldaps__heightAboveGround_10_",
}
print(f"후보 피처 풀 {len(pool)}개  /  M115 선택 {len(selected)}개\n")
print(f"{'변수군':22s} {'풀':>5} {'선택':>5}  판정")
rows = []
for name, pat in TARGETS.items():
    inp = [f for f in pool if re.search(pat, f)]
    ins = [f for f in selected if re.search(pat, f)]
    if not inp:      verdict = "**풀에 부재 — 미탐색**"
    elif not ins:    verdict = "풀에 있으나 선택 탈락"
    else:            verdict = "선택됨"
    rows.append(dict(group=name, pool=len(inp), selected=len(ins), verdict=verdict,
                     sample=inp[:2]))
    print(f"{name:22s} {len(inp):5d} {len(ins):5d}  {verdict}")

absent = [r for r in rows if r["pool"] == 0]
dropped = [r for r in rows if r["pool"] > 0 and r["selected"] == 0]
print(f"\n풀에 부재(미탐색): {[r['group'] for r in absent]}")
print(f"풀에 있으나 탈락:   {[r['group'] for r in dropped]}")

# 풀 전체의 변수 프리픽스 인벤토리
prefixes = sorted({re.sub(r"__(mean|std|min|max|q10|q50|q90)$", "", f) for f in pool})
gfs_vars = sorted({p.split("__")[1] for p in prefixes if p.startswith("gfs__") and "__" in p})
ldaps_vars = sorted({p.split("__")[1] for p in prefixes if p.startswith("ldaps__") and "__" in p})
print(f"\n풀의 GFS 변수 {len(gfs_vars)}종, LDAPS 변수 {len(ldaps_vars)}종")
print(f"  GFS: {gfs_vars[:12]}")
verdict = "ABSENT_FROM_POOL_UNEXPLORED" if absent else "PRESENT_BUT_DESELECTED"
print(f"\n판정: {verdict}")
Path("reports/D1_F3_1_feature_pool_audit.json").write_text(json.dumps(dict(
    node="D1_F3_1", stage="D1", pool_size=len(pool), selected_size=len(selected),
    groups=rows, absent_from_pool=[r["group"] for r in absent],
    present_but_deselected=[r["group"] for r in dropped],
    gfs_vars_in_pool=gfs_vars, ldaps_vars_in_pool=ldaps_vars, verdict=verdict,
    origin="research://IOP-2026-3049-4753-ae4c8d + ACP-23-3181 [directly_supported]",
    model_fits=0, dacon_upload=False), indent=1, ensure_ascii=False))
print("영수증 -> reports/D1_F3_1_feature_pool_audit.json")
