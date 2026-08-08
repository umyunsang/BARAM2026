from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import SpecRegistry
reg = SpecRegistry(Path("."))
by = {}
for s in reg.all():
    st = s.arguments.get("stage") or ("D?" if s.id.startswith("D") else "pre")
    if s.id.startswith("D") and "_" in s.id: st = s.id.split("_")[0]
    by.setdefault(st, []).append(s)
NAMES = {"D1":"전처리","D2":"피처구성","D3":"모델링","D4":"검증전략","D5":"개선전략"}
out = {"stages": {}, "totals": {}}
print("=" * 96)
for st in ("D1","D2","D3","D4","D5"):
    ns = by.get(st, [])
    if not ns: continue
    print(f"\n[{st} {NAMES[st]}]  {len(ns)}노드")
    recs = []
    for s in sorted(ns, key=lambda x: x.id):
        note = (s.outcome or {}).get("note", "-")
        acc = (s.outcome or {}).get("accepted", False)
        print(f"  {s.id:10s} {'ACCEPT' if acc else 'reject':7s} {note}")
        recs.append(dict(id=s.id, accepted=bool(acc), note=note, origin=s.origin))
    out["stages"][st] = dict(name=NAMES[st], nodes=recs)
allx = [s for v in by.values() for s in v]
acc = [s for s in allx if (s.outcome or {}).get("accepted")]
out["totals"] = dict(registry_total=len(allx), accepted=len(acc),
                     stage_nodes=sum(len(by.get(k, [])) for k in ("D1","D2","D3","D4","D5")))
print("\n" + "=" * 96)
print(f"레지스트리 총 {len(allx)}건 / 단계노드 {out['totals']['stage_nodes']}건 / 채택 {len(acc)}건")
Path("reports/STAGED_EXCAVATION_FINAL.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("-> reports/STAGED_EXCAVATION_FINAL.json")
