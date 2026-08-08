import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import NodeSpec, SpecRegistry
reg = SpecRegistry(Path("."))
items = [
 ("N521","C5.5","oil-free blend curve: best w=0.70, predicted 0.639170",{"accepted":True,"note":"VERIFIED_BLEND_W070","pred":0.639170}),
 ("N522","C5.5","per-group blend: rejected fold-outside",{"accepted":False,"note":"FOLDOUT_REJECTS_PERGROUP"}),
 ("N523","C7.1","lineage re-rank: order flipped by contamination, swap below threshold",{"accepted":False,"note":"RANK_FLIPPED_BUT_BELOW_THRESHOLD"}),
 ("N524","C7.3","upload manifest frozen: A=1 / B=6 / rejected=2",{"accepted":True,"note":"MANIFEST_FROZEN"}),
]
for sid, sub, summary, outcome in items:
    reg.register(NodeSpec(id=sid, kind="analysis", subcapability=sub, summary=summary,
                          status="retired", score_bearing=True, outcome=outcome))
reg.save()
state = {
 "session": "2026-08-06", "terminal": True,
 "best_online_measured": 0.6374708505, "best_verified_candidate_pred": 0.639170,
 "target": 0.66, "gap": 0.66 - 0.6374708505,
 "required_error_reduction": 0.108,
 "predeclared_experiments": 30, "deployable_net_gain": 0.0,
 "open_action": "user uploads grade-A/B candidates; agent never uploads",
 "stop_reason": "every axis closed with measured bounds; max verified improvement +0.0017 against a +0.0225 requirement",
}
Path("reports/SESSION_TERMINAL_STATE.json").write_text(json.dumps(state, indent=1, ensure_ascii=False))
print(json.dumps(state, indent=1, ensure_ascii=False))
print("registry specs total: " + str(len(reg.all())))
