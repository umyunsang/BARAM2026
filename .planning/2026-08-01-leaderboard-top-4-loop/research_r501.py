"""Research lane outcome, triggered by the router (stagnation 4 >= 3)."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from baram.loop.registry import NodeSpec, SpecRegistry

reg = SpecRegistry(Path("."))
finding = {
    "trigger": "router: stagnation 4 >= 3, score_bearing candidates exhausted",
    "queries": 3,
    "surfaced": [
        {"item": "Jørgensen 2025, Sequential methods for error correction of probabilistic wind power "
                 "forecast ENSEMBLES derived from NWP (Expert Syst. Appl.)",
         "admissible": False,
         "reason": "requires NWP ensemble members; the competition supplies two DETERMINISTIC sources "
                   "(LDAPS, GFS) with no members. C1N18_SPREAD_SKILL already closed the 2-source spread "
                   "proxy (SPREAD_EXPLAINED_BY_PREDICTED_LEVEL)."},
        {"item": "Li 2025, Ranking-oriented ML framework for wind power forecasting (PMC12658214)",
         "admissible": False,
         "reason": "optimises ranking consistency; the official objective is an absolute settlement band, "
                   "not a ranking. No stated translation to NMAE/FICR."},
        {"item": "Korean sources on the KPX 재생에너지 발전량 예측제도 (6%/8% bands)",
         "admissible": False,
         "reason": "confirms the settlement scheme already ported verbatim into evaluation/official.py. "
                   "No new method."},
    ],
    "new_specs": 0,
    "result": "RESEARCH_YIELDS_NO_ADMISSIBLE_SCORE_BEARING_LANE",
    "note": "third independent research pass (L2 lane, N401/N402 admission searches, this one) "
            "converging on the same closure set",
}
reg.register(NodeSpec(id="R501", kind="research", subcapability="C8.5",
                      summary="정체 트리거 딥리서치 — 신규 admissible 레인 0건",
                      origin="router-triggered", status="retired", score_bearing=False,
                      outcome={"accepted": False, "note": finding["result"], "detail": finding}))
reg.save()
print(json.dumps(finding, ensure_ascii=False, indent=1)[:1400])
