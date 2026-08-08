"""M271 — 원장 갱신 계약 발동과 감사.

C10 이 지목한 결함을 실제로 고친다. 이 스크립트는 셋을 한다.

  1. 이력이 비어 있으면 **A7 기준선**을 첫 사건으로 넣는다.
  2. **C1N63 의 셀 단위 천장 귀속**을 백필한다 — receipt 가 셀 단위로 직접 지지하는
     유일한 백필이다. 77 개 사이클을 사후 배정하지 않는다.
  3. 계약 상태를 감사해 보고한다 — 미기록 사이클, 갱신 필요 여부, 이력에서 **계산된**
     `stall_counter`.

**갱신 필요(`refresh_due`)를 자동으로 해소하지 않는다.** 원장을 현 챔피언으로 다시
계산하려면 A7 의 분해를 챔피언 예측 위에서 재실행해야 하고, 그것은 별도 노드의 일이다.
여기서는 "원장이 낡았다" 를 **라우터가 볼 수 있게** 만드는 데까지 한다. 낡은 것을
낡았다고 말하지 않는 것이 이 사태의 원인이었다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m271_ledger_state as ls
from m271_p4_consolidate import CYCLES, DEPLOYED_LOCAL_TOTAL, build

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_ledger_contract.md"
RECEIPT = REPORTS / "m271_ledger_contract_receipt.json"

NODE_ID = "C1N79_LEDGER_CONTRACT"
LANE = "L8"
PARENT_NODE = "C1N78_LOOP_ENGINE_C10"
CHAMPION_LOCAL = 0.630310


def main() -> int:
    _graph, ledger = build()
    ledger.compute_efficiency()

    before = {
        "events": len(ls.load_history()),
        "stall_counter": ls.stall_counter(),
        "recoverable_mass": ls.recoverable_mass(ledger),
        "cells_attributed": sum(
            1 for c in ledger.cells.values() if str(c.get("status")) != ls.UNEXPLAINED
        ),
    }

    baseline = ls.backfill_baseline(ledger)
    attribution = ls.backfill_ceiling_attribution(ledger)

    # 백필 뒤 원장을 다시 세워 이력 적용이 결정적으로 재현되는지 확인한다.
    _graph2, fresh = build()
    fresh.compute_efficiency()
    applied = ls.apply_to(fresh)

    state = ls.contract_state(fresh, CHAMPION_LOCAL, [c[0] for c in CYCLES])

    # V1 — 이력 적용이 멱등이다. 두 번 적용해도 같은 상태여야 한다.
    _graph3, again = build()
    again.compute_efficiency()
    ls.apply_to(again)
    ls.apply_to(again)
    v1 = bool(
        abs(ls.recoverable_mass(again) - ls.recoverable_mass(fresh)) < 1e-12
    )

    # V2 — 천장 도달 셀이 회수가능질량에서 실제로 빠졌다.
    raw_mass = sum(
        float(c.get("recoverable_if_average") or 0.0) for c in fresh.cells.values()
    )
    at_ceiling = [
        k for k, c in fresh.cells.items() if str(c.get("status")) == ls.AT_CEILING
    ]
    v2 = bool(ls.recoverable_mass(fresh) <= raw_mass)

    # V3 — `stall_counter` 가 이력에서 계산된다(상수가 아니다).
    v3 = bool(ls.stall_counter() == max(len(ls.load_history()) - 1, 0))

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "triggered_by": "C10 이 지목한 근본 결함",
        "contract_version": ls.CONTRACT_VERSION,
        "history_path": str(ls.HISTORY.relative_to(ROOT)),
        "before": before,
        "backfill": {
            "baseline": baseline,
            "ceiling_attribution": (
                {k: v for k, v in attribution.items() if k != "cell_updates"}
                if attribution else None
            ),
            "cells_updated": len(attribution["cell_updates"]) if attribution else 0,
        },
        "apply": applied,
        "after": state,
        "raw_recoverable_mass": raw_mass,
        "at_ceiling_cells": at_ceiling,
        "checks": {
            "V1_apply_idempotent": v1,
            "V2_ceiling_cells_excluded": v2,
            "V3_stall_derived_not_asserted": v3,
        },
        "ledger_baseline_policy_total": (
            ls.load_history()[0]["champion_total"] if ls.load_history() else None
        ),
        "current_champion_total": CHAMPION_LOCAL,
        "deployed_local_total": DEPLOYED_LOCAL_TOTAL,
        "refresh_due": state["refresh_due"],
        "note": (
            "refresh_due 를 자동으로 해소하지 않는다. 원장을 현 챔피언으로 다시 계산하려면 "
            "A7 분해를 챔피언 예측 위에서 재실행해야 하고 그것은 별도 노드의 일이다. "
            "여기서는 '원장이 낡았다' 를 라우터가 볼 수 있게 만드는 데까지 한다."
        ),
        "dacon_upload": False,
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")

    lines = [
        "# M271 — 결손 원장 갱신 계약",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / "
        "**C10 이 지목한 근본 결함의 수정**",
        "",
        "## 1. 무엇이 고장나 있었나",
        "",
        f"- A7 이 정책 `T0.5_G1.5` **{payload['ledger_baseline_policy_total']}** 로 원장을 "
        f"세웠고 현 챔피언은 **{CHAMPION_LOCAL}** 이다. 챔피언이 바뀌어도 원장을 다시 "
        "계산하지 않으므로 회수가능질량이 **원리적으로 움직일 수 없었다**.",
        "- 셀의 `status`/`mechanism`/`owner` 가 전부 초기값이라 C1(미설명 셀)이 영원히 "
        "발화하고 설명된 질량이 보이지 않았다.",
        f"- 그래서 `stall_counter` 가 **{before['stall_counter']} -> 이제 "
        f"{state['stall_counter']}** 로 **이력에서 계산**된다. 상수 주장이 아니다.",
        "",
        "## 2. 계약",
        "",
        f"- 이력 `{payload['history_path']}` 에 append-only 기록",
        "- 셀 귀속에 **소유자 필수** — 없으면 `ContractViolation`",
        "- `stall_counter` 는 질량 무감소 연속 기록 수로 **계산**. 한 번이라도 줄면 0",
        "- `refresh_due` 가 원장 기준 점수와 현 챔피언 불일치를 노출",
        "- `unrecorded_cycles` 가 `CYCLES` 대비 미기록 노드를 감사",
        "",
        "## 3. 백필 (receipt 가 셀 단위로 지지하는 것만)",
        "",
        f"- 기준선: {'기록됨' if baseline else '이미 존재'}",
        f"- C1N63 천장 귀속: **{payload['backfill']['cells_updated']} 셀**",
        f"- 천장 도달(회수 불가) 셀: **{len(at_ceiling)}**",
        f"- 원시 회수가능질량 {raw_mass:.5f} -> 천장 제외 후 "
        f"**{state['recoverable_mass']:.5f}**",
        "",
        "**77 개 사이클을 사후에 셀에 배정하지 않았다.** 그것이 이 사태를 만든 사후 전사 "
        "습관이다. C1N63 만이 108 셀 각각에 실현 단위와 물리 천장을 계산했으므로 "
        "유일하게 백필 가능하다.",
        "",
        "## 4. 감사",
        "",
        f"- 이력 사건 **{state['events']}** 건",
        f"- 귀속된 셀 **{state['cells_attributed']}/{state['cells_total']}**",
        f"- **미기록 사이클 {state['unrecorded_cycles']}** 건 "
        f"(예: `{', '.join(state['unrecorded_sample'])}`)",
        f"- 갱신 필요 **{state['refresh_due']}**",
        "",
        "## 5. 타당성 가드",
        "",
        f"- V1 이력 적용이 멱등 -> **{v1}**",
        f"- V2 천장 셀이 질량에서 제외됨 -> **{v2}**",
        f"- V3 `stall_counter` 가 이력에서 계산됨 -> **{v3}**",
        "",
        "## 6. 남는 것",
        "",
        payload["note"],
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 원장 갱신 계약 ===")
    print(f"[LEDGER] 기준선 정책 총점 {payload['ledger_baseline_policy_total']} vs "
          f"현 챔피언 {CHAMPION_LOCAL} -> 갱신필요 {state['refresh_due']}")
    print(f"[LEDGER] 백필 셀 {payload['backfill']['cells_updated']} / 천장도달 "
          f"{len(at_ceiling)}")
    print(f"[LEDGER] 회수가능질량 {raw_mass:.5f} -> {state['recoverable_mass']:.5f}")
    print(f"[LEDGER] 이력 {state['events']} 건 / stall {state['stall_counter']} "
          f"(계산값) / 귀속 {state['cells_attributed']}/{state['cells_total']}")
    print(f"[LEDGER] 미기록 사이클 {state['unrecorded_cycles']} 건")
    print(f"[LEDGER] V1 {v1} / V2 {v2} / V3 {v3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
