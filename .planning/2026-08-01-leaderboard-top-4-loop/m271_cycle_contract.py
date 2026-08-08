"""M271 — 사이클 계약 강제. 원장에 안 쓰면 라우팅에 들어가지 못한다.

원장 갱신 계약(`m271_ledger_state`)을 만든 직후 감사가 **미기록 76 건**을 냈고, 그 뒤
새로 만든 노드 네 개(C1N77~C1N80)도 여전히 안 써서 **79 건**으로 늘었다. 계약을
만들기만 하고 아무것도 그것을 강제하지 않은 것이다 — 이 세션의 실패와 정확히 같은
형태다(엔진을 만들고 안 돌린 것).

이 모듈이 강제한다.

  1. **감사 백필** — 계약 도입 이전에 돌았던 사이클들을 **하나의 AUDIT 사건**으로 덮는다.
     79 개의 가짜 사건을 지어내지 않는다. 대신 `cycles_covered=N` 으로 그 사건이 몇
     매크로스텝을 덮는지 명시하고, `stall_counter` 가 그것을 합산한다. 계약을 늦게
     만들었다는 이유로 정체가 짧아 보이는 왜곡이 없어진다.
  2. **전방 강제** — `require_recorded()` 가 미기록 노드를 찾아내고, 라우팅 드라이버가
     그 증거를 **거부**한다. 기록하지 않은 사이클은 라우터에 영향을 줄 수 없다.
  3. **간편 기록** — `close_cycle()` 이 receipt 를 읽어 사건 하나를 남긴다. 사이클
     스크립트가 마지막에 한 줄 호출하면 된다.

**부작용을 함께 처리한다.** 정체가 진짜 값(~82)으로 돌아가면 C10 이 또 모든 증거를
삼킨다 — 이미 서비스했는데도. 계획서의 종료 가드 사다리가 이미 규정해 뒀다:
"진행정체(N 매크로스텝간 결손질량 무감소, **단 C10 을 1 회 거친 뒤**)". 즉 C10 은 한 번
발화하고, 그 뒤에도 정체가 이어지면 그것은 **종료 조건**이지 또 다른 C10 이 아니다.
`m271_c10_loop_engine.decide_v3` 가 `loop_engine_visits` 로 그것을 강제한다.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m271_ledger_state as ls

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

AUDIT_NODE = "PRE_CONTRACT_AUDIT"


class CycleContractViolation(RuntimeError):
    """기록하지 않은 사이클이 라우팅에 들어오려 했다."""


def close_cycle(
    node_id: str,
    receipt_name: str,
    *,
    champion_total: float,
    ledger: Any,
    cell_updates: dict[str, dict[str, Any]] | None = None,
    note: str = "",
) -> dict[str, Any]:
    """사이클 종료 시 원장에 사건 하나를 남긴다. 사이클 스크립트의 마지막 한 줄.

    셀 귀속이 없어도 **기록은 남긴다** — "이 사이클이 돌았고 질량이 움직이지 않았다" 가
    정체 계산에 필요한 사실이기 때문이다. 진단 노드라고 건너뛰면 연료계가 다시 죽는다.
    """
    path = REPORTS / receipt_name
    if not path.exists():
        raise CycleContractViolation(
            f"{node_id}: receipt 가 없다({receipt_name}). 실행하지 않은 사이클은 기록할 수 없다."
        )
    return ls.record(
        node_id,
        "CYCLE",
        champion_total=champion_total,
        total_recoverable=ls.recoverable_mass(ledger),
        cell_updates=cell_updates,
        note=note or f"{receipt_name} 기준",
    )


def audit_backfill(ledger: Any, cycle_ids: list[str], champion_total: float) -> dict[str, Any] | None:
    """계약 도입 이전 사이클들을 하나의 AUDIT 사건으로 덮는다.

    **79 개의 가짜 사건을 지어내지 않는다.** 사이클별 챔피언 이력을 재구성할 수 없고,
    재구성하려 들면 그것이 이 사태를 만든 사후 전사가 된다. 대신 감사된 사실 하나를
    기록한다 — "N 개가 돌았고 질량은 움직이지 않았다".
    """
    if any(e["node_id"] == AUDIT_NODE for e in ls.load_history()):
        return None
    missing = ls.unrecorded_cycles(cycle_ids)
    if not missing:
        return None
    return ls.record(
        AUDIT_NODE,
        "AUDIT",
        champion_total=champion_total,
        total_recoverable=ls.recoverable_mass(ledger),
        cycles_covered=len(missing),
        note=(
            f"계약 도입 이전 {len(missing)} 개 사이클이 원장에 아무것도 쓰지 않았다. "
            "사이클별 챔피언 이력은 재구성할 수 없으므로 가짜 사건을 지어내지 않고 "
            "감사된 사실 하나로 덮는다. 이 사이클들도 실제로 일어난 매크로스텝이므로 "
            "`cycles_covered` 로 정체 길이에 그대로 반영된다."
        ),
    )


def require_recorded(node_ids: list[str], *, strict: bool = True) -> list[str]:
    """미기록 노드를 반환한다. `strict` 면 예외를 던진다.

    라우팅 드라이버가 이것을 호출해 **기록하지 않은 사이클의 증거를 거부**한다.
    """
    covered = {e["node_id"] for e in ls.load_history()}
    if any(e["node_id"] == AUDIT_NODE for e in ls.load_history()):
        # AUDIT 은 그 시점까지의 사이클을 덮는다. 이후 노드만 개별 기록을 요구한다.
        audit_at = next(
            e["ts"] for e in ls.load_history() if e["node_id"] == AUDIT_NODE
        )
    else:
        audit_at = None
    missing = [n for n in node_ids if n not in covered]
    if missing and strict:
        raise CycleContractViolation(
            f"원장에 기록되지 않은 노드가 라우팅에 들어오려 했다: {missing}. "
            f"`close_cycle()` 로 기록하거나 `audit_backfill()` 로 덮어라."
            + (f" (AUDIT 시각 {audit_at})" if audit_at else "")
        )
    return missing


def contract_report(ledger: Any, cycle_ids: list[str], champion_total: float) -> dict[str, Any]:
    state = ls.contract_state(ledger, champion_total, cycle_ids)
    history = ls.load_history()
    return {
        **state,
        "audit_present": any(e["node_id"] == AUDIT_NODE for e in history),
        "cycles_covered_total": sum(int(e.get("cycles_covered", 1)) for e in history),
        "generated_at": datetime.now(UTC).isoformat(),
    }
