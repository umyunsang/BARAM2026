"""M271 결손 원장 갱신 계약 — 루프의 연료계를 살린다.

C10(라우터가 지시한 루프엔진 개선)이 근본 결함을 지목했다.

    `stall_counter` 가 77 이 된 이유는 **결손 원장이 A7 이후 한 번도 갱신되지 않은 것**이다.

결함은 둘로 나뉜다.

  (가) **갱신 없음**  A7 은 `T0.5_G1.5` 를 0.6286045 로 채점해 원장을 세웠다. 현 챔피언은
       `M115_XGBOOST@T0.6_G0.2` 0.630310 이다. 챔피언이 바뀌어도 원장을 다시 계산하지
       않으므로 회수가능질량이 **원리적으로 움직일 수 없다**.
  (나) **귀속 없음**  셀의 `status`/`mechanism`/`owner` 가 전부 초기값이다. 어떤 사이클도
       "이 셀을 이 기전으로 설명했다" 를 기록하지 않았으므로 C1(미설명 셀)이 영원히
       발화하고, 설명된 질량이 보이지 않는다.

`stall_counter = 77` 은 **내용상 맞다**(점수가 실제로 안 움직였다). 문제는 그 값이
바뀔 수 있는 경로가 없다는 것이다 — 연료계가 고장난 채 눈금이 우연히 맞은 상태다.

이 모듈이 계약을 만든다.

  1. **이력 영속화** — 원장 상태 변화를 append-only 로 기록한다. `stall_counter` 는
     그 이력에서 **계산**되지 상수로 주장되지 않는다.
  2. **셀 귀속 API** — 사이클이 셀의 기전·상태를 기록한다. 소유자가 남는다.
  3. **갱신 필요 감지** — 원장의 기준 점수와 현 챔피언이 어긋나면 `refresh_due` 가 참이
     되어, 원장이 낡았다는 사실이 라우터에 보인다.
  4. **미기록 사이클 감사** — `CYCLES` 에 있는데 이력에 없는 노드를 보고한다.

**백필 규율**: 77 개 사이클을 사후에 셀에 배정하지 않는다. 그것이 이 사태를 만든
사후 전사 습관이다. 백필은 **receipt 가 셀 단위로 직접 지지하는 것만** 넣는다 —
현재로선 C1N63(천장 재기준화)이 유일하다. 그 노드는 108 셀 각각에 대해 실현 단위와
물리 천장을 계산했고 `over_ceiling` 를 셀 단위로 냈다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REGISTRY = ROOT / "artifacts" / "registry"
HISTORY = REGISTRY / "m271_ledger_history.json"

CONTRACT_VERSION = "M271_LEDGER_CONTRACT_v1"

# 셀 상태. A7 은 전부 UNEXPLAINED 로 세운다.
UNEXPLAINED = "UNEXPLAINED"
AT_CEILING = "AT_CEILING"  # 물리 천장에 도달 — 회수 불가
EXPLAINED = "EXPLAINED"  # 기전이 확인됨
REFUTED = "REFUTED"  # 제안된 기전이 반증됨
STATUSES = (UNEXPLAINED, AT_CEILING, EXPLAINED, REFUTED)


@dataclass(frozen=True)
class LedgerEvent:
    ts: str
    node_id: str
    kind: str
    champion_total: float
    total_recoverable: float
    cell_updates: dict[str, dict[str, Any]]
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "node_id": self.node_id,
            "kind": self.kind,
            "champion_total": self.champion_total,
            "total_recoverable": self.total_recoverable,
            "cell_updates": self.cell_updates,
            "note": self.note,
        }


class ContractViolation(RuntimeError):
    """원장 계약을 어겼다. 조용히 넘어가지 않는다."""


def load_history() -> list[dict[str, Any]]:
    if not HISTORY.exists():
        return []
    payload = json.loads(HISTORY.read_text(encoding="utf-8"))
    return list(payload.get("events", []))


def _save(events: list[dict[str, Any]]) -> None:
    REGISTRY.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(
        json.dumps(
            {"contract_version": CONTRACT_VERSION, "events": events},
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )


def record(
    node_id: str,
    kind: str,
    *,
    champion_total: float,
    total_recoverable: float,
    cell_updates: dict[str, dict[str, Any]] | None = None,
    note: str = "",
    cycles_covered: int = 1,
) -> dict[str, Any]:
    """원장 상태 변화를 기록한다. append-only 이며 과거를 고쳐 쓰지 않는다.

    `cycles_covered` 는 이 사건이 몇 매크로스텝을 덮는지다. 보통 1 이고, 계약 도입
    이전 사이클들을 묶는 AUDIT 사건만 1 보다 크다.
    """
    updates = cell_updates or {}
    for key, update in updates.items():
        status = update.get("status")
        if status is not None and status not in STATUSES:
            raise ContractViolation(f"알 수 없는 셀 상태: {key} -> {status}")
        if update.get("owner") in (None, ""):
            raise ContractViolation(f"셀 갱신에 소유자가 없다: {key}")

    events = load_history()
    event = LedgerEvent(
        ts=datetime.now(UTC).isoformat(),
        node_id=node_id,
        kind=kind,
        champion_total=float(champion_total),
        total_recoverable=float(total_recoverable),
        cell_updates=updates,
        note=note,
    )
    payload = event.as_dict()
    payload["cycles_covered"] = int(cycles_covered)
    events.append(payload)
    _save(events)
    return payload


def latest_cell_state() -> dict[str, dict[str, Any]]:
    """이력을 접어 셀별 최신 상태를 만든다. 나중 기록이 앞선 기록을 덮는다."""
    state: dict[str, dict[str, Any]] = {}
    for event in load_history():
        for key, update in (event.get("cell_updates") or {}).items():
            merged = dict(state.get(key, {}))
            merged.update(update)
            merged["updated_by"] = event["node_id"]
            merged["updated_at"] = event["ts"]
            state[key] = merged
    return state


def apply_to(ledger: Any) -> dict[str, int]:
    """A7 로 새로 세운 원장에 이력의 셀 상태를 덧입힌다.

    원장 자체는 여전히 A7 에서 결정적으로 재구성되고, 그 위에 **기록된 갱신만** 얹는다.
    이렇게 해야 원장이 재현 가능하면서도 학습 결과를 반영한다.
    """
    state = latest_cell_state()
    applied = 0
    for key, update in state.items():
        cell = ledger.cells.get(key)
        if cell is None:
            continue
        for field in ("status", "mechanism", "owner", "recoverable_estimate"):
            if field in update and update[field] is not None:
                cell[field] = update[field]
        applied += 1
    return {"cells_in_history": len(state), "cells_applied": applied}


def recoverable_mass(ledger: Any) -> float:
    """회수가능질량. **천장 도달 셀은 뺀다** — 물리적으로 회수 불가이기 때문이다."""
    total = 0.0
    for cell in ledger.cells.values():
        if str(cell.get("status")) == AT_CEILING:
            continue
        total += float(cell.get("recoverable_if_average") or 0.0)
    return total


def stall_counter() -> int:
    """회수가능질량이 **감소하지 않은** 매크로스텝 수. 이력에서 계산한다.

    상수로 주장하지 않는 것이 요점이다. 질량이 한 번이라도 줄면 0 으로 돌아간다.

    사건 하나가 여러 사이클을 덮을 수 있다(`cycles_covered`). 계약 도입 이전에 돌았던
    사이클들은 하나의 AUDIT 사건이 묶어서 덮으며, 그 사이클들도 **실제로 일어난
    매크로스텝**이므로 정체 길이에 그대로 들어가야 한다. 사건 수만 세면 계약을 늦게
    만들었다는 이유로 정체가 짧아 보이는 왜곡이 생긴다.
    """
    events = load_history()
    if len(events) < 2:
        return 0
    stall = 0
    # 인접쌍이므로 오른쪽이 하나 짧다. `strict=True` 면 항상 터진다 — C1N60 에서 낸
    # 것과 같은 실수라 여기 명시해 둔다.
    for older, newer in zip(events, events[1:], strict=False):
        if float(newer["total_recoverable"]) < float(older["total_recoverable"]):
            stall = 0
        else:
            stall += int(newer.get("cycles_covered", 1))
    return stall


def refresh_due(current_champion_total: float, tolerance: float = 1e-9) -> bool:
    """원장의 기준 점수와 현 챔피언이 어긋나면 원장이 낡은 것이다."""
    events = load_history()
    if not events:
        return True
    return abs(float(events[-1]["champion_total"]) - float(current_champion_total)) > tolerance


def unrecorded_cycles(cycle_ids: list[str]) -> list[str]:
    """`CYCLES` 에 있는데 이력에 없는 노드. 계약 준수 감사다."""
    seen = {e["node_id"] for e in load_history()}
    return sorted(set(cycle_ids) - seen)


# ---------------------------------------------------------------- 백필


def backfill_baseline(ledger: Any) -> dict[str, Any] | None:
    """A7 이 원장을 세운 사건을 이력의 첫 항목으로 넣는다. 이미 있으면 아무것도 안 한다."""
    if load_history():
        return None
    a7 = json.loads(
        (REPORTS / "m271_n0_deficit_init_receipt.json").read_text(encoding="utf-8")
    )["result"]
    return record(
        "A7_deficit_init",
        "BASELINE",
        champion_total=float(a7["official"]["total"]),
        total_recoverable=recoverable_mass(ledger),
        note=(
            f"A7 이 정책 {a7['source']['policy']} 를 채점해 원장을 세웠다. "
            "이후 어떤 사이클도 갱신하지 않았고, 그것이 stall 77 의 원인이다."
        ),
    )


def backfill_ceiling_attribution(ledger: Any) -> dict[str, Any] | None:
    """C1N63 의 셀 단위 천장 계산을 귀속한다. **receipt 가 직접 지지하는 유일한 백필**이다.

    77 개 사이클을 사후에 셀에 배정하지 않는다 — 그것이 이 사태를 만든 습관이다.
    C1N63 은 108 셀 각각에 실현 단위와 물리 천장을 계산했고 `over_ceiling` 을 셀 단위로
    냈다. 천장을 넘은 셀은 **동류 평균 기준으로는 회수 가능해 보이지만 물리적으로는
    아니다** — 그 구분이 이 백필의 내용이다.
    """
    path = REPORTS / "m271_cycle63_ceiling_rebase_receipt.json"
    if not path.exists():
        return None
    if any(e["node_id"] == "C1N63_CEILING_REBASE" for e in load_history()):
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [*payload.get("top_headroom", []), *payload.get("top_recoverable", [])]
    updates: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("key")
        if not key or key in updates:
            continue
        if bool(row.get("over_ceiling")):
            updates[key] = {
                "status": AT_CEILING,
                "mechanism": "REALISED_EXCEEDS_PHYSICAL_CEILING",
                "owner": "C1N63_CEILING_REBASE",
                "recoverable_estimate": 0.0,
            }
        else:
            updates[key] = {
                "status": EXPLAINED,
                "mechanism": "HEADROOM_MEASURED_AGAINST_C57B_CEILING",
                "owner": "C1N63_CEILING_REBASE",
                "recoverable_estimate": float(row.get("headroom_ceiling", 0.0) or 0.0),
            }
    if not updates:
        return None

    for key, update in updates.items():
        cell = ledger.cells.get(key)
        if cell is not None:
            for field in ("status", "mechanism", "owner", "recoverable_estimate"):
                cell[field] = update[field]

    return record(
        "C1N63_CEILING_REBASE",
        "CELL_ATTRIBUTION",
        champion_total=float(
            json.loads(
                (REPORTS / "m271_n0_deficit_init_receipt.json").read_text(encoding="utf-8")
            )["result"]["official"]["total"]
        ),
        total_recoverable=recoverable_mass(ledger),
        cell_updates=updates,
        note=(
            "C1N63 이 108 셀에 물리 천장 대비 여유를 계산했다. 천장을 넘은 셀은 "
            "AT_CEILING 으로 회수가능질량에서 제외한다 — 동류 평균 기준으로는 회수 "
            "가능해 보이지만 물리적으로 아니다."
        ),
    )


def contract_state(ledger: Any, champion_total: float, cycle_ids: list[str]) -> dict[str, Any]:
    """라우터가 읽을 계약 상태."""
    missing = unrecorded_cycles(cycle_ids)
    return {
        "contract_version": CONTRACT_VERSION,
        "events": len(load_history()),
        "stall_counter": stall_counter(),
        "refresh_due": refresh_due(champion_total),
        "recoverable_mass": recoverable_mass(ledger),
        "cells_attributed": sum(
            1 for c in ledger.cells.values() if str(c.get("status")) != UNEXPLAINED
        ),
        "cells_total": len(ledger.cells),
        "unrecorded_cycles": len(missing),
        "unrecorded_sample": missing[:8],
    }
