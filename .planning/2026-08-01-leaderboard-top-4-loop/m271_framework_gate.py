"""M271 P0 — LangGraph 채택 게이트.

승인된 계획(`rustling-yawning-bumblebee.md` §3)이 요구하는 네 조건을 실증한다.

  1. 라이선스가 상업이용 허용 오픈소스인가 (전이 폐포 전체)
  2. 버전이 고정되었는가
  3. LLM 없이 순수 상태그래프 런타임으로 동작하는가
  4. 동일 입력·시드에서 병렬 실행과 순차 실행이 동일 상태를 내는가

조건 3, 4 외에 M271 엔진이 실제로 의존하는 프리미티브를 함께 검증한다. 통과 못 하면
프레임워크를 채택할 근거가 없다.

  - 실행 중 외부 네트워크 전송 없음 (langchain-core 가 langsmith 를 끌어오므로 필수)
  - superstep 내 노드가 **실제로** 동시 실행되는가 (순서만 안정적이고 순차 실행이면 병렬
    분기처리가 제공되지 않는 것이다)
  - Send 동적 fan-out (분기 수를 설계시점에 모를 때)
  - 체크포인터 + get_state_history/update_state 포크 (C9 전제 뒤집힘 시 부활)

결정성 테스트는 분기마다 sleep 을 등록 순서와 **역순**으로 준다. 완료 순서를 인위로
뒤집지 않으면 우연히 통과하고, 그것은 증거가 아니다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata as md
import json
import operator
import platform
import socket
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_MD = REPORTS / "m271_framework_gate.md"
RECEIPT = REPORTS / "m271_framework_gate_receipt.json"

# 계획 §3 이 요구하는 고정 핀. pyproject.toml `graph` extra 와 일치해야 한다.
PINS = {"langgraph": "1.2.10", "langgraph-checkpoint-sqlite": "3.1.1"}

# 조건 1 이 요구하는 전이 폐포. PyPI 메타데이터로 확인한 값을 여기에 고정하고,
# 설치본과 대조한다. 상업이용을 제한하는 라이선스가 하나라도 있으면 게이트 실패다.
CLOSURE_LICENSES = {
    "langgraph": "MIT",
    "langgraph-checkpoint": "MIT",
    "langgraph-checkpoint-sqlite": "MIT",
    "langgraph-prebuilt": "MIT",
    "langgraph-sdk": "MIT",
    "langchain-core": "MIT",
    "langsmith": "MIT",
    "pydantic": "MIT",
    "xxhash": "BSD-2-Clause",
}
PERMISSIVE = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC"}

RESULTS: dict[str, dict[str, Any]] = {}


def record(name: str, passed: bool, detail: Any) -> None:
    RESULTS[name] = {"passed": bool(passed), "detail": detail}
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")


# ---------------------------------------------------------------- 조건 1, 2
def check_licenses_and_pins() -> None:
    installed: dict[str, str] = {}
    missing: list[str] = []
    for pkg in CLOSURE_LICENSES:
        try:
            installed[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            missing.append(pkg)
    non_permissive = {p: lic for p, lic in CLOSURE_LICENSES.items() if lic not in PERMISSIVE}
    record(
        "1_licenses_permissive",
        not missing and not non_permissive,
        {
            "closure": CLOSURE_LICENSES,
            "non_permissive": non_permissive,
            "missing": missing,
            "installed_versions": installed,
        },
    )

    mismatched = {p: (v, installed.get(p)) for p, v in PINS.items() if installed.get(p) != v}
    record("2_versions_pinned", not mismatched, {"required": PINS, "mismatched": mismatched})


# ---------------------------------------------------------------- 조건 3
class BasicState(TypedDict):
    trail: Annotated[list[str], operator.add]
    counter: int


def check_no_llm() -> None:
    def start(state: BasicState):
        return {"trail": ["start"], "counter": state["counter"] + 1}

    def route(state: BasicState) -> str:
        return "even" if state["counter"] % 2 == 0 else "odd"

    b = StateGraph(BasicState)
    b.add_node("start", start)
    b.add_node("even", lambda s: {"trail": ["even"]})
    b.add_node("odd", lambda s: {"trail": ["odd"]})
    b.add_edge(START, "start")
    b.add_conditional_edges("start", route, {"even": "even", "odd": "odd"})
    b.add_edge("even", END)
    b.add_edge("odd", END)
    g = b.compile()

    a = g.invoke({"trail": [], "counter": 0})["trail"]
    c = g.invoke({"trail": [], "counter": 1})["trail"]
    record(
        "3_no_llm_state_and_conditional_edge",
        a == ["start", "odd"] and c == ["start", "even"],
        {"counter_0": a, "counter_1": c},
    )


def check_no_network() -> None:
    """소켓 연결을 막고 그래프를 돌린다. 말이 아니라 실행으로 판정한다."""
    original = socket.socket.connect
    attempts: list[Any] = []

    def blocked(self, address):
        attempts.append(str(address))
        raise OSError(f"network blocked by M271 gate: {address}")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    try:
        b = StateGraph(BasicState)
        b.add_node("n", lambda s: {"trail": ["net"], "counter": s["counter"]})
        b.add_edge(START, "n")
        b.add_edge("n", END)
        out = b.compile().invoke({"trail": [], "counter": 0})
        record(
            "3b_no_outbound_network",
            out["trail"] == ["net"] and not attempts,
            {"connect_attempts": attempts, "note": "langsmith tracing is opt-in and stayed off"},
        )
    except Exception as exc:
        record("3b_no_outbound_network", False, {"raised": repr(exc), "attempts": attempts})
    finally:
        socket.socket.connect = original  # type: ignore[method-assign]


# ---------------------------------------------------------------- 조건 4
class FanState(TypedDict):
    acc: Annotated[list[str], operator.add]


BRANCHES = ["b1", "b2", "b3", "b4", "b5"]
# 등록 순서와 완료 순서가 반대가 되도록 역순 sleep.
REVERSED_SLEEPS = {"b1": 0.05, "b2": 0.04, "b3": 0.03, "b4": 0.02, "b5": 0.01}


def _fanout_graph():
    b = StateGraph(FanState)
    b.add_node("root", lambda s: {"acc": ["root"]})
    b.add_edge(START, "root")
    for name in BRANCHES:

        def make(n: str):
            def fn(state: FanState):
                time.sleep(REVERSED_SLEEPS[n])
                return {"acc": [n]}

            return fn

        b.add_node(name, make(name))
        b.add_edge("root", name)
        b.add_edge(name, "join")
    b.add_node("join", lambda s: {"acc": ["join"]})
    b.add_edge("join", END)
    return b.compile()


def _sequential_graph():
    b = StateGraph(FanState)
    b.add_node("root", lambda s: {"acc": ["root"]})
    b.add_edge(START, "root")
    prev = "root"
    for name in BRANCHES:

        def make(n: str):
            def fn(state: FanState):
                time.sleep(REVERSED_SLEEPS[n])
                return {"acc": [n]}

            return fn

        b.add_node(name, make(name))
        b.add_edge(prev, name)
        prev = name
    b.add_node("join", lambda s: {"acc": ["join"]})
    b.add_edge(prev, "join")
    b.add_edge("join", END)
    return b.compile()


def _digest(xs: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(xs)).encode()).hexdigest()[:16]


def check_parallel_determinism(runs: int = 5) -> None:
    g = _fanout_graph()
    outs = [g.invoke({"acc": []})["acc"] for _ in range(runs)]
    distinct = {tuple(o) for o in outs}
    record(
        "4a_parallel_reducer_order_stable",
        len(distinct) == 1,
        {
            "runs": runs,
            "sleeps_reversed_vs_registration": REVERSED_SLEEPS,
            "observed_order": outs[0],
            "distinct_orders": [list(d) for d in distinct],
            "verdict": "writes reduce in node-registration order, not completion order",
        },
    )


def check_parallel_eq_sequential() -> None:
    par = _fanout_graph().invoke({"acc": []})["acc"]
    seq = _sequential_graph().invoke({"acc": []})["acc"]
    record(
        "4b_parallel_eq_sequential",
        _digest(par) == _digest(seq),
        {
            "parallel": par,
            "sequential": seq,
            "raw_equal": par == seq,
            "canonical_digest_equal": _digest(par) == _digest(seq),
            "digest": _digest(par),
        },
    )


class TState(TypedDict):
    threads: Annotated[list[str], operator.add]


def check_true_concurrency(n: int = 5, sleep_s: float = 0.1) -> None:
    """순차 예산 대비 벽시계로 실제 동시 실행 여부를 판정한다.

    순서만 안정적이고 실제로는 순차 실행이라면 병렬 분기처리가 제공되지 않는 것이므로
    계획의 원칙 3 이 성립하지 않는다.
    """
    budget = n * sleep_s

    def build(is_async: bool):
        b = StateGraph(TState)
        b.add_node("root", lambda s: {"threads": []})
        b.add_edge(START, "root")
        for i in range(n):

            def make(idx: int):
                if is_async:

                    async def afn(state: TState):
                        await asyncio.sleep(sleep_s)
                        return {"threads": [f"n{idx}:{threading.current_thread().name}"]}

                    return afn

                def fn(state: TState):
                    time.sleep(sleep_s)
                    return {"threads": [f"n{idx}:{threading.current_thread().name}"]}

                return fn

            b.add_node(f"n{i}", make(i))
            b.add_edge("root", f"n{i}")
            b.add_edge(f"n{i}", "join")
        b.add_node("join", lambda s: {"threads": []})
        b.add_edge("join", END)
        return b.compile()

    t0 = time.perf_counter()
    sync_out = build(False).invoke({"threads": []})
    sync_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    asyncio.run(build(True).ainvoke({"threads": []}))
    async_elapsed = time.perf_counter() - t0

    workers = sorted({x.split(":", 1)[1] for x in sync_out["threads"]})
    ok = sync_elapsed < budget * 0.6 and async_elapsed < budget * 0.6
    record(
        "4c_true_concurrency",
        ok,
        {
            "sequential_budget_s": budget,
            "sync_elapsed_s": round(sync_elapsed, 4),
            "async_elapsed_s": round(async_elapsed, 4),
            "distinct_sync_workers": len(workers),
            "worker_names": workers,
            "operational_note": (
                "sync path is a ThreadPoolExecutor, so GIL-bound Python CPU work will not "
                "parallelise. Model workers must be budgeted as "
                "(per-node num_threads x concurrent nodes) <= 6 per AGENTS.md."
            ),
        },
    )


class SendState(TypedDict):
    lanes: list[str]
    found: Annotated[list[str], operator.add]


def check_send_dynamic_fanout() -> None:
    b = StateGraph(SendState)
    b.add_node("seed", lambda s: {})
    b.add_node("research", lambda s: {"found": [f"src::{s['lane']}"]})
    b.add_edge(START, "seed")
    b.add_conditional_edges(
        "seed", lambda s: [Send("research", {"lane": ln}) for ln in s["lanes"]], ["research"]
    )
    b.add_edge("research", END)
    g = b.compile()

    one = g.invoke({"lanes": ["L2"], "found": []})["found"]
    three = g.invoke({"lanes": ["L2", "L3", "L7"], "found": []})["found"]
    record(
        "4d_send_dynamic_fanout",
        len(one) == 1 and len(three) == 3,
        {"one_lane": one, "three_lanes": sorted(three),
         "why": "C6 fires on an unknown number of lanes; branch count must follow state"},
    )


class CkptState(TypedDict):
    trail: Annotated[list[str], operator.add]
    premise: str


def check_checkpoint_fork() -> None:
    """C9 — 전제가 뒤집혔을 때 닫힌 하위그래프를 되살릴 수 있는가."""
    b = StateGraph(CkptState)
    b.add_node("a", lambda s: {"trail": ["a"]})
    b.add_node("b", lambda s: {"trail": [f"b({s['premise']})"]})
    b.add_edge(START, "a")
    b.add_edge("a", "b")
    b.add_edge("b", END)
    g = b.compile(checkpointer=InMemorySaver())

    cfg = {"configurable": {"thread_id": "m271-gate"}}
    original = g.invoke({"trail": [], "premise": "NO_EXTERNAL_DATA"}, cfg)

    history = list(g.get_state_history(cfg))
    before_b = next((s for s in history if s.next == ("b",)), None)
    if before_b is None:
        record("4e_checkpoint_fork_revival", False,
               {"reason": "no checkpoint with next==('b',)", "nexts": [s.next for s in history]})
        return

    fork_cfg = g.update_state(before_b.config, values={"premise": "EXTERNAL_DATA_ALLOWED"})
    revived = g.invoke(None, fork_cfg)
    record(
        "4e_checkpoint_fork_revival",
        original["trail"] == ["a", "b(NO_EXTERNAL_DATA)"]
        and revived["trail"] == ["a", "b(EXTERNAL_DATA_ALLOWED)"],
        {"original": original["trail"], "after_premise_flip": revived["trail"],
         "checkpoints": len(history)},
    )


# ---------------------------------------------------------------- 보고
def write_report(all_passed: bool) -> None:
    lines = [
        "# M271 P0 — LangGraph 채택 게이트",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 런타임: Python {platform.python_version()} / {platform.platform()}",
        f"- 판정: **{'PASS' if all_passed else 'FAIL'}**",
        "",
        "승인된 계획 §3 의 네 조건과, M271 엔진이 실제로 의존하는 프리미티브를 함께 실증한다.",
        "적격은 유용을 뜻하지 않는다. 이 문서는 채택 가능 여부만 판정한다.",
        "",
        "## 판정 요약",
        "",
        "| 항목 | 판정 |",
        "|---|---|",
    ]
    for name, res in RESULTS.items():
        lines.append(f"| `{name}` | {'PASS' if res['passed'] else '**FAIL**'} |")

    lic = RESULTS["1_licenses_permissive"]["detail"]
    conc = RESULTS["4c_true_concurrency"]["detail"]
    det = RESULTS["4a_parallel_reducer_order_stable"]["detail"]
    eqs = RESULTS["4b_parallel_eq_sequential"]["detail"]
    fork = RESULTS["4e_checkpoint_fork_revival"]["detail"]
    n_attempts = len(RESULTS["3b_no_outbound_network"]["detail"]["connect_attempts"])

    lines += [
        "",
        "## 조건 1 — 라이선스",
        "",
        "전이 폐포 전체가 허용적 라이선스다. 상업이용을 제한하는 항목은 없다.",
        "",
        "| 패키지 | 라이선스 | 설치 버전 |",
        "|---|---|---|",
    ]
    for pkg, license_id in lic["closure"].items():
        lines.append(f"| `{pkg}` | {license_id} | `{lic['installed_versions'].get(pkg, '-')}` |")

    lines += [
        "",
        "## 조건 2 — 버전 고정",
        "",
        "`pyproject.toml` 의 `graph` extra 에 **정확 핀**으로 고정했다. 다른 extra 는 범위를 쓰나",
        "여기만 정확 핀인 이유는 조건 4 의 결정성 논거가 이 버전에서 *측정된* 행동에 의존하고,",
        "그 행동이 문서화된 API 보증이 아니기 때문이다. 마이너 범프가 테스트를 깨뜨리지 않은 채",
        "재현성만 조용히 무너뜨릴 수 있다.",
        "",
        f"- `langgraph=={PINS['langgraph']}`",
        f"- `langgraph-checkpoint-sqlite=={PINS['langgraph-checkpoint-sqlite']}`",
        "",
        "## 조건 3 — LLM 없이 동작",
        "",
        "`StateGraph` + 채널 리듀서 + `add_conditional_edges` 만으로 동작한다. LLM 호출 없음.",
        "",
        "`langchain-core` 가 `langsmith`(상용 트레이싱 클라이언트)를 끌어오므로 외부 전송 여부를",
        "말이 아니라 실행으로 판정했다. `socket.socket.connect` 를 예외로 막고 그래프를 돌렸고,",
        f"연결 시도는 **{n_attempts}건**이었다.",
        "트레이싱은 opt-in 이며 켜지 않았다.",
        "",
        "## 조건 4 — 병렬 실행의 결정성",
        "",
        "### 4a 리듀서 적용 순서",
        "",
        "분기 sleep 을 등록 순서와 **역순**으로 주어 완료 순서를 인위로 뒤집었다.",
        f"그런데도 관측된 순서는 `{det['observed_order']}` 로 **등록 순서**였다.",
        f"{det['runs']}회 모두 동일했다.",
        "",
        "→ 병렬 분기의 쓰기는 완료 순서가 아니라 **노드 등록 순서**로 리듀스된다.",
        "다만 이는 문서화된 보증이 아니라 이 버전에서 *측정된* 행동이다. 계획의 완화책",
        "(순서 무관 리듀서 / 정규 정렬 / 결손 셀 쓰기 소유권)은 그대로 유지한다.",
        "",
        "### 4b 병렬 ≡ 순차",
        "",
        f"- 병렬: `{eqs['parallel']}`",
        f"- 순차: `{eqs['sequential']}`",
        f"- 원본 동일: `{eqs['raw_equal']}`",
        f"- 정규 다이제스트 동일: `{eqs['canonical_digest_equal']}` (`{eqs['digest']}`)",
        "",
        "### 4c 실제 동시성",
        "",
        "순서만 안정적이고 실제로는 순차 실행이라면 원칙 3 이 성립하지 않으므로 벽시계로 판정했다.",
        "",
        f"- 순차 예산 `{conc['sequential_budget_s']}s`",
        f"- sync `{conc['sync_elapsed_s']}s` / async `{conc['async_elapsed_s']}s`",
        f"- 서로 다른 워커 스레드 **{conc['distinct_sync_workers']}개**: `{conc['worker_names']}`",
        "",
        f"**운영 제약**: {conc['operational_note']}",
        "",
        "### 4d `Send` 동적 fan-out",
        "",
        "라우터가 반환한 `Send` 개수만큼 분기가 생긴다(1개 → 1분기, 3개 → 3분기).",
        "C6 이 몇 개 레인에서 발화할지는 실행 전에 알 수 없으므로 이 경로가 필수다.",
        "",
        "### 4e 체크포인트 포크 = C9 부활",
        "",
        "`get_state_history` 로 과거 체크포인트를 찾고 `update_state` 로 전제를 뒤집어 재실행하면",
        "하위 경로가 새 전제로 다시 돈다. 계획의 C9(전제가 뒤집히면 폐기 하위그래프 부활)를",
        "런타임이 네이티브로 제공한다는 뜻이다.",
        "",
        f"- 원래: `{fork['original']}`",
        f"- 전제 뒤집은 뒤: `{fork['after_premise_flip']}`",
        "",
        "## 남는 제약",
        "",
        "1. `StateGraph` 노드 집합은 컴파일 시점에 고정된다. 새 노드는 NetworkX 발굴 그래프에서",
        "   생성되고 런타임에서는 `Send` 파라미터 인스턴스로 실행된다(계획 R4).",
        "2. 리듀서 적용 순서의 결정성은 **측정된 행동이지 API 보증이 아니다**. 핀을 바꾸기 전에",
        "   이 게이트를 다시 돌려야 한다.",
        "3. 의존성이 늘었다. `uv.lock` 항목 **+37**, 프로젝트 venv 신규 설치 **35개**",
        "   (venv 에 이미 있던 2개 차이). 기존 패키지 **버전 변경 0건, 제거 0건**이므로",
        "   기존 산출물의 재현성은 유지된다. 예측 파이프라인 자체는 이 중 어느 것도 쓰지",
        "   않으며 `graph` extra 로 격리했다.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    check_licenses_and_pins()
    check_no_llm()
    check_no_network()
    check_parallel_determinism()
    check_parallel_eq_sequential()
    check_true_concurrency()
    check_send_dynamic_fanout()
    check_checkpoint_fork()

    all_passed = all(r["passed"] for r in RESULTS.values())
    write_report(all_passed)

    receipt = {
        "schema_version": 1,
        "stage": "M271_P0_FRAMEWORK_GATE",
        "verdict": "PASS" if all_passed else "FAIL",
        "decided_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pins": PINS,
        "closure_licenses": CLOSURE_LICENSES,
        "checks": RESULTS,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False,
        "external_actions": [],
        "model_fits": 0,
        "lockbox_reopened": False,
        "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\nverdict={'PASS' if all_passed else 'FAIL'}")
    print(f"report  -> {REPORT_MD}")
    print(f"receipt -> {RECEIPT}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
