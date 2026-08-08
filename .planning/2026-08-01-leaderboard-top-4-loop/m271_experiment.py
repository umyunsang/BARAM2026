"""M271 실험 실행기 — 동적 후보 공간과 C11 선택.

직전 세션이 멈춘 지점이 여기다. `m270_control.py:73-118` 의 `CANDIDATE_LANES` 는 하드코딩
4-튜플이었고 `seed_queue()` 가 그 4 개에 대한 **고정** `CategoricalDistribution` 을 넣었다.
루프는 후보를 소비만 하고 생성하지 못했으므로 4 개를 다 쓰면 멈췄다.

바뀐 점:

  * 후보 집합이 **런타임에 자란다**. `enqueue_hypothesis` 는 언제든 새 가설을 받는다.
  * 선택은 Optuna 샘플러가 아니라 **C11 결정 함수**가 한다. 하이퍼파라미터 자동탐색은 계속
    금지다 — 이 프로젝트가 반복적으로 경계해 온 동일-fold 과적합을 다시 불러온다.
  * Optuna 는 **영속 부기**만 맡는다. SQLite study 로 세션 중단을 복구한다.

모델 워커 예산: `AGENTS.md` 는 동시 6 을 상한으로 둔다. P0 게이트가 LangGraph sync 경로가
`ThreadPoolExecutor` 임을 확인했으므로 상한은 **노드당 스레드 x 동시 노드 수** 로 관리해야
한다. 노드마다 6 을 주고 5 개를 동시에 돌리면 30 이 된다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))

PLAN_DIR = Path(__file__).resolve().parent
STUDY_DB = PLAN_DIR / "m271_study.db"
STUDY_NAME = "m271_excavation_loop"

MAX_MODEL_WORKERS = 6  # AGENTS.md


class WorkerBudgetExceeded(RuntimeError):
    """노드당 스레드 x 동시 노드 수가 상한을 넘었다."""


@dataclass(frozen=True)
class Candidate:
    """선택 대상. 프론티어의 LIVE 노드에서 만들어진다."""

    node_id: str
    lane: str
    deficit_cell: str | None
    expected_gain: float  # Total 단위
    expected_hours: float
    voi: float = 0.0  # 정보가치 — 점수를 직접 올리지 않는 실험용
    exploration: float = 0.0  # 탐험항 — 없으면 한 계보로 붕괴한다
    direction_id: str | None = None


def value(candidate: Candidate) -> float:
    """C11 가치 = 시간당 기대이득 + 정보가치 + 탐험항.

    **선언 관례**다. 세 항의 상대 비중에 근거가 없으며 계획 R3 의 잔여 리스크에 속한다.
    """
    return (
        candidate.expected_gain / max(candidate.expected_hours, 1e-6)
        + candidate.voi
        + candidate.exploration
    )


def select_next(candidates: list[Candidate]) -> Candidate | None:
    """결정적 선택. 가치 최대, 동률은 node_id 사전순으로 동결한다.

    탐험항이 0 이면 기대이득 최대 계보로 붕괴해 챔피언을 놓친다. 그래서 탐험항을 가치의
    1급 항으로 둔다.
    """
    if not candidates:
        return None
    return sorted(candidates, key=lambda c: (-value(c), c.node_id))[0]


def open_study(storage: Path | None = None) -> optuna.Study:
    """영속 study. 샘플러는 쓰지 않지만 시드를 고정해 부기 순서를 재현 가능하게 둔다."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    path = storage or STUDY_DB
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{path}",
        direction="maximize",  # 공식 Total on strict chronology-safe OOF
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=20260804),
    )


def enqueue_hypothesis(study: optuna.Study, candidate: Candidate) -> int:
    """새 가설을 언제든 등록한다. 고정 공간이 아니므로 개수 제한이 없다.

    이것이 직전 세션과의 결정적 차이다. `CANDIDATE_LANES` 는 4 개로 닫혀 있었고 여기는
    열려 있다.
    """
    existing = {
        t.user_attrs.get("node_id") for t in study.get_trials(deepcopy=False)
    }
    if candidate.node_id in existing:
        return -1
    trial = study.ask()
    trial.set_user_attr("node_id", candidate.node_id)
    trial.set_user_attr("lane", candidate.lane)
    trial.set_user_attr("deficit_cell", candidate.deficit_cell)
    trial.set_user_attr("direction_id", candidate.direction_id)
    trial.set_user_attr("expected_gain", candidate.expected_gain)
    trial.set_user_attr("expected_hours", candidate.expected_hours)
    trial.set_user_attr("value", value(candidate))
    trial.set_user_attr("state", "PREDECLARATION_REQUIRED")
    return trial.number


def report_result(study: optuna.Study, node_id: str, total: float) -> None:
    """실험 1 회 결과를 부기에 남긴다."""
    for trial in study.get_trials(deepcopy=False):
        if trial.user_attrs.get("node_id") == node_id and trial.state.is_finished() is False:
            study.tell(trial.number, total)
            return
    raise KeyError(f"no running trial for node {node_id}")


def check_worker_budget(concurrent_nodes: int, threads_per_node: int) -> None:
    """노드당 스레드 x 동시 노드 수가 6 을 넘으면 거부한다."""
    total = concurrent_nodes * threads_per_node
    if total > MAX_MODEL_WORKERS:
        raise WorkerBudgetExceeded(
            f"{concurrent_nodes} nodes x {threads_per_node} threads = {total} "
            f"exceeds the AGENTS.md cap of {MAX_MODEL_WORKERS}"
        )


def queue_summary(study: optuna.Study) -> dict[str, Any]:
    trials = study.get_trials(deepcopy=False)
    return {
        "study": STUDY_NAME,
        "trials": len(trials),
        "nodes": sorted(
            str(t.user_attrs.get("node_id")) for t in trials if t.user_attrs.get("node_id")
        ),
        "lanes": sorted(
            {str(t.user_attrs.get("lane")) for t in trials if t.user_attrs.get("lane")}
        ),
        "open_space": True,
        "note": (
            "후보 공간이 런타임에 자란다. m270 의 고정 4-원소 CategoricalDistribution 과 "
            "다르며, 목록 소진으로 루프가 종결되지 않는다."
        ),
    }
