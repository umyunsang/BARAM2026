"""M271 증거 정규화 — 게이트 결과를 라우터가 읽을 서명으로.

이 파일의 핵심 역할은 `m270_gate.py` 의 출력을 **C3/C4 가 구분할 수 있는 서명**으로 바꾸는
것이다. 그 게이트는 조건 키를 사람이 읽을 긴 문자열로 만든다.

    "G1 sign-test p <= 0.10 (p=0.0898, 7/9)": True
    "G3 bootstrap q05 > 0 (q05=-0.001224)": False

라우터는 `{"G1": True, "G3": False, ...}` 를 원한다. 문자열 앞의 `G<n>` 토큰만 취해 정규화
하며, 4 개가 정확히 나오지 않으면 오류다. 조용히 일부만 읽으면 C3/C4 가 잘못 갈린다.

**게이트 자체는 건드리지 않는다.** `m270_gate.py` 는 2026-08-04 에 동결됐고 재동결이
금지되어 있다. 여기서는 읽기만 한다.

부호 규약: `sign` 은 관측된 효과의 부호, `predeclared_sign` 은 실행 전에 동결한 기대 부호다.
둘이 다르면 C5(anomaly)가 발화한다. 사전확약이 없으면 `predeclared_sign = 0` 이고 C5 는
발화하지 않는다 — 기대를 정하지 않았으면 배신당할 수도 없다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m270_gate import GATE_VERSION, GateResult
from m271_router import Evidence

GATE_KEYS = ("G1", "G2", "G3", "G4")
_GATE_TOKEN = re.compile(r"^\s*(G[1-4])\b")


class GateSignatureError(RuntimeError):
    """게이트 조건을 G1~G4 로 정규화하지 못했다. 조용히 넘기면 C3/C4 가 잘못 갈린다."""


def normalize_gate(result: GateResult) -> dict[str, bool]:
    """긴 조건 문자열에서 `G<n>` 토큰만 뽑아 라우터가 읽을 평면 서명으로."""
    flags: dict[str, bool] = {}
    for label, passed in result.conditions.items():
        match = _GATE_TOKEN.match(str(label))
        if not match:
            raise GateSignatureError(f"gate condition has no G<n> token: {label!r}")
        key = match.group(1)
        if key in flags:
            raise GateSignatureError(f"duplicate gate key {key} in {list(result.conditions)}")
        flags[key] = bool(passed)
    missing = [k for k in GATE_KEYS if k not in flags]
    if missing:
        raise GateSignatureError(f"gate signature is incomplete, missing {missing}")
    return {k: flags[k] for k in GATE_KEYS}


def effect_sign(stats: dict[str, Any]) -> int:
    """관측된 효과의 부호. 중앙값 델타를 기준으로 한다(평균은 이상월에 끌린다)."""
    median = stats.get("median_total_delta")
    if median is None or median != median:  # NaN
        return 0
    if median > 0:
        return 1
    if median < 0:
        return -1
    return 0


def information_content(stats: dict[str, Any]) -> float:
    """이 실험이 만들어낸 정보량의 대용값 (0~1).

    효과크기 자체가 아니라 **판정을 얼마나 갈랐는지**를 잰다. 부호가 한쪽으로 몰릴수록,
    부트스트랩 분포가 0 에서 멀수록 정보가 있다. 회수 가능성이 아니라 구분력의 척도다.

    이 함수는 **선언 관례**다. 보정된 바 없으며 C8(정보량 미달 -> PRUNE)의 임계와 함께
    계획 R3 의 잔여 리스크에 속한다.
    """
    n = int(stats.get("months_scored", 0) or 0)
    if n == 0:
        return 0.0
    positive_fraction = float(stats.get("positive_fraction", 0.5))
    # 0.5 에서 멀수록 판정이 갈렸다.
    consistency = abs(positive_fraction - 0.5) * 2.0
    boot = float(stats.get("block_bootstrap_positive_fraction", 0.5))
    decisiveness = abs(boot - 0.5) * 2.0
    return float(max(0.0, min(1.0, 0.5 * consistency + 0.5 * decisiveness)))


def build_evidence(
    *,
    evidence_id: str,
    node_id: str,
    lane: str,
    gate_result: GateResult | None,
    stats: dict[str, Any] | None = None,
    deficit_cell: str | None = None,
    predeclared_sign: int = 0,
    refutes_mechanism: str | None = None,
    novel_mechanism: str | None = None,
    expected_gain: float = 0.0,
    expected_hours: float = 1.0,
) -> Evidence:
    """노드 실행 결과를 라우터가 읽는 유일한 입력으로 바꾼다."""
    stats = stats or (dict(gate_result.evidence) if gate_result else {})
    gate = normalize_gate(gate_result) if gate_result else None
    sign = effect_sign(stats)
    return Evidence(
        evidence_id=evidence_id,
        node_id=node_id,
        lane=lane,
        deficit_cell=deficit_cell,
        gate=gate,
        sign=sign,
        predeclared_sign=predeclared_sign,
        information=information_content(stats),
        refutes_mechanism=refutes_mechanism,
        # 게이트를 통과했고 부호가 사전확약과 같을 때만 '확인됨' 이다.
        confirms=bool(
            gate_result is not None
            and gate_result.passed
            and (predeclared_sign == 0 or sign == predeclared_sign)
        ),
        novel_mechanism=novel_mechanism,
        expected_gain=expected_gain,
        expected_hours=expected_hours,
    )


def signature_summary(evidence: Evidence) -> dict[str, Any]:
    """리포트와 receipt 에 남길 형태."""
    return {
        "evidence_id": evidence.evidence_id,
        "node_id": evidence.node_id,
        "lane": evidence.lane,
        "deficit_cell": evidence.deficit_cell,
        "gate_version": GATE_VERSION,
        "gate": evidence.gate,
        "sign": evidence.sign,
        "predeclared_sign": evidence.predeclared_sign,
        "sign_reversed": bool(
            evidence.predeclared_sign != 0
            and evidence.sign != 0
            and evidence.sign != evidence.predeclared_sign
        ),
        "information": evidence.information,
        "confirms": evidence.confirms,
        "refutes_mechanism": evidence.refutes_mechanism,
        "novel_mechanism": evidence.novel_mechanism,
    }
