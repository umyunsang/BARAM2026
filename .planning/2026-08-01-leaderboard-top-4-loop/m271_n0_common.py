"""M271 N0 자식 노드 공통 계약 — 로딩, 재현 계약, 산출물 기록.

계획의 노드별 재현 계약(§4.5)을 강제한다. 모든 노드는 예외 없이 다음을 갖는다.

    spec_hash / seed / threads / code_version / input_hashes / receipt_path
    artifact_hash / parents / premise

사양은 `m271_n0_method.py` 에서 **실행 전에** 동결됐다. 노드는 자기 사양을 여기서 읽어
가고, 사양에 없는 산출을 만들지 않는다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from baram.constants import OPEN_ZIP_SHA256
from baram.data.archive import validate_archive
from baram.data.canonical import CanonicalTables, load_canonical_tables

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PLAN_DIR = Path(__file__).resolve().parent
SOURCE_ZIP = Path("/Users/um-yunsang/Downloads/open.zip")

SEED = 20260804
THREADS = 1  # 노드 하나당 1스레드. 동시 노드 수와 곱해도 AGENTS.md 의 6 상한 안에 든다.


def code_version() -> str:
    """HEAD 커밋. 작업 트리가 더러우면 그 사실도 함께 기록한다."""
    try:
        head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        return f"{head}{'+dirty' if dirty else ''}"
    except (subprocess.SubprocessError, OSError):
        return "unavailable"


def frozen_spec(node: str) -> dict[str, Any]:
    """동결된 사양을 읽어 온다. 사양 없이 실행하는 노드는 없다."""
    import m271_n0_method

    if node not in m271_n0_method.SPECS:
        raise KeyError(f"node has no frozen spec: {node}")
    return {"spec": m271_n0_method.SPECS[node], "spec_hash": m271_n0_method.spec_hash()}


def load_tables() -> tuple[CanonicalTables, dict[str, str]]:
    """공급 ZIP 을 해시 검증한 뒤 정규 테이블로 읽는다. 추출하지 않는다."""
    manifest = validate_archive(SOURCE_ZIP, OPEN_ZIP_SHA256)
    tables = load_canonical_tables(SOURCE_ZIP)
    return tables, {"open_zip_sha256": manifest.source_sha256}


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_node_artifacts(
    node: str,
    title: str,
    report_lines: list[str],
    payload: dict[str, Any],
    input_hashes: dict[str, str],
    parents: list[str],
    script_path: Path,
) -> dict[str, Any]:
    """노드 리포트와 receipt 를 기록하고 재현 계약 필드를 채운다."""
    meta = frozen_spec(node)
    header = [
        f"# {title}",
        "",
        f"- 노드: `{node}` (M271 N0 자식)",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- `spec_hash`: `{meta['spec_hash']}`",
        f"- 부모: {', '.join(parents) if parents else '(N0 루트)'}",
        "",
        "사양은 실행 전에 동결됐다. 결과를 본 뒤 사양을 재해석하지 않는다.",
        "",
    ]
    report_md = "\n".join(header + report_lines) + "\n"
    report_path = REPORTS / f"m271_n0_{node.split('_', 1)[1]}.md"
    report_path.write_text(report_md, encoding="utf-8")

    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    artifact_hash = _digest_text(canonical)
    receipt = {
        "schema_version": 1,
        "stage": f"M271_N0_{node.upper()}",
        "node": node,
        "inner_loop_steps": ["3_code", "4_run", "5_evaluate"],
        "decided_utc": datetime.now(UTC).isoformat(),
        "spec_hash": meta["spec_hash"],
        "spec": meta["spec"],
        "seed": SEED,
        "threads": THREADS,
        "code_version": code_version(),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "input_hashes": input_hashes,
        "parents": parents,
        "premise": None,
        "artifact_hash": artifact_hash,
        "report_sha256": _digest_text(report_md),
        "result": payload,
        "dacon_upload": False,
        "external_actions": [],
        "model_fits": 0,
        "lockbox_reopened": False,
        "new_2024_evaluation": False,
    }
    receipt_path = REPORTS / f"m271_n0_{node.split('_', 1)[1]}_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {"node": node, "artifact_hash": artifact_hash, "report": str(report_path),
            "receipt": str(receipt_path)}


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"
