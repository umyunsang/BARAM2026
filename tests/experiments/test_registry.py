from dataclasses import asdict
from pathlib import Path

import pytest

from baram.contracts.types import RunManifest
from baram.exceptions import ContractError
from baram.experiments.registry import (
    append_canonical_jsonl,
    append_run_manifest,
    write_json_atomic,
)


def _run(run_id: str = "run-a", prediction: str = "f" * 64) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        source_sha256="a" * 64,
        code_sha256="b" * 64,
        split_sha256="c" * 64,
        feature_sha256="d" * 64,
        model_sha256="e" * 64,
        prediction_sha256=prediction,
        metric_sha256="1" * 64,
        environment_sha256="2" * 64,
        seed=1,
        runtime_seconds=1.25,
        hardware_tier="test",
    )


def test_json_writers_are_stable_and_idempotent(tmp_path: Path) -> None:
    """Catches noncanonical output and duplicate append records."""
    json_path = tmp_path / "receipt.json"
    first = write_json_atomic(json_path, {"b": 2, "a": 1})
    second = write_json_atomic(json_path, {"a": 1, "b": 2})
    assert first == second
    assert json_path.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'

    ledger = tmp_path / "events.jsonl"
    assert append_canonical_jsonl(ledger, {"b": 2, "a": 1}) == first
    append_canonical_jsonl(ledger, {"a": 1, "b": 2})
    assert ledger.read_text(encoding="utf-8").splitlines() == ['{"a":1,"b":2}']


def test_run_registry_rejects_conflicting_reuse(tmp_path: Path) -> None:
    """Catches a run ID being silently rebound to different lineage."""
    ledger = tmp_path / "runs.jsonl"
    first = _run()
    append_run_manifest(ledger, first)
    append_run_manifest(ledger, first)
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(ContractError, match="run_id conflict"):
        append_run_manifest(ledger, _run(prediction="9" * 64))
    assert asdict(first)["run_id"] == "run-a"
