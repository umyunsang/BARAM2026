"""Canonical JSON receipts and append-only experiment run registry."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from baram.contracts.hashing import to_canonical_value
from baram.contracts.types import RunManifest, V2StageManifest
from baram.exceptions import ContractError


def _payload(value: Any) -> str:
    return json.dumps(
        to_canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def write_json_atomic(path: Path, value: Any) -> str:
    payload = _payload(value)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)
    return digest


def append_canonical_jsonl(path: Path, value: Any) -> str:
    payload = _payload(value)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if payload not in existing:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(payload + "\n")
    return digest


def append_run_manifest(path: Path, manifest: RunManifest) -> str:
    record = asdict(manifest)
    payload = _payload(record)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                existing = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(f"run registry contains invalid JSON: {error}") from error
            if existing.get("run_id") == manifest.run_id and line != payload:
                raise ContractError(f"run_id conflict: {manifest.run_id}")
    return append_canonical_jsonl(path, record)


def write_v2_stage_manifest_atomic(path: Path, manifest: V2StageManifest) -> str:
    """Write one stage once; exact retries are idempotent and rebinding fails closed."""
    record = asdict(manifest)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ContractError(f"v2 stage registry contains invalid JSON: {error}") from error
        if existing != to_canonical_value(record):
            raise ContractError(f"v2 stage manifest conflict: {manifest.run_id}/{manifest.stage}")
    return write_json_atomic(path, record)
