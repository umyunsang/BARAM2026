"""Canonical content hashing for sources, manifests, and policies."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_canonical_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return to_canonical_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): to_canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [to_canonical_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        to_canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
