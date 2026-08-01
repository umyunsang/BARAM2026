from dataclasses import dataclass
from pathlib import Path

import pytest

from baram.contracts.hashing import canonical_sha256, to_canonical_value


def test_canonical_hash_ignores_mapping_order() -> None:
    """Catches nondeterministic manifest hashes from mapping insertion order."""
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_canonical_value_normalizes_supported_types() -> None:
    """Catches failure to normalize paths, dataclasses, integer keys, and sets."""

    @dataclass(frozen=True)
    class Example:
        path: Path
        groups: frozenset[int]

    value = {2: Example(Path("/tmp/x"), frozenset({3, 1}))}
    assert to_canonical_value(value) == {"2": {"path": "/tmp/x", "groups": [1, 3]}}


def test_canonical_hash_rejects_nan() -> None:
    """Catches nonportable NaN values in receipts."""
    with pytest.raises(ValueError):
        canonical_sha256({"score": float("nan")})


def test_canonical_hash_rejects_unsupported_objects() -> None:
    """Catches silent stringification of unknown lineage values."""
    with pytest.raises(TypeError, match="unsupported canonical value"):
        canonical_sha256(object())
